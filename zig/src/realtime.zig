//! Best-effort realtime scheduling for the isolated capture worker. Acquire
//! before audio starts. The complete RTKit exchange has one 500 ms deadline;
//! supervisor stop/cancel readiness interrupts it without consuming the command.
const std = @import("std");
const linux = std.os.linux;
const dbus = @import("dbus.zig");
const log = @import("logging.zig").scoped(.audio);
const destination = "org.freedesktop.RealtimeKit1";
const path = "/org/freedesktop/RealtimeKit1";

pub fn acquire(address: []const u8, control_fd: linux.fd_t) void {
    const params: linux.sched_param = .{ .priority = 20 };
    if (linux.errno(linux.sched_setscheduler(0, .{ .mode = .FIFO, .RESET_ON_FORK = true }, &params)) == .SUCCESS) return;
    var request: Request = .{ .deadline = now() + 500 * std.time.ns_per_ms, .control_fd = control_fd };
    defer request.connection.close();
    request.run(address) catch |err| {
        log.warn(.{}, "Realtime scheduling unavailable: operation={s}, errno={t}, bus_error=\"{f}\", detail=\"{f}\"", .{ @errorName(err), request.connection.errno, std.zig.fmtString(request.error_name[0..request.error_name_size]), std.zig.fmtString(request.error_message[0..request.error_message_size]) });
    };
}

const Request = struct {
    connection: dbus.Connection = .{},
    deadline: u64,
    control_fd: linux.fd_t,
    serial: u32 = 0,
    error_name: [256]u8 = undefined,
    error_name_size: usize = 0,
    error_message: [512]u8 = undefined,
    error_message_size: usize = 0,

    fn run(self: *Request, address_text: []const u8) !void {
        const address = try dbus.Address.parse(address_text);
        try self.connection.connect(&address);
        var uid_buffer: [10]u8 = undefined;
        const uid = try std.fmt.bufPrint(&uid_buffer, "{d}", .{linux.getuid()});
        var encoded: [20]u8 = undefined;
        for (uid, 0..) |byte, i| {
            encoded[i * 2] = std.fmt.digitToChar(byte >> 4, .lower);
            encoded[i * 2 + 1] = std.fmt.digitToChar(byte & 15, .lower);
        }
        self.connection.output_size = (try std.fmt.bufPrint(&self.connection.output, "\x00AUTH EXTERNAL {s}\r\n", .{encoded[0 .. uid.len * 2]})).len;
        while (true) {
            try self.progress();
            const input = self.connection.input[0..self.connection.input_size];
            if (std.mem.indexOf(u8, input, "\r\n")) |end| {
                if (!std.mem.startsWith(u8, input[0..end], "OK ")) return error.AuthenticationRejected;
                self.connection.consume(end + 2);
                break;
            }
        }
        @memcpy(self.connection.output[0..7], "BEGIN\r\n");
        self.connection.output_size = 7;
        while (self.connection.output_size > 0) try self.progress();
        self.serial += 1;
        var writer = try dbus.Writer.call(&self.connection.output, self.serial, "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "Hello", "");
        self.connection.output_size = writer.finish();
        var hello = try self.reply();
        _ = try hello.body.string();
        try hello.body.end();
        self.connection.consume(hello.size);

        const priority = try self.property("MaxRealtimePriority");
        const time_limit = try self.property("RTTimeUSecMax");
        if (priority <= 0 or priority > 99 or time_limit <= 0) return error.InvalidRealtimeLimits;
        const inherited = try std.posix.getrlimit(.RTTIME);
        // RTKit requires a finite CPU-time budget between blocking syscalls.
        // Match PipeWire's 200 ms cap and preserve any stricter inherited limit.
        const maximum: u64 = @min(@as(u64, @intCast(time_limit)), 200000, inherited.max);
        try std.posix.setrlimit(.RTTIME, .{ .cur = @min(maximum, inherited.cur), .max = maximum });
        self.serial += 1;
        writer = try dbus.Writer.call(&self.connection.output, self.serial, destination, path, destination, "MakeThreadRealtimeWithPID", "ttu");
        try writer.uint64(@intCast(linux.getpid()));
        try writer.uint64(@intCast(linux.gettid()));
        try writer.uint32(@intCast(@min(priority, 20)));
        self.connection.output_size = writer.finish();
        const reply_message = try self.reply();
        try reply_message.body.end();
        self.connection.consume(reply_message.size);
    }

    fn property(self: *Request, name: []const u8) !i64 {
        self.serial += 1;
        var writer = try dbus.Writer.call(&self.connection.output, self.serial, destination, path, "org.freedesktop.DBus.Properties", "Get", "ss");
        try writer.string(destination);
        try writer.string(name);
        self.connection.output_size = writer.finish();
        var message = try self.reply();
        if (!std.mem.eql(u8, message.signature, "v")) return error.InvalidRealtimeLimits;
        const signature = try message.body.signature();
        const value: i64 = if (std.mem.eql(u8, signature, "i")) @as(i32, @bitCast(try message.body.uint32())) else if (std.mem.eql(u8, signature, "x")) @bitCast(try message.body.uint64()) else return error.InvalidRealtimeLimits;
        try message.body.end();
        self.connection.consume(message.size);
        return value;
    }

    fn reply(self: *Request) !dbus.Message {
        while (true) {
            if (try dbus.Message.parse(self.connection.input[0..self.connection.input_size])) |message| {
                if (message.reply_serial == self.serial and (message.kind == .reply or message.kind == .err)) {
                    if (message.kind == .err) {
                        self.error_name_size = @min(message.error_name.len, self.error_name.len);
                        @memcpy(self.error_name[0..self.error_name_size], message.error_name[0..self.error_name_size]);
                        var body = message.body;
                        const text = body.string() catch "";
                        self.error_message_size = @min(text.len, self.error_message.len);
                        @memcpy(self.error_message[0..self.error_message_size], text[0..self.error_message_size]);
                        return error.RealtimeRequestRejected;
                    }
                    return message;
                }
                self.connection.consume(message.size);
                if (now() >= self.deadline) return error.RealtimeRequestTimedOut;
                continue;
            }
            try self.progress();
        }
    }

    fn progress(self: *Request) !void {
        const current = now();
        if (current >= self.deadline) return error.RealtimeRequestTimedOut;
        try self.connection.flush();
        var fds = [_]linux.pollfd{
            .{ .fd = self.control_fd, .events = linux.POLL.IN, .revents = 0 },
            .{ .fd = self.connection.fd.?, .events = linux.POLL.IN | (if (self.connection.output_size > 0) @as(i16, linux.POLL.OUT) else 0), .revents = 0 },
        };
        const result = linux.poll(&fds, fds.len, @intCast(@min(50, (self.deadline - current + std.time.ns_per_ms - 1) / std.time.ns_per_ms)));
        if (linux.errno(result) == .INTR) return;
        if (linux.errno(result) != .SUCCESS) return error.RealtimePollFailed;
        if (fds[0].revents != 0) return error.CaptureStopPending;
        if (fds[1].revents & (linux.POLL.ERR | linux.POLL.HUP) != 0) return error.Disconnected;
        if (fds[1].revents & linux.POLL.IN != 0) _ = try self.connection.read();
    }
};
fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
