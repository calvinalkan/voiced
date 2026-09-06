//! Process-owned diagnostics. Service records use one bounded nonblocking
//! journal datagram; recording and inference hot paths must not call this API.
//! Initialize before starting threads, and close only after they have joined.
const std = @import("std");
const linux = std.os.linux;

pub const Level = enum(u8) { critical = 2, err = 3, warn = 4, info = 6, debug = 7 };
pub const Context = struct { recording_ordinal: ?u64 = null };
pub const record_bytes_max = 4096;
const message_bytes_max = 3584;
const Sink = union(enum) { disabled, journal: struct { fd: i32, connected: bool }, cli };
var sink: Sink = .disabled;
var threshold: Level = .info;
var dropped: std.atomic.Value(u64) = .init(0);

/// Service stderr may be a connected datagram socket supplied by a test
/// receiver. Otherwise send to journald's pathname on each call, so a journal
/// restart requires no reconnect state. Never turn a pipe/file into a service
/// logging sink: even a small write to those can block the deadline owner.
pub fn init(configured_level: Level) linux.E {
    std.debug.assert(sink == .disabled);
    threshold = configured_level;
    var socket_type: i32 = 0;
    var length: linux.socklen_t = @sizeOf(i32);
    if (linux.errno(linux.getsockopt(2, linux.SOL.SOCKET, linux.SO.TYPE, @ptrCast(&socket_type), &length)) == .SUCCESS and socket_type == linux.SOCK.DGRAM) {
        const result = linux.fcntl(2, linux.F.DUPFD_CLOEXEC, 3);
        if (linux.errno(result) != .SUCCESS) return linux.errno(result);
        sink = .{ .journal = .{ .fd = @intCast(result), .connected = true } };
    } else {
        const result = linux.socket(linux.AF.UNIX, linux.SOCK.DGRAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0);
        if (linux.errno(result) != .SUCCESS) return linux.errno(result);
        sink = .{ .journal = .{ .fd = @intCast(result), .connected = false } };
    }
    return .SUCCESS;
}

/// Offline CLI tools may use stderr; they do not own service deadlines. Help,
/// argument errors, and command results need not go through the logger.
pub fn initCli(configured_level: Level) void {
    std.debug.assert(sink == .disabled);
    threshold = configured_level;
    sink = .cli;
}

pub fn deinit() void {
    if (sink == .journal) _ = linux.close(sink.journal.fd);
    sink = .disabled;
    dropped.store(0, .monotonic);
}

pub fn level() Level {
    return threshold;
}

pub fn parseLevel(text: []const u8) ?Level {
    if (std.mem.eql(u8, text, "error")) return .err;
    if (std.mem.eql(u8, text, "warn")) return .warn;
    if (std.mem.eql(u8, text, "critical")) return .critical;
    if (std.mem.eql(u8, text, "info")) return .info;
    if (std.mem.eql(u8, text, "debug")) return .debug;
    return null;
}

pub fn levelName(value: Level) []const u8 {
    return if (value == .err) "error" else @tagName(value);
}

/// The event function filters before formatting. Guard expensive argument
/// preparation at the call site with enabled(); Zig evaluates arguments first.
pub fn enabled(value: Level) bool {
    return @intFromEnum(value) <= @intFromEnum(threshold);
}

pub fn scoped(comptime component: @EnumLiteral()) type {
    return struct {
        pub fn write(severity: Level, context: Context, comptime format: []const u8, args: anytype) void {
            // An inline switch would instantiate the formatter for each level.
            emit(severity, @tagName(component), context, format, args);
        }

        /// A zero exit is still unexpected if the owner needed more work.
        /// Requested termination can legitimately end through a signal.
        pub fn processExited(context: Context, role: []const u8, information: linux.siginfo_t, expected: bool) void {
            const exited = information.code == @intFromEnum(linux.CLD.EXITED);
            const status = information.fields.common.second.sigchld.status;
            const clean = if (exited) status == 0 else information.code == @intFromEnum(linux.CLD.KILLED) and status == @intFromEnum(linux.SIG.KILL);
            const severity: Level = if (expected and clean) .debug else .err;
            write(severity, context, "Worker exited: role={s}, pid={d}, expected={}, exit_kind={s}, exit_code={?d}, signal={?d}, core_dumped={}", .{
                role,                                               information.fields.common.first.piduid.pid, expected,
                if (exited) "exited" else "signaled",               if (exited) @as(?i32, status) else null,    if (exited) null else @as(?i32, status),
                information.code == @intFromEnum(linux.CLD.DUMPED),
            });
        }

        pub fn critical(context: Context, comptime format: []const u8, args: anytype) void {
            emit(.critical, @tagName(component), context, format, args);
        }
        pub fn err(context: Context, comptime format: []const u8, args: anytype) void {
            emit(.err, @tagName(component), context, format, args);
        }
        pub fn warn(context: Context, comptime format: []const u8, args: anytype) void {
            emit(.warn, @tagName(component), context, format, args);
        }
        pub fn info(context: Context, comptime format: []const u8, args: anytype) void {
            emit(.info, @tagName(component), context, format, args);
        }
        pub fn debug(context: Context, comptime format: []const u8, args: anytype) void {
            emit(.debug, @tagName(component), context, format, args);
        }
    };
}

fn emit(severity: Level, comptime component: []const u8, context: Context, comptime format: []const u8, args: anytype) void {
    if (!enabled(severity)) return;
    comptime std.debug.assert(component.len <= 64);
    var message_buffer: [message_bytes_max]u8 = undefined;
    var message = std.Io.Writer.fixed(&message_buffer);
    const marker = " [truncated]";
    message.print(format, args) catch {
        const end = @min(message.end, message_buffer.len - marker.len);
        @memcpy(message_buffer[end..][0..marker.len], marker);
        message.end = end + marker.len;
    };
    const text = std.mem.trimEnd(u8, message.buffered(), "\n");
    // Delivery borrows the stack buffer synchronously; it must not retain text.
    sendRecord(severity, component, context, text);
}

// Keep delivery non-generic and out-of-line so journal framing and syscalls are
// not copied into every message-format specialization. The extra call is on the
// logging cold path, not capture/inference hot paths. Measured with Zig 0.16 on
// x86-64 ReleaseSafe: ~151 KiB saved, a ~10% smaller whole stripped daemon.
noinline fn sendRecord(severity: Level, component: []const u8, context: Context, text: []const u8) void {
    if (sink == .cli) {
        var line_buffer: [record_bytes_max]u8 = undefined;
        const line = std.fmt.bufPrint(&line_buffer, "{s}: {s}\n", .{ levelName(severity), text }) catch unreachable;
        _ = linux.write(2, line.ptr, line.len);
        return;
    }
    const lost = dropped.swap(0, .monotonic);
    var buffer: [record_bytes_max]u8 = undefined;
    var record = std.Io.Writer.fixed(&buffer);
    record.print("PRIORITY={d}\nSYSLOG_IDENTIFIER=voiced\nVOICED_COMPONENT={s}\n", .{ @intFromEnum(severity), component }) catch unreachable;
    if (context.recording_ordinal) |value| record.print("VOICED_RECORDING_ORDINAL={d}\n", .{value}) catch unreachable;
    if (lost != 0) record.print("VOICED_DROPPED={d}\n", .{lost}) catch unreachable;
    // Always use the native binary MESSAGE encoding. Newlines in native error
    // details remain one message and cannot inject journal metadata fields.
    record.writeAll("MESSAGE\n") catch unreachable;
    record.writeInt(u64, text.len, .little) catch unreachable;
    record.writeAll(text) catch unreachable;
    record.writeByte('\n') catch unreachable;
    const sent = switch (sink) {
        .journal => |journal| blk: {
            var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
            const path = "/run/systemd/journal/socket";
            @memcpy(address.path[0..path.len], path);
            const bytes = record.buffered();
            const result = linux.sendto(journal.fd, bytes.ptr, bytes.len, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL, if (journal.connected) null else @ptrCast(&address), if (journal.connected) 0 else @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1));
            break :blk linux.errno(result) == .SUCCESS and result == bytes.len;
        },
        .disabled => false,
        .cli => unreachable,
    };
    if (!sent) countDropped(lost +| 1);
}

fn countDropped(count: u64) void {
    var previous = dropped.load(.monotonic);
    while (dropped.cmpxchgWeak(previous, previous +| count, .monotonic, .monotonic)) |actual| previous = actual;
}
