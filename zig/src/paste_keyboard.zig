//! One persistent Linux virtual keyboard; paste frames advance on the service
//! timer. Completion means the kernel accepted all key releases, not that the
//! focused application inserted text. Never retry an uncertain paste.
const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.paste);
const linux = std.os.linux;

pub const Error = union(enum) {
    open: linux.E,
    configure: struct { request: u32, argument: union(enum) { value: usize, setup: UinputSetup }, errno: linux.E },
    write: struct { progress: Progress, errno: linux.E },
    ambiguous_write: struct { progress: Progress, bytes_written: usize },
    timed_out: struct { progress: Progress, deadline_ns: u64, observed_ns: u64 },
};

pub const Progress = struct { chord: Chord, frame: u3, frame_bytes_sent: usize };
pub fn Result(comptime T: type) type {
    return union(enum) { ok: T, err: Error };
}

pub const Chord = enum {
    @"ctrl+shift+v",
    @"ctrl+v",
    @"shift+insert",

    fn keys(self: Chord) []const u16 {
        return switch (self) {
            .@"ctrl+shift+v" => &.{ key_ctrl, key_shift, key_v },
            .@"ctrl+v" => &.{ key_ctrl, key_v },
            .@"shift+insert" => &.{ key_shift, key_insert },
        };
    }
};

const key_ctrl = 29;
const key_shift = 42;
const key_v = 47;
const key_insert = 110;

pub const Keyboard = struct {
    fd: linux.fd_t,
    usable_after_ns: u64,
    pending: ?struct {
        chord: Chord,
        key_gap_ns: u64,
        stage: u3 = 0,
        sent: usize = 0,
        next_ns: u64,
        expires_ns: u64,
    } = null,

    pub fn open(now_ns: u64) Result(Keyboard) {
        const result = linux.open("/dev/uinput", .{ .ACCMODE = .WRONLY, .NONBLOCK = true, .CLOEXEC = true }, 0);
        if (linux.errno(result) != .SUCCESS) return .{ .err = .{ .open = linux.errno(result) } };
        const fd: linux.fd_t = @intCast(result);
        var initialized = false;
        defer {
            if (!initialized) _ = linux.close(fd);
        }

        if (configure(fd, linux.IOCTL.IOW('U', 100, c_int), 1)) |err| return .{ .err = err }; // UI_SET_EVBIT(EV_KEY)

        for (1..32) |key| {
            // udev recognizes keyboards by capability bits 1..31. Only advertising
            // our chord keys can leave this device unclassified and ignored.
            if (configure(fd, linux.IOCTL.IOW('U', 101, c_int), key)) |err| return .{ .err = err };
        }

        for ([_]u16{ key_ctrl, key_shift, key_v, key_insert }) |key| {
            if (configure(fd, linux.IOCTL.IOW('U', 101, c_int), key)) |err| return .{ .err = err };
        }

        var setup = std.mem.zeroes(UinputSetup);
        setup.id.bustype = 0x06; // BUS_VIRTUAL
        setup.id.version = 1;
        const name = "Voiced paste keyboard";
        @memcpy(setup.name[0..name.len], name);

        if (configure(fd, linux.IOCTL.IOW('U', 3, UinputSetup), @intFromPtr(&setup))) |err| return .{ .err = err };
        if (configure(fd, linux.IOCTL.IO('U', 1), 0)) |err| return .{ .err = err }; // UI_DEV_CREATE

        // Enumeration is asynchronous. The kernel example's one-second
        // allowance is not proof of readiness at the destination application.
        initialized = true;
        return .{ .ok = .{ .fd = fd, .usable_after_ns = now_ns + std.time.ns_per_s } };
    }

    pub fn beginPaste(self: *Keyboard, chord: Chord, key_gap_ms: u16, now_ns: u64) void {
        std.debug.assert(self.pending == null);
        std.debug.assert(now_ns >= self.usable_after_ns);

        const key_gap_ns = @as(u64, key_gap_ms) * std.time.ns_per_ms;
        // Four frames require three intentional gaps. Keep those waits outside
        // the 200 ms allowance for scheduling delays and nonblocking writes.
        self.pending = .{
            .chord = chord,
            .key_gap_ns = key_gap_ns,
            .next_ns = now_ns,
            .expires_ns = now_ns + 3 * key_gap_ns + 200 * std.time.ns_per_ms,
        };
    }

    /// Returns true once the final key-up frame has been accepted. Errors
    /// require deinit: destroying the device also releases remaining key state.
    pub fn advance(self: *Keyboard, now_ns: u64) Result(bool) {
        const pending = if (self.pending) |*value| value else {
            return .{ .ok = false };
        };
        const progress: Progress = .{ .chord = pending.chord, .frame = pending.stage, .frame_bytes_sent = pending.sent };
        if (now_ns >= pending.expires_ns) {
            return .{ .err = .{ .timed_out = .{ .progress = progress, .deadline_ns = pending.expires_ns, .observed_ns = now_ns } } };
        }
        if (now_ns < pending.next_ns) {
            return .{ .ok = false };
        }

        const keys = pending.chord.keys();
        const modifiers = keys[0 .. keys.len - 1];

        var frame: [4]InputEvent = undefined;
        var count: usize = 0;

        switch (pending.stage) {
            0 => for (modifiers) |key| {
                frame[count] = keyEvent(key, 1);
                count += 1;
            },
            1, 2 => {
                frame[0] = keyEvent(keys[keys.len - 1], if (pending.stage == 1) 1 else 0);
                count = 1;
            },
            3 => for (0..modifiers.len) |index| {
                frame[count] = keyEvent(modifiers[modifiers.len - index - 1], 0);
                count += 1;
            },
            else => unreachable,
        }

        frame[count] = std.mem.zeroes(InputEvent); // EV_SYN / SYN_REPORT

        const bytes = std.mem.sliceAsBytes(frame[0 .. count + 1]);

        const written = linux.write(self.fd, bytes[pending.sent..].ptr, bytes.len - pending.sent);
        switch (linux.errno(written)) {
            .SUCCESS => {},
            .AGAIN, .INTR => {
                pending.next_ns = now_ns + 2 * std.time.ns_per_ms;
                return .{ .ok = false };
            },
            else => |errno| return .{ .err = .{ .write = .{ .progress = progress, .errno = errno } } },
        }

        if (written == 0 or written % @sizeOf(InputEvent) != 0) {
            // Resume only after complete input_event records. Replaying a partial
            // or ambiguous frame could insert the transcript twice.
            return .{ .err = .{ .ambiguous_write = .{ .progress = progress, .bytes_written = written } } };
        }
        pending.sent += written;

        if (pending.sent != bytes.len) {
            pending.next_ns = now_ns + 2 * std.time.ns_per_ms;

            return .{ .ok = false };
        }

        if (pending.stage == 3) {
            self.pending = null;

            return .{ .ok = true };
        }

        pending.stage += 1;
        pending.sent = 0;
        pending.next_ns = now_ns + pending.key_gap_ns;

        return .{ .ok = false };
    }

    pub fn deadlineMonotonicNs(self: *const Keyboard) ?u64 {
        const pending = self.pending orelse return null;
        return @min(pending.next_ns, pending.expires_ns);
    }

    pub fn deinit(self: *Keyboard) void {
        if (self.pending) |pending| {
            const keys = pending.chord.keys();

            var frame: [4]InputEvent = undefined;

            for (0..keys.len) |index| {
                frame[index] = keyEvent(keys[keys.len - index - 1], 0);
            }

            frame[keys.len] = std.mem.zeroes(InputEvent);

            const bytes = std.mem.sliceAsBytes(frame[0 .. keys.len + 1]);

            const written = linux.write(self.fd, bytes.ptr, bytes.len);
            if (linux.errno(written) != .SUCCESS or written != bytes.len)
                log.err(.{}, .paste_cleanup_write_failed, "system_error={f} syscall_result={d} write_size={d}", .{ logging.fmtErrno(linux.errno(written)), written, bytes.len });
        }

        const destroy_errno = linux.errno(linux.ioctl(self.fd, linux.IOCTL.IO('U', 2), 0)); // UI_DEV_DESTROY
        if (destroy_errno != .SUCCESS) log.err(.{}, .paste_cleanup_destroy_failed, "system_error={f}", .{logging.fmtErrno(destroy_errno)});
        _ = linux.close(self.fd);

        self.* = undefined;
    }
};

// Linux UAPI: input.h and uinput.h. Keep the small ABI we use here; ioctl's
// request encoding comes from Zig's architecture-specific Linux implementation.

const InputEvent = extern struct { time: linux.timeval, type: u16, code: u16, value: i32 };

const UinputSetup = extern struct {
    id: extern struct { bustype: u16, vendor: u16, product: u16, version: u16 },
    name: [80]u8,
    ff_effects_max: u32,
};

fn keyEvent(code: u16, value: i32) InputEvent {
    var event = std.mem.zeroes(InputEvent);
    event.type = 1; // EV_KEY
    event.code = code;
    event.value = value;

    return event;
}

fn configure(fd: linux.fd_t, request: u32, argument: usize) ?Error {
    const errno = linux.errno(linux.ioctl(fd, request, argument));
    // Snapshot the setup value rather than retaining its stack address.
    return if (errno == .SUCCESS) null else .{ .configure = .{ .request = request, .argument = if (request == linux.IOCTL.IOW('U', 3, UinputSetup)) .{ .setup = @as(*const UinputSetup, @ptrFromInt(argument)).* } else .{ .value = argument }, .errno = errno } };
}
