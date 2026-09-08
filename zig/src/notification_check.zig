//! Test driver for the production notification client. Commands on stdin:
//! a/b/c show errors, d shows a saved clipboard error, n resets suppression,
//! r recovers, q exits. Diagnostics
//! use the production logger; the runner supplies a private datagram receiver.
const std = @import("std");
const notifications = @import("notifications.zig");
const logging = @import("logging.zig");
const linux = std.os.linux;
pub fn main(init: std.process.Init) !void {
    std.debug.assert(logging.init(.debug) == .SUCCESS);
    defer logging.deinit();
    const fd: i32 = @intCast(linux.epoll_create1(linux.EPOLL.CLOEXEC));
    defer _ = linux.close(fd);
    var event: linux.epoll_event = .{ .events = linux.EPOLL.IN, .data = .{ .u64 = 2 } };
    std.debug.assert(linux.errno(linux.epoll_ctl(fd, linux.EPOLL.CTL_ADD, 0, &event)) == .SUCCESS);
    var client: notifications.Client = undefined;
    // Match supervisor construction, including cleanup with notifications off.
    // Poison storage so zero-filled pages cannot hide a missed metadata reset.
    @memset(std.mem.asBytes(&client), 0xa5);
    client.initEmpty();
    client.deinit(fd);
    client.init(fd, 1, init.environ_map.get("DBUS_SESSION_BUS_ADDRESS"), init.environ_map.get("XDG_RUNTIME_DIR"));
    defer client.deinit(fd);
    while (true) {
        client.advance(fd, 1);
        var ts: linux.timespec = undefined;
        _ = linux.clock_gettime(.MONOTONIC, &ts);
        const now = @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        const timeout: i32 = if (client.deadline_monotonic_ns == std.math.maxInt(u64)) -1 else @intCast(@min(std.math.maxInt(i32), (client.deadline_monotonic_ns -| now +| (std.time.ns_per_ms - 1)) / std.time.ns_per_ms));
        var events: [4]linux.epoll_event = undefined;
        const count = linux.epoll_wait(fd, &events, events.len, timeout);
        if (linux.errno(count) == .INTR) continue;
        std.debug.assert(linux.errno(count) == .SUCCESS);
        for (events[0..count]) |ready| {
            if (ready.data.u64 != 2) continue;
            var byte: [1]u8 = undefined;
            if (linux.read(0, &byte, 1) != 1) return;
            switch (byte[0]) {
                'a' => client.show(.microphone_failed),
                'b' => client.show(.transcription_failed),
                'c' => client.show(.transcript_save_failed),
                'd' => client.showOutput(.clipboard_failed, .{ .saved = notifications.transcriptDirectory("/home/test/.local/state/voiced", "/home/test") }),
                'r' => client.recover(),
                'n' => client.resetSuppression(),
                'q' => return,
                else => {},
            }
            _ = linux.write(1, "ok\n", 3);
        }
    }
}
