//! Standalone native clipboard spike. Preloads two UTF-8 files; stdin commands:
//! a/b publish, s reports ownership/transfer state, p sends one optional paste,
//! q exits. The caller drives this over pipes; stdout is a JSON event stream.
const std = @import("std");
const clipboard = @import("clipboard_wayland.zig");
const paste = @import("paste_keyboard.zig");
const logging = @import("logging.zig");
const linux = std.os.linux;

pub fn main(init: std.process.Init) !void {
    logging.initCli(.debug);
    defer logging.deinit();
    const args = try init.minimal.args.toSlice(init.gpa);
    defer init.gpa.free(args);
    var fallback = false;
    var with_paste = false;
    var paths: [2][]const u8 = undefined;
    var count: usize = 0;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--fallback")) fallback = true else if (std.mem.eql(u8, arg, "--paste")) with_paste = true else {
            if (count == paths.len) return error.InvalidArguments;
            paths[count] = arg;
            count += 1;
        }
    }
    if (count != paths.len) return error.InvalidArguments;
    const first = try std.Io.Dir.cwd().readFileAlloc(init.io, paths[0], init.gpa, .limited(4 * 1024 * 1024));
    defer init.gpa.free(first);
    const second = try std.Io.Dir.cwd().readFileAlloc(init.io, paths[1], init.gpa, .limited(4 * 1024 * 1024));
    defer init.gpa.free(second);
    _ = linux.sigaction(.PIPE, &.{ .handler = .{ .handler = linux.SIG.IGN }, .mask = linux.sigemptyset(), .flags = 0 }, null);
    const epoll = linux.epoll_create1(linux.EPOLL.CLOEXEC);
    if (linux.errno(epoll) != .SUCCESS) return error.Epoll;
    const epoll_fd: i32 = @intCast(epoll);
    defer _ = linux.close(epoll_fd);
    var input: linux.epoll_event = .{ .events = linux.EPOLL.IN, .data = .{ .u64 = 2 } };
    if (linux.errno(linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, 0, &input)) != .SUCCESS) return error.Epoll;
    var client: clipboard.Client = .{};
    defer client.deinit();
    switch (client.init(epoll_fd, 1, .{ .runtime_directory = init.environ_map.get("XDG_RUNTIME_DIR"), .display = init.environ_map.get("WAYLAND_DISPLAY") }, fallback, now())) {
        .ok => {},
        .err => |err| {
            reportError(err);
            return error.Clipboard;
        },
    }
    var keyboard: ?paste.Keyboard = null;
    defer if (keyboard) |*k| k.deinit();
    if (with_paste) keyboard = switch (paste.Keyboard.open(now())) {
        .ok => |k| k,
        .err => |err| {
            logging.scoped(.clipboard).err(.{}, "Paste setup error: detail={any}", .{err});
            return error.Keyboard;
        },
    };
    while (true) {
        switch (client.advance(now())) {
            .ok => |event| switch (event) {
                .none => {},
                .ready => emit("{{\"event\":\"ready\",\"mode\":\"{t}\"}}\n", .{client.mode}),
                .acquired => |id| emit("{{\"event\":\"acquired\",\"id\":{d}}}\n", .{id}),
            },
            .err => |err| {
                reportError(err);
                return error.Clipboard;
            },
        }
        if (keyboard) |*k| if (k.pending != null) {
            switch (k.advance(now())) {
                .ok => |done| if (done) {
                    emit("{{\"event\":\"pasted\"}}\n", .{});
                },
                .err => |err| {
                    logging.scoped(.clipboard).err(.{}, "Paste error: detail={any}", .{err});
                    return error.Keyboard;
                },
            }
        };
        var deadline = client.deadline();
        if (keyboard) |*k| if (k.deadlineMonotonicNs()) |d| {
            deadline = @min(deadline, d);
        };
        const timeout: i32 = if (deadline == std.math.maxInt(u64)) -1 else @intCast(@min(std.math.maxInt(i32), (deadline -| now() +| (std.time.ns_per_ms - 1)) / std.time.ns_per_ms));
        var events: [16]linux.epoll_event = undefined;
        const ready = linux.epoll_wait(epoll_fd, &events, events.len, timeout);
        if (linux.errno(ready) == .INTR) continue;
        if (linux.errno(ready) != .SUCCESS) return error.Epoll;
        for (events[0..ready]) |event| if (event.data.u64 == 2) {
            var byte: [1]u8 = undefined;
            if (linux.read(0, &byte, 1) != 1) return;
            switch (byte[0]) {
                'a', 'b' => switch (client.publish(if (byte[0] == 'a') 1 else 2, if (byte[0] == 'a') first else second, now())) {
                    .ok => {},
                    .err => |err| {
                        reportError(err);
                        if (err != .busy and err != .invalid_text) return error.Clipboard;
                    },
                },
                's' => emit("{{\"event\":\"status\",\"owns_a\":{},\"owns_b\":{},\"borrowed_a\":{},\"borrowed_b\":{},\"completed\":{d},\"expired\":{d},\"rejected\":{d}}}\n", .{ client.owns(1), client.owns(2), client.isBorrowed(1), client.isBorrowed(2), client.transfers_completed, client.transfers_expired, client.transfers_rejected }),
                'p' => if (keyboard) |*k| {
                    if (k.pending != null or now() < k.usable_after_ns or client.phase != .ready) return error.PasteNotReady;
                    k.beginPaste(.@"ctrl+v", 4, now());
                } else return error.PasteDisabled,
                'q' => return,
                else => {},
            }
        };
    }
}
fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
fn reportError(err: clipboard.Error) void {
    switch (err) {
        .server => |detail| logging.scoped(.clipboard).err(.{}, "Wayland error: object={d}, code={d}, message={s}, truncated={}", .{ detail.object, detail.code, detail.message[0..detail.message_size], detail.truncated }),
        else => logging.scoped(.clipboard).err(.{}, "Native clipboard error: detail={any}", .{err}),
    }
    emit("{{\"event\":\"error\",\"kind\":\"{t}\"}}\n", .{std.meta.activeTag(err)});
}
fn emit(comptime format: []const u8, args: anytype) void {
    var buffer: [1024]u8 = undefined;
    const bytes = std.fmt.bufPrint(&buffer, format, args) catch unreachable;
    _ = linux.write(1, bytes.ptr, bytes.len);
}
