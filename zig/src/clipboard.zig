//! Desktop clipboard facade. A session selects exactly one native backend:
//! Wayland when WAYLAND_DISPLAY is present, otherwise the compositor default
//! `wayland-0` socket when it exists, otherwise X11 when DISPLAY is present.
//! Both implementations retain the same bounded borrowed-text contract.
const std = @import("std");
const linux = std.os.linux;
const wayland = @import("clipboard_wayland.zig");
const x11 = @import("clipboard_x11.zig");

pub const Mode = enum { core, wlr, ext, x11 };
pub const Error = union(enum) {
    wayland: wayland.Error,
    x11: x11.Error,
    unavailable,
    busy,
    invalid_text,
};
pub const Event = union(enum) { none, ready, acquired: u64 };
pub const Environment = struct {
    runtime_directory: ?[]const u8,
    wayland_display: ?[]const u8,
    display: ?[]const u8,
    xauthority: ?[]const u8,
    home: ?[]const u8,
};

pub fn Result(comptime T: type) type {
    return union(enum) { ok: T, err: Error };
}

const Backend = union(enum) {
    none,
    wayland: wayland.Client,
    x11: x11.Client,
};

pub const Client = struct {
    backend: Backend = .none,
    mode: Mode = .core,
    transfers_completed: u64 = 0,
    transfers_expired: u64 = 0,
    transfers_rejected: u64 = 0,

    /// Initializes fresh or deinitialized storage. Presence selects one backend;
    /// an initialization failure never publishes through the other display.
    /// Unset WAYLAND_DISPLAY uses `wayland-0` when that socket exists now.
    pub fn init(self: *Client, epoll_fd: i32, tag: u64, environment: Environment, fallback_only: bool, now_ns: u64) Result(void) {
        self.backend = .none;
        self.mode = .core;
        self.transfers_completed = 0;
        self.transfers_expired = 0;
        self.transfers_rejected = 0;
        // Re-select on every init, including delivery reconnect, instead of
        // caching the first backend. A systemd user unit can start before
        // graphical-session imports WAYLAND_DISPLAY, and a compositor restart
        // after lock-screen or a new login can drop the connection while voiced
        // keeps running. Caching the first miss would leave clipboard unavailable
        // for the rest of the process.
        if (environment.wayland_display orelse defaultWaylandDisplay(environment)) |display| {
            self.backend = .{ .wayland = undefined };
            const result = self.backend.wayland.init(epoll_fd, tag, .{ .runtime_directory = environment.runtime_directory, .display = display }, fallback_only, now_ns);
            self.syncMetrics();
            return switch (result) {
                .ok => .{ .ok = {} },
                .err => |err| .{ .err = mapWaylandError(err) },
            };
        }
        if (environment.display != null) {
            self.backend = .{ .x11 = undefined };
            const result = self.backend.x11.init(epoll_fd, tag, .{ .display = environment.display, .authority = environment.xauthority, .home = environment.home }, now_ns);
            self.mode = .x11;
            self.syncMetrics();
            return switch (result) {
                .ok => .{ .ok = {} },
                .err => |err| .{ .err = mapX11Error(err) },
            };
        }
        return .{ .err = .unavailable };
    }

    pub fn deinit(self: *Client) void {
        switch (self.backend) {
            .none => {},
            .wayland => |*client| client.deinit(),
            .x11 => |*client| client.deinit(),
        }
        self.backend = .none;
    }

    pub fn publish(self: *Client, id: u64, text: []const u8, now_ns: u64) Result(void) {
        const result: Result(void) = switch (self.backend) {
            .none => .{ .err = .unavailable },
            .wayland => |*client| switch (client.publish(id, text, now_ns)) {
                .ok => .{ .ok = {} },
                .err => |err| .{ .err = mapWaylandError(err) },
            },
            .x11 => |*client| switch (client.publish(id, text, now_ns)) {
                .ok => .{ .ok = {} },
                .err => |err| .{ .err = mapX11Error(err) },
            },
        };
        self.syncMetrics();
        return result;
    }

    pub fn advance(self: *Client, now_ns: u64) Result(Event) {
        const event: Result(Event) = switch (self.backend) {
            .none => .{ .err = .unavailable },
            .wayland => |*client| switch (client.advance(now_ns)) {
                .ok => |value| .{ .ok = mapEvent(value) },
                .err => |err| .{ .err = mapWaylandError(err) },
            },
            .x11 => |*client| switch (client.advance(now_ns)) {
                .ok => |value| .{ .ok = mapEvent(value) },
                .err => |err| .{ .err = mapX11Error(err) },
            },
        };
        self.syncMetrics();
        return event;
    }

    pub fn isBorrowed(self: *const Client, id: u64) bool {
        return switch (self.backend) {
            .none => false,
            .wayland => |*client| client.isBorrowed(id),
            .x11 => |*client| client.isBorrowed(id),
        };
    }

    pub fn owns(self: *const Client, id: u64) bool {
        return switch (self.backend) {
            .none => false,
            .wayland => |*client| client.owns(id),
            .x11 => |*client| client.owns(id),
        };
    }

    pub fn ready(self: *const Client) bool {
        return switch (self.backend) {
            .none => false,
            .wayland => |*client| client.phase == .ready,
            .x11 => |*client| client.ready(),
        };
    }

    pub fn backendName(self: *const Client) []const u8 {
        return switch (self.backend) {
            .none => "none",
            .wayland => "wayland",
            .x11 => "x11",
        };
    }

    pub fn deadline(self: *const Client) u64 {
        return switch (self.backend) {
            .none => std.math.maxInt(u64),
            .wayland => |*client| client.deadline(),
            .x11 => |*client| client.deadline(),
        };
    }

    fn syncMetrics(self: *Client) void {
        switch (self.backend) {
            .none => {},
            .wayland => |*client| {
                self.mode = switch (client.mode) {
                    .core => .core,
                    .wlr => .wlr,
                    .ext => .ext,
                };
                self.transfers_completed = client.transfers_completed;
                self.transfers_expired = client.transfers_expired;
                self.transfers_rejected = client.transfers_rejected;
            },
            .x11 => |*client| {
                self.mode = .x11;
                self.transfers_completed = client.transfers_completed;
                self.transfers_expired = client.transfers_expired;
                self.transfers_rejected = client.transfers_rejected;
            },
        }
    }
};

// Probe instead of assuming `wayland-0`: a failed Wayland init never falls
// through to X11, so a missing socket must leave DISPLAY available.
fn defaultWaylandDisplay(environment: Environment) ?[]const u8 {
    const runtime_directory = environment.runtime_directory orelse return null;
    var path: [108]u8 = undefined;
    const socket = std.fmt.bufPrint(path[0 .. path.len - 1], "{s}/wayland-0", .{runtime_directory}) catch return null;
    path[socket.len] = 0;
    var stat: linux.Statx = undefined;
    if (linux.errno(linux.statx(linux.AT.FDCWD, @ptrCast(&path), 0, .BASIC_STATS, &stat)) != .SUCCESS) return null;
    if (!stat.mask.TYPE or stat.mode & linux.S.IFMT != linux.S.IFSOCK) return null;
    return "wayland-0";
}

fn mapWaylandError(err: wayland.Error) Error {
    return switch (err) {
        .busy => .busy,
        .invalid_text => .invalid_text,
        else => .{ .wayland = err },
    };
}

fn mapX11Error(err: x11.Error) Error {
    return switch (err) {
        .busy => .busy,
        .invalid_text => .invalid_text,
        else => .{ .x11 = err },
    };
}

fn mapEvent(event: anytype) Event {
    return switch (event) {
        .none => .none,
        .ready => .ready,
        .acquired => |id| .{ .acquired = id },
    };
}
