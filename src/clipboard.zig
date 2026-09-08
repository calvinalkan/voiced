//! Desktop clipboard facade. Automatic selection prefers Wayland data control,
//! then X11, then the focus-taking core Wayland protocol. An explicit backend
//! never falls through to another. Both clients retain the same bounded
//! borrowed-text contract.
const Clipboard = @This();

const std = @import("std");
const linux = std.os.linux;
const WaylandClipboard = @import("clipboard/wayland.zig");
const X11Clipboard = @import("clipboard/x11.zig");

pub const BackendSelection = enum { auto, wayland, x11 };
pub const Mode = enum { core, wlr, ext, x11 };
pub const Candidate = enum { wayland_data_control, x11, wayland_core };
pub const CandidateNotice = union(enum) {
    skipped: struct { candidate: Candidate, fallback: Candidate },
    failed: struct { candidate: Candidate, fallback: Candidate, error_detail: Error },
};
pub const TextTransfer = struct {
    id: u64,
    text_size: usize,
    started_monotonic_ns: u64,
    completed_monotonic_ns: u64,
};
pub const Error = union(enum) {
    wayland: WaylandClipboard.Error,
    x11: X11Clipboard.Error,
    unavailable,
    busy,
    invalid_text,
};
/// `text_transferred` means the owner finished serving one text request. It is
/// evidence of a clipboard read, not confirmation that an application inserted
/// the text.
pub const Event = union(enum) { none, ready, candidate_notice: CandidateNotice, acquired: u64, text_transferred: TextTransfer };
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
    wayland: WaylandClipboard,
    x11: X11Clipboard,
};
const AutoFallback = struct {
    environment: Environment,
    wayland_display: []const u8,
    candidate: Candidate = .wayland_data_control,
    next: ?Candidate = .x11,
};

backend: Backend = .none,
auto_fallback: ?AutoFallback = null,
epoll_fd: i32 = -1,
tag: u64 = 0,
mode: Mode = .core,
transfers_completed: u64 = 0,
transfers_expired: u64 = 0,
transfers_rejected: u64 = 0,

/// Initializes fresh or deinitialized storage. `auto` tries Wayland ext/wlr
/// data control, X11, then core Wayland when both displays exist. Explicit
/// selections never fall through. An unset WAYLAND_DISPLAY still selects the
/// current `wayland-0` socket when one exists.
pub fn init(self: *Clipboard, epoll_fd: i32, tag: u64, environment: Environment, selection: BackendSelection, now_ns: u64) Result(void) {
    self.backend = .none;
    self.auto_fallback = null;
    self.epoll_fd = epoll_fd;
    self.tag = tag;
    self.mode = .core;
    self.transfers_completed = 0;
    self.transfers_expired = 0;
    self.transfers_rejected = 0;

    const wayland_display = environment.wayland_display orelse defaultWaylandDisplay(environment);
    return switch (selection) {
        .wayland => if (wayland_display) |display|
            self.startWayland(environment, display, .best, now_ns)
        else
            .{ .err = .unavailable },
        .x11 => if (environment.display != null)
            self.startX11(environment, now_ns)
        else
            .{ .err = .unavailable },
        .auto => {
            if (wayland_display) |display| {
                if (environment.display != null) {
                    self.auto_fallback = .{ .environment = environment, .wayland_display = display };
                    // Auto candidates report connection failures through
                    // advance(), allowing the supervisor to log each fallback
                    // decision before the next candidate can also fail.
                    _ = self.startWayland(environment, display, .data_control_only, now_ns);
                    return .{ .ok = {} };
                }
                return self.startWayland(environment, display, .best, now_ns);
            }
            if (environment.display != null) return self.startX11(environment, now_ns);
            return .{ .err = .unavailable };
        },
    };
}

pub fn deinit(self: *Clipboard) void {
    self.closeBackend();
    self.auto_fallback = null;
}

pub fn publish(self: *Clipboard, id: u64, text: []const u8, now_ns: u64) Result(void) {
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

pub fn advance(self: *Clipboard, now_ns: u64) Result(Event) {
    const result: Result(Event) = switch (self.backend) {
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
    switch (result) {
        .ok => |event| {
            if (event == .ready) self.auto_fallback = null;
            self.syncMetrics();
            return .{ .ok = event };
        },
        .err => |err| {
            const fallback = self.auto_fallback orelse {
                self.syncMetrics();
                return .{ .err = err };
            };
            const next = fallback.next orelse {
                self.auto_fallback = null;
                self.syncMetrics();
                return .{ .err = err };
            };
            const notice: CandidateNotice = if (fallback.candidate == .wayland_data_control and dataControlUnavailable(err))
                .{ .skipped = .{ .candidate = fallback.candidate, .fallback = next } }
            else
                .{ .failed = .{ .candidate = fallback.candidate, .fallback = next, .error_detail = err } };
            self.closeBackend();
            self.auto_fallback.?.candidate = next;
            self.auto_fallback.?.next = if (next == .x11) .wayland_core else null;
            switch (next) {
                .x11 => _ = self.startX11(fallback.environment, now_ns),
                .wayland_core => _ = self.startWayland(fallback.environment, fallback.wayland_display, .core_only, now_ns),
                .wayland_data_control => unreachable,
            }
            return .{ .ok = .{ .candidate_notice = notice } };
        },
    }
}

pub fn isBorrowed(self: *const Clipboard, id: u64) bool {
    return switch (self.backend) {
        .none => false,
        .wayland => |*client| client.isBorrowed(id),
        .x11 => |*client| client.isBorrowed(id),
    };
}

pub fn owns(self: *const Clipboard, id: u64) bool {
    return switch (self.backend) {
        .none => false,
        .wayland => |*client| client.owns(id),
        .x11 => |*client| client.owns(id),
    };
}

pub fn ready(self: *const Clipboard) bool {
    return switch (self.backend) {
        .none => false,
        .wayland => |*client| client.phase == .ready,
        .x11 => |*client| client.ready(),
    };
}

pub fn backendName(self: *const Clipboard) []const u8 {
    return switch (self.backend) {
        .none => "none",
        .wayland => "wayland",
        .x11 => "x11",
    };
}

pub fn deadline(self: *const Clipboard) u64 {
    return switch (self.backend) {
        .none => std.math.maxInt(u64),
        .wayland => |*client| client.deadline(),
        .x11 => |*client| client.deadline(),
    };
}

fn startWayland(self: *Clipboard, environment: Environment, display: []const u8, policy: WaylandClipboard.ProtocolPolicy, now_ns: u64) Result(void) {
    self.backend = .{ .wayland = undefined };
    const result = self.backend.wayland.init(self.epoll_fd, self.tag, .{
        .runtime_directory = environment.runtime_directory,
        .display = display,
    }, policy, now_ns);
    self.syncMetrics();
    return switch (result) {
        .ok => .{ .ok = {} },
        .err => |err| .{ .err = mapWaylandError(err) },
    };
}

fn startX11(self: *Clipboard, environment: Environment, now_ns: u64) Result(void) {
    self.backend = .{ .x11 = undefined };
    const result = self.backend.x11.init(self.epoll_fd, self.tag, .{
        .display = environment.display,
        .authority = environment.xauthority,
        .home = environment.home,
    }, now_ns);
    self.syncMetrics();
    return switch (result) {
        .ok => .{ .ok = {} },
        .err => |err| .{ .err = mapX11Error(err) },
    };
}

fn closeBackend(self: *Clipboard) void {
    switch (self.backend) {
        .none => {},
        .wayland => |*client| client.deinit(),
        .x11 => |*client| client.deinit(),
    }
    self.backend = .none;
}

fn dataControlUnavailable(err: Error) bool {
    return switch (err) {
        .wayland => |detail| switch (detail) {
            .unsupported => |feature| feature == .data_control,
            else => false,
        },
        else => false,
    };
}

fn syncMetrics(self: *Clipboard) void {
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

// Probe instead of assuming `wayland-0`: a missing socket must leave X11
// available to automatic selection rather than producing a Wayland failure.
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

fn mapWaylandError(err: WaylandClipboard.Error) Error {
    return switch (err) {
        .busy => .busy,
        .invalid_text => .invalid_text,
        else => .{ .wayland = err },
    };
}

fn mapX11Error(err: X11Clipboard.Error) Error {
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
        .text_transferred => |transfer| .{ .text_transferred = .{
            .id = transfer.id,
            .text_size = transfer.text_size,
            .started_monotonic_ns = transfer.started_monotonic_ns,
            .completed_monotonic_ns = transfer.completed_monotonic_ns,
        } },
    };
}
