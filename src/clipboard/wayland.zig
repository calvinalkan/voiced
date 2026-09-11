//! Native Wayland clipboard for an epoll owner. Text storage is borrowed until
//! `isBorrowed(storage_index)` becomes false, including transfers surviving
//! replacement.
//! Two text generations and eight transfers bound retention. The GNOME fallback
//! temporarily maps a surface; acquisition includes destroying that surface,
//! but cannot certify which other application ultimately receives focus.
const WaylandClipboard = @This();

const std = @import("std");
const linux = std.os.linux;
const types = @import("types.zig");
const wire = @import("wayland_wire.zig");
const timeout_ns = 2 * std.time.ns_per_s;
const mime_types = [_][]const u8{ "text/plain;charset=utf-8", "text/plain", "UTF8_STRING" };
pub const Error = union(enum) {
    transport: struct { cause: anyerror, errno: linux.E, object: u32, opcode: u16 },
    server: struct { object: u32, code: u32, message: [256]u8, message_size: usize, truncated: bool },
    unsupported: enum { display, seat, keyboard, clipboard, data_control, surface },
    timed_out: Phase,
    selection_lost,
    busy,
    invalid_text,
};
pub fn Result(comptime T: type) type {
    return union(enum) { ok: T, err: Error };
}
pub const Mode = enum { core, wlr, ext };
pub const PublicationId = types.PublicationId;
pub const StorageIndex = types.StorageIndex;
pub const TextTransfer = types.TextTransfer;
/// Selects the strongest protocol, requires background-safe data control, or
/// requires the serial-acquiring core protocol respectively.
pub const ProtocolPolicy = enum { best, data_control_only, core_only };
pub const Phase = enum { discovering, binding, ready, focus, selection, restoring };
pub const Event = union(enum) { none, ready, acquired: PublicationId, text_transferred: TextTransfer };
pub const Environment = struct { runtime_directory: ?[]const u8, display: []const u8 };
const Kind = enum { display, registry, discovery, binding, selection, restoring, compositor, shm, seat, keyboard, manager, device, source, surface, pool, buffer, shell, xdg_surface, toplevel, gtk_shell, gtk_surface };
const Object = struct { kind: Kind, destroyed: bool = false };
const Global = struct { name: u32 = 0, version: u32 = 0, object: u32 = 0 };
const Text = struct { bytes: []const u8, publication: PublicationId };
const Source = union(enum) {
    empty,
    offered: struct { object: u32, text: Text },
    retired: Text,

    fn text(self: Source) ?Text {
        return switch (self) {
            .empty => null,
            .offered => |value| value.text,
            .retired => |value| value,
        };
    }
};
const Pending = struct { slot: u1, deadline: u64 };
const Popup = struct { surface: u32, xdg: u32, toplevel: u32, gtk: u32, attached: bool = false };
const Operation = union(Phase) {
    discovering: u64,
    binding: u64,
    ready,
    focus: struct { pending: Pending, popup: Popup },
    selection: struct { pending: Pending, popup: ?Popup },
    restoring: Pending,
};
const Transfer = struct { fd: i32, source: u1, offset: usize = 0, started_monotonic_ns: u64, expires_ns: u64 };

connection: wire.Connection = .{},
epoll_fd: i32 = -1,
tag: u64 = 0,
objects: [128]?Object = @splat(null),
compositor: Global = .{},
shm: Global = .{},
seat: Global = .{},
core: Global = .{},
wlr: Global = .{},
ext: Global = .{},
shell: Global = .{},
gtk: Global = .{},
mode: Mode = .core,
protocol_policy: ProtocolPolicy = .best,
phase: Operation = .{ .discovering = 0 },
keyboard_capable: bool = false,
keyboard: u32 = 0,
device: u32 = 0,
pixel: u32 = 0,
sources: [2]Source = @splat(.empty),
transfers: [8]?Transfer = @splat(null),
problem: ?Error = null,
last_object: u32 = 0,
last_opcode: u16 = 0,
transfers_completed: u64 = 0,
transfers_expired: u64 = 0,
transfers_rejected: u64 = 0,

/// Initializes fresh storage or a deinitialized client. Call deinit even
/// when initialization fails; existing connections must be closed first.
pub fn init(self: *WaylandClipboard, epoll_fd: i32, tag: u64, environment: Environment, protocol_policy: ProtocolPolicy, now_ns: u64) Result(void) {
    // PERFORMANCE: Initialize fields, not a buffer-containing WaylandClipboard value.
    // Whole-value construction can emit a large read-only memcpy template,
    // including undefined wire buffers. Declaration defaults keep metadata
    // complete when fields are added; the connection leaves storage alone.
    inline for (std.meta.fields(WaylandClipboard)) |field| {
        if (comptime std.mem.eql(u8, field.name, "connection")) {
            self.connection.initEmpty();
        } else {
            @field(self, field.name) = comptime field.defaultValue() orelse @compileError("WaylandClipboard metadata requires a default: " ++ field.name);
        }
    }

    self.epoll_fd = epoll_fd;
    self.tag = tag;
    self.protocol_policy = protocol_policy;
    self.phase = .{ .discovering = now_ns + timeout_ns };
    self.start(environment) catch |err| return .{ .err = self.failure(err) };
    return .{ .ok = {} };
}
pub fn deinit(self: *WaylandClipboard) void {
    for (0..self.transfers.len) |index| self.closeTransfer(index);
    if (self.connection.fd >= 0) _ = linux.epoll_ctl(self.epoll_fd, linux.EPOLL.CTL_DEL, self.connection.fd, null);
    self.connection.deinit();
    self.sources = @splat(.empty);
}
/// Busy and invalid_text leave the current clipboard untouched. Other errors
/// terminate the connection: the caller must deinit before reconnecting.
pub fn publish(self: *WaylandClipboard, publication: PublicationId, text: []const u8, now_ns: u64) Result(void) {
    if (self.phase != .ready or self.isBorrowed(publication.storage_index)) return .{ .err = .busy };
    if (text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return .{ .err = .invalid_text };
    const slot: u1 = for (self.sources, 0..) |source, index| {
        if (source == .empty) break @intCast(index);
    } else return .{ .err = .busy };
    self.publishInternal(slot, publication, text, now_ns) catch |err| return .{ .err = self.failure(err) };
    return .{ .ok = {} };
}
pub fn advance(self: *WaylandClipboard, now_ns: u64) Result(Event) {
    const event = self.advanceInternal(now_ns) catch |err| return .{ .err = self.failure(err) };
    return .{ .ok = event };
}
pub fn isBorrowed(self: *const WaylandClipboard, storage_index: StorageIndex) bool {
    for (self.sources) |source| if (source.text()) |text| if (text.publication.storage_index == storage_index) return true;
    return false;
}
pub fn owns(self: *const WaylandClipboard, publication: PublicationId) bool {
    for (self.sources) |source| if (source == .offered and source.offered.text.publication.eql(publication)) return true;
    return false;
}
pub fn deadline(self: *const WaylandClipboard) u64 {
    var next = self.operationDeadline();
    for (self.transfers) |transfer| if (transfer) |t| {
        next = @min(next, t.expires_ns);
    };
    return next;
}

fn start(self: *WaylandClipboard, environment: Environment) !void {
    const display = environment.display;
    var path: [108]u8 = undefined;
    const socket = if (std.fs.path.isAbsolute(display)) display else try std.fmt.bufPrint(&path, "{s}/{s}", .{ environment.runtime_directory orelse {
        self.problem = .{ .unsupported = .display };
        return error.Unsupported;
    }, display });
    try self.connection.connect(socket);
    try self.watch(self.connection.fd, linux.EPOLL.CTL_ADD, linux.EPOLL.IN | linux.EPOLL.OUT);
    self.objects[1] = .{ .kind = .display };
    const registry = try self.allocate(.registry);
    std.debug.assert(registry == 2);
    try self.words(1, 1, &.{registry});
    try self.sync(.discovery);
}
fn publishInternal(self: *WaylandClipboard, slot: u1, publication: PublicationId, text: []const u8, now_ns: u64) !void {
    const object = try self.allocate(.source);
    self.sources[slot] = .{ .offered = .{ .object = object, .text = .{ .bytes = text, .publication = publication } } };
    const pending: Pending = .{ .slot = slot, .deadline = now_ns + timeout_ns };
    try self.words(self.manager().object, 0, &.{object});
    for (mime_types) |mime| {
        var w: wire.Writer = .{};
        try w.string(mime);
        try self.connection.send(object, 0, w.data(), null);
    }
    if (self.mode == .core) {
        self.phase = .{ .focus = .{ .pending = pending, .popup = try self.createPopup() } };
    } else {
        self.phase = .{ .selection = .{ .pending = pending, .popup = null } };
        try self.setSelection(0);
    }
    try self.watch(self.connection.fd, linux.EPOLL.CTL_MOD, linux.EPOLL.IN | linux.EPOLL.OUT);
}
fn advanceInternal(self: *WaylandClipboard, now_ns: u64) !Event {
    if (self.problem != null) return error.Failed;
    if (self.phase != .ready and now_ns >= self.operationDeadline()) {
        self.problem = .{ .timed_out = std.meta.activeTag(self.phase) };
        return error.TimedOut;
    }
    var event: Event = .none;
    // A noisy compositor cannot monopolize the supervisor. receive() also
    // consumes partial frames; never block waiting for the rest of one.
    for (0..64) |_| {
        const msg = try self.connection.receive() orelse {
            if (self.connection.errno == .AGAIN or self.connection.errno == .INTR) break;
            continue;
        };
        self.last_object = msg.object;
        self.last_opcode = msg.opcode;
        var r = msg.body;
        try self.dispatch(msg.object, msg.opcode, &r, now_ns, &event);
        try r.end();
        self.connection.input_size = 0;
        if (event != .none) break;
    }
    for (&self.transfers, 0..) |*entry, index| {
        // Return every completed transfer exactly once. Deferring later writes
        // preserves their observations for the next event-loop iteration.
        if (event != .none) break;
        const transfer = if (entry.*) |*t| t else continue;
        if (now_ns >= transfer.expires_ns) {
            self.transfers_expired += 1;
            self.closeTransfer(index);
            continue;
        }
        const text = self.sources[transfer.source].text().?;
        const bytes = text.bytes[transfer.offset..];
        const written = linux.write(transfer.fd, bytes.ptr, @min(bytes.len, 64 * 1024));
        switch (linux.errno(written)) {
            .AGAIN, .INTR => continue,
            .SUCCESS => {
                if (written == 0) {
                    self.closeTransfer(index);
                    continue;
                }
                transfer.offset += written;
                if (transfer.offset == text.bytes.len) {
                    event = .{ .text_transferred = .{
                        .publication = text.publication,
                        .text_size = text.bytes.len,
                        .started_monotonic_ns = transfer.started_monotonic_ns,
                        .completed_monotonic_ns = now_ns,
                    } };
                    self.transfers_completed += 1;
                    self.closeTransfer(index);
                }
            },
            else => self.closeTransfer(index), // A paste reader may close early.
        }
    }
    for (&self.sources, 0..) |*source, index| {
        if (source.* != .retired) continue;
        const borrowed = for (self.transfers) |entry| {
            if (entry != null and entry.?.source == index) break true;
        } else false;
        if (!borrowed) source.* = .empty;
    }
    try self.connection.flush();
    try self.watch(self.connection.fd, linux.EPOLL.CTL_MOD, linux.EPOLL.IN | (if (self.connection.output_size > 0) @as(u32, linux.EPOLL.OUT) else 0));
    return event;
}
fn dispatch(self: *WaylandClipboard, object: u32, opcode: u16, r: *wire.Reader, now_ns: u64, event: *Event) !void {
    // Server-created offers use the server ID namespace. We immediately
    // destroy them: this write-only client never reads another selection.
    // Already-queued MIME events may still follow that destruction.
    if (object >= 0xff000000) {
        if (opcode != 0) return error.InvalidMessage;
        _ = try r.string();
        return;
    }
    if (object >= self.objects.len) return error.InvalidMessage;
    const entry = self.objects[object] orelse return error.InvalidMessage;
    switch (entry.kind) {
        .display => switch (opcode) {
            0 => {
                const id = try r.word();
                const code = try r.word();
                const message = try r.string();
                var detail: @FieldType(Error, "server") = .{ .object = id, .code = code, .message = @splat(0), .message_size = @min(message.len, 256), .truncated = message.len > 256 };
                @memcpy(detail.message[0..detail.message_size], message[0..detail.message_size]);
                self.problem = .{ .server = detail };
                return error.Server;
            },
            1 => {
                const id = try r.word();
                if (id < 2 or id >= self.objects.len or self.objects[id] == null) return error.InvalidMessage;
                self.objects[id] = null;
            },
            else => return error.InvalidMessage,
        },
        .registry => switch (opcode) {
            0 => {
                const name = try r.word();
                const interface = try r.string();
                const version = try r.word();
                const globals = .{ .{ "wl_compositor", &self.compositor }, .{ "wl_shm", &self.shm }, .{ "wl_seat", &self.seat }, .{ "wl_data_device_manager", &self.core }, .{ "zwlr_data_control_manager_v1", &self.wlr }, .{ "ext_data_control_manager_v1", &self.ext }, .{ "xdg_wm_base", &self.shell }, .{ "gtk_shell1", &self.gtk } };
                inline for (globals) |global| if (std.mem.eql(u8, interface, global[0]) and global[1].name == 0) {
                    global[1].* = .{ .name = name, .version = version };
                };
            },
            1 => {
                const name = try r.word();
                if (name == self.seat.name or name == self.manager().name) return error.GlobalRemoved;
            },
            else => return error.InvalidMessage,
        },
        .discovery, .binding, .selection, .restoring => {
            if (opcode != 0) return error.InvalidMessage;
            _ = try r.word();
            switch (entry.kind) {
                .discovery => try self.bindGlobals(),
                .binding => {
                    if (self.mode == .core and !self.keyboard_capable) {
                        self.problem = .{ .unsupported = .keyboard };
                        return error.Unsupported;
                    }
                    // get_keyboard is a protocol error before the seat has
                    // advertised keyboard capability. Registry discovery
                    // alone does not establish that capability.
                    if (self.mode == .core) {
                        self.keyboard = try self.allocate(.keyboard);
                        try self.words(self.seat.object, 1, &.{self.keyboard});
                    }
                    self.phase = .ready;
                    event.* = .ready;
                },
                .selection => {
                    if (self.phase != .selection) return error.InvalidMessage;
                    if (self.phase.selection.popup) |popup| {
                        try self.destroyPopup(popup);
                        const pending = self.phase.selection.pending;
                        self.phase = .{ .restoring = pending };
                        try self.sync(.restoring);
                    } else try self.acquired(event);
                },
                .restoring => try self.acquired(event),
                else => unreachable,
            }
        },
        .seat => {
            if (opcode != 0) return error.InvalidMessage;
            self.keyboard_capable = try r.word() & 2 != 0;
        },
        .keyboard => switch (opcode) {
            0 => {
                _ = try r.word();
                _ = try r.word();
                _ = linux.close(try self.connection.takeFd());
            },
            1 => {
                const serial = try r.word();
                const surface = try r.word();
                _ = try r.array();
                if (self.phase == .focus and surface == self.phase.focus.popup.surface) try self.setSelection(serial);
            },
            2 => {
                _ = try r.word();
                _ = try r.word();
            },
            3 => {
                for (0..4) |_| _ = try r.word();
            },
            4 => {
                for (0..5) |_| _ = try r.word();
            },
            else => return error.InvalidMessage,
        },
        .device => {
            if (opcode == 0) {
                const offer = try r.word();
                if (offer < 0xff000000) return error.InvalidMessage;
                try self.words(offer, if (self.mode == .core) 2 else 1, &.{});
            } else if (self.mode == .core) switch (opcode) {
                1 => {
                    for (0..5) |_| _ = try r.word();
                },
                2, 4 => {},
                3 => {
                    for (0..3) |_| _ = try r.word();
                },
                5 => {
                    _ = try r.word();
                },
                else => return error.InvalidMessage,
            } else switch (opcode) {
                1, 3 => {
                    _ = try r.word();
                },
                2 => return error.SeatFinished,
                else => return error.InvalidMessage,
            }
        },
        .source => {
            const send_opcode: u16 = if (self.mode == .core) 1 else 0;
            const cancel_opcode: u16 = if (self.mode == .core) 2 else 1;
            if (opcode == send_opcode) {
                const mime = try r.string();
                const fd = try self.connection.takeFd();
                var retained = false;
                defer if (!retained) {
                    _ = linux.close(fd);
                };
                const supported = for (mime_types) |known| {
                    if (std.mem.eql(u8, mime, known)) break true;
                } else false;
                const source: ?u1 = for (self.sources, 0..) |s, index| {
                    if (s == .offered and s.offered.object == object) break @intCast(index);
                } else null;
                if (!supported or source == null or entry.destroyed) return;
                const slot = for (&self.transfers) |*t| {
                    if (t.* == null) break t;
                } else {
                    self.transfers_rejected += 1;
                    return;
                };
                _ = try self.connection.check(linux.fcntl(fd, linux.F.SETFL, @as(u32, @bitCast(linux.O{ .NONBLOCK = true }))));
                try self.watch(fd, linux.EPOLL.CTL_ADD, linux.EPOLL.OUT);
                slot.* = .{ .fd = fd, .source = source.?, .started_monotonic_ns = now_ns, .expires_ns = now_ns + timeout_ns };
                retained = true;
            } else if (opcode == cancel_opcode) {
                for (&self.sources, 0..) |*s, index| if (s.* == .offered and s.offered.object == object) {
                    try self.destroy(object, 1);
                    // Copy before changing the union tag: Zig may write the
                    // destination tag before evaluating an aggregate field.
                    const text = s.offered.text;
                    s.* = .{ .retired = text };
                    if (self.pendingCopy()) |pending| if (pending.slot == index) {
                        self.problem = .selection_lost;
                        return error.SelectionLost;
                    };
                };
            } else return error.InvalidMessage;
        },
        .shell => {
            if (opcode != 0) return error.InvalidMessage;
            try self.words(object, 3, &.{try r.word()});
        },
        .xdg_surface => {
            if (opcode != 0) return error.InvalidMessage;
            const serial = try r.word();
            if (!entry.destroyed) {
                try self.words(object, 4, &.{serial});
                if (self.popupView()) |popup| if (popup.xdg == object and !popup.attached) {
                    try self.words(popup.surface, 1, &.{ self.pixel, 0, 0 });
                    try self.words(popup.surface, 2, &.{ 0, 0, 1, 1 });
                    if (popup.gtk != 0) try self.words(popup.gtk, 3, &.{0});
                    try self.words(popup.surface, 6, &.{});
                    popup.attached = true;
                };
            }
        },
        .toplevel => switch (opcode) {
            0 => {
                _ = try r.word();
                _ = try r.word();
                _ = try r.array();
            },
            1 => if (!entry.destroyed) return error.SurfaceClosed,
            else => return error.InvalidMessage,
        },
        .gtk_shell => {
            if (opcode != 0) return error.InvalidMessage;
            _ = try r.word();
        },
        .gtk_surface => {
            if (opcode > 1) return error.InvalidMessage;
            _ = try r.array();
        },
        .surface => {
            if (opcode > 1) return error.InvalidMessage;
            _ = try r.word();
        },
        .buffer => {
            if (opcode != 0) return error.InvalidMessage;
        },
        .shm => {
            if (opcode != 0) return error.InvalidMessage;
            _ = try r.word();
        },
        else => return error.InvalidMessage,
    }
}
fn bindGlobals(self: *WaylandClipboard) !void {
    if (self.seat.name == 0) {
        self.problem = .{ .unsupported = .seat };
        return error.Unsupported;
    }
    self.mode = switch (self.protocol_policy) {
        .best => if (self.ext.name != 0) .ext else if (self.wlr.name != 0) .wlr else .core,
        .data_control_only => if (self.ext.name != 0) .ext else if (self.wlr.name != 0) .wlr else {
            self.problem = .{ .unsupported = .data_control };
            return error.Unsupported;
        },
        .core_only => .core,
    };
    if (self.manager().name == 0) {
        self.problem = .{ .unsupported = .clipboard };
        return error.Unsupported;
    }
    try self.bind(&self.seat, "wl_seat", 1, .seat);
    switch (self.mode) {
        .core => try self.bind(&self.core, "wl_data_device_manager", 1, .manager),
        .wlr => try self.bind(&self.wlr, "zwlr_data_control_manager_v1", 1, .manager),
        .ext => try self.bind(&self.ext, "ext_data_control_manager_v1", 1, .manager),
    }
    self.device = try self.allocate(.device);
    try self.words(self.manager().object, 1, &.{ self.device, self.seat.object });
    if (self.mode == .core) {
        if (self.compositor.name == 0 or self.shm.name == 0 or self.shell.name == 0) {
            self.problem = .{ .unsupported = .surface };
            return error.Unsupported;
        }
        try self.bind(&self.compositor, "wl_compositor", 1, .compositor);
        try self.bind(&self.shm, "wl_shm", 1, .shm);
        try self.bind(&self.shell, "xdg_wm_base", 1, .shell);
        // gtk_surface.release arrived in v4. Binding only when available
        // prevents leaking a GTK extension object on every publication.
        if (self.gtk.version >= 4) try self.bind(&self.gtk, "gtk_shell1", 4, .gtk_shell);
        self.pixel = try self.createPixel();
    }
    const deadline_ns = self.operationDeadline();
    self.phase = .{ .binding = deadline_ns };
    try self.sync(.binding);
}
fn createPopup(self: *WaylandClipboard) !Popup {
    const surface = try self.allocate(.surface);
    const xdg = try self.allocate(.xdg_surface);
    const toplevel = try self.allocate(.toplevel);
    const gtk = if (self.gtk.object != 0) try self.allocate(.gtk_surface) else 0;
    try self.words(self.compositor.object, 0, &.{surface});
    try self.words(self.shell.object, 2, &.{ xdg, surface });
    try self.words(xdg, 1, &.{toplevel});
    if (gtk != 0) try self.words(self.gtk.object, 0, &.{ gtk, surface });
    try self.words(surface, 6, &.{}); // Configure before attaching a buffer.
    return .{ .surface = surface, .xdg = xdg, .toplevel = toplevel, .gtk = gtk };
}
// One immutable transparent pixel serves every temporary surface. The
// compositor may retain old attachments; no code ever writes this storage.
fn createPixel(self: *WaylandClipboard) !u32 {
    const pool = try self.allocate(.pool);
    const buffer = try self.allocate(.buffer);
    const fd: i32 = @intCast(try self.connection.check(linux.memfd_create("voiced-clipboard-surface", linux.MFD.CLOEXEC)));
    var owned = true;
    defer if (owned) {
        _ = linux.close(fd);
    };
    _ = try self.connection.check(linux.ftruncate(fd, 4));
    var w: wire.Writer = .{};
    try w.word(pool);
    try w.word(4);
    try self.connection.send(self.shm.object, 0, w.data(), fd);
    owned = false;
    // A zero-filled ARGB pixel is transparent. The pool can be destroyed
    // immediately: the wl_buffer retains the compositor's backing storage.
    try self.words(pool, 0, &.{ buffer, 0, 1, 1, 4, 0 });
    try self.destroy(pool, 1);
    return buffer;
}
fn destroyPopup(self: *WaylandClipboard, popup: Popup) !void {
    if (popup.gtk != 0) try self.destroy(popup.gtk, 5);
    try self.destroy(popup.toplevel, 0);
    try self.destroy(popup.xdg, 0);
    try self.destroy(popup.surface, 0);
}
fn setSelection(self: *WaylandClipboard, serial: u32) !void {
    const pending = self.pendingCopy() orelse return error.InvalidMessage;
    const source = self.sources[pending.slot].offered.object;
    if (self.mode == .core) try self.words(self.device, 1, &.{ source, serial }) else try self.words(self.device, 0, &.{source});
    const popup = if (self.phase == .focus) self.phase.focus.popup else null;
    self.phase = .{ .selection = .{ .pending = pending, .popup = popup } };
    try self.sync(.selection);
}
fn acquired(self: *WaylandClipboard, event: *Event) !void {
    const slot = (self.pendingCopy() orelse return error.InvalidMessage).slot;
    if (self.sources[slot] != .offered) {
        self.problem = .selection_lost;
        return error.SelectionLost;
    }
    self.phase = .ready;
    event.* = .{ .acquired = self.sources[slot].offered.text.publication };
}
fn operationDeadline(self: *const WaylandClipboard) u64 {
    return switch (self.phase) {
        .ready => std.math.maxInt(u64),
        .discovering, .binding => |deadline_ns| deadline_ns,
        .focus => |focus| focus.pending.deadline,
        .selection => |selection| selection.pending.deadline,
        .restoring => |pending| pending.deadline,
    };
}
fn pendingCopy(self: *const WaylandClipboard) ?Pending {
    return switch (self.phase) {
        .focus => |focus| focus.pending,
        .selection => |selection| selection.pending,
        .restoring => |value| value,
        else => null,
    };
}
fn popupView(self: *WaylandClipboard) ?*Popup {
    return switch (self.phase) {
        .focus => |*focus| &focus.popup,
        .selection => |*selection| if (selection.popup) |*value| value else null,
        else => null,
    };
}
fn bind(self: *WaylandClipboard, global: *Global, name: []const u8, version: u32, kind: Kind) !void {
    if (global.version == 0) return error.InvalidMessage;
    global.object = try self.allocate(kind);
    var w: wire.Writer = .{};
    try w.word(global.name);
    try w.string(name);
    try w.word(@min(global.version, version));
    try w.word(global.object);
    try self.connection.send(2, 0, w.data(), null);
}
fn manager(self: *WaylandClipboard) *Global {
    return switch (self.mode) {
        .core => &self.core,
        .wlr => &self.wlr,
        .ext => &self.ext,
    };
}
fn allocate(self: *WaylandClipboard, kind: Kind) !u32 {
    for (self.objects[2..], 2..) |entry, id| if (entry == null) {
        self.objects[id] = .{ .kind = kind };
        return @intCast(id);
    };
    return error.ObjectLimit;
}
fn destroy(self: *WaylandClipboard, object: u32, opcode: u16) !void {
    try self.words(object, opcode, &.{});
    self.objects[object].?.destroyed = true;
}
fn sync(self: *WaylandClipboard, kind: Kind) !void {
    try self.words(1, 0, &.{try self.allocate(kind)});
}
fn words(self: *WaylandClipboard, object: u32, opcode: u16, args: []const u32) !void {
    var w: wire.Writer = .{};
    for (args) |arg| try w.word(arg);
    try self.connection.send(object, opcode, w.data(), null);
}
fn watch(self: *WaylandClipboard, fd: i32, operation: u32, events: u32) !void {
    var event: linux.epoll_event = .{ .events = events, .data = .{ .u64 = self.tag } };
    _ = try self.connection.check(linux.epoll_ctl(self.epoll_fd, operation, fd, &event));
}
fn closeTransfer(self: *WaylandClipboard, index: usize) void {
    if (self.transfers[index]) |t| {
        _ = linux.epoll_ctl(self.epoll_fd, linux.EPOLL.CTL_DEL, t.fd, null);
        _ = linux.close(t.fd);
        self.transfers[index] = null;
    }
}
fn failure(self: *WaylandClipboard, cause: anyerror) Error {
    const problem = self.problem orelse Error{ .transport = .{ .cause = cause, .errno = self.connection.errno, .object = self.last_object, .opcode = self.last_opcode } };
    self.problem = problem;
    return problem;
}
