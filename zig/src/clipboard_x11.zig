//! Native X11 CLIPBOARD owner for an epoll caller. Text is borrowed while it is
//! offered or while a bounded ICCCM INCR transfer survives replacement. The
//! client uses only the core protocol and local MIT-MAGIC-COOKIE authentication.
const std = @import("std");
const linux = std.os.linux;
const wire = @import("x11_wire.zig");
const endian = @import("builtin").cpu.arch.endian();

const timeout_ns = 2 * std.time.ns_per_s;
const text_bytes_direct_max = 8 * 1024;
// A maximal INCR chunk may follow its SelectionNotify in the same dispatch.
const output_dispatch_size_max = text_bytes_direct_max + 68;
const transfers_count_max = 8;
const atom_names = [_][]const u8{
    "CLIPBOARD",
    "TARGETS",
    "TIMESTAMP",
    "UTF8_STRING",
    "TEXT",
    "text/plain",
    "text/plain;charset=utf-8",
    "INCR",
    "_VOICED_TIMESTAMP",
};

const Atom = enum(u4) {
    clipboard,
    targets,
    timestamp,
    utf8_string,
    text,
    text_plain,
    text_plain_utf8,
    incr,
    voiced_timestamp,
};

pub const Error = union(enum) {
    transport: struct { cause: anyerror, errno: linux.E, response_type: u8, sequence: u16 },
    setup: struct { status: u8, reason: [128]u8, reason_size: u8, truncated: bool },
    server: struct { code: u8, sequence: u16, major_opcode: u8, minor_opcode: u16, bad_value: u32 },
    authority: struct { cause: anyerror, errno: linux.E },
    unsupported: enum { display },
    timed_out: TimeoutPhase,
    selection_lost,
    busy,
    invalid_text,
};

pub fn Result(comptime T: type) type {
    return union(enum) { ok: T, err: Error };
}

pub const Phase = enum { setup, interning, ready, timestamp, acquiring };
pub const TimeoutPhase = enum { setup, interning, timestamp, acquiring, transfer };
pub const Event = union(enum) { none, ready, acquired: u64 };
pub const Environment = struct { display: ?[]const u8, authority: ?[]const u8, home: ?[]const u8 };

const Text = struct { bytes: []const u8, id: u64 };
const Source = union(enum) {
    empty,
    pending: Text,
    offered: struct { text: Text, timestamp: u32 },
    retired: Text,

    fn text(self: Source) ?Text {
        return switch (self) {
            .empty => null,
            .pending => |value| value,
            .offered => |value| value.text,
            .retired => |value| value,
        };
    }
};

const Pending = struct { slot: u1, deadline_ns: u64 };
const Operation = union(Phase) {
    setup: u64,
    interning: struct { remaining: u8, deadline_ns: u64 },
    ready,
    timestamp: struct { pending: Pending, sequence: u16 },
    acquiring: struct { pending: Pending, sequence: u16 },
};

const SelectionRequest = struct {
    time: u32,
    owner: u32,
    requestor: u32,
    selection: u32,
    target: u32,
    property: u32,
};

const TransferPhase = enum(u2) {
    direct_property,
    incr_announce,
    incr_wait_delete,
    incr_wait_final_delete,
};

const Transfer = struct {
    deadline_ns: u64,
    requestor: u32,
    property: u32,
    target: u32,
    property_type: u32 = 0,
    time: u32,
    offset: u32 = 0,
    sequence_start: u16,
    barrier_sequence: u16,
    phase: TransferPhase,
    source: u1 = 0,
    failed: bool = false,
    delete_pending: bool = false,
};

pub const Client = struct {
    connection: wire.Connection = .{},
    epoll_fd: i32 = -1,
    tag: u64 = 0,
    phase: Operation = .{ .setup = 0 },
    owner_window: u32 = 0,
    atoms: [atom_names.len]u32 = @splat(0),
    atom_sequences: [atom_names.len]u16 = @splat(0),
    sources: [2]Source = @splat(.empty),
    transfers: [transfers_count_max]?Transfer = @splat(null),
    problem: ?Error = null,
    last_response_type: u8 = 0,
    last_sequence: u16 = 0,
    transfers_completed: u64 = 0,
    transfers_expired: u64 = 0,
    transfers_rejected: u64 = 0,

    /// Initializes fresh storage or a deinitialized client. Call deinit even
    /// when initialization fails; existing connections must be closed first.
    pub fn init(self: *Client, epoll_fd: i32, tag: u64, environment: Environment, now_ns: u64) Result(void) {
        inline for (std.meta.fields(Client)) |field| {
            if (comptime std.mem.eql(u8, field.name, "connection")) {
                self.connection.initEmpty();
            } else {
                @field(self, field.name) = comptime field.defaultValue() orelse @compileError("Client metadata requires a default: " ++ field.name);
            }
        }
        self.epoll_fd = epoll_fd;
        self.tag = tag;
        self.phase = .{ .setup = now_ns + timeout_ns };
        self.start(environment) catch |err| return .{ .err = self.failure(err) };
        return .{ .ok = {} };
    }

    pub fn deinit(self: *Client) void {
        if (self.connection.fd >= 0) _ = linux.epoll_ctl(self.epoll_fd, linux.EPOLL.CTL_DEL, self.connection.fd, null);
        self.connection.deinit();
        self.sources = @splat(.empty);
        self.transfers = @splat(null);
    }

    /// Busy and invalid_text leave the current clipboard untouched. Other
    /// errors terminate the connection: the caller must deinit before reuse.
    pub fn publish(self: *Client, id: u64, text: []const u8, now_ns: u64) Result(void) {
        if (self.phase != .ready or self.isBorrowed(id)) return .{ .err = .busy };
        if (text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return .{ .err = .invalid_text };
        if (text.len > std.math.maxInt(u32)) return .{ .err = .invalid_text };
        const slot: u1 = for (self.sources, 0..) |source, index| {
            if (source == .empty) break @intCast(index);
        } else return .{ .err = .busy };
        self.sources[slot] = .{ .pending = .{ .bytes = text, .id = id } };
        const pending: Pending = .{ .slot = slot, .deadline_ns = now_ns + timeout_ns };
        const sequence = self.changeProperty(self.owner_window, self.atom(.voiced_timestamp), 31, 8, "") catch |err| return .{ .err = self.failure(err) };
        self.phase = .{ .timestamp = .{ .pending = pending, .sequence = sequence } };
        self.watch(linux.EPOLL.CTL_MOD, linux.EPOLL.IN | linux.EPOLL.OUT) catch |err| return .{ .err = self.failure(err) };
        return .{ .ok = {} };
    }

    pub fn advance(self: *Client, now_ns: u64) Result(Event) {
        const event = self.advanceInternal(now_ns) catch |err| return .{ .err = self.failure(err) };
        return .{ .ok = event };
    }

    pub fn isBorrowed(self: *const Client, id: u64) bool {
        for (self.sources) |source| if (source.text()) |text| if (text.id == id) return true;
        return false;
    }

    pub fn owns(self: *const Client, id: u64) bool {
        for (self.sources) |source| if (source == .offered and source.offered.text.id == id) return true;
        return false;
    }

    pub fn ready(self: *const Client) bool {
        return self.phase == .ready;
    }

    pub fn deadline(self: *const Client) u64 {
        var next = self.operationDeadline();
        for (self.transfers) |transfer| if (transfer) |value| {
            next = @min(next, value.deadline_ns);
        };
        return next;
    }

    fn start(self: *Client, environment: Environment) !void {
        const display = environment.display orelse {
            self.problem = .{ .unsupported = .display };
            return error.Unsupported;
        };
        self.connection.connectDisplay(display, environment.authority, environment.home) catch |err| switch (err) {
            error.UnsupportedDisplay => {
                self.problem = .{ .unsupported = .display };
                return error.Unsupported;
            },
            error.InvalidAuthority, error.AuthorityMissing, error.AuthoritySystem => {
                self.problem = .{ .authority = .{ .cause = err, .errno = self.connection.errno } };
                return error.Authorization;
            },
            else => return err,
        };
        try self.watch(linux.EPOLL.CTL_ADD, linux.EPOLL.IN | linux.EPOLL.OUT);
    }

    fn advanceInternal(self: *Client, now_ns: u64) !Event {
        if (self.problem != null) return error.Failed;
        if (self.phase != .ready and now_ns >= self.operationDeadline()) {
            self.problem = .{ .timed_out = switch (self.phase) {
                .setup => .setup,
                .interning => .interning,
                .timestamp => .timestamp,
                .acquiring => .acquiring,
                .ready => unreachable,
            } };
            return error.TimedOut;
        }
        try self.connection.flush();
        for (&self.transfers, 0..) |*entry, index| if (entry.*) |transfer| {
            if (now_ns < transfer.deadline_ns) continue;
            switch (transfer.phase) {
                .incr_wait_delete, .incr_wait_final_delete => {
                    self.transfers_expired += 1;
                    try self.beginCleanup(index, true);
                },
                else => {
                    self.problem = .{ .timed_out = .transfer };
                    return error.TimedOut;
                },
            }
        };

        // Accept work only when the largest dispatch fits. Input remains in
        // the socket while output is blocked.
        var event: Event = .none;
        for (0..64) |_| {
            if (self.connection.output.len - self.connection.output_size < output_dispatch_size_max) break;
            if (self.phase == .setup) {
                const setup = try self.connection.receiveSetup() orelse break;
                try self.acceptSetup(setup, now_ns);
                continue;
            }
            const message = try self.connection.receive() orelse break;
            self.last_response_type = message.responseType();
            self.last_sequence = message.sequence();
            try self.dispatch(message, now_ns, &event);
            const event_type = message.responseType();
            self.connection.consume();
            if (event != .none or event_type == 30 or self.connection.output_size >= text_bytes_direct_max) break;
        }

        self.releaseRetiredSources();
        try self.connection.flush();
        try self.watch(linux.EPOLL.CTL_MOD, linux.EPOLL.IN | (if (self.connection.output_size > 0) @as(u32, linux.EPOLL.OUT) else 0));
        return event;
    }

    fn acceptSetup(self: *Client, setup: wire.Setup, now_ns: u64) !void {
        switch (setup) {
            .rejected => |rejected| {
                self.problem = .{ .setup = .{ .status = rejected.status, .reason = rejected.reason, .reason_size = rejected.reason_size, .truncated = rejected.truncated } };
                return error.SetupRejected;
            },
            .success => |information| {
                if (information.resource_id_base == 0 or information.resource_id_mask == 0) return error.InvalidMessage;
                self.owner_window = information.resource_id_base;
                var body: [32]u8 = @splat(0);
                put32(&body, 0, self.owner_window);
                put32(&body, 4, information.root);
                put16(&body, 12, 1);
                put16(&body, 14, 1);
                put16(&body, 18, 2); // InputOnly
                put32(&body, 24, 1 << 11); // CWEventMask
                put32(&body, 28, 1 << 22); // PropertyChange
                _ = try self.connection.send(1, 0, &body);

                inline for (atom_names, 0..) |name, index| {
                    var fixed: [4]u8 = @splat(0);
                    put16(&fixed, 0, name.len);
                    self.atom_sequences[index] = try self.connection.sendParts(16, 0, &fixed, name);
                }
                self.phase = .{ .interning = .{ .remaining = atom_names.len, .deadline_ns = now_ns + timeout_ns } };
            },
        }
    }

    fn dispatch(self: *Client, message: wire.Message, now_ns: u64, event: *Event) !void {
        switch (message.responseType()) {
            0 => try self.handleServerError(message),
            1 => try self.handleReply(message, now_ns, event),
            28 => try self.handlePropertyNotify(message, now_ns),
            29 => try self.handleSelectionClear(message),
            30 => try self.handleSelectionRequest(message, now_ns),
            else => {}, // MappingNotify and unselected extension events are harmless.
        }
    }

    fn handleReply(self: *Client, message: wire.Message, now_ns: u64, event: *Event) !void {
        const sequence = message.sequence();
        if (self.phase == .interning) {
            for (&self.atom_sequences, 0..) |*expected, index| if (expected.* == sequence) {
                if (message.bytes.len != 32) return error.InvalidMessage;
                const value = word(message.bytes, 8);
                if (value == 0) return error.InvalidMessage;
                self.atoms[index] = value;
                expected.* = 0;
                self.phase.interning.remaining -= 1;
                if (self.phase.interning.remaining == 0) {
                    self.phase = .ready;
                    event.* = .ready;
                }
                return;
            };
        }
        if (self.phase == .acquiring and self.phase.acquiring.sequence == sequence) {
            if (message.bytes.len != 32) return error.InvalidMessage;
            if (word(message.bytes, 8) != self.owner_window) {
                self.problem = .selection_lost;
                return error.SelectionLost;
            }
            const slot = self.phase.acquiring.pending.slot;
            if (self.sources[slot] != .offered) return error.InvalidMessage;
            const id = self.sources[slot].offered.text.id;
            self.phase = .ready;
            event.* = .{ .acquired = id };
            return;
        }
        for (&self.transfers, 0..) |*entry, index| if (entry.*) |*transfer| {
            if ((transfer.phase != .direct_property and transfer.phase != .incr_announce) or transfer.barrier_sequence != sequence) continue;
            if (message.bytes.len != 32) return error.InvalidMessage;
            try self.finishTransferBarrier(index, now_ns);
            return;
        };
        return error.InvalidMessage;
    }

    fn handleServerError(self: *Client, message: wire.Message) !void {
        if (message.bytes.len != 32) return error.InvalidMessage;
        const sequence = message.sequence();
        for (&self.transfers, 0..) |*entry, index| if (entry.*) |*transfer| {
            switch (transfer.phase) {
                .direct_property, .incr_announce => if (sequenceInRange(sequence, transfer.sequence_start, transfer.barrier_sequence)) {
                    transfer.failed = true;
                    return;
                },
                .incr_wait_delete, .incr_wait_final_delete => if (sequence == transfer.sequence_start) {
                    self.transfers_rejected += 1;
                    try self.beginCleanup(index, true);
                    return;
                },
            }
        };
        // Requestor notification and cleanup requests are intentionally not
        // followed by barriers. Their errors cannot invalidate our owner window.
        const opcode = message.bytes[10];
        if (opcode == 2 or opcode == 19 or opcode == 25) return;
        if (opcode == 18 and !(self.phase == .timestamp and sequence == self.phase.timestamp.sequence)) return;
        self.problem = .{ .server = .{
            .code = message.bytes[1],
            .sequence = sequence,
            .major_opcode = message.bytes[10],
            .minor_opcode = std.mem.readInt(u16, message.bytes[8..10], endian),
            .bad_value = word(message.bytes, 4),
        } };
        return error.Server;
    }

    fn handlePropertyNotify(self: *Client, message: wire.Message, now_ns: u64) !void {
        if (message.bytes.len != 32) return error.InvalidMessage;
        const window = word(message.bytes, 4);
        const property = word(message.bytes, 8);
        const time = word(message.bytes, 12);
        const state = message.bytes[16];
        if (self.phase == .timestamp and message.sequence() == self.phase.timestamp.sequence and window == self.owner_window and property == self.atom(.voiced_timestamp) and state == 0) {
            const pending = self.phase.timestamp.pending;
            for (&self.sources) |*source| if (source.* == .offered) {
                const text = source.offered.text;
                source.* = .{ .retired = text };
            };
            if (self.sources[pending.slot] != .pending) return error.InvalidMessage;
            const text = self.sources[pending.slot].pending;
            self.sources[pending.slot] = .{ .offered = .{ .text = text, .timestamp = time } };
            try self.setSelectionOwner(time);
            const sequence = try self.getSelectionOwner();
            self.phase = .{ .acquiring = .{ .pending = pending, .sequence = sequence } };
            return;
        }
        if (state != 1) return;
        for (&self.transfers, 0..) |*entry, index| if (entry.*) |*transfer| {
            if (transfer.requestor != window or transfer.property != property) continue;
            switch (transfer.phase) {
                .incr_wait_delete => try self.writeIncrement(index, now_ns),
                .incr_wait_final_delete => {
                    self.transfers_completed += 1;
                    try self.beginCleanup(index, false);
                },
                .incr_announce => transfer.delete_pending = true,
                .direct_property => {},
            }
            return;
        };
    }

    fn handleSelectionClear(self: *Client, message: wire.Message) !void {
        if (message.bytes.len != 32 or word(message.bytes, 8) != self.owner_window or word(message.bytes, 12) != self.atom(.clipboard)) return error.InvalidMessage;
        for (&self.sources) |*source| if (source.* == .offered) {
            const text = source.offered.text;
            source.* = .{ .retired = text };
        };
    }

    fn handleSelectionRequest(self: *Client, message: wire.Message, now_ns: u64) !void {
        if (message.bytes.len != 32) return error.InvalidMessage;
        const request: SelectionRequest = .{
            .time = word(message.bytes, 4),
            .owner = word(message.bytes, 8),
            .requestor = word(message.bytes, 12),
            .selection = word(message.bytes, 16),
            .target = word(message.bytes, 20),
            .property = word(message.bytes, 24),
        };
        const source: ?u1 = source: {
            for (self.sources, 0..) |value, index| if (value == .offered) break :source @intCast(index);
            break :source null;
        };
        if (source == null or request.owner != self.owner_window or request.selection != self.atom(.clipboard) or
            (request.time != 0 and timestampBefore(request.time, self.sources[source.?].offered.timestamp)))
        {
            try self.refuse(request);
            return;
        }
        try self.serveTarget(request, source.?, self.sources[source.?].offered.timestamp, now_ns);
    }

    fn serveTarget(self: *Client, request: SelectionRequest, source: u1, selection_timestamp: u32, now_ns: u64) !void {
        const property = if (request.property == 0) request.target else request.property;
        if (self.propertyInUse(request.requestor, property)) return self.refuse(request);
        if (request.target == self.atom(.targets)) {
            const targets = [_]u32{
                self.atom(.targets),
                self.atom(.timestamp),
                self.atom(.utf8_string),
                self.atom(.text),
                self.atom(.text_plain),
                self.atom(.text_plain_utf8),
            };
            return self.respondDirect(request, property, 4, 32, std.mem.sliceAsBytes(&targets), now_ns);
        }
        if (request.target == self.atom(.timestamp))
            return self.respondDirect(request, property, 19, 32, std.mem.asBytes(&selection_timestamp), now_ns);
        if (request.target == self.atom(.utf8_string) or request.target == self.atom(.text) or request.target == self.atom(.text_plain) or request.target == self.atom(.text_plain_utf8)) {
            const text = (self.sources[source].text() orelse return error.InvalidMessage).bytes;
            const property_type = if (request.target == self.atom(.text)) self.atom(.utf8_string) else request.target;
            return self.respondText(request, property, source, property_type, text, now_ns);
        }
        return self.refuse(request);
    }

    fn respondText(self: *Client, request: SelectionRequest, property: u32, source: u1, property_type: u32, text: []const u8, now_ns: u64) !void {
        if (text.len <= text_bytes_direct_max)
            return self.respondDirect(request, property, property_type, 8, text, now_ns);

        const index = self.allocateTransfer() orelse return self.refuse(request);
        errdefer self.transfers[index] = null;
        const sequence_start = try self.changeWindowAttributes(request.requestor, 1 << 22);
        const size: u32 = @intCast(text.len);
        _ = try self.changeProperty(request.requestor, property, self.atom(.incr), 32, std.mem.asBytes(&size));
        const barrier_sequence = try self.getInputFocus();
        self.transfers[index] = .{
            .deadline_ns = now_ns + timeout_ns,
            .requestor = request.requestor,
            .property = property,
            .target = request.target,
            .property_type = property_type,
            .time = request.time,
            .sequence_start = sequence_start,
            .barrier_sequence = barrier_sequence,
            .phase = .incr_announce,
            .source = source,
        };
    }

    fn respondDirect(self: *Client, request: SelectionRequest, property: u32, property_type: u32, format: u8, bytes: []const u8, now_ns: u64) !void {
        const index = self.allocateTransfer() orelse return self.refuse(request);
        errdefer self.transfers[index] = null;
        const sequence_start = try self.changeProperty(request.requestor, property, property_type, format, bytes);
        const barrier_sequence = try self.getInputFocus();
        self.transfers[index] = .{
            .deadline_ns = now_ns + timeout_ns,
            .requestor = request.requestor,
            .property = property,
            .target = request.target,
            .time = request.time,
            .sequence_start = sequence_start,
            .barrier_sequence = barrier_sequence,
            .phase = .direct_property,
        };
    }

    fn refuse(self: *Client, request: SelectionRequest) !void {
        self.transfers_rejected += 1;
        _ = try self.sendNotification(request, 0);
    }

    fn finishTransferBarrier(self: *Client, index: usize, now_ns: u64) !void {
        const transfer = &self.transfers[index].?;
        const request: SelectionRequest = .{ .time = transfer.time, .owner = self.owner_window, .requestor = transfer.requestor, .selection = self.atom(.clipboard), .target = transfer.target, .property = transfer.property };
        switch (transfer.phase) {
            .direct_property => {
                _ = try self.sendNotification(request, if (transfer.failed) 0 else transfer.property);
                if (transfer.failed) self.transfers_rejected += 1 else self.transfers_completed += 1;
                self.transfers[index] = null;
            },
            .incr_announce => if (transfer.failed) {
                _ = try self.sendNotification(request, 0);
                self.transfers_rejected += 1;
                try self.beginCleanup(index, true);
            } else {
                const delete_pending = transfer.delete_pending;
                transfer.delete_pending = false;
                _ = try self.sendNotification(request, transfer.property);
                transfer.phase = .incr_wait_delete;
                transfer.deadline_ns = now_ns + timeout_ns;
                if (delete_pending) try self.writeIncrement(index, now_ns);
            },
            .incr_wait_delete, .incr_wait_final_delete => return error.InvalidMessage,
        }
    }

    fn writeIncrement(self: *Client, index: usize, now_ns: u64) !void {
        const transfer = &self.transfers[index].?;
        const text = self.sources[transfer.source].text() orelse return error.InvalidMessage;
        const remaining = text.bytes[@as(usize, transfer.offset)..];
        const bytes = remaining[0..@min(remaining.len, text_bytes_direct_max)];
        transfer.sequence_start = try self.changeProperty(transfer.requestor, transfer.property, transfer.property_type, 8, bytes);
        transfer.offset += @intCast(bytes.len);
        transfer.phase = if (remaining.len == 0) .incr_wait_final_delete else .incr_wait_delete;
        transfer.deadline_ns = now_ns + timeout_ns;
    }

    fn beginCleanup(self: *Client, index: usize, abort: bool) !void {
        const requestor = self.transfers[index].?.requestor;
        const property = self.transfers[index].?.property;
        const shared_subscription = requestor == self.owner_window or (for (self.transfers, 0..) |entry, other_index| {
            if (other_index != index and entry != null and entry.?.requestor == requestor and transferHasActiveIncrement(entry.?.phase)) break true;
        } else false);
        if (abort) _ = try self.deleteProperty(requestor, property);
        if (!shared_subscription) _ = try self.changeWindowAttributes(requestor, 0);
        self.transfers[index] = null;
    }

    fn releaseRetiredSources(self: *Client) void {
        for (&self.sources, 0..) |*source, source_index| {
            if (source.* != .retired) continue;
            const borrowed = for (self.transfers) |transfer| {
                if (transfer != null and transfer.?.source == source_index and transferHasActiveIncrement(transfer.?.phase)) break true;
            } else false;
            if (!borrowed) source.* = .empty;
        }
    }

    fn allocateTransfer(self: *Client) ?usize {
        for (&self.transfers, 0..) |*entry, index| if (entry.* == null) return index;
        return null;
    }

    fn propertyInUse(self: *const Client, requestor: u32, property: u32) bool {
        for (self.transfers) |entry| if (entry) |transfer| {
            if (transfer.requestor == requestor and transfer.property == property) return true;
        };
        return false;
    }

    fn atom(self: *const Client, name: Atom) u32 {
        return self.atoms[@intFromEnum(name)];
    }

    fn operationDeadline(self: *const Client) u64 {
        return switch (self.phase) {
            .ready => std.math.maxInt(u64),
            .setup => |deadline_ns| deadline_ns,
            .interning => |state| state.deadline_ns,
            .timestamp => |state| state.pending.deadline_ns,
            .acquiring => |state| state.pending.deadline_ns,
        };
    }

    fn setSelectionOwner(self: *Client, time: u32) !void {
        var body: [12]u8 = undefined;
        put32(&body, 0, self.owner_window);
        put32(&body, 4, self.atom(.clipboard));
        put32(&body, 8, time);
        _ = try self.connection.send(22, 0, &body);
    }

    fn getSelectionOwner(self: *Client) !u16 {
        var body: [4]u8 = undefined;
        put32(&body, 0, self.atom(.clipboard));
        return self.connection.send(23, 0, &body);
    }

    fn getInputFocus(self: *Client) !u16 {
        return self.connection.send(43, 0, "");
    }

    fn changeProperty(self: *Client, window: u32, property: u32, property_type: u32, format: u8, bytes: []const u8) !u16 {
        const element_size: usize = switch (format) {
            8 => 1,
            16 => 2,
            32 => 4,
            else => return error.InvalidMessage,
        };
        if (bytes.len % element_size != 0) return error.InvalidMessage;
        var body: [20]u8 = @splat(0);
        put32(&body, 0, window);
        put32(&body, 4, property);
        put32(&body, 8, property_type);
        body[12] = format;
        put32(&body, 16, @as(u32, @intCast(bytes.len / element_size)));
        return self.connection.sendParts(18, 0, &body, bytes);
    }

    fn deleteProperty(self: *Client, window: u32, property: u32) !u16 {
        var body: [8]u8 = undefined;
        put32(&body, 0, window);
        put32(&body, 4, property);
        return self.connection.send(19, 0, &body);
    }

    fn changeWindowAttributes(self: *Client, window: u32, event_mask: u32) !u16 {
        var body: [12]u8 = undefined;
        put32(&body, 0, window);
        put32(&body, 4, 1 << 11); // CWEventMask
        put32(&body, 8, event_mask);
        return self.connection.send(2, 0, &body);
    }

    fn sendNotification(self: *Client, request: SelectionRequest, property: u32) !u16 {
        var body: [40]u8 = @splat(0);
        put32(&body, 0, request.requestor);
        body[8] = 31; // SelectionNotify
        put32(&body, 12, request.time);
        put32(&body, 16, request.requestor);
        put32(&body, 20, request.selection);
        put32(&body, 24, request.target);
        put32(&body, 28, property);
        return self.connection.send(25, 0, &body);
    }

    fn watch(self: *Client, operation: u32, events: u32) !void {
        var event: linux.epoll_event = .{ .events = events, .data = .{ .u64 = self.tag } };
        _ = try self.connection.check(linux.epoll_ctl(self.epoll_fd, operation, self.connection.fd, &event));
    }

    fn failure(self: *Client, cause: anyerror) Error {
        const problem = self.problem orelse Error{ .transport = .{ .cause = cause, .errno = self.connection.errno, .response_type = self.last_response_type, .sequence = self.last_sequence } };
        self.problem = problem;
        return problem;
    }
};

fn transferHasActiveIncrement(phase: TransferPhase) bool {
    return phase != .direct_property;
}

fn sequenceInRange(sequence: u16, start: u16, end: u16) bool {
    return sequence -% start < end -% start;
}

fn timestampBefore(value: u32, reference: u32) bool {
    return @as(i32, @bitCast(value -% reference)) < 0;
}

fn word(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], endian);
}

fn put16(bytes: []u8, offset: usize, value: anytype) void {
    std.mem.writeInt(u16, bytes[offset..][0..2], @intCast(value), endian);
}

fn put32(bytes: []u8, offset: usize, value: anytype) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @intCast(value), endian);
}

test "sequence and timestamp comparisons wrap" {
    try std.testing.expect(sequenceInRange(0, 0xffff, 1));
    try std.testing.expect(!sequenceInRange(1, 0xffff, 1));
    try std.testing.expect(timestampBefore(0xfffffff0, 0x10));
    try std.testing.expect(!timestampBefore(0x10, 0xfffffff0));
}
