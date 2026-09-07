const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.notifications);
const linux = std.os.linux;
const dbus = @import("dbus.zig");

pub const Mode = enum { errors, off };
pub const Problem = enum {
    model_load_timed_out,
    transcription_timed_out,
    audio_start_timed_out,
    audio_stalled,

    microphone_not_found,
    microphone_ambiguous,
    microphone_connection_lost,
    microphone_changed,
    microphone_identity_unavailable,
    audio_processing_behind,
    audio_setup_failed,
    audio_teardown_failed,
    model_load_failed,
    speech_detection_conflict,
    clipboard_tool_missing,
    paste_permission_denied,
    paste_device_missing,
    paste_incomplete,
    transcript_storage_full,
    transcript_save_denied,
    transcript_directory_unsafe,

    microphone_failed,
    recording_incomplete,
    transcription_failed,
    exchange_corrupt,
    recording_timed_out,
    speech_unrecognized,
    transcript_too_large,
    transcript_chunk_too_large,
    transcript_token_limit,
    transcript_chunk_and_token_limit,
    clipboard_failed,
    paste_failed,
    transcript_save_failed,
};

pub const Output = enum { unchanged, saved, unsaved, clipboard_saved, clipboard_unsaved, partial_saved, partial_unsaved };
const Message = struct { problem: Problem, output: Output };

/// One connection, one outstanding request, and one coalesced operation. All
/// storage is reusable and all socket I/O is nonblocking. advance must run after
/// show/recover, on descriptor readiness, and at deadline_monotonic_ns.
pub const Client = struct {
    connection: dbus.Connection = .{},
    address: dbus.Address = .{},
    phase: Phase = .disabled,
    request: ?Request = null,
    pending: ?Operation = null,
    last_problem: ?Message = null,
    notification: ?struct { id: u32, owner: Name } = null,
    serial: u32 = 0,
    events: u32 = 0,
    operation_deadline_ns: u64 = std.math.maxInt(u64),
    deadline_monotonic_ns: u64 = std.math.maxInt(u64),

    const Phase = enum { disabled, retry, authenticating, begin, hello, match_closed, match_owner, ready };
    const Request = struct { serial: u32, operation: ?Operation, owner: Name = .{}, invalidated: bool = false };

    /// Environment values are borrowed only during init. A missing explicit
    /// address falls back to XDG_RUNTIME_DIR/bus; no shell or autolaunch helper.
    pub fn init(self: *Client, epoll_fd: std.posix.fd_t, tag: u64, bus_address: ?[]const u8, runtime_directory: ?[]const u8) void {
        self.configure(bus_address, runtime_directory) catch |err| {
            log.warn(.{}, "Desktop notifications unavailable: error={s}", .{@errorName(err)});
            return;
        };
        self.start(epoll_fd, tag) catch |err| {
            self.disconnected(epoll_fd, err, true);
        };
    }

    pub fn show(self: *Client, problem: Problem) void {
        self.showOutput(problem, .unchanged);
    }

    pub fn showOutput(self: *Client, problem: Problem, output: Output) void {
        const message: Message = .{ .problem = problem, .output = output };
        if (self.phase == .disabled or (self.last_problem != null and std.meta.eql(self.last_problem.?, message))) return;
        self.last_problem = message;
        self.pending = .{ .show = message };
    }

    /// Reset only suppression. The next error can replace the existing popup.
    pub fn resetSuppression(self: *Client) void {
        self.last_problem = null;
    }

    /// Close a previous error, including an ID from a still-outstanding Notify.
    pub fn recover(self: *Client) void {
        self.last_problem = null;
        self.pending = .close;
    }

    pub fn advance(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) void {
        if (self.phase == .disabled) return;
        self.process(epoll_fd, tag) catch |err| self.disconnected(epoll_fd, err, false);
    }

    pub fn deinit(self: *Client, epoll_fd: std.posix.fd_t) void {
        self.close(epoll_fd);
        self.initEmpty();
    }

    /// Initialize fresh storage, or reset it after its socket has been closed.
    /// This does not close an existing connection. Call before init or any other
    /// client operation when constructing the client from undefined storage.
    pub fn initEmpty(self: *Client) void {
        // PERFORMANCE: Initialize metadata, not a buffer-containing aggregate.
        // Both Supervisor startup and deinit must use this path: a self.* = .{}
        // reset alone can retain the same 21,208-byte template as construction.
        // Keep every Client and dbus.Connection metadata default synchronized
        // here when adding fields. Buffer contents become valid only as their
        // counts advance; optional payloads remain unreadable while null.
        // Measured 2026-09-07 with stock Zig 0.16.0/LLVM, host x86-64, ReleaseSafe
        // application/inference, static PIE, -Dcrash-diagnostics=false and GNU
        // strip --strip-all: 22,624 bytes saved, including reduced generated code.
        // This preserves buffer capacities and adds no allocation. The combined
        // initializer prototype passed private capture, inference, notification
        // lifecycle and output integration; RAM/timing changes were not measured.
        self.connection.fd = null;
        self.connection.input_size = 0;
        self.connection.output_size = 0;
        self.connection.output_sent = 0;
        self.connection.errno = .SUCCESS;
        self.address = .{};
        self.phase = .disabled;
        self.request = null;
        self.pending = null;
        self.last_problem = null;
        self.notification = null;
        self.serial = 0;
        self.events = 0;
        self.operation_deadline_ns = std.math.maxInt(u64);
        self.deadline_monotonic_ns = std.math.maxInt(u64);
    }

    fn configure(self: *Client, address: ?[]const u8, runtime_directory: ?[]const u8) !void {
        const text = address orelse {
            const directory = runtime_directory orelse return error.MissingBusAddress;
            if (!std.mem.startsWith(u8, directory, "/")) return error.InvalidAddress;
            // This is a filesystem path, not an encoded D-Bus address. Preserve
            // literal commas and percent signs in XDG_RUNTIME_DIR.
            const socket_path = try std.fmt.bufPrint(self.address.value.path[0 .. self.address.value.path.len - 1], "{s}/bus", .{directory});
            self.address.value.path[socket_path.len] = 0;
            self.address.length = @intCast(@offsetOf(linux.sockaddr.un, "path") + socket_path.len + 1);
            return;
        };
        // Desktop session addresses can list several transports. Select a
        // supported Unix endpoint; we deliberately do not implement TCP.
        var addresses = std.mem.splitScalar(u8, text, ';');
        while (addresses.next()) |candidate| {
            self.address = dbus.Address.parse(candidate) catch |err| switch (err) {
                error.UnsupportedAddress => continue,
                else => return err,
            };
            return;
        }
        return error.UnsupportedAddress;
    }

    fn start(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) !void {
        self.phase = .authenticating;
        self.operation_deadline_ns = monotonicNanoseconds() + std.time.ns_per_s;
        self.deadline_monotonic_ns = self.operation_deadline_ns;
        try self.connection.connect(&self.address);
        var event: linux.epoll_event = .{ .events = linux.EPOLL.IN | linux.EPOLL.OUT, .data = .{ .u64 = tag } };
        const result = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, self.connection.fd.?, &event);
        self.connection.errno = linux.errno(result);
        if (self.connection.errno != .SUCCESS) return error.NotificationEpollFailed;
        self.events = event.events;
        var uid_buffer: [10]u8 = undefined;
        const uid = try std.fmt.bufPrint(&uid_buffer, "{d}", .{linux.getuid()});
        var hex: [20]u8 = undefined;
        for (uid, 0..) |byte, i| {
            hex[i * 2] = std.fmt.digitToChar(byte >> 4, .lower);
            hex[i * 2 + 1] = std.fmt.digitToChar(byte & 15, .lower);
        }
        const auth = try std.fmt.bufPrint(&self.connection.output, "\x00AUTH EXTERNAL {s}\r\n", .{hex[0 .. uid.len * 2]});
        self.connection.output_size = auth.len;
    }

    fn process(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) !void {
        const now = monotonicNanoseconds();
        if (self.phase == .retry) {
            if (now < self.operation_deadline_ns) return;
            try self.start(epoll_fd, tag);
        }
        if (now >= self.operation_deadline_ns) {
            if (self.phase != .ready) return error.AuthenticationOrSetupTimedOut;
            if (self.connection.output_size != 0) return error.NotificationWriteTimedOut;
            const request = self.request orelse return error.InvalidDeadline;
            log.warn(.{}, "Desktop notification failed: operation={t}, name=\"org.freedesktop.DBus.Error.NoReply\", message=\"Reply deadline exceeded\", serial={d}", .{ std.meta.activeTag(request.operation.?), request.serial });
            self.notification = null;
            self.request = null;
            self.operation_deadline_ns = std.math.maxInt(u64);
        }
        // Each iteration performs at most one read, write and message dispatch.
        // The timer resumes buffered work when this fairness budget is exhausted.
        var budget: u8 = 16;
        while (budget > 0) : (budget -= 1) {
            try self.connection.flush();
            if (self.phase == .begin and self.connection.output_size == 0) self.phase = .hello;
            if (self.phase == .authenticating) {
                const bytes = self.connection.input[0..self.connection.input_size];
                if (std.mem.indexOf(u8, bytes, "\r\n")) |end| {
                    const line = bytes[0..end];
                    if (line.len != 35 or !std.mem.startsWith(u8, line, "OK ")) {
                        log.warn(.{}, "Desktop notification authentication rejected: response=\"{f}\"", .{std.zig.fmtString(line)});
                        return error.AuthenticationRejected;
                    }
                    for (line[3..]) |byte| if (!std.ascii.isHex(byte)) return error.InvalidAuthenticationReply;
                    if (self.connection.output_size != 0) return error.InvalidAuthenticationReply;
                    self.connection.consume(end + 2);
                    @memcpy(self.connection.output[0..7], "BEGIN\r\n");
                    self.connection.output_size = 7;
                    self.phase = .begin;
                    continue;
                }
                if (bytes.len >= 1024) return error.AuthenticationReplyTooLarge;
            } else if (self.phase != .begin) {
                if (try dbus.Message.parse(self.connection.input[0..self.connection.input_size])) |message| {
                    try self.received(message);
                    self.connection.consume(message.size);
                    if (self.request == null and self.connection.output_size == 0) try self.submit();
                    continue;
                }
                if (self.request == null and self.connection.output_size == 0) try self.submit();
            }
            if (!try self.connection.read()) break;
        }
        const events = linux.EPOLL.IN | @as(u32, if (self.connection.output_size != 0) linux.EPOLL.OUT else 0);
        if (events != self.events) {
            var event: linux.epoll_event = .{ .events = events, .data = .{ .u64 = tag } };
            self.connection.errno = linux.errno(linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_MOD, self.connection.fd.?, &event));
            if (self.connection.errno != .SUCCESS) return error.NotificationEpollFailed;
            self.events = events;
        }
        self.deadline_monotonic_ns = if (budget == 0) 0 else self.operation_deadline_ns;
    }

    fn submit(self: *Client) !void {
        const operation: ?Operation = if (self.phase == .ready) self.pending orelse return else null;
        if (operation != null and operation.? == .close and self.notification == null) {
            self.pending = null;
            return;
        }
        self.serial +%= 1;
        if (self.serial == 0) self.serial = 1;
        const owner: Name = if (operation != null and self.notification != null) self.notification.?.owner else .{};
        const recipient = if (operation == null) bus_destination else if (owner.len != 0) owner.slice() else destination;
        const member: []const u8 = switch (self.phase) {
            .hello => "Hello",
            .match_closed, .match_owner => "AddMatch",
            .ready => if (operation.? == .show) "Notify" else "CloseNotification",
            else => return error.InvalidBusPhase,
        };
        const signature: []const u8 = switch (self.phase) {
            .hello => "",
            .match_closed, .match_owner => "s",
            .ready => if (operation.? == .show) "susssasa{sv}i" else "u",
            else => unreachable,
        };
        var writer = try dbus.Writer.call(&self.connection.output, self.serial, recipient, if (operation == null) bus_path else path, if (operation == null) bus_destination else destination, member, signature);
        switch (self.phase) {
            .hello => {},
            .match_closed => try writer.string("type='signal',sender='org.freedesktop.Notifications',path='/org/freedesktop/Notifications',interface='org.freedesktop.Notifications',member='NotificationClosed'"),
            .match_owner => try writer.string("type='signal',sender='org.freedesktop.DBus',path='/org/freedesktop/DBus',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='org.freedesktop.Notifications'"),
            .ready => switch (operation.?) {
                .show => |problem| {
                    const text = problemText(problem);
                    try writer.string("voiced");
                    try writer.uint32(if (self.notification) |notification| notification.id else 0);
                    try writer.string("dialog-error");
                    try writer.string(std.mem.span(text.title));
                    try writer.string(std.mem.span(text.body));
                    try writer.uint32(0); // Empty actions array, element alignment 4.
                    try writer.uint32(0); // Empty hints array, element alignment 8.
                    try writer.alignTo(8);
                    try writer.uint32(@bitCast(@as(i32, -1)));
                },
                .close => try writer.uint32(self.notification.?.id),
            },
            else => unreachable,
        }
        self.connection.output_size = writer.finish();
        self.request = .{ .serial = self.serial, .operation = operation, .owner = owner };
        self.operation_deadline_ns = monotonicNanoseconds() + std.time.ns_per_s;
        if (operation != null) {
            self.pending = null;
            if (operation.? == .close) self.notification = null;
        }
    }

    fn received(self: *Client, message: dbus.Message) !void {
        var body = message.body;
        if (message.kind == .signal) {
            if (std.mem.eql(u8, message.sender, bus_destination) and std.mem.eql(u8, message.path, bus_path) and std.mem.eql(u8, message.interface, bus_destination) and std.mem.eql(u8, message.member, "NameOwnerChanged")) {
                if (!std.mem.eql(u8, message.signature, "sss")) return error.InvalidOwnerSignal;
                const name = try body.string();
                const previous = try body.string();
                const next = try body.string();
                try body.end();
                if (!std.mem.eql(u8, name, destination)) return;
                self.notification = null;
                // An initial activation may precede the first Notify reply.
                // Do not cancel that request just because its server appeared.
                if (previous.len != 0) {
                    if (self.request) |*request| if (request.operation != null) {
                        request.invalidated = true;
                    };
                    self.pending = null;
                }
                if (next.len != 0 and self.last_problem != null and (self.request == null or self.request.?.invalidated)) self.pending = .{ .show = self.last_problem.? };
                log.debug(.{}, "Desktop notification owner changed: previous=\"{f}\", next=\"{f}\"", .{ std.zig.fmtString(previous), std.zig.fmtString(next) });
            } else if (self.notification) |notification| {
                if (!std.mem.eql(u8, message.sender, notification.owner.slice()) or !std.mem.eql(u8, message.path, path) or !std.mem.eql(u8, message.interface, destination) or !std.mem.eql(u8, message.member, "NotificationClosed")) return;
                if (!std.mem.eql(u8, message.signature, "uu")) return error.InvalidClosedSignal;
                const id = try body.uint32();
                const reason = try body.uint32();
                try body.end();
                if (id == notification.id) {
                    self.notification = null;
                    log.debug(.{}, "Desktop notification closed: id={d}, reason={d}", .{ id, reason });
                }
            }
            return;
        }
        if (message.kind != .reply and message.kind != .err) return;
        const request = self.request orelse return;
        if (message.reply_serial != request.serial) return;
        const from_bus = std.mem.eql(u8, message.sender, bus_destination);
        if (request.operation == null) {
            if (!from_bus) return;
        } else if (request.owner.len != 0) {
            if (!std.mem.eql(u8, message.sender, request.owner.slice()) and !(from_bus and message.kind == .err)) return;
        } else if (!from_bus and !std.mem.startsWith(u8, message.sender, ":")) return;
        self.request = null;
        self.operation_deadline_ns = std.math.maxInt(u64);
        if (request.invalidated) return;
        if (message.kind == .err) {
            const detail = if (std.mem.startsWith(u8, message.signature, "s")) try body.string() else "";
            log.warn(.{}, "Desktop notification failed: operation={s}, name=\"{f}\", message=\"{f}\", serial={d}", .{ if (request.operation) |op| @tagName(op) else @tagName(self.phase), std.zig.fmtString(message.error_name), std.zig.fmtString(detail), message.reply_serial });
            self.notification = null;
            if (request.operation == null) return error.BusSetupRejected;
            return;
        }
        switch (self.phase) {
            .hello => {
                if (!std.mem.eql(u8, message.signature, "s") or !std.mem.startsWith(u8, try body.string(), ":")) return error.InvalidHelloReply;
                try body.end();
                self.phase = .match_closed;
            },
            .match_closed, .match_owner => {
                if (message.signature.len != 0) return error.InvalidMatchReply;
                try body.end();
                self.phase = if (self.phase == .match_closed) .match_owner else .ready;
                if (self.phase == .ready) log.debug(.{}, "Desktop notification bus ready: transport=native", .{});
            },
            .ready => switch (request.operation.?) {
                .show => |problem| {
                    const id = if (std.mem.eql(u8, message.signature, "u")) try body.uint32() else 0;
                    if (id == 0 or from_bus) {
                        self.notification = null;
                        log.warn(.{}, "Desktop notification failed: invalid reply", .{});
                        return;
                    }
                    try body.end();
                    self.notification = .{ .id = id, .owner = try Name.init(message.sender) };
                    log.info(.{}, "Desktop notification accepted: problem={s}", .{@tagName(problem.problem)});
                },
                .close => {
                    if (message.signature.len != 0) return error.InvalidCloseReply;
                    try body.end();
                    log.debug(.{}, "Desktop notification close accepted", .{});
                },
            },
            else => return error.UnexpectedReply,
        }
    }

    fn disconnected(self: *Client, epoll_fd: std.posix.fd_t, err: anyerror, initial: bool) void {
        log.warn(.{}, "Desktop notifications {s}: error={s}, phase={t}, errno={t}({d}), retry_duration=5s", .{ if (initial) "unavailable" else "disconnected", @errorName(err), self.phase, self.connection.errno, @intFromEnum(self.connection.errno) });
        self.close(epoll_fd);
        self.request = null;
        self.notification = null;
        self.pending = if (self.last_problem) |problem| .{ .show = problem } else null;
        self.phase = .retry;
        self.operation_deadline_ns = monotonicNanoseconds() + 5 * std.time.ns_per_s;
        self.deadline_monotonic_ns = self.operation_deadline_ns;
    }

    fn close(self: *Client, epoll_fd: std.posix.fd_t) void {
        if (self.connection.fd) |fd| _ = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_DEL, fd, null);
        self.connection.close();
        self.events = 0;
    }
};

const Name = struct {
    bytes: [255]u8 = undefined,
    len: u8 = 0,
    fn init(text: []const u8) !Name {
        if (text.len == 0 or text.len > 255) return error.InvalidBusName;
        var name: Name = .{ .len = @intCast(text.len) };
        @memcpy(name.bytes[0..text.len], text);
        return name;
    }
    fn slice(self: *const Name) []const u8 {
        return self.bytes[0..self.len];
    }
};

fn monotonicNanoseconds() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const bus_destination = "org.freedesktop.DBus";
const bus_path = "/org/freedesktop/DBus";

const Operation = union(enum) { show: Message, close };
const destination = "org.freedesktop.Notifications";
const path = "/org/freedesktop/Notifications";

fn problemText(message: Message) struct { title: [*:0]const u8, body: [*:0]const u8 } {
    const text: struct { title: [*:0]const u8, body: [*:0]const u8 } = switch (message.problem) {
        .model_load_timed_out => .{ .title = "Voiced: model loading timed out", .body = "See service logs for the stalled operation." },
        .transcription_timed_out => .{ .title = "Voiced: transcription timed out", .body = "See service logs for the stalled operation." },
        .audio_start_timed_out => .{ .title = "Voiced: microphone startup timed out", .body = "No audio arrived. Check the mic and audio service." },
        .audio_stalled => .{ .title = "Voiced: audio stopped arriving", .body = "Recording stopped. Check the mic and audio service." },

        .microphone_not_found => .{ .title = "Voiced: configured mic not found", .body = "Check the configured microphone and its connection." },
        .microphone_ambiguous => .{ .title = "Voiced: multiple mic inputs match", .body = "Replace microphone_serial with microphone_node." },
        .microphone_connection_lost => .{ .title = "Voiced: mic connection lost", .body = "Recording stopped. Check the mic connection." },
        .microphone_changed => .{ .title = "Voiced: mic changed during recording", .body = "Recording stopped to avoid mixing inputs." },
        .microphone_identity_unavailable => .{ .title = "Voiced: cannot verify the mic", .body = "Recording stopped. See service logs." },
        .audio_processing_behind => .{ .title = "Voiced: audio processing fell behind", .body = "Recording stopped. See service logs." },
        .audio_setup_failed => .{ .title = "Voiced: audio setup failed", .body = "See service logs for the cause." },
        .audio_teardown_failed => .{ .title = "Voiced: audio cleanup failed", .body = "See service logs for the cause." },
        .model_load_failed => .{ .title = "Voiced: cannot load the model", .body = "See service logs for the model error." },
        .speech_detection_conflict => .{ .title = "Voiced: speech detectors disagreed", .body = "No text was delivered. Try recording again." },
        .clipboard_tool_missing => .{ .title = "Voiced: wl-copy not found", .body = "Install wl-clipboard so /usr/bin/wl-copy is available." },
        .paste_permission_denied => .{ .title = "Voiced: paste permission denied", .body = "Voiced cannot open /dev/uinput. Paste manually." },
        .paste_device_missing => .{ .title = "Voiced: paste device unavailable", .body = "/dev/uinput is missing. Paste manually." },
        .paste_incomplete => .{ .title = "Voiced: paste may be incomplete", .body = "Check the text before pasting again." },
        .transcript_storage_full => .{ .title = "Voiced: transcript storage full", .body = "Free space in the transcript directory." },
        .transcript_save_denied => .{ .title = "Voiced: transcript save denied", .body = "Check transcript directory permissions." },
        .transcript_directory_unsafe => .{ .title = "Voiced: transcript directory owner mismatch", .body = "Check its owner. See service logs." },

        .microphone_failed => .{ .title = "Voiced: microphone unavailable", .body = "See service logs for the audio error." },
        .recording_incomplete => .{ .title = "Voiced: recording interrupted", .body = "Check the partial transcript." },
        .transcription_failed => .{ .title = "Voiced: transcription failed", .body = "Try again; see logs for details." },
        .exchange_corrupt => .{ .title = "Voiced: recording data was invalid", .body = "Recording stopped. See service logs." },
        .recording_timed_out => .{ .title = "Voiced: recording timed out", .body = "The recording exceeded its processing deadline." },
        .speech_unrecognized => .{ .title = "Voiced: speech not recognized", .body = "Try recording again." },
        .transcript_too_large => .{ .title = "Voiced: transcript size limit reached", .body = "Recording stopped. Check the available text." },
        .transcript_chunk_too_large => .{ .title = "Voiced: chunk text limit reached", .body = "Recording stopped. The last chunk is incomplete." },
        .transcript_token_limit => .{ .title = "Voiced: decoder token limit reached", .body = "Recording stopped. Check for repetition or errors." },
        .transcript_chunk_and_token_limit => .{ .title = "Voiced: text and token limits reached", .body = "Recording stopped. Check for repetition or errors." },
        .clipboard_failed => .{ .title = "Voiced: copy failed", .body = "See service logs for the clipboard error." },
        .paste_failed => .{ .title = "Voiced: paste failed", .body = "See service logs for the paste error." },
        .transcript_save_failed => .{ .title = "Voiced: transcript save failed", .body = "See service logs for the storage error." },
    };
    return .{ .title = text.title, .body = switch (message.output) {
        .unchanged => text.body,
        .saved => "Text was saved to transcript.txt. See logs.",
        .unsaved => "Copy and save failed. See service logs.",
        .clipboard_saved => "Text copied and saved. Check before pasting.",
        .clipboard_unsaved => "Text is on the clipboard; save failed. See logs.",
        .partial_saved => switch (message.problem) {
            .transcript_token_limit, .transcript_chunk_and_token_limit => "Text delivered and saved. Check for repetition.",
            else => "Partial text delivered and saved. Check it.",
        },
        .partial_unsaved => switch (message.problem) {
            .transcript_token_limit, .transcript_chunk_and_token_limit => "Text delivered; save failed. Check for repetition.",
            else => "Partial text delivered; save failed. See logs.",
        },
    } };
}
