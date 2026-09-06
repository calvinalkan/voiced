const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.notifications);
const linux = std.os.linux;
const c = @cImport({
    @cInclude("systemd/sd-bus.h");
});

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

/// The supervisor owns this client at a stable address from init through deinit.
/// show/recover only coalesce requests; advance drives all bus I/O and callbacks.
/// One outstanding call and one pending operation bound memory during a stalled
/// desktop. No bus operation waits for a reply or flushes synchronously.
pub const Client = struct {
    bus: ?*c.sd_bus = null,
    matches: [2]?*c.sd_bus_slot = .{ null, null },
    request: ?struct { slot: ?*c.sd_bus_slot, operation: Operation } = null,
    pending: ?Operation = null,
    last_problem: ?Message = null,
    // Retaining the reply retains its unique sender name. Replacement/close
    // target that owner, so an ID can never affect a restarted server's popup.
    notification: ?struct { id: u32, reply: *c.sd_bus_message } = null,
    descriptor: ?std.posix.fd_t = null,
    events: u32 = 0,
    deadline_monotonic_ns: u64 = std.math.maxInt(u64),

    pub fn init(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) void {
        self.connect(epoll_fd, tag) catch |err| {
            log.warn(.{}, "Desktop notifications unavailable: error={s}", .{@errorName(err)});
            self.deinit(epoll_fd);
        };
    }

    pub fn show(self: *Client, problem: Problem) void {
        self.showOutput(problem, .unchanged);
    }

    pub fn showOutput(self: *Client, problem: Problem, output: Output) void {
        const message: Message = .{ .problem = problem, .output = output };
        if (self.bus == null or (self.last_problem != null and std.meta.eql(self.last_problem.?, message))) return;
        self.last_problem = message;
        self.pending = .{ .show = message };
    }

    /// A new recording may report the same error again. Retain the notification
    /// ID and pending reply so the next error can replace the existing popup.
    pub fn resetSuppression(self: *Client) void {
        self.last_problem = null;
    }

    /// A successful recording ends repeat suppression and closes any old error.
    /// If Notify is still outstanding, its returned ID is closed when it arrives.
    pub fn recover(self: *Client) void {
        self.last_problem = null;
        self.pending = .close;
    }

    pub fn advance(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) void {
        if (self.bus == null) return;
        self.process(epoll_fd, tag) catch |err| {
            log.warn(.{}, "Desktop notifications disconnected: error={s}", .{@errorName(err)});
            // A lost session bus disables notifications until service restart.
            // Recording and output retain their own independent lifetimes.
            self.deinit(epoll_fd);
        };
    }

    pub fn deinit(self: *Client, epoll_fd: std.posix.fd_t) void {
        if (self.descriptor) |fd| _ = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_DEL, fd, null);
        if (self.request) |request| _ = c.sd_bus_slot_unref(request.slot);
        for (self.matches) |slot| _ = c.sd_bus_slot_unref(slot);
        self.clearNotification();
        // close_unref deliberately drops queued output; flush_close_unref can
        // block shutdown behind an unresponsive desktop bus.
        _ = c.sd_bus_close_unref(self.bus);
        self.* = .{};
    }

    fn connect(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) !void {
        // sd_bus_open_user resolves the session environment once and starts the
        // connection. Authentication and Hello are driven by sd_bus_process.
        try check(c.sd_bus_open_user(&self.bus), "sd_bus_open_user");
        try check(c.sd_bus_set_method_call_timeout(self.bus, std.time.us_per_s), "sd_bus_set_method_call_timeout");
        try check(c.sd_bus_match_signal_async(self.bus, &self.matches[0], destination, path, destination, "NotificationClosed", closed, matchInstalled, self), "sd_bus_match_signal_async");
        try check(c.sd_bus_add_match_async(self.bus, &self.matches[1], "type='signal',sender='org.freedesktop.DBus',path='/org/freedesktop/DBus',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='org.freedesktop.Notifications'", ownerChanged, matchInstalled, self), "sd_bus_add_match_async");
        const fd = c.sd_bus_get_fd(self.bus);
        try check(fd, "sd_bus_get_fd");
        var event: linux.epoll_event = .{ .events = linux.EPOLL.IN, .data = .{ .u64 = tag } };
        if (linux.errno(linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, fd, &event)) != .SUCCESS)
            return error.NotificationEpollFailed;
        self.descriptor = fd;
        self.events = event.events;
        try self.process(epoll_fd, tag);
    }

    fn process(self: *Client, epoll_fd: std.posix.fd_t, tag: u64) !void {
        // A busy bus must not monopolize worker/control dispatch. If this budget
        // runs out, the existing timer wakes us immediately to resume buffered I/O.
        var budget: u8 = 16;
        while (budget > 0) : (budget -= 1) {
            const result = c.sd_bus_process(self.bus, null);
            try check(result, "sd_bus_process");
            if (result == 0) break;
        }
        if (self.request == null and self.pending != null) {
            var queued_writes: u64 = 0;
            try check(c.sd_bus_get_n_queued_write(self.bus, &queued_writes), "sd_bus_get_n_queued_write");
            // A reply timeout need not remove an unsent message. Wait for prior
            // writes to drain before submitting again, so a blocked bus cannot
            // accumulate wire messages behind our one outstanding reply slot.
            if (queued_writes == 0 and c.sd_bus_is_ready(self.bus) > 0) {
                const operation = self.pending.?;
                self.pending = null;
                self.submit(operation) catch |err| {
                    log.warn(.{}, "Desktop notification failed: error={s}", .{@errorName(err)});
                };
            }
        }
        const interest = c.sd_bus_get_events(self.bus);
        try check(interest, "sd_bus_get_events");
        const events: u32 = (if (interest & linux.POLL.IN != 0) @as(u32, linux.EPOLL.IN) else 0) |
            (if (interest & linux.POLL.OUT != 0) @as(u32, linux.EPOLL.OUT) else 0);
        if (events != self.events) {
            var event: linux.epoll_event = .{ .events = events, .data = .{ .u64 = tag } };
            if (linux.errno(linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_MOD, self.descriptor.?, &event)) != .SUCCESS)
                return error.NotificationEpollFailed;
            self.events = events;
        }
        var timeout_us: u64 = undefined;
        try check(c.sd_bus_get_timeout(self.bus, &timeout_us), "sd_bus_get_timeout");
        self.deadline_monotonic_ns = if (budget == 0) 0 else timeout_us *| std.time.ns_per_us;
    }

    fn submit(self: *Client, operation: Operation) !void {
        if (operation == .close and self.notification == null) return;
        var message: ?*c.sd_bus_message = null;
        const recipient: [*c]const u8 = if (self.notification) |notification| c.sd_bus_message_get_sender(notification.reply) else destination;
        try check(c.sd_bus_message_new_method_call(self.bus, &message, recipient, path, destination, if (operation == .show) "Notify" else "CloseNotification"), "sd_bus_message_new_method_call");
        defer _ = c.sd_bus_message_unref(message);
        switch (operation) {
            .show => |problem| {
                const text = problemText(problem);
                try check(c.sd_bus_message_append(message, "susss", @as([*:0]const u8, "voiced"), if (self.notification) |notification| notification.id else @as(u32, 0), @as([*:0]const u8, "dialog-error"), text.title, text.body), "sd_bus_message_append");
                try check(c.sd_bus_message_open_container(message, 'a', "s"), "sd_bus_message_open_container");
                try check(c.sd_bus_message_close_container(message), "sd_bus_message_close_container");
                try check(c.sd_bus_message_open_container(message, 'a', "{sv}"), "sd_bus_message_open_container");
                try check(c.sd_bus_message_close_container(message), "sd_bus_message_close_container");
                try check(c.sd_bus_message_append(message, "i", @as(i32, -1)), "sd_bus_message_append");
            },
            .close => {
                try check(c.sd_bus_message_append(message, "u", self.notification.?.id), "sd_bus_message_append");
                self.clearNotification();
            },
        }
        var slot: ?*c.sd_bus_slot = null;
        try check(c.sd_bus_call_async(self.bus, &slot, message, replied, self, std.time.us_per_s), "sd_bus_call_async");
        self.request = .{ .slot = slot, .operation = operation };
    }

    fn replied(message: ?*c.sd_bus_message, userdata: ?*anyopaque, _: ?*c.sd_bus_error) callconv(.c) c_int {
        const self: *Client = @ptrCast(@alignCast(userdata.?));
        const request = self.request.?;
        self.request = null;
        defer _ = c.sd_bus_slot_unref(request.slot);
        if (c.sd_bus_message_is_method_error(message, null) > 0) {
            const failure = c.sd_bus_message_get_error(message);
            log.warn(.{}, "Desktop notification failed: operation={t}, name=\"{f}\", message=\"{f}\"", .{ std.meta.activeTag(request.operation), std.zig.fmtString(std.mem.span(failure.*.name)), std.zig.fmtString(if (failure.*.message) |text| std.mem.span(text) else "") });
            if (request.operation == .show) self.clearNotification();
        } else if (request.operation == .show) {
            var id: u32 = 0;
            if (c.sd_bus_message_read(message, "u", &id) > 0 and id != 0) {
                self.clearNotification();
                self.notification = .{ .id = id, .reply = c.sd_bus_message_ref(message).? };
                log.info(.{}, "Desktop notification accepted: problem={s}", .{@tagName(request.operation.show.problem)});
            } else log.warn(.{}, "Desktop notification failed: invalid reply", .{});
        }
        return 1;
    }

    fn closed(message: ?*c.sd_bus_message, userdata: ?*anyopaque, _: ?*c.sd_bus_error) callconv(.c) c_int {
        const self: *Client = @ptrCast(@alignCast(userdata.?));
        var id: u32 = 0;
        var reason: u32 = 0;
        if (c.sd_bus_message_read(message, "uu", &id, &reason) > 0 and self.notification != null and id == self.notification.?.id)
            self.clearNotification();
        return 0;
    }

    fn ownerChanged(message: ?*c.sd_bus_message, userdata: ?*anyopaque, _: ?*c.sd_bus_error) callconv(.c) c_int {
        const self: *Client = @ptrCast(@alignCast(userdata.?));
        var name: [*c]const u8 = null;
        var previous: [*c]const u8 = null;
        var next: [*c]const u8 = null;
        if (c.sd_bus_message_read(message, "sss", &name, &previous, &next) <= 0) return 0;
        self.clearNotification();
        if (self.request) |request| _ = c.sd_bus_slot_unref(request.slot);
        self.request = null;
        self.pending = if (next[0] != 0 and self.last_problem != null) .{ .show = self.last_problem.? } else null;
        return 0;
    }

    fn clearNotification(self: *Client) void {
        if (self.notification) |notification| _ = c.sd_bus_message_unref(notification.reply);
        self.notification = null;
    }

    fn matchInstalled(message: ?*c.sd_bus_message, _: ?*anyopaque, _: ?*c.sd_bus_error) callconv(.c) c_int {
        if (c.sd_bus_message_is_method_error(message, null) > 0) return -@as(c_int, @intFromEnum(linux.E.IO));
        return 0;
    }
};

const Operation = union(enum) { show: Message, close };
const destination = "org.freedesktop.Notifications";
const path = "/org/freedesktop/Notifications";

fn problemText(message: Message) struct { title: [*:0]const u8, body: [*:0]const u8 } {
    const text: struct { title: [*:0]const u8, body: [*:0]const u8 } = switch (message.problem) {
        .model_load_timed_out => .{ .title = "Voiced: model loading timed out", .body = "See service logs for the stalled operation." },
        .transcription_timed_out => .{ .title = "Voiced: transcription timed out", .body = "See service logs for the stalled operation." },
        .audio_start_timed_out => .{ .title = "Voiced: microphone startup timed out", .body = "No audio arrived. Check the mic and audio service." },
        .audio_stalled => .{ .title = "Voiced: audio stopped arriving", .body = "Recording stopped. Check the mic and audio service." },

        .microphone_not_found => .{ .title = "Voiced: configured mic not found", .body = "Check microphone_serial and connected inputs." },
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

// Share error reporting across operation names; the label is runtime data.
fn check(result: c_int, operation: []const u8) !void {
    if (result < 0) {
        log.warn(.{}, "Desktop notification bus error: operation={s}, result={d}", .{ operation, result });
        return error.NotificationBusFailed;
    }
}
