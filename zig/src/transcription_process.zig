//! Owns the resident transcription-process contract and its fixed result
//! mailbox. One `publication_state_atomic` selects empty, cancelled, or a
//! committed UTF-8 byte count. A result packet only wakes the supervisor; worker
//! death can recover an already committed mailbox without retranscribing its
//! sealed audio slot.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const descriptor_handoff = @import("descriptor_handoff.zig");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;

pub const format_version: u32 = 2;
pub const result_bytes_capacity: u32 = 4096;
pub const protocol_version: u16 = 1;

const mailbox_empty: u32 = 0;
const mailbox_cancelled: u32 = 1;
const mailbox_published_offset: u32 = 2;

pub const MailboxState = union(enum) {
    empty,
    cancelled,
    published: u32,
};

pub const TranscriptExchange = extern struct {
    version: u32,
    reserved: u32,
    session_id: u64,

    // One atomic word owns the terminal publication race: zero is empty, one
    // is cancelled, and values from two encode a committed UTF-8 byte count.
    publication_state_atomic: u32,
    reserved_3: u32,
    publication_ordinal: u32,
    reserved_2: u32,
    bytes: [result_bytes_capacity]u8,
};

pub const LaunchOptions = struct {
    session_id: u64,
    fake_behavior: FakeBehavior,
    fake_inference_duration_ms: u32,
};

const WireLaunch = extern struct {
    version: u16,
    fake_behavior: u8,
    reserved: u8,
    session_id: u64,
    fake_inference_duration_ms: u32,
    reserved_2: u32,
};

pub const FakeBehavior = enum(u8) {
    normal,
    crash_before_result,
    crash_after_result,
    hang,
};

pub const Command = union(enum) {
    transcribe: audio_exchange.SlotIndex,
    shutdown,
};

pub const WireCommand = extern struct {
    kind: u16,
    slot_index: u8,
    reserved: u8,
};

pub const ReportKind = enum(u16) {
    ready,
    result,
    stopped,
};

pub const WireReport = extern struct {
    kind: u16,
    reserved: u16,
};

comptime {
    assert(@sizeOf(TranscriptExchange) == 32 + result_bytes_capacity);
    assert(@sizeOf(WireLaunch) == 24);
    assert(@offsetOf(TranscriptExchange, "publication_state_atomic") % @alignOf(u32) == 0);
    assert(@offsetOf(TranscriptExchange, "bytes") % @alignOf(u8) == 0);
    assert(@sizeOf(WireCommand) == 4);
    assert(@sizeOf(WireReport) == 4);
}

/// `initializeExchange` assigns a new session after every process that could
/// access the previous mailbox has acknowledged idle or has been reaped.
pub fn initializeExchange(exchange: *TranscriptExchange, session_id: u64) void {
    assert(session_id > 0);

    @memset(std.mem.asBytes(exchange), 0);
    exchange.version = format_version;
    exchange.session_id = session_id;

    assert(exchange.version == format_version);
    assert(exchange.session_id == session_id);
    assert(exchange.publication_state_atomic == mailbox_empty);
}

/// `requestCancellation` prevents a cooperative worker from committing text
/// after the supervisor has chosen the session's discard path.
pub fn requestCancellation(exchange: *TranscriptExchange) void {
    assert(exchange.version == format_version);
    assert(exchange.session_id > 0);

    _ = @atomicRmw(
        u32,
        &exchange.publication_state_atomic,
        .Xchg,
        mailbox_cancelled,
        .acq_rel,
    );
}

pub fn mailboxState(exchange: *const TranscriptExchange) MailboxState {
    assert(exchange.version == format_version);
    const encoded = @atomicLoad(
        u32,
        &exchange.publication_state_atomic,
        .acquire,
    );
    return switch (encoded) {
        mailbox_empty => .empty,
        mailbox_cancelled => .cancelled,
        else => .{ .published = encoded - mailbox_published_offset },
    };
}

/// `acquireResult` returns a committed UTF-8 result. The returned slice remains
/// valid until the supervisor calls `releaseResult` after copying its bytes.
pub fn acquireResult(exchange: *const TranscriptExchange) ?struct {
    publication_ordinal: u32,
    bytes: []const u8,
} {
    const state = mailboxState(exchange);
    const bytes_count = switch (state) {
        .empty, .cancelled => return null,
        .published => |count| count,
    };
    assert(bytes_count <= exchange.bytes.len);
    return .{
        .publication_ordinal = exchange.publication_ordinal,
        .bytes = exchange.bytes[0..bytes_count],
    };
}

pub fn releaseResult(exchange: *TranscriptExchange) void {
    assert(mailboxState(exchange) == .published);

    @atomicStore(
        u32,
        &exchange.publication_state_atomic,
        mailbox_empty,
        .release,
    );
}

/// Sends one logical worker configuration and the two shared exchanges. The
/// fixed launch record remains private to this module.
pub fn sendLaunch(
    socket: std.posix.fd_t,
    audio_exchange_fd: std.posix.fd_t,
    transcript_exchange_fd: std.posix.fd_t,
    options: LaunchOptions,
) !void {
    assert(socket >= 0);
    assert(audio_exchange_fd >= 0);
    assert(transcript_exchange_fd >= 0);
    assert(options.session_id > 0);

    const wire: WireLaunch = .{
        .version = protocol_version,
        .fake_behavior = @intFromEnum(options.fake_behavior),
        .reserved = 0,
        .session_id = options.session_id,
        .fake_inference_duration_ms = options.fake_inference_duration_ms,
        .reserved_2 = 0,
    };
    const descriptors = [_]std.posix.fd_t{
        audio_exchange_fd,
        transcript_exchange_fd,
    };
    try descriptor_handoff.send(socket, &wire, &descriptors);
}

fn decodeTrustedLaunch(wire: WireLaunch) LaunchOptions {
    assert(wire.version == protocol_version);
    assert(wire.reserved == 0);
    assert(wire.session_id > 0);
    assert(wire.reserved_2 == 0);
    return .{
        .session_id = wire.session_id,
        .fake_behavior = @enumFromInt(wire.fake_behavior),
        .fake_inference_duration_ms = wire.fake_inference_duration_ms,
    };
}

/// `runFakeWorker` exercises the complete process, descriptor, shared-memory,
/// publication, crash, and cancellation contract without loading CTranslate2.
/// Production replaces only its deterministic inference body.
pub fn runFakeWorker(
    control_socket: std.posix.fd_t,
    expected_supervisor_pid: linux.pid_t,
) !void {
    assert(control_socket >= 0);
    assert(expected_supervisor_pid > 1);

    bindLifetimeToSupervisor(expected_supervisor_pid);
    unblockServiceSignals();
    defer closeDescriptor(control_socket);

    var launch_packet: WireLaunch = undefined;
    var shared_descriptors = try descriptor_handoff.receive(
        control_socket,
        &launch_packet,
    );
    defer shared_descriptors.deinit();

    const launch = decodeTrustedLaunch(launch_packet);

    const audio_mapping = try mapShared(AudioExchange, shared_descriptors.values[0]);
    defer std.posix.munmap(audio_mapping.bytes);
    const transcript_mapping = try mapShared(
        TranscriptExchange,
        shared_descriptors.values[1],
    );
    defer std.posix.munmap(transcript_mapping.bytes);
    const audio = audio_mapping.pointer;
    const transcript = transcript_mapping.pointer;

    assert(audio.version == audio_exchange.format_version);
    assert(audio.session_id == launch.session_id);
    assert(transcript.version == format_version);
    assert(transcript.session_id == launch.session_id);
    try sendReport(control_socket, .ready);

    while (true) {
        const command = try receiveCommand(control_socket);
        const slot_index = switch (command) {
            .shutdown => {
                try sendReport(control_socket, .stopped);
                return;
            },
            .transcribe => |index| index,
        };

        const slot = &audio.slots[slot_index.arrayIndex()];
        const published = audio_exchange.acquirePublishedSlot(slot).?;
        assert(mailboxState(transcript) == .empty);

        switch (launch.fake_behavior) {
            .crash_before_result => terminateSelf(),
            .hang => hangForever(),
            .normal, .crash_after_result => {},
        }
        sleepMilliseconds(launch.fake_inference_duration_ms);

        if (mailboxState(transcript) == .cancelled) {
            try sendReport(control_socket, .stopped);
            return;
        }

        transcript.publication_ordinal = published.publication_ordinal;
        const text = std.fmt.bufPrint(
            &transcript.bytes,
            "chunk-{d};",
            .{published.publication_ordinal},
        ) catch unreachable;
        const published_state = @as(u32, @intCast(text.len)) +
            mailbox_published_offset;
        if (@cmpxchgStrong(
            u32,
            &transcript.publication_state_atomic,
            mailbox_empty,
            published_state,
            .release,
            .acquire,
        ) != null) {
            assert(mailboxState(transcript) == .cancelled);
            try sendReport(control_socket, .stopped);
            return;
        }

        if (launch.fake_behavior == .crash_after_result) terminateSelf();
        try sendReport(control_socket, .result);
    }
}

fn mapShared(comptime T: type, descriptor: std.posix.fd_t) !struct {
    bytes: []align(std.heap.page_size_min) u8,
    pointer: *T,
} {
    const bytes = try std.posix.mmap(
        null,
        @sizeOf(T),
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        descriptor,
        0,
    );
    return .{
        .bytes = bytes,
        .pointer = @ptrCast(@alignCast(bytes.ptr)),
    };
}

fn sendReport(socket: std.posix.fd_t, kind: ReportKind) !void {
    const report: WireReport = .{
        .kind = @intFromEnum(kind),
        .reserved = 0,
    };
    try sendRecord(socket, std.mem.asBytes(&report));
}

fn receiveCommand(socket: std.posix.fd_t) !Command {
    var wire: WireCommand = undefined;
    try receiveRecord(socket, std.mem.asBytes(&wire));
    assert(wire.reserved == 0);

    return switch (wire.kind) {
        0 => .{ .transcribe = @enumFromInt(wire.slot_index) },
        1 => .shutdown,
        else => unreachable,
    };
}

/// `sendCommand` is the only encoder for supervisor-to-worker commands. The
/// logical union carries a slot only for transcription; shutdown has no dummy
/// slot in ordinary control flow even though its fixed wire record does.
pub fn sendCommand(socket: std.posix.fd_t, command: Command) !void {
    const wire: WireCommand = switch (command) {
        .transcribe => |slot_index| .{
            .kind = 0,
            .slot_index = @intFromEnum(slot_index),
            .reserved = 0,
        },
        .shutdown => .{
            .kind = 1,
            .slot_index = 0,
            .reserved = 0,
        },
    };
    try sendRecord(socket, std.mem.asBytes(&wire));
}

pub fn decodeTrustedReport(wire: WireReport) ReportKind {
    assert(wire.reserved == 0);
    return @enumFromInt(wire.kind);
}

fn sendRecord(socket: std.posix.fd_t, bytes: []const u8) !void {
    while (true) {
        const result = linux.sendto(
            socket,
            bytes.ptr,
            bytes.len,
            linux.MSG.NOSIGNAL,
            null,
            0,
        );
        switch (linux.errno(result)) {
            .SUCCESS => {
                assert(result == bytes.len);
                return;
            },
            .INTR => continue,
            .PIPE, .CONNRESET => return error.TranscriptionPeerClosed,
            else => return error.TranscriptionPacketSendFailed,
        }
    }
}

fn receiveRecord(socket: std.posix.fd_t, bytes: []u8) !void {
    const result = linux.recvfrom(
        socket,
        bytes.ptr,
        bytes.len,
        linux.MSG.TRUNC,
        null,
        null,
    );
    if (linux.errno(result) != .SUCCESS) return error.TranscriptionPacketReceiveFailed;
    if (result == 0) return error.TranscriptionSocketClosed;
    if (result != bytes.len) return error.TranscriptionPacketSizeMismatch;
}

fn bindLifetimeToSupervisor(expected_supervisor_pid: linux.pid_t) void {
    const result = linux.prctl(
        @intFromEnum(linux.PR.SET_PDEATHSIG),
        @intFromEnum(linux.SIG.KILL),
        0,
        0,
        0,
    );
    assert(linux.errno(result) == .SUCCESS);
    if (linux.getppid() != expected_supervisor_pid) terminateSelf();
}

fn unblockServiceSignals() void {
    var signal_mask = std.posix.sigemptyset();
    std.posix.sigaddset(&signal_mask, .TERM);
    std.posix.sigaddset(&signal_mask, .INT);
    std.posix.sigprocmask(std.posix.SIG.UNBLOCK, &signal_mask, null);
}

fn terminateSelf() noreturn {
    const result = linux.kill(linux.getpid(), .KILL);
    assert(linux.errno(result) == .SUCCESS);
    unreachable;
}

fn hangForever() noreturn {
    while (true) sleepMilliseconds(1000);
}

fn sleepMilliseconds(milliseconds: u32) void {
    var requested: linux.timespec = .{
        .sec = @intCast(milliseconds / 1000),
        .nsec = @intCast((milliseconds % 1000) * std.time.ns_per_ms),
    };
    var remaining: linux.timespec = undefined;
    while (true) {
        switch (linux.errno(linux.nanosleep(&requested, &remaining))) {
            .SUCCESS => return,
            .INTR => requested = remaining,
            else => unreachable,
        }
    }
}

fn closeDescriptor(descriptor: std.posix.fd_t) void {
    assert(descriptor >= 0);
    const result = linux.close(descriptor);
    assert(linux.errno(result) == .SUCCESS);
}
