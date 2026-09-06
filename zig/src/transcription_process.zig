//! Owns the resident transcription-process contract and its fixed result
//! mailbox. One `publication_state_atomic` selects empty, cancelled, or a
//! committed UTF-8 byte count. A result packet only wakes the supervisor; worker
//! death can recover an already committed mailbox without retranscribing its
//! sealed audio slot.

const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.transcription);
const audio_exchange = @import("audio_exchange.zig");
const descriptor_handoff = @import("descriptor_handoff.zig");
const inference = @import("inference");
const transcription_debug = @import("transcription_debug.zig");
const transcript_file = @import("transcript_file.zig");
const model_cache = @import("model_cache.zig");
const models = @import("models");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;

pub const format_version: u32 = 4;
pub const result_bytes_capacity: u32 = 4096;
pub const protocol_version: u16 = 9;
pub const inference_threads_count_max: u32 = 32;

pub const no_activity_no_speech_probability_reject_min: f32 = 0.60;
pub const active_no_speech_probability_conflict_min: f32 = 0.60;
const failure_message_bytes_capacity: u32 = 1024;
const mailbox_empty: u32 = 0;
const mailbox_cancelled: u32 = 1;
const mailbox_published_offset: u32 = 2;

pub const MailboxState = union(enum) {
    empty,
    cancelled,
    published: u32,
    corrupt,
};

pub const Limit = enum(u32) { none, text_size, tokens_count, text_size_and_tokens_count };

pub const CommittedResult = struct {
    publication_ordinal: u32,
    samples_count: u32,
    contains_activity: bool,
    no_speech_probability: f32,
    average_log_probability: f32,
    bytes: []const u8,
    limit: Limit = .none,
};

pub const ResultDisposition = enum {
    accepted,
    partial,
    no_speech,
    speech_detection_conflict,
    speech_unrecognized,
};

pub const TranscriptExchange = extern struct {
    version: u32,
    reserved: u32,
    session_id: u64,

    // One atomic word owns the terminal publication race: zero is empty, one
    // is cancelled, and values from two encode a committed UTF-8 byte count.
    // Every acceptance input is written before this word publishes the result,
    // so supervisor recovery after worker death reaches the same decision.
    publication_state_atomic: u32,
    reserved_3: u32,
    publication_ordinal: u32,
    samples_count: u32,
    contains_activity: u32,
    no_speech_probability: f32,
    average_log_probability: f32,
    limit: u32,
    bytes: [result_bytes_capacity]u8,
};

pub const ModelLaunchOptions = struct {
    session_id: u64,
    model: models.Model,
    inference_threads_count: u32,
    decoder_threads_count: ?u32 = null,
    encoder_trailing_padding: inference.EncoderTrailingPadding,
};

pub const Command = union(enum) {
    transcribe: audio_exchange.SlotIndex,
    shutdown,
};

pub const ReadyReport = struct {
    model_prepare_duration_ns: u64,
};

/// A result notification acknowledges the end of the worker's shared-memory
/// access and carries timing data. Acceptance evidence lives only in the
/// committed mailbox, including when the worker dies before this notification.
pub const ResultReport = struct {
    publication_ordinal: u32,
    features_duration_ns: u64,
    inference_duration_ns: u64,
};

const ErrorStage = enum(u16) {
    model_load,
    feature_extraction,
    inference,
    text_decode,
    exchange,
};

const Diagnostic = struct {
    stage: ErrorStage,
    message: [failure_message_bytes_capacity]u8,
    message_size: u16,
    evidence: transcription_debug.Evidence = .{},

    pub fn messageBytes(failure: *const Diagnostic) []const u8 {
        assert(failure.message_size <= failure.message.len);
        return failure.message[0..failure.message_size];
    }
};

const Report = union(enum) {
    ready: ReadyReport,
    result: ResultReport,
    stopped,
    failed: Diagnostic,
};

pub const Error = union(enum) {
    model_load: Diagnostic,
    feature_extraction: Diagnostic,
    inference: Diagnostic,
    text_decode: Diagnostic,
    exchange: Diagnostic,
};
pub const Event = union(enum) { ready: ReadyReport, result: ResultReport, stopped };
pub const Result = union(enum) { ok: Event, err: Error };

fn serviceResult(report: Report) Result {
    return switch (report) {
        .ready => |ready| .{ .ok = .{ .ready = ready } },
        .result => |result| .{ .ok = .{ .result = result } },
        .stopped => .{ .ok = .stopped },
        .failed => |detail| switch (detail.stage) {
            inline else => |stage| .{ .err = @unionInit(Error, @tagName(stage), detail) },
        },
    };
}

comptime {
    assert(@sizeOf(TranscriptExchange) == 48 + result_bytes_capacity);
    assert(@offsetOf(TranscriptExchange, "publication_state_atomic") % @alignOf(u32) == 0);
    assert(@offsetOf(TranscriptExchange, "bytes") % @alignOf(u8) == 0);
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
    assert(exchange.samples_count == 0);
    assert(exchange.contains_activity == 0);
    assert(exchange.no_speech_probability == 0);
    assert(exchange.average_log_probability == 0);
    assert(exchange.limit == @intFromEnum(Limit.none));
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
    if (exchange.version != format_version) {
        return .corrupt;
    }
    const encoded = @atomicLoad(
        u32,
        &exchange.publication_state_atomic,
        .acquire,
    );
    return switch (encoded) {
        mailbox_empty => .empty,
        mailbox_cancelled => .cancelled,
        else => {
            const bytes_count = encoded - mailbox_published_offset;
            if (bytes_count > exchange.bytes.len) {
                return .corrupt;
            }
            return .{ .published = bytes_count };
        },
    };
}

/// Shared-mailbox contents the other process published. `none` is empty or
/// cancelled; `corrupt` is a protocol failure, never a clamped prefix.
pub const AcquiredResult = union(enum) {
    none,
    committed: CommittedResult,
    corrupt,
};

/// `acquireResult` returns a committed UTF-8 result. The returned slice remains
/// valid until the supervisor calls `releaseResult` after copying its bytes.
/// Out-of-range fields or invalid UTF-8 are `corrupt`; do not clamp them.
pub fn acquireResult(exchange: *const TranscriptExchange) AcquiredResult {
    const bytes_count = switch (mailboxState(exchange)) {
        .empty, .cancelled => return .none,
        .corrupt => return .corrupt,
        .published => |count| count,
    };
    const limit = std.enums.fromInt(Limit, exchange.limit) orelse return .corrupt;
    if (bytes_count > exchange.bytes.len or
        exchange.samples_count == 0 or
        exchange.samples_count > audio_exchange.slot_samples_capacity or
        exchange.contains_activity > 1 or
        !std.math.isFinite(exchange.no_speech_probability) or
        exchange.no_speech_probability < 0 or
        exchange.no_speech_probability > 1 or
        !std.math.isFinite(exchange.average_log_probability) or
        exchange.average_log_probability > 0 or
        !std.unicode.utf8ValidateSlice(exchange.bytes[0..bytes_count]))
    {
        return .corrupt;
    }
    return .{ .committed = .{
        .publication_ordinal = exchange.publication_ordinal,
        .samples_count = exchange.samples_count,
        .contains_activity = exchange.contains_activity == 1,
        .no_speech_probability = exchange.no_speech_probability,
        .average_log_probability = exchange.average_log_probability,
        .bytes = exchange.bytes[0..bytes_count],
        .limit = limit,
    } };
}

/// Limited results retain the generated prefix for the user to judge, including
/// low-confidence or repetitive text. Otherwise, accept only when both detectors
/// support speech. A high-confidence inactive chunk is normal no-speech; any
/// other disagreement becomes an explicit error rather than publishing a
/// hallucination or silently discarding possible quiet speech.
pub fn classifyResult(result: CommittedResult) ResultDisposition {
    assert(result.samples_count > 0);
    assert(std.math.isFinite(result.no_speech_probability));
    assert(result.no_speech_probability >= 0);
    assert(result.no_speech_probability <= 1);
    assert(std.math.isFinite(result.average_log_probability));
    assert(result.average_log_probability <= 0);

    if (result.limit != .none) return .partial;
    const text = std.mem.trim(u8, result.bytes, " \t\r\n");
    if (text.len == 0) {
        return if (result.contains_activity) .speech_unrecognized else .no_speech;
    }
    if (!result.contains_activity) {
        return if (result.no_speech_probability >=
            no_activity_no_speech_probability_reject_min)
            .no_speech
        else
            .speech_detection_conflict;
    }
    if (result.no_speech_probability >= active_no_speech_probability_conflict_min) {
        return .speech_detection_conflict;
    }
    return .accepted;
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

/// `sendModelLaunch` gives a resident worker one named model and both shared
/// exchanges. The worker resolves the model's XDG installation itself, so the
/// process protocol never treats an arbitrary filesystem path as model identity.
pub fn sendModelLaunch(
    socket: std.posix.fd_t,
    audio_exchange_fd: std.posix.fd_t,
    transcript_exchange_fd: std.posix.fd_t,
    options: ModelLaunchOptions,
) !void {
    assert(socket >= 0);
    assert(audio_exchange_fd >= 0);
    assert(transcript_exchange_fd >= 0);
    assert(options.session_id > 0);
    assert(options.inference_threads_count > 0);
    assert(options.inference_threads_count <= inference_threads_count_max);

    const wire: ModelWireLaunch = .{
        .version = protocol_version,
        .model = @intFromEnum(options.model),
        .encoder_trailing_padding = @intFromEnum(options.encoder_trailing_padding),
        .inference_threads_count = options.inference_threads_count,
        .decoder_threads_count = options.decoder_threads_count orelse options.inference_threads_count,
        .session_id = options.session_id,
        .reserved = 0,
    };

    const descriptors = [_]std.posix.fd_t{
        audio_exchange_fd,
        transcript_exchange_fd,
    };
    try descriptor_handoff.send(socket, &wire, &descriptors);
}

/// `runModelWorker` loads a cached packed model and initializes one runtime,
/// reporting readiness without running inference. It borrows sealed Float32
/// slots sequentially until shutdown. After a result
/// notification the worker touches neither exchange until its next command;
/// the supervisor may reset both exchanges for a new session while it is idle.
/// Cancellation prevents publication; the supervisor contains stalled inference
/// by terminating this process and its complete CPU worker group.
pub fn runModelWorker(
    init: std.process.Init,
    control_socket: std.posix.fd_t,
    expected_supervisor_pid: linux.pid_t,
) !void {
    assert(control_socket >= 0);
    assert(expected_supervisor_pid > 1);

    bindLifetimeToSupervisor(expected_supervisor_pid);
    unblockServiceSignals();
    defer closeDescriptor(control_socket);

    var launch: ModelWireLaunch = undefined;
    var shared_descriptors = try descriptor_handoff.receive(control_socket, &launch);
    defer shared_descriptors.deinit();

    assert(launch.version == protocol_version);
    assert(launch.session_id > 0);
    assert(launch.inference_threads_count > 0);
    assert(launch.inference_threads_count <= inference_threads_count_max);
    assert(launch.reserved == 0);
    assert(launch.decoder_threads_count > 0 and launch.decoder_threads_count <= launch.inference_threads_count);
    const selected_model = std.enums.fromInt(models.Model, launch.model);
    assert(selected_model != null);
    const encoder_trailing_padding = std.enums.fromInt(inference.EncoderTrailingPadding, launch.encoder_trailing_padding);
    assert(encoder_trailing_padding != null);

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

    // ── Load The Resident Runtime ──
    //
    // The cache loader releases pristine input before workspace allocation.
    // Shutdown joins the runtime's workers before releasing its borrowed model,
    // vocabulary, and arena. No per-chunk allocation or waveform copy is needed.

    const allocator = init.gpa;
    const capture_directory = transcript_file.allocDirectoryPath(allocator, init.environ_map.get("XDG_STATE_HOME"), init.environ_map.get("HOME"), init.environ_map.get("VOICED_INSTANCE") orelse "") catch |err| unavailable: {
        log.err(.{}, "Failed transcription capture unavailable: detail=\"{f}\"", .{std.zig.fmtString(@errorName(err))});
        break :unavailable null;
    };
    defer if (capture_directory) |path| allocator.free(path);
    const model_load_started_ns = monotonicNanoseconds();
    var loaded = model_cache.loadModel(init, selected_model.?) catch |err| {
        try sendDiagnostic(control_socket, .model_load, @errorName(err), .{});
        return;
    };
    defer loaded.deinit();
    log.debug(.{ .recording_ordinal = audio.session_id }, "Model weights loaded: recording_ordinal={d}, model_weights_load_duration_ms={d:.3}", .{ audio.session_id, @as(f64, @floatFromInt(monotonicNanoseconds() - model_load_started_ns)) / std.time.ns_per_ms });
    const vocabulary_started_ns = monotonicNanoseconds();
    const vocabulary = model_cache.loadVocabulary(init, selected_model.?) catch |err| {
        try sendDiagnostic(control_socket, .model_load, @errorName(err), .{});
        return;
    };
    defer allocator.free(vocabulary);
    log.debug(.{ .recording_ordinal = audio.session_id }, "Model vocabulary loaded: recording_ordinal={d}, model_vocabulary_load_duration_ms={d:.3}", .{ audio.session_id, @as(f64, @floatFromInt(monotonicNanoseconds() - vocabulary_started_ns)) / std.time.ns_per_ms });
    const runtime_started_ns = monotonicNanoseconds();
    const policy: inference.Policy = .{
        .samples_count_max = audio_exchange.slot_samples_capacity,
        .workers_count = launch.inference_threads_count,
        .decoder_workers_count = launch.decoder_threads_count,
        .generated_tokens_count_max = 446,
    };
    const memory_size = try inference.Runtime.requiredMemorySize(loaded.model.kind, policy);
    const memory = allocator.alignedAlloc(u8, .fromByteUnits(inference.runtime_memory_alignment), memory_size) catch |err| {
        try sendDiagnostic(control_socket, .model_load, @errorName(err), .{});
        return;
    };
    defer allocator.free(memory);
    var runtime: inference.Runtime = undefined;
    runtime.init(init.io, &loaded.model, vocabulary, memory, policy) catch |err| {
        try sendDiagnostic(control_socket, .model_load, @errorName(err), .{});
        return;
    };
    defer runtime.deinit();
    log.debug(.{ .recording_ordinal = audio.session_id }, "Model runtime initialized: recording_ordinal={d}, model_runtime_init_duration_ms={d:.3}, model_runtime_size={d}", .{ audio.session_id, @as(f64, @floatFromInt(monotonicNanoseconds() - runtime_started_ns)) / std.time.ns_per_ms, memory_size });
    const model_prepare_duration_ns = monotonicNanoseconds() - model_load_started_ns;

    try sendReport(control_socket, .{ .ready = .{
        .model_prepare_duration_ns = model_prepare_duration_ns,
    } });

    // ── Serve Sealed Audio Slots ──
    //
    // The supervisor dispatches at most one slot while this mailbox is empty.
    // Audio cannot reclaim the slot until the supervisor copies the published
    // text, so both the PCM slice and its ordinal remain stable throughout the
    // synchronous native call.

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
        const published = switch (audio_exchange.acquireSlot(slot)) {
            .published => |publication| publication,
            .empty, .corrupt => {
                try sendDiagnostic(control_socket, .exchange, "CorruptAudioSlot", .{});
                return;
            },
        };
        switch (mailboxState(transcript)) {
            .empty => {},
            .cancelled, .published, .corrupt => {
                try sendDiagnostic(control_socket, .exchange, "CorruptMailbox", .{});
                return;
            },
        }

        assert(audio.session_id == transcript.session_id);
        var timings: inference.Timings = .{};
        var decoded: ?inference.Transcription = null;
        var stage: ErrorStage = .inference;
        var error_message: ?[]const u8 = null;
        var limit: Limit = .none;
        const samples = slot.samples[0..published.samples_count];
        const generated_text = runtime.transcribe(samples, &transcript.bytes, .{
            .encoder_trailing_padding = encoder_trailing_padding.?,
            .timings = &timings,
            .evidence = &decoded,
        }) catch |err| failed: {
            stage = switch (err) {
                error.AudioTooShort, error.AudioDurationExceedsLimit, error.InvalidSamples => .feature_extraction,
                error.InvalidVocabulary, error.OutputTooSmall => .text_decode,
                else => .inference,
            };
            if (err == error.OutputTooSmall) limit = .text_size;
            error_message = @errorName(err);
            break :failed null;
        };
        if (decoded) |value| {
            if (value.end == .token_limit) {
                limit = if (limit == .text_size) .text_size_and_tokens_count else .tokens_count;
                if (error_message == null) error_message = "GeneratedTokenLimitExceeded";
            }
        }
        // Commit limited text before diagnostic disk I/O. Recovery can deliver
        // it if saving stalls or the worker exits. No result notification or
        // subsequent command permits storage reuse while the snapshot borrows it.
        if (limit != .none and !publishResult(transcript, published, decoded.?, limit)) {
            try sendReport(control_socket, .stopped);
            return;
        }
        if (error_message) |message| {
            if (mailboxState(transcript) == .cancelled) {
                try sendReport(control_socket, .stopped);
                return;
            }
            var evidence: transcription_debug.Evidence = .{
                .chunk_available = 1,
                .chunk = published.publication_ordinal,
                .samples = published.samples_count,
                .token_limit = @intCast(policy.generated_tokens_count_max),
            };
            evidence.finish(decoded, timings);
            // Preserve the original error and decoder evidence even when the
            // available text will be delivered with a limit warning.
            try sendDiagnostic(control_socket, stage, message, evidence);
            if (capture_directory) |path| {
                var timestamp: linux.timespec = undefined;
                const time_result = linux.clock_gettime(.REALTIME, &timestamp);
                assert(linux.errno(time_result) == .SUCCESS);
                const metadata: transcription_debug.Metadata = .{
                    .session_id = audio.session_id,
                    .captured_unix_seconds = timestamp.sec,
                    .stage = @tagName(stage),
                    .error_name = message,
                    .evidence = evidence,
                    .contains_activity = published.contains_activity,
                    .model = selected_model.?.name(),
                    .model_revision = selected_model.?.metadata().revision,
                    .source_sha256 = selected_model.?.metadata().weights.sha256,
                    .packed_image_sha256 = transcription_debug.imageDigest(&loaded.model),
                    .model_encoder_threads = launch.inference_threads_count,
                    .model_decoder_threads = @intCast(runtime.decoder_workers_count),
                    .model_encoder_padding_seconds = transcription_debug.paddingSeconds(encoder_trailing_padding.?),
                    .text_decode_complete = generated_text != null,
                    .end = if (decoded) |value| @tagName(value.end) else null,
                };
                switch (transcription_debug.save(init.io, path, samples, if (decoded) |value| value.text else "", runtime.generated_tokens[0..evidence.tokens], metadata)) {
                    .ok => log.info(.{ .recording_ordinal = audio.session_id }, "Failed transcription saved: recording_ordinal={d}, chunk_ordinal={d}, path=\"{f}/last-failed\"", .{ audio.session_id, evidence.chunk, std.zig.fmtString(path) }),
                    .err => |err| transcription_debug.logError(.{ .recording_ordinal = audio.session_id }, err, path),
                }
            }
            if (limit == .none) return;
        }
        const features_duration_ns = timings.log_mel_ns;
        const inference_duration_ns = timings.encoder_ns + timings.cross_key_values_ns + timings.decoder_ns;
        if (limit == .none and !publishResult(transcript, published, generated_text.?, .none)) {
            try sendReport(control_socket, .stopped);
            return;
        }

        try sendReport(control_socket, .{ .result = .{
            .publication_ordinal = published.publication_ordinal,
            .features_duration_ns = features_duration_ns,
            .inference_duration_ns = inference_duration_ns,
        } });
    }
}

/// A capacity cut or final token may split a UTF-8 character. Keep only whole
/// characters; this scans the bounded chunk, never the accumulated recording.
pub fn utf8Prefix(bytes: []const u8) []const u8 {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const size = std.unicode.utf8ByteSequenceLength(bytes[offset]) catch break;
        if (size > bytes.len - offset) break;
        _ = std.unicode.utf8Decode(bytes[offset..][0..size]) catch break;
        offset += size;
    }
    return bytes[0..offset];
}

fn publishResult(transcript: *TranscriptExchange, published: audio_exchange.PublishedSlot, decoded: inference.Transcription, limit: Limit) bool {
    // Preserve token-leading spaces across chunks; trim only the assembled text.
    const text = if (limit == .none) decoded.text else utf8Prefix(decoded.text);
    transcript.publication_ordinal = published.publication_ordinal;
    transcript.samples_count = published.samples_count;
    transcript.contains_activity = @intFromBool(published.contains_activity);
    transcript.no_speech_probability = decoded.no_speech_probability;
    transcript.average_log_probability = decoded.average_log_probability;
    transcript.limit = @intFromEnum(limit);
    return @cmpxchgStrong(u32, &transcript.publication_state_atomic, mailbox_empty, @as(u32, @intCast(text.len)) + mailbox_published_offset, .release, .acquire) == null;
}

const ModelWireLaunch = extern struct {
    version: u16,
    model: u8,
    encoder_trailing_padding: u8,
    inference_threads_count: u32,
    session_id: u64,
    decoder_threads_count: u32,
    reserved: u32,
};

const WireCommand = extern struct {
    kind: u16,
    slot_index: u8,
    reserved: u8,
};

const WireReportKind = enum(u16) {
    ready,
    result,
    stopped,
    failed,
};

const WireReport = extern struct {
    kind: u16,
    failure_stage: u16,
    message_size: u16,
    reserved: u16,
    publication_ordinal: u32,
    model_prepare_duration_ns: u64,
    features_duration_ns: u64,
    inference_duration_ns: u64,
    message: [failure_message_bytes_capacity]u8,
    evidence: transcription_debug.Evidence,
};

comptime {
    assert(@sizeOf(ModelWireLaunch) == 24);
    assert(@sizeOf(WireCommand) == 4);
    assert(@sizeOf(WireReport) == 40 + failure_message_bytes_capacity + @sizeOf(transcription_debug.Evidence));
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

fn sendReport(socket: std.posix.fd_t, report: Report) !void {
    var wire: WireReport = .{
        .kind = undefined,
        .failure_stage = 0,
        .message_size = 0,
        .reserved = 0,
        .publication_ordinal = 0,
        .model_prepare_duration_ns = 0,
        .features_duration_ns = 0,
        .inference_duration_ns = 0,
        .message = @splat(0),
        .evidence = .{},
    };

    switch (report) {
        .ready => |ready| {
            wire.kind = @intFromEnum(WireReportKind.ready);
            wire.model_prepare_duration_ns = ready.model_prepare_duration_ns;
        },
        .result => |result| {
            wire.kind = @intFromEnum(WireReportKind.result);
            wire.publication_ordinal = result.publication_ordinal;
            wire.features_duration_ns = result.features_duration_ns;
            wire.inference_duration_ns = result.inference_duration_ns;
        },
        .stopped => wire.kind = @intFromEnum(WireReportKind.stopped),
        .failed => |failure| {
            assert(failure.message_size <= failure.message.len);
            wire.kind = @intFromEnum(WireReportKind.failed);
            wire.failure_stage = @intFromEnum(failure.stage);
            wire.evidence = failure.evidence;
            wire.message_size = failure.message_size;
            @memcpy(wire.message[0..failure.message_size], failure.messageBytes());
        },
    }
    try sendRecord(socket, std.mem.asBytes(&wire));
}

fn sendDiagnostic(
    socket: std.posix.fd_t,
    stage: ErrorStage,
    message: []const u8,
    evidence: transcription_debug.Evidence,
) !void {
    assert(message.len <= failure_message_bytes_capacity);
    var failure: Diagnostic = .{
        .stage = stage,
        .message = @splat(0),
        .message_size = @intCast(message.len),
        .evidence = evidence,
    };
    @memcpy(failure.message[0..failure.message_size], message[0..failure.message_size]);
    try sendReport(socket, .{ .failed = failure });
}

fn receiveCommand(socket: std.posix.fd_t) !Command {
    var wire: WireCommand = undefined;
    try receiveRecord(socket, std.mem.asBytes(&wire));
    if (wire.reserved != 0) {
        return error.InvalidCommand;
    }

    return switch (wire.kind) {
        0 => .{ .transcribe = std.enums.fromInt(audio_exchange.SlotIndex, wire.slot_index) orelse return error.InvalidCommand },
        1 => .shutdown,
        else => error.InvalidCommand,
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

/// `receiveReportNonblocking` returns one complete logical report, outer null
/// when no packet is ready, and inner null after orderly peer closure.
pub fn receiveReportNonblocking(socket: std.posix.fd_t) ??Result {
    var wire: WireReport = undefined;
    while (true) {
        const result = linux.recvfrom(
            socket,
            std.mem.asBytes(&wire).ptr,
            @sizeOf(WireReport),
            linux.MSG.TRUNC | linux.MSG.DONTWAIT,
            null,
            null,
        );
        switch (linux.errno(result)) {
            .SUCCESS => {
                if (result == 0) {
                    return @as(?Result, null);
                }
                if (result != @sizeOf(WireReport)) {
                    return @as(?Result, .{ .err = .{ .exchange = protocolDiagnostic("InvalidReport") } });
                }
                const decoded = decodeReport(wire) catch {
                    return @as(?Result, .{ .err = .{ .exchange = protocolDiagnostic("InvalidReport") } });
                };
                return serviceResult(decoded);
            },
            .INTR => continue,
            .AGAIN => {
                return null;
            },
            .CONNRESET => {
                return @as(?Result, null);
            },
            else => return @as(?Result, .{ .err = .{ .exchange = protocolDiagnostic("InvalidReport") } }),
        }
    }
}

fn decodeReport(wire: WireReport) error{InvalidReport}!Report {
    if (wire.reserved != 0 or wire.message_size > wire.message.len) {
        return error.InvalidReport;
    }
    const kind = std.enums.fromInt(WireReportKind, wire.kind) orelse return error.InvalidReport;
    return switch (kind) {
        .ready => .{ .ready = .{
            .model_prepare_duration_ns = wire.model_prepare_duration_ns,
        } },
        .result => .{ .result = .{
            .publication_ordinal = wire.publication_ordinal,
            .features_duration_ns = wire.features_duration_ns,
            .inference_duration_ns = wire.inference_duration_ns,
        } },
        .stopped => .stopped,
        .failed => failed: {
            const stage = std.enums.fromInt(ErrorStage, wire.failure_stage) orelse return error.InvalidReport;
            var failure: Diagnostic = .{
                .stage = stage,
                .message = @splat(0),
                .message_size = wire.message_size,
                .evidence = wire.evidence,
            };
            @memcpy(
                failure.message[0..failure.message_size],
                wire.message[0..failure.message_size],
            );
            break :failed .{ .failed = failure };
        },
    };
}

fn protocolDiagnostic(message: []const u8) Diagnostic {
    assert(message.len <= failure_message_bytes_capacity);
    var failure: Diagnostic = .{
        .stage = .exchange,
        .message = @splat(0),
        .message_size = @intCast(message.len),
    };
    @memcpy(failure.message[0..failure.message_size], message);
    return failure;
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
            .PIPE, .CONNRESET => {
                return error.TranscriptionPeerClosed;
            },
            else => {
                return error.TranscriptionPacketSendFailed;
            },
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
    if (linux.errno(result) != .SUCCESS) {
        return error.TranscriptionPacketReceiveFailed;
    }
    if (result == 0) {
        return error.TranscriptionSocketClosed;
    }
    if (result != bytes.len) {
        return error.TranscriptionPacketSizeMismatch;
    }
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

fn monotonicNanoseconds() u64 {
    var timestamp: linux.timespec = undefined;
    const result = linux.clock_gettime(.MONOTONIC, &timestamp);
    assert(linux.errno(result) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);
    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

fn closeDescriptor(descriptor: std.posix.fd_t) void {
    assert(descriptor >= 0);
    const result = linux.close(descriptor);
    assert(linux.errno(result) == .SUCCESS);
}
