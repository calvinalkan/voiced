//! Owns the resident transcription-process contract and its fixed result
//! mailbox. One `publication_state_atomic` selects empty, cancelled, or a
//! committed UTF-8 byte count. A result packet only wakes the supervisor; worker
//! death can recover an already committed mailbox without retranscribing its
//! sealed audio slot.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const descriptor_handoff = @import("descriptor_handoff.zig");
const gpt2_text = @import("gpt2_text.zig");
const log_mel = @import("log_mel.zig");
const models = @import("models");
const assert = std.debug.assert;
const linux = std.os.linux;

const bridge = @cImport({
    @cInclude("ctranslate2_bridge.h");
});

const AudioExchange = audio_exchange.AudioExchange;

pub const format_version: u32 = 3;
pub const result_bytes_capacity: u32 = 4096;
pub const protocol_version: u16 = 2;
pub const inference_threads_count_max: u32 = 32;

// Fixture sweeps found no normalized-word loss from greedy decoding with the
// selected small.en model, while its four-thread inference was 8–17% faster.
const decoding_beam_size: u32 = 1;
pub const no_activity_no_speech_probability_reject_min: f32 = 0.60;
pub const active_no_speech_probability_conflict_min: f32 = 0.60;
const gpt2_encoded_text_bytes_capacity: u32 = 64 * 1024;
const failure_message_bytes_capacity: u32 = @intCast(bridge.error_message_capacity);
const mailbox_empty: u32 = 0;
const mailbox_cancelled: u32 = 1;
const mailbox_published_offset: u32 = 2;

pub const MailboxState = union(enum) {
    empty,
    cancelled,
    published: u32,
};

pub const CommittedResult = struct {
    publication_ordinal: u32,
    samples_count: u32,
    contains_activity: bool,
    no_speech_probability: f32,
    average_log_probability: f32,
    bytes: []const u8,
};

pub const ResultDisposition = enum {
    accepted,
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
    reserved_2: u32,
    bytes: [result_bytes_capacity]u8,
};

pub const FakeLaunchOptions = struct {
    session_id: u64,
    behavior: FakeBehavior,
    inference_duration_ms: u32,
};

const FakeWireLaunch = extern struct {
    version: u16,
    behavior: u8,
    reserved: u8,
    session_id: u64,
    inference_duration_ms: u32,
    reserved_2: u32,
};

pub const ModelLaunchOptions = struct {
    session_id: u64,
    model: models.Model,
    inference_threads_count: u32,
};

const ModelWireLaunch = extern struct {
    version: u16,
    model_vendor: u8,
    model_variant: u8,
    inference_threads_count: u32,
    session_id: u64,
    reserved: u64,
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

const WireCommand = extern struct {
    kind: u16,
    slot_index: u8,
    reserved: u8,
};

pub const ReadyReport = struct {
    model_load_elapsed_ns: u64,
    warmup_elapsed_ns: u64,
};

const GeneratedText = struct {
    size: u32,
    no_speech_probability: f32,
    average_log_probability: f32,
};

pub const ResultReport = struct {
    publication_ordinal: u32,
    samples_count: u32,
    feature_extraction_elapsed_ns: u64,
    inference_elapsed_ns: u64,
    no_speech_probability: f32,
    average_log_probability: f32,
};

pub const FailureStage = enum(u16) {
    model_load,
    warmup_feature_extraction,
    warmup_inference,
    feature_extraction,
    inference,
    text_decode,
};

pub const FailureReport = struct {
    stage: FailureStage,
    message: [failure_message_bytes_capacity]u8,
    message_size: u16,

    pub fn messageBytes(failure: *const FailureReport) []const u8 {
        assert(failure.message_size <= failure.message.len);
        return failure.message[0..failure.message_size];
    }
};

pub const Report = union(enum) {
    ready: ReadyReport,
    result: ResultReport,
    stopped,
    failed: FailureReport,
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
    samples_count: u32,
    no_speech_probability: f32,
    average_log_probability: f32,
    model_load_elapsed_ns: u64,
    warmup_elapsed_ns: u64,
    feature_extraction_elapsed_ns: u64,
    inference_elapsed_ns: u64,
    message: [failure_message_bytes_capacity]u8,
};

comptime {
    assert(failure_message_bytes_capacity == bridge.error_message_capacity);
    assert(@sizeOf(TranscriptExchange) == 48 + result_bytes_capacity);
    assert(@sizeOf(FakeWireLaunch) == 24);
    assert(@sizeOf(ModelWireLaunch) == 24);
    assert(@offsetOf(TranscriptExchange, "publication_state_atomic") % @alignOf(u32) == 0);
    assert(@offsetOf(TranscriptExchange, "bytes") % @alignOf(u8) == 0);
    assert(@sizeOf(WireCommand) == 4);
    assert(@sizeOf(WireReport) == 56 + failure_message_bytes_capacity);
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
    assert(exchange.reserved_2 == 0);
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
pub fn acquireResult(exchange: *const TranscriptExchange) ?CommittedResult {
    const state = mailboxState(exchange);
    const bytes_count = switch (state) {
        .empty, .cancelled => return null,
        .published => |count| count,
    };
    assert(bytes_count <= exchange.bytes.len);
    assert(exchange.samples_count > 0);
    assert(exchange.contains_activity <= 1);
    assert(std.math.isFinite(exchange.no_speech_probability));
    assert(exchange.no_speech_probability >= 0);
    assert(exchange.no_speech_probability <= 1);
    assert(std.math.isFinite(exchange.average_log_probability));
    assert(exchange.average_log_probability <= 0);
    assert(exchange.reserved_2 == 0);
    return .{
        .publication_ordinal = exchange.publication_ordinal,
        .samples_count = exchange.samples_count,
        .contains_activity = exchange.contains_activity == 1,
        .no_speech_probability = exchange.no_speech_probability,
        .average_log_probability = exchange.average_log_probability,
        .bytes = exchange.bytes[0..bytes_count],
    };
}

/// `classifyResult` accepts text only when the audio detector and Whisper both
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

/// `sendFakeLaunch` gives a deterministic worker both shared exchanges and its
/// fault behavior. The sender retains ownership of all descriptors.
pub fn sendFakeLaunch(
    socket: std.posix.fd_t,
    audio_exchange_fd: std.posix.fd_t,
    transcript_exchange_fd: std.posix.fd_t,
    options: FakeLaunchOptions,
) !void {
    assert(socket >= 0);
    assert(audio_exchange_fd >= 0);
    assert(transcript_exchange_fd >= 0);
    assert(options.session_id > 0);

    const wire: FakeWireLaunch = .{
        .version = protocol_version,
        .behavior = @intFromEnum(options.behavior),
        .reserved = 0,
        .session_id = options.session_id,
        .inference_duration_ms = options.inference_duration_ms,
        .reserved_2 = 0,
    };
    const descriptors = [_]std.posix.fd_t{
        audio_exchange_fd,
        transcript_exchange_fd,
    };
    try descriptor_handoff.send(socket, &wire, &descriptors);
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
        .model_vendor = options.model.vendorCode(),
        .model_variant = options.model.variantCode(),
        .inference_threads_count = options.inference_threads_count,
        .session_id = options.session_id,
        .reserved = 0,
    };

    const descriptors = [_]std.posix.fd_t{
        audio_exchange_fd,
        transcript_exchange_fd,
    };
    try descriptor_handoff.send(socket, &wire, &descriptors);
}

fn decodeTrustedFakeLaunch(wire: FakeWireLaunch) FakeLaunchOptions {
    assert(wire.version == protocol_version);
    assert(wire.reserved == 0);
    assert(wire.session_id > 0);
    assert(wire.reserved_2 == 0);
    return .{
        .session_id = wire.session_id,
        .behavior = @enumFromInt(wire.behavior),
        .inference_duration_ms = wire.inference_duration_ms,
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

    var launch_packet: FakeWireLaunch = undefined;
    var shared_descriptors = try descriptor_handoff.receive(
        control_socket,
        &launch_packet,
    );
    defer shared_descriptors.deinit();

    const launch = decodeTrustedFakeLaunch(launch_packet);

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
    try sendReport(control_socket, .{ .ready = .{
        .model_load_elapsed_ns = 0,
        .warmup_elapsed_ns = 0,
    } });

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

        switch (launch.behavior) {
            .crash_before_result => terminateSelf(),
            .hang => hangForever(),
            .normal, .crash_after_result => {},
        }
        sleepMilliseconds(launch.inference_duration_ms);

        if (mailboxState(transcript) == .cancelled) {
            try sendReport(control_socket, .stopped);
            return;
        }

        transcript.publication_ordinal = published.publication_ordinal;
        transcript.samples_count = published.samples_count;
        transcript.contains_activity = @intFromBool(published.contains_activity);
        transcript.no_speech_probability = 0;
        transcript.average_log_probability = 0;
        transcript.reserved_2 = 0;
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

        if (launch.behavior == .crash_after_result) terminateSelf();
        try sendReport(control_socket, .{ .result = .{
            .publication_ordinal = published.publication_ordinal,
            .samples_count = published.samples_count,
            .feature_extraction_elapsed_ns = 0,
            .inference_elapsed_ns = @as(u64, launch.inference_duration_ms) *
                std.time.ns_per_ms,
            .no_speech_probability = 0,
            .average_log_probability = 0,
        } });
    }
}

/// `runModelWorker` loads and warms one CTranslate2 model, then serves sealed
/// audio slots sequentially until shutdown. The synchronous native call is
/// confined to this process; cancellation prevents publication, while the
/// supervisor may kill the process if native inference does not return.
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
    const selected_model = models.Model.fromCodes(
        launch.model_vendor,
        launch.model_variant,
    );
    assert(selected_model != null);

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

    // ── Load And Warm The Resident Model ──
    //
    // Resolve the trusted model identity through the same XDG convention used
    // by setup. CTranslate2 then creates native worker threads and lazily
    // initializes inference kernels. `ready` follows one complete silent
    // transcription, so later slot deadlines measure steady-state inference.

    const allocator = init.gpa;
    const model_path = models.allocInstalledDirectoryPath(
        init,
        selected_model.?,
    ) catch |path_error| {
        try sendFailureReport(control_socket, .model_load, @errorName(path_error));
        return;
    };
    defer allocator.free(model_path);

    var extractor: log_mel.Extractor = undefined;
    extractor.init(allocator) catch |extractor_error| {
        try sendFailureReport(
            control_socket,
            .warmup_feature_extraction,
            @errorName(extractor_error),
        );
        return;
    };
    defer extractor.deinit();

    var bridge_error: bridge.Error = undefined;
    const model_load_started_ns = monotonicNanoseconds();
    const model = bridge.model_create(
        model_path.ptr,
        launch.inference_threads_count,
        decoding_beam_size,
        &bridge_error,
    ) orelse {
        try sendFailureReport(
            control_socket,
            .model_load,
            std.mem.sliceTo(&bridge_error.message, 0),
        );
        return;
    };
    defer bridge.model_destroy(model);
    const model_load_elapsed_ns = monotonicNanoseconds() - model_load_started_ns;

    const warmup_started_ns = monotonicNanoseconds();
    var warmup_samples: [audio_exchange.sample_rate_hz]i16 = @splat(0);
    const warmup_features = extractor.calculate(&warmup_samples) catch |feature_error| {
        try sendFailureReport(
            control_socket,
            .warmup_feature_extraction,
            @errorName(feature_error),
        );
        return;
    };

    var encoded_text: [gpt2_encoded_text_bytes_capacity]u8 = undefined;
    _ = transcribeFeatures(
        model,
        warmup_features,
        &encoded_text,
        &bridge_error,
    ) catch {
        try sendFailureReport(
            control_socket,
            .warmup_inference,
            std.mem.sliceTo(&bridge_error.message, 0),
        );
        return;
    };
    const warmup_elapsed_ns = monotonicNanoseconds() - warmup_started_ns;

    try sendReport(control_socket, .{ .ready = .{
        .model_load_elapsed_ns = model_load_elapsed_ns,
        .warmup_elapsed_ns = warmup_elapsed_ns,
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
        const published = audio_exchange.acquirePublishedSlot(slot).?;
        assert(mailboxState(transcript) == .empty);

        const feature_extraction_started_ns = monotonicNanoseconds();
        const features = extractor.calculate(
            slot.samples[0..published.samples_count],
        ) catch |feature_error| {
            try sendFailureReport(
                control_socket,
                .feature_extraction,
                @errorName(feature_error),
            );
            return;
        };
        const feature_extraction_elapsed_ns = monotonicNanoseconds() -
            feature_extraction_started_ns;

        if (mailboxState(transcript) == .cancelled) {
            try sendReport(control_socket, .stopped);
            return;
        }

        const inference_started_ns = monotonicNanoseconds();
        const generated_text = transcribeFeatures(
            model,
            features,
            &encoded_text,
            &bridge_error,
        ) catch {
            try sendFailureReport(
                control_socket,
                .inference,
                std.mem.sliceTo(&bridge_error.message, 0),
            );
            return;
        };
        const inference_elapsed_ns = monotonicNanoseconds() - inference_started_ns;

        const decoded_text = gpt2_text.decodeInto(
            &transcript.bytes,
            encoded_text[0..generated_text.size],
        ) catch |decode_error| {
            try sendFailureReport(
                control_socket,
                .text_decode,
                @errorName(decode_error),
            );
            return;
        };
        // GPT-2 text tokens carry their own leading spaces. Preserve those
        // bytes across chunk publication and trim only the complete session;
        // trimming every chunk would turn `sentence.` plus ` Next` into
        // `sentence.Next` and discard the model's boundary decision.
        const text = decoded_text;

        if (mailboxState(transcript) == .cancelled) {
            try sendReport(control_socket, .stopped);
            return;
        }

        transcript.publication_ordinal = published.publication_ordinal;
        transcript.samples_count = published.samples_count;
        transcript.contains_activity = @intFromBool(published.contains_activity);
        transcript.no_speech_probability = generated_text.no_speech_probability;
        transcript.average_log_probability = generated_text.average_log_probability;
        transcript.reserved_2 = 0;
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

        try sendReport(control_socket, .{ .result = .{
            .publication_ordinal = published.publication_ordinal,
            .samples_count = published.samples_count,
            .feature_extraction_elapsed_ns = feature_extraction_elapsed_ns,
            .inference_elapsed_ns = inference_elapsed_ns,
            .no_speech_probability = generated_text.no_speech_probability,
            .average_log_probability = generated_text.average_log_probability,
        } });
    }
}

fn transcribeFeatures(
    model: *bridge.ModelHandle,
    features: log_mel.Features,
    encoded_text_buffer: *[gpt2_encoded_text_bytes_capacity]u8,
    bridge_error_out: *bridge.Error,
) !GeneratedText {
    assert(features.values.len == bridge.log_mel_values_count);
    assert(features.frames_count == bridge.log_mel_frames_count);

    var encoded_text_out: bridge.Gpt2EncodedText = .{
        .bytes = encoded_text_buffer,
        .capacity = encoded_text_buffer.len,
        .size = 0,
        .no_speech_probability = 0,
        .average_log_probability = 0,
    };
    if (!bridge.model_transcribe(
        model,
        features.values.ptr,
        &encoded_text_out,
        bridge_error_out,
    )) {
        return error.ModelTranscriptionFailed;
    }

    assert(encoded_text_out.size <= encoded_text_buffer.len);
    assert(std.math.isFinite(encoded_text_out.no_speech_probability));
    assert(encoded_text_out.no_speech_probability >= 0);
    assert(encoded_text_out.no_speech_probability <= 1);
    assert(std.math.isFinite(encoded_text_out.average_log_probability));
    assert(encoded_text_out.average_log_probability <= 0);
    return .{
        .size = encoded_text_out.size,
        .no_speech_probability = encoded_text_out.no_speech_probability,
        .average_log_probability = encoded_text_out.average_log_probability,
    };
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
        .samples_count = 0,
        .no_speech_probability = 0,
        .average_log_probability = 0,
        .model_load_elapsed_ns = 0,
        .warmup_elapsed_ns = 0,
        .feature_extraction_elapsed_ns = 0,
        .inference_elapsed_ns = 0,
        .message = @splat(0),
    };

    switch (report) {
        .ready => |ready| {
            wire.kind = @intFromEnum(WireReportKind.ready);
            wire.model_load_elapsed_ns = ready.model_load_elapsed_ns;
            wire.warmup_elapsed_ns = ready.warmup_elapsed_ns;
        },
        .result => |result| {
            assert(std.math.isFinite(result.no_speech_probability));
            assert(result.no_speech_probability >= 0);
            assert(result.no_speech_probability <= 1);
            assert(std.math.isFinite(result.average_log_probability));
            assert(result.average_log_probability <= 0);
            wire.kind = @intFromEnum(WireReportKind.result);
            wire.publication_ordinal = result.publication_ordinal;
            wire.samples_count = result.samples_count;
            wire.feature_extraction_elapsed_ns = result.feature_extraction_elapsed_ns;
            wire.inference_elapsed_ns = result.inference_elapsed_ns;
            wire.no_speech_probability = result.no_speech_probability;
            wire.average_log_probability = result.average_log_probability;
        },
        .stopped => wire.kind = @intFromEnum(WireReportKind.stopped),
        .failed => |failure| {
            assert(failure.message_size <= failure.message.len);
            wire.kind = @intFromEnum(WireReportKind.failed);
            wire.failure_stage = @intFromEnum(failure.stage);
            wire.message_size = failure.message_size;
            @memcpy(wire.message[0..failure.message_size], failure.messageBytes());
        },
    }
    try sendRecord(socket, std.mem.asBytes(&wire));
}

fn sendFailureReport(
    socket: std.posix.fd_t,
    stage: FailureStage,
    message: []const u8,
) !void {
    var failure: FailureReport = .{
        .stage = stage,
        .message = @splat(0),
        .message_size = @intCast(@min(message.len, failure_message_bytes_capacity)),
    };
    @memcpy(failure.message[0..failure.message_size], message[0..failure.message_size]);
    try sendReport(socket, .{ .failed = failure });
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

/// `receiveReportNonblocking` returns one complete logical report, outer null
/// when no packet is ready, and inner null after orderly peer closure.
pub fn receiveReportNonblocking(socket: std.posix.fd_t) ??Report {
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
                if (result == 0) return @as(?Report, null);
                assert(result == @sizeOf(WireReport));
                return decodeTrustedReport(wire);
            },
            .INTR => continue,
            .AGAIN => return null,
            .CONNRESET => return @as(?Report, null),
            else => @trap(),
        }
    }
}

fn decodeTrustedReport(wire: WireReport) Report {
    assert(wire.reserved == 0);
    assert(wire.message_size <= wire.message.len);

    return switch (@as(WireReportKind, @enumFromInt(wire.kind))) {
        .ready => .{ .ready = .{
            .model_load_elapsed_ns = wire.model_load_elapsed_ns,
            .warmup_elapsed_ns = wire.warmup_elapsed_ns,
        } },
        .result => .{ .result = .{
            .publication_ordinal = wire.publication_ordinal,
            .samples_count = wire.samples_count,
            .feature_extraction_elapsed_ns = wire.feature_extraction_elapsed_ns,
            .inference_elapsed_ns = wire.inference_elapsed_ns,
            .no_speech_probability = wire.no_speech_probability,
            .average_log_probability = wire.average_log_probability,
        } },
        .stopped => .stopped,
        .failed => failed: {
            var failure: FailureReport = .{
                .stage = @enumFromInt(wire.failure_stage),
                .message = @splat(0),
                .message_size = wire.message_size,
            };
            @memcpy(
                failure.message[0..failure.message_size],
                wire.message[0..failure.message_size],
            );
            break :failed .{ .failed = failure };
        },
    };
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
