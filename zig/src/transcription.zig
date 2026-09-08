//! A persistent transcription thread owns the resident runtime and its compute
//! pool. Typed jobs borrow sealed audio; completion releases all chunk accesses.
//! Unload joins the pool before freeing the arena and mapped model weights.

const std = @import("std");
const decimal = @import("decimal.zig");
const logging = @import("logging.zig");
const log = logging.scoped(.transcription);
const audio_exchange = @import("audio_exchange.zig");
const worker = @import("worker.zig");
const inference = @import("inference");
const transcription_debug = @import("transcription_debug.zig");
const model_cache = @import("model_cache.zig");
const models = @import("models");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;

pub const result_bytes_capacity: u32 = 4096;
pub const inference_threads_count_max: u32 = 32;

pub const no_activity_no_speech_probability_reject_min: f32 = 0.60;
pub const active_no_speech_probability_conflict_min: f32 = 0.60;
const failure_message_bytes_capacity: u32 = 1024;
pub const Limit = enum(u32) { none, text_size, tokens_count, text_size_and_tokens_count };

pub const CommittedResult = struct {
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

pub const ModelOptions = struct {
    model: models.Model = models.default,
    inference_threads_count: u32 = 4,
    decoder_threads_count: ?u32 = null,
    encoder_trailing_padding: inference.EncoderTrailingPadding = .seconds_10,
};

pub const Job = union(enum) {
    prepare: struct { recording_ordinal: u64, model: ModelOptions },
    transcribe: struct {
        recording_ordinal: u64,
        chunk_ordinal: u32,
        slot_index: audio_exchange.SlotIndex,
    },
    unload,
};

pub const ReadyReport = struct {
    model_prepare_duration_ns: u64,
};

/// Completion releases the audio borrow. Text borrows Worker.bytes until the
/// next submitted job; the supervisor copies it before submitting more work.
pub const ResultReport = struct {
    transcript: CommittedResult,
    features_duration_ns: u64,
    inference_duration_ns: u64,
};

const ErrorStage = enum(u16) {
    model_load,
    feature_extraction,
    inference,
    text_decode,
};

const Diagnostic = struct {
    message: [failure_message_bytes_capacity]u8,
    message_size: u16,
    evidence: transcription_debug.Evidence = .{},

    pub fn messageBytes(failure: *const Diagnostic) []const u8 {
        assert(failure.message_size <= failure.message.len);
        return failure.message[0..failure.message_size];
    }
};

pub const Error = union(enum) {
    model_load: Diagnostic,
    feature_extraction: Diagnostic,
    inference: Diagnostic,
    text_decode: Diagnostic,
};
pub const Event = union(enum) { ready: ReadyReport, result: ResultReport, stopped, cancelled };
pub const Result = union(enum) { ok: Event, err: Error };

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

pub const Worker = struct {
    mailbox: worker.Mailbox(Job, Result),
    context: model_cache.Context,
    audio: *AudioExchange,
    bytes: [result_bytes_capacity]u8 = undefined,
    capture_directory: ?[]const u8,
    cancel: std.atomic.Value(bool) = .init(false),

    pub fn requestCancellation(self: *Worker) void {
        self.cancel.store(true, .release);
    }

    pub fn submit(self: *Worker, job: Job) void {
        self.cancel.store(false, .release);
        self.mailbox.submit(job);
    }

    pub fn run(self: *Worker) void {
        worker.name("voiced-asr");
        defer self.mailbox.finish();
        while (self.mailbox.next()) |job| {
            switch (job) {
                .prepare => |options| {
                    // resident returns only after joining compute workers and
                    // releasing their arena, vocabulary, and mapped weights.
                    if (resident(self, options)) |err| {
                        logError(options.recording_ordinal, options.model.model, err, .err);
                        self.mailbox.complete(.{ .err = err });
                    } else self.mailbox.complete(.{ .ok = .stopped });
                },
                .transcribe, .unload => unreachable,
            }
        }
    }
};

fn resident(self: *Worker, prepare: @FieldType(Job, "prepare")) ?Error {
    const context = self.context;
    const allocator = context.allocator;
    const launch = prepare.model;
    if (self.cancel.load(.acquire)) return null;
    const model_load_started_ns = monotonicNanoseconds();
    var loaded = model_cache.loadModel(context, launch.model, &self.cancel) catch |err| {
        if (err == error.Cancelled) return null;
        return diagnostic(.model_load, @errorName(err), .{});
    };
    defer loaded.deinit();
    if (self.cancel.load(.acquire)) return null;
    log.debug(.{ .recording_id = prepare.recording_ordinal }, .model_weights_loaded, "model_weights_load_duration_ms={f}", .{decimal.fmt(@as(f64, @floatFromInt(monotonicNanoseconds() - model_load_started_ns)) / std.time.ns_per_ms, 3)});
    const vocabulary_started_ns = monotonicNanoseconds();
    const vocabulary = model_cache.loadVocabulary(context, launch.model) catch |err| {
        return diagnostic(.model_load, @errorName(err), .{});
    };
    defer allocator.free(vocabulary);
    if (self.cancel.load(.acquire)) return null;
    log.debug(.{ .recording_id = prepare.recording_ordinal }, .model_vocabulary_loaded, "model_vocabulary_load_duration_ms={f}", .{decimal.fmt(@as(f64, @floatFromInt(monotonicNanoseconds() - vocabulary_started_ns)) / std.time.ns_per_ms, 3)});
    const runtime_started_ns = monotonicNanoseconds();
    const policy: inference.Policy = .{
        .samples_count_max = audio_exchange.slot_samples_capacity,
        .workers_count = launch.inference_threads_count,
        .decoder_workers_count = if (launch.decoder_threads_count) |count| count else null,
        .generated_tokens_count_max = 446,
    };
    const memory_size = inference.Runtime.requiredMemorySize(loaded.model.kind, policy) catch |err| return diagnostic(.model_load, @errorName(err), .{});
    const memory = allocator.alignedAlloc(u8, .fromByteUnits(inference.runtime_memory_alignment), memory_size) catch |err| {
        return diagnostic(.model_load, @errorName(err), .{});
    };
    defer allocator.free(memory);
    var runtime: inference.Runtime = undefined;
    runtime.init(context.io, &loaded.model, vocabulary, memory, policy) catch |err| {
        return diagnostic(.model_load, @errorName(err), .{});
    };
    defer runtime.deinit();
    if (self.cancel.load(.acquire)) return null;
    log.debug(.{ .recording_id = prepare.recording_ordinal }, .model_runtime_initialized, "model_runtime_init_duration_ms={f} model_runtime_size={d}", .{ decimal.fmt(@as(f64, @floatFromInt(monotonicNanoseconds() - runtime_started_ns)) / std.time.ns_per_ms, 3), memory_size });
    const model_prepare_duration_ns = monotonicNanoseconds() - model_load_started_ns;

    self.mailbox.complete(.{ .ok = .{ .ready = .{
        .model_prepare_duration_ns = model_prepare_duration_ns,
    } } });

    // Runtime and executor remain at these addresses until every compute thread
    // joins. Neither the loaded model nor this stack frame may move while warm.
    while (self.mailbox.next()) |job| {
        switch (job) {
            .transcribe => |work| self.mailbox.complete(transcribe(self, launch, &loaded, &runtime, work)),
            .unload => return null,
            .prepare => unreachable,
        }
    }
    unreachable; // Shutdown is submitted only after unload completes.
}

fn transcribe(self: *Worker, launch: ModelOptions, loaded: *const model_cache.LoadedModel, runtime: *inference.Runtime, work: @FieldType(Job, "transcribe")) Result {
    const audio = self.audio;
    const slot = &audio.slots[work.slot_index.arrayIndex()];
    const published = audio_exchange.acquireSlot(slot).?;
    var timings: inference.Timings = .{};
    var decoded: ?inference.Transcription = null;
    var stage: ErrorStage = .inference;
    var error_message: ?[]const u8 = null;
    var limit: Limit = .none;
    var problem: ?Error = null;
    const samples = slot.samples[0..published.samples_count];
    const generated_text = runtime.transcribe(samples, &self.bytes, .{
        .encoder_trailing_padding = launch.encoder_trailing_padding,
        .timings = &timings,
        .evidence = &decoded,
        .cancellation = &self.cancel,
    }) catch |err| failed: {
        if (err == error.Cancelled) return .{ .ok = .cancelled };
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
    if (error_message) |message| {
        if (self.cancel.load(.acquire)) {
            return .{ .ok = .cancelled };
        }
        var evidence: transcription_debug.Evidence = .{
            .chunk_available = true,
            .chunk = work.chunk_ordinal,
            .samples = published.samples_count,
            .token_limit = @intCast(runtime.generated_tokens.len),
        };
        evidence.finish(decoded, timings);
        // Preserve the original error and decoder evidence even when the
        // available text will be delivered with a limit warning.
        problem = diagnostic(stage, message, evidence);
        // Journal the original diagnostic before filesystem work: a stuck save
        // may force a daemon restart before the typed completion reaches main.
        logError(work.recording_ordinal, launch.model, problem.?, if (limit == .none) .err else .warn);
        if (self.capture_directory) |path| {
            var timestamp: linux.timespec = undefined;
            const time_result = linux.clock_gettime(.REALTIME, &timestamp);
            assert(linux.errno(time_result) == .SUCCESS);
            const metadata: transcription_debug.Metadata = .{
                .session_id = work.recording_ordinal,
                .captured_unix_seconds = timestamp.sec,
                .stage = @tagName(stage),
                .error_name = message,
                .evidence = evidence,
                .contains_activity = published.contains_activity,
                .model = launch.model.name(),
                .model_revision = launch.model.metadata().revision,
                .source_blake3 = launch.model.metadata().weights.blake3,
                .packed_image_blake3 = transcription_debug.imageDigest(&loaded.model),
                .model_encoder_threads = launch.inference_threads_count,
                .model_decoder_threads = @intCast(runtime.decoder_workers_count),
                .model_encoder_padding_seconds = transcription_debug.paddingSeconds(launch.encoder_trailing_padding),
                .text_decode_complete = generated_text != null,
                .end = if (decoded) |value| @tagName(value.end) else null,
            };
            switch (transcription_debug.save(self.context.io, path, samples, if (decoded) |value| value.text else "", runtime.generated_tokens[0..evidence.tokens], metadata)) {
                .ok => log.debug(.{ .recording_id = work.recording_ordinal }, .transcription_capture_saved, "chunk_id={d} path=\"{f}/last-failed\"", .{ evidence.chunk, std.zig.fmtString(path) }),
                .err => |err| transcription_debug.logError(.{ .recording_id = work.recording_ordinal }, err, path),
            }
        }
        if (limit == .none) return .{ .err = problem.? };
    }
    const features_duration_ns = timings.log_mel_ns;
    const inference_duration_ns = timings.encoder_ns + timings.cross_key_values_ns + timings.decoder_ns;
    if (self.cancel.load(.acquire)) return .{ .ok = .cancelled };
    const result = decoded.?;
    return .{ .ok = .{ .result = .{
        .transcript = .{
            .samples_count = published.samples_count,
            .contains_activity = published.contains_activity,
            .no_speech_probability = result.no_speech_probability,
            .average_log_probability = result.average_log_probability,
            .bytes = if (limit == .none) result.text else utf8Prefix(result.text),
            .limit = limit,
        },
        .features_duration_ns = features_duration_ns,
        .inference_duration_ns = inference_duration_ns,
    } } };
}

noinline fn logError(recording_ordinal: u64, selected_model: models.Model, err: Error, severity: logging.Level) void {
    if (logging.enabled(severity)) switch (err) {
        .model_load, .feature_extraction, .inference, .text_decode => |detail| {
            const context: logging.Context = .{ .recording_id = recording_ordinal };
            var storage: [15]logging.Entry = undefined;
            var fields: std.ArrayList(logging.Entry) = .initBuffer(&storage);
            fields.appendSliceAssumeCapacity(&.{
                .{ "model", .{ .str = selected_model.name() } },
                .{ "problem_code", .{ .name = @tagName(err) } },
                .{ "detail", .{ .str = detail.messageBytes() } },
            });
            const evidence = detail.evidence;
            if (evidence.chunk_available) {
                fields.appendSliceAssumeCapacity(&.{
                    .{ "chunk_id", .{ .u = evidence.chunk } },
                    .{ "audio_samples_count", .{ .u = evidence.samples } },
                    .{ "audio_duration_seconds", .{ .f = .{ .value = @as(f64, @floatFromInt(evidence.samples)) / 16000, .digits = 3 } } },
                    .{ "transcription_tokens_count_max", .{ .u = evidence.token_limit } },
                    .{ "features_duration_ms", .{ .f = .{ .value = @as(f64, @floatFromInt(evidence.log_mel_ns)) / std.time.ns_per_ms, .digits = 3 } } },
                    .{ "encoder_duration_ms", .{ .f = .{ .value = @as(f64, @floatFromInt(evidence.encoder_ns)) / std.time.ns_per_ms, .digits = 3 } } },
                    .{ "cross_key_values_duration_ms", .{ .f = .{ .value = @as(f64, @floatFromInt(evidence.cross_key_values_ns)) / std.time.ns_per_ms, .digits = 3 } } },
                    .{ "decoder_duration_ms", .{ .f = .{ .value = @as(f64, @floatFromInt(evidence.decoder_ns)) / std.time.ns_per_ms, .digits = 3 } } },
                });
                if (evidence.decoding_available) fields.appendSliceAssumeCapacity(&.{
                    .{ "transcription_tokens_count", .{ .u = evidence.tokens } },
                    .{ "encoder_positions_count", .{ .u = evidence.encoder_positions } },
                    .{ "no_speech_probability", .{ .f32 = .{ .value = evidence.no_speech_probability, .digits = 6 } } },
                    .{ "average_log_probability", .{ .f32 = .{ .value = evidence.average_log_probability, .digits = 6 } } },
                });
            }
            log.kv(severity, context, if (severity == .err) "transcription_failed" else "transcription_limited", fields.items);
        },
    };
}

fn diagnostic(stage: ErrorStage, message: []const u8, evidence: transcription_debug.Evidence) Error {
    var detail: Diagnostic = .{ .message = undefined, .message_size = @intCast(@min(message.len, failure_message_bytes_capacity)), .evidence = evidence };
    @memcpy(detail.message[0..detail.message_size], message[0..detail.message_size]);
    return switch (stage) {
        inline else => |tag| @unionInit(Error, @tagName(tag), detail),
    };
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

fn monotonicNanoseconds() u64 {
    var timestamp: linux.timespec = undefined;
    const result = linux.clock_gettime(.MONOTONIC, &timestamp);
    assert(linux.errno(result) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);
    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}
