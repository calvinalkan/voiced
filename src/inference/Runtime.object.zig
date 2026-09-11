//! `Runtime` executes English Whisper transcription from normalized 16 kHz samples.
//! The caller owns its address-stable control object and tensor memory; neither
//! initialization nor transcription allocates Zig-owned storage.

const Runtime = @This();

const std = @import("std");
const abi = @import("abi.zig");
const attention = @import("attention.zig");
const decoder_module = @import("decoder.zig");
const encoder_module = @import("encoder.zig");
const Scheduler = @import("Scheduler.zig");
const log_mel = @import("log_mel.zig");
const model_module = @import("Model.zig");
const assert = std.debug.assert;
const TranscriptionResult = @import("TranscriptionResult.zig").TranscriptionResult;

const Decoder = decoder_module.Decoder;
const Encoder = encoder_module;
const Lane = Scheduler.Lane;
const ModelKind = model_module.Kind;
const ModelDimensions = model_module.Dimensions;

pub const memory_alignment: usize = abi.memory_alignment;

const decoder_prompt_tokens_count: usize = 2;
const end_of_text_token: Token = 50_256;
const start_of_transcript_token: Token = 50_257;
const no_speech_token: Token = 50_361;
const no_timestamps_token: Token = 50_362;

pub const Token = decoder_module.Token;

pub const InitError = error{
    InvalidConfig,
    InvalidModel,
    MemorySizeOverflow,
    MemoryTooSmall,
};

/// `Runtime` borrows one immutable model and tensor memory for its lifetime.
/// Keep the model and control object at stable addresses: dispatched operations
/// borrow these views until every compute lane departs.
dimensions: ModelDimensions,
model: *const model_module,
scheduler: *Scheduler,
extractor: log_mel.Extractor,
encoder_padding_max: abi.Padding,
transcript: []u8,
encoder: Encoder,
decoder: Decoder,
generated_tokens: []Token,

/// Includes control state, tensor workspace, and transcript storage;
/// excludes borrowed model storage and OS thread stacks.
pub fn requiredMemory(model: *const model_module, scheduler: *const Scheduler, config: abi.Config) InitError!usize {
    return (try RuntimeMemoryLayout.init(model, scheduler.workersCount(), config)).size;
}

/// Binds caller-owned storage and an initialized shared pool; starts no threads.
/// `runtime` occupies the header of `memory`, followed by the private workspace.
pub fn init(runtime: *Runtime, model: *const model_module, memory: []align(memory_alignment) u8, scheduler: *Scheduler, config: abi.Config) InitError!void {
    const dimensions = model_module.dimensions(model.kind);
    const layout = try RuntimeMemoryLayout.init(model, scheduler.workersCount(), config);

    if (memory.len < layout.size) {
        return error.MemoryTooSmall;
    }

    assert(@intFromPtr(runtime) == @intFromPtr(memory.ptr));

    // ── Bind Caller Memory ──

    var extractor: log_mel.Extractor = undefined;
    try extractor.init(layout.extractor_memory.bind(memory), config.audio_samples_count_max);

    const encoder_query_key_value = layout.encoder_query_key_value.bind(memory);
    const lane_quantized_rows = layout.lane_quantized_rows.bind(memory);

    const encoder_positions_count_max = log_mel.encoderPositionsCountMax(config.audio_samples_count_max, config.encoder_padding_max);

    const encoder: Encoder = .{
        .activation = layout.encoder_activation.bind(memory),
        .shared_scratch = layout.encoder_shared_scratch.bind(memory),
        .query_key_value = encoder_query_key_value,
        .attention_output = layout.encoder_attention.bind(memory),
        .encoded_audio = layout.encoded_audio.bind(memory),
        .lane_float_rows = layout.lane_float_rows.bind(memory),
        .lane_quantized_rows = lane_quantized_rows,
    };

    encoder.validate(dimensions, encoder_positions_count_max, scheduler.workersCount());

    const decoder = Decoder.init(.{
        .cross_key_values = layout.decoder_cross_key_values.bind(memory),
        .self_keys = layout.decoder_self_keys.bind(memory),
        .self_values = layout.decoder_self_values.bind(memory),
        .input = layout.decoder_input.bind(memory),
        .query_key_value = layout.decoder_query_key_value.bind(memory),
        .attention_output = layout.decoder_attention.bind(memory),
        .ffn = layout.decoder_ffn.bind(memory),
        .logits = layout.logits.bind(memory),
        .lane_quantized_rows = lane_quantized_rows,
        .lane_attention_scores = layout.lane_attention_scores.bind(memory),
    }, dimensions, encoder_positions_count_max, decoderPositionsCapacity(config), scheduler.workersCount());

    runtime.* = .{
        .dimensions = dimensions,
        // Borrow the immutable metadata too; copying its layer arrays costs
        // about 11.5 KiB per runtime without extending the model lifetime.
        .model = model,
        .scheduler = scheduler,
        .extractor = extractor,
        .encoder_padding_max = config.encoder_padding_max,
        .transcript = layout.transcript.bind(memory),
        .encoder = encoder,
        .decoder = decoder,
        .generated_tokens = layout.generated_tokens.bind(memory),
    };
}

/// Processes finite mono samples in [-1, 1] at 16 kHz. Text and tokens borrow
/// this runtime until its next call. Calls on one runtime must not overlap;
/// every compute lane departs before any result, including cancellation, returns.
pub fn transcribe(runtime: *Runtime, request: *const abi.Request) TranscriptionResult {
    if (request.encoder_padding.seconds() > runtime.encoder_padding_max.seconds()) {
        return .encoder_padding_exceeds_limit;
    }

    const samples = request.samples;

    const encoder_workers_count = request.encoder_workers_count_max orelse
        runtime.scheduler.workersCount();

    const decoder_workers_count = request.decoder_workers_count_max orelse encoder_workers_count;

    if (encoder_workers_count == 0 or encoder_workers_count > runtime.scheduler.workersCount() or
        decoder_workers_count == 0 or decoder_workers_count > runtime.scheduler.workersCount())
    {
        return .invalid_worker_count;
    }

    var timings: TranscriptionResult.Timings = .{ .log_mel_ns = 0, .encoder_ns = 0, .cross_key_values_ns = 0, .decoder_ns = 0 };

    if (request.cancellation) |cancel| if (cancel.load(.acquire)) return .{ .cancelled = timings };

    // Reject lengths before walking samples. calculate checks them too because
    // the extractor is independently callable, but no feature work starts here
    // until the complete request has passed validation.
    if (samples.len < log_mel.samples_count_min) {
        return .audio_too_short;
    }

    if (samples.len > runtime.extractor.samples_count_max) {
        return .audio_duration_exceeds_limit;
    }

    for (samples) |sample| {
        if (!std.math.isFinite(sample) or sample < -1.0 or sample > 1.0) {
            return .invalid_samples;
        }
    }

    const feature_start = monotonicNanoseconds();

    // Feature extraction and convolution finish before transformer Q/K/V
    // writes begin, so their workspace borrows that future tensor storage.
    const features = runtime.extractor.calculate(samples, request.encoder_padding, runtime.encoder.query_key_value) catch {
        unreachable;
    };

    assert(features.encoderPositionsCount() * runtime.dimensions.encoder_width <= runtime.encoder.activation.len);

    var context: TranscriptionContext = .{
        .runtime = runtime,
        .features = features,
        .timings = &timings,
        .phase_start = monotonicNanoseconds(),
    };

    timings.log_mel_ns = context.phase_start - feature_start;

    // Encoder and decoder use overlapping storage. The encoder's final
    // departure joins all writers before the decoder can repartition it.
    if (request.cancellation) |cancel| if (cancel.load(.acquire)) return .{ .cancelled = timings };

    runtime.scheduler.run(encoder_workers_count, @ptrCast(&context), encodeStep, request.cancellation) catch {
        timings.encoder_ns = monotonicNanoseconds() - context.phase_start;

        return .{ .cancelled = timings };
    };

    const now = monotonicNanoseconds();

    timings.encoder_ns = now - context.phase_start;
    context.phase_start = now;

    runtime.scheduler.run(decoder_workers_count, @ptrCast(&context), decodeStep, request.cancellation) catch {
        const elapsed = monotonicNanoseconds() - context.phase_start;

        if (context.decoder_phase == .cross) {
            timings.cross_key_values_ns = elapsed;
        } else {
            timings.decoder_ns = elapsed;
        }

        return .{ .cancelled = timings };
    };

    timings.decoder_ns = monotonicNanoseconds() - context.phase_start;

    // Selection stops immediately on EOT or when the last token slot is filled.
    // Only the former contributes an EOT probability; it needs no stored end tag.
    const token_limit = context.generated_tokens_count == runtime.generated_tokens.len;
    const scored_tokens_count = context.generated_tokens_count + @intFromBool(!token_limit);

    const average_log_probability = if (scored_tokens_count == 0)
        0
    else
        context.selected_log_probabilities_sum / @as(f32, @floatFromInt(scored_tokens_count));

    assert(context.generated_tokens_count <= runtime.generated_tokens.len);
    assert(std.math.isFinite(context.no_speech_probability));
    assert(context.no_speech_probability >= 0 and context.no_speech_probability <= 1);
    assert(std.math.isFinite(average_log_probability));
    assert(average_log_probability <= 0);

    const tokens = runtime.generated_tokens[0..context.generated_tokens_count];
    var text_size: usize = 0;

    for (tokens) |token| {
        if (token >= end_of_text_token) {
            continue;
        }

        const vocabulary = runtime.model.vocabulary;
        const token_bytes = vocabulary.bytes[vocabulary.offsets[token]..vocabulary.offsets[token + 1]];

        // init reserves token capacity times the longest vocabulary entry.
        // Immutable vocabulary and bounded generation make overflow an invariant.
        assert(token_bytes.len <= runtime.transcript.len - text_size);
        @memcpy(runtime.transcript[text_size..][0..token_bytes.len], token_bytes);

        text_size += token_bytes.len;
    }

    var output: TranscriptionResult.Output = .{
        .text = runtime.transcript[0..text_size],
        .tokens = tokens,
        .encoder_positions_count = features.encoderPositionsCount(),
        .no_speech_probability = context.no_speech_probability,
        .average_log_probability = average_log_probability,
        .timings = timings,
    };

    output.text = transcriptText(output.text, token_limit) catch {
        return .{ .invalid_transcript_encoding = output };
    };

    return if (token_limit)
        .{ .token_limit = output }
    else
        .{ .ok = output };
}

// A token-capacity cut may leave a valid prefix of the final UTF-8 character.
// Repair only that case. Validating a possible completion rejects impossible
// prefixes (overlong encodings, surrogates, values above U+10FFFF); validating
// the preceding text keeps an earlier malformed byte from being hidden.
fn transcriptText(bytes: []const u8, token_limit: bool) error{InvalidTranscriptEncoding}![]const u8 {
    if (std.unicode.utf8ValidateSlice(bytes)) {
        return bytes;
    }

    if (!token_limit) {
        return error.InvalidTranscriptEncoding;
    }

    var start = bytes.len;

    while (start > 0 and bytes[start - 1] & 0xc0 == 0x80) : (start -= 1) {}

    if (start == 0) {
        return error.InvalidTranscriptEncoding;
    }

    start -= 1;

    const size = std.unicode.utf8ByteSequenceLength(bytes[start]) catch {
        return error.InvalidTranscriptEncoding;
    };

    const tail = bytes[start..];

    if (tail.len >= size) {
        return error.InvalidTranscriptEncoding;
    }

    var completed: [4]u8 = @splat(0x80);

    @memcpy(completed[0..tail.len], tail);

    if (tail.len == 1) switch (tail[0]) {
        0xe0 => completed[1] = 0xa0,
        0xf0 => completed[1] = 0x90,
        else => {},
    };

    _ = std.unicode.utf8Decode(completed[0..size]) catch {
        return error.InvalidTranscriptEncoding;
    };

    if (!std.unicode.utf8ValidateSlice(bytes[0..start])) {
        return error.InvalidTranscriptEncoding;
    }

    return bytes[0..start];
}

const TranscriptionContext = struct {
    runtime: *Runtime,
    encoder_phase: Encoder.Phase = .frontend,
    decoder_phase: union(enum) { cross: usize, prompt_start, prompt_no_timestamps, token } = .{ .cross = 0 },
    features: log_mel.Features,
    timings: *TranscriptionResult.Timings,
    phase_start: u64,
    generated_tokens_count: usize = 0,
    no_speech_probability: f32 = 0,
    selected_log_probabilities_sum: f32 = 0,
};

fn encodeStep(raw_context: *anyopaque, lane: Lane) Scheduler.Step {
    const context: *TranscriptionContext = @ptrCast(@alignCast(raw_context));
    const runtime = context.runtime;
    const phase = context.encoder_phase;
    // Frontend and transformer work synchronize internally after every lane
    // reads phase, before the leader advances it. Final normalization does not
    // advance phase. An entry barrier would duplicate that synchronization;
    // preserve this condition if a new phase can update the shared cursor.
    const next = runtime.encoder.forwardPhase(runtime.dimensions, &runtime.model.weights, context.features, phase, lane);

    if (lane.isLeader()) if (next) |value| {
        context.encoder_phase = value;
    };

    return if (next == null) .done else .more;
}

fn decodeStep(raw_context: *anyopaque, lane: Lane) Scheduler.Step {
    const context: *TranscriptionContext = @ptrCast(@alignCast(raw_context));
    const runtime = context.runtime;
    const dimensions = runtime.dimensions;
    const positions_count = context.features.encoderPositionsCount();
    const phase = context.decoder_phase;

    // Cross-K/V tile setup and decodeToken both synchronize internally after
    // reading phase/token inputs. The leader may advance them only after those
    // reads have joined; no extra barrier is needed on entry to every token.
    switch (phase) {
        .cross => |layer_index| {
            const encoded_audio = runtime.encoder.encoded_audio[0 .. positions_count * dimensions.encoder_width];

            runtime.decoder.precomputeCrossLayer(dimensions, &runtime.model.weights, encoded_audio, positions_count, layer_index, lane);

            if (lane.isLeader()) {
                if (layer_index + 1 == dimensions.decoder_layers_count) {
                    context.decoder_phase = .prompt_start;

                    const now = monotonicNanoseconds();

                    context.timings.cross_key_values_ns = now - context.phase_start;
                    context.phase_start = now;
                } else {
                    context.decoder_phase = .{ .cross = layer_index + 1 };
                }
            }

            return .more;
        },

        .prompt_start => {
            runtime.decoder.decodeToken(dimensions, &runtime.model.weights, positions_count, start_of_transcript_token, 0, lane);

            if (lane.isLeader()) {
                context.no_speech_probability = probabilityOfToken(runtime.decoder.memory.logits, no_speech_token);
                context.decoder_phase = .prompt_no_timestamps;
            }

            return .more;
        },
        .prompt_no_timestamps => runtime.decoder.decodeToken(dimensions, &runtime.model.weights, positions_count, no_timestamps_token, 1, lane),
        .token => {
            const count = context.generated_tokens_count;
            const token = runtime.generated_tokens[count - 1];

            runtime.decoder.decodeToken(dimensions, &runtime.model.weights, positions_count, token, count + 1, lane);
        },
    }

    // decodeToken ends collectively, so every lane has consumed the preceding
    // token/count. Only the leader selects the next token and reports completion.
    if (lane.isLeader()) {
        const selection = selectGreedyToken(runtime.decoder.memory.logits, context.generated_tokens_count == 0);

        context.selected_log_probabilities_sum += selection.log_probability;

        if (selection.token == end_of_text_token) {
            return .done;
        }

        runtime.generated_tokens[context.generated_tokens_count] = selection.token;
        context.generated_tokens_count += 1;

        // The final allowed token's next logits have no consumer.
        if (context.generated_tokens_count == runtime.generated_tokens.len) {
            return .done;
        }

        context.decoder_phase = .token;
    }

    return .more;
}

// ─── Greedy Decoding And Vocabulary ────────────────────────────────────────

const TokenSelection = struct {
    token: Token,
    log_probability: f32,
};

fn selectGreedyToken(logits: []f32, is_first_generated_token: bool) TokenSelection {
    for (suppressed_tokens) |token| {
        logits[token] = -std.math.inf(f32);
    }

    if (is_first_generated_token) {
        for (suppressed_first_tokens) |token| {
            logits[token] = -std.math.inf(f32);
        }
    }

    var selected_token: Token = 0;
    var selected_logit = logits[0];
    var maximum_logit = logits[0];

    for (logits[1..], 1..) |logit, token| {
        maximum_logit = @max(maximum_logit, logit);

        if (logit > selected_logit) {
            selected_logit = logit;
            selected_token = @intCast(token);
        }
    }

    var exponential_sum: f32 = 0;

    for (logits) |logit| {
        exponential_sum += @exp(logit - maximum_logit);
    }

    return .{
        .token = selected_token,
        .log_probability = selected_logit - maximum_logit - @log(exponential_sum),
    };
}

fn probabilityOfToken(logits: []const f32, token: Token) f32 {
    assert(token < logits.len);

    var maximum = -std.math.inf(f32);

    for (logits) |logit| {
        maximum = @max(maximum, logit);
    }

    var exponential_sum: f32 = 0;

    for (logits) |logit| {
        exponential_sum += @exp(logit - maximum);
    }

    return @exp(logits[token] - maximum) / exponential_sum;
}

fn monotonicNanoseconds() u64 {
    var timestamp: std.os.linux.timespec = undefined;
    const result = std.os.linux.clock_gettime(.MONOTONIC, &timestamp);

    assert(std.os.linux.errno(result) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);

    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

pub const suppressed_first_tokens = [_]Token{ 220, end_of_text_token };
pub const suppressed_tokens = [_]Token{
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
};

// ─── Runtime Memory Layout ─────────────────────────────────────────────────

const RuntimeMemoryLayout = struct {
    transcript: Slice(u8),
    extractor_memory: Slice(u8),
    encoder_activation: Slice(f32),
    encoder_shared_scratch: Slice(f32),
    encoder_query_key_value: Slice(f32),
    encoder_attention: Slice(f32),
    encoded_audio: Slice(f32),
    decoder_cross_key_values: Slice(f16),
    decoder_self_keys: Slice(f16),
    decoder_self_values: Slice(f16),
    decoder_input: Slice(f32),
    decoder_query_key_value: Slice(f32),
    decoder_attention: Slice(f32),
    decoder_ffn: Slice(f32),
    logits: Slice(f32),
    generated_tokens: Slice(Token),
    lane_float_rows: Slice(f32),
    lane_quantized_rows: Slice(u8),
    lane_attention_scores: Slice(f32),
    size: usize,

    fn init(model: *const model_module, workers_count: usize, config: abi.Config) InitError!RuntimeMemoryLayout {
        const dimensions = model_module.dimensions(model.kind);

        try validateConfig(dimensions, config);
        assert(workers_count > 0);

        const transcript_size = try transcriptSize(model, config.transcript_tokens_count_max);
        const encoder_positions_count = log_mel.encoderPositionsCountMax(config.audio_samples_count_max, config.encoder_padding_max);
        const encoder_width = dimensions.encoder_width;
        const decoder_positions_capacity = decoderPositionsCapacity(config);
        const decoder_width = dimensions.decoder_width;
        const decoder_layers_count = dimensions.decoder_layers_count;

        // ── Sequential Phase Storage ──
        //
        // Encoder tensors occupy future decoder-cache storage. Cross projection
        // reads encoded audio while writing cross K/V, then token decoding
        // replaces that audio with self K/V and decoder work. Encoder lane
        // floats die before the final normalization writes encoded audio, so
        // they may share the later decoder-work region as well.

        var encoder_phase_size: usize = 0;
        const encoder_activation_relative = try reserve(f32, &encoder_phase_size, encoder_positions_count * encoder_width);
        const encoder_query_key_value_relative = try reserve(f32, &encoder_phase_size, attention.encoderQueryKeyValueValuesCount(encoder_positions_count, encoder_width));
        const encoder_attention_relative = try reserve(f32, &encoder_phase_size, encoder_positions_count * encoder_width);

        // Keep fixed tensor offsets independent of pool capacity. QKV holds Mel
        // features during convolution; moving it with growing attention scratch
        // would also move the lane-float lower bound below, potentially enlarging
        // the arena instead of reusing the future decoder workspace.
        const encoder_tensors_end = encoder_phase_size;

        const encoder_shared_scratch_relative = try reserve(f32, &encoder_phase_size, std.math.mul(usize, workers_count, attention.encoderScratchValuesCount(1)) catch {
            return error.MemorySizeOverflow;
        });

        var decoder_phase_size: usize = 0;
        const decoder_cross_key_values_relative = try reserve(f16, &decoder_phase_size, decoder_layers_count * (attention.packedKeyValuesCount(encoder_positions_count, decoder_width) + encoder_positions_count * decoder_width));
        var cross_projection_size = decoder_phase_size;
        const encoded_audio_relative = try reserve(f32, &cross_projection_size, encoder_positions_count * encoder_width);

        const decoder_self_keys_relative = try reserve(f16, &decoder_phase_size, decoder_layers_count * attention.packedKeyValuesCount(decoder_positions_capacity, decoder_width));
        const decoder_self_values_relative = try reserve(f16, &decoder_phase_size, decoder_layers_count * decoder_positions_capacity * decoder_width);
        const decoder_input_relative = try reserve(f32, &decoder_phase_size, decoder_width);
        const decoder_query_key_value_relative = try reserve(f32, &decoder_phase_size, 3 * decoder_width);
        const decoder_attention_relative = try reserve(f32, &decoder_phase_size, decoder_width);
        const decoder_ffn_relative = try reserve(f32, &decoder_phase_size, dimensions.decoder_ffn_width);
        const logits_relative = try reserve(f32, &decoder_phase_size, dimensions.vocabulary_tokens_count);
        const generated_tokens_relative = try reserve(Token, &decoder_phase_size, config.transcript_tokens_count_max);
        // Cross-attention scores cover encoded audio; self-attention scores
        // cover generated token history. Short audio can need fewer positions
        // than decoding, so their shared scratch must accommodate both.
        const attention_positions_capacity = @max(encoder_positions_count, decoder_positions_capacity);
        const lane_attention_scores_relative = try reserve(f32, &decoder_phase_size, attention.decoderScratchValuesCount(attention_positions_capacity, dimensions.decoder_attention_heads_count, workers_count));

        // Lane floats must stay beyond every live encoder tensor, even if the
        // future decoder-input offset is smaller. They may overlap attention
        // scratch: convolution finishes before transformer layers, and each
        // layer joins attention lanes before FFN starts reusing these bytes.
        const lane_float_rows_relative: Slice(f32) = .{
            .offset = @max(decoder_input_relative.offset, encoder_tensors_end),
            .elements_count = std.math.mul(usize, workers_count, encoder_module.laneFloatScratchValuesCount(dimensions)) catch {
                return error.MemorySizeOverflow;
            },
        };

        assert(encoder_module.laneFloatScratchValuesCount(dimensions) * @sizeOf(f32) % memory_alignment == 0);
        assert(encoder_module.laneQuantizedScratchValuesCount(dimensions) % memory_alignment == 0);

        const lane_float_size = std.math.mul(usize, lane_float_rows_relative.elements_count, @sizeOf(f32)) catch {
            return error.MemorySizeOverflow;
        };

        const lane_float_end = std.math.add(usize, lane_float_rows_relative.offset, lane_float_size) catch {
            return error.MemorySizeOverflow;
        };

        const phase_memory_size = @max(@max(encoder_phase_size, cross_projection_size), @max(decoder_phase_size, lane_float_end));

        assert(encoder_query_key_value_relative.elements_count >= log_mel.workspaceValuesCount(encoder_positions_count));

        // ── Permanent Layout ──

        var size: usize = std.mem.alignForward(usize, @sizeOf(Runtime), memory_alignment);
        const extractor_memory = try reserve(u8, &size, log_mel.Extractor.memory_size);
        const phase_memory = try reserve(u8, &size, phase_memory_size);

        const lane_quantized_rows = try reserve(u8, &size, std.math.mul(usize, workers_count, encoder_module.laneQuantizedScratchValuesCount(dimensions)) catch {
            return error.MemorySizeOverflow;
        });

        const transcript = try reserve(u8, &size, transcript_size);

        return .{
            .transcript = transcript,
            .extractor_memory = extractor_memory,
            .encoder_activation = encoder_activation_relative.rebase(phase_memory.offset),
            .encoder_shared_scratch = encoder_shared_scratch_relative.rebase(phase_memory.offset),
            .encoder_query_key_value = encoder_query_key_value_relative.rebase(phase_memory.offset),
            .encoder_attention = encoder_attention_relative.rebase(phase_memory.offset),
            .encoded_audio = encoded_audio_relative.rebase(phase_memory.offset),
            .decoder_cross_key_values = decoder_cross_key_values_relative.rebase(phase_memory.offset),
            .decoder_self_keys = decoder_self_keys_relative.rebase(phase_memory.offset),
            .decoder_self_values = decoder_self_values_relative.rebase(phase_memory.offset),
            .decoder_input = decoder_input_relative.rebase(phase_memory.offset),
            .decoder_query_key_value = decoder_query_key_value_relative.rebase(phase_memory.offset),
            .decoder_attention = decoder_attention_relative.rebase(phase_memory.offset),
            .decoder_ffn = decoder_ffn_relative.rebase(phase_memory.offset),
            .logits = logits_relative.rebase(phase_memory.offset),
            .generated_tokens = generated_tokens_relative.rebase(phase_memory.offset),
            .lane_float_rows = lane_float_rows_relative.rebase(phase_memory.offset),
            .lane_quantized_rows = lane_quantized_rows,
            .lane_attention_scores = lane_attention_scores_relative.rebase(phase_memory.offset),
            .size = size,
        };
    }

    fn reserve(comptime Element: type, size: *usize, elements_count: usize) error{MemorySizeOverflow}!Slice(Element) {
        assert(elements_count > 0);

        // Worker capacity is caller-selected. Check products and alignment before
        // binding any slices, including in builds without runtime safety.
        const rounded = std.math.add(usize, size.*, memory_alignment - 1) catch {
            return error.MemorySizeOverflow;
        };

        const offset = rounded & ~(memory_alignment - 1);

        const bytes_count = std.math.mul(usize, elements_count, @sizeOf(Element)) catch {
            return error.MemorySizeOverflow;
        };

        size.* = std.math.add(usize, offset, bytes_count) catch {
            return error.MemorySizeOverflow;
        };

        return .{ .offset = offset, .elements_count = elements_count };
    }

    fn Slice(comptime Element: type) type {
        return struct {
            offset: usize,
            elements_count: usize,

            fn rebase(slice: @This(), base_offset: usize) @This() {
                assert(base_offset % memory_alignment == 0);

                return .{
                    .offset = std.math.add(usize, base_offset, slice.offset) catch {
                        unreachable;
                    },
                    .elements_count = slice.elements_count,
                };
            }

            fn bind(slice: @This(), memory: []align(memory_alignment) u8) []align(memory_alignment) Element {
                const bytes_count = std.math.mul(usize, slice.elements_count, @sizeOf(Element)) catch {
                    unreachable;
                };

                assert(slice.offset <= memory.len);
                assert(bytes_count <= memory.len - slice.offset);

                const values: [*]align(memory_alignment) Element = @ptrCast(@alignCast(memory.ptr + slice.offset));

                return values[0..slice.elements_count];
            }
        };
    }
};

fn decoderPositionsCapacity(config: abi.Config) usize {
    assert(config.transcript_tokens_count_max > 0);

    return config.transcript_tokens_count_max + decoder_prompt_tokens_count;
}

fn validateConfig(dimensions: ModelDimensions, config: abi.Config) InitError!void {
    if (config.audio_samples_count_max < log_mel.samples_count_min or config.audio_samples_count_max > log_mel.samples_count_max) {
        return error.InvalidConfig;
    }

    if (config.transcript_tokens_count_max == 0 or config.transcript_tokens_count_max > dimensions.decoder_positions_count_max - decoder_prompt_tokens_count) {
        return error.InvalidConfig;
    }

    if (dimensions.encoder_width != dimensions.decoder_width) {
        return error.InvalidConfig;
    }

    if (dimensions.encoder_width % attention.head_width != 0 or dimensions.decoder_width % attention.head_width != 0) {
        return error.InvalidConfig;
    }
}

fn transcriptSize(model: *const model_module, tokens_count_max: usize) InitError!usize {
    const offsets = model.vocabulary.offsets;
    if (offsets.len != model_module.dimensions(model.kind).vocabulary_tokens_count + 1 or offsets[0] != 0) {
        return error.InvalidModel;
    }

    var token_size_max: usize = 0;

    for (offsets[0 .. offsets.len - 1], offsets[1..]) |start, end| {
        if (end < start or end > model.vocabulary.bytes.len) {
            return error.InvalidModel;
        }

        token_size_max = @max(token_size_max, end - start);
    }

    if (offsets[offsets.len - 1] != model.vocabulary.bytes.len) {
        return error.InvalidModel;
    }

    if (token_size_max == 0) {
        return error.InvalidModel;
    }

    return std.math.mul(usize, tokens_count_max, token_size_max) catch error.MemorySizeOverflow;
}

test "UTF-8 capacity repair preserves every valid scalar prefix and rejects malformed text" {
    const malformed = [_][]const u8{
        "\x80",  "\xc0",       "\xc1",     "\xf5",         "\xff",             "\xed\xa0", "\xf4\x90", "\xe0\x80", "\xf0\x80",
        "\xc5x", "x\x80y\xc5", "\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80",
    };

    for (malformed) |bytes| {
        try std.testing.expectError(error.InvalidTranscriptEncoding, transcriptText(bytes, true));
        try std.testing.expectError(error.InvalidTranscriptEncoding, transcriptText(bytes, false));
    }

    // Exhaustive scalar coverage also checks the E0/F0 lower bound and the
    // ED/F4 upper bound: a short but impossible prefix must never be repaired.
    var scalar: u21 = 0;

    while (scalar <= 0x10ffff) : (scalar += 1) {
        if (scalar >= 0xd800 and scalar <= 0xdfff) {
            continue;
        }

        var bytes: [5]u8 = undefined;
        bytes[0] = 'x';

        const size = try std.unicode.utf8Encode(scalar, bytes[1..]);

        try std.testing.expectEqualStrings(bytes[0 .. size + 1], try transcriptText(bytes[0 .. size + 1], true));

        for (1..size) |prefix| {
            try std.testing.expectEqualStrings("x", try transcriptText(bytes[0 .. prefix + 1], true));
            try std.testing.expectError(error.InvalidTranscriptEncoding, transcriptText(bytes[0 .. prefix + 1], false));
        }
    }

    try std.testing.expectEqualStrings("", try transcriptText("", true));
    try std.testing.expectEqualStrings("", try transcriptText("\xc5", true));
}

test "decoder token limits reject overflow before sizing memory" {
    for (std.enums.values(ModelKind)) |kind| {
        // Every supported model has 448 positions, including two prompt tokens.
        for ([_]usize{ 1, 446 }) |limit| {
            try validateConfig(model_module.dimensions(kind), .{ .audio_samples_count_max = log_mel.samples_count_max, .transcript_tokens_count_max = limit, .encoder_padding_max = .seconds_30 });
        }

        for ([_]usize{ 0, 447, std.math.maxInt(usize) - 1, std.math.maxInt(usize) }) |limit| {
            try std.testing.expectError(error.InvalidConfig, validateConfig(model_module.dimensions(kind), .{ .audio_samples_count_max = log_mel.samples_count_max, .transcript_tokens_count_max = limit, .encoder_padding_max = .seconds_30 }));
        }
    }
}

test "runtime scratch stays disjoint from live tensors and rejects overflow" {
    const allocator = std.testing.allocator;

    for (std.enums.values(ModelKind)) |kind| {
        const offsets = try allocator.alloc(u32, model_module.dimensions(kind).vocabulary_tokens_count + 1);
        defer allocator.free(offsets);

        @memset(offsets, 0);

        offsets[offsets.len - 1] = 1;

        // Sizing reads only validated dimensions and vocabulary; tensor contents
        // and owned model backing are irrelevant to this synthetic view.
        const model: model_module = .{
            .kind = kind,
            .weights = undefined,
            .vocabulary = .{ .offsets = offsets, .bytes = "x" },
            .backing_storage = .{ .allocated = .{ .allocator = allocator, .bytes = &.{} } },
        };

        for (std.enums.values(abi.Padding)) |padding| {
            for ([_]usize{ log_mel.samples_count_min, 320, 159_999, 160_000, 160_001, log_mel.samples_count_max - 1, log_mel.samples_count_max }) |samples_count| {
                for ([_]usize{ 1, 5, 6, 7, 13, 14, 15, 446 }) |tokens_count| {
                    const config: abi.Config = .{ .audio_samples_count_max = samples_count, .transcript_tokens_count_max = tokens_count, .encoder_padding_max = padding };

                    // Exercise capacity without allocating its arena or starting
                    // threads. A transcription may use only one lane of a large
                    // pool, but its workspace layout still uses pool capacity.
                    for ([_]usize{ 1, 4, 16, 32, 63, 64, 65, 255, 256, 511, 512, 640, 1024, 65_536 }) |workers_count| {
                        const layout = try RuntimeMemoryLayout.init(&model, workers_count, config);

                        const features: RuntimeMemoryLayout.Slice(f32) = .{
                            .offset = layout.encoder_query_key_value.offset,
                            .elements_count = log_mel.workspaceValuesCount(log_mel.encoderPositionsCountMax(samples_count, padding)),
                        };

                        // Convolution reads Mel features from QKV storage; FFN
                        // preserves activation. Attention needs all three tensors.
                        for ([_]RuntimeMemoryLayout.Slice(f32){ layout.lane_float_rows, layout.encoder_shared_scratch }, 0..) |scratch, scratch_index| {
                            const scratch_end = scratch.offset + scratch.elements_count * @sizeOf(f32);

                            try std.testing.expectEqual(0, scratch.offset % memory_alignment);
                            try std.testing.expect(scratch_end <= layout.lane_quantized_rows.offset);

                            const live_tensors: []const RuntimeMemoryLayout.Slice(f32) = if (scratch_index == 0)
                                &.{ layout.encoder_activation, features }
                            else
                                &.{ layout.encoder_activation, layout.encoder_query_key_value, layout.encoder_attention };

                            for (live_tensors) |tensor| {
                                const tensor_end = tensor.offset + tensor.elements_count * @sizeOf(f32);

                                std.testing.expect(tensor_end <= scratch.offset or scratch_end <= tensor.offset) catch |err| {
                                    std.debug.print("model={s}, padding={s}, samples={d}, tokens={d}, workers={d}: tensor=[{d},{d}), scratch=[{d},{d})\n", .{ @tagName(kind), @tagName(padding), samples_count, tokens_count, workers_count, tensor.offset, tensor_end, scratch.offset, scratch_end });

                                    return err;
                                };
                            }
                        }

                        try std.testing.expect(layout.lane_quantized_rows.offset + layout.lane_quantized_rows.elements_count <= layout.transcript.offset);
                        try std.testing.expect(layout.transcript.offset + layout.transcript.elements_count <= layout.size);
                    }
                }
            }
        }

        const config: abi.Config = .{ .audio_samples_count_max = 320, .transcript_tokens_count_max = 1, .encoder_padding_max = .seconds_5 };

        try std.testing.expectError(error.MemorySizeOverflow, RuntimeMemoryLayout.init(&model, std.math.maxInt(usize) / 32, config));
    }

    var size: usize = std.math.maxInt(usize) - 1;

    try std.testing.expectError(error.MemorySizeOverflow, RuntimeMemoryLayout.reserve(u8, &size, 1));

    size = 0;

    try std.testing.expectError(error.MemorySizeOverflow, RuntimeMemoryLayout.reserve(f32, &size, std.math.maxInt(usize)));
}
