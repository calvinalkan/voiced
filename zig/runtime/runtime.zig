//! `Runtime` executes English Whisper transcription from normalized 16 kHz samples.
//! The caller owns its address-stable control object and tensor memory; neither
//! initialization nor transcription allocates Zig-owned storage.

const std = @import("std");
const attention = @import("attention.zig");
const decoder_module = @import("decoder.zig");
const encoder_module = @import("encoder.zig");
const executor_module = @import("executor.zig");
const log_mel = @import("log_mel.zig");
const memory_layout = @import("memory_layout.zig");
const model_module = @import("model.zig");
const assert = std.debug.assert;

const Decoder = decoder_module.Decoder;
const Encoder = encoder_module.Encoder;
const Executor = executor_module.Executor;
const Lane = executor_module.Lane;
const Model = model_module.Model;
const ModelKind = model_module.ModelKind;
const ModelSpecification = model_module.ModelSpecification;
const InferenceWeights = model_module.InferenceWeights;

pub const runtime_memory_alignment: usize = memory_layout.alignment;

const decoder_prompt_tokens_count: usize = 2;
const end_of_text_token: Token = 50_256;
const start_of_transcript_token: Token = 50_257;
const no_speech_token: Token = 50_361;
const no_timestamps_token: Token = 50_362;

pub const EncoderTrailingPadding = log_mel.EncoderTrailingPadding;
pub const Token = decoder_module.Token;

pub const RuntimeError = error{
    AudioTooShort,
    AudioDurationExceedsLimit,
    InvalidPolicy,
    InvalidSamples,
    InvalidVocabulary,
    MemoryTooSmall,
    OutputTooSmall,
    ThreadSpawnFailed,
};

/// `Policy` fixes capacities and worker ownership for one runtime. These values
/// determine caller-owned memory size and cannot change until the runtime is
/// recreated; they do not select a transcription's logical encoder length.
pub const Policy = struct {
    /// Maximum 16 kHz samples accepted by one transcription.
    samples_count_max: usize = log_mel.samples_count_max,

    /// Maximum text tokens retained in caller-owned runtime memory.
    generated_tokens_count_max: usize = 224,

    /// Persistent executor threads; all participate in audio encoding.
    workers_count: usize = 4,

    /// Cross-K/V preparation and decoding use the first `decoder_workers_count`
    /// pool members. Null uses the full pool; a supplied count must be positive
    /// and no greater than `workers_count`. Unused members park during decoding.
    decoder_workers_count: ?usize = null,
};

/// `TranscribeOptions` selects behavior that does not alter runtime capacity.
/// Different calls on one runtime may use different trailing-padding values.
pub const TranscribeOptions = struct {
    /// Normalized silence appended after content before encoder execution.
    encoder_trailing_padding: EncoderTrailingPadding = .seconds_30,

    /// If supplied, transcription overwrites these wall-clock phase durations.
    /// The pointer is borrowed only until `transcribe` returns. Encoder and
    /// cross-K/V timing include dispatch; model loading and text decoding are
    /// excluded.
    timings: ?*Timings = null,

    /// Decoding evidence survives text conversion errors. Text and token storage
    /// remain borrowed until the next call; null means decoding did not finish.
    evidence: ?*?Transcription = null,
};

pub const Timings = struct {
    log_mel_ns: u64 = 0,
    encoder_ns: u64 = 0,
    cross_key_values_ns: u64 = 0,
    decoder_ns: u64 = 0,
};

pub const Transcription = struct {
    /// `text` aliases the supplied output without trimming token-leading spaces.
    text: []const u8,
    generated_tokens_count: usize,
    /// `end` distinguishes model completion from exhaustion of token capacity.
    /// A token-limited result is a prefix, not a complete transcript.
    end: enum { end_of_text, token_limit },

    /// Logical encoder sequence length used for this transcription. Standard
    /// 30-second Whisper input contains 1,500 positions.
    encoder_positions_count: usize,

    no_speech_probability: f32,
    average_log_probability: f32,
};

/// `Runtime` borrows an immutable model, vocabulary, and tensor memory until
/// `deinit` returns. Keep the control object at the same address from `init`
/// through `deinit`: persistent workers retain its executor's address.
pub const Runtime = struct {
    specification: ModelSpecification,
    weights: InferenceWeights,
    decoder_workers_count: usize,
    executor: Executor,
    extractor: log_mel.Extractor,
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    generated_tokens: []Token,

    /// `requiredMemorySize` returns the bytes required by `init` for `kind` and
    /// `policy`. The result excludes the packed model image and OS thread stacks.
    pub fn requiredMemorySize(kind: ModelKind, policy: Policy) RuntimeError!usize {
        const specification = kind.specification();
        try validatePolicy(specification, policy);

        return RuntimeMemoryLayout.init(specification, policy).size;
    }

    /// `init` binds caller-owned storage and starts persistent workers. `io`,
    /// `model`, `vocabulary_text`, and `memory` must outlive `deinit`. The vocabulary
    /// contains one UTF-8 token per model row in token-ID order. Call `deinit`
    /// only after successful initialization; failure leaves no running workers.
    pub fn init(runtime: *Runtime, io: std.Io, model: *const Model, vocabulary_text: []const u8, memory: []align(runtime_memory_alignment) u8, policy: Policy) RuntimeError!void {
        const specification = model.kind.specification();
        try validatePolicy(specification, policy);

        const layout = RuntimeMemoryLayout.init(specification, policy);
        if (memory.len < layout.size) {
            return error.MemoryTooSmall;
        }

        // ── Bind Caller Memory ──

        var extractor: log_mel.Extractor = undefined;
        try extractor.init(layout.extractor_memory.bind(memory), policy.samples_count_max);

        const vocabulary = try Vocabulary.init(vocabulary_text, layout.vocabulary_offsets.bind(memory));
        const encoder_query_key_value = layout.encoder_query_key_value.bind(memory);
        const lane_quantized_rows = layout.lane_quantized_rows.bind(memory);

        const encoder = Encoder.init(.{
            .activation = layout.encoder_activation.bind(memory),
            .shared_scratch = layout.encoder_shared_scratch.bind(memory),
            .query_key_value = encoder_query_key_value,
            .attention_output = layout.encoder_attention.bind(memory),
            .encoded_audio = layout.encoded_audio.bind(memory),
            .lane_float_rows = layout.lane_float_rows.bind(memory),
            .lane_quantized_rows = lane_quantized_rows,
        }, specification, policy.workers_count);

        const decoder = Decoder.init(.{
            .cross_key_values = layout.decoder_cross_key_values.bind(memory),
            .self_keys = layout.decoder_self_keys.bind(memory),
            .self_values = layout.decoder_self_values.bind(memory),
            .input = layout.decoder_input.bind(memory),
            .query_key_value = layout.decoder_query_key_value.bind(memory),
            .attention_output = layout.decoder_attention.bind(memory),
            .projection = layout.decoder_projection.bind(memory),
            .ffn = layout.decoder_ffn.bind(memory),
            .logits = layout.logits.bind(memory),
            .lane_quantized_rows = lane_quantized_rows,
            .lane_attention_scores = layout.lane_attention_scores.bind(memory),
        }, specification, decoderPositionsCapacity(policy), policy.workers_count);

        runtime.* = .{
            .specification = specification,
            .weights = model.inferenceWeights(),
            .decoder_workers_count = policy.decoder_workers_count orelse policy.workers_count,
            .executor = undefined,
            .extractor = extractor,
            .vocabulary = vocabulary,
            .encoder = encoder,
            .decoder = decoder,
            .generated_tokens = layout.generated_tokens.bind(memory),
        };

        runtime.executor.init(io, policy.workers_count) catch |err| switch (err) {
            error.InvalidWorkersCount => unreachable,
            error.ThreadSpawnFailed => {
                return error.ThreadSpawnFailed;
            },
        };
    }

    pub fn deinit(runtime: *Runtime) void {
        runtime.executor.deinit();
    }

    /// `transcribe` accepts finite mono samples in `[-1, 1]` at 16 kHz, writes
    /// one greedy English transcript into `text_output`, and returns a slice
    /// that aliases it. `options.encoder_trailing_padding` appends normalized
    /// silence after the content and caps the logical input at Whisper's
    /// 30-second context. The operation allocates nothing, reuses the runtime's
    /// sole inference slot, and is not reentrant. Later stages overwrite the
    /// intermediate features and encoded audio before this call returns.
    pub fn transcribe(runtime: *Runtime, samples: []const f32, text_output: []u8, options: TranscribeOptions) RuntimeError!Transcription {
        if (options.evidence) |evidence| evidence.* = null;
        if (options.timings) |timings| timings.* = .{};
        for (samples) |sample| {
            if (!std.math.isFinite(sample) or sample < -1.0 or sample > 1.0) {
                return error.InvalidSamples;
            }
        }

        const feature_start = if (options.timings != null) std.Io.Clock.awake.now(runtime.executor.io) else undefined;
        // Feature extraction and convolution finish before transformer Q/K/V
        // writes begin, so their workspace borrows that future tensor storage.
        const features = runtime.extractor.calculate(samples, options.encoder_trailing_padding, runtime.encoder.memory.query_key_value) catch |err| switch (err) {
            error.AudioTooShort => {
                return error.AudioTooShort;
            },
            error.AudioDurationExceedsLimit => {
                return error.AudioDurationExceedsLimit;
            },
            error.MemoryTooSmall => unreachable,
        };

        var context: TranscriptionContext = .{
            .runtime = runtime,
            .features = features,
            .timings = options.timings,
        };
        if (options.timings) |timings| {
            context.phase_start = std.Io.Clock.awake.now(runtime.executor.io);
            timings.* = .{ .log_mel_ns = @intCast(feature_start.durationTo(context.phase_start).nanoseconds) };
        }
        // Join the wide operation before narrowing: lane scratch views depend
        // on the active count, so decoding may repartition storage formerly
        // owned by encoder-only workers.
        runtime.executor.run(@ptrCast(&context), encodeWide);
        runtime.executor.runWithWorkers(runtime.decoder_workers_count, @ptrCast(&context), decodeWide);

        const scored_tokens_count = context.generated_tokens_count + @intFromBool(context.next_token == end_of_text_token);
        const average_log_probability = if (scored_tokens_count == 0) 0 else context.selected_log_probabilities_sum / @as(f32, @floatFromInt(scored_tokens_count));

        assert(context.generated_tokens_count <= runtime.generated_tokens.len);
        assert(std.math.isFinite(context.no_speech_probability));
        assert(context.no_speech_probability >= 0 and context.no_speech_probability <= 1);
        assert(std.math.isFinite(average_log_probability));
        assert(average_log_probability <= 0);

        var result: Transcription = .{
            .text = "",
            .generated_tokens_count = context.generated_tokens_count,
            .end = if (context.next_token == end_of_text_token) .end_of_text else .token_limit,
            .encoder_positions_count = features.encoderPositionsCount(),
            .no_speech_probability = context.no_speech_probability,
            .average_log_probability = average_log_probability,
        };
        var decoded_bytes: usize = 0;
        result.text = runtime.vocabulary.decode(runtime.generated_tokens[0..context.generated_tokens_count], text_output, &decoded_bytes) catch |err| {
            result.text = text_output[0..decoded_bytes];
            if (options.evidence) |evidence| evidence.* = result;
            return err;
        };
        if (options.evidence) |evidence| evidence.* = result;
        return result;
    }
};

const TranscriptionContext = struct {
    runtime: *Runtime,
    features: log_mel.Features,
    timings: ?*Timings = null,
    phase_start: std.Io.Timestamp = undefined,
    generated_tokens_count: usize = 0,
    next_token: Token = 0,
    no_speech_probability: f32 = 0,
    selected_log_probabilities_sum: f32 = 0,
};

fn encodeWide(raw_context: *anyopaque, lane: Lane) void {
    const context: *TranscriptionContext = @ptrCast(@alignCast(raw_context));
    const runtime = context.runtime;
    const specification = runtime.specification;
    const weights = &runtime.weights;

    // ── Encode Audio ──

    runtime.encoder.encode(specification, weights, context.features, lane);
    if (lane.isLeader()) {
        if (context.timings) |timings| {
            const now = std.Io.Clock.awake.now(runtime.executor.io);
            timings.encoder_ns = @intCast(context.phase_start.durationTo(now).nanoseconds);
            context.phase_start = now;
        }
    }
}

fn decodeWide(raw_context: *anyopaque, lane: Lane) void {
    const context: *TranscriptionContext = @ptrCast(@alignCast(raw_context));
    const runtime = context.runtime;
    const specification = runtime.specification;
    const weights = &runtime.weights;
    const encoder_positions_count = context.features.encoderPositionsCount();
    const encoded_audio = runtime.encoder.memory.encoded_audio[0 .. encoder_positions_count * specification.encoder_width];

    runtime.decoder.precomputeCrossKeyValues(specification, weights, encoded_audio, encoder_positions_count, lane);
    if (lane.isLeader()) {
        if (context.timings) |timings| {
            const now = std.Io.Clock.awake.now(runtime.executor.io);
            timings.cross_key_values_ns = @intCast(context.phase_start.durationTo(now).nanoseconds);
            context.phase_start = now;
        }
    }

    // ── Seed Decoder ──

    runtime.decoder.decodeToken(specification, weights, encoder_positions_count, start_of_transcript_token, 0, lane);
    if (lane.isLeader()) {
        context.no_speech_probability = probabilityOfToken(runtime.decoder.logits(), no_speech_token);
    }
    lane.sync();

    runtime.decoder.decodeToken(specification, weights, encoder_positions_count, no_timestamps_token, 1, lane);

    // ── Generate Greedy Tokens ──
    //
    // Every iteration selects from the logits produced by the previous token,
    // then feeds the selected text token to produce the next distribution.

    // Workers must enter the same iterations even if the leader advances the
    // shared token count before another worker checks the loop condition.
    const decoder_position_end = decoder_prompt_tokens_count + runtime.generated_tokens.len;
    var decoder_position = decoder_prompt_tokens_count;

    while (decoder_position < decoder_position_end) : (decoder_position += 1) {
        if (lane.isLeader()) {
            const selection = selectGreedyToken(runtime.decoder.logits(), context.generated_tokens_count == 0);
            context.next_token = selection.token;
            context.selected_log_probabilities_sum += selection.log_probability;

            if (selection.token != end_of_text_token) {
                runtime.generated_tokens[context.generated_tokens_count] = selection.token;
                context.generated_tokens_count += 1;
            }
        }
        lane.sync();

        // The final allowed token has already been selected and scored; its
        // next distribution would have no consumer. The local position keeps
        // every lane on the same side of the terminal check.
        if (context.next_token == end_of_text_token or decoder_position + 1 == decoder_position_end) {
            break;
        }

        runtime.decoder.decodeToken(specification, weights, encoder_positions_count, context.next_token, decoder_position, lane);
    }

    if (lane.isLeader()) {
        if (context.timings) |timings| {
            timings.decoder_ns = @intCast(context.phase_start.durationTo(std.Io.Clock.awake.now(runtime.executor.io)).nanoseconds);
        }
    }
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

const Vocabulary = struct {
    text: []const u8,
    token_offsets: []u32,

    fn init(text: []const u8, token_offsets: []u32) RuntimeError!Vocabulary {
        if (!std.unicode.utf8ValidateSlice(text)) {
            return error.InvalidVocabulary;
        }

        var tokens_count: usize = 0;
        var line_start_offset: usize = 0;

        while (line_start_offset < text.len) {
            if (tokens_count >= token_offsets.len - 1) {
                return error.InvalidVocabulary;
            }

            token_offsets[tokens_count] = @intCast(line_start_offset);
            const line_end_offset = std.mem.indexOfScalarPos(u8, text, line_start_offset, '\n') orelse text.len;
            line_start_offset = line_end_offset + @intFromBool(line_end_offset < text.len);
            tokens_count += 1;
        }

        if (tokens_count + 1 != token_offsets.len) {
            return error.InvalidVocabulary;
        }
        token_offsets[tokens_count] = @intCast(text.len);

        return .{ .text = text, .token_offsets = token_offsets };
    }

    fn decode(vocabulary: Vocabulary, tokens: []const Token, output: []u8, written: *usize) RuntimeError![]const u8 {
        var output_size: usize = 0;
        defer written.* = output_size;
        for (tokens) |token| {
            if (token >= end_of_text_token) {
                continue;
            }

            const token_index: usize = @intCast(token);
            const token_start_offset = vocabulary.token_offsets[token_index];
            var token_end_offset = vocabulary.token_offsets[token_index + 1];
            if (token_end_offset > token_start_offset and vocabulary.text[token_end_offset - 1] == '\n') {
                token_end_offset -= 1;
            }
            const encoded_token = vocabulary.text[token_start_offset..token_end_offset];

            const encoded_token_view = std.unicode.Utf8View.init(encoded_token) catch {
                return error.InvalidVocabulary;
            };
            var codepoints = encoded_token_view.iterator();
            while (codepoints.nextCodepoint()) |codepoint| {
                if (output_size == output.len) {
                    return error.OutputTooSmall;
                }

                const decoded_byte = gpt2ByteFromCodepoint(codepoint) orelse {
                    return error.InvalidVocabulary;
                };
                output[output_size] = decoded_byte;
                output_size += 1;
            }
        }

        const decoded = output[0..output_size];
        if (!std.unicode.utf8ValidateSlice(decoded)) {
            return error.InvalidVocabulary;
        }

        return decoded;
    }
};

fn gpt2ByteFromCodepoint(codepoint: u21) ?u8 {
    var mapped_codepoint: u16 = 0;
    for (0..256) |byte| {
        const byte_is_visible = (byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or (byte >= 174 and byte <= 255);
        if (byte_is_visible) {
            if (codepoint == byte) {
                return @intCast(byte);
            }
        } else {
            if (codepoint == 256 + mapped_codepoint) {
                return @intCast(byte);
            }
            mapped_codepoint += 1;
        }
    }

    return null;
}

pub const suppressed_first_tokens = [_]Token{ 220, end_of_text_token };
pub const suppressed_tokens = [_]Token{
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
};

// ─── Runtime Memory Layout ─────────────────────────────────────────────────

const RuntimeMemoryLayout = struct {
    extractor_memory: memory_layout.Region(u8),
    vocabulary_offsets: memory_layout.Region(u32),
    encoder_activation: memory_layout.Region(f32),
    encoder_shared_scratch: memory_layout.Region(f32),
    encoder_query_key_value: memory_layout.Region(f32),
    encoder_attention: memory_layout.Region(f32),
    encoded_audio: memory_layout.Region(f32),
    decoder_cross_key_values: memory_layout.Region(f16),
    decoder_self_keys: memory_layout.Region(f16),
    decoder_self_values: memory_layout.Region(f16),
    decoder_input: memory_layout.Region(f32),
    decoder_query_key_value: memory_layout.Region(f32),
    decoder_attention: memory_layout.Region(f32),
    decoder_projection: memory_layout.Region(f32),
    decoder_ffn: memory_layout.Region(f32),
    logits: memory_layout.Region(f32),
    generated_tokens: memory_layout.Region(Token),
    lane_float_rows: memory_layout.Region(f32),
    lane_quantized_rows: memory_layout.Region(u8),
    lane_attention_scores: memory_layout.Region(f32),
    size: usize,

    fn init(specification: ModelSpecification, policy: Policy) RuntimeMemoryLayout {
        const encoder_positions_count = specification.encoder_positions_count_max;
        const encoder_width = specification.encoder_width;
        const decoder_positions_capacity = decoderPositionsCapacity(policy);
        const decoder_width = specification.decoder_width;
        const decoder_layers_count = specification.decoder_layers_count;

        // ── Sequential Phase Storage ──
        //
        // Encoder tensors occupy future decoder-cache storage. Cross projection
        // reads encoded audio while writing cross K/V, then token decoding
        // replaces that audio with self K/V and decoder work. Encoder lane
        // floats die before the final normalization writes encoded audio, so
        // they may share the later decoder-work region as well.

        var encoder_phase_builder: memory_layout.Builder = .{};
        const encoder_activation_relative = encoder_phase_builder.add(f32, encoder_positions_count * encoder_width);
        const encoder_shared_scratch_relative = encoder_phase_builder.add(f32, encoder_module.sharedScratchValuesCount(policy.workers_count));
        const encoder_query_key_value_relative = encoder_phase_builder.add(f32, attention.encoderQueryKeyValueValuesCount(encoder_positions_count, encoder_width));
        const encoder_attention_relative = encoder_phase_builder.add(f32, encoder_positions_count * encoder_width);

        var decoder_phase_builder: memory_layout.Builder = .{};
        const decoder_cross_key_values_relative = decoder_phase_builder.add(f16, decoder_layers_count * (attention.packedKeyValuesCount(encoder_positions_count, decoder_width) + encoder_positions_count * decoder_width));
        var cross_projection_builder = decoder_phase_builder;
        const encoded_audio_relative = cross_projection_builder.add(f32, encoder_positions_count * encoder_width);

        const decoder_self_keys_relative = decoder_phase_builder.add(f16, decoder_layers_count * attention.packedKeyValuesCount(decoder_positions_capacity, decoder_width));
        const decoder_self_values_relative = decoder_phase_builder.add(f16, decoder_layers_count * decoder_positions_capacity * decoder_width);
        const decoder_input_relative = decoder_phase_builder.add(f32, decoder_width);
        const decoder_query_key_value_relative = decoder_phase_builder.add(f32, 3 * decoder_width);
        const decoder_attention_relative = decoder_phase_builder.add(f32, decoder_width);
        const decoder_projection_relative = decoder_phase_builder.add(f32, decoder_width);
        const decoder_ffn_relative = decoder_phase_builder.add(f32, specification.decoder_ffn_width);
        const logits_relative = decoder_phase_builder.add(f32, specification.vocabulary_tokens_count);
        const generated_tokens_relative = decoder_phase_builder.add(Token, policy.generated_tokens_count_max);
        const lane_attention_scores_relative = decoder_phase_builder.add(f32, attention.decoderScratchValuesCount(encoder_positions_count, specification.decoder_attention_heads_count, policy.workers_count));

        const lane_float_rows_relative: memory_layout.Region(f32) = .{
            .offset = decoder_input_relative.offset,
            .elements_count = policy.workers_count * encoder_module.laneFloatScratchValuesCount(specification),
        };
        const lane_float_end = lane_float_rows_relative.offset + lane_float_rows_relative.elements_count * @sizeOf(f32);
        const phase_memory_size = @max(@max(encoder_phase_builder.size, cross_projection_builder.size), @max(decoder_phase_builder.size, lane_float_end));
        assert(encoder_query_key_value_relative.elements_count >= log_mel.workspace_values_count);

        // ── Permanent Layout ──

        var builder: memory_layout.Builder = .{};
        const extractor_memory = builder.add(u8, log_mel.Extractor.requiredMemorySize(policy.samples_count_max));
        const vocabulary_offsets = builder.add(u32, specification.vocabulary_tokens_count + 1);
        const phase_memory = builder.add(u8, phase_memory_size);
        const lane_quantized_rows = builder.add(u8, policy.workers_count * encoder_module.laneQuantizedScratchValuesCount(specification));

        return .{
            .extractor_memory = extractor_memory,
            .vocabulary_offsets = vocabulary_offsets,
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
            .decoder_projection = decoder_projection_relative.rebase(phase_memory.offset),
            .decoder_ffn = decoder_ffn_relative.rebase(phase_memory.offset),
            .logits = logits_relative.rebase(phase_memory.offset),
            .generated_tokens = generated_tokens_relative.rebase(phase_memory.offset),
            .lane_float_rows = lane_float_rows_relative.rebase(phase_memory.offset),
            .lane_quantized_rows = lane_quantized_rows,
            .lane_attention_scores = lane_attention_scores_relative.rebase(phase_memory.offset),
            .size = builder.size,
        };
    }
};

fn decoderPositionsCapacity(policy: Policy) usize {
    assert(policy.generated_tokens_count_max > 0);

    return policy.generated_tokens_count_max + decoder_prompt_tokens_count;
}

fn validatePolicy(specification: ModelSpecification, policy: Policy) RuntimeError!void {
    if (policy.samples_count_max < log_mel.samples_count_min or policy.samples_count_max > log_mel.samples_count_max) {
        return error.InvalidPolicy;
    }
    if (policy.generated_tokens_count_max == 0 or policy.generated_tokens_count_max > specification.decoder_positions_count_max - decoder_prompt_tokens_count) {
        return error.InvalidPolicy;
    }
    if (policy.workers_count == 0 or policy.workers_count > executor_module.workers_count_max) {
        return error.InvalidPolicy;
    }
    if (policy.decoder_workers_count) |workers_count| {
        if (workers_count == 0 or workers_count > policy.workers_count) {
            return error.InvalidPolicy;
        }
    }
    if (specification.encoder_width != specification.decoder_width) {
        return error.InvalidPolicy;
    }
    if (specification.encoder_width % attention.head_width != 0 or specification.decoder_width % attention.head_width != 0) {
        return error.InvalidPolicy;
    }
}

test "decoder worker widths stay within the fixed pool and arena" {
    for ([_]ModelKind{ .base_en, .small_en }) |kind| {
        const size = try Runtime.requiredMemorySize(kind, .{ .workers_count = 8 });
        for ([_]usize{ 1, 3, 8 }) |width| {
            try std.testing.expectEqual(size, try Runtime.requiredMemorySize(kind, .{ .workers_count = 8, .decoder_workers_count = width }));
        }
        try std.testing.expectError(error.InvalidPolicy, Runtime.requiredMemorySize(kind, .{ .workers_count = 8, .decoder_workers_count = 0 }));
        try std.testing.expectError(error.InvalidPolicy, Runtime.requiredMemorySize(kind, .{ .workers_count = 8, .decoder_workers_count = 9 }));
    }
}

test "decoder token limits reject overflow before sizing memory" {
    for ([_]ModelKind{ .base_en, .small_en }) |kind| {
        // Both models have 448 positions, including two prompt tokens.
        for ([_]usize{ 1, 446 }) |limit| {
            _ = try Runtime.requiredMemorySize(kind, .{ .generated_tokens_count_max = limit });
        }
        for ([_]usize{ 0, 447, std.math.maxInt(usize) - 1, std.math.maxInt(usize) }) |limit| {
            try std.testing.expectError(error.InvalidPolicy, Runtime.requiredMemorySize(kind, .{ .generated_tokens_count_max = limit }));
        }
    }
}
