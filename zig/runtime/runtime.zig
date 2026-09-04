//! `Runtime` executes one complete English Whisper transcription from normalized 16 kHz samples. Initialization allocates one address-stable control object and binds caller-owned memory into all tensors; transcription performs no allocation.

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

pub const RuntimeInitError = RuntimeError || std.mem.Allocator.Error;

/// `Policy` fixes capacities and worker ownership for one runtime. These values
/// determine caller-owned memory size and cannot change until the runtime is
/// recreated; they do not select a transcription's logical encoder length.
pub const Policy = struct {
    /// Maximum 16 kHz samples accepted by one transcription.
    samples_count_max: usize = log_mel.samples_count_max,

    /// Maximum text tokens retained in caller-owned runtime memory.
    generated_tokens_count_max: usize = 224,

    /// Persistent executor threads cooperating on one transcription.
    workers_count: usize = 4,
};

/// `TranscribeOptions` selects behavior that does not alter runtime capacity.
/// Different calls on one runtime may use different trailing-padding values.
pub const TranscribeOptions = struct {
    /// Normalized silence appended after content before encoder execution.
    encoder_trailing_padding: EncoderTrailingPadding = .seconds_30,
};

pub const Transcription = struct {
    text: []const u8,
    generated_tokens_count: usize,

    /// Logical encoder sequence length used for this transcription. Standard
    /// 30-second Whisper input contains 1,500 positions.
    encoder_positions_count: usize,

    no_speech_probability: f32,
    average_log_probability: f32,
};

/// `Runtime` is allocated once so its persistent workers can retain the executor's address. It borrows one immutable model, vocabulary text, and caller-owned memory until `deinit` returns.
pub const Runtime = struct {
    specification: ModelSpecification,
    weights: InferenceWeights,
    generated_tokens_count_max: usize,
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

    /// `init` allocates the address-stable runtime, constructs every tensor view, and starts the persistent worker group. Pass the same allocator to `deinit`. `io`, `model`, `vocabulary_text`, and `memory` must remain valid until then. The vocabulary must contain exactly one UTF-8 token per model vocabulary row in token-ID order.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *const Model, vocabulary_text: []const u8, memory: []align(runtime_memory_alignment) u8, policy: Policy) RuntimeInitError!*Runtime {
        const specification = model.specification;
        try validatePolicy(specification, policy);

        const layout = RuntimeMemoryLayout.init(specification, policy);
        if (memory.len < layout.size) {
            return error.MemoryTooSmall;
        }

        // ── Bind Caller Memory ──

        var extractor: log_mel.Extractor = undefined;
        try extractor.init(layout.extractor_memory.bind(memory), policy.samples_count_max);

        const vocabulary = try Vocabulary.init(vocabulary_text, layout.vocabulary_offsets.bind(memory));
        const encoder_decoder_projection_scratch = layout.encoder_decoder_projection_scratch.bind(memory);
        const lane_quantized_rows = layout.lane_quantized_rows.bind(memory);

        const encoder = Encoder.init(.{
            .convolution_1_output = layout.encoder_convolution_1_output.bind(memory),
            .activation_0 = layout.encoder_activation_0.bind(memory),
            .activation_1 = layout.encoder_activation_1.bind(memory),
            .shared_scratch = layout.encoder_shared_scratch.bind(memory),
            .query_key_value = encoder_decoder_projection_scratch,
            .attention_output = layout.encoder_attention.bind(memory),
            .encoded_audio = layout.encoded_audio.bind(memory),
            .lane_float_rows = layout.lane_float_rows.bind(memory),
            .lane_quantized_rows = lane_quantized_rows,
        }, specification, policy.workers_count);

        // Encoding completes before decoder cache preparation. Those phases
        // reuse projection and quantization scratch but never overlap.
        const decoder = Decoder.init(.{
            .cross_projection_scratch = encoder_decoder_projection_scratch,
            .cross_key_values = layout.decoder_cross_key_values.bind(memory),
            .self_keys = layout.decoder_self_keys.bind(memory),
            .self_values = layout.decoder_self_values.bind(memory),
            .input = layout.decoder_input.bind(memory),
            .normalized = layout.decoder_normalized.bind(memory),
            .query_key_value = layout.decoder_query_key_value.bind(memory),
            .attention_output = layout.decoder_attention.bind(memory),
            .projection = layout.decoder_projection.bind(memory),
            .ffn = layout.decoder_ffn.bind(memory),
            .logits = layout.logits.bind(memory),
            .lane_quantized_rows = lane_quantized_rows,
            .lane_attention_scores = layout.lane_attention_scores.bind(memory),
        }, specification, policy.workers_count);

        const runtime = try allocator.create(Runtime);
        errdefer allocator.destroy(runtime);

        runtime.* = .{
            .specification = specification,
            .weights = model.inferenceWeights(),
            .generated_tokens_count_max = policy.generated_tokens_count_max,
            .executor = undefined,
            .extractor = extractor,
            .vocabulary = vocabulary,
            .encoder = encoder,
            .decoder = decoder,
            .generated_tokens = layout.generated_tokens.bind(memory),
        };

        runtime.executor.init(io, policy.workers_count) catch |err| switch (err) {
            error.InvalidWorkersCount => unreachable,
            error.ThreadSpawnFailed => return error.ThreadSpawnFailed,
        };

        return runtime;
    }

    pub fn deinit(runtime: *Runtime, allocator: std.mem.Allocator) void {
        runtime.executor.deinit();
        allocator.destroy(runtime);
    }

    /// `transcribe` accepts finite mono samples in `[-1, 1]` at 16 kHz, writes
    /// one greedy English transcript into `text_output`, and returns a slice
    /// that aliases it. `options.encoder_trailing_padding` appends normalized
    /// silence after the content and caps the logical input at Whisper's
    /// 30-second context. The operation allocates nothing, reuses the runtime's
    /// sole inference slot, and is not reentrant.
    pub fn transcribe(runtime: *Runtime, samples: []const f32, text_output: []u8, options: TranscribeOptions) RuntimeError!Transcription {
        for (samples) |sample| {
            if (!std.math.isFinite(sample) or sample < -1.0 or sample > 1.0) {
                return error.InvalidSamples;
            }
        }

        const features = runtime.extractor.calculate(samples, options.encoder_trailing_padding) catch |err| switch (err) {
            error.AudioTooShort => return error.AudioTooShort,
            error.AudioDurationExceedsLimit => return error.AudioDurationExceedsLimit,
            error.MemoryTooSmall => unreachable,
        };

        var context: TranscriptionContext = .{
            .runtime = runtime,
            .features = features,
        };
        runtime.executor.run(@ptrCast(&context), transcribeWide);

        assert(context.generated_tokens_count <= runtime.generated_tokens_count_max);
        assert(std.math.isFinite(context.no_speech_probability));
        assert(context.no_speech_probability >= 0 and context.no_speech_probability <= 1);
        assert(std.math.isFinite(context.average_log_probability));
        assert(context.average_log_probability <= 0);

        const text = try runtime.vocabulary.decode(runtime.generated_tokens[0..context.generated_tokens_count], text_output);

        return .{
            .text = std.mem.trim(u8, text, " \t\r\n"),
            .generated_tokens_count = context.generated_tokens_count,
            .encoder_positions_count = features.encoderPositionsCount(),
            .no_speech_probability = context.no_speech_probability,
            .average_log_probability = context.average_log_probability,
        };
    }
};

const TranscriptionContext = struct {
    runtime: *Runtime,
    features: log_mel.Features,
    generated_tokens_count: usize = 0,
    next_token: Token = 0,
    no_speech_probability: f32 = 0,
    selected_log_probabilities_sum: f32 = 0,
    average_log_probability: f32 = 0,
    decoding_is_complete: bool = false,
};

fn transcribeWide(raw_context: *anyopaque, lane: Lane) void {
    const context: *TranscriptionContext = @ptrCast(@alignCast(raw_context));
    const runtime = context.runtime;
    const specification = runtime.specification;
    const weights = &runtime.weights;

    // ── Encode Audio ──

    const encoded_audio = runtime.encoder.encode(specification, weights, context.features, lane);
    runtime.decoder.precomputeCrossKeyValues(specification, weights, encoded_audio.values, encoded_audio.positions_count, lane);

    // ── Seed Decoder ──

    runtime.decoder.decodeToken(specification, weights, encoded_audio.positions_count, start_of_transcript_token, 0, lane);
    if (lane.isLeader()) {
        context.no_speech_probability = probabilityOfToken(runtime.decoder.logits(), no_speech_token);
    }
    lane.sync();

    runtime.decoder.decodeToken(specification, weights, encoded_audio.positions_count, no_timestamps_token, 1, lane);

    // ── Generate Greedy Tokens ──
    //
    // Every iteration selects from the logits produced by the previous token,
    // then feeds the selected text token to produce the next distribution.

    var decoder_position = decoder_prompt_tokens_count;

    while (decoder_position < specification.decoder_positions_count_max and context.generated_tokens_count < runtime.generated_tokens_count_max) : (decoder_position += 1) {
        if (lane.isLeader()) {
            const selection = selectGreedyToken(runtime.decoder.logits(), context.generated_tokens_count == 0);
            context.next_token = selection.token;
            context.selected_log_probabilities_sum += selection.log_probability;
            context.decoding_is_complete = selection.token == end_of_text_token;

            if (!context.decoding_is_complete) {
                runtime.generated_tokens[context.generated_tokens_count] = selection.token;
                context.generated_tokens_count += 1;
            }
        }
        lane.sync();

        if (context.decoding_is_complete) {
            break;
        }

        runtime.decoder.decodeToken(specification, weights, encoded_audio.positions_count, context.next_token, decoder_position, lane);
    }

    if (lane.isLeader()) {
        const scored_tokens_count = context.generated_tokens_count + @intFromBool(context.decoding_is_complete);
        context.average_log_probability = if (scored_tokens_count == 0) 0 else context.selected_log_probabilities_sum / @as(f32, @floatFromInt(scored_tokens_count));
    }
    lane.sync();
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

    fn decode(vocabulary: Vocabulary, tokens: []const Token, output: []u8) RuntimeError![]const u8 {
        var output_size: usize = 0;
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

const suppressed_first_tokens = [_]Token{ 220, end_of_text_token };
const suppressed_tokens = [_]Token{
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
};

// ─── Runtime Memory Layout ─────────────────────────────────────────────────

const RuntimeMemoryLayout = struct {
    extractor_memory: memory_layout.Region(u8),
    vocabulary_offsets: memory_layout.Region(u32),
    encoder_convolution_1_output: memory_layout.Region(f32),
    encoder_activation_0: memory_layout.Region(f32),
    encoder_activation_1: memory_layout.Region(f32),
    encoder_shared_scratch: memory_layout.Region(f32),
    encoder_decoder_projection_scratch: memory_layout.Region(f32),
    encoder_attention: memory_layout.Region(f32),
    encoded_audio: memory_layout.Region(f32),
    decoder_cross_key_values: memory_layout.Region(f32),
    decoder_self_keys: memory_layout.Region(f32),
    decoder_self_values: memory_layout.Region(f32),
    decoder_input: memory_layout.Region(f32),
    decoder_normalized: memory_layout.Region(f32),
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
        const decoder_positions_count = specification.decoder_positions_count_max;
        const decoder_width = specification.decoder_width;
        const decoder_layers_count = specification.decoder_layers_count;
        const encoder_projection_values_count = encoder_positions_count * 3 * encoder_width;
        const decoder_cross_projection_values_count = encoder_positions_count * 2 * decoder_width;

        var builder: memory_layout.Builder = .{};
        const extractor_memory = builder.add(u8, log_mel.Extractor.requiredMemorySize(policy.samples_count_max));
        const vocabulary_offsets = builder.add(u32, specification.vocabulary_tokens_count + 1);
        const encoder_convolution_1_output = builder.add(f32, log_mel.encoder_frames_count_max * encoder_width);
        const encoder_activation_0 = builder.add(f32, encoder_positions_count * encoder_width);
        const encoder_activation_1 = builder.add(f32, encoder_positions_count * encoder_width);
        const encoder_shared_scratch = builder.add(f32, encoder_module.sharedScratchValuesCount(specification, policy.workers_count));
        const encoder_decoder_projection_scratch = builder.add(f32, @max(encoder_projection_values_count, decoder_cross_projection_values_count));
        const encoder_attention = builder.add(f32, encoder_positions_count * encoder_width);
        const encoded_audio = builder.add(f32, encoder_positions_count * encoder_width);
        const decoder_cross_key_values = builder.add(f32, decoder_layers_count * encoder_positions_count * 2 * decoder_width);
        const decoder_self_keys = builder.add(f32, decoder_layers_count * decoder_positions_count * decoder_width);
        const decoder_self_values = builder.add(f32, decoder_layers_count * decoder_positions_count * decoder_width);
        const decoder_input = builder.add(f32, decoder_width);
        const decoder_normalized = builder.add(f32, decoder_width);
        const decoder_query_key_value = builder.add(f32, 3 * decoder_width);
        const decoder_attention = builder.add(f32, decoder_width);
        const decoder_projection = builder.add(f32, decoder_width);
        const decoder_ffn = builder.add(f32, specification.decoder_ffn_width);
        const logits = builder.add(f32, specification.vocabulary_tokens_count);
        const generated_tokens = builder.add(Token, policy.generated_tokens_count_max);
        const lane_float_rows = builder.add(f32, policy.workers_count * encoder_module.laneFloatScratchValuesCount(specification));
        const lane_quantized_rows = builder.add(u8, policy.workers_count * encoder_module.laneQuantizedScratchValuesCount(specification));
        const lane_attention_scores = builder.add(f32, attention.decoderScratchValuesCount(encoder_positions_count, specification.decoder_attention_heads_count, policy.workers_count));

        return .{
            .extractor_memory = extractor_memory,
            .vocabulary_offsets = vocabulary_offsets,
            .encoder_convolution_1_output = encoder_convolution_1_output,
            .encoder_activation_0 = encoder_activation_0,
            .encoder_activation_1 = encoder_activation_1,
            .encoder_shared_scratch = encoder_shared_scratch,
            .encoder_decoder_projection_scratch = encoder_decoder_projection_scratch,
            .encoder_attention = encoder_attention,
            .encoded_audio = encoded_audio,
            .decoder_cross_key_values = decoder_cross_key_values,
            .decoder_self_keys = decoder_self_keys,
            .decoder_self_values = decoder_self_values,
            .decoder_input = decoder_input,
            .decoder_normalized = decoder_normalized,
            .decoder_query_key_value = decoder_query_key_value,
            .decoder_attention = decoder_attention,
            .decoder_projection = decoder_projection,
            .decoder_ffn = decoder_ffn,
            .logits = logits,
            .generated_tokens = generated_tokens,
            .lane_float_rows = lane_float_rows,
            .lane_quantized_rows = lane_quantized_rows,
            .lane_attention_scores = lane_attention_scores,
            .size = builder.size,
        };
    }
};

fn validatePolicy(specification: ModelSpecification, policy: Policy) RuntimeError!void {
    if (policy.samples_count_max < log_mel.samples_count_min or policy.samples_count_max > log_mel.samples_count_max) {
        return error.InvalidPolicy;
    }
    if (policy.generated_tokens_count_max == 0 or policy.generated_tokens_count_max + decoder_prompt_tokens_count > specification.decoder_positions_count_max) {
        return error.InvalidPolicy;
    }
    if (policy.workers_count == 0 or policy.workers_count > executor_module.workers_count_max) {
        return error.InvalidPolicy;
    }
    if (specification.encoder_width != specification.decoder_width) {
        return error.InvalidPolicy;
    }
    if (specification.encoder_width % attention.head_width != 0 or specification.decoder_width % attention.head_width != 0) {
        return error.InvalidPolicy;
    }
}
