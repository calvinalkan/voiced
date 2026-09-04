//! `Runtime` executes one complete English Whisper transcription from normalized 16 kHz samples. Initialization allocates one address-stable control object and partitions caller-owned memory into all tensors; transcription performs no allocation.

const std = @import("std");
const audio = @import("audio.zig");
const dense_kernel = @import("dense.zig");
const encoder_attention_kernel = @import("attention.zig");
const executor_module = @import("executor.zig");
const model_module = @import("model.zig");
const assert = std.debug.assert;

const Executor = executor_module.Executor;
const Lane = executor_module.Lane;
const Model = model_module.Model;
const ModelKind = model_module.ModelKind;
const ModelSpecification = model_module.ModelSpecification;
const InferenceWeights = model_module.InferenceWeights;
const QuantizedWeight = model_module.QuantizedWeight;

const sample_rate_hz: usize = 16_000;
const mel_frames_count: usize = 3000;
const convolution_kernel_width: usize = 3;
const attention_head_width: usize = encoder_attention_kernel.head_width;
pub const runtime_memory_alignment: usize = 64;
const memory_alignment = runtime_memory_alignment;
const simd_lanes_count: usize = 8;
const packed_output_rows_count: usize = 8;
const packed_depth_values_count: usize = 4;
const packed_group_bytes_count: usize = packed_output_rows_count * packed_depth_values_count;
const decoder_prompt_tokens_count: usize = 2;
const end_of_text_token: Token = 50_256;
const start_of_transcript_token: Token = 50_257;
const no_speech_token: Token = 50_361;
const no_timestamps_token: Token = 50_362;

const F32x8 = @Vector(simd_lanes_count, f32);

pub const Token = u32;

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

pub const Policy = struct {
    samples_count_max: usize = 30 * sample_rate_hz,
    generated_tokens_count_max: usize = 224,
    workers_count: usize = 4,
};

pub const Transcription = struct {
    text: []const u8,
    generated_tokens_count: usize,
    no_speech_probability: f32,
    average_log_probability: f32,
};

/// `Runtime` is allocated once so its persistent workers can retain the executor's address. It borrows one immutable model, vocabulary text, and caller-owned memory until `deinit` returns.
pub const Runtime = struct {
    model: *const Model,
    specification: ModelSpecification,
    weights: InferenceWeights,
    policy: Policy,
    memory: []align(memory_alignment) u8,
    executor: Executor,
    extractor: audio.Extractor,
    vocabulary: Vocabulary,
    convolution_1_weight: QuantizedWeight,
    convolution_2_weight: QuantizedWeight,
    convolution_1_packed_values: []i8,
    convolution_2_packed_values: []i8,
    convolution_1_output: []f32,
    encoder_activation_0: []f32,
    encoder_activation_1: []f32,
    encoder_normalized: []f32,
    encoder_query_key_value: []f32,
    encoder_attention: []f32,
    encoder_attention_scratch: []f32,
    encoded_audio: []f32,
    decoder_cross_key_values: []f32,
    decoder_self_keys: []f32,
    decoder_self_values: []f32,
    decoder_input: []f32,
    decoder_normalized: []f32,
    decoder_query_key_value: []f32,
    decoder_attention: []f32,
    decoder_projection: []f32,
    decoder_ffn: []f32,
    logits: []f32,
    generated_tokens: []Token,
    lane_float_rows: []f32,
    lane_quantized_rows: []u8,
    lane_attention_scores: []f32,

    /// `requiredMemorySize` returns the bytes required by `init` for `kind` and
    /// `policy`. The result excludes the packed model image and OS thread stacks.
    pub fn requiredMemorySize(kind: ModelKind, policy: Policy) RuntimeError!usize {
        try validatePolicy(kind.specification(), policy);

        const specification = kind.specification();
        const width = specification.encoder_width;
        const positions_count = specification.encoder_positions_count_max;
        const decoder_width = specification.decoder_width;
        const decoder_positions_count = specification.decoder_positions_count_max;
        const layers_count = specification.decoder_layers_count;
        const convolution_1_depth = specification.mel_bins_count * convolution_kernel_width;
        const convolution_2_depth = width * convolution_kernel_width;
        const lane_row_values_count = @max(specification.encoder_ffn_width, convolution_2_depth);
        const lane_float_scratch_values_count = @max(lane_row_values_count, dense_kernel.ffn_rows_per_tile * specification.encoder_ffn_width);
        const lane_quantized_scratch_values_count = dense_kernel.rows_per_tile * lane_row_values_count;
        const encoder_normalized_values_count = positions_count * width;
        const encoder_attention_scratch_values_count = encoder_attention_kernel.scratchValuesCount(positions_count, specification.encoder_attention_heads_count, policy.workers_count);
        const encoder_shared_scratch_values_count = @max(encoder_normalized_values_count, encoder_attention_scratch_values_count);

        var size: usize = 0;
        size = addMemoryRegion(size, u8, audio.Extractor.requiredMemorySize(policy.samples_count_max));
        size = addMemoryRegion(size, u8, width * convolution_1_depth);
        size = addMemoryRegion(size, u8, width * convolution_2_depth);
        size = addMemoryRegion(size, u32, specification.vocabulary_tokens_count + 1);
        size = addMemoryRegion(size, f32, mel_frames_count * width);
        size = addMemoryRegion(size, f32, positions_count * width);
        size = addMemoryRegion(size, f32, positions_count * width);
        size = addMemoryRegion(size, f32, encoder_shared_scratch_values_count);
        size = addMemoryRegion(size, f32, positions_count * 3 * width);
        size = addMemoryRegion(size, f32, positions_count * width);
        size = addMemoryRegion(size, f32, positions_count * width);
        size = addMemoryRegion(size, f32, layers_count * positions_count * 2 * decoder_width);
        size = addMemoryRegion(size, f32, layers_count * decoder_positions_count * decoder_width);
        size = addMemoryRegion(size, f32, layers_count * decoder_positions_count * decoder_width);
        size = addMemoryRegion(size, f32, decoder_width);
        size = addMemoryRegion(size, f32, decoder_width);
        size = addMemoryRegion(size, f32, 3 * decoder_width);
        size = addMemoryRegion(size, f32, decoder_width);
        size = addMemoryRegion(size, f32, decoder_width);
        size = addMemoryRegion(size, f32, specification.decoder_ffn_width);
        size = addMemoryRegion(size, f32, specification.vocabulary_tokens_count);
        size = addMemoryRegion(size, Token, policy.generated_tokens_count_max);
        size = addMemoryRegion(size, f32, policy.workers_count * lane_float_scratch_values_count);
        size = addMemoryRegion(size, u8, policy.workers_count * lane_quantized_scratch_values_count);
        size = addMemoryRegion(size, f32, policy.workers_count * positions_count);

        return size;
    }

    /// `init` allocates the address-stable runtime, constructs every tensor view, and starts the persistent worker group. Pass the same allocator to `deinit`. `io`, `model`, `vocabulary_text`, and `memory` must remain valid until then. The vocabulary must contain exactly one UTF-8 token per model vocabulary row in token-ID order.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *const Model, vocabulary_text: []const u8, memory: []align(memory_alignment) u8, policy: Policy) RuntimeInitError!*Runtime {
        try validatePolicy(model.specification, policy);

        const required_memory_size = try requiredMemorySize(model.kind, policy);
        if (memory.len < required_memory_size) {
            return error.MemoryTooSmall;
        }

        const specification = model.specification;
        const weights = model.inferenceWeights();
        const width = specification.encoder_width;
        const positions_count = specification.encoder_positions_count_max;
        const decoder_width = specification.decoder_width;
        const decoder_positions_count = specification.decoder_positions_count_max;
        const layers_count = specification.decoder_layers_count;
        const convolution_1_depth = specification.mel_bins_count * convolution_kernel_width;
        const convolution_2_depth = width * convolution_kernel_width;
        const lane_row_values_count = @max(specification.encoder_ffn_width, convolution_2_depth);
        const lane_float_scratch_values_count = @max(lane_row_values_count, dense_kernel.ffn_rows_per_tile * specification.encoder_ffn_width);
        const lane_quantized_scratch_values_count = dense_kernel.rows_per_tile * lane_row_values_count;
        const encoder_normalized_values_count = positions_count * width;
        const encoder_attention_scratch_values_count = encoder_attention_kernel.scratchValuesCount(positions_count, specification.encoder_attention_heads_count, policy.workers_count);
        const encoder_shared_scratch_values_count = @max(encoder_normalized_values_count, encoder_attention_scratch_values_count);

        // ── Partition Caller Memory ──

        var cursor: MemoryCursor = .{ .memory = memory };
        const extractor_memory = cursor.takeBytes(audio.Extractor.requiredMemorySize(policy.samples_count_max));
        const convolution_1_packed_values = cursor.take(i8, width * convolution_1_depth);
        const convolution_2_packed_values = cursor.take(i8, width * convolution_2_depth);
        const vocabulary_offsets = cursor.take(u32, specification.vocabulary_tokens_count + 1);
        const convolution_1_output = cursor.take(f32, mel_frames_count * width);
        const encoder_activation_0 = cursor.take(f32, positions_count * width);
        const encoder_activation_1 = cursor.take(f32, positions_count * width);

        // LayerNorm output dies after QKV projection. Blocked attention reuses
        // the same physical region instead of adding another persistent arena.
        const encoder_shared_scratch = cursor.take(f32, encoder_shared_scratch_values_count);
        const encoder_normalized = encoder_shared_scratch[0..encoder_normalized_values_count];
        const encoder_attention_scratch = encoder_shared_scratch[0..encoder_attention_scratch_values_count];
        const encoder_query_key_value = cursor.take(f32, positions_count * 3 * width);
        const encoder_attention = cursor.take(f32, positions_count * width);
        const encoded_audio = cursor.take(f32, positions_count * width);
        const decoder_cross_key_values = cursor.take(f32, layers_count * positions_count * 2 * decoder_width);
        const decoder_self_keys = cursor.take(f32, layers_count * decoder_positions_count * decoder_width);
        const decoder_self_values = cursor.take(f32, layers_count * decoder_positions_count * decoder_width);
        const decoder_input = cursor.take(f32, decoder_width);
        const decoder_normalized = cursor.take(f32, decoder_width);
        const decoder_query_key_value = cursor.take(f32, 3 * decoder_width);
        const decoder_attention = cursor.take(f32, decoder_width);
        const decoder_projection = cursor.take(f32, decoder_width);
        const decoder_ffn = cursor.take(f32, specification.decoder_ffn_width);
        const logits = cursor.take(f32, specification.vocabulary_tokens_count);
        const generated_tokens = cursor.take(Token, policy.generated_tokens_count_max);
        const lane_float_rows = cursor.take(f32, policy.workers_count * lane_float_scratch_values_count);
        const lane_quantized_rows = cursor.take(u8, policy.workers_count * lane_quantized_scratch_values_count);
        const lane_attention_scores = cursor.take(f32, policy.workers_count * positions_count);
        assert(cursor.offset <= memory.len);

        // ── Initialize Immutable Runtime State ──

        var extractor: audio.Extractor = undefined;
        try extractor.init(extractor_memory, policy.samples_count_max);

        const vocabulary = try Vocabulary.init(vocabulary_text, vocabulary_offsets);
        packRowMajorWeight(weights.encoder_convolution_1_weight, convolution_1_packed_values);
        packRowMajorWeight(weights.encoder_convolution_2_weight, convolution_2_packed_values);

        const runtime = try allocator.create(Runtime);
        errdefer allocator.destroy(runtime);

        runtime.* = .{
            .model = model,
            .specification = specification,
            .weights = weights,
            .policy = policy,
            .memory = memory,
            .executor = undefined,
            .extractor = extractor,
            .vocabulary = vocabulary,
            .convolution_1_weight = repackedWeight(weights.encoder_convolution_1_weight, convolution_1_packed_values),
            .convolution_2_weight = repackedWeight(weights.encoder_convolution_2_weight, convolution_2_packed_values),
            .convolution_1_packed_values = convolution_1_packed_values,
            .convolution_2_packed_values = convolution_2_packed_values,
            .convolution_1_output = convolution_1_output,
            .encoder_activation_0 = encoder_activation_0,
            .encoder_activation_1 = encoder_activation_1,
            .encoder_normalized = encoder_normalized,
            .encoder_query_key_value = encoder_query_key_value,
            .encoder_attention = encoder_attention,
            .encoder_attention_scratch = encoder_attention_scratch,
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
        };

        runtime.executor.init(io, policy.workers_count) catch {
            return error.ThreadSpawnFailed;
        };

        return runtime;
    }

    pub fn deinit(runtime: *Runtime, allocator: std.mem.Allocator) void {
        runtime.executor.deinit();
        allocator.destroy(runtime);
    }

    /// `transcribe` accepts finite mono samples in `[-1, 1]` at 16 kHz, writes
    /// one greedy English transcript into `text_output`, and returns a slice
    /// that aliases it. The operation reuses the runtime's sole inference slot
    /// and is not reentrant.
    pub fn transcribe(runtime: *Runtime, samples: []const f32, text_output: []u8) RuntimeError!Transcription {
        for (samples) |sample| {
            if (!std.math.isFinite(sample) or sample < -1.0 or sample > 1.0) {
                return error.InvalidSamples;
            }
        }

        const features = runtime.extractor.calculate(samples) catch |err| switch (err) {
            error.AudioTooShort => return error.AudioTooShort,
            error.AudioDurationExceedsLimit => return error.AudioDurationExceedsLimit,
            error.MemoryTooSmall => unreachable,
        };
        assert(features.frames_count == mel_frames_count);

        var context: TranscriptionContext = .{
            .runtime = runtime,
            .features = features.values,
        };
        runtime.executor.run(@ptrCast(&context), transcribeWide);

        assert(context.generated_tokens_count <= runtime.policy.generated_tokens_count_max);
        assert(std.math.isFinite(context.no_speech_probability));
        assert(context.no_speech_probability >= 0 and context.no_speech_probability <= 1);
        assert(std.math.isFinite(context.average_log_probability));
        assert(context.average_log_probability <= 0);

        const text = try runtime.vocabulary.decode(runtime.generated_tokens[0..context.generated_tokens_count], text_output);

        return .{
            .text = std.mem.trim(u8, text, " \t\r\n"),
            .generated_tokens_count = context.generated_tokens_count,
            .no_speech_probability = context.no_speech_probability,
            .average_log_probability = context.average_log_probability,
        };
    }
};

const TranscriptionContext = struct {
    runtime: *Runtime,
    features: []const f32,
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

    // ── Encode Audio ──

    encodeAudio(runtime, context.features, lane);
    precomputeDecoderCrossKeyValues(runtime, lane);

    // ── Seed Decoder ──

    decodeToken(runtime, start_of_transcript_token, 0, lane);
    if (lane.isLeader()) {
        context.no_speech_probability = probabilityOfToken(runtime.logits, no_speech_token);
    }
    lane.sync();

    decodeToken(runtime, no_timestamps_token, 1, lane);

    // ── Generate Greedy Tokens ──
    //
    // Every iteration selects from the logits produced by the previous token,
    // then feeds the selected text token to produce the next distribution.

    var decoder_position = decoder_prompt_tokens_count;

    while (decoder_position < specification.decoder_positions_count_max and context.generated_tokens_count < runtime.policy.generated_tokens_count_max) : (decoder_position += 1) {
        if (lane.isLeader()) {
            const selection = selectGreedyToken(runtime.logits, context.generated_tokens_count == 0);
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

        decodeToken(runtime, context.next_token, decoder_position, lane);
    }

    if (lane.isLeader()) {
        const scored_tokens_count = context.generated_tokens_count + @intFromBool(context.decoding_is_complete);
        context.average_log_probability = if (scored_tokens_count == 0) 0 else context.selected_log_probabilities_sum / @as(f32, @floatFromInt(scored_tokens_count));
    }
    lane.sync();
}

// ─── Encoder ───────────────────────────────────────────────────────────────

fn encodeAudio(runtime: *Runtime, features: []const f32, lane: Lane) void {
    const specification = runtime.specification;
    const width = specification.encoder_width;
    const positions_count = specification.encoder_positions_count_max;

    convolution(runtime, features, mel_frames_count, specification.mel_bins_count, runtime.convolution_1_weight, runtime.weights.encoder_convolution_1_bias, 1, true, runtime.convolution_1_output, lane);
    lane.sync();

    convolution(runtime, runtime.convolution_1_output, mel_frames_count, width, runtime.convolution_2_weight, runtime.weights.encoder_convolution_2_bias, 2, false, runtime.encoder_activation_0, lane);
    lane.sync();

    const position_values_range = lane.range(positions_count * width);
    for (position_values_range.start_index..position_values_range.end_index) |value_index| {
        runtime.encoder_activation_0[value_index] += runtime.weights.encoder_position_encodings[value_index];
    }
    lane.sync();

    var layer_input = runtime.encoder_activation_0;
    var layer_output = runtime.encoder_activation_1;
    for (runtime.weights.encoder_layers[0..specification.encoder_layers_count]) |layer_weights| {
        dense_kernel.forwardNormalizedRows(layer_input, layer_weights.self_attention_layer_norm_gamma, layer_weights.self_attention_layer_norm_beta, positions_count, layer_weights.self_attention_query_key_value_weight, layer_weights.self_attention_query_key_value_bias, runtime.encoder_query_key_value, laneQuantizedRow(runtime, lane), lane);
        lane.sync();

        encoderAttention(runtime, runtime.encoder_query_key_value, runtime.encoder_attention, lane);
        lane.sync();

        dense(runtime, runtime.encoder_attention, positions_count, layer_weights.self_attention_output_weight, layer_weights.self_attention_output_bias, .none, layer_output, lane);
        lane.sync();

        addInto(layer_input, layer_output, lane);
        lane.sync();

        layerNorm(layer_output, layer_weights.ffn_layer_norm_gamma, layer_weights.ffn_layer_norm_beta, positions_count, width, runtime.encoder_normalized, lane);
        lane.sync();

        dense_kernel.forwardFeedForwardRows(runtime.encoder_normalized, positions_count, layer_weights.ffn_expansion_weight, layer_weights.ffn_expansion_bias, layer_weights.ffn_contraction_weight, layer_weights.ffn_contraction_bias, runtime.encoder_attention, laneFloatRow(runtime, lane), laneQuantizedRow(runtime, lane), lane);
        lane.sync();

        addInto(runtime.encoder_attention, layer_output, lane);
        lane.sync();

        const previous_input = layer_input;
        layer_input = layer_output;
        layer_output = previous_input;
    }

    layerNorm(layer_input, runtime.weights.encoder_layer_norm_gamma, runtime.weights.encoder_layer_norm_beta, positions_count, width, runtime.encoded_audio, lane);
    lane.sync();
}

fn convolution(runtime: *Runtime, input: []const f32, input_positions_count: usize, input_channels_count: usize, weight: QuantizedWeight, bias: []const f32, stride: usize, input_is_channel_major: bool, output: []f32, lane: Lane) void {
    const output_positions_count = @divFloor(input_positions_count + 2 - convolution_kernel_width, stride) + 1;
    const output_channels_count = weight.output_rows_count;
    const input_depth = input_channels_count * convolution_kernel_width;
    assert(weight.input_values_count == input_depth);
    assert(output.len == output_positions_count * output_channels_count);

    const output_positions_range = lane.range(output_positions_count);
    const float_row = laneFloatRow(runtime, lane)[0..input_depth];
    const quantized_row = laneQuantizedRow(runtime, lane)[0..input_depth];

    for (output_positions_range.start_index..output_positions_range.end_index) |output_position| {
        const input_origin = @as(isize, @intCast(output_position * stride)) - 1;
        for (0..input_channels_count) |channel_index| {
            for (0..convolution_kernel_width) |kernel_index| {
                const input_position = input_origin + @as(isize, @intCast(kernel_index));
                const row_index = channel_index * convolution_kernel_width + kernel_index;
                if (input_position < 0 or input_position >= input_positions_count) {
                    float_row[row_index] = 0;
                } else {
                    const position_index: usize = @intCast(input_position);
                    const input_index = if (input_is_channel_major) channel_index * input_positions_count + position_index else position_index * input_channels_count + channel_index;
                    float_row[row_index] = input[input_index];
                }
            }
        }

        const input_scale = dense_kernel.quantizeRow(float_row, quantized_row);
        const output_row = output[output_position * output_channels_count ..][0..output_channels_count];
        dense_kernel.forwardQuantizedOne(quantized_row, input_scale, weight, bias, .gelu, output_row);
    }
}

fn encoderAttention(runtime: *Runtime, query_key_value: []const f32, output: []f32, lane: Lane) void {
    const specification = runtime.specification;
    encoder_attention_kernel.forward(query_key_value, specification.encoder_positions_count_max, specification.encoder_width, specification.encoder_attention_heads_count, output, runtime.encoder_attention_scratch, lane);
}

fn precomputeDecoderCrossKeyValues(runtime: *Runtime, lane: Lane) void {
    const specification = runtime.specification;
    const positions_count = specification.encoder_positions_count_max;
    const width = specification.decoder_width;
    const heads_count = specification.decoder_attention_heads_count;
    const projection_values_count = positions_count * 2 * width;
    const head_values_count = positions_count * attention_head_width;
    const projected_key_values = runtime.encoder_query_key_value[0..projection_values_count];

    for (runtime.weights.decoder_layers[0..specification.decoder_layers_count], 0..) |layer_weights, layer_index| {
        dense(runtime, runtime.encoded_audio, positions_count, layer_weights.cross_attention_key_value_weight, layer_weights.cross_attention_key_value_bias, .none, projected_key_values, lane);
        lane.sync();

        // Decoder attention revisits these values for every generated token.
        // Store K and V head-major once so each head reads contiguous rows.
        const head_major_key_values = runtime.decoder_cross_key_values[layer_index * projection_values_count ..][0..projection_values_count];
        const projection_heads_range = lane.range(2 * heads_count);
        for (projection_heads_range.start_index..projection_heads_range.end_index) |projection_head_index| {
            const projection_index = projection_head_index / heads_count;
            const head_index = projection_head_index % heads_count;
            const destination_head_offset = projection_index * positions_count * width + head_index * head_values_count;
            const source_column = projection_index * width + head_index * attention_head_width;
            for (0..positions_count) |position_index| {
                const source_offset = position_index * 2 * width + source_column;
                const destination_offset = destination_head_offset + position_index * attention_head_width;
                var depth: usize = 0;
                while (depth < attention_head_width) : (depth += simd_lanes_count) {
                    head_major_key_values[destination_offset + depth ..][0..simd_lanes_count].* = projected_key_values[source_offset + depth ..][0..simd_lanes_count].*;
                }
            }
        }
        lane.sync();
    }
}

// ─── Decoder ───────────────────────────────────────────────────────────────

fn decodeToken(runtime: *Runtime, token: Token, decoder_position: usize, lane: Lane) void {
    const specification = runtime.specification;
    const width = specification.decoder_width;
    assert(token < specification.vocabulary_tokens_count);
    assert(decoder_position < specification.decoder_positions_count_max);

    const input_range = lane.range(width);
    for (input_range.start_index..input_range.end_index) |column| {
        runtime.decoder_input[column] = embeddingValue(runtime.weights.decoder_embeddings_weight, token, column) + runtime.weights.decoder_position_encodings[decoder_position * width + column];
    }
    lane.sync();

    for (runtime.weights.decoder_layers[0..specification.decoder_layers_count], 0..) |layer_weights, layer_index| {
        // ── Self-Attention ──

        layerNorm(runtime.decoder_input, layer_weights.self_attention_layer_norm_gamma, layer_weights.self_attention_layer_norm_beta, 1, width, runtime.decoder_normalized, lane);
        lane.sync();

        dense(runtime, runtime.decoder_normalized, 1, layer_weights.self_attention_query_key_value_weight, layer_weights.self_attention_query_key_value_bias, .none, runtime.decoder_query_key_value, lane);
        lane.sync();

        storeDecoderSelfKeyValues(runtime, layer_index, decoder_position, lane);
        lane.sync();

        decoderSelfAttention(runtime, layer_index, decoder_position, lane);
        lane.sync();

        dense(runtime, runtime.decoder_attention, 1, layer_weights.self_attention_output_weight, layer_weights.self_attention_output_bias, .none, runtime.decoder_projection, lane);
        lane.sync();

        addInto(runtime.decoder_projection, runtime.decoder_input, lane);
        lane.sync();

        // ── Cross-Attention ──

        layerNorm(runtime.decoder_input, layer_weights.cross_attention_layer_norm_gamma, layer_weights.cross_attention_layer_norm_beta, 1, width, runtime.decoder_normalized, lane);
        lane.sync();

        dense(runtime, runtime.decoder_normalized, 1, layer_weights.cross_attention_query_weight, layer_weights.cross_attention_query_bias, .none, runtime.decoder_query_key_value[0..width], lane);
        lane.sync();

        decoderCrossAttention(runtime, layer_index, lane);
        lane.sync();

        dense(runtime, runtime.decoder_attention, 1, layer_weights.cross_attention_output_weight, layer_weights.cross_attention_output_bias, .none, runtime.decoder_projection, lane);
        lane.sync();

        addInto(runtime.decoder_projection, runtime.decoder_input, lane);
        lane.sync();

        // ── Feed-Forward Network ──

        layerNorm(runtime.decoder_input, layer_weights.ffn_layer_norm_gamma, layer_weights.ffn_layer_norm_beta, 1, width, runtime.decoder_normalized, lane);
        lane.sync();

        dense(runtime, runtime.decoder_normalized, 1, layer_weights.ffn_expansion_weight, layer_weights.ffn_expansion_bias, .gelu, runtime.decoder_ffn, lane);
        lane.sync();

        dense(runtime, runtime.decoder_ffn, 1, layer_weights.ffn_contraction_weight, layer_weights.ffn_contraction_bias, .none, runtime.decoder_projection, lane);
        lane.sync();

        addInto(runtime.decoder_projection, runtime.decoder_input, lane);
        lane.sync();
    }

    layerNorm(runtime.decoder_input, runtime.weights.decoder_layer_norm_gamma, runtime.weights.decoder_layer_norm_beta, 1, width, runtime.decoder_normalized, lane);
    lane.sync();

    dense(runtime, runtime.decoder_normalized, 1, runtime.weights.decoder_embeddings_weight, &.{}, .none, runtime.logits, lane);
    lane.sync();
}

fn storeDecoderSelfKeyValues(runtime: *Runtime, layer_index: usize, decoder_position: usize, lane: Lane) void {
    const specification = runtime.specification;
    const width = specification.decoder_width;
    const positions_capacity = specification.decoder_positions_count_max;
    const layer_offset = layer_index * positions_capacity * width;
    const columns_range = lane.range(width);

    for (columns_range.start_index..columns_range.end_index) |column| {
        const head_index = column / attention_head_width;
        const head_depth = column % attention_head_width;
        const cache_offset = layer_offset + head_index * positions_capacity * attention_head_width + decoder_position * attention_head_width + head_depth;
        runtime.decoder_self_keys[cache_offset] = runtime.decoder_query_key_value[width + column];
        runtime.decoder_self_values[cache_offset] = runtime.decoder_query_key_value[2 * width + column];
    }
}

fn decoderSelfAttention(runtime: *Runtime, layer_index: usize, decoder_position: usize, lane: Lane) void {
    const specification = runtime.specification;
    const width = specification.decoder_width;
    const positions_capacity = specification.decoder_positions_count_max;
    const layer_offset = layer_index * positions_capacity * width;
    const keys = runtime.decoder_self_keys[layer_offset..][0 .. positions_capacity * width];
    const values = runtime.decoder_self_values[layer_offset..][0 .. positions_capacity * width];
    encoder_attention_kernel.decoderForward(runtime.decoder_query_key_value[0..width], keys, values, decoder_position + 1, positions_capacity, width, specification.decoder_attention_heads_count, runtime.decoder_attention, runtime.lane_attention_scores, lane);
}

fn decoderCrossAttention(runtime: *Runtime, layer_index: usize, lane: Lane) void {
    const specification = runtime.specification;
    const width = specification.decoder_width;
    const positions_count = specification.encoder_positions_count_max;
    const tensor_values_count = positions_count * width;
    const layer_values_count = 2 * tensor_values_count;
    const key_values = runtime.decoder_cross_key_values[layer_index * layer_values_count ..][0..layer_values_count];
    const keys = key_values[0..tensor_values_count];
    const values = key_values[tensor_values_count..][0..tensor_values_count];
    encoder_attention_kernel.decoderForward(runtime.decoder_query_key_value[0..width], keys, values, positions_count, positions_count, width, specification.decoder_attention_heads_count, runtime.decoder_attention, runtime.lane_attention_scores, lane);
}

// ─── Numerical Operations ──────────────────────────────────────────────────

fn dense(runtime: *Runtime, input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, activation: dense_kernel.Activation, output: []f32, lane: Lane) void {
    const quantized_scratch = laneQuantizedRow(runtime, lane);
    if (rows_count == 1) {
        dense_kernel.forwardOne(input, weight, bias, activation, output, quantized_scratch, lane);
        return;
    }

    dense_kernel.forwardRows(input, rows_count, weight, bias, activation, output, quantized_scratch, lane);
}

fn layerNorm(input: []const f32, gamma: []const f32, beta: []const f32, rows_count: usize, width: usize, output: []f32, lane: Lane) void {
    assert(input.len == rows_count * width);
    assert(output.len == input.len);
    assert(gamma.len == width);
    assert(beta.len == width);
    assert(width % (4 * simd_lanes_count) == 0);

    const rows_range = lane.range(rows_count);
    for (rows_range.start_index..rows_range.end_index) |row_index| {
        const input_row = input[row_index * width ..][0..width];
        const output_row = output[row_index * width ..][0..width];

        var sums_0: F32x8 = @splat(0);
        var sums_1: F32x8 = @splat(0);
        var sums_2: F32x8 = @splat(0);
        var sums_3: F32x8 = @splat(0);

        var square_sums_0: F32x8 = @splat(0);
        var square_sums_1: F32x8 = @splat(0);
        var square_sums_2: F32x8 = @splat(0);
        var square_sums_3: F32x8 = @splat(0);

        var column: usize = 0;

        while (column < width) : (column += 4 * simd_lanes_count) {
            const values_0: F32x8 = input_row[column + 0 * simd_lanes_count ..][0..simd_lanes_count].*;
            const values_1: F32x8 = input_row[column + 1 * simd_lanes_count ..][0..simd_lanes_count].*;
            const values_2: F32x8 = input_row[column + 2 * simd_lanes_count ..][0..simd_lanes_count].*;
            const values_3: F32x8 = input_row[column + 3 * simd_lanes_count ..][0..simd_lanes_count].*;
            sums_0 += values_0;
            sums_1 += values_1;
            sums_2 += values_2;
            sums_3 += values_3;
            square_sums_0 = @mulAdd(F32x8, values_0, values_0, square_sums_0);
            square_sums_1 = @mulAdd(F32x8, values_1, values_1, square_sums_1);
            square_sums_2 = @mulAdd(F32x8, values_2, values_2, square_sums_2);
            square_sums_3 = @mulAdd(F32x8, values_3, values_3, square_sums_3);
        }

        const reciprocal_width = 1.0 / @as(f32, @floatFromInt(width));
        const mean = reduceAdd(((sums_0 + sums_1) + sums_2) + sums_3) * reciprocal_width;
        const mean_square = reduceAdd((square_sums_0 + square_sums_1) + (square_sums_2 + square_sums_3)) * reciprocal_width;
        const variance = @max(@mulAdd(f32, -mean, mean, mean_square), 0.0);
        const reciprocal_standard_deviation = 1.0 / @sqrt(variance + 1.0e-5);
        const means: F32x8 = @splat(mean);
        const reciprocal_standard_deviations: F32x8 = @splat(reciprocal_standard_deviation);

        column = 0;
        while (column < width) : (column += simd_lanes_count) {
            const values: F32x8 = input_row[column..][0..simd_lanes_count].*;
            const gammas: F32x8 = gamma[column..][0..simd_lanes_count].*;
            const betas: F32x8 = beta[column..][0..simd_lanes_count].*;
            output_row[column..][0..simd_lanes_count].* = @mulAdd(F32x8, (values - means) * gammas, reciprocal_standard_deviations, betas);
        }
    }
}

fn addInto(addend: []const f32, output: []f32, lane: Lane) void {
    assert(addend.len == output.len);

    const values_range = lane.range(output.len);
    var value_index = values_range.start_index;

    while (value_index + simd_lanes_count <= values_range.end_index) : (value_index += simd_lanes_count) {
        const left: F32x8 = output[value_index..][0..simd_lanes_count].*;
        const right: F32x8 = addend[value_index..][0..simd_lanes_count].*;
        output[value_index..][0..simd_lanes_count].* = left + right;
    }

    while (value_index < values_range.end_index) : (value_index += 1) {
        output[value_index] += addend[value_index];
    }
}

inline fn reduceAdd(values: F32x8) f32 {
    var reduced = values;
    var shuffled: F32x8 = undefined;

    asm volatile (
        \\ vperm2f128 $0x1, %[reduced], %[reduced], %[shuffled]
        \\ vaddps %[shuffled], %[reduced], %[reduced]
        \\ vshufps $0x4e, %[reduced], %[reduced], %[shuffled]
        \\ vaddps %[shuffled], %[reduced], %[reduced]
        \\ vshufps $0xb1, %[reduced], %[reduced], %[shuffled]
        \\ vaddps %[shuffled], %[reduced], %[reduced]
        : [reduced] "+x" (reduced),
          [shuffled] "=&x" (shuffled),
    );
    return reduced[0];
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

// ─── Packed Weight And Memory Support ──────────────────────────────────────

fn embeddingValue(weight: QuantizedWeight, token: Token, column: usize) f32 {
    assert(weight.encoding == .vnni_o8_k4);
    assert(token < weight.output_rows_count);
    assert(column < weight.input_values_count);

    const output_row: usize = @intCast(token);
    const output_block_index = output_row / packed_output_rows_count;
    const block_row_index = output_row % packed_output_rows_count;
    const depth_group_index = column / packed_depth_values_count;
    const depth_group_value_index = column % packed_depth_values_count;
    const output_block_size = packed_output_rows_count * weight.input_values_count;
    const packed_offset = output_block_index * output_block_size + depth_group_index * packed_group_bytes_count + block_row_index * packed_depth_values_count + depth_group_value_index;

    return @as(f32, @floatFromInt(weight.values[packed_offset])) / weight.scales[output_row];
}

fn packRowMajorWeight(weight: QuantizedWeight, destination: []i8) void {
    assert(weight.encoding == .row_major);
    assert(destination.len == weight.values.len);
    assert(weight.output_rows_count % packed_output_rows_count == 0);
    assert(weight.input_values_count % packed_depth_values_count == 0);

    const output_block_size = packed_output_rows_count * weight.input_values_count;
    var output_row_start: usize = 0;
    while (output_row_start < weight.output_rows_count) : (output_row_start += packed_output_rows_count) {
        const destination_block = destination[output_row_start / packed_output_rows_count * output_block_size ..][0..output_block_size];
        var depth_start: usize = 0;
        while (depth_start < weight.input_values_count) : (depth_start += packed_depth_values_count) {
            const destination_group = destination_block[depth_start / packed_depth_values_count * packed_group_bytes_count ..][0..packed_group_bytes_count];
            for (0..packed_output_rows_count) |block_row_index| {
                const source_offset = (output_row_start + block_row_index) * weight.input_values_count + depth_start;
                const destination_offset = block_row_index * packed_depth_values_count;
                @memcpy(destination_group[destination_offset..][0..packed_depth_values_count], weight.values[source_offset..][0..packed_depth_values_count]);
            }
        }
    }
}

fn repackedWeight(weight: QuantizedWeight, packed_values: []const i8) QuantizedWeight {
    assert(weight.encoding == .row_major);
    assert(weight.values.len == packed_values.len);

    return .{
        .values = packed_values,
        .scales = weight.scales,
        .compensation = weight.compensation,
        .output_rows_count = weight.output_rows_count,
        .input_values_count = weight.input_values_count,
        .encoding = .vnni_o8_k4,
    };
}

fn laneFloatRow(runtime: *Runtime, lane: Lane) []f32 {
    const values_count = runtime.lane_float_rows.len / lane.count;
    return runtime.lane_float_rows[lane.index * values_count ..][0..values_count];
}

fn laneQuantizedRow(runtime: *Runtime, lane: Lane) []u8 {
    const values_count = runtime.lane_quantized_rows.len / lane.count;
    return runtime.lane_quantized_rows[lane.index * values_count ..][0..values_count];
}

fn validatePolicy(specification: ModelSpecification, policy: Policy) RuntimeError!void {
    if (policy.samples_count_max < 400 / 2 + 1 or policy.samples_count_max > 30 * sample_rate_hz) {
        return error.InvalidPolicy;
    }
    if (policy.generated_tokens_count_max == 0 or policy.generated_tokens_count_max + decoder_prompt_tokens_count > specification.decoder_positions_count_max) {
        return error.InvalidPolicy;
    }
    if (policy.workers_count == 0 or policy.workers_count > 32) {
        return error.InvalidPolicy;
    }
    if (specification.encoder_width != specification.decoder_width) {
        return error.InvalidPolicy;
    }
    if (specification.encoder_width % attention_head_width != 0 or specification.decoder_width % attention_head_width != 0) {
        return error.InvalidPolicy;
    }
}

fn addMemoryRegion(size: usize, comptime Element: type, elements_count: usize) usize {
    const aligned_size = std.mem.alignForward(usize, size, memory_alignment);

    return aligned_size + elements_count * @sizeOf(Element);
}

const MemoryCursor = struct {
    memory: []align(memory_alignment) u8,
    offset: usize = 0,

    fn takeBytes(cursor: *MemoryCursor, bytes_count: usize) []u8 {
        return cursor.take(u8, bytes_count);
    }

    fn take(cursor: *MemoryCursor, comptime Element: type, elements_count: usize) []Element {
        cursor.offset = std.mem.alignForward(usize, cursor.offset, memory_alignment);
        const region_size = elements_count * @sizeOf(Element);
        assert(cursor.offset + region_size <= cursor.memory.len);

        const values: [*]Element = @ptrCast(@alignCast(cursor.memory.ptr + cursor.offset));
        cursor.offset += region_size;

        return values[0..elements_count];
    }
};
