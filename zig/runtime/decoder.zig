//! The Whisper decoder prepares cross-attention storage once, then advances one
//! token through explicit self-attention, cross-attention, and feed-forward
//! stages while retaining its head-major caches.

const std = @import("std");
const attention = @import("attention.zig");
const linear = @import("linear.zig");
const model_module = @import("model.zig");
const normalization = @import("normalization.zig");
const vnni_weight = @import("vnni_weight.zig");
const Lane = @import("executor.zig").Lane;
const InferenceWeights = model_module.InferenceWeights;
const ModelSpecification = model_module.ModelSpecification;
const QuantizedWeight = vnni_weight.QuantizedWeight;
const assert = std.debug.assert;

const attention_head_width: usize = attention.head_width;
const simd_lanes_count: usize = 8;

const F32x8 = @Vector(simd_lanes_count, f32);

pub const Token = u32;

pub const Memory = struct {
    cross_projection_scratch: []f32,
    cross_key_values: []f32,
    self_keys: []f32,
    self_values: []f32,
    input: []f32,
    normalized: []f32,
    query_key_value: []f32,
    attention_output: []f32,
    projection: []f32,
    ffn: []f32,
    logits: []f32,
    lane_quantized_rows: []u8,
    lane_attention_scores: []f32,
};

pub const Decoder = struct {
    memory: Memory,

    pub fn init(memory: Memory, specification: ModelSpecification, workers_count: usize) Decoder {
        const encoder_positions_count = specification.encoder_positions_count_max;
        const decoder_positions_count = specification.decoder_positions_count_max;
        const width = specification.decoder_width;
        const layers_count = specification.decoder_layers_count;

        assert(memory.cross_projection_scratch.len >= encoder_positions_count * 2 * width);
        assert(memory.cross_key_values.len == layers_count * encoder_positions_count * 2 * width);
        assert(memory.self_keys.len == layers_count * decoder_positions_count * width);
        assert(memory.self_values.len == layers_count * decoder_positions_count * width);
        assert(memory.input.len == width);
        assert(memory.normalized.len == width);
        assert(memory.query_key_value.len == 3 * width);
        assert(memory.attention_output.len == width);
        assert(memory.projection.len == width);
        assert(memory.ffn.len == specification.decoder_ffn_width);
        assert(memory.logits.len == specification.vocabulary_tokens_count);
        assert(memory.lane_quantized_rows.len % workers_count == 0);
        assert(memory.lane_quantized_rows.len / workers_count >= specification.decoder_ffn_width);
        assert(memory.lane_attention_scores.len == attention.decoderScratchValuesCount(encoder_positions_count, specification.decoder_attention_heads_count, workers_count));

        return .{ .memory = memory };
    }

    /// Cross-attention storage keeps its maximum physical stride so one runtime
    /// can transcribe different encoder lengths without reallocating. Only the
    /// initialized `positions_count` prefix participates in decoder attention.
    pub fn precomputeCrossKeyValues(decoder: *Decoder, specification: ModelSpecification, weights: *const InferenceWeights, encoded_audio: []const f32, positions_count: usize, lane: Lane) void {
        const positions_capacity = specification.encoder_positions_count_max;
        const width = specification.decoder_width;
        const heads_count = specification.decoder_attention_heads_count;
        const projection_values_count = positions_count * 2 * width;
        const tensor_values_capacity = positions_capacity * width;
        const layer_values_capacity = 2 * tensor_values_capacity;
        const head_values_capacity = positions_capacity * attention_head_width;
        const projected_key_values = decoder.memory.cross_projection_scratch[0..projection_values_count];

        assert(positions_count > 0);
        assert(positions_count <= positions_capacity);
        assert(encoded_audio.len == positions_count * width);

        for (weights.decoder_layers[0..specification.decoder_layers_count], 0..) |layer_weights, layer_index| {
            linear.forwardRows(encoded_audio, positions_count, layer_weights.cross_attention_key_value_weight, layer_weights.cross_attention_key_value_bias, .none, projected_key_values, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            // Decoder attention revisits these values for every generated token.
            // Store K and V head-major once so each head reads contiguous rows.
            const head_major_key_values = decoder.memory.cross_key_values[layer_index * layer_values_capacity ..][0..layer_values_capacity];
            const projection_heads_range = lane.range(2 * heads_count);
            for (projection_heads_range.start_index..projection_heads_range.end_index) |projection_head_index| {
                const projection_index = projection_head_index / heads_count;
                const head_index = projection_head_index % heads_count;
                const destination_head_offset = projection_index * tensor_values_capacity + head_index * head_values_capacity;
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

    pub fn decodeToken(decoder: *Decoder, specification: ModelSpecification, weights: *const InferenceWeights, encoder_positions_count: usize, token: Token, decoder_position: usize, lane: Lane) void {
        const width = specification.decoder_width;
        assert(encoder_positions_count > 0);
        assert(encoder_positions_count <= specification.encoder_positions_count_max);
        assert(token < specification.vocabulary_tokens_count);
        assert(decoder_position < specification.decoder_positions_count_max);

        const input_range = lane.range(width);
        for (input_range.start_index..input_range.end_index) |column| {
            decoder.memory.input[column] = embeddingValue(weights.decoder_embeddings_weight, token, column) + weights.decoder_position_encodings[decoder_position * width + column];
        }
        lane.sync();

        for (weights.decoder_layers[0..specification.decoder_layers_count], 0..) |layer_weights, layer_index| {
            // ── Self-Attention ──

            normalization.forwardRows(decoder.memory.input, layer_weights.self_attention_layer_norm_gamma, layer_weights.self_attention_layer_norm_beta, 1, width, decoder.memory.normalized, lane);
            lane.sync();

            linear.forwardOne(decoder.memory.normalized, layer_weights.self_attention_query_key_value_weight, layer_weights.self_attention_query_key_value_bias, .none, decoder.memory.query_key_value, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            storeSelfKeyValues(decoder, specification, layer_index, decoder_position, lane);
            lane.sync();

            selfAttention(decoder, specification, layer_index, decoder_position, lane);
            lane.sync();

            linear.forwardOne(decoder.memory.attention_output, layer_weights.self_attention_output_weight, layer_weights.self_attention_output_bias, .none, decoder.memory.projection, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            addInto(decoder.memory.projection, decoder.memory.input, lane);
            lane.sync();

            // ── Cross-Attention ──

            normalization.forwardRows(decoder.memory.input, layer_weights.cross_attention_layer_norm_gamma, layer_weights.cross_attention_layer_norm_beta, 1, width, decoder.memory.normalized, lane);
            lane.sync();

            linear.forwardOne(decoder.memory.normalized, layer_weights.cross_attention_query_weight, layer_weights.cross_attention_query_bias, .none, decoder.memory.query_key_value[0..width], laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            crossAttention(decoder, specification, encoder_positions_count, layer_index, lane);
            lane.sync();

            linear.forwardOne(decoder.memory.attention_output, layer_weights.cross_attention_output_weight, layer_weights.cross_attention_output_bias, .none, decoder.memory.projection, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            addInto(decoder.memory.projection, decoder.memory.input, lane);
            lane.sync();

            // ── Feed-Forward Network ──

            normalization.forwardRows(decoder.memory.input, layer_weights.ffn_layer_norm_gamma, layer_weights.ffn_layer_norm_beta, 1, width, decoder.memory.normalized, lane);
            lane.sync();

            linear.forwardOne(decoder.memory.normalized, layer_weights.ffn_expansion_weight, layer_weights.ffn_expansion_bias, .gelu, decoder.memory.ffn, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            linear.forwardOne(decoder.memory.ffn, layer_weights.ffn_contraction_weight, layer_weights.ffn_contraction_bias, .none, decoder.memory.projection, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            addInto(decoder.memory.projection, decoder.memory.input, lane);
            lane.sync();
        }

        normalization.forwardRows(decoder.memory.input, weights.decoder_layer_norm_gamma, weights.decoder_layer_norm_beta, 1, width, decoder.memory.normalized, lane);
        lane.sync();

        linear.forwardOne(decoder.memory.normalized, weights.decoder_embeddings_weight, &.{}, .none, decoder.memory.logits, laneQuantizedRow(decoder, lane), lane);
        lane.sync();
    }

    pub fn logits(decoder: *Decoder) []f32 {
        return decoder.memory.logits;
    }
};

fn storeSelfKeyValues(decoder: *Decoder, specification: ModelSpecification, layer_index: usize, decoder_position: usize, lane: Lane) void {
    const width = specification.decoder_width;
    const positions_capacity = specification.decoder_positions_count_max;
    const layer_offset = layer_index * positions_capacity * width;
    const columns_range = lane.range(width);

    for (columns_range.start_index..columns_range.end_index) |column| {
        const head_index = column / attention_head_width;
        const head_depth = column % attention_head_width;
        const cache_offset = layer_offset + head_index * positions_capacity * attention_head_width + decoder_position * attention_head_width + head_depth;
        decoder.memory.self_keys[cache_offset] = decoder.memory.query_key_value[width + column];
        decoder.memory.self_values[cache_offset] = decoder.memory.query_key_value[2 * width + column];
    }
}

fn selfAttention(decoder: *Decoder, specification: ModelSpecification, layer_index: usize, decoder_position: usize, lane: Lane) void {
    const width = specification.decoder_width;
    const positions_capacity = specification.decoder_positions_count_max;
    const layer_offset = layer_index * positions_capacity * width;
    const keys = decoder.memory.self_keys[layer_offset..][0 .. positions_capacity * width];
    const values = decoder.memory.self_values[layer_offset..][0 .. positions_capacity * width];
    attention.forwardDecoder(decoder.memory.query_key_value[0..width], keys, values, decoder_position + 1, positions_capacity, width, specification.decoder_attention_heads_count, decoder.memory.attention_output, decoder.memory.lane_attention_scores, lane);
}

fn crossAttention(decoder: *Decoder, specification: ModelSpecification, positions_count: usize, layer_index: usize, lane: Lane) void {
    const width = specification.decoder_width;
    const positions_capacity = specification.encoder_positions_count_max;
    const tensor_values_capacity = positions_capacity * width;
    const layer_values_capacity = 2 * tensor_values_capacity;
    const key_values = decoder.memory.cross_key_values[layer_index * layer_values_capacity ..][0..layer_values_capacity];
    const keys = key_values[0..tensor_values_capacity];
    const values = key_values[tensor_values_capacity..][0..tensor_values_capacity];

    assert(positions_count > 0);
    assert(positions_count <= positions_capacity);

    attention.forwardDecoder(decoder.memory.query_key_value[0..width], keys, values, positions_count, positions_capacity, width, specification.decoder_attention_heads_count, decoder.memory.attention_output, decoder.memory.lane_attention_scores, lane);
}

fn embeddingValue(weight: QuantizedWeight, token: Token, column: usize) f32 {
    assert(token < weight.output_rows_count);
    assert(column < weight.input_values_count);

    const output_row: usize = @intCast(token);
    const packed_offset = weight.layout().valueOffset(output_row, column);

    return @as(f32, @floatFromInt(weight.values[packed_offset])) / weight.scales[output_row];
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

fn laneQuantizedRow(decoder: *Decoder, lane: Lane) []u8 {
    const values_count = decoder.memory.lane_quantized_rows.len / lane.count;
    return decoder.memory.lane_quantized_rows[lane.index * values_count ..][0..values_count];
}
