//! The Whisper decoder prepares cross-attention storage once, then advances one
//! token through explicit self-attention, cross-attention, and feed-forward
//! stages. Keys use depth-major SIMD storage; values use position-major rows.
//! Both caches retain Float16 values to reduce the dominant mutable working set
//! and the bandwidth paid again for every generated token.

const std = @import("std");
const attention = @import("attention.zig");
const linear = @import("linear.zig");
const model_module = @import("Model.zig");
const vnni_weight = @import("vnni_weight.zig");
const Lane = @import("Scheduler.zig").Lane;
const InferenceWeights = model_module.Weights;
const ModelDimensions = model_module.Dimensions;
const QuantizedWeight = vnni_weight.QuantizedWeight;
const assert = std.debug.assert;

const attention_head_width: usize = attention.head_width;
const simd_lanes_count: usize = 8;

pub const Token = u32;

pub const Memory = struct {
    cross_key_values: []f16,
    self_keys: []f16,
    self_values: []f16,
    input: []f32,
    query_key_value: []f32,
    attention_output: []f32,
    ffn: []f32,
    logits: []f32,
    lane_quantized_rows: []u8,
    lane_attention_scores: []f32,
};

pub const Decoder = struct {
    memory: Memory,
    self_positions_capacity: usize,
    normalized_input_scale: f32 = 1.0,

    pub fn init(memory: Memory, dimensions: ModelDimensions, encoder_positions_capacity: usize, self_positions_capacity: usize, workers_count: usize) Decoder {
        const width = dimensions.decoder_width;
        const layers_count = dimensions.decoder_layers_count;

        assert(encoder_positions_capacity > 0);
        assert(encoder_positions_capacity <= dimensions.encoder_positions_count_max);
        assert(self_positions_capacity > 0);
        assert(self_positions_capacity <= dimensions.decoder_positions_count_max);
        assert(memory.cross_key_values.len == layers_count * crossLayerValuesCapacity(encoder_positions_capacity, width));
        assert(memory.self_keys.len == layers_count * attention.packedKeyValuesCount(self_positions_capacity, width));
        assert(memory.self_values.len == layers_count * self_positions_capacity * width);
        assert(memory.input.len == width);
        assert(memory.query_key_value.len == 3 * width);
        assert(memory.attention_output.len == width);
        assert(memory.ffn.len == dimensions.decoder_ffn_width);
        assert(memory.logits.len == dimensions.vocabulary_tokens_count);
        assert(memory.lane_quantized_rows.len % workers_count == 0);
        assert(memory.lane_quantized_rows.len / workers_count >= dimensions.decoder_ffn_width);
        assert(memory.lane_attention_scores.len == attention.decoderScratchValuesCount(@max(encoder_positions_capacity, self_positions_capacity), dimensions.decoder_attention_heads_count, workers_count));

        return .{ .memory = memory, .self_positions_capacity = self_positions_capacity };
    }

    /// Cross-attention uses this transcription's encoder length for its physical
    /// strides within the maximum backing allocation. Subsequent `decodeToken`
    /// calls must use this same encoder length. Projection rewrites every
    /// logical K/V value; SIMD key padding and unused capacity are never read.
    /// It writes Float16 K depth-major and V position-major without creating a
    /// generic row-major intermediate. Float16 conversion is not bit-exact to
    /// the former Float32 cache and is covered by corpus gates.
    pub fn precomputeCrossLayer(decoder: *Decoder, dimensions: ModelDimensions, weights: *const InferenceWeights, encoded_audio: []const f32, positions_count: usize, layer_index: usize, lane: Lane) void {
        assert(positions_count > 0);
        assert(positions_count <= dimensions.encoder_positions_count_max);
        assert(layer_index < dimensions.decoder_layers_count);

        const width = dimensions.decoder_width;
        const layer_values_capacity = crossLayerValuesCapacity(positions_count, width);

        assert(encoded_audio.len == positions_count * width);

        const layer_weights = weights.decoder_layers[layer_index];
        const key_values = decoder.memory.cross_key_values[layer_index * layer_values_capacity ..][0..layer_values_capacity];

        linear.forwardDecoderCrossKeyValues(encoded_audio, positions_count, layer_weights.cross_attention_key_value_weight, layer_weights.cross_attention_key_value_bias, positions_count, key_values, laneQuantizedRow(decoder, lane), lane);
    }

    pub fn decodeToken(decoder: *Decoder, dimensions: ModelDimensions, weights: *const InferenceWeights, encoder_positions_count: usize, token: Token, decoder_position: usize, lane: Lane) void {
        const width = dimensions.decoder_width;

        assert(encoder_positions_count > 0);
        assert(encoder_positions_count <= dimensions.encoder_positions_count_max);
        assert(token < dimensions.vocabulary_tokens_count);
        assert(decoder_position < decoder.self_positions_capacity);

        const input_range = lane.range(width);

        for (input_range.start_index..input_range.end_index) |column| {
            decoder.memory.input[column] = embeddingValue(weights.decoder_embeddings_weight, token, column) + weights.decoder_position_encodings[decoder_position * width + column];
        }

        lane.sync();

        for (weights.decoder_layers[0..dimensions.decoder_layers_count], 0..) |layer_weights, layer_index| {
            // ── Self-Attention ──

            forwardNormalizedProjection(decoder, decoder.memory.input, layer_weights.self_attention_layer_norm_gamma, layer_weights.self_attention_layer_norm_beta, layer_weights.self_attention_query_key_value_weight, layer_weights.self_attention_query_key_value_bias, .none, decoder.memory.query_key_value, lane);
            lane.sync();

            storeSelfKeyValues(decoder, dimensions, layer_index, decoder_position, lane);
            lane.sync();

            selfAttention(decoder, dimensions, layer_index, decoder_position, lane);
            lane.sync();

            // Each lane adds the completed projection into its own output
            // columns. Its source is separate from input, so no intermediate
            // projection buffer or barrier before the residual addition is needed.
            linear.forwardResidualOne(decoder.memory.attention_output, layer_weights.self_attention_output_weight, layer_weights.self_attention_output_bias, decoder.memory.input, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            // ── Cross-Attention ──

            forwardNormalizedProjection(decoder, decoder.memory.input, layer_weights.cross_attention_layer_norm_gamma, layer_weights.cross_attention_layer_norm_beta, layer_weights.cross_attention_query_weight, layer_weights.cross_attention_query_bias, .none, decoder.memory.query_key_value[0..width], lane);
            lane.sync();

            crossAttention(decoder, dimensions, encoder_positions_count, layer_index, lane);
            lane.sync();

            linear.forwardResidualOne(decoder.memory.attention_output, layer_weights.cross_attention_output_weight, layer_weights.cross_attention_output_bias, decoder.memory.input, laneQuantizedRow(decoder, lane), lane);
            lane.sync();

            // ── Feed-Forward Network ──

            forwardNormalizedProjection(decoder, decoder.memory.input, layer_weights.ffn_layer_norm_gamma, layer_weights.ffn_layer_norm_beta, layer_weights.ffn_expansion_weight, layer_weights.ffn_expansion_bias, .gelu, decoder.memory.ffn, lane);
            lane.sync();

            linear.forwardResidualOne(decoder.memory.ffn, layer_weights.ffn_contraction_weight, layer_weights.ffn_contraction_bias, decoder.memory.input, laneQuantizedRow(decoder, lane), lane);
            lane.sync();
        }

        forwardNormalizedProjection(decoder, decoder.memory.input, weights.decoder_layer_norm_gamma, weights.decoder_layer_norm_beta, weights.decoder_embeddings_weight, &.{}, .none, decoder.memory.logits, lane);
        lane.sync();
    }
};

fn storeSelfKeyValues(decoder: *Decoder, dimensions: ModelDimensions, layer_index: usize, decoder_position: usize, lane: Lane) void {
    const width = dimensions.decoder_width;
    const positions_capacity = decoder.self_positions_capacity;
    const packed_positions_capacity = std.mem.alignForward(usize, positions_capacity, simd_lanes_count);
    const packed_key_layer_values_capacity = attention.packedKeyValuesCount(positions_capacity, width);
    const value_layer_values_capacity = positions_capacity * width;
    const packed_key_layer_offset = layer_index * packed_key_layer_values_capacity;
    const value_layer_offset = layer_index * value_layer_values_capacity;
    const columns_range = lane.range(width);

    for (columns_range.start_index..columns_range.end_index) |column| {
        const head_index = column / attention_head_width;
        const head_depth = column % attention_head_width;
        const packed_key_offset = packed_key_layer_offset + head_index * packed_positions_capacity * attention_head_width + head_depth * packed_positions_capacity + decoder_position;
        const value_offset = value_layer_offset + head_index * positions_capacity * attention_head_width + decoder_position * attention_head_width + head_depth;

        decoder.memory.self_keys[packed_key_offset] = @floatCast(decoder.memory.query_key_value[width + column]);
        decoder.memory.self_values[value_offset] = @floatCast(decoder.memory.query_key_value[2 * width + column]);
    }
}

fn selfAttention(decoder: *Decoder, dimensions: ModelDimensions, layer_index: usize, decoder_position: usize, lane: Lane) void {
    const width = dimensions.decoder_width;
    const positions_capacity = decoder.self_positions_capacity;
    const packed_key_layer_values_capacity = attention.packedKeyValuesCount(positions_capacity, width);
    const value_layer_values_capacity = positions_capacity * width;
    const packed_keys = decoder.memory.self_keys[layer_index * packed_key_layer_values_capacity ..][0..packed_key_layer_values_capacity];
    const values = decoder.memory.self_values[layer_index * value_layer_values_capacity ..][0..value_layer_values_capacity];

    attention.forwardDecoder(decoder.memory.query_key_value[0..width], packed_keys, values, decoder_position + 1, positions_capacity, width, dimensions.decoder_attention_heads_count, decoder.memory.attention_output, decoder.memory.lane_attention_scores, lane);
}

fn crossAttention(decoder: *Decoder, dimensions: ModelDimensions, positions_count: usize, layer_index: usize, lane: Lane) void {
    const width = dimensions.decoder_width;
    const packed_keys_values_capacity = attention.packedKeyValuesCount(positions_count, width);
    const values_capacity = positions_count * width;
    const layer_values_capacity = packed_keys_values_capacity + values_capacity;
    const key_values = decoder.memory.cross_key_values[layer_index * layer_values_capacity ..][0..layer_values_capacity];
    const packed_keys = key_values[0..packed_keys_values_capacity];
    const values = key_values[packed_keys_values_capacity..][0..values_capacity];

    assert(positions_count > 0);
    assert(positions_count <= dimensions.encoder_positions_count_max);

    attention.forwardDecoder(decoder.memory.query_key_value[0..width], packed_keys, values, positions_count, positions_count, width, dimensions.decoder_attention_heads_count, decoder.memory.attention_output, decoder.memory.lane_attention_scores, lane);
}

fn crossLayerValuesCapacity(positions_capacity: usize, width: usize) usize {
    return attention.packedKeyValuesCount(positions_capacity, width) + positions_capacity * width;
}

fn forwardNormalizedProjection(decoder: *Decoder, input: []const f32, gamma: []const f32, beta: []const f32, weight: QuantizedWeight, bias: []const f32, activation: linear.Activation, output: []f32, lane: Lane) void {
    assert(input.len <= decoder.memory.lane_quantized_rows.len / lane.count);

    // All lanes consume the leader's quantized row and scale after publication.
    const quantized_input = decoder.memory.lane_quantized_rows[0..input.len];

    if (lane.isLeader()) {
        decoder.normalized_input_scale = linear.quantizeNormalized(input, gamma, beta, quantized_input);
    }

    lane.sync();

    linear.forwardQuantizedOneParallel(quantized_input, decoder.normalized_input_scale, weight, bias, activation, output, lane);
}

fn embeddingValue(weight: QuantizedWeight, token: Token, column: usize) f32 {
    assert(token < weight.scales.len);
    assert(column < weight.input_values_count);

    const output_row: usize = @intCast(token);
    const packed_offset = weight.layout().valueOffset(output_row, column);

    return @as(f32, @floatFromInt(weight.values[packed_offset])) / weight.scales[output_row];
}

fn laneQuantizedRow(decoder: *Decoder, lane: Lane) []u8 {
    const values_count = decoder.memory.lane_quantized_rows.len / lane.count;

    return decoder.memory.lane_quantized_rows[lane.index * values_count ..][0..values_count];
}
