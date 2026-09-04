//! The Whisper encoder transforms a logical prefix of its fixed 30-second
//! capacity into encoded audio. Backing buffers retain maximum capacity while
//! every convolution and transformer operation uses the current frame count.

const std = @import("std");
const attention = @import("attention.zig");
const linear = @import("linear.zig");
const log_mel = @import("log_mel.zig");
const model_module = @import("model.zig");
const normalization = @import("normalization.zig");
const vnni_weight = @import("vnni_weight.zig");
const Lane = @import("executor.zig").Lane;
const InferenceWeights = model_module.InferenceWeights;
const ModelSpecification = model_module.ModelSpecification;
const QuantizedWeight = vnni_weight.QuantizedWeight;
const assert = std.debug.assert;

const convolution_kernel_width: usize = 3;
const simd_lanes_count: usize = 8;

const F32x8 = @Vector(simd_lanes_count, f32);

pub const Memory = struct {
    convolution_1_output: []f32,
    activation_0: []f32,
    activation_1: []f32,
    shared_scratch: []f32,
    query_key_value: []f32,
    attention_output: []f32,
    encoded_audio: []f32,
    lane_float_rows: []f32,
    lane_quantized_rows: []u8,
};

pub const EncodedAudio = struct {
    values: []const f32,
    positions_count: usize,
};

pub const Encoder = struct {
    memory: Memory,

    pub fn init(memory: Memory, specification: ModelSpecification, workers_count: usize) Encoder {
        const positions_count_max = specification.encoder_positions_count_max;
        const width = specification.encoder_width;

        assert(memory.convolution_1_output.len == log_mel.encoder_frames_count_max * width);
        assert(memory.activation_0.len == positions_count_max * width);
        assert(memory.activation_1.len == positions_count_max * width);
        assert(memory.shared_scratch.len == sharedScratchValuesCount(specification, workers_count));
        assert(memory.query_key_value.len == positions_count_max * 3 * width);
        assert(memory.attention_output.len == positions_count_max * width);
        assert(memory.encoded_audio.len == positions_count_max * width);
        assert(memory.lane_float_rows.len == workers_count * laneFloatScratchValuesCount(specification));
        assert(memory.lane_quantized_rows.len == workers_count * laneQuantizedScratchValuesCount(specification));

        return .{ .memory = memory };
    }

    /// `encode` consumes the logical feature shape rather than the backing
    /// buffers' maximum capacity. Every returned row has `encoder_width` values.
    pub fn encode(encoder: *Encoder, specification: ModelSpecification, weights: *const InferenceWeights, features: log_mel.Features, lane: Lane) EncodedAudio {
        const frames_count = features.frames_count;
        const positions_count = features.encoderPositionsCount();
        const width = specification.encoder_width;

        assert(frames_count > 0);
        assert(frames_count <= log_mel.encoder_frames_count_max);
        assert(positions_count <= specification.encoder_positions_count_max);
        assert(features.values.len == log_mel.mel_bins_count * frames_count);

        const convolution_1_output = encoder.memory.convolution_1_output[0 .. frames_count * width];
        const activation_0 = encoder.memory.activation_0[0 .. positions_count * width];
        const activation_1 = encoder.memory.activation_1[0 .. positions_count * width];
        const query_key_value = encoder.memory.query_key_value[0 .. positions_count * 3 * width];
        const attention_output = encoder.memory.attention_output[0 .. positions_count * width];
        const encoded_audio = encoder.memory.encoded_audio[0 .. positions_count * width];

        // ── Convolutional Frontend ──

        convolution(encoder, features.values, frames_count, specification.mel_bins_count, weights.encoder_convolution_1_weight, weights.encoder_convolution_1_bias, 1, true, convolution_1_output, lane);
        lane.sync();

        convolution(encoder, convolution_1_output, frames_count, width, weights.encoder_convolution_2_weight, weights.encoder_convolution_2_bias, 2, false, activation_0, lane);
        lane.sync();

        const position_values_range = lane.range(positions_count * width);
        for (position_values_range.start_index..position_values_range.end_index) |value_index| {
            activation_0[value_index] += weights.encoder_position_encodings[value_index];
        }
        lane.sync();

        // ── Transformer Layers ──

        const normalized_values_count = positions_count * width;
        const attention_scratch_values_count = attention.encoderScratchValuesCount(positions_count, specification.encoder_attention_heads_count, lane.count);
        const normalized = encoder.memory.shared_scratch[0..normalized_values_count];
        const attention_scratch = encoder.memory.shared_scratch[0..attention_scratch_values_count];
        var layer_input = activation_0;
        var layer_output = activation_1;

        for (weights.encoder_layers[0..specification.encoder_layers_count]) |layer_weights| {
            linear.forwardNormalizedRows(layer_input, layer_weights.self_attention_layer_norm_gamma, layer_weights.self_attention_layer_norm_beta, positions_count, layer_weights.self_attention_query_key_value_weight, layer_weights.self_attention_query_key_value_bias, query_key_value, laneQuantizedRow(encoder, lane), lane);
            lane.sync();

            attention.forwardEncoder(query_key_value, positions_count, width, specification.encoder_attention_heads_count, attention_output, attention_scratch, lane);
            lane.sync();

            linear.forwardRows(attention_output, positions_count, layer_weights.self_attention_output_weight, layer_weights.self_attention_output_bias, .none, layer_output, laneQuantizedRow(encoder, lane), lane);
            lane.sync();

            addInto(layer_input, layer_output, lane);
            lane.sync();

            normalization.forwardRows(layer_output, layer_weights.ffn_layer_norm_gamma, layer_weights.ffn_layer_norm_beta, positions_count, width, normalized, lane);
            lane.sync();

            linear.forwardFeedForwardRows(normalized, positions_count, layer_weights.ffn_expansion_weight, layer_weights.ffn_expansion_bias, layer_weights.ffn_contraction_weight, layer_weights.ffn_contraction_bias, attention_output, laneFloatRow(encoder, lane), laneQuantizedRow(encoder, lane), lane);
            lane.sync();

            addInto(attention_output, layer_output, lane);
            lane.sync();

            const previous_input = layer_input;
            layer_input = layer_output;
            layer_output = previous_input;
        }

        normalization.forwardRows(layer_input, weights.encoder_layer_norm_gamma, weights.encoder_layer_norm_beta, positions_count, width, encoded_audio, lane);
        lane.sync();

        return .{ .values = encoded_audio, .positions_count = positions_count };
    }
};

pub fn sharedScratchValuesCount(specification: ModelSpecification, workers_count: usize) usize {
    const normalized_values_count = specification.encoder_positions_count_max * specification.encoder_width;
    const attention_scratch_values_count = attention.encoderScratchValuesCount(specification.encoder_positions_count_max, specification.encoder_attention_heads_count, workers_count);

    return @max(normalized_values_count, attention_scratch_values_count);
}

pub fn laneFloatScratchValuesCount(specification: ModelSpecification) usize {
    const convolution_2_depth = specification.encoder_width * convolution_kernel_width;
    const lane_row_values_count = @max(specification.encoder_ffn_width, convolution_2_depth);

    return @max(lane_row_values_count, linear.ffn_rows_per_tile * specification.encoder_ffn_width);
}

pub fn laneQuantizedScratchValuesCount(specification: ModelSpecification) usize {
    const convolution_2_depth = specification.encoder_width * convolution_kernel_width;
    const lane_row_values_count = @max(specification.encoder_ffn_width, convolution_2_depth);

    return linear.rows_per_tile * lane_row_values_count;
}

fn convolution(encoder: *Encoder, input: []const f32, input_positions_count: usize, input_channels_count: usize, weight: QuantizedWeight, bias: []const f32, stride: usize, input_is_channel_major: bool, output: []f32, lane: Lane) void {
    const output_positions_count = @divFloor(input_positions_count + 2 - convolution_kernel_width, stride) + 1;
    const output_channels_count = weight.output_rows_count;
    const input_depth = input_channels_count * convolution_kernel_width;
    assert(weight.input_values_count == input_depth);
    assert(output.len == output_positions_count * output_channels_count);

    const output_positions_range = lane.range(output_positions_count);
    const float_row = laneFloatRow(encoder, lane)[0..input_depth];
    const quantized_row = laneQuantizedRow(encoder, lane)[0..input_depth];

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

        const input_scale = linear.quantizeRow(float_row, quantized_row);
        const output_row = output[output_position * output_channels_count ..][0..output_channels_count];
        linear.forwardQuantizedOne(quantized_row, input_scale, weight, bias, .gelu, output_row);
    }
}

// This short residual leaf is intentionally mirrored in decoder.zig; keeping
// it local avoids a generic transformer seam between distinct lane pipelines.
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

fn laneFloatRow(encoder: *Encoder, lane: Lane) []f32 {
    const values_count = encoder.memory.lane_float_rows.len / lane.count;
    return encoder.memory.lane_float_rows[lane.index * values_count ..][0..values_count];
}

fn laneQuantizedRow(encoder: *Encoder, lane: Lane) []u8 {
    const values_count = encoder.memory.lane_quantized_rows.len / lane.count;
    return encoder.memory.lane_quantized_rows[lane.index * values_count ..][0..values_count];
}
