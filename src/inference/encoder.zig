//! The Whisper encoder transforms a logical prefix of its fixed 30-second
//! capacity into encoded audio. Backing buffers retain maximum capacity while
//! every convolution and transformer operation uses the current frame count.

const Encoder = @This();

const std = @import("std");
const attention = @import("attention.zig");
const linear = @import("linear.zig");
const log_mel = @import("log_mel.zig");
const model_module = @import("Model.zig");
const normalization = @import("normalization.zig");
const Lane = @import("Scheduler.zig").Lane;
const InferenceWeights = model_module.Weights;
const ModelDimensions = model_module.Dimensions;
const assert = std.debug.assert;

const convolution_kernel_width: usize = 3;
const convolution_positions_per_tile: usize = 6;

activation: []f32,
shared_scratch: []f32,
query_key_value: []f32,
attention_output: []f32,
encoded_audio: []f32,
lane_float_rows: []f32,
lane_quantized_rows: []u8,

pub fn validate(encoder: *const Encoder, dimensions: ModelDimensions, positions_count_max: usize, workers_count: usize) void {
    const width = dimensions.encoder_width;

    assert(positions_count_max > 0);
    assert(positions_count_max <= dimensions.encoder_positions_count_max);
    assert(encoder.activation.len == positions_count_max * width);
    assert(encoder.shared_scratch.len == attention.encoderScratchValuesCount(workers_count));
    assert(encoder.query_key_value.len == attention.encoderQueryKeyValueValuesCount(positions_count_max, width));
    assert(encoder.attention_output.len == positions_count_max * width);
    assert(encoder.encoded_audio.len == positions_count_max * width);
    assert(encoder.lane_float_rows.len == workers_count * laneFloatScratchValuesCount(dimensions));
    assert(encoder.lane_quantized_rows.len == workers_count * laneQuantizedScratchValuesCount(dimensions));
}

pub const Phase = union(enum) { frontend, layer: usize, normalization };

/// Runs one dependency-complete phase. The caller joins all lanes before
/// consuming its output or starting the returned phase with a new width.
pub fn forwardPhase(encoder: *Encoder, dimensions: ModelDimensions, weights: *const InferenceWeights, features: log_mel.Features, phase: Phase, lane: Lane) ?Phase {
    assert(features.frames_count > 0);
    assert(features.frames_count <= log_mel.encoder_frames_count_max);
    assert(features.values.len == log_mel.mel_bins_count * features.frames_count);
    const positions_count = features.encoderPositionsCount();
    assert(positions_count <= dimensions.encoder_positions_count_max);
    const width = dimensions.encoder_width;
    const activation = encoder.activation[0 .. positions_count * width];
    switch (phase) {
        .frontend => {
            forwardConvolutionFrontend(features, weights, activation, laneFloatRow(encoder, lane), laneQuantizedRow(encoder, lane), lane);
            lane.sync();
            const indices = lane.range(activation.len);
            for (indices.start_index..indices.end_index) |index| activation[index] += weights.encoder_position_encodings[index];
            return .{ .layer = 0 };
        },
        .layer => |index| {
            const layer_weights = weights.encoder_layers[index];
            const query_key_value_storage = encoder.query_key_value[0..attention.encoderQueryKeyValueValuesCount(positions_count, width)];
            const query_key_value = attention.encoderQueryKeyValue(query_key_value_storage, positions_count, width);
            const attention_output = encoder.attention_output[0 .. positions_count * width];
            const scratch = encoder.shared_scratch[0..attention.encoderScratchValuesCount(lane.count)];
            linear.forwardNormalizedEncoderQueryKeyValue(activation, layer_weights.self_attention_layer_norm_gamma, layer_weights.self_attention_layer_norm_beta, positions_count, layer_weights.self_attention_query_key_value_weight, layer_weights.self_attention_query_key_value_bias, query_key_value_storage, laneQuantizedRow(encoder, lane), lane);
            lane.sync();
            attention.forwardEncoder(query_key_value, positions_count, width, dimensions.encoder_attention_heads_count, attention_output, scratch, lane);
            lane.sync();
            linear.forwardResidualRows(attention_output, positions_count, layer_weights.self_attention_output_weight, layer_weights.self_attention_output_bias, activation, laneQuantizedRow(encoder, lane), lane);
            lane.sync();
            linear.forwardFeedForwardResidualRows(activation, layer_weights.ffn_layer_norm_gamma, layer_weights.ffn_layer_norm_beta, positions_count, layer_weights.ffn_expansion_weight, layer_weights.ffn_expansion_bias, layer_weights.ffn_contraction_weight, layer_weights.ffn_contraction_bias, laneFloatRow(encoder, lane), laneQuantizedRow(encoder, lane), lane);
            return if (index + 1 == dimensions.encoder_layers_count) .normalization else .{ .layer = index + 1 };
        },
        .normalization => {
            normalization.forwardRows(activation, weights.encoder_layer_norm_gamma, weights.encoder_layer_norm_beta, positions_count, width, encoder.encoded_audio[0 .. positions_count * width], lane);
            return null;
        },
    }
}

pub fn laneFloatScratchValuesCount(dimensions: ModelDimensions) usize {
    const convolution_rows_count = 2 * convolution_positions_per_tile + 1;
    const convolution_values_count = (convolution_rows_count + convolution_kernel_width) * dimensions.encoder_width;

    return @max(convolution_values_count, linear.ffn_rows_per_tile * dimensions.encoder_ffn_width);
}

pub fn laneQuantizedScratchValuesCount(dimensions: ModelDimensions) usize {
    const first_convolution_values_count = (2 * convolution_positions_per_tile + 1) * log_mel.mel_bins_count * convolution_kernel_width;
    const second_convolution_values_count = convolution_positions_per_tile * dimensions.encoder_width * convolution_kernel_width;
    const convolution_values_count = @max(first_convolution_values_count, second_convolution_values_count);
    // Fourteen ordinary rows also cover the eight-row QKV and cross-K/V tiles.
    const projection_values_count = linear.rows_per_tile * dimensions.encoder_width;
    const ffn_values_count = linear.ffn_rows_per_tile * @max(dimensions.encoder_width, dimensions.encoder_ffn_width);

    return @max(convolution_values_count, @max(projection_values_count, ffn_values_count));
}

/// `forwardConvolutionFrontend` writes GELU(Conv2(GELU(Conv1(features)))) before
/// positional encoding, with ceil(frames_count / 2) output rows. Each lane owns
/// disjoint output rows and private scratch sized by the encoder's lane scratch
/// functions. Scheduler lanes synchronize before claiming tiles; no intermediate
/// rows survive the operation.
pub fn forwardConvolutionFrontend(features: log_mel.Features, weights: *const InferenceWeights, output: []f32, float_scratch: []f32, quantized_scratch: []u8, lane: Lane) void {
    const frames_count = features.frames_count;
    const width = weights.encoder_convolution_1_weight.scales.len;
    const first_depth = log_mel.mel_bins_count * convolution_kernel_width;
    const second_depth = width * convolution_kernel_width;
    const first_rows_capacity = 2 * convolution_positions_per_tile + 1;

    assert(frames_count > 0);
    assert(features.values.len == log_mel.mel_bins_count * frames_count);
    assert(weights.encoder_convolution_1_weight.input_values_count == first_depth);
    assert(weights.encoder_convolution_2_weight.input_values_count == second_depth);
    assert(weights.encoder_convolution_2_weight.scales.len == width);
    assert(output.len == ((frames_count + 1) / 2) * width);
    assert(float_scratch.len >= first_rows_capacity * width + @max(first_depth, second_depth));
    assert(quantized_scratch.len >= @max(first_rows_capacity * first_depth, convolution_positions_per_tile * second_depth));

    const first_rows = float_scratch[0 .. first_rows_capacity * width];
    const float_row = float_scratch[first_rows.len..];
    var first_scales: [first_rows_capacity]f32 = undefined;
    var second_scales: [convolution_positions_per_tile]f32 = undefined;
    const output_positions_count = (frames_count + 1) / 2;
    var tiles = lane.tiles(std.math.divCeil(usize, output_positions_count, convolution_positions_per_tile) catch unreachable, 4);

    // PERFORMANCE: Six Conv2 rows need only thirteen Conv1 rows. Keeping that
    // window in lane scratch avoids a full [frames, width] intermediate and
    // lets both projections reuse weights across rows. Adjacent tiles repeat
    // one halo row; lanes never read another lane's unfinished Conv1 output.
    while (tiles.next()) |tile_index| {
        const position_begin = tile_index * convolution_positions_per_tile;
        const positions_count: usize = @min(convolution_positions_per_tile, output_positions_count - position_begin);
        const first_origin = @as(isize, @intCast(2 * position_begin)) - 1;
        const first_begin: usize = @intCast(@max(first_origin, 0));
        const first_end = @min(2 * (position_begin + positions_count), frames_count);
        const first_rows_count = first_end - first_begin;
        const first_row_begin: usize = @intFromBool(first_origin < 0);

        // Conv2 padding is zero AFTER Conv1 + GELU, not Conv1 evaluated on
        // padded Mel rows (which would introduce its bias at the boundary).
        // Real rows are fully overwritten below; only the boundary rows that
        // Conv2 actually reads need clearing, including a partial final tile.
        @memset(first_rows[0 .. first_row_begin * width], 0);
        @memset(first_rows[(first_row_begin + first_rows_count) * width .. (2 * positions_count + 1) * width], 0);
        for (0..first_rows_count) |row| {
            const frame = first_begin + row;
            for (0..log_mel.mel_bins_count) |channel| {
                for (0..convolution_kernel_width) |kernel_index| {
                    const input_position = @as(isize, @intCast(frame + kernel_index)) - 1;
                    float_row[channel * convolution_kernel_width + kernel_index] = if (input_position < 0 or input_position >= frames_count)
                        0
                    else
                        features.values[channel * frames_count + @as(usize, @intCast(input_position))];
                }
            }
            first_scales[row] = linear.quantizeRow(float_row[0..first_depth], quantized_scratch[row * first_depth ..][0..first_depth]);
        }
        linear.forwardQuantizedRows(quantized_scratch[0 .. first_rows_count * first_depth], first_scales[0..first_rows_count], weights.encoder_convolution_1_weight, weights.encoder_convolution_1_bias, .gelu, first_rows[first_row_begin * width ..][0 .. first_rows_count * width]);

        for (0..positions_count) |row| {
            for (0..width) |channel| {
                for (0..convolution_kernel_width) |kernel_index| {
                    float_row[channel * convolution_kernel_width + kernel_index] = first_rows[(2 * row + kernel_index) * width + channel];
                }
            }
            second_scales[row] = linear.quantizeRow(float_row[0..second_depth], quantized_scratch[row * second_depth ..][0..second_depth]);
        }
        linear.forwardQuantizedRows(quantized_scratch[0 .. positions_count * second_depth], second_scales[0..positions_count], weights.encoder_convolution_2_weight, weights.encoder_convolution_2_bias, .gelu, output[position_begin * width ..][0 .. positions_count * width]);
    }
}

fn laneFloatRow(encoder: *Encoder, lane: Lane) []f32 {
    const values_count = encoder.lane_float_rows.len / lane.count;
    return encoder.lane_float_rows[lane.index * values_count ..][0..values_count];
}

fn laneQuantizedRow(encoder: *Encoder, lane: Lane) []u8 {
    const values_count = encoder.lane_quantized_rows.len / lane.count;
    return encoder.lane_quantized_rows[lane.index * values_count ..][0..values_count];
}
