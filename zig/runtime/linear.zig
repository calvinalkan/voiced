//! Quantized linear projections over the runtime's permanent o8/k4 weight layout.
//!
//! Multi-row projections reuse each weight vector across fourteen activation
//! rows. Token-serial projections retain up to fourteen independent output
//! accumulators. Encoder FFN pairs use four-row tiles and never materialize the
//! full `[positions, ffn_width]` expansion tensor.

const builtin = @import("builtin");
const std = @import("std");
const normalization = @import("normalization.zig");
const vnni_weight = @import("vnni_weight.zig");
const Lane = @import("executor.zig").Lane;
const QuantizedWeight = vnni_weight.QuantizedWeight;
const assert = std.debug.assert;

pub const rows_per_tile: usize = 14;
pub const ffn_rows_per_tile: usize = 4;

const output_rows_per_block = vnni_weight.output_rows_per_block;
const output_blocks_per_token_iteration: usize = 14;
const output_blocks_per_ffn_iteration: usize = 3;
const accumulators_count_max: usize = 14;
const depth_values_per_group = vnni_weight.input_values_per_group;
const bytes_per_weight_group = vnni_weight.values_per_group;
const simd_lanes_count: usize = 8;

const F32x8 = @Vector(simd_lanes_count, f32);
const I32x8 = @Vector(simd_lanes_count, i32);
const U32x8 = @Vector(simd_lanes_count, u32);
const I8x32 = @Vector(bytes_per_weight_group, i8);
const U8x8 = @Vector(simd_lanes_count, u8);

pub const Activation = enum {
    none,
    gelu,
};

// ─── Multi-Row Projection ──────────────────────────────────────────────────

pub fn forwardNormalizedRows(input: []const f32, gamma: []const f32, beta: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    validateProjection(input, rows_count, weight, bias, output);
    assert(gamma.len == weight.input_values_count);
    assert(beta.len == weight.input_values_count);
    assert(quantized_scratch.len >= rows_per_tile * weight.input_values_count);

    const rows_range = lane.range(rows_count);
    const quantized_rows = quantized_scratch[0 .. rows_per_tile * weight.input_values_count];
    var input_scales: [rows_per_tile]f32 = undefined;

    var tile_row_begin = rows_range.start_index;

    while (tile_row_begin < rows_range.end_index) : (tile_row_begin += rows_per_tile) {
        const tile_rows_count = @min(rows_per_tile, rows_range.end_index - tile_row_begin);
        for (0..tile_rows_count) |tile_row| {
            const input_row = input[(tile_row_begin + tile_row) * weight.input_values_count ..][0..weight.input_values_count];
            const statistics = normalization.calculateRowStatistics(input_row);
            const maximum = calculateNormalizedAbsoluteMaximum(input_row, gamma, beta, statistics);
            const input_scale = if (maximum == 0) 1.0 else 127.0 / maximum;
            const quantized_row = quantized_rows[tile_row * weight.input_values_count ..][0..weight.input_values_count];
            quantizeNormalizedRow(input_row, gamma, beta, statistics, input_scale, quantized_row);
            input_scales[tile_row] = input_scale;
        }

        const tile_output = output[tile_row_begin * weight.output_rows_count ..][0 .. tile_rows_count * weight.output_rows_count];
        if (tile_rows_count == rows_per_tile) {
            projectRowsTile(rows_per_tile, quantized_rows, &input_scales, weight, bias, .none, tile_output);
        } else {
            projectRowsTileForCount(tile_rows_count, quantized_rows, input_scales[0..tile_rows_count], weight, bias, .none, tile_output);
        }
    }
}

pub fn forwardRows(input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    validateProjection(input, rows_count, weight, bias, output);
    assert(quantized_scratch.len >= rows_per_tile * weight.input_values_count);

    const rows_range = lane.range(rows_count);
    const quantized_rows = quantized_scratch[0 .. rows_per_tile * weight.input_values_count];
    var input_scales: [rows_per_tile]f32 = undefined;

    var tile_row_begin = rows_range.start_index;
    while (tile_row_begin < rows_range.end_index) : (tile_row_begin += rows_per_tile) {
        const tile_rows_count = @min(rows_per_tile, rows_range.end_index - tile_row_begin);
        for (0..tile_rows_count) |tile_row| {
            const input_row = input[(tile_row_begin + tile_row) * weight.input_values_count ..][0..weight.input_values_count];
            const quantized_row = quantized_rows[tile_row * weight.input_values_count ..][0..weight.input_values_count];
            input_scales[tile_row] = quantizeRow(input_row, quantized_row);
        }

        const tile_output = output[tile_row_begin * weight.output_rows_count ..][0 .. tile_rows_count * weight.output_rows_count];
        if (tile_rows_count == rows_per_tile) {
            projectRowsTile(rows_per_tile, quantized_rows, &input_scales, weight, bias, activation, tile_output);
        } else {
            projectRowsTileForCount(tile_rows_count, quantized_rows, input_scales[0..tile_rows_count], weight, bias, activation, tile_output);
        }
    }
}

fn projectRowsTileForCount(rows_count: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(rows_count > 0);
    assert(rows_count < rows_per_tile);

    switch (rows_count) {
        1 => projectRowsTile(1, quantized_rows, input_scales, weight, bias, activation, output),
        2 => projectRowsTile(2, quantized_rows, input_scales, weight, bias, activation, output),
        3 => projectRowsTile(3, quantized_rows, input_scales, weight, bias, activation, output),
        4 => projectRowsTile(4, quantized_rows, input_scales, weight, bias, activation, output),
        5 => projectRowsTile(5, quantized_rows, input_scales, weight, bias, activation, output),
        6 => projectRowsTile(6, quantized_rows, input_scales, weight, bias, activation, output),
        7 => projectRowsTile(7, quantized_rows, input_scales, weight, bias, activation, output),
        8 => projectRowsTile(8, quantized_rows, input_scales, weight, bias, activation, output),
        9 => projectRowsTile(9, quantized_rows, input_scales, weight, bias, activation, output),
        10 => projectRowsTile(10, quantized_rows, input_scales, weight, bias, activation, output),
        11 => projectRowsTile(11, quantized_rows, input_scales, weight, bias, activation, output),
        12 => projectRowsTile(12, quantized_rows, input_scales, weight, bias, activation, output),
        13 => projectRowsTile(13, quantized_rows, input_scales, weight, bias, activation, output),
        else => unreachable,
    }
}

fn projectRowsTile(comptime rows_count: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(rows_count > 0);
    assert(rows_count <= rows_per_tile);
    assert(input_scales.len == rows_count);
    assert(quantized_rows.len >= rows_count * weight.input_values_count);
    assert(output.len == rows_count * weight.output_rows_count);

    const output_blocks_count = weight.output_rows_count / output_rows_per_block;
    for (0..output_blocks_count) |output_block_begin| {
        projectOutputBlocks(rows_count, 1, quantized_rows, input_scales, weight, bias, activation, output_block_begin, output);
    }
}

// ─── Token-Serial Projection ───────────────────────────────────────────────

pub fn forwardOne(input: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    validateProjection(input, 1, weight, bias, output);
    assert(quantized_scratch.len >= weight.input_values_count);

    const quantized_input = quantized_scratch[0..weight.input_values_count];
    const input_scale = quantizeRow(input, quantized_input);
    const output_blocks_count = weight.output_rows_count / output_rows_per_block;
    const output_blocks_range = lane.range(output_blocks_count);
    projectOneOutputRange(quantized_input, input_scale, weight, bias, activation, output_blocks_range.start_index, output_blocks_range.end_index, output);
}

pub fn forwardQuantizedOne(quantized_input: []const u8, input_scale: f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(quantized_input.len == weight.input_values_count);
    assert(std.math.isFinite(input_scale));
    assert(input_scale > 0);
    assert(output.len == weight.output_rows_count);
    assert(bias.len == 0 or bias.len == weight.output_rows_count);

    const output_blocks_count = weight.output_rows_count / output_rows_per_block;
    projectOneOutputRange(quantized_input, input_scale, weight, bias, activation, 0, output_blocks_count, output);
}

fn projectOneOutputRange(quantized_input: []const u8, input_scale: f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output_block_begin: usize, output_block_end: usize, output: []f32) void {
    assert(output_block_begin <= output_block_end);
    assert(output_block_end <= weight.output_rows_count / output_rows_per_block);

    const input_scales = [1]f32{input_scale};
    var block_begin = output_block_begin;
    while (block_begin + output_blocks_per_token_iteration <= output_block_end) : (block_begin += output_blocks_per_token_iteration) {
        projectOutputBlocks(1, output_blocks_per_token_iteration, quantized_input, &input_scales, weight, bias, activation, block_begin, output);
    }

    const remaining_blocks_count = output_block_end - block_begin;
    switch (remaining_blocks_count) {
        0 => {},
        1 => projectOutputBlocks(1, 1, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        2 => projectOutputBlocks(1, 2, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        3 => projectOutputBlocks(1, 3, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        4 => projectOutputBlocks(1, 4, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        5 => projectOutputBlocks(1, 5, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        6 => projectOutputBlocks(1, 6, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        7 => projectOutputBlocks(1, 7, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        8 => projectOutputBlocks(1, 8, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        9 => projectOutputBlocks(1, 9, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        10 => projectOutputBlocks(1, 10, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        11 => projectOutputBlocks(1, 11, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        12 => projectOutputBlocks(1, 12, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        13 => projectOutputBlocks(1, 13, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        else => unreachable,
    }
}

// ─── Joined Encoder FFN ────────────────────────────────────────────────────

pub fn forwardFeedForwardRows(input: []const f32, rows_count: usize, expansion_weight: QuantizedWeight, expansion_bias: []const f32, contraction_weight: QuantizedWeight, contraction_bias: []const f32, output: []f32, float_scratch: []f32, quantized_scratch: []u8, lane: Lane) void {
    assert(expansion_weight.input_values_count == contraction_weight.output_rows_count);
    assert(expansion_weight.output_rows_count == contraction_weight.input_values_count);
    assert(input.len == rows_count * expansion_weight.input_values_count);
    assert(output.len == rows_count * contraction_weight.output_rows_count);
    assert(expansion_bias.len == expansion_weight.output_rows_count);
    assert(contraction_bias.len == contraction_weight.output_rows_count);

    const model_width = expansion_weight.input_values_count;
    const ffn_width = expansion_weight.output_rows_count;
    assert(float_scratch.len >= ffn_rows_per_tile * ffn_width);
    assert(quantized_scratch.len >= ffn_rows_per_tile * (model_width + ffn_width));

    const expanded = float_scratch[0 .. ffn_rows_per_tile * ffn_width];
    const quantized_expansion_input = quantized_scratch[0 .. ffn_rows_per_tile * model_width];
    const quantized_contraction_input = quantized_scratch[ffn_rows_per_tile * model_width ..][0 .. ffn_rows_per_tile * ffn_width];
    var expansion_input_scales: [ffn_rows_per_tile]f32 = undefined;
    var contraction_input_scales: [ffn_rows_per_tile]f32 = undefined;
    const rows_range = lane.range(rows_count);

    var tile_row_begin = rows_range.start_index;
    while (tile_row_begin < rows_range.end_index) : (tile_row_begin += ffn_rows_per_tile) {
        const tile_rows_count = @min(ffn_rows_per_tile, rows_range.end_index - tile_row_begin);
        for (0..tile_rows_count) |tile_row| {
            const input_row = input[(tile_row_begin + tile_row) * model_width ..][0..model_width];
            const quantized_row = quantized_expansion_input[tile_row * model_width ..][0..model_width];
            expansion_input_scales[tile_row] = quantizeRow(input_row, quantized_row);
        }

        const expanded_values = expanded[0 .. tile_rows_count * ffn_width];
        projectFeedForwardTileForRows(tile_rows_count, quantized_expansion_input, expansion_input_scales[0..tile_rows_count], expansion_weight, expansion_bias, .gelu, expanded_values);
        for (0..tile_rows_count) |tile_row| {
            const expanded_row = expanded_values[tile_row * ffn_width ..][0..ffn_width];
            const quantized_row = quantized_contraction_input[tile_row * ffn_width ..][0..ffn_width];
            contraction_input_scales[tile_row] = quantizeRow(expanded_row, quantized_row);
        }

        const tile_output = output[tile_row_begin * model_width ..][0 .. tile_rows_count * model_width];
        projectFeedForwardTileForRows(tile_rows_count, quantized_contraction_input, contraction_input_scales[0..tile_rows_count], contraction_weight, contraction_bias, .none, tile_output);
    }
}

fn projectFeedForwardTileForRows(rows_count: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    switch (rows_count) {
        1 => projectFeedForwardTile(1, quantized_rows, input_scales, weight, bias, activation, output),
        2 => projectFeedForwardTile(2, quantized_rows, input_scales, weight, bias, activation, output),
        3 => projectFeedForwardTile(3, quantized_rows, input_scales, weight, bias, activation, output),
        4 => projectFeedForwardTile(4, quantized_rows, input_scales, weight, bias, activation, output),
        else => unreachable,
    }
}

fn projectFeedForwardTile(comptime rows_count: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(rows_count > 0);
    assert(rows_count <= ffn_rows_per_tile);
    assert(input_scales.len == rows_count);
    assert(quantized_rows.len >= ffn_rows_per_tile * weight.input_values_count);
    assert(output.len == input_scales.len * weight.output_rows_count);

    const complete_iterations_count = weight.output_rows_count / (output_blocks_per_ffn_iteration * output_rows_per_block);
    for (0..complete_iterations_count) |iteration| {
        const output_block_begin = iteration * output_blocks_per_ffn_iteration;
        projectOutputBlocks(rows_count, output_blocks_per_ffn_iteration, quantized_rows, input_scales, weight, bias, activation, output_block_begin, output);
    }

    const remaining_output_rows_count = weight.output_rows_count % (output_blocks_per_ffn_iteration * output_rows_per_block);
    if (remaining_output_rows_count != 0) {
        assert(remaining_output_rows_count == output_rows_per_block);
        projectOutputBlocks(rows_count, 1, quantized_rows, input_scales, weight, bias, activation, complete_iterations_count * output_blocks_per_ffn_iteration, output);
    }
}

inline fn projectOutputBlocks(comptime rows_count: usize, comptime output_blocks_count: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output_block_begin: usize, output: []f32) void {
    assert(rows_count > 0);
    assert(rows_count <= rows_per_tile);
    assert(output_blocks_count > 0);
    assert(output_blocks_count <= output_blocks_per_token_iteration);
    assert(rows_count * output_blocks_count <= accumulators_count_max);
    assert(quantized_rows.len >= rows_count * weight.input_values_count);
    assert(input_scales.len == rows_count);
    assert(output_block_begin + output_blocks_count <= weight.output_rows_count / output_rows_per_block);
    assert(output.len == rows_count * weight.output_rows_count);
    assert(bias.len == 0 or bias.len == weight.output_rows_count);

    const weight_layout = weight.layout();
    var accumulators: [rows_count][output_blocks_count]I32x8 = undefined;
    inline for (0..output_blocks_count) |block_offset| {
        const output_begin = (output_block_begin + block_offset) * output_rows_per_block;
        const compensation: I32x8 = weight.compensation[output_begin..][0..output_rows_per_block].*;
        inline for (0..rows_count) |tile_row| {
            accumulators[tile_row][block_offset] = compensation;
        }
    }

    var depth_begin: usize = 0;
    while (depth_begin < weight.input_values_count) : (depth_begin += depth_values_per_group) {

        // PERFORMANCE: One-row tiles load and consume each weight immediately.
        // Fourteen accumulators leave only two of AVX2's sixteen YMM registers
        // for the activation and current weight, so preloading all weights would
        // spill. Multi-row tiles preload each output-block weight once and reuse
        // it across rows. The supported shapes fit accumulators, weights, and one
        // activation in sixteen registers; loading inside the row loop would
        // repeat each weight load for every row.
        if (rows_count == 1) {
            const activation_bytes = quantized_rows[depth_begin..][0..depth_values_per_group].*;
            const activations: I32x8 = @splat(@as(i32, @bitCast(activation_bytes)));
            inline for (0..output_blocks_count) |block_offset| {
                const packed_block_index = output_block_begin + block_offset;
                const packed_offset = weight_layout.groupOffset(packed_block_index, depth_begin);
                const weights: I8x32 = weight.values[packed_offset..][0..bytes_per_weight_group].*;
                accumulators[0][block_offset] = dotUnsignedSignedBytes(accumulators[0][block_offset], activations, weights);
            }
        } else {
            var weights: [output_blocks_count]I8x32 = undefined;
            inline for (0..output_blocks_count) |block_offset| {
                const packed_block_index = output_block_begin + block_offset;
                const packed_offset = weight_layout.groupOffset(packed_block_index, depth_begin);
                weights[block_offset] = weight.values[packed_offset..][0..bytes_per_weight_group].*;
            }
            inline for (0..rows_count) |tile_row| {
                const activation_bytes = quantized_rows[tile_row * weight.input_values_count + depth_begin ..][0..depth_values_per_group].*;
                const activations: I32x8 = @splat(@as(i32, @bitCast(activation_bytes)));
                inline for (0..output_blocks_count) |block_offset| {
                    accumulators[tile_row][block_offset] = dotUnsignedSignedBytes(accumulators[tile_row][block_offset], activations, weights[block_offset]);
                }
            }
        }
    }

    const reciprocal_input_scale: F32x8 = if (rows_count == 1) @splat(1.0 / input_scales[0]) else undefined;

    inline for (0..output_blocks_count) |block_offset| {
        const output_begin = (output_block_begin + block_offset) * output_rows_per_block;
        const output_scales: F32x8 = weight.scales[output_begin..][0..output_rows_per_block].*;
        const output_bias: F32x8 = if (bias.len == 0) @splat(0) else bias[output_begin..][0..output_rows_per_block].*;

        inline for (0..rows_count) |tile_row| {
            const tile_row_reciprocal_input_scale: F32x8 = if (rows_count == 1) reciprocal_input_scale else @splat(1.0 / input_scales[tile_row]);
            var values: F32x8 = @floatFromInt(accumulators[tile_row][block_offset]);
            values *= tile_row_reciprocal_input_scale;
            values /= output_scales;
            values += output_bias;
            if (activation == .gelu) {
                values = gelu(values);
            }
            output[tile_row * weight.output_rows_count + output_begin ..][0..output_rows_per_block].* = values;
        }
    }
}

// ─── Numerical Primitives ──────────────────────────────────────────────────

inline fn calculateNormalizedAbsoluteMaximum(input: []const f32, gamma: []const f32, beta: []const f32, statistics: normalization.RowStatistics) f32 {
    var maximums: F32x8 = @splat(0);
    var column: usize = 0;
    while (column < input.len) : (column += simd_lanes_count) {
        maximums = @max(maximums, @abs(normalization.normalizedVector(input, gamma, beta, statistics, column)));
    }
    return @reduce(.Max, maximums);
}

fn quantizeNormalizedRow(input: []const f32, gamma: []const f32, beta: []const f32, statistics: normalization.RowStatistics, scale: f32, quantized: []u8) void {
    assert(input.len == gamma.len);
    assert(input.len == beta.len);
    assert(input.len == quantized.len);

    const scales: F32x8 = @splat(scale);
    const shifts: F32x8 = @splat(@as(f32, @floatFromInt(vnni_weight.activation_zero_point)));
    var column: usize = 0;
    while (column < input.len) : (column += simd_lanes_count) {
        const normalized = normalization.normalizedVector(input, gamma, beta, statistics, column);
        const shifted = @mulAdd(F32x8, normalized, scales, shifts);
        const integers: I32x8 = @intFromFloat(vnni_weight.roundFloat32VectorToNearestEven(shifted));
        quantized[column..][0..simd_lanes_count].* = @as(U8x8, @intCast(integers));
    }
}

pub fn quantizeRow(values: []const f32, quantized: []u8) f32 {
    assert(values.len == quantized.len);
    assert(values.len % simd_lanes_count == 0);

    var maximums: F32x8 = @splat(0);
    var value_index: usize = 0;
    while (value_index < values.len) : (value_index += simd_lanes_count) {
        const value_vector: F32x8 = values[value_index..][0..simd_lanes_count].*;
        maximums = @max(maximums, @abs(value_vector));
    }

    const maximum = @reduce(.Max, maximums);
    const scale = if (maximum == 0) 1.0 else 127.0 / maximum;
    const scales: F32x8 = @splat(scale);
    const shifts: F32x8 = @splat(@as(f32, @floatFromInt(vnni_weight.activation_zero_point)));

    value_index = 0;
    while (value_index < values.len) : (value_index += simd_lanes_count) {
        const shifted = @mulAdd(F32x8, values[value_index..][0..simd_lanes_count].*, scales, shifts);
        const integers: I32x8 = @intFromFloat(vnni_weight.roundFloat32VectorToNearestEven(shifted));
        quantized[value_index..][0..simd_lanes_count].* = @as(U8x8, @intCast(integers));
    }

    return scale;
}

inline fn dotUnsignedSignedBytes(accumulator: I32x8, activations: I32x8, weights: I8x32) I32x8 {
    if (builtin.zig_backend != .stage2_llvm) {
        // Zig's self-hosted x86 backend does not support the AVX-VNNI mnemonic yet. Keep those builds usable with an equivalent scalar implementation.
        const activation_bytes: [depth_values_per_group]u8 = @bitCast(activations[0]);
        const weight_bytes: [bytes_per_weight_group]i8 = weights;
        var result: [simd_lanes_count]i32 = accumulator;
        for (0..result.len) |output_index| {
            for (0..activation_bytes.len) |depth_index| {
                result[output_index] += @as(i32, activation_bytes[depth_index]) * @as(i32, weight_bytes[output_index * activation_bytes.len + depth_index]);
            }
        }
        return result;
    }

    var result = accumulator;
    asm volatile ("{vex} vpdpbusd %[weights], %[activations], %[result]"
        : [result] "+x" (result),
        : [activations] "x" (activations),
          [weights] "x" (weights),
    );
    return result;
}

inline fn gelu(values: F32x8) F32x8 {
    const scaled = values * @as(F32x8, @splat(0.7071067811865475));
    return @as(F32x8, @splat(0.5)) * values * (@as(F32x8, @splat(1.0)) + errorFunction(scaled));
}

inline fn errorFunction(values: F32x8) F32x8 {
    const sign_bit: U32x8 = @splat(0x8000_0000);
    const value_bits: U32x8 = @bitCast(values);
    const sign_mask = value_bits & sign_bit;
    const absolute_values: F32x8 = @bitCast(value_bits ^ sign_mask);
    const ones: F32x8 = @splat(1.0);
    const t = ones / @mulAdd(F32x8, @as(F32x8, @splat(0.3275911)), absolute_values, ones);

    var polynomial = @mulAdd(F32x8, @as(F32x8, @splat(1.061405429)), t, @as(F32x8, @splat(-1.453152027)));
    polynomial = @mulAdd(F32x8, polynomial, t, @as(F32x8, @splat(1.421413741)));
    polynomial = @mulAdd(F32x8, polynomial, t, @as(F32x8, @splat(-0.284496736)));
    polynomial = @mulAdd(F32x8, polynomial, t, @as(F32x8, @splat(0.254829592)));

    const negative_squares: F32x8 = @bitCast(@as(U32x8, @bitCast(values * values)) ^ sign_bit);
    const negative_exponentials: F32x8 = @bitCast(@as(U32x8, @bitCast(expApproximation(negative_squares))) ^ sign_bit);
    const magnitudes = @mulAdd(F32x8, negative_exponentials * t, polynomial, ones);

    return @bitCast(@as(U32x8, @bitCast(magnitudes)) ^ sign_mask);
}

// Attention retains a separate exponential approximation because its strict
// softmax evaluation and reduction order are a distinct numerical contract.
inline fn expApproximation(input: F32x8) F32x8 {
    const ones: F32x8 = @splat(1.0);
    var values = @max(@min(input, @as(F32x8, @splat(88.3762626647949))), @as(F32x8, @splat(-88.3762626647949)));
    var powers = values * @as(F32x8, @splat(1.44269504088896341)) + @as(F32x8, @splat(0.5));
    const floored = @floor(powers);
    powers = floored - @select(f32, floored > powers, ones, @as(F32x8, @splat(0.0)));
    values -= powers * @as(F32x8, @splat(0.693359375));
    values -= powers * @as(F32x8, @splat(-2.12194440e-4));
    const squares = values * values;

    var polynomial: F32x8 = @splat(1.9875691500e-4);
    polynomial = polynomial * values + @as(F32x8, @splat(1.3981999507e-3));
    polynomial = polynomial * values + @as(F32x8, @splat(8.3334519073e-3));
    polynomial = polynomial * values + @as(F32x8, @splat(4.1665795894e-2));
    polynomial = polynomial * values + @as(F32x8, @splat(1.6666665459e-1));
    polynomial = polynomial * values + @as(F32x8, @splat(5.0000001201e-1));
    polynomial = polynomial * squares + values + ones;

    const integer_powers: I32x8 = @intFromFloat(powers);
    const biased_powers: U32x8 = @bitCast(integer_powers + @as(I32x8, @splat(127)));
    const exponent_bits = biased_powers << @as(@Vector(simd_lanes_count, u5), @splat(23));

    return polynomial * @as(F32x8, @bitCast(exponent_bits));
}

fn validateProjection(input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, output: []f32) void {
    assert(rows_count > 0);
    _ = weight.layout();
    assert(input.len == rows_count * weight.input_values_count);
    assert(output.len == rows_count * weight.output_rows_count);
    assert(bias.len == 0 or bias.len == weight.output_rows_count);
}
