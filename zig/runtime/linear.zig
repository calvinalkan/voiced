//! Quantized linear projections over the runtime's permanent o8/k4 weight layout.
//!
//! Multi-row projections reuse each weight vector across fourteen activation
//! rows. Token-serial projections retain up to fourteen independent output
//! accumulators. Encoder FFN pairs use four-row tiles and never materialize the
//! full `[positions, ffn_width]` expansion tensor.

const builtin = @import("builtin");
const std = @import("std");
const attention = @import("attention.zig");
const normalization = @import("normalization.zig");
const vnni_weight = @import("vnni_weight.zig");
const Lane = @import("executor.zig").Lane;
const QuantizedWeight = vnni_weight.QuantizedWeight;
const assert = std.debug.assert;

pub const rows_per_tile: usize = 14;
pub const ffn_rows_per_tile: usize = 4;
const direct_layout_rows_per_tile: usize = 8;

const output_rows_per_block = vnni_weight.output_rows_per_block;
const output_blocks_per_token_iteration: usize = 14;
const output_blocks_per_ffn_iteration: usize = 3;
const accumulators_count_max: usize = 14;
const depth_values_per_group = vnni_weight.input_values_per_group;
const bytes_per_weight_group = vnni_weight.values_per_group;
const simd_lanes_count: usize = 8;

const F16x8 = @Vector(simd_lanes_count, f16);
const F32x8 = @Vector(simd_lanes_count, f32);
const I32x8 = @Vector(simd_lanes_count, i32);
const U32x8 = @Vector(simd_lanes_count, u32);
const I8x32 = @Vector(bytes_per_weight_group, i8);
const U8x8 = @Vector(simd_lanes_count, u8);

comptime {
    @setEvalBranchQuota(10_000);
}

pub const Activation = enum {
    none,
    gelu,
};

// ─── Multi-Row Projection ──────────────────────────────────────────────────

/// Every executor lane must participate; callers synchronize after return.
pub fn forwardNormalizedEncoderQueryKeyValue(input: []const f32, gamma: []const f32, beta: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    const model_width = @divExact(weight.scales.len, 3);

    assert(rows_count > 0);
    assert(weight.scales.len == 3 * model_width);
    assert(model_width % attention.head_width == 0);
    assert(input.len == rows_count * weight.input_values_count);
    assert(gamma.len == weight.input_values_count);
    assert(beta.len == weight.input_values_count);
    assert(bias.len == weight.scales.len);
    assert(output.len == attention.encoderQueryKeyValueValuesCount(rows_count, model_width));
    assert(quantized_scratch.len >= direct_layout_rows_per_tile * weight.input_values_count);

    // A lane owns complete eight-position key blocks, so the transpose stores
    // never cross a block or another lane's cache line.
    var tiles = lane.tiles(std.math.divCeil(usize, rows_count, direct_layout_rows_per_tile) catch unreachable, 4);
    const quantized_rows = quantized_scratch[0 .. direct_layout_rows_per_tile * weight.input_values_count];
    var input_scales: [direct_layout_rows_per_tile]f32 = undefined;

    while (tiles.next()) |tile_index| {
        const tile_row_begin = tile_index * direct_layout_rows_per_tile;
        const tile_rows_count = @min(direct_layout_rows_per_tile, rows_count - tile_row_begin);
        for (0..tile_rows_count) |tile_row| {
            const input_row = input[(tile_row_begin + tile_row) * weight.input_values_count ..][0..weight.input_values_count];
            const quantized_row = quantized_rows[tile_row * weight.input_values_count ..][0..weight.input_values_count];
            input_scales[tile_row] = quantizeNormalized(input_row, gamma, beta, quantized_row);
        }

        projectDirectLayoutRows(.encoder_query_key_value, tile_rows_count, tile_row_begin, quantized_rows, input_scales[0..tile_rows_count], weight, bias, rows_count, output);
    }
}

/// `forwardRows` permits identical `input` and `output` slices for square
/// weights; partial overlap is unsupported. Each lane's `quantized_scratch`
/// must be disjoint from inputs, outputs, and other lanes' scratch.
/// Every executor lane must participate; callers synchronize after return.
pub fn forwardRows(input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    forwardRowProjection(false, input, rows_count, weight, bias, activation, output, quantized_scratch, lane);
}

/// Adds the projection to `output`. Identical input/output slices are supported
/// for square weights; partial overlap is unsupported. Scratch must be disjoint
/// from all input/output storage and other lanes. Every executor lane must
/// participate; callers synchronize after return.
pub fn forwardResidualRows(input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    forwardRowProjection(true, input, rows_count, weight, bias, .none, output, quantized_scratch, lane);
}

fn forwardRowProjection(comptime add_to_output: bool, input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    validateProjection(input, rows_count, weight, bias, output);
    assert(quantized_scratch.len >= rows_per_tile * weight.input_values_count);

    var tiles = lane.tiles(std.math.divCeil(usize, rows_count, rows_per_tile) catch unreachable, 2);
    const quantized_rows = quantized_scratch[0 .. rows_per_tile * weight.input_values_count];
    var input_scales: [rows_per_tile]f32 = undefined;

    while (tiles.next()) |tile_index| {
        const tile_row_begin = tile_index * rows_per_tile;
        const tile_rows_count = @min(rows_per_tile, rows_count - tile_row_begin);
        // Keep the complete input tile in scratch before any output writes so
        // square projections can replace their input in place.
        for (0..tile_rows_count) |tile_row| {
            const input_row = input[(tile_row_begin + tile_row) * weight.input_values_count ..][0..weight.input_values_count];
            const quantized_row = quantized_rows[tile_row * weight.input_values_count ..][0..weight.input_values_count];
            input_scales[tile_row] = quantizeRow(input_row, quantized_row);
        }

        const tile_output = output[tile_row_begin * weight.scales.len ..][0 .. tile_rows_count * weight.scales.len];
        // PERFORMANCE: Full tiles pass fixed-size scale storage directly to
        // the kernel, avoiding a runtime row-count switch for every full tile.
        // Only the partial final tile needs generic dispatch.
        if (tile_rows_count == rows_per_tile) {
            projectRowsTile(rows_per_tile, add_to_output, quantized_rows, &input_scales, weight, bias, activation, tile_output);
        } else {
            projectRowsForCount(add_to_output, quantized_rows, input_scales[0..tile_rows_count], weight, bias, activation, tile_output);
        }
    }
}

/// Every executor lane must participate; callers synchronize after return.
pub fn forwardDecoderCrossKeyValues(input: []const f32, rows_count: usize, weight: QuantizedWeight, bias: []const f32, positions_capacity: usize, output: []f16, quantized_scratch: []u8, lane: Lane) void {
    const model_width = @divExact(weight.scales.len, 2);
    const expected_output_values_count = attention.packedKeyValuesCount(positions_capacity, model_width) + positions_capacity * model_width;

    assert(rows_count > 0);
    assert(rows_count <= positions_capacity);
    assert(weight.scales.len == 2 * model_width);
    assert(model_width % attention.head_width == 0);
    assert(input.len == rows_count * weight.input_values_count);
    assert(bias.len == weight.scales.len);
    assert(output.len == expected_output_values_count);
    assert(quantized_scratch.len >= direct_layout_rows_per_tile * weight.input_values_count);

    var tiles = lane.tiles(std.math.divCeil(usize, rows_count, direct_layout_rows_per_tile) catch unreachable, 4);
    const quantized_rows = quantized_scratch[0 .. direct_layout_rows_per_tile * weight.input_values_count];
    var input_scales: [direct_layout_rows_per_tile]f32 = undefined;

    while (tiles.next()) |tile_index| {
        const tile_row_begin = tile_index * direct_layout_rows_per_tile;
        const tile_rows_count = @min(direct_layout_rows_per_tile, rows_count - tile_row_begin);
        for (0..tile_rows_count) |tile_row| {
            const input_row = input[(tile_row_begin + tile_row) * weight.input_values_count ..][0..weight.input_values_count];
            const quantized_row = quantized_rows[tile_row * weight.input_values_count ..][0..weight.input_values_count];
            input_scales[tile_row] = quantizeRow(input_row, quantized_row);
        }

        projectDirectLayoutRows(.decoder_cross_key_value, tile_rows_count, tile_row_begin, quantized_rows, input_scales[0..tile_rows_count], weight, bias, positions_capacity, output);
    }
}

/// `forwardQuantizedRows` projects one tile of independently quantized rows.
/// The caller owns row partitioning; this operation does not synchronize.
pub fn forwardQuantizedRows(quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    projectRowsForCount(false, quantized_rows, input_scales, weight, bias, activation, output);
}

fn projectRowsForCount(comptime add_to_output: bool, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    const rows_count = input_scales.len;
    assert(rows_count > 0 and rows_count <= rows_per_tile);
    switch (rows_count) {
        inline 1...rows_per_tile => |count| projectRowsTile(count, add_to_output, quantized_rows, input_scales, weight, bias, activation, output),
        else => unreachable,
    }
}

fn projectRowsTile(comptime rows_count: usize, comptime add_to_output: bool, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(rows_count > 0);
    assert(rows_count <= rows_per_tile);
    assert(input_scales.len == rows_count);
    assert(quantized_rows.len >= rows_count * weight.input_values_count);
    assert(output.len == rows_count * weight.scales.len);

    const output_blocks_count = weight.scales.len / output_rows_per_block;
    for (0..output_blocks_count) |output_block_begin| {
        projectOutputBlocks(rows_count, 1, add_to_output, quantized_rows, input_scales, weight, bias, activation, output_block_begin, output);
    }
}

// ─── Token-Serial Projection ───────────────────────────────────────────────

pub fn forwardOne(input: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32, quantized_scratch: []u8, lane: Lane) void {
    validateProjection(input, 1, weight, bias, output);
    assert(quantized_scratch.len >= weight.input_values_count);

    const quantized_input = quantized_scratch[0..weight.input_values_count];
    const input_scale = quantizeRow(input, quantized_input);
    forwardQuantizedOneParallel(quantized_input, input_scale, weight, bias, activation, output, lane);
}

pub fn forwardQuantizedOne(quantized_input: []const u8, input_scale: f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(quantized_input.len == weight.input_values_count);
    assert(std.math.isFinite(input_scale));
    assert(input_scale > 0);
    assert(output.len == weight.scales.len);
    assert(bias.len == 0 or bias.len == weight.scales.len);

    const output_blocks_count = weight.scales.len / output_rows_per_block;
    projectOneOutputRange(quantized_input, input_scale, weight, bias, activation, 0, output_blocks_count, output);
}

pub fn forwardQuantizedOneParallel(quantized_input: []const u8, input_scale: f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32, lane: Lane) void {
    assert(quantized_input.len == weight.input_values_count);
    assert(std.math.isFinite(input_scale));
    assert(input_scale > 0);
    assert(output.len == weight.scales.len);
    assert(bias.len == 0 or bias.len == weight.scales.len);

    const output_blocks_count = weight.scales.len / output_rows_per_block;
    const output_blocks_range = lane.range(output_blocks_count);
    projectOneOutputRange(quantized_input, input_scale, weight, bias, activation, output_blocks_range.start_index, output_blocks_range.end_index, output);
}

fn projectOneOutputRange(quantized_input: []const u8, input_scale: f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output_block_begin: usize, output_block_end: usize, output: []f32) void {
    assert(output_block_begin <= output_block_end);
    assert(output_block_end <= weight.scales.len / output_rows_per_block);

    const input_scales = [1]f32{input_scale};
    var block_begin = output_block_begin;
    while (block_begin + output_blocks_per_token_iteration <= output_block_end) : (block_begin += output_blocks_per_token_iteration) {
        projectOutputBlocks(1, output_blocks_per_token_iteration, false, quantized_input, &input_scales, weight, bias, activation, block_begin, output);
    }

    const remaining_blocks_count = output_block_end - block_begin;
    switch (remaining_blocks_count) {
        0 => {},
        inline 1...output_blocks_per_token_iteration - 1 => |count| projectOutputBlocks(1, count, false, quantized_input, &input_scales, weight, bias, activation, block_begin, output),
        else => unreachable,
    }
}

// ─── Joined Encoder FFN ────────────────────────────────────────────────────

/// Replaces `values` with values + FFN(LayerNorm(values)). Each lane owns whole
/// row tiles and disjoint scratch; every executor lane must participate.
/// Callers synchronize after return.
pub fn forwardFeedForwardResidualRows(values: []f32, gamma: []const f32, beta: []const f32, rows_count: usize, expansion_weight: QuantizedWeight, expansion_bias: []const f32, contraction_weight: QuantizedWeight, contraction_bias: []const f32, float_scratch: []f32, quantized_scratch: []u8, lane: Lane) void {
    assert(expansion_weight.input_values_count == contraction_weight.scales.len);
    assert(expansion_weight.scales.len == contraction_weight.input_values_count);
    assert(values.len == rows_count * expansion_weight.input_values_count);
    assert(expansion_bias.len == expansion_weight.scales.len);
    assert(contraction_bias.len == contraction_weight.scales.len);

    const model_width = expansion_weight.input_values_count;
    const ffn_width = expansion_weight.scales.len;
    assert(gamma.len == model_width);
    assert(beta.len == model_width);
    assert(float_scratch.len >= ffn_rows_per_tile * ffn_width);
    assert(quantized_scratch.len >= ffn_rows_per_tile * @max(model_width, ffn_width));

    const expanded = float_scratch[0 .. ffn_rows_per_tile * ffn_width];
    const quantized_rows = quantized_scratch[0 .. ffn_rows_per_tile * @max(model_width, ffn_width)];
    var input_scales: [ffn_rows_per_tile]f32 = undefined;
    var tiles = lane.tiles(std.math.divCeil(usize, rows_count, ffn_rows_per_tile) catch unreachable, 4);

    while (tiles.next()) |tile_index| {
        const tile_row_begin = tile_index * ffn_rows_per_tile;
        const tile_rows_count = @min(ffn_rows_per_tile, rows_count - tile_row_begin);
        for (0..tile_rows_count) |tile_row| {
            const input_row = values[(tile_row_begin + tile_row) * model_width ..][0..model_width];
            const quantized_row = quantized_rows[tile_row * model_width ..][0..model_width];
            // Expansion storage is dead until this tile is quantized. Reuse one
            // row to retain normalized floats while finding their quantization
            // scale, avoiding both a full tensor and repeated normalization.
            input_scales[tile_row] = quantizeNormalizedWithScratch(input_row, gamma, beta, expanded[0..model_width], quantized_row);
        }

        const expanded_values = expanded[0 .. tile_rows_count * ffn_width];
        projectFeedForwardTileForRows(false, tile_rows_count, quantized_rows, input_scales[0..tile_rows_count], expansion_weight, expansion_bias, .gelu, expanded_values);

        // Expansion has consumed the quantized rows and scales. Reuse them at
        // the wider row stride for contraction while expanded floats stay live.
        for (0..tile_rows_count) |tile_row| {
            const expanded_row = expanded_values[tile_row * ffn_width ..][0..ffn_width];
            const quantized_row = quantized_rows[tile_row * ffn_width ..][0..ffn_width];
            input_scales[tile_row] = quantizeRow(expanded_row, quantized_row);
        }

        const tile_output = values[tile_row_begin * model_width ..][0 .. tile_rows_count * model_width];
        projectFeedForwardTileForRows(true, tile_rows_count, quantized_rows, input_scales[0..tile_rows_count], contraction_weight, contraction_bias, .none, tile_output);
    }
}

fn projectFeedForwardTileForRows(comptime add_to_output: bool, rows_count: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    switch (rows_count) {
        inline 1...ffn_rows_per_tile => |count| projectFeedForwardTile(count, add_to_output, quantized_rows, input_scales, weight, bias, activation, output),
        else => unreachable,
    }
}

fn projectFeedForwardTile(comptime rows_count: usize, comptime add_to_output: bool, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output: []f32) void {
    assert(rows_count > 0);
    assert(rows_count <= ffn_rows_per_tile);
    assert(input_scales.len == rows_count);
    assert(quantized_rows.len >= ffn_rows_per_tile * weight.input_values_count);
    assert(output.len == input_scales.len * weight.scales.len);

    const complete_iterations_count = weight.scales.len / (output_blocks_per_ffn_iteration * output_rows_per_block);
    for (0..complete_iterations_count) |iteration| {
        const output_block_begin = iteration * output_blocks_per_ffn_iteration;
        projectOutputBlocks(rows_count, output_blocks_per_ffn_iteration, add_to_output, quantized_rows, input_scales, weight, bias, activation, output_block_begin, output);
    }

    const remaining_output_rows_count = weight.scales.len % (output_blocks_per_ffn_iteration * output_rows_per_block);
    if (remaining_output_rows_count != 0) {
        assert(remaining_output_rows_count == output_rows_per_block);
        projectOutputBlocks(rows_count, 1, add_to_output, quantized_rows, input_scales, weight, bias, activation, complete_iterations_count * output_blocks_per_ffn_iteration, output);
    }
}

const ProjectionOutputLayout = enum {
    row_major,
    encoder_query_key_value,
    decoder_cross_key_value,
};

fn projectDirectLayoutRows(comptime output_layout: ProjectionOutputLayout, rows_count: usize, output_row_begin: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, positions_capacity: usize, output: anytype) void {
    assert(output_layout != .row_major);
    assert(rows_count > 0);
    assert(rows_count <= direct_layout_rows_per_tile);

    switch (rows_count) {
        inline 1...direct_layout_rows_per_tile => |count| projectDirectLayoutRowsForCount(count, output_layout, output_row_begin, quantized_rows, input_scales, weight, bias, positions_capacity, output),
        else => unreachable,
    }
}

inline fn projectDirectLayoutRowsForCount(comptime rows_count: usize, comptime output_layout: ProjectionOutputLayout, output_row_begin: usize, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, positions_capacity: usize, output: anytype) void {
    @setEvalBranchQuota(10_000);

    const output_blocks_count = weight.scales.len / output_rows_per_block;
    for (0..output_blocks_count) |output_block_begin| {
        projectOutputBlocksWithLayout(rows_count, 1, output_layout, false, quantized_rows, input_scales, weight, bias, .none, output_block_begin, output_row_begin, positions_capacity, output);
    }
}

inline fn projectOutputBlocks(comptime rows_count: usize, comptime output_blocks_count: usize, comptime add_to_output: bool, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output_block_begin: usize, output: []f32) void {
    projectOutputBlocksWithLayout(rows_count, output_blocks_count, .row_major, add_to_output, quantized_rows, input_scales, weight, bias, activation, output_block_begin, 0, 0, output);
}

inline fn projectOutputBlocksWithLayout(comptime rows_count: usize, comptime output_blocks_count: usize, comptime output_layout: ProjectionOutputLayout, comptime add_to_output: bool, quantized_rows: []const u8, input_scales: []const f32, weight: QuantizedWeight, bias: []const f32, activation: Activation, output_block_begin: usize, output_row_begin: usize, positions_capacity: usize, output: anytype) void {
    @setEvalBranchQuota(10_000);

    assert(rows_count > 0);
    assert(rows_count <= rows_per_tile);
    assert(output_blocks_count > 0);
    assert(output_blocks_count <= output_blocks_per_token_iteration);
    assert(rows_count * output_blocks_count <= accumulators_count_max);
    assert(quantized_rows.len >= rows_count * weight.input_values_count);
    assert(input_scales.len == rows_count);
    assert(output_block_begin + output_blocks_count <= weight.scales.len / output_rows_per_block);
    assert(bias.len == 0 or bias.len == weight.scales.len);
    assert(!add_to_output or output_layout == .row_major);

    switch (output_layout) {
        .row_major => {
            assert(output.len == rows_count * weight.scales.len);
            assert(output_row_begin == 0);
            assert(positions_capacity == 0);
        },
        .encoder_query_key_value => {
            const model_width = @divExact(weight.scales.len, 3);
            assert(output_row_begin + rows_count <= positions_capacity);
            assert(output.len == attention.encoderQueryKeyValueValuesCount(positions_capacity, model_width));
        },
        .decoder_cross_key_value => {
            const model_width = @divExact(weight.scales.len, 2);
            assert(output_row_begin + rows_count <= positions_capacity);
            assert(output.len == attention.packedKeyValuesCount(positions_capacity, model_width) + positions_capacity * model_width);
        },
    }

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

        // PERFORMANCE: One-row tiles let LLVM fold the weight load into VNNI,
        // rather than spending a separate instruction and YMM register on it.
        // Fourteen output accumulators hide dot-product latency. Multi-row tiles
        // instead pin each weight in a register and reuse it across rows;
        // folding those loads would reread a weight for every activation row.
        if (rows_count == 1) {
            const activation_bytes = quantized_rows[depth_begin..][0..depth_values_per_group].*;
            const activations: I32x8 = @splat(@as(i32, @bitCast(activation_bytes)));
            inline for (0..output_blocks_count) |block_offset| {
                const packed_block_index = output_block_begin + block_offset;
                const packed_offset = weight_layout.groupOffset(packed_block_index, depth_begin);
                const weights = weight.values[packed_offset..][0..bytes_per_weight_group];
                accumulators[0][block_offset] = dotUnsignedSignedBytesFromMemory(accumulators[0][block_offset], activations, weights);
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

        if (output_layout == .encoder_query_key_value or (output_layout == .decoder_cross_key_value and rows_count == simd_lanes_count)) {
            // Encoder lanes own complete key blocks. Zero rows complete the
            // final partial block for QK's full-vector reads; Q and V store
            // only real positions, and softmax excludes the padding scores.
            var projected_rows: [simd_lanes_count]F32x8 = @splat(@as(F32x8, @splat(0)));
            inline for (0..rows_count) |tile_row| {
                const tile_row_reciprocal_input_scale: F32x8 = @splat(1.0 / input_scales[tile_row]);
                projected_rows[tile_row] = finishProjection(accumulators[tile_row][block_offset], tile_row_reciprocal_input_scale, output_scales, output_bias, activation);
            }

            if (directOutputColumnIsPackedKey(output_layout, output_begin, weight.scales.len)) {
                storeDirectOutputPackedKeyTile(output_layout, projected_rows, output_row_begin, output_begin, weight.scales.len, positions_capacity, output);
            } else {
                inline for (0..rows_count) |tile_row| {
                    storeDirectOutputBlock(output_layout, projected_rows[tile_row], output_row_begin + tile_row, output_begin, weight.scales.len, positions_capacity, output);
                }
            }
        } else {
            inline for (0..rows_count) |tile_row| {
                const tile_row_reciprocal_input_scale: F32x8 = if (rows_count == 1) reciprocal_input_scale else @splat(1.0 / input_scales[tile_row]);
                const values = finishProjection(accumulators[tile_row][block_offset], tile_row_reciprocal_input_scale, output_scales, output_bias, activation);

                if (output_layout == .row_major) {
                    const output_values = output[tile_row * weight.scales.len + output_begin ..][0..output_rows_per_block];
                    // Keep the completed projection's rounding before the
                    // residual addition, matching the former separate pass.
                    output_values.* = if (add_to_output) @as(F32x8, output_values.*) + values else values;
                } else {
                    storeDirectOutputBlock(output_layout, values, output_row_begin + tile_row, output_begin, weight.scales.len, positions_capacity, output);
                }
            }
        }
    }
}

inline fn finishProjection(accumulator: I32x8, reciprocal_input_scale: F32x8, output_scales: F32x8, output_bias: F32x8, activation: Activation) F32x8 {
    var values: F32x8 = @floatFromInt(accumulator);
    values *= reciprocal_input_scale;
    values /= output_scales;
    values += output_bias;
    if (activation == .gelu) {
        values = gelu(values);
    }

    return values;
}

inline fn directOutputColumnIsPackedKey(comptime output_layout: ProjectionOutputLayout, output_column_begin: usize, output_rows_count: usize) bool {
    return switch (output_layout) {
        .row_major => false,
        .encoder_query_key_value => output_column_begin >= output_rows_count / 3 and output_column_begin < 2 * (output_rows_count / 3),
        .decoder_cross_key_value => output_column_begin < output_rows_count / 2,
    };
}

inline fn storeDirectOutputPackedKeyTile(comptime output_layout: ProjectionOutputLayout, rows: [simd_lanes_count]F32x8, position_begin: usize, output_column_begin: usize, output_rows_count: usize, positions_capacity: usize, output: anytype) void {
    const model_width = switch (output_layout) {
        .row_major => unreachable,
        .encoder_query_key_value => output_rows_count / 3,
        .decoder_cross_key_value => output_rows_count / 2,
    };
    const key_column_begin = if (output_layout == .encoder_query_key_value) model_width else 0;
    const packed_keys_offset = if (output_layout == .encoder_query_key_value) positions_capacity * model_width else 0;
    const packed_positions_capacity = std.mem.alignForward(usize, positions_capacity, simd_lanes_count);
    const key_column = output_column_begin - key_column_begin;
    const head_index = key_column / attention.head_width;
    const head_depth_begin = key_column % attention.head_width;
    const packed_head_offset = packed_keys_offset + head_index * packed_positions_capacity * attention.head_width;
    const packed_head = output[packed_head_offset..][0 .. packed_positions_capacity * attention.head_width];

    assert(directOutputColumnIsPackedKey(output_layout, output_column_begin, output_rows_count));
    assert(position_begin < positions_capacity);

    if (output_layout == .encoder_query_key_value) {
        assert(position_begin % simd_lanes_count == 0);
        assert(position_begin + simd_lanes_count <= packed_positions_capacity);
        const key_block = packed_head[position_begin * attention.head_width ..][0 .. simd_lanes_count * attention.head_width];
        attention.storeTransposedTile(rows, key_block, simd_lanes_count, head_depth_begin, 0);
    } else {
        assert(position_begin + simd_lanes_count <= positions_capacity);
        attention.storeTransposedTileFloat16(rows, packed_head, packed_positions_capacity, head_depth_begin, position_begin);
    }
}

inline fn storeDirectOutputBlock(comptime output_layout: ProjectionOutputLayout, values: F32x8, position: usize, output_column_begin: usize, output_rows_count: usize, positions_capacity: usize, output: anytype) void {
    const model_width = switch (output_layout) {
        .row_major => unreachable,
        .encoder_query_key_value => output_rows_count / 3,
        .decoder_cross_key_value => output_rows_count / 2,
    };
    const packed_positions_capacity = std.mem.alignForward(usize, positions_capacity, simd_lanes_count);
    const packed_keys_values_count = attention.packedKeyValuesCount(positions_capacity, model_width);

    assert(position < positions_capacity);
    assert(output_column_begin % output_rows_per_block == 0);
    assert(output_column_begin + output_rows_per_block <= output_rows_count);

    if (output_layout == .encoder_query_key_value) {
        assert(output.len == attention.encoderQueryKeyValueValuesCount(positions_capacity, model_width));
        assert(!directOutputColumnIsPackedKey(output_layout, output_column_begin, output_rows_count));
        if (output_column_begin < model_width) {
            const head_index = output_column_begin / attention.head_width;
            const head_depth_begin = output_column_begin % attention.head_width;
            const destination_offset = head_index * positions_capacity * attention.head_width + position * attention.head_width + head_depth_begin;
            output[destination_offset..][0..output_rows_per_block].* = values;
        } else {
            const value_column_begin = output_column_begin - 2 * model_width;
            const head_index = value_column_begin / attention.head_width;
            const head_depth_begin = value_column_begin % attention.head_width;
            const values_offset = positions_capacity * model_width + packed_keys_values_count;
            const head_offset = values_offset + head_index * positions_capacity * attention.head_width;
            const destination_offset = head_offset + position * attention.head_width + head_depth_begin;
            output[destination_offset..][0..output_rows_per_block].* = values;
        }
    } else {
        assert(output.len == packed_keys_values_count + positions_capacity * model_width);
        if (output_column_begin < model_width) {
            const head_index = output_column_begin / attention.head_width;
            const head_depth_begin = output_column_begin % attention.head_width;
            const head_offset = head_index * packed_positions_capacity * attention.head_width;
            inline for (0..output_rows_per_block) |depth_offset| {
                output[head_offset + (head_depth_begin + depth_offset) * packed_positions_capacity + position] = @floatCast(values[depth_offset]);
            }
        } else {
            const value_column_begin = output_column_begin - model_width;
            const head_index = value_column_begin / attention.head_width;
            const head_depth_begin = value_column_begin % attention.head_width;
            const head_offset = head_index * positions_capacity * attention.head_width;
            const destination_offset = packed_keys_values_count + head_offset + position * attention.head_width + head_depth_begin;
            output[destination_offset..][0..output_rows_per_block].* = @as(F16x8, @floatCast(values));
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

pub fn quantizeNormalized(input: []const f32, gamma: []const f32, beta: []const f32, quantized: []u8) f32 {
    assert(input.len == gamma.len);
    assert(input.len == beta.len);
    assert(input.len == quantized.len);

    const statistics = normalization.calculateRowStatistics(input);
    const maximum = calculateNormalizedAbsoluteMaximum(input, gamma, beta, statistics);
    const scale = if (maximum == 0) 1.0 else 127.0 / maximum;
    const scales: F32x8 = @splat(scale);
    const shifts: F32x8 = @splat(@as(f32, @floatFromInt(vnni_weight.activation_zero_point)));
    var column: usize = 0;
    while (column < input.len) : (column += simd_lanes_count) {
        const normalized = normalization.normalizedVector(input, gamma, beta, statistics, column);
        const shifted = @mulAdd(F32x8, normalized, scales, shifts);
        const integers: I32x8 = @intFromFloat(vnni_weight.roundFloat32VectorToNearestEven(shifted));
        quantized[column..][0..simd_lanes_count].* = @as(U8x8, @intCast(integers));
    }

    return scale;
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

    return quantizeRowWithMaximum(values, @reduce(.Max, maximums), quantized);
}

inline fn quantizeNormalizedWithScratch(input: []const f32, gamma: []const f32, beta: []const f32, normalized: []f32, quantized: []u8) f32 {
    assert(input.len == gamma.len);
    assert(input.len == beta.len);
    assert(input.len == normalized.len);
    assert(input.len == quantized.len);

    const statistics = normalization.calculateRowStatistics(input);
    var maximums: F32x8 = @splat(0);
    var column: usize = 0;
    while (column < input.len) : (column += simd_lanes_count) {
        const values = normalization.normalizedVector(input, gamma, beta, statistics, column);
        normalized[column..][0..simd_lanes_count].* = values;
        maximums = @max(maximums, @abs(values));
    }

    return quantizeRowWithMaximum(normalized, @reduce(.Max, maximums), quantized);
}

inline fn quantizeRowWithMaximum(values: []const f32, maximum: f32, quantized: []u8) f32 {
    const scale = if (maximum == 0) 1.0 else 127.0 / maximum;
    const scales: F32x8 = @splat(scale);
    const shifts: F32x8 = @splat(@as(f32, @floatFromInt(vnni_weight.activation_zero_point)));

    var value_index: usize = 0;
    while (value_index < values.len) : (value_index += simd_lanes_count) {
        const shifted = @mulAdd(F32x8, values[value_index..][0..simd_lanes_count].*, scales, shifts);
        const integers: I32x8 = @intFromFloat(vnni_weight.roundFloat32VectorToNearestEven(shifted));
        quantized[value_index..][0..simd_lanes_count].* = @as(U8x8, @intCast(integers));
    }

    return scale;
}

inline fn dotUnsignedSignedBytesFromMemory(accumulator: I32x8, activations: I32x8, weights: *const [bytes_per_weight_group]i8) I32x8 {
    if (builtin.zig_backend != .stage2_llvm) {
        return dotUnsignedSignedBytes(accumulator, activations, weights.*);
    }

    return @"llvm.x86.avx512.vpdpbusd.256"(accumulator, activations, @bitCast(weights.*));
}

// LLVM also lowers this intrinsic to AVX-VNNI's VEX encoding on AVX2 targets;
// its historical name does not require AVX-512. Unlike opaque register-only
// assembly, it exposes the single-use weight load to instruction selection.
extern fn @"llvm.x86.avx512.vpdpbusd.256"(I32x8, I32x8, I32x8) I32x8;

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
    assert(output.len == rows_count * weight.scales.len);
    assert(bias.len == 0 or bias.len == weight.scales.len);
}

test "token projections match signed scalar dot products across tile tails" {
    const allocator = std.testing.allocator;
    for ([_]usize{ 8, 240, 512, 768, 3072 }) |depth| {
        for (1..30) |blocks_count| {
            const outputs_count = blocks_count * output_rows_per_block;
            const layout = vnni_weight.Layout.init(outputs_count, depth);
            const packed_values = try allocator.alloc(i8, layout.valuesCount());
            defer allocator.free(packed_values);
            const scales = try allocator.alloc(f32, outputs_count);
            defer allocator.free(scales);
            const compensation = try allocator.alloc(i32, outputs_count);
            defer allocator.free(compensation);
            const bias = try allocator.alloc(f32, outputs_count);
            defer allocator.free(bias);
            const input = try allocator.alloc(u8, depth);
            defer allocator.free(input);
            const actual = try allocator.alloc(f32, outputs_count);
            defer allocator.free(actual);
            const sums = try allocator.alloc(i32, outputs_count);
            defer allocator.free(sums);
            var random = std.Random.DefaultPrng.init(42);
            random.random().bytes(input);
            random.random().bytes(std.mem.sliceAsBytes(packed_values));
            for (0..outputs_count) |row| {
                scales[row] = 64 + @as(f32, @floatFromInt(row));
                bias[row] = @as(f32, @floatFromInt(row)) * 0.01;
                var weight_sum: i32 = 0;
                var sum: i32 = 0;
                for (input, 0..) |value, column| {
                    const weight_value: i32 = packed_values[layout.valueOffset(row, column)];
                    weight_sum += weight_value;
                    sum += (@as(i32, value) - vnni_weight.activation_zero_point) * weight_value;
                }
                compensation[row] = -vnni_weight.activation_zero_point * weight_sum;
                sums[row] = sum;
            }
            const weight: QuantizedWeight = .{ .values = packed_values, .scales = scales, .compensation = compensation, .input_values_count = depth };
            const input_scale: f32 = 17.3;
            for ([_]Activation{ .none, .gelu }) |activation| {
                for ([_]usize{ 1, 4, 32 }) |lanes_count| {
                    @memset(actual, std.math.nan(f32));
                    for (0..lanes_count) |lane_index| {
                        forwardQuantizedOneParallel(input, input_scale, weight, bias, activation, actual, .{ .index = lane_index, .count = lanes_count, .barrier = undefined });
                    }
                    for (actual, 0..) |value, row| {
                        const expected = finishProjection(@splat(sums[row]), @splat(1.0 / input_scale), @splat(scales[row]), @splat(bias[row]), activation);
                        try std.testing.expectEqual(expected[0], value);
                    }
                }
            }
        }
    }
}

test "normalized FFN residual matches materialized normalization and separate projections with minimum scratch" {
    const allocator = std.testing.allocator;
    const model_width = 32;
    const ffn_width = 128;
    const rows_count_max = 17;
    var packed_values: [2][model_width * ffn_width]i8 = undefined;
    var scales: [ffn_width + model_width]f32 = undefined;
    var compensation: [ffn_width + model_width]i32 = undefined;
    var biases: [ffn_width + model_width]f32 = undefined;
    var gamma: [model_width]f32 = undefined;
    var beta: [model_width]f32 = undefined;
    var input: [rows_count_max * model_width]f32 = undefined;
    var random = std.Random.DefaultPrng.init(42);
    random.random().bytes(std.mem.asBytes(&packed_values));
    for (&scales, &biases, 0..) |*scale, *bias, row| {
        scale.* = 64 + @as(f32, @floatFromInt(row));
        bias.* = random.random().float(f32) - 0.5;
    }
    for (&input) |*value| value.* = random.random().float(f32) * 4 - 2;
    for (&gamma, &beta) |*gain, *offset| {
        gain.* = random.random().float(f32) * 2 - 1;
        offset.* = random.random().float(f32) - 0.5;
    }
    @memset(input[0..model_width], 0);

    const weights = [2]QuantizedWeight{
        .{ .values = &packed_values[0], .scales = scales[0..ffn_width], .compensation = compensation[0..ffn_width], .input_values_count = model_width },
        .{ .values = &packed_values[1], .scales = scales[ffn_width..], .compensation = compensation[ffn_width..], .input_values_count = ffn_width },
    };
    for (weights, 0..) |weight, index| {
        const layout = weight.layout();
        for (0..weight.scales.len) |row| {
            var sum: i32 = 0;
            for (0..weight.input_values_count) |column| sum += weight.values[layout.valueOffset(row, column)];
            compensation[(if (index == 0) @as(usize, 0) else ffn_width) + row] = -vnni_weight.activation_zero_point * sum;
        }
    }

    const Check = struct {
        weights: [2]QuantizedWeight,
        biases: []const f32,
        gamma: []const f32,
        beta: []const f32,
        values: []f32,
        float_scratch: []f32,
        quantized_scratch: []u8,

        fn run(raw_context: *anyopaque, lane: Lane) void {
            const context: *@This() = @ptrCast(@alignCast(raw_context));
            const float_values_count = context.float_scratch.len / lane.count;
            const quantized_values_count = context.quantized_scratch.len / lane.count;
            const expansion_outputs_count = context.weights[0].scales.len;
            forwardFeedForwardResidualRows(context.values, context.gamma, context.beta, context.values.len / context.weights[0].input_values_count, context.weights[0], context.biases[0..expansion_outputs_count], context.weights[1], context.biases[expansion_outputs_count..], context.float_scratch[lane.index * float_values_count ..][0..float_values_count], context.quantized_scratch[lane.index * quantized_values_count ..][0..quantized_values_count], lane);
        }
    };

    var normalized: [rows_count_max * model_width]f32 = undefined;
    var expanded: [rows_count_max * ffn_width]f32 = undefined;
    var expected: [rows_count_max * model_width]f32 = undefined;
    var actual: [rows_count_max * model_width]f32 = undefined;
    var reference_scratch: [rows_per_tile * ffn_width]u8 = undefined;
    const standalone_lane: Lane = .{ .index = 0, .count = 1, .barrier = undefined };
    for ([_]usize{ 1, 4, 8 }) |workers_count| {
        var executor: @import("executor.zig").Executor = undefined;
        try executor.init(std.testing.io, workers_count);
        defer executor.deinit();
        const float_scratch = try allocator.alloc(f32, workers_count * ffn_rows_per_tile * ffn_width);
        defer allocator.free(float_scratch);
        const quantized_scratch = try allocator.alloc(u8, workers_count * ffn_rows_per_tile * @max(model_width, ffn_width));
        defer allocator.free(quantized_scratch);

        for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, rows_count_max }) |rows_count| {
            const input_rows = input[0 .. rows_count * model_width];
            const expanded_rows = expanded[0 .. rows_count * ffn_width];
            const expected_rows = expected[0 .. rows_count * model_width];
            const actual_rows = actual[0 .. rows_count * model_width];
            // The reference retains the full expansion and uses independent
            // fourteen-row projection tiles instead of the joined FFN tiles.
            const normalized_rows = normalized[0 .. rows_count * model_width];
            normalization.forwardRows(input_rows, &gamma, &beta, rows_count, model_width, normalized_rows, standalone_lane);
            forwardRows(normalized_rows, rows_count, weights[0], biases[0..ffn_width], .gelu, expanded_rows, &reference_scratch, standalone_lane);
            forwardRows(expanded_rows, rows_count, weights[1], biases[ffn_width..], .none, expected_rows, &reference_scratch, standalone_lane);
            for (input_rows, expected_rows) |value, *projected| projected.* = value + projected.*;

            @memset(&actual, std.math.nan(f32));
            @memcpy(actual_rows, input_rows);
            @memset(float_scratch, std.math.nan(f32));
            @memset(quantized_scratch, 0xaa);
            var check: Check = .{ .weights = weights, .biases = &biases, .gamma = &gamma, .beta = &beta, .values = actual_rows, .float_scratch = float_scratch, .quantized_scratch = quantized_scratch };
            executor.run(&check, Check.run);
            try std.testing.expectEqualSlices(f32, expected_rows, actual_rows);
            for (actual[actual_rows.len..]) |value| try std.testing.expect(std.math.isNan(value));
        }
    }
}

test "square row projections and residuals preserve results across tile boundaries" {
    const allocator = std.testing.allocator;
    const width = 32;
    const rows_count_max = 71;
    const layout = vnni_weight.Layout.init(width, width);
    var packed_values: [width * width]i8 = undefined;
    var scales: [width]f32 = undefined;
    var compensation: [width]i32 = undefined;
    var biases: [width]f32 = undefined;
    var input: [rows_count_max * width]f32 = undefined;
    var residual: [rows_count_max * width]f32 = undefined;
    var random = std.Random.DefaultPrng.init(42);
    random.random().bytes(std.mem.asBytes(&packed_values));
    for (&scales, &compensation, &biases, 0..) |*scale, *row_compensation, *bias, row| {
        scale.* = 64 + @as(f32, @floatFromInt(row));
        bias.* = random.random().float(f32) - 0.5;
        var sum: i32 = 0;
        for (0..width) |column| sum += packed_values[layout.valueOffset(row, column)];
        row_compensation.* = -vnni_weight.activation_zero_point * sum;
    }
    for (&input) |*value| value.* = random.random().float(f32) * 4 - 2;
    for (&residual) |*value| value.* = random.random().float(f32) * 4 - 2;
    @memset(input[0..width], 0);
    const weight: QuantizedWeight = .{ .values = &packed_values, .scales = &scales, .compensation = &compensation, .input_values_count = width };

    const Check = struct {
        weight: QuantizedWeight,
        biases: []const f32,
        activation: Activation,
        add_to_output: bool,
        input: []const f32,
        values: []f32,
        quantized_scratch: []u8,

        fn run(raw_context: *anyopaque, lane: Lane) void {
            const context: *@This() = @ptrCast(@alignCast(raw_context));
            const scratch_values_count = context.quantized_scratch.len / lane.count;
            const rows_count = context.input.len / context.weight.input_values_count;
            const scratch = context.quantized_scratch[lane.index * scratch_values_count ..][0..scratch_values_count];
            if (context.add_to_output) {
                forwardResidualRows(context.input, rows_count, context.weight, context.biases, context.values, scratch, lane);
            } else {
                forwardRows(context.input, rows_count, context.weight, context.biases, context.activation, context.values, scratch, lane);
            }
        }
    };

    var expected: [rows_count_max * width]f32 = undefined;
    var actual: [rows_count_max * width]f32 = undefined;
    var reference_scratch: [width]u8 = undefined;
    const standalone_lane: Lane = .{ .index = 0, .count = 1, .barrier = undefined };
    const Mode = enum { none, gelu, residual_in_place, residual_disjoint };
    for ([_]Mode{ .none, .gelu, .residual_in_place, .residual_disjoint }) |mode| {
        const activation: Activation = if (mode == .gelu) .gelu else .none;
        const add_to_output = mode == .residual_in_place or mode == .residual_disjoint;
        const initial_values = if (mode == .residual_disjoint) &residual else &input;
        // Token projections provide an independent output and use output-block
        // tiling instead of the fourteen input rows used by forwardRows.
        for (0..rows_count_max) |row| {
            forwardOne(input[row * width ..][0..width], weight, &biases, activation, expected[row * width ..][0..width], &reference_scratch, standalone_lane);
        }
        if (add_to_output) {
            for (initial_values, &expected) |value, *projected| projected.* = value + projected.*;
        }
        for ([_]usize{ 1, 4, 8 }) |workers_count| {
            var executor: @import("executor.zig").Executor = undefined;
            try executor.init(std.testing.io, workers_count);
            defer executor.deinit();
            const quantized_scratch = try allocator.alloc(u8, workers_count * rows_per_tile * width);
            defer allocator.free(quantized_scratch);

            for (1..rows_count_max + 1) |rows_count| {
                const values_count = rows_count * width;
                const actual_rows = actual[0..values_count];
                @memset(&actual, std.math.nan(f32));
                @memcpy(actual_rows, initial_values[0..values_count]);
                @memset(quantized_scratch, 0xa5);
                const input_rows = if (mode == .residual_disjoint) input[0..values_count] else actual_rows;
                var check: Check = .{ .weight = weight, .biases = &biases, .activation = activation, .add_to_output = add_to_output, .input = input_rows, .values = actual_rows, .quantized_scratch = quantized_scratch };
                executor.run(&check, Check.run);
                try std.testing.expectEqualSlices(f32, expected[0..values_count], actual_rows);
                for (actual[values_count..]) |value| try std.testing.expect(std.math.isNan(value));
            }
        }
    }
}
