//! LayerNorm statistics and vector transformation shared by materialized
//! normalization and the fused normalized linear projection.

const std = @import("std");
const Lane = @import("executor.zig").Lane;
const assert = std.debug.assert;

const simd_lanes_count: usize = 8;

pub const F32x8 = @Vector(simd_lanes_count, f32);

pub const RowStatistics = struct {
    mean: f32,
    reciprocal_standard_deviation: f32,
};

pub fn forwardRows(input: []const f32, gamma: []const f32, beta: []const f32, rows_count: usize, width: usize, output: []f32, lane: Lane) void {
    assert(input.len == rows_count * width);
    assert(output.len == input.len);
    assert(gamma.len == width);
    assert(beta.len == width);
    assert(width % (4 * simd_lanes_count) == 0);

    const rows_range = lane.range(rows_count);
    for (rows_range.start_index..rows_range.end_index) |row_index| {
        const input_row = input[row_index * width ..][0..width];
        const output_row = output[row_index * width ..][0..width];
        const statistics = calculateRowStatistics(input_row);

        var column: usize = 0;
        while (column < width) : (column += simd_lanes_count) {
            output_row[column..][0..simd_lanes_count].* = normalizedVector(input_row, gamma, beta, statistics, column);
        }
    }
}

pub inline fn calculateRowStatistics(input: []const f32) RowStatistics {
    assert(input.len % (4 * simd_lanes_count) == 0);

    var sums_0: F32x8 = @splat(0);
    var sums_1: F32x8 = @splat(0);
    var sums_2: F32x8 = @splat(0);
    var sums_3: F32x8 = @splat(0);
    var square_sums_0: F32x8 = @splat(0);
    var square_sums_1: F32x8 = @splat(0);
    var square_sums_2: F32x8 = @splat(0);
    var square_sums_3: F32x8 = @splat(0);

    var column: usize = 0;

    while (column < input.len) : (column += 4 * simd_lanes_count) {
        const values_0: F32x8 = input[column + 0 * simd_lanes_count ..][0..simd_lanes_count].*;
        const values_1: F32x8 = input[column + 1 * simd_lanes_count ..][0..simd_lanes_count].*;
        const values_2: F32x8 = input[column + 2 * simd_lanes_count ..][0..simd_lanes_count].*;
        const values_3: F32x8 = input[column + 3 * simd_lanes_count ..][0..simd_lanes_count].*;
        sums_0 += values_0;
        sums_1 += values_1;
        sums_2 += values_2;
        sums_3 += values_3;
        square_sums_0 = @mulAdd(F32x8, values_0, values_0, square_sums_0);
        square_sums_1 = @mulAdd(F32x8, values_1, values_1, square_sums_1);
        square_sums_2 = @mulAdd(F32x8, values_2, values_2, square_sums_2);
        square_sums_3 = @mulAdd(F32x8, values_3, values_3, square_sums_3);
    }

    const reciprocal_width = 1.0 / @as(f32, @floatFromInt(input.len));
    const mean = reduceAddAvx2(((sums_0 + sums_1) + sums_2) + sums_3) * reciprocal_width;
    const mean_square = reduceAddAvx2((square_sums_0 + square_sums_1) + (square_sums_2 + square_sums_3)) * reciprocal_width;
    const variance = @max(@mulAdd(f32, -mean, mean, mean_square), 0.0);

    return .{ .mean = mean, .reciprocal_standard_deviation = 1.0 / @sqrt(variance + 1.0e-5) };
}

pub inline fn normalizedVector(input: []const f32, gamma: []const f32, beta: []const f32, statistics: RowStatistics, column: usize) F32x8 {
    const values: F32x8 = input[column..][0..simd_lanes_count].*;
    const gammas: F32x8 = gamma[column..][0..simd_lanes_count].*;
    const betas: F32x8 = beta[column..][0..simd_lanes_count].*;
    const means: F32x8 = @splat(statistics.mean);
    const reciprocal_standard_deviations: F32x8 = @splat(statistics.reciprocal_standard_deviation);

    return @mulAdd(F32x8, (values - means) * gammas, reciprocal_standard_deviations, betas);
}

inline fn reduceAddAvx2(values: F32x8) f32 {
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
