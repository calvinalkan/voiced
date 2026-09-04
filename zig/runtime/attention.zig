//! Specialized Whisper attention for blocked encoder heads and token-serial
//! decoder heads. Encoder attention uses online softmax without materializing
//! an N-by-N tensor; decoder attention consumes persistent head-major K/V.

const std = @import("std");
const Lane = @import("executor.zig").Lane;
const assert = std.debug.assert;

pub const head_width: usize = 64;

const query_positions_per_tile: usize = 8;
const key_positions_per_tile: usize = 64;
const simd_lanes_count: usize = 8;
const score_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));

const F32x8 = @Vector(simd_lanes_count, f32);

// ─── Encoder Attention ─────────────────────────────────────────────────────

pub fn encoderScratchValuesCount(positions_count: usize, heads_count: usize, lanes_count: usize) usize {
    assert(positions_count > 0);
    assert(heads_count > 0);
    assert(lanes_count > 0);

    return @min(heads_count, lanes_count) * laneScratchValuesCount(positions_count);
}

/// `forwardEncoder` calculates unmasked encoder self-attention from one fused
/// row-major `[positions_count, 3 * model_width]` QKV projection. Every lane in
/// one executor operation must call it with the same arguments and disjoint
/// lane identity. The operation allocates nothing and does not synchronize.
pub fn forwardEncoder(query_key_value: []const f32, positions_count: usize, model_width: usize, heads_count: usize, output: []f32, scratch: []f32, lane: Lane) void {
    assert(positions_count > 0);
    assert(heads_count > 0);
    assert(model_width == heads_count * head_width);
    assert(query_key_value.len == positions_count * 3 * model_width);
    assert(output.len == positions_count * model_width);
    assert(scratch.len == encoderScratchValuesCount(positions_count, heads_count, lane.count));

    const heads_range = lane.range(heads_count);
    if (heads_range.start_index == heads_range.end_index) {
        // More executor lanes than attention heads leaves this lane idle.
        return;
    }

    const lane_scratch_values_count = laneScratchValuesCount(positions_count);
    const lane_scratch = scratch[lane.index * lane_scratch_values_count ..][0..lane_scratch_values_count];
    const packed_positions_count = packedPositionsCount(positions_count);
    const packed_keys_values_count = head_width * packed_positions_count;
    const values_count = positions_count * head_width;
    const queries_count = query_positions_per_tile * head_width;
    const scores_count = query_positions_per_tile * key_positions_per_tile;
    const packed_keys = lane_scratch[0..packed_keys_values_count];
    const values = lane_scratch[packed_keys_values_count..][0..values_count];
    const queries = lane_scratch[packed_keys_values_count + values_count ..][0..queries_count];
    const scores = lane_scratch[packed_keys_values_count + values_count + queries_count ..][0..scores_count];

    for (heads_range.start_index..heads_range.end_index) |head_index| {
        copyValueHead(query_key_value, positions_count, model_width, head_index, values);
        packKeyHead(query_key_value, positions_count, model_width, head_index, packed_positions_count, packed_keys);

        var query_position_begin: usize = 0;
        while (query_position_begin < positions_count) : (query_position_begin += query_positions_per_tile) {
            const query_positions_count = @min(query_positions_per_tile, positions_count - query_position_begin);
            copyQueryTile(query_key_value, positions_count, model_width, head_index, query_position_begin, query_positions_count, queries);
            calculateAttentionTile(queries, packed_keys, values, output, scores, positions_count, packed_positions_count, model_width, head_index, query_position_begin, query_positions_count);
        }
    }
}

// ─── Decoder Attention ─────────────────────────────────────────────────────

pub fn decoderScratchValuesCount(positions_count: usize, heads_count: usize, lanes_count: usize) usize {
    assert(positions_count > 0);
    assert(heads_count > 0);
    assert(lanes_count > 0);

    return @min(heads_count, lanes_count) * positions_count;
}

/// `forwardDecoder` calculates one token's attention over head-major K/V
/// storage. `head_positions_stride` is the physical capacity between heads;
/// `positions_count` selects the initialized prefix for this invocation.
pub fn forwardDecoder(query: []const f32, keys: []const f32, values: []const f32, positions_count: usize, head_positions_stride: usize, model_width: usize, heads_count: usize, output: []f32, scores_scratch: []f32, lane: Lane) void {
    assert(positions_count > 0);
    assert(positions_count <= head_positions_stride);
    assert(model_width == heads_count * head_width);
    assert(query.len == model_width);
    assert(keys.len == heads_count * head_positions_stride * head_width);
    assert(values.len == keys.len);
    assert(output.len == model_width);
    assert(scores_scratch.len >= decoderScratchValuesCount(positions_count, heads_count, lane.count));

    const heads_range = lane.range(heads_count);
    if (heads_range.start_index == heads_range.end_index) {
        // More executor lanes than attention heads leaves this lane idle.
        return;
    }

    const scores = scores_scratch[lane.index * positions_count ..][0..positions_count];
    for (heads_range.start_index..heads_range.end_index) |head_index| {
        const head_offset = head_index * head_positions_stride * head_width;
        const query_head = query[head_index * head_width ..][0..head_width];
        const key_head = keys[head_offset..][0 .. positions_count * head_width];
        const value_head = values[head_offset..][0 .. positions_count * head_width];
        calculateDecoderScoreRow(query_head, key_head, scores);

        const maximum = maximumScore(scores);
        const probability_sum = exponentiateScores(scores, maximum);
        assert(probability_sum > 0);
        scaleScores(scores, 1.0 / probability_sum);

        const output_rows = [1][]f32{output[head_index * head_width ..][0..head_width]};
        const probability_rows = [1][]const f32{scores};
        accumulateValueRows(1, 0, output_rows, probability_rows, .{0}, value_head, positions_count, false);
    }
}

// ─── Head Layout Preparation ───────────────────────────────────────────────

fn copyValueHead(query_key_value: []const f32, positions_count: usize, model_width: usize, head_index: usize, values: []f32) void {
    assert(head_index < model_width / head_width);
    assert(values.len == positions_count * head_width);

    const query_key_value_width = 3 * model_width;
    const value_column = 2 * model_width + head_index * head_width;
    for (0..positions_count) |position_index| {
        var depth: usize = 0;
        while (depth < head_width) : (depth += simd_lanes_count) {
            const source_offset = position_index * query_key_value_width + value_column + depth;
            const destination_offset = position_index * head_width + depth;
            values[destination_offset..][0..simd_lanes_count].* = query_key_value[source_offset..][0..simd_lanes_count].*;
        }
    }
}

fn packKeyHead(query_key_value: []const f32, positions_count: usize, model_width: usize, head_index: usize, packed_positions_count: usize, packed_keys: []f32) void {
    assert(head_index < model_width / head_width);
    assert(packed_positions_count == packedPositionsCount(positions_count));
    assert(packed_keys.len == head_width * packed_positions_count);

    const query_key_value_width = 3 * model_width;
    const key_column = model_width + head_index * head_width;
    var position_begin: usize = 0;
    while (position_begin + simd_lanes_count <= positions_count) : (position_begin += simd_lanes_count) {
        var depth_begin: usize = 0;
        while (depth_begin < head_width) : (depth_begin += simd_lanes_count) {
            var rows: [simd_lanes_count]F32x8 = undefined;
            for (0..simd_lanes_count) |row_index| {
                const source_offset = (position_begin + row_index) * query_key_value_width + key_column + depth_begin;
                rows[row_index] = query_key_value[source_offset..][0..simd_lanes_count].*;
            }
            storeTransposedTile(rows, packed_keys, packed_positions_count, depth_begin, position_begin);
        }
    }

    for (position_begin..positions_count) |position_index| {
        for (0..head_width) |depth| {
            const source_offset = position_index * query_key_value_width + key_column + depth;
            packed_keys[depth * packed_positions_count + position_index] = query_key_value[source_offset];
        }
    }
    for (0..head_width) |depth| {
        @memset(packed_keys[depth * packed_positions_count + positions_count ..][0 .. packed_positions_count - positions_count], 0);
    }
}

fn copyQueryTile(query_key_value: []const f32, positions_count: usize, model_width: usize, head_index: usize, query_position_begin: usize, query_positions_count: usize, queries: []f32) void {
    assert(head_index < model_width / head_width);
    assert(query_position_begin < positions_count);
    assert(query_positions_count > 0);
    assert(query_positions_count <= query_positions_per_tile);
    assert(query_position_begin + query_positions_count <= positions_count);
    assert(queries.len == query_positions_per_tile * head_width);

    const query_key_value_width = 3 * model_width;
    const query_column = head_index * head_width;
    for (0..query_positions_count) |query_offset| {
        const position_index = query_position_begin + query_offset;
        var depth: usize = 0;
        while (depth < head_width) : (depth += simd_lanes_count) {
            const source_offset = position_index * query_key_value_width + query_column + depth;
            const destination_offset = query_offset * head_width + depth;
            queries[destination_offset..][0..simd_lanes_count].* = query_key_value[source_offset..][0..simd_lanes_count].*;
        }
    }
}

// ─── Blocked Online Attention ──────────────────────────────────────────────

noinline fn calculateAttentionTile(queries: []const f32, packed_keys: []const f32, values: []const f32, output: []f32, scores: []f32, positions_count: usize, packed_positions_count: usize, model_width: usize, head_index: usize, query_position_begin: usize, query_positions_count: usize) void {
    assert(queries.len == query_positions_per_tile * head_width);
    assert(packed_keys.len == head_width * packed_positions_count);
    assert(values.len == positions_count * head_width);
    assert(output.len == positions_count * model_width);
    assert(scores.len == query_positions_per_tile * key_positions_per_tile);
    assert(query_positions_count > 0);
    assert(query_positions_count <= query_positions_per_tile);

    var row_maximums: [query_positions_per_tile]f32 = @splat(-std.math.inf(f32));
    var row_sums: [query_positions_per_tile]f32 = @splat(0);

    var key_position_begin: usize = 0;
    while (key_position_begin < positions_count) : (key_position_begin += key_positions_per_tile) {
        const key_positions_count = @min(key_positions_per_tile, positions_count - key_position_begin);
        calculateScoreTile(queries, packed_keys, scores, query_positions_count, key_position_begin, key_positions_count, packed_positions_count);

        var next_maximums: [query_positions_per_tile]f32 = undefined;
        for (0..query_positions_count) |query_offset| {
            const score_row = scores[query_offset * key_positions_per_tile ..][0..key_positions_count];
            next_maximums[query_offset] = @max(row_maximums[query_offset], maximumScore(score_row));
        }

        var previous_scales: [query_positions_per_tile]f32 = @splat(0);
        if (key_position_begin != 0) {
            var maximum_differences: [query_positions_per_tile]f32 = @splat(0);
            for (0..query_positions_count) |query_offset| {
                maximum_differences[query_offset] = row_maximums[query_offset] - next_maximums[query_offset];
            }
            previous_scales = @bitCast(vectorExp(@bitCast(maximum_differences)));
        }

        for (0..query_positions_count) |query_offset| {
            const score_row = scores[query_offset * key_positions_per_tile ..][0..key_positions_count];
            const probability_sum = exponentiateScores(score_row, next_maximums[query_offset]);
            row_sums[query_offset] = @mulAdd(f32, previous_scales[query_offset], row_sums[query_offset], probability_sum);
            row_maximums[query_offset] = next_maximums[query_offset];
        }

        const value_rows = values[key_position_begin * head_width ..][0 .. key_positions_count * head_width];
        var query_offset: usize = 0;
        while (query_offset + 1 < query_positions_count) : (query_offset += 2) {
            const output_rows = [2][]f32{
                outputHead(output, model_width, head_index, query_position_begin + query_offset),
                outputHead(output, model_width, head_index, query_position_begin + query_offset + 1),
            };
            const probability_rows = [2][]const f32{
                scores[(query_offset + 0) * key_positions_per_tile ..][0..key_positions_count],
                scores[(query_offset + 1) * key_positions_per_tile ..][0..key_positions_count],
            };
            const previous_output_scales = [2]f32{ previous_scales[query_offset], previous_scales[query_offset + 1] };
            accumulateValueRows(2, 0, output_rows, probability_rows, previous_output_scales, value_rows, key_positions_count, key_position_begin != 0);
            accumulateValueRows(2, 32, output_rows, probability_rows, previous_output_scales, value_rows, key_positions_count, key_position_begin != 0);
        }
        if (query_offset < query_positions_count) {
            const output_rows = [1][]f32{outputHead(output, model_width, head_index, query_position_begin + query_offset)};
            const probability_rows = [1][]const f32{scores[query_offset * key_positions_per_tile ..][0..key_positions_count]};
            accumulateValueRows(1, 0, output_rows, probability_rows, .{previous_scales[query_offset]}, value_rows, key_positions_count, key_position_begin != 0);
        }
    }

    for (0..query_positions_count) |query_offset| {
        assert(row_sums[query_offset] > 0);
        scaleVector(outputHead(output, model_width, head_index, query_position_begin + query_offset), 1.0 / row_sums[query_offset]);
    }
}

inline fn outputHead(output: []f32, model_width: usize, head_index: usize, position_index: usize) []f32 {
    const offset = position_index * model_width + head_index * head_width;
    return output[offset..][0..head_width];
}

// ─── Query-Key Scores ──────────────────────────────────────────────────────

inline fn calculateScoreTile(queries: []const f32, packed_keys: []const f32, scores: []f32, query_positions_count: usize, key_position_begin: usize, key_positions_count: usize, packed_positions_count: usize) void {
    assert(query_positions_count > 0);
    assert(query_positions_count <= query_positions_per_tile);
    assert(key_positions_count > 0);
    assert(key_positions_count <= key_positions_per_tile);

    const rounded_key_positions_count = std.mem.alignForward(usize, key_positions_count, simd_lanes_count);
    var query_offset: usize = 0;
    while (query_offset + 4 <= query_positions_count) : (query_offset += 4) {
        calculatePackedScoreRows(4, queries, packed_keys, scores[query_offset * key_positions_per_tile ..][0 .. 4 * key_positions_per_tile], query_offset, key_position_begin, rounded_key_positions_count, packed_positions_count);
    }
    while (query_offset < query_positions_count) : (query_offset += 1) {
        calculatePackedScoreRows(1, queries, packed_keys, scores[query_offset * key_positions_per_tile ..][0..key_positions_per_tile], query_offset, key_position_begin, rounded_key_positions_count, packed_positions_count);
    }
}

inline fn calculatePackedScoreRows(comptime rows_count: usize, queries: []const f32, packed_keys: []const f32, scores: []f32, query_offset: usize, key_position_begin: usize, rounded_key_positions_count: usize, packed_positions_count: usize) void {
    assert(rows_count == 1 or rows_count == 4);
    assert(query_offset + rows_count <= query_positions_per_tile);
    assert(scores.len == rows_count * key_positions_per_tile);

    var key_offset: usize = 0;
    while (key_offset < rounded_key_positions_count) : (key_offset += simd_lanes_count) {
        var sums_a: [rows_count]F32x8 = @splat(@as(F32x8, @splat(0)));
        var sums_b: [rows_count]F32x8 = @splat(@as(F32x8, @splat(0)));

        var depth: usize = 0;
        while (depth < head_width) : (depth += 2) {
            const keys_a: F32x8 = packed_keys[(depth + 0) * packed_positions_count + key_position_begin + key_offset ..][0..simd_lanes_count].*;
            const keys_b: F32x8 = packed_keys[(depth + 1) * packed_positions_count + key_position_begin + key_offset ..][0..simd_lanes_count].*;
            inline for (0..rows_count) |row_offset| {
                const query = queries[(query_offset + row_offset) * head_width ..][0..head_width];
                sums_a[row_offset] = @mulAdd(F32x8, @as(F32x8, @splat(query[depth + 0])), keys_a, sums_a[row_offset]);
                sums_b[row_offset] = @mulAdd(F32x8, @as(F32x8, @splat(query[depth + 1])), keys_b, sums_b[row_offset]);
            }
        }

        const scales: F32x8 = @splat(score_scale);
        inline for (0..rows_count) |row_offset| {
            const score_row = scores[row_offset * key_positions_per_tile ..][0..key_positions_per_tile];
            score_row[key_offset..][0..simd_lanes_count].* = (sums_a[row_offset] + sums_b[row_offset]) * scales;
        }
    }
}

noinline fn calculateDecoderScoreRow(query: []const f32, keys: []const f32, scores: []f32) void {
    assert(query.len == head_width);
    assert(keys.len == scores.len * head_width);

    const key_positions_per_iteration = 4;
    var key_position: usize = 0;
    while (key_position + key_positions_per_iteration <= scores.len) : (key_position += key_positions_per_iteration) {
        var products: [key_positions_per_iteration]F32x8 = @splat(@as(F32x8, @splat(0)));
        var depth: usize = 0;
        while (depth < head_width) : (depth += simd_lanes_count) {
            const query_values: F32x8 = query[depth..][0..simd_lanes_count].*;
            inline for (0..key_positions_per_iteration) |key_offset| {
                const key_values: F32x8 = keys[(key_position + key_offset) * head_width + depth ..][0..simd_lanes_count].*;
                products[key_offset] = @mulAdd(F32x8, query_values, key_values, products[key_offset]);
            }
        }
        inline for (0..key_positions_per_iteration) |key_offset| {
            scores[key_position + key_offset] = reduceSumInMklGemvOrder(products[key_offset]) * score_scale;
        }
    }

    while (key_position < scores.len) : (key_position += 1) {
        var products: F32x8 = @splat(0);
        var depth: usize = 0;
        while (depth < head_width) : (depth += simd_lanes_count) {
            const query_values: F32x8 = query[depth..][0..simd_lanes_count].*;
            const key_values: F32x8 = keys[key_position * head_width + depth ..][0..simd_lanes_count].*;
            products = @mulAdd(F32x8, query_values, key_values, products);
        }
        scores[key_position] = reduceSumInMklGemvOrder(products) * score_scale;
    }
}

// ─── Online Softmax ────────────────────────────────────────────────────────

inline fn maximumScore(scores: []const f32) f32 {
    var maximums: F32x8 = @splat(-std.math.inf(f32));
    var score_index: usize = 0;
    while (score_index + simd_lanes_count <= scores.len) : (score_index += simd_lanes_count) {
        const score_vector: F32x8 = scores[score_index..][0..simd_lanes_count].*;
        maximums = @max(maximums, score_vector);
    }

    var maximum = reduceMaximumCTranslate2(maximums);
    while (score_index < scores.len) : (score_index += 1) {
        maximum = @max(maximum, scores[score_index]);
    }
    return maximum;
}

inline fn exponentiateScores(scores: []f32, maximum: f32) f32 {
    const maximums: F32x8 = @splat(maximum);
    if (scores.len <= simd_lanes_count) {
        var score_lanes: [simd_lanes_count]f32 = @splat(0);
        @memcpy(score_lanes[0..scores.len], scores);
        const probability_vector = vectorExp(@as(F32x8, @bitCast(score_lanes)) - maximums);
        const probabilities: [simd_lanes_count]f32 = @bitCast(probability_vector);

        var sum: f32 = 0;
        for (0..scores.len) |lane_index| {
            scores[lane_index] = probabilities[lane_index];
            sum += probabilities[lane_index];
        }
        return sum;
    }

    var sums: F32x8 = @splat(0);
    var score_index: usize = 0;
    while (score_index + simd_lanes_count <= scores.len) : (score_index += simd_lanes_count) {
        const probabilities = vectorExp(scores[score_index..][0..simd_lanes_count].* - maximums);
        scores[score_index..][0..simd_lanes_count].* = probabilities;
        sums += probabilities;
    }

    var sum = reduceSumCTranslate2(sums);
    if (score_index < scores.len) {
        const active_lanes_count = scores.len - score_index;
        var tail_scores: [simd_lanes_count]f32 = @splat(0);
        for (0..active_lanes_count) |lane_index| {
            tail_scores[lane_index] = scores[score_index + lane_index];
        }
        const probability_vector = vectorExp(@as(F32x8, @bitCast(tail_scores)) - maximums);
        const probabilities: [simd_lanes_count]f32 = @bitCast(probability_vector);
        for (0..active_lanes_count) |lane_index| {
            scores[score_index + lane_index] = probabilities[lane_index];
            sum += probabilities[lane_index];
        }
    }
    return sum;
}

inline fn reduceMaximumCTranslate2(values: F32x8) f32 {
    var reduced = values;
    reduced = @max(reduced, @shuffle(f32, reduced, undefined, @Vector(8, i32){ 4, 5, 6, 7, 0, 1, 2, 3 }));
    reduced = @max(reduced, @shuffle(f32, reduced, undefined, @Vector(8, i32){ 2, 3, 0, 1, 6, 7, 4, 5 }));
    reduced = @max(reduced, @shuffle(f32, reduced, undefined, @Vector(8, i32){ 1, 0, 3, 2, 5, 4, 7, 6 }));
    return reduced[0];
}

inline fn reduceSumInMklGemvOrder(values: F32x8) f32 {
    @setFloatMode(.strict);

    const adjacent_pairs = values + @shuffle(f32, values, undefined, @Vector(8, i32){ 1, 0, 3, 2, 5, 4, 7, 6 });
    const low_high_pairs = adjacent_pairs + @shuffle(f32, adjacent_pairs, undefined, @Vector(8, i32){ 4, 5, 6, 7, 0, 1, 2, 3 });
    return low_high_pairs[0] + low_high_pairs[2];
}

inline fn reduceSumCTranslate2(values: F32x8) f32 {
    @setFloatMode(.strict);

    var reduced = values;
    reduced += @shuffle(f32, reduced, undefined, @Vector(8, i32){ 4, 5, 6, 7, 0, 1, 2, 3 });
    reduced += @shuffle(f32, reduced, undefined, @Vector(8, i32){ 2, 3, 0, 1, 6, 7, 4, 5 });
    reduced += @shuffle(f32, reduced, undefined, @Vector(8, i32){ 1, 0, 3, 2, 5, 4, 7, 6 });
    return reduced[0];
}

inline fn vectorExp(input: F32x8) F32x8 {
    @setFloatMode(.strict);

    const maximum_input: F32x8 = @splat(88.3762626647949);
    const minimum_input: F32x8 = @splat(-88.3762626647949);
    var values = @max(@min(input, maximum_input), minimum_input);
    const exponent = @floor(values * @as(F32x8, @splat(1.44269504088896341)) + @as(F32x8, @splat(0.5)));
    values -= exponent * @as(F32x8, @splat(0.693359375));
    values -= exponent * @as(F32x8, @splat(-2.12194440e-4));
    const squares = values * values;

    var polynomial: F32x8 = @splat(1.9875691500e-4);
    polynomial = polynomial * values + @as(F32x8, @splat(1.3981999507e-3));
    polynomial = polynomial * values + @as(F32x8, @splat(8.3334519073e-3));
    polynomial = polynomial * values + @as(F32x8, @splat(4.1665795894e-2));
    polynomial = polynomial * values + @as(F32x8, @splat(1.6666665459e-1));
    polynomial = polynomial * values + @as(F32x8, @splat(5.0000001201e-1));
    polynomial = polynomial * squares + values;
    polynomial += @as(F32x8, @splat(1));

    const integer_exponents: @Vector(simd_lanes_count, i32) = @intFromFloat(exponent);
    const biased_exponents: @Vector(simd_lanes_count, u32) = @bitCast(integer_exponents + @as(@Vector(simd_lanes_count, i32), @splat(127)));
    const exponent_bits = biased_exponents << @as(@Vector(simd_lanes_count, u5), @splat(23));
    return polynomial * @as(F32x8, @bitCast(exponent_bits));
}

// ─── Probability-Value Accumulation ───────────────────────────────────────

inline fn accumulateValueRows(comptime query_rows_count: usize, comptime column_begin: usize, output_rows: [query_rows_count][]f32, probability_rows: [query_rows_count][]const f32, previous_scales: [query_rows_count]f32, values: []const f32, key_positions_count: usize, output_has_previous_keys: bool) void {
    assert(query_rows_count == 1 or query_rows_count == 2);

    const output_vectors_count = @divExact(simd_lanes_count, query_rows_count);
    assert(column_begin + output_vectors_count * simd_lanes_count <= head_width);
    assert(values.len == key_positions_count * head_width);
    inline for (0..query_rows_count) |query_row| {
        assert(output_rows[query_row].len == head_width);
        assert(probability_rows[query_row].len == key_positions_count);
    }

    var accumulators: [query_rows_count][output_vectors_count]F32x8 = undefined;
    inline for (0..query_rows_count) |query_row| {
        const previous_scale: F32x8 = @splat(previous_scales[query_row]);
        inline for (0..output_vectors_count) |output_vector| {
            const output_column = column_begin + output_vector * simd_lanes_count;
            accumulators[query_row][output_vector] = if (output_has_previous_keys) output_rows[query_row][output_column..][0..simd_lanes_count].* * previous_scale else @splat(0);
        }
    }

    for (0..key_positions_count) |key_offset| {
        const value_row = values[key_offset * head_width ..][0..head_width];
        if (query_rows_count == 1) {
            const probability: F32x8 = @splat(probability_rows[0][key_offset]);
            inline for (0..output_vectors_count) |output_vector| {
                const output_column = column_begin + output_vector * simd_lanes_count;
                accumulators[0][output_vector] = @mulAdd(F32x8, probability, value_row[output_column..][0..simd_lanes_count].*, accumulators[0][output_vector]);
            }
        } else {
            var value_vectors: [output_vectors_count]F32x8 = undefined;
            inline for (0..output_vectors_count) |output_vector| {
                const output_column = column_begin + output_vector * simd_lanes_count;
                value_vectors[output_vector] = value_row[output_column..][0..simd_lanes_count].*;
            }
            inline for (0..query_rows_count) |query_row| {
                const probability: F32x8 = @splat(probability_rows[query_row][key_offset]);
                inline for (0..output_vectors_count) |output_vector| {
                    accumulators[query_row][output_vector] = @mulAdd(F32x8, probability, value_vectors[output_vector], accumulators[query_row][output_vector]);
                }
            }
        }
    }

    inline for (0..query_rows_count) |query_row| {
        inline for (0..output_vectors_count) |output_vector| {
            const output_column = column_begin + output_vector * simd_lanes_count;
            output_rows[query_row][output_column..][0..simd_lanes_count].* = accumulators[query_row][output_vector];
        }
    }
}

inline fn scaleScores(scores: []f32, scale: f32) void {
    const scales: F32x8 = @splat(scale);
    var score_index: usize = 0;
    while (score_index + simd_lanes_count <= scores.len) : (score_index += simd_lanes_count) {
        const score_vector: F32x8 = scores[score_index..][0..simd_lanes_count].*;
        scores[score_index..][0..simd_lanes_count].* = score_vector * scales;
    }
    while (score_index < scores.len) : (score_index += 1) {
        scores[score_index] *= scale;
    }
}

inline fn scaleVector(values: []f32, scale: f32) void {
    assert(values.len == head_width);

    const scales: F32x8 = @splat(scale);
    var value_index: usize = 0;
    while (value_index < values.len) : (value_index += simd_lanes_count) {
        const value_vector: F32x8 = values[value_index..][0..simd_lanes_count].*;
        values[value_index..][0..simd_lanes_count].* = value_vector * scales;
    }
}

// ─── Packed-Key Transpose ──────────────────────────────────────────────────

inline fn storeTransposedTile(rows: [simd_lanes_count]F32x8, packed_output: []f32, packed_positions_count: usize, depth_begin: usize, position_begin: usize) void {
    const columns = transposeTileAvx2(rows);
    for (0..simd_lanes_count) |depth_offset| {
        const offset = (depth_begin + depth_offset) * packed_positions_count + position_begin;
        packed_output[offset..][0..simd_lanes_count].* = columns[depth_offset];
    }
}

inline fn transposeTileAvx2(rows: [simd_lanes_count]F32x8) [simd_lanes_count]F32x8 {
    var row_0 = rows[0];
    var row_1 = rows[1];
    var row_2 = rows[2];
    var row_3 = rows[3];
    var row_4 = rows[4];
    var row_5 = rows[5];
    var row_6 = rows[6];
    var row_7 = rows[7];
    var scratch: F32x8 = undefined;

    // PERFORMANCE: Zig lowers an equivalent arbitrary `@shuffle` transpose to
    // scalar `vinsertps` chains. This fixed AVX2 network keeps every row in YMM
    // registers; changes require checking the generated assembly again.
    asm volatile (
        \\ vunpcklps %[row_1], %[row_0], %[scratch]
        \\ vunpckhps %[row_1], %[row_0], %[row_1]
        \\ vmovaps %[scratch], %[row_0]
        \\ vunpcklps %[row_3], %[row_2], %[scratch]
        \\ vunpckhps %[row_3], %[row_2], %[row_3]
        \\ vmovaps %[scratch], %[row_2]
        \\ vunpcklps %[row_5], %[row_4], %[scratch]
        \\ vunpckhps %[row_5], %[row_4], %[row_5]
        \\ vmovaps %[scratch], %[row_4]
        \\ vunpcklps %[row_7], %[row_6], %[scratch]
        \\ vunpckhps %[row_7], %[row_6], %[row_7]
        \\ vmovaps %[scratch], %[row_6]
        \\ vshufps $0x44, %[row_2], %[row_0], %[scratch]
        \\ vshufps $0xee, %[row_2], %[row_0], %[row_2]
        \\ vmovaps %[scratch], %[row_0]
        \\ vshufps $0x44, %[row_3], %[row_1], %[scratch]
        \\ vshufps $0xee, %[row_3], %[row_1], %[row_3]
        \\ vmovaps %[scratch], %[row_1]
        \\ vshufps $0x44, %[row_6], %[row_4], %[scratch]
        \\ vshufps $0xee, %[row_6], %[row_4], %[row_6]
        \\ vmovaps %[scratch], %[row_4]
        \\ vshufps $0x44, %[row_7], %[row_5], %[scratch]
        \\ vshufps $0xee, %[row_7], %[row_5], %[row_7]
        \\ vmovaps %[scratch], %[row_5]
        \\ vperm2f128 $0x20, %[row_4], %[row_0], %[scratch]
        \\ vperm2f128 $0x31, %[row_4], %[row_0], %[row_4]
        \\ vmovaps %[scratch], %[row_0]
        \\ vperm2f128 $0x20, %[row_6], %[row_2], %[scratch]
        \\ vperm2f128 $0x31, %[row_6], %[row_2], %[row_6]
        \\ vmovaps %[scratch], %[row_2]
        \\ vperm2f128 $0x20, %[row_5], %[row_1], %[scratch]
        \\ vperm2f128 $0x31, %[row_5], %[row_1], %[row_5]
        \\ vmovaps %[scratch], %[row_1]
        \\ vperm2f128 $0x20, %[row_7], %[row_3], %[scratch]
        \\ vperm2f128 $0x31, %[row_7], %[row_3], %[row_7]
        \\ vmovaps %[scratch], %[row_3]
        : [row_0] "+x" (row_0),
          [row_1] "+x" (row_1),
          [row_2] "+x" (row_2),
          [row_3] "+x" (row_3),
          [row_4] "+x" (row_4),
          [row_5] "+x" (row_5),
          [row_6] "+x" (row_6),
          [row_7] "+x" (row_7),
          [scratch] "=&x" (scratch),
    );

    return .{ row_0, row_2, row_1, row_3, row_4, row_6, row_5, row_7 };
}

// ─── Scratch Layout ────────────────────────────────────────────────────────

inline fn packedPositionsCount(positions_count: usize) usize {
    return std.mem.alignForward(usize, positions_count, simd_lanes_count);
}

inline fn laneScratchValuesCount(positions_count: usize) usize {
    return head_width * packedPositionsCount(positions_count) + positions_count * head_width + query_positions_per_tile * head_width + query_positions_per_tile * key_positions_per_tile;
}
