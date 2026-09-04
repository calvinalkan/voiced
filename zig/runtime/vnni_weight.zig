//! This file owns the packed weight representation shared by model conversion,
//! embedding lookup, and quantized linear inference. Every 32-byte group stores four
//! adjacent input values for each of eight adjacent output rows:
//!
//!   [o0k0 o0k1 o0k2 o0k3 ... o7k0 o7k1 o7k2 o7k3]
//!
//! Activations add 128 before unsigned-by-signed VNNI multiplication. Each
//! output row therefore carries compensation equal to -128 times the sum of
//! its quantized weights.

const std = @import("std");
const assert = std.debug.assert;

pub const output_rows_per_block: usize = 8;
pub const input_values_per_group: usize = 4;
pub const values_per_group: usize = output_rows_per_block * input_values_per_group;
pub const activation_zero_point: i32 = 128;

/// `QuantizedWeight` exposes one o8/k4-packed INT8 matrix and the scale and
/// compensation values needed to recover its Float32 result. The views borrow
/// one packed model image.
pub const QuantizedWeight = struct {
    values: []const i8,
    scales: []const f32,
    compensation: []const i32,
    output_rows_count: usize,
    input_values_count: usize,

    pub fn layout(weight: QuantizedWeight) Layout {
        const weight_layout = Layout.init(weight.output_rows_count, weight.input_values_count);
        assert(weight.values.len == weight_layout.valuesCount());
        assert(weight.scales.len == weight.output_rows_count);
        assert(weight.compensation.len == weight.output_rows_count);

        return weight_layout;
    }
};

/// `Layout` centralizes the o8/k4 shape and byte-offset contract. Offsets select
/// INT8 elements, so they are also byte offsets within a packed values slice.
pub const Layout = struct {
    output_rows_count: usize,
    input_values_count: usize,

    pub fn init(output_rows_count: usize, input_values_count: usize) Layout {
        assert(output_rows_count > 0);
        assert(input_values_count > 0);
        assert(output_rows_count % output_rows_per_block == 0);
        assert(input_values_count % input_values_per_group == 0);

        return .{
            .output_rows_count = output_rows_count,
            .input_values_count = input_values_count,
        };
    }

    pub fn valuesCount(layout: Layout) usize {
        return std.math.mul(usize, layout.output_rows_count, layout.input_values_count) catch unreachable;
    }

    pub fn outputBlocksCount(layout: Layout) usize {
        return @divExact(layout.output_rows_count, output_rows_per_block);
    }

    pub fn outputBlockSize(layout: Layout) usize {
        return output_rows_per_block * layout.input_values_count;
    }

    pub fn outputBlockOffset(layout: Layout, output_block_index: usize) usize {
        assert(output_block_index < layout.outputBlocksCount());

        return output_block_index * layout.outputBlockSize();
    }

    pub fn depthGroupOffset(layout: Layout, input_value_start_index: usize) usize {
        assert(input_value_start_index < layout.input_values_count);
        assert(input_value_start_index % input_values_per_group == 0);

        return @divExact(input_value_start_index, input_values_per_group) * values_per_group;
    }

    pub fn groupOffset(layout: Layout, output_block_index: usize, input_value_start_index: usize) usize {
        return layout.outputBlockOffset(output_block_index) + layout.depthGroupOffset(input_value_start_index);
    }

    pub fn valueOffset(layout: Layout, output_row_index: usize, input_value_index: usize) usize {
        assert(output_row_index < layout.output_rows_count);
        assert(input_value_index < layout.input_values_count);

        const output_block_index = output_row_index / output_rows_per_block;
        const block_row_index = output_row_index % output_rows_per_block;
        const input_value_start_index = input_value_index / input_values_per_group * input_values_per_group;
        const group_value_index = input_value_index % input_values_per_group;

        return layout.groupOffset(output_block_index, input_value_start_index) + block_row_index * input_values_per_group + group_value_index;
    }
};

pub inline fn roundFloat32VectorToNearestEven(values: anytype) @TypeOf(values) {
    const Float32Vector = @TypeOf(values);
    comptime {
        if (Float32Vector != @Vector(4, f32) and Float32Vector != @Vector(8, f32)) {
            @compileError("expected a four- or eight-lane Float32 vector");
        }
    }

    // CTranslate2 quantization rounds with `vroundps` in nearest-even mode.
    // Zig's `@round` rounds midpoints away from zero and would change packed
    // weights or shifted activations for exact `.5` inputs.
    var rounded: Float32Vector = undefined;
    asm volatile ("vroundps $0, %[values], %[rounded]"
        : [rounded] "=x" (rounded),
        : [values] "x" (values),
    );
    return rounded;
}
