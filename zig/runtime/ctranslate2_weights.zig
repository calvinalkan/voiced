//! This file recognizes the pristine CTranslate2 version-6 Whisper model and
//! converts it directly into a packed model image. It owns source names, source
//! ordering, metadata values, Float16 decoding, and quantization; it does not
//! own the destination image ABI.

const std = @import("std");
const model_specification = @import("model_specification.zig");
const packed_model_image = @import("packed_model_image.zig");
const vnni_weight = @import("vnni_weight.zig");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ModelKind = model_specification.ModelKind;
const ModelSpecification = model_specification.ModelSpecification;
const Builder = packed_model_image.Builder;
const SectionKind = packed_model_image.SectionKind;
const TensorDimensions = packed_model_image.TensorDimensions;
const QuantizedSectionLayout = packed_model_image.QuantizedSectionLayout;

pub const weights_size_max: usize = 512 * 1024 * 1024;

pub const ConvertError = error{
    OutOfMemory,
    InvalidPristineWeights,
};

/// `convertToPackedImage` validates the complete canonical CTranslate2 source
/// and returns one owned target-2 image. It writes each tensor directly into
/// its final section and retains no reference to `weights`.
pub fn convertToPackedImage(allocator: Allocator, kind: ModelKind, weights: []const u8) ConvertError![]align(packed_model_image.alignment) u8 {
    // ── Read And Validate The Source Header ──

    if (weights.len > weights_size_max) {
        return error.InvalidPristineWeights;
    }

    var reader: WeightsReader = .{ .bytes = weights };

    const binary_version = try reader.readInt(u32);
    if (binary_version != binary_version_supported) {
        return error.InvalidPristineWeights;
    }

    const specification_name = try reader.readString();
    if (!std.mem.eql(u8, specification_name, "WhisperSpec")) {
        return error.InvalidPristineWeights;
    }

    const specification_revision = try reader.readInt(u32);
    if (specification_revision != whisper_specification_revision) {
        return error.InvalidPristineWeights;
    }

    const specification = kind.specification();
    const tensors_count = try reader.readInt(u32);
    if (tensors_count != countPristineTensors(specification)) {
        return error.InvalidPristineWeights;
    }

    var builder = try Builder.init(allocator, kind);
    errdefer builder.deinit();

    // ── Convert Decoder Weights ──
    //
    // The explicit sequence rejects missing, duplicate, reordered, or unknown
    // records while writing accepted tensors directly into their final image
    // sections.

    for (decoder_tensors_before_layers) |plan| {
        try convertTensor(&reader, &builder, specification, packed_model_image.model_wide_layer_index, plan);
    }

    for (0..specification.decoder_layers_count) |layer_ordinal| {
        const layer_index = pristineLayerIndexFromOrdinal(layer_ordinal, specification.decoder_layers_count);
        for (decoder_layer_tensors) |plan| {
            try convertLayerTensor(&reader, &builder, specification, "decoder", layer_index, plan);
        }
    }

    for (decoder_tensors_after_layers) |plan| {
        try convertTensor(&reader, &builder, specification, packed_model_image.model_wide_layer_index, plan);
    }

    // ── Convert Encoder Weights ──

    for (encoder_tensors_before_layers) |plan| {
        try convertTensor(&reader, &builder, specification, packed_model_image.model_wide_layer_index, plan);
    }

    for (0..specification.encoder_layers_count) |layer_ordinal| {
        const layer_index = pristineLayerIndexFromOrdinal(layer_ordinal, specification.encoder_layers_count);
        for (encoder_layer_tensors) |plan| {
            try convertLayerTensor(&reader, &builder, specification, "encoder", layer_index, plan);
        }
    }

    for (encoder_tensors_after_layers) |plan| {
        try convertTensor(&reader, &builder, specification, packed_model_image.model_wide_layer_index, plan);
    }

    // ── Validate The Tied-Embedding Alias ──
    //
    // Whisper uses its decoder embedding matrix again for vocabulary
    // projection. The packed image stores one matrix for both operations.

    const aliases_count = try reader.readInt(u32);
    if (aliases_count != 1) {
        return error.InvalidPristineWeights;
    }

    const alias = try reader.readString();
    if (!std.mem.eql(u8, alias, "decoder/projection/weight")) {
        return error.InvalidPristineWeights;
    }

    const aliased_tensor = try reader.readString();
    if (!std.mem.eql(u8, aliased_tensor, "decoder/embeddings/weight")) {
        return error.InvalidPristineWeights;
    }

    if (reader.offset != weights.len) {
        return error.InvalidPristineWeights;
    }

    return builder.finish();
}

// ─── CTranslate2 Weight Reader ─────────────────────────────────────────────
//
// The reader borrows names and tensor payloads from the caller's image, copies
// only small dimension arrays, and rejects every read that crosses the image.
// CTranslate2 calls every stored tensor or scalar a variable; this reader uses
// tensor throughout because a scalar is a rank-zero tensor.

const WeightsReader = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn readTensor(reader: *WeightsReader) ConvertError!Tensor {
        const name = try reader.readString();
        if (name.len == 0) {
            return error.InvalidPristineWeights;
        }

        const dimensions_count: usize = @intCast(try reader.readInt(u8));
        if (dimensions_count > Tensor.dimensions_count_max) {
            return error.InvalidPristineWeights;
        }

        var dimensions: [Tensor.dimensions_count_max]usize = @splat(0);
        var elements_count: usize = 1;
        for (0..dimensions_count) |dimension_index| {
            const dimension: usize = @intCast(try reader.readInt(u32));
            dimensions[dimension_index] = dimension;

            elements_count = std.math.mul(usize, elements_count, dimension) catch {
                return error.InvalidPristineWeights;
            };
        }

        const data_type = std.enums.fromInt(DataType, try reader.readInt(u8)) orelse {
            return error.InvalidPristineWeights;
        };
        const data_size: usize = @intCast(try reader.readInt(u32));
        const expected_data_size = std.math.mul(usize, elements_count, data_type.elementSize()) catch {
            return error.InvalidPristineWeights;
        };
        if (data_size != expected_data_size) {
            return error.InvalidPristineWeights;
        }

        const data = try reader.readBytes(data_size);
        return .{
            .name = name,
            .dimensions = dimensions,
            .dimensions_count = dimensions_count,
            .data_type = data_type,
            .data = data,
        };
    }

    fn readString(reader: *WeightsReader) ConvertError![]const u8 {
        const encoded_size = try reader.readInt(u16);
        if (encoded_size == 0) {
            return error.InvalidPristineWeights;
        }

        const encoded = try reader.readBytes(encoded_size);
        if (encoded[encoded.len - 1] != 0) {
            return error.InvalidPristineWeights;
        }

        const value = encoded[0 .. encoded.len - 1];
        if (std.mem.indexOfScalar(u8, value, 0) != null) {
            return error.InvalidPristineWeights;
        }

        return value;
    }

    fn readInt(reader: *WeightsReader, comptime Int: type) ConvertError!Int {
        const bytes = try reader.readBytes(@sizeOf(Int));

        return std.mem.readInt(Int, bytes[0..@sizeOf(Int)], .little);
    }

    fn readBytes(reader: *WeightsReader, size: usize) ConvertError![]const u8 {
        assert(reader.offset <= reader.bytes.len);
        if (size > reader.bytes.len - reader.offset) {
            return error.InvalidPristineWeights;
        }

        const bytes = reader.bytes[reader.offset..][0..size];
        reader.offset += size;
        return bytes;
    }
};

const Tensor = struct {
    const dimensions_count_max = 3;

    name: []const u8,
    dimensions: [dimensions_count_max]usize,
    dimensions_count: usize,
    data_type: DataType,
    data: []const u8,

    fn hasDimensions(tensor: *const Tensor, expected_dimensions: []const usize) bool {
        if (tensor.dimensions_count != expected_dimensions.len) {
            return false;
        }

        return std.mem.eql(usize, tensor.dimensions[0..tensor.dimensions_count], expected_dimensions);
    }

    fn hasScalarValue(tensor: *const Tensor, comptime Int: type, expected_value: Int) bool {
        const expected_data_type: DataType = switch (Int) {
            i8 => .int8,
            i16 => .int16,
            else => @compileError("unsupported CTranslate2 scalar type"),
        };
        if (tensor.dimensions_count != 0 or tensor.data_type != expected_data_type) {
            return false;
        }

        const actual_value = std.mem.readInt(Int, tensor.data[0..@sizeOf(Int)], .little);
        return actual_value == expected_value;
    }
};

// These numeric values are part of CTranslate2's version-6 `model.bin` ABI.
const DataType = enum(u8) {
    float32 = 0,
    int8 = 1,
    int16 = 2,
    int32 = 3,
    float16 = 4,
    bfloat16 = 5,

    fn elementSize(data_type: DataType) usize {
        return switch (data_type) {
            .int8 => @sizeOf(i8),
            .int16, .float16, .bfloat16 => @sizeOf(u16),
            .float32 => @sizeOf(f32),
            .int32 => @sizeOf(i32),
        };
    }
};

// ─── Pristine Whisper Schema ───────────────────────────────────────────────
//
// These plans own only CTranslate2 names, ordering, and metadata. Packed image
// shape, domain, and encoding come from `sectionDefinition`.

const binary_version_supported: u32 = 6;
const whisper_specification_revision: u32 = 3;
const tensor_name_size_max: usize = 128;

const TensorPlan = struct {
    name: []const u8,
    handling: union(enum) {
        packed_section: SectionKind,
        scalar_i8: i8,
        scalar_i16: i16,
        encoder_attention_heads_count,
        decoder_attention_heads_count,
    },
};

const decoder_tensors_before_layers = [_]TensorPlan{
    .{ .name = "decoder/activation", .handling = .{ .scalar_i8 = 3 } },
    .{ .name = "decoder/alibi", .handling = .{ .scalar_i8 = 0 } },
    .{ .name = "decoder/alignment_heads", .handling = .{ .scalar_i16 = 1 } },
    .{ .name = "decoder/alignment_layer", .handling = .{ .scalar_i16 = -1 } },
    .{ .name = "decoder/embeddings/weight", .handling = .{ .packed_section = .decoder_embeddings_weight } },
};

const decoder_layer_tensors = [_]TensorPlan{
    .{ .name = "attention/layer_norm/beta", .handling = .{ .packed_section = .decoder_layer_cross_attention_layer_norm_beta } },
    .{ .name = "attention/layer_norm/gamma", .handling = .{ .packed_section = .decoder_layer_cross_attention_layer_norm_gamma } },
    .{ .name = "attention/linear_0/bias", .handling = .{ .packed_section = .decoder_layer_cross_attention_query_bias } },
    .{ .name = "attention/linear_0/weight", .handling = .{ .packed_section = .decoder_layer_cross_attention_query_weight } },
    .{ .name = "attention/linear_1/bias", .handling = .{ .packed_section = .decoder_layer_cross_attention_key_value_bias } },
    .{ .name = "attention/linear_1/weight", .handling = .{ .packed_section = .decoder_layer_cross_attention_key_value_weight } },
    .{ .name = "attention/linear_2/bias", .handling = .{ .packed_section = .decoder_layer_cross_attention_output_bias } },
    .{ .name = "attention/linear_2/weight", .handling = .{ .packed_section = .decoder_layer_cross_attention_output_weight } },
    .{ .name = "ffn/layer_norm/beta", .handling = .{ .packed_section = .decoder_layer_ffn_layer_norm_beta } },
    .{ .name = "ffn/layer_norm/gamma", .handling = .{ .packed_section = .decoder_layer_ffn_layer_norm_gamma } },
    .{ .name = "ffn/linear_0/bias", .handling = .{ .packed_section = .decoder_layer_ffn_expansion_bias } },
    .{ .name = "ffn/linear_0/weight", .handling = .{ .packed_section = .decoder_layer_ffn_expansion_weight } },
    .{ .name = "ffn/linear_1/bias", .handling = .{ .packed_section = .decoder_layer_ffn_contraction_bias } },
    .{ .name = "ffn/linear_1/weight", .handling = .{ .packed_section = .decoder_layer_ffn_contraction_weight } },
    .{ .name = "self_attention/layer_norm/beta", .handling = .{ .packed_section = .decoder_layer_self_attention_layer_norm_beta } },
    .{ .name = "self_attention/layer_norm/gamma", .handling = .{ .packed_section = .decoder_layer_self_attention_layer_norm_gamma } },
    .{ .name = "self_attention/linear_0/bias", .handling = .{ .packed_section = .decoder_layer_self_attention_query_key_value_bias } },
    .{ .name = "self_attention/linear_0/weight", .handling = .{ .packed_section = .decoder_layer_self_attention_query_key_value_weight } },
    .{ .name = "self_attention/linear_1/bias", .handling = .{ .packed_section = .decoder_layer_self_attention_output_bias } },
    .{ .name = "self_attention/linear_1/weight", .handling = .{ .packed_section = .decoder_layer_self_attention_output_weight } },
};

const decoder_tensors_after_layers = [_]TensorPlan{
    .{ .name = "decoder/layer_norm/beta", .handling = .{ .packed_section = .decoder_layer_norm_beta } },
    .{ .name = "decoder/layer_norm/gamma", .handling = .{ .packed_section = .decoder_layer_norm_gamma } },
    .{ .name = "decoder/num_heads", .handling = .decoder_attention_heads_count },
    .{ .name = "decoder/position_encodings/encodings", .handling = .{ .packed_section = .decoder_position_encodings } },
    .{ .name = "decoder/pre_norm", .handling = .{ .scalar_i8 = 1 } },
    .{ .name = "decoder/scale_embeddings", .handling = .{ .scalar_i8 = 0 } },
    .{ .name = "decoder/start_from_zero_embedding", .handling = .{ .scalar_i8 = 0 } },
};

const encoder_tensors_before_layers = [_]TensorPlan{
    .{ .name = "encoder/conv1/bias", .handling = .{ .packed_section = .encoder_convolution_1_bias } },
    .{ .name = "encoder/conv1/weight", .handling = .{ .packed_section = .encoder_convolution_1_weight } },
    .{ .name = "encoder/conv2/bias", .handling = .{ .packed_section = .encoder_convolution_2_bias } },
    .{ .name = "encoder/conv2/weight", .handling = .{ .packed_section = .encoder_convolution_2_weight } },
};

const encoder_layer_tensors = [_]TensorPlan{
    .{ .name = "ffn/layer_norm/beta", .handling = .{ .packed_section = .encoder_layer_ffn_layer_norm_beta } },
    .{ .name = "ffn/layer_norm/gamma", .handling = .{ .packed_section = .encoder_layer_ffn_layer_norm_gamma } },
    .{ .name = "ffn/linear_0/bias", .handling = .{ .packed_section = .encoder_layer_ffn_expansion_bias } },
    .{ .name = "ffn/linear_0/weight", .handling = .{ .packed_section = .encoder_layer_ffn_expansion_weight } },
    .{ .name = "ffn/linear_1/bias", .handling = .{ .packed_section = .encoder_layer_ffn_contraction_bias } },
    .{ .name = "ffn/linear_1/weight", .handling = .{ .packed_section = .encoder_layer_ffn_contraction_weight } },
    .{ .name = "self_attention/layer_norm/beta", .handling = .{ .packed_section = .encoder_layer_self_attention_layer_norm_beta } },
    .{ .name = "self_attention/layer_norm/gamma", .handling = .{ .packed_section = .encoder_layer_self_attention_layer_norm_gamma } },
    .{ .name = "self_attention/linear_0/bias", .handling = .{ .packed_section = .encoder_layer_self_attention_query_key_value_bias } },
    .{ .name = "self_attention/linear_0/weight", .handling = .{ .packed_section = .encoder_layer_self_attention_query_key_value_weight } },
    .{ .name = "self_attention/linear_1/bias", .handling = .{ .packed_section = .encoder_layer_self_attention_output_bias } },
    .{ .name = "self_attention/linear_1/weight", .handling = .{ .packed_section = .encoder_layer_self_attention_output_weight } },
};

const encoder_tensors_after_layers = [_]TensorPlan{
    .{ .name = "encoder/layer_norm/beta", .handling = .{ .packed_section = .encoder_layer_norm_beta } },
    .{ .name = "encoder/layer_norm/gamma", .handling = .{ .packed_section = .encoder_layer_norm_gamma } },
    .{ .name = "encoder/num_heads", .handling = .encoder_attention_heads_count },
    .{ .name = "encoder/position_encodings/encodings", .handling = .{ .packed_section = .encoder_position_encodings } },
};

comptime {
    assert(decoder_tensors_before_layers.len == 5);
    assert(decoder_layer_tensors.len == 20);
    assert(decoder_tensors_after_layers.len == 7);
    assert(encoder_tensors_before_layers.len == 4);
    assert(encoder_layer_tensors.len == 12);
    assert(encoder_tensors_after_layers.len == 4);
}

fn countPristineTensors(specification: ModelSpecification) u32 {
    const tensors_count = decoder_tensors_before_layers.len +
        specification.decoder_layers_count * decoder_layer_tensors.len +
        decoder_tensors_after_layers.len +
        encoder_tensors_before_layers.len +
        specification.encoder_layers_count * encoder_layer_tensors.len +
        encoder_tensors_after_layers.len;
    assert(tensors_count <= std.math.maxInt(u32));

    return @intCast(tensors_count);
}

fn pristineLayerIndexFromOrdinal(layer_ordinal: usize, layers_count: usize) u16 {
    assert(layer_ordinal < layers_count);
    assert(layers_count == 6 or layers_count == 12);

    if (layers_count == 6) {
        return @intCast(layer_ordinal);
    }

    const lexicographically_sorted_layer_indexes = [_]u16{ 0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9 };
    return lexicographically_sorted_layer_indexes[layer_ordinal];
}

fn convertLayerTensor(reader: *WeightsReader, builder: *Builder, specification: ModelSpecification, scope: []const u8, layer_index: u16, plan: TensorPlan) ConvertError!void {
    assert(scope.len > 0);
    assert(plan.name.len > 0);

    var tensor_name_buffer: [tensor_name_size_max]u8 = undefined;
    const tensor_name = std.fmt.bufPrint(&tensor_name_buffer, "{s}/layer_{d}/{s}", .{ scope, layer_index, plan.name }) catch unreachable;
    const complete_plan: TensorPlan = .{ .name = tensor_name, .handling = plan.handling };

    try convertTensor(reader, builder, specification, layer_index, complete_plan);
}

fn convertTensor(reader: *WeightsReader, builder: *Builder, specification: ModelSpecification, layer_index: u16, plan: TensorPlan) ConvertError!void {
    const tensor = try reader.readTensor();
    if (!std.mem.eql(u8, tensor.name, plan.name)) {
        return error.InvalidPristineWeights;
    }

    switch (plan.handling) {
        .scalar_i8 => |expected_value| {
            if (!tensor.hasScalarValue(i8, expected_value)) {
                return error.InvalidPristineWeights;
            }
        },
        .scalar_i16 => |expected_value| {
            if (!tensor.hasScalarValue(i16, expected_value)) {
                return error.InvalidPristineWeights;
            }
        },
        .encoder_attention_heads_count => {
            const expected_value: i16 = @intCast(specification.encoder_attention_heads_count);
            if (!tensor.hasScalarValue(i16, expected_value)) {
                return error.InvalidPristineWeights;
            }
        },
        .decoder_attention_heads_count => {
            const expected_value: i16 = @intCast(specification.decoder_attention_heads_count);
            if (!tensor.hasScalarValue(i16, expected_value)) {
                return error.InvalidPristineWeights;
            }
        },
        .packed_section => |section_kind| {
            const definition = packed_model_image.sectionDefinition(section_kind, specification);
            if (tensor.data_type != .float16 or !tensor.hasDimensions(definition.dimensions.slice())) {
                return error.InvalidPristineWeights;
            }

            const section = builder.claimSection(section_kind, layer_index);
            assert(std.meta.eql(definition, section.definition));
            switch (section.definition.encoding) {
                .float32_little_endian => try convertFloat16TensorToFloat32(tensor.data, section.payload),
                .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation => try quantizeVnniWeight(tensor.data, section.definition.dimensions, section.payload),
            }
        },
    }
}

// ─── Packed Tensor Conversion ──────────────────────────────────────────────

fn convertFloat16TensorToFloat32(source: []const u8, destination: []u8) ConvertError!void {
    assert(source.len % @sizeOf(f16) == 0);
    assert(destination.len == source.len / @sizeOf(f16) * @sizeOf(f32));

    const values_count = source.len / @sizeOf(f16);
    for (0..values_count) |value_index| {
        const value = readFloat16(source, value_index);
        if (!std.math.isFinite(value)) {
            return error.InvalidPristineWeights;
        }

        packed_model_image.writeFloat32(destination, value_index * @sizeOf(f32), value);
    }
}

fn quantizeVnniWeight(source: []const u8, dimensions: TensorDimensions, destination: []u8) ConvertError!void {
    const layout = packed_model_image.quantizedSectionLayout(dimensions);
    assert(dimensions.count == 2 or dimensions.count == 3);
    assert(source.len == layout.weights_size * @sizeOf(f16));
    assert(destination.len == layout.section_size);
    assert(layout.weight.input_values_count % 8 == 0);

    try calculateVnniWeightScales(source, layout, destination);
    quantizeAndPackVnniWeight(source, layout, destination);
}

fn calculateVnniWeightScales(source: []const u8, layout: QuantizedSectionLayout, destination: []u8) ConvertError!void {
    const F16x8 = @Vector(8, f16);
    const F32x8 = @Vector(8, f32);
    const U16x8 = @Vector(8, u16);
    const exponent_mask: U16x8 = @splat(0x7c00);

    assert(layout.weight.input_values_count % 8 == 0);

    for (0..layout.weight.output_rows_count) |output_row_index| {
        var absolute_maximums: F32x8 = @splat(0);
        var input_value_start_index: usize = 0;
        while (input_value_start_index < layout.weight.input_values_count) : (input_value_start_index += 8) {
            const source_value_index = output_row_index * layout.weight.input_values_count + input_value_start_index;
            const source_offset = source_value_index * @sizeOf(f16);
            const half_bits: U16x8 = @bitCast(source[source_offset..][0 .. 8 * @sizeOf(f16)].*);
            if (@reduce(.Or, half_bits & exponent_mask == exponent_mask)) {
                return error.InvalidPristineWeights;
            }

            const half_values: F16x8 = @bitCast(half_bits);
            const values: F32x8 = @floatCast(half_values);
            absolute_maximums = @max(absolute_maximums, @abs(values));
        }

        const absolute_maximum = @reduce(.Max, absolute_maximums);
        const scale: f32 = if (absolute_maximum == 0) 1 else 127.0 / absolute_maximum;
        assert(std.math.isFinite(scale));
        assert(scale > 0);
        packed_model_image.writeFloat32(destination, layout.scales_offset + output_row_index * @sizeOf(f32), scale);
    }
}

fn quantizeAndPackVnniWeight(source: []const u8, layout: QuantizedSectionLayout, destination: []u8) void {
    const F16x4 = @Vector(4, f16);
    const F32x4 = @Vector(4, f32);
    const F32x8 = @Vector(8, f32);
    const I32x4 = @Vector(4, i32);
    const I32x8 = @Vector(8, i32);
    const I8x4 = @Vector(4, i8);
    const U16x4 = @Vector(4, u16);
    const minimums: F32x4 = @splat(-127.0);
    const maximums: F32x4 = @splat(127.0);

    assert(source.len == layout.weights_size * @sizeOf(f16));
    assert(destination.len == layout.section_size);

    var output_row_start_index: usize = 0;
    while (output_row_start_index < layout.weight.output_rows_count) : (output_row_start_index += vnni_weight.output_rows_per_block) {
        var output_scales: F32x8 = undefined;
        inline for (0..vnni_weight.output_rows_per_block) |block_row_index| {
            const scale_offset = layout.scales_offset + (output_row_start_index + block_row_index) * @sizeOf(f32);
            output_scales[block_row_index] = packed_model_image.readFloat32(destination, scale_offset);
        }

        var quantized_values_sums: I32x8 = @splat(0);
        var input_value_start_index: usize = 0;
        while (input_value_start_index < layout.weight.input_values_count) : (input_value_start_index += vnni_weight.input_values_per_group) {
            var quantized_group_sums: I32x8 = undefined;

            inline for (0..vnni_weight.output_rows_per_block) |block_row_index| {
                const output_row_index = output_row_start_index + block_row_index;
                const source_value_index = output_row_index * layout.weight.input_values_count + input_value_start_index;
                const source_offset = source_value_index * @sizeOf(f16);
                const half_bits: U16x4 = @bitCast(source[source_offset..][0 .. vnni_weight.input_values_per_group * @sizeOf(f16)].*);
                const half_values: F16x4 = @bitCast(half_bits);
                const values: F32x4 = @floatCast(half_values);
                const scaled_values = values * @as(F32x4, @splat(output_scales[block_row_index]));
                const clamped_values = @min(@max(scaled_values, minimums), maximums);
                const rounded_values = vnni_weight.roundFloat32VectorToNearestEven(clamped_values);
                const quantized_integers: I32x4 = @intFromFloat(rounded_values);
                const quantized_values: I8x4 = @intCast(quantized_integers);
                const quantized_bytes: [vnni_weight.input_values_per_group]u8 = @bitCast(quantized_values);

                const packed_output_offset = layout.weight.valueOffset(output_row_index, input_value_start_index);
                destination[packed_output_offset..][0..vnni_weight.input_values_per_group].* = quantized_bytes;
                quantized_group_sums[block_row_index] = @reduce(.Add, quantized_integers);
            }

            quantized_values_sums += quantized_group_sums;
        }

        quantized_values_sums *= @as(I32x8, @splat(-vnni_weight.activation_zero_point));
        inline for (0..vnni_weight.output_rows_per_block) |block_row_index| {
            const compensation_offset = layout.compensation_offset + (output_row_start_index + block_row_index) * @sizeOf(i32);
            packed_model_image.writeInt(i32, destination, compensation_offset, quantized_values_sums[block_row_index]);
        }
    }
}

fn readFloat16(source: []const u8, value_index: usize) f32 {
    const byte_offset = value_index * @sizeOf(f16);
    assert(byte_offset + @sizeOf(f16) <= source.len);

    const bits = std.mem.readInt(u16, source[byte_offset..][0..@sizeOf(f16)], .little);
    const value: f16 = @bitCast(bits);
    return @floatCast(value);
}

// ─── Tests ─────────────────────────────────────────────────────────────────

test "rank-three VNNI fixture verifies quantization boundary values" {
    const allocator = std.testing.allocator;
    const dimensions = TensorDimensions.init3(8, 2, 4);
    const layout = packed_model_image.quantizedSectionLayout(dimensions);

    // Each output row contains two channels of four kernel values. Distinct
    // channel halves make the required `[channel][kernel]` depth flattening
    // observable in the literal packed bytes below.
    const source_values = [8][2][4]f32{
        .{ .{ 0.0, 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0, 0.0 } },
        .{ .{ 0.5, 1.5, 2.5, 3.5 }, .{ -0.5, -1.5, -2.5, 127.0 } },
        .{ .{ -127.0, -126.0, -1.0, 0.0 }, .{ 1.0, 126.0, 127.0, -0.0 } },
        .{ .{ -127.0, -64.0, -32.0, -16.0 }, .{ -8.0, -4.0, -2.0, -1.0 } },
        .{ .{ 64.0, -64.0, 32.0, -32.0 }, .{ 0.5, -0.5, 1.5, -1.5 } },
        .{ .{ 0.25 / 1024.0, -0.25 / 1024.0, 0.5 / 1024.0, -0.5 / 1024.0 }, .{ 0.75 / 1024.0, -0.75 / 1024.0, 1.0 / 1024.0, -1.0 / 1024.0 } },
        .{ .{ -0.5, -1.5, -2.5, -3.5 }, .{ 0.5, 1.5, 2.5, 127.0 } },
        .{ .{ -127.0, -0.25, 0.25, -0.75 }, .{ 0.75, -1.25, 1.25, 0.0 } },
    };
    const expected_weights = [8][8]i8{
        .{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 0, 2, 2, 4, 0, -2, -2, 127 },
        .{ -127, -126, -1, 0, 1, 126, 127, 0 },
        .{ -127, -64, -32, -16, -8, -4, -2, -1 },
        .{ 127, -127, 64, -64, 1, -1, 3, -3 },
        .{ 32, -32, 64, -64, 95, -95, 127, -127 },
        .{ 0, -2, -2, -4, 0, 2, 2, 127 },
        .{ -127, 0, 0, -1, 1, -1, 1, 0 },
    };
    const expected_packed_weights = [64]i8{
        0,   0,    0,  0,   0,  2,   2,   4,    -127, -126, -1,  0,   -127, -64, -32, -16,
        127, -127, 64, -64, 32, -32, 64,  -64,  0,    -2,   -2,  -4,  -127, 0,   0,   -1,
        0,   0,    0,  0,   0,  -2,  -2,  127,  1,    126,  127, 0,   -8,   -4,  -2,  -1,
        1,   -1,   3,  -3,  95, -95, 127, -127, 0,    2,    2,   127, 1,    -1,  1,   0,
    };
    const expected_scales = [8]f32{ 1.0, 1.0, 1.0, 1.0, 127.0 / 64.0, 127.0 * 1024.0, 1.0, 1.0 };

    const source = try allocator.alloc(u8, layout.weights_size * @sizeOf(f16));
    defer allocator.free(source);
    testWriteFloat16Values(source, layout, &source_values);

    const destination = try allocator.alloc(u8, layout.section_size);
    defer allocator.free(destination);
    @memset(destination, 0);

    try quantizeVnniWeight(source, dimensions, destination);

    try testExpectQuantizedPadding(destination, layout);

    const expected_packed_bytes: [expected_packed_weights.len]u8 = @bitCast(expected_packed_weights);
    try std.testing.expectEqual(expected_packed_bytes.len, layout.weights_size);
    try std.testing.expectEqualSlices(u8, &expected_packed_bytes, destination[0..layout.weights_size]);

    for (0..layout.weight.output_rows_count) |output_row_index| {
        try testExpectPackedScale(destination, layout, output_row_index, expected_scales[output_row_index]);
        try testExpectPackedCompensation(destination, layout, output_row_index, &expected_weights[output_row_index]);
    }
}

fn testWriteFloat16Values(destination: []u8, layout: QuantizedSectionLayout, source_values: *const [8][2][4]f32) void {
    assert(destination.len == layout.weights_size * @sizeOf(f16));
    assert(source_values.len == layout.weight.output_rows_count);
    assert(source_values[0].len * source_values[0][0].len == layout.weight.input_values_count);

    for (source_values, 0..) |channels, output_row_index| {
        for (channels, 0..) |kernel_values, channel_index| {
            for (kernel_values, 0..) |source_value, kernel_index| {
                const input_value_index = channel_index * kernel_values.len + kernel_index;
                const source_value_index = output_row_index * layout.weight.input_values_count + input_value_index;
                const value: f16 = @floatCast(source_value);
                std.mem.writeInt(u16, destination[source_value_index * @sizeOf(f16) ..][0..@sizeOf(f16)], @bitCast(value), .little);
            }
        }
    }
}

fn testExpectQuantizedPadding(destination: []const u8, layout: QuantizedSectionLayout) !void {
    try std.testing.expect(std.mem.allEqual(u8, destination[layout.weights_size..layout.scales_offset], 0));
    try std.testing.expect(std.mem.allEqual(u8, destination[layout.scales_offset + layout.scales_size .. layout.compensation_offset], 0));
}

fn testExpectPackedScale(destination: []const u8, layout: QuantizedSectionLayout, output_row_index: usize, expected_scale: f32) !void {
    const scale_offset = layout.scales_offset + output_row_index * @sizeOf(f32);
    const actual_scale_bits = std.mem.readInt(u32, destination[scale_offset..][0..@sizeOf(u32)], .little);
    try std.testing.expectEqual(@as(u32, @bitCast(expected_scale)), actual_scale_bits);
}

fn testExpectPackedCompensation(destination: []const u8, layout: QuantizedSectionLayout, output_row_index: usize, expected_weights: []const i8) !void {
    assert(expected_weights.len == layout.weight.input_values_count);

    var quantized_values_sum: i32 = 0;
    for (expected_weights) |quantized_value| {
        quantized_values_sum += quantized_value;
    }

    const compensation_offset = layout.compensation_offset + output_row_index * @sizeOf(i32);
    const expected_compensation = std.math.mul(i32, quantized_values_sum, -vnni_weight.activation_zero_point) catch unreachable;
    const actual_compensation = std.mem.readInt(i32, destination[compensation_offset..][0..@sizeOf(i32)], .little);
    try std.testing.expectEqual(expected_compensation, actual_compensation);
}
