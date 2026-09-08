//! This file owns the packed model image ABI shared by image construction and
//! zero-copy loading. The format stores no directory: model kind, section kind,
//! layer index, and packed target determine every payload shape and offset.
//!
//! Version 1 is a little-endian, relocatable, uncompressed image. Target 1 used
//! row-major convolution weights and o8/k4 linear weights. Target 2 stores every
//! quantized weight in the o8/k4 representation owned by `vnni_weight.zig`.
//! Changing physical weight encoding requires a new target; changing the header
//! or canonical section sequence requires a new format version.

const std = @import("std");
const model_specification = @import("model_specification.zig");
const vnni_weight = @import("vnni_weight.zig");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ModelKind = model_specification.ModelKind;
const ModelSpecification = model_specification.ModelSpecification;
const QuantizedWeight = vnni_weight.QuantizedWeight;

/// Alignment of every packed image. Buffers filled from persistent storage
/// must satisfy it before `Model.fromPackedImage` accepts them.
pub const alignment: usize = 64;

/// `format_version` identifies the header and canonical section sequence. The
/// packed target identifies the physical encoding of sections in that sequence.
pub const format_version: u32 = 1;

pub const model_wide_layer_index: u16 = std.math.maxInt(u16);

pub const LoadError = error{
    InvalidPackedImage,
    UnsupportedModelImageFormatVersion,
    UnsupportedModelKind,
    UnsupportedPackedTarget,
};

// Declaration order defines packed-image payload order. Explicit values protect
// the format from accidental insertion or reuse.
pub const SectionKind = enum(u16) {
    decoder_embeddings_weight = 1,
    decoder_layer_norm_beta = 2,
    decoder_layer_norm_gamma = 3,
    decoder_position_encodings = 4,
    encoder_convolution_1_bias = 5,
    encoder_convolution_1_weight = 6,
    encoder_convolution_2_bias = 7,
    encoder_convolution_2_weight = 8,
    encoder_layer_norm_beta = 9,
    encoder_layer_norm_gamma = 10,
    encoder_position_encodings = 11,

    encoder_layer_self_attention_layer_norm_beta = 12,
    encoder_layer_self_attention_layer_norm_gamma = 13,
    encoder_layer_self_attention_query_key_value_bias = 14,
    encoder_layer_self_attention_query_key_value_weight = 15,
    encoder_layer_self_attention_output_bias = 16,
    encoder_layer_self_attention_output_weight = 17,
    encoder_layer_ffn_layer_norm_beta = 18,
    encoder_layer_ffn_layer_norm_gamma = 19,
    encoder_layer_ffn_expansion_bias = 20,
    encoder_layer_ffn_expansion_weight = 21,
    encoder_layer_ffn_contraction_bias = 22,
    encoder_layer_ffn_contraction_weight = 23,

    decoder_layer_self_attention_layer_norm_beta = 24,
    decoder_layer_self_attention_layer_norm_gamma = 25,
    decoder_layer_self_attention_query_key_value_bias = 26,
    decoder_layer_self_attention_query_key_value_weight = 27,
    decoder_layer_self_attention_output_bias = 28,
    decoder_layer_self_attention_output_weight = 29,
    decoder_layer_cross_attention_layer_norm_beta = 30,
    decoder_layer_cross_attention_layer_norm_gamma = 31,
    decoder_layer_cross_attention_query_bias = 32,
    decoder_layer_cross_attention_query_weight = 33,
    decoder_layer_cross_attention_key_value_bias = 34,
    decoder_layer_cross_attention_key_value_weight = 35,
    decoder_layer_cross_attention_output_bias = 36,
    decoder_layer_cross_attention_output_weight = 37,
    decoder_layer_ffn_layer_norm_beta = 38,
    decoder_layer_ffn_layer_norm_gamma = 39,
    decoder_layer_ffn_expansion_bias = 40,
    decoder_layer_ffn_expansion_weight = 41,
    decoder_layer_ffn_contraction_bias = 42,
    decoder_layer_ffn_contraction_weight = 43,
};

comptime {
    assert(std.enums.values(SectionKind).len == 43);
    assert(@intFromEnum(SectionKind.decoder_embeddings_weight) == 1);
    assert(@intFromEnum(SectionKind.encoder_position_encodings) == 11);
    assert(@intFromEnum(SectionKind.encoder_layer_ffn_contraction_weight) == 23);
    assert(@intFromEnum(SectionKind.decoder_layer_ffn_contraction_weight) == 43);
}

pub const SectionDomain = enum {
    model,
    encoder_layer,
    decoder_layer,
};

pub const SectionEncoding = enum {
    float32_little_endian,
    vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation,
};

pub const TensorDimensions = struct {
    values: [3]usize,
    count: usize,

    pub fn init1(first: usize) TensorDimensions {
        assert(first > 0);

        return .{ .values = .{ first, 0, 0 }, .count = 1 };
    }

    pub fn init2(first: usize, second: usize) TensorDimensions {
        assert(first > 0);
        assert(second > 0);

        return .{ .values = .{ first, second, 0 }, .count = 2 };
    }

    pub fn init3(first: usize, second: usize, third: usize) TensorDimensions {
        assert(first > 0);
        assert(second > 0);
        assert(third > 0);

        return .{ .values = .{ first, second, third }, .count = 3 };
    }

    pub fn slice(dimensions: *const TensorDimensions) []const usize {
        assert(dimensions.count <= dimensions.values.len);

        return dimensions.values[0..dimensions.count];
    }

    pub fn elementsCount(dimensions: TensorDimensions) usize {
        var elements_count: usize = 1;
        for (dimensions.slice()) |dimension| {
            assert(dimension > 0);
            elements_count = std.math.mul(usize, elements_count, dimension) catch unreachable;
        }

        return elements_count;
    }
};

pub const SectionDefinition = struct {
    domain: SectionDomain,
    encoding: SectionEncoding,
    dimensions: TensorDimensions,
};

/// `sectionDefinition` is the canonical logical and physical schema for every
/// image section. Conversion, measurement, iteration, and inference views all
/// consume this one definition.
pub fn sectionDefinition(kind: SectionKind, specification: ModelSpecification) SectionDefinition {
    const encoder_width = specification.encoder_width;
    const decoder_width = specification.decoder_width;
    const encoder_ffn_width = specification.encoder_ffn_width;
    const decoder_ffn_width = specification.decoder_ffn_width;

    return switch (kind) {
        .decoder_embeddings_weight => vnniDefinition(.model, .init2(specification.vocabulary_tokens_count, decoder_width)),
        .decoder_layer_norm_beta, .decoder_layer_norm_gamma => float32Definition(.model, .init1(decoder_width)),
        .decoder_position_encodings => float32Definition(.model, .init2(specification.decoder_positions_count_max, decoder_width)),
        .encoder_convolution_1_bias => float32Definition(.model, .init1(encoder_width)),
        .encoder_convolution_1_weight => vnniDefinition(.model, .init3(encoder_width, specification.mel_bins_count, 3)),
        .encoder_convolution_2_bias => float32Definition(.model, .init1(encoder_width)),
        .encoder_convolution_2_weight => vnniDefinition(.model, .init3(encoder_width, encoder_width, 3)),
        .encoder_layer_norm_beta, .encoder_layer_norm_gamma => float32Definition(.model, .init1(encoder_width)),
        .encoder_position_encodings => float32Definition(.model, .init2(specification.encoder_positions_count_max, encoder_width)),

        .encoder_layer_self_attention_layer_norm_beta,
        .encoder_layer_self_attention_layer_norm_gamma,
        .encoder_layer_self_attention_output_bias,
        .encoder_layer_ffn_layer_norm_beta,
        .encoder_layer_ffn_layer_norm_gamma,
        .encoder_layer_ffn_contraction_bias,
        => float32Definition(.encoder_layer, .init1(encoder_width)),
        .encoder_layer_self_attention_query_key_value_bias => float32Definition(.encoder_layer, .init1(3 * encoder_width)),
        .encoder_layer_self_attention_query_key_value_weight => vnniDefinition(.encoder_layer, .init2(3 * encoder_width, encoder_width)),
        .encoder_layer_self_attention_output_weight => vnniDefinition(.encoder_layer, .init2(encoder_width, encoder_width)),
        .encoder_layer_ffn_expansion_bias => float32Definition(.encoder_layer, .init1(encoder_ffn_width)),
        .encoder_layer_ffn_expansion_weight => vnniDefinition(.encoder_layer, .init2(encoder_ffn_width, encoder_width)),
        .encoder_layer_ffn_contraction_weight => vnniDefinition(.encoder_layer, .init2(encoder_width, encoder_ffn_width)),

        .decoder_layer_self_attention_layer_norm_beta,
        .decoder_layer_self_attention_layer_norm_gamma,
        .decoder_layer_self_attention_output_bias,
        .decoder_layer_cross_attention_layer_norm_beta,
        .decoder_layer_cross_attention_layer_norm_gamma,
        .decoder_layer_cross_attention_query_bias,
        .decoder_layer_cross_attention_output_bias,
        .decoder_layer_ffn_layer_norm_beta,
        .decoder_layer_ffn_layer_norm_gamma,
        .decoder_layer_ffn_contraction_bias,
        => float32Definition(.decoder_layer, .init1(decoder_width)),
        .decoder_layer_self_attention_query_key_value_bias => float32Definition(.decoder_layer, .init1(3 * decoder_width)),
        .decoder_layer_self_attention_query_key_value_weight => vnniDefinition(.decoder_layer, .init2(3 * decoder_width, decoder_width)),
        .decoder_layer_self_attention_output_weight,
        .decoder_layer_cross_attention_query_weight,
        .decoder_layer_cross_attention_output_weight,
        => vnniDefinition(.decoder_layer, .init2(decoder_width, decoder_width)),
        .decoder_layer_cross_attention_key_value_bias => float32Definition(.decoder_layer, .init1(2 * decoder_width)),
        .decoder_layer_cross_attention_key_value_weight => vnniDefinition(.decoder_layer, .init2(2 * decoder_width, encoder_width)),
        .decoder_layer_ffn_expansion_bias => float32Definition(.decoder_layer, .init1(decoder_ffn_width)),
        .decoder_layer_ffn_expansion_weight => vnniDefinition(.decoder_layer, .init2(decoder_ffn_width, decoder_width)),
        .decoder_layer_ffn_contraction_weight => vnniDefinition(.decoder_layer, .init2(decoder_width, decoder_ffn_width)),
    };
}

fn float32Definition(domain: SectionDomain, dimensions: TensorDimensions) SectionDefinition {
    return .{ .domain = domain, .encoding = .float32_little_endian, .dimensions = dimensions };
}

fn vnniDefinition(domain: SectionDomain, dimensions: TensorDimensions) SectionDefinition {
    _ = vnni_weight.Layout.init(dimensions.values[0], @divExact(dimensions.elementsCount(), dimensions.values[0]));

    return .{ .domain = domain, .encoding = .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation, .dimensions = dimensions };
}

pub const QuantizedSectionLayout = struct {
    weight: vnni_weight.Layout,
    weights_size: usize,
    scales_offset: usize,
    scales_size: usize,
    compensation_offset: usize,
    section_size: usize,
};

pub fn quantizedSectionLayout(dimensions: TensorDimensions) QuantizedSectionLayout {
    assert(dimensions.count == 2 or dimensions.count == 3);

    const output_rows_count = dimensions.values[0];
    const input_values_count = @divExact(dimensions.elementsCount(), output_rows_count);
    const weight = vnni_weight.Layout.init(output_rows_count, input_values_count);
    const weights_size = weight.valuesCount();
    const scales_offset = std.mem.alignForward(usize, weights_size, alignment);
    const scales_size = std.math.mul(usize, output_rows_count, @sizeOf(f32)) catch unreachable;
    const scales_end = std.math.add(usize, scales_offset, scales_size) catch unreachable;
    const compensation_offset = std.mem.alignForward(usize, scales_end, alignment);
    const compensation_size = std.math.mul(usize, output_rows_count, @sizeOf(i32)) catch unreachable;
    const section_size = std.math.add(usize, compensation_offset, compensation_size) catch unreachable;

    return .{
        .weight = weight,
        .weights_size = weights_size,
        .scales_offset = scales_offset,
        .scales_size = scales_size,
        .compensation_offset = compensation_offset,
        .section_size = section_size,
    };
}

/// `validate` checks header validity, compatibility, and exact byte size for a
/// trusted target-2 image, then returns its model kind. Payload validity is a
/// caller precondition: tensor values and quantization metadata are not
/// inspected. It borrows `image` and performs no allocation.
pub fn validate(image: []align(alignment) const u8) LoadError!ModelKind {
    if (image.len < header_size or image.len > image_size_max) {
        return error.InvalidPackedImage;
    }
    if (!std.mem.eql(u8, image[magic_offset..format_version_offset], magic)) {
        return error.InvalidPackedImage;
    }

    const image_format_version = readInt(u32, image, format_version_offset);
    if (image_format_version != format_version) {
        return error.UnsupportedModelImageFormatVersion;
    }

    const kind = std.enums.fromInt(ModelKind, readInt(u16, image, model_kind_offset)) orelse {
        return error.UnsupportedModelKind;
    };

    const target = std.enums.fromInt(PackedTarget, readInt(u16, image, target_offset)) orelse {
        return error.UnsupportedPackedTarget;
    };
    if (target != current_target) {
        return error.UnsupportedPackedTarget;
    }

    const specification = kind.specification();
    const expected_image_size = measureImageSize(specification);
    if (readInt(u64, image, image_size_offset) != image.len or image.len != expected_image_size) {
        return error.InvalidPackedImage;
    }
    if (readInt(u32, image, flags_offset) != 0) {
        return error.InvalidPackedImage;
    }
    if (!std.mem.allEqual(u8, image[reserved_offset..header_size], 0)) {
        return error.InvalidPackedImage;
    }

    return kind;
}

pub fn floatSection(image: []align(alignment) const u8, specification: ModelSpecification, kind: SectionKind, layer_index: u16) []const f32 {
    const section = findSection(specification, kind, layer_index);
    assert(section.definition.encoding == .float32_little_endian);

    const values_count = @divExact(section.payload_size, @sizeOf(f32));
    const values: [*]const f32 = @ptrCast(@alignCast(image.ptr + section.payload_offset));

    return values[0..values_count];
}

pub fn quantizedSection(image: []align(alignment) const u8, specification: ModelSpecification, kind: SectionKind, layer_index: u16) QuantizedWeight {
    const section = findSection(specification, kind, layer_index);
    assert(section.definition.encoding == .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation);

    const layout = quantizedSectionLayout(section.definition.dimensions);
    const payload = image[section.payload_offset..][0..section.payload_size];
    const values: [*]const i8 = @ptrCast(payload.ptr);
    const scales: [*]const f32 = @ptrCast(@alignCast(payload.ptr + layout.scales_offset));
    const compensation: [*]const i32 = @ptrCast(@alignCast(payload.ptr + layout.compensation_offset));

    return .{
        .values = values[0..layout.weights_size],
        .scales = scales[0..layout.weight.output_rows_count],
        .compensation = compensation[0..layout.weight.output_rows_count],
        .input_values_count = layout.weight.input_values_count,
    };
}

/// `Builder` owns one incomplete target-2 image. `claimSection` returns each
/// final payload exactly once; the caller must initialize the complete payload
/// before calling `finish`.
pub const Builder = struct {
    allocator: Allocator,
    kind: ModelKind,
    specification: ModelSpecification,
    image: []align(alignment) u8,
    sections_count: usize,
    sections_written: SectionsWritten = .empty,

    pub fn init(allocator: Allocator, kind: ModelKind) Allocator.Error!Builder {
        // ── Measure And Allocate ──

        const specification = kind.specification();
        const image_size = measureImageSize(specification);
        assert(image_size <= image_size_max);
        const image = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), image_size);

        // ── Initialize Header And Padding ──
        //
        // PERFORMANCE: Conversion overwrites every payload value. Zero only
        // the header and alignment gaps so construction does not write the
        // complete image twice.

        var iterator = SectionIterator.init(specification);
        const payload_start_offset = iterator.payload_offset;
        assert(payload_start_offset == header_size);
        @memset(image[0..payload_start_offset], 0);

        var previous_payload_end = payload_start_offset;
        while (iterator.next()) |section| {
            assert(previous_payload_end <= section.payload_offset);
            @memset(image[previous_payload_end..section.payload_offset], 0);

            if (section.definition.encoding == .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation) {
                const layout = quantizedSectionLayout(section.definition.dimensions);
                const payload = image[section.payload_offset..][0..section.payload_size];
                @memset(payload[layout.weights_size..layout.scales_offset], 0);
                @memset(payload[layout.scales_offset + layout.scales_size .. layout.compensation_offset], 0);
            }

            previous_payload_end = section.payload_offset + section.payload_size;
        }
        assert(iterator.payload_offset == image.len);
        @memset(image[previous_payload_end..], 0);

        const sections_count = iterator.section_index;
        assert(sections_count > 0);
        assert(sections_count <= sections_count_max);

        return .{
            .allocator = allocator,
            .kind = kind,
            .specification = specification,
            .image = image,
            .sections_count = sections_count,
        };
    }

    pub fn claimSection(builder: *Builder, kind: SectionKind, layer_index: u16) WritableSection {
        const section = findSection(builder.specification, kind, layer_index);
        assert(section.section_index < builder.sections_count);
        assert(!builder.sections_written.isSet(section.section_index));

        builder.sections_written.set(section.section_index);
        return .{
            .definition = section.definition,
            .payload = builder.image[section.payload_offset..][0..section.payload_size],
        };
    }

    pub fn finish(builder: *Builder) []align(alignment) u8 {
        assert(builder.sections_written.count() == builder.sections_count);
        assert(builder.image.len <= image_size_max);

        @memcpy(builder.image[magic_offset..format_version_offset], magic);
        writeInt(u32, builder.image, format_version_offset, format_version);
        writeInt(u16, builder.image, model_kind_offset, @intFromEnum(builder.kind));
        writeInt(u16, builder.image, target_offset, @intFromEnum(current_target));
        writeInt(u64, builder.image, image_size_offset, builder.image.len);
        writeInt(u32, builder.image, flags_offset, 0);

        const image = builder.image;
        builder.* = undefined;
        return image;
    }

    pub fn deinit(builder: *Builder) void {
        builder.allocator.free(builder.image);
        builder.* = undefined;
    }
};

pub const WritableSection = struct {
    definition: SectionDefinition,
    payload: []u8,
};

pub fn writeFloat32(destination: []u8, byte_offset: usize, value: f32) void {
    writeInt(u32, destination, byte_offset, @as(u32, @bitCast(value)));
}

pub fn readFloat32(source: []const u8, byte_offset: usize) f32 {
    return @bitCast(readInt(u32, source, byte_offset));
}

pub fn writeInt(comptime Int: type, destination: []u8, byte_offset: usize, value: anytype) void {
    assert(byte_offset <= destination.len);
    assert(@sizeOf(Int) <= destination.len - byte_offset);

    std.mem.writeInt(Int, destination[byte_offset..][0..@sizeOf(Int)], @intCast(value), .little);
}

const magic = "VOICEDM\x00";
const header_size: usize = 128;
const magic_offset: usize = 0;
const format_version_offset: usize = 8;
const model_kind_offset: usize = 12;
const target_offset: usize = 14;
const image_size_offset: usize = 16;
const flags_offset: usize = 24;
const reserved_offset: usize = 28;
const image_size_max: usize = 512 * 1024 * 1024;

comptime {
    assert(magic.len == format_version_offset);
    assert(flags_offset + @sizeOf(u32) == reserved_offset);
    assert(reserved_offset + 100 == header_size);
}

const PackedTarget = enum(u16) {
    x86_64_avx_vnni_with_row_major_convolution = 1,
    x86_64_avx_vnni_o8_k4 = 2,
};

const current_target = PackedTarget.x86_64_avx_vnni_o8_k4;

const SectionLocation = struct {
    kind: SectionKind,
    definition: SectionDefinition,
    layer_index: u16,
    section_index: usize,
    payload_offset: usize,
    payload_size: usize,
};

const SectionIterator = struct {
    specification: ModelSpecification,
    kind_ordinal: usize = 0,
    layer_ordinal: usize = 0,
    section_index: usize = 0,
    payload_offset: usize = header_size,

    fn init(specification: ModelSpecification) SectionIterator {
        assert(header_size % alignment == 0);

        return .{ .specification = specification };
    }

    fn next(iterator: *SectionIterator) ?SectionLocation {
        const section_kinds = std.enums.values(SectionKind);
        while (iterator.kind_ordinal < section_kinds.len) {
            const kind = section_kinds[iterator.kind_ordinal];
            const definition = sectionDefinition(kind, iterator.specification);
            const instances_count = sectionInstancesCount(definition, iterator.specification);
            if (iterator.layer_ordinal == instances_count) {
                iterator.kind_ordinal += 1;
                iterator.layer_ordinal = 0;
                continue;
            }

            const layer_index = sectionLayerIndex(definition, iterator.layer_ordinal);
            const payload_size = sectionSize(definition);
            const location: SectionLocation = .{
                .kind = kind,
                .definition = definition,
                .layer_index = layer_index,
                .section_index = iterator.section_index,
                .payload_offset = iterator.payload_offset,
                .payload_size = payload_size,
            };

            const payload_end = std.math.add(usize, iterator.payload_offset, payload_size) catch unreachable;
            iterator.payload_offset = std.mem.alignForward(usize, payload_end, alignment);
            iterator.layer_ordinal += 1;
            iterator.section_index += 1;

            return location;
        }

        return null;
    }
};

const sections_count_max = std.enums.values(SectionKind).len * model_specification.layers_count_max;
const SectionsWritten = std.StaticBitSet(sections_count_max);

fn measureImageSize(specification: ModelSpecification) usize {
    var iterator = SectionIterator.init(specification);
    while (iterator.next() != null) {}

    return iterator.payload_offset;
}

fn findSection(specification: ModelSpecification, kind: SectionKind, layer_index: u16) SectionLocation {
    var iterator = SectionIterator.init(specification);
    while (iterator.next()) |section| {
        if (section.kind == kind and section.layer_index == layer_index) {
            return section;
        }
    }

    unreachable;
}

fn sectionInstancesCount(definition: SectionDefinition, specification: ModelSpecification) usize {
    return switch (definition.domain) {
        .model => 1,
        .encoder_layer => specification.encoder_layers_count,
        .decoder_layer => specification.decoder_layers_count,
    };
}

fn sectionLayerIndex(definition: SectionDefinition, layer_ordinal: usize) u16 {
    return switch (definition.domain) {
        .model => model_wide_layer_index,
        .encoder_layer, .decoder_layer => @intCast(layer_ordinal),
    };
}

fn sectionSize(definition: SectionDefinition) usize {
    return switch (definition.encoding) {
        .float32_little_endian => std.math.mul(usize, definition.dimensions.elementsCount(), @sizeOf(f32)) catch unreachable,
        .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation => quantizedSectionLayout(definition.dimensions).section_size,
    };
}

fn readInt(comptime Int: type, source: []const u8, byte_offset: usize) Int {
    assert(byte_offset <= source.len);
    assert(@sizeOf(Int) <= source.len - byte_offset);

    return std.mem.readInt(Int, source[byte_offset..][0..@sizeOf(Int)], .little);
}

test "legacy packed target is rejected" {
    var image: [header_size]u8 align(alignment) = @splat(0);
    @memcpy(image[magic_offset..format_version_offset], magic);
    writeInt(u32, &image, format_version_offset, format_version);
    writeInt(u16, &image, model_kind_offset, @intFromEnum(ModelKind.base_en));
    writeInt(u16, &image, target_offset, @intFromEnum(PackedTarget.x86_64_avx_vnni_with_row_major_convolution));

    try std.testing.expectError(error.UnsupportedPackedTarget, validate(&image));
}
