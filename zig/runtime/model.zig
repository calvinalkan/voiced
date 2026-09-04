//! This file owns the packed model image ABI shared by pristine conversion
//! and packed-image loading. Both paths derive payload order, shape, offset,
//! and size from `ModelKind`; the image stores no directory.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// `ModelKind` selects one supported English Whisper architecture. Its numeric
/// values are part of the packed-image ABI and must not be reordered or reused.
pub const ModelKind = enum(u16) {
    base_en = 1,
    small_en = 2,

    /// `specification` returns the fixed tensor dimensions for the model kind.
    pub fn specification(kind: ModelKind) ModelSpecification {
        const model_spec: ModelSpecification = switch (kind) {
            .base_en => .{
                .mel_bins_count = 80,
                .encoder_positions_count_max = 1500,
                .encoder_width = 512,
                .encoder_attention_heads_count = 8,
                .encoder_layers_count = 6,
                .encoder_ffn_width = 2048,
                .decoder_positions_count_max = 448,
                .decoder_width = 512,
                .decoder_attention_heads_count = 8,
                .decoder_layers_count = 6,
                .decoder_ffn_width = 2048,
                .vocabulary_tokens_count = 51_864,
            },
            .small_en => .{
                .mel_bins_count = 80,
                .encoder_positions_count_max = 1500,
                .encoder_width = 768,
                .encoder_attention_heads_count = 12,
                .encoder_layers_count = 12,
                .encoder_ffn_width = 3072,
                .decoder_positions_count_max = 448,
                .decoder_width = 768,
                .decoder_attention_heads_count = 12,
                .decoder_layers_count = 12,
                .decoder_ffn_width = 3072,
                .vocabulary_tokens_count = 51_864,
            },
        };

        assert(model_spec.mel_bins_count > 0);
        assert(model_spec.encoder_positions_count_max > 0);
        assert(model_spec.encoder_width > 0);
        assert(model_spec.encoder_attention_heads_count > 0);
        assert(model_spec.encoder_layers_count > 0);
        assert(model_spec.encoder_layers_count <= 12);
        assert(model_spec.encoder_ffn_width > 0);
        assert(model_spec.decoder_positions_count_max > 0);
        assert(model_spec.decoder_width > 0);
        assert(model_spec.decoder_attention_heads_count > 0);
        assert(model_spec.decoder_layers_count > 0);
        assert(model_spec.decoder_layers_count <= 12);
        assert(model_spec.decoder_ffn_width > 0);
        assert(model_spec.vocabulary_tokens_count > 0);
        assert(model_spec.encoder_width % model_spec.encoder_attention_heads_count == 0);
        assert(model_spec.decoder_width % model_spec.decoder_attention_heads_count == 0);
        assert(model_spec.encoder_ffn_width == 4 * model_spec.encoder_width);
        assert(model_spec.decoder_ffn_width == 4 * model_spec.decoder_width);
        assert(model_spec.encoder_width % 8 == 0);
        assert(model_spec.decoder_width % 8 == 0);
        assert(model_spec.encoder_ffn_width % 8 == 0);
        assert(model_spec.decoder_ffn_width % 8 == 0);
        assert(model_spec.vocabulary_tokens_count % 8 == 0);

        return model_spec;
    }
};

/// `ModelSpecification` describes the fixed tensor dimensions of one
/// `ModelKind`. They determine the packed image's section layout and size.
/// Counts use logical tensor elements, not bytes or allocated capacity.
pub const ModelSpecification = struct {
    mel_bins_count: usize,

    encoder_positions_count_max: usize,
    encoder_width: usize,
    encoder_attention_heads_count: usize,
    encoder_layers_count: usize,
    encoder_ffn_width: usize,

    decoder_positions_count_max: usize,
    decoder_width: usize,
    decoder_attention_heads_count: usize,
    decoder_layers_count: usize,
    decoder_ffn_width: usize,

    vocabulary_tokens_count: usize,
};

/// `ModelLoadError` reports malformed or incompatible model input and
/// allocation failure. A failed load returns no partially initialized `Model`.
pub const ModelLoadError = error{
    OutOfMemory,
    InvalidPackedImage,
    InvalidPristineWeights,
    UnsupportedModelImageFormatVersion,
    UnsupportedModelKind,
    UnsupportedPackedTarget,
};

/// `Model` is a handle to one inference-ready Whisper model. A model loaded
/// from a packed image borrows that image; a model loaded from pristine
/// weights owns its generated packed image. Do not mutate or copy the value;
/// call `deinit` exactly once.
///
/// The allocator passed to `fromPristineWeights` must remain valid until
/// `deinit` returns. A borrowed packed image must remain mapped and immutable
/// until `deinit` returns.
pub const Model = struct {
    kind: ModelKind,
    specification: ModelSpecification,
    backing: Backing,

    /// `fromPackedImage` validates an aligned native image and constructs model
    /// views directly over its bytes. It performs no allocation or copying and
    /// does not take ownership of `image`.
    ///
    /// It rejects a malformed image with `error.InvalidPackedImage`. It rejects
    /// a well-formed image with an unsupported format version, model kind, or
    /// packed target with the corresponding `Unsupported*` error.
    pub fn fromPackedImage(image: []align(packed_model_image_alignment) const u8) ModelLoadError!Model {
        const kind = try validatePackedImage(image);

        return .{
            .kind = kind,
            .specification = kind.specification(),
            .backing = .{ .borrowed = image },
        };
    }

    /// `fromPristineWeights` converts the unmodified CTranslate2 `model.bin`
    /// from the matching `Systran/faster-whisper-base.en` or
    /// `Systran/faster-whisper-small.en` Hugging Face repository. These are not
    /// OpenAI PyTorch checkpoints or whisper.cpp GGML files.
    ///
    /// `zig build setup-models` downloads a pinned revision, verifies its hash,
    /// and installs it below `$XDG_DATA_HOME/voiced/models`, or below
    /// `~/.local/share/voiced/models` when `XDG_DATA_HOME` is unset.
    ///
    /// The function rejects any input that is not the canonical CTranslate2
    /// version-6 serialization for `kind`—including truncated, reordered, or
    /// dimension-mismatched tensors—with `error.InvalidPristineWeights`. It
    /// allocates one packed image through `allocator`; on failure it frees that
    /// allocation, and on success the returned `Model` owns it. The model
    /// retains no reference to `weights`, which the caller may free after this
    /// function returns.
    pub fn fromPristineWeights(allocator: Allocator, kind: ModelKind, weights: []const u8) ModelLoadError!Model {
        // ── Read And Validate The Source Header ──

        if (weights.len > pristine_weights_size_max) {
            return error.InvalidPristineWeights;
        }

        var reader: CT2WeightsReader = .{ .bytes = weights };

        const binary_version = try reader.readInt(u32);
        if (binary_version != ctranslate2_binary_version) {
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

        const model_specification = kind.specification();
        const tensors_count = try reader.readInt(u32);
        const expected_tensors_count = countPristineTensors(model_specification);
        if (tensors_count != expected_tensors_count) {
            return error.InvalidPristineWeights;
        }

        // ── Convert Decoder Weights ──
        //
        // The explicit sequence rejects missing, duplicate, reordered, or
        // unknown records while writing each accepted tensor directly into its
        // final image section.

        var builder = try PackedImageBuilder.init(allocator, kind);
        errdefer builder.deinit();

        for (decoder_pristine_tensors_before_layers) |tensor_plan| {
            try convertPristineTensor(&reader, &builder, model_specification, model_wide_layer_index, tensor_plan);
        }

        for (0..model_specification.decoder_layers_count) |layer_ordinal| {
            const layer_index = pristineLayerIndexFromOrdinal(layer_ordinal, model_specification.decoder_layers_count);
            for (decoder_layer_pristine_tensors) |tensor_plan| {
                try convertPristineLayerTensor(&reader, &builder, model_specification, "decoder", layer_index, tensor_plan);
            }
        }

        for (decoder_pristine_tensors_after_layers) |tensor_plan| {
            try convertPristineTensor(&reader, &builder, model_specification, model_wide_layer_index, tensor_plan);
        }

        // ── Convert Encoder Weights ──

        for (encoder_pristine_tensors_before_layers) |tensor_plan| {
            try convertPristineTensor(&reader, &builder, model_specification, model_wide_layer_index, tensor_plan);
        }

        for (0..model_specification.encoder_layers_count) |layer_ordinal| {
            const layer_index = pristineLayerIndexFromOrdinal(layer_ordinal, model_specification.encoder_layers_count);
            for (encoder_layer_pristine_tensors) |tensor_plan| {
                try convertPristineLayerTensor(&reader, &builder, model_specification, "encoder", layer_index, tensor_plan);
            }
        }

        for (encoder_pristine_tensors_after_layers) |tensor_plan| {
            try convertPristineTensor(&reader, &builder, model_specification, model_wide_layer_index, tensor_plan);
        }

        // ── Validate The Tied-Embedding Alias ──
        //
        // Whisper uses its decoder embedding matrix again for vocabulary
        // projection. The packed image stores one quantized and VNNI-packed
        // matrix for both operations.

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

        return builder.intoModel();
    }

    /// `packedImage` returns the complete relocatable native image backing the
    /// model. The returned bytes remain valid until `deinit`; callers may write
    /// them verbatim to persistent storage and later pass a mapped copy to
    /// `fromPackedImage`.
    pub fn packedImage(model: *const Model) []align(packed_model_image_alignment) const u8 {
        const image = model.backing.packedImage();
        assert(image.len > 0);

        return image;
    }

    /// `deinit` releases an image created by `fromPristineWeights` and leaves a
    /// borrowed image untouched. It invalidates the model in both cases.
    pub fn deinit(model: *Model) void {
        model.backing.deinit();
        model.* = undefined;
    }

    /// `inferenceWeights` constructs typed, zero-copy views over every packed
    /// tensor. The returned views remain valid while the model image remains
    /// mapped and immutable.
    pub fn inferenceWeights(model: *const Model) InferenceWeights {
        var encoder_layers: [model_layers_count_max]EncoderLayerWeights = undefined;
        for (encoder_layers[0..model.specification.encoder_layers_count], 0..) |*layer, layer_index| {
            layer.* = .{
                .self_attention_layer_norm_beta = model.floatSection(.encoder_layer_self_attention_layer_norm_beta, @intCast(layer_index)),
                .self_attention_layer_norm_gamma = model.floatSection(.encoder_layer_self_attention_layer_norm_gamma, @intCast(layer_index)),
                .self_attention_query_key_value_bias = model.floatSection(.encoder_layer_self_attention_query_key_value_bias, @intCast(layer_index)),
                .self_attention_query_key_value_weight = model.quantizedSection(.encoder_layer_self_attention_query_key_value_weight, @intCast(layer_index)),
                .self_attention_output_bias = model.floatSection(.encoder_layer_self_attention_output_bias, @intCast(layer_index)),
                .self_attention_output_weight = model.quantizedSection(.encoder_layer_self_attention_output_weight, @intCast(layer_index)),
                .ffn_layer_norm_beta = model.floatSection(.encoder_layer_ffn_layer_norm_beta, @intCast(layer_index)),
                .ffn_layer_norm_gamma = model.floatSection(.encoder_layer_ffn_layer_norm_gamma, @intCast(layer_index)),
                .ffn_expansion_bias = model.floatSection(.encoder_layer_ffn_expansion_bias, @intCast(layer_index)),
                .ffn_expansion_weight = model.quantizedSection(.encoder_layer_ffn_expansion_weight, @intCast(layer_index)),
                .ffn_contraction_bias = model.floatSection(.encoder_layer_ffn_contraction_bias, @intCast(layer_index)),
                .ffn_contraction_weight = model.quantizedSection(.encoder_layer_ffn_contraction_weight, @intCast(layer_index)),
            };
        }

        var decoder_layers: [model_layers_count_max]DecoderLayerWeights = undefined;
        for (decoder_layers[0..model.specification.decoder_layers_count], 0..) |*layer, layer_index| {
            layer.* = .{
                .self_attention_layer_norm_beta = model.floatSection(.decoder_layer_self_attention_layer_norm_beta, @intCast(layer_index)),
                .self_attention_layer_norm_gamma = model.floatSection(.decoder_layer_self_attention_layer_norm_gamma, @intCast(layer_index)),
                .self_attention_query_key_value_bias = model.floatSection(.decoder_layer_self_attention_query_key_value_bias, @intCast(layer_index)),
                .self_attention_query_key_value_weight = model.quantizedSection(.decoder_layer_self_attention_query_key_value_weight, @intCast(layer_index)),
                .self_attention_output_bias = model.floatSection(.decoder_layer_self_attention_output_bias, @intCast(layer_index)),
                .self_attention_output_weight = model.quantizedSection(.decoder_layer_self_attention_output_weight, @intCast(layer_index)),
                .cross_attention_layer_norm_beta = model.floatSection(.decoder_layer_cross_attention_layer_norm_beta, @intCast(layer_index)),
                .cross_attention_layer_norm_gamma = model.floatSection(.decoder_layer_cross_attention_layer_norm_gamma, @intCast(layer_index)),
                .cross_attention_query_bias = model.floatSection(.decoder_layer_cross_attention_query_bias, @intCast(layer_index)),
                .cross_attention_query_weight = model.quantizedSection(.decoder_layer_cross_attention_query_weight, @intCast(layer_index)),
                .cross_attention_key_value_bias = model.floatSection(.decoder_layer_cross_attention_key_value_bias, @intCast(layer_index)),
                .cross_attention_key_value_weight = model.quantizedSection(.decoder_layer_cross_attention_key_value_weight, @intCast(layer_index)),
                .cross_attention_output_bias = model.floatSection(.decoder_layer_cross_attention_output_bias, @intCast(layer_index)),
                .cross_attention_output_weight = model.quantizedSection(.decoder_layer_cross_attention_output_weight, @intCast(layer_index)),
                .ffn_layer_norm_beta = model.floatSection(.decoder_layer_ffn_layer_norm_beta, @intCast(layer_index)),
                .ffn_layer_norm_gamma = model.floatSection(.decoder_layer_ffn_layer_norm_gamma, @intCast(layer_index)),
                .ffn_expansion_bias = model.floatSection(.decoder_layer_ffn_expansion_bias, @intCast(layer_index)),
                .ffn_expansion_weight = model.quantizedSection(.decoder_layer_ffn_expansion_weight, @intCast(layer_index)),
                .ffn_contraction_bias = model.floatSection(.decoder_layer_ffn_contraction_bias, @intCast(layer_index)),
                .ffn_contraction_weight = model.quantizedSection(.decoder_layer_ffn_contraction_weight, @intCast(layer_index)),
            };
        }

        return .{
            .decoder_embeddings_weight = model.quantizedSection(.decoder_embeddings_weight, model_wide_layer_index),
            .decoder_layer_norm_beta = model.floatSection(.decoder_layer_norm_beta, model_wide_layer_index),
            .decoder_layer_norm_gamma = model.floatSection(.decoder_layer_norm_gamma, model_wide_layer_index),
            .decoder_position_encodings = model.floatSection(.decoder_position_encodings, model_wide_layer_index),
            .encoder_convolution_1_bias = model.floatSection(.encoder_convolution_1_bias, model_wide_layer_index),
            .encoder_convolution_1_weight = model.quantizedSection(.encoder_convolution_1_weight, model_wide_layer_index),
            .encoder_convolution_2_bias = model.floatSection(.encoder_convolution_2_bias, model_wide_layer_index),
            .encoder_convolution_2_weight = model.quantizedSection(.encoder_convolution_2_weight, model_wide_layer_index),
            .encoder_layer_norm_beta = model.floatSection(.encoder_layer_norm_beta, model_wide_layer_index),
            .encoder_layer_norm_gamma = model.floatSection(.encoder_layer_norm_gamma, model_wide_layer_index),
            .encoder_position_encodings = model.floatSection(.encoder_position_encodings, model_wide_layer_index),
            .encoder_layers = encoder_layers,
            .decoder_layers = decoder_layers,
        };
    }

    fn floatSection(model: *const Model, kind: PackedSectionKind, layer_index: u16) []const f32 {
        assert(packedSectionEncoding(kind) == .float32_little_endian);

        const section = findPackedSection(model.specification, kind, layer_index);
        const image = model.backing.packedImage();
        const values_count = @divExact(section.payload_size, @sizeOf(f32));
        const values: [*]const f32 = @ptrCast(@alignCast(image.ptr + section.payload_offset));

        return values[0..values_count];
    }

    fn quantizedSection(model: *const Model, kind: PackedSectionKind, layer_index: u16) QuantizedWeight {
        assert(packedSectionEncoding(kind) != .float32_little_endian);

        const section = findPackedSection(model.specification, kind, layer_index);
        const dimensions = packedSectionDimensions(kind, model.specification);
        const layout = quantizedWeightLayout(dimensions);
        const image = model.backing.packedImage();
        const payload = image[section.payload_offset..][0..section.payload_size];
        const values: [*]const i8 = @ptrCast(payload.ptr);
        const scales: [*]const f32 = @ptrCast(@alignCast(payload.ptr + layout.scales_offset));
        const compensation: [*]const i32 = @ptrCast(@alignCast(payload.ptr + layout.compensation_offset));

        return .{
            .values = values[0..layout.weights_size],
            .scales = scales[0..layout.output_rows_count],
            .compensation = compensation[0..layout.output_rows_count],
            .output_rows_count = layout.output_rows_count,
            .input_values_count = layout.input_values_count,
            .encoding = switch (packedSectionEncoding(kind)) {
                .int8_row_major_with_float32_scales_and_int32_compensation => .row_major,
                .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation => .vnni_o8_k4,
                .float32_little_endian => unreachable,
            },
        };
    }
};

/// `QuantizedWeight` exposes one packed INT8 matrix and the scale and
/// compensation values needed to recover its Float32 result. The matrix views
/// borrow the model image.
pub const QuantizedWeight = struct {
    values: []const i8,
    scales: []const f32,
    compensation: []const i32,
    output_rows_count: usize,
    input_values_count: usize,
    encoding: enum { row_major, vnni_o8_k4 },
};

pub const EncoderLayerWeights = struct {
    self_attention_layer_norm_beta: []const f32,
    self_attention_layer_norm_gamma: []const f32,
    self_attention_query_key_value_bias: []const f32,
    self_attention_query_key_value_weight: QuantizedWeight,
    self_attention_output_bias: []const f32,
    self_attention_output_weight: QuantizedWeight,
    ffn_layer_norm_beta: []const f32,
    ffn_layer_norm_gamma: []const f32,
    ffn_expansion_bias: []const f32,
    ffn_expansion_weight: QuantizedWeight,
    ffn_contraction_bias: []const f32,
    ffn_contraction_weight: QuantizedWeight,
};

pub const DecoderLayerWeights = struct {
    self_attention_layer_norm_beta: []const f32,
    self_attention_layer_norm_gamma: []const f32,
    self_attention_query_key_value_bias: []const f32,
    self_attention_query_key_value_weight: QuantizedWeight,
    self_attention_output_bias: []const f32,
    self_attention_output_weight: QuantizedWeight,
    cross_attention_layer_norm_beta: []const f32,
    cross_attention_layer_norm_gamma: []const f32,
    cross_attention_query_bias: []const f32,
    cross_attention_query_weight: QuantizedWeight,
    cross_attention_key_value_bias: []const f32,
    cross_attention_key_value_weight: QuantizedWeight,
    cross_attention_output_bias: []const f32,
    cross_attention_output_weight: QuantizedWeight,
    ffn_layer_norm_beta: []const f32,
    ffn_layer_norm_gamma: []const f32,
    ffn_expansion_bias: []const f32,
    ffn_expansion_weight: QuantizedWeight,
    ffn_contraction_bias: []const f32,
    ffn_contraction_weight: QuantizedWeight,
};

/// `InferenceWeights` indexes every model tensor without copying the packed
/// image. Only the layer prefixes selected by the model specification are
/// initialized.
pub const InferenceWeights = struct {
    decoder_embeddings_weight: QuantizedWeight,
    decoder_layer_norm_beta: []const f32,
    decoder_layer_norm_gamma: []const f32,
    decoder_position_encodings: []const f32,
    encoder_convolution_1_bias: []const f32,
    encoder_convolution_1_weight: QuantizedWeight,
    encoder_convolution_2_bias: []const f32,
    encoder_convolution_2_weight: QuantizedWeight,
    encoder_layer_norm_beta: []const f32,
    encoder_layer_norm_gamma: []const f32,
    encoder_position_encodings: []const f32,
    encoder_layers: [model_layers_count_max]EncoderLayerWeights,
    decoder_layers: [model_layers_count_max]DecoderLayerWeights,
};

const model_layers_count_max: usize = 12;

/// Alignment of every packed image. Buffers filled from persistent storage
/// must satisfy it before `fromPackedImage` accepts them; `packedImage` always
/// returns bytes with this alignment.
pub const packed_model_image_alignment = 64;

const Backing = union(enum) {
    borrowed: []align(packed_model_image_alignment) const u8,
    owned: struct {
        allocator: Allocator,
        image: []align(packed_model_image_alignment) u8,
    },

    fn packedImage(backing: *const Backing) []align(packed_model_image_alignment) const u8 {
        return switch (backing.*) {
            .borrowed => |image| image,
            .owned => |owned| owned.image,
        };
    }

    fn deinit(backing: *Backing) void {
        switch (backing.*) {
            .borrowed => {},
            .owned => |owned| owned.allocator.free(owned.image),
        }

        backing.* = undefined;
    }
};

// ─── CTranslate2 Weight Reader ─────────────────────────────────────────────
//
// The reader recognizes the version-6 `model.bin` emitted by CTranslate2. It
// borrows names and tensor payloads from the caller's image, copies only the
// small dimension arrays, and rejects every read that crosses the image.
// CTranslate2 calls every stored tensor or scalar a variable; this reader uses
// tensor throughout because a scalar is simply a rank-zero tensor.

const CT2WeightsReader = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn readTensor(reader: *CT2WeightsReader) ModelLoadError!CT2Tensor {
        const name = try reader.readString();
        if (name.len == 0) {
            return error.InvalidPristineWeights;
        }

        const dimensions_count: usize = @intCast(try reader.readInt(u8));
        if (dimensions_count > CT2Tensor.dimensions_count_max) {
            return error.InvalidPristineWeights;
        }

        var dimensions: [CT2Tensor.dimensions_count_max]usize = @splat(0);
        var elements_count: usize = 1;

        for (0..dimensions_count) |dimension_index| {
            const dimension: usize = @intCast(try reader.readInt(u32));
            dimensions[dimension_index] = dimension;

            elements_count = std.math.mul(usize, elements_count, dimension) catch {
                return error.InvalidPristineWeights;
            };
        }

        const data_type = std.enums.fromInt(CT2DataType, try reader.readInt(u8)) orelse {
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

        return .{ .name = name, .dimensions = dimensions, .dimensions_count = dimensions_count, .data_type = data_type, .data = data };
    }

    fn readString(reader: *CT2WeightsReader) ModelLoadError![]const u8 {
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

    fn readInt(reader: *CT2WeightsReader, comptime Int: type) ModelLoadError!Int {
        const bytes = try reader.readBytes(@sizeOf(Int));

        return std.mem.readInt(Int, bytes[0..@sizeOf(Int)], .little);
    }

    fn readBytes(reader: *CT2WeightsReader, size: usize) ModelLoadError![]const u8 {
        assert(reader.offset <= reader.bytes.len);

        if (size > reader.bytes.len - reader.offset) {
            return error.InvalidPristineWeights;
        }

        const bytes = reader.bytes[reader.offset..][0..size];
        reader.offset += size;

        return bytes;
    }
};

const CT2Tensor = struct {
    const dimensions_count_max = 3;

    name: []const u8,
    dimensions: [dimensions_count_max]usize,
    dimensions_count: usize,
    data_type: CT2DataType,
    data: []const u8,

    fn hasDimensions(tensor: *const CT2Tensor, expected_dimensions: []const usize) bool {
        if (tensor.dimensions_count != expected_dimensions.len) {
            return false;
        }

        return std.mem.eql(usize, tensor.dimensions[0..tensor.dimensions_count], expected_dimensions);
    }

    fn hasScalarValue(tensor: *const CT2Tensor, comptime Int: type, expected_value: Int) bool {
        const expected_data_type: CT2DataType = switch (Int) {
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
const CT2DataType = enum(u8) {
    float32 = 0,
    int8 = 1,
    int16 = 2,
    int32 = 3,
    float16 = 4,
    bfloat16 = 5,

    fn elementSize(data_type: CT2DataType) usize {
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
// The converter accepts the canonical, lexicographically sorted variable
// order emitted by CTranslate2's version-6 serializer. Each plan binds one
// source name to either an exact metadata value or one packed-image section;
// no source tensor is collected in an intermediate model representation.

const ctranslate2_binary_version: u32 = 6;
const whisper_specification_revision: u32 = 3;
const pristine_weights_size_max: usize = 512 * 1024 * 1024;
const pristine_tensor_name_size_max: usize = 128;
const pristine_metadata_tensors_count: usize = 9;

const PristineTensorPlan = struct {
    name: []const u8,
    handling: union(enum) {
        packed_section: PackedSectionKind,
        scalar_i8: i8,
        scalar_i16: i16,
        encoder_attention_heads_count,
        decoder_attention_heads_count,
    },
};

const decoder_pristine_tensors_before_layers = [_]PristineTensorPlan{
    .{ .name = "decoder/activation", .handling = .{ .scalar_i8 = 3 } },
    .{ .name = "decoder/alibi", .handling = .{ .scalar_i8 = 0 } },
    .{ .name = "decoder/alignment_heads", .handling = .{ .scalar_i16 = 1 } },
    .{ .name = "decoder/alignment_layer", .handling = .{ .scalar_i16 = -1 } },
    .{ .name = "decoder/embeddings/weight", .handling = .{ .packed_section = .decoder_embeddings_weight } },
};

const decoder_layer_pristine_tensors = [_]PristineTensorPlan{
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

const decoder_pristine_tensors_after_layers = [_]PristineTensorPlan{
    .{ .name = "decoder/layer_norm/beta", .handling = .{ .packed_section = .decoder_layer_norm_beta } },
    .{ .name = "decoder/layer_norm/gamma", .handling = .{ .packed_section = .decoder_layer_norm_gamma } },
    .{ .name = "decoder/num_heads", .handling = .decoder_attention_heads_count },
    .{ .name = "decoder/position_encodings/encodings", .handling = .{ .packed_section = .decoder_position_encodings } },
    .{ .name = "decoder/pre_norm", .handling = .{ .scalar_i8 = 1 } },
    .{ .name = "decoder/scale_embeddings", .handling = .{ .scalar_i8 = 0 } },
    .{ .name = "decoder/start_from_zero_embedding", .handling = .{ .scalar_i8 = 0 } },
};

const encoder_pristine_tensors_before_layers = [_]PristineTensorPlan{
    .{ .name = "encoder/conv1/bias", .handling = .{ .packed_section = .encoder_convolution_1_bias } },
    .{ .name = "encoder/conv1/weight", .handling = .{ .packed_section = .encoder_convolution_1_weight } },
    .{ .name = "encoder/conv2/bias", .handling = .{ .packed_section = .encoder_convolution_2_bias } },
    .{ .name = "encoder/conv2/weight", .handling = .{ .packed_section = .encoder_convolution_2_weight } },
};

const encoder_layer_pristine_tensors = [_]PristineTensorPlan{
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

const encoder_pristine_tensors_after_layers = [_]PristineTensorPlan{
    .{ .name = "encoder/layer_norm/beta", .handling = .{ .packed_section = .encoder_layer_norm_beta } },
    .{ .name = "encoder/layer_norm/gamma", .handling = .{ .packed_section = .encoder_layer_norm_gamma } },
    .{ .name = "encoder/num_heads", .handling = .encoder_attention_heads_count },
    .{ .name = "encoder/position_encodings/encodings", .handling = .{ .packed_section = .encoder_position_encodings } },
};

comptime {
    assert(decoder_pristine_tensors_before_layers.len == 5);
    assert(decoder_layer_pristine_tensors.len == 20);
    assert(decoder_pristine_tensors_after_layers.len == 7);
    assert(encoder_pristine_tensors_before_layers.len == 4);
    assert(encoder_layer_pristine_tensors.len == 12);
    assert(encoder_pristine_tensors_after_layers.len == 4);
}

fn countPristineTensors(model_specification: ModelSpecification) u32 {
    const tensors_count = pristine_metadata_tensors_count + countPackedSections(model_specification);
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

fn convertPristineLayerTensor(reader: *CT2WeightsReader, builder: *PackedImageBuilder, model_specification: ModelSpecification, scope: []const u8, layer_index: u16, tensor_plan: PristineTensorPlan) ModelLoadError!void {
    assert(scope.len > 0);
    assert(tensor_plan.name.len > 0);

    var tensor_name_buffer: [pristine_tensor_name_size_max]u8 = undefined;
    const tensor_name = std.fmt.bufPrint(&tensor_name_buffer, "{s}/layer_{d}/{s}", .{ scope, layer_index, tensor_plan.name }) catch unreachable;
    const complete_tensor_plan: PristineTensorPlan = .{ .name = tensor_name, .handling = tensor_plan.handling };

    try convertPristineTensor(reader, builder, model_specification, layer_index, complete_tensor_plan);
}

fn convertPristineTensor(reader: *CT2WeightsReader, builder: *PackedImageBuilder, model_specification: ModelSpecification, layer_index: u16, tensor_plan: PristineTensorPlan) ModelLoadError!void {
    const tensor = try reader.readTensor();
    if (!std.mem.eql(u8, tensor.name, tensor_plan.name)) {
        return error.InvalidPristineWeights;
    }

    switch (tensor_plan.handling) {
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
            const expected_value: i16 = @intCast(model_specification.encoder_attention_heads_count);
            if (!tensor.hasScalarValue(i16, expected_value)) {
                return error.InvalidPristineWeights;
            }
        },
        .decoder_attention_heads_count => {
            const expected_value: i16 = @intCast(model_specification.decoder_attention_heads_count);
            if (!tensor.hasScalarValue(i16, expected_value)) {
                return error.InvalidPristineWeights;
            }
        },
        .packed_section => |section_kind| {
            const expected_dimensions = packedSectionDimensions(section_kind, model_specification);
            if (tensor.data_type != .float16) {
                return error.InvalidPristineWeights;
            }
            if (!tensor.hasDimensions(expected_dimensions.slice())) {
                return error.InvalidPristineWeights;
            }

            try builder.writeTensor(section_kind, layer_index, tensor);
        },
    }
}

// ─── Packed Model Image Format ─────────────────────────────────────────────
//
// Version 1 is a little-endian, relocatable, uncompressed image. It stores no
// tensor names, section directory, process pointers, `usize` values, or
// serialized Zig structs. The format version, `ModelKind`, and packed target
// determine every payload's order, logical shape, physical encoding, offset,
// and size. Converting the same source with the same packer is deterministic.
//
// Header, 128 bytes:
//
//   0x00  [8]u8    magic: "VOICEDM\0"
//   0x08  u32      image format version
//   0x0c  u16      ModelKind
//   0x0e  u16      packed target (1 = x86-64 AVX-VNNI)
//   0x10  u64      complete image size in bytes
//   0x18  u32      flags; zero in version 1
//   0x1c  [100]u8  reserved; all zero
//
// Payloads begin immediately after the header and appear in canonical
// section-kind and layer-index order. Every payload begins at a 64-byte
// boundary. The writer zeros alignment padding for deterministic output; the
// image stores no payload offsets or sizes, so the loader derives them from
// `ModelKind` and cross-checks the header's recorded total size.
//
// Each quantized weight section contains the INT8 matrix, 64-byte-aligned
// per-output FP32 scales, and 64-byte-aligned INT32 unsigned-activation
// compensation. Compensation is −128 × the row's quantized-weight sum: VNNI
// kernels add 128 to signed activations to form unsigned operands, and this
// term removes the bias contribution from each output. Convolution weights
// retain row-major order. Dense and tied embedding weights use the
// `vnni_u8s8_o8_k4` order: each 32-byte group contains four input-depth values
// for each of eight output rows. Packing permutes the INT8 matrix bytes but
// does not increase their count.
//
// The packed target and format version define the physical encoding implied by
// each section kind. A faster kernel may consume the same encoding without a
// format change. Changing output blocking, depth grouping, signedness, scale or
// compensation semantics, or element width requires a new packed target or
// image format version.

/// `packed_model_image_format_version` identifies the header and canonical
/// payload sequence of the packed model image format. The packed target selects
/// the physical encoding implied by each section in that sequence.
pub const packed_model_image_format_version: u32 = 1;

const packed_image_magic = "VOICEDM\x00";
const packed_image_header_size: usize = 128;
const packed_image_magic_offset: usize = 0;
const packed_image_format_version_offset: usize = 8;
const packed_image_model_kind_offset: usize = 12;
const packed_image_target_offset: usize = 14;
const packed_image_size_offset: usize = 16;
const packed_image_flags_offset: usize = 24;
const packed_image_reserved_offset: usize = 28;
const packed_image_size_max: usize = 512 * 1024 * 1024;
const packed_weight_output_block_rows_count: usize = 8;
const packed_weight_depth_group_values_count: usize = 4;
const model_wide_layer_index: u16 = std.math.maxInt(u16);

comptime {
    assert(packed_image_magic.len == packed_image_format_version_offset);
    assert(packed_image_flags_offset + @sizeOf(u32) == packed_image_reserved_offset);
    assert(packed_image_reserved_offset + 100 == packed_image_header_size);
}

const PackedTarget = enum(u16) {
    x86_64_avx_vnni = 1,
};

// Declaration order defines the packed-image payload order; numeric ranges
// place each kind in the model-wide, encoder-layer, or decoder-layer domain.
// Reordering declarations or moving a value across a domain boundary changes
// the image and requires a format-version change.
const PackedSectionKind = enum(u16) {
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
    assert(std.enums.values(PackedSectionKind).len == 43);
    assert(@intFromEnum(PackedSectionKind.decoder_embeddings_weight) == 1);
    assert(@intFromEnum(PackedSectionKind.encoder_position_encodings) == 11);
    assert(@intFromEnum(PackedSectionKind.encoder_layer_ffn_contraction_weight) == 23);
    assert(@intFromEnum(PackedSectionKind.decoder_layer_ffn_contraction_weight) == 43);
}

const PackedSectionEncoding = enum(u16) {
    float32_little_endian = 1,
    int8_row_major_with_float32_scales_and_int32_compensation = 2,
    vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation = 3,
};

const PackedSectionDomain = enum {
    model,
    encoder_layer,
    decoder_layer,
};

const PackedTensorDimensions = struct {
    values: [CT2Tensor.dimensions_count_max]usize,
    count: usize,

    fn slice(dimensions: *const PackedTensorDimensions) []const usize {
        assert(dimensions.count <= dimensions.values.len);

        return dimensions.values[0..dimensions.count];
    }

    fn elementsCount(dimensions: PackedTensorDimensions) usize {
        var elements_count: usize = 1;
        for (dimensions.slice()) |dimension| {
            assert(dimension > 0);
            elements_count = std.math.mul(usize, elements_count, dimension) catch unreachable;
        }

        return elements_count;
    }
};

const QuantizedWeightLayout = struct {
    weights_size: usize,
    scales_offset: usize,
    scales_size: usize,
    compensation_offset: usize,
    compensation_size: usize,
    section_size: usize,
    output_rows_count: usize,
    input_values_count: usize,
};

const PackedSectionLocation = struct {
    kind: PackedSectionKind,
    layer_index: u16,
    section_index: usize,
    payload_offset: usize,
    payload_size: usize,
};

const PackedSectionIterator = struct {
    model_specification: ModelSpecification,
    kind_ordinal: usize = 0,
    layer_ordinal: usize = 0,
    section_index: usize = 0,
    payload_offset: usize = packed_image_header_size,

    fn init(model_specification: ModelSpecification) PackedSectionIterator {
        assert(packed_image_header_size % packed_model_image_alignment == 0);

        return .{ .model_specification = model_specification };
    }

    fn next(iterator: *PackedSectionIterator) ?PackedSectionLocation {
        const section_kinds = std.enums.values(PackedSectionKind);
        while (iterator.kind_ordinal < section_kinds.len) {
            const kind = section_kinds[iterator.kind_ordinal];
            const layers_count = packedSectionInstancesCount(kind, iterator.model_specification);
            if (iterator.layer_ordinal == layers_count) {
                iterator.kind_ordinal += 1;
                iterator.layer_ordinal = 0;
                continue;
            }

            const layer_index = packedSectionLayerIndex(kind, iterator.layer_ordinal);
            const payload_size = packedSectionSize(kind, iterator.model_specification);
            const location: PackedSectionLocation = .{
                .kind = kind,
                .layer_index = layer_index,
                .section_index = iterator.section_index,
                .payload_offset = iterator.payload_offset,
                .payload_size = payload_size,
            };

            const payload_end = std.math.add(usize, iterator.payload_offset, payload_size) catch unreachable;
            iterator.payload_offset = std.mem.alignForward(usize, payload_end, packed_model_image_alignment);
            iterator.layer_ordinal += 1;
            iterator.section_index += 1;

            return location;
        }

        return null;
    }
};

const packed_sections_count_max: usize = 11 + 12 * 12 + 12 * 20;
const PackedSectionsWritten = std.StaticBitSet(packed_sections_count_max);

const PackedImageBuilder = struct {
    allocator: Allocator,
    kind: ModelKind,
    model_specification: ModelSpecification,
    image: []align(packed_model_image_alignment) u8,
    sections_count: usize,
    sections_written: PackedSectionsWritten = .empty,

    fn init(allocator: Allocator, kind: ModelKind) ModelLoadError!PackedImageBuilder {
        // ── Measure And Allocate ──

        const model_specification = kind.specification();
        const sections_count = countPackedSections(model_specification);
        assert(sections_count > 0);
        assert(sections_count <= packed_sections_count_max);

        const image_size = measurePackedImageSize(model_specification);
        if (image_size > packed_image_size_max) {
            return error.InvalidPristineWeights;
        }

        const image = try allocator.alignedAlloc(u8, .fromByteUnits(packed_model_image_alignment), image_size);

        // ── Initialize Header And Padding ──
        //
        // PERFORMANCE: Tensor conversion overwrites every payload value. Zero
        // only the header and alignment gaps so model construction does not
        // write the complete image twice.

        var section_iterator = PackedSectionIterator.init(model_specification);
        const payload_start_offset = section_iterator.payload_offset;
        assert(payload_start_offset == packed_image_header_size);
        @memset(image[0..payload_start_offset], 0);

        var previous_payload_end = payload_start_offset;
        while (section_iterator.next()) |section| {
            assert(previous_payload_end <= section.payload_offset);
            @memset(image[previous_payload_end..section.payload_offset], 0);

            switch (packedSectionEncoding(section.kind)) {
                .float32_little_endian => {},
                .int8_row_major_with_float32_scales_and_int32_compensation, .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation => {
                    const dimensions = packedSectionDimensions(section.kind, model_specification);
                    const weight_layout = quantizedWeightLayout(dimensions);
                    const payload = image[section.payload_offset..][0..section.payload_size];
                    @memset(payload[weight_layout.weights_size..weight_layout.scales_offset], 0);
                    @memset(payload[weight_layout.scales_offset + weight_layout.scales_size .. weight_layout.compensation_offset], 0);
                },
            }

            previous_payload_end = section.payload_offset + section.payload_size;
        }
        assert(section_iterator.section_index == sections_count);
        assert(section_iterator.payload_offset == image.len);
        @memset(image[previous_payload_end..], 0);

        return .{
            .allocator = allocator,
            .kind = kind,
            .model_specification = model_specification,
            .image = image,
            .sections_count = sections_count,
        };
    }

    fn writeTensor(builder: *PackedImageBuilder, section_kind: PackedSectionKind, layer_index: u16, tensor: CT2Tensor) ModelLoadError!void {
        const section = findPackedSection(builder.model_specification, section_kind, layer_index);
        assert(section.section_index < builder.sections_count);
        assert(!builder.sections_written.isSet(section.section_index));

        const destination = builder.image[section.payload_offset..][0..section.payload_size];
        switch (packedSectionEncoding(section.kind)) {
            .float32_little_endian => try convertFloat16TensorToFloat32(tensor.data, destination),
            .int8_row_major_with_float32_scales_and_int32_compensation => try quantizeRowMajorWeight(tensor.data, packedSectionDimensions(section_kind, builder.model_specification), destination),
            .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation => try quantizeVnniWeight(tensor.data, packedSectionDimensions(section_kind, builder.model_specification), destination),
        }

        builder.sections_written.set(section.section_index);
    }

    fn intoModel(builder: *PackedImageBuilder) Model {
        // ── Seal Image ──

        assert(builder.sections_written.count() == builder.sections_count);
        assert(builder.image.len <= packed_image_size_max);

        @memcpy(builder.image[packed_image_magic_offset..packed_image_format_version_offset], packed_image_magic);
        writePackedInt(u32, builder.image, packed_image_format_version_offset, packed_model_image_format_version);
        writePackedInt(u16, builder.image, packed_image_model_kind_offset, @intFromEnum(builder.kind));
        writePackedInt(u16, builder.image, packed_image_target_offset, @intFromEnum(PackedTarget.x86_64_avx_vnni));
        writePackedInt(u64, builder.image, packed_image_size_offset, builder.image.len);
        writePackedInt(u32, builder.image, packed_image_flags_offset, 0);

        // ── Transfer Ownership ──

        const model: Model = .{
            .kind = builder.kind,
            .specification = builder.model_specification,
            .backing = .{ .owned = .{ .allocator = builder.allocator, .image = builder.image } },
        };
        builder.* = undefined;

        return model;
    }

    fn deinit(builder: *PackedImageBuilder) void {
        builder.allocator.free(builder.image);
        builder.* = undefined;
    }
};

fn validatePackedImage(image: []align(packed_model_image_alignment) const u8) ModelLoadError!ModelKind {
    if (image.len < packed_image_header_size or image.len > packed_image_size_max) {
        return error.InvalidPackedImage;
    }
    if (!std.mem.eql(u8, image[packed_image_magic_offset..packed_image_format_version_offset], packed_image_magic)) {
        return error.InvalidPackedImage;
    }

    const format_version = readPackedInt(u32, image, packed_image_format_version_offset);
    if (format_version != packed_model_image_format_version) {
        return error.UnsupportedModelImageFormatVersion;
    }

    const kind = std.enums.fromInt(ModelKind, readPackedInt(u16, image, packed_image_model_kind_offset)) orelse {
        return error.UnsupportedModelKind;
    };

    const packed_target = std.enums.fromInt(PackedTarget, readPackedInt(u16, image, packed_image_target_offset)) orelse {
        return error.UnsupportedPackedTarget;
    };
    if (packed_target != .x86_64_avx_vnni) {
        return error.UnsupportedPackedTarget;
    }

    const model_specification = kind.specification();
    const expected_image_size = measurePackedImageSize(model_specification);
    if (readPackedInt(u64, image, packed_image_size_offset) != image.len or image.len != expected_image_size) {
        return error.InvalidPackedImage;
    }
    if (readPackedInt(u32, image, packed_image_flags_offset) != 0) {
        return error.InvalidPackedImage;
    }
    if (!std.mem.allEqual(u8, image[packed_image_reserved_offset..packed_image_header_size], 0)) {
        return error.InvalidPackedImage;
    }

    return kind;
}

fn countPackedSections(model_specification: ModelSpecification) usize {
    const model_sections_count: usize = 11;
    const encoder_sections_count = model_specification.encoder_layers_count * 12;
    const decoder_sections_count = model_specification.decoder_layers_count * 20;
    const sections_count = model_sections_count + encoder_sections_count + decoder_sections_count;

    assert(sections_count <= packed_sections_count_max);
    return sections_count;
}

fn measurePackedImageSize(model_specification: ModelSpecification) usize {
    var section_iterator = PackedSectionIterator.init(model_specification);
    while (section_iterator.next() != null) {}

    return section_iterator.payload_offset;
}

fn findPackedSection(model_specification: ModelSpecification, kind: PackedSectionKind, layer_index: u16) PackedSectionLocation {
    var section_iterator = PackedSectionIterator.init(model_specification);
    while (section_iterator.next()) |section| {
        if (section.kind == kind and section.layer_index == layer_index) {
            return section;
        }
    }

    unreachable;
}

fn packedSectionInstancesCount(kind: PackedSectionKind, model_specification: ModelSpecification) usize {
    return switch (packedSectionDomain(kind)) {
        .model => 1,
        .encoder_layer => model_specification.encoder_layers_count,
        .decoder_layer => model_specification.decoder_layers_count,
    };
}

fn packedSectionLayerIndex(kind: PackedSectionKind, layer_ordinal: usize) u16 {
    return switch (packedSectionDomain(kind)) {
        .model => model_wide_layer_index,
        .encoder_layer, .decoder_layer => @intCast(layer_ordinal),
    };
}

fn packedSectionDomain(kind: PackedSectionKind) PackedSectionDomain {
    const value = @intFromEnum(kind);
    if (value <= @intFromEnum(PackedSectionKind.encoder_position_encodings)) {
        return .model;
    }
    if (value <= @intFromEnum(PackedSectionKind.encoder_layer_ffn_contraction_weight)) {
        return .encoder_layer;
    }

    return .decoder_layer;
}

fn packedSectionEncoding(kind: PackedSectionKind) PackedSectionEncoding {
    return switch (kind) {
        .encoder_convolution_1_weight, .encoder_convolution_2_weight => .int8_row_major_with_float32_scales_and_int32_compensation,
        .decoder_embeddings_weight,
        .encoder_layer_self_attention_query_key_value_weight,
        .encoder_layer_self_attention_output_weight,
        .encoder_layer_ffn_expansion_weight,
        .encoder_layer_ffn_contraction_weight,
        .decoder_layer_self_attention_query_key_value_weight,
        .decoder_layer_self_attention_output_weight,
        .decoder_layer_cross_attention_query_weight,
        .decoder_layer_cross_attention_key_value_weight,
        .decoder_layer_cross_attention_output_weight,
        .decoder_layer_ffn_expansion_weight,
        .decoder_layer_ffn_contraction_weight,
        => .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation,
        else => .float32_little_endian,
    };
}

fn packedSectionDimensions(kind: PackedSectionKind, model_specification: ModelSpecification) PackedTensorDimensions {
    const encoder_width = model_specification.encoder_width;
    const decoder_width = model_specification.decoder_width;
    const encoder_ffn_width = model_specification.encoder_ffn_width;
    const decoder_ffn_width = model_specification.decoder_ffn_width;

    return switch (kind) {
        .decoder_embeddings_weight => tensorDimensions2(model_specification.vocabulary_tokens_count, decoder_width),
        .decoder_layer_norm_beta, .decoder_layer_norm_gamma => tensorDimensions1(decoder_width),
        .decoder_position_encodings => tensorDimensions2(model_specification.decoder_positions_count_max, decoder_width),
        .encoder_convolution_1_bias => tensorDimensions1(encoder_width),
        .encoder_convolution_1_weight => tensorDimensions3(encoder_width, model_specification.mel_bins_count, 3),
        .encoder_convolution_2_bias => tensorDimensions1(encoder_width),
        .encoder_convolution_2_weight => tensorDimensions3(encoder_width, encoder_width, 3),
        .encoder_layer_norm_beta, .encoder_layer_norm_gamma => tensorDimensions1(encoder_width),
        .encoder_position_encodings => tensorDimensions2(model_specification.encoder_positions_count_max, encoder_width),
        .encoder_layer_self_attention_layer_norm_beta,
        .encoder_layer_self_attention_layer_norm_gamma,
        .encoder_layer_self_attention_output_bias,
        .encoder_layer_ffn_layer_norm_beta,
        .encoder_layer_ffn_layer_norm_gamma,
        .encoder_layer_ffn_contraction_bias,
        => tensorDimensions1(encoder_width),
        .encoder_layer_self_attention_query_key_value_bias => tensorDimensions1(3 * encoder_width),
        .encoder_layer_self_attention_query_key_value_weight => tensorDimensions2(3 * encoder_width, encoder_width),
        .encoder_layer_self_attention_output_weight => tensorDimensions2(encoder_width, encoder_width),
        .encoder_layer_ffn_expansion_bias => tensorDimensions1(encoder_ffn_width),
        .encoder_layer_ffn_expansion_weight => tensorDimensions2(encoder_ffn_width, encoder_width),
        .encoder_layer_ffn_contraction_weight => tensorDimensions2(encoder_width, encoder_ffn_width),
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
        => tensorDimensions1(decoder_width),
        .decoder_layer_self_attention_query_key_value_bias => tensorDimensions1(3 * decoder_width),
        .decoder_layer_self_attention_query_key_value_weight => tensorDimensions2(3 * decoder_width, decoder_width),
        .decoder_layer_self_attention_output_weight,
        .decoder_layer_cross_attention_query_weight,
        .decoder_layer_cross_attention_output_weight,
        => tensorDimensions2(decoder_width, decoder_width),
        .decoder_layer_cross_attention_key_value_bias => tensorDimensions1(2 * decoder_width),
        .decoder_layer_cross_attention_key_value_weight => tensorDimensions2(2 * decoder_width, encoder_width),
        .decoder_layer_ffn_expansion_bias => tensorDimensions1(decoder_ffn_width),
        .decoder_layer_ffn_expansion_weight => tensorDimensions2(decoder_ffn_width, decoder_width),
        .decoder_layer_ffn_contraction_weight => tensorDimensions2(decoder_width, decoder_ffn_width),
    };
}

fn packedSectionSize(kind: PackedSectionKind, model_specification: ModelSpecification) usize {
    const dimensions = packedSectionDimensions(kind, model_specification);
    return switch (packedSectionEncoding(kind)) {
        .float32_little_endian => std.math.mul(usize, dimensions.elementsCount(), @sizeOf(f32)) catch unreachable,
        .int8_row_major_with_float32_scales_and_int32_compensation, .vnni_u8s8_o8_k4_with_float32_scales_and_int32_compensation => quantizedWeightLayout(dimensions).section_size,
    };
}

fn quantizedWeightLayout(dimensions: PackedTensorDimensions) QuantizedWeightLayout {
    assert(dimensions.count == 2 or dimensions.count == 3);

    const output_rows_count = dimensions.values[0];
    const weights_size = dimensions.elementsCount();
    const input_values_count = @divExact(weights_size, output_rows_count);
    const scales_offset = std.mem.alignForward(usize, weights_size, packed_model_image_alignment);
    const scales_size = std.math.mul(usize, output_rows_count, @sizeOf(f32)) catch unreachable;
    const scales_end = std.math.add(usize, scales_offset, scales_size) catch unreachable;
    const compensation_offset = std.mem.alignForward(usize, scales_end, packed_model_image_alignment);
    const compensation_size = std.math.mul(usize, output_rows_count, @sizeOf(i32)) catch unreachable;
    const section_size = std.math.add(usize, compensation_offset, compensation_size) catch unreachable;

    return .{
        .weights_size = weights_size,
        .scales_offset = scales_offset,
        .scales_size = scales_size,
        .compensation_offset = compensation_offset,
        .compensation_size = compensation_size,
        .section_size = section_size,
        .output_rows_count = output_rows_count,
        .input_values_count = input_values_count,
    };
}

fn convertFloat16TensorToFloat32(source: []const u8, destination: []u8) ModelLoadError!void {
    assert(source.len % @sizeOf(f16) == 0);
    assert(destination.len == source.len / @sizeOf(f16) * @sizeOf(f32));

    const values_count = source.len / @sizeOf(f16);
    for (0..values_count) |value_index| {
        const value = readFloat16(source, value_index);
        if (!std.math.isFinite(value)) {
            return error.InvalidPristineWeights;
        }

        writePackedFloat32(destination, value_index * @sizeOf(f32), value);
    }
}

fn quantizeRowMajorWeight(source: []const u8, dimensions: PackedTensorDimensions, destination: []u8) ModelLoadError!void {
    const layout = quantizedWeightLayout(dimensions);
    assert(source.len == layout.weights_size * @sizeOf(f16));
    assert(destination.len == layout.section_size);

    for (0..layout.output_rows_count) |output_row_index| {
        var absolute_maximum: f32 = 0;
        for (0..layout.input_values_count) |input_value_index| {
            const source_value_index = output_row_index * layout.input_values_count + input_value_index;
            const value = readFloat16(source, source_value_index);
            if (!std.math.isFinite(value)) {
                return error.InvalidPristineWeights;
            }

            absolute_maximum = @max(absolute_maximum, @abs(value));
        }

        const scale: f32 = if (absolute_maximum == 0) 1 else 127.0 / absolute_maximum;
        assert(std.math.isFinite(scale));
        assert(scale > 0);
        writePackedFloat32(destination, layout.scales_offset + output_row_index * @sizeOf(f32), scale);

        var quantized_values_sum: i32 = 0;
        for (0..layout.input_values_count) |input_value_index| {
            const source_value_index = output_row_index * layout.input_values_count + input_value_index;
            const value = readFloat16(source, source_value_index);
            const scaled_value = std.math.clamp(value * scale, -127.0, 127.0);
            const quantized_value = roundFloat32ToNearestEven(scaled_value);
            assert(quantized_value >= -127 and quantized_value <= 127);

            destination[source_value_index] = @bitCast(@as(i8, @intCast(quantized_value)));
            quantized_values_sum += quantized_value;
        }

        const compensation = std.math.mul(i32, quantized_values_sum, -128) catch unreachable;
        writePackedInt(i32, destination, layout.compensation_offset + output_row_index * @sizeOf(i32), compensation);
    }
}

fn quantizeVnniWeight(source: []const u8, dimensions: PackedTensorDimensions, destination: []u8) ModelLoadError!void {
    const layout = quantizedWeightLayout(dimensions);
    assert(dimensions.count == 2);
    assert(source.len == layout.weights_size * @sizeOf(f16));
    assert(destination.len == layout.section_size);
    assert(layout.output_rows_count % packed_weight_output_block_rows_count == 0);
    assert(layout.input_values_count % 8 == 0);

    try calculateVnniWeightScales(source, layout, destination);
    quantizeAndPackVnniWeight(source, layout, destination);
}

fn calculateVnniWeightScales(source: []const u8, layout: QuantizedWeightLayout, destination: []u8) ModelLoadError!void {
    const F16x8 = @Vector(8, f16);
    const F32x8 = @Vector(8, f32);
    const U16x8 = @Vector(8, u16);
    const exponent_mask: U16x8 = @splat(0x7c00);

    assert(layout.input_values_count % 8 == 0);

    for (0..layout.output_rows_count) |output_row_index| {
        var absolute_maximums: F32x8 = @splat(0);
        var input_value_start_index: usize = 0;
        while (input_value_start_index < layout.input_values_count) : (input_value_start_index += 8) {
            const source_value_index = output_row_index * layout.input_values_count + input_value_start_index;
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
        writePackedFloat32(destination, layout.scales_offset + output_row_index * @sizeOf(f32), scale);
    }
}

fn quantizeAndPackVnniWeight(source: []const u8, layout: QuantizedWeightLayout, destination: []u8) void {
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
    assert(layout.output_rows_count % packed_weight_output_block_rows_count == 0);
    assert(layout.input_values_count % packed_weight_depth_group_values_count == 0);

    var output_row_start_index: usize = 0;
    while (output_row_start_index < layout.output_rows_count) : (output_row_start_index += packed_weight_output_block_rows_count) {
        var output_scales: F32x8 = undefined;
        inline for (0..packed_weight_output_block_rows_count) |block_row_index| {
            const scale_offset = layout.scales_offset + (output_row_start_index + block_row_index) * @sizeOf(f32);
            output_scales[block_row_index] = readPackedFloat32(destination, scale_offset);
        }

        var quantized_values_sums: I32x8 = @splat(0);
        var input_value_start_index: usize = 0;
        while (input_value_start_index < layout.input_values_count) : (input_value_start_index += packed_weight_depth_group_values_count) {
            const packed_group_offset = output_row_start_index * layout.input_values_count + input_value_start_index * packed_weight_output_block_rows_count;
            var quantized_group_sums: I32x8 = undefined;

            inline for (0..packed_weight_output_block_rows_count) |block_row_index| {
                const output_row_index = output_row_start_index + block_row_index;
                const source_value_index = output_row_index * layout.input_values_count + input_value_start_index;
                const source_offset = source_value_index * @sizeOf(f16);
                const half_bits: U16x4 = @bitCast(source[source_offset..][0 .. packed_weight_depth_group_values_count * @sizeOf(f16)].*);
                const half_values: F16x4 = @bitCast(half_bits);
                const values: F32x4 = @floatCast(half_values);
                const scaled_values = values * @as(F32x4, @splat(output_scales[block_row_index]));
                const clamped_values = @min(@max(scaled_values, minimums), maximums);
                const rounded_values = roundFloat32x4ToNearestEven(clamped_values);
                const quantized_integers: I32x4 = @intFromFloat(rounded_values);
                const quantized_values: I8x4 = @intCast(quantized_integers);
                const quantized_bytes: [packed_weight_depth_group_values_count]u8 = @bitCast(quantized_values);

                const packed_output_offset = packed_group_offset + block_row_index * packed_weight_depth_group_values_count;
                destination[packed_output_offset..][0..packed_weight_depth_group_values_count].* = quantized_bytes;
                quantized_group_sums[block_row_index] = @reduce(.Add, quantized_integers);
            }

            quantized_values_sums += quantized_group_sums;
        }

        quantized_values_sums *= @as(I32x8, @splat(-128));
        inline for (0..packed_weight_output_block_rows_count) |block_row_index| {
            const compensation_offset = layout.compensation_offset + (output_row_start_index + block_row_index) * @sizeOf(i32);
            writePackedInt(i32, destination, compensation_offset, quantized_values_sums[block_row_index]);
        }
    }
}

inline fn roundFloat32x4ToNearestEven(values: @Vector(4, f32)) @Vector(4, f32) {
    // CTranslate2 rounds quantized weights with `vroundps` in nearest-even
    // mode. Substituting Zig's `@round` here or in `roundFloat32ToNearestEven`
    // would round midpoints away from zero, diverging from the reference
    // conversion on exact `.5` inputs and changing the packed image bytes.
    var rounded: @Vector(4, f32) = undefined;
    asm volatile ("vroundps $0, %[values], %[rounded]"
        : [rounded] "=x" (rounded),
        : [values] "x" (values),
    );
    return rounded;
}

fn roundFloat32ToNearestEven(value: f32) i32 {
    assert(std.math.isFinite(value));
    assert(value >= @as(f32, @floatFromInt(std.math.minInt(i32))));
    assert(value <= @as(f32, @floatFromInt(std.math.maxInt(i32))));

    const lower_float = @floor(value);
    const lower_integer: i32 = @intFromFloat(lower_float);
    const fraction = value - lower_float;
    if (fraction < 0.5) {
        return lower_integer;
    }
    if (fraction > 0.5) {
        return lower_integer + 1;
    }
    if (@mod(lower_integer, 2) == 0) {
        return lower_integer;
    }

    return lower_integer + 1;
}

fn readFloat16(source: []const u8, value_index: usize) f32 {
    const byte_offset = value_index * @sizeOf(f16);
    assert(byte_offset + @sizeOf(f16) <= source.len);

    const bits = std.mem.readInt(u16, source[byte_offset..][0..@sizeOf(f16)], .little);
    const value: f16 = @bitCast(bits);
    return @floatCast(value);
}

fn tensorDimensions1(first: usize) PackedTensorDimensions {
    assert(first > 0);

    return .{ .values = .{ first, 0, 0 }, .count = 1 };
}

fn tensorDimensions2(first: usize, second: usize) PackedTensorDimensions {
    assert(first > 0);
    assert(second > 0);

    return .{ .values = .{ first, second, 0 }, .count = 2 };
}

fn tensorDimensions3(first: usize, second: usize, third: usize) PackedTensorDimensions {
    assert(first > 0);
    assert(second > 0);
    assert(third > 0);

    return .{ .values = .{ first, second, third }, .count = 3 };
}

fn writePackedFloat32(destination: []u8, byte_offset: usize, value: f32) void {
    writePackedInt(u32, destination, byte_offset, @as(u32, @bitCast(value)));
}

fn readPackedFloat32(source: []const u8, byte_offset: usize) f32 {
    return @bitCast(readPackedInt(u32, source, byte_offset));
}

fn writePackedInt(comptime Int: type, destination: []u8, byte_offset: usize, value: anytype) void {
    assert(byte_offset <= destination.len);
    assert(@sizeOf(Int) <= destination.len - byte_offset);

    std.mem.writeInt(Int, destination[byte_offset..][0..@sizeOf(Int)], @intCast(value), .little);
}

fn readPackedInt(comptime Int: type, source: []const u8, byte_offset: usize) Int {
    assert(byte_offset <= source.len);
    assert(@sizeOf(Int) <= source.len - byte_offset);

    return std.mem.readInt(Int, source[byte_offset..][0..@sizeOf(Int)], .little);
}

// ─── Tests ─────────────────────────────────────────────────────────────────
//
// The model-kind tests use the real model files installed by `zig build
// setup-models`; each converts pristine weights, persists the packed image,
// reloads independent bytes, and compares the reconstructed model. The
// synthetic fixture tests verify quantization boundaries without model files.

test "base.en pristine weights and packed image produce the same model" {
    const expected_specification: ModelSpecification = .{
        .mel_bins_count = 80,
        .encoder_positions_count_max = 1500,
        .encoder_width = 512,
        .encoder_attention_heads_count = 8,
        .encoder_layers_count = 6,
        .encoder_ffn_width = 2048,
        .decoder_positions_count_max = 448,
        .decoder_width = 512,
        .decoder_attention_heads_count = 8,
        .decoder_layers_count = 6,
        .decoder_ffn_width = 2048,
        .vocabulary_tokens_count = 51_864,
    };

    try testExpectPristineAndPackedModelsEqual(.base_en, "Systran/faster-whisper-base.en", expected_specification);
}

test "small.en pristine weights and packed image produce the same model" {
    const expected_specification: ModelSpecification = .{
        .mel_bins_count = 80,
        .encoder_positions_count_max = 1500,
        .encoder_width = 768,
        .encoder_attention_heads_count = 12,
        .encoder_layers_count = 12,
        .encoder_ffn_width = 3072,
        .decoder_positions_count_max = 448,
        .decoder_width = 768,
        .decoder_attention_heads_count = 12,
        .decoder_layers_count = 12,
        .decoder_ffn_width = 3072,
        .vocabulary_tokens_count = 51_864,
    };

    try testExpectPristineAndPackedModelsEqual(.small_en, "Systran/faster-whisper-small.en", expected_specification);
}

test "synthetic VNNI fixture verifies quantization boundary values" {
    const allocator = std.testing.allocator;
    const dimensions = tensorDimensions2(8, 8);
    const layout = quantizedWeightLayout(dimensions);

    const source_values = [8][8]f32{
        .{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        .{ 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 127.0 },
        .{ -127.0, -126.0, -1.0, 0.0, 1.0, 126.0, 127.0, -0.0 },
        .{ -127.0, -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0 },
        .{ 64.0, -64.0, 32.0, -32.0, 0.5, -0.5, 1.5, -1.5 },
        .{ 0.25 / 1024.0, -0.25 / 1024.0, 0.5 / 1024.0, -0.5 / 1024.0, 0.75 / 1024.0, -0.75 / 1024.0, 1.0 / 1024.0, -1.0 / 1024.0 },
        .{ -0.5, -1.5, -2.5, -3.5, 0.5, 1.5, 2.5, 127.0 },
        .{ -127.0, -0.25, 0.25, -0.75, 0.75, -1.25, 1.25, 0.0 },
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
    const expected_scales = [8]f32{ 1.0, 1.0, 1.0, 1.0, 127.0 / 64.0, 127.0 * 1024.0, 1.0, 1.0 };

    const source = try allocator.alloc(u8, layout.weights_size * @sizeOf(f16));
    defer allocator.free(source);
    testWriteFloat16Values(source, layout, &source_values);

    const destination = try allocator.alloc(u8, layout.section_size);
    defer allocator.free(destination);
    @memset(destination, 0);

    try quantizeVnniWeight(source, dimensions, destination);

    try testExpectQuantizedPadding(destination, layout);
    for (0..layout.output_rows_count) |output_row_index| {
        try testExpectPackedScale(destination, layout, output_row_index, expected_scales[output_row_index]);
        try testExpectPackedCompensation(destination, layout, output_row_index, &expected_weights[output_row_index]);
        for (0..layout.input_values_count) |input_value_index| {
            const actual_byte = testReadVnniWeight(destination, layout, output_row_index, input_value_index);
            try std.testing.expectEqual(@as(u8, @bitCast(expected_weights[output_row_index][input_value_index])), actual_byte);
        }
    }
}

test "synthetic row-major fixture verifies quantization and compensation" {
    const allocator = std.testing.allocator;
    const dimensions = tensorDimensions2(3, 4);
    const layout = quantizedWeightLayout(dimensions);

    const source_values = [3][4]f32{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ -2.0, -1.0, 1.0, 2.0 },
        .{ 0.5, 1.0, 1.5, 2.0 },
    };
    const expected_weights = [3][4]i8{
        .{ 0, 0, 0, 0 },
        .{ -127, -64, 64, 127 },
        .{ 32, 64, 95, 127 },
    };
    const expected_scales = [3]f32{ 1.0, 63.5, 63.5 };

    const source = try allocator.alloc(u8, layout.weights_size * @sizeOf(f16));
    defer allocator.free(source);
    testWriteFloat16Values(source, layout, &source_values);

    const destination = try allocator.alloc(u8, layout.section_size);
    defer allocator.free(destination);
    @memset(destination, 0);

    try quantizeRowMajorWeight(source, dimensions, destination);

    try testExpectQuantizedPadding(destination, layout);
    for (0..layout.output_rows_count) |output_row_index| {
        try testExpectPackedScale(destination, layout, output_row_index, expected_scales[output_row_index]);
        try testExpectPackedCompensation(destination, layout, output_row_index, &expected_weights[output_row_index]);
        for (0..layout.input_values_count) |input_value_index| {
            const value_index = output_row_index * layout.input_values_count + input_value_index;
            try std.testing.expectEqual(@as(u8, @bitCast(expected_weights[output_row_index][input_value_index])), destination[value_index]);
        }
    }
}

fn testExpectPristineAndPackedModelsEqual(kind: ModelKind, installed_directory_name: []const u8, expected_specification: ModelSpecification) !void {
    // ── Convert ──
    //
    // Successful conversion means every source tensor was classified,
    // validated, and written into its canonical packed-image section.

    const allocator = std.testing.allocator;
    const pristine_weights = try testReadInstalledPristineWeights(allocator, installed_directory_name);
    defer allocator.free(pristine_weights);

    var pristine_model = try Model.fromPristineWeights(allocator, kind, pristine_weights);
    defer pristine_model.deinit();

    try std.testing.expectEqual(kind, pristine_model.kind);
    try std.testing.expectEqualDeep(expected_specification, pristine_model.specification);

    const in_memory_packed_image = pristine_model.packedImage();
    try std.testing.expect(in_memory_packed_image.len > 0);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(in_memory_packed_image.ptr) % packed_model_image_alignment);

    // ── Reconvert ──
    //
    // A second live allocation proves conversion is independent of destination
    // address and initializes every byte deterministically.

    {
        var repeated_model = try Model.fromPristineWeights(allocator, kind, pristine_weights);
        defer repeated_model.deinit();

        const repeated_packed_image = repeated_model.packedImage();
        try std.testing.expect(@intFromPtr(in_memory_packed_image.ptr) != @intFromPtr(repeated_packed_image.ptr));
        try std.testing.expectEqualSlices(u8, in_memory_packed_image, repeated_packed_image);
    }

    // ── Persist ──
    //
    // Reading into a separate allocation ensures byte equality is not merely
    // the result of comparing two aliases of the converter's allocation.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const packed_image_file_name = "model.voiced";
    try tmp_dir.dir.writeFile(std.testing.io, .{ .sub_path = packed_image_file_name, .data = in_memory_packed_image });

    const disk_packed_image = try allocator.alignedAlloc(u8, .fromByteUnits(packed_model_image_alignment), in_memory_packed_image.len);
    defer allocator.free(disk_packed_image);

    const disk_bytes_read = try tmp_dir.dir.readFile(std.testing.io, packed_image_file_name, disk_packed_image);
    try std.testing.expectEqual(disk_packed_image.len, disk_bytes_read.len);
    try std.testing.expectEqualSlices(u8, in_memory_packed_image, disk_packed_image);
    try std.testing.expect(@intFromPtr(in_memory_packed_image.ptr) != @intFromPtr(disk_packed_image.ptr));

    // ── Map ──
    //
    // Packed loading accepts a mapped image under the `fromPackedImage`
    // contract. Map the persisted image independently and prove the model
    // borrows those bytes directly.

    const mapped_file = try tmp_dir.dir.openFile(std.testing.io, packed_image_file_name, .{ .mode = .read_only });
    defer mapped_file.close(std.testing.io);

    const mapped_packed_image_bytes = try std.posix.mmap(null, in_memory_packed_image.len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, mapped_file.handle, 0);
    defer std.posix.munmap(mapped_packed_image_bytes);

    const mapped_packed_image: []align(packed_model_image_alignment) const u8 = @alignCast(mapped_packed_image_bytes);
    var mapped_model = try Model.fromPackedImage(mapped_packed_image);
    defer mapped_model.deinit();

    try std.testing.expectEqual(pristine_model.kind, mapped_model.kind);
    try std.testing.expectEqualDeep(pristine_model.specification, mapped_model.specification);
    try std.testing.expectEqual(@intFromPtr(mapped_packed_image.ptr), @intFromPtr(mapped_model.packedImage().ptr));

    // ── Reload ──
    //
    // Packed loading must borrow the independent bytes while reconstructing
    // the same model properties produced by pristine conversion.

    var packed_model = try Model.fromPackedImage(disk_packed_image);
    defer packed_model.deinit();

    try std.testing.expectEqual(pristine_model.kind, packed_model.kind);
    try std.testing.expectEqualDeep(pristine_model.specification, packed_model.specification);
    try std.testing.expectEqualSlices(u8, in_memory_packed_image, packed_model.packedImage());

    // Packed loading is zero-copy: the loaded model borrows the allocation
    // filled from disk rather than copying it or referring to the pristine model.
    try std.testing.expectEqual(@intFromPtr(disk_packed_image.ptr), @intFromPtr(packed_model.packedImage().ptr));
}

fn testWriteFloat16Values(destination: []u8, layout: QuantizedWeightLayout, source_values: anytype) void {
    assert(destination.len == layout.weights_size * @sizeOf(f16));

    for (0..layout.output_rows_count) |output_row_index| {
        for (0..layout.input_values_count) |input_value_index| {
            const source_value_index = output_row_index * layout.input_values_count + input_value_index;
            const value: f16 = @floatCast(source_values[output_row_index][input_value_index]);
            std.mem.writeInt(u16, destination[source_value_index * @sizeOf(f16) ..][0..@sizeOf(f16)], @bitCast(value), .little);
        }
    }
}

fn testExpectQuantizedPadding(destination: []const u8, layout: QuantizedWeightLayout) !void {
    try std.testing.expect(std.mem.allEqual(u8, destination[layout.weights_size..layout.scales_offset], 0));
    try std.testing.expect(std.mem.allEqual(u8, destination[layout.scales_offset + layout.scales_size .. layout.compensation_offset], 0));
}

fn testExpectPackedScale(destination: []const u8, layout: QuantizedWeightLayout, output_row_index: usize, expected_scale: f32) !void {
    const scale_offset = layout.scales_offset + output_row_index * @sizeOf(f32);
    const actual_scale_bits = std.mem.readInt(u32, destination[scale_offset..][0..@sizeOf(u32)], .little);
    try std.testing.expectEqual(@as(u32, @bitCast(expected_scale)), actual_scale_bits);
}

fn testExpectPackedCompensation(destination: []const u8, layout: QuantizedWeightLayout, output_row_index: usize, expected_weights: []const i8) !void {
    assert(expected_weights.len == layout.input_values_count);

    var quantized_values_sum: i32 = 0;
    for (expected_weights) |quantized_value| {
        quantized_values_sum += quantized_value;
    }

    const compensation_offset = layout.compensation_offset + output_row_index * @sizeOf(i32);
    const expected_compensation = std.math.mul(i32, quantized_values_sum, -128) catch unreachable;
    const actual_compensation = std.mem.readInt(i32, destination[compensation_offset..][0..@sizeOf(i32)], .little);
    try std.testing.expectEqual(expected_compensation, actual_compensation);
}

fn testReadVnniWeight(destination: []const u8, layout: QuantizedWeightLayout, output_row_index: usize, input_value_index: usize) u8 {
    assert(output_row_index < layout.output_rows_count);
    assert(input_value_index < layout.input_values_count);
    assert(layout.output_rows_count % packed_weight_output_block_rows_count == 0);
    assert(layout.input_values_count % packed_weight_depth_group_values_count == 0);

    const output_block_index = output_row_index / packed_weight_output_block_rows_count;
    const block_row_index = output_row_index % packed_weight_output_block_rows_count;
    const depth_group_index = input_value_index / packed_weight_depth_group_values_count;
    const group_value_index = input_value_index % packed_weight_depth_group_values_count;
    const output_block_stride = packed_weight_output_block_rows_count * layout.input_values_count;
    const depth_group_stride = packed_weight_output_block_rows_count * packed_weight_depth_group_values_count;
    const value_offset = output_block_index * output_block_stride + depth_group_index * depth_group_stride + block_row_index * packed_weight_depth_group_values_count + group_value_index;
    return destination[value_offset];
}

fn testReadInstalledPristineWeights(allocator: Allocator, installed_directory_name: []const u8) ![]u8 {
    const env = std.testing.environ;

    const data_home = env.getPosix("XDG_DATA_HOME");
    const path_parts: []const []const u8 = if (data_home) |path|
        &.{ path, "voiced/models", installed_directory_name, "model.bin" }
    else
        &.{ env.getPosix("HOME") orelse return error.HomeNotSet, ".local/share/voiced/models", installed_directory_name, "model.bin" };

    const path = try std.fs.path.join(allocator, path_parts);
    defer allocator.free(path);

    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(pristine_weights_size_max));
}
