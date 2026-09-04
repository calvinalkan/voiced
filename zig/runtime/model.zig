//! `Model` is the public lifecycle facade over one validated packed model
//! image. Packed-image layout and pristine CTranslate2 conversion remain in
//! their representation-owning modules.

const std = @import("std");
const ctranslate2_weights = @import("ctranslate2_weights.zig");
const model_specification = @import("model_specification.zig");
const packed_model_image = @import("packed_model_image.zig");
const vnni_weight = @import("vnni_weight.zig");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const SectionKind = packed_model_image.SectionKind;

pub const ModelKind = model_specification.ModelKind;
pub const ModelSpecification = model_specification.ModelSpecification;
pub const QuantizedWeight = vnni_weight.QuantizedWeight;

/// Alignment of every packed image. Buffers filled from persistent storage
/// must satisfy it before `fromPackedImage` accepts them; `packedImage` always
/// returns bytes with this alignment.
pub const packed_model_image_alignment = packed_model_image.alignment;

/// `packed_model_image_format_version` identifies the header and canonical
/// payload sequence. The image header separately identifies packed target 2.
pub const packed_model_image_format_version = packed_model_image.format_version;

/// `ModelLoadError` reports malformed or incompatible model input and
/// allocation failure. A failed load returns no partially initialized `Model`.
pub const ModelLoadError = packed_model_image.LoadError || ctranslate2_weights.ConvertError;

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
    encoder_layers: [model_specification.layers_count_max]EncoderLayerWeights,
    decoder_layers: [model_specification.layers_count_max]DecoderLayerWeights,
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

    /// `fromPackedImage` validates an aligned target-2 native image and
    /// constructs model views directly over its bytes. It performs no
    /// allocation or copying and does not take ownership of `image`.
    pub fn fromPackedImage(image: []align(packed_model_image_alignment) const u8) ModelLoadError!Model {
        const kind = try packed_model_image.validate(image);

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
    /// The function rejects any input that is not the canonical CTranslate2
    /// version-6 serialization for `kind` with
    /// `error.InvalidPristineWeights`. It allocates one packed image through
    /// `allocator`; the returned model owns that image and retains no reference
    /// to `weights`.
    pub fn fromPristineWeights(allocator: Allocator, kind: ModelKind, weights: []const u8) ModelLoadError!Model {
        const image = try ctranslate2_weights.convertToPackedImage(allocator, kind, weights);

        return .{
            .kind = kind,
            .specification = kind.specification(),
            .backing = .{ .owned = .{ .allocator = allocator, .image = image } },
        };
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
        var encoder_layers: [model_specification.layers_count_max]EncoderLayerWeights = undefined;
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

        var decoder_layers: [model_specification.layers_count_max]DecoderLayerWeights = undefined;
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
            .decoder_embeddings_weight = model.quantizedSection(.decoder_embeddings_weight, packed_model_image.model_wide_layer_index),
            .decoder_layer_norm_beta = model.floatSection(.decoder_layer_norm_beta, packed_model_image.model_wide_layer_index),
            .decoder_layer_norm_gamma = model.floatSection(.decoder_layer_norm_gamma, packed_model_image.model_wide_layer_index),
            .decoder_position_encodings = model.floatSection(.decoder_position_encodings, packed_model_image.model_wide_layer_index),
            .encoder_convolution_1_bias = model.floatSection(.encoder_convolution_1_bias, packed_model_image.model_wide_layer_index),
            .encoder_convolution_1_weight = model.quantizedSection(.encoder_convolution_1_weight, packed_model_image.model_wide_layer_index),
            .encoder_convolution_2_bias = model.floatSection(.encoder_convolution_2_bias, packed_model_image.model_wide_layer_index),
            .encoder_convolution_2_weight = model.quantizedSection(.encoder_convolution_2_weight, packed_model_image.model_wide_layer_index),
            .encoder_layer_norm_beta = model.floatSection(.encoder_layer_norm_beta, packed_model_image.model_wide_layer_index),
            .encoder_layer_norm_gamma = model.floatSection(.encoder_layer_norm_gamma, packed_model_image.model_wide_layer_index),
            .encoder_position_encodings = model.floatSection(.encoder_position_encodings, packed_model_image.model_wide_layer_index),
            .encoder_layers = encoder_layers,
            .decoder_layers = decoder_layers,
        };
    }

    fn floatSection(model: *const Model, kind: SectionKind, layer_index: u16) []const f32 {
        return packed_model_image.floatSection(model.backing.packedImage(), model.specification, kind, layer_index);
    }

    fn quantizedSection(model: *const Model, kind: SectionKind, layer_index: u16) QuantizedWeight {
        return packed_model_image.quantizedSection(model.backing.packedImage(), model.specification, kind, layer_index);
    }
};

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

// ─── Tests ─────────────────────────────────────────────────────────────────
//
// These tests convert installed pristine weights, persist the target-2 image,
// reload independent bytes, and verify deterministic and zero-copy behavior.

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

    const inference_weights = pristine_model.inferenceWeights();
    try std.testing.expectEqual(expected_specification.encoder_width, inference_weights.encoder_convolution_1_weight.output_rows_count);
    try std.testing.expectEqual(expected_specification.mel_bins_count * 3, inference_weights.encoder_convolution_1_weight.input_values_count);
    try std.testing.expectEqual(expected_specification.encoder_width * 3, inference_weights.encoder_convolution_2_weight.input_values_count);

    // ── Reconvert ──
    //
    // A second live allocation proves conversion is independent of destination
    // address and initializes every byte deterministically.

    {
        var repeated_model = try Model.fromPristineWeights(allocator, kind, pristine_weights);
        defer repeated_model.deinit();

        try std.testing.expect(@intFromPtr(in_memory_packed_image.ptr) != @intFromPtr(repeated_model.packedImage().ptr));
        try std.testing.expectEqualSlices(u8, in_memory_packed_image, repeated_model.packedImage());
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

    // ── Map ──
    //
    // Packed loading accepts a mapped image under the `fromPackedImage`
    // contract. Map the persisted image independently and prove the model
    // borrows those bytes directly.

    const mapped_file = try tmp_dir.dir.openFile(std.testing.io, packed_image_file_name, .{ .mode = .read_only });
    defer mapped_file.close(std.testing.io);

    const mapped_bytes = try std.posix.mmap(null, in_memory_packed_image.len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, mapped_file.handle, 0);
    defer std.posix.munmap(mapped_bytes);

    const mapped_image: []align(packed_model_image_alignment) const u8 = @alignCast(mapped_bytes);
    var mapped_model = try Model.fromPackedImage(mapped_image);
    defer mapped_model.deinit();

    try std.testing.expectEqual(pristine_model.kind, mapped_model.kind);
    try std.testing.expectEqualDeep(pristine_model.specification, mapped_model.specification);
    try std.testing.expectEqual(@intFromPtr(mapped_image.ptr), @intFromPtr(mapped_model.packedImage().ptr));

    // ── Reload ──
    //
    // Packed loading must borrow the independent bytes while reconstructing
    // the same model properties produced by pristine conversion.

    var packed_model = try Model.fromPackedImage(disk_packed_image);
    defer packed_model.deinit();

    try std.testing.expectEqual(pristine_model.kind, packed_model.kind);
    try std.testing.expectEqualDeep(pristine_model.specification, packed_model.specification);
    try std.testing.expectEqualSlices(u8, in_memory_packed_image, packed_model.packedImage());
    try std.testing.expectEqual(@intFromPtr(disk_packed_image.ptr), @intFromPtr(packed_model.packedImage().ptr));
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

    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(ctranslate2_weights.weights_size_max));
}
