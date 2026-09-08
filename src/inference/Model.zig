//! Owns one immutable Whisper model and the storage backing its tensor and
//! vocabulary views. This module knows no source format, installation path, or
//! packed-file schema; the loader supplies already-validated views and storage.

const Model = @This();

const std = @import("std");
const vnni_weight = @import("vnni_weight.zig");
const assert = std.debug.assert;

pub const layers_count_max: usize = 24;
const QuantizedWeight = vnni_weight.QuantizedWeight;

// State
backing_storage: BackingStorage,
kind: Kind,
weights: Weights,
vocabulary: Vocabulary,

pub const BackingStorage = union(enum) {
    mapped: []align(std.heap.page_size_min) u8,
    allocated: struct {
        allocator: std.mem.Allocator,
        bytes: []align(64) u8,
    },
};

pub fn deinit(model: *Model) void {
    switch (model.backing_storage) {
        .mapped => |mapping| std.posix.munmap(mapping),
        .allocated => |allocation| allocation.allocator.free(allocation.bytes),
    }
    model.* = undefined;
}

pub const Kind = enum(u16) {
    // Numeric values are part of the packed-model ABI and must not be
    // reordered or reused.
    whisper_base_en = 1,
    whisper_small_en = 2,
    whisper_medium_en = 3,

    pub fn name(kind: Kind) []const u8 {
        return switch (kind) {
            .whisper_base_en => "whisper.base.en",
            .whisper_small_en => "whisper.small.en",
            .whisper_medium_en => "whisper.medium.en",
        };
    }
};

pub const Weights = struct {
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
    encoder_layers: [layers_count_max]EncoderLayer,
    decoder_layers: [layers_count_max]DecoderLayer,

    pub const EncoderLayer = struct {
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

    pub const DecoderLayer = struct {
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
};

pub const Vocabulary = struct {
    offsets: []const u32,
    bytes: []const u8,
};

pub const Dimensions = struct {
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

pub fn parseKind(text: []const u8) ?Kind {
    inline for (std.meta.tags(Kind)) |kind| {
        if (std.mem.eql(u8, text, kind.name())) {
            return kind;
        }
    }

    return null;
}

pub fn dimensions(kind: Kind) Dimensions {
    const result: Dimensions = switch (kind) {
        .whisper_base_en => .{
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
        .whisper_small_en => .{
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
        .whisper_medium_en => .{
            .mel_bins_count = 80,
            .encoder_positions_count_max = 1500,
            .encoder_width = 1024,
            .encoder_attention_heads_count = 16,
            .encoder_layers_count = 24,
            .encoder_ffn_width = 4096,
            .decoder_positions_count_max = 448,
            .decoder_width = 1024,
            .decoder_attention_heads_count = 16,
            .decoder_layers_count = 24,
            .decoder_ffn_width = 4096,
            .vocabulary_tokens_count = 51_864,
        },
    };

    assert(result.mel_bins_count > 0);
    assert(result.encoder_positions_count_max > 0);
    assert(result.encoder_width > 0);
    assert(result.encoder_attention_heads_count > 0);
    assert(result.encoder_layers_count > 0);
    assert(result.encoder_layers_count <= layers_count_max);
    assert(result.encoder_ffn_width > 0);
    assert(result.decoder_positions_count_max > 0);
    assert(result.decoder_width > 0);
    assert(result.decoder_attention_heads_count > 0);
    assert(result.decoder_layers_count > 0);
    assert(result.decoder_layers_count <= layers_count_max);
    assert(result.decoder_ffn_width > 0);
    assert(result.vocabulary_tokens_count > 0);
    assert(result.encoder_width % result.encoder_attention_heads_count == 0);
    assert(result.decoder_width % result.decoder_attention_heads_count == 0);
    assert(result.encoder_ffn_width == 4 * result.encoder_width);
    assert(result.decoder_ffn_width == 4 * result.decoder_width);
    assert(result.encoder_width % 8 == 0);
    assert(result.decoder_width % 8 == 0);
    assert(result.encoder_ffn_width % 8 == 0);
    assert(result.decoder_ffn_width % 8 == 0);
    assert(result.vocabulary_tokens_count % 8 == 0);

    return result;
}
