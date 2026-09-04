const std = @import("std");
const assert = std.debug.assert;

pub const layers_count_max: usize = 12;

/// `ModelKind` selects one supported English Whisper architecture. Its numeric
/// values are part of the packed-image ABI and must not be reordered or reused.
pub const ModelKind = enum(u16) {
    base_en = 1,
    small_en = 2,

    /// `specification` returns the fixed tensor dimensions for the model kind.
    pub fn specification(kind: ModelKind) ModelSpecification {
        const model_specification: ModelSpecification = switch (kind) {
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

        assert(model_specification.mel_bins_count > 0);
        assert(model_specification.encoder_positions_count_max > 0);
        assert(model_specification.encoder_width > 0);
        assert(model_specification.encoder_attention_heads_count > 0);
        assert(model_specification.encoder_layers_count > 0);
        assert(model_specification.encoder_layers_count <= layers_count_max);
        assert(model_specification.encoder_ffn_width > 0);
        assert(model_specification.decoder_positions_count_max > 0);
        assert(model_specification.decoder_width > 0);
        assert(model_specification.decoder_attention_heads_count > 0);
        assert(model_specification.decoder_layers_count > 0);
        assert(model_specification.decoder_layers_count <= layers_count_max);
        assert(model_specification.decoder_ffn_width > 0);
        assert(model_specification.vocabulary_tokens_count > 0);
        assert(model_specification.encoder_width % model_specification.encoder_attention_heads_count == 0);
        assert(model_specification.decoder_width % model_specification.decoder_attention_heads_count == 0);
        assert(model_specification.encoder_ffn_width == 4 * model_specification.encoder_width);
        assert(model_specification.decoder_ffn_width == 4 * model_specification.decoder_width);
        assert(model_specification.encoder_width % 8 == 0);
        assert(model_specification.decoder_width % 8 == 0);
        assert(model_specification.encoder_ffn_width % 8 == 0);
        assert(model_specification.decoder_ffn_width % 8 == 0);
        assert(model_specification.vocabulary_tokens_count % 8 == 0);

        return model_specification;
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
