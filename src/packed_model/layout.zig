//! `writer.zig` turns one CTranslate2 Whisper model and its vocabulary into one
//! self-contained model file. It widens non-quantized Float16 tensors to the
//! Float32 values consumed by inference, quantizes and rearranges weight tensors
//! for the VNNI kernels, and converts the GPT-2 vocabulary from its
//! byte-to-Unicode representation to token offsets and raw token bytes.
//!
//! Keeping the converted weights and vocabulary in the same checksummed file
//! prevents an installed model from being paired with a different vocabulary.
//!
//! `reader.zig` memory-maps that file and returns inference views directly into
//! its runtime-ready payloads; startup performs no tensor conversion, vocabulary
//! decoding, or payload allocation.
//!
//! This file defines the byte layout used by both operations. `Model.Kind`
//! determines every tensor's dimensions, encoding, and directory capacity, so
//! the file stores only a table of tensor payload offsets rather than repeating
//! that schema. A layout change for an existing kind therefore requires a new
//! `format_version`; otherwise an updated reader could silently reinterpret an
//! already-installed model.

const std = @import("std");
const inference = @import("../inference/root.zig");
const assert = std.debug.assert;

const ModelKind = inference.Model.Kind;
const ModelDimensions = inference.Model.Dimensions;

// ─── Packed Model File ─────────────────────────────────────────────────────
//
// The file stores inference's runtime representation rather than preserving
// CTranslate2's source representation. A fixed directory maps each tensor kind
// and layer to its payload. Vocabulary offsets index the decoded byte payload;
// the final offset is the payload size, so token `i` is the byte range
// `[offsets[i], offsets[i + 1])`.
//
// byte 0
// ┌──────────────────────────────────────┐
// │ fixed 128-byte header                │
// ├──────────────────────────────────────┤ byte 128
// │ tensor payload offset directory      │
// ├──────────────────────────────────────┤
// │ alignment padding                    │
// ├──────────────────────────────────────┤
// │ runtime-ready tensor payloads        │
// │                                      │
// │ • one payload per model tensor       │
// │ • one payload per layer tensor       │
// │ • every payload begins at 64-byte    │
// │   alignment                          │
// ├──────────────────────────────────────┤
// │ vocabulary offset index              │
// │                                      │
// │ (token count + 1) little-endian u32s │
// ├──────────────────────────────────────┤
// │ alignment padding                    │
// ├──────────────────────────────────────┤
// │ decoded vocabulary bytes             │
// ├──────────────────────────────────────┤
// │ final alignment padding              │
// └──────────────────────────────────────┘
//                                       file_size
//
// The 128-byte header identifies the packed format and Whisper model and records
// the variable file boundaries needed before constructing any views. Its BLAKE3
// digest detects corruption anywhere in the file. To avoid making the digest
// depend on itself, the writer and reader treat bytes `[48, 80)` as zero while
// calculating it. Header bytes that carry no value must also be zero.
//
// byte 0
// ┌──────────────────────────────────────┐
// │ magic: "VOICEDP\0"           8 bytes │
// ├──────────────────────────────────────┤ byte 8
// │ format_version: u32                  │
// ├──────────────────────────────────────┤ byte 12
// │ model_kind: u16                      │
// ├──────────────────────────────────────┤ byte 14
// │ reserved: zero                2 bytes│
// ├──────────────────────────────────────┤ byte 16
// │ file_size: u64                       │
// ├──────────────────────────────────────┤ byte 24
// │ vocabulary_index_offset: u64         │
// ├──────────────────────────────────────┤ byte 32
// │ vocabulary_bytes_start_offset: u64   │
// ├──────────────────────────────────────┤ byte 40
// │ vocabulary_bytes_end_offset: u64     │
// ├──────────────────────────────────────┤ byte 48
// │ file_blake3: [32]u8                  │
// ├──────────────────────────────────────┤ byte 80
// │ reserved: zero               48 bytes│
// └──────────────────────────────────────┘ byte 128

pub const alignment: usize = 64;
pub const format_version: u32 = 1;

// Every supported English model uses the same GPT-2 vocabulary. Decoding each
// pinned `vocabulary.txt` produces this exact payload size.
pub const whisper_vocabulary_bytes_size: usize = 334_537;

pub const header_magic = "VOICEDP\x00";
pub const header_size: usize = 128;
pub const header_format_version_offset: usize = 8;
pub const header_model_kind_offset: usize = 12;
pub const header_model_kind_end_offset: usize = 14;
pub const header_file_size_offset: usize = 16;
pub const header_vocabulary_index_offset: usize = 24;
pub const header_vocabulary_bytes_start_offset: usize = 32;
pub const header_vocabulary_bytes_end_offset: usize = 40;
pub const header_file_blake3_offset: usize = 48;
pub const header_reserved_offset: usize = 80;

comptime {
    assert(header_magic.len == header_format_version_offset);
    assert(header_format_version_offset + @sizeOf(u32) == header_model_kind_offset);
    assert(header_model_kind_offset + @sizeOf(u16) == header_model_kind_end_offset);
    assert(header_model_kind_end_offset + 2 == header_file_size_offset);
    assert(header_file_size_offset + @sizeOf(u64) == header_vocabulary_index_offset);
    assert(header_vocabulary_index_offset + @sizeOf(u64) == header_vocabulary_bytes_start_offset);
    assert(header_vocabulary_bytes_start_offset + @sizeOf(u64) == header_vocabulary_bytes_end_offset);
    assert(header_vocabulary_bytes_end_offset + @sizeOf(u64) == header_file_blake3_offset);
    assert(header_file_blake3_offset + std.crypto.hash.Blake3.digest_length == header_reserved_offset);
    assert(header_reserved_offset + 48 == header_size);
    assert(header_size % alignment == 0);
}

pub const tensor_directory_offset = header_size;
pub const tensor_directory_entries_count_max = std.enums.values(TensorSection.Kind).len * inference.Model.layers_count_max;

pub const FileLayout = struct {
    tensor_payloads_start_offset: usize,
    tensor_payloads_end_offset: usize,
    vocabulary_index_offset: usize,
    vocabulary_bytes_start_offset: usize,
    vocabulary_bytes_end_offset: usize,
    size: usize,
};

pub fn calculate(model_kind: ModelKind) FileLayout {
    const dimensions = inference.Model.dimensions(model_kind);

    // ── Reserve The Tensor Directory And Payloads ──
    //
    // The directory has one fixed `u64` offset slot for every tensor kind and
    // possible layer. Tensor payloads begin at the first alignment boundary
    // after that table. Their total size depends only on model dimensions and
    // encodings, not on the order in which the writer appends them.

    const tensor_directory_entries_count = std.enums.values(TensorSection.Kind).len * tensorDirectoryLayerSlotsCount(model_kind);
    const tensor_directory_size = tensor_directory_entries_count * @sizeOf(u64);
    const tensor_payloads_start_offset = std.mem.alignForward(usize, tensor_directory_offset + tensor_directory_size, alignment);
    const tensor_payloads_end_offset = tensor_payloads_start_offset + TensorSection.calculatePayloadsSize(model_kind);

    // ── Place The Vocabulary Index And Bytes ──
    //
    // Every token needs a start offset, and one terminal offset supplies the
    // exclusive end of the final token. The index therefore contains
    // `vocabulary_tokens_count + 1` little-endian `u32` offsets. Token `i`
    // occupies vocabulary bytes `[offsets[i], offsets[i + 1])`.

    const vocabulary_index_offset = tensor_payloads_end_offset;
    const vocabulary_offsets_size = (dimensions.vocabulary_tokens_count + 1) * @sizeOf(u32);
    const vocabulary_bytes_start_offset = std.mem.alignForward(usize, vocabulary_index_offset + vocabulary_offsets_size, alignment);
    const vocabulary_bytes_end_offset = vocabulary_bytes_start_offset + whisper_vocabulary_bytes_size;

    // The complete file ends at the next boundary after the vocabulary bytes;
    // the writer leaves both alignment gaps as zero.
    const size = std.mem.alignForward(usize, vocabulary_bytes_end_offset, alignment);

    return .{
        .tensor_payloads_start_offset = tensor_payloads_start_offset,
        .tensor_payloads_end_offset = tensor_payloads_end_offset,
        .vocabulary_index_offset = vocabulary_index_offset,
        .vocabulary_bytes_start_offset = vocabulary_bytes_start_offset,
        .vocabulary_bytes_end_offset = vocabulary_bytes_end_offset,
        .size = size,
    };
}

/// Kind values 1 and 2 retain the twelve-slot directory defined by packed
/// format version 1, including Base.en's unused slots. New kinds can use their
/// exact layer count without changing the interpretation of an existing packed
/// file.
pub fn tensorDirectoryLayerSlotsCount(model_kind: ModelKind) usize {
    return switch (model_kind) {
        .whisper_base_en, .whisper_small_en => 12,
        .whisper_medium_en => 24,
        .whisper_tiny_en => 4,
    };
}

// ─── Tensor Sections ───────────────────────────────────────────────────────
//
// `TensorSection.Kind` describes a tensor role, while `layer_index` selects the
// concrete layer that owns one tensor with that role. For example, every encoder
// layer has its own independently trained FFN expansion weight:
//
//     TensorSection.Kind
//     .encoder_layer_ffn_expansion_weight
//                       │
//                       ├── layer_index 0 → encoder_layers[0].ffn_expansion_weight
//                       ├── layer_index 1 → encoder_layers[1].ffn_expansion_weight
//                       ├── layer_index 2 → encoder_layers[2].ffn_expansion_weight
//                       ├── ...
//                       └── layer_index N → encoder_layers[N].ffn_expansion_weight
//
// These sections have the same role and dimensions but contain different learned
// values. Small.en has 12 encoder layers, so that one kind expands into 12 file
// sections:
//
//     encoder_layer_ffn_expansion_weight
//     ┌─────────────┬──────────────────────────────────────┐
//     │ layer_index │ packed-file payload                  │
//     ├─────────────┼──────────────────────────────────────┤
//     │ 0           │ expansion weights for encoder layer 0│
//     │ 1           │ expansion weights for encoder layer 1│
//     │ 2           │ expansion weights for encoder layer 2│
//     │ ...         │ ...                                  │
//     │ 11          │ expansion weights for layer 11       │
//     └─────────────┴──────────────────────────────────────┘
//
// The tensor directory is kind-major: it stores every layer slot for one kind
// before advancing to the next kind. Each entry is a little-endian `u64` payload
// offset. Model-wide tensors use slot zero; every inapplicable slot remains zero.
// Payloads may appear in a different order; their directory offsets provide the
// mapping.
//
//     Kind: encoder layer-norm beta
//       ├── slot 0 → layer 0 payload offset
//       ├── slot 1 → layer 1 payload offset
//       └── ...
//
//     Kind: encoder layer-norm gamma
//       ├── slot 0 → layer 0 payload offset
//       ├── slot 1 → layer 1 payload offset
//       └── ...
//
//     Kind: encoder FFN expansion weight
//       ├── slot 0 → layer 0 payload offset
//       ├── slot 1 → layer 1 payload offset
//       └── ...
//
// Model-wide tensors do not belong to a repeated Transformer layer. They produce
// one section with no layer index:
//
//     encoder_convolution_1_weight
//     └── layer_index = null → Model.Weights.encoder_convolution_1_weight
//
// Therefore `layer_index = null` selects a field directly on `Model.Weights`,
// while `layer_index = 0`, `1`, and so on select the corresponding element of
// `encoder_layers` or `decoder_layers`.
//
// Neither kind nor layer identity is serialized beside a payload. Together they
// select one fixed directory slot; that slot stores the payload offset, while
// the shared schema supplies its size and encoding:
//
//     (kind, layer_index) → directory slot → payload offset
//     (Model.Kind, kind)  → schema         → payload size and encoding
//
// `payload_size` excludes the zero padding that aligns the following section.

pub const TensorSection = struct {
    encoding: Encoding,
    dimensions: Dimensions,

    // The section occupies
    // `[payload_offset, payload_offset + payload_size)` in the packed file.
    // `payload_size` includes encoding-internal padding but excludes alignment
    // before the following section.
    payload_offset: usize,
    payload_size: usize,

    // `kind` identifies the corresponding `Model.Weights` field. Declaration
    // order determines directory slots; inference uses direct views after
    // loading and does not traverse tensors in this order.
    //
    // The groups follow Whisper's encoder-then-decoder execution structure.
    pub const Kind = enum {
        // Stored once for the encoder.
        encoder_convolution_1_bias,
        encoder_convolution_1_weight,
        encoder_convolution_2_bias,
        encoder_convolution_2_weight,
        encoder_layer_norm_beta,
        encoder_layer_norm_gamma,
        encoder_position_encodings,

        // Stored once for every encoder layer.
        encoder_layer_self_attention_layer_norm_beta,
        encoder_layer_self_attention_layer_norm_gamma,
        encoder_layer_self_attention_query_key_value_bias,
        encoder_layer_self_attention_query_key_value_weight,
        encoder_layer_self_attention_output_bias,
        encoder_layer_self_attention_output_weight,
        encoder_layer_ffn_layer_norm_beta,
        encoder_layer_ffn_layer_norm_gamma,
        encoder_layer_ffn_expansion_bias,
        encoder_layer_ffn_expansion_weight,
        encoder_layer_ffn_contraction_bias,
        encoder_layer_ffn_contraction_weight,

        // Stored once for the decoder.
        decoder_embeddings_weight,
        decoder_layer_norm_beta,
        decoder_layer_norm_gamma,
        decoder_position_encodings,

        // Stored once for every decoder layer.
        decoder_layer_self_attention_layer_norm_beta,
        decoder_layer_self_attention_layer_norm_gamma,
        decoder_layer_self_attention_query_key_value_bias,
        decoder_layer_self_attention_query_key_value_weight,
        decoder_layer_self_attention_output_bias,
        decoder_layer_self_attention_output_weight,
        decoder_layer_cross_attention_layer_norm_beta,
        decoder_layer_cross_attention_layer_norm_gamma,
        decoder_layer_cross_attention_query_bias,
        decoder_layer_cross_attention_query_weight,
        decoder_layer_cross_attention_key_value_bias,
        decoder_layer_cross_attention_key_value_weight,
        decoder_layer_cross_attention_output_bias,
        decoder_layer_cross_attention_output_weight,
        decoder_layer_ffn_layer_norm_beta,
        decoder_layer_ffn_layer_norm_gamma,
        decoder_layer_ffn_expansion_bias,
        decoder_layer_ffn_expansion_weight,
        decoder_layer_ffn_contraction_bias,
        decoder_layer_ffn_contraction_weight,
    };

    // `encoding` determines how the payload is measured, written, and viewed.
    pub const Encoding = enum {
        // The payload contains contiguous little-endian Float32 values.
        float32,

        // The payload uses the packed representation consumed by
        // `inference.VnniWeight`.
        vnni_o8_k4,
    };

    // `dimensions` preserves the logical CTranslate2 shape used to validate
    // source tensors and calculate payload sizes. The shape is not serialized.
    pub const Dimensions = struct {
        // Active dimensions in outer-to-inner order. Entries after `rank` are
        // zero.
        values: [3]usize,

        // Number of active entries in `values`. Packed Whisper tensors have
        // rank one, two, or three.
        rank: usize,

        pub fn init1(first: usize) Dimensions {
            assert(first > 0);

            return .{ .values = .{ first, 0, 0 }, .rank = 1 };
        }

        pub fn init2(first: usize, second: usize) Dimensions {
            assert(first > 0);
            assert(second > 0);

            return .{ .values = .{ first, second, 0 }, .rank = 2 };
        }

        pub fn init3(first: usize, second: usize, third: usize) Dimensions {
            assert(first > 0);
            assert(second > 0);
            assert(third > 0);

            return .{ .values = .{ first, second, third }, .rank = 3 };
        }

        pub fn slice(dimensions: *const Dimensions) []const usize {
            return dimensions.values[0..dimensions.rank];
        }

        pub fn countElements(dimensions: Dimensions) usize {
            var elements_count: usize = 1;

            for (dimensions.slice()) |dimension| {
                assert(dimension > 0);

                elements_count = std.math.mul(usize, elements_count, dimension) catch {
                    unreachable;
                };
            }

            return elements_count;
        }
    };

    // A VNNI payload contains packed weights and per-output-row metadata.
    pub const VnniPayload = struct {
        // Runtime addressing for the packed O8K4 signed weights.
        weight_layout: inference.VnniWeight.Layout,

        // The packed weight bytes occupy `[0, weights_size)`.
        weights_size: usize,

        // The Float32 scales occupy
        // `[scales_offset, scales_offset + scales_size)`.
        scales_offset: usize,
        scales_size: usize,

        // One Int32 activation compensation value per output row begins here.
        compensation_offset: usize,

        // Complete payload size, including both internal alignment gaps.
        size: usize,

        pub fn calculate(dimensions: Dimensions) VnniPayload {
            assert(dimensions.rank == 2 or dimensions.rank == 3);

            const output_rows_count = dimensions.values[0];
            const input_values_count = @divExact(dimensions.countElements(), output_rows_count);
            const weight_layout = inference.VnniWeight.Layout.init(output_rows_count, input_values_count);
            const weights_size = weight_layout.valuesCount();
            const scales_offset = std.mem.alignForward(usize, weights_size, alignment);

            const scales_size = std.math.mul(usize, output_rows_count, @sizeOf(f32)) catch {
                unreachable;
            };

            const scales_end = std.math.add(usize, scales_offset, scales_size) catch {
                unreachable;
            };

            const compensation_offset = std.mem.alignForward(usize, scales_end, alignment);

            const compensation_size = std.math.mul(usize, output_rows_count, @sizeOf(i32)) catch {
                unreachable;
            };

            const size = std.math.add(usize, compensation_offset, compensation_size) catch {
                unreachable;
            };

            return .{
                .weight_layout = weight_layout,
                .weights_size = weights_size,
                .scales_offset = scales_offset,
                .scales_size = scales_size,
                .compensation_offset = compensation_offset,
                .size = size,
            };
        }
    };

    pub fn layerCount(model_kind: ModelKind, section_kind: Kind) ?usize {
        const model_dimensions = inference.Model.dimensions(model_kind);

        return switch (definition(section_kind, model_dimensions).scope) {
            .model => null,
            .encoder_layer => model_dimensions.encoder_layers_count,
            .decoder_layer => model_dimensions.decoder_layers_count,
        };
    }

    pub fn count(model_kind: ModelKind) usize {
        const model_dimensions = inference.Model.dimensions(model_kind);
        var sections_count: usize = 0;

        for (std.enums.values(Kind)) |section_kind| {
            sections_count += definition(section_kind, model_dimensions).instancesCount(model_dimensions);
        }

        return sections_count;
    }

    pub fn calculate(model_kind: ModelKind, section_kind: Kind, layer_index: ?u16, payload_offset: usize) TensorSection {
        const model_dimensions = inference.Model.dimensions(model_kind);
        const section_definition = definition(section_kind, model_dimensions);

        section_definition.assertValidLayerIndex(model_dimensions, layer_index);

        return .{
            .encoding = section_definition.encoding,
            .dimensions = section_definition.dimensions,
            .payload_offset = payload_offset,
            .payload_size = section_definition.payloadSize(),
        };
    }

    pub fn directoryEntryOffset(model_kind: ModelKind, section_kind: Kind, layer_slot: usize) usize {
        const layer_slots_count = tensorDirectoryLayerSlotsCount(model_kind);
        assert(layer_slot < layer_slots_count);

        const section_kind_index: usize = @intFromEnum(section_kind);
        const directory_index = section_kind_index * layer_slots_count + layer_slot;

        return tensor_directory_offset + directory_index * @sizeOf(u64);
    }

    fn calculatePayloadsSize(model_kind: ModelKind) usize {
        const model_dimensions = inference.Model.dimensions(model_kind);
        var payloads_size: usize = 0;

        for (std.enums.values(Kind)) |section_kind| {
            const section_definition = definition(section_kind, model_dimensions);
            const section_size = std.mem.alignForward(usize, section_definition.payloadSize(), alignment);

            const instances_size = std.math.mul(usize, section_size, section_definition.instancesCount(model_dimensions)) catch {
                unreachable;
            };

            payloads_size = std.math.add(usize, payloads_size, instances_size) catch {
                unreachable;
            };
        }

        return payloads_size;
    }

    // `Scope` determines whether one logical kind expands once or once per
    // corresponding encoder or decoder layer. It is not serialized.
    const Scope = enum {
        model,
        encoder_layer,
        decoder_layer,
    };

    // `Definition` contains the model-dependent facts shared by every concrete
    // layer occurrence of one kind. It is not serialized.
    const Definition = struct {
        // `scope` determines how many concrete sections the model contains.
        scope: Scope,

        // `encoding` determines the payload representation and size calculation.
        encoding: Encoding,

        // `dimensions` is the logical source shape validated by the writer.
        dimensions: Dimensions,

        fn instancesCount(section: Definition, model_dimensions: ModelDimensions) usize {
            return switch (section.scope) {
                .model => 1,
                .encoder_layer => model_dimensions.encoder_layers_count,
                .decoder_layer => model_dimensions.decoder_layers_count,
            };
        }

        fn assertValidLayerIndex(section: Definition, model_dimensions: ModelDimensions, layer_index: ?u16) void {
            switch (section.scope) {
                .model => assert(layer_index == null),
                .encoder_layer => assert(layer_index != null and layer_index.? < model_dimensions.encoder_layers_count),
                .decoder_layer => assert(layer_index != null and layer_index.? < model_dimensions.decoder_layers_count),
            }
        }

        fn payloadSize(section: Definition) usize {
            return switch (section.encoding) {
                .float32 => std.math.mul(usize, section.dimensions.countElements(), @sizeOf(f32)) catch {
                    unreachable;
                },
                .vnni_o8_k4 => VnniPayload.calculate(section.dimensions).size,
            };
        }
    };

    fn definition(kind: Kind, model_dimensions: ModelDimensions) Definition {
        const encoder_width = model_dimensions.encoder_width;
        const decoder_width = model_dimensions.decoder_width;
        const encoder_ffn_width = model_dimensions.encoder_ffn_width;
        const decoder_ffn_width = model_dimensions.decoder_ffn_width;

        return switch (kind) {
            .encoder_convolution_1_bias => float32Definition(.model, .init1(encoder_width)),
            .encoder_convolution_1_weight => vnniDefinition(.model, .init3(encoder_width, model_dimensions.mel_bins_count, 3)),
            .encoder_convolution_2_bias => float32Definition(.model, .init1(encoder_width)),
            .encoder_convolution_2_weight => vnniDefinition(.model, .init3(encoder_width, encoder_width, 3)),
            .encoder_layer_norm_beta, .encoder_layer_norm_gamma => float32Definition(.model, .init1(encoder_width)),
            .encoder_position_encodings => float32Definition(.model, .init2(model_dimensions.encoder_positions_count_max, encoder_width)),

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

            .decoder_embeddings_weight => vnniDefinition(.model, .init2(model_dimensions.vocabulary_tokens_count, decoder_width)),
            .decoder_layer_norm_beta, .decoder_layer_norm_gamma => float32Definition(.model, .init1(decoder_width)),
            .decoder_position_encodings => float32Definition(.model, .init2(model_dimensions.decoder_positions_count_max, decoder_width)),

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

    fn float32Definition(scope: Scope, dimensions: Dimensions) Definition {
        return .{ .scope = scope, .encoding = .float32, .dimensions = dimensions };
    }

    fn vnniDefinition(scope: Scope, dimensions: Dimensions) Definition {
        return .{ .scope = scope, .encoding = .vnni_o8_k4, .dimensions = dimensions };
    }
};
