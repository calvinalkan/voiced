//! Maps one packed model and constructs inference views over its runtime-ready
//! tensor and vocabulary payloads.
//!
//! File bytes are untrusted; `layout.zig` and `inference.Model.Kind` provide the
//! authoritative shape. `load` validates every byte-derived range before any
//! pointer or slice escapes in the returned model.

const std = @import("std");
const inference = @import("../inference/root.zig");
const layout = @import("layout.zig");
const assert = std.debug.assert;
const Blake3 = std.crypto.hash.Blake3;

/// `load` maps and validates `file_name` from `directory` for `expected_kind`.
/// The returned model owns the mapping; the caller releases it with
/// `inference.Model.deinit`.
pub fn load(io: std.Io, directory: std.Io.Dir, file_name: []const u8, expected_kind: inference.Model.Kind) !inference.Model {
    var model: inference.Model = undefined;
    try loadInto(io, directory, file_name, expected_kind, &model);

    return model;
}

// PERFORMANCE: Keep validation errors in a small !void result. Returning them
// directly as !Model made Zig 0.16/LLVM emit five 11,808-byte constants containing
// unused model payloads. This helper saved 57.8 KiB in the stripped static binary
// (application/stdlib ReleaseSmall, packed-model code ReleaseFast). Preserve this
// boundary when changing validation. The complete model is assigned only after
// every check succeeds; on error, load returns without reading its local model.
fn loadInto(io: std.Io, directory: std.Io.Dir, file_name: []const u8, expected_kind: inference.Model.Kind, model: *inference.Model) !void {
    // ── Map The Exact Expected File ──
    //
    // The model kind determines the complete file size. Rejecting every other
    // size before `mmap` makes all fixed header reads below safe without trusting
    // a length stored inside the file.

    const expected = layout.calculate(expected_kind);

    const file = try directory.openFile(io, file_name, .{ .mode = .read_only, .allow_directory = false });
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) {
        // Only regular files provide the stable extent required by the mapping.
        return error.InvalidPackedModel;
    }

    if (stat.size != expected.size) {
        // The expected model kind fixes the exact packed-file size.
        return error.InvalidPackedModel;
    }

    const mapping = try std.posix.mmap(
        null,
        expected.size,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    errdefer std.posix.munmap(mapping);

    const bytes: []align(layout.alignment) const u8 = @alignCast(mapping);

    // ── Validate The Header ──

    if (!std.mem.eql(u8, bytes[0..layout.header_format_version_offset], layout.header_magic)) {
        // Without the magic, these bytes are not a Voiced packed model.
        return error.InvalidPackedModel;
    }

    if (readInt(u32, bytes, layout.header_format_version_offset) != layout.format_version) {
        // Another format version requires a reader for that packed-model ABI.
        return error.UnsupportedPackedModelFormatVersion;
    }

    if (!std.mem.allEqual(u8, bytes[layout.header_model_kind_end_offset..layout.header_file_size_offset], 0)) {
        // Nonzero reserved bytes make this header noncanonical.
        return error.InvalidPackedModel;
    }

    if (!std.mem.allEqual(u8, bytes[layout.header_reserved_offset..layout.header_size], 0)) {
        // The trailing reserve must remain zero until a later format assigns it.
        return error.InvalidPackedModel;
    }

    const kind = std.enums.fromInt(
        inference.Model.Kind,
        readInt(u16, bytes, layout.header_model_kind_offset),
    ) orelse {
        // The stored integer does not identify a model supported by this build.
        return error.UnsupportedModelKind;
    };

    if (kind != expected_kind) {
        // Callers select the model explicitly; loading another kind would pair
        // the wrong dimensions and runtime policy with these weights.
        return error.UnexpectedModelKind;
    }

    if (readInt(u64, bytes, layout.header_file_size_offset) != expected.size) {
        return error.InvalidPackedModel;
    }

    if (readInt(u64, bytes, layout.header_vocabulary_index_offset) != expected.vocabulary_index_offset) {
        return error.InvalidPackedModel;
    }

    if (readInt(u64, bytes, layout.header_vocabulary_bytes_start_offset) != expected.vocabulary_bytes_start_offset) {
        return error.InvalidPackedModel;
    }

    if (readInt(u64, bytes, layout.header_vocabulary_bytes_end_offset) != expected.vocabulary_bytes_end_offset) {
        return error.InvalidPackedModel;
    }

    // ── Validate The Whole-File Digest ──
    //
    // The writer hashes the digest field while it is still zero. Substitute the
    // same zero bytes here so the digest covers every other header and payload
    // byte without depending on itself.

    var hash = Blake3.init(.{});
    hash.update(bytes[0..layout.header_file_blake3_offset]);
    hash.update(&([_]u8{0} ** Blake3.digest_length));
    hash.update(bytes[layout.header_file_blake3_offset + Blake3.digest_length ..]);

    var digest: [Blake3.digest_length]u8 = undefined;
    hash.final(&digest);

    if (!std.mem.eql(u8, &digest, bytes[layout.header_file_blake3_offset..layout.header_reserved_offset])) {
        // A mismatch means at least one packed byte changed after conversion.
        return error.PackedModelChecksumMismatch;
    }

    // ── Validate The Tensor Directory ──

    try validateTensorDirectory(bytes, kind, expected);

    // ── Construct And Validate The Vocabulary ──
    //
    // The packed format is the native little-endian x86-64 runtime format. Its
    // aligned vocabulary index can therefore become a borrowed `u32` view with
    // no decoding or allocation.

    const dimensions = inference.Model.dimensions(kind);

    const vocabulary_offsets_pointer: [*]const u32 = @ptrCast(
        @alignCast(bytes.ptr + expected.vocabulary_index_offset),
    );

    const vocabulary: inference.Model.Vocabulary = .{
        .offsets = vocabulary_offsets_pointer[0 .. dimensions.vocabulary_tokens_count + 1],
        .bytes = bytes[expected.vocabulary_bytes_start_offset..expected.vocabulary_bytes_end_offset],
    };
    if (vocabulary.offsets[0] != 0) {
        // Token zero must begin at the first vocabulary payload byte.
        return error.InvalidPackedModel;
    }
    if (vocabulary.offsets[vocabulary.offsets.len - 1] != layout.whisper_vocabulary_bytes_size) {
        // The terminal offset must cover the complete vocabulary payload.
        return error.InvalidPackedModel;
    }

    // Adjacent offsets define each token's half-open byte range. Monotonicity
    // keeps every range forward and inside the one validated vocabulary payload.

    for (
        vocabulary.offsets[1..],
        vocabulary.offsets[0 .. vocabulary.offsets.len - 1],
    ) |token_bytes_end_offset, token_bytes_start_offset| {
        if (token_bytes_end_offset < token_bytes_start_offset) {
            // Reversed offsets would create an invalid token byte range.
            return error.InvalidPackedModel;
        }

        if (token_bytes_end_offset > layout.whisper_vocabulary_bytes_size) {
            // Every token must end inside the vocabulary payload.
            return error.InvalidPackedModel;
        }
    }

    // ── Publish The Model Views ──
    //
    // No fallible work remains. Tensor construction performs only validated
    // directory lookups, and the model takes ownership of `mapping` on return.

    model.* = .{
        .backing_storage = .{ .mapped = mapping },
        .kind = kind,
        .weights = modelWeights(bytes, kind),
        .vocabulary = vocabulary,
    };
}

fn validateTensorDirectory(
    bytes: []align(layout.alignment) const u8,
    kind: inference.Model.Kind,
    file_layout: layout.FileLayout,
) !void {
    // The directory is a rectangle of tensor kinds by maximum layer slots.
    // Model-wide kinds use only slot zero; layer kinds use the active layer
    // prefix. Every remaining slot must retain the writer's zero sentinel.
    //
    // Schema-derived padded sizes exactly fill the payload region. Proving that
    // every required range is in bounds and disjoint therefore also proves that
    // the ranges form one complete tiling, regardless of their source-file order.

    const PayloadRange = struct { start: usize, end: usize };

    var payload_ranges: [layout.tensor_directory_entries_count_max]PayloadRange = undefined;
    var payload_ranges_count: usize = 0;

    for (std.enums.values(layout.TensorSection.Kind)) |section_kind| {
        const layers_count = layout.TensorSection.layerCount(kind, section_kind);
        const required_entries_count = layers_count orelse 1;

        for (0..layout.tensorDirectoryLayerSlotsCount(kind)) |layer_slot_index| {
            const directory_entry_offset = layout.TensorSection.directoryEntryOffset(kind, section_kind, layer_slot_index);
            const stored_payload_offset = readInt(u64, bytes, directory_entry_offset);
            const entry_is_required = layer_slot_index < required_entries_count;

            const unused_entry_is_nonzero = !entry_is_required and stored_payload_offset != 0;
            if (unused_entry_is_nonzero) {
                // A nonzero unused slot would assign a second meaning to bytes
                // outside this model kind's tensor set.
                return error.InvalidPackedModel;
            }

            if (!entry_is_required) {
                // This canonical zero slot contributes no payload range.
                continue;
            }

            const layer_index: ?u16 = if (layers_count != null) @intCast(layer_slot_index) else null;

            const payload_offset = std.math.cast(usize, stored_payload_offset) orelse {
                // The file offset cannot be represented by this runtime target.
                return error.InvalidPackedModel;
            };

            if (payload_offset % layout.alignment != 0) {
                // Tensor views require the packed format's 64-byte alignment.
                return error.InvalidPackedModel;
            }
            if (payload_offset < file_layout.tensor_payloads_start_offset) {
                // A tensor may not point backward into the header or directory.
                return error.InvalidPackedModel;
            }
            if (payload_offset > file_layout.tensor_payloads_end_offset) {
                // A tensor may not start beyond the tensor payload region.
                return error.InvalidPackedModel;
            }

            const section = layout.TensorSection.calculate(kind, section_kind, layer_index, payload_offset);

            const padded_payload_size = std.mem.alignForward(usize, section.payload_size, layout.alignment);
            if (padded_payload_size > file_layout.tensor_payloads_end_offset - payload_offset) {
                // The complete payload and its canonical padding must fit before
                // the vocabulary index begins.
                return error.InvalidPackedModel;
            }

            const payload_range: PayloadRange = .{ .start = payload_offset, .end = payload_offset + padded_payload_size };

            for (payload_ranges[0..payload_ranges_count]) |existing_payload_range| {
                const payload_ranges_overlap = payload_range.start < existing_payload_range.end and
                    existing_payload_range.start < payload_range.end;

                if (payload_ranges_overlap) {
                    // Each directory entry owns distinct bytes; aliasing would
                    // make two logical tensors share one payload.
                    return error.InvalidPackedModel;
                }
            }

            payload_ranges[payload_ranges_count] = payload_range;
            payload_ranges_count += 1;
        }
    }

    assert(payload_ranges_count == layout.TensorSection.count(kind));
}

// ─── Runtime Tensor Views ─────────────────────────────────────────────────
//
// These constructors assume `validateTensorDirectory` has accepted every
// directory offset and payload range. They perform no further validation, so
// `load` must not call them before that phase succeeds.

fn modelWeights(
    bytes: []align(layout.alignment) const u8,
    kind: inference.Model.Kind,
) inference.Model.Weights {
    const dimensions = inference.Model.dimensions(kind);

    // ── Construct Encoder Layer Views ──
    //
    // Only the architecture's active array prefix is initialized. Inference
    // uses the same model dimensions as its iteration bounds and never reads the
    // undefined tail.

    var encoder_layers: [inference.Model.layers_count_max]inference.Model.Weights.EncoderLayer = undefined;

    for (encoder_layers[0..dimensions.encoder_layers_count], 0..) |*layer, layer_index| {
        layer.* = .{
            .self_attention_layer_norm_beta = floatValues(bytes, kind, .encoder_layer_self_attention_layer_norm_beta, @intCast(layer_index)),
            .self_attention_layer_norm_gamma = floatValues(bytes, kind, .encoder_layer_self_attention_layer_norm_gamma, @intCast(layer_index)),
            .self_attention_query_key_value_bias = floatValues(bytes, kind, .encoder_layer_self_attention_query_key_value_bias, @intCast(layer_index)),
            .self_attention_query_key_value_weight = vnniWeight(bytes, kind, .encoder_layer_self_attention_query_key_value_weight, @intCast(layer_index)),
            .self_attention_output_bias = floatValues(bytes, kind, .encoder_layer_self_attention_output_bias, @intCast(layer_index)),
            .self_attention_output_weight = vnniWeight(bytes, kind, .encoder_layer_self_attention_output_weight, @intCast(layer_index)),
            .ffn_layer_norm_beta = floatValues(bytes, kind, .encoder_layer_ffn_layer_norm_beta, @intCast(layer_index)),
            .ffn_layer_norm_gamma = floatValues(bytes, kind, .encoder_layer_ffn_layer_norm_gamma, @intCast(layer_index)),
            .ffn_expansion_bias = floatValues(bytes, kind, .encoder_layer_ffn_expansion_bias, @intCast(layer_index)),
            .ffn_expansion_weight = vnniWeight(bytes, kind, .encoder_layer_ffn_expansion_weight, @intCast(layer_index)),
            .ffn_contraction_bias = floatValues(bytes, kind, .encoder_layer_ffn_contraction_bias, @intCast(layer_index)),
            .ffn_contraction_weight = vnniWeight(bytes, kind, .encoder_layer_ffn_contraction_weight, @intCast(layer_index)),
        };
    }

    // ── Construct Decoder Layer Views ──

    var decoder_layers: [inference.Model.layers_count_max]inference.Model.Weights.DecoderLayer = undefined;

    for (decoder_layers[0..dimensions.decoder_layers_count], 0..) |*layer, layer_index| {
        layer.* = .{
            .self_attention_layer_norm_beta = floatValues(bytes, kind, .decoder_layer_self_attention_layer_norm_beta, @intCast(layer_index)),
            .self_attention_layer_norm_gamma = floatValues(bytes, kind, .decoder_layer_self_attention_layer_norm_gamma, @intCast(layer_index)),
            .self_attention_query_key_value_bias = floatValues(bytes, kind, .decoder_layer_self_attention_query_key_value_bias, @intCast(layer_index)),
            .self_attention_query_key_value_weight = vnniWeight(bytes, kind, .decoder_layer_self_attention_query_key_value_weight, @intCast(layer_index)),
            .self_attention_output_bias = floatValues(bytes, kind, .decoder_layer_self_attention_output_bias, @intCast(layer_index)),
            .self_attention_output_weight = vnniWeight(bytes, kind, .decoder_layer_self_attention_output_weight, @intCast(layer_index)),
            .cross_attention_layer_norm_beta = floatValues(bytes, kind, .decoder_layer_cross_attention_layer_norm_beta, @intCast(layer_index)),
            .cross_attention_layer_norm_gamma = floatValues(bytes, kind, .decoder_layer_cross_attention_layer_norm_gamma, @intCast(layer_index)),
            .cross_attention_query_bias = floatValues(bytes, kind, .decoder_layer_cross_attention_query_bias, @intCast(layer_index)),
            .cross_attention_query_weight = vnniWeight(bytes, kind, .decoder_layer_cross_attention_query_weight, @intCast(layer_index)),
            .cross_attention_key_value_bias = floatValues(bytes, kind, .decoder_layer_cross_attention_key_value_bias, @intCast(layer_index)),
            .cross_attention_key_value_weight = vnniWeight(bytes, kind, .decoder_layer_cross_attention_key_value_weight, @intCast(layer_index)),
            .cross_attention_output_bias = floatValues(bytes, kind, .decoder_layer_cross_attention_output_bias, @intCast(layer_index)),
            .cross_attention_output_weight = vnniWeight(bytes, kind, .decoder_layer_cross_attention_output_weight, @intCast(layer_index)),
            .ffn_layer_norm_beta = floatValues(bytes, kind, .decoder_layer_ffn_layer_norm_beta, @intCast(layer_index)),
            .ffn_layer_norm_gamma = floatValues(bytes, kind, .decoder_layer_ffn_layer_norm_gamma, @intCast(layer_index)),
            .ffn_expansion_bias = floatValues(bytes, kind, .decoder_layer_ffn_expansion_bias, @intCast(layer_index)),
            .ffn_expansion_weight = vnniWeight(bytes, kind, .decoder_layer_ffn_expansion_weight, @intCast(layer_index)),
            .ffn_contraction_bias = floatValues(bytes, kind, .decoder_layer_ffn_contraction_bias, @intCast(layer_index)),
            .ffn_contraction_weight = vnniWeight(bytes, kind, .decoder_layer_ffn_contraction_weight, @intCast(layer_index)),
        };
    }

    // ── Construct Model-Wide Views ──

    return .{
        .decoder_embeddings_weight = vnniWeight(bytes, kind, .decoder_embeddings_weight, null),
        .decoder_layer_norm_beta = floatValues(bytes, kind, .decoder_layer_norm_beta, null),
        .decoder_layer_norm_gamma = floatValues(bytes, kind, .decoder_layer_norm_gamma, null),
        .decoder_position_encodings = floatValues(bytes, kind, .decoder_position_encodings, null),
        .encoder_convolution_1_bias = floatValues(bytes, kind, .encoder_convolution_1_bias, null),
        .encoder_convolution_1_weight = vnniWeight(bytes, kind, .encoder_convolution_1_weight, null),
        .encoder_convolution_2_bias = floatValues(bytes, kind, .encoder_convolution_2_bias, null),
        .encoder_convolution_2_weight = vnniWeight(bytes, kind, .encoder_convolution_2_weight, null),
        .encoder_layer_norm_beta = floatValues(bytes, kind, .encoder_layer_norm_beta, null),
        .encoder_layer_norm_gamma = floatValues(bytes, kind, .encoder_layer_norm_gamma, null),
        .encoder_position_encodings = floatValues(bytes, kind, .encoder_position_encodings, null),
        .encoder_layers = encoder_layers,
        .decoder_layers = decoder_layers,
    };
}

fn floatValues(
    bytes: []align(layout.alignment) const u8,
    kind: inference.Model.Kind,
    section_kind: layout.TensorSection.Kind,
    layer_index: ?u16,
) []const f32 {
    const section = tensorSection(bytes, kind, section_kind, layer_index);
    assert(section.encoding == .float32);

    const values_count = @divExact(section.payload_size, @sizeOf(f32));
    const values: [*]const f32 = @ptrCast(@alignCast(bytes.ptr + section.payload_offset));

    return values[0..values_count];
}

fn vnniWeight(
    bytes: []align(layout.alignment) const u8,
    kind: inference.Model.Kind,
    section_kind: layout.TensorSection.Kind,
    layer_index: ?u16,
) inference.VnniWeight.QuantizedWeight {
    const section = tensorSection(bytes, kind, section_kind, layer_index);
    assert(section.encoding == .vnni_o8_k4);

    // One VNNI section contains three separately aligned arrays. Rebuild their
    // slices from the same sub-layout that the writer used to pack them.
    const vnni_layout = layout.TensorSection.VnniPayload.calculate(section.dimensions);
    const payload = bytes[section.payload_offset..][0..section.payload_size];
    const values: [*]const i8 = @ptrCast(payload.ptr);
    const scales: [*]const f32 = @ptrCast(@alignCast(payload.ptr + vnni_layout.scales_offset));
    const compensation: [*]const i32 = @ptrCast(@alignCast(payload.ptr + vnni_layout.compensation_offset));

    return .{
        .values = values[0..vnni_layout.weights_size],
        .scales = scales[0..vnni_layout.weight_layout.output_rows_count],
        .compensation = compensation[0..vnni_layout.weight_layout.output_rows_count],
        .input_values_count = vnni_layout.weight_layout.input_values_count,
    };
}

fn tensorSection(
    bytes: []const u8,
    kind: inference.Model.Kind,
    section_kind: layout.TensorSection.Kind,
    layer_index: ?u16,
) layout.TensorSection {
    const layer_slot_index: usize = if (layer_index) |index| @intCast(index) else 0;
    const directory_entry_offset = layout.TensorSection.directoryEntryOffset(kind, section_kind, layer_slot_index);

    // Directory validation already proved that every required `u64` offset fits
    // `usize`; view construction can use the direct cast without another branch.
    const payload_offset: usize = @intCast(readInt(u64, bytes, directory_entry_offset));

    return layout.TensorSection.calculate(kind, section_kind, layer_index, payload_offset);
}

fn readInt(
    comptime Int: type,
    source: []const u8,
    byte_offset: usize,
) Int {
    assert(byte_offset <= source.len);
    assert(@sizeOf(Int) <= source.len - byte_offset);

    return std.mem.readInt(Int, source[byte_offset..][0..@sizeOf(Int)], .little);
}
