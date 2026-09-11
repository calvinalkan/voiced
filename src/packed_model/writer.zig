//! Converts canonical CTranslate2 Whisper weights and GPT-2 vocabulary text into
//! Voiced's runtime-ready packed model.
//!
//! The source bytes are untrusted and remain borrowed. Conversion validates the
//! complete CTranslate2 stream in its serialized order, builds every packed
//! tensor and vocabulary view in owned scratch output, then writes the completed
//! checksummed file. No CTranslate2 representation escapes this module.

const std = @import("std");
const inference = @import("../inference/root.zig");
const layout = @import("layout.zig");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;
const alignment = layout.alignment;

const ModelKind = inference.Model.Kind;
const ModelDimensions = inference.Model.Dimensions;
const TensorSection = layout.TensorSection;
const SectionKind = TensorSection.Kind;
const TensorDimensions = TensorSection.Dimensions;
const VnniPayload = TensorSection.VnniPayload;

const ConvertError = error{
    OutOfMemory,
    InvalidCTranslate2Weights,
    InvalidCTranslate2Vocabulary,
};
const WriteError = ConvertError || std.Io.File.Writer.Error;

/// `writeFromCTranslate2` validates one canonical CTranslate2 Whisper model and
/// writes its complete packed representation to `output_file`. It neither closes
/// the file nor retains either source slice after returning. Caller is responsible
/// for closing `output_file`; and atomic persistence.
pub fn writeFromCTranslate2(
    io: std.Io,
    allocator: Allocator,
    output_file: std.Io.File,
    kind: ModelKind,
    weights_bytes: []const u8,
    vocabulary_text: []const u8,
) WriteError!void {
    // ── Decode The GPT-2 Vocabulary ──

    const prepared_vocabulary = try prepareVocabulary(allocator, kind, vocabulary_text);
    defer prepared_vocabulary.deinit(allocator);

    // ── Read And Validate The Source Header ──

    var reader: WeightsReader = .{ .remaining = weights_bytes };

    const binary_version = try reader.readInt(u32);

    if (binary_version != binary_version_supported) {
        return error.InvalidCTranslate2Weights;
    }

    const specification_name = try reader.readString();
    if (!std.mem.eql(u8, specification_name, "WhisperSpec")) {
        return error.InvalidCTranslate2Weights;
    }

    const specification_revision = try reader.readInt(u32);
    if (specification_revision != whisper_specification_revision) {
        return error.InvalidCTranslate2Weights;
    }

    const source_tensors_count = try reader.readInt(u32);
    const dimensions = inference.Model.dimensions(kind);

    if (source_tensors_count != expectedTensorsCount(dimensions)) {
        return error.InvalidCTranslate2Weights;
    }

    // ── Allocate The Packed Output ──

    var output = try Output.init(allocator, kind);
    defer output.deinit();

    // ── Convert Decoder Weights ──
    //
    // The source serializes decoder records before encoder records. These
    // plans make that order visible and select each scalar check or packed
    // tensor section. A name mismatch rejects missing, duplicate, reordered,
    // or unknown records at its first position.

    // PERFORMANCE: Keep these schema loops non-inline and pass plans as runtime
    // values. Conversion is a cold setup operation; inline loops and comptime
    // plans duplicated record parsing, name formatting, and validation. Sharing
    // that code saved 18.1 KiB in the stripped static Zig 0.16/LLVM binary
    // (application/stdlib ReleaseSmall, packed-model code ReleaseFast). Paired
    // Base.en/Small.en conversion timings on an i7-13700HX, with memory-backed
    // output, stayed within 0.4% at the median.
    // Dispatch happens once per tensor; the loops over its weights are unchanged.
    for (decoder_tensors_before_layers) |plan| {
        try convertTensor(&reader, &output, dimensions, null, plan.name, plan.handling);
    }

    for (0..dimensions.decoder_layers_count) |layer_ordinal| {
        const layer_index = sourceLayerIndexFromOrdinal(layer_ordinal, dimensions.decoder_layers_count);

        for (decoder_layer_tensors) |plan| {
            try convertLayerTensor(&reader, &output, dimensions, "decoder", layer_index, plan);
        }
    }

    for (decoder_tensors_after_layers) |plan| {
        try convertTensor(&reader, &output, dimensions, null, plan.name, plan.handling);
    }

    // ── Convert Encoder Weights ──

    for (encoder_tensors_before_layers) |plan| {
        try convertTensor(&reader, &output, dimensions, null, plan.name, plan.handling);
    }

    for (0..dimensions.encoder_layers_count) |layer_ordinal| {
        const layer_index = sourceLayerIndexFromOrdinal(layer_ordinal, dimensions.encoder_layers_count);

        for (encoder_layer_tensors) |plan| {
            try convertLayerTensor(&reader, &output, dimensions, "encoder", layer_index, plan);
        }
    }

    for (encoder_tensors_after_layers) |plan| {
        try convertTensor(&reader, &output, dimensions, null, plan.name, plan.handling);
    }

    // ── Validate The Tied-Embedding Alias ──
    //
    // Whisper uses its decoder embedding matrix again for vocabulary
    // projection. The packed model stores one matrix for both operations.

    const aliases_count = try reader.readInt(u32);
    if (aliases_count != 1) {
        // Canonical Whisper has exactly one tied-weight alias.
        return error.InvalidCTranslate2Weights;
    }

    const alias = try reader.readString();
    if (!std.mem.eql(u8, alias, "decoder/projection/weight")) {
        // Only the vocabulary projection may name the shared tensor.
        return error.InvalidCTranslate2Weights;
    }

    const aliased_tensor = try reader.readString();
    if (!std.mem.eql(u8, aliased_tensor, "decoder/embeddings/weight")) {
        // Projection must reuse the embedding matrix that conversion retained.
        return error.InvalidCTranslate2Weights;
    }

    if (reader.remaining.len != 0) {
        // A canonical stream ends immediately after the alias table.
        return error.InvalidCTranslate2Weights;
    }

    // ── Finalize And Write ──

    const packed_bytes = output.finish(prepared_vocabulary);

    try output_file.writeStreamingAll(io, packed_bytes);
}

// ─── GPT-2 Byte Vocabulary ────────────────────────────────────────────────
//
// CTranslate2's `vocabulary.txt` stores one UTF-8 token per line. GPT-2's
// byte-level BPE first maps every possible source byte to one printable Unicode
// codepoint, so a token can pass through ordinary Unicode text without losing
// arbitrary byte values:
//
//   source byte                         vocabulary codepoint
//   '!'...'~'                           unchanged
//   '¡'...'¬'                           unchanged
//   '®'...'ÿ'                           unchanged
//   every remaining byte, in byte order U+0100, U+0101, ...
//
// For example, byte `0x20` (space) becomes U+0120 (`Ġ`) and byte `0x0A`
// (line feed) becomes U+010A (`Ċ`). Encoding line feed this way leaves the raw
// `\n` byte available exclusively as the token delimiter. The packed vocabulary
// reverses this mapping and stores the original token bytes plus an offset index.

const PreparedVocabulary = struct {
    offsets: []u32,
    bytes: []u8,

    fn deinit(prepared: PreparedVocabulary, allocator: Allocator) void {
        allocator.free(prepared.offsets);
        allocator.free(prepared.bytes);
    }
};

fn prepareVocabulary(
    allocator: Allocator,
    kind: inference.Model.Kind,
    vocabulary_text: []const u8,
) ConvertError!PreparedVocabulary {
    const tokens_count = inference.Model.dimensions(kind).vocabulary_tokens_count;

    // Both supported models use the same fixed GPT-2 byte payload. Allocate the
    // final sizes up front, then validate and decode the source in one pass.
    const offsets = try allocator.alloc(u32, tokens_count + 1);
    errdefer allocator.free(offsets);

    const token_bytes = try allocator.alloc(u8, layout.whisper_vocabulary_bytes_size);
    errdefer allocator.free(token_bytes);

    var token_count: usize = 0;
    var byte_offset: usize = 0;
    var line_start: usize = 0;

    while (line_start < vocabulary_text.len) {
        if (token_count == tokens_count) {
            // The source contains more token lines than this model kind permits.
            return error.InvalidCTranslate2Vocabulary;
        }

        offsets[token_count] = @intCast(byte_offset);

        const line_end = std.mem.indexOfScalarPos(u8, vocabulary_text, line_start, '\n') orelse vocabulary_text.len;
        const token_text = vocabulary_text[line_start..line_end];

        const token_view = std.unicode.Utf8View.init(token_text) catch {
            // Every vocabulary token must be valid UTF-8 before GPT-2 decoding.
            return error.InvalidCTranslate2Vocabulary;
        };

        var codepoints = token_view.iterator();

        while (codepoints.nextCodepoint()) |codepoint| {
            const byte = gpt2ByteFromCodepoint(codepoint) orelse {
                // GPT-2's alphabet contains only the 256 codepoints documented
                // above; any other codepoint cannot represent a source byte.
                return error.InvalidCTranslate2Vocabulary;
            };

            if (byte_offset == token_bytes.len) {
                // Decoding may not exceed the packed format's fixed byte payload.
                return error.InvalidCTranslate2Vocabulary;
            }

            token_bytes[byte_offset] = byte;
            byte_offset += 1;
        }

        token_count += 1;
        line_start = line_end + @intFromBool(line_end < vocabulary_text.len);
    }

    if (token_count != tokens_count) {
        // The source ended before supplying every model vocabulary token.
        return error.InvalidCTranslate2Vocabulary;
    }

    if (byte_offset != token_bytes.len) {
        // The decoded tokens must exactly fill the format's vocabulary payload.
        return error.InvalidCTranslate2Vocabulary;
    }

    offsets[token_count] = @intCast(byte_offset);

    return .{ .offsets = offsets, .bytes = token_bytes };
}

const gpt2_bytes_count = @as(usize, std.math.maxInt(u8)) + 1;
const gpt2_direct_bytes_count = ('~' - '!' + 1) + ('¬' - '¡' + 1) + ('ÿ' - '®' + 1);
const gpt2_remapped_bytes_count = gpt2_bytes_count - gpt2_direct_bytes_count;
const gpt2_codepoints_count = gpt2_bytes_count + gpt2_remapped_bytes_count;

// Invert GPT-2's byte encoder at compile time. Runtime vocabulary decoding is
// then one bounds check and one table lookup per Unicode codepoint rather than a
// scan over all 256 source bytes.
const gpt2_byte_by_codepoint: [gpt2_codepoints_count]?u8 = mapping: {
    var byte_by_codepoint: [gpt2_codepoints_count]?u8 = @splat(null);
    var next_remapped_codepoint = gpt2_bytes_count;

    for (0..gpt2_bytes_count) |byte_value| {
        const byte: u8 = @intCast(byte_value);

        const codepoint = if (gpt2ByteIsDirect(byte)) direct: {
            break :direct byte_value;
        } else remapped: {
            defer next_remapped_codepoint += 1;

            break :remapped next_remapped_codepoint;
        };

        assert(byte_by_codepoint[codepoint] == null);

        byte_by_codepoint[codepoint] = byte;
    }

    assert(next_remapped_codepoint == gpt2_codepoints_count);

    break :mapping byte_by_codepoint;
};

fn gpt2ByteFromCodepoint(codepoint: u21) ?u8 {
    if (codepoint >= gpt2_byte_by_codepoint.len) {
        // GPT-2's remapped alphabet ends before this Unicode codepoint.
        return null;
    }

    return gpt2_byte_by_codepoint[codepoint];
}

fn gpt2ByteIsDirect(byte: u8) bool {
    return (byte >= '!' and byte <= '~') or
        (byte >= '¡' and byte <= '¬') or
        (byte >= '®' and byte <= 'ÿ');
}

// ─── Packed Output Construction ───────────────────────────────────────────
//
// `Output` owns one zero-initialized final-size buffer. Tensor conversion claims
// payloads in source order and records each location in the fixed directory.
// `finish` publishes the vocabulary and header only after every tensor claim has
// succeeded, then seals all bytes with the file digest.

const Output = struct {
    const ClaimedSection = struct {
        encoding: TensorSection.Encoding,
        dimensions: TensorDimensions,
        payload: []u8,
    };

    allocator: Allocator,
    kind: ModelKind,
    file_layout: layout.FileLayout,
    bytes: []align(alignment) u8,
    next_payload_offset: usize,
    sections_written_count: usize = 0,

    fn init(allocator: Allocator, kind: ModelKind) Allocator.Error!Output {
        const file_layout = layout.calculate(kind);
        const bytes = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), file_layout.size);

        // One initialization establishes every unused directory entry, alignment
        // gap, reserved header byte, and the not-yet-written digest as zero.
        @memset(bytes, 0);

        return .{
            .allocator = allocator,
            .kind = kind,
            .file_layout = file_layout,
            .bytes = bytes,
            .next_payload_offset = file_layout.tensor_payloads_start_offset,
        };
    }

    fn claimSection(output: *Output, section_kind: SectionKind, layer_index: ?u16) ClaimedSection {
        // Conversion claims tensors as the source reader encounters them. Append
        // each payload in that order and publish its location through the fixed
        // kind/layer directory rather than rearranging the source stream.
        const section = TensorSection.calculate(output.kind, section_kind, layer_index, output.next_payload_offset);

        const layer_slot_index: usize = if (layer_index) |index|
            @intCast(index)
        else
            0;

        const directory_entry_offset = TensorSection.directoryEntryOffset(output.kind, section_kind, layer_slot_index);
        const directory_entry = output.bytes[directory_entry_offset..][0..@sizeOf(u64)];

        // Zero is the unclaimed sentinel and cannot be a payload offset. This
        // assertion rejects a conversion plan that targets one destination twice.
        assert(std.mem.readInt(u64, directory_entry, .little) == 0);

        writeInt(u64, output.bytes, directory_entry_offset, section.payload_offset);

        output.next_payload_offset = std.mem.alignForward(usize, section.payload_offset + section.payload_size, alignment);
        output.sections_written_count += 1;

        return .{
            .encoding = section.encoding,
            .dimensions = section.dimensions,
            .payload = output.bytes[section.payload_offset..][0..section.payload_size],
        };
    }

    fn finish(output: *Output, vocabulary: PreparedVocabulary) []align(alignment) u8 {
        // ── Verify Complete Tensor Conversion ──

        const dimensions = inference.Model.dimensions(output.kind);

        assert(output.sections_written_count == TensorSection.count(output.kind));
        assert(output.next_payload_offset == output.file_layout.tensor_payloads_end_offset);
        assert(vocabulary.offsets.len == dimensions.vocabulary_tokens_count + 1);
        assert(vocabulary.bytes.len == output.file_layout.vocabulary_bytes_end_offset - output.file_layout.vocabulary_bytes_start_offset);

        // ── Publish The Vocabulary ──

        for (vocabulary.offsets, 0..) |offset, index| {
            writeInt(u32, output.bytes, output.file_layout.vocabulary_index_offset + index * @sizeOf(u32), offset);
        }

        @memcpy(output.bytes[output.file_layout.vocabulary_bytes_start_offset..output.file_layout.vocabulary_bytes_end_offset], vocabulary.bytes);

        // ── Publish The Header ──

        @memcpy(output.bytes[0..layout.header_format_version_offset], layout.header_magic);
        writeInt(u32, output.bytes, layout.header_format_version_offset, layout.format_version);
        writeInt(u16, output.bytes, layout.header_model_kind_offset, @intFromEnum(output.kind));
        writeInt(u64, output.bytes, layout.header_file_size_offset, output.file_layout.size);
        writeInt(u64, output.bytes, layout.header_vocabulary_index_offset, output.file_layout.vocabulary_index_offset);
        writeInt(u64, output.bytes, layout.header_vocabulary_bytes_start_offset, output.file_layout.vocabulary_bytes_start_offset);
        writeInt(u64, output.bytes, layout.header_vocabulary_bytes_end_offset, output.file_layout.vocabulary_bytes_end_offset);

        // ── Seal The File ──
        //
        // The digest field is still zero from `init`, so hashing the complete
        // output matches the reader's rule of substituting zeros for that field.

        var hash = Blake3.init(.{});

        hash.update(output.bytes);
        hash.final(output.bytes[layout.header_file_blake3_offset..layout.header_reserved_offset]);

        return output.bytes;
    }

    fn deinit(output: *Output) void {
        output.allocator.free(output.bytes);

        output.* = undefined;
    }
};

// ─── CTranslate2 Weight Reader ─────────────────────────────────────────────
//
// The reader borrows names and tensor payloads from the source weights, copies
// only small dimension arrays, and rejects every read that crosses the input.
// CTranslate2 calls every stored tensor or scalar a variable; this reader uses
// tensor throughout because a scalar is a rank-zero tensor.
//
// Version 6 uses little-endian integers and this top-level sequence:
//
//   u32 binary version
//   string specification name
//   u32 specification revision
//   u32 tensor count
//   tensor[tensor count]
//   u32 alias count
//   (string alias, string target)[alias count]
//
// A `string` is `u16 byte_size`, followed by `byte_size - 1` content bytes and
// one `\x00` terminator. A tensor record is:
//
//   string name
//   u8 rank
//   u32[rank] dimensions
//   u8 data type
//   u32 payload byte size
//   u8[payload byte size] payload

const WeightsReader = struct {
    remaining: []const u8,

    fn readTensor(reader: *WeightsReader) ConvertError!Tensor {
        const name = try reader.readString();
        if (name.len == 0) {
            // Tensor names are the keys used to validate serialized order.
            return error.InvalidCTranslate2Weights;
        }

        const dimensions_count: usize = @intCast(try reader.readInt(u8));
        if (dimensions_count > Tensor.dimensions_count_max) {
            // Supported Whisper tensors are scalars, vectors, matrices, or
            // rank-three convolution kernels.
            return error.InvalidCTranslate2Weights;
        }

        var dimensions: [Tensor.dimensions_count_max]usize = @splat(0);
        var elements_count: usize = 1;

        for (0..dimensions_count) |dimension_index| {
            const dimension: usize = @intCast(try reader.readInt(u32));

            dimensions[dimension_index] = dimension;

            elements_count = std.math.mul(usize, elements_count, dimension) catch {
                // Dimensions whose product cannot fit `usize` cannot describe a
                // payload addressable by this process.
                return error.InvalidCTranslate2Weights;
            };
        }

        const data_type = std.enums.fromInt(DataType, try reader.readInt(u8)) orelse {
            // Unassigned tags have no defined CTranslate2 element width.
            return error.InvalidCTranslate2Weights;
        };

        const data_size: usize = @intCast(try reader.readInt(u32));

        const expected_data_size = std.math.mul(usize, elements_count, data_type.elementSize()) catch {
            // The declared shape and element type cannot fit an addressable
            // payload size.
            return error.InvalidCTranslate2Weights;
        };

        if (data_size != expected_data_size) {
            // The payload must contain exactly one encoded value per element.
            return error.InvalidCTranslate2Weights;
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
            // The encoded size always includes at least the terminator.
            return error.InvalidCTranslate2Weights;
        }

        const encoded = try reader.readBytes(encoded_size);
        if (encoded[encoded.len - 1] != '\x00') {
            // Every source string must end at its declared boundary.
            return error.InvalidCTranslate2Weights;
        }

        const value = encoded[0 .. encoded.len - 1];
        if (std.mem.indexOfScalar(u8, value, '\x00') != null) {
            // An earlier terminator would make the source string ambiguous.
            return error.InvalidCTranslate2Weights;
        }

        return value;
    }

    fn readInt(reader: *WeightsReader, comptime Int: type) ConvertError!Int {
        const bytes = try reader.readBytes(@sizeOf(Int));

        return std.mem.readInt(Int, bytes[0..@sizeOf(Int)], .little);
    }

    fn readBytes(reader: *WeightsReader, size: usize) ConvertError![]const u8 {
        if (size > reader.remaining.len) {
            // A declared field may not extend beyond the borrowed source stream.
            return error.InvalidCTranslate2Weights;
        }

        const bytes = reader.remaining[0..size];

        reader.remaining = reader.remaining[size..];

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

        if (tensor.dimensions_count != 0) {
            return false;
        }

        if (tensor.data_type != expected_data_type) {
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

// ─── CTranslate2 Whisper Schema ───────────────────────────────────────────────
//
// These plans own only CTranslate2 names, serialized order, and metadata.
// Packed-model shape, numeric domain, and encoding come from
// `TensorSection.calculate`.
//
// Scalar records are source-model configuration rather than learned weights.
// Voiced's inference implementation commits to the values below, so conversion
// validates and discards them instead of carrying duplicate runtime flags into
// the packed format.

const binary_version_supported: u32 = 6;
const whisper_specification_revision: u32 = 3;

// This is the longest full layer name generated by the supported schemas.
const tensor_name_size_max = "decoder/layer_23/self_attention/layer_norm/gamma".len;

// CTranslate2 stores its `ActivationType` ordinal and Boolean flags as i8
// scalars. Alignment settings are signed i16 attributes; `-1` selects the final
// decoder layer.
const ctranslate2_activation_gelu: i8 = 3;
const ctranslate2_flag_disabled: i8 = 0;
const ctranslate2_flag_enabled: i8 = 1;
const ctranslate2_alignment_heads_default: i16 = 1;
const ctranslate2_alignment_layer_last: i16 = -1;

const TensorPlan = struct {
    name: []const u8,
    handling: Handling,

    const Handling = union(enum) {
        packed_section: SectionKind,
        scalar_i8: i8,
        scalar_i16: i16,
        encoder_attention_heads_count,
        decoder_attention_heads_count,
    };
};

fn expectedTensorsCount(dimensions: ModelDimensions) u32 {
    const tensors_count = decoder_tensors_before_layers.len +
        dimensions.decoder_layers_count * decoder_layer_tensors.len +
        decoder_tensors_after_layers.len +
        encoder_tensors_before_layers.len +
        dimensions.encoder_layers_count * encoder_layer_tensors.len +
        encoder_tensors_after_layers.len;

    assert(tensors_count <= std.math.maxInt(u32));

    return @intCast(tensors_count);
}

fn sourceLayerIndexFromOrdinal(layer_ordinal: usize, layers_count: usize) u16 {
    assert(layer_ordinal < layers_count);

    // CTranslate2 serializes tensor names lexicographically rather than by
    // numeric layer index. Tiny.en and Base.en have only single-digit indexes,
    // while Small.en and Medium.en need explicit source-order mappings.
    return switch (layers_count) {
        4, 6 => @intCast(layer_ordinal),
        12 => ([_]u16{ 0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9 })[layer_ordinal],
        24 => ([_]u16{ 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9 })[layer_ordinal],
        else => unreachable,
    };
}

const decoder_tensors_before_layers = [_]TensorPlan{
    .{ .name = "decoder/activation", .handling = .{ .scalar_i8 = ctranslate2_activation_gelu } },
    .{ .name = "decoder/alibi", .handling = .{ .scalar_i8 = ctranslate2_flag_disabled } },
    .{ .name = "decoder/alignment_heads", .handling = .{ .scalar_i16 = ctranslate2_alignment_heads_default } },
    .{ .name = "decoder/alignment_layer", .handling = .{ .scalar_i16 = ctranslate2_alignment_layer_last } },
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
    .{ .name = "decoder/pre_norm", .handling = .{ .scalar_i8 = ctranslate2_flag_enabled } },
    .{ .name = "decoder/scale_embeddings", .handling = .{ .scalar_i8 = ctranslate2_flag_disabled } },
    .{ .name = "decoder/start_from_zero_embedding", .handling = .{ .scalar_i8 = ctranslate2_flag_disabled } },
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

fn convertLayerTensor(
    reader: *WeightsReader,
    output: *Output,
    dimensions: ModelDimensions,
    scope: []const u8,
    layer_index: u16,
    plan: TensorPlan,
) ConvertError!void {
    assert(scope.len > 0);
    assert(plan.name.len > 0);

    var tensor_name_buffer: [tensor_name_size_max]u8 = undefined;
    const tensor_name = std.fmt.bufPrint(&tensor_name_buffer, "{s}/layer_{d}/{s}", .{ scope, layer_index, plan.name }) catch {
        unreachable;
    };

    try convertTensor(reader, output, dimensions, layer_index, tensor_name, plan.handling);
}

fn convertTensor(
    reader: *WeightsReader,
    output: *Output,
    dimensions: ModelDimensions,
    layer_index: ?u16,
    expected_name: []const u8,
    handling: TensorPlan.Handling,
) ConvertError!void {
    const tensor = try reader.readTensor();
    if (!std.mem.eql(u8, tensor.name, expected_name)) {
        // Exact positional matching makes missing, extra, and reordered source
        // records observable without building a name lookup table.
        return error.InvalidCTranslate2Weights;
    }

    // Select the validation or conversion from the tensor plan.
    switch (handling) {
        .scalar_i8 => |expected_value| {
            if (!tensor.hasScalarValue(i8, expected_value)) {
                // This source configuration does not match Voiced's fixed model.
                return error.InvalidCTranslate2Weights;
            }
        },

        .scalar_i16 => |expected_value| {
            if (!tensor.hasScalarValue(i16, expected_value)) {
                // This source configuration does not match Voiced's fixed model.
                return error.InvalidCTranslate2Weights;
            }
        },

        .encoder_attention_heads_count => {
            const expected_value: i16 = @intCast(dimensions.encoder_attention_heads_count);
            if (!tensor.hasScalarValue(i16, expected_value)) {
                // Encoder metadata must agree with the selected model kind.
                return error.InvalidCTranslate2Weights;
            }
        },

        .decoder_attention_heads_count => {
            const expected_value: i16 = @intCast(dimensions.decoder_attention_heads_count);
            if (!tensor.hasScalarValue(i16, expected_value)) {
                // Decoder metadata must agree with the selected model kind.
                return error.InvalidCTranslate2Weights;
            }
        },

        .packed_section => |section_kind| {
            const section = output.claimSection(section_kind, layer_index);

            if (tensor.data_type != .float16) {
                // The pinned source models store every retained tensor as Float16.
                return error.InvalidCTranslate2Weights;
            }

            if (!tensor.hasDimensions(section.dimensions.slice())) {
                // Runtime section dimensions are authoritative for the model kind.
                return error.InvalidCTranslate2Weights;
            }

            switch (section.encoding) {
                .float32 => try convertFloat16TensorToFloat32(tensor.data, section.payload),
                .vnni_o8_k4 => try quantizeVnniWeight(tensor.data, section.dimensions, section.payload),
            }
        },
    }
}

// ─── Packed Tensor Conversion ──────────────────────────────────────────────
//
// Float sections widen validated Float16 values to native Float32. Weight
// sections use symmetric per-output-row quantization, then rearrange signed
// bytes into the O8/K4 order consumed directly by `VnniWeight`. Each VNNI
// payload stores packed bytes first, followed by aligned Float32 scales and i32
// activation-zero-point compensation values.

const float16_values_per_validation_vector: usize = 8;
const quantized_weight_magnitude_max: f32 = @floatFromInt(std.math.maxInt(i8));

fn convertFloat16TensorToFloat32(source: []const u8, destination: []u8) ConvertError!void {
    assert(source.len % @sizeOf(f16) == 0);
    assert(destination.len == source.len / @sizeOf(f16) * @sizeOf(f32));

    const values_count = source.len / @sizeOf(f16);

    for (0..values_count) |value_index| {
        const source_offset = value_index * @sizeOf(f16);
        const value: f32 = @floatCast(readFloat(f16, source, source_offset));

        if (!std.math.isFinite(value)) {
            // Runtime tensors admit only finite numeric values.
            return error.InvalidCTranslate2Weights;
        }

        writeFloat(f32, destination, value_index * @sizeOf(f32), value);
    }
}

fn quantizeVnniWeight(source: []const u8, dimensions: TensorDimensions, destination: []u8) ConvertError!void {
    const vnni_layout = TensorSection.VnniPayload.calculate(dimensions);

    assert(dimensions.rank == 2 or dimensions.rank == 3);
    assert(source.len == vnni_layout.weights_size * @sizeOf(f16));
    assert(destination.len == vnni_layout.size);
    assert(vnni_layout.weight_layout.input_values_count % float16_values_per_validation_vector == 0);

    // Scales must exist before packing because the second pass reads each scale
    // from its final payload location; no temporary scale allocation is needed.
    try calculateVnniWeightScales(source, vnni_layout, destination);
    quantizeAndPackVnniWeight(source, vnni_layout, destination);
}

fn calculateVnniWeightScales(source: []const u8, vnni_layout: VnniPayload, destination: []u8) ConvertError!void {
    const F16Values = @Vector(float16_values_per_validation_vector, f16);
    const F32Values = @Vector(float16_values_per_validation_vector, f32);
    const U16Bits = @Vector(float16_values_per_validation_vector, u16);

    // Positive Float16 infinity has every exponent bit set and no sign or
    // mantissa bits, so its representation is exactly the exponent mask.
    const exponent_mask: U16Bits = @splat(@as(u16, @bitCast(std.math.inf(f16))));

    assert(vnni_layout.weight_layout.input_values_count % float16_values_per_validation_vector == 0);

    for (0..vnni_layout.weight_layout.output_rows_count) |output_row_index| {
        var absolute_maximums: F32Values = @splat(0);
        var input_value_start_index: usize = 0;

        while (input_value_start_index < vnni_layout.weight_layout.input_values_count) : (input_value_start_index += float16_values_per_validation_vector) {
            const source_value_index = output_row_index * vnni_layout.weight_layout.input_values_count + input_value_start_index;
            const source_offset = source_value_index * @sizeOf(f16);
            const half_bits: U16Bits = @bitCast(source[source_offset..][0 .. float16_values_per_validation_vector * @sizeOf(f16)].*);

            if (@reduce(.Or, half_bits & exponent_mask == exponent_mask)) {
                // An all-ones exponent identifies either infinity or NaN.
                return error.InvalidCTranslate2Weights;
            }

            const half_values: F16Values = @bitCast(half_bits);
            const values: F32Values = @floatCast(half_values);

            absolute_maximums = @max(absolute_maximums, @abs(values));
        }

        // The scale maps this row's largest magnitude to signed byte magnitude
        // 127. An all-zero row uses scale one to avoid division by zero while
        // still quantizing every value to zero.
        const absolute_maximum = @reduce(.Max, absolute_maximums);

        const scale: f32 = if (absolute_maximum == 0)
            1
        else
            quantized_weight_magnitude_max / absolute_maximum;

        assert(std.math.isFinite(scale));
        assert(scale > 0);
        writeFloat(f32, destination, vnni_layout.scales_offset + output_row_index * @sizeOf(f32), scale);
    }
}

fn quantizeAndPackVnniWeight(source: []const u8, vnni_layout: VnniPayload, destination: []u8) void {
    const F16InputGroup = @Vector(inference.VnniWeight.input_values_per_group, f16);
    const F32InputGroup = @Vector(inference.VnniWeight.input_values_per_group, f32);
    const I32InputGroup = @Vector(inference.VnniWeight.input_values_per_group, i32);
    const I8InputGroup = @Vector(inference.VnniWeight.input_values_per_group, i8);
    const U16InputGroup = @Vector(inference.VnniWeight.input_values_per_group, u16);
    const F32OutputBlock = @Vector(inference.VnniWeight.output_rows_per_block, f32);
    const I32OutputBlock = @Vector(inference.VnniWeight.output_rows_per_block, i32);
    const minimums: F32InputGroup = @splat(-quantized_weight_magnitude_max);
    const maximums: F32InputGroup = @splat(quantized_weight_magnitude_max);

    assert(source.len == vnni_layout.weights_size * @sizeOf(f16));
    assert(destination.len == vnni_layout.size);

    var output_row_start_index: usize = 0;

    while (output_row_start_index < vnni_layout.weight_layout.output_rows_count) : (output_row_start_index += inference.VnniWeight.output_rows_per_block) {
        // Scales live in the final payload. Load one output-row block before
        // traversing its K dimension so every group uses the same row scale.
        var output_scales: F32OutputBlock = undefined;
        inline for (0..inference.VnniWeight.output_rows_per_block) |block_row_index| {
            const scale_offset = vnni_layout.scales_offset + (output_row_start_index + block_row_index) * @sizeOf(f32);

            output_scales[block_row_index] = readFloat(f32, destination, scale_offset);
        }

        var quantized_values_sums: I32OutputBlock = @splat(0);
        var input_value_start_index: usize = 0;

        while (input_value_start_index < vnni_layout.weight_layout.input_values_count) : (input_value_start_index += inference.VnniWeight.input_values_per_group) {
            var quantized_group_sums: I32OutputBlock = undefined;
            inline for (0..inference.VnniWeight.output_rows_per_block) |block_row_index| {
                const output_row_index = output_row_start_index + block_row_index;
                const source_value_index = output_row_index * vnni_layout.weight_layout.input_values_count + input_value_start_index;
                const source_offset = source_value_index * @sizeOf(f16);
                const half_bits: U16InputGroup = @bitCast(source[source_offset..][0 .. inference.VnniWeight.input_values_per_group * @sizeOf(f16)].*);
                const half_values: F16InputGroup = @bitCast(half_bits);
                const values: F32InputGroup = @floatCast(half_values);
                const scaled_values = values * @as(F32InputGroup, @splat(output_scales[block_row_index]));
                const clamped_values = @min(@max(scaled_values, minimums), maximums);
                const rounded_values = inference.VnniWeight.roundFloat32VectorToNearestEven(clamped_values);
                const quantized_integers: I32InputGroup = @intFromFloat(rounded_values);
                const quantized_values: I8InputGroup = @intCast(quantized_integers);
                const quantized_bytes: [inference.VnniWeight.input_values_per_group]u8 = @bitCast(quantized_values);

                const packed_output_offset = vnni_layout.weight_layout.valueOffset(output_row_index, input_value_start_index);

                destination[packed_output_offset..][0..inference.VnniWeight.input_values_per_group].* = quantized_bytes;
                quantized_group_sums[block_row_index] = @reduce(.Add, quantized_integers);
            }

            quantized_values_sums += quantized_group_sums;
        }

        // The kernel consumes unsigned activations centered at
        // `activation_zero_point`. Precompute `-zero_point * sum(weights)` so the
        // runtime dot product equals one over centered activation values.
        quantized_values_sums *= @as(I32OutputBlock, @splat(-inference.VnniWeight.activation_zero_point));

        inline for (0..inference.VnniWeight.output_rows_per_block) |block_row_index| {
            const compensation_offset = vnni_layout.compensation_offset + (output_row_start_index + block_row_index) * @sizeOf(i32);

            writeInt(i32, destination, compensation_offset, quantized_values_sums[block_row_index]);
        }
    }
}

fn readFloat(comptime Float: type, source: []const u8, byte_offset: usize) Float {
    const Bits = switch (Float) {
        f16 => u16,
        f32 => u32,
        else => @compileError("unsupported packed float type"),
    };

    assert(byte_offset <= source.len);
    assert(@sizeOf(Bits) <= source.len - byte_offset);

    return @bitCast(std.mem.readInt(Bits, source[byte_offset..][0..@sizeOf(Bits)], .little));
}

fn writeFloat(comptime Float: type, destination: []u8, byte_offset: usize, value: Float) void {
    const Bits = switch (Float) {
        f16 => u16,
        f32 => u32,
        else => @compileError("unsupported packed float type"),
    };

    writeInt(Bits, destination, byte_offset, @as(Bits, @bitCast(value)));
}

fn writeInt(comptime Int: type, destination: []u8, byte_offset: usize, value: anytype) void {
    assert(byte_offset <= destination.len);
    assert(@sizeOf(Int) <= destination.len - byte_offset);

    std.mem.writeInt(
        Int,
        destination[byte_offset..][0..@sizeOf(Int)],
        @intCast(value),
        .little,
    );
}
