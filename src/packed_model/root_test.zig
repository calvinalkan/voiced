const std = @import("std");
const assert = std.debug.assert;

const inference = @import("../inference/root.zig");
const packed_model = @import("root.zig");
const setup = @import("../setup.zig");
const Blake3 = std.crypto.hash.Blake3;

// ─── Golden Model Round Trip ──────────────────────────────────────────────────
//
// Each entry pins the whole-file BLAKE3 of a packed model known to work for
// inference. The test rebuilds every model from its pinned CTranslate2 source,
// compares the exact on-disk bytes, then loads the result through the production
// reader. Together the matching digest and successful load provide high
// confidence in the writer-reader round trip without storing model-sized
// fixtures in Git.
//
// Source artifacts persist in the project Zig cache across test processes.
// Generated packed files remain private to each test run.

const golden_models = [_]struct {
    kind: inference.Model.Kind,
    packed_file_name: []const u8,
    expected_file_blake3_hex: []const u8,
}{
    .{
        .kind = .whisper_base_en,
        .packed_file_name = "whisper.base.en.voiced",
        .expected_file_blake3_hex = "c99d3970637f812ca81ce62da9dd30e8f26f477e792be86a07b67a6488b428fc",
    },
    .{
        .kind = .whisper_small_en,
        .packed_file_name = "whisper.small.en.voiced",
        .expected_file_blake3_hex = "e1eed702e095c94e1b66f164cff0902de2ce885522ac16fca1b961dcc86f8168",
    },
    .{
        .kind = .whisper_medium_en,
        .packed_file_name = "whisper.medium.en.voiced",
        .expected_file_blake3_hex = "c2343a47b30eec063cf13463d09ca5796c655543b626c99754118be46a04ed9d",
    },
    .{
        .kind = .whisper_tiny_en,
        .packed_file_name = "whisper.tiny.en.voiced",
        .expected_file_blake3_hex = "2fe13e291b2d70f6a89c4509c0a9ef7be70e4e3a5bbf2c395aa927fff87a9c11",
    },
};
comptime {
    const supported_model_kinds = std.enums.values(inference.Model.Kind);
    assert(golden_models.len == supported_model_kinds.len);

    for (golden_models, 0..) |golden_model, model_index| {
        assert(golden_model.kind == supported_model_kinds[model_index]);
    }
}

test "supported model conversions match the inference-blessed packed files" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var http_client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer http_client.deinit();

    for (golden_models) |golden_model| {
        // ── Rebuild The Packed Model ──

        var tmp_dir = try generatePackedModel(
            io,
            allocator,
            &http_client,
            golden_model.kind,
            golden_model.packed_file_name,
        );
        defer tmp_dir.cleanup();

        // ── Verify The Exact File Bytes ──
        //
        // Hash the generated file exactly as it exists on disk. Unlike the
        // packed format's internal checksum, this golden digest includes the
        // checksum field and detects any writer output change.

        var expected_file_blake3: [Blake3.digest_length]u8 = undefined;
        _ = try std.fmt.hexToBytes(&expected_file_blake3, golden_model.expected_file_blake3_hex);

        var actual_file_blake3: [Blake3.digest_length]u8 = undefined;
        {
            const packed_file = try tmp_dir.dir.openFile(
                io,
                golden_model.packed_file_name,
                .{ .mode = .read_only, .allow_directory = false },
            );
            defer packed_file.close(io);

            var file_hash = Blake3.init(.{});
            var packed_file_reader = packed_file.reader(io, &.{});

            var hash_buffer: [64 * 1024]u8 = undefined;
            while (true) {
                const bytes_read_count = try packed_file_reader.interface.readSliceShort(&hash_buffer);
                if (bytes_read_count == 0) {
                    break;
                }

                file_hash.update(hash_buffer[0..bytes_read_count]);
            }

            file_hash.final(&actual_file_blake3);
        }

        try std.testing.expectEqualSlices(u8, &expected_file_blake3, &actual_file_blake3);

        // ── Validate With The Production Reader ──
        //
        // The golden digest proves byte identity. Loading separately exercises
        // the production format validation and model-view construction.

        var loaded_model = try packed_model.load(io, tmp_dir.dir, golden_model.packed_file_name, golden_model.kind);
        defer loaded_model.deinit();
    }
}

fn generatePackedModel(
    io: std.Io,
    allocator: std.mem.Allocator,
    http_client: *std.http.Client,
    kind: inference.Model.Kind,
    packed_file_name: []const u8,
) !std.testing.TmpDir {
    // ── Acquire The Pinned Source ──

    const source_directory_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/model-sources/{s}",
        .{kind.name()},
    );
    defer allocator.free(source_directory_path);

    const cwd = std.Io.Dir.cwd();

    try cwd.createDirPath(io, source_directory_path);

    var source_directory = try cwd.openDir(io, source_directory_path, .{});
    defer source_directory.close(io);

    const source = try setup.ensureModuleCTranslate2Source(io, http_client, source_directory, kind);

    const weights_bytes = try source_directory.readFileAlloc(
        io,
        source.weights.file_name,
        allocator,
        .limited(source.weights.expected_size + 1),
    );
    defer allocator.free(weights_bytes);

    const vocabulary_text = try source_directory.readFileAlloc(
        io,
        source.vocabulary.file_name,
        allocator,
        .limited(source.vocabulary.expected_size + 1),
    );
    defer allocator.free(vocabulary_text);

    // ── Pack Into A Private Test Directory ──

    var temporary_directory = std.testing.tmpDir(.{});
    errdefer temporary_directory.cleanup();

    const packed_file = try temporary_directory.dir.createFile(io, packed_file_name, .{});
    defer packed_file.close(io);

    try packed_model.writeFromCTranslate2(io, allocator, packed_file, kind, weights_bytes, vocabulary_text);
    try packed_file.sync(io);

    return temporary_directory;
}

// ─── Reader Corruption Rejection ──────────────────────────────────────────────
//
// Header and checksum validation are shared by every model kind. These cases use
// Base.en so checksum failures scan the smaller packed file.

test "reader rejects representative packed-file corruption" {
    // These offsets intentionally duplicate the version-1 on-disk ABI. Using
    // the reader's layout constants would make the mutations follow an
    // accidental field movement instead of detecting it.
    const byte_corruptions = [_]struct {
        name: []const u8,
        byte_offset: u64,
        xor_mask: u8,
        expected_error: anyerror,
    }{
        .{ .name = "magic", .byte_offset = 0, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        .{ .name = "format version", .byte_offset = 8, .xor_mask = 1, .expected_error = error.UnsupportedPackedModelFormatVersion },
        .{ .name = "unsupported model kind", .byte_offset = 12, .xor_mask = 1, .expected_error = error.UnsupportedModelKind },
        .{ .name = "unexpected model kind", .byte_offset = 12, .xor_mask = 3, .expected_error = error.UnexpectedModelKind },
        .{ .name = "header reserve", .byte_offset = 14, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        .{ .name = "stored file size", .byte_offset = 16, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        .{ .name = "vocabulary index offset", .byte_offset = 24, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        .{ .name = "vocabulary bytes start", .byte_offset = 32, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        .{ .name = "vocabulary bytes end", .byte_offset = 40, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        .{ .name = "stored checksum", .byte_offset = 48, .xor_mask = 1, .expected_error = error.PackedModelChecksumMismatch },
        .{ .name = "trailing header reserve", .byte_offset = 80, .xor_mask = 1, .expected_error = error.InvalidPackedModel },
        // Offset 8192 lies inside the first tensor payload in the version-1 Base.en layout.
        .{ .name = "tensor payload", .byte_offset = 8192, .xor_mask = 1, .expected_error = error.PackedModelChecksumMismatch },
    };

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const base_model = golden_models[0];

    assert(base_model.kind == .whisper_base_en);

    var http_client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer http_client.deinit();

    // ── Generate And Validate The Baseline ──

    var tmp_dir = try generatePackedModel(
        io,
        allocator,
        &http_client,
        base_model.kind,
        base_model.packed_file_name,
    );
    defer tmp_dir.cleanup();

    {
        var loaded_model = try packed_model.load(io, tmp_dir.dir, base_model.packed_file_name, base_model.kind);
        defer loaded_model.deinit();
    }

    const packed_file = try tmp_dir.dir.openFile(
        io,
        base_model.packed_file_name,
        .{ .mode = .read_write, .allow_directory = false },
    );
    defer packed_file.close(io);

    // ── Reject Header And Checksum Corruption ──
    //
    // Each loop-body defer restores the original byte before the next table
    // entry, including when its expectation fails.

    for (byte_corruptions) |corruption| {
        var original_byte: [1]u8 = undefined;
        const bytes_read = try packed_file.readPositionalAll(io, &original_byte, corruption.byte_offset);

        try std.testing.expectEqual(original_byte.len, bytes_read);

        const corrupted_byte = [_]u8{original_byte[0] ^ corruption.xor_mask};

        try packed_file.writePositionalAll(io, &corrupted_byte, corruption.byte_offset);
        errdefer std.debug.print("reader corruption case failed: {s}\n", .{corruption.name});
        defer packed_file.writePositionalAll(io, &original_byte, corruption.byte_offset) catch {
            @panic("could not restore the packed model after a corruption case");
        };

        try std.testing.expectError(
            corruption.expected_error,
            packed_model.load(io, tmp_dir.dir, base_model.packed_file_name, base_model.kind),
        );
    }

    // ── Reject Resealed Tensor Directory Corruption ──
    //
    // A fresh internal checksum lets each mutation pass integrity validation and
    // reach the tensor-directory validator. The offsets below independently pin
    // the version-1 directory's first required slot, its first unused slot, and
    // the next tensor kind's required slot.

    const first_tensor_entry_offset: u64 = 128;
    const first_unused_tensor_entry_offset: u64 = 136;
    const second_tensor_entry_offset: u64 = 224;
    const first_tensor_payload_offset = try fileReadInt(u64, io, packed_file, first_tensor_entry_offset);
    const vocabulary_index_offset = try fileReadInt(u64, io, packed_file, 24);

    var tensor_offset_bytes: [@sizeOf(u64)]u8 = undefined;
    // Replace a required entry with the directory's zero sentinel. The reader
    // rejects it because required tensors must begin inside the payload region.
    std.mem.writeInt(u64, &tensor_offset_bytes, 0, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = first_tensor_entry_offset,
        .replacement_bytes = &tensor_offset_bytes,
    });

    // Point an unused layer slot at the first tensor payload. The reader rejects
    // every nonzero unused entry so one payload cannot acquire another meaning.
    std.mem.writeInt(u64, &tensor_offset_bytes, first_tensor_payload_offset, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = first_unused_tensor_entry_offset,
        .replacement_bytes = &tensor_offset_bytes,
    });

    // Point the second required tensor at the first tensor's payload. The reader
    // detects that the two schema-owned payload ranges overlap.
    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = second_tensor_entry_offset,
        .replacement_bytes = &tensor_offset_bytes,
    });

    // Move the first tensor one byte forward. The reader rejects the resulting
    // payload address because every tensor requires 64-byte alignment.
    std.mem.writeInt(u64, &tensor_offset_bytes, first_tensor_payload_offset + 1, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = first_tensor_entry_offset,
        .replacement_bytes = &tensor_offset_bytes,
    });

    // Move the first tensor beyond the tensor region and into vocabulary space.
    // The reader rejects the payload before constructing a tensor view.
    std.mem.writeInt(u64, &tensor_offset_bytes, vocabulary_index_offset + 64, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = first_tensor_entry_offset,
        .replacement_bytes = &tensor_offset_bytes,
    });

    // ── Reject Resealed Vocabulary Corruption ──
    //
    // These cases reach the reader's first-offset, monotonicity, bounds, and
    // terminal-offset checks with an otherwise valid packed file.

    const second_vocabulary_offset = try fileReadInt(u32, io, packed_file, vocabulary_index_offset + @sizeOf(u32));
    assert(second_vocabulary_offset > 0);

    var vocabulary_offset_bytes: [@sizeOf(u32)]u8 = undefined;
    // Move the first token start to byte one. The reader rejects a vocabulary
    // whose index does not begin at the first payload byte.
    std.mem.writeInt(u32, &vocabulary_offset_bytes, 1, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = vocabulary_index_offset,
        .replacement_bytes = &vocabulary_offset_bytes,
    });

    // Move the third token start below the second token start. The reader
    // rejects the resulting reversed token byte range.
    std.mem.writeInt(u32, &vocabulary_offset_bytes, second_vocabulary_offset - 1, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = vocabulary_index_offset + 2 * @sizeOf(u32),
        .replacement_bytes = &vocabulary_offset_bytes,
    });

    // Move the second token end beyond the complete vocabulary payload. The
    // reader rejects the range before exposing the token bytes.
    std.mem.writeInt(u32, &vocabulary_offset_bytes, std.math.maxInt(u32), .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = vocabulary_index_offset + @sizeOf(u32),
        .replacement_bytes = &vocabulary_offset_bytes,
    });

    // Replace the terminal offset with zero. The reader rejects an index whose
    // final entry does not cover the complete vocabulary payload.
    const vocabulary_tokens_count = inference.Model.dimensions(base_model.kind).vocabulary_tokens_count;
    const vocabulary_terminal_offset = vocabulary_index_offset + vocabulary_tokens_count * @sizeOf(u32);

    std.mem.writeInt(u32, &vocabulary_offset_bytes, 0, .little);

    try expectMutationRejected(.{
        .io = io,
        .packed_file = packed_file,
        .packed_directory = tmp_dir.dir,
        .packed_file_name = base_model.packed_file_name,
        .model_kind = base_model.kind,
        .byte_offset = vocabulary_terminal_offset,
        .replacement_bytes = &vocabulary_offset_bytes,
    });

    // ── Reject An Incorrect Physical File Size ──
    //
    // One extra byte is enough to violate the model kind's exact file extent;
    // checksum validation and mapping must not begin.

    const packed_file_size = (try packed_file.stat(io)).size;

    {
        try packed_file.setLength(io, packed_file_size + 1);
        defer packed_file.setLength(io, packed_file_size) catch {
            @panic("could not restore the packed model after extending it");
        };

        try std.testing.expectError(
            error.InvalidPackedModel,
            packed_model.load(io, tmp_dir.dir, base_model.packed_file_name, base_model.kind),
        );
    }
}

fn expectMutationRejected(options: struct {
    io: std.Io,
    packed_file: std.Io.File,
    packed_directory: std.Io.Dir,
    packed_file_name: []const u8,
    model_kind: inference.Model.Kind,
    byte_offset: u64,
    replacement_bytes: []const u8,
}) !void {
    const io = options.io;
    const packed_file = options.packed_file;
    const byte_offset = options.byte_offset;
    const replacement_bytes = options.replacement_bytes;

    assert(replacement_bytes.len > 0);
    assert(replacement_bytes.len <= @sizeOf(u64));
    assert(byte_offset >= 80);

    // Preserve both regions changed by this operation. Restoring their original
    // bytes avoids another whole-file checksum pass between cases.
    var original_bytes: [@sizeOf(u64)]u8 = undefined;
    const original = original_bytes[0..replacement_bytes.len];

    const read_size = try packed_file.readPositionalAll(io, original, byte_offset);

    try std.testing.expectEqual(original.len, read_size);

    var original_digest: [Blake3.digest_length]u8 = undefined;
    const digest_bytes_read = try packed_file.readPositionalAll(io, &original_digest, 48);

    try std.testing.expectEqual(original_digest.len, digest_bytes_read);
    defer {
        packed_file.writePositionalAll(io, original, byte_offset) catch {
            @panic("could not restore a structurally corrupted packed model");
        };

        packed_file.writePositionalAll(io, &original_digest, 48) catch {
            @panic("could not restore a structurally corrupted packed model checksum");
        };
    }

    try packed_file.writePositionalAll(io, replacement_bytes, byte_offset);

    try resealPackedFile(io, packed_file);

    try std.testing.expectError(
        error.InvalidPackedModel,
        packed_model.load(io, options.packed_directory, options.packed_file_name, options.model_kind),
    );
}

fn resealPackedFile(io: std.Io, packed_file: std.Io.File) !void {
    // The packed checksum hashes its own field as zero. Write that canonical
    // value before streaming the complete mutated file.
    const zero_digest: [Blake3.digest_length]u8 = @splat(0);

    try packed_file.writePositionalAll(io, &zero_digest, 48);

    var hash = Blake3.init(.{});
    var packed_file_reader = packed_file.reader(io, &.{});

    var buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const read_size = try packed_file_reader.interface.readSliceShort(&buffer);
        if (read_size == 0) {
            break;
        }

        hash.update(buffer[0..read_size]);
    }

    var digest: [Blake3.digest_length]u8 = undefined;
    hash.final(&digest);

    try packed_file.writePositionalAll(io, &digest, 48);
}

fn fileReadInt(
    comptime Int: type,
    io: std.Io,
    file: std.Io.File,
    byte_offset: u64,
) !Int {
    var bytes: [@sizeOf(Int)]u8 = undefined;
    const read_size = try file.readPositionalAll(io, &bytes, byte_offset);

    try std.testing.expectEqual(bytes.len, read_size);

    return std.mem.readInt(Int, &bytes, .little);
}
