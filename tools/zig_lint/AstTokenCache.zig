//! `AstTokenCache` owns one immutable source-and-token pack per scanned root.
//! Each entry couples source bytes with the exact token tags and byte offsets
//! produced from those bytes; callers can safely rebuild an owned `Ast` with
//! `Entry.parse`. Loaded entry storage remains stable until `deinit`.
//!
//! The binary representation is private and versioned. Serialization writes
//! explicit little-endian fields and canonical padding rather than Zig struct
//! bytes. A completed scan atomically replaces the previous pack; malformed,
//! incompatible, or unavailable packs become ordinary cache misses. A
//! successful complete scan retains only entries observed during that scan.
//! Packs live beneath `base_dir/syntax/<linter Zig version>/`, separated from
//! lint facts.

const AstTokenCache = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const Ast = std.zig.Ast;
const Blake3 = std.crypto.hash.Blake3;
const Token = std.zig.Token;
const CachePath = @import("CachePath.zig");

entry_by_path: std.StringHashMapUnmanaged(Record) = .empty,
storage_arena: std.heap.ArenaAllocator,
pack_path: []const u8,
dirty: bool = false,

const Record = struct {
    inode: u64,
    source_size: u32,
    mtime_ns: i128,
    source: [:0]const u8,
    token_tags: []const Token.Tag,
    token_starts: []const Ast.ByteOffset,

    // Per-scan state, never serialized. A completed scan omits records whose
    // source path was deleted, renamed, excluded, or otherwise not selected.
    seen_in_scan: bool = false,

    fn matches(record: Record, stat: std.Io.File.Stat) bool {
        return record.inode == stat.inode and record.source_size == stat.size and
            record.mtime_ns == stat.mtime.nanoseconds;
    }
};

pub const Entry = struct {
    source: [:0]const u8,
    token_tags: []const Token.Tag,
    token_starts: []const Ast.ByteOffset,

    /// `parse` returns an AST that borrows `source` from the cache and owns all
    /// of its other storage. Keep the cache alive until calling `Ast.deinit`.
    pub fn parse(entry: Entry, allocator: Allocator) !Ast {
        var tokens: Ast.TokenList = .{};
        defer tokens.deinit(allocator);

        try tokens.ensureTotalCapacity(allocator, entry.token_tags.len);

        tokens.len = entry.token_tags.len;

        const token_slice = tokens.slice();

        @memcpy(token_slice.items(.tag), entry.token_tags);
        @memcpy(token_slice.items(.start), entry.token_starts);

        var owned_tokens = tokens.toOwnedSlice();
        errdefer owned_tokens.deinit(allocator);

        return Ast.parseTokens(allocator, entry.source, owned_tokens, .zig);
    }
};

comptime {
    assert(@sizeOf(Token.Tag) == 1);
    assert(@sizeOf(Ast.ByteOffset) == 4);
}

// ─── Pack Format ─────────────────────────────────────────────────────────────
//
// The entry table contains offsets into the path and file-blob regions:
//
//   header | fixed entries | paths | source\0 tags padding starts | ...
//
// Paths and file blobs appear in entry-table order. Every start offset is a
// little-endian u32. `loadRecords` validates the complete canonical layout
// before any decoded prefix can become a cache hit.

const magic = "ZASTTOKS";
const format_version: u32 = 1;
const header_size: usize = 96;
const entry_size: usize = 56;
const pack_size_max: usize = 256 * 1024 * 1024;
const token_tag_is_valid = valid: {
    var values = [_]bool{false} ** 256;

    for (std.meta.fields(Token.Tag)) |field| {
        values[field.value] = true;
    }

    break :valid values;
};

/// `init` opens the pack for one directory scan. Cache loading is best effort:
/// an unreadable or invalid existing pack produces an empty, dirty cache that
/// `save` can replace after the scan completes.
pub fn init(
    allocator: Allocator,
    io: std.Io,
    root_dir: std.Io.Dir,
    cache_base_dir: []const u8,
) !AstTokenCache {
    if (cache_base_dir.len == 0) {
        return error.InvalidCacheDir;
    }

    var storage_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer storage_arena.deinit();

    const storage_allocator = storage_arena.allocator();
    const root_path = try root_dir.realPathFileAlloc(io, ".", storage_allocator);

    const linter_version_shard = try CachePath.allocVersionShard(
        storage_allocator,
        builtin.zig_version,
    );

    var root_digest: [Blake3.digest_length]u8 = undefined;
    Blake3.hash(root_path, &root_digest, .{});

    const pack_path = try std.fmt.allocPrint(
        storage_allocator,
        "{s}/syntax/{s}/{s}.bin",
        .{
            cache_base_dir,
            linter_version_shard,
            std.fmt.bytesToHex(root_digest, .lower),
        },
    );

    var cache: AstTokenCache = .{
        .storage_arena = storage_arena,
        .pack_path = pack_path,
    };

    cache.loadRecords(allocator, io) catch {
        // Decoding may already have inserted a valid prefix. No prefix is
        // trusted unless the checksum and every record in the pack are valid.
        cache.entry_by_path.clearRetainingCapacity();

        cache.dirty = true;
    };

    return cache;
}

pub fn deinit(cache: *AstTokenCache, allocator: Allocator) void {
    cache.entry_by_path.deinit(allocator);
    cache.storage_arena.deinit();
}

/// `lookup` returns a borrowed entry only when the path, metadata, and scan
/// size limit match. A returned entry remains valid until `deinit`.
pub fn lookup(
    cache: *AstTokenCache,
    relative_path: []const u8,
    stat: std.Io.File.Stat,
    file_size_max: u32,
) ?Entry {
    const record = cache.entry_by_path.getPtr(relative_path) orelse {
        return null;
    };

    if (record.source_size <= file_size_max and record.matches(stat)) {
        record.seen_in_scan = true;

        return .{
            .source = record.source,
            .token_tags = record.token_tags,
            .token_starts = record.token_starts,
        };
    }

    _ = cache.entry_by_path.remove(relative_path);
    cache.dirty = true;

    return null;
}

/// `put` copies one source snapshot and its tokenizer output. The caller may
/// release or reuse the source and AST after this function returns.
pub fn put(
    cache: *AstTokenCache,
    allocator: Allocator,
    relative_path: []const u8,
    stat: std.Io.File.Stat,
    source: [:0]const u8,
    tokens: Ast.TokenList.Slice,
) !void {
    if (relative_path.len == 0 or source.len > std.math.maxInt(u32) or
        stat.size != source.len or tokens.len == 0 or tokens.len > source.len + 1 or
        tokens.items(.tag).len != tokens.items(.start).len)
    {
        return error.InvalidCacheEntry;
    }

    const token_tags = tokens.items(.tag);
    const token_starts = tokens.items(.start);

    if (token_tags[token_tags.len - 1] != .eof or token_starts[token_starts.len - 1] != source.len) {
        return error.InvalidCacheEntry;
    }

    const storage = cache.storage_arena.allocator();
    const owned_path = try storage.dupe(u8, relative_path);
    const owned_source = try storage.dupeZ(u8, source);
    const owned_tags = try storage.dupe(Token.Tag, token_tags);
    const owned_starts = try storage.dupe(Ast.ByteOffset, token_starts);

    try cache.entry_by_path.put(allocator, owned_path, .{
        .inode = stat.inode,
        .source_size = @intCast(source.len),
        .mtime_ns = stat.mtime.nanoseconds,
        .source = owned_source,
        .token_tags = owned_tags,
        .token_starts = owned_starts,
        .seen_in_scan = true,
    });

    cache.dirty = true;
}

/// `save` atomically publishes the entries observed during one successful
/// complete scan and drops every unvisited entry.
pub fn save(cache: *AstTokenCache, allocator: Allocator, io: std.Io) !void {
    const SavedEntry = struct {
        path: []const u8,
        record: *const Record,
    };

    if (!cache.dirty) {
        var records = cache.entry_by_path.valueIterator();

        while (records.next()) |record| {
            if (!record.seen_in_scan) {
                break;
            }
        } else {
            return;
        }
    }

    var saved_entries: std.ArrayList(SavedEntry) = .empty;
    defer saved_entries.deinit(allocator);

    var iterator = cache.entry_by_path.iterator();

    while (iterator.next()) |map_entry| {
        if (!map_entry.value_ptr.seen_in_scan) {
            continue;
        }

        try saved_entries.append(allocator, .{
            .path = map_entry.key_ptr.*,
            .record = map_entry.value_ptr,
        });
    }

    if (saved_entries.items.len > std.math.maxInt(u32)) {
        return error.CacheTooLarge;
    }

    std.mem.sort(SavedEntry, saved_entries.items, {}, struct {
        fn lessThan(_: void, left: SavedEntry, right: SavedEntry) bool {
            return std.mem.lessThan(u8, left.path, right.path);
        }
    }.lessThan);

    // ── Measure Canonical Regions ──

    var paths_size: usize = 0;

    for (saved_entries.items) |saved_entry| {
        paths_size = try std.math.add(usize, paths_size, saved_entry.path.len);
    }

    const entries_size = try std.math.mul(usize, saved_entries.items.len, entry_size);
    const records_end = try std.math.add(usize, header_size, entries_size);
    const file_blobs_offset = try std.math.add(usize, records_end, paths_size);
    var encoded_size = file_blobs_offset;

    for (saved_entries.items) |saved_entry| {
        const record = saved_entry.record;

        encoded_size = try fileBlobEnd(encoded_size, record.source_size, record.token_tags.len);
    }

    if (encoded_size > pack_size_max) {
        return error.CacheTooLarge;
    }

    // ── Encode And Publish ──

    const encoded = try allocator.alloc(u8, encoded_size);
    defer allocator.free(encoded);

    @memset(encoded, 0);

    @memcpy(encoded[0..8], magic);
    std.mem.writeInt(u32, encoded[8..12], format_version, .little);
    std.mem.writeInt(u32, encoded[12..16], @intCast(saved_entries.items.len), .little);
    std.mem.writeInt(u64, encoded[16..24], encoded_size, .little);
    std.mem.writeInt(u64, encoded[24..32], file_blobs_offset, .little);

    var compatibility_digest: [Blake3.digest_length]u8 = undefined;
    tokenizerCompatibilityDigest(&compatibility_digest);

    @memcpy(encoded[32..64], &compatibility_digest);

    var path_write_offset = records_end;
    var blob_write_offset = file_blobs_offset;

    for (saved_entries.items, 0..) |saved_entry, entry_index| {
        const record = saved_entry.record;
        const encoded_entry_offset = header_size + entry_index * entry_size;
        const encoded_entry = encoded[encoded_entry_offset..][0..entry_size];

        std.mem.writeInt(u64, encoded_entry[0..8], path_write_offset, .little);
        std.mem.writeInt(u32, encoded_entry[8..12], @intCast(saved_entry.path.len), .little);
        std.mem.writeInt(u32, encoded_entry[12..16], record.source_size, .little);
        std.mem.writeInt(u32, encoded_entry[16..20], @intCast(record.token_tags.len), .little);
        std.mem.writeInt(u64, encoded_entry[24..32], blob_write_offset, .little);
        std.mem.writeInt(u64, encoded_entry[32..40], record.inode, .little);
        std.mem.writeInt(i128, encoded_entry[40..56], record.mtime_ns, .little);

        @memcpy(encoded[path_write_offset..][0..saved_entry.path.len], saved_entry.path);

        path_write_offset += saved_entry.path.len;

        @memcpy(encoded[blob_write_offset..][0..record.source.len], record.source);

        blob_write_offset += record.source.len + 1;

        for (record.token_tags) |tag| {
            encoded[blob_write_offset] = @intFromEnum(tag);
            blob_write_offset += 1;
        }

        blob_write_offset = std.mem.alignForward(usize, blob_write_offset, @sizeOf(Ast.ByteOffset));

        for (record.token_starts) |start| {
            std.mem.writeInt(
                Ast.ByteOffset,
                encoded[blob_write_offset..][0..@sizeOf(Ast.ByteOffset)],
                start,
                .little,
            );

            blob_write_offset += @sizeOf(Ast.ByteOffset);
        }
    }

    assert(path_write_offset == file_blobs_offset);
    assert(blob_write_offset == encoded_size);

    var payload_digest: [Blake3.digest_length]u8 = undefined;
    Blake3.hash(encoded[header_size..], &payload_digest, .{});

    @memcpy(encoded[64..96], &payload_digest);

    // A reader sees either the previous complete generation or this one.
    // Concurrent writers may lose cache hits, but metadata validation prevents
    // an entry from being associated with a different observed file identity.
    var output = try std.Io.Dir.cwd().createFileAtomic(io, cache.pack_path, .{
        .make_path = true,
        .replace = true,
    });
    defer output.deinit(io);

    try output.file.writeStreamingAll(io, encoded);
    try output.replace(io);

    cache.dirty = false;
}

fn loadRecords(cache: *AstTokenCache, allocator: Allocator, io: std.Io) !void {
    const encoded = try std.Io.Dir.cwd().readFileAllocOptions(
        io,
        cache.pack_path,
        cache.storage_arena.allocator(),
        .limited(pack_size_max + 1),
        .of(Ast.ByteOffset),
        null,
    );

    if (encoded.len < header_size or !std.mem.eql(u8, encoded[0..8], magic) or
        std.mem.readInt(u32, encoded[8..12], .little) != format_version or
        std.mem.readInt(u64, encoded[16..24], .little) != encoded.len)
    {
        return error.InvalidCache;
    }

    var expected_compatibility_digest: [Blake3.digest_length]u8 = undefined;
    tokenizerCompatibilityDigest(&expected_compatibility_digest);

    if (!std.mem.eql(u8, encoded[32..64], &expected_compatibility_digest)) {
        return error.InvalidCache;
    }

    var actual_payload_digest: [Blake3.digest_length]u8 = undefined;
    Blake3.hash(encoded[header_size..], &actual_payload_digest, .{});

    if (!std.mem.eql(u8, encoded[64..96], &actual_payload_digest)) {
        return error.InvalidCache;
    }

    const entries_count = std.mem.readInt(u32, encoded[12..16], .little);

    const entries_size = std.math.mul(usize, entries_count, entry_size) catch {
        return error.InvalidCache;
    };

    const records_end = std.math.add(usize, header_size, entries_size) catch {
        return error.InvalidCache;
    };

    const file_blobs_offset = std.mem.readInt(u64, encoded[24..32], .little);

    if (records_end > encoded.len or file_blobs_offset < records_end or file_blobs_offset > encoded.len) {
        return error.InvalidCache;
    }

    try cache.entry_by_path.ensureTotalCapacity(allocator, entries_count);

    var expected_path_offset = records_end;
    var expected_blob_offset: usize = @intCast(file_blobs_offset);
    var previous_path: ?[]const u8 = null;

    for (0..entries_count) |entry_index| {
        const encoded_entry_offset = header_size + entry_index * entry_size;
        const encoded_entry = encoded[encoded_entry_offset..][0..entry_size];

        const path_offset = std.mem.readInt(u64, encoded_entry[0..8], .little);
        const path_size = std.mem.readInt(u32, encoded_entry[8..12], .little);
        const source_size = std.mem.readInt(u32, encoded_entry[12..16], .little);
        const token_count = std.mem.readInt(u32, encoded_entry[16..20], .little);
        const reserved = std.mem.readInt(u32, encoded_entry[20..24], .little);
        const file_blob_offset = std.mem.readInt(u64, encoded_entry[24..32], .little);
        const inode = std.mem.readInt(u64, encoded_entry[32..40], .little);
        const mtime_ns = std.mem.readInt(i128, encoded_entry[40..56], .little);

        if (reserved != 0 or path_size == 0 or path_offset != expected_path_offset or
            file_blob_offset != expected_blob_offset or token_count == 0 or
            @as(u64, token_count) > @as(u64, source_size) + 1)
        {
            return error.InvalidCache;
        }

        const path_end = std.math.add(usize, expected_path_offset, path_size) catch {
            return error.InvalidCache;
        };

        if (path_end > file_blobs_offset) {
            return error.InvalidCache;
        }

        const relative_path = encoded[expected_path_offset..path_end];

        if (previous_path) |path| {
            if (!std.mem.lessThan(u8, path, relative_path)) {
                return error.InvalidCache;
            }
        }

        previous_path = relative_path;
        expected_path_offset = path_end;

        const blob_end = fileBlobEnd(expected_blob_offset, source_size, token_count) catch {
            return error.InvalidCache;
        };

        if (blob_end > encoded.len) {
            return error.InvalidCache;
        }

        const source_end = expected_blob_offset + source_size;
        if (encoded[source_end] != 0) {
            return error.InvalidCache;
        }

        const source: [:0]const u8 = encoded[expected_blob_offset..source_end :0];

        const tags_offset = source_end + 1;
        const tags_end = tags_offset + token_count;
        const starts_offset = std.mem.alignForward(usize, tags_end, @sizeOf(Ast.ByteOffset));

        for (encoded[tags_end..starts_offset]) |padding| {
            if (padding != 0) {
                return error.InvalidCache;
            }
        }

        for (encoded[tags_offset..tags_end]) |encoded_tag| {
            if (!token_tag_is_valid[encoded_tag]) {
                return error.InvalidCache;
            }
        }

        const token_tags: []const Token.Tag = @ptrCast(encoded[tags_offset..tags_end]);

        const starts_bytes = @as(
            []align(@alignOf(Ast.ByteOffset)) const u8,
            @alignCast(encoded[starts_offset..blob_end]),
        );

        const token_starts: []const Ast.ByteOffset = if (builtin.cpu.arch.endian() == .little)
            std.mem.bytesAsSlice(Ast.ByteOffset, starts_bytes)
        else starts: {
            const decoded = try cache.storage_arena.allocator().alloc(Ast.ByteOffset, token_count);

            for (decoded, 0..) |*start, token_index| {
                const start_offset = token_index * @sizeOf(Ast.ByteOffset);

                start.* = std.mem.readInt(
                    Ast.ByteOffset,
                    starts_bytes[start_offset..][0..@sizeOf(Ast.ByteOffset)],
                    .little,
                );
            }

            break :starts decoded;
        };

        var previous_start: Ast.ByteOffset = 0;

        for (token_starts, 0..) |start, token_index| {
            if (start > source_size or (token_index != 0 and start < previous_start)) {
                return error.InvalidCache;
            }

            previous_start = start;
        }

        if (token_tags[token_tags.len - 1] != .eof or token_starts[token_starts.len - 1] != source_size) {
            return error.InvalidCache;
        }

        cache.entry_by_path.putAssumeCapacity(relative_path, .{
            .inode = inode,
            .source_size = source_size,
            .mtime_ns = mtime_ns,
            .source = source,
            .token_tags = token_tags,
            .token_starts = token_starts,
        });

        expected_blob_offset = blob_end;
    }

    if (expected_path_offset != file_blobs_offset or expected_blob_offset != encoded.len) {
        return error.InvalidCache;
    }
}

fn fileBlobEnd(blob_offset: usize, source_size: usize, token_count: usize) !usize {
    const source_end = try std.math.add(usize, blob_offset, source_size);
    const tags_offset = try std.math.add(usize, source_end, 1);
    const tags_end = try std.math.add(usize, tags_offset, token_count);
    const starts_offset = std.mem.alignForward(usize, tags_end, @sizeOf(Ast.ByteOffset));
    const starts_size = try std.math.mul(usize, token_count, @sizeOf(Ast.ByteOffset));

    return std.math.add(usize, starts_offset, starts_size);
}

fn tokenizerCompatibilityDigest(digest: *[Blake3.digest_length]u8) void {
    var hasher = Blake3.init(.{});

    hasher.update(builtin.zig_version_string);

    inline for (std.meta.fields(Token.Tag)) |field| {
        hasher.update(field.name);

        const value: Token.Tag = @enumFromInt(field.value);
        const encoded_value = [_]u8{@intFromEnum(value)};

        hasher.update(&encoded_value);
    }

    hasher.final(digest);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test "saved source and token entries parse after loading" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const source = "const answer: u32 = 42;\n";

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    var ast = try Ast.parse(allocator, source, .zig);
    defer ast.deinit(allocator);

    const stat = try root_dir.statFile(io, "a.zig", .{});

    {
        var cache = try AstTokenCache.init(allocator, io, root_dir, cache_base);
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", stat, source, ast.tokens);
        try cache.save(allocator, io);

        try std.testing.expect(!cache.dirty);
    }

    var loaded = try AstTokenCache.init(allocator, io, root_dir, cache_base);
    defer loaded.deinit(allocator);

    const entry = loaded.lookup("a.zig", stat, std.math.maxInt(u32)) orelse {
        return error.MissingCacheEntry;
    };

    var cached_ast = try entry.parse(allocator);
    defer cached_ast.deinit(allocator);

    const rendered = try cached_ast.renderAlloc(allocator);
    defer allocator.free(rendered);

    try std.testing.expectEqualStrings(source, entry.source);
    try std.testing.expectEqualSlices(Token.Tag, ast.tokens.items(.tag), entry.token_tags);
    try std.testing.expectEqualSlices(Ast.ByteOffset, ast.tokens.items(.start), entry.token_starts);
    try std.testing.expectEqualStrings(source, rendered);
}

test "saving drops entries not observed during a complete scan" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const source = "const answer = 1;\n";

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = source });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    var ast = try Ast.parse(allocator, source, .zig);
    defer ast.deinit(allocator);

    const a_stat = try root_dir.statFile(io, "a.zig", .{});
    const b_stat = try root_dir.statFile(io, "b.zig", .{});

    {
        var cache = try AstTokenCache.init(allocator, io, root_dir, cache_base);
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", a_stat, source, ast.tokens);
        try cache.put(allocator, "b.zig", b_stat, source, ast.tokens);
        try cache.save(allocator, io);
    }

    {
        var cache = try AstTokenCache.init(allocator, io, root_dir, cache_base);
        defer cache.deinit(allocator);

        try std.testing.expect(cache.lookup(
            "a.zig",
            a_stat,
            std.math.maxInt(u32),
        ) != null);

        try cache.save(allocator, io);
    }

    var reloaded = try AstTokenCache.init(allocator, io, root_dir, cache_base);
    defer reloaded.deinit(allocator);

    try std.testing.expect(reloaded.lookup(
        "a.zig",
        a_stat,
        std.math.maxInt(u32),
    ) != null);

    try std.testing.expect(reloaded.lookup(
        "b.zig",
        b_stat,
        std.math.maxInt(u32),
    ) == null);
}

test "metadata mismatch removes an entry from the next generation" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const source = "const answer = 1;\n";

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    var ast = try Ast.parse(allocator, source, .zig);
    defer ast.deinit(allocator);

    const original_stat = try root_dir.statFile(io, "a.zig", .{});

    {
        var cache = try AstTokenCache.init(allocator, io, root_dir, cache_base);
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", original_stat, source, ast.tokens);
        try cache.save(allocator, io);
    }

    try root_dir.writeFile(io, .{ .sub_path = "a.zig", .data = "const answer = 22;\n" });

    const changed_stat = try root_dir.statFile(io, "a.zig", .{});

    {
        var cache = try AstTokenCache.init(allocator, io, root_dir, cache_base);
        defer cache.deinit(allocator);

        try std.testing.expect(cache.lookup(
            "a.zig",
            changed_stat,
            std.math.maxInt(u32),
        ) == null);

        try cache.save(allocator, io);
    }

    var reloaded = try AstTokenCache.init(allocator, io, root_dir, cache_base);
    defer reloaded.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), reloaded.entry_by_path.count());
}
