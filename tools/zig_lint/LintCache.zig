//! Verified lint and Zig formatting facts -> cache metadata
//! Unknown or failed checks                -> run that check again
//!
//! base_dir/lint/<target Zig version>/
//! └── hex(BLAKE3(canonical root, linter identity)).bin
//!     └── root-relative/path.zig -> { inode, file_size, mtime_ns, verified }
//!
//! Lookup:
//!   path found + metadata matches + size <= limit -> hit
//!   missing / changed / oversized                -> miss for this file only
//!
//! Contents are NOT hashed -> metadata-preserving edits can be missed.
//! A result-affecting cache identity change
//!   -> selects separate state or makes every previous result a miss.

const LintCache = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;
const CachePath = @import("CachePath.zig");

metadata_by_path: std.StringHashMapUnmanaged(PathMetadata) = .empty,
storage_arena: std.heap.ArenaAllocator,
index_path: []const u8,

// Insertions, invalidations, or an unusable loaded index require a write.
// Unseen paths are detected separately by save's retained count.
dirty: bool = false,

// The scan limit is u32, so every admitted file size fits without loss.
// Keeping size at that width leaves room for seen_in_scan within a 32-byte record,
// rather than padding a u64-sized record to 48 bytes around the i128 mtime.
pub const Verified = packed struct(u8) {
    lint_clean: bool = false,
    zig_fmt_clean: bool = false,
    reserved: u6 = 0,
};

const PathMetadata = struct {
    inode: u64,
    file_size: u32,
    mtime_ns: i128,
    verified: Verified,

    // Per-scan state, never serialized:
    //
    //   loaded from disk             -> false
    //   matching cache hit           -> true
    //   newly linted without errors  -> true
    //
    //   completed scan + still false -> omit when saving
    //
    // Deleted or excluded paths are never marked. Changed paths must
    // pass linting again before being retained. Failed scans never save.
    seen_in_scan: bool = false,

    fn matches(metadata: PathMetadata, stat: std.Io.File.Stat) bool {
        return metadata.inode == stat.inode and metadata.file_size == stat.size and
            metadata.mtime_ns == stat.mtime.nanoseconds;
    }
};

comptime {
    assert(@sizeOf(PathMetadata) == 32);
}

// Explicit little-endian fields, never a dump of Zig struct padding:
//
//   header:  magic[8] | version:u32 | record_count:u32
//   record:  path_len:u32 | verified:u8 | inode:u64 | size:u32 |
//            mtime_ns:i128 | path bytes
//
// Keys borrow slices of the loaded buffer. The map is freed before the
// storage arena that owns that buffer and newly discovered paths.
const magic = "ZLCLNIDX";

// Advance for any storage or lint-semantic change that could make a previous
// clean fact unsafe to reuse. Cache compatibility is intentionally one private
// epoch rather than separate format and behavior versions.
const cache_version: u32 = 3;

const index_header_size: usize = 16;
const record_header_size: usize = 33;
const index_size_max: usize = 100 * 1024 * 1024; // 100 MB

/// Create cache state for one complete directory walk.
pub fn init(
    allocator: Allocator,
    io: std.Io,
    root_dir: std.Io.Dir,
    cache_base_dir: []const u8,
    target_zig_version: std.SemanticVersion,
) !LintCache {
    if (cache_base_dir.len == 0) {
        return error.InvalidCacheDir;
    }

    // ── Locate Index ──
    //
    // Separate scanned roots and target Zig contracts beneath the private base
    // directory. This identity does not include file contents.

    var storage_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer storage_arena.deinit();

    const storage_allocator = storage_arena.allocator();
    const root_path = try root_dir.realPathFileAlloc(io, ".", storage_allocator);

    const target_version_shard = try CachePath.allocVersionShard(
        storage_allocator,
        target_zig_version,
    );

    var index_digest: [Blake3.digest_length]u8 = undefined;
    lintIdentityDigest(root_path, &index_digest);

    const index_path = try std.fmt.allocPrint(
        storage_allocator,
        "{s}/lint/{s}/{s}.bin",
        .{
            cache_base_dir,
            target_version_shard,
            std.fmt.bytesToHex(index_digest, .lower),
        },
    );

    // ── Load Records ──

    var cache: LintCache = .{ .storage_arena = storage_arena, .index_path = index_path };

    cache.loadRecords(allocator, io) catch {
        // Loading may stop after inserting some records. No partial index is
        // trusted: one malformed record makes every file a cache miss.
        cache.metadata_by_path.clearRetainingCapacity();

        cache.dirty = true;
    };

    return cache;
}

fn lintIdentityDigest(root_path: []const u8, digest: *[Blake3.digest_length]u8) void {
    var hasher = Blake3.init(.{});

    var encoded_version: [4]u8 = undefined;
    std.mem.writeInt(u32, &encoded_version, cache_version, .little);

    hasher.update("zig-lint-cache\x00");
    hasher.update(&encoded_version);
    hasher.update(builtin.zig_version_string);
    hasher.update(&.{0});
    hasher.update(root_path);
    hasher.final(digest);
}

pub fn deinit(cache: *LintCache, allocator: Allocator) void {
    cache.metadata_by_path.deinit(allocator);
    cache.storage_arena.deinit();
}

fn loadRecords(cache: *LintCache, allocator: Allocator, io: std.Io) !void {
    // readFileAlloc's limit is exclusive; save's byte budget is inclusive:
    //
    //   index_size_max bytes     -> accept after observing EOF
    //   index_size_max + 1 bytes -> reject as too long
    const encoded_index = try std.Io.Dir.cwd().readFileAlloc(
        io,
        cache.index_path,
        cache.storage_arena.allocator(),
        .limited(index_size_max + 1),
    );

    const index_size = encoded_index.len;

    if (index_size < index_header_size or !std.mem.eql(u8, encoded_index[0..8], magic) or
        std.mem.readInt(u32, encoded_index[8..12], .little) != cache_version)
    {
        return error.InvalidCache;
    }

    const record_count = std.mem.readInt(u32, encoded_index[12..16], .little);
    const max_record_count = (index_size - index_header_size) / record_header_size;

    if (record_count > max_record_count) {
        // Bound capacity before trusting a disk-provided count. A tiny
        // corrupt file must not trigger a huge hashmap reservation.
        return error.InvalidCache;
    }

    try cache.metadata_by_path.ensureTotalCapacity(allocator, record_count);

    var read_offset: usize = index_header_size;

    for (0..record_count) |_| {
        if (index_size - read_offset < record_header_size) {
            return error.InvalidCache;
        }

        const record_header = encoded_index[read_offset..][0..record_header_size];
        const path_len = std.mem.readInt(u32, record_header[0..4], .little);
        const verified: Verified = @bitCast(record_header[4]);

        if (!verified.lint_clean or verified.reserved != 0) {
            return error.InvalidCache;
        }

        const metadata: PathMetadata = .{
            .inode = std.mem.readInt(u64, record_header[5..13], .little),
            .file_size = std.mem.readInt(u32, record_header[13..17], .little),
            .mtime_ns = std.mem.readInt(i128, record_header[17..33], .little),
            .verified = verified,
        };

        read_offset += record_header_size;

        // The header has already been consumed. Only the bytes after it
        // can satisfy the encoded path length.
        const remaining_bytes = index_size - read_offset;
        if (path_len == 0 or path_len > remaining_bytes) {
            return error.InvalidCache;
        }

        const relative_path = encoded_index[read_offset..][0..path_len];

        read_offset += path_len;

        const entry = cache.metadata_by_path.getOrPutAssumeCapacity(relative_path);
        if (entry.found_existing) {
            return error.InvalidCache;
        }

        entry.value_ptr.* = metadata;
    }

    if (read_offset != index_size) {
        return error.InvalidCache;
    }
}

/// `lookup` returns verified facts only when the opened file's metadata and
/// this scan's size limit still match. A hit marks the path seen in this scan;
/// a mismatch removes the stale entry so it cannot survive the next save.
pub fn lookup(
    cache: *LintCache,
    relative_path: []const u8,
    stat: std.Io.File.Stat,
    file_size_max: u32,
) ?Verified {
    const metadata = cache.metadata_by_path.getPtr(relative_path) orelse {
        return null;
    };

    if (metadata.file_size <= file_size_max and metadata.matches(stat)) {
        metadata.seen_in_scan = true;

        return metadata.verified;
    }

    _ = cache.metadata_by_path.remove(relative_path);
    cache.dirty = true;

    return null;
}

/// Remember verified facts for one metadata-identified source and mark it seen
/// in this scan. `lint_clean` must be true because records with unknown or
/// failing lint results cannot skip rule execution.
pub fn put(
    cache: *LintCache,
    allocator: Allocator,
    relative_path: []const u8,
    stat: std.Io.File.Stat,
    verified: Verified,
) !void {
    assert(stat.size <= std.math.maxInt(u32));
    assert(verified.lint_clean);
    assert(verified.reserved == 0);

    if (cache.metadata_by_path.getPtr(relative_path)) |metadata| {
        const metadata_changed = !metadata.matches(stat) or
            @as(u8, @bitCast(metadata.verified)) != @as(u8, @bitCast(verified));

        metadata.* = .{
            .inode = stat.inode,
            .file_size = @intCast(stat.size),
            .mtime_ns = stat.mtime.nanoseconds,
            .verified = verified,
            .seen_in_scan = true,
        };
        cache.dirty = cache.dirty or metadata_changed;

        return;
    }

    const metadata: PathMetadata = .{
        .inode = stat.inode,
        .file_size = @intCast(stat.size),
        .mtime_ns = stat.mtime.nanoseconds,
        .verified = verified,
        .seen_in_scan = true,
    };

    // Input paths are borrowed. Unlike loaded keys, new keys need one copy
    // before the caller advances or releases that storage.
    const owned_path = try cache.storage_arena.allocator().dupe(u8, relative_path);

    try cache.metadata_by_path.put(allocator, owned_path, metadata);

    cache.dirty = true;
}

/// Publish only after the complete scan. Unseen paths are omitted; calling
/// this after a partial scan would also discard paths not yet visited.
pub fn save(cache: *LintCache, allocator: Allocator, io: std.Io) !void {
    // Two passes avoid growing and copying the output buffer:
    //
    //   measure fitting records -> header count + exact allocation size
    //   encode the same records -> fill that allocation without growth

    // ── Measure Retained Records ──
    //
    // Only seen records survive. Deleted files disappear, and a warm scan
    // with exactly the same records avoids a cache write altogether.

    var retained_count: u32 = 0;
    var index_size: usize = index_header_size;
    var iterator = cache.metadata_by_path.iterator();

    while (iterator.next()) |entry| {
        if (!entry.value_ptr.seen_in_scan) {
            continue;
        }

        const remaining_bytes = index_size_max - index_size;

        const path_len = entry.key_ptr.len;

        if (remaining_bytes < record_header_size or
            path_len > remaining_bytes - record_header_size)
        {
            // Omit this record rather than abandon the whole index. Keep
            // looking: a later, shorter path may still fit in the budget.
            continue;
        }

        index_size += record_header_size + path_len;
        retained_count += 1;
    }

    if (!cache.dirty and retained_count == cache.metadata_by_path.count()) {
        return;
    }

    // ── Encode And Publish ──

    const encoded_index = try allocator.alloc(u8, index_size);
    defer allocator.free(encoded_index);

    @memcpy(encoded_index[0..8], magic);
    std.mem.writeInt(u32, encoded_index[8..12], cache_version, .little);
    std.mem.writeInt(u32, encoded_index[12..16], retained_count, .little);

    var write_offset: usize = index_header_size;

    iterator = cache.metadata_by_path.iterator();

    // Replay the measurement pass's size check in the same map order.
    // Do not mutate the map between passes: both must select exactly the
    // same records so the header count and allocated byte size stay valid.
    while (iterator.next()) |entry| {
        if (!entry.value_ptr.seen_in_scan) {
            continue;
        }

        const remaining_bytes = index_size_max - write_offset;

        const metadata = entry.value_ptr;
        const relative_path = entry.key_ptr.*;
        const path_len = relative_path.len;

        if (remaining_bytes < record_header_size or
            path_len > remaining_bytes - record_header_size)
        {
            continue;
        }

        const record_header = encoded_index[write_offset..][0..record_header_size];

        std.mem.writeInt(u32, record_header[0..4], @intCast(path_len), .little);

        record_header[4] = @bitCast(metadata.verified);

        std.mem.writeInt(u64, record_header[5..13], metadata.inode, .little);
        std.mem.writeInt(u32, record_header[13..17], metadata.file_size, .little);
        std.mem.writeInt(i128, record_header[17..33], metadata.mtime_ns, .little);

        write_offset += record_header_size;

        @memcpy(encoded_index[write_offset..][0..path_len], relative_path);

        write_offset += path_len;
    }

    // A concurrent reader sees the previous complete index or this one,
    // never a truncated overwrite. Concurrent writers may lose cache hits,
    // but every retained record must still pass the metadata check.
    var output = try std.Io.Dir.cwd().createFileAtomic(io, cache.index_path, .{ .make_path = true, .replace = true });
    defer output.deinit(io);

    try output.file.writeStreamingAll(io, encoded_index);
    try output.replace(io);

    cache.dirty = false;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test "saved records round trip with exact metadata" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const source = "const value = 1;\n";

    // ── Create One Source File And Its Cache Location ──
    //
    // `LintCache.init` hashes the canonical source root and places that root's
    // index beneath `cache_base`. Keeping both paths inside this temporary
    // directory isolates the complete save-and-load cycle.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    // Keep generated cache files outside the source root under test.
    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    const stat = try root_dir.statFile(io, "a.zig", .{});

    // ── Publish The First Scan ──
    //
    // `put` records the metadata read for `a.zig` and marks the path seen in
    // this scan. `save` therefore serializes exactly this one clean record.

    {
        var cache = try LintCache.init(
            allocator,
            io,
            root_dir,
            cache_base,
            @import("builtin").zig_version,
        );
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", stat, .{ .lint_clean = true, .zig_fmt_clean = true });
        try cache.save(allocator, io);

        try std.testing.expect(!cache.dirty);
    }

    // ── Load The Next Scan ──
    //
    // A fresh cache instance must reconstruct every serialized field. Loaded
    // records begin unseen because no path in this new scan has matched yet.

    var loaded = try LintCache.init(
        allocator,
        io,
        root_dir,
        cache_base,
        @import("builtin").zig_version,
    );
    defer loaded.deinit(allocator);

    const metadata = loaded.metadata_by_path.getPtr("a.zig") orelse {
        return error.MissingCacheRecord;
    };

    try std.testing.expectEqual(@as(usize, 1), loaded.metadata_by_path.count());
    try std.testing.expectEqual(stat.inode, metadata.inode);
    try std.testing.expectEqual(@as(u32, @intCast(stat.size)), metadata.file_size);
    try std.testing.expectEqual(stat.mtime.nanoseconds, metadata.mtime_ns);
    try std.testing.expect(metadata.verified.lint_clean);
    try std.testing.expect(metadata.verified.zig_fmt_clean);
    try std.testing.expect(!metadata.seen_in_scan);
    try std.testing.expect(!loaded.dirty);

    // An exact opened-file identity returns a hit and marks this record seen so
    // a subsequent save retains it.
    const verified = loaded.lookup(
        "a.zig",
        stat,
        std.math.maxInt(u32),
    ) orelse {
        return error.MissingCacheHit;
    };

    try std.testing.expect(verified.lint_clean);
    try std.testing.expect(verified.zig_fmt_clean);

    try std.testing.expect(metadata.seen_in_scan);
}

test "cache state does not cross target Zig versions" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const first_target: std.SemanticVersion = .{ .major = 0, .minor = 14, .patch = 0 };
    const second_target: std.SemanticVersion = .{ .major = 0, .minor = 15, .patch = 1 };

    const metadata_target: std.SemanticVersion = .{
        .major = 0,
        .minor = 14,
        .patch = 0,
        .build = "custom.1",
    };

    // ── Publish One Target's Observable Result ──

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "source/a.zig",
        .data = "const value = 1;\n",
    });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base_dir = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base_dir);

    const stat = try root_dir.statFile(io, "a.zig", .{});

    {
        var cache = try LintCache.init(
            allocator,
            io,
            root_dir,
            cache_base_dir,
            first_target,
        );
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", stat, .{ .lint_clean = true });
        try cache.save(allocator, io);
    }

    // ── Observe Version Isolation Without Inspecting Storage ──

    var other_target_cache = try LintCache.init(
        allocator,
        io,
        root_dir,
        cache_base_dir,
        second_target,
    );
    defer other_target_cache.deinit(allocator);

    try std.testing.expect(other_target_cache.lookup(
        "a.zig",
        stat,
        std.math.maxInt(u32),
    ) == null);

    var metadata_target_cache = try LintCache.init(
        allocator,
        io,
        root_dir,
        cache_base_dir,
        metadata_target,
    );
    defer metadata_target_cache.deinit(allocator);

    try std.testing.expect(metadata_target_cache.lookup(
        "a.zig",
        stat,
        std.math.maxInt(u32),
    ) == null);

    var matching_target_cache = try LintCache.init(
        allocator,
        io,
        root_dir,
        cache_base_dir,
        first_target,
    );
    defer matching_target_cache.deinit(allocator);

    try std.testing.expect(matching_target_cache.lookup(
        "a.zig",
        stat,
        std.math.maxInt(u32),
    ) != null);
}

test "saving drops records not seen in the current scan" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const source = "const value = 1;\n";

    // ── Create Two Clean Source Files ──

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = source });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    // Keep generated cache files outside the source root under test.
    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    // ── Publish Both Paths As Seen ──
    //
    // The first completed scan visits both files, so its index starts with two
    // records available to the next scan.

    {
        var cache = try LintCache.init(
            allocator,
            io,
            root_dir,
            cache_base,
            @import("builtin").zig_version,
        );
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", try root_dir.statFile(io, "a.zig", .{}), .{ .lint_clean = true });
        try cache.put(allocator, "b.zig", try root_dir.statFile(io, "b.zig", .{}), .{ .lint_clean = true });
        try cache.save(allocator, io);
    }

    // ── Visit Only `a.zig` In The Next Scan ──
    //
    // Loading resets both records to unseen. The hit marks only `a.zig`; saving
    // then publishes that seen subset and drops the stale `b.zig` record.

    {
        var cache = try LintCache.init(
            allocator,
            io,
            root_dir,
            cache_base,
            @import("builtin").zig_version,
        );
        defer cache.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 2), cache.metadata_by_path.count());

        try std.testing.expect(cache.lookup(
            "a.zig",
            try root_dir.statFile(io, "a.zig", .{}),
            std.math.maxInt(u32),
        ) != null);

        try cache.save(allocator, io);
    }

    // ── Verify The Published Seen Subset ──

    var retained = try LintCache.init(
        allocator,
        io,
        root_dir,
        cache_base,
        @import("builtin").zig_version,
    );
    defer retained.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), retained.metadata_by_path.count());
    try std.testing.expect(retained.metadata_by_path.contains("a.zig"));
    try std.testing.expect(!retained.metadata_by_path.contains("b.zig"));
}

test "malformed indexes never become partial cache hits" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const source = "const value = 1;\n";

    // ── Create A Valid Two-Record Index ──
    //
    // Two records let truncation and trailing-data cases fail after the decoder
    // has already accepted a valid prefix. The cache must still discard that
    // prefix rather than expose a partial hit.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = source });

    var root_dir = try tmp_dir.dir.openDir(io, "source", .{});
    defer root_dir.close(io);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    // Keep generated cache files outside the source root under test.
    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    const index_path = path: {
        var cache = try LintCache.init(
            allocator,
            io,
            root_dir,
            cache_base,
            @import("builtin").zig_version,
        );
        defer cache.deinit(allocator);

        try cache.put(allocator, "a.zig", try root_dir.statFile(io, "a.zig", .{}), .{ .lint_clean = true });
        try cache.put(allocator, "b.zig", try root_dir.statFile(io, "b.zig", .{}), .{ .lint_clean = true });
        try cache.save(allocator, io);

        break :path try allocator.dupe(u8, cache.index_path);
    };
    defer allocator.free(index_path);

    const encoded = try std.Io.Dir.cwd().readFileAlloc(
        io,
        index_path,
        allocator,
        .limited(4096),
    );
    defer allocator.free(encoded);

    // ── Corrupt One Format Invariant At A Time ──
    //
    // Every case starts from the same valid bytes:
    //
    //   header | first complete record | second complete record
    //
    // `LintCache.init` catches a decoding failure, clears every record already
    // inserted from the valid prefix, and marks the index dirty for replacement.
    const Corruption = enum {
        bad_magic,
        unsupported_version,
        invalid_verified_bits,
        impossible_record_count,
        truncated_final_record,
        trailing_byte,
        duplicate_path,
        empty_path,
    };

    for (std.enums.values(Corruption)) |corruption| {
        var malformed: std.ArrayList(u8) = .empty;
        defer malformed.deinit(allocator);

        try malformed.appendSlice(allocator, encoded);

        switch (corruption) {
            .bad_magic => malformed.items[0] ^= 1,
            .unsupported_version => std.mem.writeInt(u32, malformed.items[8..12], 99, .little),
            .invalid_verified_bits => malformed.items[index_header_size + 4] |= 0x80,
            .impossible_record_count => std.mem.writeInt(
                u32,
                malformed.items[12..16],
                std.math.maxInt(u32),
                .little,
            ),
            .truncated_final_record => {
                // The complete first record may already be in the map when the
                // missing final path byte invalidates the second record.
                malformed.shrinkRetainingCapacity(encoded.len - 1);
            },

            .trailing_byte => {
                // Both declared records parse successfully, but the unconsumed
                // byte proves the index is not the canonical complete encoding.
                try malformed.append(allocator, 0);
            },

            .duplicate_path => {
                // Append a byte-for-byte copy of the first record and include it
                // in the declared count. Duplicate paths make the index ambiguous.
                const first_path_len = std.mem.readInt(
                    u32,
                    encoded[index_header_size..][0..4],
                    .little,
                );

                const first_record_end = index_header_size + record_header_size + first_path_len;
                const record_count = std.mem.readInt(u32, encoded[12..16], .little);

                try malformed.appendSlice(allocator, encoded[index_header_size..first_record_end]);
                std.mem.writeInt(u32, malformed.items[12..16], record_count + 1, .little);
            },
            .empty_path => std.mem.writeInt(
                u32,
                malformed.items[index_header_size..][0..4],
                0,
                .little,
            ),
        }

        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = index_path,
            .data = malformed.items,
        });

        var loaded = try LintCache.init(
            allocator,
            io,
            root_dir,
            cache_base,
            @import("builtin").zig_version,
        );
        defer loaded.deinit(allocator);

        // No valid prefix survives. `dirty` records that the malformed index
        // must be replaced if this scan later completes and saves.
        try std.testing.expectEqual(@as(usize, 0), loaded.metadata_by_path.count());
        try std.testing.expect(loaded.dirty);
    }
}
