//! Lint clean       -> cache metadata
//! Any diagnostics  -> never cache
//!
//! base_dir/
//! └── hex(BLAKE3(canonical_root_path)).bin
//!     └── root-relative/path.zig -> { inode, file_size, mtime_ns }
//!
//! Lookup:
//!   path found + metadata matches + size <= limit -> hit
//!   missing / changed / oversized                -> miss for this file only
//!
//! Contents are NOT hashed -> metadata-preserving edits can be missed.
//! Linter / Zig / rules change
//!   -> caller versions base_dir, clears the cache, or disables caching.

const LintCache = @This();

const std = @import("std");
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

metadata_by_path: std.StringHashMapUnmanaged(PathMetadata) = .empty,
storage_arena: std.heap.ArenaAllocator,
index_path: []const u8,

// Insertions, invalidations, or an unusable loaded index require a write.
// Unseen paths are detected separately by save's retained count.
dirty: bool = false,

// The scan limit is u32, so every admitted file size fits without loss.
// Keeping size at that width leaves room for seen_in_scan within a 32-byte record,
// rather than padding a u64-sized record to 48 bytes around the i128 mtime.
const PathMetadata = struct {
    inode: u64,
    file_size: u32,
    mtime_ns: i128,

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
//   record:  path_len:u32 | inode:u64 | size:u32 | mtime_ns:i128 | path bytes
//
// Keys borrow slices of the loaded buffer. The map is freed before the
// storage arena that owns that buffer and newly discovered paths.
const magic = "ZLCLNIDX";

const format_version: u32 = 1;
const index_header_size: usize = 16;
const record_header_size: usize = 32;
const index_size_max: usize = 100 * 1024 * 1024; // 100 MB

/// Create per-scan state; initialize a fresh cache for each directory walk.
pub fn init(
    allocator: Allocator,
    io: std.Io,
    root_dir: std.Io.Dir,
    cache_base_dir: []const u8,
) !LintCache {
    if (cache_base_dir.len == 0) {
        return error.InvalidCacheDirectory;
    }

    // ── Locate Index ──
    //
    // Separate scanned roots beneath the caller's base directory. The
    // caller may include an implementation version in that base path.
    // This hash identifies the scanned root; file contents are not hashed.

    var storage_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer storage_arena.deinit();

    const storage_allocator = storage_arena.allocator();
    const root_path = try root_dir.realPathFileAlloc(io, ".", storage_allocator);

    var root_digest: [Blake3.digest_length]u8 = undefined;
    Blake3.hash(root_path, &root_digest, .{});

    const index_path = try std.fmt.allocPrint(storage_allocator, "{s}/{s}.bin", .{
        cache_base_dir,
        std.fmt.bytesToHex(root_digest, .lower),
    });

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
        std.mem.readInt(u32, encoded_index[8..12], .little) != format_version)
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

        const metadata: PathMetadata = .{
            .inode = std.mem.readInt(u64, record_header[4..12], .little),
            .file_size = std.mem.readInt(u32, record_header[12..16], .little),
            .mtime_ns = std.mem.readInt(i128, record_header[16..32], .little),
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

/// A hit marks the path seen in this scan. A failed metadata check or an
/// oversized cached file removes that entry so it cannot survive the next save.
pub fn checkHit(
    cache: *LintCache,
    io: std.Io,
    parent_dir: std.Io.Dir,
    basename: []const u8,
    relative_path: []const u8,
    file_size_max: u32,
) bool {
    const metadata = cache.metadata_by_path.getPtr(relative_path) orelse {
        return false;
    };

    // The size limit belongs to this scan, not the binary format. Lowering
    // it invalidates only this entry; the other clean records remain useful.
    const stat = if (metadata.file_size <= file_size_max)
        parent_dir.statFile(io, basename, .{}) catch null
    else
        null;

    if (stat) |current_stat| {
        if (metadata.matches(current_stat)) {
            metadata.seen_in_scan = true;

            return true;
        }
    }

    // A failed metadata read or a changed file must never leave the old
    // clean result reusable.
    // The same applies when the recorded size exceeds this scan's limit.
    _ = cache.metadata_by_path.remove(relative_path);
    cache.dirty = true;

    return false;
}

/// Remember a file linted without diagnostics and mark it seen in this scan.
/// Pass its pre-read stat, with size verified against the linted byte count
/// and the scan's u32 file-size limit.
pub fn put(
    cache: *LintCache,
    allocator: Allocator,
    relative_path: []const u8,
    stat: std.Io.File.Stat,
) !void {
    assert(stat.size <= std.math.maxInt(u32));

    const metadata: PathMetadata = .{
        .inode = stat.inode,
        .file_size = @intCast(stat.size),
        .mtime_ns = stat.mtime.nanoseconds,
        .seen_in_scan = true,
    };

    // Walker paths are temporary. Unlike loaded keys, new keys need one
    // copy before advancing the directory walker.
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
    std.mem.writeInt(u32, encoded_index[8..12], format_version, .little);
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
        std.mem.writeInt(u64, record_header[4..12], metadata.inode, .little);
        std.mem.writeInt(u32, record_header[12..16], metadata.file_size, .little);
        std.mem.writeInt(i128, record_header[16..32], metadata.mtime_ns, .little);

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
}
