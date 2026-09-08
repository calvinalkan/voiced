const Linter = @This();

const std = @import("std");
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

const Diagnostics = @import("Diagnostics.zig");
const FileContext = @import("FileContext.zig");

pub const File = struct {
    path: []const u8,
    text: [:0]const u8,
};

const LintRule = *const fn (
    Allocator,
    Allocator,
    []const u8,
    *const FileContext,
    *Diagnostics,
) Allocator.Error!void;

const lint_rules: []const LintRule = &.{
    @import("rules/hidden_control_flow.zig").lint,
    @import("rules/explicit_optional_unwrap.zig").lint,
    @import("rules/blank_line_before_control_flow_exit.zig").lint,
    @import("rules/visible_resource_lifetime.zig").lint,
};

pub const empty: Linter = .{};

diagnostics: Diagnostics = .empty,

/// Return `error.FileTooLarge` before parsing if `file.text` exceeds
/// `file_size_max` bytes. Files exactly at the limit are accepted.
/// Syntax errors and lint violations are appended to `diagnostics`.
pub fn lint_file(
    linter: *Linter,
    file_allocator: Allocator,
    diagnostic_allocator: Allocator,
    file: File,
    file_size_max: u32,
) (Allocator.Error || error{FileTooLarge})!void {
    if (file.text.len > file_size_max) {
        return error.FileTooLarge;
    }

    var ast = try std.zig.Ast.parse(file_allocator, file.text, .zig);
    defer ast.deinit(file_allocator);

    const context = try FileContext.init(file_allocator, ast, file_size_max);
    defer file_allocator.free(context.line_starts);

    if (ast.errors.len != 0) {
        try linter.diagnostics.add_at_token(
            diagnostic_allocator,
            file.path,
            &context,
            ast.errors[0].token,
            "parse",
            "this file does not parse; fix the syntax error before linting",
        );

        return;
    }

    for (lint_rules) |run_rule| {
        try run_rule(file_allocator, diagnostic_allocator, file.path, &context, &linter.diagnostics);
    }
}

const file_size_max_default = 1 * 1024 * 1024; // 1 MiB

pub const LintDirectoryOptions = struct {
    /// Null disables caching. The caller supplies a trusted, writable cache
    /// directory; the library does not consult XDG or HOME.
    cache: ?struct {
        /// The caller owns invalidation when the linter implementation, Zig
        /// version, or effective rules change. Use a versioned base directory
        /// (for example `/tmp/lint-cache/v2`), clear the cache, or disable it.
        /// The library adds no implementation-version directory automatically.
        /// A commit ID alone does not cover dirty builds.
        base_dir: []const u8,
    } = null,
    /// Inclusive source-byte limit; zero permits only empty files. Cache entries
    /// exceeding this limit are discarded individually, without invalidating
    /// other entries or requiring a different cache directory.
    file_size_max: u32 = file_size_max_default,
};

/// Recursively lint `.zig` files beneath `dir_path`, appending diagnostics.
///
/// Skips descendant directories named `.git`, `.zig-cache`, `.cache`,
/// `zig-cache`, `zig-out`, `zig-pkg`, or `__fixtures__` at any depth.
///
/// Files in `__fixtures__` can still be checked explicitly with `lint_file`.
///
/// Files larger than `options.file_size_max` produce a file-level diagnostic without
/// being parsed; scanning continues with the remaining files.
///
/// Optional clean-file caching uses relative path, inode, size, and nanosecond
/// mtime, not content hashes. Changes preserving that metadata can be missed;
/// pass `.{}` for an unconditional scan. Cache files live directly beneath
/// the supplied base: `<root-hash>.bin`. Cache failures
/// do not fail linting. Only a completed scan publishes updated cache records.
pub fn lintDirectory(
    linter: *Linter,
    scratch_allocator: Allocator,
    diagnostic_allocator: Allocator,
    io: std.Io,
    dir_path: []const u8,
    options: LintDirectoryOptions,
) !void {
    // ── Prepare Scan ──

    var dir_handle = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir_handle.close(io);

    var cache: ?CleanFilesCache = CleanFilesCache.init(scratch_allocator, io, dir_handle, options) catch null;
    defer if (cache) |*clean_cache| {
        clean_cache.deinit(scratch_allocator);
    };

    var walker = try dir_handle.walk(scratch_allocator);
    defer walker.deinit();

    // Allocate only on the first miss. An entirely cached scan needs neither
    // this configurable read buffer nor its ReleaseSafe allocation/free poison fills.
    var file_buffer: ?[]u8 = null;
    defer if (file_buffer) |buffer| {
        scratch_allocator.free(buffer);
    };

    var file_arena = std.heap.ArenaAllocator.init(scratch_allocator);
    defer file_arena.deinit();

    // ── Check Files ──

    while (true) {
        const entry = try walker.next(io) orelse {
            break;
        };

        if (entry.kind == .directory) {
            if (std.mem.eql(u8, entry.basename, ".git") or
                std.mem.eql(u8, entry.basename, ".zig-cache") or
                std.mem.eql(u8, entry.basename, ".cache") or
                std.mem.eql(u8, entry.basename, "zig-cache") or
                std.mem.eql(u8, entry.basename, "zig-out") or
                std.mem.eql(u8, entry.basename, "zig-pkg") or
                std.mem.eql(u8, entry.basename, "__fixtures__"))
            {
                walker.leave(io);
            }

            continue;
        }

        if (!std.mem.endsWith(u8, entry.basename, ".zig")) {
            continue;
        }

        if (cache) |*clean_files_cache| {
            if (clean_files_cache.has(io, entry.dir, entry.basename, entry.path, options.file_size_max)) {
                continue;
            }
        }

        const buffer = file_buffer orelse buffer: {
            // Read one byte beyond the accepted file limit. A full buffer means the
            // file is either exactly this size or larger; both exceed `file_size_max`.
            // A shorter read leaves that extra byte available for the zero sentinel
            // required by `Ast.parse`'s `[:0]const u8` input.
            const buffer_size = std.math.add(usize, options.file_size_max, 1) catch {
                return error.OutOfMemory;
            };

            const allocated = try scratch_allocator.alloc(u8, buffer_size);
            file_buffer = allocated;

            break :buffer allocated;
        };

        var source_file = try entry.dir.openFile(io, entry.basename, .{});
        defer source_file.close(io);

        // Cache the pre-read metadata, even if the file changes during linting.
        // A metadata change makes this entry miss on the next scan. Using newer
        // metadata instead could associate a clean result with unread contents.
        const stat_before = if (cache) |_|
            source_file.stat(io) catch null
        else
            null;

        var reader = source_file.reader(io, &.{});

        const read_len = reader.interface.readSliceShort(buffer) catch |err|
            switch (err) {
                error.ReadFailed => return reader.err orelse {
                    unreachable;
                },
            };

        if (read_len == buffer.len) {
            // Messages are borrowed by Diagnostics, so this formatted limit
            // must live with the diagnostic rather than the reusable file arena.
            const message = try std.fmt.allocPrint(diagnostic_allocator, "file is larger than {d} bytes; split it.", .{options.file_size_max});
            errdefer diagnostic_allocator.free(message);

            try linter.diagnostics.add(diagnostic_allocator, .{
                .path = entry.path,
                .line = 1,
                .column = 1,
                .rule_name = "parse",
                .message = message,
            });

            continue;
        }

        // Terminate the populated prefix using the extra byte reserved above.
        // The sentinel is available at `text[text.len]` but is not part of `text`.
        buffer[read_len] = 0;

        _ = file_arena.reset(.retain_capacity);

        const diagnostics_before = linter.diagnostics.count();

        try linter.lint_file(
            file_arena.allocator(),
            diagnostic_allocator,
            .{ .path = entry.path, .text = buffer[0..read_len :0] },
            options.file_size_max,
        );

        const clean_cache = if (cache) |*value|
            value
        else {
            continue;
        };

        const metadata = stat_before orelse {
            continue;
        };

        if (linter.diagnostics.count() != diagnostics_before or metadata.size != read_len) {
            continue;
        }

        const record: CleanFilesCache.Record = .{
            .inode = metadata.inode,
            .size = @intCast(metadata.size),
            .mtime_ns = metadata.mtime.nanoseconds,
            .seen = true,
        };

        clean_cache.put(scratch_allocator, entry.path, record) catch {
            // Ignore errors if cache is full; the cache is best effort and not critical for linting.
        };
    }

    // Reaching here means the walk completed. Failed scans leave the previous
    // disk index intact; its metadata is still checked on the next invocation.
    if (cache) |*clean_cache| {
        clean_cache.save(scratch_allocator, io) catch {};
    }
}

const CleanFilesCache = struct {
    records: std.StringHashMapUnmanaged(Record) = .empty,
    storage: std.heap.ArenaAllocator,
    abs_path: []const u8,
    dirty: bool = false,

    // The scan limit is u32, so every admitted file size fits without loss.
    // Keeping size at that width leaves room for seen within a 32-byte record,
    // rather than padding a u64-sized record to 48 bytes around the i128 mtime.
    const Record = struct {
        inode: u64,
        size: u32,
        mtime_ns: i128,
        seen: bool = false,

        fn matches(record: Record, stat: std.Io.File.Stat) bool {
            return record.inode == stat.inode and record.size == stat.size and
                record.mtime_ns == stat.mtime.nanoseconds;
        }
    };

    comptime {
        assert(@sizeOf(Record) == 32);
    }

    // Explicit little-endian fields, never a dump of Zig struct padding:
    //
    //   header:  magic[8] | version:u32 | record_count:u32
    //   record:  path_len:u32 | inode:u64 | size:u32 | mtime_ns:i128 | path bytes
    //
    // Keys borrow slices of the loaded buffer. The map is freed before the
    // storage arena that owns that buffer and newly discovered paths.
    const magic = "ZLCLNIDX";

    // Version 2 narrows the encoded size too. Version 1 indexes rebuild once;
    // within this format, a lower scan limit still drops only oversized entries.
    const version = 2;
    const header_size_expected = 16;
    const record_size_expected = 32;
    const cache_size_max = 100 * 1024 * 1024; // 100 MB

    fn init(
        allocator: Allocator,
        io: std.Io,
        base_dir_handle: std.Io.Dir,
        options: LintDirectoryOptions,
    ) !?CleanFilesCache {
        const configuration = options.cache orelse {
            return null;
        };

        if (configuration.base_dir.len == 0) {
            return error.InvalidCacheDirectory;
        }

        // ── Locate Index ──
        //
        // Separate scanned roots beneath the caller's base directory. The
        // caller may include an implementation version in that base path.
        // This hash identifies the scanned root; file contents are not hashed.

        var storage_arena = std.heap.ArenaAllocator.init(allocator);
        errdefer storage_arena.deinit();

        const storage = storage_arena.allocator();
        const base_dir_path = try base_dir_handle.realPathFileAlloc(io, ".", storage);

        var base_dir_digest: [Blake3.digest_length]u8 = undefined;
        Blake3.hash(base_dir_path, &base_dir_digest, .{});

        const cache_path = try std.fmt.allocPrint(storage, "{s}/{s}.bin", .{
            configuration.base_dir,
            std.fmt.bytesToHex(base_dir_digest, .lower),
        });

        // ── Load Records ──

        var cache: CleanFilesCache = .{ .storage = storage_arena, .abs_path = cache_path };

        cache.init_load_records(allocator, io) catch {
            // Loading may stop after inserting some records. No partial index is
            // trusted: one malformed record makes every file a cache miss.
            cache.records.clearRetainingCapacity();
            cache.dirty = true;
        };

        return cache;
    }

    fn deinit(cache: *CleanFilesCache, allocator: Allocator) void {
        cache.records.deinit(allocator);
        cache.storage.deinit();
    }

    fn init_load_records(cache: *CleanFilesCache, allocator: Allocator, io: std.Io) !void {
        const cache_bin = try std.Io.Dir.cwd().readFileAlloc(
            io,
            cache.abs_path,
            cache.storage.allocator(),
            .limited(cache_size_max),
        );
        const cache_size = cache_bin.len;

        if (cache_size < header_size_expected or !std.mem.eql(u8, cache_bin[0..8], magic) or
            std.mem.readInt(u32, cache_bin[8..12], .little) != version)
        {
            return error.InvalidCache;
        }

        const count = std.mem.readInt(u32, cache_bin[12..16], .little);

        if (count > (cache_size - header_size_expected) / record_size_expected) {
            // Bound capacity before trusting a disk-provided count. A tiny
            // corrupt file must not trigger a huge hashmap reservation.
            return error.InvalidCache;
        }

        try cache.records.ensureTotalCapacity(allocator, count);

        var scan_offset: usize = header_size_expected;

        for (0..count) |_| {
            if (cache_size - scan_offset < record_size_expected) {
                return error.InvalidCache;
            }

            const fields = cache_bin[scan_offset..][0..record_size_expected];
            const path_size = std.mem.readInt(u32, fields[0..4], .little);

            const record: Record = .{
                .inode = std.mem.readInt(u64, fields[4..12], .little),
                .size = std.mem.readInt(u32, fields[12..16], .little),
                .mtime_ns = std.mem.readInt(i128, fields[16..32], .little),
            };

            scan_offset += record_size_expected;

            if (path_size == 0 or path_size > cache_size - scan_offset) {
                return error.InvalidCache;
            }

            const path = cache_bin[scan_offset..][0..path_size];

            scan_offset += path_size;

            const entry = cache.records.getOrPutAssumeCapacity(path);
            if (entry.found_existing) {
                return error.InvalidCache;
            }

            entry.value_ptr.* = record;
        }

        if (scan_offset != cache_size) {
            return error.InvalidCache;
        }
    }

    fn has(
        cache: *CleanFilesCache,
        io: std.Io,
        dir_handle: std.Io.Dir,
        name: []const u8,
        path: []const u8,
        file_size_max: u32,
    ) bool {
        const record = cache.records.getPtr(path) orelse {
            return false;
        };

        // The size limit belongs to this scan, not the binary format. Lowering
        // it invalidates only this entry; the other clean records remain useful.
        const stat = if (record.size <= file_size_max)
            dir_handle.statFile(io, name, .{}) catch null
        else
            null;

        if (stat) |metadata| {
            if (record.matches(metadata)) {
                record.seen = true;

                return true;
            }
        }

        // A failed metadata read or a changed file must never leave the old
        // clean result reusable.
        // The same applies when the recorded size exceeds this scan's limit.
        _ = cache.records.remove(path);
        cache.dirty = true;

        return false;
    }

    fn put(
        cache: *CleanFilesCache,
        allocator: Allocator,
        path: []const u8,
        record: Record,
    ) !void {
        // Walker paths are temporary. Unlike loaded keys, new keys need one
        // copy before advancing the directory walker.
        const owned_path = try cache.storage.allocator().dupe(u8, path);

        try cache.records.put(allocator, owned_path, record);

        cache.dirty = true;
    }

    fn save(cache: *CleanFilesCache, allocator: Allocator, io: std.Io) !void {
        // ── Measure Retained Records ──
        //
        // Only seen records survive. Deleted files disappear, and a warm scan
        // with exactly the same records avoids a cache write altogether.

        var record_count: u32 = 0;
        var cache_size: usize = header_size_expected;
        var iterator = cache.records.iterator();

        while (iterator.next()) |entry| {
            if (!entry.value_ptr.seen) {
                continue;
            }

            const remaining_space = cache_size_max - cache_size;

            const path_size = entry.key_ptr.len;
            if (remaining_space < record_size_expected or
                path_size > remaining_space - record_size_expected)
            {
                // Omit this record rather than abandon the whole index. Keep
                // looking: a later, shorter path may still fit in the budget.
                continue;
            }

            cache_size += record_size_expected + path_size;
            record_count += 1;
        }

        if (!cache.dirty and record_count == cache.records.count()) {
            return;
        }

        // ── Encode And Publish ──

        const bytes = try allocator.alloc(u8, cache_size);
        defer allocator.free(bytes);

        @memcpy(bytes[0..8], magic);
        std.mem.writeInt(u32, bytes[8..12], version, .little);
        std.mem.writeInt(u32, bytes[12..16], record_count, .little);

        var scan_offset: usize = header_size_expected;
        iterator = cache.records.iterator();

        // Replay the measurement pass's size check in the same map order.
        // Do not mutate the map between passes: both must select exactly the
        // same records so the header count and allocated byte size stay valid.
        while (iterator.next()) |entry| {
            if (!entry.value_ptr.seen) {
                continue;
            }

            const remaining_space = cache_size_max - scan_offset;

            const record = entry.value_ptr;
            const key = entry.key_ptr.*;
            const key_size = key.len;

            if (remaining_space < record_size_expected or
                key_size > remaining_space - record_size_expected)
            {
                continue;
            }

            const fields = bytes[scan_offset..][0..record_size_expected];

            std.mem.writeInt(u32, fields[0..4], @intCast(key_size), .little);
            std.mem.writeInt(u64, fields[4..12], record.inode, .little);
            std.mem.writeInt(u32, fields[12..16], record.size, .little);
            std.mem.writeInt(i128, fields[16..32], record.mtime_ns, .little);

            scan_offset += record_size_expected;
            @memcpy(bytes[scan_offset..][0..key_size], key);
            scan_offset += key_size;
        }

        // A concurrent reader sees the previous complete index or this one,
        // never a truncated overwrite. Concurrent writers may lose cache hits,
        // but every retained record must still pass the metadata check.
        var output = try std.Io.Dir.cwd().createFileAtomic(io, cache.abs_path, .{ .make_path = true, .replace = true });
        defer output.deinit(io);

        try output.file.writeStreamingAll(io, bytes);
        try output.replace(io);
    }
};
