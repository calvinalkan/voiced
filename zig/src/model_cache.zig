//! Owns verified model installation reads and atomic packed-cache publication.
//! Cache entries wrap the runtime image in an aligned provenance/checksum header;
//! the runtime itself remains independent of filesystems and XDG policy.

const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.model_cache);
const inference = @import("inference");
const models = @import("models");
const linux = std.os.linux;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Blake3 = std.crypto.hash.Blake3;

// Cache ABI 1 stores the pinned source SHA-256 at 16..48 and a BLAKE3-256
// digest of the complete packed image at 48..80. Version the envelope separately
// from model packing so a different checksum algorithm cannot be misread.
const cache_format_version: u32 = 1;
const cache_header_size: usize = 128;
const cache_magic = "VOICEDC\x00";

/// `LoadedModel` owns a converted image or a read-only cache mapping. Destroy
/// borrowing runtimes before calling `deinit`; moving an initialized runtime's
/// model or changing the underlying cache inode in place is not supported.
pub const LoadedModel = struct {
    model: inference.Model,
    mapping: ?[]align(std.heap.page_size_min) u8 = null,

    pub fn deinit(loaded: *LoadedModel) void {
        loaded.model.deinit();
        if (loaded.mapping) |mapping| {
            std.posix.munmap(mapping);
        }
        loaded.* = undefined;
    }
};

/// `loadModel` loads one pinned model from a checksummed cache, or converts its
/// verified installed weights on a miss. Cache failures are warnings, not a
/// requirement for inference. Successful cache reads borrow file-backed pages;
/// converted results retain no pristine source buffer.
pub fn loadModel(init: std.process.Init, selected: models.Model) !LoadedModel {
    const cache_started = std.Io.Clock.awake.now(init.io);
    const allocator = init.gpa;
    const source = selected.metadata().weights;

    var source_digest: [Sha256.digest_length]u8 = undefined;

    _ = try std.fmt.hexToBytes(&source_digest, source.sha256);

    const kind: inference.ModelKind = switch (selected) {
        .systran_base_en => .base_en,
        .systran_small_en => .small_en,
    };

    // ── Resolve And Lock The Cache Entry ──
    // A directory lock survives atomic file replacement and is released on
    // worker death. Readers recheck after acquiring it, so concurrent misses
    // do not each convert a large pristine model.

    const directory = openCacheDirectory(init, selected) catch |err| unavailable: {
        log.warn(.{}, "Model cache unavailable: error={s}\n", .{@errorName(err)});
        break :unavailable null;
    };
    defer if (directory) |dir| dir.close(init.io);

    if (directory) |dir| {
        if (tryReadCached(init.io, dir, kind, source_digest)) |loaded| {
            log.info(.{}, "Model cache loaded: model={s}, model_cache_load_duration_ms={d:.3}", .{ selected.name(), @as(f64, @floatFromInt(cache_started.untilNow(init.io, .awake).nanoseconds)) / std.time.ns_per_ms });
            return loaded;
        }
    }

    // ── Convert Verified Pristine Weights ──
    // Unmap the source before inference workspace allocation. Keeping both
    // model representations for the worker's lifetime doubles retained data.

    log.debug(.{}, "Model cache lookup completed: model={s}, cache_available={}, cache_hit=false, model_cache_lookup_duration_ms={d:.3}", .{ selected.name(), directory != null, @as(f64, @floatFromInt(cache_started.untilNow(init.io, .awake).nanoseconds)) / std.time.ns_per_ms });
    const conversion_started = std.Io.Clock.awake.now(init.io);
    var loaded: LoadedModel = block: {
        const model_directory = try models.allocInstalledDirectoryPath(init, selected);
        defer allocator.free(model_directory);

        const path = try std.fs.path.join(allocator, &.{ model_directory, "model.bin" });
        defer allocator.free(path);

        const file = try std.Io.Dir.cwd().openFile(init.io, path, .{ .mode = .read_only, .allow_directory = false });
        defer file.close(init.io);
        const stat = try file.stat(init.io);
        if (stat.kind != .file or stat.size != source.size) {
            return error.InvalidPristineWeights;
        }
        const mapping = try std.posix.mmap(null, source.size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        defer std.posix.munmap(mapping);
        var digest: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(mapping, &digest, .{});
        if (!std.mem.eql(u8, &digest, &source_digest)) {
            return error.InvalidPristineWeights;
        }
        break :block .{ .model = try inference.Model.fromPristineWeights(allocator, kind, mapping) };
    };
    errdefer loaded.deinit();
    log.info(.{}, "Model converted: model={s}, model_convert_duration_ms={d:.3}", .{ selected.name(), @as(f64, @floatFromInt(conversion_started.untilNow(init.io, .awake).nanoseconds)) / std.time.ns_per_ms });

    // ── Publish One Complete Cache Entry ──
    // The envelope and image share one atomic rename; there is no separately
    // published checksum sidecar that could describe a different generation.

    if (directory) |dir| {
        const publication_started = std.Io.Clock.awake.now(init.io);
        saveCached(init.io, dir, loaded.model.packedImage(), source_digest) catch |err| {
            log.warn(.{}, "Model cache save failed: error={s}\n", .{@errorName(err)});
            return loaded;
        };
        log.debug(.{}, "Model cache published: model={s}, model_cache_publish_duration_ms={d:.3}", .{ selected.name(), @as(f64, @floatFromInt(publication_started.untilNow(init.io, .awake).nanoseconds)) / std.time.ns_per_ms });
        const remap_started = std.Io.Clock.awake.now(init.io);
        if (tryReadCached(init.io, dir, kind, source_digest)) |mapped| {
            log.debug(.{}, "Model cache remapped: model={s}, model_cache_remap_duration_ms={d:.3}", .{ selected.name(), @as(f64, @floatFromInt(remap_started.untilNow(init.io, .awake).nanoseconds)) / std.time.ns_per_ms });
            loaded.deinit();
            return mapped;
        }
    }
    return loaded;
}

/// `loadVocabulary` returns verified GPT-2 token spellings owned by `init.gpa`.
/// Keep the bytes alive until the borrowing runtime has been deinitialized.
pub fn loadVocabulary(init: std.process.Init, selected: models.Model) ![]u8 {
    const directory = try models.allocInstalledDirectoryPath(init, selected);
    defer init.gpa.free(directory);

    const path = try std.fs.path.join(init.gpa, &.{ directory, "vocabulary.txt" });
    defer init.gpa.free(path);

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, .limited(4 * 1024 * 1024));
    errdefer init.gpa.free(text);

    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(text, &digest, .{});

    const hex = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &hex, models.vocabulary.sha256)) {
        return error.InvalidVocabulary;
    }

    return text;
}

fn openCacheDirectory(init: std.process.Init, selected: models.Model) !std.Io.Dir {
    const allocator = init.gpa;
    const configured = init.environ_map.get("XDG_CACHE_HOME");
    const root = if (configured != null and std.fs.path.isAbsolute(configured.?))
        try allocator.dupe(u8, configured.?)
    else
        try std.fs.path.join(allocator, &.{ init.environ_map.get("HOME") orelse {
            return error.HomeNotSet;
        }, ".cache" });
    defer allocator.free(root);
    if (!std.fs.path.isAbsolute(root)) {
        return error.CacheHomeNotAbsolute;
    }
    const revision = try std.fmt.allocPrint(allocator, "cache-{d}-packed-{d}-format-{d}", .{ cache_format_version, inference.packed_model_cache_version, inference.packed_model_image_format_version });
    defer allocator.free(revision);
    const path = try std.fs.path.join(allocator, &.{ root, "voiced/models", selected.name(), selected.metadata().weights.sha256, revision });
    defer allocator.free(path);
    // Locking and fsync require a readable directory descriptor, not O_PATH.
    const directory = try std.Io.Dir.cwd().createDirPathOpen(init.io, path, .{ .permissions = .fromMode(0o700), .open_options = .{ .iterate = true, .follow_symlinks = false } });
    errdefer directory.close(init.io);
    try requirePrivateOwner(directory.handle);
    const lock: std.Io.File = .{ .handle = directory.handle, .flags = .{ .nonblocking = false } };
    try lock.lock(init.io, .exclusive);
    return directory;
}

fn tryReadCached(io: std.Io, directory: std.Io.Dir, kind: inference.ModelKind, source_digest: [32]u8) ?LoadedModel {
    return readCached(io, directory, kind, source_digest) catch |err| {
        if (err != error.FileNotFound) log.warn(.{}, "Model cache rejected: error={s}\n", .{@errorName(err)});
        return null;
    };
}

fn readCached(io: std.Io, directory: std.Io.Dir, kind: inference.ModelKind, source_digest: [32]u8) !LoadedModel {
    const file = try directory.openFile(io, "model.voiced", .{ .mode = .read_only, .follow_symlinks = false, .allow_directory = false });
    defer file.close(io);
    try requirePrivateOwner(file.handle);
    const stat = try file.stat(io);
    if (stat.kind != .file or stat.size <= cache_header_size or stat.size > 512 * 1024 * 1024 + cache_header_size) {
        return error.InvalidCacheImage;
    }
    const mapping = try std.posix.mmap(null, @intCast(stat.size), .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    errdefer std.posix.munmap(mapping);
    const header = mapping[0..cache_header_size];
    if (!std.mem.eql(u8, header[0..8], cache_magic) or
        std.mem.readInt(u32, header[8..12], .little) != inference.packed_model_cache_version or
        !std.mem.eql(u8, header[16..48], &source_digest) or
        std.mem.readInt(u64, header[80..88], .little) != mapping.len - cache_header_size or
        std.mem.readInt(u32, header[12..16], .little) != cache_format_version or !std.mem.allEqual(u8, header[88..], 0))
    {
        return error.InvalidCacheImage;
    }
    const image: []align(inference.packed_model_image_alignment) const u8 = @alignCast(mapping[cache_header_size..]);
    var digest: [Blake3.digest_length]u8 = undefined;
    Blake3.hash(image, &digest, .{});
    if (!std.mem.eql(u8, &digest, header[48..80])) {
        return error.CacheChecksumMismatch;
    }
    var model = try inference.Model.fromPackedImage(image);
    errdefer model.deinit();
    if (model.kind != kind) {
        return error.InvalidCacheModel;
    }
    return .{ .model = model, .mapping = mapping };
}

fn saveCached(io: std.Io, directory: std.Io.Dir, image: []const u8, source_digest: [32]u8) !void {
    var header: [cache_header_size]u8 = @splat(0);
    @memcpy(header[0..8], cache_magic);
    std.mem.writeInt(u32, header[8..12], inference.packed_model_cache_version, .little);
    std.mem.writeInt(u32, header[12..16], cache_format_version, .little);
    @memcpy(header[16..48], &source_digest);
    Blake3.hash(image, header[48..80], .{});
    std.mem.writeInt(u64, header[80..88], image.len, .little);
    var output = try directory.createFileAtomic(io, "model.voiced", .{ .replace = true, .permissions = .fromMode(0o600) });
    defer output.deinit(io);
    try output.file.writeStreamingAll(io, &header);
    try output.file.writeStreamingAll(io, image);
    try output.file.sync(io);
    try output.replace(io);
    const directory_file: std.Io.File = .{ .handle = directory.handle, .flags = .{ .nonblocking = false } };
    try directory_file.sync(io);
}

fn requirePrivateOwner(descriptor: std.posix.fd_t) !void {
    var stat: linux.Statx = undefined;

    const result = linux.statx(descriptor, "", linux.AT.EMPTY_PATH, .BASIC_STATS, &stat);
    if (linux.errno(result) != .SUCCESS) {
        return error.CacheStatFailed;
    }

    if (!stat.mask.UID or !stat.mask.MODE or stat.uid != linux.geteuid() or stat.mode & 0o077 != 0) {
        return error.UnsafeCachePermissions;
    }
}
