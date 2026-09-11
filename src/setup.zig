//! Installs every supported Whisper model as one self-contained packed model.
//! Upstream selection, XDG paths, staging, downloads, and publication belong to
//! this command; source files are removed after conversion.

const std = @import("std");
const artifact_downloader = @import("artifact_downloader.zig");
const inference = @import("inference/root.zig");
const packed_model = @import("packed_model/root.zig");
const Io = std.Io;
const Blake3 = std.crypto.hash.Blake3;

/// `ModelSource` identifies the two pinned CTranslate2 files from which setup
/// constructs one packed model. Its strings have static lifetime.
pub const ModelSource = struct {
    weights: artifact_downloader.Artifact,
    vocabulary: artifact_downloader.Artifact,

    fn forKind(kind: inference.Model.Kind) ModelSource {
        return switch (kind) {
            .whisper_base_en => .{
                .weights = .{
                    .url = "https://huggingface.co/Systran/faster-whisper-base.en/resolve/3d3d5dee26484f91867d81cb899cfcf72b96be6c/model.bin",
                    .file_name = "model.bin",
                    .expected_size = 145_216_508,
                    .expected_blake3 = "46fa7ff77f6613205ae186b6b763e74b5e0f0ee19ce008f96e85094afaa17f4d",
                },
                .vocabulary = .{
                    .url = "https://huggingface.co/Systran/faster-whisper-base.en/resolve/3d3d5dee26484f91867d81cb899cfcf72b96be6c/vocabulary.txt",
                    .file_name = "vocabulary.txt",
                    .expected_size = 422_309,
                    .expected_blake3 = "5ba2618f5d7940b9cebc94299dcc42f056848f660e602ace121ac29af488cb15",
                },
            },
            .whisper_small_en => .{
                .weights = .{
                    .url = "https://huggingface.co/Systran/faster-whisper-small.en/resolve/d1d751a5f8271d482d14ca55d9e2deeebbae577f/model.bin",
                    .file_name = "model.bin",
                    .expected_size = 483_545_366,
                    .expected_blake3 = "6f8da5f2d48b1133b5bc150c5f59a3d3861209637ee15ecf55a09e704fc0254d",
                },
                .vocabulary = .{
                    .url = "https://huggingface.co/Systran/faster-whisper-small.en/resolve/d1d751a5f8271d482d14ca55d9e2deeebbae577f/vocabulary.txt",
                    .file_name = "vocabulary.txt",
                    .expected_size = 422_309,
                    .expected_blake3 = "5ba2618f5d7940b9cebc94299dcc42f056848f660e602ace121ac29af488cb15",
                },
            },
            .whisper_medium_en => .{
                .weights = .{
                    .url = "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/a29b04bd15381511a9af671baec01072039215e3/model.bin",
                    .file_name = "model.bin",
                    .expected_size = 1_527_904_330,
                    .expected_blake3 = "c5feee2b4e796143a5706c89a6c4c4078f23fbf7697611a9bc8f12cf4da15df9",
                },
                .vocabulary = .{
                    .url = "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/a29b04bd15381511a9af671baec01072039215e3/vocabulary.txt",
                    .file_name = "vocabulary.txt",
                    .expected_size = 422_309,
                    .expected_blake3 = "5ba2618f5d7940b9cebc94299dcc42f056848f660e602ace121ac29af488cb15",
                },
            },
        };
    }
};

pub fn run(init: std.process.Init) !void {
    var http_client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer http_client.deinit();

    const env = init.environ_map;

    // ── Determine The Output Directory ──

    const xdg_data_home_path = env.get("XDG_DATA_HOME");
    const data_home_path = xdg_data_home_path orelse env.get("HOME") orelse {
        return error.HomeNotSet;
    };

    if (!std.fs.path.isAbsolute(data_home_path)) {
        return error.DataHomeNotAbsolute;
    }

    const output_dir_parts: []const []const u8 = if (xdg_data_home_path != null)
        &.{ data_home_path, "voiced/models" }
    else
        &.{ data_home_path, ".local/share/voiced/models" };

    const output_dir_path = try std.fs.path.join(init.gpa, output_dir_parts);
    defer init.gpa.free(output_dir_path);

    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(init.io, output_dir_path);

    var output_dir_handle = try cwd.openDir(init.io, output_dir_path, .{ .iterate = true });
    defer output_dir_handle.close(init.io);

    inline for (comptime std.meta.tags(inference.Model.Kind)) |kind| {
        try setupModel(init, output_dir_path, output_dir_handle, &http_client, kind);
    }
}

fn setupModel(
    init: std.process.Init,
    installed_root: []const u8,
    installed_directory: Io.Dir,
    http_client: *std.http.Client,
    kind: inference.Model.Kind,
) !void {
    const io = init.io;
    const allocator = init.gpa;

    const output_file_name = try std.fmt.allocPrint(allocator, "{s}.voiced", .{kind.name()});
    defer allocator.free(output_file_name);

    const install_path = try std.fs.path.join(allocator, &.{ installed_root, output_file_name });
    defer allocator.free(install_path);

    // ── Reuse A Valid Installed Model ──

    if (packed_model.load(io, installed_directory, output_file_name, kind)) |loaded_model| {
        var model = loaded_model;
        defer model.deinit();
        std.debug.print("{s} already installed in {s}\n", .{ kind.name(), install_path });

        return;
    } else |err| switch (err) {
        error.FileNotFound,
        error.InvalidPackedModel,
        error.PackedModelChecksumMismatch,
        error.UnsupportedPackedModelFormatVersion,
        error.UnsupportedModelKind,
        error.UnexpectedModelKind,
        => {},
        else => return err,
    }

    // ── Prepare Staging Directory ──

    const staging_path = try std.fmt.allocPrint(allocator, "{s}/.{s}.setup", .{ installed_root, kind.name() });
    defer allocator.free(staging_path);

    const cwd = Io.Dir.cwd();

    try cwd.deleteTree(io, staging_path);
    try cwd.createDirPath(io, staging_path);
    errdefer cwd.deleteTree(io, staging_path) catch |err| {
        std.debug.print("Could not cleanup the staging directory: {any}\n", .{err});
    };

    var staging_dir_handle = try cwd.openDir(io, staging_path, .{});
    defer staging_dir_handle.close(io);

    // ── Download Source Model ──

    std.debug.print("Downloading {s}\n", .{kind.name()});

    const ctranslate2_source = try ensureModuleCTranslate2Source(io, http_client, staging_dir_handle, kind);

    // ── Load The CTranslate2 Inputs ──
    //
    // Downloads stream to disk; conversion needs random access to the complete
    // source files. Setup is the only command that holds these allocations.

    const ctranslate2_weights_bytes = try staging_dir_handle.readFileAlloc(
        io,
        ctranslate2_source.weights.file_name,
        allocator,
        .limited(ctranslate2_source.weights.expected_size + 1),
    );
    defer allocator.free(ctranslate2_weights_bytes);

    const ctranslate2_vocabulary_text = try staging_dir_handle.readFileAlloc(
        io,
        ctranslate2_source.vocabulary.file_name,
        allocator,
        .limited(ctranslate2_source.vocabulary.expected_size + 1),
    );
    defer allocator.free(ctranslate2_vocabulary_text);

    std.debug.print("Packing {s}\n", .{kind.name()});

    // ── Publish ──

    var output_file_atomic_writer = try installed_directory.createFileAtomic(io, output_file_name, .{ .replace = true, .permissions = .fromMode(0o600) });
    defer output_file_atomic_writer.deinit(io);

    try packed_model.writeFromCTranslate2(io, allocator, output_file_atomic_writer.file, kind, ctranslate2_weights_bytes, ctranslate2_vocabulary_text);
    try output_file_atomic_writer.file.sync(io);
    try output_file_atomic_writer.replace(io);

    const directory_handle: Io.File = .{ .handle = installed_directory.handle, .flags = .{ .nonblocking = false } };
    try directory_handle.sync(io);

    // Reopen the published file through the production reader before deleting
    // its sources. A successful load verifies the writer-reader round trip and
    // the final on-disk bytes.
    var installed_model = try packed_model.load(io, installed_directory, output_file_name, kind);
    defer installed_model.deinit();

    try cwd.deleteTree(io, staging_path);

    std.debug.print("Installed {s} in {s}\n", .{ kind.name(), install_path });
}

/// Ensures `directory` contains the pinned CTranslate2 weights and vocabulary
/// for `kind`, downloading any missing or invalid artifact. Returns metadata
/// naming the verified files.
///
/// Use a distinct directory for each model kind because their source artifacts
/// share the names `model.bin` and `vocabulary.txt`. The caller retains
/// ownership of `directory` and `http_client`.
pub fn ensureModuleCTranslate2Source(
    io: Io,
    http_client: *std.http.Client,
    directory: Io.Dir,
    kind: inference.Model.Kind,
) !ModelSource {
    const source = ModelSource.forKind(kind);

    try ensureArtifact(io, http_client, directory, source.weights);
    try ensureArtifact(io, http_client, directory, source.vocabulary);

    return source;
}

fn ensureArtifact(
    io: Io,
    http_client: *std.http.Client,
    directory: Io.Dir,
    artifact: artifact_downloader.Artifact,
) !void {
    if (try cachedArtifactMatches(io, directory, artifact)) {
        return;
    }

    directory.deleteFile(io, artifact.file_name) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };

    try downloadArtifact(io, http_client, directory, artifact);
}

fn cachedArtifactMatches(
    io: Io,
    directory: Io.Dir,
    artifact: artifact_downloader.Artifact,
) !bool {
    const file = directory.openFile(
        io,
        artifact.file_name,
        .{ .mode = .read_only, .allow_directory = false },
    ) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) {
        return false;
    }

    if (stat.size != artifact.expected_size) {
        return false;
    }

    var hash = Blake3.init(.{});
    var file_reader = file.reader(io, &.{});
    var buffer: [64 * 1024]u8 = undefined;

    while (true) {
        const size = try file_reader.interface.readSliceShort(&buffer);
        if (size == 0) {
            break;
        }

        hash.update(buffer[0..size]);
    }

    var digest_bytes: [Blake3.digest_length]u8 = undefined;
    hash.final(&digest_bytes);

    const digest = std.fmt.bytesToHex(digest_bytes, .lower);

    return std.mem.eql(u8, &digest, artifact.expected_blake3);
}

fn downloadArtifact(io: Io, client: *std.http.Client, directory: Io.Dir, artifact: artifact_downloader.Artifact) !void {
    switch (try artifact_downloader.downloadAndVerify(io, client, artifact, directory)) {
        .ok => {},
        .err => |download_error| switch (download_error) {
            .http_status => |status| {
                std.debug.print("error: HTTP request failed\n  output: {s}\n  url: {s}\n  status: {d}\n", .{ artifact.file_name, artifact.url, status });

                return error.UnexpectedHttpStatus;
            },
            .size_exceeded => |actual_size| {
                std.debug.print("error: artifact size mismatch\n  output: {s}\n  expected: {d} bytes\n  actual: at least {d} bytes\n", .{ artifact.file_name, artifact.expected_size, actual_size });

                return error.DownloadSizeMismatch;
            },
            .size_incomplete => |actual_size| {
                std.debug.print("error: artifact size mismatch\n  output: {s}\n  expected: {d} bytes\n  actual: {d} bytes\n", .{ artifact.file_name, artifact.expected_size, actual_size });

                return error.DownloadSizeMismatch;
            },
            .blake3_mismatch => |actual_digest_bytes| {
                const actual_digest = std.fmt.bytesToHex(actual_digest_bytes, .lower);

                std.debug.print("error: artifact BLAKE3 mismatch\n  output: {s}\n  expected: {s}\n  actual: {s}\n", .{ artifact.file_name, artifact.expected_blake3, &actual_digest });

                return error.DownloadHashMismatch;
            },
        },
    }
}
