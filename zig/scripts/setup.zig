//! Installs verified Whisper checkpoints under the user's XDG data directory.
//! Each checkpoint is assembled under a sibling temporary directory and is
//! published only after both files pass their pinned size and BLAKE3-256.

const std = @import("std");
const models = @import("models");
const assert = std.debug.assert;
const Io = std.Io;
const Blake3 = std.crypto.hash.Blake3;

pub fn main(init: std.process.Init) !void {
    var arguments = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer arguments.deinit();

    assert(arguments.skip());
    if (arguments.next() != null) {
        return error.InvalidArguments;
    }

    var http_client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer http_client.deinit();

    const model_files_dir = try models.allocInstalledRootPath(init);
    defer init.gpa.free(model_files_dir);

    inline for (comptime std.meta.tags(models.Model)) |model| {
        const metadata = comptime model.metadata();
        const url = "https://huggingface.co/" ++ metadata.name ++ "/resolve/" ++ metadata.revision ++ "/";

        try setupModel(
            init,
            model_files_dir,
            &http_client,
            model,
            &.{
                .{
                    .display_name = metadata.name ++ " weights",
                    .url = url ++ "model.bin",
                    .file_name = "model.bin",
                    .expected_size = metadata.weights.size,
                    .expected_blake3 = metadata.weights.blake3,
                },
                .{
                    .display_name = metadata.name ++ " vocabulary",
                    .url = url ++ "vocabulary.txt",
                    .file_name = "vocabulary.txt",
                    .expected_size = models.vocabulary.size,
                    .expected_blake3 = models.vocabulary.blake3,
                },
            },
        );
    }
}

const Artifact = struct {
    display_name: []const u8,
    url: []const u8,
    file_name: []const u8,
    expected_size: u64,
    expected_blake3: []const u8,
};

fn setupModel(
    init: std.process.Init,
    model_files_dir: []const u8,
    http_client: *std.http.Client,
    model: models.Model,
    artifacts: []const Artifact,
) !void {
    const io = init.io;
    const allocator = init.gpa;

    const model_dir_path = try models.allocInstalledDirectoryPath(init.gpa, model_files_dir, model);
    defer allocator.free(model_dir_path);

    const cwd = Io.Dir.cwd();

    const model_dir_handle = cwd.openDir(io, model_dir_path, .{}) catch |err| switch (err) {
        error.FileNotFound => null,
        else => {
            return err;
        },
    };

    // ── Check For Existing Files First ──
    //
    // If all required files exist, have the correct size AND hash,
    // then we can skip the downloads.

    if (model_dir_handle) |dir_handle| {
        defer dir_handle.close(io);

        const all_model_files_valid = for (artifacts) |artifact| {
            const file = dir_handle.openFile(io, artifact.file_name, .{}) catch |err| switch (err) {
                error.FileNotFound => {
                    break false;
                },
                else => {
                    return err;
                },
            };
            defer file.close(io);

            const stat = try file.stat(io);
            if (stat.kind != .file or stat.size != artifact.expected_size) {
                break false;
            }

            var file_reader = file.readerStreaming(io, &.{});
            var hash = Blake3.init(.{});
            var buffer: [64 * 1024]u8 = undefined;
            var bytes_read: u64 = 0;

            while (true) {
                const read_size = try file_reader.interface.readSliceShort(&buffer);
                if (read_size == 0) {
                    break;
                }

                hash.update(buffer[0..read_size]);
                bytes_read += read_size;
            }

            if (bytes_read != artifact.expected_size) {
                break false;
            }

            var digest_bytes: [Blake3.digest_length]u8 = undefined;
            hash.final(&digest_bytes);

            const digest = std.fmt.bytesToHex(digest_bytes, .lower);
            if (!std.mem.eql(u8, &digest, artifact.expected_blake3)) {
                break false;
            }
        } else true;

        if (all_model_files_valid) {
            std.debug.print("{s} already installed in {s}\n", .{ model.name(), model_dir_path });

            return;
        }
    }

    // ── Download Model Files ──

    // Build the path for a staging directory beside the final installation.
    const model_staging_dir_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{model_dir_path});
    defer allocator.free(model_staging_dir_path);

    // Remove leftovers from an interrupted installation.
    try cwd.deleteTree(io, model_staging_dir_path);

    // Create the staging directory and any missing parent directories.
    try cwd.createDirPath(io, model_staging_dir_path);

    errdefer cwd.deleteTree(io, model_staging_dir_path) catch {};

    {
        var model_staging_dir_handle = try cwd.openDir(io, model_staging_dir_path, .{});
        defer model_staging_dir_handle.close(io);

        for (artifacts) |artifact| {
            try downloadAndVerifyArtifact(io, http_client, artifact, model_staging_dir_handle);
        }
    }

    // Remove the previous installation: rename cannot replace a nonempty directory.
    try cwd.deleteTree(io, model_dir_path);

    try cwd.rename(model_staging_dir_path, cwd, model_dir_path, io);

    std.debug.print("Installed {s} in {s}\n", .{ model.name(), model_dir_path });
}

// The HTTP body streams through BLAKE3 into an unpublished file; only the
// exact pinned size and digest allow its final name to become visible.
fn downloadAndVerifyArtifact(
    io: Io,
    http_client: *std.http.Client,
    artifact: Artifact,
    dir_handle: Io.Dir,
) !void {
    assert(artifact.display_name.len > 0);
    assert(artifact.url.len > 0);
    assert(artifact.file_name.len > 0);
    assert(artifact.expected_size > 0);
    assert(artifact.expected_blake3.len == Blake3.digest_length * 2);

    std.debug.print("Downloading {s}", .{artifact.display_name});

    var download_line_is_open = true;
    defer {
        // An early error must end the unterminated progress line.
        if (download_line_is_open) std.debug.print("\n", .{});
    }

    // ── Request Artifact ──
    //
    // Hugging Face redirects to storage backends. Disable content encoding so
    // the pins describe the actual bytes written to disk.

    var request = try http_client.request(
        .GET,
        try std.Uri.parse(artifact.url),
        .{
            .redirect_behavior = .init(3),
            .headers = .{
                .user_agent = .{ .override = "voiced-setup" },
                .accept_encoding = .omit,
            },
        },
    );
    defer request.deinit();

    try request.sendBodiless();

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var response = try request.receiveHead(&redirect_buffer);
    if (response.head.status != .ok) {
        std.debug.print(
            "\r\x1b[2Kerror: HTTP request failed\n  output: {s}\n  url: {s}\n  status: {d}\n",
            .{ artifact.file_name, artifact.url, @intFromEnum(response.head.status) },
        );
        download_line_is_open = false;

        return error.UnexpectedHttpStatus;
    }

    // ── Stream Into Unpublished File ──
    //
    // Every exit before `link` removes the atomic file without exposing partial
    // data. Streaming keeps memory independent of checkpoint size.

    var artifact_file = try dir_handle.createFileAtomic(io, artifact.file_name, .{});
    defer artifact_file.deinit(io);

    const expected_mib_tenths = artifact.expected_size * 10 / (1024 * 1024);

    var response_transfer_buffer: [64]u8 = undefined;
    const response_body = response.reader(&response_transfer_buffer);

    var artifact_writer = artifact_file.file.writerStreaming(io, &.{});
    var blake3_writer = artifact_writer.interface.hashed(Blake3.init(.{}), &.{});

    var download_buffer: [64 * 1024]u8 = undefined;
    var downloaded_size: u64 = 0;
    var progress_percent_reported: u8 = 0;

    while (true) {
        const read_size = response_body.readSliceShort(&download_buffer) catch {
            return response.bodyErr().?;
        };
        if (read_size == 0) {
            break;
        }

        if (downloaded_size + read_size > artifact.expected_size) {
            std.debug.print(
                "\r\x1b[2Kerror: artifact size mismatch\n  output: {s}\n  expected: {d} bytes\n  actual: more than {d} bytes\n",
                .{ artifact.file_name, artifact.expected_size, artifact.expected_size },
            );
            download_line_is_open = false;

            return error.DownloadSizeMismatch;
        }

        try blake3_writer.writer.writeAll(download_buffer[0..read_size]);
        downloaded_size += read_size;

        const progress_percent: u8 = @intCast(@min(downloaded_size * 100 / artifact.expected_size, 100));
        if (progress_percent > progress_percent_reported) {
            const downloaded_mib_tenths = downloaded_size * 10 / (1024 * 1024);
            std.debug.print(
                "\r\x1b[2KDownloading {s} - {d}.{d}/{d}.{d} MiB ({d}%)",
                .{
                    artifact.display_name,
                    downloaded_mib_tenths / 10,
                    downloaded_mib_tenths % 10,
                    expected_mib_tenths / 10,
                    expected_mib_tenths % 10,
                    progress_percent,
                },
            );
            progress_percent_reported = progress_percent;
        }
    }

    assert(downloaded_size == artifact_writer.pos);

    // ── Verify And Publish Complete Artifact ──

    if (artifact_writer.pos != artifact.expected_size) {
        std.debug.print(
            "\r\x1b[2Kerror: artifact size mismatch\n  output: {s}\n  expected: {d} bytes\n  actual: {d} bytes\n",
            .{ artifact.file_name, artifact.expected_size, artifact_writer.pos },
        );
        download_line_is_open = false;

        return error.DownloadSizeMismatch;
    }

    var digest_bytes: [Blake3.digest_length]u8 = undefined;
    blake3_writer.hasher.final(&digest_bytes);

    const digest = std.fmt.bytesToHex(digest_bytes, .lower);
    if (!std.mem.eql(u8, &digest, artifact.expected_blake3)) {
        std.debug.print(
            "\r\x1b[2Kerror: artifact BLAKE3 mismatch\n  output: {s}\n  expected: {s}\n  actual: {s}\n",
            .{ artifact.file_name, artifact.expected_blake3, &digest },
        );
        download_line_is_open = false;

        return error.DownloadHashMismatch;
    }

    // Persist the bytes before the file name becomes visible.
    try artifact_file.file.sync(io);

    try artifact_file.link(io);

    std.debug.print("\r\x1b[2KDownloaded {s} as {s}\n", .{ artifact.display_name, artifact.file_name });

    download_line_is_open = false;
}
