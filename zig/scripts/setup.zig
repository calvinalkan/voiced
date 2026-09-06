//! Installs verified Whisper checkpoints under the user's XDG data directory.
//! Each checkpoint is assembled under a sibling temporary directory and is
//! published only after both files pass their pinned size and SHA-256.

const std = @import("std");
const models = @import("models");
const assert = std.debug.assert;
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub fn main(init: std.process.Init) !void {
    var arguments = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer arguments.deinit();
    assert(arguments.skip());
    if (arguments.next() != null) {
        return error.InvalidArguments;
    }

    var http_client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer http_client.deinit();

    // The runtime imports only the weights and GPT-2 vocabulary. Model shapes
    // and decoding policy are defined in Zig, not upstream JSON sidecars.
    // A failed second installation leaves the first published and usable.
    inline for (comptime std.meta.tags(models.Model)) |model| {
        const metadata = comptime model.metadata();
        const url = "https://huggingface.co/" ++ metadata.name ++ "/resolve/" ++ metadata.revision ++ "/";
        try installCheckpoint(init, &http_client, .{
            .model = model,
            .files = &.{
                .{
                    .display_name = metadata.name ++ " weights",
                    .url = url ++ "model.bin",
                    .file_name = "model.bin",
                    .expected_size = metadata.weights.size,
                    .expected_sha256 = metadata.weights.sha256,
                },
                .{
                    .display_name = metadata.name ++ " vocabulary",
                    .url = url ++ "vocabulary.txt",
                    .file_name = "vocabulary.txt",
                    .expected_size = models.vocabulary.size,
                    .expected_sha256 = models.vocabulary.sha256,
                },
            },
        });
    }
}

const Checkpoint = struct {
    model: models.Model,
    files: []const Artifact,
};

const Artifact = struct {
    display_name: []const u8,
    url: []const u8,
    file_name: []const u8,
    expected_size: u64,
    expected_sha256: []const u8,
};

fn installCheckpoint(init: std.process.Init, http_client: *std.http.Client, checkpoint: Checkpoint) !void {
    const io = init.io;
    const allocator = init.gpa;
    const directory_path = try models.allocInstalledDirectoryPath(init, checkpoint.model);
    defer allocator.free(directory_path);

    // ── Verify An Existing Installation ──
    //
    // The two payloads define completeness. Verify their bytes directly so an
    // installation needs neither a marker nor unrelated distribution files.
    const cwd = Io.Dir.cwd();
    const existing_directory = cwd.openDir(io, directory_path, .{}) catch |err| switch (err) {
        error.FileNotFound => null,
        else => {
            return err;
        },
    };
    if (existing_directory) |directory| {
        defer directory.close(io);
        const complete = for (checkpoint.files) |artifact| {
            const file = directory.openFile(io, artifact.file_name, .{}) catch |err| switch (err) {
                error.FileNotFound => break false,
                else => {
                    return err;
                },
            };
            defer file.close(io);
            const stat = try file.stat(io);
            if (stat.kind != .file or stat.size != artifact.expected_size) break false;

            var reader = file.readerStreaming(io, &.{});
            var buffer: [64 * 1024]u8 = undefined;
            var hash = Sha256.init(.{});
            var size: u64 = 0;
            while (true) {
                const read_size = try reader.interface.readSliceShort(&buffer);
                if (read_size == 0) break;
                hash.update(buffer[0..read_size]);
                size += read_size;
            }
            const digest = std.fmt.bytesToHex(hash.finalResult(), .lower);
            if (size != artifact.expected_size or
                !std.mem.eql(u8, &digest, artifact.expected_sha256)) break false;
        } else true;
        if (complete) {
            std.debug.print("{s} already installed in {s}\n", .{ checkpoint.model.name(), directory_path });
            return;
        }
    }

    // ── Assemble And Publish A Complete Checkpoint ──
    //
    // Downloads never expose a partially written installation. Both files are
    // verified before publication; failure removes the unpublished directory.
    const parent_path = std.fs.path.dirname(directory_path).?;
    const temporary_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{directory_path});
    defer allocator.free(temporary_path);
    try cwd.createDirPath(io, parent_path);
    try cwd.deleteTree(io, temporary_path);
    try cwd.createDirPath(io, temporary_path);
    errdefer cwd.deleteTree(io, temporary_path) catch {};
    {
        var directory = try cwd.openDir(io, temporary_path, .{});
        defer directory.close(io);
        for (checkpoint.files) |artifact| try downloadAndVerifyArtifact(io, http_client, artifact, directory);
    }
    try cwd.deleteTree(io, directory_path);
    try cwd.rename(temporary_path, cwd, directory_path, io);
    std.debug.print("Installed {s} in {s}\n", .{ checkpoint.model.name(), directory_path });
}

// The HTTP body streams through SHA-256 into an unpublished file; only the
// exact pinned size and digest allow its final name to become visible.
fn downloadAndVerifyArtifact(
    io: Io,
    http_client: *std.http.Client,
    artifact: Artifact,
    destination_dir: Io.Dir,
) !void {
    assert(artifact.display_name.len > 0);
    assert(artifact.url.len > 0);
    assert(artifact.file_name.len > 0);
    assert(artifact.expected_size > 0);
    assert(artifact.expected_sha256.len == Sha256.digest_length * 2);

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
    var artifact_file = try destination_dir.createFileAtomic(io, artifact.file_name, .{});
    defer artifact_file.deinit(io);
    var response_transfer_buffer: [64]u8 = undefined;
    var download_buffer: [64 * 1024]u8 = undefined;
    const response_body = response.reader(&response_transfer_buffer);
    var artifact_writer = artifact_file.file.writerStreaming(io, &.{});
    var sha256_writer = artifact_writer.interface.hashed(Sha256.init(.{}), &.{});
    const expected_mib_tenths = artifact.expected_size * 10 / (1024 * 1024);
    var downloaded_size: u64 = 0;
    var progress_percent_reported: u8 = 0;
    while (true) {
        const read_size = response_body.readSliceShort(&download_buffer) catch {
            return response.bodyErr().?;
        };
        if (read_size == 0) break;
        if (downloaded_size + read_size > artifact.expected_size) {
            std.debug.print(
                "\r\x1b[2Kerror: artifact size mismatch\n  output: {s}\n  expected: {d} bytes\n  actual: more than {d} bytes\n",
                .{ artifact.file_name, artifact.expected_size, artifact.expected_size },
            );
            download_line_is_open = false;
            return error.DownloadSizeMismatch;
        }
        try sha256_writer.writer.writeAll(download_buffer[0..read_size]);
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
    var actual_sha256: [Sha256.digest_length]u8 = undefined;
    sha256_writer.hasher.final(&actual_sha256);
    const actual_sha256_hex = std.fmt.bytesToHex(actual_sha256, .lower);
    if (!std.mem.eql(u8, &actual_sha256_hex, artifact.expected_sha256)) {
        std.debug.print(
            "\r\x1b[2Kerror: artifact SHA-256 mismatch\n  output: {s}\n  expected: {s}\n  actual: {s}\n",
            .{ artifact.file_name, artifact.expected_sha256, &actual_sha256_hex },
        );
        download_line_is_open = false;
        return error.DownloadHashMismatch;
    }
    try artifact_file.file.sync(io);
    try artifact_file.link(io);
    std.debug.print("\r\x1b[2KDownloaded {s} as {s}\n", .{ artifact.display_name, artifact.file_name });
    download_line_is_open = false;
}
