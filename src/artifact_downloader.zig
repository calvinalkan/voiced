//! Downloads one size- and BLAKE3-pinned artifact into an atomic file. The
//! caller owns artifact selection, directory staging, and user-facing output.

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const Blake3 = std.crypto.hash.Blake3;

pub const Artifact = struct {
    url: []const u8,
    file_name: []const u8,
    expected_size: u64,
    expected_blake3: []const u8,
};

pub const Result = union(enum) {
    ok,
    err: Error,
};

pub const Error = union(enum) {
    http_status: u16,
    size_incomplete: u64,
    size_exceeded: u64,
    blake3_mismatch: [Blake3.digest_length]u8,
};

// The HTTP body streams through BLAKE3 into an unpublished file; only the
// exact pinned size and digest allow its final name to become visible.
pub fn downloadAndVerify(
    io: Io,
    http_client: *std.http.Client,
    artifact: Artifact,
    dir_handle: Io.Dir,
) !Result {
    assert(artifact.url.len > 0);
    assert(artifact.file_name.len > 0);
    assert(artifact.expected_size > 0);
    assert(artifact.expected_blake3.len == Blake3.digest_length * 2);

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
        return .{ .err = .{ .http_status = @intFromEnum(response.head.status) } };
    }

    // ── Stream Into Unpublished File ──
    //
    // Every exit before `link` removes the atomic file without exposing partial
    // data. Streaming keeps memory independent of checkpoint size.

    var artifact_file = try dir_handle.createFileAtomic(io, artifact.file_name, .{});
    defer artifact_file.deinit(io);

    var response_transfer_buffer: [64]u8 = undefined;
    const response_body = response.reader(&response_transfer_buffer);

    var artifact_writer = artifact_file.file.writerStreaming(io, &.{});
    var blake3_writer = artifact_writer.interface.hashed(Blake3.init(.{}), &.{});

    var download_buffer: [64 * 1024]u8 = undefined;
    var downloaded_size: u64 = 0;

    while (true) {
        const read_size = response_body.readSliceShort(&download_buffer) catch {
            return response.bodyErr().?;
        };

        if (read_size == 0) {
            break;
        }

        if (downloaded_size + read_size > artifact.expected_size) {
            return .{ .err = .{ .size_exceeded = downloaded_size + read_size } };
        }

        try blake3_writer.writer.writeAll(download_buffer[0..read_size]);

        downloaded_size += read_size;
    }

    assert(downloaded_size == artifact_writer.pos);

    // ── Verify And Publish Complete Artifact ──

    if (artifact_writer.pos != artifact.expected_size) {
        return .{ .err = .{ .size_incomplete = artifact_writer.pos } };
    }

    var digest_bytes: [Blake3.digest_length]u8 = undefined;
    blake3_writer.hasher.final(&digest_bytes);

    const digest = std.fmt.bytesToHex(digest_bytes, .lower);
    if (!std.mem.eql(u8, &digest, artifact.expected_blake3)) {
        return .{ .err = .{ .blake3_mismatch = digest_bytes } };
    }

    // Persist the bytes before the file name becomes visible.
    try artifact_file.file.sync(io);

    try artifact_file.link(io);

    return .ok;
}
