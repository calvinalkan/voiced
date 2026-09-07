//! One private, replaceable failed chunk. The worker borrows sealed audio and
//! decoder storage; only this error path hashes weights and writes files.
const std = @import("std");
const decimal = @import("decimal.zig");
const logging = @import("logging.zig");
const log = logging.scoped(.transcription_capture);
const builtin = @import("builtin");
const inference = @import("inference");
const linux = std.os.linux;

/// Fixed diagnostic evidence shared with the supervisor. Availability flags
/// distinguish an unknown decoder result from a real zero probability/count.
pub const Evidence = extern struct {
    chunk_available: u32 = 0,
    decoding_available: u32 = 0,
    chunk: u32 = 0,
    samples: u32 = 0,
    tokens: u32 = 0,
    token_limit: u32 = 0,
    encoder_positions: u32 = 0,
    no_speech_probability: f32 = 0,
    average_log_probability: f32 = 0,
    reserved: u32 = 0,
    log_mel_ns: u64 = 0,
    encoder_ns: u64 = 0,
    cross_key_values_ns: u64 = 0,
    decoder_ns: u64 = 0,

    pub fn finish(self: *Evidence, decoded: ?inference.Transcription, timings: inference.Timings) void {
        self.log_mel_ns = timings.log_mel_ns;
        self.encoder_ns = timings.encoder_ns;
        self.cross_key_values_ns = timings.cross_key_values_ns;
        self.decoder_ns = timings.decoder_ns;
        if (decoded) |value| {
            self.decoding_available = 1;
            self.tokens = @intCast(value.generated_tokens_count);
            self.encoder_positions = @intCast(value.encoder_positions_count);
            self.no_speech_probability = value.no_speech_probability;
            self.average_log_probability = value.average_log_probability;
        }
    }
};

pub const Metadata = struct {
    format_version: u32 = 2,
    session_id: u64,
    captured_unix_seconds: i64,
    stage: []const u8,
    error_name: []const u8,
    evidence: Evidence,
    contains_activity: bool,
    model: []const u8,
    model_revision: []const u8,
    source_sha256: []const u8,
    packed_image_sha256: [64]u8,
    packed_image_format_version: u32 = inference.packed_model_image_format_version,
    packed_model_cache_version: u32 = inference.packed_model_cache_version,
    zig_version: []const u8 = builtin.zig_version_string,
    optimize: []const u8 = @tagName(builtin.mode),
    model_encoder_threads: u32,
    model_decoder_threads: u32,
    model_encoder_padding_seconds: u32,
    sample_rate: u32 = 16000,
    sample_format: []const u8 = "float32_le",
    text_decode_complete: bool,
    end: ?[]const u8,
    prompt: [2]u16 = .{ 50257, 50362 },
    suppressed_tokens: []const inference.Token = &inference.suppressed_tokens,
    suppressed_first_tokens: []const inference.Token = &inference.suppressed_first_tokens,
    beam_size: u32 = 1,
    temperature_fallback: bool = false,
};

pub const Error = union(enum) {
    open_directory: std.Io.Dir.CreateDirPathOpenError,
    stat_directory: linux.E,
    unsafe_directory: struct { uid: u32, expected_uid: u32, mode: u16, uid_available: bool, mode_available: bool },
    permissions: std.Io.Dir.SetPermissionsError,
    lock_directory: std.Io.File.LockError,
    clear_staging: std.Io.Dir.DeleteFileError,
    write_audio: std.Io.Dir.WriteFileError,
    write_text: std.Io.Dir.WriteFileError,
    write_tokens: std.Io.Dir.WriteFileError,
    write_metadata: std.Io.Dir.WriteFileError,
    publish: linux.E,
};
pub const Result = union(enum) { ok: void, err: Error };

pub fn logError(context: logging.Context, err: Error, path: []const u8) void {
    switch (err) {
        .publish, .stat_directory => |errno| log.err(context, "Failed transcription capture: stage={t}, errno={t}, path=\"{f}\"", .{ std.meta.activeTag(err), errno, std.zig.fmtString(path) }),
        .unsafe_directory => |detail| log.err(context, "Failed transcription capture: stage=unsafe_directory, uid={d}, expected_uid={d}, mode={o}, uid_available={}, mode_available={}, path=\"{f}\"", .{ detail.uid, detail.expected_uid, detail.mode, detail.uid_available, detail.mode_available, std.zig.fmtString(path) }),
        inline else => |cause| log.err(context, "Failed transcription capture: stage={t}, detail=\"{f}\", path=\"{f}\"", .{ std.meta.activeTag(err), std.zig.fmtString(@errorName(cause)), std.zig.fmtString(path) }),
    }
}

/// directory_path is resolved once at worker initialization. Writers and replay
/// readers lock its directory inode, which survives generation replacement.
/// There is at most one complete bundle plus one bounded staging bundle after
/// interruption. Publication is atomic, but does not promise power-loss durability.
pub fn save(io: std.Io, directory_path: []const u8, samples: []const f32, text: []const u8, tokens: []const inference.Token, metadata: Metadata) Result {
    const base = switch (openPrivate(io, std.Io.Dir.cwd(), directory_path)) {
        .ok => |dir| dir,
        .err => |err| return .{ .err = err },
    };
    defer base.close(io);
    const lock: std.Io.File = .{ .handle = base.handle, .flags = .{ .nonblocking = false } };
    lock.lock(io, .exclusive) catch |err| return .{ .err = .{ .lock_directory = err } };
    // Reuse fixed names after an interrupted save; never follow their symlinks.
    const pending = switch (openPrivate(io, base, "last-failed.pending")) {
        .ok => |dir| dir,
        .err => |err| return .{ .err = err },
    };
    defer pending.close(io);
    clearFiles(io, pending) catch |err| return .{ .err = .{ .clear_staging = err } };
    writeAudio(io, pending, samples) catch |err| return .{ .err = .{ .write_audio = err } };
    pending.writeFile(io, .{ .sub_path = "generated.txt", .data = text, .flags = .{ .exclusive = true, .permissions = .fromMode(0o600) } }) catch |err| return .{ .err = .{ .write_text = err } };
    writeTokens(io, pending, tokens) catch |err| return .{ .err = .{ .write_tokens = err } };
    writeMetadata(io, pending, metadata) catch |err| return .{ .err = .{ .write_metadata = err } };
    var errno = linux.errno(linux.renameat2(base.handle, "last-failed.pending", base.handle, "last-failed", .{ .NOREPLACE = true }));
    if (errno == .EXIST) {
        errno = linux.errno(linux.renameat2(base.handle, "last-failed.pending", base.handle, "last-failed", .{ .EXCHANGE = true }));
        if (errno != .SUCCESS) return .{ .err = .{ .publish = errno } };
        // The old generation now has the staging name. A cleanup problem does
        // not invalidate the new capture; retain and log it independently.
        const old = base.openDir(io, "last-failed.pending", .{ .iterate = true, .follow_symlinks = false }) catch |err| {
            log.err(.{ .recording_ordinal = metadata.session_id }, "Failed transcription capture cleanup: detail=\"{f}\"", .{std.zig.fmtString(@errorName(err))});
            return .{ .ok = {} };
        };
        defer old.close(io);
        clearFiles(io, old) catch |err| {
            log.err(.{ .recording_ordinal = metadata.session_id }, "Failed transcription capture cleanup: detail=\"{f}\"", .{std.zig.fmtString(@errorName(err))});
            return .{ .ok = {} };
        };
        base.deleteDir(io, "last-failed.pending") catch |err| log.err(.{ .recording_ordinal = metadata.session_id }, "Failed transcription capture cleanup: detail=\"{f}\"", .{std.zig.fmtString(@errorName(err))});
    } else if (errno != .SUCCESS) return .{ .err = .{ .publish = errno } };
    return .{ .ok = {} };
}

pub fn imageDigest(model: *const inference.Model) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(model.packedImage(), &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn openPrivate(io: std.Io, parent: std.Io.Dir, path: []const u8) union(enum) { ok: std.Io.Dir, err: Error } {
    const dir = parent.createDirPathOpen(io, path, .{ .permissions = .fromMode(0o700), .open_options = .{ .iterate = true, .follow_symlinks = false } }) catch |err| return .{ .err = .{ .open_directory = err } };
    var keep = false;
    defer if (!keep) dir.close(io);
    var stat: linux.Statx = undefined;
    const errno = linux.errno(linux.statx(dir.handle, "", linux.AT.EMPTY_PATH, .BASIC_STATS, &stat));
    if (errno != .SUCCESS) return .{ .err = .{ .stat_directory = errno } };
    if (!stat.mask.UID or !stat.mask.MODE or stat.uid != linux.geteuid()) return .{ .err = .{ .unsafe_directory = .{ .uid = stat.uid, .expected_uid = linux.geteuid(), .mode = stat.mode, .uid_available = stat.mask.UID, .mode_available = stat.mask.MODE } } };
    dir.setPermissions(io, .fromMode(0o700)) catch |err| return .{ .err = .{ .permissions = err } };
    keep = true;
    return .{ .ok = dir };
}

fn clearFiles(io: std.Io, dir: std.Io.Dir) !void {
    // Also remove v1 names when replacing a JSON generation or reusing staging.
    for ([_][]const u8{ "audio.wav", "generated.txt", "tokens.txt", "metadata.txt", "tokens.json", "metadata.json" }) |name| {
        dir.deleteFile(io, name) catch |err| if (err != error.FileNotFound) return err;
    }
}

fn writeAudio(io: std.Io, dir: std.Io.Dir, samples: []const f32) !void {
    // IEEE-float WAV, including a fact chunk. No PCM16 quantization or padding.
    comptime std.debug.assert(builtin.cpu.arch.endian() == .little);
    var header: [56]u8 = @splat(0);
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], @intCast(48 + samples.len * 4), .little);
    @memcpy(header[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little);
    std.mem.writeInt(u16, header[20..22], 3, .little);
    std.mem.writeInt(u16, header[22..24], 1, .little);
    std.mem.writeInt(u32, header[24..28], 16000, .little);
    std.mem.writeInt(u32, header[28..32], 64000, .little);
    std.mem.writeInt(u16, header[32..34], 4, .little);
    std.mem.writeInt(u16, header[34..36], 32, .little);
    @memcpy(header[36..40], "fact");
    std.mem.writeInt(u32, header[40..44], 4, .little);
    std.mem.writeInt(u32, header[44..48], @intCast(samples.len), .little);
    @memcpy(header[48..52], "data");
    std.mem.writeInt(u32, header[52..56], @intCast(samples.len * 4), .little);
    const file = try dir.createFile(io, "audio.wav", .{ .exclusive = true, .permissions = .fromMode(0o600) });
    defer file.close(io);
    try file.writeStreamingAll(io, &header);
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(samples));
}

fn writeTokens(io: std.Io, dir: std.Io.Dir, tokens: []const inference.Token) !void {
    const file = try dir.createFile(io, "tokens.txt", .{ .exclusive = true, .permissions = .fromMode(0o600) });
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    writeTokenList(&writer.interface, tokens) catch return writer.err.?;
    writer.interface.flush() catch return writer.err.?;
}

// V2 is key=value, with one physical line per field and dotted evidence keys.
// Strings use Zig byte escapes without surrounding quotes; no value is trimmed.
// end_available distinguishes null from an empty end string. Keep the schema in
// sync with read_metadata in scripts/replay-transcription.py.
fn writeMetadata(io: std.Io, dir: std.Io.Dir, metadata: Metadata) !void {
    std.debug.assert(metadata.format_version == 2);
    const file = try dir.createFile(io, "metadata.txt", .{ .exclusive = true, .permissions = .fromMode(0o600) });
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    const evidence = metadata.evidence;
    writer.interface.print(
        "format_version={d}\nsession_id={d}\ncaptured_unix_seconds={d}\n" ++
            "stage={f}\nerror_name={f}\n" ++
            "evidence.chunk_available={d}\nevidence.decoding_available={d}\n" ++
            "evidence.chunk={d}\nevidence.samples={d}\nevidence.tokens={d}\nevidence.token_limit={d}\n" ++
            "evidence.encoder_positions={d}\nevidence.no_speech_probability={f}\nevidence.average_log_probability={f}\n" ++
            "evidence.reserved={d}\nevidence.log_mel_ns={d}\nevidence.encoder_ns={d}\n" ++
            "evidence.cross_key_values_ns={d}\nevidence.decoder_ns={d}\n",
        .{
            metadata.format_version,                                     metadata.session_id,                                           metadata.captured_unix_seconds,
            std.zig.fmtString(metadata.stage),                           std.zig.fmtString(metadata.error_name),                        evidence.chunk_available,
            evidence.decoding_available,                                 evidence.chunk,                                                evidence.samples,
            evidence.tokens,                                             evidence.token_limit,                                          evidence.encoder_positions,
            // Widen before formatting so an f64 reader recovers the exact f32
            // evidence value, rather than only a decimal that rounds back to it.
            // Null precision is intentional: fixed log precision loses evidence.
            decimal.fmt(@as(f64, evidence.no_speech_probability), null), decimal.fmt(@as(f64, evidence.average_log_probability), null), evidence.reserved,
            evidence.log_mel_ns,                                         evidence.encoder_ns,                                           evidence.cross_key_values_ns,
            evidence.decoder_ns,
        },
    ) catch return writer.err.?;
    writer.interface.print(
        "contains_activity={}\nmodel={f}\nmodel_revision={f}\nsource_sha256={f}\npacked_image_sha256={s}\n" ++
            "packed_image_format_version={d}\npacked_model_cache_version={d}\nzig_version={f}\noptimize={f}\n" ++
            "model_encoder_threads={d}\nmodel_decoder_threads={d}\nmodel_encoder_padding_seconds={d}\n" ++
            "sample_rate={d}\nsample_format={f}\ntext_decode_complete={}\nend_available={}\nend={f}\n" ++
            "prompt={d} {d}\nbeam_size={d}\ntemperature_fallback={}\nsuppressed_tokens=",
        .{
            metadata.contains_activity,                std.zig.fmtString(metadata.model),         std.zig.fmtString(metadata.model_revision),
            std.zig.fmtString(metadata.source_sha256), &metadata.packed_image_sha256,             metadata.packed_image_format_version,
            metadata.packed_model_cache_version,       std.zig.fmtString(metadata.zig_version),   std.zig.fmtString(metadata.optimize),
            metadata.model_encoder_threads,            metadata.model_decoder_threads,            metadata.model_encoder_padding_seconds,
            metadata.sample_rate,                      std.zig.fmtString(metadata.sample_format), metadata.text_decode_complete,
            metadata.end != null,                      std.zig.fmtString(metadata.end orelse ""), metadata.prompt[0],
            metadata.prompt[1],                        metadata.beam_size,                        metadata.temperature_fallback,
        },
    ) catch return writer.err.?;
    writeTokenList(&writer.interface, metadata.suppressed_tokens) catch return writer.err.?;
    writer.interface.writeAll("suppressed_first_tokens=") catch return writer.err.?;
    writeTokenList(&writer.interface, metadata.suppressed_first_tokens) catch return writer.err.?;
    writer.interface.flush() catch return writer.err.?;
}

fn writeTokenList(writer: *std.Io.Writer, tokens: []const inference.Token) std.Io.Writer.Error!void {
    for (tokens, 0..) |token, index| {
        if (index != 0) try writer.writeByte(' ');
        try writer.print("{d}", .{token});
    }
    try writer.writeByte('\n');
}

pub fn paddingSeconds(padding: inference.EncoderTrailingPadding) u32 {
    return switch (padding) {
        .seconds_5 => 5,
        .seconds_10 => 10,
        .seconds_30 => 30,
    };
}
