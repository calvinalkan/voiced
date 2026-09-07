//! Offline single-chunk decoder. The Python driver supplies captured settings
//! and exact float32 PCM, and compares the resulting private JSON artifacts.
const std = @import("std");
const logging = @import("logging.zig");
const inference = @import("inference");
const models = @import("models");
const model_cache = @import("model_cache.zig");
const debug_capture = @import("transcription_debug.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 8 and args.len != 9) {
        std.debug.print("Usage: voiced-replay MODEL ENCODER_THREADS DECODER_THREADS PADDING TOKEN_LIMIT AUDIO.f32 OUTPUT.json [CAPTURE_DIRECTORY]\n", .{});
        return error.InvalidArguments;
    }
    const selected = models.Model.parse(args[1]) orelse return error.InvalidModel;
    const encoder_threads = try std.fmt.parseInt(u32, args[2], 10);
    const decoder_threads = try std.fmt.parseInt(u32, args[3], 10);
    const padding = switch (try std.fmt.parseInt(u8, args[4], 10)) {
        5 => inference.EncoderTrailingPadding.seconds_5,
        10 => .seconds_10,
        30 => .seconds_30,
        else => return error.InvalidPadding,
    };
    const token_limit = try std.fmt.parseInt(u32, args[5], 10);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[6], init.gpa, .limited(480000 * 4));
    defer init.gpa.free(bytes);
    if (bytes.len % 4 != 0) return error.InvalidAudioLength;
    const samples = try init.gpa.alloc(f32, bytes.len / 4);
    defer init.gpa.free(samples);
    @memcpy(std.mem.sliceAsBytes(samples), bytes);
    logging.initCli(.info);
    defer logging.deinit();
    var loaded = try model_cache.loadModel(init, selected);
    defer loaded.deinit();
    const vocabulary = try model_cache.loadVocabulary(init, selected);
    defer init.gpa.free(vocabulary);
    const policy: inference.Policy = .{ .samples_count_max = 480000, .workers_count = encoder_threads, .decoder_workers_count = decoder_threads, .generated_tokens_count_max = token_limit };
    const size = try inference.Runtime.requiredMemorySize(loaded.model.kind, policy);
    const memory = try init.gpa.alignedAlloc(u8, .fromByteUnits(inference.runtime_memory_alignment), size);
    defer init.gpa.free(memory);
    var runtime: inference.Runtime = undefined;
    try runtime.init(init.io, &loaded.model, vocabulary, memory, policy);
    defer runtime.deinit();
    // Match the service's text capacity, including its text conversion errors.
    var text: [4096]u8 = undefined;
    var timings: inference.Timings = .{};
    var decoded: ?inference.Transcription = null;
    var error_name: ?[]const u8 = null;
    const result = runtime.transcribe(samples, &text, .{ .encoder_trailing_padding = padding, .timings = &timings, .evidence = &decoded }) catch |err| failed: {
        error_name = @errorName(err);
        break :failed null;
    };
    if (result) |value| {
        if (value.end == .token_limit) error_name = "GeneratedTokenLimitExceeded";
    }
    var evidence: debug_capture.Evidence = .{ .chunk_available = 1, .samples = @intCast(samples.len), .token_limit = token_limit };
    evidence.finish(decoded, timings);
    const tokens = runtime.generated_tokens[0..evidence.tokens];
    const generated = if (decoded) |value| value.text else "";
    const metadata: debug_capture.Metadata = .{
        .session_id = 0,
        .captured_unix_seconds = 0,
        .stage = "offline_replay",
        .error_name = error_name orelse "",
        .evidence = evidence,
        .contains_activity = false,
        .model = selected.name(),
        .model_revision = selected.metadata().revision,
        .source_sha256 = selected.metadata().weights.sha256,
        .packed_image_sha256 = debug_capture.imageDigest(&loaded.model),
        .model_encoder_threads = encoder_threads,
        .model_decoder_threads = decoder_threads,
        .model_encoder_padding_seconds = debug_capture.paddingSeconds(padding),
        .text_decode_complete = result != null,
        .end = if (decoded) |value| @tagName(value.end) else null,
    };
    const valid_utf8 = std.unicode.utf8ValidateSlice(generated);
    var hex_buffer: [8192]u8 = undefined;
    const invalid_text_hex: ?[]const u8 = if (valid_utf8) null else try std.fmt.bufPrint(&hex_buffer, "{x}", .{generated});
    // JSON belongs to this offline executable, not the daemon's capture writer.
    {
        const file = try std.Io.Dir.cwd().createFile(init.io, args[7], .{ .exclusive = true, .permissions = .fromMode(0o600) });
        defer file.close(init.io);
        var buffer: [4096]u8 = undefined;
        var writer = file.writerStreaming(init.io, &buffer);
        std.json.Stringify.value(.{ .metadata = metadata, .text = if (valid_utf8) generated else "", .invalid_text_hex = invalid_text_hex, .tokens = tokens }, .{ .whitespace = .indent_2 }, &writer.interface) catch return writer.err.?;
        writer.interface.writeByte('\n') catch return writer.err.?;
        writer.interface.flush() catch return writer.err.?;
    }
    if (args.len == 9 and error_name != null) switch (debug_capture.save(init.io, args[8], samples, generated, tokens, metadata)) {
        .ok => {},
        .err => |err| {
            debug_capture.logError(.{}, err, args[8]);
            return error.CaptureFailed;
        },
    };
}
