//! Loads one 16 kHz mono PCM WAV fixture, computes Whisper log-Mel features in
//! Zig, and transcribes them through the narrow CTranslate2 C ABI.

const std = @import("std");
const log_mel = @import("log_mel.zig");
const assert = std.debug.assert;
const stderr = std.debug.print;

const Io = std.Io;
const Allocator = std.mem.Allocator;

const bridge = @cImport({
    @cInclude("ctranslate2_bridge.h");
});

const sample_rate_hz: u32 = 16_000;
const log_mel_bins_count: u32 = @intCast(bridge.log_mel_bins_count);
const log_mel_frames_count: u32 = @intCast(bridge.log_mel_frames_count);
const log_mel_values_count: u32 = @intCast(bridge.log_mel_values_count);
const transcript_size_max: u32 = 64 * 1024;
const threads_count_default: u8 = 4;
const threads_count_max: u8 = 32;
const measurement_runs_count_default: u8 = 3;
const measurement_runs_count_max: u8 = 9;

comptime {
    assert(log_mel_values_count == log_mel_bins_count * log_mel_frames_count);
}

const Arguments = struct {
    model_path: [:0]const u8,
    audio_path: [:0]const u8,
    threads_count: u8,
    measurement_runs_count: u8,
};

const Benchmark = struct {
    feature_extraction_elapsed_ns: u64,
    model_load_elapsed_ns: u64,
    warmup_elapsed_ns: u64,
    measurement_elapsed_ns_min: u64,
    measurement_elapsed_ns_median: u64,
    measurement_elapsed_ns_max: u64,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var arguments: Arguments = undefined;
    try parse_arguments(&arguments, init.minimal.args, gpa);

    assert(arguments.threads_count > 0);
    assert(arguments.threads_count <= threads_count_max);
    assert(arguments.measurement_runs_count > 0);
    assert(arguments.measurement_runs_count <= measurement_runs_count_max);
    assert(arguments.measurement_runs_count % 2 == 1);

    // ── Load Fixture ──

    const samples = try load_fixture_audio(io, gpa, arguments.audio_path);
    defer gpa.free(samples);

    // ── Extract Log-Mel Features ──

    var extractor: log_mel.Extractor = undefined;
    try extractor.init(gpa);
    defer extractor.deinit();

    const feature_extraction_started = Io.Clock.awake.now(io);
    const features = try extractor.calculate(samples);
    defer gpa.free(features.values);
    const feature_extraction_elapsed_ns = measure_elapsed_awake_ns(io, feature_extraction_started);

    // ── Load Model ──

    var bridge_error: bridge.Error = undefined;

    const model_load_started = Io.Clock.awake.now(io);
    const model = bridge.model_create(
        arguments.model_path.ptr,
        arguments.threads_count,
        &bridge_error,
    ) orelse {
        stderr(
            "CTranslate2 model load failed: {s}\n",
            .{std.mem.sliceTo(&bridge_error.message, 0)},
        );

        return error.ModelLoadFailed;
    };
    defer bridge.model_destroy(model);

    const model_load_elapsed_ns = measure_elapsed_awake_ns(io, model_load_started);

    // ── Warm And Measure Transcription ──

    var gpt2_encoded_text_buffer: [transcript_size_max]u8 = undefined;

    const warmup_started = Io.Clock.awake.now(io);
    _ = try transcribe_log_mel_features(
        model,
        features,
        &gpt2_encoded_text_buffer,
        &bridge_error,
    );
    const warmup_elapsed_ns = measure_elapsed_awake_ns(io, warmup_started);

    var gpt2_encoded_text_size: u32 = 0;
    var measurement_elapsed_ns: [measurement_runs_count_max]u64 = undefined;
    for (measurement_elapsed_ns[0..arguments.measurement_runs_count]) |*elapsed_ns| {
        const transcription_started = Io.Clock.awake.now(io);
        gpt2_encoded_text_size = try transcribe_log_mel_features(
            model,
            features,
            &gpt2_encoded_text_buffer,
            &bridge_error,
        );
        elapsed_ns.* = measure_elapsed_awake_ns(io, transcription_started);
    }

    const measurements = measurement_elapsed_ns[0..arguments.measurement_runs_count];
    std.mem.sort(u64, measurements, {}, std.sort.asc(u64));
    const measurement_median_index = @divFloor(measurements.len, 2);

    // ── Decode And Report ──

    var transcript_buffer: [transcript_size_max]u8 = undefined;
    const transcript = try decode_gpt2_encoded_text(
        &transcript_buffer,
        gpt2_encoded_text_buffer[0..gpt2_encoded_text_size],
    );
    const benchmark: Benchmark = .{
        .feature_extraction_elapsed_ns = feature_extraction_elapsed_ns,
        .model_load_elapsed_ns = model_load_elapsed_ns,
        .warmup_elapsed_ns = warmup_elapsed_ns,
        .measurement_elapsed_ns_min = measurements[0],
        .measurement_elapsed_ns_median = measurements[measurement_median_index],
        .measurement_elapsed_ns_max = measurements[measurements.len - 1],
    };

    report_fixture(&arguments, samples.len, features.frames_count, transcript, &benchmark);
}

fn transcribe_log_mel_features(
    model: *bridge.ModelHandle,
    features: log_mel.Features,
    gpt2_encoded_text_buffer: *[transcript_size_max]u8,
    bridge_error_out: *bridge.Error,
) !u32 {
    assert(features.values.len == @as(usize, log_mel_values_count));
    assert(features.frames_count == log_mel_frames_count);
    assert(gpt2_encoded_text_buffer.len == transcript_size_max);

    var gpt2_encoded_text_out: bridge.Gpt2EncodedText = .{
        .bytes = gpt2_encoded_text_buffer,
        .capacity = gpt2_encoded_text_buffer.len,
        .size = 0,
    };

    const transcription_succeeded = bridge.model_transcribe(
        model,
        features.values.ptr,
        &gpt2_encoded_text_out,
        bridge_error_out,
    );

    if (!transcription_succeeded) {
        stderr(
            "CTranslate2 transcription failed: {s}\n",
            .{std.mem.sliceTo(&bridge_error_out.message, 0)},
        );

        return error.TranscriptionFailed;
    }

    assert(gpt2_encoded_text_out.size <= gpt2_encoded_text_buffer.len);

    return gpt2_encoded_text_out.size;
}

fn decode_gpt2_encoded_text(
    transcript_buffer: *[transcript_size_max]u8,
    gpt2_encoded_text: []const u8,
) ![]const u8 {
    const unmapped: u16 = 256;
    var byte_by_codepoint: [512]u16 = @splat(unmapped);
    var codepoint: u16 = 0;

    // GPT-2 leaves visible Latin-1 bytes at their Unicode code points and maps
    // every remaining byte, in byte order, to consecutive code points at 256.
    for (0..256) |byte| {
        const is_visible = (byte >= 33 and byte <= 126) or
            (byte >= 161 and byte <= 172) or
            (byte >= 174 and byte <= 255);

        if (is_visible) {
            byte_by_codepoint[byte] = @intCast(byte);
        } else {
            while (byte_by_codepoint[codepoint] != unmapped) : (codepoint += 1) {}
            byte_by_codepoint[256 + codepoint] = @intCast(byte);
            codepoint += 1;
        }
    }

    var transcript_size: usize = 0;
    var iterator = (try std.unicode.Utf8View.init(gpt2_encoded_text)).iterator();

    while (iterator.nextCodepoint()) |gpt2_codepoint| {
        if (gpt2_codepoint >= byte_by_codepoint.len) {
            return error.InvalidGpt2TokenCodepoint;
        }

        const byte = byte_by_codepoint[gpt2_codepoint];
        if (byte == unmapped) {
            return error.InvalidGpt2TokenCodepoint;
        }
        if (transcript_size == transcript_buffer.len) {
            return error.TranscriptExceedsLimit;
        }

        transcript_buffer[transcript_size] = @intCast(byte);
        transcript_size += 1;
    }

    if (!std.unicode.utf8ValidateSlice(transcript_buffer[0..transcript_size])) {
        return error.InvalidTranscriptUtf8;
    }

    return std.mem.trim(u8, transcript_buffer[0..transcript_size], " \t\r\n");
}

fn measure_elapsed_awake_ns(io: Io, started: Io.Timestamp) u64 {
    const elapsed = started.durationTo(Io.Clock.awake.now(io));

    assert(elapsed.nanoseconds >= 0);

    return @intCast(elapsed.nanoseconds);
}

fn parse_arguments(
    arguments: *Arguments,
    process_arguments: std.process.Args,
    gpa: Allocator,
) !void {
    var iterator = try std.process.Args.Iterator.initAllocator(process_arguments, gpa);
    defer iterator.deinit();

    assert(iterator.skip());

    // ── Parse Options ──

    var model_path: ?[:0]const u8 = null;
    var audio_path: ?[:0]const u8 = null;
    var threads_count: ?u8 = null;
    var measurement_runs_count: ?u8 = null;

    while (iterator.next()) |option| {
        if (std.mem.eql(u8, option, "--model")) {
            if (model_path != null) {
                return usage();
            }
            model_path = iterator.next() orelse return usage();
            continue;
        }
        if (std.mem.eql(u8, option, "--audio")) {
            if (audio_path != null) {
                return usage();
            }
            audio_path = iterator.next() orelse return usage();
            continue;
        }
        if (std.mem.eql(u8, option, "--threads")) {
            if (threads_count != null) {
                return usage();
            }

            const count_text = iterator.next() orelse return usage();
            const count = std.fmt.parseInt(u8, count_text, 10) catch return usage();
            if (count == 0 or count > threads_count_max) {
                return usage();
            }
            threads_count = count;
            continue;
        }
        if (std.mem.eql(u8, option, "--runs")) {
            if (measurement_runs_count != null) {
                return usage();
            }

            const count_text = iterator.next() orelse return usage();
            const count = std.fmt.parseInt(u8, count_text, 10) catch return usage();
            if (count == 0 or count > measurement_runs_count_max) {
                return usage();
            }
            // An odd count selects one observed run as the median instead of
            // averaging two measurements with different system interference.
            if (count % 2 == 0) {
                return usage();
            }
            measurement_runs_count = count;
            continue;
        }

        return usage();
    }

    // ── Resolve Defaults And Publish ──

    const model_path_required = model_path orelse return usage();
    const audio_path_required = audio_path orelse return usage();
    const threads_count_resolved = threads_count orelse threads_count_default;
    const measurement_runs_count_resolved =
        measurement_runs_count orelse measurement_runs_count_default;

    if (model_path_required.len == 0) {
        return usage();
    }
    if (audio_path_required.len == 0) {
        return usage();
    }

    assert(threads_count_resolved > 0);
    assert(threads_count_resolved <= threads_count_max);
    assert(measurement_runs_count_resolved > 0);
    assert(measurement_runs_count_resolved <= measurement_runs_count_max);
    assert(measurement_runs_count_resolved % 2 == 1);

    arguments.* = .{
        .model_path = model_path_required,
        .audio_path = audio_path_required,
        .threads_count = threads_count_resolved,
        .measurement_runs_count = measurement_runs_count_resolved,
    };
}

fn usage() error{InvalidArguments} {
    stderr(
        "usage: ctranslate2-fixture --model <model-directory> --audio <wav-path> " ++
            "[--threads <1-32>] [--runs <odd 1-9>]\n",
        .{},
    );
    return error.InvalidArguments;
}

fn load_fixture_audio(io: Io, gpa: Allocator, audio_path: []const u8) ![]f32 {
    const wav_file_size_max = 32 * 1024 * 1024;
    const samples_count_max = 15 * 60 * sample_rate_hz;

    assert(audio_path.len > 0);
    assert(samples_count_max <= std.math.maxInt(c_int));

    const wav_bytes = try Io.Dir.cwd().readFileAlloc(
        io,
        audio_path,
        gpa,
        .limited(wav_file_size_max),
    );
    defer gpa.free(wav_bytes);

    const sample_bytes = try parse_pcm_wav(wav_bytes);
    const samples_count = @divExact(sample_bytes.len, @sizeOf(i16));
    if (samples_count > samples_count_max) {
        return error.AudioDurationExceedsLimit;
    }

    const samples = try gpa.alloc(f32, samples_count);
    errdefer gpa.free(samples);

    for (samples, 0..) |*sample, sample_index| {
        const sample_offset = sample_index * @sizeOf(i16);
        const sample_i16 = std.mem.readInt(
            i16,
            sample_bytes[sample_offset..][0..@sizeOf(i16)],
            .little,
        );
        sample.* = @as(f32, @floatFromInt(sample_i16)) / 32768.0;
    }

    assert(samples.len == samples_count);
    assert(samples.len > 0);

    return samples;
}

fn parse_pcm_wav(wav_bytes: []const u8) ![]const u8 {
    const wav_chunks_count_max = 64;

    try validate_pcm_wav_header(wav_bytes);

    var format_bytes: ?[]const u8 = null;
    var sample_bytes: ?[]const u8 = null;
    var chunk_offset: usize = 12;
    var chunks_count: u8 = 0;

    while (chunks_count < wav_chunks_count_max) : (chunks_count += 1) {
        if (chunk_offset == wav_bytes.len) {
            break;
        }
        if (chunk_offset > wav_bytes.len or wav_bytes.len - chunk_offset < 8) {
            return error.InvalidWavChunk;
        }

        const chunk_name = wav_bytes[chunk_offset..][0..4];
        const chunk_size = std.mem.readInt(u32, wav_bytes[chunk_offset + 4 ..][0..4], .little);
        const chunk_data_offset = chunk_offset + 8;
        const chunk_data_end = chunk_data_offset + chunk_size;
        if (chunk_data_end > wav_bytes.len) {
            return error.InvalidWavChunk;
        }

        const chunk_padding_size = chunk_size & 1;
        if (chunk_data_end + chunk_padding_size > wav_bytes.len) {
            return error.InvalidWavChunk;
        }

        if (std.mem.eql(u8, chunk_name, "fmt ")) {
            if (format_bytes != null) {
                return error.DuplicateWavFormatChunk;
            }
            format_bytes = wav_bytes[chunk_data_offset..chunk_data_end];
        } else if (std.mem.eql(u8, chunk_name, "data")) {
            if (sample_bytes != null) {
                return error.DuplicateWavDataChunk;
            }
            sample_bytes = wav_bytes[chunk_data_offset..chunk_data_end];
        }

        chunk_offset = chunk_data_end + chunk_padding_size;
    }

    if (chunk_offset != wav_bytes.len) {
        return error.TooManyWavChunks;
    }

    const format = format_bytes orelse return error.MissingWavFormatChunk;
    const samples = sample_bytes orelse return error.MissingWavDataChunk;
    try validate_pcm_wav_format(format);

    if (samples.len == 0) {
        return error.EmptyWavData;
    }
    if (samples.len % @sizeOf(i16) != 0) {
        return error.InvalidWavDataSize;
    }

    return samples;
}

fn validate_pcm_wav_header(wav_bytes: []const u8) !void {
    if (wav_bytes.len < 12) {
        return error.InvalidWavHeader;
    }
    if (!std.mem.eql(u8, wav_bytes[0..4], "RIFF")) {
        return error.InvalidWavHeader;
    }
    if (!std.mem.eql(u8, wav_bytes[8..12], "WAVE")) {
        return error.InvalidWavHeader;
    }

    const riff_payload_size = std.mem.readInt(u32, wav_bytes[4..8], .little);
    if (@as(u64, riff_payload_size) + 8 != wav_bytes.len) {
        return error.InvalidWavFileSize;
    }
}

fn validate_pcm_wav_format(format: []const u8) !void {
    if (format.len < 16) {
        return error.InvalidWavFormat;
    }

    const encoding = std.mem.readInt(u16, format[0..2], .little);
    const channels_count = std.mem.readInt(u16, format[2..4], .little);
    const wav_sample_rate_hz = std.mem.readInt(u32, format[4..8], .little);
    const bytes_rate = std.mem.readInt(u32, format[8..12], .little);
    const sample_frame_size = std.mem.readInt(u16, format[12..14], .little);
    const sample_bits_count = std.mem.readInt(u16, format[14..16], .little);

    if (encoding != 1) {
        return error.UnsupportedWavEncoding;
    }
    if (channels_count != 1) {
        return error.UnsupportedWavChannels;
    }
    if (wav_sample_rate_hz != sample_rate_hz) {
        return error.UnsupportedWavSampleRate;
    }
    if (bytes_rate != sample_rate_hz * @sizeOf(i16)) {
        return error.InvalidWavByteRate;
    }
    if (sample_frame_size != @sizeOf(i16)) {
        return error.UnsupportedWavFrameSize;
    }
    if (sample_bits_count != @bitSizeOf(i16)) {
        return error.UnsupportedWavSampleSize;
    }
}

fn report_fixture(
    arguments: *const Arguments,
    samples_count: usize,
    frames_count: u32,
    transcript: []const u8,
    benchmark: *const Benchmark,
) void {
    assert(arguments.model_path.len > 0);
    assert(arguments.audio_path.len > 0);
    assert(samples_count > 0);
    assert(frames_count > 0);
    assert(benchmark.measurement_elapsed_ns_min <= benchmark.measurement_elapsed_ns_median);
    assert(benchmark.measurement_elapsed_ns_median <= benchmark.measurement_elapsed_ns_max);

    const audio_duration_ns = @divExact(
        @as(u64, samples_count) * std.time.ns_per_s,
        sample_rate_hz,
    );
    const realtime_factor_thousandths = @divFloor(
        benchmark.measurement_elapsed_ns_median * 1000,
        audio_duration_ns,
    );
    const resource_usage = std.posix.getrusage(std.posix.rusage.SELF);
    assert(resource_usage.maxrss >= 0);
    const peak_rss_mib_tenths = @divFloor(@as(u64, @intCast(resource_usage.maxrss)) * 10, 1024);

    stderr(
        "\nTranscript:\n{s}\n\n" ++
            "Model: {s}\n" ++
            "Audio: {s}\n" ++
            "Audio duration: {d}.{d} s\n" ++
            "Log-Mel frames: {d}\n" ++
            "Threads: {d}\n" ++
            "Measured runs: {d}\n" ++
            "Log-Mel extraction: {d} ms\n" ++
            "Model load: {d} ms\n" ++
            "Warm-up: {d} ms\n" ++
            "Transcription: {d}/{d}/{d} ms min/median/max\n" ++
            "Realtime factor: {d}.{d:0>3}x median\n" ++
            "Peak RSS: {d}.{d} MiB\n",
        .{
            transcript,
            arguments.model_path,
            arguments.audio_path,
            @divFloor(audio_duration_ns, std.time.ns_per_s),
            @divFloor(audio_duration_ns * 10, std.time.ns_per_s) % 10,
            frames_count,
            arguments.threads_count,
            arguments.measurement_runs_count,
            @divFloor(benchmark.feature_extraction_elapsed_ns, std.time.ns_per_ms),
            @divFloor(benchmark.model_load_elapsed_ns, std.time.ns_per_ms),
            @divFloor(benchmark.warmup_elapsed_ns, std.time.ns_per_ms),
            @divFloor(benchmark.measurement_elapsed_ns_min, std.time.ns_per_ms),
            @divFloor(benchmark.measurement_elapsed_ns_median, std.time.ns_per_ms),
            @divFloor(benchmark.measurement_elapsed_ns_max, std.time.ns_per_ms),
            @divFloor(realtime_factor_thousandths, 1000),
            realtime_factor_thousandths % 1000,
            @divFloor(peak_rss_mib_tenths, 10),
            peak_rss_mib_tenths % 10,
        },
    );
}
