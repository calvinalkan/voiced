//! End-to-end transcription checks for the pure Zig inference runtime.
//!
//! Audio fixtures are discovered from `zig/audio-fixtures/audio`; each `.wav` file must have an adjacent `.transcript` reference.

const std = @import("std");
const voiced_runtime_module = @import("root.zig");
const assert = std.debug.assert;

// ─── Test Limits ───────────────────────────────────────────────────────────

const transcript_size_max: usize = 64 * 1024;
const model_file_size_max: usize = 512 * 1024 * 1024;
const vocabulary_file_size_max: usize = 4 * 1024 * 1024;
const wav_file_size_max: usize = 2 * 1024 * 1024;

const TestModel = struct {
    name: []const u8,
    kind: voiced_runtime_module.ModelKind,
    installed_directory_name: []const u8,
};

// ─── Fixture Transcription ─────────────────────────────────────────────────

test "transcribes audio fixtures" {
    const allocator = std.testing.allocator;

    // ── Load Model Inputs ──
    //
    // Resolve the installed model once, then retain its pristine weights and vocabulary for the complete fixture run.

    const test_model = try selectedTestModel();
    const model_directory = try modelsDirectory(allocator, test_model.installed_directory_name);
    defer allocator.free(model_directory);

    const model_path = try std.fs.path.join(allocator, &.{ model_directory, "model.bin" });
    defer allocator.free(model_path);

    const vocabulary_path = try std.fs.path.join(allocator, &.{ model_directory, "vocabulary.txt" });
    defer allocator.free(vocabulary_path);

    const pristine_weights = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, model_path, allocator, .limited(model_file_size_max));
    defer allocator.free(pristine_weights);

    const vocabulary_text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, vocabulary_path, allocator, .limited(vocabulary_file_size_max));
    defer allocator.free(vocabulary_text);

    // ── Convert Model ──
    //
    // Conversion validates the pristine CTranslate2 tensors and produces the runtime's permanent packed representation.

    var model = try voiced_runtime_module.Model.fromPristineWeights(allocator, test_model.kind, pristine_weights);
    defer model.deinit();

    // ── Initialize Runtime ──
    //
    // Allocate the complete tensor arena before initialization. Runtime.init allocates one address-stable control object; transcribe() performs no dynamic allocation.

    const policy: voiced_runtime_module.Policy = .{ .workers_count = 4 };
    const runtime_memory_size = try voiced_runtime_module.Runtime.requiredMemorySize(model.kind, policy);

    const runtime_memory = try allocator.alignedAlloc(u8, .fromByteUnits(voiced_runtime_module.runtime_memory_alignment), runtime_memory_size);
    defer allocator.free(runtime_memory);

    const runtime: *voiced_runtime_module.Runtime = try voiced_runtime_module.Runtime.init(
        allocator,
        std.testing.io,
        &model,
        vocabulary_text,
        runtime_memory,
        policy,
    );
    defer runtime.deinit(allocator);

    // ── Discover Fixtures ──
    //
    // The directory is the test index. An optional exact fixture name keeps development runs deterministic without a manifest.

    const audio_fixtures = try openAudioFixturesDirectory();
    defer audio_fixtures.close(std.testing.io);

    const selected_fixture = std.testing.environ.getPosix("VOICED_RUNTIME_AUDIO_FIXTURE");
    const encoder_trailing_padding = try selectedEncoderTrailingPadding();
    const transcribe_options: voiced_runtime_module.TranscribeOptions = .{ .encoder_trailing_padding = encoder_trailing_padding };
    var transcript_output: [transcript_size_max]u8 = undefined;

    // ── Accumulate Suite Results ──

    var fixtures_count: usize = 0;
    var reference_words_count: usize = 0;
    var word_errors_count: usize = 0;
    var fixtures = audio_fixtures.iterate();

    while (try fixtures.next(std.testing.io)) |entry| {
        // ── Select Fixture ──

        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".wav")) {
            continue;
        }

        const id = entry.name[0 .. entry.name.len - ".wav".len];
        if (selected_fixture) |selected_id| {
            if (!std.mem.eql(u8, id, selected_id)) {
                continue;
            }
        }

        // ── Load Fixture ──
        //
        // The WAV filename determines the adjacent human-reference transcript filename.

        const wav_bytes = try audio_fixtures.readFileAlloc(std.testing.io, entry.name, allocator, .limited(wav_file_size_max));
        defer allocator.free(wav_bytes);

        const transcript_name = try std.fmt.allocPrint(allocator, "{s}.transcript", .{id});
        defer allocator.free(transcript_name);

        const expected_transcript = try audio_fixtures.readFileAlloc(
            std.testing.io,
            transcript_name,
            allocator,
            .limited(transcript_size_max),
        );
        defer allocator.free(expected_transcript);

        // ── Decode Samples ──
        //
        // parsePcmWav borrows the PCM payload.
        // The runtime consumes normalized Float32 samples,
        // so the test allocates and converts that representation here.

        const sample_bytes = try parsePcmWav(wav_bytes);
        const encoded_sample_size = @sizeOf(i16);
        const samples_count = @divExact(sample_bytes.len, encoded_sample_size);
        const samples = try allocator.alloc(f32, samples_count);
        defer allocator.free(samples);

        // Signed 16-bit PCM spans -32768 through 32767.
        // Dividing by 32768 maps the complete range into the runtime's
        // accepted [-1, 1] interval without clipping the negative endpoint.
        const pcm_full_scale: f32 = 32_768.0;

        for (samples, 0..) |*sample, sample_index| {
            const sample_offset = sample_index * encoded_sample_size;
            const encoded_sample = sample_bytes[sample_offset..][0..encoded_sample_size];
            const integer_sample = std.mem.readInt(i16, encoded_sample, .little);

            sample.* = @as(f32, @floatFromInt(integer_sample)) / pcm_full_scale;
        }

        // ── Transcribe ──

        const transcription_start = std.Io.Clock.awake.now(std.testing.io);

        const transcription = try runtime.transcribe(samples, &transcript_output, transcribe_options);

        const transcription_elapsed = transcription_start.durationTo(std.Io.Clock.awake.now(std.testing.io));
        assert(transcription_elapsed.nanoseconds >= 0);
        const transcription_elapsed_ms = @as(f64, @floatFromInt(transcription_elapsed.nanoseconds)) / std.time.ns_per_ms;

        try std.testing.expect(transcription.text.len > 0);
        try std.testing.expect(std.unicode.utf8ValidateSlice(transcription.text));

        // ── Score Transcript ──
        //
        // Word error rate ignores ASCII case and punctuation while the report retains both original strings.

        const expected_words = try splitAndNormalizeWords(allocator, expected_transcript);
        defer allocator.free(expected_words);

        const actual_words = try splitAndNormalizeWords(allocator, transcription.text);
        defer allocator.free(actual_words);

        const errors_count = try levenshteinDistance(allocator, expected_words, actual_words);

        reference_words_count += expected_words.len;
        word_errors_count += errors_count;
        fixtures_count += 1;

        // ── Report Fixture ──

        std.debug.print("\n{s}\n  expected: {s}\n  actual:   {s}\n  word errors: {d}/{d}\n  encoder positions: {d}\n  no-speech probability: {d:.6}\n  average log probability: {d:.6}\n  latency: {d:.1} ms\n", .{ id, std.mem.trim(u8, expected_transcript, " \t\r\n"), transcription.text, errors_count, expected_words.len, transcription.encoder_positions_count, transcription.no_speech_probability, transcription.average_log_probability, transcription_elapsed_ms });
    }

    // ── Report Suite ──
    //
    // A missing selected fixture fails here instead of silently producing an empty successful run.

    try std.testing.expect(fixtures_count > 0);
    try std.testing.expect(reference_words_count > 0);

    const word_error_rate = @as(f64, @floatFromInt(word_errors_count)) / @as(f64, @floatFromInt(reference_words_count));

    std.debug.print("\nPure Zig {s} transcription with {s} trailing padding: {d} fixtures, {d}/{d} word errors ({d:.2}%)\n", .{ test_model.name, @tagName(encoder_trailing_padding), fixtures_count, word_errors_count, reference_words_count, 100.0 * word_error_rate });

    try std.testing.expect(word_error_rate <= 0.06);
}

fn openAudioFixturesDirectory() !std.Io.Dir {
    const cwd = std.Io.Dir.cwd();
    if (std.testing.environ.getPosix("VOICED_RUNTIME_AUDIO_FIXTURES_DIRECTORY")) |path| {
        return cwd.openDir(std.testing.io, path, .{ .iterate = true });
    }

    const directory = cwd.openDir(std.testing.io, "zig/audio-fixtures/audio", .{ .iterate = true }) catch |err| {
        if (err != error.FileNotFound) {
            return err;
        }

        // Zig package commands run with zig/ as their working directory.
        const package_directory = try cwd.openDir(std.testing.io, "audio-fixtures/audio", .{ .iterate = true });
        return package_directory;
    };

    return directory;
}

// ─── Test Configuration ───────────────────────────────────────────────────

fn selectedEncoderTrailingPadding() !voiced_runtime_module.EncoderTrailingPadding {
    const seconds = std.testing.environ.getPosix("VOICED_RUNTIME_TRAILING_PADDING_SECONDS") orelse "30";

    if (std.mem.eql(u8, seconds, "5")) {
        return .seconds_5;
    }
    if (std.mem.eql(u8, seconds, "10")) {
        return .seconds_10;
    }
    if (std.mem.eql(u8, seconds, "30")) {
        return .seconds_30;
    }

    return error.UnsupportedEncoderTrailingPadding;
}

fn selectedTestModel() !TestModel {
    const name = std.testing.environ.getPosix("VOICED_RUNTIME_MODEL") orelse "base.en";

    if (std.mem.eql(u8, name, "base.en")) {
        return .{ .name = name, .kind = .base_en, .installed_directory_name = "Systran/faster-whisper-base.en" };
    }
    if (std.mem.eql(u8, name, "small.en")) {
        return .{ .name = name, .kind = .small_en, .installed_directory_name = "Systran/faster-whisper-small.en" };
    }

    return error.UnsupportedTestModel;
}

// ─── Installed Model Location ──────────────────────────────────────────────

fn modelsDirectory(allocator: std.mem.Allocator, directory_name: []const u8) ![]u8 {
    const env = std.testing.environ;
    if (env.getPosix("XDG_DATA_HOME")) |data_home| {
        return std.fs.path.join(allocator, &.{ data_home, "voiced/models", directory_name });
    }

    const home = env.getPosix("HOME") orelse {
        return error.HomeNotSet;
    };

    return std.fs.path.join(allocator, &.{ home, ".local/share/voiced/models", directory_name });
}

fn parsePcmWav(wav_bytes: []const u8) ![]const u8 {
    const wav_size = wav_bytes.len;

    // ── Validate RIFF Header ──

    if (wav_size < 12 or !std.mem.eql(u8, wav_bytes[0..4], "RIFF") or !std.mem.eql(u8, wav_bytes[8..12], "WAVE")) {
        return error.InvalidWav;
    }

    const riff_payload_size = std.mem.readInt(u32, wav_bytes[4..8], .little);
    if (@as(u64, riff_payload_size) + 8 != wav_size) {
        return error.InvalidWav;
    }

    // ── Find Format and Sample Chunks ──
    //
    // WAV permits unrelated chunks and does not require fmt/data to appear at fixed offsets.

    var format: ?[]const u8 = null;
    var samples: ?[]const u8 = null;
    var chunk_offset: usize = 12;

    while (chunk_offset < wav_size) {
        if (wav_size - chunk_offset < 8) {
            return error.InvalidWav;
        }

        const chunk_name = wav_bytes[chunk_offset..][0..4];
        const chunk_size = std.mem.readInt(u32, wav_bytes[chunk_offset + 4 ..][0..4], .little);
        const data_offset = chunk_offset + 8;
        const data_end = data_offset + chunk_size;

        if (data_end > wav_size) {
            return error.InvalidWav;
        }

        if (std.mem.eql(u8, chunk_name, "fmt ")) {
            format = wav_bytes[data_offset..data_end];
        } else if (std.mem.eql(u8, chunk_name, "data")) {
            samples = wav_bytes[data_offset..data_end];
        }

        chunk_offset = data_end + (chunk_size & 1);
    }

    // ── Validate PCM Representation ──
    //
    // Runtime fixtures are mono, 16 kHz, signed 16-bit little-endian PCM.

    const format_bytes = format orelse {
        return error.InvalidWav;
    };
    const sample_bytes = samples orelse {
        return error.InvalidWav;
    };
    if (format_bytes.len < 16 or std.mem.readInt(u16, format_bytes[0..2], .little) != 1 or std.mem.readInt(u16, format_bytes[2..4], .little) != 1 or std.mem.readInt(u32, format_bytes[4..8], .little) != 16_000 or std.mem.readInt(u16, format_bytes[14..16], .little) != 16) {
        return error.InvalidWav;
    }
    if (sample_bytes.len == 0 or sample_bytes.len % @sizeOf(i16) != 0) {
        return error.InvalidWav;
    }

    return sample_bytes;
}

fn splitAndNormalizeWords(allocator: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var normalized_words: std.ArrayList([]const u8) = .empty;
    defer normalized_words.deinit(allocator);

    var word_start_offset: ?usize = null;

    for (text, 0..) |byte, i| {
        if (std.ascii.isAlphanumeric(byte)) {
            if (word_start_offset == null) {
                // This byte begins a new word.
                word_start_offset = i;
            }
        } else if (word_start_offset) |start| {
            // A non-alphanumeric byte ends the current word and appends it to the list of normalized words.
            try normalized_words.append(allocator, text[start..i]);

            word_start_offset = null;
        }
    }

    if (word_start_offset) |start| {
        // Handle the last word, which may not be followed by a non-alphanumeric byte.
        try normalized_words.append(allocator, text[start..]);
    }

    return normalized_words.toOwnedSlice(allocator);
}

fn levenshteinDistance(allocator: std.mem.Allocator, expected: []const []const u8, actual: []const []const u8) !usize {
    // ── Allocate Matrix ──
    //
    // Cell [row, column] is the minimum edits needed to turn the expected prefix into the actual prefix.

    const rows_count = expected.len + 1;
    const columns_count = actual.len + 1;
    const distances = try allocator.alloc(usize, rows_count * columns_count);
    defer allocator.free(distances);

    // ── Initialize Empty Prefixes ──
    //
    // The first column deletes expected words; the first row inserts actual words.

    for (0..rows_count) |row| {
        distances[row * columns_count] = row;
    }

    for (0..columns_count) |column| {
        distances[column] = column;
    }

    // ── Fill Matrix ──
    //
    // Each remaining cell chooses the cheapest deletion, insertion, or case-insensitive replacement.

    for (expected, 0..) |expected_word, expected_index| {
        const row = expected_index + 1;

        for (actual, 0..) |actual_word, actual_index| {
            const column = actual_index + 1;
            const delete_expected_word = distances[(row - 1) * columns_count + column] + 1;
            const insert_actual_word = distances[row * columns_count + column - 1] + 1;
            const words_are_different = !std.ascii.eqlIgnoreCase(expected_word, actual_word);
            const replace_expected_word = distances[(row - 1) * columns_count + column - 1] + @intFromBool(words_are_different);

            distances[row * columns_count + column] = @min(delete_expected_word, @min(insert_actual_word, replace_expected_word));
        }
    }

    // ── Read Complete-Transcript Distance ──

    return distances[(rows_count - 1) * columns_count + columns_count - 1];
}
