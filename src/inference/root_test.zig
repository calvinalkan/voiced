//! End-to-end transcription checks for the pure Zig inference runtime.
//!
//! Audio fixtures are discovered from `__fixtures__/librispeech/audio`; each `.wav` file must have an adjacent `.transcript` reference.

const std = @import("std");
const inference = @import("root.zig");
const packed_model = @import("../packed_model/root.zig");
const vnni_weight = @import("vnni_weight.zig");
const assert = std.debug.assert;

// ─── Test Limits ───────────────────────────────────────────────────────────

const transcript_size_max: usize = 64 * 1024;
const wav_file_size_max: usize = 2 * 1024 * 1024;

const TestModel = struct {
    name: []const u8,
    kind: inference.Model.Kind,
};

fn loadTestModel(allocator: std.mem.Allocator, test_model: TestModel) !inference.Model {
    const io = std.testing.io;
    const directory_path = try modelDirectoryPath(allocator);
    defer allocator.free(directory_path);
    var directory = try std.Io.Dir.cwd().openDir(io, directory_path, .{});
    defer directory.close(io);
    var file_name_buffer: [32]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&file_name_buffer, "{s}.voiced", .{test_model.name});

    return packed_model.load(io, directory, file_name, test_model.kind);
}

// ─── Fixture Transcription ─────────────────────────────────────────────────

test "transcribes audio fixtures" {
    const allocator = std.testing.allocator;

    // ── Load Model Inputs ──
    //
    // Load the same packed model that setup publishes.

    const test_model = try selectedTestModel();
    var model = try loadTestModel(allocator, test_model);
    defer model.deinit();

    // ── Initialize Runtime ──
    //
    // The test owns both the tensor arena and address-stable runtime. Neither
    // runtime initialization nor transcription allocates tensor/control storage.

    const workers_count = if (std.testing.environ.getPosix("VOICED_RUNTIME_WORKERS")) |value| try std.fmt.parseInt(usize, value, 10) else 4;
    const decoder_workers_count = if (std.testing.environ.getPosix("VOICED_RUNTIME_DECODER_WORKERS")) |value| try std.fmt.parseInt(usize, value, 10) else null;
    const pool_config: inference.WorkerPool.Config = .{ .workers_count = workers_count };
    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), try inference.WorkerPool.requiredMemory(pool_config));
    defer allocator.free(pool_memory);
    const pool = try inference.WorkerPool.init(pool_memory, pool_config);
    defer pool.deinit();
    const config: inference.Runtime.Config = .{
        .audio_samples_count_max = log_mel.samples_count_max,
        .transcript_tokens_count_max = 224,
        .encoder_padding_max = .seconds_30,
    };
    const runtime_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), try inference.Runtime.requiredMemory(&model, pool, config));
    defer allocator.free(runtime_memory);
    const runtime = try inference.Runtime.init(runtime_memory, &model, pool, config);
    defer runtime.deinit();
    if (std.testing.environ.getPosix("VOICED_RUNTIME_WORKER_CPUS")) |cpu_list| {
        var ids = std.mem.splitScalar(u8, cpu_list, ',');
        for (pool.workers) |thread| {
            const id = ids.next() orelse return error.InvalidWorkerCpus;
            const cpu = try std.fmt.parseInt(usize, id, 10);
            var mask: std.os.linux.cpu_set_t = @splat(0);
            if (cpu >= @bitSizeOf(@TypeOf(mask))) return error.InvalidWorkerCpus;
            mask[cpu / @bitSizeOf(usize)] = @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
            try std.os.linux.sched_setaffinity(thread.getHandle(), &mask);
        }
        if (ids.next() != null) return error.InvalidWorkerCpus;
    }

    // ── Discover Fixtures ──
    //
    // The directory is the test index. An optional exact fixture name keeps development runs deterministic without a manifest.

    const audio_fixtures = try openAudioFixturesDirectory();
    defer audio_fixtures.close(std.testing.io);

    const selected_fixture = std.testing.environ.getPosix("VOICED_RUNTIME_AUDIO_FIXTURE");
    const encoder_trailing_padding = try selectedEncoderTrailingPadding();
    const transcribe_options: inference.Runtime.TranscriptionOptions = .{ .encoder_padding = encoder_trailing_padding, .decoder_workers_count_max = decoder_workers_count };

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

        const result = runtime.transcribe(samples, transcribe_options);
        const transcription = switch (result) {
            .ok, .token_limit => |output| output,
            else => return error.UnexpectedTranscriptionResult,
        };
        const timings = transcription.timings;

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
        std.debug.print("  phases ms: mel={d:.3} encoder={d:.3} cross={d:.3} decoder={d:.3}\n", .{
            @as(f64, @floatFromInt(timings.log_mel_ns)) / std.time.ns_per_ms,
            @as(f64, @floatFromInt(timings.encoder_ns)) / std.time.ns_per_ms,
            @as(f64, @floatFromInt(timings.cross_key_values_ns)) / std.time.ns_per_ms,
            @as(f64, @floatFromInt(timings.decoder_ns)) / std.time.ns_per_ms,
        });
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

test "object boundary validates capacities and preserves failure evidence" {
    const allocator = std.testing.allocator;
    var model = try loadTestModel(allocator, try selectedTestModel());
    defer model.deinit();
    const pool_config: inference.WorkerPool.Config = .{ .workers_count = 2 };
    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), try inference.WorkerPool.requiredMemory(pool_config));
    defer allocator.free(pool_memory);
    const pool = try inference.WorkerPool.init(pool_memory, pool_config);
    defer pool.deinit();

    const config: inference.Runtime.Config = .{ .audio_samples_count_max = 320, .transcript_tokens_count_max = 1, .encoder_padding_max = .seconds_5 };
    for ([_]usize{ 0, 200, 480_001, std.math.maxInt(usize) }) |limit| {
        var invalid = config;
        invalid.audio_samples_count_max = limit;
        try std.testing.expectError(error.InvalidConfig, inference.Runtime.requiredMemory(&model, pool, invalid));
    }
    for ([_]usize{ 0, 447, std.math.maxInt(usize) }) |limit| {
        var invalid = config;
        invalid.transcript_tokens_count_max = limit;
        try std.testing.expectError(error.InvalidConfig, inference.Runtime.requiredMemory(&model, pool, invalid));
    }
    var invalid_model = model;
    invalid_model.vocabulary.offsets = &.{};
    try std.testing.expectError(error.InvalidModel, inference.Runtime.requiredMemory(&invalid_model, pool, config));

    const size = try inference.Runtime.requiredMemory(&model, pool, config);
    const memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), size + 64);
    defer allocator.free(memory);
    @memset(memory[size..], 0xa5);
    for ([_]usize{ 0, 1, 63, 64, 65, size - 1 }) |short_size| {
        try std.testing.expectError(error.MemoryTooSmall, inference.Runtime.init(memory[0..short_size], &model, pool, config));
    }
    const runtime = try inference.Runtime.init(memory[0..size], &model, pool, config);
    defer runtime.deinit();
    var samples: [321]f32 = @splat(0);
    const options: inference.Runtime.TranscriptionOptions = .{ .encoder_padding = .seconds_5 };
    const result = runtime.transcribe(samples[0..320], options);
    try std.testing.expect(result == .token_limit);
    const expected_token = result.token_limit.tokens[0];
    try std.testing.expectEqual(@as(usize, 1), result.token_limit.tokens.len);
    try std.testing.expect(result.token_limit.timings.encoder_ns > 0);

    // Rejected requests carry only their own variant, never a stale output.
    for (0..5) |case| {
        var rejected = options;
        var cancellation: std.atomic.Value(bool) = .init(true);
        switch (case) {
            0 => {
                rejected.encoder_padding = .seconds_10;
                try std.testing.expectEqual(.encoder_padding_exceeds_limit, runtime.transcribe(samples[0..320], rejected));
            },
            1 => {
                rejected.encoder_workers_count_max = 0;
                try std.testing.expectEqual(.invalid_worker_count, runtime.transcribe(samples[0..320], rejected));
            },
            2 => {
                rejected.decoder_workers_count_max = 3;
                try std.testing.expectEqual(.invalid_worker_count, runtime.transcribe(samples[0..320], rejected));
            },
            3 => {
                for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 1.001, -1.001 }) |invalid| {
                    samples[0] = invalid;
                    defer samples[0] = 0;
                    try std.testing.expectEqual(.invalid_samples, runtime.transcribe(samples[0..320], rejected));
                }
            },
            4 => {
                rejected.cancellation = &cancellation;
                const cancelled = runtime.transcribe(samples[0..320], rejected);
                try std.testing.expect(cancelled == .cancelled);
                try std.testing.expectEqualDeep(inference.TranscriptionResult.Timings{ .log_mel_ns = 0, .encoder_ns = 0, .cross_key_values_ns = 0, .decoder_ns = 0 }, cancelled.cancelled);
            },
            else => unreachable,
        }
    }
    try std.testing.expectEqual(.audio_too_short, runtime.transcribe(samples[0..200], options));
    try std.testing.expectEqual(.audio_duration_exceeds_limit, runtime.transcribe(&samples, options));
    // A rejected/cancelled request leaves the same arena immediately reusable.
    const resumed = runtime.transcribe(samples[0..320], options);
    try std.testing.expect(resumed == .token_limit);
    try std.testing.expectEqual(expected_token, resumed.token_limit.tokens[0]);
    for (memory[size..]) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);

    // Private vocabulary bytes exercise the late error without changing weights
    // or the mapped model. The first token cannot be EOT, so decoding emits bytes.
    const invalid_bytes = try allocator.alloc(u8, model.vocabulary.bytes.len);
    defer allocator.free(invalid_bytes);
    @memset(invalid_bytes, 0xff);
    invalid_model = model;
    invalid_model.vocabulary.bytes = invalid_bytes;
    const invalid_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), try inference.Runtime.requiredMemory(&invalid_model, pool, config));
    defer allocator.free(invalid_memory);
    const invalid_runtime = try inference.Runtime.init(invalid_memory, &invalid_model, pool, config);
    defer invalid_runtime.deinit();
    const invalid = invalid_runtime.transcribe(samples[0..320], options);
    try std.testing.expect(invalid == .invalid_transcript_encoding);
    const output = invalid.invalid_transcript_encoding;
    try std.testing.expectEqual(@as(usize, 1), output.tokens.len);
    try std.testing.expect(output.text.len > 0);
    try std.testing.expect(std.math.isFinite(output.average_log_probability));
    try std.testing.expect(output.timings.encoder_ns > 0 and output.timings.decoder_ns > 0);
    for (output.text) |byte| try std.testing.expectEqual(@as(u8, 0xff), byte);
}

test "token limits preserve complete UTF-8 characters and all tokens" {
    const allocator = std.testing.allocator;
    var model = try loadTestModel(allocator, .{ .name = "whisper.small.en", .kind = .whisper_small_en });
    defer model.deinit();
    const wav = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "__fixtures__/inference/token_limit_utf8.wav", allocator, .limited(wav_file_size_max));
    defer allocator.free(wav);
    const pcm = try parsePcmWav(wav);
    const samples = try allocator.alloc(f32, pcm.len / 2);
    defer allocator.free(samples);
    for (samples, 0..) |*sample, index| sample.* = @as(f32, @floatFromInt(std.mem.readInt(i16, pcm[index * 2 ..][0..2], .little))) / 32768;

    const pool_config: inference.WorkerPool.Config = .{ .workers_count = 4 };
    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), try inference.WorkerPool.requiredMemory(pool_config));
    defer allocator.free(pool_memory);
    const pool = try inference.WorkerPool.init(pool_memory, pool_config);
    defer pool.deinit();
    // The unmodified model splits ń (C5 84) across tokens 14 and 15.
    // The synthetic speech is intentionally an encoding fixture, not a WER oracle.
    const tokens = [_]u32{ 383, 16639, 26777, 19632, 449, 2271, 16639, 371, 5092, 404, 430, 449, 2271, 129, 226, 64 };
    const text = " The Czech composer Leo Jana Czech Rotiopra Janańa";
    const cases = [_]struct { capacity: usize, tokens_count: usize, text_size: usize }{
        .{ .capacity = 14, .tokens_count = 14, .text_size = text.len - 3 },
        .{ .capacity = 15, .tokens_count = 15, .text_size = text.len - 1 },
        .{ .capacity = 446, .tokens_count = tokens.len, .text_size = text.len },
    };
    for (cases) |case| {
        const config: inference.Runtime.Config = .{ .audio_samples_count_max = 480000, .transcript_tokens_count_max = case.capacity, .encoder_padding_max = .seconds_30 };
        const memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), try inference.Runtime.requiredMemory(&model, pool, config));
        defer allocator.free(memory);
        const runtime = try inference.Runtime.init(memory, &model, pool, config);
        defer runtime.deinit();
        for (0..3) |_| {
            const options: inference.Runtime.TranscriptionOptions = .{ .encoder_padding = .seconds_5 };
            const result = runtime.transcribe(samples, options);
            const output = if (case.capacity < 446) output: {
                try std.testing.expect(result == .token_limit);
                break :output result.token_limit;
            } else output: {
                try std.testing.expect(result == .ok);
                break :output result.ok;
            };
            try std.testing.expectEqualSlices(u32, tokens[0..case.tokens_count], output.tokens);
            try std.testing.expectEqualStrings(text[0..case.text_size], output.text);
            try std.testing.expect(std.unicode.utf8ValidateSlice(output.text));
            try std.testing.expectEqual(.audio_too_short, runtime.transcribe(samples[0..200], options));
        }
    }
}

test "cancellation retains elapsed work and permits immediate reuse" {
    const allocator = std.testing.allocator;
    var model = try loadTestModel(allocator, try selectedTestModel());
    defer model.deinit();
    const pool_config: inference.WorkerPool.Config = .{ .workers_count = 2 };
    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), try inference.WorkerPool.requiredMemory(pool_config));
    defer allocator.free(pool_memory);
    const pool = try inference.WorkerPool.init(pool_memory, pool_config);
    defer pool.deinit();
    const config: inference.Runtime.Config = .{ .audio_samples_count_max = 320, .transcript_tokens_count_max = 1, .encoder_padding_max = .seconds_30 };
    const memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), try inference.Runtime.requiredMemory(&model, pool, config));
    defer allocator.free(memory);
    const runtime = try inference.Runtime.init(memory, &model, pool, config);
    defer runtime.deinit();
    const Context = struct {
        runtime: *inference.Runtime,
        started: std.atomic.Value(bool) = .init(false),
        cancel: std.atomic.Value(bool) = .init(false),
        result: inference.TranscriptionResult = undefined,
        fn run(context: *@This()) void {
            const samples: [320]f32 = @splat(0);
            context.started.store(true, .release);
            context.result = context.runtime.transcribe(&samples, .{ .encoder_padding = .seconds_30, .cancellation = &context.cancel });
        }
    };
    // Delay after entry to exercise cancellation during feature/encoder work.
    // The pre-entry cancellation case is covered by the validation test above.
    var observed_elapsed_work = false;
    for (0..3) |_| {
        var context: Context = .{ .runtime = runtime };
        const thread = try std.Thread.spawn(.{}, Context.run, .{&context});
        while (!context.started.load(.acquire)) std.atomic.spinLoopHint();
        std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch |err| {
            context.cancel.store(true, .release);
            thread.join();
            return err;
        };
        context.cancel.store(true, .release);
        thread.join();
        try std.testing.expect(context.result == .cancelled);
        const timings = context.result.cancelled;
        observed_elapsed_work = observed_elapsed_work or timings.log_mel_ns > 0 or timings.encoder_ns > 0;
        const samples: [320]f32 = @splat(0);
        const resumed = runtime.transcribe(&samples, .{ .encoder_padding = .seconds_5 });
        try std.testing.expect(resumed == .token_limit);
    }
    try std.testing.expect(observed_elapsed_work);
}

test "object boundary supports a pool larger than one bitset word" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidConfig, inference.WorkerPool.requiredMemory(.{ .workers_count = 0 }));
    for ([_]usize{ std.math.maxInt(usize), std.math.maxInt(usize) / 20 }) |count| {
        try std.testing.expectError(error.MemorySizeOverflow, inference.WorkerPool.requiredMemory(.{ .workers_count = count }));
    }
    const pool_config: inference.WorkerPool.Config = .{ .workers_count = 65 };
    const pool_size = try inference.WorkerPool.requiredMemory(pool_config);
    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), pool_size + 64);
    defer allocator.free(pool_memory);
    @memset(pool_memory[pool_size..], 0xa5);
    try std.testing.expectError(error.MemoryTooSmall, inference.WorkerPool.init(pool_memory[0 .. pool_size - 1], pool_config));
    const pool = try inference.WorkerPool.init(pool_memory[0..pool_size], pool_config);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 65), pool.workersCount());

    var model = try loadTestModel(allocator, try selectedTestModel());
    defer model.deinit();
    const config: inference.Runtime.Config = .{ .audio_samples_count_max = 320, .transcript_tokens_count_max = 1, .encoder_padding_max = .seconds_5 };
    const memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), try inference.Runtime.requiredMemory(&model, pool, config));
    defer allocator.free(memory);
    const runtime = try inference.Runtime.init(memory, &model, pool, config);
    defer runtime.deinit();
    const samples: [320]f32 = @splat(0);
    // Exercise the object layout and public pool binding without oversubscribing
    // every math barrier: scheduler tests separately dispatch all 65 workers.
    const result = runtime.transcribe(&samples, .{ .encoder_padding = .seconds_5, .encoder_workers_count_max = 1 });
    try std.testing.expect(result == .token_limit);
    try std.testing.expectEqual(@as(usize, 1), result.token_limit.tokens.len);
    for (pool_memory[pool_size..]) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
}

test "object boundary shares a pool between concurrent runtimes" {
    const allocator = std.testing.allocator;
    var model = try loadTestModel(allocator, try selectedTestModel());
    defer model.deinit();
    const pool_config: inference.WorkerPool.Config = .{ .workers_count = 4 };
    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), try inference.WorkerPool.requiredMemory(pool_config));
    defer allocator.free(pool_memory);
    const pool = try inference.WorkerPool.init(pool_memory, pool_config);
    defer pool.deinit();
    const config: inference.Runtime.Config = .{ .audio_samples_count_max = 320, .transcript_tokens_count_max = 1, .encoder_padding_max = .seconds_5 };
    const size = try inference.Runtime.requiredMemory(&model, pool, config);
    var memories: [2][]align(inference.Runtime.memory_alignment) u8 = undefined;
    var runtimes: [2]*inference.Runtime = undefined;
    var initialized: usize = 0;
    defer for (0..initialized) |index| {
        runtimes[index].deinit();
        allocator.free(memories[index]);
    };
    while (initialized < runtimes.len) : (initialized += 1) {
        memories[initialized] = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), size);
        errdefer allocator.free(memories[initialized]);
        runtimes[initialized] = try inference.Runtime.init(memories[initialized], &model, pool, config);
    }
    const samples: [320]f32 = @splat(0);
    const expected = runtimes[0].transcribe(&samples, .{ .encoder_padding = .seconds_5 });
    try std.testing.expect(expected == .token_limit);
    const expected_token = expected.token_limit.tokens[0];
    const Context = struct {
        runtime: *inference.Runtime,
        samples: []const f32,
        width: usize,
        start: *std.atomic.Value(bool),
        outcome: ?inference.TranscriptionResult = null,
        fn run(context: *@This()) void {
            while (!context.start.load(.acquire)) std.atomic.spinLoopHint();
            context.outcome = context.runtime.transcribe(context.samples, .{ .encoder_padding = .seconds_5, .encoder_workers_count_max = context.width, .decoder_workers_count_max = 1 });
        }
    };
    for (0..8) |round| {
        var start: std.atomic.Value(bool) = .init(false);
        var contexts = [_]Context{
            .{ .runtime = runtimes[0], .samples = &samples, .width = 4, .start = &start },
            .{ .runtime = runtimes[1], .samples = &samples, .width = 1 + round % 4, .start = &start },
        };
        const first = try std.Thread.spawn(.{}, Context.run, .{&contexts[0]});
        const second = std.Thread.spawn(.{}, Context.run, .{&contexts[1]}) catch |err| {
            start.store(true, .release);
            first.join();
            return err;
        };
        start.store(true, .release);
        first.join();
        second.join();
        for (contexts) |context| {
            const result = context.outcome orelse return error.MissingResult;
            try std.testing.expect(result == .token_limit);
            try std.testing.expectEqual(expected_token, result.token_limit.tokens[0]);
        }
    }
}

// ─── Convolution Frontend Equivalence ──────────────────────────────────────

const encoder_module = @import("encoder.zig");
const linear = @import("linear.zig");
const log_mel = @import("log_mel.zig");
const Scheduler = @import("Scheduler.zig");
const model_module = inference.Model;

test "blocked convolution frontend matches materialized reference" {
    const allocator = std.testing.allocator;
    const test_model = try selectedTestModel();
    var model = try loadTestModel(allocator, test_model);
    defer model.deinit();
    const weights = &model.weights;
    const dimensions = inference.Model.dimensions(model.kind);
    const width = dimensions.encoder_width;

    // Use actual Mel values, retaining channel-major storage when varying the
    // logical length. Tiny/odd lengths and lane boundaries exercise both halos.
    const fixtures = try openAudioFixturesDirectory();
    defer fixtures.close(std.testing.io);
    var entries = fixtures.iterate();
    var wav: ?[]u8 = null;
    while (try entries.next(std.testing.io)) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".wav")) {
            wav = try fixtures.readFileAlloc(std.testing.io, entry.name, allocator, .limited(wav_file_size_max));
            break;
        }
    }
    const wav_bytes = wav orelse {
        return error.MissingAudioFixture;
    };
    defer allocator.free(wav_bytes);
    const sample_bytes = try parsePcmWav(wav_bytes);
    const samples = try allocator.alloc(f32, sample_bytes.len / 2);
    defer allocator.free(samples);
    for (samples, 0..) |*sample, index| {
        sample.* = @as(f32, @floatFromInt(std.mem.readInt(i16, sample_bytes[index * 2 ..][0..2], .little))) / 32768.0;
    }
    const mel_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), log_mel.Extractor.memory_size);
    defer allocator.free(mel_memory);
    var extractor: log_mel.Extractor = undefined;
    try extractor.init(mel_memory, log_mel.samples_count_max);
    const mel_workspace = try allocator.alloc(f32, log_mel.workspace_values_count);
    defer allocator.free(mel_workspace);
    const actual_features = try extractor.calculate(samples, .seconds_30, mel_workspace);
    const mel = try allocator.alloc(f32, log_mel.mel_bins_count * log_mel.encoder_frames_count_max);
    defer allocator.free(mel);
    const first = try allocator.alloc(f32, log_mel.encoder_frames_count_max * width);
    defer allocator.free(first);
    const expected = try allocator.alloc(f32, dimensions.encoder_positions_count_max * width);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, expected.len);
    defer allocator.free(actual);

    for ([_]usize{ 1, 3, 4, 8, 16 }) |workers_count| {
        const float_scratch = try allocator.alloc(f32, workers_count * encoder_module.laneFloatScratchValuesCount(dimensions));
        defer allocator.free(float_scratch);
        const quantized_scratch = try allocator.alloc(u8, workers_count * encoder_module.laneQuantizedScratchValuesCount(dimensions));
        defer allocator.free(quantized_scratch);
        var pool: Scheduler.TestPool = undefined;
        try pool.init(workers_count);
        defer pool.deinit();

        for ([_]usize{ 1, 2, 3, 11, 12, 13, 23, 24, 25, 47, 48, 49, 2119, 2120, 2999, 3000 }) |frames_count| {
            for (0..log_mel.mel_bins_count) |channel| {
                @memcpy(mel[channel * frames_count ..][0..frames_count], actual_features.values[channel * actual_features.frames_count ..][0..frames_count]);
            }
            const positions_count = (frames_count + 1) / 2;
            var context: FrontendComparison = .{
                .features = .{ .values = mel[0 .. log_mel.mel_bins_count * frames_count], .frames_count = frames_count },
                .weights = weights,
                .first = first[0 .. frames_count * width],
                .output = expected[0 .. positions_count * width],
                .float_scratch = float_scratch,
                .quantized_scratch = quantized_scratch,
                .use_reference = true,
            };
            pool.run(runFrontendComparison, .{&context});
            context.output = actual[0 .. positions_count * width];
            context.use_reference = false;
            @memset(actual, std.math.nan(f32));
            @memset(float_scratch, std.math.nan(f32));
            pool.run(runFrontendComparison, .{&context});
            try std.testing.expectEqualSlices(f32, expected[0 .. positions_count * width], context.output);

            if (workers_count == 4 and frames_count == 3000) {
                if (std.testing.environ.getPosix("VOICED_RUNTIME_FRONTEND_BENCHMARK")) |variant| {
                    context.use_reference = std.mem.eql(u8, variant, "reference");
                    const start = std.Io.Clock.awake.now(std.testing.io);
                    for (0..20) |_| pool.run(runFrontendComparison, .{&context});
                    const elapsed = start.durationTo(std.Io.Clock.awake.now(std.testing.io));
                    std.debug.print("\nfrontend {s}: {d:.3} ms/iteration\n", .{ variant, @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms / 20 });
                }
            }
        }
    }
}

const FrontendComparison = struct {
    features: log_mel.Features,
    weights: *const model_module.Weights,
    first: []f32,
    output: []f32,
    float_scratch: []f32,
    quantized_scratch: []u8,
    use_reference: bool,
};

fn runFrontendComparison(raw_context: *anyopaque, lane: Scheduler.Lane) void {
    const context: *FrontendComparison = @ptrCast(@alignCast(raw_context));
    const float_count = context.float_scratch.len / lane.count;
    const quantized_count = context.quantized_scratch.len / lane.count;
    const floats = context.float_scratch[lane.index * float_count ..][0..float_count];
    const quantized = context.quantized_scratch[lane.index * quantized_count ..][0..quantized_count];
    if (context.use_reference) {
        referenceConvolution(context.features.values, context.features.frames_count, log_mel.mel_bins_count, context.weights.encoder_convolution_1_weight, context.weights.encoder_convolution_1_bias, 1, true, context.first, floats, quantized, lane);
        lane.sync();
        referenceConvolution(context.first, context.features.frames_count, context.weights.encoder_convolution_1_weight.scales.len, context.weights.encoder_convolution_2_weight, context.weights.encoder_convolution_2_bias, 2, false, context.output, floats, quantized, lane);
    } else {
        encoder_module.forwardConvolutionFrontend(context.features, context.weights, context.output, floats, quantized, lane);
    }
}

// Deliberately materialize the complete first convolution using one-row
// projections. This independent indexing path is the frontend's exact oracle.
fn referenceConvolution(input: []const f32, input_positions_count: usize, channels_count: usize, weight: vnni_weight.QuantizedWeight, bias: []const f32, stride: usize, channel_major: bool, output: []f32, float_scratch: []f32, quantized_scratch: []u8, lane: Scheduler.Lane) void {
    const depth = channels_count * 3;
    const float_row = float_scratch[0..depth];
    const quantized_row = quantized_scratch[0..depth];
    const positions_range = lane.range(output.len / weight.scales.len);
    for (positions_range.start_index..positions_range.end_index) |position| {
        const origin = @as(isize, @intCast(position * stride)) - 1;
        for (0..channels_count) |channel| {
            for (0..3) |kernel_index| {
                const source = origin + @as(isize, @intCast(kernel_index));
                float_row[channel * 3 + kernel_index] = if (source < 0 or source >= input_positions_count) 0 else input[if (channel_major) channel * input_positions_count + @as(usize, @intCast(source)) else @as(usize, @intCast(source)) * channels_count + channel];
            }
        }
        const scale = linear.quantizeRow(float_row, quantized_row);
        linear.forwardQuantizedOne(quantized_row, scale, weight, bias, .gelu, output[position * weight.scales.len ..][0..weight.scales.len]);
    }
}

// ─── Opt-In Hot-Kernel Benchmarks ──────────────────────────────────────────
//
// VOICED_RUNTIME_KERNEL_BENCHMARK selects attention, vocabulary, or ffn.
// Setup and model loading are outside the timed region. Repeated projections
// reuse one matrix, so these checks supplement rather than replace corpus
// timing, where decoder layers compete for cache capacity.

const attention_module = @import("attention.zig");

test "benchmarks encoder attention" {
    const selected = std.testing.environ.getPosix("VOICED_RUNTIME_KERNEL_BENCHMARK") orelse {
        return error.SkipZigTest;
    };
    if (!std.mem.eql(u8, selected, "attention")) {
        return error.SkipZigTest;
    }
    const allocator = std.testing.allocator;
    const positions_count = 1500;
    const width = 768;
    const storage = try allocator.alloc(f32, attention_module.encoderQueryKeyValueValuesCount(positions_count, width));
    defer allocator.free(storage);
    var random = std.Random.DefaultPrng.init(42);
    for (storage) |*value| value.* = random.random().float(f32) * 4 - 2;
    const output = try allocator.alloc(f32, positions_count * width);
    defer allocator.free(output);
    const scratch = try allocator.alloc(f32, attention_module.encoderScratchValuesCount(4));
    defer allocator.free(scratch);
    var pool: Scheduler.TestPool = undefined;
    try pool.init(4);
    defer pool.deinit();
    var context: AttentionBenchmark = .{ .qkv = attention_module.encoderQueryKeyValue(storage, positions_count, width), .output = output, .scratch = scratch };
    const start = std.Io.Clock.awake.now(std.testing.io);
    pool.run(runAttentionBenchmark, .{&context});
    const elapsed = start.durationTo(std.Io.Clock.awake.now(std.testing.io));
    for (output) |value| try std.testing.expect(std.math.isFinite(value));
    std.debug.print("\nattention: {d:.3} ms/iteration\n", .{@as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms / 100});
}

const AttentionBenchmark = struct {
    qkv: attention_module.EncoderQueryKeyValue,
    output: []f32,
    scratch: []f32,
};

fn runAttentionBenchmark(raw_context: *anyopaque, lane: Scheduler.Lane) void {
    const context: *AttentionBenchmark = @ptrCast(@alignCast(raw_context));
    for (0..100) |_| {
        attention_module.forwardEncoder(context.qkv, 1500, 768, 12, context.output, context.scratch, lane);
        lane.sync();
    }
}

test "benchmarks token projections" {
    const selected = std.testing.environ.getPosix("VOICED_RUNTIME_KERNEL_BENCHMARK") orelse {
        return error.SkipZigTest;
    };
    if (!std.mem.eql(u8, selected, "vocabulary") and !std.mem.eql(u8, selected, "ffn")) {
        return error.SkipZigTest;
    }
    const allocator = std.testing.allocator;
    const test_model = try selectedTestModel();
    var model = try loadTestModel(allocator, test_model);
    defer model.deinit();
    const weights = model.weights;
    const weight = if (std.mem.eql(u8, selected, "vocabulary")) weights.decoder_embeddings_weight else weights.decoder_layers[0].ffn_contraction_weight;
    const input = try allocator.alloc(u8, weight.input_values_count);
    defer allocator.free(input);
    var random = std.Random.DefaultPrng.init(42);
    random.random().bytes(input);
    const output = try allocator.alloc(f32, weight.scales.len);
    defer allocator.free(output);
    var pool: Scheduler.TestPool = undefined;
    try pool.init(4);
    defer pool.deinit();
    var context: ProjectionBenchmark = .{ .input = input, .weight = weight, .output = output, .iterations_count = if (std.mem.eql(u8, selected, "vocabulary")) 5000 else 50_000 };
    const start = std.Io.Clock.awake.now(std.testing.io);
    pool.run(runProjectionBenchmark, .{&context});
    const elapsed = start.durationTo(std.Io.Clock.awake.now(std.testing.io));
    for (output) |value| try std.testing.expect(std.math.isFinite(value));
    std.debug.print("\n{s}: {d:.6} ms/iteration\n", .{ selected, @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms / @as(f64, @floatFromInt(context.iterations_count)) });
}

const ProjectionBenchmark = struct {
    input: []const u8,
    weight: vnni_weight.QuantizedWeight,
    output: []f32,
    iterations_count: usize,
};

fn runProjectionBenchmark(raw_context: *anyopaque, lane: Scheduler.Lane) void {
    const context: *ProjectionBenchmark = @ptrCast(@alignCast(raw_context));
    for (0..context.iterations_count) |_| {
        linear.forwardQuantizedOneParallel(context.input, 17.3, context.weight, &.{}, .none, context.output, lane);
        lane.sync();
    }
}

fn openAudioFixturesDirectory() !std.Io.Dir {
    const cwd = std.Io.Dir.cwd();
    if (std.testing.environ.getPosix("VOICED_RUNTIME_AUDIO_FIXTURES_DIRECTORY")) |path| {
        return cwd.openDir(std.testing.io, path, .{ .iterate = true });
    }

    return cwd.openDir(std.testing.io, "__fixtures__/librispeech/audio", .{ .iterate = true });
}

// ─── Test Configuration ───────────────────────────────────────────────────

fn selectedEncoderTrailingPadding() !inference.EncoderTrailingPadding {
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
    const name = std.testing.environ.getPosix("VOICED_RUNTIME_MODEL") orelse "whisper.base.en";

    if (std.mem.eql(u8, name, "whisper.base.en")) {
        return .{ .name = name, .kind = .whisper_base_en };
    }
    if (std.mem.eql(u8, name, "whisper.small.en")) {
        return .{ .name = name, .kind = .whisper_small_en };
    }
    if (std.mem.eql(u8, name, "whisper.medium.en")) {
        return .{ .name = name, .kind = .whisper_medium_en };
    }

    return error.UnsupportedTestModel;
}

// ─── Installed Model Location ──────────────────────────────────────────────

fn modelDirectoryPath(allocator: std.mem.Allocator) ![]u8 {
    const env = std.testing.environ;
    const data_home = if (env.getPosix("XDG_DATA_HOME")) |path|
        path
    else
        env.getPosix("HOME") orelse return error.HomeNotSet;
    const relative_directory = if (env.getPosix("XDG_DATA_HOME") != null)
        "voiced/models"
    else
        ".local/share/voiced/models";

    return std.fs.path.join(allocator, &.{ data_home, relative_directory });
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
