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
    // The test owns both the tensor arena and address-stable runtime. Neither
    // runtime initialization nor transcription allocates tensor/control storage.

    const workers_count = if (std.testing.environ.getPosix("VOICED_RUNTIME_WORKERS")) |value| try std.fmt.parseInt(usize, value, 10) else 4;
    const decoder_workers_count = if (std.testing.environ.getPosix("VOICED_RUNTIME_DECODER_WORKERS")) |value| try std.fmt.parseInt(usize, value, 10) else null;
    const policy: voiced_runtime_module.Policy = .{ .workers_count = workers_count, .decoder_workers_count = decoder_workers_count };
    const runtime_memory_size = try voiced_runtime_module.Runtime.requiredMemorySize(model.kind, policy);

    const runtime_memory = try allocator.alignedAlloc(u8, .fromByteUnits(voiced_runtime_module.runtime_memory_alignment), runtime_memory_size);
    defer allocator.free(runtime_memory);

    var runtime: voiced_runtime_module.Runtime = undefined;
    try runtime.init(
        std.testing.io,
        &model,
        vocabulary_text,
        runtime_memory,
        policy,
    );
    defer runtime.deinit();
    if (std.testing.environ.getPosix("VOICED_RUNTIME_WORKER_CPUS")) |cpu_list| {
        var binding: WorkerCpuBinding = .{};
        var ids = std.mem.splitScalar(u8, cpu_list, ',');
        var count: usize = 0;
        while (ids.next()) |id| {
            if (count == workers_count) {
                return error.InvalidWorkerCpus;
            }
            binding.cpu_ids[count] = try std.fmt.parseInt(usize, id, 10);
            count += 1;
        }
        if (count != workers_count) {
            return error.InvalidWorkerCpus;
        }
        runtime.executor.run(&binding, bindWorkerCpu);
        try std.testing.expect(!binding.failed.load(.acquire));
    }

    // ── Discover Fixtures ──
    //
    // The directory is the test index. An optional exact fixture name keeps development runs deterministic without a manifest.

    const audio_fixtures = try openAudioFixturesDirectory();
    defer audio_fixtures.close(std.testing.io);

    const selected_fixture = std.testing.environ.getPosix("VOICED_RUNTIME_AUDIO_FIXTURE");
    const encoder_trailing_padding = try selectedEncoderTrailingPadding();
    var timings: voiced_runtime_module.Timings = .{};
    const transcribe_options: voiced_runtime_module.TranscribeOptions = .{ .encoder_trailing_padding = encoder_trailing_padding, .timings = &timings };
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

const WorkerCpuBinding = struct {
    cpu_ids: [32]usize = @splat(0),
    failed: std.atomic.Value(bool) = .init(false),
};

fn bindWorkerCpu(raw_context: *anyopaque, lane: executor_module.Lane) void {
    const binding: *WorkerCpuBinding = @ptrCast(@alignCast(raw_context));
    const cpu = binding.cpu_ids[lane.index];
    var mask: std.os.linux.cpu_set_t = @splat(0);
    if (cpu >= @bitSizeOf(@TypeOf(mask))) {
        binding.failed.store(true, .release);
        return;
    }
    mask[cpu / @bitSizeOf(usize)] = @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    std.os.linux.sched_setaffinity(0, &mask) catch binding.failed.store(true, .release);
}

// ─── Convolution Frontend Equivalence ──────────────────────────────────────

const encoder_module = @import("encoder.zig");
const linear = @import("linear.zig");
const log_mel = @import("log_mel.zig");
const executor_module = @import("executor.zig");
const model_module = @import("model.zig");

test "blocked convolution frontend matches materialized reference" {
    const allocator = std.testing.allocator;
    const test_model = try selectedTestModel();
    const model_directory = try modelsDirectory(allocator, test_model.installed_directory_name);
    defer allocator.free(model_directory);
    const model_path = try std.fs.path.join(allocator, &.{ model_directory, "model.bin" });
    defer allocator.free(model_path);
    const pristine = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, model_path, allocator, .limited(model_file_size_max));
    defer allocator.free(pristine);
    var model = try model_module.Model.fromPristineWeights(allocator, test_model.kind, pristine);
    defer model.deinit();
    const weights = model.inferenceWeights();
    const specification = model.kind.specification();
    const width = specification.encoder_width;

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
    const mel_memory = try allocator.alignedAlloc(u8, .fromByteUnits(voiced_runtime_module.runtime_memory_alignment), log_mel.Extractor.requiredMemorySize(log_mel.samples_count_max));
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
    const expected = try allocator.alloc(f32, specification.encoder_positions_count_max * width);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, expected.len);
    defer allocator.free(actual);

    for ([_]usize{ 1, 3, 4, 8, 16 }) |workers_count| {
        const float_scratch = try allocator.alloc(f32, workers_count * encoder_module.laneFloatScratchValuesCount(specification));
        defer allocator.free(float_scratch);
        const quantized_scratch = try allocator.alloc(u8, workers_count * encoder_module.laneQuantizedScratchValuesCount(specification));
        defer allocator.free(quantized_scratch);
        var executor: executor_module.Executor = undefined;
        try executor.init(std.testing.io, workers_count);
        defer executor.deinit();

        for ([_]usize{ 1, 2, 3, 11, 12, 13, 23, 24, 25, 47, 48, 49, 2119, 2120, 2999, 3000 }) |frames_count| {
            for (0..log_mel.mel_bins_count) |channel| {
                @memcpy(mel[channel * frames_count ..][0..frames_count], actual_features.values[channel * actual_features.frames_count ..][0..frames_count]);
            }
            const positions_count = (frames_count + 1) / 2;
            var context: FrontendComparison = .{
                .features = .{ .values = mel[0 .. log_mel.mel_bins_count * frames_count], .frames_count = frames_count },
                .weights = &weights,
                .first = first[0 .. frames_count * width],
                .output = expected[0 .. positions_count * width],
                .float_scratch = float_scratch,
                .quantized_scratch = quantized_scratch,
                .use_reference = true,
            };
            executor.run(&context, runFrontendComparison);
            context.output = actual[0 .. positions_count * width];
            context.use_reference = false;
            @memset(actual, std.math.nan(f32));
            @memset(float_scratch, std.math.nan(f32));
            executor.run(&context, runFrontendComparison);
            try std.testing.expectEqualSlices(f32, expected[0 .. positions_count * width], context.output);

            if (workers_count == 4 and frames_count == 3000) {
                if (std.testing.environ.getPosix("VOICED_RUNTIME_FRONTEND_BENCHMARK")) |variant| {
                    context.use_reference = std.mem.eql(u8, variant, "reference");
                    const start = std.Io.Clock.awake.now(std.testing.io);
                    for (0..20) |_| executor.run(&context, runFrontendComparison);
                    const elapsed = start.durationTo(std.Io.Clock.awake.now(std.testing.io));
                    std.debug.print("\nfrontend {s}: {d:.3} ms/iteration\n", .{ variant, @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms / 20 });
                }
            }
        }
    }
}

const FrontendComparison = struct {
    features: log_mel.Features,
    weights: *const model_module.InferenceWeights,
    first: []f32,
    output: []f32,
    float_scratch: []f32,
    quantized_scratch: []u8,
    use_reference: bool,
};

fn runFrontendComparison(raw_context: *anyopaque, lane: executor_module.Lane) void {
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
fn referenceConvolution(input: []const f32, input_positions_count: usize, channels_count: usize, weight: model_module.QuantizedWeight, bias: []const f32, stride: usize, channel_major: bool, output: []f32, float_scratch: []f32, quantized_scratch: []u8, lane: executor_module.Lane) void {
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
    var executor: executor_module.Executor = undefined;
    try executor.init(std.testing.io, 4);
    defer executor.deinit();
    var context: AttentionBenchmark = .{ .qkv = attention_module.encoderQueryKeyValue(storage, positions_count, width), .output = output, .scratch = scratch };
    const start = std.Io.Clock.awake.now(std.testing.io);
    executor.run(&context, runAttentionBenchmark);
    const elapsed = start.durationTo(std.Io.Clock.awake.now(std.testing.io));
    for (output) |value| try std.testing.expect(std.math.isFinite(value));
    std.debug.print("\nattention: {d:.3} ms/iteration\n", .{@as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms / 100});
}

const AttentionBenchmark = struct {
    qkv: attention_module.EncoderQueryKeyValue,
    output: []f32,
    scratch: []f32,
};

fn runAttentionBenchmark(raw_context: *anyopaque, lane: executor_module.Lane) void {
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
    const model_directory = try modelsDirectory(allocator, test_model.installed_directory_name);
    defer allocator.free(model_directory);
    const model_path = try std.fs.path.join(allocator, &.{ model_directory, "model.bin" });
    defer allocator.free(model_path);
    const pristine = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, model_path, allocator, .limited(model_file_size_max));
    defer allocator.free(pristine);
    var model = try model_module.Model.fromPristineWeights(allocator, test_model.kind, pristine);
    defer model.deinit();
    const weights = model.inferenceWeights();
    const weight = if (std.mem.eql(u8, selected, "vocabulary")) weights.decoder_embeddings_weight else weights.decoder_layers[0].ffn_contraction_weight;
    const input = try allocator.alloc(u8, weight.input_values_count);
    defer allocator.free(input);
    var random = std.Random.DefaultPrng.init(42);
    random.random().bytes(input);
    const output = try allocator.alloc(f32, weight.scales.len);
    defer allocator.free(output);
    var executor: executor_module.Executor = undefined;
    try executor.init(std.testing.io, 4);
    defer executor.deinit();
    var context: ProjectionBenchmark = .{ .input = input, .weight = weight, .output = output, .iterations_count = if (std.mem.eql(u8, selected, "vocabulary")) 5000 else 50_000 };
    const start = std.Io.Clock.awake.now(std.testing.io);
    executor.run(&context, runProjectionBenchmark);
    const elapsed = start.durationTo(std.Io.Clock.awake.now(std.testing.io));
    for (output) |value| try std.testing.expect(std.math.isFinite(value));
    std.debug.print("\n{s}: {d:.6} ms/iteration\n", .{ selected, @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms / @as(f64, @floatFromInt(context.iterations_count)) });
}

const ProjectionBenchmark = struct {
    input: []const u8,
    weight: model_module.QuantizedWeight,
    output: []f32,
    iterations_count: usize,
};

fn runProjectionBenchmark(raw_context: *anyopaque, lane: executor_module.Lane) void {
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
