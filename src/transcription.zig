//! A persistent transcription thread owns the resident runtime and its compute
//! pool. Typed jobs borrow sealed audio; completion releases all chunk accesses.
//! Unload joins the pool before freeing the arena and mapped model weights.
const Transcription = @This();

const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.transcription);
const ModelLoadedFields = logging.FieldSet(.{
    .model_load_duration_ms,
});
const ModelRuntimeInitializedFields = logging.FieldSet(.{
    .model_runtime_init_duration_ms,
    .model_runtime_size,
});
const ModelLoaded = log.Event(.model_loaded, ModelLoadedFields);
const ModelRuntimeInitialized = log.Event(.model_runtime_initialized, ModelRuntimeInitializedFields);
const AudioExchange = @import("audio_exchange.zig");
const worker = @import("worker.zig");
const inference = @import("inference/root.zig");
const packed_model = @import("packed_model/root.zig");
const assert = std.debug.assert;
const linux = std.os.linux;

pub const transcript_tokens_count_max: usize = 446;

pub const ModelOptions = struct {
    model: inference.Model.Kind = .whisper_small_en,
    /// Pool size: one persistent OS thread per worker, excluding the caller.
    inference_threads_count: usize = 4,
    /// Decoder limit within that same pool; null uses every worker.
    decoder_threads_count: ?usize = null,
    encoder_trailing_padding: inference.EncoderTrailingPadding = .seconds_10,
};

pub const Job = union(enum) {
    prepare: struct { recording_ordinal: u64, model: ModelOptions },
    transcribe: struct {
        slot_index: AudioExchange.SlotIndex,
    },
    unload,
};

/// Completion releases the audio borrow. Inference output borrows the runtime
/// until the next job; consume its diagnostics and copy text before submitting.
pub const Result = union(enum) {
    ready: struct { model_prepare_duration_ns: u64 },
    model_load_error: ModelLoadError,
    transcription: inference.TranscriptionResult,
    stopped,
};

// Retain the actual loader, allocation, and initialization errors. No formatted
// diagnostic buffer or second transcription error taxonomy crosses the mailbox.
pub const ModelLoadError = @typeInfo(@typeInfo(@TypeOf(packed_model.load)).@"fn".return_type.?).error_union.error_set ||
    std.Io.Dir.OpenError || std.mem.Allocator.Error || inference.WorkerPool.InitError || inference.Runtime.InitError;

pub const Context = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    models_directory_path: []const u8,
};

mailbox: worker.Mailbox(Job, Result),
context: Context,
audio: *AudioExchange,
cancel: std.atomic.Value(bool) = .init(false),

pub fn requestCancellation(self: *Transcription) void {
    self.cancel.store(true, .release);
}

pub fn submit(self: *Transcription, job: Job) void {
    self.cancel.store(false, .release);
    self.mailbox.submit(job);
}

pub fn run(self: *Transcription) void {
    worker.name("voiced-asr");
    defer self.mailbox.finish();

    while (self.mailbox.next()) |job| {
        switch (job) {
            .prepare => |options| {
                // resident returns only after joining compute workers and
                // releasing their arena, vocabulary, and mapped weights.
                resident(self, options) catch |err| {
                    self.mailbox.complete(.{ .model_load_error = err });

                    continue;
                };

                self.mailbox.complete(.stopped);
            },
            .transcribe, .unload => unreachable,
        }
    }
}

fn resident(self: *Transcription, prepare: @FieldType(Job, "prepare")) ModelLoadError!void {
    const context = self.context;
    const allocator = context.allocator;
    const launch = prepare.model;

    if (self.cancel.load(.acquire)) {
        return;
    }

    const model_load_started_ns = monotonicNanoseconds();

    var model = load: {
        var directory = try std.Io.Dir.cwd().openDir(context.io, context.models_directory_path, .{});
        defer directory.close(context.io);

        break :load try packed_model.load(context.io, directory, modelFileName(launch.model), launch.model);
    };
    defer model.deinit();

    if (self.cancel.load(.acquire)) {
        return;
    }

    ModelLoaded.emit(.debug, .{ .recording_id = prepare.recording_ordinal }, .{
        .model_load_duration_ms = logging.float(@as(f64, @floatFromInt(monotonicNanoseconds() - model_load_started_ns)) / std.time.ns_per_ms, 3),
    });

    const runtime_started_ns = monotonicNanoseconds();
    const pool_config: inference.WorkerPool.Config = .{ .workers_count = launch.inference_threads_count };
    const pool_size = try inference.WorkerPool.requiredMemory(pool_config);

    const pool_memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.WorkerPool.memory_alignment), pool_size);
    defer allocator.free(pool_memory);

    const pool = try inference.WorkerPool.init(pool_memory, pool_config);
    defer pool.deinit();

    const config: inference.Runtime.Config = .{
        .audio_samples_count_max = AudioExchange.slot_samples_capacity,
        .transcript_tokens_count_max = transcript_tokens_count_max,
        .encoder_padding_max = .seconds_30,
    };

    const memory_size = try inference.Runtime.requiredMemory(&model, pool, config);

    const memory = try allocator.alignedAlloc(u8, .fromByteUnits(inference.Runtime.memory_alignment), memory_size);
    defer allocator.free(memory);

    const runtime = try inference.Runtime.init(memory, &model, pool, config);
    defer runtime.deinit();

    if (self.cancel.load(.acquire)) {
        return;
    }

    ModelRuntimeInitialized.emit(.debug, .{ .recording_id = prepare.recording_ordinal }, .{
        .model_runtime_init_duration_ms = logging.float(@as(f64, @floatFromInt(monotonicNanoseconds() - runtime_started_ns)) / std.time.ns_per_ms, 3),
        .model_runtime_size = memory_size,
    });

    const model_prepare_duration_ns = monotonicNanoseconds() - model_load_started_ns;

    self.mailbox.complete(.{ .ready = .{
        .model_prepare_duration_ns = model_prepare_duration_ns,
    } });

    // Runtime, model, and pool remain at these addresses until
    // every compute thread joins.
    while (self.mailbox.next()) |job| {
        switch (job) {
            .transcribe => |work| {
                const slot = &self.audio.slots[work.slot_index.arrayIndex()];
                const published = AudioExchange.acquireSlot(slot).?;

                const result = runtime.transcribe(slot.samples[0..published.samples_count], .{
                    .encoder_padding = launch.encoder_trailing_padding,
                    .encoder_workers_count_max = launch.inference_threads_count,
                    .decoder_workers_count_max = launch.decoder_threads_count,
                    .cancellation = &self.cancel,
                });

                self.mailbox.complete(.{ .transcription = result });
            },
            .unload => return,
            .prepare => unreachable,
        }
    }

    unreachable; // Shutdown is submitted only after unload completes.
}

fn modelFileName(kind: inference.Model.Kind) []const u8 {
    return switch (kind) {
        .whisper_base_en => "whisper.base.en.voiced",
        .whisper_small_en => "whisper.small.en.voiced",
        .whisper_medium_en => "whisper.medium.en.voiced",
        .whisper_tiny_en => "whisper.tiny.en.voiced",
    };
}

fn monotonicNanoseconds() u64 {
    var timestamp: linux.timespec = undefined;
    const result = linux.clock_gettime(.MONOTONIC, &timestamp);

    assert(linux.errno(result) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);

    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}
