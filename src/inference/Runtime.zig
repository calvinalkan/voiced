//! `Runtime` binds immutable model views and a shared worker pool to one workspace.
//! Callers allocate its complete permanent memory, initialize it once at that
//! address, perform serial transcriptions, and deinitialize it after the final
//! transcription.

const std = @import("std");
const abi = @import("abi.zig");
const Model = @import("Model.zig");
const WorkerPool = @import("WorkerPool.zig");
const TranscriptionResult = @import("TranscriptionResult.zig").TranscriptionResult;

/// Opaque view of the runtime stored directly in caller memory. There is no
/// separately stored facade or pointer header.
pub const Runtime = opaque {
    pub const memory_alignment: usize = abi.memory_alignment;

    pub const EncoderPadding = abi.Padding;
    pub const Config = abi.Config;

    pub const InitError = error{
        InvalidConfig,
        InvalidModel,
        MemorySizeOverflow,
        MemoryTooSmall,
    };

    /// `requiredMemory` returns the permanent caller-owned memory required by
    /// `init`, including its control state and transcript storage. Model
    /// storage, pool storage, and the pool's OS thread stacks are separate. The
    /// bound pool capacity sizes per-lane scratch; request widths do not resize it.
    pub fn requiredMemory(model: *const Model, pool: *const WorkerPool, config: Config) InitError!usize {
        var size: usize = undefined;
        try checkInit(abi.voiced_inference_runtime_memory(model, pool.scheduler, &config, &size));

        return size;
    }

    /// Constructs the runtime inside memory without allocating or spawning threads.
    /// The caller keeps the immutable model, pool, and memory alive at stable
    /// addresses until deinit returns; model metadata is borrowed with its tensors.
    /// On error no resources remain owned; do not call deinit for a failed init.
    pub fn init(memory: []align(memory_alignment) u8, model: *const Model, pool: *WorkerPool, config: Config) InitError!*Runtime {
        try checkInit(abi.voiced_inference_runtime_init(model, memory.ptr, memory.len, pool.scheduler, &config));

        return @ptrCast(memory.ptr);
    }

    /// No transcription may be active. Leaves the borrowed pool and model intact.
    pub fn deinit(runtime: *Runtime) void {
        // All object work joins before transcribe returns. No OS resources need
        // releasing here; deinit ends the borrow before the caller reuses memory.
        _ = runtime;
    }

    fn checkInit(status: abi.InitStatus) InitError!void {
        switch (status) {
            .ok => {},
            .invalid_config => return error.InvalidConfig,
            .invalid_model => return error.InvalidModel,
            .memory_size_overflow => return error.MemorySizeOverflow,
            .memory_too_small => return error.MemoryTooSmall,
        }
    }

    pub const TranscriptionOptions = struct {
        /// `encoder_padding` must not exceed `Config.encoder_padding_max`.
        encoder_padding: EncoderPadding = .seconds_30,
        /// Null selects the bound pool's capacity. A limit must be in 1..capacity;
        /// contention may grant fewer workers, with width fixed during each step.
        encoder_workers_count_max: ?usize = null,
        /// Null selects the resolved encoder maximum; an explicit value may be wider.
        decoder_workers_count_max: ?usize = null,
        /// Borrowed until return. Once set, leave it set until this call completes.
        cancellation: ?*const std.atomic.Value(bool) = null,
    };

    /// `transcribe` processes one recording synchronously. Calls on the same
    /// runtime must not overlap. Different runtimes may call concurrently, including
    /// when sharing this pool and model. Feature extraction runs on the caller;
    /// encoder and decoder collectives run on pool workers.
    ///
    /// Success or cancellation joins every worker reference before returning, so
    /// workspace reuse requires no separate wait or reset. Neither execution nor
    /// dispatch allocates memory. Cancellation is cooperative at compute boundaries;
    /// it does not interrupt a running kernel.
    pub fn transcribe(runtime: *Runtime, samples: []const f32, options: TranscriptionOptions) TranscriptionResult {
        const request: abi.Request = .{
            .samples = samples,
            .encoder_padding = options.encoder_padding,
            .encoder_workers_count_max = options.encoder_workers_count_max,
            .decoder_workers_count_max = options.decoder_workers_count_max,
            .cancellation = options.cancellation,
        };

        var result: TranscriptionResult = undefined;
        abi.voiced_inference_runtime_transcribe(runtime, &request, &result);

        return result;
    }
};
