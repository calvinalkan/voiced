//! WorkerPool owns persistent threads and a model-independent shared scheduler.
//! Its caller-owned memory and address remain stable until deinit joins workers.
const WorkerPool = @This();
const std = @import("std");
const abi = @import("abi.zig");

pub const memory_alignment: usize = abi.memory_alignment;
scheduler: *abi.Scheduler,
workers: []std.Thread,

pub const Config = struct {
    /// Positive number of persistent OS threads, excluding callers.
    /// Capacity is independent of the machine's CPU count.
    /// Excess threads can increase contention and scheduling overhead;
    /// a larger pool does not necessarily finish work sooner.
    workers_count: usize,
};
pub const InitError = error{ InvalidConfig, MemorySizeOverflow, MemoryTooSmall, ThreadSpawnFailed };

/// Includes scheduler state and handles, excluding OS thread stacks.
pub fn requiredMemory(config: Config) InitError!usize {
    var scheduler_size: usize = undefined;
    try checkInit(abi.voiced_inference_scheduler_memory(config.workers_count, &scheduler_size));

    const workers_size = std.math.mul(usize, config.workers_count, @sizeOf(std.Thread)) catch {
        return error.MemorySizeOverflow;
    };

    const workers_offset = std.math.add(usize, schedulerOffset(), scheduler_size) catch {
        return error.MemorySizeOverflow;
    };

    return std.math.add(usize, workers_offset, workers_size) catch error.MemorySizeOverflow;
}

/// Starts every configured worker; failure stops and joins any started prefix.
/// Pool control and thread handles live in memory; the OS allocates thread
/// stacks here, before any transcription. The pool borrows no model or runtime.
pub fn init(memory: []align(memory_alignment) u8, config: Config) InitError!*WorkerPool {
    const size = try requiredMemory(config);
    if (memory.len < size) {
        return error.MemoryTooSmall;
    }

    const pool: *WorkerPool = @ptrCast(memory.ptr);
    const workers_offset = size - config.workers_count * @sizeOf(std.Thread);
    const workers: [*]std.Thread = @ptrCast(@alignCast(memory.ptr + workers_offset));

    pool.* = .{ .scheduler = @ptrCast(memory.ptr + schedulerOffset()), .workers = workers[0..config.workers_count] };

    try checkInit(abi.voiced_inference_scheduler_init(pool.scheduler, workers_offset - schedulerOffset(), config.workers_count));

    var started: usize = 0;
    errdefer {
        abi.voiced_inference_scheduler_stop(pool.scheduler);

        for (pool.workers[0..started]) |thread| {
            thread.join();
        }
    }

    // Keep spawn/join on the application side of the object boundary. Zig's
    // executable startup initializes the stdlib TLS layout used by spawn;
    // an independently compiled object's private copy is uninitialized.
    // Moving spawn into that object produced an invalid optimized init body.
    while (started < pool.workers.len) : (started += 1) {
        pool.workers[started] = std.Thread.spawn(.{}, workerMain, .{ pool.scheduler, started }) catch {
            return error.ThreadSpawnFailed;
        };
    }

    return pool;
}

/// Destroy bound runtimes before this pool. All runtime calls must have returned. The caller releases
/// this block only after deinit returns; it does not free the block itself.
pub fn deinit(pool: *WorkerPool) void {
    abi.voiced_inference_scheduler_stop(pool.scheduler);

    for (pool.workers) |thread| {
        thread.join();
    }

    pool.* = undefined;
}

pub fn workersCount(pool: *const WorkerPool) usize {
    return pool.workers.len;
}

fn workerMain(scheduler: *abi.Scheduler, index: usize) void {
    abi.voiced_inference_scheduler_worker(scheduler, index);
}

fn schedulerOffset() usize {
    return std.mem.alignForward(usize, @sizeOf(WorkerPool), memory_alignment);
}

fn checkInit(status: abi.InitStatus) InitError!void {
    switch (status) {
        .ok => {},
        .invalid_config => return error.InvalidConfig,
        .memory_size_overflow => return error.MemorySizeOverflow,
        .memory_too_small => return error.MemoryTooSmall,
        else => unreachable,
    }
}
