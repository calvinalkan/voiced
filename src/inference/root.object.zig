//! Object build root. Exports the compute implementation; thread creation
//! remains in the application-side WorkerPool, alongside Zig startup state.

const std = @import("std");
const abi = @import("abi.zig");
const Runtime = @import("Runtime.object.zig");
const Scheduler = @import("Scheduler.zig");
const Model = @import("Model.zig");
const TranscriptionResult = @import("TranscriptionResult.zig").TranscriptionResult;

pub const panic = std.debug.FullPanic(panicInApplication);

fn panicInApplication(message: []const u8, address: ?usize) noreturn {
    abi.voiced_inference_panic(message.ptr, message.len, address orelse 0);
}

pub export fn voiced_inference_runtime_memory(model: *const Model, scheduler_storage: *const abi.Scheduler, config: *const abi.Config, size: *usize) abi.InitStatus {
    const scheduler: *const Scheduler = @ptrCast(@alignCast(scheduler_storage));

    size.* = Runtime.requiredMemory(model, scheduler, config.*) catch |err| {
        return runtimeError(err);
    };

    return .ok;
}

pub export fn voiced_inference_runtime_init(
    model: *const Model,
    memory_bytes: [*]u8,
    memory_size: usize,
    scheduler_storage: *abi.Scheduler,
    config: *const abi.Config,
) abi.InitStatus {
    const memory: []align(abi.memory_alignment) u8 = @alignCast(memory_bytes[0..memory_size]);
    const runtime: *Runtime = @ptrCast(memory.ptr);
    const scheduler: *Scheduler = @ptrCast(@alignCast(scheduler_storage));

    runtime.init(model, memory, scheduler, config.*) catch |err| {
        return runtimeError(err);
    };

    return .ok;
}

pub export fn voiced_inference_scheduler_memory(count: usize, size: *usize) abi.InitStatus {
    size.* = Scheduler.requiredMemory(count) catch |err| {
        return schedulerError(err);
    };

    return .ok;
}

pub export fn voiced_inference_scheduler_init(storage: *abi.Scheduler, memory_size: usize, count: usize) abi.InitStatus {
    const scheduler: *Scheduler = @ptrCast(@alignCast(storage));
    const bytes: [*]align(abi.memory_alignment) u8 = @ptrCast(@alignCast(storage));

    scheduler.init(bytes[0..memory_size], count) catch |err| {
        return schedulerError(err);
    };

    return .ok;
}

fn schedulerError(err: Scheduler.InitError) abi.InitStatus {
    return switch (err) {
        error.InvalidConfig => .invalid_config,
        error.MemorySizeOverflow => .memory_size_overflow,
        error.MemoryTooSmall => .memory_too_small,
    };
}

pub export fn voiced_inference_scheduler_worker(storage: *abi.Scheduler, index: usize) void {
    const scheduler: *Scheduler = @ptrCast(@alignCast(storage));

    Scheduler.workerMain(scheduler, index);
}

pub export fn voiced_inference_scheduler_stop(storage: *abi.Scheduler) void {
    const scheduler: *Scheduler = @ptrCast(@alignCast(storage));

    scheduler.requestStop();
}

pub export fn voiced_inference_runtime_transcribe(runtime_storage: *abi.Runtime, request: *const abi.Request, result: *TranscriptionResult) void {
    const runtime: *Runtime = @ptrCast(@alignCast(runtime_storage));

    result.* = runtime.transcribe(request);
}

fn runtimeError(err: Runtime.InitError) abi.InitStatus {
    return switch (err) {
        error.InvalidConfig => .invalid_config,
        error.InvalidModel => .invalid_model,
        error.MemorySizeOverflow => .memory_size_overflow,
        error.MemoryTooSmall => .memory_too_small,
    };
}

comptime {
    std.debug.assert(@alignOf(Runtime) <= abi.memory_alignment);
    std.debug.assert(Runtime.memory_alignment == abi.memory_alignment);
}
