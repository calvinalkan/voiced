const std = @import("std");
const builtin = @import("builtin");
const Model = @import("Model.zig");

// Private ABI: both sides use this source, compiler, and target and are rebuilt
// together. Shared Zig records cross by pointer, never by value: this preserves
// slices and optionals without a second encoding. Their layouts and model views
// do not constitute a versioned external C ABI.

// The executable owns initialized crash-reporting state as well as thread
// startup. Route object panics to its existing panic policy; do not instantiate
// a second stdlib crash handler with private, uninitialized process globals.
pub extern fn voiced_inference_panic(message: [*]const u8, size: usize, address: usize) noreturn;

comptime {
    if (builtin.output_mode == .Exe) {
        @export(&panicFromObject, .{ .name = "voiced_inference_panic" });
    }
}

fn panicFromObject(message: [*]const u8, size: usize, address: usize) callconv(.c) noreturn {
    std.builtin.panic.call(message[0..size], if (address == 0) null else address);
}

pub const memory_alignment: usize = std.atomic.cache_line;
pub const Runtime = @import("Runtime.zig").Runtime;
pub const Scheduler = opaque {};

pub const InitStatus = enum(u32) {
    ok,
    invalid_config,
    invalid_model,
    memory_size_overflow,
    memory_too_small,
};

pub const Padding = enum(u32) {
    seconds_5,
    seconds_10,
    seconds_30,

    pub fn seconds(padding: Padding) u8 {
        return switch (padding) {
            .seconds_5 => 5,
            .seconds_10 => 10,
            .seconds_30 => 30,
        };
    }
};

pub const Config = struct {
    audio_samples_count_max: usize,
    transcript_tokens_count_max: usize,
    /// Individual transcriptions may select this padding or a smaller value.
    encoder_padding_max: Padding,
};

pub const Request = struct {
    samples: []const f32,
    encoder_padding: Padding,
    encoder_workers_count_max: ?usize,
    decoder_workers_count_max: ?usize,
    cancellation: ?*const std.atomic.Value(bool),
};

// The runtime starts at memory.ptr. Its complete layout, including transcript
// storage, is private to the object. The bound scheduler supplies scratch capacity.
pub extern fn voiced_inference_runtime_memory(model: *const Model, scheduler: *const Scheduler, config: *const Config, size: *usize) InitStatus;
pub extern fn voiced_inference_runtime_init(model: *const Model, memory: [*]u8, memory_size: usize, scheduler: *Scheduler, config: *const Config) InitStatus;
pub extern fn voiced_inference_scheduler_memory(workers_count: usize, size: *usize) InitStatus;
pub extern fn voiced_inference_scheduler_init(scheduler: *Scheduler, memory_size: usize, workers_count: usize) InitStatus;
pub extern fn voiced_inference_scheduler_worker(scheduler: *Scheduler, worker_index: usize) void;
pub extern fn voiced_inference_scheduler_stop(scheduler: *Scheduler) void;
// Every call fills exactly one result variant. The shared type crosses by
// pointer; its text and token slices keep the runtime's ordinary borrow lifetime.
pub extern fn voiced_inference_runtime_transcribe(
    runtime: *Runtime,
    request: *const Request,
    result: *TranscriptionResult,
) void;

const TranscriptionResult = @import("TranscriptionResult.zig").TranscriptionResult;

comptime {
    std.debug.assert(@sizeOf(InitStatus) == @sizeOf(u32));
    std.debug.assert(@sizeOf(Padding) == @sizeOf(u32));
}
