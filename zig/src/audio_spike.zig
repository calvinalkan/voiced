//! Exercises the production audio process boundary and writes its ordered
//! published samples to a WAV. Process creation, worker protocol, shared-memory
//! ownership, deadlines, stop/cancel transport, and child containment belong to
//! `audio_process.zig`; this file owns only arguments and experiment output.

const std = @import("std");
const audio_process = @import("audio_process.zig");
const assert = std.debug.assert;
const stderr = std.debug.print;

const Io = std.Io;
const Allocator = std.mem.Allocator;
const AudioProcessReport = audio_process.CaptureReport;

const target_name_capacity = 256;
const wav_header_size: u32 = 44;

const RequestedControl = struct {
    command: audio_process.ControlCommand,
    after_ms: u32,
};

const Arguments = struct {
    source: audio_process.Source,
    output_path: []const u8,
    recording_seconds: u8,
    slot_seconds: u8,
    consumer_delay_ms: u32,
    requested_control: ?RequestedControl,
    process_realtime: bool,
};

const ConsumptionSummary = struct {
    samples_count: usize,
    next_publication_ordinal: u32,
    pending_slot_index: ?u8,
    slots_consumed_before_final_report: u32,
};

pub fn main(init: std.process.Init) !void {
    var process_arguments = try std.process.Args.Iterator.initAllocator(
        init.minimal.args,
        init.gpa,
    );
    defer process_arguments.deinit();
    assert(process_arguments.skip());

    const first_argument = process_arguments.next();
    const arguments = parseArguments(first_argument, &process_arguments) catch |err| {
        stderr(
            "usage: audio-spike --output <wav-path> " ++
                "[--target <pipewire-node-name> | --device-serial <serial>] " ++
                "[--seconds <1-90>] " ++
                "[--stop-after-ms <0-90000> | --cancel-after-ms <0-90000>] " ++
                "[--slot-seconds <1-30>] [--consumer-delay-ms <0-10000>] " ++
                "[--main-loop]\n",
            .{},
        );
        return err;
    };
    const requested_source: []const u8 = switch (arguments.source) {
        .default => "default source",
        .node_name, .device_serial => |source| source,
    };

    // Remove stale experiment output before microphone setup. A setup timeout,
    // worker crash, or cancellation must never leave an older WAV looking like
    // the result of this invocation.
    Io.Dir.cwd().deleteFile(init.io, arguments.output_path) catch |delete_error| switch (delete_error) {
        error.FileNotFound => {},
        else => return delete_error,
    };

    const process = try audio_process.start(init, .{
        .session_id = 1,
        .source = arguments.source,
        .recording_samples_target = @as(u32, arguments.recording_seconds) * audio_process.sample_rate_hz,
        .slot_samples_boundary = @as(u32, arguments.slot_seconds) * audio_process.sample_rate_hz,
        .consumer_delay_ms = arguments.consumer_delay_ms,
        .process_realtime = arguments.process_realtime,
    });
    defer audio_process.killAndReap(process) catch |cleanup_error| {
        stderr("Failed to clean up audio process: {s}\n", .{@errorName(cleanup_error)});
    };

    const started_ns = monotonicNanoseconds();
    var control_was_requested = false;
    const result = while (true) {
        if (arguments.requested_control) |requested_control| {
            const request_at_ns = started_ns +
                @as(u64, requested_control.after_ms) * std.time.ns_per_ms;
            if (!control_was_requested and monotonicNanoseconds() >= request_at_ns) {
                switch (requested_control.command) {
                    .stop => try audio_process.requestStop(process),
                    .cancel => try audio_process.requestCancel(process),
                }
                control_was_requested = true;
            }
        }

        if (try audio_process.receiveReport(process, 10)) |completed| {
            break completed;
        }
    };
    const capture = switch (result) {
        .setup_failed => |failure| {
            stderr(
                "Audio worker setup error\n" ++
                    "Stage: {s}\n" ++
                    "Error: domain={s}, code={d}\n" ++
                    "PipeWire library: {s}\n" ++
                    "Target: {s}\n" ++
                    "Processing: {s}\n" ++
                    "Detail: {s}\n",
                .{
                    @tagName(failure.stage),
                    @tagName(failure.domain),
                    failure.code,
                    failure.pipewire_version[0..failure.pipewire_version_size],
                    requested_source,
                    if (arguments.process_realtime) "realtime" else "main loop",
                    failure.message[0..failure.message_size],
                },
            );
            return error.AudioWorkerFailed;
        },
        .captured => |completed| completed,
    };
    const process_report = &capture.report;

    if (process_report.end != .cancelled) {
        try writePcm16Wav(
            init.io,
            init.gpa,
            arguments.output_path,
            capture.samples,
        );
    }

    const consumer: ConsumptionSummary = .{
        .samples_count = capture.samples.len,
        .next_publication_ordinal = process_report.slot_publications_count,
        .pending_slot_index = null,
        .slots_consumed_before_final_report = capture.slots_consumed_before_final_report,
    };
    report(&arguments, process_report, &consumer);
}

fn monotonicNanoseconds() u64 {
    var timestamp: std.os.linux.timespec = undefined;
    const result = std.os.linux.clock_gettime(.MONOTONIC, &timestamp);
    assert(std.os.linux.errno(result) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);
    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

fn parseArguments(
    first_argument: ?[:0]const u8,
    process_arguments: *std.process.Args.Iterator,
) !Arguments {
    var source: audio_process.Source = .default;
    var output_path: ?[:0]const u8 = null;
    var recording_seconds: ?u8 = null;
    var slot_seconds: ?u8 = null;
    var consumer_delay_ms: ?u32 = null;
    var requested_control: ?RequestedControl = null;
    var process_realtime = true;
    var main_loop_was_requested = false;

    var option = first_argument;
    while (option) |option_text| : (option = process_arguments.next()) {
        if (std.mem.eql(u8, option_text, "--target")) {
            if (source != .default) return error.InvalidArguments;
            source = .{ .node_name = process_arguments.next() orelse
                return error.InvalidArguments };
        } else if (std.mem.eql(u8, option_text, "--device-serial")) {
            if (source != .default) return error.InvalidArguments;
            source = .{ .device_serial = process_arguments.next() orelse
                return error.InvalidArguments };
        } else if (std.mem.eql(u8, option_text, "--output")) {
            if (output_path != null) return error.InvalidArguments;
            output_path = process_arguments.next() orelse return error.InvalidArguments;
        } else if (std.mem.eql(u8, option_text, "--seconds")) {
            if (recording_seconds != null) return error.InvalidArguments;
            const seconds_text = process_arguments.next() orelse {
                return error.InvalidArguments;
            };
            recording_seconds = std.fmt.parseInt(u8, seconds_text, 10) catch {
                return error.InvalidArguments;
            };
            if (recording_seconds.? == 0 or recording_seconds.? > 90) {
                return error.InvalidArguments;
            }
        } else if (std.mem.eql(u8, option_text, "--stop-after-ms") or
            std.mem.eql(u8, option_text, "--cancel-after-ms"))
        {
            if (requested_control != null) return error.InvalidArguments;
            const delay_text = process_arguments.next() orelse {
                return error.InvalidArguments;
            };
            const after_ms = std.fmt.parseInt(u32, delay_text, 10) catch {
                return error.InvalidArguments;
            };
            if (after_ms > 90_000) return error.InvalidArguments;
            requested_control = .{
                .command = if (std.mem.eql(u8, option_text, "--stop-after-ms"))
                    .stop
                else
                    .cancel,
                .after_ms = after_ms,
            };
        } else if (std.mem.eql(u8, option_text, "--slot-seconds")) {
            if (slot_seconds != null) return error.InvalidArguments;
            const seconds_text = process_arguments.next() orelse {
                return error.InvalidArguments;
            };
            slot_seconds = std.fmt.parseInt(u8, seconds_text, 10) catch {
                return error.InvalidArguments;
            };
            if (slot_seconds.? == 0 or
                slot_seconds.? > audio_process.slot_duration_seconds_max)
            {
                return error.InvalidArguments;
            }
        } else if (std.mem.eql(u8, option_text, "--consumer-delay-ms")) {
            if (consumer_delay_ms != null) return error.InvalidArguments;
            const delay_text = process_arguments.next() orelse {
                return error.InvalidArguments;
            };
            consumer_delay_ms = std.fmt.parseInt(u32, delay_text, 10) catch {
                return error.InvalidArguments;
            };
            if (consumer_delay_ms.? > 10_000) return error.InvalidArguments;
        } else if (std.mem.eql(u8, option_text, "--main-loop")) {
            if (main_loop_was_requested) return error.InvalidArguments;
            main_loop_was_requested = true;
            process_realtime = false;
        } else {
            return error.InvalidArguments;
        }
    }

    const output_path_required = output_path orelse return error.InvalidArguments;
    if (output_path_required.len == 0) return error.InvalidArguments;
    switch (source) {
        .default => {},
        .node_name, .device_serial => |source_text| {
            if (source_text.len == 0 or source_text.len >= target_name_capacity) {
                return error.InvalidArguments;
            }
        },
    }

    const arguments: Arguments = .{
        .source = source,
        .output_path = output_path_required,
        .recording_seconds = recording_seconds orelse 5,
        .slot_seconds = slot_seconds orelse audio_process.slot_duration_seconds_max,
        .consumer_delay_ms = consumer_delay_ms orelse 0,
        .requested_control = requested_control,
        .process_realtime = process_realtime,
    };
    assert(arguments.output_path.len > 0);
    assert(arguments.recording_seconds > 0);
    assert(arguments.recording_seconds <= 90);
    assert(arguments.slot_seconds > 0);
    assert(arguments.slot_seconds <= audio_process.slot_duration_seconds_max);
    assert(arguments.consumer_delay_ms <= 10_000);
    if (arguments.requested_control) |control| {
        assert(control.after_ms <= 90_000);
    }
    switch (arguments.source) {
        .default => {},
        .node_name, .device_serial => |source_text| {
            assert(source_text.len > 0);
            assert(source_text.len < target_name_capacity);
        },
    }
    return arguments;
}

fn writePcm16Wav(
    io: Io,
    gpa: Allocator,
    output_path: []const u8,
    samples: []const f32,
) !void {
    assert(output_path.len > 0);
    assert(samples.len <=
        90 * audio_process.sample_rate_hz + audio_process.callback_samples_count_max);

    const sample_data_size = samples.len * @sizeOf(i16);
    const wav_size = wav_header_size + sample_data_size;
    assert(wav_size <= std.math.maxInt(u32));

    const wav = try gpa.alloc(u8, wav_size);
    defer gpa.free(wav);

    @memcpy(wav[0..4], "RIFF");
    std.mem.writeInt(u32, wav[4..8], @intCast(wav_size - 8), .little);
    @memcpy(wav[8..12], "WAVE");
    @memcpy(wav[12..16], "fmt ");
    std.mem.writeInt(u32, wav[16..20], 16, .little);
    std.mem.writeInt(u16, wav[20..22], 1, .little);
    std.mem.writeInt(u16, wav[22..24], 1, .little);
    std.mem.writeInt(u32, wav[24..28], audio_process.sample_rate_hz, .little);
    std.mem.writeInt(
        u32,
        wav[28..32],
        audio_process.sample_rate_hz * @sizeOf(i16),
        .little,
    );
    std.mem.writeInt(u16, wav[32..34], @sizeOf(i16), .little);
    std.mem.writeInt(u16, wav[34..36], @bitSizeOf(i16), .little);
    @memcpy(wav[36..40], "data");
    std.mem.writeInt(u32, wav[40..44], @intCast(sample_data_size), .little);

    for (samples, 0..) |sample, sample_index| {
        // The audio worker rejects non-finite values and clamps normalized F32
        // before publication. Assert that producer guarantee again at the final
        // artifact boundary rather than silently repairing an internal defect.
        assert(std.math.isFinite(sample));
        assert(sample >= -1.0);
        assert(sample <= 1.0);
        const scaled_sample = @max(-32768.0, @min(32767.0, sample * 32768.0));
        const pcm_sample: i16 = @intFromFloat(scaled_sample);
        const sample_offset = wav_header_size + sample_index * @sizeOf(i16);
        std.mem.writeInt(i16, wav[sample_offset..][0..@sizeOf(i16)], pcm_sample, .little);
    }

    // Build the complete artifact in an unnamed or randomized sibling file,
    // then replace the requested path with one filesystem rename. Directly
    // truncating the destination would let ENOSPC, cancellation, or a crash
    // leave a plausible RIFF header followed by partial audio. Concurrent spike
    // runs may race, but each observable winner is now one complete WAV rather
    // than interleaved or truncated bytes.
    var atomic_output = try Io.Dir.cwd().createFileAtomic(io, output_path, .{
        .replace = true,
    });
    defer atomic_output.deinit(io);
    try atomic_output.file.writeStreamingAll(io, wav);
    try atomic_output.replace(io);
}

fn schedulerPolicyName(policy: i32) []const u8 {
    return switch (policy) {
        @intFromEnum(std.os.linux.SCHED.Mode.FIFO) => "FIFO",
        @intFromEnum(std.os.linux.SCHED.Mode.RR) => "round-robin",
        @intFromEnum(std.os.linux.SCHED.Mode.NORMAL) => "normal",
        else => "other",
    };
}

fn schedulerPolicyIsRealtime(policy: i32) bool {
    return policy == @intFromEnum(std.os.linux.SCHED.Mode.FIFO) or
        policy == @intFromEnum(std.os.linux.SCHED.Mode.RR);
}

fn linuxErrorNameFromCode(error_code: i32) []const u8 {
    assert(error_code > 0);
    return @tagName(@as(std.os.linux.E, @enumFromInt(error_code)));
}

fn report(
    arguments: *const Arguments,
    capture_report: *const AudioProcessReport,
    consumer: *const ConsumptionSummary,
) void {
    const outcome: audio_process.Outcome = switch (capture_report.end) {
        .completed => .completed,
        .stopped => .stopped,
        .cancelled => .cancelled,
        .failed => |failure| @enumFromInt(@intFromEnum(failure.outcome)),
    };
    const failure: ?audio_process.Failure = switch (capture_report.end) {
        .completed, .stopped, .cancelled => null,
        .failed => |capture_failure| capture_failure.failure,
    };
    const source = capture_report.source;
    const callback = capture_report.callback;
    const process_report = &.{
        .worker_succeeded = @as(u8, 1),
        .shared_memory_is_locked = @as(u8, @intFromBool(capture_report.memory_lock == .locked)),
        .shared_memory_lock_error_code = switch (capture_report.memory_lock) {
            .locked => 0,
            .unavailable => |unavailable| unavailable.error_code,
        },
        .shared_memory_lock_limit_bytes = switch (capture_report.memory_lock) {
            .locked => |limit| limit,
            .unavailable => |unavailable| unavailable.limit_bytes,
        },
        .outcome = @intFromEnum(outcome),
        .error_message = if (failure) |value| value.message else @as([4096]u8, @splat(0)),
        .error_message_size = if (failure) |value| value.message_size else 0,
        .runtime_error_stage = if (failure) |value| @intFromEnum(value.stage) else 0,
        .runtime_error_domain = if (failure) |value| @intFromEnum(value.domain) else 0,
        .runtime_error_code = if (failure) |value| value.code else 0,
        .teardown_error_message = if (capture_report.teardown_failure) |value|
            value.message
        else
            @as([4096]u8, @splat(0)),
        .teardown_error_message_size = if (capture_report.teardown_failure) |value|
            value.message_size
        else
            0,
        .teardown_error_stage = if (capture_report.teardown_failure) |value|
            @intFromEnum(value.stage)
        else
            0,
        .teardown_error_domain = if (capture_report.teardown_failure) |value|
            @intFromEnum(value.domain)
        else
            0,
        .teardown_error_code = if (capture_report.teardown_failure) |value| value.code else 0,
        .timeline_validation = @intFromEnum(capture_report.timeline_validation),
        .pipewire_headers_version = capture_report.pipewire_headers_version,
        .pipewire_headers_version_size = capture_report.pipewire_headers_version_size,
        .pipewire_library_version = capture_report.pipewire_library_version,
        .pipewire_library_version_size = capture_report.pipewire_library_version_size,
        .pipewire_server_version = capture_report.pipewire_server_version,
        .pipewire_server_version_size = capture_report.pipewire_server_version_size,
        .source_is_resolved = @as(u8, @intFromBool(source != null)),
        .source_node_id = if (source) |value| value.node_id else std.math.maxInt(u32),
        .source_node_object_serial = if (source) |value| value.node_object_serial else 0,
        .source_device_id = if (source) |value| value.device_id else std.math.maxInt(u32),
        .source_device_object_serial = if (source) |value| value.device_object_serial else 0,
        .source_node_name = if (source) |value| value.node_name else @as([256]u8, @splat(0)),
        .source_node_name_size = if (source) |value| value.node_name_size else 0,
        .source_node_description = if (source) |value| value.node_description else @as([256]u8, @splat(0)),
        .source_node_description_size = if (source) |value| value.node_description_size else 0,
        .source_device_serial = if (source) |value| value.device_serial else @as([256]u8, @splat(0)),
        .source_device_serial_size = if (source) |value| value.device_serial_size else 0,
        .source_device_description = if (source) |value| value.device_description else @as([256]u8, @splat(0)),
        .source_device_description_size = if (source) |value| value.device_description_size else 0,
        .negotiated_sample_rate_hz = if (capture_report.negotiated_format) |format|
            format.sample_rate_hz
        else
            0,
        .negotiated_channels_count = if (capture_report.negotiated_format) |format|
            format.channels_count
        else
            0,
        .samples_count = capture_report.samples_count,
        .published_samples_count = capture_report.published_samples_count,
        .slot_publications_count = capture_report.slot_publications_count,
        .main_loop_thread_id = capture_report.main_loop_thread_id,
        .callback_thread_id = if (callback) |value| value.thread_id else 0,
        .callback_scheduler_policy = if (callback) |value| value.scheduler_policy orelse -1 else -1,
        .callback_scheduler_priority = if (callback) |value| value.scheduler_priority orelse -1 else -1,
        .callbacks_count = if (callback) |value| value.callbacks_count else 0,
        .missing_buffers_count = if (callback) |value| value.missing_buffers_count else 0,
        .clipped_samples_count = if (callback) |value| value.clipped_samples_count else 0,
        .header_metadata_buffers_count = if (callback) |value| value.header_metadata_buffers_count else 0,
        .header_gap_buffers_count = if (callback) |value| value.header_gap_buffers_count else 0,
        .header_gap_samples_count = if (callback) |value| value.header_gap_samples_count else 0,
        .block_samples_count_min = if (callback) |value|
            if (value.samples_range) |range| range.minimum else 0
        else
            0,
        .block_samples_count_max = if (callback) |value|
            if (value.samples_range) |range| range.maximum else 0
        else
            0,
        .callback_duration_ns_max = if (callback) |value| value.duration_ns_max else 0,
        .callback_gap_ns_max = if (callback) |value| value.gap_ns_max else 0,
    };
    assert(arguments.output_path.len > 0);
    assert(arguments.consumer_delay_ms <= 10_000);
    assert(process_report.worker_succeeded == 1);
    assert(process_report.shared_memory_is_locked <= 1);
    assert(process_report.error_message_size <= process_report.error_message.len);
    assert(process_report.pipewire_headers_version_size > 0);
    assert(process_report.pipewire_headers_version_size <=
        process_report.pipewire_headers_version.len);
    assert(process_report.pipewire_library_version_size > 0);
    assert(process_report.pipewire_library_version_size <=
        process_report.pipewire_library_version.len);
    assert(process_report.pipewire_server_version_size <=
        process_report.pipewire_server_version.len);
    assert(process_report.source_is_resolved <= 1);
    assert(process_report.source_node_name_size <= process_report.source_node_name.len);
    assert(process_report.source_node_description_size <=
        process_report.source_node_description.len);
    assert(process_report.source_device_serial_size <= process_report.source_device_serial.len);
    assert(process_report.source_device_description_size <=
        process_report.source_device_description.len);
    if (process_report.samples_count > 0) {
        assert(process_report.source_is_resolved == 1);
    }
    assert(process_report.published_samples_count <= process_report.samples_count);
    assert(consumer.samples_count == process_report.published_samples_count);
    assert(consumer.next_publication_ordinal == process_report.slot_publications_count);
    assert(consumer.pending_slot_index == null);

    assert(process_report.outcome == @intFromEnum(outcome));
    const timeline_validation: audio_process.TimelineValidation =
        @enumFromInt(process_report.timeline_validation);
    const timeline_validation_name = switch (timeline_validation) {
        .header_only => "header-only",
        .full => "full",
    };
    const pipewire_server_version = if (process_report.pipewire_server_version_size == 0)
        "unknown"
    else
        process_report.pipewire_server_version[0..process_report.pipewire_server_version_size];
    const selection_mode = switch (arguments.source) {
        .default => "current default",
        .node_name => "explicit node name",
        .device_serial => "configured device.serial",
    };
    const target: []const u8 = switch (arguments.source) {
        .default => "default source",
        .node_name, .device_serial => |source_text| source_text,
    };
    const actual_source = if (process_report.source_is_resolved == 0)
        "unresolved"
    else if (process_report.source_node_description_size > 0)
        process_report.source_node_description[0..process_report.source_node_description_size]
    else if (process_report.source_node_name_size > 0)
        process_report.source_node_name[0..process_report.source_node_name_size]
    else
        "unnamed PipeWire source";
    const duration_tenths = process_report.published_samples_count * 10 /
        audio_process.sample_rate_hz;
    const output = if (outcome == .cancelled)
        "not published (cancelled)"
    else
        arguments.output_path;

    stderr(
        "\nAudio exchange spike complete\n" ++
            "Outcome: {s}\n" ++
            "Selection: {s}\n" ++
            "Requested source: {s}\n" ++
            "Actual source: {s}\n" ++
            "Output: {s}\n",
        .{
            @tagName(outcome),
            selection_mode,
            target,
            actual_source,
            output,
        },
    );
    stderr(
        "PipeWire headers/library/server: {s}/{s}/{s}\n" ++
            "Timeline validation: {s}\n" ++
            "Format: float32, {d} Hz, {d} channel\n" ++
            "Published duration: {d}.{d} s\n" ++
            "Samples captured/published: {d}/{d}\n" ++
            "Physical slots: {d}\n" ++
            "Slot publications: {d}\n" ++
            "Consumed before final report: {d}\n" ++
            "Consumer delay: {d} ms\n" ++
            "Shared memory locked: {s}\n" ++
            "Processing: {s}\n" ++
            "Main-loop/callback TID: {d}/{d}\n" ++
            "Callback scheduler: {s}, priority {d}\n" ++
            "Callbacks: {d}\n" ++
            "Missing buffers: {d}\n" ++
            "Clipped samples: {d}\n" ++
            "SPA Header buffers: {d}\n" ++
            "SPA Header gaps: {d} buffers, {d} samples\n" ++
            "Block samples: {d}-{d}\n" ++
            "Longest callback: {d} us\n" ++
            "Largest callback gap: {d} ms\n",
        .{
            process_report.pipewire_headers_version[0..process_report.pipewire_headers_version_size],
            process_report.pipewire_library_version[0..process_report.pipewire_library_version_size],
            pipewire_server_version,
            timeline_validation_name,
            process_report.negotiated_sample_rate_hz,
            process_report.negotiated_channels_count,
            duration_tenths / 10,
            duration_tenths % 10,
            process_report.samples_count,
            process_report.published_samples_count,
            audio_process.slots_count,
            process_report.slot_publications_count,
            consumer.slots_consumed_before_final_report,
            arguments.consumer_delay_ms,
            if (process_report.shared_memory_is_locked == 1) "yes" else "no",
            if (arguments.process_realtime) "realtime" else "main loop",
            process_report.main_loop_thread_id,
            process_report.callback_thread_id,
            schedulerPolicyName(process_report.callback_scheduler_policy),
            process_report.callback_scheduler_priority,
            process_report.callbacks_count,
            process_report.missing_buffers_count,
            process_report.clipped_samples_count,
            process_report.header_metadata_buffers_count,
            process_report.header_gap_buffers_count,
            process_report.header_gap_samples_count,
            process_report.block_samples_count_min,
            process_report.block_samples_count_max,
            process_report.callback_duration_ns_max / std.time.ns_per_us,
            process_report.callback_gap_ns_max / std.time.ns_per_ms,
        },
    );

    if (process_report.source_is_resolved == 1) {
        stderr(
            "Source node: id={d}, object_serial={d}, name={s}\n",
            .{
                process_report.source_node_id,
                process_report.source_node_object_serial,
                if (process_report.source_node_name_size == 0)
                    "not provided"
                else
                    process_report.source_node_name[0..process_report.source_node_name_size],
            },
        );
        if (process_report.source_device_id == std.math.maxInt(u32)) {
            stderr("Source device: not provided by PipeWire\n", .{});
        } else {
            stderr(
                "Source device: id={d}, object_serial={d}, serial={s}, description={s}\n",
                .{
                    process_report.source_device_id,
                    process_report.source_device_object_serial,
                    if (process_report.source_device_serial_size == 0)
                        "not provided"
                    else
                        process_report.source_device_serial[0..process_report.source_device_serial_size],
                    if (process_report.source_device_description_size == 0)
                        "not provided"
                    else
                        process_report.source_device_description[0..process_report.source_device_description_size],
                },
            );
        }
    }

    if (timeline_validation == .header_only) {
        stderr(
            "Warning: this PipeWire build cannot detect every dropped audio interval\n",
            .{},
        );
    }

    if (arguments.process_realtime and process_report.callbacks_count > 0 and
        (!schedulerPolicyIsRealtime(process_report.callback_scheduler_policy) or
            process_report.callback_scheduler_priority <= 0))
    {
        stderr(
            "Warning: realtime processing was requested, but the callback " ++
                "scheduler is {s} at priority {d}; capture remains protected " ++
                "by progress deadlines and timeline validation\n",
            .{
                schedulerPolicyName(process_report.callback_scheduler_policy),
                process_report.callback_scheduler_priority,
            },
        );
    }

    if (process_report.shared_memory_is_locked == 0) {
        assert(process_report.shared_memory_lock_error_code > 0);
        stderr(
            "Warning: mlock could not lock the {d}-byte audio exchange: " ++
                "errno={s} ({d}), RLIMIT_MEMLOCK={d} bytes; prefaulted pages " ++
                "remain usable but may be reclaimed under memory pressure\n",
            .{
                audio_process.audio_exchange_size,
                linuxErrorNameFromCode(process_report.shared_memory_lock_error_code),
                process_report.shared_memory_lock_error_code,
                process_report.shared_memory_lock_limit_bytes,
            },
        );
    }

    if (process_report.error_message_size > 0) {
        const error_stage: audio_process.RuntimeErrorStage =
            @enumFromInt(process_report.runtime_error_stage);
        const error_domain: audio_process.ErrorDomain =
            @enumFromInt(process_report.runtime_error_domain);
        stderr(
            "Audio error: stage={s}, domain={s}, code={d}\n" ++
                "Audio detail: {s}\n",
            .{
                @tagName(error_stage),
                @tagName(error_domain),
                process_report.runtime_error_code,
                process_report.error_message[0..process_report.error_message_size],
            },
        );
    }

    if (process_report.teardown_error_message_size > 0) {
        const error_stage: audio_process.RuntimeErrorStage =
            @enumFromInt(process_report.teardown_error_stage);
        const error_domain: audio_process.ErrorDomain =
            @enumFromInt(process_report.teardown_error_domain);
        stderr(
            "Audio teardown error: stage={s}, domain={s}, code={d}\n" ++
                "Audio teardown detail: {s}\n",
            .{
                @tagName(error_stage),
                @tagName(error_domain),
                process_report.teardown_error_code,
                process_report.teardown_error_message[0..process_report.teardown_error_message_size],
            },
        );
    }
}
