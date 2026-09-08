//! One native PipeWire capture session. The worker owns graph discovery,
//! mappings, mono conversion and shared-slot publication through one poll loop.
//! Graph processing uses fixed storage and no allocation or blocking I/O.
const std = @import("std");
const logging = @import("../logging.zig");
const linux = std.os.linux;
const assert = std.debug.assert;
const PipeWireClient = @import("pipewire_client.zig");
const audio_resampler = @import("resampler.zig");
const realtime_scheduling = @import("realtime.zig");
const audio_activity = @import("activity.zig");
const AudioExchange = @import("../audio_exchange.zig");
const audio_policy = @import("policy.zig");
const worker = @import("../worker.zig");
pub const runtime_error_message_capacity = 512;
const callback_error_message_capacity = runtime_error_message_capacity;
pub const setup_error_message_capacity = 4096;
pub const pipewire_version_capacity = 64;
pub const source_identity_text_capacity = 256;

/// One supervisor command ends an active recording. `stop` retains every
/// complete callback block, including the final partially filled exchange slot;
/// `cancel` abandons that private final slot and tells the supervisor to discard
/// every earlier publication belonging to the session.
pub const ControlCommand = enum(u32) {
    none,
    stop,
    cancel,
};

pub const Source = union(enum) {
    default,
    node_name: [:0]const u8,
    device_serial: [:0]const u8,
};

pub const Launch = struct {
    recording_id: u64,
    exchange: *AudioExchange,
    publication_event_fd: std.posix.fd_t,
    control_event_fd: std.posix.fd_t,
    control: *const std.atomic.Value(ControlCommand),
    source: Source,
    recording_samples_target: u32,
    environment: Environment,
};

pub const End = enum {
    completed,
    stopped,
    cancelled,
};

pub const AudioError = enum {
    configured_device_not_found,
    configured_device_ambiguous,
    exchange_full,
};

pub const FailureCause = union(enum) {
    zig: anyerror,
    linux: linux.E,
    pipewire: i64,
    audio: AudioError,
};

pub const RuntimeFailure = struct {
    stage: FailureStage,
    cause: FailureCause,
    message: [callback_error_message_capacity]u8,
    message_size: u16,
};

pub const RuntimeError = struct {
    report: Report,
    detail: RuntimeFailure,
};

pub const Error = union(enum) {
    source_not_found: SetupFailure,
    source_ambiguous: SetupFailure,
    setup: SetupFailure,
    source_connection_lost: RuntimeError,
    source_changed: RuntimeError,
    pipeline_full: RuntimeError,
    unexpected_format: RuntimeError,
    timeline_discontinuity: RuntimeError,
    stream_error: RuntimeError,
    stream_disconnected: RuntimeError,
    invalid_buffer: RuntimeError,
    corrupted_buffer: RuntimeError,
};

pub const Success = struct { end: End, report: Report };
pub const Result = union(enum) { ok: Success, err: Error };

pub const MemoryLockResult = union(enum) {
    locked: u64,
    unavailable: struct {
        errno: linux.E,
        limit_bytes: u64,
    },
};

pub const Activity = audio_activity.Detector.Activity;
pub const ActivityReport = audio_activity.Detector.Report;

pub const ReportCallback = struct {
    thread_id: i32,
    scheduler_policy: ?i32,
    scheduler_priority: ?i32,
    callbacks_count: u64,
    missing_buffers_count: u32,
    clipped_samples_count: u32,
    header_metadata_buffers_count: u32,
    header_gap_buffers_count: u32,
    header_gap_samples_count: u32,
    samples_range: ?CallbackSamplesRange,
    duration_ns_max: u64,
    gap_ns_max: u64,
    activity: ActivityReport,
};

/// Stable measurements collected before the stream stops. The surrounding
/// `Result` supplies the terminal outcome or typed error.
pub const NegotiatedFormat = struct {
    sample_rate_hz: u32,
    channels_count: u32,
};

pub const Report = struct {
    client_node_version_advertised: u32,
    client_node_version_selected: u32,
    pipewire_server_version: [pipewire_version_capacity]u8,
    pipewire_server_version_size: u8,

    source_identity: ?SourceIdentity,
    negotiated_format: ?NegotiatedFormat,
    // Accepted callback samples include a cancelled final private slot. The
    // published count includes only samples release-published to consumers; the
    // two counts are equal for every outcome except cancellation.
    samples_count: u32,
    published_samples_count: u32,
    slot_publications_count: u32,
    memory_lock: MemoryLockResult,

    main_loop_thread_id: i32,
    callback: ?ReportCallback,
};

pub const SetupFailureStage = enum {
    source_resolution,
    stream_connect,
};

/// Names the capture operation that observed a runtime error.
pub const FailureStage = enum {
    main_loop,
    stream_state,
    source_identity,
    buffer_data,
    timeline_continuity,
    exchange_publication,
};

/// Identifies the first concrete PipeWire source linked during one recording.
/// Global IDs and object serials diagnose that graph instance; `device_serial`
/// is the stable hardware identity expected to survive unplug and replug.
pub const SourceIdentity = struct {
    node_id: u32,
    node_object_serial: u64,
    device_id: u32,
    device_object_serial: u64,
    node_name: [source_identity_text_capacity]u8,
    node_name_size: u16,
    node_description: [source_identity_text_capacity]u8,
    node_description_size: u16,
    device_serial: [source_identity_text_capacity]u8,
    device_serial_size: u16,
    device_description: [source_identity_text_capacity]u8,
    device_description_size: u16,
};

/// Setup diagnostics retain the server release and protocol negotiation even
/// when no audio callback was accepted.
pub const SetupFailure = struct {
    stage: SetupFailureStage,
    cause: FailureCause,
    message: [setup_error_message_capacity]u8,
    message_size: u16,
    pipewire_version: [pipewire_version_capacity]u8,
    pipewire_version_size: u8,
    client_node_version_advertised: u32,
    client_node_version_selected: u32,
};

pub const Environment = struct {
    runtime_directory: ?[]const u8,
    remote: []const u8,
    system_bus_address: []const u8,
};

/// Own the native connection, shared graph mappings and recording publication
/// for one worker session. No other thread accesses this client. Control work is
/// interleaved one message at a time with graph wakes and supervisor commands.
pub fn run(launch: Launch) Result {
    // PERFORMANCE: Do not restore aggregate defaults for connection or ports.
    // Zig can materialize their large buffer-containing initializers in .rodata
    // even though the buffers themselves are undefined. Initialize connection
    // metadata below; PipeWireClient.createStream writes each complete Port before
    // increasing ports_count, and readers use only that initialized prefix.
    // Measured 2026-09-07 with stock Zig 0.16.0/LLVM, host x86-64, ReleaseSafe
    // application/inference, static PIE, crash diagnostics disabled and GNU strip:
    // connection metadata-only initialization saved about 131 KB; leaving unused
    // port slots untouched removed an 8,576-byte template. Capacities, layouts
    // and allocation policy are unchanged; no resident-memory saving is claimed.
    // Applied together with Notification.initEmpty on the same size
    // basis after the formatting/logging changes, these three initializer fixes
    // reduced 1,301,832 to 1,139,208 bytes (162,624 saved). All three templates
    // disappeared. Historical savings are not additive across compiler builds.
    var client: PipeWireClient = .{ .ports = undefined, .connection = undefined, .source = switch (launch.source) {
        .default => .default,
        .node_name => |value| .{ .node_name = value },
        .device_serial => |value| .{ .device_serial = value },
    } };
    // Establish a closed, empty connection before cleanup or any fallible work;
    // otherwise an early error could close an uninitialized descriptor.
    client.connection.initEmpty();
    defer client.deinit();
    var report: Report = undefined;
    // Keep the large Report out of the error union: constant error returns must
    // not materialize zero-filled report payloads.
    const capture_result = runCapture(launch, &client, &report) catch |err| {
        var detail: SetupFailure = undefined;
        detail.message = @splat(0);
        detail.pipewire_version = @splat(0);
        detail.stage = if (client.source != .default and (err == error.SourceNotFound or err == error.SourceAmbiguous)) .source_resolution else .stream_connect;
        detail.cause = switch (err) {
            error.SourceNotFound => .{ .audio = .configured_device_not_found },
            error.SourceAmbiguous => .{ .audio = .configured_device_ambiguous },
            error.GraphError => .{ .pipewire = client.error_code },
            else => .{ .zig = err },
        };
        detail.pipewire_version_size = @intCast(@min(client.server_version.size, detail.pipewire_version.len));
        @memcpy(detail.pipewire_version[0..detail.pipewire_version_size], client.server_version.get()[0..detail.pipewire_version_size]);
        detail.client_node_version_advertised = client.client_node_version;
        detail.client_node_version_selected = if (client.stream_created) @min(client.client_node_version, 5) else 0;
        detail.message_size = describeError(&client, err, &detail.message);
        return .{ .err = if (err == error.SourceNotFound)
            .{ .source_not_found = detail }
        else if (err == error.SourceAmbiguous)
            .{ .source_ambiguous = detail }
        else
            .{ .setup = detail } };
    };
    return switch (capture_result) {
        .ok => |end| .{ .ok = .{ .end = end, .report = report } },
        .err => |failure| .{ .err = switch (failure.kind) {
            .source_disconnected => .{ .source_connection_lost = .{ .report = report, .detail = failure.detail } },
            .source_changed => .{ .source_changed = .{ .report = report, .detail = failure.detail } },
            .pipeline_full => .{ .pipeline_full = .{ .report = report, .detail = failure.detail } },
            .unexpected_format => .{ .unexpected_format = .{ .report = report, .detail = failure.detail } },
            .timeline_discontinuity => .{ .timeline_discontinuity = .{ .report = report, .detail = failure.detail } },
            .stream_error => .{ .stream_error = .{ .report = report, .detail = failure.detail } },
            .stream_disconnected => .{ .stream_disconnected = .{ .report = report, .detail = failure.detail } },
            .invalid_buffer => .{ .invalid_buffer = .{ .report = report, .detail = failure.detail } },
            .corrupted_buffer => .{ .corrupted_buffer = .{ .report = report, .detail = failure.detail } },
        } },
    };
}

const CaptureFailure = struct { kind: TerminalFailure, detail: RuntimeFailure };
const CaptureResult = union(enum) { ok: End, err: CaptureFailure };

fn runCapture(launch: Launch, client: *PipeWireClient, report: *Report) !CaptureResult {
    assert(launch.recording_samples_target > 0);
    const limits = try std.posix.getrlimit(.MEMLOCK);
    const bytes = std.mem.asBytes(launch.exchange);
    const lock_errno = linux.errno(linux.mlock(bytes.ptr, bytes.len));
    defer if (lock_errno == .SUCCESS) {
        _ = linux.munlock(bytes.ptr, bytes.len);
    };
    var socket_buffer: [108]u8 = undefined;
    const socket_path = if (std.fs.path.isAbsolute(launch.environment.remote)) launch.environment.remote else try std.fmt.bufPrint(&socket_buffer, "{s}/{s}", .{ launch.environment.runtime_directory orelse return error.PipeWireRuntimeDirectoryMissing, launch.environment.remote });
    realtime_scheduling.acquire(launch.recording_id, launch.environment.system_bus_address, launch.control_event_fd);
    try client.init(socket_path);
    var capture: RealtimeCapture = .{
        .exchange = launch.exchange,
        .publication_event_fd = launch.publication_event_fd,
        .recording_samples_target = launch.recording_samples_target,
        .activity_detector = audio_activity.Detector.init(AudioExchange.sample_rate_hz),
    };
    if (!beginNextAvailableSlot(&capture)) return error.AudioExchangeFull;
    defer abandonUnpublishedActiveSlot(&capture);
    var resampler: audio_resampler.Resampler = .{};
    var negotiated: ?NegotiatedFormat = null;
    var outcome: TerminalOutcome = .none;
    var capture_error: ?anyerror = null;
    var polling_errno: linux.E = .SUCCESS;
    while (outcome == .none) {
        client.connection.flush() catch |err| {
            capture_error = err;
            break;
        };
        var fds = [_]linux.pollfd{
            .{ .fd = launch.control_event_fd, .events = linux.POLL.IN, .revents = 0 },
            .{ .fd = client.connection.fd, .events = linux.POLL.IN | (if (client.connection.output_size > 0) @as(i16, linux.POLL.OUT) else 0), .revents = 0 },
            .{ .fd = client.wake_fd, .events = linux.POLL.IN, .revents = 0 },
        };
        const result = linux.poll(&fds, fds.len, if (launch.control.load(.acquire) == .none) -1 else 0);
        polling_errno = linux.errno(result);
        if (polling_errno == .INTR) continue;
        if (polling_errno != .SUCCESS) {
            capture_error = error.CapturePollFailed;
            break;
        }
        // A stop applies at the last completed graph cycle. Its borrowed
        // buffers have been returned before re-entering this poll loop.
        if (fds[0].revents != 0) {
            worker.drain(launch.control_event_fd);
        }
        // The wake may have been drained while taking the job. The atomic
        // command is authoritative, so check it independently of poll readiness.
        switch (launch.control.load(.acquire)) {
            .none => {},
            .stop => {
                outcome = .stopped;
                break;
            },
            .cancel => {
                outcome = .cancelled;
                break;
            },
        }
        // Process at most one control message before returning to the poller.
        // New mappings and identity changes cannot race a graph buffer read.
        if (fds[1].revents & (linux.POLL.IN | linux.POLL.HUP | linux.POLL.ERR) != 0) {
            _ = client.dispatch() catch |err| {
                capture_error = err;
                break;
            };
        }
        if (fds[2].revents & (linux.POLL.HUP | linux.POLL.ERR) != 0) {
            capture_error = error.WakeFailed;
            break;
        }
        if (fds[2].revents & linux.POLL.IN == 0) continue;
        {
            const start = monotonicNanoseconds();
            const block = client.process() catch |err| {
                capture_error = captureErrorAfterPendingEvents(client, err);
                break;
            } orelse {
                if (capture.callback_state == .observed) {
                    observeCallback(&capture, start);
                    capture.callback_state.observed.missing_buffers_count += 1;
                }
                continue;
            };
            defer client.finishCycle();
            observeCallback(&capture, start);
            if (block.header_present) capture.callback_state.observed.header_metadata_buffers_count += 1;

            resampler.configure(block.rate) catch |err| {
                capture_error = err;
                break;
            };
            const count = resampler.outputCount(block.samples_count);
            if (count > AudioExchange.callback_samples_count_max) {
                capture_error = error.OutputFull;
                break;
            }
            if (count > AudioExchange.slot_samples_capacity - capture.active_slot.?.samples_count) {
                publishActiveSlotIfNonEmpty(&capture);
                if (!beginNextAvailableSlot(&capture)) {
                    outcome = .pipeline_full;
                    break;
                }
            }
            const active = capture.active_slot.?;
            const destination = active.writer.slot.samples[active.samples_count..][0..count];
            const samples = resampler.process(&block, destination) catch |err| {
                capture_error = err;
                break;
            };
            if (samples.len > 0) {
                outcome = publishCompleteBlock(&capture, samples);
                if (block.silence and block.header_present and outcome != .pipeline_full) {
                    capture.callback_state.observed.header_gap_buffers_count += 1;
                    capture.callback_state.observed.header_gap_samples_count += @intCast(samples.len);
                }
            }
            capture.callback_state.observed.duration_ns_max = @max(capture.callback_state.observed.duration_ns_max, monotonicNanoseconds() -| start);
        }
        if (negotiated == null or negotiated.?.sample_rate_hz != resampler.input_rate) {
            client.publishFormat(resampler.input_rate) catch |err| {
                capture_error = err;
                break;
            };
            negotiated = .{ .sample_rate_hz = resampler.input_rate, .channels_count = @intCast(client.ports_count) };
        }
    }
    if (capture_error) |err| {
        // Before the first retained callback, discovery/format errors remain
        // setup errors and shared liveness must remain unpublished.
        if (capture.callback_state == .unobserved) return err;
        outcome = classifyError(err);
    }
    if (outcome != .cancelled) publishActiveSlotIfNonEmpty(&capture);
    report.* = undefined;
    report.source_identity = null;
    report.callback = null;
    report.pipewire_server_version = @splat(0);
    const result: CaptureResult = switch (outcome) {
        .completed => .{ .ok = .completed },
        .stopped => .{ .ok = .stopped },
        .cancelled => .{ .ok = .cancelled },
        else => failed: {
            var failure: RuntimeFailure = undefined;
            failure.message = @splat(0);
            failure.stage = switch (outcome) {
                .pipeline_full => .exchange_publication,
                .source_changed, .source_disconnected => .source_identity,
                .timeline_discontinuity => .timeline_continuity,
                .invalid_buffer, .corrupted_buffer => .buffer_data,
                else => .stream_state,
            };
            failure.cause = if ((capture_error orelse error.NoCaptureError) == error.GraphError)
                .{ .pipewire = client.error_code }
            else if (outcome == .pipeline_full)
                .{ .audio = .exchange_full }
            else if (capture_error) |err|
                .{ .zig = err }
            else
                unreachable;
            failure.message_size = describeError(client, capture_error orelse error.AudioExchangeFull, &failure.message);
            if ((capture_error orelse error.NoCaptureError) == error.CapturePollFailed) {
                failure.stage = .main_loop;
                failure.cause = .{ .linux = polling_errno };
            }
            break :failed .{ .err = .{
                .kind = switch (outcome) {
                    .pipeline_full => .pipeline_full,
                    .unexpected_format => .unexpected_format,
                    .timeline_discontinuity => .timeline_discontinuity,
                    .stream_error => .stream_error,
                    .stream_disconnected => .stream_disconnected,
                    .source_disconnected => .source_disconnected,
                    .source_changed => .source_changed,
                    .invalid_buffer => .invalid_buffer,
                    .corrupted_buffer => .corrupted_buffer,
                    .none, .completed, .stopped, .cancelled => unreachable,
                },
                .detail = failure,
            } };
        },
    };
    report.pipewire_server_version_size = @intCast(@min(client.server_version.size, report.pipewire_server_version.len));
    @memcpy(report.pipewire_server_version[0..report.pipewire_server_version_size], client.server_version.get()[0..report.pipewire_server_version_size]);
    report.client_node_version_advertised = client.client_node_version;
    report.client_node_version_selected = if (client.stream_created) @min(client.client_node_version, 5) else 0;
    if (client.selected) |identity| {
        var source: SourceIdentity = std.mem.zeroes(SourceIdentity);
        source.node_id = identity.node.id;
        source.node_object_serial = identity.node.serial;
        source.device_id = std.math.maxInt(u32);
        source.node_name = identity.node.name.bytes;
        source.node_name_size = identity.node.name.size;
        source.node_description = identity.node.description.bytes;
        source.node_description_size = identity.node.description.size;
        if (identity.device) |device| {
            source.device_id = device.id;
            source.device_object_serial = device.serial;
            source.device_serial = device.device_serial.bytes;
            source.device_serial_size = device.device_serial.size;
            source.device_description = device.description.bytes;
            source.device_description_size = device.description.size;
        }
        report.source_identity = source;
    }
    report.negotiated_format = negotiated;
    report.samples_count = capture.samples_count;
    report.published_samples_count = capture.published_samples_count;
    report.slot_publications_count = capture.publications_count;
    report.memory_lock = if (lock_errno == .SUCCESS) .{ .locked = limits.cur } else .{ .unavailable = .{ .errno = lock_errno, .limit_bytes = limits.cur } };
    report.main_loop_thread_id = linux.gettid();
    if (capture.callback_state == .observed) {
        const metrics = capture.callback_state.observed;
        report.callback = .{
            .thread_id = metrics.thread_id,
            .scheduler_policy = metrics.scheduler_policy,
            .scheduler_priority = metrics.scheduler_priority,
            .callbacks_count = metrics.callbacks_count,
            .missing_buffers_count = metrics.missing_buffers_count,
            .clipped_samples_count = metrics.clipped_samples_count,
            .header_metadata_buffers_count = metrics.header_metadata_buffers_count,
            .header_gap_buffers_count = metrics.header_gap_buffers_count,
            .header_gap_samples_count = metrics.header_gap_samples_count,
            .samples_range = metrics.samples_range,
            .duration_ns_max = metrics.duration_ns_max,
            .gap_ns_max = metrics.gap_ns_max,
            .activity = capture.activity_detector.report(),
        };
    }
    return result;
}

fn captureErrorAfterPendingEvents(client: *PipeWireClient, cycle_error: anyerror) anyerror {
    // On 0.3.48, unplug can wake an already-invalid cycle before the queued
    // registry removal reaches us. Stop retaining audio immediately, but drain
    // bounded, already-readable control work to report the concrete source loss
    // instead of only its buffer/timeline symptom. Never wait for future events.
    const cycle_errno = client.connection.errno;
    defer client.connection.errno = cycle_errno;
    for (0..256) |_| {
        const dispatched = client.dispatch() catch |err| {
            if (err == error.SourceDisconnected or err == error.SourceChanged) {
                if (client.error_message_size == 0) {
                    const message = std.fmt.bufPrint(&client.error_message, "audio_cycle_error={s}", .{@errorName(cycle_error)}) catch unreachable;
                    client.error_message_size = message.len;
                }
                return err;
            }
            return cycle_error;
        };
        if (dispatched) continue;
        // receive() can consume just the header and still have the body
        // waiting in the socket. Readiness distinguishes that from waiting.
        var fd = [_]linux.pollfd{.{ .fd = client.connection.fd, .events = linux.POLL.IN, .revents = 0 }};
        const result = linux.poll(&fd, 1, 0);
        if (linux.errno(result) != .SUCCESS or fd[0].revents & linux.POLL.IN == 0) break;
    }
    return cycle_error;
}

fn observeCallback(capture: *RealtimeCapture, start: u64) void {
    if (capture.callback_state == .unobserved) {
        var metrics: CallbackMetrics = std.mem.zeroes(CallbackMetrics);
        metrics.thread_id = linux.gettid();
        metrics.started_ns_previous = start;
        const result = linux.sched_getscheduler(0);
        if (linux.errno(result) == .SUCCESS) {
            const scheduler: linux.SCHED = @bitCast(@as(i32, @intCast(result)));
            metrics.scheduler_policy = @intFromEnum(scheduler.mode);
        }
        var params: linux.sched_param = undefined;
        if (linux.errno(linux.sched_getparam(0, &params)) == .SUCCESS) metrics.scheduler_priority = params.priority;
        capture.callback_state = .{ .observed = metrics };
    }
    const callback = &capture.callback_state.observed;
    callback.gap_ns_max = @max(callback.gap_ns_max, start -| callback.started_ns_previous);
    callback.started_ns_previous = start;
    callback.callbacks_count += 1;
    AudioExchange.publishAudioCallbacksCount(capture.exchange, callback.callbacks_count);
}

fn classifyError(err: anyerror) TerminalOutcome {
    return switch (err) {
        error.SourceDisconnected => .source_disconnected,
        error.SourceChanged => .source_changed,
        error.TimelineDiscontinuity => .timeline_discontinuity,
        error.CorruptedBuffer => .corrupted_buffer,
        error.InvalidBuffer, error.InvalidSample, error.OutputFull => .invalid_buffer,
        error.UnsupportedFormat, error.UnsupportedRate => .unexpected_format,
        error.Disconnected => .stream_disconnected,
        else => .stream_error,
    };
}

fn describeError(client: *const PipeWireClient, err: anyerror, buffer: []u8) u16 {
    var writer = std.Io.Writer.fixed(buffer);
    writeErrorDescription(client, err, &writer) catch {
        const marker = " [truncated]";
        const end = @min(writer.end, buffer.len - marker.len);
        @memcpy(buffer[end..][0..marker.len], marker);
        writer.end = end + marker.len;
    };
    return @intCast(writer.end);
}

fn writeErrorDescription(client: *const PipeWireClient, err: anyerror, writer: *std.Io.Writer) error{WriteFailed}!void {
    try writer.print("{s}: errno={f}, server_error_code={d}, object_id={d}, sequence={d}, detail=\"{f}\"", .{ @errorName(err), logging.fmtErrno(client.connection.errno), client.error_code, client.error_object, client.error_sequence, std.zig.fmtString(client.error_message[0..client.error_message_size]) });
    switch (err) {
        error.UnsupportedDefaultSourceMetadata => try writer.writeAll("; PipeWire default.audio.source must use {\"name\":\"NODE\"} without JSON escapes; configure microphone_node or microphone_serial explicitly"),
        error.DefaultSourceNameInvalidUtf8 => try writer.writeAll("; PipeWire default.audio.source name is not valid UTF-8; configure microphone_node or microphone_serial explicitly"),
        error.DefaultSourceNameTooLong => try writer.writeAll("; PipeWire default.audio.source name exceeds 256 bytes; configure microphone_node or microphone_serial explicitly"),
        else => {},
    }
    if (client.diagnostic.wake_count > 0) {
        const d = client.diagnostic;
        try writer.print("; graph_rate={d}/{d}, graph_position={d}, graph_duration={d}, wake_count={d}, channel={d}, buffer_id={d}, chunk_offset={d}, chunk_size={d}, chunk_stride={d}, chunk_flags={d}, header_flags={d}, header_sequence={d}, header_pts_ns={d}", .{ d.graph_rate_num, d.graph_rate_hz, d.graph_position, d.graph_duration, d.wake_count, d.channel, d.buffer_id, d.chunk_offset, d.chunk_size, d.chunk_stride, d.chunk_flags, d.header_flags, d.header_sequence, d.header_pts_ns });
    }
    if (client.previous) |previous| try writer.print("; previous_position={d}, previous_duration={d}, previous_rate_hz={d}", .{ previous.position, previous.duration, previous.rate });
    if (client.diagnostic.invalid_sample_index) |index| try writer.print("; invalid_sample_index={d}", .{index});
    if (err == error.SourceNotFound or err == error.SourceAmbiguous) {
        try writer.print("; available sources:", .{});
        for (client.catalog) |entry| if (entry) |stored| {
            if (stored.data != .node or !stored.data.node.is_source) continue;
            const object = stored.expand(&client.text);
            var serial: PipeWireClient.Text = .{};
            for (client.catalog) |candidate| if (candidate) |stored_device| {
                if (stored_device.data == .device and stored_device.id == object.parent) {
                    // Own the expanded text until the diagnostic is formatted.
                    serial = stored_device.expand(&client.text).device_serial;
                    break;
                }
            };
            try writer.print(" [name=\"{f}\", description=\"{f}\", serial=\"{f}\"]", .{ std.zig.fmtString(object.name.get()), std.zig.fmtString(object.description.get()), std.zig.fmtString(serial.get()) });
        };
    }
}

const FillingSlot = struct {
    writer: AudioExchange.SlotWriter,
    samples_count: u32,
    contains_activity: bool,
};

const CallbackSamplesRange = struct {
    minimum: u32,
    maximum: u32,
};

const CallbackMetrics = struct {
    thread_id: i32,
    scheduler_policy: ?i32,
    scheduler_priority: ?i32,
    started_ns_previous: u64,
    samples_range: ?CallbackSamplesRange,
    callbacks_count: u64,
    missing_buffers_count: u32,
    clipped_samples_count: u32,
    header_metadata_buffers_count: u32,
    header_gap_buffers_count: u32,
    header_gap_samples_count: u32,
    duration_ns_max: u64,
    gap_ns_max: u64,
};

const CallbackState = union(enum) {
    unobserved,
    observed: CallbackMetrics,
};

const TerminalOutcome = enum(u8) {
    none,
    completed,
    stopped,
    cancelled,
    pipeline_full,
    unexpected_format,
    timeline_discontinuity,
    stream_error,
    stream_disconnected,
    source_disconnected,
    source_changed,
    invalid_buffer,
    corrupted_buffer,
};

const TerminalFailure = enum {
    pipeline_full,
    unexpected_format,
    timeline_discontinuity,
    stream_error,
    stream_disconnected,
    source_disconnected,
    source_changed,
    invalid_buffer,
    corrupted_buffer,
};

const RealtimeCapture = struct {
    exchange: *AudioExchange,
    publication_event_fd: linux.fd_t,
    recording_samples_target: u32,
    active_slot: ?FillingSlot = null,
    publications_count: u32 = 0,
    samples_count: u32 = 0,
    published_samples_count: u32 = 0,
    callback_state: CallbackState = .unobserved,
    activity_detector: audio_activity.Detector,
};
fn publishCompleteBlock(
    realtime: *RealtimeCapture,
    destination_samples: []f32,
) TerminalOutcome {
    const block_samples_count: u32 = @intCast(destination_samples.len);
    assert(block_samples_count > 0);
    assert(block_samples_count <= AudioExchange.callback_samples_count_max);
    assert(block_samples_count <= AudioExchange.slot_samples_capacity);
    if (realtime.active_slot) |active_slot| {
        assert(active_slot.samples_count <= AudioExchange.slot_samples_capacity);
    }
    assert(realtime.samples_count <
        realtime.recording_samples_target + AudioExchange.callback_samples_count_max);

    const active_slot = &realtime.active_slot.?;
    // Conversion wrote only into this unpublished suffix. Normalize in place;
    // a rejected suffix never advances the publication count.
    var clipped: u32 = 0;
    for (destination_samples) |*sample| {
        if (!std.math.isFinite(sample.*)) return .invalid_buffer;
        if (sample.* > 1) {
            sample.* = 1;
            clipped += 1;
        } else if (sample.* < -1) {
            sample.* = -1;
            clipped += 1;
        }
    }
    realtime.callback_state.observed.clipped_samples_count += clipped;

    const activity = realtime.activity_detector.observe(destination_samples);

    active_slot.samples_count += block_samples_count;
    if (activity.activity == .active) active_slot.contains_activity = true;
    realtime.samples_count += block_samples_count;
    assert(active_slot.samples_count <= AudioExchange.slot_samples_capacity);
    assert(realtime.samples_count <=
        realtime.recording_samples_target + AudioExchange.callback_samples_count_max);
    const callback = &realtime.callback_state.observed;
    if (callback.samples_range) |*range| {
        range.minimum = @min(range.minimum, block_samples_count);
        range.maximum = @max(range.maximum, block_samples_count);
    } else {
        callback.samples_range = .{
            .minimum = block_samples_count,
            .maximum = block_samples_count,
        };
    }

    // The absolute duration is authoritative even when its final callback also
    // completes a quiet interval. It is a caller-selected hard limit, while
    // activity supplies only earlier publication decisions.
    if (realtime.samples_count >= realtime.recording_samples_target) {
        return .completed;
    }

    // Once one physical slot contains at least twenty seconds, a sustained
    // quiet run supplies a word-safe boundary. Publication wakes Whisper while
    // capture immediately claims another slot. Experimental boundaries below
    // twenty seconds remain fixed so the supervisor's pressure scenarios keep
    // their deliberate one-second behavior.
    if (active_slot.samples_count >= audio_policy.internal_chunk_samples_count_min and
        activity.activity == .quiet and
        activity.activity_samples_count >=
            audio_policy.natural_boundary_quiet_samples_count)
    {
        publishActiveSlotIfNonEmpty(realtime);
        if (!beginNextAvailableSlot(realtime)) {
            return .pipeline_full;
        }
    }

    return .none;
}

fn beginNextAvailableSlot(realtime: *RealtimeCapture) bool {
    assert(realtime.active_slot == null);

    // Publication order determines physical placement. If this slot is still
    // occupied, every later publication is also unavailable to the ordered
    // consumer, so scanning the other slots cannot create useful capacity.
    const slot_index = AudioExchange.SlotIndex.fromPublicationOrdinal(
        realtime.publications_count,
    );
    const writer = AudioExchange.tryAcquireWriter(
        &realtime.exchange.slots[slot_index.arrayIndex()],
    ) orelse return false;
    realtime.active_slot = .{
        .writer = writer,
        .samples_count = 0,
        .contains_activity = false,
    };
    return true;
}

fn publishActiveSlotIfNonEmpty(realtime: *RealtimeCapture) void {
    const active_slot = realtime.active_slot orelse return;
    assert(active_slot.samples_count <= AudioExchange.slot_samples_capacity);
    if (active_slot.samples_count == 0) return;

    AudioExchange.publishWrittenSlot(active_slot.writer, .{
        .samples_count = active_slot.samples_count,
        .contains_activity = active_slot.contains_activity,
    });

    // The eventfd is a doorbell, not a slot queue. Several publications may
    // coalesce into one counter value; the supervisor drains ring order after
    // every wake. At most three increments can remain unread
    // because audio stops rather than overwriting a published slot.
    writeEventCounter(realtime.publication_event_fd);

    realtime.active_slot = null;
    realtime.published_samples_count += active_slot.samples_count;
    realtime.publications_count += 1;

    assert(realtime.published_samples_count <= realtime.samples_count);
}

fn abandonUnpublishedActiveSlot(realtime: *RealtimeCapture) void {
    const active_slot = realtime.active_slot orelse return;
    assert(active_slot.samples_count <= AudioExchange.slot_samples_capacity);

    // Samples may have been copied into this slot and counted as callback
    // progress, but its writer never release-published a positive count.
    AudioExchange.abandonEmptyWrite(active_slot.writer);
    realtime.active_slot = null;

    assert(realtime.published_samples_count <= realtime.samples_count);
}

fn writeEventCounter(event_fd: std.posix.fd_t) void {
    assert(event_fd >= 0);

    const increment: u64 = 1;
    while (true) {
        const write_result = linux.write(
            event_fd,
            std.mem.asBytes(&increment).ptr,
            @sizeOf(u64),
        );
        switch (linux.errno(write_result)) {
            .SUCCESS => {
                assert(write_result == @sizeOf(u64));
                return;
            },
            // The syscall has not changed the counter when a signal interrupts
            // it, so retrying preserves exactly one notification.
            .INTR => continue,
            // A saturated counter is already readable. This cannot lose the
            // wakeup that asks the receiver to inspect authoritative shared
            // state rather than treating counter values as individual records.
            .AGAIN => return,
            // Both eventfds outlive the capture thread. Any other failure breaks
            // that internal contract, so terminate the daemon for systemd to
            // restart rather than continue with lost publication wakes.
            else => @trap(),
        }
    }
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
pub fn schedulerPolicyName(policy: i32) []const u8 {
    const name = switch (policy) {
        @intFromEnum(linux.SCHED.Mode.FIFO) => "fifo",
        @intFromEnum(linux.SCHED.Mode.RR) => "round_robin",
        @intFromEnum(linux.SCHED.Mode.NORMAL) => "normal",
        else => "other",
    };
    assert(name.len > 0);
    return name;
}

pub fn schedulerPolicyIsRealtime(policy: i32) bool {
    return policy == @intFromEnum(linux.SCHED.Mode.FIFO) or
        policy == @intFromEnum(linux.SCHED.Mode.RR);
}
