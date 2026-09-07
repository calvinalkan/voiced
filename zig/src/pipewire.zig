//! One native PipeWire capture session. The worker owns graph discovery,
//! mappings, mono conversion and shared-slot publication through one poll loop.
//! Graph processing uses fixed storage and no allocation or blocking I/O.
const std = @import("std");
const logging = @import("logging.zig");
const linux = std.os.linux;
const assert = std.debug.assert;
const native = @import("pipewire_native.zig");
const audio_resampler = @import("audio_resampler.zig");
const realtime_scheduling = @import("realtime.zig");
const audio_activity = @import("audio_activity.zig");
const audio_exchange = @import("audio_exchange.zig");
const audio_policy = @import("audio_policy.zig");
const AudioExchange = audio_exchange.AudioExchange;
pub const runtime_error_message_capacity = 512;
const callback_error_message_capacity = runtime_error_message_capacity;
pub const setup_error_message_capacity = 4096;
pub const pipewire_version_capacity = 64;
pub const source_identity_text_capacity = 256;
pub const TimelineValidation = enum(u32) {
    header_only,
    full,
};

/// One supervisor command ends an active recording. `stop` retains every
/// complete callback block, including the final partially filled exchange slot;
/// `cancel` abandons that private final slot and tells the supervisor to discard
/// every earlier publication belonging to the session.
pub const ControlCommand = enum(u32) {
    stop,
    cancel,
};

/// Fixed control-socket record shared by the supervisor and audio worker.
/// Reserved bytes make accidental protocol drift an asserted internal defect.
pub const ControlPacket = extern struct {
    command: u32,
    reserved: u32,
};

pub const Source = union(enum) {
    default,
    node_name: [:0]const u8,
    device_serial: [:0]const u8,
};

pub const AutomaticStop = audio_policy.AutomaticStop;

pub const Launch = struct {
    exchange: *AudioExchange,
    publication_event_fd: std.posix.fd_t,
    control_socket: std.posix.fd_t,
    source: Source,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    automatic_stop: AutomaticStop,
    process_realtime: bool,
    environment: Environment,
};

/// Classifies why capture stopped so the supervisor can choose policy without
/// parsing diagnostic text. Duration completion, automatic quiet, stop, and
/// cancellation are expected terminal states and therefore carry no
/// `RuntimeError`; every other value does.
pub const Outcome = enum(u32) {
    completed,
    automatic_stop,
    stopped,
    cancelled,
    control_error,
    pipeline_full,
    format_parse_error,
    unexpected_format,
    buffer_configuration_error,
    timeline_error,
    timeline_discontinuity,
    stream_error,
    stream_disconnected,
    source_disconnected,
    source_changed,
    source_observation_error,
    invalid_buffer,
    corrupted_buffer,
    buffer_return_error,
};

pub const FailureOutcome = enum(u32) {
    control_error = @intFromEnum(Outcome.control_error),
    pipeline_full = @intFromEnum(Outcome.pipeline_full),
    format_parse_error = @intFromEnum(Outcome.format_parse_error),
    unexpected_format = @intFromEnum(Outcome.unexpected_format),
    buffer_configuration_error = @intFromEnum(Outcome.buffer_configuration_error),
    timeline_error = @intFromEnum(Outcome.timeline_error),
    timeline_discontinuity = @intFromEnum(Outcome.timeline_discontinuity),
    stream_error = @intFromEnum(Outcome.stream_error),
    stream_disconnected = @intFromEnum(Outcome.stream_disconnected),
    source_disconnected = @intFromEnum(Outcome.source_disconnected),
    source_changed = @intFromEnum(Outcome.source_changed),
    source_observation_error = @intFromEnum(Outcome.source_observation_error),
    invalid_buffer = @intFromEnum(Outcome.invalid_buffer),
    corrupted_buffer = @intFromEnum(Outcome.corrupted_buffer),
    buffer_return_error = @intFromEnum(Outcome.buffer_return_error),
};

pub const RuntimeFailure = struct {
    coordinate: FailureCoordinate,
    message: [callback_error_message_capacity]u8,
    message_size: u16,
};

pub const CaptureFailure = struct {
    outcome: FailureOutcome,
    detail: RuntimeFailure,
};

pub const CaptureEnd = union(enum) {
    completed,
    automatic_stop,
    stopped,
    cancelled,
    failed: CaptureFailure,
};

pub const MemoryLockResult = union(enum) {
    locked: u64,
    unavailable: struct {
        error_code: i32,
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

/// Returns the first terminal capture cause and all stable measurements after
/// the stream has stopped. Optional and tagged payloads retain only fields that
/// exist for that outcome; teardown remains independent because it happens
/// after capture has already selected its terminal cause.
pub const NegotiatedFormat = struct {
    sample_rate_hz: u32,
    channels_count: u32,
};

pub const Report = struct {
    end: CaptureEnd,
    teardown_failure: ?RuntimeFailure,

    timeline_validation: TimelineValidation,
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

pub const SetupErrorStage = enum(u32) {
    none,
    source_resolution,
    stream_connect,
};

/// Numeric errors retain their native Zig error, Linux errno or PipeWire
/// result. Voiced policy codes use the explicit enums below.
pub const ErrorDomain = enum(u32) {
    none,
    zig_error,
    linux_errno,
    pipewire_result,
    voiced_audio,
};

/// Names the capture operation that observed a runtime or teardown error.
pub const RuntimeErrorStage = enum(u32) {
    none,
    main_loop,
    callback_event,
    format_negotiation,
    buffer_negotiation,
    stream_state,
    source_identity,
    supervisor_control,
    stream_disconnect,
    buffer_metadata,
    buffer_data,
    sample_validation,
    timeline_continuity,
    exchange_publication,
    buffer_return,
};

pub const FailureStage = enum(u32) {
    main_loop = @intFromEnum(RuntimeErrorStage.main_loop),
    callback_event = @intFromEnum(RuntimeErrorStage.callback_event),
    format_negotiation = @intFromEnum(RuntimeErrorStage.format_negotiation),
    buffer_negotiation = @intFromEnum(RuntimeErrorStage.buffer_negotiation),
    stream_state = @intFromEnum(RuntimeErrorStage.stream_state),
    source_identity = @intFromEnum(RuntimeErrorStage.source_identity),
    supervisor_control = @intFromEnum(RuntimeErrorStage.supervisor_control),
    stream_disconnect = @intFromEnum(RuntimeErrorStage.stream_disconnect),
    buffer_metadata = @intFromEnum(RuntimeErrorStage.buffer_metadata),
    buffer_data = @intFromEnum(RuntimeErrorStage.buffer_data),
    sample_validation = @intFromEnum(RuntimeErrorStage.sample_validation),
    timeline_continuity = @intFromEnum(RuntimeErrorStage.timeline_continuity),
    exchange_publication = @intFromEnum(RuntimeErrorStage.exchange_publication),
    buffer_return = @intFromEnum(RuntimeErrorStage.buffer_return),
};

pub const FailureDomain = enum(u32) {
    zig_error = @intFromEnum(ErrorDomain.zig_error),
    linux_errno = @intFromEnum(ErrorDomain.linux_errno),
    pipewire_result = @intFromEnum(ErrorDomain.pipewire_result),
    voiced_audio = @intFromEnum(ErrorDomain.voiced_audio),
};

pub const FailureCoordinate = struct {
    stage: FailureStage,
    domain: FailureDomain,
    code: i64,
};

/// Stable codes for Voiced policy errors that have no native code.
pub const SourceResolutionErrorCode = enum(u32) {
    none,
    configured_device_not_found,
    configured_device_ambiguous,
};

pub const AudioErrorCode = enum(u32) { none, exchange_full };

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

pub const SetupFailureStage = enum(u32) {
    source_resolution = @intFromEnum(SetupErrorStage.source_resolution),
    stream_connect = @intFromEnum(SetupErrorStage.stream_connect),
};

/// Setup diagnostics retain the server release and protocol negotiation even
/// when no audio callback was accepted.
pub const SetupFailure = struct {
    stage: SetupFailureStage,
    domain: FailureDomain,
    code: i64,
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

pub const RunResult = union(enum) { captured: Report, setup_failed: SetupFailure };

/// Own the native connection, shared graph mappings and recording publication
/// for one worker session. No other thread accesses this client. Control work is
/// interleaved one message at a time with graph wakes and supervisor commands.
pub fn run(launch: Launch) RunResult {
    // PERFORMANCE: Do not restore aggregate defaults for connection or ports.
    // Zig can materialize their large buffer-containing initializers in .rodata
    // even though the buffers themselves are undefined. Initialize connection
    // metadata below; native.Client.createStream writes each complete Port before
    // increasing ports_count, and readers use only that initialized prefix.
    // Measured 2026-09-07 with stock Zig 0.16.0/LLVM, host x86-64, ReleaseSafe
    // application/inference, static PIE, -Dcrash-diagnostics=false and GNU strip:
    // connection metadata-only initialization saved about 131 KB; leaving unused
    // port slots untouched removed an 8,576-byte template. Capacities, layouts
    // and allocation policy are unchanged; no resident-memory saving is claimed.
    // Applied together with notifications.Client.initEmpty on the same size
    // basis after the formatting/logging changes, these three initializer fixes
    // reduced 1,301,832 to 1,139,208 bytes (162,624 saved). All three templates
    // disappeared. Historical savings are not additive across compiler builds.
    var client: native.Client = .{ .ports = undefined, .connection = undefined, .source = switch (launch.source) {
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
    runCapture(launch, &client, &report) catch |err| {
        var detail: SetupFailure = undefined;
        detail.message = @splat(0);
        detail.pipewire_version = @splat(0);
        detail.stage = if (client.source != .default and (err == error.SourceNotFound or err == error.SourceAmbiguous)) .source_resolution else .stream_connect;
        detail.domain = if (err == error.SourceNotFound or err == error.SourceAmbiguous) .voiced_audio else if (err == error.GraphError) .pipewire_result else .zig_error;
        detail.code = switch (err) {
            error.SourceNotFound => @intFromEnum(SourceResolutionErrorCode.configured_device_not_found),
            error.SourceAmbiguous => @intFromEnum(SourceResolutionErrorCode.configured_device_ambiguous),
            error.GraphError => client.error_code,
            else => @intFromError(err),
        };
        detail.pipewire_version_size = @intCast(@min(client.server_version.size, detail.pipewire_version.len));
        @memcpy(detail.pipewire_version[0..detail.pipewire_version_size], client.server_version.get()[0..detail.pipewire_version_size]);
        detail.client_node_version_advertised = client.client_node_version;
        detail.client_node_version_selected = if (client.stream_created) @min(client.client_node_version, 5) else 0;
        detail.message_size = describeError(&client, err, &detail.message);
        return .{ .setup_failed = detail };
    };
    return .{ .captured = report };
}

fn runCapture(launch: Launch, client: *native.Client, report: *Report) !void {
    assert(launch.recording_samples_target > 0);
    assert(launch.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(launch.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    const limits = try std.posix.getrlimit(.MEMLOCK);
    const bytes = std.mem.asBytes(launch.exchange);
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += std.heap.page_size_min) {
        const byte: *volatile u8 = @ptrCast(&bytes[offset]);
        byte.* = byte.*;
    }
    const lock_errno = linux.errno(linux.mlock(bytes.ptr, bytes.len));
    defer if (lock_errno == .SUCCESS) {
        _ = linux.munlock(bytes.ptr, bytes.len);
    };
    var socket_buffer: [108]u8 = undefined;
    const socket_path = if (std.fs.path.isAbsolute(launch.environment.remote)) launch.environment.remote else try std.fmt.bufPrint(&socket_buffer, "{s}/{s}", .{ launch.environment.runtime_directory orelse return error.PipeWireRuntimeDirectoryMissing, launch.environment.remote });
    if (launch.process_realtime) realtime_scheduling.acquire(launch.environment.system_bus_address, launch.control_socket);
    try client.init(socket_path);
    var capture: RealtimeCapture = .{
        .exchange = launch.exchange,
        .publication_event_fd = launch.publication_event_fd,
        .recording_samples_target = launch.recording_samples_target,
        .slot_samples_boundary = launch.slot_samples_boundary,
        .automatic_stop = launch.automatic_stop,
        .activity_detector = audio_activity.Detector.init(audio_exchange.sample_rate_hz),
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
            .{ .fd = launch.control_socket, .events = linux.POLL.IN, .revents = 0 },
            .{ .fd = client.connection.fd, .events = linux.POLL.IN | (if (client.connection.output_size > 0) @as(i16, linux.POLL.OUT) else 0), .revents = 0 },
            .{ .fd = client.wake_fd, .events = linux.POLL.IN, .revents = 0 },
        };
        const result = linux.poll(&fds, fds.len, -1);
        polling_errno = linux.errno(result);
        if (polling_errno == .INTR) continue;
        if (polling_errno != .SUCCESS) {
            capture_error = error.CapturePollFailed;
            break;
        }
        // A stop applies at the last completed graph cycle. Its borrowed
        // buffers have been returned before re-entering this poll loop.
        if (fds[0].revents != 0) {
            var command: ControlPacket = undefined;
            const received = linux.recvfrom(launch.control_socket, std.mem.asBytes(&command).ptr, @sizeOf(ControlPacket), linux.MSG.DONTWAIT | linux.MSG.TRUNC, null, null);
            if (linux.errno(received) == .INTR or linux.errno(received) == .AGAIN) continue;
            if (received != @sizeOf(ControlPacket) or command.reserved != 0 or command.command > @intFromEnum(ControlCommand.cancel)) {
                capture_error = error.InvalidControlCommand;
                break;
            }
            outcome = if (command.command == @intFromEnum(ControlCommand.cancel)) .cancelled else .stopped;
            break;
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
            if (capture.callback_state == .unobserved) audio_exchange.publishTimelineValidation(launch.exchange, .full);
            observeCallback(&capture, start);
            if (block.header_present) capture.callback_state.observed.header_metadata_buffers_count += 1;

            resampler.configure(block.rate) catch |err| {
                capture_error = err;
                break;
            };
            const count = resampler.outputCount(block.samples_count);
            if (count > audio_exchange.callback_samples_count_max) {
                capture_error = error.OutputFull;
                break;
            }
            if (count > capture.slot_samples_boundary - capture.active_slot.?.samples_count) {
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
    if (outcome != .cancelled and !(outcome == .automatic_stop and capture.automatic_stop_confirmation_tail_is_private)) publishActiveSlotIfNonEmpty(&capture);
    report.* = undefined;
    report.teardown_failure = null;
    report.source_identity = null;
    report.callback = null;
    report.pipewire_server_version = @splat(0);
    if (capture.callback_state == .unobserved) audio_exchange.publishTimelineValidation(launch.exchange, .full);
    report.end = switch (outcome) {
        .completed => .completed,
        .automatic_stop => .automatic_stop,
        .stopped => .stopped,
        .cancelled => .cancelled,
        else => failed: {
            var failure: RuntimeFailure = undefined;
            failure.message = @splat(0);
            failure.coordinate = .{ .stage = switch (outcome) {
                .pipeline_full => .exchange_publication,
                .source_changed, .source_disconnected => .source_identity,
                .control_error => .supervisor_control,
                .timeline_discontinuity => .timeline_continuity,
                .invalid_buffer, .corrupted_buffer => .buffer_data,
                else => .stream_state,
            }, .domain = if ((capture_error orelse error.NoCaptureError) == error.GraphError) .pipewire_result else if (outcome == .pipeline_full) .voiced_audio else .zig_error, .code = if (capture_error) |err| if (err == error.GraphError) client.error_code else @intFromError(err) else @intFromEnum(AudioErrorCode.exchange_full) };
            failure.message_size = describeError(client, capture_error orelse error.AudioExchangeFull, &failure.message);
            if ((capture_error orelse error.NoCaptureError) == error.CapturePollFailed) failure.coordinate = .{ .stage = .main_loop, .domain = .linux_errno, .code = @intFromEnum(polling_errno) };
            break :failed .{ .failed = .{ .outcome = std.meta.stringToEnum(FailureOutcome, @tagName(outcome)).?, .detail = failure } };
        },
    };
    report.timeline_validation = .full;
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
    report.memory_lock = if (lock_errno == .SUCCESS) .{ .locked = limits.cur } else .{ .unavailable = .{ .error_code = @intFromEnum(lock_errno), .limit_bytes = limits.cur } };
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
}

fn captureErrorAfterPendingEvents(client: *native.Client, cycle_error: anyerror) anyerror {
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
    audio_exchange.publishAudioCallbacksCount(capture.exchange, callback.callbacks_count);
}

fn classifyError(err: anyerror) TerminalOutcome {
    return switch (err) {
        error.SourceDisconnected => .source_disconnected,
        error.SourceChanged => .source_changed,
        error.TimelineDiscontinuity => .timeline_discontinuity,
        error.CorruptedBuffer => .corrupted_buffer,
        error.InvalidBuffer, error.InvalidSample, error.OutputFull => .invalid_buffer,
        error.UnsupportedFormat, error.UnsupportedRate => .unexpected_format,
        error.InvalidControlCommand => .control_error,
        error.Disconnected => .stream_disconnected,
        else => .stream_error,
    };
}

fn describeError(client: *const native.Client, err: anyerror, buffer: []u8) u16 {
    var writer = std.Io.Writer.fixed(buffer);
    writeErrorDescription(client, err, &writer) catch {
        const marker = " [truncated]";
        const end = @min(writer.end, buffer.len - marker.len);
        @memcpy(buffer[end..][0..marker.len], marker);
        writer.end = end + marker.len;
    };
    return @intCast(writer.end);
}

fn writeErrorDescription(client: *const native.Client, err: anyerror, writer: *std.Io.Writer) error{WriteFailed}!void {
    try writer.print("{s}: errno={f}, server_error_code={d}, object_id={d}, sequence={d}, detail=\"{f}\"", .{ @errorName(err), logging.fmtErrno(client.connection.errno), client.error_code, client.error_object, client.error_sequence, std.zig.fmtString(client.error_message[0..client.error_message_size]) });
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
            var serial: native.Text = .{};
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
    writer: audio_exchange.SlotWriter,
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
    automatic_stop,
    stopped,
    cancelled,
    control_error,
    pipeline_full,
    format_parse_error,
    unexpected_format,
    buffer_configuration_error,
    timeline_error,
    timeline_discontinuity,
    stream_error,
    stream_disconnected,
    source_disconnected,
    source_changed,
    source_observation_error,
    invalid_buffer,
    corrupted_buffer,
    buffer_return_error,
};

const RealtimeCapture = struct {
    exchange: *AudioExchange,
    publication_event_fd: linux.fd_t,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    automatic_stop: AutomaticStop,
    active_slot: ?FillingSlot = null,
    publications_count: u32 = 0,
    samples_count: u32 = 0,
    published_samples_count: u32 = 0,
    automatic_stop_confirmation_tail_is_private: bool = false,
    callback_state: CallbackState = .unobserved,
    activity_detector: audio_activity.Detector,
};
fn publishCompleteBlock(
    realtime: *RealtimeCapture,
    destination_samples: []f32,
) TerminalOutcome {
    const block_samples_count: u32 = @intCast(destination_samples.len);
    assert(block_samples_count > 0);
    assert(block_samples_count <= audio_exchange.callback_samples_count_max);
    assert(block_samples_count <= realtime.slot_samples_boundary);
    if (realtime.active_slot) |active_slot| {
        assert(active_slot.samples_count <= realtime.slot_samples_boundary);
    }
    assert(realtime.samples_count <
        realtime.recording_samples_target + audio_exchange.callback_samples_count_max);

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
    assert(active_slot.samples_count <= realtime.slot_samples_boundary);
    assert(realtime.samples_count <=
        realtime.recording_samples_target + audio_exchange.callback_samples_count_max);
    audio_exchange.publishAudioSamplesCount(
        realtime.exchange,
        realtime.samples_count,
    );

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
    // activity supplies only earlier publication and automatic-stop decisions.
    if (realtime.samples_count >= realtime.recording_samples_target) {
        return .completed;
    }

    if (activity.activity == .active) {
        // Speech after a natural boundary makes the current slot part of the
        // utterance. A later automatic stop must publish it rather than treating
        // it as the quiet confirmation tail of the preceding chunk.
        realtime.automatic_stop_confirmation_tail_is_private = false;
    } else if (activity.activity == .quiet and
        realtime.automatic_stop == .after_quiet and
        realtime.activity_detector.active_samples_count > 0 and
        activity.activity_samples_count >=
            audio_policy.automatic_stop_quiet_samples_count)
    {
        return .automatic_stop;
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
        realtime.automatic_stop_confirmation_tail_is_private =
            realtime.automatic_stop == .after_quiet;
        if (!beginNextAvailableSlot(realtime)) {
            return .pipeline_full;
        }
    }

    return .none;
}

fn beginNextAvailableSlot(realtime: *RealtimeCapture) bool {
    assert(realtime.exchange.session_id > 0);
    assert(realtime.active_slot == null);

    // Prefer cyclic physical reuse so a healthy consumer spreads writes across
    // all slots, but accept any released slot. The publication count is also
    // the next ordinal; claiming a private slot does not create another counter.
    const preferred_slot_index =
        realtime.publications_count % audio_exchange.slots_count;
    for (0..audio_exchange.slots_count) |slot_offset| {
        const slot_index = audio_exchange.SlotIndex.fromArrayIndex(
            (preferred_slot_index + slot_offset) % audio_exchange.slots_count,
        );
        const writer = audio_exchange.tryAcquireWriter(
            realtime.exchange,
            slot_index,
            realtime.publications_count,
        ) orelse continue;

        realtime.active_slot = .{
            .writer = writer,
            .samples_count = 0,
            .contains_activity = false,
        };
        return true;
    }

    return false;
}

fn publishActiveSlotIfNonEmpty(realtime: *RealtimeCapture) void {
    const active_slot = realtime.active_slot orelse return;
    assert(active_slot.samples_count <= realtime.slot_samples_boundary);
    if (active_slot.samples_count == 0) return;

    audio_exchange.publishWrittenSlot(active_slot.writer, .{
        .samples_count = active_slot.samples_count,
        .contains_activity = active_slot.contains_activity,
    });

    // The eventfd is a doorbell, not a slot queue. Several publications may
    // coalesce into one counter value; the supervisor scans all three slots by
    // ordinal after every wake. At most three increments can remain unread
    // because audio stops rather than overwriting a published slot.
    writeEventCounter(realtime.publication_event_fd);

    realtime.active_slot = null;
    realtime.published_samples_count += active_slot.samples_count;
    realtime.publications_count += 1;

    assert(realtime.published_samples_count <= realtime.samples_count);
}

fn abandonUnpublishedActiveSlot(realtime: *RealtimeCapture) void {
    const active_slot = realtime.active_slot orelse return;
    assert(active_slot.samples_count <= realtime.slot_samples_boundary);

    // Samples may have been copied into this slot and counted as callback
    // progress, but its writer never release-published a positive count.
    audio_exchange.abandonEmptyWrite(active_slot.writer);
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
            // Both eventfds are worker-owned for the complete capture. Any other
            // failure violates that internal lifetime contract; trap only this
            // process and let supervisor pidfd/deadline handling contain it.
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
        @intFromEnum(linux.SCHED.Mode.FIFO) => "FIFO",
        @intFromEnum(linux.SCHED.Mode.RR) => "round-robin",
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

pub fn linuxErrorNameFromCode(error_code: i32) []const u8 {
    assert(error_code > 0);
    return @tagName(@as(linux.E, @enumFromInt(error_code)));
}
