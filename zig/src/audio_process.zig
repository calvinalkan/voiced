//! Owns the capture worker's process protocol and execution. The supervisor
//! creates shared resources, owns deadlines, consumes publications, and reaps
//! workers. This module transfers descriptors, validates value-only packets,
//! and enters the single PipeWire capture implementation after exec.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const audio_policy = @import("audio_policy.zig");
const pipewire = @import("pipewire.zig");
const descriptor_handoff = @import("descriptor_handoff.zig");
const assert = std.debug.assert;

const AudioExchange = audio_exchange.AudioExchange;

const target_name_capacity = 256;
const worker_error_message_capacity = pipewire.setup_error_message_capacity;

// These are process-contract vocabulary, re-exported so callers never need to
// import the worker-private PipeWire or exchange modules.
pub const ControlCommand = pipewire.ControlCommand;
pub const AutomaticStop = pipewire.AutomaticStop;
pub const Outcome = pipewire.Outcome;
pub const CaptureErrorKind = pipewire.FailureOutcome;
pub const TimelineValidation = pipewire.TimelineValidation;
pub const SetupErrorStage = pipewire.SetupErrorStage;
pub const SetupDiagnosticStage = pipewire.SetupFailureStage;
pub const RuntimeErrorStage = pipewire.RuntimeErrorStage;
pub const DiagnosticStage = pipewire.FailureStage;
pub const ErrorDomain = pipewire.ErrorDomain;
pub const DiagnosticDomain = pipewire.FailureDomain;
pub const sample_rate_hz = audio_exchange.sample_rate_hz;
pub const slots_count = audio_exchange.slots_count;
pub const slot_duration_seconds_max = audio_exchange.slot_duration_seconds_max;
pub const internal_chunk_duration_seconds_min =
    audio_policy.internal_chunk_duration_seconds_min;
pub const natural_boundary_quiet_duration_ms =
    audio_policy.natural_boundary_quiet_duration_ms;
pub const automatic_stop_quiet_duration_ms =
    audio_policy.automatic_stop_quiet_duration_ms;
pub const callback_samples_count_max = audio_exchange.callback_samples_count_max;
pub const target_name_bytes_capacity = target_name_capacity;
pub const schedulerPolicyName = pipewire.schedulerPolicyName;
pub const schedulerPolicyIsRealtime = pipewire.schedulerPolicyIsRealtime;

pub const Source = pipewire.Source;

pub const NativeDiagnostic = pipewire.RuntimeFailure;
pub const SetupDiagnostic = pipewire.SetupFailure;
pub const MemoryLockResult = pipewire.MemoryLockResult;
pub const ResolvedSource = pipewire.SourceIdentity;
pub const CallbackSamplesRange = pipewire.CallbackSamplesRange;
pub const ActivityReport = pipewire.ActivityReport;
pub const CallbackObservation = pipewire.ReportCallback;
pub const CaptureError = pipewire.CaptureFailure;
pub const CaptureEnd = pipewire.CaptureEnd;
pub const NegotiatedFormat = pipewire.NegotiatedFormat;
pub const CaptureReport = pipewire.Report;
const WorkerReport = pipewire.RunResult;

// An error owns the complete original report: valid audio, source identity,
// native diagnostics, and a possible second teardown error travel together.
pub const Error = union(enum) {
    source_not_found: SetupDiagnostic,
    source_ambiguous: SetupDiagnostic,
    setup: SetupDiagnostic,
    source_connection_lost: CaptureReport,
    source_changed: CaptureReport,
    source_observation: CaptureReport,
    pipeline_full: CaptureReport,
    capture: CaptureReport,
    teardown: CaptureReport,
    invalid_report,
};
pub const Result = union(enum) { ok: CaptureReport, err: Error };

fn serviceResult(report: WorkerReport) Result {
    return switch (report) {
        .setup_failed => |detail| if (detail.stage == .source_resolution and detail.domain == .voiced_audio)
            switch (detail.code) {
                @intFromEnum(pipewire.SourceResolutionErrorCode.configured_device_not_found) => .{ .err = .{ .source_not_found = detail } },
                @intFromEnum(pipewire.SourceResolutionErrorCode.configured_device_ambiguous) => .{ .err = .{ .source_ambiguous = detail } },
                else => .{ .err = .{ .setup = detail } },
            }
        else
            .{ .err = .{ .setup = detail } },
        .captured => |capture| if (capture.end == .failed) switch (capture.end.failed.outcome) {
            .source_disconnected => .{ .err = .{ .source_connection_lost = capture } },
            .source_changed => .{ .err = .{ .source_changed = capture } },
            .source_observation_error => .{ .err = .{ .source_observation = capture } },
            .pipeline_full => .{ .err = .{ .pipeline_full = capture } },
            else => .{ .err = .{ .capture = capture } },
        } else if (capture.teardown_failure != null)
            .{ .err = .{ .teardown = capture } }
        else
            .{ .ok = capture },
    };
}

fn wireEnum(comptime Enum: type, value: anytype) error{InvalidReport}!Enum {
    return std.enums.fromInt(Enum, value) orelse error.InvalidReport;
}

/// Encodes the audio control protocol behind the process boundary. Supervisors
/// choose a logical command and never import PipeWire transport records.
pub fn sendControl(socket: std.posix.fd_t, command: ControlCommand) !void {
    assert(socket >= 0);
    const packet: pipewire.ControlPacket = .{
        .command = @intFromEnum(command),
        .reserved = 0,
    };
    while (true) {
        const bytes = std.mem.asBytes(&packet);
        const result = std.os.linux.sendto(
            socket,
            bytes.ptr,
            bytes.len,
            std.os.linux.MSG.NOSIGNAL,
            null,
            0,
        );
        switch (std.os.linux.errno(result)) {
            .SUCCESS => {
                assert(result == bytes.len);
                return;
            },
            .INTR => continue,
            .PIPE, .CONNRESET => {
                return error.AudioControlPeerClosed;
            },
            else => {
                return error.AudioControlSendFailed;
            },
        }
    }
}

/// `PipeWireWorker` is the real audio role used by the one-binary supervisor.
/// The supervisor sends bounded capture settings and transfers the audio memfd
/// plus publication eventfd in one launch message. The worker returns the same
/// logical final report used by service recordings.
pub const PipeWireWorker = struct {
    pub const protocol_version: u16 = 5;

    pub const LaunchOptions = struct {
        session_id: u64,
        source: Source,
        recording_samples_target: u32,
        automatic_stop: AutomaticStop,
    };

    /// `sendLaunch` keeps descriptor transfer and fixed-record construction on
    /// one side of the process API. The sender retains both descriptors; the
    /// worker receives close-on-exec duplicates referring to the same objects.
    pub fn sendLaunch(
        control_socket: std.posix.fd_t,
        audio_exchange_fd: std.posix.fd_t,
        publication_event_fd: std.posix.fd_t,
        options: LaunchOptions,
    ) !void {
        assert(control_socket >= 0);
        assert(audio_exchange_fd >= 0);
        assert(publication_event_fd >= 0);
        assert(options.session_id > 0);
        assert(options.recording_samples_target > 0);
        assert(options.recording_samples_target <=
            std.math.maxInt(u32) - audio_exchange.callback_samples_count_max);
        switch (options.source) {
            .default => {},
            .node_name, .device_serial => |source| {
                assert(source.len > 0);
                assert(source.len < target_name_capacity);
            },
        }

        const launch_packet = buildLaunchPacket(options);
        const descriptors = [_]std.posix.fd_t{
            audio_exchange_fd,
            publication_event_fd,
        };
        try descriptor_handoff.send(control_socket, &launch_packet, &descriptors);
    }

    /// `decodeReport` checks wire enums before any `@enumFromInt`. Field
    /// consistency stays in `validateReportPacket` as encoder asserts.
    fn decodeReport(
        report: WireReport,
        exchange: *const AudioExchange,
        recording_samples_target: u32,
        automatic_stop: AutomaticStop,
    ) error{InvalidReport}!WorkerReport {
        if (report.worker_succeeded == 0) {
            const stage = try wireEnum(pipewire.SetupFailureStage, report.setup_error_stage);
            const domain = try wireEnum(pipewire.FailureDomain, report.setup_error_domain);
            validateReportPacket(&report, recording_samples_target, automatic_stop, exchange);
            return .{ .setup_failed = .{
                .stage = stage,
                .domain = domain,
                .code = report.setup_error_code,
                .message = report.error_message,
                .message_size = report.error_message_size,
                .pipewire_version = report.pipewire_version,
                .pipewire_version_size = report.pipewire_version_size,
            } };
        }

        const outcome = try wireEnum(Outcome, report.outcome);
        const timeline_validation = try wireEnum(TimelineValidation, report.timeline_validation);
        const activity = try wireEnum(pipewire.Activity, report.activity.activity);
        validateReportPacket(&report, recording_samples_target, automatic_stop, exchange);
        assert(report.error_message_size <= pipewire.runtime_error_message_capacity);
        assert(report.teardown_error_message_size <= pipewire.runtime_error_message_capacity);
        const end: CaptureEnd = switch (outcome) {
            .completed => .completed,
            .automatic_stop => .automatic_stop,
            .stopped => .stopped,
            .cancelled => .cancelled,
            else => .{ .failed = .{
                .outcome = @enumFromInt(@intFromEnum(outcome)),
                .detail = .{
                    .coordinate = .{
                        .stage = try wireEnum(pipewire.FailureStage, report.runtime_error_stage),
                        .domain = try wireEnum(pipewire.FailureDomain, report.runtime_error_domain),
                        .code = report.runtime_error_code,
                    },
                    .message = report.error_message[0..pipewire.runtime_error_message_capacity].*,
                    .message_size = report.error_message_size,
                },
            } },
        };
        const teardown_failure: ?NativeDiagnostic = if (report.teardown_error_stage == @intFromEnum(RuntimeErrorStage.none))
            null
        else
            .{
                .coordinate = .{
                    .stage = try wireEnum(pipewire.FailureStage, report.teardown_error_stage),
                    .domain = try wireEnum(pipewire.FailureDomain, report.teardown_error_domain),
                    .code = report.teardown_error_code,
                },
                .message = report.teardown_error_message[0..pipewire.runtime_error_message_capacity].*,
                .message_size = report.teardown_error_message_size,
            };
        const source: ?ResolvedSource = if (report.source_is_resolved == 1)
            .{
                .node_id = report.source_node_id,
                .node_object_serial = report.source_node_object_serial,
                .device_id = report.source_device_id,
                .device_object_serial = report.source_device_object_serial,
                .node_name = report.source_node_name,
                .node_name_size = report.source_node_name_size,
                .node_description = report.source_node_description,
                .node_description_size = report.source_node_description_size,
                .device_serial = report.source_device_serial,
                .device_serial_size = report.source_device_serial_size,
                .device_description = report.source_device_description,
                .device_description_size = report.source_device_description_size,
            }
        else
            null;
        const callback: ?CallbackObservation = if (report.callback_thread_id == 0)
            null
        else
            .{
                .thread_id = report.callback_thread_id,
                .scheduler_policy = if (report.callback_scheduler_policy < 0)
                    null
                else
                    report.callback_scheduler_policy,
                .scheduler_priority = if (report.callback_scheduler_priority < 0)
                    null
                else
                    report.callback_scheduler_priority,
                .callbacks_count = report.callbacks_count,
                .missing_buffers_count = report.missing_buffers_count,
                .clipped_samples_count = report.clipped_samples_count,
                .header_metadata_buffers_count = report.header_metadata_buffers_count,
                .header_gap_buffers_count = report.header_gap_buffers_count,
                .header_gap_samples_count = report.header_gap_samples_count,
                .samples_range = if (report.block_samples_count_min == 0)
                    null
                else
                    .{
                        .minimum = report.block_samples_count_min,
                        .maximum = report.block_samples_count_max,
                    },
                .duration_ns_max = report.callback_duration_ns_max,
                .gap_ns_max = report.callback_gap_ns_max,
                .activity = .{
                    .activity = activity,
                    .activity_samples_count = report.activity.activity_samples_count,
                    .observed_samples_count = report.activity.observed_samples_count,
                    .unknown_samples_count = report.activity.unknown_samples_count,
                    .quiet_samples_count = report.activity.quiet_samples_count,
                    .active_samples_count = report.activity.active_samples_count,
                    .activity_changes_count = report.activity.changes_count,
                    .active_run_samples_count_max = report.activity.active_run_samples_count_max,
                    .quiet_run_samples_count_max = report.activity.quiet_run_samples_count_max,
                    .noise_floor_rms = report.activity.noise_floor_rms,
                    .quiet_threshold_rms = report.activity.quiet_threshold_rms,
                    .active_threshold_rms = report.activity.active_threshold_rms,
                },
            };
        const memory_lock: MemoryLockResult = if (report.shared_memory_is_locked == 1)
            .{ .locked = report.shared_memory_lock_limit_bytes }
        else
            .{ .unavailable = .{
                .error_code = report.shared_memory_lock_error_code,
                .limit_bytes = report.shared_memory_lock_limit_bytes,
            } };
        return .{ .captured = .{
            .end = end,
            .teardown_failure = teardown_failure,
            .memory_lock = memory_lock,
            .timeline_validation = timeline_validation,
            .pipewire_headers_version = report.pipewire_headers_version,
            .pipewire_headers_version_size = report.pipewire_headers_version_size,
            .pipewire_library_version = report.pipewire_library_version,
            .pipewire_library_version_size = report.pipewire_library_version_size,
            .pipewire_server_version = report.pipewire_server_version,
            .pipewire_server_version_size = report.pipewire_server_version_size,
            .source_identity = source,
            .negotiated_format = if (report.negotiated_sample_rate_hz == 0)
                null
            else
                .{
                    .sample_rate_hz = report.negotiated_sample_rate_hz,
                    .channels_count = report.negotiated_channels_count,
                },
            .samples_count = report.samples_count,
            .published_samples_count = report.published_samples_count,
            .slot_publications_count = report.slot_publications_count,
            .main_loop_thread_id = report.main_loop_thread_id,
            .callback = callback,
        } };
    }

    /// Receives and decodes one complete worker report without exposing the
    /// fixed transport record. Outer null means not ready; inner null means the
    /// socket closed before another report.
    pub fn receiveReportNonblocking(
        socket: std.posix.fd_t,
        exchange: *const AudioExchange,
        recording_samples_target: u32,
        automatic_stop: AutomaticStop,
    ) ??Result {
        var wire: WireReport = undefined;
        const result = std.os.linux.recvfrom(
            socket,
            std.mem.asBytes(&wire).ptr,
            @sizeOf(WireReport),
            std.os.linux.MSG.TRUNC | std.os.linux.MSG.DONTWAIT,
            null,
            null,
        );
        switch (std.os.linux.errno(result)) {
            .SUCCESS => {
                if (result == 0) {
                    return @as(?Result, null);
                }
                if (result != @sizeOf(WireReport)) {
                    return @as(?Result, .{ .err = .invalid_report });
                }
                const decoded = decodeReport(wire, exchange, recording_samples_target, automatic_stop) catch {
                    return @as(?Result, .{ .err = .invalid_report });
                };
                return serviceResult(decoded);
            },
            .AGAIN => {
                return null;
            },
            .CONNRESET => {
                return @as(?Result, null);
            },
            else => return @as(?Result, .{ .err = .invalid_report }),
        }
    }

    /// `run` enters one real PipeWire capture after exec. It binds its lifetime
    /// to the expected supervisor before receiving shared resources, then owns
    /// and closes every descriptor installed by `SCM_RIGHTS`.
    pub fn run(
        control_socket: std.posix.fd_t,
        expected_supervisor_pid: std.os.linux.pid_t,
    ) !void {
        assert(control_socket >= 0);
        assert(expected_supervisor_pid > 1);

        try bindWorkerLifetimeToSupervisor(expected_supervisor_pid);
        unblockServiceSignals();
        defer closeFileDescriptor(control_socket);

        var launch_packet: WireLaunch = undefined;
        var shared_descriptors = try descriptor_handoff.receive(
            control_socket,
            &launch_packet,
        );
        defer shared_descriptors.deinit();

        assert(shared_descriptors.values[0] != control_socket);
        assert(shared_descriptors.values[1] != control_socket);
        assert(shared_descriptors.values[0] != shared_descriptors.values[1]);
        try runAudioWorkerSession(
            control_socket,
            shared_descriptors.values[0],
            shared_descriptors.values[1],
            launch_packet,
        );
    }

    const WireLaunch = extern struct {
        version: u16,
        reserved: u16,
        session_id: u64,
        recording_samples_target: u32,
        automatic_stop: u8,
        source_kind: u8,
        source_size: u16,
        source: [target_name_capacity]u8,
    };

    fn buildLaunchPacket(options: LaunchOptions) WireLaunch {
        var launch_packet: WireLaunch = std.mem.zeroes(WireLaunch);
        launch_packet.version = protocol_version;
        launch_packet.session_id = options.session_id;
        launch_packet.recording_samples_target = options.recording_samples_target;
        launch_packet.automatic_stop = @intFromEnum(options.automatic_stop);
        launch_packet.source_kind = @intFromEnum(std.meta.activeTag(options.source));
        switch (options.source) {
            .default => {},
            .node_name, .device_serial => |source| {
                @memcpy(launch_packet.source[0..source.len], source);
                launch_packet.source_size = @intCast(source.len);
            },
        }
        return launch_packet;
    }

    fn decodeTrustedLaunch(wire: *const WireLaunch) LaunchOptions {
        assert(wire.version == protocol_version);
        assert(wire.reserved == 0);
        assert(wire.session_id > 0);
        assert(wire.recording_samples_target > 0);
        assert(wire.recording_samples_target <=
            std.math.maxInt(u32) - audio_exchange.callback_samples_count_max);
        assert(wire.automatic_stop <= @intFromEnum(AutomaticStop.after_quiet));
        assert(wire.source_size < wire.source.len);
        assert(std.mem.indexOfScalar(u8, wire.source[0..wire.source_size], 0) == null);
        assert(wire.source[wire.source_size] == 0);

        const source_text = wire.source[0..wire.source_size :0];
        const source: Source = switch (@as(std.meta.Tag(Source), @enumFromInt(wire.source_kind))) {
            .default => source: {
                assert(wire.source_size == 0);
                break :source .default;
            },
            .node_name => source: {
                assert(wire.source_size > 0);
                break :source .{ .node_name = source_text };
            },
            .device_serial => source: {
                assert(wire.source_size > 0);
                break :source .{ .device_serial = source_text };
            },
        };
        return .{
            .session_id = wire.session_id,
            .source = source,
            .recording_samples_target = wire.recording_samples_target,
            .automatic_stop = @enumFromInt(wire.automatic_stop),
        };
    }

    comptime {
        assert(@sizeOf(WireLaunch) == 280);
    }
};

// The fixed process report is value-only: strings own bounded arrays and no
// field contains an address from the worker. All PipeWire version observations
// survive encoding, including the separate version available on setup failure.
const WireReport = extern struct {
    worker_succeeded: u8,
    shared_memory_is_locked: u8,
    shared_memory_lock_error_code: i32,
    shared_memory_lock_limit_bytes: u64,
    outcome: u32,
    error_message_size: u16,
    runtime_error_stage: u32,
    runtime_error_domain: u32,
    runtime_error_code: i64,
    teardown_error_message_size: u16,
    teardown_error_stage: u32,
    teardown_error_domain: u32,
    teardown_error_code: i64,
    setup_error_stage: u32,
    setup_error_domain: u32,
    setup_error_code: i64,
    source_matches_count: u32,
    source_matches_present: u8,
    pipewire_version_size: u8,
    pipewire_version: [pipewire.pipewire_version_capacity]u8,

    timeline_validation: u32,
    pipewire_headers_version_size: u8,
    pipewire_headers_version: [pipewire.pipewire_version_capacity]u8,
    pipewire_library_version_size: u8,
    pipewire_library_version: [pipewire.pipewire_version_capacity]u8,
    pipewire_server_version_size: u8,
    pipewire_server_version: [pipewire.pipewire_version_capacity]u8,

    source_is_resolved: u8,
    source_node_id: u32,
    source_node_object_serial: u64,
    source_device_id: u32,
    source_device_object_serial: u64,
    source_node_name_size: u16,
    source_node_description_size: u16,
    source_device_serial_size: u16,
    source_device_description_size: u16,
    source_node_name: [pipewire.source_identity_text_capacity]u8,
    source_node_description: [pipewire.source_identity_text_capacity]u8,
    source_device_serial: [pipewire.source_identity_text_capacity]u8,
    source_device_description: [pipewire.source_identity_text_capacity]u8,

    negotiated_sample_rate_hz: u32,
    negotiated_channels_count: u32,
    samples_count: u32,
    published_samples_count: u32,
    slot_publications_count: u32,

    main_loop_thread_id: i32,
    callback_thread_id: i32,
    callback_scheduler_policy: i32,
    callback_scheduler_priority: i32,

    callbacks_count: u64,
    missing_buffers_count: u32,
    clipped_samples_count: u32,
    header_metadata_buffers_count: u32,
    header_gap_buffers_count: u32,
    header_gap_samples_count: u32,
    block_samples_count_min: u32,
    block_samples_count_max: u32,
    callback_duration_ns_max: u64,
    callback_gap_ns_max: u64,
    activity: WireActivityReport,

    error_message: [worker_error_message_capacity]u8,
    teardown_error_message: [pipewire.runtime_error_message_capacity]u8,
};

const WireActivityReport = extern struct {
    activity: u32,
    activity_samples_count: u32,
    observed_samples_count: u32,
    unknown_samples_count: u32,
    quiet_samples_count: u32,
    active_samples_count: u32,
    changes_count: u32,
    active_run_samples_count_max: u32,
    quiet_run_samples_count_max: u32,
    noise_floor_rms: f32,
    quiet_threshold_rms: f32,
    active_threshold_rms: f32,
};

const MappedAudioExchange = struct {
    bytes: []align(std.heap.page_size_min) u8,
    exchange: *AudioExchange,

    fn unmap(mapped: MappedAudioExchange) void {
        assert(mapped.bytes.len == @sizeOf(AudioExchange));
        assert(@intFromPtr(mapped.exchange) == @intFromPtr(mapped.bytes.ptr));
        std.posix.munmap(mapped.bytes);
    }
};

fn validateReportPacket(
    report: *const WireReport,
    recording_samples_target: u32,
    automatic_stop: AutomaticStop,
    exchange: *const AudioExchange,
) void {
    assert(report.worker_succeeded <= 1);
    assert(report.source_matches_present <= 1);
    if (report.source_matches_present == 0) assert(report.source_matches_count == 0);
    assert(report.shared_memory_is_locked <= 1);
    assert(report.error_message_size <= report.error_message.len);
    assert(report.teardown_error_message_size <= report.teardown_error_message.len);
    assert(report.pipewire_version_size <= report.pipewire_version.len);
    assert(report.pipewire_headers_version_size <= report.pipewire_headers_version.len);
    assert(report.pipewire_library_version_size <= report.pipewire_library_version.len);
    assert(report.pipewire_server_version_size <= report.pipewire_server_version.len);
    assert(report.source_is_resolved <= 1);
    assert(report.source_node_name_size <= report.source_node_name.len);
    assert(report.source_node_description_size <= report.source_node_description.len);
    assert(report.source_device_serial_size <= report.source_device_serial.len);
    assert(report.source_device_description_size <= report.source_device_description.len);
    assert(report.samples_count <= recording_samples_target + audio_exchange.callback_samples_count_max);
    assert(report.published_samples_count <= report.samples_count);

    if (report.worker_succeeded == 0) {
        assert(std.enums.fromInt(pipewire.SetupErrorStage, report.setup_error_stage) != null);
        assert(std.enums.fromInt(pipewire.ErrorDomain, report.setup_error_domain) != null);
        assert(report.setup_error_stage != @intFromEnum(pipewire.SetupErrorStage.none));
        assert(report.setup_error_domain != @intFromEnum(pipewire.ErrorDomain.none));
        assert(report.pipewire_version_size > 0);
        assert(report.error_message_size > 0);
        assert(report.shared_memory_is_locked == 0);
        assert(report.shared_memory_lock_error_code == 0);
        assert(report.shared_memory_lock_limit_bytes == 0);
        assert(report.runtime_error_stage == @intFromEnum(pipewire.RuntimeErrorStage.none));
        assert(report.runtime_error_domain == @intFromEnum(pipewire.ErrorDomain.none));
        assert(report.runtime_error_code == 0);
        assert(report.teardown_error_stage == @intFromEnum(pipewire.RuntimeErrorStage.none));
        assert(report.teardown_error_domain == @intFromEnum(pipewire.ErrorDomain.none));
        assert(report.teardown_error_code == 0);
        assert(report.teardown_error_message_size == 0);
        assert(report.pipewire_headers_version_size == 0);
        assert(report.pipewire_library_version_size == 0);
        assert(report.pipewire_server_version_size == 0);
        assert(report.source_is_resolved == 0);
        assert(report.negotiated_sample_rate_hz == 0);
        assert(report.negotiated_channels_count == 0);
        assert(report.samples_count == 0);
        assert(report.published_samples_count == 0);
        assert(report.slot_publications_count == 0);
        assert(report.callbacks_count == 0);
        assert(audio_exchange.acquireAudioCallbacksCount(exchange) == 0);
        assert(audio_exchange.acquireAudioSamplesCount(exchange) == 0);
        return;
    }

    assert(std.enums.fromInt(pipewire.TimelineValidation, report.timeline_validation) != null);
    assert(std.enums.fromInt(pipewire.Outcome, report.outcome) != null);
    assert(std.enums.fromInt(pipewire.Activity, report.activity.activity) != null);
    assert(std.enums.fromInt(pipewire.RuntimeErrorStage, report.runtime_error_stage) != null);
    assert(std.enums.fromInt(pipewire.ErrorDomain, report.runtime_error_domain) != null);
    assert(std.enums.fromInt(pipewire.RuntimeErrorStage, report.teardown_error_stage) != null);
    assert(std.enums.fromInt(pipewire.ErrorDomain, report.teardown_error_domain) != null);
    assert(report.setup_error_stage == @intFromEnum(pipewire.SetupErrorStage.none));
    assert(report.setup_error_domain == @intFromEnum(pipewire.ErrorDomain.none));
    assert(report.setup_error_code == 0);
    assert(report.pipewire_version_size == 0);
    assert(report.pipewire_headers_version_size > 0);
    assert(report.pipewire_library_version_size > 0);
    assert(report.main_loop_thread_id > 0);
    if (report.shared_memory_is_locked == 1) {
        assert(report.shared_memory_lock_error_code == 0);
    } else {
        assert(report.shared_memory_lock_error_code > 0);
    }

    const outcome: pipewire.Outcome = @enumFromInt(report.outcome);
    const runtime_stage: pipewire.RuntimeErrorStage =
        @enumFromInt(report.runtime_error_stage);
    const runtime_domain: pipewire.ErrorDomain =
        @enumFromInt(report.runtime_error_domain);
    switch (outcome) {
        .completed, .automatic_stop, .stopped, .cancelled => {
            assert(runtime_stage == .none);
            assert(runtime_domain == .none);
            assert(report.runtime_error_code == 0);
            assert(report.error_message_size == 0);
        },
        else => {
            assert(runtime_stage != .none);
            assert(runtime_domain != .none);
            assert(report.error_message_size > 0);
        },
    }

    const teardown_stage: pipewire.RuntimeErrorStage =
        @enumFromInt(report.teardown_error_stage);
    const teardown_domain: pipewire.ErrorDomain =
        @enumFromInt(report.teardown_error_domain);
    if (teardown_stage == .none) {
        assert(teardown_domain == .none);
        assert(report.teardown_error_code == 0);
        assert(report.teardown_error_message_size == 0);
    } else {
        assert(teardown_stage == .stream_disconnect);
        assert(teardown_domain == .pipewire_result);
        assert(report.teardown_error_code < 0);
        assert(report.teardown_error_message_size > 0);
    }

    assert(report.slot_publications_count <= report.published_samples_count);
    assert(report.missing_buffers_count <= report.callbacks_count);
    assert(report.clipped_samples_count <= report.samples_count);
    assert(report.header_metadata_buffers_count <= report.callbacks_count);
    assert(report.header_gap_buffers_count <= report.header_metadata_buffers_count);
    assert(report.header_gap_samples_count <= report.samples_count);
    // An absent callback has no detector measurements. Do not invent threshold
    // values in the encoder merely to satisfy checks on an absent observation.
    if (report.callback_thread_id != 0) {
        assert(report.activity.observed_samples_count == report.samples_count);
        assert(report.activity.observed_samples_count ==
            report.activity.unknown_samples_count +
                report.activity.quiet_samples_count +
                report.activity.active_samples_count);
        assert(report.activity.active_run_samples_count_max <=
            report.activity.active_samples_count);
        assert(report.activity.quiet_run_samples_count_max <=
            report.activity.quiet_samples_count);
        assert(std.math.isFinite(report.activity.noise_floor_rms));
        assert(std.math.isFinite(report.activity.quiet_threshold_rms));
        assert(std.math.isFinite(report.activity.active_threshold_rms));
        assert(report.activity.noise_floor_rms >= 0);
        assert(report.activity.quiet_threshold_rms >= 0);
        assert(report.activity.active_threshold_rms >= 0);
        assert(report.activity.quiet_threshold_rms <=
            report.activity.active_threshold_rms);
    }
    assert(report.callbacks_count == audio_exchange.acquireAudioCallbacksCount(exchange));
    assert(report.samples_count == audio_exchange.acquireAudioSamplesCount(exchange));
    if (report.callbacks_count == 0) {
        assert(report.callback_thread_id == 0);
    } else {
        assert(report.callback_thread_id > 0);
    }
    if (report.samples_count == 0) {
        assert(report.activity.activity == @intFromEnum(pipewire.Activity.unknown));
        assert(report.activity.activity_samples_count == 0);
        assert(report.published_samples_count == 0);
        assert(report.slot_publications_count == 0);
        assert(report.block_samples_count_min == 0);
        assert(report.block_samples_count_max == 0);
    } else {
        assert(report.activity.activity_samples_count > 0);
        if (report.activity.activity == @intFromEnum(pipewire.Activity.unknown)) {
            assert(report.activity.activity_samples_count ==
                report.activity.unknown_samples_count);
        } else if (report.activity.activity == @intFromEnum(pipewire.Activity.quiet)) {
            assert(report.activity.activity_samples_count <=
                report.activity.quiet_samples_count);
        } else {
            assert(report.activity.activity == @intFromEnum(pipewire.Activity.active));
            assert(report.activity.activity_samples_count <=
                report.activity.active_samples_count);
        }
        assert(report.source_is_resolved == 1);
        assert(report.callbacks_count > 0);
        assert(report.negotiated_sample_rate_hz == audio_exchange.sample_rate_hz);
        assert(report.negotiated_channels_count == audio_exchange.channels_count);
        assert(report.block_samples_count_min > 0);
        assert(report.block_samples_count_min <= report.block_samples_count_max);
        assert(report.block_samples_count_max <= audio_exchange.callback_samples_count_max);
        assert(report.block_samples_count_max <= audio_exchange.slot_samples_capacity);
    }

    switch (outcome) {
        .completed => {
            assert(report.samples_count >= recording_samples_target);
            assert(report.published_samples_count == report.samples_count);
        },
        .automatic_stop => {
            assert(automatic_stop == .after_quiet);
            assert(report.samples_count < recording_samples_target);
            assert(report.activity.active_samples_count > 0);
            assert(report.activity.activity == @intFromEnum(pipewire.Activity.quiet));
            assert(report.activity.activity_samples_count >=
                audio_policy.automatic_stop_quiet_samples_count);
            assert(report.samples_count - report.published_samples_count <=
                audio_policy.automatic_stop_quiet_samples_count +
                    audio_exchange.callback_samples_count_max);
        },
        .stopped => assert(report.published_samples_count == report.samples_count),
        .cancelled => assert(report.samples_count - report.published_samples_count <=
            audio_exchange.slot_samples_capacity),
        else => assert(report.published_samples_count == report.samples_count),
    }
    if (report.published_samples_count == 0) {
        assert(report.slot_publications_count == 0);
    } else {
        assert(report.slot_publications_count > 0);
    }
}

fn runAudioWorkerSession(
    supervisor_socket: std.posix.fd_t,
    exchange_fd: std.posix.fd_t,
    publication_event_fd: std.posix.fd_t,
    launch_packet: PipeWireWorker.WireLaunch,
) !void {
    assert(supervisor_socket >= 0);
    assert(exchange_fd >= 0);
    assert(publication_event_fd >= 0);
    assert(supervisor_socket != exchange_fd);
    assert(supervisor_socket != publication_event_fd);
    assert(exchange_fd != publication_event_fd);

    // Decode the fixed transport record once. Ordinary worker setup below uses
    // the logical source union, so a default source cannot carry stale text and
    // named source variants cannot exist without their required name.
    const launch = PipeWireWorker.decodeTrustedLaunch(&launch_packet);

    const mapped_exchange = try mapAudioExchange(exchange_fd);
    defer mapped_exchange.unmap();

    // The supervisor asserts the same postconditions immediately after it
    // initializes this trusted memfd. Repeat them here so the worker's launch
    // assumptions are visible without following an assertion helper.
    assert(mapped_exchange.exchange.version == audio_exchange.format_version);
    assert(mapped_exchange.exchange.timeline_validation_atomic == 0);
    assert(mapped_exchange.exchange.session_id == launch.session_id);
    assert(mapped_exchange.exchange.audio_callbacks_count_atomic == 0);
    assert(mapped_exchange.exchange.audio_samples_count_atomic == 0);
    assert(mapped_exchange.exchange.reserved_2 == 0);
    for (&mapped_exchange.exchange.slots) |*slot| {
        assert(@atomicLoad(
            u32,
            &slot.published_samples_count_atomic,
            .acquire,
        ) == 0);
        assert(slot.publication_ordinal == 0);
    }

    const worker_result = pipewire.run(.{
        .exchange = mapped_exchange.exchange,
        .publication_event_fd = publication_event_fd,
        .control_socket = supervisor_socket,
        .source = launch.source,
        .recording_samples_target = launch.recording_samples_target,
        .slot_samples_boundary = audio_exchange.slot_samples_capacity,
        .automatic_stop = launch.automatic_stop,
        .process_realtime = true,
    });

    var report_packet: WireReport = std.mem.zeroes(WireReport);
    switch (worker_result) {
        .captured => |capture| {
            report_packet.worker_succeeded = 1;
            report_packet.outcome = switch (capture.end) {
                .completed => @intFromEnum(pipewire.Outcome.completed),
                .automatic_stop => @intFromEnum(pipewire.Outcome.automatic_stop),
                .stopped => @intFromEnum(pipewire.Outcome.stopped),
                .cancelled => @intFromEnum(pipewire.Outcome.cancelled),
                .failed => |failure| failed: {
                    const detail = &failure.detail;
                    assert(detail.message_size > 0);
                    assert(detail.message_size <= detail.message.len);
                    report_packet.runtime_error_stage = @intFromEnum(detail.coordinate.stage);
                    report_packet.runtime_error_domain = @intFromEnum(detail.coordinate.domain);
                    report_packet.runtime_error_code = detail.coordinate.code;
                    report_packet.error_message_size = detail.message_size;
                    @memcpy(
                        report_packet.error_message[0..detail.message_size],
                        detail.message[0..detail.message_size],
                    );
                    break :failed @intFromEnum(failure.outcome);
                },
            };
            if (capture.teardown_failure) |failure| {
                assert(failure.message_size > 0);
                assert(failure.message_size <= failure.message.len);
                report_packet.teardown_error_stage = @intFromEnum(failure.coordinate.stage);
                report_packet.teardown_error_domain = @intFromEnum(failure.coordinate.domain);
                report_packet.teardown_error_code = failure.coordinate.code;
                report_packet.teardown_error_message_size = failure.message_size;
                @memcpy(
                    report_packet.teardown_error_message[0..failure.message_size],
                    failure.message[0..failure.message_size],
                );
            }
            switch (capture.memory_lock) {
                .locked => |limit| {
                    report_packet.shared_memory_is_locked = 1;
                    report_packet.shared_memory_lock_limit_bytes = limit;
                },
                .unavailable => |unavailable| {
                    report_packet.shared_memory_lock_error_code = unavailable.error_code;
                    report_packet.shared_memory_lock_limit_bytes = unavailable.limit_bytes;
                },
            }

            report_packet.timeline_validation = @intFromEnum(capture.timeline_validation);
            report_packet.pipewire_headers_version_size = capture.pipewire_headers_version_size;
            @memcpy(
                report_packet.pipewire_headers_version[0..capture.pipewire_headers_version_size],
                capture.pipewire_headers_version[0..capture.pipewire_headers_version_size],
            );
            report_packet.pipewire_library_version_size = capture.pipewire_library_version_size;
            @memcpy(
                report_packet.pipewire_library_version[0..capture.pipewire_library_version_size],
                capture.pipewire_library_version[0..capture.pipewire_library_version_size],
            );
            report_packet.pipewire_server_version_size = capture.pipewire_server_version_size;
            @memcpy(
                report_packet.pipewire_server_version[0..capture.pipewire_server_version_size],
                capture.pipewire_server_version[0..capture.pipewire_server_version_size],
            );
            report_packet.source_node_id = std.math.maxInt(u32);
            report_packet.source_device_id = std.math.maxInt(u32);
            if (capture.source_identity) |source| {
                report_packet.source_is_resolved = 1;
                report_packet.source_node_id = source.node_id;
                report_packet.source_node_object_serial = source.node_object_serial;
                report_packet.source_device_id = source.device_id;
                report_packet.source_device_object_serial = source.device_object_serial;
                report_packet.source_node_name_size = source.node_name_size;
                report_packet.source_node_description_size = source.node_description_size;
                report_packet.source_device_serial_size = source.device_serial_size;
                report_packet.source_device_description_size = source.device_description_size;
                @memcpy(
                    report_packet.source_node_name[0..source.node_name_size],
                    source.node_name[0..source.node_name_size],
                );
                @memcpy(
                    report_packet.source_node_description[0..source.node_description_size],
                    source.node_description[0..source.node_description_size],
                );
                @memcpy(
                    report_packet.source_device_serial[0..source.device_serial_size],
                    source.device_serial[0..source.device_serial_size],
                );
                @memcpy(
                    report_packet.source_device_description[0..source.device_description_size],
                    source.device_description[0..source.device_description_size],
                );
            }
            if (capture.negotiated_format) |format| {
                report_packet.negotiated_sample_rate_hz = format.sample_rate_hz;
                report_packet.negotiated_channels_count = format.channels_count;
            }
            report_packet.samples_count = capture.samples_count;
            report_packet.published_samples_count = capture.published_samples_count;
            report_packet.slot_publications_count = capture.slot_publications_count;
            report_packet.main_loop_thread_id = capture.main_loop_thread_id;
            report_packet.callback_scheduler_policy = -1;
            report_packet.callback_scheduler_priority = -1;
            report_packet.activity.activity = @intFromEnum(pipewire.Activity.unknown);
            if (capture.callback) |callback| {
                report_packet.callback_thread_id = callback.thread_id;
                report_packet.callback_scheduler_policy = callback.scheduler_policy orelse -1;
                report_packet.callback_scheduler_priority = callback.scheduler_priority orelse -1;
                report_packet.callbacks_count = callback.callbacks_count;
                report_packet.missing_buffers_count = callback.missing_buffers_count;
                report_packet.clipped_samples_count = callback.clipped_samples_count;
                report_packet.header_metadata_buffers_count = callback.header_metadata_buffers_count;
                report_packet.header_gap_buffers_count = callback.header_gap_buffers_count;
                report_packet.header_gap_samples_count = callback.header_gap_samples_count;
                if (callback.samples_range) |range| {
                    report_packet.block_samples_count_min = range.minimum;
                    report_packet.block_samples_count_max = range.maximum;
                }
                report_packet.callback_duration_ns_max = callback.duration_ns_max;
                report_packet.callback_gap_ns_max = callback.gap_ns_max;
                report_packet.activity = .{
                    .activity = @intFromEnum(callback.activity.activity),
                    .activity_samples_count = callback.activity.activity_samples_count,
                    .observed_samples_count = callback.activity.observed_samples_count,
                    .unknown_samples_count = callback.activity.unknown_samples_count,
                    .quiet_samples_count = callback.activity.quiet_samples_count,
                    .active_samples_count = callback.activity.active_samples_count,
                    .changes_count = callback.activity.activity_changes_count,
                    .active_run_samples_count_max = callback.activity.active_run_samples_count_max,
                    .quiet_run_samples_count_max = callback.activity.quiet_run_samples_count_max,
                    .noise_floor_rms = callback.activity.noise_floor_rms,
                    .quiet_threshold_rms = callback.activity.quiet_threshold_rms,
                    .active_threshold_rms = callback.activity.active_threshold_rms,
                };
            }
        },
        .setup_failed => |setup_error| {
            assert(setup_error.message_size > 0);
            assert(setup_error.message_size <= setup_error.message.len);
            assert(setup_error.pipewire_version_size > 0);
            assert(setup_error.pipewire_version_size <= setup_error.pipewire_version.len);

            report_packet.error_message_size = setup_error.message_size;
            @memcpy(
                report_packet.error_message[0..setup_error.message_size],
                setup_error.message[0..setup_error.message_size],
            );
            report_packet.setup_error_stage = @intFromEnum(setup_error.stage);
            report_packet.setup_error_domain = @intFromEnum(setup_error.domain);
            report_packet.setup_error_code = setup_error.code;
            report_packet.pipewire_version_size = setup_error.pipewire_version_size;
            @memcpy(
                report_packet.pipewire_version[0..setup_error.pipewire_version_size],
                setup_error.pipewire_version[0..setup_error.pipewire_version_size],
            );
        },
    }

    // Validate the exact transport record on the sending side. The supervisor
    // repeats the same trusted contract after receipt before exposing the report
    // to its caller.
    validateReportPacket(
        &report_packet,
        launch.recording_samples_target,
        launch.automatic_stop,
        mapped_exchange.exchange,
    );
    try sendPacket(supervisor_socket, std.mem.asBytes(&report_packet));
}

fn bindWorkerLifetimeToSupervisor(expected_supervisor_pid: std.os.linux.pid_t) !void {
    assert(expected_supervisor_pid > 1);

    const parent_death_signal_result = std.os.linux.prctl(
        @intFromEnum(std.os.linux.PR.SET_PDEATHSIG),
        @intFromEnum(std.os.linux.SIG.KILL),
        0,
        0,
        0,
    );
    if (std.os.linux.errno(parent_death_signal_result) != .SUCCESS) {
        return error.ParentDeathSignalConfigurationFailed;
    }

    if (std.os.linux.getppid() != expected_supervisor_pid) {
        return error.SupervisorExitedDuringWorkerLaunch;
    }
}

fn mapAudioExchange(exchange_fd: std.posix.fd_t) !MappedAudioExchange {
    assert(exchange_fd >= 0);

    // `mmap` can succeed when its backing file is shorter than the mapping and
    // then SIGBUS on access. The supervisor created and sealed this memfd, so
    // size and seal mismatches are internal protocol defects asserted on both
    // sides; syscall errors remain operating errors.
    var exchange_status: std.os.linux.Statx = undefined;
    const status_result = std.os.linux.statx(
        exchange_fd,
        "",
        std.os.linux.AT.EMPTY_PATH,
        .{ .SIZE = true },
        &exchange_status,
    );
    if (std.os.linux.errno(status_result) != .SUCCESS) {
        return error.AudioExchangeStatusFailed;
    }
    assert(exchange_status.mask.SIZE);
    assert(exchange_status.size == @sizeOf(AudioExchange));

    const required_seals = std.os.linux.F.SEAL_GROW |
        std.os.linux.F.SEAL_SHRINK |
        std.os.linux.F.SEAL_SEAL;
    const seals_result = std.os.linux.fcntl(
        exchange_fd,
        std.os.linux.F.GET_SEALS,
        0,
    );
    if (std.os.linux.errno(seals_result) != .SUCCESS) {
        return error.AudioExchangeSealStatusFailed;
    }
    assert(seals_result & required_seals == required_seals);

    const bytes = try std.posix.mmap(
        null,
        @sizeOf(AudioExchange),
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        exchange_fd,
        0,
    );
    const exchange: *AudioExchange = @ptrCast(@alignCast(bytes.ptr));
    assert(bytes.len == @sizeOf(AudioExchange));
    assert(@intFromPtr(exchange) % @alignOf(AudioExchange) == 0);

    return .{
        .bytes = bytes,
        .exchange = exchange,
    };
}

fn sendPacket(socket: std.posix.fd_t, packet: []const u8) !void {
    assert(socket >= 0);
    assert(packet.len > 0);

    while (true) {
        const send_result = std.os.linux.sendto(
            socket,
            packet.ptr,
            packet.len,
            std.os.linux.MSG.NOSIGNAL,
            null,
            0,
        );
        switch (std.os.linux.errno(send_result)) {
            .SUCCESS => {
                // SOCK_SEQPACKET sends one complete record or reports an error.
                // A successful short send violates the internal transport
                // contract established by the paired receiver assertion.
                assert(send_result == packet.len);
                return;
            },
            .INTR => continue,
            else => {
                return error.WorkerPacketSendFailed;
            },
        }
    }
}

fn unblockServiceSignals() void {
    var signal_mask = std.posix.sigemptyset();
    std.posix.sigaddset(&signal_mask, .TERM);
    std.posix.sigaddset(&signal_mask, .INT);
    std.posix.sigprocmask(std.posix.SIG.UNBLOCK, &signal_mask, null);
}

fn closeFileDescriptor(file_descriptor: std.posix.fd_t) void {
    assert(file_descriptor >= 0);
    const result = std.os.linux.close(file_descriptor);
    assert(std.os.linux.errno(result) == .SUCCESS);
}
