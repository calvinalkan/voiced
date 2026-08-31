//! Owns the complete process boundary around one PipeWire recording. The
//! supervisor-side API creates and maps the shared exchange, launches the audio
//! worker, transmits stop or cancel, consumes publications while supervising
//! deadlines, receives the final report, and contains a worker that does not
//! exit. The worker entry maps the same exchange after exec and runs exactly one
//! `pipewire.run` operation.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const audio_policy = @import("audio_policy.zig");
const pipewire = @import("pipewire.zig");
const descriptor_handoff = @import("descriptor_handoff.zig");
const assert = std.debug.assert;
const stderr = std.debug.print;

const Io = std.Io;
const Allocator = std.mem.Allocator;
const AudioExchange = audio_exchange.AudioExchange;

const target_name_capacity = 256;
const worker_error_message_capacity = pipewire.setup_error_message_capacity;

// These are process-contract vocabulary, re-exported so callers never need to
// import the worker-private PipeWire or exchange modules.
pub const ControlCommand = pipewire.ControlCommand;
pub const AutomaticStop = pipewire.AutomaticStop;
pub const Outcome = pipewire.Outcome;
pub const FailureOutcome = pipewire.FailureOutcome;
pub const TimelineValidation = pipewire.TimelineValidation;
pub const SetupErrorStage = pipewire.SetupErrorStage;
pub const SetupFailureStage = enum(u32) {
    file_descriptor_limits_query = @intFromEnum(SetupErrorStage.file_descriptor_limits_query),
    file_descriptor_limit = @intFromEnum(SetupErrorStage.file_descriptor_limit),
    memory_lock_limits_query = @intFromEnum(SetupErrorStage.memory_lock_limits_query),
    main_loop_create = @intFromEnum(SetupErrorStage.main_loop_create),
    callback_event_create = @intFromEnum(SetupErrorStage.callback_event_create),
    callback_event_register = @intFromEnum(SetupErrorStage.callback_event_register),
    control_socket_register = @intFromEnum(SetupErrorStage.control_socket_register),
    properties_create = @intFromEnum(SetupErrorStage.properties_create),
    property_config = @intFromEnum(SetupErrorStage.property_config),
    property_media_type = @intFromEnum(SetupErrorStage.property_media_type),
    property_media_category = @intFromEnum(SetupErrorStage.property_media_category),
    property_media_role = @intFromEnum(SetupErrorStage.property_media_role),
    property_target = @intFromEnum(SetupErrorStage.property_target),
    stream_create = @intFromEnum(SetupErrorStage.stream_create),
    source_resolution = @intFromEnum(SetupErrorStage.source_resolution),
    source_resolution_main_loop_create = @intFromEnum(SetupErrorStage.source_resolution_main_loop_create),
    source_resolution_context_create = @intFromEnum(SetupErrorStage.source_resolution_context_create),
    source_resolution_core_connect = @intFromEnum(SetupErrorStage.source_resolution_core_connect),
    source_resolution_core_observer_register = @intFromEnum(SetupErrorStage.source_resolution_core_observer_register),
    source_resolution_registry_create = @intFromEnum(SetupErrorStage.source_resolution_registry_create),
    source_resolution_registry_observer_register = @intFromEnum(SetupErrorStage.source_resolution_registry_observer_register),
    source_resolution_node_bind = @intFromEnum(SetupErrorStage.source_resolution_node_bind),
    source_resolution_device_bind = @intFromEnum(SetupErrorStage.source_resolution_device_bind),
    source_resolution_sync = @intFromEnum(SetupErrorStage.source_resolution_sync),
    source_resolution_main_loop = @intFromEnum(SetupErrorStage.source_resolution_main_loop),
    source_observer_create = @intFromEnum(SetupErrorStage.source_observer_create),
    source_registry_create = @intFromEnum(SetupErrorStage.source_registry_create),
    source_observer_register = @intFromEnum(SetupErrorStage.source_observer_register),
    server_observer_register = @intFromEnum(SetupErrorStage.server_observer_register),
    stream_connect = @intFromEnum(SetupErrorStage.stream_connect),
};
pub const RuntimeErrorStage = pipewire.RuntimeErrorStage;
pub const FailureStage = pipewire.FailureStage;
pub const ErrorDomain = pipewire.ErrorDomain;
pub const FailureDomain = pipewire.FailureDomain;
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
pub const audio_exchange_size = @sizeOf(AudioExchange);
pub const schedulerPolicyName = pipewire.schedulerPolicyName;

pub const Source = pipewire.Source;

pub const StartOptions = struct {
    session_id: u64,
    source: Source,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    automatic_stop: AutomaticStop = .disabled,
    consumer_delay_ms: u32,
    process_realtime: bool,
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

/// The fixed process report is intentionally value-only: every string owns a
/// bounded array, and no field contains an address from the worker process.
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
    teardown_error_message: [worker_error_message_capacity]u8,
};

/// Setup failure has no sample payload. Successful setup returns one logical
/// capture report and borrows ordered samples until `killAndReap`.
pub const Result = union(enum) {
    setup_failed: SetupFailure,
    captured: struct {
        report: CaptureReport,
        samples: []const i16,
        slots_consumed_before_final_report: u32,
    },
};

pub const Failure = struct {
    stage: FailureStage,
    domain: FailureDomain,
    code: i64,
    message: [worker_error_message_capacity]u8,
    message_size: u16,

    pub fn messageBytes(failure: *const Failure) []const u8 {
        return failure.message[0..failure.message_size];
    }
};

pub const SetupFailure = struct {
    stage: SetupFailureStage,
    domain: FailureDomain,
    code: i64,
    message: [worker_error_message_capacity]u8,
    message_size: u16,
    pipewire_version: [pipewire.pipewire_version_capacity]u8,
    pipewire_version_size: u8,
};

pub const MemoryLockResult = union(enum) {
    locked: u64,
    unavailable: struct {
        error_code: i32,
        limit_bytes: u64,
    },
};

pub const ResolvedSource = struct {
    node_id: u32,
    node_object_serial: u64,
    device_id: u32,
    device_object_serial: u64,
    node_name: [pipewire.source_identity_text_capacity]u8,
    node_name_size: u16,
    node_description: [pipewire.source_identity_text_capacity]u8,
    node_description_size: u16,
    device_serial: [pipewire.source_identity_text_capacity]u8,
    device_serial_size: u16,
    device_description: [pipewire.source_identity_text_capacity]u8,
    device_description_size: u16,
};

pub const CallbackSamplesRange = struct {
    minimum: u32,
    maximum: u32,
};

pub const ActivityReport = pipewire.ActivityReport;

pub const CallbackObservation = struct {
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

pub const CaptureFailure = struct {
    outcome: FailureOutcome,
    failure: Failure,
};

pub const CaptureEnd = union(enum) {
    completed,
    automatic_stop,
    stopped,
    cancelled,
    failed: CaptureFailure,
};

pub const NegotiatedFormat = struct {
    sample_rate_hz: u32,
    channels_count: u32,
};

pub const CaptureReport = struct {
    end: CaptureEnd,
    teardown_failure: ?Failure,
    memory_lock: MemoryLockResult,
    timeline_validation: TimelineValidation,
    pipewire_headers_version: [pipewire.pipewire_version_capacity]u8,
    pipewire_headers_version_size: u8,
    pipewire_library_version: [pipewire.pipewire_version_capacity]u8,
    pipewire_library_version_size: u8,
    pipewire_server_version: [pipewire.pipewire_version_capacity]u8,
    pipewire_server_version_size: u8,
    source: ?ResolvedSource,
    negotiated_format: ?NegotiatedFormat,
    samples_count: u32,
    published_samples_count: u32,
    slot_publications_count: u32,
    main_loop_thread_id: i32,
    callback: ?CallbackObservation,
};

/// The raw fixed record exists only at the process transport boundary. Ordinary
/// supervisors receive one of these payload-shaped logical alternatives.
pub const WorkerReport = union(enum) {
    setup_failed: SetupFailure,
    captured: CaptureReport,
};

pub const AudioProcess = opaque {};

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
            .PIPE, .CONNRESET => return error.AudioControlPeerClosed,
            else => return error.AudioControlSendFailed,
        }
    }
}

// The installed `audio-process` worker executable enters through this private
// function. Exporting it as the root entry is separate from the five operations
// imported by the spike and, later, the supervisor.
pub const main = workerMain;

/// `PipeWireWorker` is the real audio role used by the one-binary supervisor.
/// The supervisor sends bounded capture settings and transfers the audio memfd
/// plus publication eventfd in one launch message. The worker returns the same
/// structured final report already proven by the standalone audio process.
pub const PipeWireWorker = struct {
    pub const protocol_version: u16 = 3;

    pub const LaunchOptions = struct {
        session_id: u64,
        source: Source,
        recording_samples_target: u32,
        slot_samples_boundary: u32,
        automatic_stop: AutomaticStop,
        process_realtime: bool,
    };

    pub const WireLaunch = extern struct {
        version: u16,
        reserved: u16,
        session_id: u64,
        recording_samples_target: u32,
        slot_samples_boundary: u32,
        automatic_stop: u8,
        process_realtime: u8,
        source_kind: u8,
        reserved_2: u8,
        source_size: u16,
        source: [target_name_capacity]u8,
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
        assert(options.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
        assert(options.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
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

    /// `decodeTrustedReport` validates the fixed transport record and returns
    /// exactly one logical outcome. The worker is trusted process-internal code,
    /// so malformed records assert rather than becoming an external-input path.
    pub fn decodeTrustedReport(
        report: WireReport,
        exchange: *const AudioExchange,
        options: LaunchOptions,
    ) WorkerReport {
        const launch_packet = buildLaunchPacket(options);
        validateReportPacket(
            &report,
            launch_packet,
            options.recording_samples_target +
                audio_exchange.callback_samples_count_max,
            exchange,
        );
        if (report.worker_succeeded == 0) {
            return .{ .setup_failed = .{
                .stage = @enumFromInt(report.setup_error_stage),
                .domain = @enumFromInt(report.setup_error_domain),
                .code = report.setup_error_code,
                .message = report.error_message,
                .message_size = report.error_message_size,
                .pipewire_version = report.pipewire_version,
                .pipewire_version_size = report.pipewire_version_size,
            } };
        }

        const outcome: Outcome = @enumFromInt(report.outcome);
        const end: CaptureEnd = switch (outcome) {
            .completed => .completed,
            .automatic_stop => .automatic_stop,
            .stopped => .stopped,
            .cancelled => .cancelled,
            else => .{ .failed = .{
                .outcome = @enumFromInt(@intFromEnum(outcome)),
                .failure = .{
                    .stage = @enumFromInt(report.runtime_error_stage),
                    .domain = @enumFromInt(report.runtime_error_domain),
                    .code = report.runtime_error_code,
                    .message = report.error_message,
                    .message_size = report.error_message_size,
                },
            } },
        };
        const teardown_failure: ?Failure = if (report.teardown_error_stage == @intFromEnum(RuntimeErrorStage.none))
            null
        else
            .{
                .stage = @enumFromInt(report.teardown_error_stage),
                .domain = @enumFromInt(report.teardown_error_domain),
                .code = report.teardown_error_code,
                .message = report.teardown_error_message,
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
                    .activity = @enumFromInt(report.activity.activity),
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
            .timeline_validation = @enumFromInt(report.timeline_validation),
            .pipewire_headers_version = report.pipewire_headers_version,
            .pipewire_headers_version_size = report.pipewire_headers_version_size,
            .pipewire_library_version = report.pipewire_library_version,
            .pipewire_library_version_size = report.pipewire_library_version_size,
            .pipewire_server_version = report.pipewire_server_version,
            .pipewire_server_version_size = report.pipewire_server_version_size,
            .source = source,
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
        options: LaunchOptions,
    ) ??WorkerReport {
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
                if (result == 0) return @as(?WorkerReport, null);
                assert(result == @sizeOf(WireReport));
                return decodeTrustedReport(wire, exchange, options);
            },
            .AGAIN => return null,
            .CONNRESET => return @as(?WorkerReport, null),
            else => @trap(),
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

    fn buildLaunchPacket(options: LaunchOptions) WireLaunch {
        var launch_packet: WireLaunch = std.mem.zeroes(WireLaunch);
        launch_packet.version = protocol_version;
        launch_packet.session_id = options.session_id;
        launch_packet.recording_samples_target = options.recording_samples_target;
        launch_packet.slot_samples_boundary = options.slot_samples_boundary;
        launch_packet.automatic_stop = @intFromEnum(options.automatic_stop);
        launch_packet.process_realtime = @intFromBool(options.process_realtime);
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
        assert(wire.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
        assert(wire.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
        assert(wire.automatic_stop <= @intFromEnum(AutomaticStop.after_quiet));
        assert(wire.process_realtime <= 1);
        assert(wire.reserved_2 == 0);
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
            .slot_samples_boundary = wire.slot_samples_boundary,
            .automatic_stop = @enumFromInt(wire.automatic_stop),
            .process_realtime = wire.process_realtime == 1,
        };
    }

    comptime {
        assert(@sizeOf(WireLaunch) == 288);
    }
};

const WorkerLaunchPacket = PipeWireWorker.WireLaunch;

/// `FakeWorker` drives the production process and shared-exchange contracts
/// without PipeWire. The supervisor uses it to establish lifecycle, deadline,
/// cancellation, slot-pressure, and transcription behavior deterministically.
pub const FakeWorker = struct {
    pub const protocol_version: u16 = 1;

    pub const LaunchOptions = struct {
        session_id: u64,
        chunks_count: u32,
        publication_interval_ms: u32,
    };

    const WireLaunch = extern struct {
        version: u16,
        reserved: u16,
        session_id: u64,
        chunks_count: u32,
        publication_interval_ms: u32,
    };

    pub const ReportKind = enum(u16) {
        ready,
        completed,
        pipeline_full,
        cancelled,
    };

    pub const ReportPacket = extern struct {
        kind: u16,
        reserved: u16,
    };

    comptime {
        assert(@sizeOf(WireLaunch) == 24);
        assert(@sizeOf(ReportPacket) == 4);
    }

    pub fn sendLaunch(
        socket: std.posix.fd_t,
        audio_exchange_fd: std.posix.fd_t,
        publication_event_fd: std.posix.fd_t,
        options: LaunchOptions,
    ) !void {
        assert(options.session_id > 0);
        assert(options.chunks_count > 0);
        assert(options.chunks_count <= 32);
        const wire: WireLaunch = .{
            .version = protocol_version,
            .reserved = 0,
            .session_id = options.session_id,
            .chunks_count = options.chunks_count,
            .publication_interval_ms = options.publication_interval_ms,
        };
        const descriptors = [_]std.posix.fd_t{
            audio_exchange_fd,
            publication_event_fd,
        };
        try descriptor_handoff.send(socket, &wire, &descriptors);
    }

    pub fn decodeTrustedReport(report: ReportPacket) ReportKind {
        assert(report.reserved == 0);
        return @enumFromInt(report.kind);
    }

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

        assert(launch_packet.version == protocol_version);
        assert(launch_packet.reserved == 0);
        assert(launch_packet.session_id > 0);
        assert(launch_packet.chunks_count > 0);
        assert(launch_packet.chunks_count <= 32);

        const mapped_exchange = try mapAudioExchange(shared_descriptors.values[0]);
        defer mapped_exchange.unmap();
        const exchange = mapped_exchange.exchange;
        assert(exchange.version == audio_exchange.format_version);
        assert(exchange.session_id == launch_packet.session_id);

        try sendFakeReport(control_socket, .ready);
        for (0..launch_packet.chunks_count) |publication_ordinal_usize| {
            if (receiveFakeControl(control_socket)) |command| {
                assert(command == .cancel);
                try sendFakeReport(control_socket, .cancelled);
                return;
            }

            const publication_ordinal: u32 = @intCast(publication_ordinal_usize);
            var selected_writer: ?audio_exchange.SlotWriter = null;
            const preferred_slot_index = publication_ordinal % audio_exchange.slots_count;
            for (0..audio_exchange.slots_count) |slot_offset| {
                const slot_index = (preferred_slot_index + slot_offset) %
                    audio_exchange.slots_count;
                selected_writer = audio_exchange.tryAcquireWriter(
                    exchange,
                    audio_exchange.SlotIndex.fromArrayIndex(slot_index),
                    publication_ordinal,
                );
                if (selected_writer != null) break;
            }
            const writer = selected_writer orelse {
                try sendFakeReport(control_socket, .pipeline_full);
                return;
            };

            const fake_samples_count: u32 = audio_exchange.sample_rate_hz;
            for (writer.slot.samples[0..fake_samples_count], 0..) |*sample, sample_index| {
                sample.* = @intCast(sample_index % 100);
            }
            audio_exchange.publishWrittenSlot(writer, .{
                .samples_count = fake_samples_count,
                .contains_activity = true,
            });
            writeFakePublicationEvent(shared_descriptors.values[1]);

            for (0..launch_packet.publication_interval_ms) |_| {
                sleepMilliseconds(1);
                if (receiveFakeControl(control_socket)) |command| {
                    assert(command == .cancel);
                    try sendFakeReport(control_socket, .cancelled);
                    return;
                }
            }
        }

        try sendFakeReport(control_socket, .completed);
    }

    fn sendFakeReport(socket: std.posix.fd_t, kind: ReportKind) !void {
        const report: ReportPacket = .{
            .kind = @intFromEnum(kind),
            .reserved = 0,
        };
        try sendPacket(socket, std.mem.asBytes(&report));
    }

    fn receiveFakeControl(socket: std.posix.fd_t) ?pipewire.ControlCommand {
        var control_packet: pipewire.ControlPacket = undefined;
        const receive_result = std.os.linux.recvfrom(
            socket,
            std.mem.asBytes(&control_packet).ptr,
            @sizeOf(pipewire.ControlPacket),
            std.os.linux.MSG.TRUNC | std.os.linux.MSG.DONTWAIT,
            null,
            null,
        );
        switch (std.os.linux.errno(receive_result)) {
            .SUCCESS => {
                assert(receive_result == @sizeOf(pipewire.ControlPacket));
                assert(control_packet.reserved == 0);
                return @enumFromInt(control_packet.command);
            },
            .AGAIN => return null,
            else => @trap(),
        }
    }

    fn writeFakePublicationEvent(event_fd: std.posix.fd_t) void {
        const increment: u64 = 1;
        const write_result = std.os.linux.write(
            event_fd,
            std.mem.asBytes(&increment).ptr,
            @sizeOf(u64),
        );
        assert(std.os.linux.errno(write_result) == .SUCCESS);
        assert(write_result == @sizeOf(u64));
    }
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

const PendingPublishedSlot = struct {
    index: audio_exchange.SlotIndex,
    ready_at_monotonic_ns: u64,
};

const PublishedAudioConsumer = struct {
    samples: []i16,
    samples_count: usize,
    next_publication_ordinal: u32,
    pending: ?PendingPublishedSlot,
    slots_consumed_before_final_report: u32,
};

const RequestedControl = struct {
    command: pipewire.ControlCommand,
    deadline_monotonic_ns: u64,
};

const SupervisionState = union(enum) {
    awaiting_callbacks: u64,
    recording: u64,
    control_requested: RequestedControl,
    reported,
};

const ProgressObservation = struct {
    callbacks_count: u64,
    samples_count: u32,
};

const Process = struct {
    allocator: Allocator,
    io: Io,
    exchange_fd: std.posix.fd_t,
    mapped_exchange: MappedAudioExchange,
    publication_event_fd: std.posix.fd_t,
    supervisor_socket: std.posix.fd_t,
    worker: std.process.Child,
    launch_packet: WorkerLaunchPacket,
    consumer_delay_ms: u32,
    consumer: PublishedAudioConsumer,

    recording_deadline_ns: u64,
    supervision: SupervisionState,
    progress: ProgressObservation,
};

/// Creates the shared exchange, launches one isolated worker, sends its fixed
/// launch record, and returns an opaque handle owned by `killAndReap`.
pub fn start(
    init: std.process.Init,
    options: StartOptions,
) !*AudioProcess {
    assert(options.session_id > 0);
    assert(options.recording_samples_target > 0);
    assert(options.recording_samples_target <=
        std.math.maxInt(u32) - audio_exchange.callback_samples_count_max);
    assert(options.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(options.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(options.consumer_delay_ms <= 10_000);
    switch (options.source) {
        .default => {},
        .node_name, .device_serial => |source| {
            assert(source.len > 0);
            assert(source.len < target_name_capacity);
        },
    }

    const exchange_fd = try std.posix.memfd_create(
        "voiced-audio",
        std.os.linux.MFD.ALLOW_SEALING,
    );
    var exchange_fd_is_owned = true;
    errdefer if (exchange_fd_is_owned) closeFileDescriptor(exchange_fd);

    const truncate_result = std.os.linux.ftruncate(
        exchange_fd,
        @intCast(@sizeOf(AudioExchange)),
    );
    if (std.os.linux.errno(truncate_result) != .SUCCESS) {
        return error.AudioExchangeResizeFailed;
    }

    const required_seals = std.os.linux.F.SEAL_GROW |
        std.os.linux.F.SEAL_SHRINK |
        std.os.linux.F.SEAL_SEAL;
    const seal_result = std.os.linux.fcntl(
        exchange_fd,
        std.os.linux.F.ADD_SEALS,
        required_seals,
    );
    if (std.os.linux.errno(seal_result) != .SUCCESS) {
        return error.AudioExchangeSealFailed;
    }

    const mapped_exchange = try mapAudioExchange(exchange_fd);
    var exchange_is_mapped = true;
    errdefer if (exchange_is_mapped) mapped_exchange.unmap();
    audio_exchange.initialize(mapped_exchange.exchange, options.session_id);

    const publication_event_result = std.os.linux.eventfd(
        0,
        std.os.linux.EFD.NONBLOCK,
    );
    if (std.os.linux.errno(publication_event_result) != .SUCCESS) {
        return error.AudioPublicationEventCreationFailed;
    }
    const publication_event_fd: std.posix.fd_t = @intCast(publication_event_result);
    var publication_event_fd_is_owned = true;
    errdefer if (publication_event_fd_is_owned) closeFileDescriptor(publication_event_fd);

    var sockets: [2]std.posix.fd_t = undefined;
    const socket_pair_result = std.os.linux.socketpair(
        std.os.linux.AF.UNIX,
        std.os.linux.SOCK.SEQPACKET,
        0,
        &sockets,
    );
    if (std.os.linux.errno(socket_pair_result) != .SUCCESS) {
        return error.WorkerSocketCreationFailed;
    }
    const supervisor_socket = sockets[0];
    const worker_socket = sockets[1];
    var supervisor_socket_is_owned = true;
    var worker_socket_is_owned = true;
    errdefer if (supervisor_socket_is_owned) closeFileDescriptor(supervisor_socket);
    errdefer if (worker_socket_is_owned) closeFileDescriptor(worker_socket);

    assert(supervisor_socket >= 0);
    assert(worker_socket >= 0);
    assert(supervisor_socket != worker_socket);
    const close_on_exec_result = std.os.linux.fcntl(
        supervisor_socket,
        std.os.linux.F.SETFD,
        std.os.linux.FD_CLOEXEC,
    );
    if (std.os.linux.errno(close_on_exec_result) != .SUCCESS) {
        return error.WorkerSocketConfigurationFailed;
    }

    var worker_socket_text_buffer: [32]u8 = undefined;
    const worker_socket_text = try std.fmt.bufPrint(
        &worker_socket_text_buffer,
        "{d}",
        .{worker_socket},
    );
    var exchange_fd_text_buffer: [32]u8 = undefined;
    const exchange_fd_text = try std.fmt.bufPrint(
        &exchange_fd_text_buffer,
        "{d}",
        .{exchange_fd},
    );
    var publication_event_fd_text_buffer: [32]u8 = undefined;
    const publication_event_fd_text = try std.fmt.bufPrint(
        &publication_event_fd_text_buffer,
        "{d}",
        .{publication_event_fd},
    );
    var supervisor_pid_text_buffer: [32]u8 = undefined;
    const supervisor_pid_text = try std.fmt.bufPrint(
        &supervisor_pid_text_buffer,
        "{d}",
        .{std.os.linux.getpid()},
    );

    // Both installed audio artifacts are siblings. Resolve the running spike or
    // supervisor through `/proc/self/exe`, then launch the dedicated worker
    // entry without asking the caller to know an installation path or internal
    // role arguments.
    var executable_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const executable_path_size = try Io.Dir.readLinkAbsolute(
        init.io,
        "/proc/self/exe",
        &executable_path_buffer,
    );
    const executable_directory = std.fs.path.dirname(
        executable_path_buffer[0..executable_path_size],
    ) orelse return error.AudioExecutableDirectoryUnavailable;
    var worker_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const worker_path = try std.fmt.bufPrint(
        &worker_path_buffer,
        "{s}/audio-process",
        .{executable_directory},
    );

    var worker = try std.process.spawn(init.io, .{
        .argv = &.{
            worker_path,
            worker_socket_text,
            exchange_fd_text,
            publication_event_fd_text,
            supervisor_pid_text,
        },
    });
    errdefer forceStopWorker(&worker, init.io) catch {};
    closeFileDescriptor(worker_socket);
    worker_socket_is_owned = false;

    const launch_packet = PipeWireWorker.buildLaunchPacket(.{
        .session_id = options.session_id,
        .source = options.source,
        .recording_samples_target = options.recording_samples_target,
        .slot_samples_boundary = options.slot_samples_boundary,
        .automatic_stop = options.automatic_stop,
        .process_realtime = options.process_realtime,
    });

    assert(launch_packet.session_id > 0);
    assert(launch_packet.recording_samples_target > 0);
    assert(launch_packet.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(launch_packet.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(launch_packet.automatic_stop <= @intFromEnum(AutomaticStop.after_quiet));
    assert(launch_packet.process_realtime <= 1);
    assert(launch_packet.source_size < launch_packet.source.len);
    assert(launch_packet.source[launch_packet.source_size] == 0);
    try sendPacket(supervisor_socket, std.mem.asBytes(&launch_packet));

    const samples_capacity = options.recording_samples_target +
        audio_exchange.callback_samples_count_max;
    const samples = try init.gpa.alloc(i16, samples_capacity);
    errdefer init.gpa.free(samples);

    const supervision_started_ns = monotonicNanoseconds();
    const recording_seconds =
        (options.recording_samples_target + audio_exchange.sample_rate_hz - 1) /
        audio_exchange.sample_rate_hz;
    const process = try init.gpa.create(Process);
    process.* = .{
        .allocator = init.gpa,
        .io = init.io,
        .exchange_fd = exchange_fd,
        .mapped_exchange = mapped_exchange,
        .publication_event_fd = publication_event_fd,
        .supervisor_socket = supervisor_socket,
        .worker = worker,
        .launch_packet = launch_packet,
        .consumer_delay_ms = options.consumer_delay_ms,
        .consumer = .{
            .samples = samples,
            .samples_count = 0,
            .next_publication_ordinal = 0,
            .pending = null,
            .slots_consumed_before_final_report = 0,
        },
        .recording_deadline_ns = supervision_started_ns +
            (@as(u64, recording_seconds) + 10) * std.time.ns_per_s,
        .supervision = .{
            .awaiting_callbacks = supervision_started_ns + 3 * std.time.ns_per_s,
        },
        .progress = .{
            .callbacks_count = 0,
            .samples_count = 0,
        },
    };

    exchange_fd_is_owned = false;
    exchange_is_mapped = false;
    publication_event_fd_is_owned = false;
    supervisor_socket_is_owned = false;
    return @ptrCast(process);
}

/// Sends the recording-preserving terminal command exactly once.
pub fn requestStop(process_opaque: *AudioProcess) !void {
    try requestControl(processImplementation(process_opaque), .stop);
}

/// Sends the discard-only terminal command exactly once.
pub fn requestCancel(process_opaque: *AudioProcess) !void {
    try requestControl(processImplementation(process_opaque), .cancel);
}

/// Advances publication consumption and process supervision for at most
/// `timeout_ms`. `null` means the worker remains active. A returned result owns
/// its report and borrows only its sample slice from the process.
pub fn receiveReport(
    process_opaque: *AudioProcess,
    timeout_ms: u16,
) !?Result {
    const process = processImplementation(process_opaque);
    assert(timeout_ms > 0);
    assert(process.worker.id != null);
    assert(process.supervision != .reported);

    const now_ns = monotonicNanoseconds();
    while (consumeNextPublishedSlot(
        &process.consumer,
        process.mapped_exchange.exchange,
        process.launch_packet.session_id,
        process.launch_packet.slot_samples_boundary,
        process.consumer_delay_ms,
        now_ns,
        false,
    )) {
        process.consumer.slots_consumed_before_final_report += 1;
    }

    var poll_descriptors = [_]std.posix.pollfd{
        .{
            .fd = process.supervisor_socket,
            .events = std.posix.POLL.IN,
            .revents = 0,
        },
        .{
            .fd = process.publication_event_fd,
            .events = std.posix.POLL.IN,
            .revents = 0,
        },
    };
    if (try std.posix.poll(&poll_descriptors, timeout_ms) > 0) {
        if (poll_descriptors[1].revents & std.posix.POLL.IN != 0) {
            _ = try readEventCounter(process.publication_event_fd);
            while (consumeNextPublishedSlot(
                &process.consumer,
                process.mapped_exchange.exchange,
                process.launch_packet.session_id,
                process.launch_packet.slot_samples_boundary,
                process.consumer_delay_ms,
                monotonicNanoseconds(),
                false,
            )) {
                process.consumer.slots_consumed_before_final_report += 1;
            }
        }
        if (poll_descriptors[1].revents &
            (std.posix.POLL.ERR | std.posix.POLL.HUP | std.posix.POLL.NVAL) != 0)
        {
            return error.AudioPublicationEventFailed;
        }

        if (poll_descriptors[0].revents & std.posix.POLL.IN != 0) {
            var report: WireReport = undefined;
            receivePacket(
                process.supervisor_socket,
                std.mem.asBytes(&report),
            ) catch |packet_error| switch (packet_error) {
                error.WorkerSocketClosed => {
                    try reportUnexpectedWorkerExit(&process.worker, process.io);
                },
                else => return packet_error,
            };
            return try finishReport(process, report);
        }
        if (poll_descriptors[0].revents &
            (std.posix.POLL.ERR | std.posix.POLL.HUP | std.posix.POLL.NVAL) != 0)
        {
            try reportUnexpectedWorkerExit(&process.worker, process.io);
        }
    }

    const deadline_now_ns = monotonicNanoseconds();
    const callbacks_count = audio_exchange.acquireAudioCallbacksCount(
        process.mapped_exchange.exchange,
    );
    const callbacks_count_previous = process.progress.callbacks_count;
    assert(callbacks_count >= callbacks_count_previous);
    process.progress.callbacks_count = callbacks_count;
    if (callbacks_count > callbacks_count_previous) {
        switch (process.supervision) {
            .awaiting_callbacks, .recording => process.supervision = .{
                .recording = deadline_now_ns + 2 * std.time.ns_per_s,
            },
            .control_requested => {},
            .reported => unreachable,
        }
    }

    const samples_count = audio_exchange.acquireAudioSamplesCount(
        process.mapped_exchange.exchange,
    );
    assert(samples_count >= process.progress.samples_count);
    assert(samples_count <= process.consumer.samples.len);
    process.progress.samples_count = samples_count;

    if (process.supervision == .control_requested) {
        const control = process.supervision.control_requested;
        if (deadline_now_ns >= control.deadline_monotonic_ns) {
            stderr(
                "Audio worker control deadline exceeded\n" ++
                    "Command: {s}\n" ++
                    "Deadline: 2000 ms\n" ++
                    "Action: forced termination\n" ++
                    "Callbacks observed: {d}\n" ++
                    "Samples observed: {d}\n",
                .{
                    @tagName(control.command),
                    process.progress.callbacks_count,
                    process.progress.samples_count,
                },
            );
            return error.AudioWorkerControlDeadlineExceeded;
        }
        return null;
    }

    const requested_target = if (process.launch_packet.source_size > 0)
        process.launch_packet.source[0..process.launch_packet.source_size]
    else
        "default source";
    const processing_mode = if (process.launch_packet.process_realtime == 1)
        "realtime"
    else
        "main loop";

    if (process.supervision == .awaiting_callbacks and
        deadline_now_ns >= process.supervision.awaiting_callbacks)
    {
        stderr(
            "Audio worker deadline exceeded\n" ++
                "Stage: waiting_for_callback_progress\n" ++
                "Deadline: 3000 ms\n" ++
                "Target: {s}\n" ++
                "Processing: {s}\n" ++
                "Callbacks observed: {d}\n" ++
                "Samples observed: {d}\n",
            .{
                requested_target,
                processing_mode,
                process.progress.callbacks_count,
                process.progress.samples_count,
            },
        );
        return error.AudioWorkerSetupDeadlineExceeded;
    }
    if (process.supervision == .recording and
        deadline_now_ns >= process.supervision.recording)
    {
        stderr(
            "Audio worker deadline exceeded\n" ++
                "Stage: waiting_for_callback_progress\n" ++
                "Deadline: 2000 ms\n" ++
                "Target: {s}\n" ++
                "Processing: {s}\n" ++
                "Callbacks observed: {d}\n" ++
                "Samples observed: {d}\n",
            .{
                requested_target,
                processing_mode,
                process.progress.callbacks_count,
                process.progress.samples_count,
            },
        );
        return error.AudioWorkerProgressDeadlineExceeded;
    }
    if (deadline_now_ns >= process.recording_deadline_ns) {
        stderr(
            "Audio worker deadline exceeded\n" ++
                "Stage: recording_containment\n" ++
                "Target: {s}\n" ++
                "Processing: {s}\n" ++
                "Callbacks observed: {d}\n" ++
                "Samples observed: {d}\n",
            .{
                requested_target,
                processing_mode,
                process.progress.callbacks_count,
                process.progress.samples_count,
            },
        );
        return error.AudioWorkerRecordingDeadlineExceeded;
    }
    return null;
}

/// Kills and reaps a still-running child, then releases every descriptor,
/// mapping, sample buffer, and allocation owned by the opaque handle. Calling it
/// after a normal report only performs the resource-release half.
pub fn killAndReap(process_opaque: *AudioProcess) !void {
    const process = processImplementation(process_opaque);
    const allocator = process.allocator;
    var stop_error: ?anyerror = null;
    if (process.worker.id != null) {
        forceStopWorker(&process.worker, process.io) catch |err| {
            stop_error = err;
        };
    }

    // Resource ownership ends even when forced termination itself reports an
    // operating error. Returning early would leak the mapping and descriptors
    // while also hiding which cleanup operation still requires diagnosis.
    closeFileDescriptor(process.supervisor_socket);
    closeFileDescriptor(process.publication_event_fd);
    process.mapped_exchange.unmap();
    closeFileDescriptor(process.exchange_fd);
    allocator.free(process.consumer.samples);
    allocator.destroy(process);

    if (stop_error) |err| return err;
}

fn processImplementation(process: *AudioProcess) *Process {
    return @ptrCast(@alignCast(process));
}

fn requestControl(process: *Process, command: pipewire.ControlCommand) !void {
    assert(process.worker.id != null);
    assert(process.supervision == .awaiting_callbacks or
        process.supervision == .recording);

    try sendControl(process.supervisor_socket, command);
    process.supervision = .{ .control_requested = .{
        .command = command,
        .deadline_monotonic_ns = monotonicNanoseconds() + 2 * std.time.ns_per_s,
    } };

    assert(process.supervision.control_requested.command == command);
    assert(process.supervision.control_requested.deadline_monotonic_ns > 0);
}

fn finishReport(process: *Process, report: WireReport) !Result {
    validateReportPacket(
        &report,
        process.launch_packet,
        process.consumer.samples.len,
        process.mapped_exchange.exchange,
    );

    const launch = PipeWireWorker.decodeTrustedLaunch(&process.launch_packet);
    const worker_report = PipeWireWorker.decodeTrustedReport(
        report,
        process.mapped_exchange.exchange,
        launch,
    );
    const publications_count = switch (worker_report) {
        .setup_failed => 0,
        .captured => |capture| capture.slot_publications_count,
    };
    const published_samples_count = switch (worker_report) {
        .setup_failed => 0,
        .captured => |capture| capture.published_samples_count,
    };

    while (process.consumer.next_publication_ordinal < publications_count) {
        assert(consumeNextPublishedSlot(
            &process.consumer,
            process.mapped_exchange.exchange,
            process.launch_packet.session_id,
            process.launch_packet.slot_samples_boundary,
            process.consumer_delay_ms,
            monotonicNanoseconds(),
            true,
        ));
    }
    assert(process.consumer.samples_count == published_samples_count);

    const worker_term = try waitForWorkerExit(&process.worker, process.io, 1_000);
    switch (worker_term) {
        .exited => |exit_code| assert(exit_code == 0),
        else => unreachable,
    }
    assert(process.worker.id == null);
    process.supervision = .reported;

    return switch (worker_report) {
        .setup_failed => |failure| .{ .setup_failed = failure },
        .captured => |capture| .{ .captured = .{
            .report = capture,
            .samples = process.consumer.samples[0..process.consumer.samples_count],
            .slots_consumed_before_final_report = process.consumer.slots_consumed_before_final_report,
        } },
    };
}

fn validateReportPacket(
    report: *const WireReport,
    launch: WorkerLaunchPacket,
    samples_capacity: usize,
    exchange: *const AudioExchange,
) void {
    assert(report.worker_succeeded <= 1);
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
    assert(report.samples_count <= samples_capacity);
    assert(report.published_samples_count <= report.samples_count);

    if (report.worker_succeeded == 0) {
        var setup_stage_is_valid = false;
        inline for (@typeInfo(pipewire.SetupErrorStage).@"enum".fields) |field| {
            if (report.setup_error_stage == field.value) setup_stage_is_valid = true;
        }
        var setup_domain_is_valid = false;
        inline for (@typeInfo(pipewire.ErrorDomain).@"enum".fields) |field| {
            if (report.setup_error_domain == field.value) setup_domain_is_valid = true;
        }

        assert(setup_stage_is_valid);
        assert(setup_domain_is_valid);
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

    var timeline_validation_is_valid = false;
    inline for (@typeInfo(pipewire.TimelineValidation).@"enum".fields) |field| {
        if (report.timeline_validation == field.value) {
            timeline_validation_is_valid = true;
        }
    }
    var outcome_is_valid = false;
    inline for (@typeInfo(pipewire.Outcome).@"enum".fields) |field| {
        if (report.outcome == field.value) outcome_is_valid = true;
    }
    var activity_is_valid = false;
    inline for (@typeInfo(pipewire.Activity).@"enum".fields) |field| {
        if (report.activity.activity == field.value) activity_is_valid = true;
    }
    var runtime_stage_is_valid = false;
    inline for (@typeInfo(pipewire.RuntimeErrorStage).@"enum".fields) |field| {
        if (report.runtime_error_stage == field.value) runtime_stage_is_valid = true;
    }
    var runtime_domain_is_valid = false;
    inline for (@typeInfo(pipewire.ErrorDomain).@"enum".fields) |field| {
        if (report.runtime_error_domain == field.value) runtime_domain_is_valid = true;
    }
    var teardown_stage_is_valid = false;
    inline for (@typeInfo(pipewire.RuntimeErrorStage).@"enum".fields) |field| {
        if (report.teardown_error_stage == field.value) teardown_stage_is_valid = true;
    }
    var teardown_domain_is_valid = false;
    inline for (@typeInfo(pipewire.ErrorDomain).@"enum".fields) |field| {
        if (report.teardown_error_domain == field.value) teardown_domain_is_valid = true;
    }

    assert(timeline_validation_is_valid);
    assert(outcome_is_valid);
    assert(activity_is_valid);
    assert(runtime_stage_is_valid);
    assert(runtime_domain_is_valid);
    assert(teardown_stage_is_valid);
    assert(teardown_domain_is_valid);
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

    assert(report.samples_count <= launch.recording_samples_target +
        audio_exchange.callback_samples_count_max);
    assert(report.slot_publications_count <= report.published_samples_count);
    assert(report.missing_buffers_count <= report.callbacks_count);
    assert(report.clipped_samples_count <= report.samples_count);
    assert(report.header_metadata_buffers_count <= report.callbacks_count);
    assert(report.header_gap_buffers_count <= report.header_metadata_buffers_count);
    assert(report.header_gap_samples_count <= report.samples_count);
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
    assert(report.activity.quiet_threshold_rms >= 0.003);
    assert(report.activity.active_threshold_rms >= 0.008);
    assert(report.activity.quiet_threshold_rms <=
        report.activity.active_threshold_rms);
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
        assert(report.block_samples_count_max <= launch.slot_samples_boundary);
    }

    switch (outcome) {
        .completed => {
            assert(report.samples_count >= launch.recording_samples_target);
            assert(report.published_samples_count == report.samples_count);
        },
        .automatic_stop => {
            assert(launch.automatic_stop == @intFromEnum(AutomaticStop.after_quiet));
            assert(report.samples_count < launch.recording_samples_target);
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
            launch.slot_samples_boundary),
        else => assert(report.published_samples_count == report.samples_count),
    }
    if (report.published_samples_count == 0) {
        assert(report.slot_publications_count == 0);
    } else {
        assert(report.slot_publications_count > 0);
    }
}

fn workerMain(init: std.process.Init) !void {
    var arguments = try std.process.Args.Iterator.initAllocator(
        init.minimal.args,
        init.gpa,
    );
    defer arguments.deinit();
    assert(arguments.skip());

    const supervisor_socket_text = arguments.next() orelse {
        return error.InvalidWorkerArguments;
    };
    const exchange_fd_text = arguments.next() orelse {
        return error.InvalidWorkerArguments;
    };
    const publication_event_fd_text = arguments.next() orelse {
        return error.InvalidWorkerArguments;
    };
    const supervisor_pid_text = arguments.next() orelse {
        return error.InvalidWorkerArguments;
    };
    if (arguments.next() != null) return error.InvalidWorkerArguments;

    const supervisor_socket = try std.fmt.parseInt(
        std.posix.fd_t,
        supervisor_socket_text,
        10,
    );
    const exchange_fd = try std.fmt.parseInt(
        std.posix.fd_t,
        exchange_fd_text,
        10,
    );
    const publication_event_fd = try std.fmt.parseInt(
        std.posix.fd_t,
        publication_event_fd_text,
        10,
    );
    const supervisor_pid = try std.fmt.parseInt(
        std.os.linux.pid_t,
        supervisor_pid_text,
        10,
    );
    try runAudioWorker(
        supervisor_socket,
        exchange_fd,
        publication_event_fd,
        supervisor_pid,
    );
}

fn runAudioWorker(
    supervisor_socket: std.posix.fd_t,
    exchange_fd: std.posix.fd_t,
    publication_event_fd: std.posix.fd_t,
    expected_supervisor_pid: std.os.linux.pid_t,
) !void {
    assert(supervisor_socket >= 0);
    assert(exchange_fd >= 0);
    assert(publication_event_fd >= 0);
    assert(supervisor_socket != exchange_fd);
    assert(supervisor_socket != publication_event_fd);
    assert(exchange_fd != publication_event_fd);
    assert(expected_supervisor_pid > 1);

    // Arm the kernel-enforced lifetime relationship before touching inherited
    // resources. Passing the expected PID closes the classic race where the
    // parent exits between `spawn` and `PR_SET_PDEATHSIG`: after arming SIGKILL,
    // a changed parent PID proves that the signal opportunity was already lost,
    // so this worker exits instead of becoming an orphaned microphone process.
    try bindWorkerLifetimeToSupervisor(expected_supervisor_pid);

    defer closeFileDescriptor(supervisor_socket);
    defer closeFileDescriptor(exchange_fd);
    defer closeFileDescriptor(publication_event_fd);

    var launch_packet: WorkerLaunchPacket = undefined;
    try receivePacket(supervisor_socket, std.mem.asBytes(&launch_packet));

    try runAudioWorkerSession(
        supervisor_socket,
        exchange_fd,
        publication_event_fd,
        launch_packet,
    );
}

fn runAudioWorkerSession(
    supervisor_socket: std.posix.fd_t,
    exchange_fd: std.posix.fd_t,
    publication_event_fd: std.posix.fd_t,
    launch_packet: WorkerLaunchPacket,
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
        .slot_samples_boundary = launch.slot_samples_boundary,
        .automatic_stop = launch.automatic_stop,
        .process_realtime = launch.process_realtime,
    });

    var report_packet: WireReport = std.mem.zeroes(WireReport);
    switch (worker_result) {
        .captured => |capture| {
            const capture_failure: ?pipewire.RuntimeFailure = switch (capture.end) {
                .completed, .automatic_stop, .stopped, .cancelled => null,
                .failed => |failure| failure.detail,
            };
            const outcome: pipewire.Outcome = switch (capture.end) {
                .completed => .completed,
                .automatic_stop => .automatic_stop,
                .stopped => .stopped,
                .cancelled => .cancelled,
                .failed => |failure| @enumFromInt(@intFromEnum(failure.outcome)),
            };
            const empty_runtime_error: pipewire.RuntimeError = .{
                .stage = .none,
                .domain = .none,
                .code = 0,
            };
            const source = capture.source_identity;
            const callback = capture.callback;
            const worker_succeeded = .{
                .outcome = outcome,
                .runtime_error = if (capture_failure) |failure|
                    pipewire.RuntimeError{
                        .stage = @enumFromInt(@intFromEnum(failure.coordinate.stage)),
                        .domain = @enumFromInt(@intFromEnum(failure.coordinate.domain)),
                        .code = failure.coordinate.code,
                    }
                else
                    empty_runtime_error,
                .error_message = if (capture_failure) |failure|
                    failure.message
                else
                    @as([pipewire.runtime_error_message_capacity]u8, @splat(0)),
                .error_message_size = if (capture_failure) |failure|
                    failure.message_size
                else
                    0,
                .teardown_error = if (capture.teardown_failure) |failure|
                    pipewire.RuntimeError{
                        .stage = @enumFromInt(@intFromEnum(failure.coordinate.stage)),
                        .domain = @enumFromInt(@intFromEnum(failure.coordinate.domain)),
                        .code = failure.coordinate.code,
                    }
                else
                    empty_runtime_error,
                .teardown_error_message = if (capture.teardown_failure) |failure|
                    failure.message
                else
                    @as([pipewire.runtime_error_message_capacity]u8, @splat(0)),
                .teardown_error_message_size = if (capture.teardown_failure) |failure|
                    failure.message_size
                else
                    0,
                .timeline_validation = capture.timeline_validation,
                .pipewire_headers_version = capture.pipewire_headers_version,
                .pipewire_headers_version_size = capture.pipewire_headers_version_size,
                .pipewire_library_version = capture.pipewire_library_version,
                .pipewire_library_version_size = capture.pipewire_library_version_size,
                .pipewire_server_version = capture.pipewire_server_version,
                .pipewire_server_version_size = capture.pipewire_server_version_size,
                .source_identity = .{
                    .is_resolved = source != null,
                    .node_id = if (source) |value| value.node_id else std.math.maxInt(u32),
                    .node_object_serial = if (source) |value| value.node_object_serial else 0,
                    .device_id = if (source) |value| value.device_id else std.math.maxInt(u32),
                    .device_object_serial = if (source) |value| value.device_object_serial else 0,
                    .node_name = if (source) |value| value.node_name else @as([pipewire.source_identity_text_capacity]u8, @splat(0)),
                    .node_name_size = if (source) |value| value.node_name_size else 0,
                    .node_description = if (source) |value| value.node_description else @as([pipewire.source_identity_text_capacity]u8, @splat(0)),
                    .node_description_size = if (source) |value| value.node_description_size else 0,
                    .device_serial = if (source) |value| value.device_serial else @as([pipewire.source_identity_text_capacity]u8, @splat(0)),
                    .device_serial_size = if (source) |value| value.device_serial_size else 0,
                    .device_description = if (source) |value| value.device_description else @as([pipewire.source_identity_text_capacity]u8, @splat(0)),
                    .device_description_size = if (source) |value| value.device_description_size else 0,
                },
                .negotiated_sample_rate_hz = if (capture.negotiated_format) |format|
                    format.sample_rate_hz
                else
                    0,
                .negotiated_channels_count = if (capture.negotiated_format) |format|
                    format.channels_count
                else
                    0,
                .samples_count = capture.samples_count,
                .published_samples_count = capture.published_samples_count,
                .slot_publications_count = capture.slot_publications_count,
                .shared_memory_is_locked = capture.memory_lock == .locked,
                .shared_memory_lock_error_code = switch (capture.memory_lock) {
                    .locked => 0,
                    .unavailable => |unavailable| unavailable.error_code,
                },
                .shared_memory_lock_limit_bytes = switch (capture.memory_lock) {
                    .locked => |limit| limit,
                    .unavailable => |unavailable| unavailable.limit_bytes,
                },
                .main_loop_thread_id = capture.main_loop_thread_id,
                .callback_thread_id = if (callback) |value| value.thread_id else 0,
                .callback_scheduler_policy = if (callback) |value|
                    value.scheduler_policy orelse -1
                else
                    -1,
                .callback_scheduler_priority = if (callback) |value|
                    value.scheduler_priority orelse -1
                else
                    -1,
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
                .activity = if (callback) |value| value.activity else ActivityReport{
                    .activity = .unknown,
                    .activity_samples_count = 0,
                    .observed_samples_count = 0,
                    .unknown_samples_count = 0,
                    .quiet_samples_count = 0,
                    .active_samples_count = 0,
                    .activity_changes_count = 0,
                    .active_run_samples_count_max = 0,
                    .quiet_run_samples_count_max = 0,
                    .noise_floor_rms = 0,
                    .quiet_threshold_rms = 0.003,
                    .active_threshold_rms = 0.008,
                },
            };
            assert(worker_succeeded.error_message_size <= worker_succeeded.error_message.len);
            assert(worker_succeeded.teardown_error_message_size <=
                worker_succeeded.teardown_error_message.len);
            assert(worker_succeeded.pipewire_headers_version_size > 0);
            assert(worker_succeeded.pipewire_headers_version_size <=
                worker_succeeded.pipewire_headers_version.len);
            assert(worker_succeeded.pipewire_library_version_size > 0);
            assert(worker_succeeded.pipewire_library_version_size <=
                worker_succeeded.pipewire_library_version.len);
            assert(worker_succeeded.pipewire_server_version_size <=
                worker_succeeded.pipewire_server_version.len);
            assert(worker_succeeded.source_identity.node_name_size <=
                worker_succeeded.source_identity.node_name.len);
            assert(worker_succeeded.source_identity.node_description_size <=
                worker_succeeded.source_identity.node_description.len);
            assert(worker_succeeded.source_identity.device_serial_size <=
                worker_succeeded.source_identity.device_serial.len);
            assert(worker_succeeded.source_identity.device_description_size <=
                worker_succeeded.source_identity.device_description.len);
            if (worker_succeeded.samples_count > 0) {
                assert(worker_succeeded.source_identity.is_resolved);
            }
            switch (worker_succeeded.outcome) {
                .completed, .automatic_stop, .stopped, .cancelled => {
                    assert(worker_succeeded.runtime_error.stage == .none);
                    assert(worker_succeeded.runtime_error.domain == .none);
                    assert(worker_succeeded.runtime_error.code == 0);
                    assert(worker_succeeded.error_message_size == 0);
                },
                else => {
                    assert(worker_succeeded.runtime_error.stage != .none);
                    assert(worker_succeeded.runtime_error.domain != .none);
                    assert(worker_succeeded.error_message_size > 0);
                },
            }
            if (worker_succeeded.teardown_error.stage == .none) {
                assert(worker_succeeded.teardown_error.domain == .none);
                assert(worker_succeeded.teardown_error.code == 0);
                assert(worker_succeeded.teardown_error_message_size == 0);
            } else {
                assert(worker_succeeded.teardown_error.stage == .stream_disconnect);
                assert(worker_succeeded.teardown_error.domain == .pipewire_result);
                assert(worker_succeeded.teardown_error.code < 0);
                assert(worker_succeeded.teardown_error_message_size > 0);
            }
            assert(worker_succeeded.samples_count <=
                launch_packet.recording_samples_target +
                    audio_exchange.callback_samples_count_max);
            assert(worker_succeeded.published_samples_count <=
                worker_succeeded.samples_count);
            report_packet.worker_succeeded = 1;
            report_packet.shared_memory_is_locked =
                @intFromBool(worker_succeeded.shared_memory_is_locked);
            report_packet.shared_memory_lock_error_code =
                worker_succeeded.shared_memory_lock_error_code;
            report_packet.shared_memory_lock_limit_bytes =
                worker_succeeded.shared_memory_lock_limit_bytes;
            report_packet.outcome = @intFromEnum(worker_succeeded.outcome);
            report_packet.error_message_size = worker_succeeded.error_message_size;
            report_packet.runtime_error_stage = @intFromEnum(worker_succeeded.runtime_error.stage);
            report_packet.runtime_error_domain = @intFromEnum(worker_succeeded.runtime_error.domain);
            report_packet.runtime_error_code = worker_succeeded.runtime_error.code;
            assert(report_packet.runtime_error_stage ==
                @intFromEnum(worker_succeeded.runtime_error.stage));
            assert(report_packet.runtime_error_domain ==
                @intFromEnum(worker_succeeded.runtime_error.domain));
            assert(report_packet.runtime_error_code == worker_succeeded.runtime_error.code);
            @memcpy(
                report_packet.error_message[0..worker_succeeded.error_message_size],
                worker_succeeded.error_message[0..worker_succeeded.error_message_size],
            );
            report_packet.teardown_error_message_size =
                worker_succeeded.teardown_error_message_size;
            report_packet.teardown_error_stage =
                @intFromEnum(worker_succeeded.teardown_error.stage);
            report_packet.teardown_error_domain =
                @intFromEnum(worker_succeeded.teardown_error.domain);
            report_packet.teardown_error_code = worker_succeeded.teardown_error.code;
            assert(report_packet.teardown_error_stage ==
                @intFromEnum(worker_succeeded.teardown_error.stage));
            assert(report_packet.teardown_error_domain ==
                @intFromEnum(worker_succeeded.teardown_error.domain));
            assert(report_packet.teardown_error_code == worker_succeeded.teardown_error.code);
            @memcpy(
                report_packet.teardown_error_message[0..worker_succeeded.teardown_error_message_size],
                worker_succeeded.teardown_error_message[0..worker_succeeded.teardown_error_message_size],
            );
            report_packet.timeline_validation =
                @intFromEnum(worker_succeeded.timeline_validation);
            report_packet.pipewire_headers_version_size =
                worker_succeeded.pipewire_headers_version_size;
            @memcpy(
                report_packet.pipewire_headers_version[0..worker_succeeded.pipewire_headers_version_size],
                worker_succeeded.pipewire_headers_version[0..worker_succeeded.pipewire_headers_version_size],
            );
            report_packet.pipewire_library_version_size =
                worker_succeeded.pipewire_library_version_size;
            @memcpy(
                report_packet.pipewire_library_version[0..worker_succeeded.pipewire_library_version_size],
                worker_succeeded.pipewire_library_version[0..worker_succeeded.pipewire_library_version_size],
            );
            report_packet.pipewire_server_version_size =
                worker_succeeded.pipewire_server_version_size;
            @memcpy(
                report_packet.pipewire_server_version[0..worker_succeeded.pipewire_server_version_size],
                worker_succeeded.pipewire_server_version[0..worker_succeeded.pipewire_server_version_size],
            );
            report_packet.source_is_resolved =
                @intFromBool(worker_succeeded.source_identity.is_resolved);
            report_packet.source_node_id = worker_succeeded.source_identity.node_id;
            report_packet.source_node_object_serial =
                worker_succeeded.source_identity.node_object_serial;
            report_packet.source_device_id = worker_succeeded.source_identity.device_id;
            report_packet.source_device_object_serial =
                worker_succeeded.source_identity.device_object_serial;
            report_packet.source_node_name_size =
                worker_succeeded.source_identity.node_name_size;
            report_packet.source_node_description_size =
                worker_succeeded.source_identity.node_description_size;
            report_packet.source_device_serial_size =
                worker_succeeded.source_identity.device_serial_size;
            report_packet.source_device_description_size =
                worker_succeeded.source_identity.device_description_size;
            @memcpy(
                report_packet.source_node_name[0..worker_succeeded.source_identity.node_name_size],
                worker_succeeded.source_identity.node_name[0..worker_succeeded.source_identity.node_name_size],
            );
            @memcpy(
                report_packet.source_node_description[0..worker_succeeded.source_identity.node_description_size],
                worker_succeeded.source_identity.node_description[0..worker_succeeded.source_identity.node_description_size],
            );
            @memcpy(
                report_packet.source_device_serial[0..worker_succeeded.source_identity.device_serial_size],
                worker_succeeded.source_identity.device_serial[0..worker_succeeded.source_identity.device_serial_size],
            );
            @memcpy(
                report_packet.source_device_description[0..worker_succeeded.source_identity.device_description_size],
                worker_succeeded.source_identity.device_description[0..worker_succeeded.source_identity.device_description_size],
            );
            report_packet.negotiated_sample_rate_hz =
                worker_succeeded.negotiated_sample_rate_hz;
            report_packet.negotiated_channels_count =
                worker_succeeded.negotiated_channels_count;
            report_packet.samples_count = worker_succeeded.samples_count;
            report_packet.published_samples_count =
                worker_succeeded.published_samples_count;
            report_packet.slot_publications_count =
                worker_succeeded.slot_publications_count;
            report_packet.main_loop_thread_id = worker_succeeded.main_loop_thread_id;
            report_packet.callback_thread_id = worker_succeeded.callback_thread_id;
            report_packet.callback_scheduler_policy =
                worker_succeeded.callback_scheduler_policy;
            report_packet.callback_scheduler_priority =
                worker_succeeded.callback_scheduler_priority;
            report_packet.callbacks_count = worker_succeeded.callbacks_count;
            report_packet.missing_buffers_count = worker_succeeded.missing_buffers_count;
            report_packet.clipped_samples_count = worker_succeeded.clipped_samples_count;
            report_packet.header_metadata_buffers_count =
                worker_succeeded.header_metadata_buffers_count;
            report_packet.header_gap_buffers_count = worker_succeeded.header_gap_buffers_count;
            report_packet.header_gap_samples_count = worker_succeeded.header_gap_samples_count;
            report_packet.block_samples_count_min = worker_succeeded.block_samples_count_min;
            report_packet.block_samples_count_max = worker_succeeded.block_samples_count_max;
            report_packet.callback_duration_ns_max = worker_succeeded.callback_duration_ns_max;
            report_packet.callback_gap_ns_max = worker_succeeded.callback_gap_ns_max;
            report_packet.activity = .{
                .activity = @intFromEnum(worker_succeeded.activity.activity),
                .activity_samples_count = worker_succeeded.activity.activity_samples_count,
                .observed_samples_count = worker_succeeded.activity.observed_samples_count,
                .unknown_samples_count = worker_succeeded.activity.unknown_samples_count,
                .quiet_samples_count = worker_succeeded.activity.quiet_samples_count,
                .active_samples_count = worker_succeeded.activity.active_samples_count,
                .changes_count = worker_succeeded.activity.activity_changes_count,
                .active_run_samples_count_max = worker_succeeded.activity.active_run_samples_count_max,
                .quiet_run_samples_count_max = worker_succeeded.activity.quiet_run_samples_count_max,
                .noise_floor_rms = worker_succeeded.activity.noise_floor_rms,
                .quiet_threshold_rms = worker_succeeded.activity.quiet_threshold_rms,
                .active_threshold_rms = worker_succeeded.activity.active_threshold_rms,
            };
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
        launch_packet,
        launch_packet.recording_samples_target +
            audio_exchange.callback_samples_count_max,
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

fn consumeNextPublishedSlot(
    consumer: *PublishedAudioConsumer,
    exchange: *AudioExchange,
    expected_session_id: u64,
    slot_samples_boundary: u32,
    consumer_delay_ms: u32,
    now_ns: u64,
    ignore_consumer_delay: bool,
) bool {
    assert(expected_session_id > 0);
    assert(exchange.session_id == expected_session_id);
    assert(slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(consumer_delay_ms <= 10_000);
    assert(consumer.samples_count <= consumer.samples.len);
    if (consumer.pending == null) {
        // Physical indices are unrelated to publication order after recycling.
        // Find the one release-published slot carrying the next ordinal; later
        // ordinals remain untouched until every earlier prefix has been copied.
        for (&exchange.slots, 0..) |*slot, slot_index| {
            const publication = audio_exchange.acquirePublishedSlot(slot) orelse continue;
            assert(publication.publication_ordinal >=
                consumer.next_publication_ordinal);
            if (publication.publication_ordinal == consumer.next_publication_ordinal) {
                consumer.pending = .{
                    .index = audio_exchange.SlotIndex.fromArrayIndex(slot_index),
                    .ready_at_monotonic_ns = if (consumer_delay_ms == 0)
                        0
                    else
                        now_ns + @as(u64, consumer_delay_ms) * std.time.ns_per_ms,
                };
                break;
            }
        }
        if (consumer.pending == null) return false;

        // Keeping the slot published during this delay models a consumer that
        // owns shared audio while performing expensive work. Audio can continue
        // through other slots, then reports pipeline pressure if all three stay
        // occupied. A zero delay exercises the normal immediate handoff path.
        if (!ignore_consumer_delay and consumer_delay_ms > 0) return false;
    }

    const pending = consumer.pending.?;
    if (!ignore_consumer_delay and now_ns < pending.ready_at_monotonic_ns) {
        return false;
    }

    const slot = &exchange.slots[pending.index.arrayIndex()];
    const publication_optional = audio_exchange.acquirePublishedSlot(slot);
    assert(publication_optional != null);
    const publication = publication_optional.?;
    assert(publication.publication_ordinal == consumer.next_publication_ordinal);
    assert(publication.samples_count <= slot_samples_boundary);
    assert(publication.samples_count <=
        consumer.samples.len - consumer.samples_count);

    const samples_count_before = consumer.samples_count;
    const publication_ordinal_before = consumer.next_publication_ordinal;
    @memcpy(
        consumer.samples[consumer.samples_count..][0..publication.samples_count],
        slot.samples[0..publication.samples_count],
    );
    consumer.samples_count += publication.samples_count;
    consumer.next_publication_ordinal += 1;
    consumer.pending = null;
    audio_exchange.releaseConsumedSlot(slot);

    assert(consumer.samples_count == samples_count_before + publication.samples_count);
    assert(consumer.next_publication_ordinal == publication_ordinal_before + 1);
    assert(consumer.pending == null);
    return true;
}

fn forceStopWorker(worker: *std.process.Child, io: Io) !void {
    const worker_pid = worker.id orelse return;

    while (true) {
        switch (std.os.linux.errno(std.os.linux.kill(worker_pid, .KILL))) {
            .SUCCESS, .SRCH => break,
            .INTR => continue,
            else => return error.AudioWorkerKillFailed,
        }
    }

    _ = try worker.wait(io);
    assert(worker.id == null);
}

fn waitForWorkerExit(
    worker: *std.process.Child,
    io: Io,
    timeout_ms: i32,
) !std.process.Child.Term {
    assert(worker.id != null);
    assert(timeout_ms > 0);
    const worker_pid = worker.id.?;
    const pidfd_result = std.os.linux.pidfd_open(worker_pid, 0);
    if (std.os.linux.errno(pidfd_result) != .SUCCESS) {
        return error.AudioWorkerPidfdCreationFailed;
    }
    const pidfd: std.posix.fd_t = @intCast(pidfd_result);
    defer closeFileDescriptor(pidfd);

    // Receiving a report or socket EOF ends protocol supervision, not process
    // supervision. Native teardown or an internal defect can still stop the
    // worker before it exits. A pidfd becomes readable only when that exact
    // process exits, so this deadline cannot be confused by PID reuse and keeps
    // the following reap bounded after either protocol outcome.
    var poll_descriptors = [_]std.posix.pollfd{.{
        .fd = pidfd,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};
    if (try std.posix.poll(&poll_descriptors, timeout_ms) == 0) {
        return error.AudioWorkerExitDeadlineExceeded;
    }
    if (poll_descriptors[0].revents & std.posix.POLL.IN == 0) {
        return error.AudioWorkerPidfdFailed;
    }
    const term = try worker.wait(io);
    assert(worker.id == null);
    return term;
}

fn reportUnexpectedWorkerExit(worker: *std.process.Child, io: Io) !noreturn {
    assert(worker.id != null);

    // SOCK_SEQPACKET EOF says only that no report can arrive; it does not prove
    // the process exited. Wait through the bounded pidfd path before classifying
    // an assertion, signal, or exit code so a worker that closes the socket and
    // hangs cannot wedge its supervisor.
    const term = try waitForWorkerExit(worker, io, 1_000);
    assert(worker.id == null);
    switch (term) {
        .exited => |exit_code| stderr(
            "Audio worker exited with code {d} before sending a report\n",
            .{exit_code},
        ),
        .signal => |signal| stderr(
            "Audio worker terminated by signal {d} before sending a report\n",
            .{@intFromEnum(signal)},
        ),
        .stopped => |signal| stderr(
            "Audio worker stopped by signal {d} before sending a report\n",
            .{@intFromEnum(signal)},
        ),
        .unknown => |status| stderr(
            "Audio worker ended with unknown status 0x{x} before sending a report\n",
            .{status},
        ),
    }
    return error.AudioWorkerExitedWithoutReport;
}

fn readEventCounter(event_fd: std.posix.fd_t) !u64 {
    assert(event_fd >= 0);

    var counter: u64 = 0;
    while (true) {
        const read_result = std.os.linux.read(
            event_fd,
            std.mem.asBytes(&counter).ptr,
            @sizeOf(u64),
        );
        switch (std.os.linux.errno(read_result)) {
            .SUCCESS => {
                assert(read_result == @sizeOf(u64));
                assert(counter > 0);
                return counter;
            },
            .INTR => continue,
            .AGAIN => return error.AudioPublicationEventNotReady,
            else => return error.AudioPublicationEventReadFailed,
        }
    }
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
            else => return error.WorkerPacketSendFailed,
        }
    }
}

fn receivePacket(socket: std.posix.fd_t, packet: []u8) !void {
    assert(socket >= 0);
    assert(packet.len > 0);

    while (true) {
        // Linux returns a SOCK_SEQPACKET record's complete length when
        // `MSG_TRUNC` is requested, even though it copies at most `packet.len`
        // bytes. This lets the receiver enforce the same fixed packet size that
        // the sender asserts, rather than silently accepting only a prefix.
        const receive_result = std.os.linux.recvfrom(
            socket,
            packet.ptr,
            packet.len,
            std.os.linux.MSG.TRUNC,
            null,
            null,
        );
        switch (std.os.linux.errno(receive_result)) {
            .SUCCESS => {
                if (receive_result == 0) return error.WorkerSocketClosed;
                assert(receive_result == packet.len);
                return;
            },
            .INTR => continue,
            else => return error.WorkerPacketReceiveFailed,
        }
    }
}

fn unblockServiceSignals() void {
    var signal_mask = std.posix.sigemptyset();
    std.posix.sigaddset(&signal_mask, .TERM);
    std.posix.sigaddset(&signal_mask, .INT);
    std.posix.sigprocmask(std.posix.SIG.UNBLOCK, &signal_mask, null);
}

fn sleepMilliseconds(milliseconds: u32) void {
    var requested: std.os.linux.timespec = .{
        .sec = @intCast(milliseconds / 1000),
        .nsec = @intCast((milliseconds % 1000) * std.time.ns_per_ms),
    };
    var remaining: std.os.linux.timespec = undefined;
    while (true) {
        switch (std.os.linux.errno(std.os.linux.nanosleep(&requested, &remaining))) {
            .SUCCESS => return,
            .INTR => requested = remaining,
            else => unreachable,
        }
    }
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

fn closeFileDescriptor(file_descriptor: std.posix.fd_t) void {
    assert(file_descriptor >= 0);
    const result = std.os.linux.close(file_descriptor);
    assert(std.os.linux.errno(result) == .SUCCESS);
}
