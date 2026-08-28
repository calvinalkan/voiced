//! Owns the complete process boundary around one PipeWire recording. The
//! supervisor-side API creates and maps the shared exchange, launches the audio
//! worker, transmits stop or cancel, consumes publications while supervising
//! deadlines, receives the final report, and contains a worker that does not
//! exit. The worker entry maps the same exchange after exec and runs exactly one
//! `pipewire.run` operation.

const std = @import("std");
const pipewire = @import("pipewire.zig");
const audio_exchange = @import("audio_exchange.zig");
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
pub const Outcome = pipewire.Outcome;
pub const TimelineValidation = pipewire.TimelineValidation;
pub const SetupErrorStage = pipewire.SetupErrorStage;
pub const RuntimeErrorStage = pipewire.RuntimeErrorStage;
pub const ErrorDomain = pipewire.ErrorDomain;
pub const sample_rate_hz = audio_exchange.sample_rate_hz;
pub const slots_count = audio_exchange.slots_count;
pub const slot_duration_seconds_max = audio_exchange.slot_duration_seconds_max;
pub const callback_samples_count_max = audio_exchange.callback_samples_count_max;
pub const audio_exchange_size = @sizeOf(AudioExchange);

pub const StartOptions = struct {
    generation: u64,
    target: ?[:0]const u8,
    configured_device_serial: ?[:0]const u8,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    consumer_delay_ms: u32,
    process_realtime: bool,
};

/// The fixed process report is intentionally value-only: every string owns a
/// bounded array, and no field contains an address from the worker process.
pub const Report = extern struct {
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

    error_message: [worker_error_message_capacity]u8,
    teardown_error_message: [worker_error_message_capacity]u8,
};

/// Borrows the final report and ordered published samples from `AudioProcess`.
/// Both remain valid until `killAndReap` releases the process handle.
pub const Result = struct {
    report: *const Report,
    samples: []const f32,
    slots_consumed_before_final_report: u32,
};

pub const AudioProcess = opaque {};

// The installed `audio-process` worker executable enters through this private
// function. Exporting it as the root entry is separate from the five operations
// imported by the spike and, later, the supervisor.
pub const main = workerMain;

const WorkerLaunchPacket = extern struct {
    exchange_generation: u64,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    process_realtime: u8,
    target_name_size: u16,
    configured_device_serial_size: u16,
    target_name: [target_name_capacity]u8,
    configured_device_serial: [target_name_capacity]u8,
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

const PublishedAudioConsumer = struct {
    samples: []f32,
    samples_count: usize,
    next_publication_ordinal: u32,
    pending_slot_index: ?u8,
    pending_slot_ready_at_ns: u64,
    slots_consumed_before_final_report: u32,
};

const Process = struct {
    allocator: Allocator,
    io: Io,
    exchange_fd: std.posix.fd_t,
    mapped_exchange: MappedAudioExchange,
    supervisor_socket: std.posix.fd_t,
    worker: std.process.Child,
    launch_packet: WorkerLaunchPacket,
    consumer_delay_ms: u32,
    consumer: PublishedAudioConsumer,

    setup_deadline_ns: u64,
    recording_deadline_ns: u64,
    sample_progress_deadline_ns: u64,
    control_deadline_ns: u64,
    observed_callbacks_count: u64,
    observed_samples_count: u32,
    control_requested: ?pipewire.ControlCommand,

    report: Report,
    result: ?Result,
};

/// Creates the shared exchange, launches one isolated worker, sends its fixed
/// launch record, and returns an opaque handle owned by `killAndReap`.
pub fn start(
    init: std.process.Init,
    options: StartOptions,
) !*AudioProcess {
    assert(options.generation > 0);
    assert(options.recording_samples_target > 0);
    assert(options.recording_samples_target <= 90 * audio_exchange.sample_rate_hz);
    assert(options.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(options.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(options.consumer_delay_ms <= 10_000);
    assert(options.target == null or options.configured_device_serial == null);
    if (options.target) |target| {
        assert(target.len > 0);
        assert(target.len < target_name_capacity);
    }
    if (options.configured_device_serial) |device_serial| {
        assert(device_serial.len > 0);
        assert(device_serial.len < target_name_capacity);
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
    audio_exchange.initialize(mapped_exchange.exchange, options.generation);

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
            supervisor_pid_text,
        },
    });
    errdefer forceStopWorker(&worker, init.io) catch {};
    closeFileDescriptor(worker_socket);
    worker_socket_is_owned = false;

    var launch_packet: WorkerLaunchPacket = std.mem.zeroes(WorkerLaunchPacket);
    launch_packet.exchange_generation = options.generation;
    launch_packet.recording_samples_target = options.recording_samples_target;
    launch_packet.slot_samples_boundary = options.slot_samples_boundary;
    launch_packet.process_realtime = @intFromBool(options.process_realtime);
    if (options.target) |target| {
        @memcpy(launch_packet.target_name[0..target.len], target);
        launch_packet.target_name_size = @intCast(target.len);
    }
    if (options.configured_device_serial) |device_serial| {
        @memcpy(
            launch_packet.configured_device_serial[0..device_serial.len],
            device_serial,
        );
        launch_packet.configured_device_serial_size = @intCast(device_serial.len);
    }

    assert(launch_packet.exchange_generation > 0);
    assert(launch_packet.recording_samples_target > 0);
    assert(launch_packet.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(launch_packet.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(launch_packet.process_realtime <= 1);
    assert(launch_packet.target_name_size == 0 or
        launch_packet.configured_device_serial_size == 0);
    assert(launch_packet.target_name_size < launch_packet.target_name.len);
    assert(launch_packet.configured_device_serial_size <
        launch_packet.configured_device_serial.len);
    assert(launch_packet.target_name[launch_packet.target_name_size] == 0);
    assert(launch_packet.configured_device_serial[
        launch_packet.configured_device_serial_size
    ] == 0);
    try sendPacket(supervisor_socket, std.mem.asBytes(&launch_packet));

    const samples_capacity = options.recording_samples_target +
        audio_exchange.callback_samples_count_max;
    const samples = try init.gpa.alloc(f32, samples_capacity);
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
        .supervisor_socket = supervisor_socket,
        .worker = worker,
        .launch_packet = launch_packet,
        .consumer_delay_ms = options.consumer_delay_ms,
        .consumer = .{
            .samples = samples,
            .samples_count = 0,
            .next_publication_ordinal = 0,
            .pending_slot_index = null,
            .pending_slot_ready_at_ns = 0,
            .slots_consumed_before_final_report = 0,
        },
        .setup_deadline_ns = supervision_started_ns + 3 * std.time.ns_per_s,
        .recording_deadline_ns = supervision_started_ns +
            (@as(u64, recording_seconds) + 10) * std.time.ns_per_s,
        .sample_progress_deadline_ns = 0,
        .control_deadline_ns = 0,
        .observed_callbacks_count = 0,
        .observed_samples_count = 0,
        .control_requested = null,
        .report = undefined,
        .result = null,
    };

    exchange_fd_is_owned = false;
    exchange_is_mapped = false;
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
/// `timeout_ms`. `null` means the worker remains active; a returned result stays
/// borrowed from the process until `killAndReap`.
pub fn receiveReport(
    process_opaque: *AudioProcess,
    timeout_ms: u16,
) !?*const Result {
    const process = processImplementation(process_opaque);
    assert(timeout_ms > 0);
    assert(process.result == null);
    assert(process.worker.id != null);

    const now_ns = monotonicNanoseconds();
    while (consumeNextPublishedSlot(
        &process.consumer,
        process.mapped_exchange.exchange,
        process.launch_packet.exchange_generation,
        process.launch_packet.slot_samples_boundary,
        process.consumer_delay_ms,
        now_ns,
        false,
    )) {
        process.consumer.slots_consumed_before_final_report += 1;
    }

    var poll_descriptors = [_]std.posix.pollfd{.{
        .fd = process.supervisor_socket,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};
    if (try std.posix.poll(&poll_descriptors, timeout_ms) > 0) {
        if (poll_descriptors[0].revents & std.posix.POLL.IN != 0) {
            receivePacket(
                process.supervisor_socket,
                std.mem.asBytes(&process.report),
            ) catch |packet_error| switch (packet_error) {
                error.WorkerSocketClosed => {
                    try reportUnexpectedWorkerExit(&process.worker, process.io);
                },
                else => return packet_error,
            };
            try finishReport(process);
            return &process.result.?;
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
    assert(callbacks_count >= process.observed_callbacks_count);
    process.observed_callbacks_count = callbacks_count;

    const samples_count = audio_exchange.acquireAudioSamplesCount(
        process.mapped_exchange.exchange,
    );
    assert(samples_count >= process.observed_samples_count);
    assert(samples_count <= process.consumer.samples.len);
    if (samples_count > process.observed_samples_count) {
        process.observed_samples_count = samples_count;
        process.sample_progress_deadline_ns = deadline_now_ns + 2 * std.time.ns_per_s;
    }

    if (process.control_requested) |command| {
        if (deadline_now_ns >= process.control_deadline_ns) {
            stderr(
                "Audio worker control deadline exceeded\n" ++
                    "Command: {s}\n" ++
                    "Deadline: 2000 ms\n" ++
                    "Action: forced termination\n" ++
                    "Callbacks observed: {d}\n" ++
                    "Samples observed: {d}\n",
                .{
                    @tagName(command),
                    process.observed_callbacks_count,
                    process.observed_samples_count,
                },
            );
            return error.AudioWorkerControlDeadlineExceeded;
        }
        return null;
    }

    const requested_target = if (process.launch_packet.target_name_size > 0)
        process.launch_packet.target_name[0..process.launch_packet.target_name_size]
    else if (process.launch_packet.configured_device_serial_size > 0)
        process.launch_packet.configured_device_serial[0..process.launch_packet.configured_device_serial_size]
    else
        "default source";
    const processing_mode = if (process.launch_packet.process_realtime == 1)
        "realtime"
    else
        "main loop";

    if (process.observed_samples_count == 0 and
        deadline_now_ns >= process.setup_deadline_ns)
    {
        stderr(
            "Audio worker deadline exceeded\n" ++
                "Stage: waiting_for_first_samples\n" ++
                "Deadline: 3000 ms\n" ++
                "Target: {s}\n" ++
                "Processing: {s}\n" ++
                "Callbacks observed: {d}\n" ++
                "Samples observed: {d}\n",
            .{
                requested_target,
                processing_mode,
                process.observed_callbacks_count,
                process.observed_samples_count,
            },
        );
        return error.AudioWorkerSetupDeadlineExceeded;
    }
    if (process.observed_samples_count > 0 and
        deadline_now_ns >= process.sample_progress_deadline_ns)
    {
        stderr(
            "Audio worker deadline exceeded\n" ++
                "Stage: waiting_for_sample_progress\n" ++
                "Deadline: 2000 ms\n" ++
                "Target: {s}\n" ++
                "Processing: {s}\n" ++
                "Callbacks observed: {d}\n" ++
                "Samples observed: {d}\n",
            .{
                requested_target,
                processing_mode,
                process.observed_callbacks_count,
                process.observed_samples_count,
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
                process.observed_callbacks_count,
                process.observed_samples_count,
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
    assert(process.result == null);
    assert(process.control_requested == null);

    const packet: pipewire.ControlPacket = .{
        .command = @intFromEnum(command),
        .reserved = 0,
    };
    try sendPacket(process.supervisor_socket, std.mem.asBytes(&packet));
    process.control_requested = command;
    process.control_deadline_ns = monotonicNanoseconds() + 2 * std.time.ns_per_s;

    assert(process.control_requested == command);
    assert(process.control_deadline_ns > 0);
}

fn finishReport(process: *Process) !void {
    assert(process.result == null);
    validateReportPacket(
        &process.report,
        process.launch_packet,
        process.consumer.samples.len,
        process.mapped_exchange.exchange,
    );

    while (process.consumer.next_publication_ordinal <
        process.report.slot_publications_count)
    {
        assert(consumeNextPublishedSlot(
            &process.consumer,
            process.mapped_exchange.exchange,
            process.launch_packet.exchange_generation,
            process.launch_packet.slot_samples_boundary,
            process.consumer_delay_ms,
            monotonicNanoseconds(),
            true,
        ));
    }
    assert(process.consumer.samples_count == process.report.published_samples_count);

    const worker_term = try waitForWorkerExit(&process.worker, process.io, 1_000);
    switch (worker_term) {
        .exited => |exit_code| assert(exit_code == 0),
        else => unreachable,
    }
    assert(process.worker.id == null);

    process.result = .{
        .report = &process.report,
        .samples = process.consumer.samples[0..process.consumer.samples_count],
        .slots_consumed_before_final_report = process.consumer.slots_consumed_before_final_report,
    };
}

fn validateReportPacket(
    report: *const Report,
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
        .completed, .stopped, .cancelled => {
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
    assert(report.callbacks_count == audio_exchange.acquireAudioCallbacksCount(exchange));
    assert(report.samples_count == audio_exchange.acquireAudioSamplesCount(exchange));
    if (report.samples_count == 0) {
        assert(report.published_samples_count == 0);
        assert(report.slot_publications_count == 0);
        assert(report.callback_thread_id == 0);
        assert(report.block_samples_count_min == 0);
        assert(report.block_samples_count_max == 0);
    } else {
        assert(report.source_is_resolved == 1);
        assert(report.callback_thread_id > 0);
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
    const supervisor_pid = try std.fmt.parseInt(
        std.os.linux.pid_t,
        supervisor_pid_text,
        10,
    );
    try runAudioWorker(supervisor_socket, exchange_fd, supervisor_pid);
}

fn runAudioWorker(
    supervisor_socket: std.posix.fd_t,
    exchange_fd: std.posix.fd_t,
    expected_supervisor_pid: std.os.linux.pid_t,
) !void {
    assert(supervisor_socket >= 0);
    assert(exchange_fd >= 0);
    assert(supervisor_socket != exchange_fd);
    assert(expected_supervisor_pid > 1);

    // Arm the kernel-enforced lifetime relationship before touching inherited
    // resources. Passing the expected PID closes the classic race where the
    // parent exits between `spawn` and `PR_SET_PDEATHSIG`: after arming SIGKILL,
    // a changed parent PID proves that the signal opportunity was already lost,
    // so this worker exits instead of becoming an orphaned microphone process.
    try bindWorkerLifetimeToSupervisor(expected_supervisor_pid);

    defer closeFileDescriptor(supervisor_socket);
    defer closeFileDescriptor(exchange_fd);

    var launch_packet: WorkerLaunchPacket = undefined;
    try receivePacket(supervisor_socket, std.mem.asBytes(&launch_packet));

    // Assert the launch assumptions where the worker consumes them, paired with
    // the supervisor's assertions immediately before send.
    assert(launch_packet.exchange_generation > 0);
    assert(launch_packet.recording_samples_target > 0);
    assert(launch_packet.recording_samples_target <= 90 * audio_exchange.sample_rate_hz);
    assert(launch_packet.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(launch_packet.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(launch_packet.process_realtime <= 1);
    assert(launch_packet.target_name_size == 0 or
        launch_packet.configured_device_serial_size == 0);
    assert(launch_packet.target_name_size < launch_packet.target_name.len);
    assert(launch_packet.configured_device_serial_size <
        launch_packet.configured_device_serial.len);
    assert(std.mem.indexOfScalar(
        u8,
        launch_packet.target_name[0..launch_packet.target_name_size],
        0,
    ) == null);
    assert(launch_packet.target_name[launch_packet.target_name_size] == 0);
    assert(std.mem.indexOfScalar(
        u8,
        launch_packet.configured_device_serial[0..launch_packet.configured_device_serial_size],
        0,
    ) == null);
    assert(launch_packet.configured_device_serial[
        launch_packet.configured_device_serial_size
    ] == 0);

    const mapped_exchange = try mapAudioExchange(exchange_fd);
    defer mapped_exchange.unmap();

    // The supervisor asserts the same postconditions immediately after it
    // initializes this trusted memfd. Repeat them here so the worker's launch
    // assumptions are visible without following an assertion helper.
    assert(mapped_exchange.exchange.version == audio_exchange.format_version);
    assert(mapped_exchange.exchange.reserved == 0);
    assert(mapped_exchange.exchange.generation == launch_packet.exchange_generation);
    assert(mapped_exchange.exchange.audio_callbacks_count == 0);
    assert(mapped_exchange.exchange.audio_samples_count == 0);
    assert(mapped_exchange.exchange.reserved_2 == 0);
    for (&mapped_exchange.exchange.slots) |*slot| {
        assert(slot.state == audio_exchange.slot_state_available);
        assert(slot.samples_count == 0);
        assert(slot.generation == 0);
        assert(slot.publication_ordinal == 0);
        assert(slot.reserved == 0);
    }

    const target: ?[:0]const u8 = if (launch_packet.target_name_size == 0)
        null
    else target: {
        launch_packet.target_name[launch_packet.target_name_size] = 0;
        break :target launch_packet.target_name[0..launch_packet.target_name_size :0];
    };
    const configured_device_serial: ?[:0]const u8 =
        if (launch_packet.configured_device_serial_size == 0)
            null
        else configured: {
            launch_packet.configured_device_serial[
                launch_packet.configured_device_serial_size
            ] = 0;
            break :configured launch_packet.configured_device_serial[0..launch_packet.configured_device_serial_size :0];
        };
    assert(target == null or configured_device_serial == null);

    var setup_error: pipewire.SetupError = std.mem.zeroes(pipewire.SetupError);

    const worker_report = pipewire.run(.{
        .exchange = mapped_exchange.exchange,
        .control_socket = supervisor_socket,
        .target = target,
        .configured_device_serial = configured_device_serial,
        .recording_samples_target = launch_packet.recording_samples_target,
        .slot_samples_boundary = launch_packet.slot_samples_boundary,
        .process_realtime = launch_packet.process_realtime == 1,
    }, &setup_error);

    var report_packet: Report = std.mem.zeroes(Report);
    if (worker_report) |worker_succeeded| {
        assert(setup_error.stage == .none);
        assert(setup_error.domain == .none);
        assert(setup_error.code == 0);
        assert(setup_error.message_size == 0);
        assert(setup_error.pipewire_version_size == 0);
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
            .completed, .stopped, .cancelled => {
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
    } else |_| {
        assert(setup_error.stage != .none);
        assert(setup_error.domain != .none);
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
    expected_generation: u64,
    slot_samples_boundary: u32,
    consumer_delay_ms: u32,
    now_ns: u64,
    ignore_consumer_delay: bool,
) bool {
    assert(expected_generation > 0);
    assert(exchange.generation == expected_generation);
    assert(slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(consumer_delay_ms <= 10_000);
    assert(consumer.samples_count <= consumer.samples.len);
    if (consumer.pending_slot_index) |slot_index| {
        assert(slot_index < audio_exchange.slots_count);
    }

    if (consumer.pending_slot_index == null) {
        // Physical indices are unrelated to publication order after recycling.
        // Find the one release-published slot carrying the next ordinal; later
        // ordinals remain untouched until every earlier prefix has been copied.
        for (&exchange.slots, 0..) |*slot, slot_index| {
            const publication = audio_exchange.acquirePublishedSlot(slot) orelse continue;
            assert(publication.generation == expected_generation);
            assert(publication.publication_ordinal >=
                consumer.next_publication_ordinal);
            if (publication.publication_ordinal == consumer.next_publication_ordinal) {
                consumer.pending_slot_index = @intCast(slot_index);
                break;
            }
        }
        if (consumer.pending_slot_index == null) return false;

        // Keeping the slot published during this delay models a consumer that
        // owns shared audio while performing expensive work. Audio can continue
        // through other slots, then reports pipeline pressure if all three stay
        // occupied. A zero delay exercises the normal immediate handoff path.
        if (!ignore_consumer_delay and consumer_delay_ms > 0) {
            consumer.pending_slot_ready_at_ns = now_ns +
                @as(u64, consumer_delay_ms) * std.time.ns_per_ms;
            return false;
        }
    }

    if (!ignore_consumer_delay and now_ns < consumer.pending_slot_ready_at_ns) {
        return false;
    }

    assert(consumer.pending_slot_index != null);
    const slot = &exchange.slots[consumer.pending_slot_index.?];
    const publication_optional = audio_exchange.acquirePublishedSlot(slot);
    assert(publication_optional != null);
    const publication = publication_optional.?;
    assert(publication.generation == expected_generation);
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
    consumer.pending_slot_index = null;
    consumer.pending_slot_ready_at_ns = 0;
    audio_exchange.releaseConsumedSlot(slot);

    assert(consumer.samples_count == samples_count_before + publication.samples_count);
    assert(consumer.next_publication_ordinal == publication_ordinal_before + 1);
    assert(consumer.pending_slot_index == null);
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

    // Receiving a Report or socket EOF ends protocol supervision, not process
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

    // SOCK_SEQPACKET EOF says only that no Report can arrive; it does not prove
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
