//! Owns one bounded end-to-end session while the production supervisor boundary
//! is established. The event loop is the only lifecycle owner: workers
//! publish bytes and reports, but only this file chooses finish, discard, retry,
//! slot release, and final transcript acceptance.
//!
//! The implementation deliberately has no worker or deadline incarnation. A
//! replacement starts only after the complete returned epoll batch is consumed,
//! so stale events from the old descriptors have no replacement to affect.
//! Timer events likewise ask the supervisor to evaluate current absolute
//! deadlines; they do not identify an earlier deadline instance.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const audio_process = @import("audio_process.zig");
const models = @import("models");
const transcription_process = @import("transcription_process.zig");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;
const TranscriptExchange = transcription_process.TranscriptExchange;

const epoll_events_count_max: u32 = 16;
const session_id: u64 = 1;

pub const FakeScenario = enum {
    normal,
    burst_publications,
    slow_transcription,
    transcription_crash_before_result,
    transcription_crash_after_result,
    repeated_transcription_crash,
    transcription_hang,
};

pub const ModelResidency = enum {
    session,
    service,
};

pub const ModelOptions = struct {
    model: models.Model = models.default,
    inference_threads_count: u32 = 4,
    residency: ModelResidency = .session,
};

pub const ModelTimingReport = struct {
    residency: ModelResidency,
    workers_started_count: u32,
    workers_ready_count: u32,
    worker_start_to_ready_elapsed_ns_total: u64,
    worker_start_to_ready_elapsed_ns_max: u64,
    model_load_elapsed_ns_total: u64,
    model_load_elapsed_ns_max: u64,
    warmup_elapsed_ns_total: u64,
    warmup_elapsed_ns_max: u64,
    feature_extraction_elapsed_ns_total: u64,
    feature_extraction_elapsed_ns_max: u64,
    inference_elapsed_ns_total: u64,
    inference_elapsed_ns_max: u64,
    transcriptions_count: u32,
};

pub const AudioTimingReport = struct {
    start_to_finish_elapsed_ns: u64,
    finish_to_complete_elapsed_ns: u64,
    callbacks_count: u64,
    callback_duration_ns_max: u64,
    callback_gap_ns_max: u64,
};

pub const SessionTimingReport = struct {
    command_elapsed_ns: u64,
    audio: ?AudioTimingReport,
    model: ?ModelTimingReport,
};

pub const TranscriptionBackend = union(enum) {
    fake,
    model: ModelOptions,
};

pub const recording_duration_seconds_default: u16 = 60 * 60;
pub const recording_duration_seconds_limit: u16 = std.math.maxInt(u16);

comptime {
    // The configured duration remains a u16, while the audio launch protocol
    // carries its corresponding 16 kHz sample target in a u32.
    assert(@as(u64, recording_duration_seconds_limit) *
        audio_process.sample_rate_hz <= std.math.maxInt(u32));
}

pub const PipeWireOptions = struct {
    source: audio_process.Source = .default,
    recording_seconds: u16 = recording_duration_seconds_default,
    slot_seconds: u8 = audio_process.slot_duration_seconds_max,
    automatic_stop: audio_process.AutomaticStop = .disabled,
    process_realtime: bool = true,
    transcription: TranscriptionBackend = .fake,
};

const SessionConfiguration = union(enum) {
    fake: FakeScenario,
    pipewire: PipeWireOptions,
};

const EventSource = enum(u64) {
    audio_publication = 1,
    audio_packet = 2,
    audio_exit = 3,
    transcription_packet = 4,
    transcription_exit = 5,
    deadline = 6,
    service_signal = 7,
};

const FinishReason = enum {
    audio_completed,
    audio_automatic_stop,
    audio_stopped,
    pipeline_full,
    audio_failed_with_valid_prefix,
};

const TranscriptionRejection = struct {
    model: models.Model,
    publication_ordinal: u32,
    samples_count: u32,
    contains_activity: bool,
    no_speech_probability: f32,
    average_log_probability: f32,
};

const TranscriptCapacityExceeded = struct {
    publication_ordinal: u32,
    bytes_capacity: u32,
    bytes_committed: u32,
    result_bytes_count: u32,
};

const DiscardReason = union(enum) {
    service_signal,
    audio_failed,
    transcription_failed,
    deadline,
    transcript_capacity_exceeded: TranscriptCapacityExceeded,
    speech_detection_conflict: TranscriptionRejection,
    speech_unrecognized: TranscriptionRejection,
};

const SessionPhase = union(enum) {
    active,
    finishing: FinishReason,
    discarding: DiscardReason,
};

const ChildProcess = struct {
    socket: ?std.posix.fd_t,
    pid_fd: std.posix.fd_t,
};

// A role operation and its deadline are one fact. Backend-specific operation
// types prevent fake capture from entering PipeWire progress states and vice
// versa.
const AudioProgress = struct {
    callbacks_count: u64,
    deadline_monotonic_ns: u64,
};

const FakeAudioOperation = union(enum) {
    starting: u64,
    capturing,
    canceling: u64,
    exiting: u64,
    terminating: u64,

    fn deadlineMonotonicNs(operation: FakeAudioOperation) ?u64 {
        return switch (operation) {
            .capturing => null,
            inline else => |deadline_monotonic_ns| deadline_monotonic_ns,
        };
    }
};

const PipeWireAudioOperation = union(enum) {
    starting: u64,
    capturing: AudioProgress,
    canceling: u64,
    exiting: u64,
    terminating: u64,

    fn deadlineMonotonicNs(operation: PipeWireAudioOperation) ?u64 {
        return switch (operation) {
            .capturing => |progress| progress.deadline_monotonic_ns,
            inline else => |deadline_monotonic_ns| deadline_monotonic_ns,
        };
    }
};

const FakeAudioProcess = struct {
    process: ChildProcess,
    operation: FakeAudioOperation,
};

const PipeWireAudioProcess = struct {
    process: ChildProcess,
    operation: PipeWireAudioOperation,
};

const SessionAudio = union(enum) {
    fake: struct {
        scenario: FakeScenario,
        process: ?FakeAudioProcess,
    },
    pipewire: struct {
        options: PipeWireOptions,
        process: ?PipeWireAudioProcess,
    },
};

const TranscriptionWork = struct {
    slot_index: audio_exchange.SlotIndex,
    publication_ordinal: u32,
};

const WorkAttempt = union(enum) {
    first: TranscriptionWork,
    retry: TranscriptionWork,

    fn work(attempt: WorkAttempt) TranscriptionWork {
        return switch (attempt) {
            inline else => |value| value,
        };
    }
};

const StartingTranscription = struct {
    started_monotonic_ns: u64,
    deadline_monotonic_ns: u64,
    retry_work: ?TranscriptionWork,
};

const PendingTranscription = struct {
    deadline_monotonic_ns: u64,
    work: WorkAttempt,
};

const TranscriptionOperation = union(enum) {
    starting: StartingTranscription,
    idle: ?TranscriptionWork,
    busy: PendingTranscription,

    // Normal shutdown preserves accepted text; forced termination during
    // startup or inference discards it. Keep those operations distinct while
    // still storing each operation and its deadline as one tagged value.
    shutdown_sent: u64,
    exiting: u64,
    terminating: u64,

    fn deadlineMonotonicNs(operation: TranscriptionOperation) ?u64 {
        return switch (operation) {
            .idle => null,
            .starting => |starting| starting.deadline_monotonic_ns,
            .busy => |pending| pending.deadline_monotonic_ns,
            inline else => |deadline_monotonic_ns| deadline_monotonic_ns,
        };
    }
};

const TranscriptionProcess = struct {
    process: ChildProcess,
    operation: TranscriptionOperation,
};

const TranscriptionState = union(enum) {
    absent,
    running: TranscriptionProcess,
    restart_pending: TranscriptionWork,
};

const TranscriptProgress = struct {
    next_publication_ordinal: u32,
    accepted_chunks_count: u32,
    no_speech_chunks_count: u32,
    bytes: []u8,
    bytes_count: u32,
};

const TimingProgress = struct {
    command_started_monotonic_ns: u64,
    audio_started_monotonic_ns: ?u64,
    audio_finished_monotonic_ns: ?u64,
    audio_callbacks_count: u64,
    audio_callback_duration_ns_max: u64,
    audio_callback_gap_ns_max: u64,
    model: ?ModelTimingReport,
};

const SharedMapping = struct {
    bytes: []align(std.heap.page_size_min) u8,

    fn pointer(mapping: SharedMapping, comptime T: type) *T {
        assert(mapping.bytes.len == @sizeOf(T));
        return @ptrCast(@alignCast(mapping.bytes.ptr));
    }

    fn unmap(mapping: SharedMapping) void {
        std.posix.munmap(mapping.bytes);
    }
};

const Supervisor = struct {
    io: std.Io,
    audio: SessionAudio,

    epoll_fd: std.posix.fd_t,

    audio_exchange_fd: std.posix.fd_t,
    audio_exchange: *AudioExchange,
    transcript_exchange_fd: std.posix.fd_t,
    transcript_exchange: *TranscriptExchange,

    transcription: TranscriptionState,
    phase: SessionPhase,
    session_deadline_monotonic_ns: u64,

    transcript: TranscriptProgress,
    timing: TimingProgress,
};

/// `runFakeSession` drives deterministic workers through the same process,
/// descriptor, exchange, deadline, and recovery boundaries used by real roles.
pub fn runFakeSession(
    init: std.process.Init,
    scenario: FakeScenario,
) !SessionTimingReport {
    return runSession(init, .{ .fake = scenario });
}

/// `runPipeWireSession` runs one real capture with deterministic or model
/// transcription and returns the timing observations collected across workers.
/// Service residency waits for model readiness before opening the microphone.
/// Retaining that worker between calls still belongs to the long-running
/// supervisor, which this one-session entry point does not yet implement.
pub fn runPipeWireSession(
    init: std.process.Init,
    options: PipeWireOptions,
) !SessionTimingReport {
    switch (options.source) {
        .default => {},
        .node_name, .device_serial => |source| {
            assert(source.len > 0);
            assert(source.len < audio_process.target_name_bytes_capacity);
        },
    }
    assert(options.recording_seconds > 0);
    assert(options.recording_seconds <= recording_duration_seconds_limit);
    assert(options.slot_seconds > 0);
    assert(options.slot_seconds <= audio_process.slot_duration_seconds_max);
    switch (options.transcription) {
        .fake => {},
        .model => |model| {
            assert(model.inference_threads_count > 0);
            assert(model.inference_threads_count <=
                transcription_process.inference_threads_count_max);
        },
    }
    return runSession(init, .{ .pipewire = options });
}

/// `runSession` owns one bounded recording from worker launch through complete
/// process reaping. It prints text only after the audio prefix has been drained
/// in publication order and the transcription process has stopped, then returns
/// the same pipeline timings as structured data.
fn runSession(
    init: std.process.Init,
    configuration: SessionConfiguration,
) !SessionTimingReport {
    const command_started_monotonic_ns = monotonicNanoseconds();

    var signal_mask = std.posix.sigemptyset();
    std.posix.sigaddset(&signal_mask, .TERM);
    std.posix.sigaddset(&signal_mask, .INT);
    std.posix.sigprocmask(std.posix.SIG.BLOCK, &signal_mask, null);

    const signal_fd = try std.posix.signalfd(
        -1,
        &signal_mask,
        linux.SFD.CLOEXEC | linux.SFD.NONBLOCK,
    );
    defer closeDescriptor(signal_fd);
    const epoll_fd = descriptorFromResult(linux.epoll_create1(linux.EPOLL.CLOEXEC));
    defer closeDescriptor(epoll_fd);
    const timer_fd = descriptorFromResult(linux.timerfd_create(.MONOTONIC, .{
        .CLOEXEC = true,
        .NONBLOCK = true,
    }));
    defer closeDescriptor(timer_fd);
    const publication_event_fd = descriptorFromResult(linux.eventfd(
        0,
        linux.EFD.CLOEXEC | linux.EFD.NONBLOCK,
    ));
    defer closeDescriptor(publication_event_fd);

    const audio_exchange_fd = try createSharedMemory(
        "voiced-supervisor-audio",
        @sizeOf(AudioExchange),
    );
    defer closeDescriptor(audio_exchange_fd);
    const audio_mapping = try mapSharedMemory(audio_exchange_fd, AudioExchange);
    defer audio_mapping.unmap();
    audio_exchange.initialize(audio_mapping.pointer(AudioExchange), session_id);

    const transcript_exchange_fd = try createSharedMemory(
        "voiced-supervisor-transcript",
        @sizeOf(TranscriptExchange),
    );
    defer closeDescriptor(transcript_exchange_fd);
    const transcript_mapping = try mapSharedMemory(
        transcript_exchange_fd,
        TranscriptExchange,
    );
    defer transcript_mapping.unmap();
    transcription_process.initializeExchange(
        transcript_mapping.pointer(TranscriptExchange),
        session_id,
    );

    // The exchange retains only three PCM slots regardless of session length.
    // Text is the one session-sized value: reserve 64 UTF-8 bytes per configured
    // audio second plus one complete 4 KiB mailbox result. The rate budget keeps
    // the one-hour default near 229 KiB without depending on chunk frequency;
    // the result reserve lets any one valid mailbox publication fit.
    // Accepted text that exceeds the fixed budget ends the session explicitly
    // instead of growing memory or silently truncating output.
    const transcript_bytes_capacity = sessionTranscriptBytesCapacity(configuration);
    const transcript_bytes = try init.gpa.alloc(u8, transcript_bytes_capacity);
    defer init.gpa.free(transcript_bytes);
    @memset(transcript_bytes, 0);

    var supervisor: Supervisor = .{
        .io = init.io,
        .audio = switch (configuration) {
            .fake => |scenario| .{ .fake = .{
                .scenario = scenario,
                .process = null,
            } },
            .pipewire => |options| .{ .pipewire = .{
                .options = options,
                .process = null,
            } },
        },
        .epoll_fd = epoll_fd,
        .audio_exchange_fd = audio_exchange_fd,
        .audio_exchange = audio_mapping.pointer(AudioExchange),
        .transcript_exchange_fd = transcript_exchange_fd,
        .transcript_exchange = transcript_mapping.pointer(TranscriptExchange),
        .transcription = .absent,
        .phase = .active,
        .session_deadline_monotonic_ns = sessionDeadline(configuration),
        .transcript = .{
            .next_publication_ordinal = 0,
            .accepted_chunks_count = 0,
            .no_speech_chunks_count = 0,
            .bytes = transcript_bytes,
            .bytes_count = 0,
        },
        .timing = .{
            .command_started_monotonic_ns = command_started_monotonic_ns,
            .audio_started_monotonic_ns = null,
            .audio_finished_monotonic_ns = null,
            .audio_callbacks_count = 0,
            .audio_callback_duration_ns_max = 0,
            .audio_callback_gap_ns_max = 0,
            .model = switch (configuration) {
                .fake => null,
                .pipewire => |options| switch (options.transcription) {
                    .fake => null,
                    .model => |model| .{
                        .residency = model.residency,
                        .workers_started_count = 0,
                        .workers_ready_count = 0,
                        .worker_start_to_ready_elapsed_ns_total = 0,
                        .worker_start_to_ready_elapsed_ns_max = 0,
                        .model_load_elapsed_ns_total = 0,
                        .model_load_elapsed_ns_max = 0,
                        .warmup_elapsed_ns_total = 0,
                        .warmup_elapsed_ns_max = 0,
                        .feature_extraction_elapsed_ns_total = 0,
                        .feature_extraction_elapsed_ns_max = 0,
                        .inference_elapsed_ns_total = 0,
                        .inference_elapsed_ns_max = 0,
                        .transcriptions_count = 0,
                    },
                },
            },
        },
    };

    try register(epoll_fd, publication_event_fd, .audio_publication);
    try register(epoll_fd, timer_fd, .deadline);
    try register(epoll_fd, signal_fd, .service_signal);

    const model_is_service_resident = switch (configuration) {
        .fake => false,
        .pipewire => |options| switch (options.transcription) {
            .fake => false,
            .model => |model| model.residency == .service,
        },
    };
    supervisor.transcription = .{
        .running = try startTranscription(&supervisor, null),
    };
    if (model_is_service_resident) {
        // A production service performs this work before accepting a recording.
        // The one-session spike waits here to reproduce that ready-at-capture
        // boundary without pretending its process survives command exit.
        try awaitServiceModelReadiness(&supervisor, timer_fd, signal_fd);
        if (supervisor.phase == .active) {
            supervisor.session_deadline_monotonic_ns = sessionDeadline(configuration);
            try startAudio(&supervisor, publication_event_fd);
        }
    } else {
        try startAudio(&supervisor, publication_event_fd);
    }
    armNearestDeadline(&supervisor, timer_fd);

    // Every iteration consumes one complete readiness snapshot. Worker
    // replacement happens only in the final maintenance paragraph, after no old
    // event from this snapshot can be applied to the replacement.
    while (!sessionIsComplete(&supervisor)) {
        var events: [epoll_events_count_max]linux.epoll_event = undefined;
        // timerfd represents the nearest absolute deadline, so an unbounded
        // epoll wait is still bounded by current session policy. A separate
        // polling timeout would merely create a second, inconsistent deadline.
        const events_count = waitForEvents(epoll_fd, &events, -1);
        assert(events_count > 0);

        for (events[0..events_count]) |event| {
            if (eventSource(event) != .audio_publication) continue;
            _ = readEventCounter(publication_event_fd);
        }

        // Final packets and process exit frequently arrive in one batch. Drain
        // role sockets first; pidfd readiness never means the packet is lost.
        for (events[0..events_count]) |event| {
            switch (eventSource(event)) {
                .audio_packet => if (audioProcessExists(&supervisor))
                    try drainAudioPackets(&supervisor),
                .transcription_packet => if (supervisor.transcription == .running)
                    try drainTranscriptionPackets(&supervisor),
                else => {},
            }
        }
        try dispatchTranscription(&supervisor);

        observePipeWireProgress(&supervisor);

        for (events[0..events_count]) |event| {
            switch (eventSource(event)) {
                .audio_exit => if (audioProcessExists(&supervisor))
                    try reapAudio(&supervisor),
                .transcription_exit => if (supervisor.transcription == .running)
                    try reapTranscription(&supervisor),
                .deadline => {
                    _ = readEventCounter(timer_fd);
                    try applyExpiredDeadlines(&supervisor);
                },
                .service_signal => {
                    _ = readSignal(signal_fd);
                    try beginDiscard(&supervisor, .service_signal);
                },
                else => {},
            }
        }

        // Shared counts remain authoritative when eventfd increments coalesce or
        // when a worker publishes text and dies before sending its report.
        try dispatchTranscription(&supervisor);
        try maintainWorkersAndSession(&supervisor);
        armNearestDeadline(&supervisor, timer_fd);
    }

    return finishSession(&supervisor);
}

/// Service residency requires the model to finish loading and warming before
/// capture can begin. This bounded wait uses the same packet, pidfd, signal, and
/// deadline handling as the session event loop; it opens no second lifecycle.
fn awaitServiceModelReadiness(
    supervisor: *Supervisor,
    timer_fd: std.posix.fd_t,
    signal_fd: std.posix.fd_t,
) !void {
    assert(!audioProcessExists(supervisor));
    assert(supervisor.transcription == .running);
    assert(supervisor.transcription.running.operation == .starting);

    armNearestDeadline(supervisor, timer_fd);
    while (supervisor.phase == .active and
        supervisor.transcription == .running and
        supervisor.transcription.running.operation == .starting)
    {
        var events: [epoll_events_count_max]linux.epoll_event = undefined;
        const events_count = waitForEvents(supervisor.epoll_fd, &events, -1);
        assert(events_count > 0);

        for (events[0..events_count]) |event| {
            if (eventSource(event) == .transcription_packet and
                supervisor.transcription == .running)
            {
                try drainTranscriptionPackets(supervisor);
            }
        }
        for (events[0..events_count]) |event| {
            switch (eventSource(event)) {
                .transcription_exit => if (supervisor.transcription == .running)
                    try reapTranscription(supervisor),
                .deadline => {
                    _ = readEventCounter(timer_fd);
                    try applyExpiredDeadlines(supervisor);
                },
                .service_signal => {
                    _ = readSignal(signal_fd);
                    try beginDiscard(supervisor, .service_signal);
                },
                else => {},
            }
        }

        try maintainWorkersAndSession(supervisor);
        armNearestDeadline(supervisor, timer_fd);
    }

    if (supervisor.phase == .active) {
        assert(supervisor.transcription == .running);
        assert(supervisor.transcription.running.operation == .idle);
    }
}

fn sessionTranscriptBytesCapacity(configuration: SessionConfiguration) usize {
    const bytes_per_recording_second_max: u64 = 64;
    const result_bytes_reserve: u64 =
        transcription_process.result_bytes_capacity;

    const bytes_capacity: u64 = switch (configuration) {
        // Deterministic scenarios intentionally stress publication counts rather
        // than recording duration. Preserve one complete result per fake chunk
        // so these protocol tests cannot fail on the production text-rate policy.
        .fake => |scenario| @as(u64, scenarioBehavior(scenario).chunks_count) *
            transcription_process.result_bytes_capacity,
        .pipewire => |options| @as(u64, options.recording_seconds) *
            bytes_per_recording_second_max + result_bytes_reserve,
    };
    assert(bytes_capacity > 0);
    assert(bytes_capacity <= std.math.maxInt(u32));
    return @intCast(bytes_capacity);
}

fn startAudio(
    supervisor: *Supervisor,
    publication_event_fd: std.posix.fd_t,
) !void {
    const internal_role = switch (supervisor.audio) {
        .fake => "audio-fake",
        .pipewire => "audio-pipewire",
    };
    var process = try startChild(
        supervisor,
        .audio_packet,
        .audio_exit,
        internal_role,
    );
    errdefer forceStopAndReap(&process) catch {};

    assert(supervisor.timing.audio_started_monotonic_ns == null);

    switch (supervisor.audio) {
        .fake => |*audio| {
            assert(audio.process == null);
            const behavior = scenarioBehavior(audio.scenario);
            try audio_process.FakeWorker.sendLaunch(
                process.socket.?,
                supervisor.audio_exchange_fd,
                publication_event_fd,
                .{
                    .session_id = session_id,
                    .chunks_count = behavior.chunks_count,
                    .publication_interval_ms = behavior.audio_interval_ms,
                },
            );
            audio.process = .{
                .process = process,
                .operation = .{
                    .starting = monotonicNanoseconds() + std.time.ns_per_s,
                },
            };
        },
        .pipewire => |*audio| {
            assert(audio.process == null);
            const options = audio.options;
            try audio_process.PipeWireWorker.sendLaunch(
                process.socket.?,
                supervisor.audio_exchange_fd,
                publication_event_fd,
                .{
                    .session_id = session_id,
                    .source = options.source,
                    .recording_samples_target = @as(u32, options.recording_seconds) *
                        audio_process.sample_rate_hz,
                    .slot_samples_boundary = @as(u32, options.slot_seconds) *
                        audio_process.sample_rate_hz,
                    .automatic_stop = options.automatic_stop,
                    .process_realtime = options.process_realtime,
                },
            );
            audio.process = .{
                .process = process,
                .operation = .{
                    .starting = monotonicNanoseconds() + 3 * std.time.ns_per_s,
                },
            };
        },
    }

    supervisor.timing.audio_started_monotonic_ns = monotonicNanoseconds();
}

fn startTranscription(
    supervisor: *Supervisor,
    retry_work: ?TranscriptionWork,
) !TranscriptionProcess {
    const backend: TranscriptionBackend = switch (supervisor.audio) {
        .fake => .fake,
        .pipewire => |audio| audio.options.transcription,
    };
    const internal_role = switch (backend) {
        .fake => "transcription-fake",
        .model => "transcription-model",
    };
    var process = try startChild(
        supervisor,
        .transcription_packet,
        .transcription_exit,
        internal_role,
    );
    errdefer forceStopAndReap(&process) catch {};

    // A replacement receives the same model configuration and exact retained
    // slot. Deterministic fault scenarios alone alter behavior on replacement:
    // one-shot crash cases recover, while the repeated-crash case proves the
    // retry bound by failing the same work twice.
    switch (backend) {
        .fake => {
            const behavior = transcriptionBehavior(supervisor.audio);
            const repeats_crash = switch (supervisor.audio) {
                .fake => |audio| audio.scenario == .repeated_transcription_crash,
                .pipewire => false,
            };
            const is_replacement = retry_work != null;
            const fake_behavior = if (!is_replacement or repeats_crash)
                behavior.behavior
            else
                transcription_process.FakeBehavior.normal;
            try transcription_process.sendFakeLaunch(
                process.socket.?,
                supervisor.audio_exchange_fd,
                supervisor.transcript_exchange_fd,
                .{
                    .session_id = session_id,
                    .behavior = fake_behavior,
                    .inference_duration_ms = behavior.duration_ms,
                },
            );
        },
        .model => |model| try transcription_process.sendModelLaunch(
            process.socket.?,
            supervisor.audio_exchange_fd,
            supervisor.transcript_exchange_fd,
            .{
                .session_id = session_id,
                .model = model.model,
                .inference_threads_count = model.inference_threads_count,
            },
        ),
    }

    const startup_duration_ns: u64 = switch (backend) {
        .fake => std.time.ns_per_s,
        .model => 15 * std.time.ns_per_s,
    };
    const started_monotonic_ns = monotonicNanoseconds();
    switch (backend) {
        .fake => {},
        .model => {
            const timing = &supervisor.timing.model.?;
            timing.workers_started_count += 1;
        },
    }
    return .{
        .process = process,
        .operation = .{ .starting = .{
            .started_monotonic_ns = started_monotonic_ns,
            .deadline_monotonic_ns = started_monotonic_ns + startup_duration_ns,
            .retry_work = retry_work,
        } },
    };
}

const ScenarioBehavior = struct {
    chunks_count: u32,
    audio_interval_ms: u32,
    transcription_duration_ms: u32,
    transcription_behavior: transcription_process.FakeBehavior,
};

const TranscriptionBehavior = struct {
    duration_ms: u32,
    behavior: transcription_process.FakeBehavior,
};

fn transcriptionBehavior(audio: SessionAudio) TranscriptionBehavior {
    return switch (audio) {
        .fake => |fake_audio| behavior: {
            const scenario_behavior = scenarioBehavior(fake_audio.scenario);
            break :behavior .{
                .duration_ms = scenario_behavior.transcription_duration_ms,
                .behavior = scenario_behavior.transcription_behavior,
            };
        },
        .pipewire => .{
            .duration_ms = 4,
            .behavior = .normal,
        },
    };
}

fn scenarioBehavior(scenario: FakeScenario) ScenarioBehavior {
    return switch (scenario) {
        .normal => .{
            .chunks_count = 8,
            .audio_interval_ms = 8,
            .transcription_duration_ms = 4,
            .transcription_behavior = .normal,
        },
        .burst_publications => .{
            .chunks_count = 3,
            .audio_interval_ms = 0,
            .transcription_duration_ms = 4,
            .transcription_behavior = .normal,
        },
        .slow_transcription => .{
            .chunks_count = 8,
            .audio_interval_ms = 2,
            .transcription_duration_ms = 50,
            .transcription_behavior = .normal,
        },
        .transcription_crash_before_result => .{
            .chunks_count = 8,
            .audio_interval_ms = 8,
            .transcription_duration_ms = 4,
            .transcription_behavior = .crash_before_result,
        },
        .transcription_crash_after_result => .{
            .chunks_count = 8,
            .audio_interval_ms = 8,
            .transcription_duration_ms = 4,
            .transcription_behavior = .crash_after_result,
        },
        .repeated_transcription_crash => .{
            .chunks_count = 8,
            .audio_interval_ms = 8,
            .transcription_duration_ms = 4,
            .transcription_behavior = .crash_before_result,
        },
        .transcription_hang => .{
            .chunks_count = 20,
            .audio_interval_ms = 35,
            .transcription_duration_ms = 4,
            .transcription_behavior = .hang,
        },
    };
}

fn startChild(
    supervisor: *Supervisor,
    socket_source: EventSource,
    pid_source: EventSource,
    internal_role: []const u8,
) !ChildProcess {
    var sockets: [2]std.posix.fd_t = undefined;
    checkSyscall(linux.socketpair(
        linux.AF.UNIX,
        linux.SOCK.SEQPACKET,
        0,
        &sockets,
    ));
    const supervisor_socket = sockets[0];
    const worker_socket = sockets[1];
    var worker_socket_is_owned = true;
    errdefer closeDescriptor(supervisor_socket);
    errdefer if (worker_socket_is_owned) closeDescriptor(worker_socket);
    setCloseOnExec(supervisor_socket);

    var worker_socket_text_buffer: [32]u8 = undefined;
    const worker_socket_text = try std.fmt.bufPrint(
        &worker_socket_text_buffer,
        "{d}",
        .{worker_socket},
    );
    var supervisor_pid_text_buffer: [32]u8 = undefined;
    const supervisor_pid_text = try std.fmt.bufPrint(
        &supervisor_pid_text_buffer,
        "{d}",
        .{linux.getpid()},
    );

    var child = try std.process.spawn(supervisor.io, .{
        .argv = &.{
            "/proc/self/exe",
            "--internal-role",
            internal_role,
            worker_socket_text,
            supervisor_pid_text,
        },
    });
    const pid = child.id.?;
    assert(child.stdin == null);
    assert(child.stdout == null);
    assert(child.stderr == null);
    errdefer child.kill(supervisor.io);
    closeDescriptor(worker_socket);
    worker_socket_is_owned = false;

    const pid_fd = descriptorFromResult(linux.pidfd_open(pid, 0));
    errdefer closeDescriptor(pid_fd);
    try register(supervisor.epoll_fd, supervisor_socket, socket_source);
    errdefer unregister(supervisor.epoll_fd, supervisor_socket);
    try register(supervisor.epoll_fd, pid_fd, pid_source);
    errdefer unregister(supervisor.epoll_fd, pid_fd);

    return .{
        .socket = supervisor_socket,
        .pid_fd = pid_fd,
    };
}

fn drainAudioPackets(supervisor: *Supervisor) !void {
    switch (supervisor.audio) {
        .fake => try drainFakeAudioPackets(supervisor),
        .pipewire => try drainPipeWireAudioPackets(supervisor),
    }
}

fn drainFakeAudioPackets(supervisor: *Supervisor) !void {
    const audio = &supervisor.audio.fake.process.?;
    while (receiveRecordNonblocking(
        audio.process.socket.?,
        audio_process.FakeWorker.ReportPacket,
    )) |received| {
        const report = received orelse {
            unregisterSocket(supervisor.epoll_fd, &audio.process);
            return;
        };
        const report_kind = audio_process.FakeWorker.decodeTrustedReport(report);
        switch (report_kind) {
            .ready => switch (audio.operation) {
                .starting => audio.operation = .capturing,
                .canceling, .terminating => {},
                .capturing, .exiting => unreachable,
            },
            .completed, .pipeline_full => switch (audio.operation) {
                .capturing, .canceling => {
                    if (supervisor.timing.audio_finished_monotonic_ns == null) {
                        supervisor.timing.audio_finished_monotonic_ns =
                            monotonicNanoseconds();
                    }
                    audio.operation = .{
                        .exiting = monotonicNanoseconds() + std.time.ns_per_s,
                    };
                    if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = if (report_kind == .completed)
                            .audio_completed
                        else
                            .pipeline_full };
                    }
                },
                // SIGKILL may race a terminal packet already queued by the
                // worker. The pidfd remains authoritative for reap progress.
                .terminating => {},
                .starting, .exiting => unreachable,
            },
            .cancelled => switch (audio.operation) {
                .canceling => {
                    if (supervisor.timing.audio_finished_monotonic_ns == null) {
                        supervisor.timing.audio_finished_monotonic_ns =
                            monotonicNanoseconds();
                    }
                    audio.operation = .{
                        .exiting = monotonicNanoseconds() + std.time.ns_per_s,
                    };
                },
                .terminating => {},
                .starting, .capturing, .exiting => unreachable,
            },
        }
    }
}

fn drainPipeWireAudioPackets(supervisor: *Supervisor) !void {
    const session_audio = &supervisor.audio.pipewire;
    const options = session_audio.options;
    const audio = &session_audio.process.?;
    const launch: audio_process.PipeWireWorker.LaunchOptions = .{
        .session_id = session_id,
        .source = options.source,
        .recording_samples_target = @as(u32, options.recording_seconds) *
            audio_process.sample_rate_hz,
        .slot_samples_boundary = @as(u32, options.slot_seconds) *
            audio_process.sample_rate_hz,
        .automatic_stop = options.automatic_stop,
        .process_realtime = options.process_realtime,
    };
    while (audio_process.PipeWireWorker.receiveReportNonblocking(
        audio.process.socket.?,
        supervisor.audio_exchange,
        launch,
    )) |received| {
        const worker_report = received orelse {
            unregisterSocket(supervisor.epoll_fd, &audio.process);
            return;
        };
        audio.operation = .{
            .exiting = monotonicNanoseconds() + std.time.ns_per_s,
        };
        if (supervisor.timing.audio_finished_monotonic_ns == null) {
            supervisor.timing.audio_finished_monotonic_ns = monotonicNanoseconds();
        }

        switch (worker_report) {
            .setup_failed => |setup_failure| {
                printPipeWireSetupFailure(&setup_failure);
                try beginDiscard(supervisor, .audio_failed);
            },
            .captured => |capture| {
                const timeline_validation = audio_exchange.acquireTimelineValidation(
                    supervisor.audio_exchange,
                ).?;
                assert(switch (capture.timeline_validation) {
                    .header_only => timeline_validation == .header_only,
                    .full => timeline_validation == .full,
                });
                if (capture.callback) |callback| {
                    supervisor.timing.audio_callbacks_count = callback.callbacks_count;
                    supervisor.timing.audio_callback_duration_ns_max =
                        callback.duration_ns_max;
                    supervisor.timing.audio_callback_gap_ns_max = callback.gap_ns_max;
                }
                printPipeWireCapture(&capture);
                switch (capture.end) {
                    .completed => if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = .audio_completed };
                    },
                    .automatic_stop => if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = .audio_automatic_stop };
                    },
                    .stopped => if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = .audio_stopped };
                    },
                    .cancelled => assert(supervisor.phase == .discarding),
                    .failed => |failure| if (supervisor.phase == .active) {
                        if (failure.outcome == .pipeline_full) {
                            supervisor.phase = .{ .finishing = .pipeline_full };
                        } else if (capture.published_samples_count == 0) {
                            try beginDiscard(supervisor, .audio_failed);
                        } else {
                            // PipeWire reports preserve every complete block before a
                            // disconnect, malformed buffer, or timeline failure. Drain
                            // and transcribe that valid prefix, but retain the failure
                            // outcome so policy cannot mistake it for completion.
                            supervisor.phase = .{
                                .finishing = .audio_failed_with_valid_prefix,
                            };
                        }
                    },
                }
            },
        }
    }
}

fn observePipeWireProgress(supervisor: *Supervisor) void {
    if (supervisor.audio != .pipewire) return;
    const audio = if (supervisor.audio.pipewire.process) |*value| value else return;
    const capture_is_starting = audio.operation == .starting;
    const callbacks_count_previous = switch (audio.operation) {
        .starting => 0,
        .capturing => |progress| progress.callbacks_count,
        .canceling, .exiting, .terminating => return,
    };

    const callbacks_count = audio_exchange.acquireAudioCallbacksCount(
        supervisor.audio_exchange,
    );
    assert(callbacks_count >= callbacks_count_previous);
    if (callbacks_count == callbacks_count_previous) return;

    if (capture_is_starting) {
        const timeline_validation = audio_exchange.acquireTimelineValidation(
            supervisor.audio_exchange,
        ).?;
        if (timeline_validation == .header_only) {
            std.debug.print(
                "Warning: this PipeWire build supplies Header-only timeline " ++
                    "validation; some dropped audio intervals cannot be detected\n",
                .{},
            );
        }
    }

    audio.operation = .{ .capturing = .{
        .callbacks_count = callbacks_count,
        .deadline_monotonic_ns = monotonicNanoseconds() +
            2 * std.time.ns_per_s,
    } };
}

fn drainTranscriptionPackets(supervisor: *Supervisor) !void {
    const transcription = &supervisor.transcription.running;
    while (transcription_process.receiveReportNonblocking(
        transcription.process.socket.?,
    )) |received| {
        const report = received orelse {
            unregisterSocket(supervisor.epoll_fd, &transcription.process);
            return;
        };
        switch (report) {
            .ready => |ready| switch (transcription.operation) {
                .starting => |starting| {
                    if (supervisor.timing.model) |*timing| {
                        const start_to_ready_elapsed_ns =
                            monotonicNanoseconds() - starting.started_monotonic_ns;
                        timing.workers_ready_count += 1;
                        timing.worker_start_to_ready_elapsed_ns_total +=
                            start_to_ready_elapsed_ns;
                        timing.worker_start_to_ready_elapsed_ns_max = @max(
                            timing.worker_start_to_ready_elapsed_ns_max,
                            start_to_ready_elapsed_ns,
                        );
                        timing.model_load_elapsed_ns_total +=
                            ready.model_load_elapsed_ns;
                        timing.model_load_elapsed_ns_max = @max(
                            timing.model_load_elapsed_ns_max,
                            ready.model_load_elapsed_ns,
                        );
                        timing.warmup_elapsed_ns_total += ready.warmup_elapsed_ns;
                        timing.warmup_elapsed_ns_max = @max(
                            timing.warmup_elapsed_ns_max,
                            ready.warmup_elapsed_ns,
                        );
                    }
                    if (ready.model_load_elapsed_ns > 0 or ready.warmup_elapsed_ns > 0) {
                        std.debug.print(
                            "Whisper ready\n" ++
                                "Model load: {d} ms\n" ++
                                "Warm-up: {d} ms\n",
                            .{
                                ready.model_load_elapsed_ns / std.time.ns_per_ms,
                                ready.warmup_elapsed_ns / std.time.ns_per_ms,
                            },
                        );
                    }
                    transcription.operation = .{ .idle = starting.retry_work };
                },
                .terminating => {},
                .idle, .busy, .shutdown_sent, .exiting => unreachable,
            },
            .result => |result| {
                if (supervisor.phase == .discarding or
                    transcription.operation == .terminating)
                {
                    // Cancellation owns the mailbox terminal state. A result
                    // packet already queued before termination cannot revive it.
                    continue;
                }
                assert(transcription.operation == .busy);
                const work = transcription.operation.busy.work.work();
                assert(result.publication_ordinal == work.publication_ordinal);
                const committed = transcription_process.acquireResult(
                    supervisor.transcript_exchange,
                ).?;
                assert(committed.publication_ordinal == result.publication_ordinal);
                assert(committed.samples_count == result.samples_count);
                assert(committed.no_speech_probability == result.no_speech_probability);
                assert(committed.average_log_probability == result.average_log_probability);
                std.debug.print(
                    "Whisper chunk {d}: {d} samples, features {d} ms, " ++
                        "inference {d} ms, no-speech {d:.6}, " ++
                        "average-log-probability {d:.6}\n",
                    .{
                        result.publication_ordinal,
                        result.samples_count,
                        result.feature_extraction_elapsed_ns / std.time.ns_per_ms,
                        result.inference_elapsed_ns / std.time.ns_per_ms,
                        result.no_speech_probability,
                        result.average_log_probability,
                    },
                );
                if (supervisor.timing.model) |*timing| {
                    timing.feature_extraction_elapsed_ns_total +=
                        result.feature_extraction_elapsed_ns;
                    timing.feature_extraction_elapsed_ns_max = @max(
                        timing.feature_extraction_elapsed_ns_max,
                        result.feature_extraction_elapsed_ns,
                    );
                    timing.inference_elapsed_ns_total += result.inference_elapsed_ns;
                    timing.inference_elapsed_ns_max = @max(
                        timing.inference_elapsed_ns_max,
                        result.inference_elapsed_ns,
                    );
                    timing.transcriptions_count += 1;
                }
                try consumeTranscript(supervisor);
                if (supervisor.phase != .discarding) {
                    transcription.operation = .{ .idle = null };
                }
            },
            .stopped => {
                assert(transcription.operation == .shutdown_sent or
                    transcription.operation == .terminating);
                transcription.operation = .{
                    .exiting = monotonicNanoseconds() + std.time.ns_per_s,
                };
            },
            .failed => |failure| {
                std.debug.print(
                    "Whisper failure\nStage: {s}\nDetail: {s}\n",
                    .{ @tagName(failure.stage), failure.messageBytes() },
                );
                // The worker exits after reporting. pidfd reaping below decides
                // whether exact in-flight work receives its one bounded retry;
                // this diagnostic packet does not create a second failure owner.
            },
        }
    }
}

fn dispatchTranscription(supervisor: *Supervisor) !void {
    if (supervisor.phase == .discarding) return;
    if (supervisor.transcription != .running) return;
    const transcription = &supervisor.transcription.running;
    const retry_work = switch (transcription.operation) {
        .idle => |work| work,
        else => return,
    };
    const socket = transcription.process.socket orelse return;

    const work_attempt: WorkAttempt = if (retry_work) |work|
        .{ .retry = work }
    else fresh: {
        const pending = nextPublishedAudio(supervisor) orelse return;
        break :fresh .{ .first = .{
            .slot_index = pending.index,
            .publication_ordinal = pending.publication.publication_ordinal,
        } };
    };
    const work = work_attempt.work();
    const inference_duration_ns: u64 = switch (supervisor.audio) {
        .fake => 80 * std.time.ns_per_ms,
        .pipewire => |audio| switch (audio.options.transcription) {
            .fake => 80 * std.time.ns_per_ms,
            .model => 10 * std.time.ns_per_s,
        },
    };
    transcription.operation = .{ .busy = .{
        .deadline_monotonic_ns = monotonicNanoseconds() + inference_duration_ns,
        .work = work_attempt,
    } };
    transcription_process.sendCommand(
        socket,
        .{ .transcribe = work.slot_index },
    ) catch |send_error| switch (send_error) {
        error.TranscriptionPeerClosed => {
            unregisterSocket(supervisor.epoll_fd, &transcription.process);
            return;
        },
        else => return send_error,
    };
}

fn consumeTranscript(supervisor: *Supervisor) !void {
    assert(supervisor.transcription == .running);
    assert(supervisor.transcription.running.operation == .busy);
    const work = supervisor.transcription.running.operation.busy.work.work();
    const slot = &supervisor.audio_exchange.slots[work.slot_index.arrayIndex()];
    const publication = audio_exchange.acquirePublishedSlot(slot).?;
    assert(publication.publication_ordinal == work.publication_ordinal);

    const result = transcription_process.acquireResult(
        supervisor.transcript_exchange,
    ).?;
    assert(result.publication_ordinal == work.publication_ordinal);
    assert(result.publication_ordinal ==
        supervisor.transcript.next_publication_ordinal);
    assert(result.samples_count == publication.samples_count);
    assert(result.contains_activity == publication.contains_activity);

    // The deterministic fake worker produces protocol text, not Whisper
    // confidence scores. Only model results can be accepted or rejected by
    // comparing Whisper's score with the audio activity detector.
    const disposition: transcription_process.ResultDisposition = switch (supervisor.audio) {
        .fake => .accepted,
        .pipewire => |audio| switch (audio.options.transcription) {
            .fake => .accepted,
            .model => transcription_process.classifyResult(result),
        },
    };
    switch (disposition) {
        .accepted => {
            assert(supervisor.transcript.bytes_count <= supervisor.transcript.bytes.len);
            const bytes_remaining = supervisor.transcript.bytes.len -
                supervisor.transcript.bytes_count;
            if (result.bytes.len > bytes_remaining) {
                const capacity_exceeded: TranscriptCapacityExceeded = .{
                    .publication_ordinal = result.publication_ordinal,
                    .bytes_capacity = @intCast(supervisor.transcript.bytes.len),
                    .bytes_committed = supervisor.transcript.bytes_count,
                    .result_bytes_count = @intCast(result.bytes.len),
                };
                transcription_process.releaseResult(supervisor.transcript_exchange);
                try beginDiscard(supervisor, .{
                    .transcript_capacity_exceeded = capacity_exceeded,
                });
                return;
            }

            @memcpy(
                supervisor.transcript.bytes[supervisor.transcript.bytes_count..][0..result.bytes.len],
                result.bytes,
            );
            supervisor.transcript.bytes_count += @intCast(result.bytes.len);
            supervisor.transcript.accepted_chunks_count += 1;
        },
        .no_speech => {
            supervisor.transcript.no_speech_chunks_count += 1;
            if (std.mem.trim(u8, result.bytes, " \t\r\n").len == 0) {
                std.debug.print(
                    "Whisper chunk {d} rejected as no speech: model output is empty, " ++
                        "activity={s}, no-speech={d:.6}\n",
                    .{
                        result.publication_ordinal,
                        if (result.contains_activity) "active" else "not observed",
                        result.no_speech_probability,
                    },
                );
            } else {
                std.debug.print(
                    "Whisper chunk {d} rejected as no speech: activity={s}, " ++
                        "no-speech={d:.6}, inactive threshold={d:.2}\n",
                    .{
                        result.publication_ordinal,
                        if (result.contains_activity) "active" else "not observed",
                        result.no_speech_probability,
                        transcription_process.no_activity_no_speech_probability_reject_min,
                    },
                );
            }
        },
        .speech_detection_conflict, .speech_unrecognized => {
            const selected_model = switch (supervisor.audio) {
                .fake => unreachable,
                .pipewire => |audio| switch (audio.options.transcription) {
                    .fake => unreachable,
                    .model => |model| model.model,
                },
            };
            const rejection: TranscriptionRejection = .{
                .model = selected_model,
                .publication_ordinal = result.publication_ordinal,
                .samples_count = result.samples_count,
                .contains_activity = result.contains_activity,
                .no_speech_probability = result.no_speech_probability,
                .average_log_probability = result.average_log_probability,
            };
            transcription_process.releaseResult(supervisor.transcript_exchange);
            try beginDiscard(supervisor, switch (disposition) {
                .speech_detection_conflict => .{
                    .speech_detection_conflict = rejection,
                },
                .speech_unrecognized => .{ .speech_unrecognized = rejection },
                .accepted, .no_speech => unreachable,
            });
            return;
        },
    }
    supervisor.transcript.next_publication_ordinal += 1;

    transcription_process.releaseResult(supervisor.transcript_exchange);
    audio_exchange.releaseConsumedSlot(slot);
}

fn nextPublishedAudio(supervisor: *Supervisor) ?audio_exchange.PublishedSlotRef {
    for (&supervisor.audio_exchange.slots, 0..) |*slot, slot_index| {
        const publication = audio_exchange.acquirePublishedSlot(slot) orelse
            continue;
        if (publication.publication_ordinal !=
            supervisor.transcript.next_publication_ordinal) continue;
        return .{
            .index = audio_exchange.SlotIndex.fromArrayIndex(slot_index),
            .publication = publication,
        };
    }
    return null;
}

fn reapAudio(supervisor: *Supervisor) !void {
    if (supervisor.timing.audio_finished_monotonic_ns == null) {
        supervisor.timing.audio_finished_monotonic_ns = monotonicNanoseconds();
    }

    switch (supervisor.audio) {
        inline else => |*session_audio| {
            if (session_audio.process.?.process.socket != null) {
                try drainAudioPackets(supervisor);
            }
            var audio = session_audio.process.?;
            unregisterChild(supervisor.epoll_fd, &audio.process);
            _ = try reapChild(&audio.process);
            closeDescriptor(audio.process.pid_fd);
            session_audio.process = null;
        },
    }

    if (supervisor.phase == .active) {
        try finishValidAudioPrefixOrDiscard(supervisor);
    }
}

fn reapTranscription(supervisor: *Supervisor) !void {
    assert(supervisor.transcription == .running);
    if (supervisor.transcription.running.process.socket != null) {
        try drainTranscriptionPackets(supervisor);
    }
    var transcription = supervisor.transcription.running;
    unregisterChild(supervisor.epoll_fd, &transcription.process);
    _ = try reapChild(&transcription.process);
    closeDescriptor(transcription.process.pid_fd);

    if (supervisor.phase == .discarding or
        transcription.operation == .exiting or
        transcription.operation == .terminating)
    {
        supervisor.transcription = .absent;
        return;
    }

    if (transcription.operation != .busy) {
        supervisor.transcription = .absent;
        try beginDiscard(supervisor, .transcription_failed);
        return;
    }
    const failed_work = transcription.operation.busy.work;
    if (transcription_process.acquireResult(supervisor.transcript_exchange) != null) {
        try consumeTranscript(supervisor);
        supervisor.transcription = .absent;
    } else switch (failed_work) {
        .retry => {
            supervisor.transcription = .absent;
            try beginDiscard(supervisor, .transcription_failed);
        },
        .first => |work| supervisor.transcription = .{ .restart_pending = work },
    }
}

fn maintainWorkersAndSession(supervisor: *Supervisor) !void {
    if (supervisor.phase == .discarding) return;

    // A retained retry has its own state while no process exists. Starting the
    // replacement here prevents events from the reaped process's epoll batch
    // from being applied to its successor.
    if (supervisor.transcription == .restart_pending) {
        const work = supervisor.transcription.restart_pending;
        supervisor.transcription = .{
            .running = try startTranscription(supervisor, work),
        };
        return;
    }
    if (supervisor.transcription == .absent and
        countPublishedSlots(supervisor) > 0)
    {
        supervisor.transcription = .{
            .running = try startTranscription(supervisor, null),
        };
        return;
    }

    if (supervisor.phase != .finishing) return;
    if (audioProcessExists(supervisor)) return;
    if (countPublishedSlots(supervisor) != 0) return;
    if (supervisor.transcription != .running) return;
    const transcription = &supervisor.transcription.running;
    if (transcription.operation != .idle) return;
    const socket = transcription.process.socket orelse return;

    transcription.operation = .{
        .shutdown_sent = monotonicNanoseconds() + std.time.ns_per_s,
    };
    transcription_process.sendCommand(
        socket,
        .shutdown,
    ) catch |send_error| switch (send_error) {
        error.TranscriptionPeerClosed => {
            unregisterSocket(supervisor.epoll_fd, &transcription.process);
            return;
        },
        else => return send_error,
    };
}

fn finishValidAudioPrefixOrDiscard(supervisor: *Supervisor) !void {
    assert(supervisor.phase == .active);

    if (supervisor.transcript.next_publication_ordinal > 0 or
        countPublishedSlots(supervisor) > 0)
    {
        supervisor.phase = .{
            .finishing = .audio_failed_with_valid_prefix,
        };
    } else {
        try beginDiscard(supervisor, .audio_failed);
    }
}

fn beginDiscard(supervisor: *Supervisor, reason: DiscardReason) !void {
    if (supervisor.phase == .discarding) return;
    supervisor.phase = .{ .discarding = reason };
    transcription_process.requestCancellation(supervisor.transcript_exchange);

    switch (supervisor.audio) {
        inline else => |*session_audio| if (session_audio.process) |*audio| {
            try requestAudioDiscard(supervisor, audio);
        },
    }
    switch (supervisor.transcription) {
        .absent => {},
        .restart_pending => supervisor.transcription = .absent,
        .running => |*transcription| {
            try signalChild(&transcription.process);
            transcription.operation = .{
                .terminating = monotonicNanoseconds() + std.time.ns_per_s,
            };
        },
    }
}

fn requestAudioDiscard(supervisor: *Supervisor, audio: anytype) !void {
    switch (audio.operation) {
        .canceling, .exiting, .terminating => {},
        .starting, .capturing => {
            if (audio.process.socket) |socket| {
                audio_process.sendControl(socket, .cancel) catch |send_error| switch (send_error) {
                    error.AudioControlPeerClosed => {
                        unregisterSocket(supervisor.epoll_fd, &audio.process);
                        try signalChild(&audio.process);
                        audio.operation = .{
                            .terminating = monotonicNanoseconds() + std.time.ns_per_s,
                        };
                        return;
                    },
                    else => return send_error,
                };
                audio.operation = .{
                    .canceling = monotonicNanoseconds() + std.time.ns_per_s,
                };
            } else {
                try signalChild(&audio.process);
                audio.operation = .{
                    .terminating = monotonicNanoseconds() + std.time.ns_per_s,
                };
            }
        },
    }
}

fn applyExpiredDeadlines(supervisor: *Supervisor) !void {
    const now_monotonic_ns = monotonicNanoseconds();
    if (now_monotonic_ns >= supervisor.session_deadline_monotonic_ns) {
        try beginDiscard(supervisor, .deadline);
    }
    switch (supervisor.audio) {
        inline else => |*session_audio| if (session_audio.process) |*audio| {
            try applyAudioDeadline(supervisor, audio, now_monotonic_ns);
        },
    }
    if (supervisor.transcription == .running) {
        const transcription = &supervisor.transcription.running;
        if (transcription.operation.deadlineMonotonicNs()) |deadline_monotonic_ns| {
            if (now_monotonic_ns >= deadline_monotonic_ns) {
                switch (transcription.operation) {
                    .starting, .busy => {
                        try signalChild(&transcription.process);
                        transcription.operation = .{
                            .terminating = now_monotonic_ns + std.time.ns_per_s,
                        };
                        try beginDiscard(supervisor, .deadline);
                    },
                    .shutdown_sent, .exiting => {
                        assert(supervisor.phase != .active);
                        try signalChild(&transcription.process);
                        transcription.operation = .{
                            .terminating = now_monotonic_ns + std.time.ns_per_s,
                        };
                    },
                    .terminating => return error.TranscriptionWorkerExitDeadlineExceeded,
                    .idle => unreachable,
                }
            }
        }
    }
}

fn applyAudioDeadline(
    supervisor: *Supervisor,
    audio: anytype,
    now_monotonic_ns: u64,
) !void {
    const deadline_monotonic_ns =
        audio.operation.deadlineMonotonicNs() orelse return;
    if (now_monotonic_ns < deadline_monotonic_ns) return;

    switch (audio.operation) {
        .starting, .capturing => {
            try signalChild(&audio.process);
            audio.operation = .{
                .terminating = now_monotonic_ns + std.time.ns_per_s,
            };
            if (supervisor.phase == .active) {
                try finishValidAudioPrefixOrDiscard(supervisor);
            }
        },
        .canceling, .exiting => {
            assert(supervisor.phase != .active);
            try signalChild(&audio.process);
            audio.operation = .{
                .terminating = now_monotonic_ns + std.time.ns_per_s,
            };
        },
        .terminating => return error.AudioWorkerExitDeadlineExceeded,
    }
}

fn armNearestDeadline(
    supervisor: *Supervisor,
    timer_fd: std.posix.fd_t,
) void {
    var nearest_deadline_monotonic_ns = supervisor.session_deadline_monotonic_ns;
    switch (supervisor.audio) {
        inline else => |session_audio| if (session_audio.process) |audio| {
            if (audio.operation.deadlineMonotonicNs()) |deadline_monotonic_ns| {
                nearest_deadline_monotonic_ns = @min(
                    nearest_deadline_monotonic_ns,
                    deadline_monotonic_ns,
                );
            }
        },
    }
    if (supervisor.transcription == .running) {
        const transcription = supervisor.transcription.running;
        if (transcription.operation.deadlineMonotonicNs()) |deadline_monotonic_ns| {
            nearest_deadline_monotonic_ns = @min(
                nearest_deadline_monotonic_ns,
                deadline_monotonic_ns,
            );
        }
    }

    const now_monotonic_ns = monotonicNanoseconds();
    const remaining_ns = nearest_deadline_monotonic_ns -| now_monotonic_ns;
    const specification: linux.itimerspec = .{
        .it_interval = .{ .sec = 0, .nsec = 0 },
        .it_value = .{
            .sec = @intCast(remaining_ns / std.time.ns_per_s),
            .nsec = @intCast(remaining_ns % std.time.ns_per_s),
        },
    };
    // A zero timer disarms timerfd. Use one nanosecond when a deadline is
    // already due so the next epoll wait observes it immediately.
    var armed = specification;
    if (remaining_ns == 0) armed.it_value.nsec = 1;
    checkSyscall(linux.timerfd_settime(
        timer_fd,
        .{},
        &armed,
        null,
    ));
}

fn sessionIsComplete(supervisor: *Supervisor) bool {
    return supervisor.phase != .active and
        !audioProcessExists(supervisor) and
        supervisor.transcription == .absent;
}

fn finishSession(supervisor: *Supervisor) SessionTimingReport {
    assert(!audioProcessExists(supervisor));
    assert(supervisor.transcription == .absent);

    const completed_monotonic_ns = monotonicNanoseconds();
    const timing_report: SessionTimingReport = .{
        .command_elapsed_ns = completed_monotonic_ns -
            supervisor.timing.command_started_monotonic_ns,
        .audio = if (supervisor.timing.audio_started_monotonic_ns) |started| audio: {
            const finished = supervisor.timing.audio_finished_monotonic_ns.?;
            break :audio .{
                .start_to_finish_elapsed_ns = finished - started,
                .finish_to_complete_elapsed_ns = completed_monotonic_ns - finished,
                .callbacks_count = supervisor.timing.audio_callbacks_count,
                .callback_duration_ns_max = supervisor.timing.audio_callback_duration_ns_max,
                .callback_gap_ns_max = supervisor.timing.audio_callback_gap_ns_max,
            };
        } else null,
        .model = supervisor.timing.model,
    };

    switch (supervisor.phase) {
        .active => unreachable,
        .finishing => |reason| {
            assert(countPublishedSlots(supervisor) == 0);
            const transcript = std.mem.trim(
                u8,
                supervisor.transcript.bytes[0..supervisor.transcript.bytes_count],
                " \t\r\n",
            );
            const outcome_name = if (supervisor.transcript.accepted_chunks_count == 0 and
                supervisor.transcript.no_speech_chunks_count > 0)
                "no_speech"
            else
                @tagName(reason);
            std.debug.print(
                "Supervisor session complete\n" ++
                    "Outcome: {s}\n" ++
                    "Chunks accepted/no-speech/total: {d}/{d}/{d}\n" ++
                    "Transcript: {s}\n",
                .{
                    outcome_name,
                    supervisor.transcript.accepted_chunks_count,
                    supervisor.transcript.no_speech_chunks_count,
                    supervisor.transcript.next_publication_ordinal,
                    transcript,
                },
            );
        },
        .discarding => |reason| {
            // This command owns one session and unmaps both exchanges on return.
            // A future long-running loop must initialize the next session at its
            // start, after these worker-reaping assertions still hold.
            std.debug.print(
                "Supervisor session discarded\n" ++
                    "Reason: {s}\n" ++
                    "Transcript: not committed\n",
                .{@tagName(std.meta.activeTag(reason))},
            );
            switch (reason) {
                .speech_detection_conflict, .speech_unrecognized => |rejection| {
                    std.debug.print(
                        "Transcription evidence: model={s}, chunk={d}, samples={d}, " ++
                            "activity={s}, no-speech={d:.6}, " ++
                            "thresholds inactive/active={d:.2}/{d:.2}, " ++
                            "average-log-probability={d:.6}\n",
                        .{
                            rejection.model.name(),
                            rejection.publication_ordinal,
                            rejection.samples_count,
                            if (rejection.contains_activity) "active" else "not observed",
                            rejection.no_speech_probability,
                            transcription_process.no_activity_no_speech_probability_reject_min,
                            transcription_process.active_no_speech_probability_conflict_min,
                            rejection.average_log_probability,
                        },
                    );
                },
                .transcript_capacity_exceeded => |capacity| {
                    std.debug.print(
                        "Transcript capacity exceeded: chunk={d}, capacity={d}, " ++
                            "committed={d}, result={d}\n",
                        .{
                            capacity.publication_ordinal,
                            capacity.bytes_capacity,
                            capacity.bytes_committed,
                            capacity.result_bytes_count,
                        },
                    );
                },
                .service_signal,
                .audio_failed,
                .transcription_failed,
                .deadline,
                => {},
            }
        },
    }

    std.debug.print(
        "Pipeline timing\nCommand: {d} ms\n",
        .{timing_report.command_elapsed_ns / std.time.ns_per_ms},
    );
    if (timing_report.audio) |audio| {
        std.debug.print(
            "Audio start to finish: {d} ms\n" ++
                "Audio finish to session completion: {d} ms\n" ++
                "Audio callbacks: {d}\n" ++
                "Audio callback duration max: {d} us\n" ++
                "Audio callback gap max: {d} ms\n",
            .{
                audio.start_to_finish_elapsed_ns / std.time.ns_per_ms,
                audio.finish_to_complete_elapsed_ns / std.time.ns_per_ms,
                audio.callbacks_count,
                audio.callback_duration_ns_max / std.time.ns_per_us,
                audio.callback_gap_ns_max / std.time.ns_per_ms,
            },
        );
    }
    if (timing_report.model) |model| {
        assert(model.workers_ready_count <= model.workers_started_count);
        assert(model.feature_extraction_elapsed_ns_max <=
            model.feature_extraction_elapsed_ns_total);
        assert(model.inference_elapsed_ns_max <= model.inference_elapsed_ns_total);
        std.debug.print(
            "Model residency: {s}\n" ++
                "Model workers started/ready: {d}/{d}\n" ++
                "Model startup total/max: {d}/{d} ms\n" ++
                "Model load total/max: {d}/{d} ms\n" ++
                "Model warm-up total/max: {d}/{d} ms\n" ++
                "Feature extraction total/max: {d}/{d} ms\n" ++
                "Inference total/max: {d}/{d} ms\n" ++
                "Timed transcriptions: {d}\n",
            .{
                @tagName(model.residency),
                model.workers_started_count,
                model.workers_ready_count,
                model.worker_start_to_ready_elapsed_ns_total / std.time.ns_per_ms,
                model.worker_start_to_ready_elapsed_ns_max / std.time.ns_per_ms,
                model.model_load_elapsed_ns_total / std.time.ns_per_ms,
                model.model_load_elapsed_ns_max / std.time.ns_per_ms,
                model.warmup_elapsed_ns_total / std.time.ns_per_ms,
                model.warmup_elapsed_ns_max / std.time.ns_per_ms,
                model.feature_extraction_elapsed_ns_total / std.time.ns_per_ms,
                model.feature_extraction_elapsed_ns_max / std.time.ns_per_ms,
                model.inference_elapsed_ns_total / std.time.ns_per_ms,
                model.inference_elapsed_ns_max / std.time.ns_per_ms,
                model.transcriptions_count,
            },
        );
    }

    return timing_report;
}

fn printPipeWireSetupFailure(failure: *const audio_process.SetupFailure) void {
    std.debug.print(
        "PipeWire setup failed\n" ++
            "Stage: {s}\n" ++
            "Error: {s}/{d}\n" ++
            "PipeWire library: {s}\n" ++
            "Detail: {s}\n",
        .{
            @tagName(failure.stage),
            @tagName(failure.domain),
            failure.code,
            failure.pipewire_version[0..failure.pipewire_version_size],
            failure.message[0..failure.message_size],
        },
    );
}

fn printPipeWireCapture(report: *const audio_process.CaptureReport) void {
    const outcome_name = switch (report.end) {
        .completed => "completed",
        .automatic_stop => "automatic_stop",
        .stopped => "stopped",
        .cancelled => "cancelled",
        .failed => |failure| @tagName(failure.outcome),
    };
    const source_description = if (report.source) |source|
        source.node_description[0..source.node_description_size]
    else
        "not resolved";
    const scheduler_policy = if (report.callback) |callback|
        callback.scheduler_policy orelse -1
    else
        -1;
    const scheduler_priority = if (report.callback) |callback|
        callback.scheduler_priority orelse -1
    else
        -1;
    std.debug.print(
        "PipeWire capture\n" ++
            "Outcome: {s}\n" ++
            "Samples published/captured: {d}/{d}\n" ++
            "Source: {s}\n" ++
            "PipeWire headers/library/server: {s}/{s}/{s}\n" ++
            "Timeline validation: {s}\n" ++
            "Scheduler: {s}, priority {d}\n",
        .{
            outcome_name,
            report.published_samples_count,
            report.samples_count,
            source_description,
            report.pipewire_headers_version[0..report.pipewire_headers_version_size],
            report.pipewire_library_version[0..report.pipewire_library_version_size],
            report.pipewire_server_version[0..report.pipewire_server_version_size],
            @tagName(report.timeline_validation),
            audio_process.schedulerPolicyName(scheduler_policy),
            scheduler_priority,
        },
    );
    if (report.end == .failed) {
        const failure = report.end.failed.failure;
        std.debug.print(
            "Capture error: stage={s}, error={s}/{d}\n" ++
                "Detail: {s}\n",
            .{
                @tagName(failure.stage),
                @tagName(failure.domain),
                failure.code,
                failure.messageBytes(),
            },
        );
    }
}

fn audioProcessExists(supervisor: *const Supervisor) bool {
    return switch (supervisor.audio) {
        inline else => |audio| audio.process != null,
    };
}

fn countPublishedSlots(supervisor: *Supervisor) u32 {
    var count: u32 = 0;
    for (&supervisor.audio_exchange.slots) |*slot| {
        if (audio_exchange.acquirePublishedSlot(slot) != null) count += 1;
    }
    return count;
}

fn receiveRecordNonblocking(socket: std.posix.fd_t, comptime T: type) ??T {
    var record: T = undefined;
    const result = linux.recvfrom(
        socket,
        std.mem.asBytes(&record).ptr,
        @sizeOf(T),
        linux.MSG.TRUNC | linux.MSG.DONTWAIT,
        null,
        null,
    );
    switch (linux.errno(result)) {
        .SUCCESS => {
            if (result == 0) return @as(?T, null);
            assert(result == @sizeOf(T));
            return record;
        },
        .AGAIN => return null,
        // SIGKILL can reset a seqpacket endpoint before pidfd readiness from
        // the same epoll snapshot is processed. Treat reset like orderly EOF;
        // reap and session policy remain owned by the pidfd path.
        .CONNRESET => return @as(?T, null),
        else => @trap(),
    }
}

fn sendRecord(socket: std.posix.fd_t, bytes: []const u8) !void {
    const result = linux.sendto(
        socket,
        bytes.ptr,
        bytes.len,
        linux.MSG.NOSIGNAL,
        null,
        0,
    );
    if (linux.errno(result) != .SUCCESS) return error.SupervisorPacketSendFailed;
    assert(result == bytes.len);
}

fn createSharedMemory(name: [:0]const u8, size: usize) !std.posix.fd_t {
    const descriptor = try std.posix.memfd_create(
        name,
        linux.MFD.CLOEXEC | linux.MFD.ALLOW_SEALING,
    );
    checkSyscall(linux.ftruncate(descriptor, @intCast(size)));
    checkSyscall(linux.fcntl(
        descriptor,
        linux.F.ADD_SEALS,
        linux.F.SEAL_GROW | linux.F.SEAL_SHRINK | linux.F.SEAL_SEAL,
    ));
    return descriptor;
}

fn mapSharedMemory(
    descriptor: std.posix.fd_t,
    comptime T: type,
) !SharedMapping {
    const bytes = try std.posix.mmap(
        null,
        @sizeOf(T),
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        descriptor,
        0,
    );
    return .{ .bytes = bytes };
}

fn register(
    epoll_fd: std.posix.fd_t,
    descriptor: std.posix.fd_t,
    source: EventSource,
) !void {
    var event: linux.epoll_event = .{
        .events = linux.EPOLL.IN | linux.EPOLL.ERR | linux.EPOLL.HUP |
            linux.EPOLL.RDHUP,
        .data = .{ .u64 = @intFromEnum(source) },
    };
    const result = linux.epoll_ctl(
        epoll_fd,
        linux.EPOLL.CTL_ADD,
        descriptor,
        &event,
    );
    if (linux.errno(result) != .SUCCESS) return error.SupervisorEpollRegisterFailed;
}

fn unregister(epoll_fd: std.posix.fd_t, descriptor: std.posix.fd_t) void {
    const result = linux.epoll_ctl(
        epoll_fd,
        linux.EPOLL.CTL_DEL,
        descriptor,
        null,
    );
    assert(linux.errno(result) == .SUCCESS);
}

fn unregisterSocket(epoll_fd: std.posix.fd_t, process: *ChildProcess) void {
    const socket = process.socket orelse return;
    unregister(epoll_fd, socket);
    closeDescriptor(socket);
    process.socket = null;
}

fn unregisterChild(
    epoll_fd: std.posix.fd_t,
    process: *ChildProcess,
) void {
    unregisterSocket(epoll_fd, process);
    unregister(epoll_fd, process.pid_fd);
}

fn forceStopAndReap(process: *ChildProcess) !void {
    try signalChild(process);
    _ = try reapChild(process);
    if (process.socket) |socket| closeDescriptor(socket);
    process.socket = null;
    closeDescriptor(process.pid_fd);
}

fn waitForEvents(
    epoll_fd: std.posix.fd_t,
    events: []linux.epoll_event,
    timeout_ms: i32,
) usize {
    while (true) {
        const result = linux.epoll_wait(
            epoll_fd,
            events.ptr,
            @intCast(events.len),
            timeout_ms,
        );
        switch (linux.errno(result)) {
            .SUCCESS => return result,
            .INTR => continue,
            else => @trap(),
        }
    }
}

fn eventSource(event: linux.epoll_event) EventSource {
    return @enumFromInt(event.data.u64);
}

fn readEventCounter(descriptor: std.posix.fd_t) ?u64 {
    var counter: u64 = 0;
    const result = linux.read(
        descriptor,
        std.mem.asBytes(&counter).ptr,
        @sizeOf(u64),
    );
    switch (linux.errno(result)) {
        .SUCCESS => {
            assert(result == @sizeOf(u64));
            assert(counter > 0);
            return counter;
        },
        // Rearming timerfd clears an unread old expiration. An event already
        // copied into the current epoll batch may therefore find no counter;
        // current absolute deadlines still decide whether anything expires.
        .AGAIN => return null,
        else => @trap(),
    }
}

fn readSignal(descriptor: std.posix.fd_t) linux.signalfd_siginfo {
    var signal: linux.signalfd_siginfo = undefined;
    const result = linux.read(
        descriptor,
        std.mem.asBytes(&signal).ptr,
        @sizeOf(linux.signalfd_siginfo),
    );
    assert(linux.errno(result) == .SUCCESS);
    assert(result == @sizeOf(linux.signalfd_siginfo));
    return signal;
}

fn setCloseOnExec(descriptor: std.posix.fd_t) void {
    checkSyscall(linux.fcntl(descriptor, linux.F.SETFD, linux.FD_CLOEXEC));
}

fn signalChild(process: *const ChildProcess) !void {
    while (true) {
        switch (linux.errno(linux.pidfd_send_signal(
            process.pid_fd,
            .KILL,
            null,
            0,
        ))) {
            .SUCCESS, .SRCH => return,
            .INTR => continue,
            else => return error.SupervisorWorkerKillFailed,
        }
    }
}

fn reapChild(process: *const ChildProcess) !linux.siginfo_t {
    var information: linux.siginfo_t = std.mem.zeroes(linux.siginfo_t);
    while (true) {
        const result = linux.waitid(
            .PIDFD,
            process.pid_fd,
            &information,
            linux.W.EXITED,
            null,
        );
        switch (linux.errno(result)) {
            .SUCCESS => return information,
            .INTR => continue,
            else => return error.SupervisorWorkerReapFailed,
        }
    }
}

fn sessionDeadline(configuration: SessionConfiguration) u64 {
    const duration_ns: u64 = switch (configuration) {
        .fake => 5 * std.time.ns_per_s,
        .pipewire => |options| (@as(u64, options.recording_seconds) +
            @as(u64, switch (options.transcription) {
                .fake => 10,
                // The model starts concurrently with capture. After audio
                // stops, at most three sealed slots can remain, each with its
                // own ten-second inference deadline plus bounded shutdown.
                .model => 35,
            })) * std.time.ns_per_s,
    };
    return monotonicNanoseconds() + duration_ns;
}

fn monotonicNanoseconds() u64 {
    var timestamp: linux.timespec = undefined;
    checkSyscall(linux.clock_gettime(.MONOTONIC, &timestamp));
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);
    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

fn descriptorFromResult(result: usize) std.posix.fd_t {
    checkSyscall(result);
    return @intCast(result);
}

fn checkSyscall(result: usize) void {
    assert(linux.errno(result) == .SUCCESS);
}

fn closeDescriptor(descriptor: std.posix.fd_t) void {
    assert(descriptor >= 0);
    checkSyscall(linux.close(descriptor));
}
