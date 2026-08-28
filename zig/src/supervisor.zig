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
const transcription_process = @import("transcription_process.zig");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;
const TranscriptExchange = transcription_process.TranscriptExchange;

const session_transcript_bytes_capacity: u32 = 4096;
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

pub const PipeWireOptions = struct {
    source: audio_process.Source = .default,
    recording_seconds: u8 = 5,
    slot_seconds: u8 = 1,
    process_realtime: bool = true,
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
    audio_stopped,
    pipeline_full,
    audio_failed_with_valid_prefix,
};

const DiscardReason = enum {
    service_signal,
    audio_failed,
    transcription_failed,
    deadline,
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
    bytes: [session_transcript_bytes_capacity]u8,
    bytes_count: u32,
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
};

/// `runFakeSession` drives deterministic workers through the same process,
/// descriptor, exchange, deadline, and recovery boundaries used by real roles.
pub fn runFakeSession(init: std.process.Init, scenario: FakeScenario) !void {
    try runSession(init, .{ .fake = scenario });
}

/// `runPipeWireSession` replaces only deterministic audio with the proven real
/// PipeWire role. Fake transcription remains intentional here: this slice tests
/// live capture under the new supervisor before CTranslate2 is attached.
pub fn runPipeWireSession(init: std.process.Init, options: PipeWireOptions) !void {
    switch (options.source) {
        .default => {},
        .node_name, .device_serial => |source| {
            assert(source.len > 0);
            assert(source.len < audio_process.target_name_bytes_capacity);
        },
    }
    assert(options.recording_seconds > 0);
    assert(options.recording_seconds <= 90);
    assert(options.slot_seconds > 0);
    assert(options.slot_seconds <= audio_process.slot_duration_seconds_max);
    try runSession(init, .{ .pipewire = options });
}

/// `runSession` owns one bounded recording from worker launch through complete
/// process reaping. It prints text only after the audio prefix has been drained
/// in publication order and the transcription process has stopped.
fn runSession(init: std.process.Init, configuration: SessionConfiguration) !void {
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
            .bytes = @splat(0),
            .bytes_count = 0,
        },
    };

    try register(epoll_fd, publication_event_fd, .audio_publication);
    try register(epoll_fd, timer_fd, .deadline);
    try register(epoll_fd, signal_fd, .service_signal);
    try startAudio(&supervisor, publication_event_fd);
    supervisor.transcription = .{
        .running = try startTranscription(&supervisor, null),
    };
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

    finishSession(&supervisor);
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
}

fn startTranscription(
    supervisor: *Supervisor,
    retry_work: ?TranscriptionWork,
) !TranscriptionProcess {
    var process = try startChild(
        supervisor,
        .transcription_packet,
        .transcription_exit,
        "transcription-fake",
    );
    errdefer forceStopAndReap(&process) catch {};

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
    try transcription_process.sendLaunch(
        process.socket.?,
        supervisor.audio_exchange_fd,
        supervisor.transcript_exchange_fd,
        .{
            .session_id = session_id,
            .fake_behavior = fake_behavior,
            .fake_inference_duration_ms = behavior.duration_ms,
        },
    );

    return .{
        .process = process,
        .operation = .{ .starting = .{
            .deadline_monotonic_ns = monotonicNanoseconds() + std.time.ns_per_s,
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
                .canceling => audio.operation = .{
                    .exiting = monotonicNanoseconds() + std.time.ns_per_s,
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

        switch (worker_report) {
            .setup_failed => |setup_failure| {
                printPipeWireSetupFailure(&setup_failure);
                try beginDiscard(supervisor, .audio_failed);
            },
            .captured => |capture| {
                printPipeWireCapture(&capture);
                switch (capture.end) {
                    .completed => if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = .audio_completed };
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

    audio.operation = .{ .capturing = .{
        .callbacks_count = callbacks_count,
        .deadline_monotonic_ns = monotonicNanoseconds() +
            2 * std.time.ns_per_s,
    } };
}

fn drainTranscriptionPackets(supervisor: *Supervisor) !void {
    const transcription = &supervisor.transcription.running;
    while (receiveRecordNonblocking(
        transcription.process.socket.?,
        transcription_process.WireReport,
    )) |received| {
        const report = received orelse {
            unregisterSocket(supervisor.epoll_fd, &transcription.process);
            return;
        };
        const report_kind = transcription_process.decodeTrustedReport(report);
        switch (report_kind) {
            .ready => switch (transcription.operation) {
                .starting => |starting| transcription.operation = .{
                    .idle = starting.retry_work,
                },
                .terminating => {},
                .idle, .busy, .shutdown_sent, .exiting => unreachable,
            },
            .result => {
                if (supervisor.phase == .discarding or
                    transcription.operation == .terminating)
                {
                    // Cancellation owns the mailbox terminal state. A result
                    // packet already queued before termination cannot revive it.
                    continue;
                }
                assert(transcription.operation == .busy);
                try acceptTranscript(supervisor);
                transcription.operation = .{ .idle = null };
            },
            .stopped => {
                assert(transcription.operation == .shutdown_sent or
                    transcription.operation == .terminating);
                transcription.operation = .{
                    .exiting = monotonicNanoseconds() + std.time.ns_per_s,
                };
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
    transcription.operation = .{ .busy = .{
        .deadline_monotonic_ns = monotonicNanoseconds() + 80 * std.time.ns_per_ms,
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

fn acceptTranscript(supervisor: *Supervisor) !void {
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
    assert(supervisor.transcript.bytes_count + result.bytes.len <=
        supervisor.transcript.bytes.len);
    @memcpy(
        supervisor.transcript.bytes[supervisor.transcript.bytes_count..][0..result.bytes.len],
        result.bytes,
    );
    supervisor.transcript.bytes_count += @intCast(result.bytes.len);
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
        try acceptTranscript(supervisor);
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

fn finishSession(supervisor: *Supervisor) void {
    assert(!audioProcessExists(supervisor));
    assert(supervisor.transcription == .absent);

    switch (supervisor.phase) {
        .active => unreachable,
        .finishing => |reason| {
            assert(countPublishedSlots(supervisor) == 0);
            std.debug.print(
                "Supervisor session complete\n" ++
                    "Outcome: {s}\n" ++
                    "Chunks: {d}\n" ++
                    "Transcript: {s}\n",
                .{
                    @tagName(reason),
                    supervisor.transcript.next_publication_ordinal,
                    supervisor.transcript.bytes[0..supervisor.transcript.bytes_count],
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
                .{@tagName(reason)},
            );
        },
    }
}

fn printPipeWireSetupFailure(failure: *const audio_process.SetupFailure) void {
    std.debug.print(
        "PipeWire setup failed\n" ++
            "Stage: {s}\n" ++
            "Error: {s}/{d}\n" ++
            "Detail: {s}\n",
        .{
            @tagName(failure.stage),
            @tagName(failure.domain),
            failure.code,
            failure.message[0..failure.message_size],
        },
    );
}

fn printPipeWireCapture(report: *const audio_process.CaptureReport) void {
    const outcome_name = switch (report.end) {
        .completed => "completed",
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
            "Scheduler: {s}, priority {d}\n",
        .{
            outcome_name,
            report.published_samples_count,
            report.samples_count,
            source_description,
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
        .pipewire => |options| (@as(u64, options.recording_seconds) + 10) *
            std.time.ns_per_s,
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
