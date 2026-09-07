//! Owns bounded recordings and the service's idle model lifetime. The event
//! loop is the only lifecycle owner: workers
//! publish bytes and reports, but only this file chooses finish, discard, retry,
//! slot release, and final transcript acceptance.
//!
//! The implementation deliberately has no worker or deadline incarnation. A
//! replacement starts only after the complete returned epoll batch is consumed,
//! so stale events from the old descriptors have no replacement to affect.
//! Timer events likewise ask the supervisor to evaluate current absolute
//! deadlines; they do not identify an earlier deadline instance.

const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.supervisor);
const notifications = @import("notifications.zig");
const transcript_file = @import("transcript_file.zig");
const audio_exchange = @import("audio_exchange.zig");
const audio_process = @import("audio_process.zig");
const models = @import("models");
const control_socket = @import("control_socket.zig");
const clipboard_process = @import("clipboard_process.zig");
const paste_keyboard = @import("paste_keyboard.zig");
const transcription_process = @import("transcription_process.zig");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;
const TranscriptExchange = transcription_process.TranscriptExchange;

const epoll_events_count_max: u32 = 16;

pub const ModelOptions = struct {
    model: models.Model = models.default,
    inference_threads_count: u32 = 4,
    decoder_threads_count: ?u32 = null,
    encoder_trailing_padding: @FieldType(transcription_process.ModelLaunchOptions, "encoder_trailing_padding") = .seconds_10,
};

pub const recording_duration_seconds_default: u16 = 60 * 60;
pub const recording_duration_seconds_limit: u16 = std.math.maxInt(u16);

comptime {
    // The configured duration remains a u16, while the audio launch protocol
    // carries its corresponding 16 kHz sample target in a u32.
    assert(@as(u64, recording_duration_seconds_limit) *
        audio_process.sample_rate_hz <= std.math.maxInt(u32));
}

pub const CaptureOptions = struct {
    source: audio_process.Source = .default,
    recording_seconds: u16 = recording_duration_seconds_default,
    automatic_stop: audio_process.AutomaticStop = .disabled,
    transcription: ModelOptions = .{},
};

pub const ServiceOptions = struct {
    log_level: logging.Level = .info,
    capture: CaptureOptions = .{},
    /// Zero releases the complete model process after every recording; otherwise
    /// the runtime, worker group, weights, and workspace survive this idle window.
    model_keep_warm_seconds: u32 = 300,
    output: enum { desktop, clipboard, stdout } = .desktop,
    notification_mode: notifications.Mode = .errors,
    paste_key: paste_keyboard.Chord = .@"ctrl+shift+v",
    paste_settle_ms: u16 = 10,
    paste_key_gap_ms: u16 = 4,
};

const EventSource = enum(u64) {
    audio_publication = 1,
    audio_packet = 2,
    audio_exit = 3,
    transcription_packet = 4,
    transcription_exit = 5,
    deadline = 6,
    service_signal = 7,
    control_listener = 8,
    control_client = 9,
    clipboard = 10,
    notification_bus = 11,
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

const AbortReason = union(enum) {
    service_signal,
    user_cancelled,
    audio_failed,
    transcription_failed,
    exchange_corrupt: enum { audio_slot, timeline, mailbox },
    deadline: enum { recording, model_load, transcription },
    transcript_limit: enum { recording_size, chunk_size, decoder_tokens, chunk_size_and_decoder_tokens },
    speech_detection_conflict: TranscriptionRejection,
    speech_unrecognized: TranscriptionRejection,
};

const SessionPhase = union(enum) {
    idle: ?u64,
    active,
    finishing: FinishReason,
    delivering: struct {
        slot: u1,
        // One boundary reused after acquisition: delivery start, then clipboard
        // acquisition. The paste interval intentionally includes settle waits.
        boundary_monotonic_ns: u64,
        problem: ?notifications.Problem = null,
        paste: union(enum) { waiting, settling: u64, sending, done } = .waiting,
    },
    aborting: AbortReason,
};

const ChildProcess = struct {
    socket: ?std.posix.fd_t,
    pid_fd: std.posix.fd_t,
};

// A role operation and its deadline are one fact; changing the operation also
// replaces the deadline that owns its progress or termination.
const AudioProgress = struct {
    callbacks_count: u64,
    deadline_monotonic_ns: u64,
};

const PipeWireAudioOperation = union(enum) {
    starting: u64,
    capturing: AudioProgress,
    stopping: u64,
    canceling: u64,
    exiting: u64,
    terminating: u64,

    fn deadlineMonotonicNs(operation: PipeWireAudioOperation) u64 {
        return switch (operation) {
            .capturing => |progress| progress.deadline_monotonic_ns,
            inline else => |deadline_monotonic_ns| deadline_monotonic_ns,
        };
    }
};

const PipeWireAudioProcess = struct {
    process: ChildProcess,
    operation: PipeWireAudioOperation,
};

const SessionAudio = struct {
    options: CaptureOptions,
    process: ?PipeWireAudioProcess,
    // Full worker diagnostics are journaled before retaining presentation state.
    problem: ?notifications.Problem = null,
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
    accepted_chunks_count: u32,
    no_speech_chunks_count: u32,
    processed_samples_count: u64 = 0,
    // A recovered mailbox has text but may have no timing notification. A
    // partial timing sum cannot describe the whole recording's realtime speed.
    compute_duration_ns: ?u64 = 0,
    bytes: []u8,
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
    recording_ordinal: u64 = 0,
    recording_requested_monotonic_ns: u64 = 0,
    recording_stop: ?struct {
        monotonic_ns: u64,
        origin: enum { command, capture_end },
    } = null,
    // At most one selection owner plus its candidate/retiring predecessor.
    // An owner retains its recording ordinal after a later recording starts.
    // Slots are reused only after the complete epoll batch has been consumed.
    clipboard: [2]?struct {
        recording_ordinal: u64,
        process: clipboard_process.Process,
    } = .{ null, null },
    keyboard: ?paste_keyboard.Keyboard = null,
    notifications: notifications.Client = .{},
    transcription_problem: ?notifications.Problem = null,
    paste_problem: ?notifications.Problem = null,
    service: struct {
        control: *control_socket.Server,
        transcript_directory: ?[]const u8,
        model_keep_warm_seconds: u32,
        output: @FieldType(ServiceOptions, "output"),
        paste_key: paste_keyboard.Chord,
        paste_settle_ms: u16,
        paste_key_gap_ms: u16,
        shutdown_requested: bool = false,
        pending_recording: ?struct { automatic_stop: audio_process.AutomaticStop, requested_monotonic_ns: u64 } = null,
    },
};

/// `runService` accepts newline-JSON commands on the selected instance's control
/// socket. Capture and model loading start together on the first recording.
/// Successful recordings retain the entire model process for the idle window;
/// discarded recordings terminate it before another recording can reset shared
/// storage. Final delivery borrows the transcript until wl-copy consumes it;
/// clipboard ownership can then outlive this recording and its model process.
pub fn runService(init: std.process.Init, options: ServiceOptions) !void {
    const configuration = options.capture;

    // ── Own Signals And Event Sources ──

    try checkSyscall("prctl", linux.prctl(@intFromEnum(linux.PR.SET_CHILD_SUBREAPER), 1, 0, 0, 0));
    var previous_pipe_action: linux.Sigaction = undefined;
    try checkSyscall("sigaction", linux.sigaction(.PIPE, &.{ .handler = .{ .handler = linux.SIG.IGN }, .mask = linux.sigemptyset(), .flags = 0 }, &previous_pipe_action));
    defer logCleanupSyscall("sigaction", linux.sigaction(.PIPE, &previous_pipe_action, null));
    var previous_child_action: linux.Sigaction = undefined;
    try checkSyscall("sigaction", linux.sigaction(.CHLD, &.{ .handler = .{ .handler = linux.SIG.DFL }, .mask = linux.sigemptyset(), .flags = 0 }, &previous_child_action));
    defer logCleanupSyscall("sigaction", linux.sigaction(.CHLD, &previous_child_action, null));

    var signal_mask = std.posix.sigemptyset();
    std.posix.sigaddset(&signal_mask, .TERM);
    std.posix.sigaddset(&signal_mask, .INT);
    var previous_signal_mask = std.posix.sigemptyset();
    std.posix.sigprocmask(std.posix.SIG.BLOCK, &signal_mask, &previous_signal_mask);
    defer std.posix.sigprocmask(std.posix.SIG.SETMASK, &previous_signal_mask, null);

    const signal_fd = try std.posix.signalfd(
        -1,
        &signal_mask,
        linux.SFD.CLOEXEC | linux.SFD.NONBLOCK,
    );
    defer closeDescriptor(signal_fd);
    const epoll_fd = try descriptorFromResult("epoll_create1", linux.epoll_create1(linux.EPOLL.CLOEXEC));
    defer closeDescriptor(epoll_fd);
    const timer_fd = try descriptorFromResult("timerfd_create", linux.timerfd_create(.MONOTONIC, .{
        .CLOEXEC = true,
        .NONBLOCK = true,
    }));
    defer closeDescriptor(timer_fd);
    const publication_event_fd = try descriptorFromResult("eventfd", linux.eventfd(
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
    audio_exchange.initialize(audio_mapping.pointer(AudioExchange), 1);

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
        1,
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

    var control = try control_socket.Server.open(
        init,
        epoll_fd,
        @intFromEnum(EventSource.control_listener),
        @intFromEnum(EventSource.control_client),
    );
    defer control.deinit(init.io);

    const transcript_directory = if (options.output == .stdout) null else transcript_file.allocDirectoryPath(
        init.gpa,
        init.environ_map.get("XDG_STATE_HOME"),
        init.environ_map.get("HOME"),
        init.environ_map.get("VOICED_INSTANCE") orelse "",
    ) catch |err| path: {
        log.err(.{}, "Transcript saving unavailable: error={s}", .{@errorName(err)});
        break :path null;
    };
    defer if (transcript_directory) |path| init.gpa.free(path);

    var supervisor: Supervisor = .{
        .io = init.io,
        .audio = .{ .options = configuration, .process = null },
        .epoll_fd = epoll_fd,
        .audio_exchange_fd = audio_exchange_fd,
        .audio_exchange = audio_mapping.pointer(AudioExchange),
        .transcript_exchange_fd = transcript_exchange_fd,
        .transcript_exchange = transcript_mapping.pointer(TranscriptExchange),
        .transcription = .absent,
        .phase = .{ .idle = null },
        .session_deadline_monotonic_ns = sessionDeadline(configuration),
        .service = .{
            .control = &control,
            .transcript_directory = transcript_directory,
            .model_keep_warm_seconds = options.model_keep_warm_seconds,
            .output = options.output,
            .paste_key = options.paste_key,
            .paste_settle_ms = options.paste_settle_ms,
            .paste_key_gap_ms = options.paste_key_gap_ms,
        },
        .transcript = .{
            .next_publication_ordinal = 0,
            .accepted_chunks_count = 0,
            .no_speech_chunks_count = 0,
            .bytes = transcript_bytes,
            .bytes_count = 0,
        },
    };

    try register(epoll_fd, publication_event_fd, .audio_publication);
    try register(epoll_fd, timer_fd, .deadline);
    try register(epoll_fd, signal_fd, .service_signal);

    if (options.notification_mode == .errors and options.output != .stdout)
        supervisor.notifications.init(epoll_fd, @intFromEnum(EventSource.notification_bus), init.environ_map.get("DBUS_SESSION_BUS_ADDRESS"), init.environ_map.get("XDG_RUNTIME_DIR"));
    defer {
        supervisor.notifications.recover();
        supervisor.notifications.advance(epoll_fd, @intFromEnum(EventSource.notification_bus));
        supervisor.notifications.deinit(epoll_fd);
    }

    const model = configuration.transcription;
    const microphone: struct { key: []const u8, value: []const u8 } = switch (configuration.source) {
        .default => .{ .key = "microphone", .value = "default" },
        .node_name => |name| .{ .key = "microphone_node", .value = name },
        .device_serial => |serial| .{ .key = "microphone_serial", .value = serial },
    };
    log.info(.{}, "Effective config: log_level={s}, model={s}, model_encoder_threads={d}, model_decoder_threads={d}, model_encoder_padding_seconds={d}, model_idle_seconds_max={d}, recording_seconds_max={d}, transcript_output={s}, paste_shortcut={s}, paste_settle_ms={d}, paste_key_gap_ms={d}, notification_mode={s}, {s}=\"{f}\"", .{
        logging.levelName(options.log_level),
        model.model.name(),
        model.inference_threads_count,
        model.decoder_threads_count orelse model.inference_threads_count,
        @as(u8, switch (model.encoder_trailing_padding) {
            .seconds_5 => 5,
            .seconds_10 => 10,
            .seconds_30 => 30,
        }),
        options.model_keep_warm_seconds,
        configuration.recording_seconds,
        @tagName(options.output),
        @tagName(options.paste_key),
        options.paste_settle_ms,
        options.paste_key_gap_ms,
        @tagName(if (options.output == .stdout) notifications.Mode.off else options.notification_mode),
        microphone.key,
        std.zig.fmtString(microphone.value),
    });
    log.info(.{}, "Supervisor service ready\n", .{});
    // Error cleanup owns descriptor closure as well as bounded reaping. A
    // worker stuck in uninterruptible kernel work must not prevent service exit;
    // its own memfd mappings remain valid after the supervisor unmaps its views.
    defer {
        if (supervisor.keyboard) |*keyboard| keyboard.deinit();
        if (supervisor.audio.process) |*process| {
            forceStopAndReap(&process.process) catch |err| log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Worker cleanup failed: role=audio, error={s}", .{@errorName(err)});
        }
        if (supervisor.transcription == .running) {
            const process = &supervisor.transcription.running.process;
            forceStopAndReap(process) catch |err| log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Worker cleanup failed: role=transcription, error={s}", .{@errorName(err)});
        }
        for (&supervisor.clipboard) |*slot| if (slot.*) |*owner| owner.process.deinit(epoll_fd);
    }

    if (options.output == .desktop) openPasteKeyboard(&supervisor);

    // Every iteration consumes one complete readiness snapshot. Worker
    // replacement happens only in the final maintenance paragraph, after no old
    // event from this snapshot can be applied to the replacement.
    while (true) {
        if (sessionIsComplete(&supervisor)) {
            try finishSession(&supervisor);
            if (supervisor.phase != .delivering) enterIdle(&supervisor);
        }
        if (supervisor.service.shutdown_requested and supervisor.phase == .idle and
            supervisor.transcription == .absent and supervisor.clipboard[0] == null and supervisor.clipboard[1] == null and
            !supervisor.service.control.hasPendingReplies())
        {
            return;
        }
        supervisor.notifications.advance(epoll_fd, @intFromEnum(EventSource.notification_bus));
        try armNearestDeadline(&supervisor, timer_fd);
        var events: [epoll_events_count_max]linux.epoll_event = undefined;
        // timerfd represents the nearest absolute deadline, so an unbounded
        // epoll wait is still bounded by current session policy. A separate
        // polling timeout would merely create a second, inconsistent deadline.
        const events_count = try waitForEvents(epoll_fd, &events, -1);
        assert(events_count > 0);

        for (events[0..events_count]) |event| {
            if (eventSource(event) != .audio_publication) continue;
            _ = try readEventCounter(publication_event_fd);
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
        try dispatchPublishedAudio(&supervisor);

        try observePipeWireProgress(&supervisor);

        for (events[0..events_count]) |event| {
            switch (eventSource(event)) {
                .audio_exit => if (audioProcessExists(&supervisor))
                    try reapAudio(&supervisor),
                .transcription_exit => if (supervisor.transcription == .running)
                    try reapTranscription(&supervisor),
                .deadline => {
                    _ = try readEventCounter(timer_fd);
                    try applyExpiredDeadlines(&supervisor);
                },
                .service_signal => {
                    if (try readSignal(signal_fd) == null) continue;
                    supervisor.service.shutdown_requested = true;
                    supervisor.service.pending_recording = null;
                    if (supervisor.phase == .idle) {
                        supervisor.phase.idle = null;
                        try requestTranscriptionShutdown(&supervisor);
                    } else {
                        try beginAbort(&supervisor, .service_signal);
                    }
                },
                else => {},
            }
        }

        {
            const service = &supervisor.service;
            service.control.expire(monotonicNanoseconds());
            // Scan existing clients before accepting replacements, after all
            // worker events. Requests can queue a start, never launch mid-batch.
            for (0..control_socket.clients_count_max) |index| {
                if (service.control.receive(index)) |request| {
                    try dispatchControl(&supervisor, index, request);
                }
            }
            if (service.shutdown_requested) {
                try service.control.stopAccepting();
            } else {
                for (events[0..events_count]) |event| {
                    if (eventSource(event) == .control_listener) try service.control.acceptClients(monotonicNanoseconds());
                }
            }
        }

        // Shared counts remain authoritative when eventfd increments coalesce or
        // when a worker publishes text and dies before sending its report.
        try dispatchPublishedAudio(&supervisor);
        try maintainWorkersAndSession(&supervisor);
        try advanceOutput(&supervisor);
        {
            const service = &supervisor.service;
            if (service.pending_recording != null and supervisor.phase == .idle and clipboardSlotsSettled(&supervisor) and
                (supervisor.transcription == .absent or (supervisor.transcription == .running and supervisor.transcription.running.operation == .idle)))
            {
                const pending = service.pending_recording.?;
                supervisor.recording_requested_monotonic_ns = pending.requested_monotonic_ns;
                const automatic_stop = pending.automatic_stop;
                service.pending_recording = null;
                assert(!audioProcessExists(&supervisor));
                // PCM payload is not cleared. Reset logical lengths only when
                // capture is reaped and the retained model awaits a command.
                supervisor.recording_ordinal += 1;
                supervisor.notifications.resetSuppression();
                supervisor.recording_stop = null;
                supervisor.audio.problem = null;
                supervisor.transcription_problem = null;
                audio_exchange.initialize(supervisor.audio_exchange, supervisor.recording_ordinal);
                transcription_process.initializeExchange(supervisor.transcript_exchange, supervisor.recording_ordinal);
                supervisor.transcript = .{ .next_publication_ordinal = 0, .accepted_chunks_count = 0, .no_speech_chunks_count = 0, .bytes = transcript_bytes, .bytes_count = 0 };
                supervisor.audio.options.automatic_stop = automatic_stop;
                supervisor.phase = .active;
                supervisor.session_deadline_monotonic_ns = sessionDeadline(configuration);
                log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Recording requested: recording_ordinal={d}", .{supervisor.recording_ordinal});
                if (supervisor.transcription == .absent) supervisor.transcription = .{ .running = try startTranscription(&supervisor, null) };
                try startAudio(&supervisor, publication_event_fd);
            }
        }
    }
}

fn dispatchControl(supervisor: *Supervisor, client_index: usize, request: control_socket.Request) !void {
    const request_received_monotonic_ns = monotonicNanoseconds();
    const service = &supervisor.service;
    var ignored = false;
    switch (request.cmd) {
        .record, .listen => {
            if (service.shutdown_requested or supervisor.phase == .finishing or supervisor.phase == .delivering or supervisor.phase == .aborting) {
                ignored = true;
            } else if (supervisor.phase == .active) {
                if (request.toggle) try requestAudioFinish(supervisor, request_received_monotonic_ns) else ignored = true;
            } else if (service.pending_recording != null) {
                if (request.toggle) service.pending_recording = null else ignored = true;
            } else {
                assert(supervisor.phase == .idle);
                service.pending_recording = .{ .automatic_stop = if (request.cmd == .listen) .after_quiet else .disabled, .requested_monotonic_ns = request_received_monotonic_ns };
            }
        },
        .stop => {
            service.pending_recording = null;
            if (supervisor.phase == .active) try requestAudioFinish(supervisor, request_received_monotonic_ns);
        },
        .cancel => {
            service.pending_recording = null;
            if (supervisor.phase != .idle) try beginAbort(supervisor, .user_cancelled);
        },
        .kill => {
            service.shutdown_requested = true;
            service.pending_recording = null;
            if (supervisor.phase == .idle) {
                supervisor.phase.idle = null;
                try requestTranscriptionShutdown(supervisor);
            } else try beginAbort(supervisor, .service_signal);
        },
        .status => {},
    }
    if (ignored) log.debug(.{ .recording_ordinal = supervisor.recording_ordinal }, "Command ignored: recording_ordinal={d}, command={s}, phase={s}", .{ supervisor.recording_ordinal, @tagName(request.cmd), @tagName(supervisor.phase) });
    const phase: control_socket.Phase = switch (supervisor.phase) {
        .idle => if (service.pending_recording != null) .capturing else .idle,
        .active => .capturing,
        .finishing => if (audioProcessExists(supervisor)) .stopping else .transcribing,
        .delivering => .delivering,
        .aborting => .stopping,
    };
    const model_state: control_socket.ModelState = switch (supervisor.transcription) {
        .absent => .absent,
        .restart_pending => .restarting,
        .running => |running| switch (running.operation) {
            .starting => .loading,
            .idle, .busy => .warm,
            .shutdown_sent, .exiting, .terminating => .unloading,
        },
    };
    service.control.respond(client_index, .{
        .ignored = ignored,
        .phase = phase,
        .model = model_state,
        .session_id = supervisor.audio_exchange.session_id,
        .model_keep_warm_seconds = service.model_keep_warm_seconds,
    });
}

fn requestAudioFinish(supervisor: *Supervisor, request_received_monotonic_ns: u64) !void {
    assert(supervisor.phase == .active);
    assert(supervisor.recording_stop == null);
    supervisor.recording_stop = .{ .monotonic_ns = request_received_monotonic_ns, .origin = .command };
    log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Recording stop requested: recording_ordinal={d}, recording_stop_origin=command", .{supervisor.recording_ordinal});
    supervisor.phase = .{ .finishing = .audio_stopped };
    const audio = if (supervisor.audio.process) |*process| process else {
        return;
    };
    switch (audio.operation) {
        .starting, .capturing => {
            if (audio.process.socket) |socket| {
                audio_process.sendControl(socket, .stop) catch |err| switch (err) {
                    error.AudioControlPeerClosed => {
                        unregisterSocket(supervisor.epoll_fd, &audio.process);
                        try signalChild(&audio.process);
                        audio.operation = .{ .terminating = monotonicNanoseconds() + std.time.ns_per_s };
                        return;
                    },
                    else => {
                        return err;
                    },
                };
                audio.operation = .{ .stopping = monotonicNanoseconds() + std.time.ns_per_s };
            } else {
                try signalChild(&audio.process);
                audio.operation = .{ .terminating = monotonicNanoseconds() + std.time.ns_per_s };
            }
        },
        .stopping, .canceling, .exiting, .terminating => {},
    }
}

fn sessionTranscriptBytesCapacity(configuration: CaptureOptions) usize {
    const bytes_per_recording_second_max: u64 = 64;
    const result_bytes_reserve: u64 =
        transcription_process.result_bytes_capacity;

    const bytes_capacity = @as(u64, configuration.recording_seconds) *
        bytes_per_recording_second_max + result_bytes_reserve;
    assert(bytes_capacity > 0);
    assert(bytes_capacity <= std.math.maxInt(u32));
    return @intCast(bytes_capacity);
}

fn startAudio(
    supervisor: *Supervisor,
    publication_event_fd: std.posix.fd_t,
) !void {
    var process = try startChild(
        supervisor,
        .audio_packet,
        .audio_exit,
        "audio-pipewire",
    );
    errdefer forceStopAndReap(&process) catch |err| log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Worker launch cleanup failed: error={s}", .{@errorName(err)});

    const audio = &supervisor.audio;
    assert(audio.process == null);
    const options = audio.options;

    try audio_process.PipeWireWorker.sendLaunch(
        process.socket.?,
        supervisor.audio_exchange_fd,
        publication_event_fd,
        .{
            .session_id = supervisor.audio_exchange.session_id,
            .source = options.source,
            .recording_samples_target = @as(u32, options.recording_seconds) *
                audio_process.sample_rate_hz,
            .automatic_stop = options.automatic_stop,
        },
    );
    audio.process = .{
        .process = process,
        .operation = .{
            .starting = monotonicNanoseconds() + 3 * std.time.ns_per_s,
        },
    };
}

fn startTranscription(
    supervisor: *Supervisor,
    retry_work: ?TranscriptionWork,
) !TranscriptionProcess {
    const model = supervisor.audio.options.transcription;
    var process = try startChild(
        supervisor,
        .transcription_packet,
        .transcription_exit,
        "transcription-model",
    );
    errdefer forceStopAndReap(&process) catch |err| log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Worker launch cleanup failed: error={s}", .{@errorName(err)});

    // A replacement loads the same model and receives the exact retained slot;
    // it does not reconnect capture or advance the publication ordinal.
    try transcription_process.sendModelLaunch(
        process.socket.?,
        supervisor.audio_exchange_fd,
        supervisor.transcript_exchange_fd,
        .{
            .session_id = supervisor.audio_exchange.session_id,
            .model = model.model,
            .inference_threads_count = model.inference_threads_count,
            .decoder_threads_count = model.decoder_threads_count,
            .encoder_trailing_padding = model.encoder_trailing_padding,
        },
    );

    const startup_duration_ns = 15 * std.time.ns_per_s;
    const started_monotonic_ns = monotonicNanoseconds();
    return .{
        .process = process,
        .operation = .{ .starting = .{
            .deadline_monotonic_ns = started_monotonic_ns + startup_duration_ns,
            .retry_work = retry_work,
        } },
    };
}

fn startChild(
    supervisor: *Supervisor,
    socket_source: EventSource,
    pid_source: EventSource,
    internal_role: []const u8,
) !ChildProcess {
    var sockets: [2]std.posix.fd_t = undefined;
    try checkSyscall("socketpair", linux.socketpair(
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
    try setCloseOnExec(supervisor_socket);

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
            logging.levelName(logging.level()),
        },
    });
    const pid = child.id.?;
    assert(child.stdin == null);
    assert(child.stdout == null);
    assert(child.stderr == null);
    errdefer child.kill(supervisor.io);
    closeDescriptor(worker_socket);
    worker_socket_is_owned = false;

    const pid_fd = try descriptorFromResult("pidfd_open", linux.pidfd_open(pid, 0));
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
    const session_audio = &supervisor.audio;
    const options = session_audio.options;
    const audio = &session_audio.process.?;
    while (audio_process.PipeWireWorker.receiveReportNonblocking(
        audio.process.socket.?,
        supervisor.audio_exchange,
        @as(u32, options.recording_seconds) * audio_process.sample_rate_hz,
        options.automatic_stop,
    )) |received| {
        const worker_report = received orelse {
            unregisterSocket(supervisor.epoll_fd, &audio.process);
            return;
        };
        audio.operation = .{
            .exiting = monotonicNanoseconds() + std.time.ns_per_s,
        };

        const capture = switch (worker_report) {
            .ok => |capture| capture,
            .err => |err| capture: {
                supervisor.audio.problem = audioProblem(err);
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio service error: recording_ordinal={d}, kind={t}", .{ supervisor.recording_ordinal, std.meta.activeTag(err) });
                switch (err) {
                    .source_not_found, .source_ambiguous, .setup => |detail| {
                        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Capture setup error: recording_ordinal={d}, kind={t}, stage={t}, domain={t}, code={d}, pipewire_server_version=\"{f}\", client_node_version_advertised={d}, client_node_version_selected={d}, detail=\"{f}\"", .{ supervisor.recording_ordinal, std.meta.activeTag(err), detail.stage, detail.domain, detail.code, std.zig.fmtString(detail.pipewire_version[0..detail.pipewire_version_size]), detail.client_node_version_advertised, detail.client_node_version_selected, std.zig.fmtString(detail.message[0..detail.message_size]) });
                        if (err == .source_ambiguous) log.err(.{}, "Microphone selection: sources_expected_count=1; replace microphone_serial with a microphone_node from the candidate list", .{});
                        try beginAbort(supervisor, .audio_failed);
                        continue;
                    },
                    .invalid_report => {
                        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Worker report corrupt: recording_ordinal={d}, role=audio", .{supervisor.recording_ordinal});
                        try beginAbort(supervisor, .{ .exchange_corrupt = .audio_slot });
                        continue;
                    },
                    .source_connection_lost, .source_changed, .source_observation, .pipeline_full, .capture, .teardown => |report| break :capture report,
                }
            },
        };
        if (supervisor.recording_stop == null and supervisor.phase == .active) {
            // Automatic and source-driven stops occur inside capture.
            // This baseline observes its final report, not that earlier
            // decision; it must not masquerade as command-to-done time.
            supervisor.recording_stop = .{ .monotonic_ns = monotonicNanoseconds(), .origin = .capture_end };
            log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Recording stop observed: recording_ordinal={d}, recording_stop_origin=capture_end", .{supervisor.recording_ordinal});
        }
        const timeline_validation = switch (audio_exchange.acquireTimelineValidation(
            supervisor.audio_exchange,
        )) {
            .published => |value| value,
            .unpublished, .corrupt => {
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=timeline", .{supervisor.recording_ordinal});
                try beginAbort(supervisor, .{ .exchange_corrupt = .timeline });
                continue;
            },
        };
        if ((capture.timeline_validation == .header_only) != (timeline_validation == .header_only)) {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=timeline", .{supervisor.recording_ordinal});
            try beginAbort(supervisor, .{ .exchange_corrupt = .timeline });
            continue;
        }
        logCaptureReport(supervisor.recording_ordinal, &capture);
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
            .cancelled => assert(supervisor.phase == .aborting),
            .failed => |failure| if (supervisor.phase == .active) {
                if (failure.outcome == .pipeline_full) {
                    supervisor.phase = .{ .finishing = .pipeline_full };
                } else if (capture.published_samples_count == 0) {
                    try beginAbort(supervisor, .audio_failed);
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
    }
}

fn observePipeWireProgress(supervisor: *Supervisor) !void {
    const audio = if (supervisor.audio.process) |*value| value else {
        return;
    };
    const capture_is_starting = audio.operation == .starting;
    const callbacks_count_previous = switch (audio.operation) {
        .starting => 0,
        .capturing => |progress| progress.callbacks_count,
        .stopping, .canceling, .exiting, .terminating => {
            return;
        },
    };

    const callbacks_count = audio_exchange.acquireAudioCallbacksCount(
        supervisor.audio_exchange,
    );
    assert(callbacks_count >= callbacks_count_previous);
    if (callbacks_count == callbacks_count_previous) {
        return;
    }

    if (capture_is_starting) {
        log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Capture started: recording_ordinal={d}, capture_start_duration_ms={d:.3}", .{
            supervisor.recording_ordinal,
            @as(f64, @floatFromInt(monotonicNanoseconds() - supervisor.recording_requested_monotonic_ns)) / std.time.ns_per_ms,
        });
        const timeline_validation = switch (audio_exchange.acquireTimelineValidation(
            supervisor.audio_exchange,
        )) {
            .published => |value| value,
            .unpublished, .corrupt => {
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=timeline", .{supervisor.recording_ordinal});
                try beginAbort(supervisor, .{ .exchange_corrupt = .timeline });
                return;
            },
        };
        if (timeline_validation == .header_only) {
            log.warn(
                .{ .recording_ordinal = supervisor.recording_ordinal },
                "Capture timeline validation limited: recording_ordinal={d}, validation=header_only; some dropped audio intervals cannot be detected",
                .{supervisor.recording_ordinal},
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
            if (transcription.operation == .idle) transcription.operation = .{ .exiting = monotonicNanoseconds() + std.time.ns_per_s };
            return;
        };
        const event = switch (report) {
            .ok => |event| event,
            .err => |err| {
                supervisor.transcription_problem = switch (err) {
                    .model_load => .model_load_failed,
                    .feature_extraction, .inference, .text_decode => .transcription_failed,
                    .exchange => .exchange_corrupt,
                };
                if (err == .exchange) try beginAbort(supervisor, .{ .exchange_corrupt = .audio_slot });
                switch (err) {
                    inline else => |detail| {
                        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription error: recording_ordinal={d}, model={s}, stage={t}, detail=\"{f}\"", .{ supervisor.recording_ordinal, supervisor.audio.options.transcription.model.name(), std.meta.activeTag(err), std.zig.fmtString(detail.messageBytes()) });
                        const evidence = detail.evidence;
                        if (evidence.chunk_available == 1) {
                            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription error evidence: recording_ordinal={d}, chunk_ordinal={d}, audio_samples_count={d}, audio_duration_seconds={d:.3}, tokens_count_max={d}, features_duration_ms={d:.3}, encoder_duration_ms={d:.3}, cross_key_values_duration_ms={d:.3}, decoder_duration_ms={d:.3}", .{
                                supervisor.recording_ordinal,                                      evidence.chunk,                                                             evidence.samples,
                                @as(f64, @floatFromInt(evidence.samples)) / 16000,                 evidence.token_limit,                                                       @as(f64, @floatFromInt(evidence.log_mel_ns)) / std.time.ns_per_ms,
                                @as(f64, @floatFromInt(evidence.encoder_ns)) / std.time.ns_per_ms, @as(f64, @floatFromInt(evidence.cross_key_values_ns)) / std.time.ns_per_ms, @as(f64, @floatFromInt(evidence.decoder_ns)) / std.time.ns_per_ms,
                            });
                            if (evidence.decoding_available == 1) log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Decoder error evidence: recording_ordinal={d}, chunk_ordinal={d}, tokens_count={d}, tokens_count_max={d}, encoder_positions_count={d}, no_speech_probability={d:.6}, average_log_probability={d:.6}", .{
                                supervisor.recording_ordinal, evidence.chunk,                 evidence.tokens,                  evidence.token_limit,
                                evidence.encoder_positions,   evidence.no_speech_probability, evidence.average_log_probability,
                            });
                        }
                    },
                }
                // The original diagnostic is retained in the journal. Reaping
                // still owns the bounded retry and the eventual session outcome.
                continue;
            },
        };
        switch (event) {
            .ready => |ready| switch (transcription.operation) {
                .starting => |starting| {
                    if (ready.model_prepare_duration_ns > 0) {
                        log.info(
                            .{ .recording_ordinal = supervisor.recording_ordinal },
                            "Model prepared: recording_ordinal={d}, model_prepare_duration_ms={d:.3}\n",
                            .{ supervisor.recording_ordinal, @as(f64, @floatFromInt(ready.model_prepare_duration_ns)) / std.time.ns_per_ms },
                        );
                    }
                    transcription.operation = .{ .idle = starting.retry_work };
                },
                .terminating => {},
                .idle, .busy, .shutdown_sent, .exiting => unreachable,
            },
            .result => |result| {
                if (supervisor.phase == .aborting or
                    transcription.operation == .terminating)
                {
                    // Cancellation owns the mailbox terminal state. A result
                    // packet already queued before termination cannot revive it.
                    continue;
                }
                assert(transcription.operation == .busy);
                const work = transcription.operation.busy.work.work();
                assert(result.publication_ordinal == work.publication_ordinal);
                try consumeTranscript(supervisor, work, result);
                if (supervisor.phase != .aborting) {
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
        }
    }
}

fn dispatchPublishedAudio(supervisor: *Supervisor) !void {
    if (supervisor.phase == .aborting or supervisor.phase == .idle or supervisor.phase == .delivering) {
        return;
    }
    if (supervisor.transcription != .running) {
        return;
    }
    const transcription = &supervisor.transcription.running;
    const retry_work = switch (transcription.operation) {
        .idle => |work| work,
        else => {
            return;
        },
    };
    const socket = transcription.process.socket orelse {
        return;
    };

    const work_attempt: WorkAttempt = if (retry_work) |work|
        .{ .retry = work }
    else fresh: {
        const pending = switch (nextPublishedAudio(supervisor)) {
            .none => return,
            .pending => |value| value,
            .corrupt => {
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=audio_slot", .{supervisor.recording_ordinal});
                try beginAbort(supervisor, .{ .exchange_corrupt = .audio_slot });
                return;
            },
        };
        break :fresh .{ .first = .{
            .slot_index = pending.index,
            .publication_ordinal = pending.publication.publication_ordinal,
        } };
    };
    const work = work_attempt.work();
    const inference_duration_ns = 10 * std.time.ns_per_s;
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
        else => {
            return send_error;
        },
    };
}

fn consumeTranscript(supervisor: *Supervisor, work: TranscriptionWork, timings: ?transcription_process.ResultReport) !void {
    const slot = &supervisor.audio_exchange.slots[work.slot_index.arrayIndex()];
    const publication = switch (audio_exchange.acquireSlot(slot)) {
        .published => |value| value,
        .empty, .corrupt => {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=audio_slot, chunk_ordinal={d}", .{ supervisor.recording_ordinal, work.publication_ordinal });
            try beginAbort(supervisor, .{ .exchange_corrupt = .audio_slot });
            return;
        },
    };

    const result = switch (transcription_process.acquireResult(
        supervisor.transcript_exchange,
    )) {
        .committed => |value| value,
        .none, .corrupt => {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=mailbox, chunk_ordinal={d}", .{ supervisor.recording_ordinal, work.publication_ordinal });
            try beginAbort(supervisor, .{ .exchange_corrupt = .mailbox });
            return;
        },
    };
    if (publication.publication_ordinal != work.publication_ordinal or
        result.publication_ordinal != work.publication_ordinal or
        result.publication_ordinal != supervisor.transcript.next_publication_ordinal or
        result.samples_count != publication.samples_count or
        result.contains_activity != publication.contains_activity)
    {
        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=mailbox, chunk_ordinal={d}", .{ supervisor.recording_ordinal, work.publication_ordinal });
        try beginAbort(supervisor, .{ .exchange_corrupt = .mailbox });
        return;
    }

    const disposition = transcription_process.classifyResult(result);
    assert(supervisor.transcript.bytes_count <= supervisor.transcript.bytes.len);
    const output_fits = result.bytes.len <= supervisor.transcript.bytes.len - supervisor.transcript.bytes_count;
    const compute_duration_ns = if (timings) |measured|
        measured.features_duration_ns + measured.inference_duration_ns
    else
        null;
    supervisor.transcript.processed_samples_count += result.samples_count;
    supervisor.transcript.compute_duration_ns = if (supervisor.transcript.compute_duration_ns != null and compute_duration_ns != null)
        supervisor.transcript.compute_duration_ns.? + compute_duration_ns.?
    else
        null;

    if (logging.enabled(.info)) {
        var features_buffer: [32]u8 = undefined;
        var inference_buffer: [32]u8 = undefined;
        var processing_buffer: [32]u8 = undefined;
        var speed_buffer: [32]u8 = undefined;
        log.info(
            .{ .recording_ordinal = supervisor.recording_ordinal },
            "Transcription chunk: recording_ordinal={d}, chunk_ordinal={d}, audio_duration_seconds={d:.3}, audio_samples_count={d}, " ++
                "features_duration_ms={s}, inference_duration_ms={s}, transcription_compute_duration_ms={s}, transcription_compute_speed_ratio={s}, " ++
                "transcript_size={d}, disposition={s}, chunk_limit={t}, recording_size_exceeded={}, activity_observed={}, text_empty={}, no_speech_probability={d:.6}, no_speech_inactive_probability_min={d:.2}, average_log_probability={d:.6}\n",
            .{
                supervisor.recording_ordinal,
                result.publication_ordinal,
                @as(f64, @floatFromInt(result.samples_count)) / audio_process.sample_rate_hz,
                result.samples_count,
                formatDurationMilliseconds(&features_buffer, if (timings) |measured| measured.features_duration_ns else null),
                formatDurationMilliseconds(&inference_buffer, if (timings) |measured| measured.inference_duration_ns else null),
                formatDurationMilliseconds(&processing_buffer, compute_duration_ns),
                formatComputeSpeedRatio(&speed_buffer, result.samples_count, compute_duration_ns),
                result.bytes.len,
                @tagName(disposition),
                result.limit,
                !output_fits,
                result.contains_activity,
                std.mem.trim(u8, result.bytes, " \t\r\n").len == 0,
                result.no_speech_probability,
                transcription_process.no_activity_no_speech_probability_reject_min,
                result.average_log_probability,
            },
        );
    }
    switch (disposition) {
        .accepted, .partial => {
            const available = supervisor.transcript.bytes.len - supervisor.transcript.bytes_count;
            const text = if (output_fits) result.bytes else transcription_process.utf8Prefix(result.bytes[0..available]);
            if (!output_fits) log.warn(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript capacity reached: recording_ordinal={d}, chunk_ordinal={d}, transcript_size_max={d}, transcript_committed_size={d}, chunk_transcript_size={d}, chunk_retained_size={d}", .{
                supervisor.recording_ordinal,      result.publication_ordinal, supervisor.transcript.bytes.len,
                supervisor.transcript.bytes_count, result.bytes.len,           text.len,
            });
            @memcpy(supervisor.transcript.bytes[supervisor.transcript.bytes_count..][0..text.len], text);
            supervisor.transcript.bytes_count += @intCast(text.len);
            supervisor.transcript.accepted_chunks_count += 1;
        },
        .no_speech => {
            supervisor.transcript.no_speech_chunks_count += 1;
        },
        .speech_detection_conflict, .speech_unrecognized => {
            const selected_model = supervisor.audio.options.transcription.model;
            const rejection: TranscriptionRejection = .{
                .model = selected_model,
                .publication_ordinal = result.publication_ordinal,
                .samples_count = result.samples_count,
                .contains_activity = result.contains_activity,
                .no_speech_probability = result.no_speech_probability,
                .average_log_probability = result.average_log_probability,
            };
            transcription_process.releaseResult(supervisor.transcript_exchange);
            try beginAbort(supervisor, switch (disposition) {
                .speech_detection_conflict => .{
                    .speech_detection_conflict = rejection,
                },
                .speech_unrecognized => .{ .speech_unrecognized = rejection },
                .accepted, .partial, .no_speech => unreachable,
            });
            return;
        },
    }
    supervisor.transcript.next_publication_ordinal += 1;

    transcription_process.releaseResult(supervisor.transcript_exchange);
    if (result.limit != .none or (disposition == .accepted and !output_fits)) {
        // Keep the audio slot sealed: deadline recovery can reach here while
        // diagnostic capture still reads it. Cancellation is asynchronous, so
        // releasing it first could let capture overwrite those borrowed samples.
        // Stop further work, retaining all copied text until delivery finishes.
        // Prefer the decoder warning if both chunk and recording limits apply;
        // the chunk event and capacity event preserve every limit in the journal.
        try beginAbort(supervisor, .{ .transcript_limit = switch (result.limit) {
            .none => .recording_size,
            .text_size => .chunk_size,
            .tokens_count => .decoder_tokens,
            .text_size_and_tokens_count => .chunk_size_and_decoder_tokens,
        } });
    } else audio_exchange.releaseConsumedSlot(slot);
}

fn nextPublishedAudio(supervisor: *Supervisor) union(enum) {
    none,
    pending: audio_exchange.PublishedSlotRef,
    corrupt,
} {
    for (&supervisor.audio_exchange.slots, 0..) |*slot, slot_index| {
        const publication = switch (audio_exchange.acquireSlot(slot)) {
            .empty => continue,
            .published => |value| value,
            .corrupt => return .corrupt,
        };
        if (publication.publication_ordinal !=
            supervisor.transcript.next_publication_ordinal) continue;
        return .{ .pending = .{
            .index = audio_exchange.SlotIndex.fromArrayIndex(slot_index),
            .publication = publication,
        } };
    }
    return .none;
}

fn reapAudio(supervisor: *Supervisor) !void {
    if (supervisor.audio.process.?.process.socket != null) {
        try drainAudioPackets(supervisor);
    }

    var audio = supervisor.audio.process.?;
    unregisterChild(supervisor.epoll_fd, &audio.process);
    const audio_exit = try reapChild(&audio.process);
    log.processExited(.{ .recording_ordinal = supervisor.recording_ordinal }, "audio", audio_exit, switch (audio.operation) {
        .exiting, .terminating => true,
        else => false,
    });
    closeDescriptor(audio.process.pid_fd);
    supervisor.audio.process = null;

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
    const transcription_exit = try reapChild(&transcription.process);
    log.processExited(.{ .recording_ordinal = supervisor.recording_ordinal }, "transcription", transcription_exit, switch (transcription.operation) {
        .shutdown_sent, .exiting, .terminating => true,
        else => false,
    });
    closeDescriptor(transcription.process.pid_fd);

    if (supervisor.phase == .aborting or supervisor.phase == .idle or supervisor.phase == .delivering or
        transcription.operation == .shutdown_sent or transcription.operation == .exiting or
        transcription.operation == .terminating)
    {
        supervisor.transcription = .absent;
        return;
    }

    if (transcription.operation != .busy) {
        supervisor.transcription = .absent;
        try beginAbort(supervisor, .transcription_failed);
        return;
    }
    const failed_work = transcription.operation.busy.work;
    switch (transcription_process.acquireResult(supervisor.transcript_exchange)) {
        .committed => {
            // Confidence rejection can discard the session. Publish the reaped
            // state first so that path cannot signal an already-closed pidfd.
            supervisor.transcription = .absent;
            try consumeTranscript(supervisor, failed_work.work(), null);
        },
        .corrupt => {
            supervisor.transcription = .absent;
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=mailbox", .{supervisor.recording_ordinal});
            try beginAbort(supervisor, .{ .exchange_corrupt = .mailbox });
        },
        .none => switch (failed_work) {
            .retry => {
                supervisor.transcription = .absent;
                try beginAbort(supervisor, .transcription_failed);
            },
            .first => |work| supervisor.transcription = .{ .restart_pending = work },
        },
    }
}

fn maintainWorkersAndSession(supervisor: *Supervisor) !void {
    if (supervisor.phase == .aborting or supervisor.phase == .idle or supervisor.phase == .delivering) {
        return;
    }

    // A retained retry has its own state while no process exists. Starting the
    // replacement here prevents events from the reaped process's epoll batch
    // from being applied to its successor.
    if (supervisor.transcription == .restart_pending) {
        const work = supervisor.transcription.restart_pending;
        log.warn(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription retry: recording_ordinal={d}, chunk_ordinal={d}, reason=worker_exited_without_result, attempt=2, attempts_count_max=2", .{ supervisor.recording_ordinal, work.publication_ordinal });
        supervisor.transcription = .{
            .running = try startTranscription(supervisor, work),
        };
        return;
    }
    switch (publishedSlots(supervisor)) {
        .corrupt => {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=audio_slot", .{supervisor.recording_ordinal});
            try beginAbort(supervisor, .{ .exchange_corrupt = .audio_slot });
            return;
        },
        .count => |count| {
            if (supervisor.transcription == .absent and count > 0) {
                supervisor.transcription = .{
                    .running = try startTranscription(supervisor, null),
                };
                return;
            }
            if (supervisor.phase != .finishing) {
                return;
            }
            if (audioProcessExists(supervisor)) {
                return;
            }
            if (count != 0) {
                return;
            }
        },
    }
    if (!supervisor.service.shutdown_requested and
        supervisor.service.model_keep_warm_seconds > 0)
    {
        return;
    }
    try requestTranscriptionShutdown(supervisor);
}

fn requestTranscriptionShutdown(supervisor: *Supervisor) !void {
    assert(!audioProcessExists(supervisor));
    if (supervisor.transcription != .running) {
        return;
    }
    const transcription = &supervisor.transcription.running;
    if (transcription.operation != .idle) {
        return;
    }
    const socket = transcription.process.socket orelse {
        try signalChild(&transcription.process);
        transcription.operation = .{ .terminating = monotonicNanoseconds() + std.time.ns_per_s };
        return;
    };
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
        else => {
            return send_error;
        },
    };
}

fn finishValidAudioPrefixOrDiscard(supervisor: *Supervisor) !void {
    assert(supervisor.phase == .active);

    switch (publishedSlots(supervisor)) {
        .corrupt => {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=audio_slot", .{supervisor.recording_ordinal});
            try beginAbort(supervisor, .{ .exchange_corrupt = .audio_slot });
        },
        .count => |count| if (supervisor.transcript.next_publication_ordinal > 0 or count > 0) {
            supervisor.phase = .{
                .finishing = .audio_failed_with_valid_prefix,
            };
        } else {
            try beginAbort(supervisor, .audio_failed);
        },
    }
}

fn beginAbort(supervisor: *Supervisor, reason: AbortReason) !void {
    if (supervisor.phase == .delivering) {
        // Accepted text may already be on the clipboard or pasted. Cancellation
        // stops further output; it cannot undo those external effects.
        cancelDelivery(supervisor);
        // stop() closed the clipboard input; no writer still borrows this text.
        // Explicit cancellation preserves the previous saved transcript.
        if (reason == .user_cancelled or reason == .service_signal) supervisor.transcript.bytes_count = 0;
        if (supervisor.service.shutdown_requested) try requestTranscriptionShutdown(supervisor);
        return;
    }
    if (supervisor.phase == .aborting) {
        // An explicit cancel/shutdown also cancels a pending partial delivery.
        if (reason == .user_cancelled or reason == .service_signal) supervisor.phase.aborting = reason;
        return;
    }
    supervisor.phase = .{ .aborting = reason };
    transcription_process.requestCancellation(supervisor.transcript_exchange);

    if (supervisor.audio.process) |*audio| {
        try requestAudioDiscard(supervisor, audio);
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

fn requestAudioDiscard(supervisor: *Supervisor, audio: *PipeWireAudioProcess) !void {
    switch (audio.operation) {
        .canceling, .exiting, .terminating => {},
        .starting, .capturing, .stopping => {
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
                    else => {
                        return send_error;
                    },
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
    if (supervisor.phase == .idle) {
        if (supervisor.phase.idle) |deadline| {
            if (now_monotonic_ns >= deadline and supervisor.service.pending_recording == null) {
                supervisor.phase.idle = null;
                try requestTranscriptionShutdown(supervisor);
            }
        }
    } else if (supervisor.phase != .aborting and supervisor.phase != .delivering and now_monotonic_ns >= supervisor.session_deadline_monotonic_ns) {
        if (supervisor.transcription == .running and supervisor.transcription.running.operation == .busy) {
            switch (transcription_process.acquireResult(supervisor.transcript_exchange)) {
                .committed => |result| if (result.limit != .none) try consumeTranscript(supervisor, supervisor.transcription.running.operation.busy.work.work(), null),
                .none => {},
                .corrupt => {
                    log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=mailbox", .{supervisor.recording_ordinal});
                    try beginAbort(supervisor, .{ .exchange_corrupt = .mailbox });
                    return;
                },
            }
        }
        try beginAbort(supervisor, .{ .deadline = .recording });
    }
    if (supervisor.audio.process) |*audio| {
        try applyAudioDeadline(supervisor, audio, now_monotonic_ns);
    }
    if (supervisor.transcription == .running) {
        const transcription = &supervisor.transcription.running;
        if (transcription.operation.deadlineMonotonicNs()) |deadline_monotonic_ns| {
            if (now_monotonic_ns >= deadline_monotonic_ns) {
                switch (transcription.operation) {
                    .starting, .busy => {
                        if (transcription.operation == .busy) {
                            switch (transcription_process.acquireResult(supervisor.transcript_exchange)) {
                                .committed => |result| if (result.limit != .none) {
                                    log.warn(.{ .recording_ordinal = supervisor.recording_ordinal }, "Limited transcription recovered at deadline: recording_ordinal={d}, chunk_ordinal={d}", .{ supervisor.recording_ordinal, result.publication_ordinal });
                                    try consumeTranscript(supervisor, transcription.operation.busy.work.work(), null);
                                    // Abort waits for both workers before any storage is reused.
                                    return;
                                },
                                .none => {},
                                .corrupt => {
                                    log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio exchange corrupt: recording_ordinal={d}, field=mailbox", .{supervisor.recording_ordinal});
                                    try beginAbort(supervisor, .{ .exchange_corrupt = .mailbox });
                                    return;
                                },
                            }
                        }
                        const stage = std.meta.activeTag(transcription.operation);
                        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription deadline exceeded: recording_ordinal={d}, stage={t}, deadline_monotonic_ns={d}, observed_monotonic_ns={d}", .{
                            supervisor.recording_ordinal, stage, deadline_monotonic_ns, now_monotonic_ns,
                        });
                        try signalChild(&transcription.process);
                        transcription.operation = .{
                            .terminating = now_monotonic_ns + std.time.ns_per_s,
                        };
                        try beginAbort(supervisor, .{ .deadline = if (stage == .starting) .model_load else .transcription });
                    },
                    .shutdown_sent, .exiting => {
                        assert(supervisor.phase != .active);
                        try signalChild(&transcription.process);
                        transcription.operation = .{
                            .terminating = now_monotonic_ns + std.time.ns_per_s,
                        };
                    },
                    .terminating => {
                        return error.TranscriptionWorkerExitDeadlineExceeded;
                    },
                    .idle => unreachable,
                }
            }
        }
    }
}

fn applyAudioDeadline(
    supervisor: *Supervisor,
    audio: *PipeWireAudioProcess,
    now_monotonic_ns: u64,
) !void {
    const deadline_monotonic_ns = audio.operation.deadlineMonotonicNs();
    if (now_monotonic_ns < deadline_monotonic_ns) {
        return;
    }

    const target: []const u8 = switch (supervisor.audio.options.source) {
        .default => "default source",
        .node_name, .device_serial => |text| text,
    };
    log.warn(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio worker deadline exceeded: recording_ordinal={d}, stage={s}, target=\"{f}\", audio_callbacks_count={d}, audio_samples_count={d}, deadline_monotonic_ns={d}, observed_monotonic_ns={d}", .{ supervisor.recording_ordinal, @tagName(audio.operation), std.zig.fmtString(target), audio_exchange.acquireAudioCallbacksCount(supervisor.audio_exchange), audio_exchange.acquireAudioSamplesCount(supervisor.audio_exchange), deadline_monotonic_ns, now_monotonic_ns });
    switch (audio.operation) {
        .starting, .capturing => {
            supervisor.audio.problem = if (audio.operation == .starting) .audio_start_timed_out else .audio_stalled;
            try signalChild(&audio.process);
            audio.operation = .{
                .terminating = now_monotonic_ns + std.time.ns_per_s,
            };
            if (supervisor.phase == .active) {
                try finishValidAudioPrefixOrDiscard(supervisor);
            }
        },
        .stopping, .canceling, .exiting => {
            assert(supervisor.phase != .active);
            try signalChild(&audio.process);
            audio.operation = .{
                .terminating = now_monotonic_ns + std.time.ns_per_s,
            };
        },
        .terminating => {
            return error.AudioWorkerExitDeadlineExceeded;
        },
    }
}

fn armNearestDeadline(
    supervisor: *Supervisor,
    timer_fd: std.posix.fd_t,
) !void {
    var nearest_deadline_monotonic_ns: u64 = switch (supervisor.phase) {
        .idle => |deadline| deadline orelse std.math.maxInt(u64),
        .aborting, .delivering => std.math.maxInt(u64),
        else => supervisor.session_deadline_monotonic_ns,
    };
    nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, supervisor.notifications.deadline_monotonic_ns);
    if (supervisor.service.control.deadline()) |deadline| {
        nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);
    }
    for (&supervisor.clipboard) |*slot| if (slot.*) |*owner| {
        if (owner.process.deadlineMonotonicNs()) |deadline| nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);
    };
    if (supervisor.phase == .delivering and supervisor.phase.delivering.paste == .settling)
        nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, supervisor.phase.delivering.paste.settling);
    if (supervisor.keyboard) |*keyboard| {
        if (keyboard.deadlineMonotonicNs()) |deadline| nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);
    }
    if (supervisor.audio.process) |audio| {
        nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, audio.operation.deadlineMonotonicNs());
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
    if (nearest_deadline_monotonic_ns == std.math.maxInt(u64)) armed.it_value = .{ .sec = 0, .nsec = 0 };
    try checkSyscall("timerfd_settime", linux.timerfd_settime(
        timer_fd,
        .{},
        &armed,
        null,
    ));
}

fn sessionIsComplete(supervisor: *Supervisor) bool {
    if (supervisor.phase == .active or supervisor.phase == .idle or supervisor.phase == .delivering or audioProcessExists(supervisor)) {
        return false;
    }
    if (supervisor.transcription == .absent) {
        return switch (publishedSlots(supervisor)) {
            .count => |count| count == 0 or supervisor.phase == .aborting,
            .corrupt => supervisor.phase == .aborting,
        };
    }
    return supervisor.phase == .finishing and
        !supervisor.service.shutdown_requested and supervisor.service.model_keep_warm_seconds > 0 and
        supervisor.transcription == .running and supervisor.transcription.running.operation == .idle and
        supervisor.transcription.running.operation.idle == null and switch (publishedSlots(supervisor)) {
        .count => |count| count == 0,
        .corrupt => false,
    };
}

fn finishSession(supervisor: *Supervisor) !void {
    assert(sessionIsComplete(supervisor));

    var processing_buffer: [32]u8 = undefined;
    var speed_buffer: [32]u8 = undefined;
    var stop_buffer: [32]u8 = undefined;
    const stop_origin = if (supervisor.recording_stop) |stop| @tagName(stop.origin) else "unavailable";
    const audio_seconds = @as(f64, @floatFromInt(supervisor.transcript.processed_samples_count)) / audio_process.sample_rate_hz;
    const processing = formatDurationMilliseconds(&processing_buffer, supervisor.transcript.compute_duration_ns);
    const realtime_speed = formatComputeSpeedRatio(&speed_buffer, supervisor.transcript.processed_samples_count, supervisor.transcript.compute_duration_ns);

    switch (supervisor.phase) {
        .active, .idle, .delivering => unreachable,
        .finishing => |reason| {
            assert(switch (publishedSlots(supervisor)) {
                .count => |count| count == 0,
                .corrupt => false,
            });
            const outcome = if (supervisor.transcript.accepted_chunks_count == 0 and supervisor.transcript.no_speech_chunks_count > 0)
                "no_speech"
            else
                @tagName(reason);
            try finishTranscription(supervisor, outcome, switch (reason) {
                .audio_failed_with_valid_prefix, .pipeline_full => supervisor.audio.problem orelse .recording_incomplete,
                else => supervisor.audio.problem,
            }, false);
        },
        .aborting => |reason| {
            if (reason == .transcript_limit) {
                try finishTranscription(supervisor, @tagName(reason.transcript_limit), switch (reason.transcript_limit) {
                    .recording_size => .transcript_too_large,
                    .chunk_size => .transcript_chunk_too_large,
                    .decoder_tokens => .transcript_token_limit,
                    .chunk_size_and_decoder_tokens => .transcript_chunk_and_token_limit,
                }, true);
                return;
            }
            // No process can still publish into cancelled storage. The next
            // service recording resets both exchanges before either role starts.
            log.write(if (reason == .user_cancelled or reason == .service_signal) .info else .warn, .{ .recording_ordinal = supervisor.recording_ordinal }, "Recording discarded: recording_ordinal={d}, reason={s}, audio_duration_seconds={d:.3}, transcription_compute_duration_ms={s}, transcription_compute_speed_ratio={s}, transcript_size=0, recording_stop_origin={s}, recording_stop_elapsed_ms={s}", .{
                supervisor.recording_ordinal, @tagName(std.meta.activeTag(reason)),                             audio_seconds, processing, realtime_speed,
                stop_origin,                  formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
            });
            switch (reason) {
                .service_signal, .user_cancelled => {},
                .audio_failed => supervisor.notifications.show(supervisor.audio.problem orelse .microphone_failed),
                .transcription_failed => supervisor.notifications.show(supervisor.transcription_problem orelse .transcription_failed),
                .exchange_corrupt => supervisor.notifications.show(.exchange_corrupt),
                .deadline => |stage| supervisor.notifications.show(switch (stage) {
                    .recording => .recording_timed_out,
                    .model_load => .model_load_timed_out,
                    .transcription => .transcription_timed_out,
                }),
                .transcript_limit => unreachable,
                .speech_detection_conflict => supervisor.notifications.show(.speech_detection_conflict),
                .speech_unrecognized => supervisor.notifications.show(.speech_unrecognized),
            }
            switch (reason) {
                .speech_detection_conflict, .speech_unrecognized => |rejection| {
                    log.err(
                        .{ .recording_ordinal = supervisor.recording_ordinal },
                        "Transcription evidence: recording_ordinal={d}, model={s}, chunk_ordinal={d}, audio_samples_count={d}, " ++
                            "activity_observed={}, no_speech_probability={d:.6}, " ++
                            "no_speech_inactive_probability_min={d:.2}, no_speech_active_probability_min={d:.2}, " ++
                            "average_log_probability={d:.6}\n",
                        .{
                            supervisor.recording_ordinal,
                            rejection.model.name(),
                            rejection.publication_ordinal,
                            rejection.samples_count,
                            rejection.contains_activity,
                            rejection.no_speech_probability,
                            transcription_process.no_activity_no_speech_probability_reject_min,
                            transcription_process.active_no_speech_probability_conflict_min,
                            rejection.average_log_probability,
                        },
                    );
                },
                .transcript_limit => unreachable,
                .service_signal,
                .user_cancelled,
                .audio_failed,
                .transcription_failed,
                .exchange_corrupt,
                .deadline,
                => {},
            }
        },
    }
}

// Both normal completion and a capacity stop deliver the one accumulated buffer.
// A limit may leave sealed audio unprocessed; workers are reaped before this call.
fn finishTranscription(supervisor: *Supervisor, outcome_name: []const u8, problem: ?notifications.Problem, limited: bool) !void {
    var processing_buffer: [32]u8 = undefined;
    var speed_buffer: [32]u8 = undefined;
    var stop_buffer: [32]u8 = undefined;
    const stop_origin = if (supervisor.recording_stop) |stop| @tagName(stop.origin) else "unavailable";
    const audio_seconds = @as(f64, @floatFromInt(supervisor.transcript.processed_samples_count)) / audio_process.sample_rate_hz;
    const processing = formatDurationMilliseconds(&processing_buffer, supervisor.transcript.compute_duration_ns);
    const realtime_speed = formatComputeSpeedRatio(&speed_buffer, supervisor.transcript.processed_samples_count, supervisor.transcript.compute_duration_ns);

    const transcript = std.mem.trim(
        u8,
        supervisor.transcript.bytes[0..supervisor.transcript.bytes_count],
        " \t\r\n",
    );
    log.write(if (limited) .warn else .info, .{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription complete: recording_ordinal={d}, outcome={s}, chunks_accepted_count={d}, chunks_no_speech_count={d}, chunks_count={d}, audio_duration_seconds={d:.3}, transcription_compute_duration_ms={s}, transcription_compute_speed_ratio={s}, transcript_size={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}", .{
        supervisor.recording_ordinal,
        outcome_name,
        supervisor.transcript.accepted_chunks_count,
        supervisor.transcript.no_speech_chunks_count,
        supervisor.transcript.next_publication_ordinal,
        audio_seconds,
        processing,
        realtime_speed,
        transcript.len,
        stop_origin,
        formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
    });

    if (transcript.len == 0) {
        if (problem) |cause| supervisor.notifications.show(cause);
        return;
    }
    if (supervisor.service.output == .stdout) {
        try printOutput(supervisor.io, transcript);
        log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript written: recording_ordinal={d}, transcript_size={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}", .{
            supervisor.recording_ordinal, transcript.len, stop_origin, formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
        });
        if (problem) |cause| supervisor.notifications.show(cause);
    } else beginDelivery(supervisor, problem);
}

fn formatDurationMilliseconds(buffer: *[32]u8, elapsed_ns: ?u64) []const u8 {
    const elapsed = elapsed_ns orelse return "unavailable";
    return std.fmt.bufPrint(buffer, "{d:.3}", .{@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_ms}) catch unreachable;
}

fn formatComputeSpeedRatio(buffer: *[32]u8, samples_count: u64, compute_duration_ns: ?u64) []const u8 {
    const elapsed_ns = compute_duration_ns orelse return "unavailable";
    if (elapsed_ns == 0 or samples_count == 0) return "unavailable";
    const audio_seconds = @as(f64, @floatFromInt(samples_count)) / audio_process.sample_rate_hz;
    const processing_seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    return std.fmt.bufPrint(buffer, "{d:.2}", .{audio_seconds / processing_seconds}) catch unreachable;
}

// Explicit diagnostic mode: synchronous stdout after capture/inference drain.
fn printOutput(io: std.Io, transcript: []const u8) !void {
    const stdout = std.Io.File.stdout();
    try stdout.writeStreamingAll(io, transcript);
    try stdout.writeStreamingAll(io, "\n");
}

fn beginDelivery(supervisor: *Supervisor, problem: ?notifications.Problem) void {
    const delivery_started_ns = monotonicNanoseconds();
    if (supervisor.service.output == .desktop and supervisor.keyboard == null) openPasteKeyboard(supervisor);
    const slot: u1 = if (supervisor.clipboard[0] == null) 0 else 1;
    assert(supervisor.clipboard[slot] == null);
    const process = switch (clipboard_process.Process.start(
        supervisor.io,
        supervisor.epoll_fd,
        @intFromEnum(EventSource.clipboard),
        monotonicNanoseconds(),
    )) {
        .ok => |process| process,
        .err => |err| {
            logClipboardError(supervisor.recording_ordinal, err);
            completeDelivery(supervisor, clipboardProblem(err));
            return;
        },
    };
    supervisor.clipboard[slot] = .{ .recording_ordinal = supervisor.recording_ordinal, .process = process };
    supervisor.phase = .{ .delivering = .{ .slot = slot, .boundary_monotonic_ns = delivery_started_ns, .problem = problem } };
}

fn advanceOutput(supervisor: *Supervisor) !void {
    const now_ns = monotonicNanoseconds();
    var stop_buffer: [32]u8 = undefined;
    const stop_origin = if (supervisor.recording_stop) |stop| @tagName(stop.origin) else "unavailable";
    const text = std.mem.trim(u8, supervisor.transcript.bytes[0..supervisor.transcript.bytes_count], " \t\r\n");
    for (&supervisor.clipboard, 0..) |*slot, index| {
        const owner = if (slot.*) |*value| value else continue;
        const process = &owner.process;
        if (supervisor.service.shutdown_requested) process.stop(supervisor.epoll_fd, now_ns);
        const event = switch (process.advance(supervisor.epoll_fd, text, now_ns)) {
            .ok => |event| event,
            .err => |err| {
                logClipboardError(owner.recording_ordinal, err);
                if (process.operation == .reaping or process.operation == .terminating) {
                    if (supervisor.phase == .delivering) {
                        cancelDelivery(supervisor);
                        completeDelivery(supervisor, .clipboard_failed);
                    }
                    return error.ClipboardCleanupFailed;
                }
                if (supervisor.phase == .delivering and supervisor.phase.delivering.slot == index)
                    supervisor.phase.delivering.problem = clipboardProblem(err);
                process.stop(supervisor.epoll_fd, now_ns);
                continue;
            },
        };
        switch (event) {
            .none => {},
            .finished => slot.* = null,
            .acquired => {
                assert(supervisor.phase == .delivering and supervisor.phase.delivering.slot == index);
                if (supervisor.clipboard[1 - index]) |*previous| previous.process.stop(supervisor.epoll_fd, now_ns);
                log.info(.{ .recording_ordinal = owner.recording_ordinal }, "Clipboard acquired: recording_ordinal={d}, transcript_size={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}, clipboard_acquire_duration_ms={d:.3}", .{
                    owner.recording_ordinal,                                                                                  text.len, stop_origin, formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
                    @as(f64, @floatFromInt(now_ns - supervisor.phase.delivering.boundary_monotonic_ns)) / std.time.ns_per_ms,
                });
                supervisor.phase.delivering.boundary_monotonic_ns = now_ns;
                if (supervisor.service.output == .desktop and supervisor.keyboard == null)
                    supervisor.phase.delivering.problem = supervisor.paste_problem orelse .paste_failed;
                supervisor.phase.delivering.paste = if (supervisor.service.output == .desktop and supervisor.keyboard != null)
                    .{ .settling = @max(now_ns + @as(u64, supervisor.service.paste_settle_ms) * std.time.ns_per_ms, supervisor.keyboard.?.usable_after_ns) }
                else
                    .done;
            },
        }
    }
    if (supervisor.phase != .delivering) return;
    const delivery = &supervisor.phase.delivering;
    const current = &supervisor.clipboard[delivery.slot];
    if (current.* == null or (current.*.?.process.operation != .copying and current.*.?.process.operation != .owning)) {
        if (delivery.paste != .done and delivery.problem == null) delivery.problem = .clipboard_failed;
        cancelDelivery(supervisor);
    }
    switch (delivery.paste) {
        .waiting, .done => {},
        .settling => |deadline_ns| if (now_ns >= deadline_ns) {
            supervisor.keyboard.?.beginPaste(supervisor.service.paste_key, supervisor.service.paste_key_gap_ms, now_ns);
            delivery.paste = .sending;
        },
        .sending => {},
    }
    if (delivery.paste == .sending) {
        const complete = switch (supervisor.keyboard.?.advance(now_ns)) {
            .ok => |complete| complete,
            .err => |err| failed: {
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Paste error; clipboard retained, no retry: recording_ordinal={d}, detail={any}", .{ supervisor.recording_ordinal, err });
                supervisor.keyboard.?.deinit();
                supervisor.keyboard = null;
                delivery.paste = .done;
                delivery.problem = pasteProblem(err);
                break :failed false;
            },
        };
        if (complete) {
            log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Paste shortcut sent: recording_ordinal={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}, paste_duration_ms={d:.3}", .{
                supervisor.recording_ordinal,                                                          stop_origin, formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
                @as(f64, @floatFromInt(now_ns - delivery.boundary_monotonic_ns)) / std.time.ns_per_ms,
            });
            delivery.paste = .done;
        }
    }
    if (delivery.paste == .done and clipboardSlotsSettled(supervisor)) {
        completeDelivery(supervisor, delivery.problem);
        enterIdle(supervisor);
    }
}

// Save only after delivery releases the text. Choose the popup from both
// outcomes, so it never promises a saved file before the rename succeeds.
fn completeDelivery(supervisor: *Supervisor, problem: ?notifications.Problem) void {
    const text = std.mem.trim(u8, supervisor.transcript.bytes[0..supervisor.transcript.bytes_count], " \t\r\n");
    if (text.len == 0) return;
    const save_error = saveTranscript(supervisor, text);
    const saved = save_error == null;
    if (supervisor.service.shutdown_requested) return;
    if (problem) |cause| {
        const output: notifications.Output = switch (cause) {
            .clipboard_failed, .clipboard_tool_missing => if (saved) .saved else .unsaved,
            .paste_failed, .paste_permission_denied, .paste_device_missing, .paste_incomplete => if (saved) .clipboard_saved else .clipboard_unsaved,
            else => if (saved) .partial_saved else .partial_unsaved,
        };
        supervisor.notifications.showOutput(cause, output);
    } else if (save_error) |err| {
        supervisor.notifications.showOutput(saveProblem(err), .clipboard_unsaved);
    } else supervisor.notifications.recover();
}

fn saveTranscript(supervisor: *const Supervisor, text: []const u8) ?transcript_file.Error {
    const directory_path = supervisor.service.transcript_directory.?;
    const started = monotonicNanoseconds();
    var elapsed_buffer: [32]u8 = undefined;
    switch (transcript_file.save(supervisor.io, directory_path, text)) {
        .ok => {},
        .err => |err| {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript save error: recording_ordinal={d}, path=\"{f}/transcript.txt\", detail={any}, transcript_save_duration_ms={s}", .{ supervisor.recording_ordinal, std.zig.fmtString(directory_path), err, formatDurationMilliseconds(&elapsed_buffer, monotonicNanoseconds() - started) });
            return err;
        },
    }
    log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript saved: recording_ordinal={d}, transcript_size={d}, transcript_save_duration_ms={s}", .{
        supervisor.recording_ordinal, text.len, formatDurationMilliseconds(&elapsed_buffer, monotonicNanoseconds() - started),
    });
    return null;
}

fn formatRecordingStopElapsedMilliseconds(buffer: *[32]u8, supervisor: *const Supervisor) []const u8 {
    const stop = supervisor.recording_stop orelse return "unavailable";
    return formatDurationMilliseconds(buffer, monotonicNanoseconds() - stop.monotonic_ns);
}

fn cancelDelivery(supervisor: *Supervisor) void {
    const delivery = &supervisor.phase.delivering;
    if (supervisor.clipboard[delivery.slot]) |*owner| owner.process.stop(supervisor.epoll_fd, monotonicNanoseconds());
    if (supervisor.keyboard) |*keyboard| {
        if (keyboard.pending != null) {
            keyboard.deinit();
            supervisor.keyboard = null;
        }
    }
    delivery.paste = .done;
}

fn clipboardSlotsSettled(supervisor: *const Supervisor) bool {
    for (supervisor.clipboard) |slot| if (slot) |owner| {
        if (owner.process.operation != .owning) return false;
    };
    return true;
}

fn openPasteKeyboard(supervisor: *Supervisor) void {
    switch (paste_keyboard.Keyboard.open(monotonicNanoseconds())) {
        .ok => |keyboard| {
            supervisor.keyboard = keyboard;
            supervisor.paste_problem = null;
        },
        .err => |err| {
            supervisor.paste_problem = pasteProblem(err);
            log.warn(.{}, "Automatic paste unavailable; clipboard delivery remains enabled: detail={any}", .{err});
        },
    }
}

fn audioProblem(err: audio_process.Error) notifications.Problem {
    return switch (err) {
        .source_not_found => .microphone_not_found,
        .source_ambiguous => .microphone_ambiguous,
        .source_connection_lost => .microphone_connection_lost,
        .source_changed => .microphone_changed,
        .source_observation => .microphone_identity_unavailable,
        .pipeline_full => .audio_processing_behind,
        .setup => .audio_setup_failed,
        .capture => .microphone_failed,
        .teardown => .audio_teardown_failed,
        .invalid_report => .exchange_corrupt,
    };
}

fn clipboardProblem(err: clipboard_process.Error) notifications.Problem {
    return switch (err) {
        .worker => |detail| switch (detail.kind) {
            .tool_not_found => .clipboard_tool_missing,
            .launcher_exited, .owner_unavailable, .system, .cleanup => .clipboard_failed,
        },
        .launch, .system, .input_write, .timed_out, .invalid_report => .clipboard_failed,
    };
}

fn logClipboardError(recording: u64, err: clipboard_process.Error) void {
    switch (err) {
        .worker => |*detail| {
            log.err(.{ .recording_ordinal = recording }, "Clipboard error: recording_ordinal={d}, kind={t}", .{ recording, detail.kind });
            // Worker journals full stderr and native/exit details before cleanup;
            // it remains observable even when this endpoint has been cancelled.
        },
        else => log.err(.{ .recording_ordinal = recording }, "Clipboard error: recording_ordinal={d}, detail={any}", .{ recording, err }),
    }
}

fn pasteProblem(err: paste_keyboard.Error) notifications.Problem {
    return switch (err) {
        .open => |errno| switch (errno) {
            .ACCES, .PERM => .paste_permission_denied,
            .NOENT => .paste_device_missing,
            else => .paste_failed,
        },
        .configure => .paste_failed,
        .write, .ambiguous_write, .timed_out => .paste_incomplete,
    };
}

fn saveProblem(err: transcript_file.Error) notifications.Problem {
    const cause: anyerror = switch (err) {
        .open_directory, .permissions, .create_temporary => |cause| cause,
        .replace => |detail| detail.cause,
        .write => |detail| detail.cause,
        .unsafe_directory => return .transcript_directory_unsafe,
        .stat_directory => return .transcript_save_failed,
    };
    return switch (cause) {
        error.NoSpaceLeft, error.DiskQuota => .transcript_storage_full,
        error.AccessDenied, error.PermissionDenied, error.ReadOnlyFileSystem => .transcript_save_denied,
        else => .transcript_save_failed,
    };
}

fn enterIdle(supervisor: *Supervisor) void {
    // Delivery no longer borrows session text. An acknowledged result or reaped
    // worker released every audio borrow before this idle timer starts.
    supervisor.phase = .{ .idle = if (supervisor.transcription == .running and !supervisor.service.shutdown_requested)
        monotonicNanoseconds() + @as(u64, supervisor.service.model_keep_warm_seconds) * std.time.ns_per_s
    else
        null };
}

fn logCaptureReport(recording_ordinal: u64, report: *const audio_process.CaptureReport) void {
    const outcome_name = switch (report.end) {
        .completed => "completed",
        .automatic_stop => "automatic_stop",
        .stopped => "stopped",
        .cancelled => "cancelled",
        .failed => |failure| @tagName(failure.outcome),
    };
    const source_description = if (report.source_identity) |source|
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

    log.info(.{ .recording_ordinal = recording_ordinal }, "Capture ended: recording_ordinal={d}, outcome={s}, audio_duration_seconds={d:.3}, audio_samples_published_count={d}, audio_samples_captured_count={d}, microphone_description=\"{f}\"", .{ recording_ordinal, outcome_name, @as(f64, @floatFromInt(report.samples_count)) / audio_process.sample_rate_hz, report.published_samples_count, report.samples_count, std.zig.fmtString(source_description) });

    log.info(.{ .recording_ordinal = recording_ordinal }, "Capture protocol: recording_ordinal={d}, pipewire_server_version=\"{f}\", client_node_version_advertised={d}, client_node_version_selected={d}, graph_rate_hz={d}, graph_channels_count={d}, callback_duration_ms_max={d:.3}, callback_gap_ms_max={d:.3}", .{ recording_ordinal, std.zig.fmtString(report.pipewire_server_version[0..report.pipewire_server_version_size]), report.client_node_version_advertised, report.client_node_version_selected, if (report.negotiated_format) |format| format.sample_rate_hz else 0, if (report.negotiated_format) |format| format.channels_count else 0, if (report.callback) |callback| @as(f64, @floatFromInt(callback.duration_ns_max)) / std.time.ns_per_ms else 0, if (report.callback) |callback| @as(f64, @floatFromInt(callback.gap_ns_max)) / std.time.ns_per_ms else 0 });

    if (report.end == .failed or report.teardown_failure != null) {
        if (report.source_identity) |source| log.err(.{ .recording_ordinal = recording_ordinal }, "Capture source: recording_ordinal={d}, node_id={d}, node_object_serial={d}, device_id={d}, device_object_serial={d}, node_name=\"{f}\", node_description=\"{f}\", device_serial=\"{f}\", device_description=\"{f}\"", .{ recording_ordinal, source.node_id, source.node_object_serial, source.device_id, source.device_object_serial, std.zig.fmtString(source.node_name[0..source.node_name_size]), std.zig.fmtString(source.node_description[0..source.node_description_size]), std.zig.fmtString(source.device_serial[0..source.device_serial_size]), std.zig.fmtString(source.device_description[0..source.device_description_size]) });
    }
    if (report.end == .failed) {
        const failure = report.end.failed.detail;
        log.err(.{ .recording_ordinal = recording_ordinal }, "Capture failed: recording_ordinal={d}, stage={s}, error_domain={s}, error_code={d}, detail=\"{f}\"", .{ recording_ordinal, @tagName(failure.coordinate.stage), @tagName(failure.coordinate.domain), failure.coordinate.code, std.zig.fmtString(failure.message[0..failure.message_size]) });
    }
    if (report.teardown_failure) |failure| {
        log.err(.{ .recording_ordinal = recording_ordinal }, "Capture teardown failed: recording_ordinal={d}, stage={s}, error_domain={s}, error_code={d}, detail=\"{f}\"", .{ recording_ordinal, @tagName(failure.coordinate.stage), @tagName(failure.coordinate.domain), failure.coordinate.code, std.zig.fmtString(failure.message[0..failure.message_size]) });
    }

    // These failures reduce scheduling guarantees, not the validity of already
    // captured samples. Preserve the warnings without discarding valid audio.
    if (report.callback != null and
        (scheduler_priority <= 0 or !audio_process.schedulerPolicyIsRealtime(scheduler_policy)))
    {
        log.warn(.{ .recording_ordinal = recording_ordinal }, "Capture did not obtain realtime scheduling: recording_ordinal={d}, policy={s}, priority={d}", .{
            recording_ordinal, audio_process.schedulerPolicyName(scheduler_policy), scheduler_priority,
        });
    }
    if (report.memory_lock == .unavailable) {
        const unavailable = report.memory_lock.unavailable;
        log.warn(.{ .recording_ordinal = recording_ordinal }, "Capture memory was not locked: recording_ordinal={d}, errno={d}, memory_lock_size_max={d}", .{
            recording_ordinal, unavailable.error_code, unavailable.limit_bytes,
        });
    }
}

fn audioProcessExists(supervisor: *const Supervisor) bool {
    return supervisor.audio.process != null;
}

fn publishedSlots(supervisor: *Supervisor) union(enum) { count: u32, corrupt } {
    var count: u32 = 0;
    for (&supervisor.audio_exchange.slots) |*slot| {
        switch (audio_exchange.acquireSlot(slot)) {
            .empty => {},
            .published => count += 1,
            .corrupt => return .corrupt,
        }
    }
    return .{ .count = count };
}

fn createSharedMemory(name: [:0]const u8, size: usize) !std.posix.fd_t {
    const descriptor = try std.posix.memfd_create(
        name,
        linux.MFD.CLOEXEC | linux.MFD.ALLOW_SEALING,
    );
    errdefer closeDescriptor(descriptor);
    try checkSyscall("ftruncate", linux.ftruncate(descriptor, @intCast(size)));
    try checkSyscall("fcntl", linux.fcntl(
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
    if (linux.errno(result) != .SUCCESS) {
        log.err(.{}, "Event registration failed: operation=epoll_ctl_add, fd={d}, source={t}, errno={t}", .{ descriptor, source, linux.errno(result) });
        return error.SupervisorEpollRegisterFailed;
    }
}

fn unregister(epoll_fd: std.posix.fd_t, descriptor: std.posix.fd_t) void {
    const result = linux.epoll_ctl(
        epoll_fd,
        linux.EPOLL.CTL_DEL,
        descriptor,
        null,
    );
    logCleanupSyscall("epoll_ctl_del", result);
}

fn unregisterSocket(epoll_fd: std.posix.fd_t, process: *ChildProcess) void {
    const socket = process.socket orelse {
        return;
    };
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

// This rollback path runs outside the event loop, including partially failed
// launches. It closes every owned descriptor exactly once even if SIGKILL
// cannot complete; only pidfd readiness permits the otherwise blocking reap.
fn forceStopAndReap(process: *ChildProcess) !void {
    defer {
        if (process.socket) |socket| closeDescriptor(socket);
        process.socket = null;
        closeDescriptor(process.pid_fd);
    }
    try signalChild(process);
    const deadline_monotonic_ns = monotonicNanoseconds() + std.time.ns_per_s;
    var descriptors = [_]linux.pollfd{.{ .fd = process.pid_fd, .events = linux.POLL.IN, .revents = 0 }};
    while (true) {
        const remaining_ns = deadline_monotonic_ns -| monotonicNanoseconds();
        if (remaining_ns == 0) {
            return error.SupervisorWorkerCleanupDeadlineExceeded;
        }
        const timeout_ms: i32 = @intCast(std.math.divCeil(u64, remaining_ns, std.time.ns_per_ms) catch unreachable);
        const result = linux.poll(&descriptors, descriptors.len, timeout_ms);
        switch (linux.errno(result)) {
            .SUCCESS => {},
            .INTR => continue,
            else => {
                log.err(.{}, "Worker cleanup failed: operation=poll, errno={t}", .{linux.errno(result)});
                return error.SupervisorWorkerCleanupPollFailed;
            },
        }
        if (result == 0) continue;
        if (descriptors[0].revents & linux.POLL.IN == 0) {
            log.err(.{}, "Worker cleanup failed: operation=poll, events=0x{x}", .{descriptors[0].revents});
            return error.SupervisorWorkerCleanupPollFailed;
        }
        _ = try reapChild(process);
        return;
    }
}

fn waitForEvents(
    epoll_fd: std.posix.fd_t,
    events: []linux.epoll_event,
    timeout_ms: i32,
) !usize {
    while (true) {
        const result = linux.epoll_wait(
            epoll_fd,
            events.ptr,
            @intCast(events.len),
            timeout_ms,
        );
        switch (linux.errno(result)) {
            .SUCCESS => {
                return result;
            },
            .INTR => continue,
            else => {
                try checkSyscall("epoll_wait", result);
                unreachable;
            },
        }
    }
}

fn eventSource(event: linux.epoll_event) EventSource {
    return @enumFromInt(event.data.u64);
}

fn readEventCounter(descriptor: std.posix.fd_t) !?u64 {
    while (true) {
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
            .AGAIN => {
                return null;
            },
            .INTR => continue,
            else => {
                try checkSyscall("read_event_counter", result);
                unreachable;
            },
        }
    }
}

fn readSignal(descriptor: std.posix.fd_t) !?linux.signalfd_siginfo {
    while (true) {
        var signal: linux.signalfd_siginfo = undefined;
        const result = linux.read(
            descriptor,
            std.mem.asBytes(&signal).ptr,
            @sizeOf(linux.signalfd_siginfo),
        );
        if (linux.errno(result) == .INTR) continue;
        if (linux.errno(result) == .AGAIN) return null;
        try checkSyscall("read_signal", result);
        assert(result == @sizeOf(linux.signalfd_siginfo));
        return signal;
    }
}

fn setCloseOnExec(descriptor: std.posix.fd_t) !void {
    try checkSyscall("fcntl", linux.fcntl(descriptor, linux.F.SETFD, linux.FD_CLOEXEC));
}

fn signalChild(process: *const ChildProcess) !void {
    while (true) {
        const result = linux.pidfd_send_signal(
            process.pid_fd,
            .KILL,
            null,
            0,
        );
        switch (linux.errno(result)) {
            .SUCCESS, .SRCH => {
                return;
            },
            .INTR => continue,
            else => {
                log.err(.{}, "Worker termination failed: operation=pidfd_send_signal, pid_fd={d}, signal=KILL, errno={t}", .{ process.pid_fd, linux.errno(result) });
                return error.SupervisorWorkerKillFailed;
            },
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
            .SUCCESS => {
                return information;
            },
            .INTR => continue,
            else => {
                log.err(.{}, "Worker reap failed: operation=waitid, pid_fd={d}, errno={t}", .{ process.pid_fd, linux.errno(result) });
                return error.SupervisorWorkerReapFailed;
            },
        }
    }
}

fn sessionDeadline(configuration: CaptureOptions) u64 {
    // Model loading overlaps capture. After capture stops, at most three sealed
    // slots remain, each with a ten-second inference deadline and bounded exit.
    const duration_ns = (@as(u64, configuration.recording_seconds) + 35) * std.time.ns_per_s;
    return monotonicNanoseconds() + duration_ns;
}

fn monotonicNanoseconds() u64 {
    var timestamp: linux.timespec = undefined;
    // CLOCK_MONOTONIC and a valid stack pointer cannot produce an operational
    // failure; unlike event sources, neither depends on a descriptor/resource.
    assert(linux.errno(linux.clock_gettime(.MONOTONIC, &timestamp)) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);
    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

// Operation names are diagnostic data, not reasons to specialize these helpers.
fn descriptorFromResult(operation: []const u8, result: usize) !std.posix.fd_t {
    try checkSyscall(operation, result);
    return @intCast(result);
}

fn checkSyscall(operation: []const u8, result: usize) !void {
    const errno = linux.errno(result);
    if (errno == .SUCCESS) return;
    log.err(.{}, "Supervisor system call failed: operation={s}, errno={t}", .{ operation, errno });
    return error.SupervisorSystemCallFailed;
}

// Cleanup must retain its own diagnostics without replacing a primary error.
// Closing the descriptor also removes its epoll registration. Never retry close:
// Linux releases the descriptor even when close reports a late I/O error.
fn logCleanupSyscall(operation: []const u8, result: usize) void {
    const errno = linux.errno(result);
    if (errno != .SUCCESS) log.err(.{}, "Supervisor cleanup failed: operation={s}, errno={t}", .{ operation, errno });
}

fn closeDescriptor(descriptor: std.posix.fd_t) void {
    assert(descriptor >= 0);
    logCleanupSyscall("close", linux.close(descriptor));
}
