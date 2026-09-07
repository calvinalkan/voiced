//! The main thread owns recording lifecycle, deadlines, commands and desktop
//! delivery. Capture and transcription return typed results through fixed
//! mailboxes; only this loop releases their borrowed slots or starts another job.
//! An unacknowledged stop is fatal to the daemon, never permission to reuse data.

const std = @import("std");
const decimal = @import("decimal.zig");
const logging = @import("logging.zig");
const log = logging.scoped(.supervisor);
const notifications = @import("notifications.zig");
const transcript_file = @import("transcript_file.zig");
const audio_exchange = @import("audio_exchange.zig");
const capture_module = @import("capture.zig");
const model_cache = @import("model_cache.zig");
const models = @import("models");
const control_socket = @import("control_socket.zig");
const clipboard_module = @import("clipboard.zig");
const paste_keyboard = @import("paste_keyboard.zig");
const transcription_module = @import("transcription.zig");
const assert = std.debug.assert;
const linux = std.os.linux;

const AudioExchange = audio_exchange.AudioExchange;

const epoll_events_count_max: u32 = 16;

pub const ModelOptions = transcription_module.ModelOptions;

pub const recording_duration_seconds_default: u16 = 60 * 60;
pub const recording_duration_seconds_limit: u16 = std.math.maxInt(u16);

comptime {
    // The configured duration remains a u16, while the capture job
    // carries its corresponding 16 kHz sample target in a u32.
    assert(@as(u64, recording_duration_seconds_limit) *
        capture_module.sample_rate_hz <= std.math.maxInt(u32));
}

pub const CaptureOptions = struct {
    source: capture_module.Source = .default,
    recording_seconds: u16 = recording_duration_seconds_default,
    transcription: ModelOptions = .{},
};

pub const ServiceOptions = struct {
    log_level: logging.Level = .info,
    capture: CaptureOptions = .{},
    /// Zero releases the model runtime and compute pool after every recording; otherwise
    /// the runtime, worker group, weights, and workspace survive this idle window.
    model_keep_warm_seconds: u32 = 300,
    output: enum { desktop, clipboard, stdout } = .desktop,
    notification_mode: notifications.Mode = .errors,
    paste_key: paste_keyboard.Chord = .@"ctrl+shift+v",
    paste_settle_ms: u16 = 10,
    paste_key_gap_ms: u16 = 4,
};

const EventSource = enum(u64) {
    worker_notification = 1,
    deadline,
    service_signal,
    control_listener,
    control_client,
    clipboard,
    notification_bus,
};

const FinishReason = union(enum) {
    audio_completed,
    audio_stopped: ?notifications.Problem,
    pipeline_full: notifications.Problem,
    audio_failed_with_valid_prefix: notifications.Problem,
};

const TranscriptionRejection = struct {
    samples_count: u32,
    contains_activity: bool,
    no_speech_probability: f32,
    average_log_probability: f32,
};

const AbortReason = union(enum) {
    service_signal,
    user_cancelled,
    audio_failed: notifications.Problem,
    transcription_failed: notifications.Problem,
    deadline: enum { recording, model_load, transcription },
    transcript_limit: enum { recording_size, chunk_size, decoder_tokens, chunk_size_and_decoder_tokens },
    speech_detection_conflict: TranscriptionRejection,
    speech_unrecognized: TranscriptionRejection,
};

const SessionPhase = union(enum) {
    idle,
    active,
    finishing: FinishReason,
    delivering: struct {
        // One boundary reused after acquisition: delivery start, then clipboard
        // acquisition. The paste interval intentionally includes settle waits.
        boundary_monotonic_ns: u64,
        problem: ?notifications.Problem = null,
        paste: union(enum) { waiting, acquiring, settling: u64, sending, done } = .waiting,
    },
    aborting: AbortReason,
};

// Deadlines belong to operations, never to a second parallel state machine.
const AudioProgress = struct { callbacks_count: u64, deadline_monotonic_ns: u64 };
const CaptureOperation = union(enum) {
    idle,
    starting: u64,
    capturing: AudioProgress,
    stopping: u64,
    canceling: u64,

    fn deadlineMonotonicNs(operation: CaptureOperation) ?u64 {
        return switch (operation) {
            .idle => null,
            .capturing => |progress| progress.deadline_monotonic_ns,
            inline else => |deadline| deadline,
        };
    }
};
const SessionAudio = struct {
    worker: *capture_module.Worker,
    operation: CaptureOperation = .idle,
};
const TranscriptionState = union(enum) {
    absent,
    starting: u64,
    idle: ?u64,
    busy: struct { deadline_monotonic_ns: u64, slot_index: audio_exchange.SlotIndex },
    stopping: u64,

    fn deadlineMonotonicNs(state: TranscriptionState) ?u64 {
        return switch (state) {
            .absent => null,
            .idle => |deadline| deadline,
            .busy => |work| work.deadline_monotonic_ns,
            inline else => |deadline| deadline,
        };
    }
};

const TranscriptProgress = struct {
    accepted_chunks_count: u32,
    no_speech_chunks_count: u32,
    processed_samples_count: u64 = 0,
    compute_duration_ns: u64 = 0,
    storage_index: u1,
    bytes_count: u32,
};

const KeyboardState = union(enum) {
    disabled,
    unavailable: notifications.Problem,
    ready: paste_keyboard.Keyboard,
};

const Supervisor = struct {
    io: std.Io,
    options: *const ServiceOptions,
    audio: SessionAudio,

    epoll_fd: std.posix.fd_t,

    audio_exchange: *AudioExchange,

    transcription: TranscriptionState,
    model_worker: *transcription_module.Worker,
    phase: SessionPhase,
    session_deadline_monotonic_ns: u64,

    transcript: TranscriptProgress,
    recording_ordinal: u64 = 0,
    recording_requested_monotonic_ns: u64 = 0,
    recording_stop: ?struct {
        monotonic_ns: u64,
        origin: enum { command, capture_end },
    } = null,
    // The active transcript borrows one of these slices. Clipboard sources and
    // transfers retain the others; availability is derived from those owners.
    transcript_storage: []u8,
    clipboard: union(enum) { disconnected, connected: clipboard_module.Client } = .disconnected,
    clipboard_environment: clipboard_module.Environment,
    keyboard: KeyboardState = .disabled,
    notifications: notifications.Client = .{},
    service: struct {
        control: *control_socket.Server,
        transcript_directory: ?[]const u8,
        shutdown_requested: bool = false,
        pending_recording: ?u64 = null,
    },
};

/// `runService` accepts fixed-record commands on the instance's control socket.
/// Capture and model loading start together on the first recording.
/// Successful recordings retain the resident runtime for the idle window;
/// discarded recordings unload it before another recording can reset shared
/// storage. Native clipboard ownership and transfers borrow the completed
/// transcript and can outlive this recording and its model runtime.
pub fn runService(init: std.process.Init, options: ServiceOptions) !void {
    const configuration = &options.capture;

    // ── Own Signals And Event Sources ──

    var previous_pipe_action: linux.Sigaction = undefined;
    try checkSyscall("sigaction", linux.sigaction(.PIPE, &.{ .handler = .{ .handler = linux.SIG.IGN }, .mask = linux.sigemptyset(), .flags = 0 }, &previous_pipe_action));
    defer logCleanupSyscall("sigaction", linux.sigaction(.PIPE, &previous_pipe_action, null));
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
    const worker_event_fd = try descriptorFromResult("eventfd", linux.eventfd(
        0,
        linux.EFD.CLOEXEC | linux.EFD.NONBLOCK,
    ));
    defer closeDescriptor(worker_event_fd);

    const audio_storage = try std.heap.page_allocator.create(AudioExchange);
    defer std.heap.page_allocator.destroy(audio_storage);
    // Fault in writable PCM pages once, before any thread can borrow them.
    // Retouching arbitrary exchange bytes during capture setup would race the
    // supervisor's atomic metadata reads, even when storing the same byte back.
    const audio_bytes = std.mem.asBytes(audio_storage);
    var audio_offset: usize = 0;
    while (audio_offset < audio_bytes.len) : (audio_offset += std.heap.page_size_min) {
        const byte: *volatile u8 = &audio_bytes[audio_offset];
        byte.* = 0;
    }
    audio_exchange.initialize(audio_storage);

    // The exchange retains only three PCM slots regardless of session length.
    // Text is the one session-sized value: reserve 64 UTF-8 bytes per configured
    // audio second plus one complete 4 KiB mailbox result. The rate budget keeps
    // each one-hour transcript near 229 KiB; desktop output reserves two
    // generations so clipboard readers may outlive the next recording;
    // the result reserve lets any one valid mailbox publication fit.
    // Accepted text that exceeds the fixed budget ends the session explicitly
    // instead of growing memory or silently truncating output.
    const transcript_bytes_capacity = sessionTranscriptBytesCapacity(configuration.*);
    const transcript_bytes = try init.gpa.alloc(u8, transcript_bytes_capacity * @as(usize, if (options.output == .stdout) 1 else 2));
    defer init.gpa.free(transcript_bytes);

    var control = try control_socket.Server.open(
        init,
        epoll_fd,
        @intFromEnum(EventSource.control_listener),
        @intFromEnum(EventSource.control_client),
    );
    defer control.deinit(init.io);

    // Resolve the one state directory once. Desktop transcript saving and the
    // transcription worker's bounded last-failure capture borrow the same path.
    const state_directory = transcript_file.allocDirectoryPath(
        init.gpa,
        init.environ_map.get("XDG_STATE_HOME"),
        init.environ_map.get("HOME"),
        init.environ_map.get("VOICED_INSTANCE") orelse "",
    ) catch |err| unavailable: {
        log.err(.{}, "State storage unavailable: error={s}; transcript saving and failed-transcription capture disabled", .{@errorName(err)});
        break :unavailable null;
    };
    defer if (state_directory) |path| init.gpa.free(path);

    const installed_models_root = try models.allocInstalledRootPath(init);
    defer init.gpa.free(installed_models_root);
    const model_cache_root = try model_cache.allocCacheRootPath(init);
    defer init.gpa.free(model_cache_root);
    var capture_worker: capture_module.Worker = .{
        .mailbox = undefined,
        .exchange = audio_storage,
        .environment = .{
            .runtime_directory = init.environ_map.get("PIPEWIRE_RUNTIME_DIR") orelse init.environ_map.get("XDG_RUNTIME_DIR"),
            .remote = init.environ_map.get("PIPEWIRE_REMOTE") orelse "pipewire-0",
            .system_bus_address = init.environ_map.get("DBUS_SYSTEM_BUS_ADDRESS") orelse "unix:path=/run/dbus/system_bus_socket",
        },
    };
    try capture_worker.mailbox.init(worker_event_fd);
    defer capture_worker.mailbox.deinit();
    var model_worker: transcription_module.Worker = .{
        .mailbox = undefined,
        .context = .{
            .io = init.io,
            .allocator = init.gpa,
            .installed_models_root = installed_models_root,
            .cache_root = model_cache_root,
        },
        .audio = audio_storage,
        .capture_directory = state_directory,
    };
    try model_worker.mailbox.init(worker_event_fd);
    defer model_worker.mailbox.deinit();

    var supervisor: Supervisor = .{
        // initEmpty below establishes all notification metadata without the
        // buffer-containing aggregate template; see notifications.Client.initEmpty.
        .notifications = undefined,
        .io = init.io,
        .options = &options,
        .audio = .{ .worker = &capture_worker },
        .epoll_fd = epoll_fd,
        .audio_exchange = audio_storage,
        .model_worker = &model_worker,
        .transcription = .absent,
        .transcript_storage = transcript_bytes,
        .clipboard_environment = .{
            .runtime_directory = init.environ_map.get("XDG_RUNTIME_DIR"),
            .wayland_display = init.environ_map.get("WAYLAND_DISPLAY"),
            .display = init.environ_map.get("DISPLAY"),
            .xauthority = init.environ_map.get("XAUTHORITY"),
            .home = init.environ_map.get("HOME"),
        },
        .phase = .idle,
        .session_deadline_monotonic_ns = sessionDeadline(configuration.*),
        .service = .{
            .control = &control,
            .transcript_directory = if (options.output == .stdout) null else state_directory,
        },
        .transcript = .{
            .accepted_chunks_count = 0,
            .no_speech_chunks_count = 0,
            .storage_index = 0,
            .bytes_count = 0,
        },
    };

    // Initialize even when notifications are off: cleanup still uses the client.
    supervisor.notifications.initEmpty();

    try register(epoll_fd, worker_event_fd, .worker_notification);
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
    defer {
        if (supervisor.keyboard == .ready) supervisor.keyboard.ready.deinit();
        closeClipboard(&supervisor);
    }

    if (options.output == .desktop) openPasteKeyboard(&supervisor);
    if (options.output != .stdout) openClipboard(&supervisor);
    const capture_thread = try std.Thread.spawn(.{ .stack_size = 2 * 1024 * 1024 }, capture_module.Worker.run, .{&capture_worker});
    // Once threads can borrow this frame, an unrecoverable supervisor error must
    // exit the process before any defer frees their storage. Linux terminates
    // every thread; systemd Restart=on-failure supplies the fresh daemon.
    errdefer |err| {
        log.critical(.{ .recording_ordinal = supervisor.recording_ordinal }, "Supervisor stopped: error={s}; exiting with worker storage retained", .{@errorName(err)});
        linux.exit_group(1);
    }
    const model_thread = try std.Thread.spawn(.{ .stack_size = 2 * 1024 * 1024 }, transcription_module.Worker.run, .{&model_worker});

    while (true) {
        if (supervisor.service.shutdown_requested and supervisor.phase == .idle and
            supervisor.transcription == .absent and supervisor.audio.operation == .idle and
            !supervisor.service.control.hasPendingReplies())
        {
            break;
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
            if (eventSource(event) != .worker_notification) continue;
            _ = try readEventCounter(worker_event_fd);
        }

        // Completion releases each worker's borrows before dispatch can reuse
        // audio or transcript storage. Wake counts carry no independent state.
        try drainAudioResult(&supervisor);
        try drainTranscriptionResult(&supervisor);
        try dispatchPublishedAudio(&supervisor);

        try observePipeWireProgress(&supervisor);

        for (events[0..events_count]) |event| {
            switch (eventSource(event)) {
                .deadline => {
                    _ = try readEventCounter(timer_fd);
                    try applyExpiredDeadlines(&supervisor);
                },
                .service_signal => {
                    if (try readSignal(signal_fd) == null) continue;
                    supervisor.service.shutdown_requested = true;
                    supervisor.service.pending_recording = null;
                    if (supervisor.phase == .idle) {
                        requestTranscriptionShutdown(&supervisor);
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
        // when a worker finishes between draining the wake counter and its mailbox.
        try dispatchPublishedAudio(&supervisor);
        try maintainWorkersAndSession(&supervisor);
        // Finish and start desktop delivery before the next epoll wait. A ready
        // clipboard has no pending I/O or deadline until publication begins;
        // sleeping between these operations can strand text until another command.
        if (sessionIsComplete(&supervisor)) {
            try finishSession(&supervisor);
            if (supervisor.phase != .delivering) enterIdle(&supervisor);
        }
        try advanceOutput(&supervisor);
        {
            const service = &supervisor.service;
            if (service.pending_recording != null and supervisor.phase == .idle and availableTranscript(&supervisor) != null and
                supervisor.audio.operation == .idle and
                (supervisor.transcription == .absent or supervisor.transcription == .idle))
            {
                supervisor.recording_requested_monotonic_ns = service.pending_recording.?;
                service.pending_recording = null;
                assert(!captureActive(&supervisor));
                // PCM payload is not cleared. Reset logical lengths only when
                // capture acknowledged idle and the model awaits a command.
                supervisor.recording_ordinal += 1;
                supervisor.notifications.resetSuppression();
                supervisor.recording_stop = null;
                audio_exchange.initialize(supervisor.audio_exchange);
                supervisor.transcript = .{ .accepted_chunks_count = 0, .no_speech_chunks_count = 0, .storage_index = availableTranscript(&supervisor).?, .bytes_count = 0 };
                supervisor.phase = .active;
                if (supervisor.transcription == .idle) supervisor.transcription.idle = null;
                supervisor.session_deadline_monotonic_ns = sessionDeadline(configuration.*);
                log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Recording requested: recording_ordinal={d}", .{supervisor.recording_ordinal});
                if (supervisor.transcription == .absent) startTranscription(&supervisor);
                startAudio(&supervisor);
            }
        }
    }
    capture_worker.mailbox.shutdown();
    model_worker.mailbox.shutdown();
    const join_deadline = monotonicNanoseconds() + std.time.ns_per_s;
    while (!capture_worker.mailbox.exited() or !model_worker.mailbox.exited()) {
        const remaining = join_deadline -| monotonicNanoseconds();
        if (remaining == 0) return error.WorkerJoinDeadlineExceeded;
        var fd = [_]linux.pollfd{.{ .fd = worker_event_fd, .events = linux.POLL.IN, .revents = 0 }};
        const result = linux.poll(&fd, 1, @intCast((remaining + std.time.ns_per_ms - 1) / std.time.ns_per_ms));
        if (linux.errno(result) == .INTR) continue;
        try checkSyscall("worker_join_poll", result);
        _ = try readEventCounter(worker_event_fd);
    }
    capture_thread.join();
    model_thread.join();
}

// PERFORMANCE: Keep request dispatch out of `runService`. Inlining this
// user-paced branch and response path expands the event loop's ReleaseSafe
// machine code; one call per control request is outside capture and inference.
noinline fn dispatchControl(supervisor: *Supervisor, client_index: usize, request: control_socket.Request) !void {
    const request_received_monotonic_ns = monotonicNanoseconds();
    const service = &supervisor.service;
    var ignored = false;
    switch (request.cmd) {
        .record => {
            if (service.shutdown_requested or supervisor.phase == .finishing or supervisor.phase == .delivering or supervisor.phase == .aborting) {
                ignored = true;
            } else if (supervisor.phase == .active) {
                if (request.toggle) try requestAudioFinish(supervisor, request_received_monotonic_ns) else ignored = true;
            } else if (service.pending_recording != null) {
                if (request.toggle) service.pending_recording = null else ignored = true;
            } else {
                assert(supervisor.phase == .idle);
                service.pending_recording = request_received_monotonic_ns;
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
                requestTranscriptionShutdown(supervisor);
            } else try beginAbort(supervisor, .service_signal);
        },
        .status => {},
    }
    if (ignored) log.debug(.{ .recording_ordinal = supervisor.recording_ordinal }, "Command ignored: recording_ordinal={d}, command={s}, phase={s}", .{ supervisor.recording_ordinal, @tagName(request.cmd), @tagName(supervisor.phase) });
    const phase: control_socket.Phase = switch (supervisor.phase) {
        .idle => if (service.pending_recording != null) .capturing else .idle,
        .active => .capturing,
        .finishing => if (captureActive(supervisor)) .stopping else .transcribing,
        .delivering => .delivering,
        .aborting => .stopping,
    };
    const model_state: control_socket.ModelState = switch (supervisor.transcription) {
        .absent => .absent,
        .starting => .loading,
        .idle, .busy => .warm,
        .stopping => .unloading,
    };
    service.control.respond(client_index, .{
        .ignored = ignored,
        .phase = phase,
        .model = model_state,
        .session_id = supervisor.recording_ordinal,
        .model_keep_warm_seconds = supervisor.options.model_keep_warm_seconds,
    });
}

fn requestAudioFinish(supervisor: *Supervisor, request_received_monotonic_ns: u64) !void {
    assert(supervisor.phase == .active);
    assert(supervisor.recording_stop == null);
    supervisor.recording_stop = .{ .monotonic_ns = request_received_monotonic_ns, .origin = .command };
    log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Recording stop requested: recording_ordinal={d}, recording_stop_origin=command", .{supervisor.recording_ordinal});
    supervisor.phase = .{ .finishing = .{ .audio_stopped = null } };
    switch (supervisor.audio.operation) {
        .starting, .capturing => {
            supervisor.audio.worker.requestStop(.stop);
            supervisor.audio.operation = .{ .stopping = monotonicNanoseconds() + std.time.ns_per_s };
        },
        .idle, .stopping, .canceling => {},
    }
}

fn sessionTranscriptBytesCapacity(configuration: CaptureOptions) usize {
    const bytes_per_recording_second_max: u64 = 64;
    const result_bytes_reserve: u64 =
        transcription_module.result_bytes_capacity;

    const bytes_capacity = @as(u64, configuration.recording_seconds) *
        bytes_per_recording_second_max + result_bytes_reserve;
    assert(bytes_capacity > 0);
    assert(bytes_capacity <= std.math.maxInt(u32));
    return @intCast(bytes_capacity);
}

fn startAudio(supervisor: *Supervisor) void {
    const audio = &supervisor.audio;
    assert(audio.operation == .idle);
    audio.worker.start(.{
        .source = supervisor.options.capture.source,
        .recording_samples_target = @as(u32, supervisor.options.capture.recording_seconds) * capture_module.sample_rate_hz,
    });
    audio.operation = .{ .starting = monotonicNanoseconds() + 3 * std.time.ns_per_s };
}

fn startTranscription(supervisor: *Supervisor) void {
    assert(supervisor.transcription == .absent);
    supervisor.model_worker.submit(.{ .prepare = .{
        .recording_ordinal = supervisor.recording_ordinal,
        .model = supervisor.options.capture.transcription,
    } });
    supervisor.transcription = .{ .starting = monotonicNanoseconds() + 15 * std.time.ns_per_s };
}

fn drainAudioResult(supervisor: *Supervisor) !void {
    while (supervisor.audio.worker.mailbox.receive()) |worker_report| {
        supervisor.audio.operation = .idle;
        switch (worker_report) {
            .ok => |success| {
                observeCaptureEnd(supervisor);
                logCaptureReport(supervisor.recording_ordinal, @tagName(success.end), &success.report, null);
                switch (success.end) {
                    .completed => if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = .audio_completed };
                    },
                    .stopped => if (supervisor.phase == .active) {
                        supervisor.phase = .{ .finishing = .{ .audio_stopped = null } };
                    },
                    .cancelled => assert(supervisor.phase == .aborting or supervisor.phase == .finishing),
                }
            },
            .err => |err| {
                const problem = audioProblem(err);
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Audio service error: recording_ordinal={d}, kind={t}", .{ supervisor.recording_ordinal, std.meta.activeTag(err) });
                switch (err) {
                    .source_not_found, .source_ambiguous, .setup => |detail| {
                        const cause = captureFailureCause(detail.cause);
                        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Capture setup error: recording_ordinal={d}, kind={t}, stage={t}, error_domain={s}, error_code={d}, pipewire_server_version=\"{f}\", client_node_version_advertised={d}, client_node_version_selected={d}, detail=\"{f}\"", .{ supervisor.recording_ordinal, std.meta.activeTag(err), detail.stage, cause.domain, cause.code, std.zig.fmtString(detail.pipewire_version[0..detail.pipewire_version_size]), detail.client_node_version_advertised, detail.client_node_version_selected, std.zig.fmtString(detail.message[0..detail.message_size]) });
                        if (std.meta.activeTag(err) == .source_ambiguous) log.err(.{}, "Microphone selection: sources_expected_count=1; replace microphone_serial with a microphone_node from the candidate list", .{});
                        try beginAbort(supervisor, .{ .audio_failed = problem });
                        continue;
                    },
                    .source_connection_lost,
                    .source_changed,
                    .pipeline_full,
                    .unexpected_format,
                    .timeline_discontinuity,
                    .stream_error,
                    .stream_disconnected,
                    .invalid_buffer,
                    .corrupted_buffer,
                    => |runtime| {
                        observeCaptureEnd(supervisor);
                        logCaptureReport(supervisor.recording_ordinal, @tagName(std.meta.activeTag(err)), &runtime.report, &runtime.detail);
                        if (supervisor.phase == .active) {
                            if (std.meta.activeTag(err) == .pipeline_full) {
                                supervisor.phase = .{ .finishing = .{ .pipeline_full = problem } };
                            } else if (runtime.report.published_samples_count == 0) {
                                try beginAbort(supervisor, .{ .audio_failed = problem });
                            } else {
                                // PipeWire reports preserve every complete block before a
                                // disconnect, malformed buffer, or timeline failure. Drain
                                // and transcribe that valid prefix, but retain the failure
                                // outcome so policy cannot mistake it for completion.
                                supervisor.phase = .{ .finishing = .{ .audio_failed_with_valid_prefix = problem } };
                            }
                        } else if (supervisor.phase == .finishing and supervisor.phase.finishing == .audio_stopped) {
                            supervisor.phase.finishing.audio_stopped = problem;
                        }
                    },
                }
            },
        }
    }
}

fn observeCaptureEnd(supervisor: *Supervisor) void {
    if (supervisor.recording_stop != null or supervisor.phase != .active) return;
    // Automatic and source-driven stops occur inside capture. This observer
    // must not masquerade as command-to-done time.
    supervisor.recording_stop = .{ .monotonic_ns = monotonicNanoseconds(), .origin = .capture_end };
    log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Recording stop observed: recording_ordinal={d}, recording_stop_origin=capture_end", .{supervisor.recording_ordinal});
}

fn observePipeWireProgress(supervisor: *Supervisor) !void {
    const audio = &supervisor.audio;
    const capture_is_starting = audio.operation == .starting;
    const callbacks_count_previous = switch (audio.operation) {
        .starting => 0,
        .capturing => |progress| progress.callbacks_count,
        .idle, .stopping, .canceling => {
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
        log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Capture started: recording_ordinal={d}, capture_start_duration_ms={f}", .{
            supervisor.recording_ordinal,
            decimal.fmt(@as(f64, @floatFromInt(monotonicNanoseconds() - supervisor.recording_requested_monotonic_ns)) / std.time.ns_per_ms, 3),
        });
    }

    audio.operation = .{ .capturing = .{
        .callbacks_count = callbacks_count,
        .deadline_monotonic_ns = monotonicNanoseconds() +
            2 * std.time.ns_per_s,
    } };
}

fn drainTranscriptionResult(supervisor: *Supervisor) !void {
    const report = supervisor.model_worker.mailbox.receive() orelse return;
    switch (report) {
        .err => |err| {
            supervisor.transcription = if (err == .model_load) .absent else .{ .idle = null };
            try beginAbort(supervisor, .{ .transcription_failed = transcriptionProblem(err) });
        },
        .ok => |event| switch (event) {
            .ready => |ready| {
                assert(supervisor.transcription == .starting or supervisor.transcription == .stopping);
                supervisor.transcription = .{ .idle = null };
                log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Model prepared: recording_ordinal={d}, model_prepare_duration_ms={f}", .{ supervisor.recording_ordinal, decimal.fmt(@as(f64, @floatFromInt(ready.model_prepare_duration_ns)) / std.time.ns_per_ms, 3) });
            },
            .result => |result| {
                const previous = supervisor.transcription;
                supervisor.transcription = .{ .idle = null };
                if (supervisor.phase != .aborting) {
                    assert(previous == .busy);
                    try consumeTranscript(supervisor, previous.busy.slot_index, result);
                }
            },
            .cancelled => {
                assert(supervisor.transcription == .stopping);
                supervisor.transcription = .{ .idle = null };
            },
            .stopped => {
                assert(supervisor.transcription == .stopping);
                supervisor.transcription = .absent;
            },
        },
    }
}

fn transcriptionProblem(err: transcription_module.Error) notifications.Problem {
    return switch (err) {
        .model_load => .model_load_failed,
        .feature_extraction, .inference, .text_decode => .transcription_failed,
    };
}

fn dispatchPublishedAudio(supervisor: *Supervisor) !void {
    if (supervisor.phase != .active and supervisor.phase != .finishing) return;
    if (supervisor.transcription != .idle) return;
    const chunk_ordinal = processedChunksCount(&supervisor.transcript);
    const slot_index = nextPublishedAudio(supervisor) orelse return;
    supervisor.model_worker.submit(.{ .transcribe = .{
        .recording_ordinal = supervisor.recording_ordinal,
        .chunk_ordinal = chunk_ordinal,
        .slot_index = slot_index,
    } });
    supervisor.transcription = .{ .busy = .{
        .deadline_monotonic_ns = monotonicNanoseconds() + 10 * std.time.ns_per_s,
        .slot_index = slot_index,
    } };
}

fn consumeTranscript(supervisor: *Supervisor, slot_index: audio_exchange.SlotIndex, timings: transcription_module.ResultReport) !void {
    const slot = &supervisor.audio_exchange.slots[slot_index.arrayIndex()];
    const result = timings.transcript;
    const transcript_bytes = transcriptBytes(supervisor);
    const chunk_ordinal = processedChunksCount(&supervisor.transcript);
    assert(audio_exchange.acquireSlot(slot) != null);

    const disposition = transcription_module.classifyResult(result);
    assert(supervisor.transcript.bytes_count <= transcript_bytes.len);
    const output_fits = result.bytes.len <= transcript_bytes.len - supervisor.transcript.bytes_count;
    const compute_duration_ns = timings.features_duration_ns + timings.inference_duration_ns;
    supervisor.transcript.processed_samples_count += result.samples_count;
    supervisor.transcript.compute_duration_ns += compute_duration_ns;

    if (logging.enabled(.info)) {
        var features_buffer: [32]u8 = undefined;
        var inference_buffer: [32]u8 = undefined;
        var processing_buffer: [32]u8 = undefined;
        var speed_buffer: [32]u8 = undefined;
        log.info(
            .{ .recording_ordinal = supervisor.recording_ordinal },
            "Transcription chunk: recording_ordinal={d}, chunk_ordinal={d}, audio_duration_seconds={f}, audio_samples_count={d}, " ++
                "features_duration_ms={s}, inference_duration_ms={s}, transcription_compute_duration_ms={s}, transcription_compute_speed_ratio={s}, " ++
                "transcript_size={d}, disposition={s}, chunk_limit={t}, recording_size_exceeded={}, activity_observed={}, text_empty={}, no_speech_probability={f}, no_speech_inactive_probability_min={f}, average_log_probability={f}\n",
            .{
                supervisor.recording_ordinal,
                chunk_ordinal,
                decimal.fmt(@as(f64, @floatFromInt(result.samples_count)) / capture_module.sample_rate_hz, 3),
                result.samples_count,
                formatDurationMilliseconds(&features_buffer, timings.features_duration_ns),
                formatDurationMilliseconds(&inference_buffer, timings.inference_duration_ns),
                formatDurationMilliseconds(&processing_buffer, compute_duration_ns),
                formatComputeSpeedRatio(&speed_buffer, result.samples_count, compute_duration_ns),
                result.bytes.len,
                @tagName(disposition),
                result.limit,
                !output_fits,
                result.contains_activity,
                std.mem.trim(u8, result.bytes, " \t\r\n").len == 0,
                decimal.fmt(result.no_speech_probability, 6),
                decimal.fmt(transcription_module.no_activity_no_speech_probability_reject_min, 2),
                decimal.fmt(result.average_log_probability, 6),
            },
        );
    }
    switch (disposition) {
        .accepted, .partial => {
            const available = transcript_bytes.len - supervisor.transcript.bytes_count;
            const text = if (output_fits) result.bytes else transcription_module.utf8Prefix(result.bytes[0..available]);
            if (!output_fits) log.warn(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript capacity reached: recording_ordinal={d}, chunk_ordinal={d}, transcript_size_max={d}, transcript_committed_size={d}, chunk_transcript_size={d}, chunk_retained_size={d}", .{
                supervisor.recording_ordinal,      chunk_ordinal,    transcript_bytes.len,
                supervisor.transcript.bytes_count, result.bytes.len, text.len,
            });
            @memcpy(transcript_bytes[supervisor.transcript.bytes_count..][0..text.len], text);
            supervisor.transcript.bytes_count += @intCast(text.len);
            supervisor.transcript.accepted_chunks_count += 1;
        },
        .no_speech => {
            supervisor.transcript.no_speech_chunks_count += 1;
        },
        .speech_detection_conflict, .speech_unrecognized => {
            const rejection: TranscriptionRejection = .{
                .samples_count = result.samples_count,
                .contains_activity = result.contains_activity,
                .no_speech_probability = result.no_speech_probability,
                .average_log_probability = result.average_log_probability,
            };
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
    if (result.limit != .none or (disposition == .accepted and !output_fits)) {
        // Stop further work while retaining copied text until delivery finishes.
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

fn nextPublishedAudio(supervisor: *Supervisor) ?audio_exchange.SlotIndex {
    const index = audio_exchange.SlotIndex.fromPublicationOrdinal(
        processedChunksCount(&supervisor.transcript),
    );
    return if (audio_exchange.acquireSlot(
        &supervisor.audio_exchange.slots[index.arrayIndex()],
    ) != null) index else null;
}

fn maintainWorkersAndSession(supervisor: *Supervisor) !void {
    if (supervisor.phase == .aborting) {
        requestTranscriptionShutdown(supervisor);
        return;
    }
    if (supervisor.phase != .active and supervisor.phase != .finishing) return;
    const count = publishedSlots(supervisor);
    if (supervisor.transcription == .absent and count > 0) {
        startTranscription(supervisor);
        return;
    }
    if (supervisor.phase == .finishing and !captureActive(supervisor) and count == 0 and
        (supervisor.service.shutdown_requested or supervisor.options.model_keep_warm_seconds == 0))
        requestTranscriptionShutdown(supervisor);
}

fn requestTranscriptionShutdown(supervisor: *Supervisor) void {
    if (supervisor.transcription != .idle) return;
    supervisor.model_worker.submit(.unload);
    supervisor.transcription = .{ .stopping = monotonicNanoseconds() + std.time.ns_per_s };
}

fn finishValidAudioPrefixOrDiscard(supervisor: *Supervisor, problem: notifications.Problem) !void {
    assert(supervisor.phase == .active);

    if (processedChunksCount(&supervisor.transcript) > 0 or publishedSlots(supervisor) > 0) {
        supervisor.phase = .{ .finishing = .{ .audio_failed_with_valid_prefix = problem } };
    } else try beginAbort(supervisor, .{ .audio_failed = problem });
}

fn beginAbort(supervisor: *Supervisor, reason: AbortReason) !void {
    if (supervisor.phase == .delivering) {
        // Accepted text may already be on the clipboard or pasted. Cancellation
        // stops further output; it cannot undo those external effects.
        cancelDelivery(supervisor);
        // Closing the clipboard also closes every transfer before text can be reused.
        // Explicit cancellation preserves the previous saved transcript.
        if (reason == .user_cancelled or reason == .service_signal) supervisor.transcript.bytes_count = 0;
        if (supervisor.service.shutdown_requested) requestTranscriptionShutdown(supervisor);
        return;
    }
    if (supervisor.phase == .aborting) {
        // An explicit cancel/shutdown also cancels a pending partial delivery.
        if (reason == .user_cancelled or reason == .service_signal) supervisor.phase.aborting = reason;
        return;
    }
    supervisor.phase = .{ .aborting = reason };
    supervisor.model_worker.requestCancellation();
    requestAudioDiscard(supervisor);
    switch (supervisor.transcription) {
        .starting, .busy => supervisor.transcription = .{ .stopping = monotonicNanoseconds() + std.time.ns_per_s },
        .idle => requestTranscriptionShutdown(supervisor),
        .absent, .stopping => {},
    }
}

fn requestAudioDiscard(supervisor: *Supervisor) void {
    switch (supervisor.audio.operation) {
        .idle, .canceling => {},
        .starting, .capturing, .stopping => {
            supervisor.audio.worker.requestStop(.cancel);
            supervisor.audio.operation = .{ .canceling = monotonicNanoseconds() + std.time.ns_per_s };
        },
    }
}

// PERFORMANCE: Keep deadline policy out of `runService`. Inlining this
// error-heavy state machine expands the event loop's ReleaseSafe machine code;
// one call after a timer expiration is negligible beside the wake and actions.
noinline fn applyExpiredDeadlines(supervisor: *Supervisor) !void {
    const now = monotonicNanoseconds();
    if (supervisor.phase != .idle and supervisor.phase != .aborting and supervisor.phase != .delivering and now >= supervisor.session_deadline_monotonic_ns) {
        try beginAbort(supervisor, .{ .deadline = .recording });
    }
    if (supervisor.audio.operation.deadlineMonotonicNs()) |deadline| {
        if (now >= deadline) {
            log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Capture deadline exceeded: recording_ordinal={d}, stage={t}", .{ supervisor.recording_ordinal, std.meta.activeTag(supervisor.audio.operation) });
            switch (supervisor.audio.operation) {
                .starting, .capturing => {
                    const problem: notifications.Problem = if (supervisor.audio.operation == .starting) .audio_start_timed_out else .audio_stalled;
                    requestAudioDiscard(supervisor);
                    if (supervisor.phase == .active) try finishValidAudioPrefixOrDiscard(supervisor, problem);
                },
                .stopping, .canceling => return error.CaptureCancellationDeadlineExceeded,
                .idle => unreachable,
            }
        }
    }
    if (supervisor.transcription.deadlineMonotonicNs()) |deadline| {
        if (now >= deadline) switch (supervisor.transcription) {
            .starting, .busy => {
                const loading = supervisor.transcription == .starting;
                log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription deadline exceeded: recording_ordinal={d}, stage={t}", .{ supervisor.recording_ordinal, std.meta.activeTag(supervisor.transcription) });
                try beginAbort(supervisor, .{ .deadline = if (loading) .model_load else .transcription });
            },
            .stopping => return error.TranscriptionStopDeadlineExceeded,
            .idle => |warm_deadline| {
                assert(warm_deadline != null);
                if (supervisor.service.pending_recording == null) requestTranscriptionShutdown(supervisor);
            },
            .absent => unreachable,
        };
    }
}

fn armNearestDeadline(
    supervisor: *Supervisor,
    timer_fd: std.posix.fd_t,
) !void {
    var nearest_deadline_monotonic_ns: u64 = switch (supervisor.phase) {
        .idle, .aborting, .delivering => std.math.maxInt(u64),
        else => supervisor.session_deadline_monotonic_ns,
    };
    nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, supervisor.notifications.deadline_monotonic_ns);
    if (supervisor.service.control.deadline()) |deadline| {
        nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);
    }
    if (supervisor.clipboard == .connected)
        nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, supervisor.clipboard.connected.deadline());
    if (supervisor.phase == .delivering and supervisor.phase.delivering.paste == .settling)
        nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, supervisor.phase.delivering.paste.settling);
    if (supervisor.keyboard == .ready) {
        const keyboard = &supervisor.keyboard.ready;
        if (keyboard.deadlineMonotonicNs()) |deadline| nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);
    }
    if (supervisor.audio.operation.deadlineMonotonicNs()) |deadline| nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);
    if (supervisor.transcription.deadlineMonotonicNs()) |deadline| nearest_deadline_monotonic_ns = @min(nearest_deadline_monotonic_ns, deadline);

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
    if (supervisor.phase == .active or supervisor.phase == .idle or supervisor.phase == .delivering or captureActive(supervisor)) {
        return false;
    }
    if (supervisor.transcription == .absent) {
        return publishedSlots(supervisor) == 0 or supervisor.phase == .aborting;
    }
    return supervisor.phase == .finishing and
        !supervisor.service.shutdown_requested and supervisor.options.model_keep_warm_seconds > 0 and
        supervisor.transcription == .idle and publishedSlots(supervisor) == 0;
}

fn finishSession(supervisor: *Supervisor) !void {
    assert(sessionIsComplete(supervisor));

    var processing_buffer: [32]u8 = undefined;
    var speed_buffer: [32]u8 = undefined;
    var stop_buffer: [32]u8 = undefined;
    const stop_origin = if (supervisor.recording_stop) |stop| @tagName(stop.origin) else "unavailable";
    const audio_seconds = @as(f64, @floatFromInt(supervisor.transcript.processed_samples_count)) / capture_module.sample_rate_hz;
    const processing = formatDurationMilliseconds(&processing_buffer, supervisor.transcript.compute_duration_ns);
    const realtime_speed = formatComputeSpeedRatio(&speed_buffer, supervisor.transcript.processed_samples_count, supervisor.transcript.compute_duration_ns);

    switch (supervisor.phase) {
        .active, .idle, .delivering => unreachable,
        .finishing => |reason| {
            assert(publishedSlots(supervisor) == 0);
            const outcome = if (supervisor.transcript.accepted_chunks_count == 0 and supervisor.transcript.no_speech_chunks_count > 0)
                "no_speech"
            else
                @tagName(std.meta.activeTag(reason));
            try finishTranscription(supervisor, outcome, switch (reason) {
                .audio_failed_with_valid_prefix, .pipeline_full => |problem| problem,
                .audio_stopped => |problem| problem,
                .audio_completed => null,
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
            // Neither thread can still publish into cancelled storage. The next
            // service recording resets both exchanges before either role starts.
            log.write(if (reason == .user_cancelled or reason == .service_signal) .info else .warn, .{ .recording_ordinal = supervisor.recording_ordinal }, "Recording discarded: recording_ordinal={d}, reason={s}, audio_duration_seconds={f}, transcription_compute_duration_ms={s}, transcription_compute_speed_ratio={s}, transcript_size=0, recording_stop_origin={s}, recording_stop_elapsed_ms={s}", .{
                supervisor.recording_ordinal, @tagName(std.meta.activeTag(reason)),                             decimal.fmt(audio_seconds, 3), processing, realtime_speed,
                stop_origin,                  formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
            });
            switch (reason) {
                .service_signal, .user_cancelled => {},
                .audio_failed, .transcription_failed => |problem| supervisor.notifications.show(problem),
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
                            "activity_observed={}, no_speech_probability={f}, " ++
                            "no_speech_inactive_probability_min={f}, no_speech_active_probability_min={f}, " ++
                            "average_log_probability={f}\n",
                        .{
                            supervisor.recording_ordinal,
                            supervisor.options.capture.transcription.model.name(),
                            processedChunksCount(&supervisor.transcript),
                            rejection.samples_count,
                            rejection.contains_activity,
                            decimal.fmt(rejection.no_speech_probability, 6),
                            decimal.fmt(transcription_module.no_activity_no_speech_probability_reject_min, 2),
                            decimal.fmt(transcription_module.active_no_speech_probability_conflict_min, 2),
                            decimal.fmt(rejection.average_log_probability, 6),
                        },
                    );
                },
                .transcript_limit => unreachable,
                .service_signal,
                .user_cancelled,
                .audio_failed,
                .transcription_failed,
                .deadline,
                => {},
            }
        },
    }
}

// Both normal completion and a capacity stop deliver the one accumulated buffer.
// A limit may leave sealed audio unprocessed; capture has released its stream
// and no worker is consuming a slot before this call.
fn finishTranscription(supervisor: *Supervisor, outcome_name: []const u8, problem: ?notifications.Problem, limited: bool) !void {
    var processing_buffer: [32]u8 = undefined;
    var speed_buffer: [32]u8 = undefined;
    var stop_buffer: [32]u8 = undefined;
    const stop_origin = if (supervisor.recording_stop) |stop| @tagName(stop.origin) else "unavailable";
    const audio_seconds = @as(f64, @floatFromInt(supervisor.transcript.processed_samples_count)) / capture_module.sample_rate_hz;
    const processing = formatDurationMilliseconds(&processing_buffer, supervisor.transcript.compute_duration_ns);
    const realtime_speed = formatComputeSpeedRatio(&speed_buffer, supervisor.transcript.processed_samples_count, supervisor.transcript.compute_duration_ns);

    const transcript = std.mem.trim(
        u8,
        transcriptBytes(supervisor)[0..supervisor.transcript.bytes_count],
        " \t\r\n",
    );
    log.write(if (limited) .warn else .info, .{ .recording_ordinal = supervisor.recording_ordinal }, "Transcription complete: recording_ordinal={d}, outcome={s}, chunks_accepted_count={d}, chunks_no_speech_count={d}, chunks_count={d}, audio_duration_seconds={f}, transcription_compute_duration_ms={s}, transcription_compute_speed_ratio={s}, transcript_size={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}", .{
        supervisor.recording_ordinal,
        outcome_name,
        supervisor.transcript.accepted_chunks_count,
        supervisor.transcript.no_speech_chunks_count,
        processedChunksCount(&supervisor.transcript),
        decimal.fmt(audio_seconds, 3),
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
    if (supervisor.options.output == .stdout) {
        try printOutput(supervisor.io, transcript);
        log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript written: recording_ordinal={d}, transcript_size={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}", .{
            supervisor.recording_ordinal, transcript.len, stop_origin, formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
        });
        if (problem) |cause| supervisor.notifications.show(cause);
    } else beginDelivery(supervisor, problem);
}

fn formatDurationMilliseconds(buffer: *[32]u8, elapsed_ns: ?u64) []const u8 {
    const elapsed = elapsed_ns orelse return "unavailable";
    return std.fmt.bufPrint(buffer, "{f}", .{decimal.fmt(@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_ms, 3)}) catch unreachable;
}

fn formatComputeSpeedRatio(buffer: *[32]u8, samples_count: u64, compute_duration_ns: ?u64) []const u8 {
    const elapsed_ns = compute_duration_ns orelse return "unavailable";
    if (elapsed_ns == 0 or samples_count == 0) return "unavailable";
    const audio_seconds = @as(f64, @floatFromInt(samples_count)) / capture_module.sample_rate_hz;
    const processing_seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    return std.fmt.bufPrint(buffer, "{f}", .{decimal.fmt(audio_seconds / processing_seconds, 2)}) catch unreachable;
}

// Explicit diagnostic mode: synchronous stdout after capture/inference drain.
fn printOutput(io: std.Io, transcript: []const u8) !void {
    const stdout = std.Io.File.stdout();
    try stdout.writeStreamingAll(io, transcript);
    try stdout.writeStreamingAll(io, "\n");
}

fn beginDelivery(supervisor: *Supervisor, problem: ?notifications.Problem) void {
    if (supervisor.options.output == .desktop and supervisor.keyboard != .ready) openPasteKeyboard(supervisor);
    supervisor.phase = .{ .delivering = .{ .boundary_monotonic_ns = monotonicNanoseconds(), .problem = problem } };
    if (supervisor.clipboard == .disconnected) openClipboard(supervisor);
}

fn openClipboard(supervisor: *Supervisor) void {
    supervisor.clipboard = .{ .connected = undefined };
    switch (supervisor.clipboard.connected.init(supervisor.epoll_fd, @intFromEnum(EventSource.clipboard), supervisor.clipboard_environment, false, monotonicNanoseconds())) {
        .ok => {},
        .err => |err| clipboardError(supervisor, err),
    }
}

fn advanceOutput(supervisor: *Supervisor) !void {
    const now_ns = monotonicNanoseconds();
    var stop_buffer: [32]u8 = undefined;
    const stop_origin = if (supervisor.recording_stop) |stop| @tagName(stop.origin) else "unavailable";
    const text = std.mem.trim(u8, transcriptBytes(supervisor)[0..supervisor.transcript.bytes_count], " \t\r\n");
    if (supervisor.service.shutdown_requested) closeClipboard(supervisor);
    if (supervisor.clipboard == .connected) {
        const event = switch (supervisor.clipboard.connected.advance(now_ns)) {
            .ok => |event| event,
            .err => |err| failed: {
                clipboardError(supervisor, err);
                break :failed clipboard_module.Event.none;
            },
        };
        if (event == .acquired and supervisor.phase == .delivering) {
            const delivery = &supervisor.phase.delivering;
            assert(event.acquired == transcriptId(supervisor));
            log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Clipboard acquired: recording_ordinal={d}, transcript_size={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}, clipboard_backend={s}, clipboard_mode={s}, clipboard_acquire_duration_ms={f}", .{
                supervisor.recording_ordinal,
                text.len,
                stop_origin,
                formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
                supervisor.clipboard.connected.backendName(),
                @tagName(supervisor.clipboard.connected.mode),
                decimal.fmt(@as(f64, @floatFromInt(now_ns - delivery.boundary_monotonic_ns)) / std.time.ns_per_ms, 3),
            });
            delivery.boundary_monotonic_ns = now_ns;
            if (supervisor.options.output == .desktop and supervisor.keyboard != .ready)
                delivery.problem = switch (supervisor.keyboard) {
                    .unavailable => |problem| problem,
                    .disabled, .ready => .paste_failed,
                };
            delivery.paste = if (supervisor.options.output == .desktop and supervisor.keyboard == .ready)
                .{ .settling = @max(now_ns + @as(u64, supervisor.options.paste_settle_ms) * std.time.ns_per_ms, supervisor.keyboard.ready.usable_after_ns) }
            else
                .done;
        }
    }
    if (supervisor.phase != .delivering) return;
    const delivery = &supervisor.phase.delivering;
    if (supervisor.clipboard == .connected and delivery.paste == .waiting and supervisor.clipboard.connected.ready()) {
        switch (supervisor.clipboard.connected.publish(transcriptId(supervisor), text, now_ns)) {
            .ok => delivery.paste = .acquiring,
            .err => |err| clipboardError(supervisor, err),
        }
    }
    if (delivery.paste == .settling or delivery.paste == .sending) {
        if (supervisor.clipboard != .connected or !supervisor.clipboard.connected.owns(transcriptId(supervisor))) {
            delivery.problem = .clipboard_failed;
            cancelDelivery(supervisor);
        }
    }
    switch (delivery.paste) {
        .waiting, .acquiring, .done => {},
        .settling => |deadline_ns| if (now_ns >= deadline_ns) {
            supervisor.keyboard.ready.beginPaste(supervisor.options.paste_key, supervisor.options.paste_key_gap_ms, now_ns);
            delivery.paste = .sending;
        },
        .sending => {},
    }
    if (delivery.paste == .sending) {
        const complete = switch (supervisor.keyboard.ready.advance(now_ns)) {
            .ok => |complete| complete,
            .err => |err| failed: {
                logPasteError(.err, .{ .recording_ordinal = supervisor.recording_ordinal }, "Paste error; clipboard retained, no retry", &err);
                supervisor.keyboard.ready.deinit();
                supervisor.keyboard = .{ .unavailable = pasteProblem(err) };
                delivery.paste = .done;
                delivery.problem = pasteProblem(err);
                break :failed false;
            },
        };
        if (complete) {
            log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Paste shortcut sent: recording_ordinal={d}, recording_stop_origin={s}, recording_stop_elapsed_ms={s}, paste_duration_ms={f}", .{
                supervisor.recording_ordinal,                                                                          stop_origin, formatRecordingStopElapsedMilliseconds(&stop_buffer, supervisor),
                decimal.fmt(@as(f64, @floatFromInt(now_ns - delivery.boundary_monotonic_ns)) / std.time.ns_per_ms, 3),
            });
            delivery.paste = .done;
        }
    }
    if (delivery.paste == .done) {
        completeDelivery(supervisor, delivery.problem);
        enterIdle(supervisor);
    }
}

// Save after paste completion, borrowing the same immutable clipboard text.
// Choose the popup from both outcomes, so it never promises a saved file before
// the rename succeeds.
fn completeDelivery(supervisor: *Supervisor, problem: ?notifications.Problem) void {
    const text = std.mem.trim(u8, transcriptBytes(supervisor)[0..supervisor.transcript.bytes_count], " \t\r\n");
    if (text.len == 0) return;
    const save_error = saveTranscript(supervisor, text);
    const saved = save_error == null;
    if (supervisor.service.shutdown_requested) return;
    if (problem) |cause| {
        const output: notifications.Output = switch (cause) {
            .clipboard_failed => if (saved) .saved else .unsaved,
            .paste_failed, .paste_permission_denied, .paste_device_missing, .paste_incomplete => if (saved) .clipboard_saved else .clipboard_unsaved,
            else => if (saved) .partial_saved else .partial_unsaved,
        };
        supervisor.notifications.showOutput(cause, output);
    } else if (save_error) |err| {
        supervisor.notifications.showOutput(err, .clipboard_unsaved);
    } else supervisor.notifications.recover();
}

fn saveTranscript(supervisor: *const Supervisor, text: []const u8) ?notifications.Problem {
    const directory_path = supervisor.service.transcript_directory orelse {
        log.err(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript not saved: recording_ordinal={d}, reason=state_directory_unavailable", .{supervisor.recording_ordinal});
        return .transcript_save_failed;
    };
    const started = monotonicNanoseconds();
    var elapsed_buffer: [32]u8 = undefined;
    switch (transcript_file.save(supervisor.io, directory_path, text)) {
        .ok => {},
        .err => |err| {
            logTranscriptSaveError(.{ .recording_ordinal = supervisor.recording_ordinal }, directory_path, monotonicNanoseconds() - started, &err);
            return saveProblem(err);
        },
    }
    log.info(.{ .recording_ordinal = supervisor.recording_ordinal }, "Transcript saved: recording_ordinal={d}, transcript_size={d}, transcript_save_duration_ms={s}", .{
        supervisor.recording_ordinal, text.len, formatDurationMilliseconds(&elapsed_buffer, monotonicNanoseconds() - started),
    });
    return null;
}

noinline fn logTranscriptSaveError(context: logging.Context, directory_path: []const u8, duration_ns: u64, err: *const transcript_file.Error) void {
    if (!logging.enabled(.err)) return;
    // The largest case is unsafe_directory: five common fields plus five
    // inspection fields. Cleanup adds two fields only to write/replace errors.
    var storage: [10]logging.Entry = undefined;
    var fields: std.ArrayList(logging.Entry) = .initBuffer(&storage);
    if (context.recording_ordinal) |ordinal| fields.appendAssumeCapacity(.{ "recording_ordinal", .{ .u = ordinal } });
    fields.appendSliceAssumeCapacity(&.{
        .{ "directory_path", .{ .str = directory_path } },
        .{ "file_name", .{ .str = "transcript.txt" } },
        .{ "operation", .{ .str = @tagName(err.*) } },
        .{ "transcript_save_duration_ms", .{ .f = .{ .value = @as(f64, @floatFromInt(duration_ns)) / std.time.ns_per_ms, .digits = 3 } } },
    });
    var cleanup: ?*const transcript_file.CleanupError = null;
    switch (err.*) {
        .open_directory, .permissions, .create_temporary => |cause| fields.appendAssumeCapacity(.{ "error", .{ .str = @errorName(cause) } }),
        .stat_directory => |errno| fields.appendAssumeCapacity(.{ "errno", .{ .errno = errno } }),
        .unsafe_directory => |detail| fields.appendSliceAssumeCapacity(&.{
            .{ "uid", .{ .u = detail.uid } },
            .{ "expected_uid", .{ .u = detail.expected_uid } },
            .{ "mode", .{ .u = detail.mode } },
            .{ "uid_available", .{ .b = detail.uid_available } },
            .{ "mode_available", .{ .b = detail.mode_available } },
        }),
        .write => |*detail| {
            fields.appendSliceAssumeCapacity(&.{
                .{ "error", .{ .str = @errorName(detail.cause) } },
                .{ "bytes_written", .{ .u = detail.bytes_written } },
                .{ "bytes_total", .{ .u = detail.bytes_total } },
            });
            if (detail.cleanup) |*failure| cleanup = failure;
        },
        .replace => |*detail| {
            fields.appendAssumeCapacity(.{ "error", .{ .str = @errorName(detail.cause) } });
            if (detail.cleanup) |*failure| cleanup = failure;
        },
    }
    if (cleanup) |failure| fields.appendSliceAssumeCapacity(&.{
        .{ "cleanup_error", .{ .str = @errorName(failure.cause) } },
        .{ "temporary_name", .{ .str = &failure.temporary_name } },
    });
    log.kv(.err, context, "Transcript save error", fields.items);
}

fn formatRecordingStopElapsedMilliseconds(buffer: *[32]u8, supervisor: *const Supervisor) []const u8 {
    const stop = supervisor.recording_stop orelse return "unavailable";
    return formatDurationMilliseconds(buffer, monotonicNanoseconds() - stop.monotonic_ns);
}

fn cancelDelivery(supervisor: *Supervisor) void {
    const delivery = &supervisor.phase.delivering;
    closeClipboard(supervisor);
    if (supervisor.keyboard == .ready) {
        const keyboard = &supervisor.keyboard.ready;
        if (keyboard.pending != null) {
            keyboard.deinit();
            supervisor.keyboard = .disabled;
        }
    }
    delivery.paste = .done;
}

fn transcriptId(supervisor: *const Supervisor) u64 {
    return @as(u64, supervisor.transcript.storage_index) + 1;
}

fn transcriptBytes(supervisor: *const Supervisor) []u8 {
    const capacity = supervisor.transcript_storage.len / transcriptStorageCount(supervisor);
    const offset = @as(usize, supervisor.transcript.storage_index) * capacity;
    return supervisor.transcript_storage[offset..][0..capacity];
}

fn availableTranscript(supervisor: *const Supervisor) ?u1 {
    for (0..transcriptStorageCount(supervisor)) |index| {
        if (supervisor.clipboard != .connected or !supervisor.clipboard.connected.isBorrowed(index + 1)) return @intCast(index);
    }
    return null;
}

fn transcriptStorageCount(supervisor: *const Supervisor) usize {
    return if (supervisor.options.output == .stdout) 1 else 2;
}

fn closeClipboard(supervisor: *Supervisor) void {
    if (supervisor.clipboard == .connected) supervisor.clipboard.connected.deinit();
    supervisor.clipboard = .disconnected;
}

fn clipboardError(supervisor: *Supervisor, err: clipboard_module.Error) void {
    if (logging.enabled(.err)) {
        var storage: [10]logging.Entry = undefined;
        var fields: std.ArrayList(logging.Entry) = .initBuffer(&storage);
        var protocol_error = false;
        switch (err) {
            .wayland => |*detail| {
                fields.appendSliceAssumeCapacity(&.{
                    .{ "backend", .{ .str = "wayland" } },
                    .{ "kind", .{ .str = @tagName(detail.*) } },
                });
                switch (detail.*) {
                    .transport => |failure| fields.appendSliceAssumeCapacity(&.{
                        .{ "error", .{ .str = @errorName(failure.cause) } },
                        .{ "errno", .{ .errno = failure.errno } },
                        .{ "object", .{ .u = failure.object } },
                        .{ "opcode", .{ .u = failure.opcode } },
                    }),
                    .server => |*failure| {
                        protocol_error = true;
                        fields.appendSliceAssumeCapacity(&.{
                            .{ "object", .{ .u = failure.object } },
                            .{ "code", .{ .u = failure.code } },
                            .{ "message", .{ .str = failure.message[0..failure.message_size] } },
                            .{ "truncated", .{ .b = failure.truncated } },
                        });
                    },
                    .unsupported => |feature| fields.appendAssumeCapacity(.{ "feature", .{ .str = @tagName(feature) } }),
                    .timed_out => |phase| fields.appendAssumeCapacity(.{ "phase", .{ .str = @tagName(phase) } }),
                    .selection_lost, .busy, .invalid_text => {},
                }
            },
            .x11 => |*detail| {
                fields.appendSliceAssumeCapacity(&.{
                    .{ "backend", .{ .str = "x11" } },
                    .{ "kind", .{ .str = @tagName(detail.*) } },
                });
                switch (detail.*) {
                    .transport => |failure| fields.appendSliceAssumeCapacity(&.{
                        .{ "error", .{ .str = @errorName(failure.cause) } },
                        .{ "errno", .{ .errno = failure.errno } },
                        .{ "response_type", .{ .u = failure.response_type } },
                        .{ "sequence", .{ .u = failure.sequence } },
                    }),
                    .setup => |*failure| fields.appendSliceAssumeCapacity(&.{
                        .{ "status", .{ .u = failure.status } },
                        .{ "message", .{ .str = failure.reason[0..failure.reason_size] } },
                        .{ "truncated", .{ .b = failure.truncated } },
                    }),
                    .server => |failure| {
                        protocol_error = true;
                        fields.appendSliceAssumeCapacity(&.{
                            .{ "code", .{ .u = failure.code } },
                            .{ "sequence", .{ .u = failure.sequence } },
                            .{ "major_opcode", .{ .u = failure.major_opcode } },
                            .{ "minor_opcode", .{ .u = failure.minor_opcode } },
                            .{ "bad_value", .{ .u = failure.bad_value } },
                        });
                    },
                    .authority => |failure| fields.appendSliceAssumeCapacity(&.{
                        .{ "error", .{ .str = @errorName(failure.cause) } },
                        .{ "errno", .{ .errno = failure.errno } },
                    }),
                    .unsupported => |feature| fields.appendAssumeCapacity(.{ "feature", .{ .str = @tagName(feature) } }),
                    .timed_out => |phase| fields.appendAssumeCapacity(.{ "phase", .{ .str = @tagName(phase) } }),
                    .selection_lost, .busy, .invalid_text => {},
                }
            },
            .unavailable, .busy, .invalid_text => fields.appendAssumeCapacity(.{ "kind", .{ .str = @tagName(err) } }),
        }
        fields.appendSliceAssumeCapacity(&.{
            .{ "transfers_completed", .{ .u = supervisor.clipboard.connected.transfers_completed } },
            .{ "transfers_expired", .{ .u = supervisor.clipboard.connected.transfers_expired } },
            .{ "transfers_rejected", .{ .u = supervisor.clipboard.connected.transfers_rejected } },
        });
        log.kv(.err, .{ .recording_ordinal = supervisor.recording_ordinal }, if (protocol_error) "Clipboard protocol error" else "Clipboard error", fields.items);
    }
    closeClipboard(supervisor);
    if (supervisor.phase == .delivering) {
        supervisor.phase.delivering.problem = .clipboard_failed;
        cancelDelivery(supervisor);
    }
}

fn openPasteKeyboard(supervisor: *Supervisor) void {
    switch (paste_keyboard.Keyboard.open(monotonicNanoseconds())) {
        .ok => |keyboard| {
            supervisor.keyboard = .{ .ready = keyboard };
        },
        .err => |err| {
            supervisor.keyboard = .{ .unavailable = pasteProblem(err) };
            logPasteError(.warn, .{}, "Automatic paste unavailable; clipboard delivery remains enabled", &err);
        },
    }
}

// Share the schema between setup and delivery failures rather than specializing
// the error union's formatting for each caller's severity and message.
noinline fn logPasteError(severity: logging.Level, context: logging.Context, message: []const u8, err: *const paste_keyboard.Error) void {
    if (!logging.enabled(severity)) return;
    // Setup has the most fields: ordinal, operation, request, errno and six
    // setup members. Only the initialized entries reach the synchronous logger.
    var storage: [10]logging.Entry = undefined;
    var fields: std.ArrayList(logging.Entry) = .initBuffer(&storage);
    if (context.recording_ordinal) |ordinal| fields.appendAssumeCapacity(.{ "recording_ordinal", .{ .u = ordinal } });
    fields.appendAssumeCapacity(.{ "operation", .{ .str = @tagName(err.*) } });
    const progress: ?paste_keyboard.Progress = switch (err.*) {
        .open => |errno| blk: {
            fields.appendAssumeCapacity(.{ "errno", .{ .errno = errno } });
            break :blk null;
        },
        .configure => |*detail| blk: {
            fields.appendSliceAssumeCapacity(&.{
                .{ "request", .{ .u = detail.request } },
                .{ "errno", .{ .errno = detail.errno } },
            });
            switch (detail.argument) {
                .value => |value| fields.appendAssumeCapacity(.{ "argument", .{ .u = value } }),
                .setup => |*setup| fields.appendSliceAssumeCapacity(&.{
                    .{ "bustype", .{ .u = setup.id.bustype } },
                    .{ "vendor", .{ .u = setup.id.vendor } },
                    .{ "product", .{ .u = setup.id.product } },
                    .{ "version", .{ .u = setup.id.version } },
                    .{ "name", .{ .str = std.mem.sliceTo(&setup.name, 0) } },
                    .{ "ff_effects_max", .{ .u = setup.ff_effects_max } },
                }),
            }
            break :blk null;
        },
        .write => |detail| blk: {
            fields.appendAssumeCapacity(.{ "errno", .{ .errno = detail.errno } });
            break :blk detail.progress;
        },
        .ambiguous_write => |detail| blk: {
            fields.appendAssumeCapacity(.{ "bytes_written", .{ .u = detail.bytes_written } });
            break :blk detail.progress;
        },
        .timed_out => |detail| blk: {
            fields.appendSliceAssumeCapacity(&.{
                .{ "deadline_ns", .{ .u = detail.deadline_ns } },
                .{ "observed_ns", .{ .u = detail.observed_ns } },
            });
            break :blk detail.progress;
        },
    };
    if (progress) |value| fields.appendSliceAssumeCapacity(&.{
        .{ "chord", .{ .str = @tagName(value.chord) } },
        .{ "frame", .{ .u = value.frame } },
        .{ "frame_bytes_sent", .{ .u = value.frame_bytes_sent } },
    });
    log.kv(severity, context, message, fields.items);
}

fn audioProblem(err: capture_module.Error) notifications.Problem {
    return switch (err) {
        .source_not_found => .microphone_not_found,
        .source_ambiguous => .microphone_ambiguous,
        .source_connection_lost => .microphone_connection_lost,
        .source_changed => .microphone_changed,
        .pipeline_full => .audio_processing_behind,
        .setup => .audio_setup_failed,
        .unexpected_format,
        .timeline_discontinuity,
        .stream_error,
        .stream_disconnected,
        .invalid_buffer,
        .corrupted_buffer,
        => .microphone_failed,
    };
}

const CaptureFailureCause = struct { domain: []const u8, code: i64 };

fn captureFailureCause(cause: capture_module.FailureCause) CaptureFailureCause {
    return switch (cause) {
        .zig => |err| .{ .domain = "zig", .code = @intFromError(err) },
        .linux => |err| .{ .domain = "linux", .code = @intFromEnum(err) },
        .pipewire => |code| .{ .domain = "pipewire", .code = code },
        .audio => |err| .{ .domain = "audio", .code = @intFromEnum(err) },
    };
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
    // Delivery no longer borrows session text. Acknowledged completion released every audio borrow before this idle timer starts.
    supervisor.phase = .idle;
    if (supervisor.transcription == .idle) supervisor.transcription.idle = if (!supervisor.service.shutdown_requested)
        monotonicNanoseconds() + @as(u64, supervisor.options.model_keep_warm_seconds) * std.time.ns_per_s
    else
        null;
}

// PERFORMANCE: Keep final-report formatting outside result dispatch. Inlining
// exposes the report's tagged payloads to the caller's branches and expands the
// diagnostic code in ReleaseSafe. One call per completed capture is off the
// graph-processing path; the report remains borrowed only during this call.
noinline fn logCaptureReport(recording_ordinal: u64, outcome_name: []const u8, report: *const capture_module.CaptureReport, failure: ?*const capture_module.RuntimeFailure) void {
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

    log.info(.{ .recording_ordinal = recording_ordinal }, "Capture ended: recording_ordinal={d}, outcome={s}, audio_duration_seconds={f}, audio_samples_published_count={d}, audio_samples_captured_count={d}, microphone_description=\"{f}\"", .{ recording_ordinal, outcome_name, decimal.fmt(@as(f64, @floatFromInt(report.samples_count)) / capture_module.sample_rate_hz, 3), report.published_samples_count, report.samples_count, std.zig.fmtString(source_description) });

    log.info(.{ .recording_ordinal = recording_ordinal }, "Capture protocol: recording_ordinal={d}, pipewire_server_version=\"{f}\", client_node_version_advertised={d}, client_node_version_selected={d}, graph_rate_hz={d}, graph_channels_count={d}, callback_duration_ms_max={f}, callback_gap_ms_max={f}", .{ recording_ordinal, std.zig.fmtString(report.pipewire_server_version[0..report.pipewire_server_version_size]), report.client_node_version_advertised, report.client_node_version_selected, if (report.negotiated_format) |format| format.sample_rate_hz else 0, if (report.negotiated_format) |format| format.channels_count else 0, decimal.fmt(if (report.callback) |callback| @as(f64, @floatFromInt(callback.duration_ns_max)) / std.time.ns_per_ms else 0, 3), decimal.fmt(if (report.callback) |callback| @as(f64, @floatFromInt(callback.gap_ns_max)) / std.time.ns_per_ms else 0, 3) });

    if (failure != null) {
        if (report.source_identity) |source| log.err(.{ .recording_ordinal = recording_ordinal }, "Capture source: recording_ordinal={d}, node_id={d}, node_object_serial={d}, device_id={d}, device_object_serial={d}, node_name=\"{f}\", node_description=\"{f}\", device_serial=\"{f}\", device_description=\"{f}\"", .{ recording_ordinal, source.node_id, source.node_object_serial, source.device_id, source.device_object_serial, std.zig.fmtString(source.node_name[0..source.node_name_size]), std.zig.fmtString(source.node_description[0..source.node_description_size]), std.zig.fmtString(source.device_serial[0..source.device_serial_size]), std.zig.fmtString(source.device_description[0..source.device_description_size]) });
    }
    if (failure) |detail| {
        const cause = captureFailureCause(detail.cause);
        log.err(.{ .recording_ordinal = recording_ordinal }, "Capture failed: recording_ordinal={d}, stage={s}, error_domain={s}, error_code={d}, detail=\"{f}\"", .{ recording_ordinal, @tagName(detail.stage), cause.domain, cause.code, std.zig.fmtString(detail.message[0..detail.message_size]) });
    }

    // These failures reduce scheduling guarantees, not the validity of already
    // captured samples. Preserve the warnings without discarding valid audio.
    if (report.callback != null and
        (scheduler_priority <= 0 or !capture_module.schedulerPolicyIsRealtime(scheduler_policy)))
    {
        log.warn(.{ .recording_ordinal = recording_ordinal }, "Capture did not obtain realtime scheduling: recording_ordinal={d}, policy={s}, priority={d}", .{
            recording_ordinal, capture_module.schedulerPolicyName(scheduler_policy), scheduler_priority,
        });
    }
    if (report.memory_lock == .unavailable) {
        const unavailable = report.memory_lock.unavailable;
        log.warn(.{ .recording_ordinal = recording_ordinal }, "Capture memory was not locked: recording_ordinal={d}, errno={d}, memory_lock_size_max={d}", .{
            recording_ordinal, unavailable.error_code, unavailable.limit_bytes,
        });
    }
}

fn captureActive(supervisor: *const Supervisor) bool {
    return supervisor.audio.operation != .idle;
}

fn processedChunksCount(transcript: *const TranscriptProgress) u32 {
    return transcript.accepted_chunks_count + transcript.no_speech_chunks_count;
}

fn publishedSlots(supervisor: *Supervisor) u32 {
    var count: u32 = 0;
    for (&supervisor.audio_exchange.slots) |*slot| {
        if (audio_exchange.acquireSlot(slot) != null) count += 1;
    }
    return count;
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
        log.err(.{}, "Event registration failed: operation=epoll_ctl_add, fd={d}, source={t}, errno={f}", .{ descriptor, source, logging.fmtErrno(linux.errno(result)) });
        return error.SupervisorEpollRegisterFailed;
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
    log.err(.{}, "Supervisor system call failed: operation={s}, errno={f}", .{ operation, logging.fmtErrno(errno) });
    return error.SupervisorSystemCallFailed;
}

// Cleanup must retain its own diagnostics without replacing a primary error.
// Closing the descriptor also removes its epoll registration. Never retry close:
// Linux releases the descriptor even when close reports a late I/O error.
fn logCleanupSyscall(operation: []const u8, result: usize) void {
    const errno = linux.errno(result);
    if (errno != .SUCCESS) log.err(.{}, "Supervisor cleanup failed: operation={s}, errno={f}", .{ operation, logging.fmtErrno(errno) });
}

fn closeDescriptor(descriptor: std.posix.fd_t) void {
    assert(descriptor >= 0);
    logCleanupSyscall("close", linux.close(descriptor));
}
