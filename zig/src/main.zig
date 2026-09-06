const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.service);
const audio_process = @import("audio_process.zig");
const control_socket = @import("control_socket.zig");
const clipboard_process = @import("clipboard_process.zig");
const service_config = @import("service_config.zig");
const supervisor = @import("supervisor.zig");
const transcription_process = @import("transcription_process.zig");
const stderr = std.debug.print;

pub const std_options: std.Options = .{
    // Keep panic stack tracing even when building a compact executable.
    .allow_stack_tracing = true,
    // IPC uses raw Unix sockets; PipeWire and libsystemd use their own APIs.
    // The separate model-setup executable retains std.Io networking.
    .networking = false,
};

pub fn main(init: std.process.Init) u8 {
    // ── Parse Before Acting ──
    //
    // No socket, microphone, or worker is opened until the complete command is
    // valid. Helpers return errors and borrowed argument context; only main
    // formats diagnostics. Returning a status lets normal cleanup run:
    //   0 = success or help, 1 = operational failure, 2 = invalid command line.
    const arguments = init.minimal.args.toSlice(init.arena.allocator()) catch |err| {
        stderr("voiced: could not read command-line arguments ({s}).\n", .{@errorName(err)});

        return 1;
    };

    var diagnostic: ArgumentDiagnostic = .{};
    const invocation = parseCommand(init, arguments[1..], &diagnostic) catch |err| {
        if (diagnostic.path.len > 0) stderr("{s}:{d}: ", .{ diagnostic.path, diagnostic.line });
        switch (err) {
            error.UnknownCommand => stderr("voiced: unknown command '{s}'.\n", .{diagnostic.argument}),
            error.UnknownOption => stderr("voiced: unknown option '{s}'.\n", .{diagnostic.argument}),
            error.UnexpectedArgument => stderr("voiced: unexpected argument '{s}'; this command takes no positional arguments.\n", .{diagnostic.argument}),
            error.MissingValue => stderr("voiced: option '{s}' requires a value.\n", .{diagnostic.argument}),
            error.DuplicateOption => stderr("voiced: option '{s}' was supplied more than once; specify it only once.\n", .{diagnostic.argument}),
            error.ConflictingOptions => stderr("voiced: options '{s}' and '{s}' cannot be used together; choose one source.\n", .{ diagnostic.other_option, diagnostic.argument }),
            error.InvalidValue => stderr("voiced: invalid value '{s}' for '{s}'; expected {s}.\n", .{ diagnostic.value, diagnostic.argument, diagnostic.expected }),
            error.OptionTakesNoValue => stderr("voiced: option '{s}' does not take a value; use 'voiced record --toggle'.\n", .{diagnostic.argument}),
            error.InvalidConfigLine => stderr("voiced: expected key=value.\n", .{}),
            error.InvalidInternalRole => stderr("voiced: invalid internal worker invocation; use 'voiced serve' to start the daemon.\n", .{}),
            else => stderr("voiced: could not load configuration ({s}).\n", .{@errorName(err)}),
        }

        if (diagnostic.command.len > 0) {
            stderr("Run 'voiced {s} --help' for usage and examples.\n", .{diagnostic.command});
        } else {
            stderr("Run 'voiced --help' for commands and examples.\n", .{});
        }

        return 2;
    };

    // ── Help Without Side Effects ──
    //
    // Explicit help goes to stdout and succeeds even when other options are
    // incomplete: `voiced serve --recording-seconds-max --help` never starts the service.
    if (invocation == .usage or invocation == .help) {
        var help_buffer: [4096]u8 = undefined;
        const text = if (invocation == .usage)
            brief_help
        else text: {
            const command = invocation.help;
            if (command.len == 0) {
                break :text general_help;
            }

            if (std.mem.eql(u8, command, "serve")) {
                const defaults: supervisor.ServiceOptions = .{};

                break :text std.fmt.bufPrint(&help_buffer, serve_help, .{
                    defaults.capture.transcription.model.name(),
                    defaults.capture.transcription.inference_threads_count,
                    @as(u8, switch (defaults.capture.transcription.encoder_trailing_padding) {
                        .seconds_5 => 5,
                        .seconds_10 => 10,
                        .seconds_30 => 30,
                    }),
                    defaults.capture.recording_seconds,
                    defaults.model_keep_warm_seconds,
                    defaults.paste_settle_ms,
                    defaults.paste_key_gap_ms,
                }) catch unreachable;
            }

            if (std.mem.eql(u8, command, "record")) {
                break :text record_help;
            }

            const client_command = std.meta.stringToEnum(@FieldType(control_socket.Request, "cmd"), command).?;

            const description = switch (client_command) {
                .listen => "Start recording and stop automatically after speech followed by silence.",
                .stop => "Stop recording and finish transcribing the captured audio.",
                .cancel => "Cancel the current recording and discard its transcription.",
                .status => "Show daemon, recording, and model status as JSON.",
                .kill => "Shut down the daemon and its workers.",
                .record => unreachable,
            };

            break :text std.fmt.bufPrint(
                &help_buffer,
                "{s}\n\nUsage:\n  voiced {s}\n\n" ++
                    "Options:\n  -h, --help  Show this help.\n\n" ++
                    "The daemon must already be running; start it with 'voiced serve'.\n" ++
                    "Use the same VOICED_INSTANCE in both terminals.\n" ++
                    "Responses are written to stdout as JSON; errors go to stderr.\n",
                .{ description, command },
            ) catch unreachable;
        };

        std.Io.File.stdout().writeStreamingAll(init.io, text) catch |err| {
            stderr("voiced: could not write help ({s}).\n", .{@errorName(err)});

            return 1;
        };

        return 0;
    }

    // ── Execute The Validated Command ──
    //
    // Worker invocations are private re-executions of this binary. Their argv
    // carries the role, control FD, parent PID, and logging level. Shared
    // descriptors and model/capture settings arrive in the first seqpacket.
    // The clipboard role instead inherits stdin and waits for a start packet.
    if (invocation == .serve or invocation == .worker) {
        const configured_level = if (invocation == .serve) invocation.serve.log_level else invocation.worker.log_level;
        const errno = logging.init(configured_level);
        if (errno != .SUCCESS) {
            // This is still CLI startup, before service deadlines or threads.
            stderr("voiced: could not initialize logging (errno={t}).\n", .{errno});
            return 1;
        }
    }
    defer logging.deinit();
    const process_name: ?[:0]const u8 = switch (invocation) {
        .serve => "voiced",
        .worker => |worker| switch (worker.role) {
            .@"audio-pipewire" => "voiced-capture",
            .@"transcription-model" => "voiced-asr",
            .clipboard => "voiced-clip",
        },
        else => null,
    };
    if (process_name) |name| {
        const linux = std.os.linux;
        std.debug.assert(name.len <= 15);
        const result = linux.prctl(@intFromEnum(linux.PR.SET_NAME), @intFromPtr(name.ptr), 0, 0, 0);
        if (linux.errno(result) != .SUCCESS) log.warn(.{}, "Process name unavailable: operation=prctl_set_name, errno={t}", .{linux.errno(result)});
    }

    if (invocation == .worker) log.debug(.{}, "Worker starting: role={s}, log_level={s}", .{ @tagName(invocation.worker.role), logging.levelName(invocation.worker.log_level) });

    const execution = switch (invocation) {
        .serve => |options| supervisor.runService(init, options),
        .client => |request| control_socket.sendRequest(init, request),
        .worker => |worker| switch (worker.role) {
            .@"audio-pipewire" => audio_process.PipeWireWorker.run(worker.socket, worker.supervisor_pid),
            .@"transcription-model" => transcription_process.runModelWorker(init, worker.socket, worker.supervisor_pid),
            .clipboard => clipboard_process.runWorker(init, worker.socket, worker.supervisor_pid),
        },
        .usage, .help => unreachable,
    };

    execution catch |err| {
        if (invocation == .worker) {
            log.err(.{}, "Worker stopped: role={s}, detail=\"{f}\"", .{ @tagName(invocation.worker.role), std.zig.fmtString(@errorName(err)) });
            return 1;
        }
        if (invocation == .serve) {
            switch (err) {
                error.DaemonAlreadyRunning, error.RuntimeDirectoryNotSet, error.RuntimeDirectoryNotAbsolute, error.InvalidInstance, error.UnsafeRuntimeDirectory, error.UnsafeControlSocket, error.SocketPathTooLong => log.err(.{}, "Service startup refused: error={s}", .{@errorName(err)}),
                else => log.critical(.{}, "Service stopped: error={s}", .{@errorName(err)}),
            }
            return 1;
        }
        switch (err) {
            error.DaemonNotRunning => stderr("voiced: cannot connect to the daemon. Start 'voiced serve' in another terminal with the same VOICED_INSTANCE.\n", .{}),
            error.DaemonAlreadyRunning => stderr("voiced: a daemon is already running for this instance. Use 'voiced status' or select another VOICED_INSTANCE.\n", .{}),
            error.RuntimeDirectoryNotSet, error.RuntimeDirectoryNotAbsolute => stderr("voiced: XDG_RUNTIME_DIR must name an absolute runtime directory for your user session.\n", .{}),
            error.InvalidInstance => stderr("voiced: VOICED_INSTANCE must contain at most 40 ASCII letters, digits, hyphens, or underscores.\n", .{}),
            error.UnsafeRuntimeDirectory => stderr("voiced: XDG_RUNTIME_DIR and the voiced instance directory must be owned by your user with no group or other permissions (typically 0700).\n", .{}),
            error.CommandRejected => stderr("voiced: the daemon rejected the command; see its JSON response.\n", .{}),
            error.ControlSendFailed, error.ControlReceiveFailed => stderr("voiced: communication with the daemon failed or timed out. Check 'voiced status' before retrying a recording command.\n", .{}),
            else => stderr("voiced: could not complete the command ({s}).\n", .{@errorName(err)}),
        }

        return 1;
    };

    if (invocation == .serve) log.info(.{}, "Service stopped", .{});
    return 0;
}

const Invocation = union(enum) {
    usage,
    help: []const u8,
    serve: supervisor.ServiceOptions,
    client: control_socket.Request,
    worker: struct {
        role: enum { @"audio-pipewire", @"transcription-model", clipboard },
        socket: std.posix.fd_t,
        supervisor_pid: std.os.linux.pid_t,
        log_level: logging.Level,
    },
};

const ArgumentDiagnostic = service_config.Diagnostic;

fn parseCommand(init: std.process.Init, arguments: []const [:0]const u8, diagnostic: *ArgumentDiagnostic) !Invocation {
    if (arguments.len == 0) {
        return .usage;
    }

    const command = arguments[0];

    // Private worker argv must never be interpreted as public help or commands.
    if (std.mem.eql(u8, command, "--internal-role")) {
        if (arguments.len != 5) {
            return error.InvalidInternalRole;
        }

        const role = std.meta.stringToEnum(@FieldType(@FieldType(Invocation, "worker"), "role"), arguments[1]) orelse {
            return error.InvalidInternalRole;
        };

        const socket = std.fmt.parseInt(std.posix.fd_t, arguments[2], 10) catch {
            return error.InvalidInternalRole;
        };
        const parent_pid = std.fmt.parseInt(std.os.linux.pid_t, arguments[3], 10) catch {
            return error.InvalidInternalRole;
        };
        if (socket < 0 or parent_pid <= 1) {
            return error.InvalidInternalRole;
        }

        return .{ .worker = .{ .role = role, .socket = socket, .supervisor_pid = parent_pid, .log_level = logging.parseLevel(arguments[4]) orelse return error.InvalidInternalRole } };
    }

    // Scan help before validating other flags, but respect `--` as the end of
    // options. An explicit `--microphone-node=--help` remains a value, not a help request.
    for (arguments) |argument| {
        if (std.mem.eql(u8, argument, "--")) {
            break;
        }
        if (std.mem.eql(u8, argument, "-h") or std.mem.eql(u8, argument, "--help")) {
            const topic = if (std.mem.eql(u8, command, "serve") or
                std.meta.stringToEnum(@FieldType(control_socket.Request, "cmd"), command) != null)
                command
            else
                "";
            return .{ .help = topic };
        }
    }

    if (std.mem.eql(u8, command, "help")) {
        if (arguments.len == 1) {
            return .{ .help = "" };
        }

        const topic = arguments[1];
        diagnostic.argument = topic;

        if (!std.mem.eql(u8, topic, "serve") and
            std.meta.stringToEnum(@FieldType(control_socket.Request, "cmd"), topic) == null)
        {
            return error.UnknownCommand;
        }

        if (arguments.len > 2) {
            diagnostic.argument = arguments[2];
            return error.UnexpectedArgument;
        }

        return .{ .help = topic };
    }

    if (std.mem.eql(u8, command, "serve")) {
        diagnostic.command = command;
        return .{ .serve = try service_config.load(init, arguments[1..], diagnostic) };
    }

    diagnostic.argument = command;

    const client_command = std.meta.stringToEnum(@FieldType(control_socket.Request, "cmd"), command) orelse {
        if (std.mem.startsWith(u8, command, "-")) {
            return error.UnknownOption;
        }

        return error.UnknownCommand;
    };

    diagnostic.command = command;

    // Only record takes a toggle flag; aliases identify the same option:
    //   record                 → start a recording.
    //   record -t / --toggle    → toggle recording.
    //   record -t --toggle      → DuplicateOption, not two toggles.
    //   stop -t                → UnknownOption; stop has no toggle mode.
    var request: control_socket.Request = .{ .cmd = client_command };

    for (arguments[1..], 1..) |argument, index| {
        diagnostic.argument = argument;

        if (std.mem.eql(u8, argument, "--")) {
            if (index + 1 < arguments.len) {
                diagnostic.argument = arguments[index + 1];
                return error.UnexpectedArgument;
            }

            break;
        }

        if (client_command == .record and
            (std.mem.eql(u8, argument, "-t") or std.mem.eql(u8, argument, "--toggle")))
        {
            if (request.toggle) {
                return error.DuplicateOption;
            }

            request.toggle = true;
            continue;
        }

        if (client_command == .record and std.mem.startsWith(u8, argument, "--toggle=")) {
            diagnostic.argument = "--toggle";
            return error.OptionTakesNoValue;
        }

        if (std.mem.startsWith(u8, argument, "-")) {
            return error.UnknownOption;
        }

        return error.UnexpectedArgument;
    }

    return .{ .client = request };
}

const brief_help =
    \\voiced - voice dictation daemon
    \\
    \\Usage: voiced <command> [options]
    \\
    \\  voiced serve       Start the daemon in the foreground.
    \\  voiced record -t   Toggle recording from another terminal.
    \\
    \\Run 'voiced --help' for all commands and examples.
    \\
;

const general_help =
    \\voiced - voice dictation daemon
    \\
    \\Examples:
    \\  voiced serve                 Start the daemon in one terminal.
    \\  voiced record --toggle       Start or stop recording in another.
    \\  voiced listen                Record until speech is followed by silence.
    \\  voiced status                Inspect recording and model status.
    \\
    \\Usage: voiced <command> [options]
    \\
    \\Commands:
    \\  serve    Run the daemon in the foreground.
    \\  listen   Record with automatic silence stopping.
    \\  record   Record manually; -t or --toggle toggles recording.
    \\  stop     Stop recording and finish transcription.
    \\  cancel   Discard the current recording.
    \\  status   Print daemon status as JSON.
    \\  kill     Shut down the daemon.
    \\  help     Show help, optionally for a command.
    \\
    \\Options:
    \\  -h, --help  Show help without executing a command.
    \\
    \\Run 'voiced help serve' or 'voiced <command> --help' for details.
    \\Client commands print JSON to stdout; errors go to stderr.
    \\The daemon copies final text with wl-copy and sends a paste shortcut.
    \\Use 'voiced serve --transcript-output stdout' for diagnostic transcript output.
    \\
    \\Environment:
    \\  XDG_RUNTIME_DIR Your user session's absolute runtime directory.
    \\  VOICED_INSTANCE Optional isolated instance name; use the same value
    \\                  for the daemon and its clients (e.g. 'test').
    \\
    \\Exit status: 0 success/help, 1 operational failure, 2 command-line error.
    \\
;

const serve_help =
    \\Run the voice dictation daemon in the foreground.
    \\
    \\Examples:
    \\  voiced serve
    \\  voiced serve --model-encoder-threads 8 --recording-seconds-max 120
    \\  voiced serve --microphone-serial ABC123 --model-idle-seconds-max 0
    \\
    \\Usage: voiced serve [options]
    \\
    \\Options:
    \\  --log-level <critical|error|warn|info|debug>  Diagnostic threshold (default: info).
    \\  --config <path>                           Read this file instead of the default.
    \\  --model <name>                            Model repository (default: {s}).
    \\  --model-encoder-threads <1-32>            CPU workers (default: {d}).
    \\  --model-decoder-threads <1-32>            Decoder subset (default: encoder count).
    \\  --model-encoder-padding-seconds <5|10|30>
    \\                                     Encoder silence tail (default: {d} seconds).
    \\  --recording-seconds-max <1-65535>         Recording limit (default: {d} seconds).
    \\  --model-idle-seconds-max <seconds>        Idle retention (default: {d}; 0 disables).
    \\  --microphone-node <node-name>             Select a PipeWire source node.
    \\  --microphone-serial <serial>              Select a physical microphone.
    \\  --transcript-output <mode>                desktop (default), clipboard, or stdout.
    \\  --notification-mode <errors|off>          Error popups (default: errors; stdout disables).
    \\  --paste-shortcut <chord>                  ctrl+shift+v (default), ctrl+v, shift+insert.
    \\  --paste-settle-ms <0-65535>               Wait after clipboard acquisition (default: {d} ms).
    \\  --paste-key-gap-ms <0-65535>              Gap between key-event frames (default: {d} ms).
    \\  -h, --help                         Show this help.
    \\
    \\Config: $XDG_CONFIG_HOME/voiced/config or ~/.config/voiced/config.
    \\Use snake_case keys corresponding to flags, without the leading --.
    \\Blank lines and # comment lines are ignored. Values are literal.
    \\CLI settings override the file; restart the service after editing.
    \\Decoder threads cannot exceed the encoder count (the shared pool size).
    \\
    \\Models: Systran/faster-whisper-base.en, Systran/faster-whisper-small.en.
    \\Values accept '--model-encoder-threads 8' or '--model-encoder-threads=8'. Supply each option once.
    \\Choose at most one of --microphone-node and --microphone-serial; omitting both uses
    \\the default source. A trailing '--' ends options; no positional arguments
    \\are accepted. Idle retention accepts 0 through 4294967295 seconds.
    \\
    \\The service opens no microphone or model until a recording is requested.
    \\Desktop output requires wl-copy and access to /dev/uinput. If the keyboard
    \\is unavailable, copying still works. Clipboard mode sends no shortcut;
    \\stdout mode writes final text without contacting the desktop.
    \\Run 'voiced record -t' in another terminal to start recording.
    \\Use the same VOICED_INSTANCE in both terminals. Ctrl-C stops the daemon.
    \\
;

const record_help =
    \\Record audio manually, or toggle recording on and off.
    \\
    \\Examples:
    \\  voiced record              Start recording; finish with 'voiced stop'.
    \\  voiced record -t           Start if idle, stop if already recording.
    \\
    \\Usage: voiced record [-t | --toggle]
    \\
    \\Options:
    \\  -t, --toggle  Toggle recording; takes no value and may appear only once.
    \\  -h, --help    Show this help.
    \\
    \\Toggles during stopping, transcription, or delivery are ignored.
    \\The daemon must already be running; start it with 'voiced serve'.
    \\Use the same VOICED_INSTANCE in both terminals.
    \\Responses are written to stdout as JSON; errors go to stderr.
    \\
;
