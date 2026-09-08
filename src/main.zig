const std = @import("std");
const builtin = @import("builtin");
const setup = @import("setup.zig");
const voiced = @import("voiced.zig");
const logging = voiced.logging;
const log = logging.scoped(.service);
const service = voiced.service;
const stderr = std.debug.print;

// ─── Binary Size Optimizations ────────────────────────────────────────────────

// ── Crash Diagnostics ──
//
// `-Ddeveloper=false` omits in-process panic and fault stack traces. Runtime
// safety checks and normal logs remain; panics print a best-effort message to
// stderr and abort with SIGABRT. Keeping both panic and fault paths out of stack
// symbolization discards ELF/DWARF parsing, symbol lookup/sorting, unwind tables,
// and their diagnostic dependencies.
//
// Configure external core collection separately. GDB needs the exact unstripped
// build companion to resolve deployment addresses to source lines, but the
// deployment profile does not yet emit one. Stripping removes debug data, not
// the code that prints stack traces; the non-developer profile also removes that
// code through this compile-time value.
//
// Measured on this workstation: Intel Core i7-13700HX (x86-64), Ubuntu 24.04.3 LTS,
// Zig 0.16.0/LLVM, native PipeWire, ReleaseSafe + PIE. GNU-stripped size fell from
// 1,565,784 to 1,314,072 bytes (-251,712, or 16.1%).
const developer = @import("build_options").developer;

pub const std_options: std.Options = options: {
    var configured: std.Options = .{
        .allow_stack_tracing = developer,
        .enable_segfault_handler = developer and std.debug.default_enable_segfault_handler,
        // Service IPC, D-Bus, and PipeWire use native protocols, but the same
        // executable provides `voiced setup` through std.http.
        .networking = true,
    };
    // Thread entry allocates an alternate signal stack independently of the
    // segfault-handler setting. Keep the standard size only when that handler
    // is enabled by the developer profile.
    if (!developer) configured.signal_stack_size = null;
    break :options configured;
};

// Zig 0.16's self-hosted x86 backend can misalign globals in ReleaseSmall.
// An under-aligned stderr singleton makes checked debug printing panic
// recursively. The stdlib's root hook selects this explicitly aligned instance
// for debug output and crash diagnostics.
// https://codeberg.org/ziglang/zig/issues/36806
var stderr_io: std.Io.Threaded align(@alignOf(std.Io.Threaded)) = .init_single_threaded;
pub const std_options_debug_threaded_io = &stderr_io;

pub const panic = std.debug.FullPanic(if (developer) std.debug.defaultPanic else panicWithoutTrace);

fn panicWithoutTrace(message: []const u8, first_trace_addr: ?usize) noreturn {
    @branchHint(.cold);
    _ = first_trace_addr;
    const linux = std.os.linux;

    // A closed stderr pipe must not terminate us with SIGPIPE before abort can
    // request a core. No restoration is needed: this thread never resumes.
    var blocked = linux.sigemptyset();
    linux.sigaddset(&blocked, .PIPE);
    _ = linux.sigprocmask(linux.SIG.BLOCK, &blocked, null);

    // Best-effort raw writes avoid allocation, logging locks, and recursive
    // formatting during a panic. A failed write must not prevent termination.
    print: {
        for ([_][]const u8{ "voiced: panic: ", message, "\n" }) |part| {
            var remaining = part;
            while (remaining.len != 0) {
                const result = linux.write(2, remaining.ptr, remaining.len);
                switch (linux.errno(result)) {
                    .SUCCESS => {
                        if (result == 0) break :print;
                        remaining = remaining[result..];
                    },
                    .INTR => continue,
                    else => break :print,
                }
            }
        }
    }
    // Normal exit cannot produce a core; the host's collector and policy decide
    // whether this SIGABRT actually saves one. Keep the matching debug binary.
    std.process.abort();
}

// ─── Process Initialization ──────────────────────────────────────────────────

// PERFORMANCE: The full `std.process.Init` selects DebugAllocator for a
// libc-free ReleaseSafe program, retaining its diagnostics and allocation
// metadata in the resident daemon. Our process needs only the facilities below;
// Debug restores the diagnostic allocator while release modes use the
// thread-safe SMP allocator.
var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub fn main(minimal: std.process.Init.Minimal) u8 {
    const gpa = switch (builtin.mode) {
        .Debug => debug_allocator.allocator(),
        .ReleaseSafe, .ReleaseFast, .ReleaseSmall => std.heap.smp_allocator,
    };
    defer if (builtin.mode == .Debug) {
        _ = debug_allocator.deinit();
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var threaded = std.Io.Threaded.init(gpa, .{
        .argv0 = .init(minimal.args),
        .environ = minimal.environ,
    });
    defer threaded.deinit();

    var environ_map = std.process.Environ.createMap(minimal.environ, gpa) catch |err| {
        stderr("voiced: could not read environment ({s}).\n", .{@errorName(err)});
        return 1;
    };
    defer environ_map.deinit();

    return runCommandLine(.{
        .minimal = minimal,
        .arena = &arena,
        .gpa = gpa,
        .io = threaded.io(),
        .environ_map = &environ_map,
        .preopens = .empty,
    });
}

// ─── Command Dispatch ─────────────────────────────────────────────────────────

fn runCommandLine(init: std.process.Init) u8 {
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
        var help_buffer: [16384]u8 = undefined;
        const text = if (invocation == .usage)
            main_help
        else text: {
            const command = invocation.help;
            if (command.len == 0) {
                break :text main_help;
            }

            if (std.mem.eql(u8, command, "serve")) {
                const defaults: service.Options = .{};

                break :text std.fmt.bufPrint(&help_buffer, serve_help, .{
                    defaults.capture.recording_seconds,
                    defaults.capture.transcription.model.name(),
                    defaults.capture.transcription.inference_threads_count,
                    @as(u8, switch (defaults.capture.transcription.encoder_trailing_padding) {
                        .seconds_5 => 5,
                        .seconds_10 => 10,
                        .seconds_30 => 30,
                    }),
                    defaults.model_keep_warm_seconds,
                    @tagName(defaults.clipboard_backend),
                    @tagName(defaults.output),
                    @tagName(defaults.notification_mode),
                    @tagName(defaults.paste_key),
                    defaults.paste_settle_ms,
                    defaults.paste_key_gap_ms,
                    defaults.paste_observation_ms,
                    @tagName(defaults.log_target),
                }) catch unreachable;
            }

            if (std.mem.eql(u8, command, "record")) {
                break :text record_help;
            }

            if (std.mem.eql(u8, command, "setup")) {
                break :text setup_help;
            }

            if (std.mem.eql(u8, command, "status")) {
                break :text status_help;
            }

            const client_command = std.meta.stringToEnum(@FieldType(service.Request, "cmd"), command).?;

            const description = switch (client_command) {
                .stop => "Stop recording and finish transcribing the captured audio.",
                .cancel => "Cancel the current recording and discard its transcription.",
                .status => "Show daemon, recording, and model status as text.",
                .kill => "Shut down the daemon and its workers.",
                .record => unreachable,
            };

            break :text std.fmt.bufPrint(
                &help_buffer,
                "{s}\n\nUsage:\n  voiced {s}\n\n" ++
                    "The daemon must already be running; start it with 'voiced serve'.\n",
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
    if (invocation == .serve) {
        const errno = logging.init(init.io, invocation.serve.log_target, invocation.serve.log_level);
        if (errno != .SUCCESS) {
            stderr("voiced: could not initialize logging (errno={f}).\n", .{logging.fmtErrno(errno)});
            return 1;
        }
    }
    defer logging.deinit();

    const execution = switch (invocation) {
        .setup => setup.run(init),
        .serve => |options| service.run(init, options),
        .client => |request| service.send(init, request),
        .usage, .help => unreachable,
    };

    execution catch |err| {
        if (invocation == .serve) {
            switch (err) {
                error.ClipboardBackendUnavailable => {}, // The supervisor logged the typed clipboard failure.
                error.DaemonAlreadyRunning, error.RuntimeDirectoryNotSet, error.RuntimeDirectoryNotAbsolute, error.UnsafeRuntimeDirectory, error.UnsafeControlSocket, error.SocketPathTooLong => log.event(.err, .{}, "service_startup_refused", &.{.{ "error", .{ .verbatim = @errorName(err) } }}),
                else => log.event(.critical, .{}, "service_failed", &.{.{ "error", .{ .verbatim = @errorName(err) } }}),
            }
            return 1;
        }
        if (invocation == .setup) {
            switch (err) {
                error.UnexpectedHttpStatus, error.DownloadSizeMismatch, error.DownloadHashMismatch => {},
                else => stderr("voiced: could not install models ({s}).\n", .{@errorName(err)}),
            }
            return 1;
        }
        switch (err) {
            error.DaemonNotRunning => stderr("voiced: cannot connect to the daemon. Start 'voiced serve'.\n", .{}),
            error.DaemonAlreadyRunning => stderr("voiced: the daemon is already running. Use 'voiced status'.\n", .{}),
            error.RuntimeDirectoryNotSet, error.RuntimeDirectoryNotAbsolute => stderr("voiced: XDG_RUNTIME_DIR must name an absolute runtime directory for your user session.\n", .{}),
            error.UnsafeRuntimeDirectory => stderr("voiced: XDG_RUNTIME_DIR and its voiced directory must be owned by your user with no group or other permissions (typically 0700).\n", .{}),
            error.CommandRejected => stderr("voiced: the daemon rejected the command.\n", .{}),
            error.IncompatibleControlProtocol => stderr("voiced: incompatible control protocol. Restart the daemon using the same version as this CLI.\n", .{}),
            error.InvalidControlReply => stderr("voiced: invalid reply from the daemon. Check its version and status before retrying a recording command.\n", .{}),
            error.ControlSendFailed, error.ControlReceiveFailed => stderr("voiced: communication with the daemon failed or timed out. Check 'voiced status' before retrying a recording command.\n", .{}),
            else => stderr("voiced: could not complete the command ({s}).\n", .{@errorName(err)}),
        }

        return 1;
    };

    if (invocation == .serve) log.event(.info, .{}, "service_stopped", &.{});
    return 0;
}

const Invocation = union(enum) {
    usage,
    help: []const u8,
    setup,
    serve: service.Options,
    client: service.Request,
};

const ArgumentDiagnostic = service.ConfigDiagnostic;

fn parseCommand(init: std.process.Init, arguments: []const [:0]const u8, diagnostic: *ArgumentDiagnostic) !Invocation {
    if (arguments.len == 0) {
        return .usage;
    }

    const command = arguments[0];

    // Scan help before validating other flags, but respect `--` as the end of
    // options. An explicit `--microphone-node=--help` remains a value, not a help request.
    for (arguments) |argument| {
        if (std.mem.eql(u8, argument, "--")) {
            break;
        }
        if (std.mem.eql(u8, argument, "-h") or std.mem.eql(u8, argument, "--help")) {
            const topic = if (std.mem.eql(u8, command, "setup") or
                std.mem.eql(u8, command, "serve") or
                std.meta.stringToEnum(@FieldType(service.Request, "cmd"), command) != null)
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

        if (!std.mem.eql(u8, topic, "setup") and
            !std.mem.eql(u8, topic, "serve") and
            std.meta.stringToEnum(@FieldType(service.Request, "cmd"), topic) == null)
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
        return .{ .serve = try service.load(init, arguments[1..], diagnostic) };
    }

    if (std.mem.eql(u8, command, "setup")) {
        diagnostic.command = command;
        if (arguments.len > 1) {
            diagnostic.argument = arguments[1];
            if (std.mem.eql(u8, diagnostic.argument, "--")) {
                if (arguments.len == 2) return .setup;
                diagnostic.argument = arguments[2];
                return error.UnexpectedArgument;
            }
            if (std.mem.startsWith(u8, diagnostic.argument, "-")) return error.UnknownOption;
            return error.UnexpectedArgument;
        }
        return .setup;
    }

    diagnostic.argument = command;

    const client_command = std.meta.stringToEnum(@FieldType(service.Request, "cmd"), command) orelse {
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
    var request: service.Request = .{ .cmd = client_command };
    var index: usize = 1;
    while (index < arguments.len) : (index += 1) {
        const argument = arguments[index];
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
            if (request.toggle) return error.DuplicateOption;
            request.toggle = true;
            continue;
        }

        if (client_command == .record and std.mem.startsWith(u8, argument, "--toggle=")) {
            diagnostic.argument = "--toggle";
            return error.OptionTakesNoValue;
        }

        if (std.mem.startsWith(u8, argument, "-")) return error.UnknownOption;
        return error.UnexpectedArgument;
    }

    return .{ .client = request };
}

const main_help =
    \\voiced - voice dictation daemon
    \\
    \\Usage: voiced <command> [options]
    \\
    \\Commands:
    \\  setup    Install and verify the supported models.
    \\  serve    Run the daemon in the foreground.
    \\  record   Record manually; -t or --toggle toggles recording.
    \\  stop     Stop recording and finish transcription.
    \\  cancel   Discard the current recording.
    \\  status   Print daemon status as text.
    \\  kill     Shut down the daemon.
    \\  help     Show help, optionally for a command (voiced help <command>)
    \\
    \\Global options:
    \\  -h, --help  Show help without executing a command.
    \\              Works in any position and always has the highest precedence.
    \\
    \\Output:
    \\  'voiced status' and 'voiced serve --transcript-output stdout' write their
    \\  requested data to stdout.
    \\
    \\  Successful record, stop, cancel, and kill commands are silent on stdout and
    \\  write one brief acknowledgement to stderr.
    \\
    \\  All diagnostics and error output goes to stderr.
    \\
    \\Exit codes:
    \\  0  Success or help.
    \\  1  Operational failure.
    \\  2  Command-line or configuration error.
    \\
    \\Documentation and issues:
    \\  https://github.com/calvinalkan/voiced
    \\
;

const setup_help =
    \\Install every Whisper speech-recognition model supported by this build.
    \\
    \\Installed models:
    \\  whisper.base.en
    \\  whisper.small.en
    \\  whisper.medium.en
    \\
    \\Each model is built from weights and a vocabulary downloaded from a pinned
    \\Hugging Face revision. Existing models are reused when their packed contents
    \\and source digests remain valid.
    \\
    \\Setup downloads about 2.1 GiB. The completed installation uses about 1.1 GiB.
    \\
    \\Storage:
    \\  $XDG_DATA_HOME/voiced/models/whisper.base.en.voiced
    \\  $XDG_DATA_HOME/voiced/models/whisper.small.en.voiced
    \\  $XDG_DATA_HOME/voiced/models/whisper.medium.en.voiced
    \\
    \\  When XDG_DATA_HOME is unset:
    \\  $HOME/.local/share/voiced/models/whisper.base.en.voiced
    \\  $HOME/.local/share/voiced/models/whisper.small.en.voiced
    \\  $HOME/.local/share/voiced/models/whisper.medium.en.voiced
    \\
;

const serve_help =
    \\Run the voice dictation daemon in the foreground.
    \\
    \\Voiced opens no microphone or model until a recording is requested.
    \\
    \\Examples:
    \\  voiced serve
    \\  voiced serve --microphone-serial DEVICE_SERIAL
    \\  voiced serve --transcript-output stdout --log-level debug
    \\
    \\Usage: voiced serve [options]
    \\
    \\Recording:
    \\  --recording-seconds-max <1-65535>
    \\      Stop capture and process the recording after this many seconds
    \\      (default: {d}).
    \\
    \\Microphone:
    \\  --microphone-serial <serial>
    \\      Require exactly one source belonging to this physical PipeWire device.
    \\  --microphone-node <node-name>
    \\      Require exactly one source with this PipeWire node.name.
    \\
    \\Transcription:
    \\  --model <model>
    \\      Select an installed English Whisper model (default:
    \\      {s}).
    \\  --model-encoder-threads <count>
    \\      Set the shared inference pool size: one OS thread per worker
    \\      (positive integer; default: {d}).
    \\  --model-decoder-threads <count>
    \\      Use up to this many workers from the same pool during decoding
    \\      (default: all encoder workers).
    \\  --model-encoder-padding-seconds <5|10|30>
    \\      Append normalized silence to each inference chunk (default: {d} seconds).
    \\  --model-idle-seconds-max <seconds>
    \\      Keep the loaded model and worker pool after a successful recording
    \\      (default: {d}; 0 unloads them after every recording).
    \\
    \\Transcript delivery:
    \\  --clipboard-backend <auto|wayland|x11>
    \\      Select automatic hierarchy, Wayland only, or X11 only (default: {s}).
    \\  --transcript-output <desktop|clipboard|stdout>
    \\      Choose paste, clipboard-only, or terminal output (default: {s}).
    \\  --notification-mode <errors|off>
    \\      Show desktop popups for operational failures or disable them
    \\      (default: {s}; stdout output disables them).
    \\  --paste-shortcut <ctrl+shift+v|ctrl+v|shift+insert>
    \\      Select the shortcut injected in desktop mode (default: {s}).
    \\  --paste-settle-ms <0-65535>
    \\      Wait after acquiring the clipboard before injecting the shortcut
    \\      (default: {d} ms).
    \\  --paste-key-gap-ms <0-65535>
    \\      Wait between the shortcut's press and release stages (default: {d} ms).
    \\  --paste-observation-ms <0-65535>
    \\      Wait this long for a post-shortcut clipboard transfer before saving
    \\      (default: {d} ms; 0 disables observation and its warning).
    \\
    \\Diagnostics and configuration:
    \\  --log-level <critical|error|warn|info|debug>
    \\      Emit diagnostics at this severity and above (default: info).
    \\  --log-target <auto|journal|stderr>
    \\      Select the diagnostic destination (default: {s}).
    \\  --config <path>
    \\      Read this configuration file instead of the default location.
    \\
    \\Microphone selection:
    \\  Without a microphone option, every recording resolves PipeWire's current
    \\  default source. --microphone-serial matches a physical Device's device.serial;
    \\  --microphone-node instead matches one exact source node's node.name. The two
    \\  options are mutually exclusive.
    \\
    \\  Each recording resolves the configured value, which must match exactly one
    \\  source. A missing or ambiguous source fails the recording; Voiced never
    \\  silently falls back to another microphone. A serial can be ambiguous when one
    \\  physical device exposes multiple source nodes; use --microphone-node to choose
    \\  one of them. If the selected source disappears or changes during capture,
    \\  Voiced stops capture instead of continuing from a different source.
    \\
    \\  List PipeWire devices and source node names:
    \\    wpctl status --name
    \\  For --microphone-node, copy a name under Audio > Sources. For
    \\  --microphone-serial, choose its ID under Audio > Devices and run:
    \\    pw-cli info 55 | grep 'device.serial'
    \\  Replace 55 with that device ID and copy only the value inside quotes. PipeWire
    \\  IDs can change; do not store the numeric ID in the Voiced configuration.
    \\
    \\PipeWire connection:
    \\  PIPEWIRE_RUNTIME_DIR selects the socket directory and falls back to
    \\  XDG_RUNTIME_DIR. PIPEWIRE_REMOTE selects the server (default: pipewire-0).
    \\
    \\Transcript delivery:
    \\  desktop    Own the clipboard, inject the configured shortcut into the focused
    \\             application, and save the transcript (default).
    \\  clipboard  Own the clipboard and save the transcript without pressing keys.
    \\  stdout     Write the transcript to stdout without clipboard ownership, key
    \\             injection, transcript saving, or desktop notifications.
    \\
    \\  Desktop and clipboard modes save accepted text to
    \\  $XDG_STATE_HOME/voiced/transcript.txt or ~/.local/state/voiced/transcript.txt.
    \\
    \\Desktop integration:
    \\  Desktop output requires Wayland or local X11 plus access to /dev/uinput.
    \\  Automatic clipboard selection tries Wayland ext-data-control, Wayland
    \\  wlr-data-control, X11, then core Wayland. An explicit backend never falls
    \\  through and must initialize successfully at startup. WAYLAND_DISPLAY selects
    \\  Wayland; when unset, Voiced uses wayland-0 if its socket exists. DISPLAY
    \\  selects local X11, and XAUTHORITY selects its authority file (default:
    \\  ~/.Xauthority). Stdout output initializes neither clipboard backend.
    \\
    \\  If keyboard setup fails, clipboard delivery and transcript saving can still
    \\  succeed. paste_outcome=sent means Linux accepted the shortcut, not that the
    \\  focused application inserted the text. Voiced waits for a post-shortcut
    \\  clipboard transfer before saving; a missing transfer emits a diagnostic
    \\  warning but never retries. Physically held modifier keys can also
    \\  affect the shortcut.
    \\
    \\Notifications:
    \\  errors shows popups for failures such as a missing microphone, incomplete
    \\  recording, failed transcription, or clipboard, paste, and save failures.
    \\  Successful recording, progress, ordinary silence, and cancellation produce no
    \\  popup. off disables the notification connection. Notifications do not change
    \\  which diagnostics are logged. DBUS_SESSION_BUS_ADDRESS selects the Unix
    \\  session bus; when unset, Voiced uses $XDG_RUNTIME_DIR/bus.
    \\
    \\Model behavior:
    \\  Models: whisper.base.en, whisper.small.en, whisper.medium.en.
    \\  Model loading begins alongside capture. Encoder and decoder work share one
    \\  worker pool, so decoder threads cannot exceed encoder threads.
    \\  Counts may exceed the CPU count; excessive threads can increase contention
    \\  and scheduling overhead and slow transcription down.
    \\
    \\  Encoder padding adds normalized silence after each audio chunk; it does not
    \\  change microphone buffering or the recording limit. The complete Whisper input
    \\  remains capped at 30 seconds; longer recordings use chunked inference.
    \\  After successful delivery, the idle limit controls how long mapped weights,
    \\  workspace, and worker threads remain resident. Cancellation unloads the model.
    \\
    \\Logging:
    \\  Help is written to stdout. Invalid command lines and failures before logging
    \\  initializes are written to stderr. After initialization, daemon diagnostics go
    \\  to the selected target. Transcript contents and token IDs are never logged.
    \\
    \\  Levels are critical, error, warn, info, and debug. The selected level includes
    \\  all more severe events; warn therefore includes warn, error, and critical.
    \\  auto selects stderr when stderr is a terminal and journal otherwise. journal
    \\  emits native fields including PRIORITY, VOICED_COMPONENT, VOICED_EVENT, and a
    \\  VOICED_RECORDING_ID on recording events. stderr writes one readable line per
    \\  event. Its writes are synchronous, so keep a redirected pipe drained.
    \\
    \\Journal examples:
    \\  journalctl --user -u voiced -f -o short-precise
    \\  journalctl --user -u voiced -p warning
    \\  journalctl --user -u voiced VOICED_EVENT=recording_finished
    \\  journalctl --user -u voiced VOICED_COMPONENT=capture VOICED_RECORDING_ID=21
    \\  # Follow foreground 'voiced serve --log-target journal':
    \\  journalctl --user -t voiced -f
    \\
    \\Configuration:
    \\  The default file is $XDG_CONFIG_HOME/voiced/config, falling back to
    \\  ~/.config/voiced/config. A missing default file is allowed; a path supplied
    \\  with --config must exist. Service options use corresponding snake_case keys.
    \\  Blank lines and # comment lines are ignored; values are literal.
    \\
    \\  Command-line settings override file settings. A command-line microphone choice
    \\  replaces the file's choice. Values accept either '--option value' or
    \\  '--option=value'; supply each option only once per input. A trailing '--' ends
    \\  options, and serve accepts no positional arguments. Restart the service after
    \\  changing its configuration.
    \\
    \\Service lifecycle:
    \\  Run 'voiced record -t' in another terminal to start recording.
    \\  Ctrl-C stops the daemon gracefully.
    \\
    \\Documentation and issues:
    \\  https://github.com/calvinalkan/voiced
    \\
;

const status_help =
    \\Show the daemon's current recording and model state.
    \\
    \\Examples:
    \\  voiced status
    \\  voiced status | grep '^phase='
    \\  voiced status | grep '^model_state='
    \\
    \\Output:
    \\  One key=value field per line. Values contain no spaces.
    \\
    \\  phase=<idle|capturing|stopping|transcribing|delivering>
    \\      Current recording workflow:
    \\        idle          Ready for another recording.
    \\        capturing     Opening or reading the selected microphone.
    \\        stopping      Stopping or cancelling active work.
    \\        transcribing  Processing captured audio.
    \\        delivering    Copying, pasting, or saving the transcript.
    \\
    \\  recording_id=<integer>
    \\      Identifies the current recording, or the most recently started recording
    \\      while idle. A newly started daemon reports 0. The first recording uses 1,
    \\      and each subsequent recording increments the value.
    \\
    \\      Find its journal records with:
    \\        journalctl --user -u voiced VOICED_RECORDING_ID=127
    \\
    \\  recording_elapsed_seconds=<integer|unavailable>
    \\      Whole seconds since the current recording was requested. Available only
    \\      while phase=capturing.
    \\
    \\  model_state=<unloaded|loading|loaded|unloading>
    \\      Current model lifecycle:
    \\        unloaded   The next recording must load the model.
    \\        loading    Model weights and inference workers are starting.
    \\        loaded     The model and workers are resident.
    \\        unloading  Model resources and workers are being released.
    \\
    \\  model_idle_seconds_remaining=<integer|unavailable>
    \\      Whole seconds before a loaded model unloads. Unavailable means no idle
    \\      countdown is currently active.
    \\
    \\  model_idle_seconds_max=<integer>
    \\      Configured idle retention limit. Zero unloads the model after every
    \\      recording.
    \\
    \\  daemon_uptime_seconds=<integer>
    \\      Whole seconds since this daemon started.
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
    \\
    \\Toggles during stopping, transcription, or delivery are ignored.
    \\The daemon must already be running; start it with 'voiced serve'.
    \\
;
