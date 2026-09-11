# Voiced Zig daemon

The native executable contains the daemon and its CLI client. One process owns
three persistent threads: the supervisor, capture, and transcription coordinator.
The resident runtime adds the configured inference pool while a model is loaded.

```text
src/main.zig
├── serve → service.run                    main-thread epoll loop
│            ├── service/control.zig       public commands and status
│            ├── capture.zig               persistent capture thread
│            │   ├── capture/pipewire.zig  stream ownership and graph cycles
│            │   └── audio_exchange.zig    three reusable Float32 slots
│            ├── transcription.zig         persistent model coordinator thread
│            │   ├── packed_model/reader.zig validated mapping and model views
│            │   └── inference/root.zig    Whisper runtime and shared worker pool
│            ├── worker.zig                typed job/result mailboxes and wakeups
│            ├── clipboard.zig             Wayland/X11 clipboard selection
│            │   ├── clipboard/wayland.zig native Wayland ownership and transfers
│            │   └── clipboard/x11.zig     native X11 ownership and transfers
│            └── paste.zig                 native uinput keyboard
├── setup → setup.zig → packed_model/writer.zig CTranslate2 conversion and installation
└── record / stop / cancel / status / kill → control socket
```

Each worker has one fixed job/result mailbox and an eventfd wakeup. Completion
releases its borrowed data before the supervisor submits another job. Capture
has a separate atomic stop command; transcription checks cancellation collectively
between encoder layers and decoder steps. Capture alone requests realtime
scheduling. Native clipboard and notification I/O remain on the main event loop.

No internal exec roles, launch packets, descriptor handoff, child processes,
or pidfds are needed. A worker crash or overdue cancellation exits the entire
daemon; `Restart=on-failure` restarts it. An in-progress recording may be lost.
The supervisor never frees or reuses storage while a worker might still access it.

## Build and install models

All executable targets explicitly use static linkage and PIE: address
randomization remains enabled, with no ELF interpreter or shared-library
dependencies. Native Zig clients speak
PipeWire and D-Bus directly; neither libc nor client development headers are
required. Inference uses host-targeted
AVX-VNNI kernels; this is not a portable baseline binary. There is
no external inference engine or C++ bridge. The checkpoint importer reads the
published CTranslate2 serialization format, not CTranslate2 code or libraries.

```bash
sudo apt install pipewire
zig build
./zig-out/bin/voiced setup
```

Setup downloads the pinned Systran weights and vocabularies from Hugging Face,
verifies their sizes and BLAKE3-256 pins, and converts them into the native files
`whisper.base.en.voiced`, `whisper.small.en.voiced`, and
`whisper.medium.en.voiced`. An existing file is reused only when it loads as the
requested model under the runtime's current packed format; otherwise setup
atomically replaces it. Source downloads are temporary. The complete installed
files use about 1.1 GiB under
`$XDG_DATA_HOME/voiced/models`, or `~/.local/share/voiced/models` when the XDG
path is unset. `zig build setup` builds Voiced and invokes the same command;
ordinary builds neither download models nor run inference.

Plain `zig build` creates a developer build and defaults to ReleaseSafe for the
application, inference runtime, and stdlib. It retains symbols, in-process panic
and fault stack traces, lifecycle assertions, and Zig runtime safety checks.
`-Doptimize` selects the application mode. `-Doptimize-inference-runtime` and
`-Doptimize-stdlib` optionally override the complete inference and Zig standard
library modules; each otherwise follows the application mode. ReleaseSafe
inference keeps the projection kernel's entry assertions. Its validated k4
depth loop uses a proven packed offset instead of repeating per-update layout
assertions and disables generated bounds and overflow checks. Debug restores
the generated loop checks. `inference/linear.zig` owns the complete safety and
performance fence.

Use fully Debug code only when its unoptimized execution is acceptable:

```bash
zig build -Doptimize=Debug
```

For ordinary application debugging without unoptimized inference, retain
ReleaseSafe inference:

```bash
zig build -Doptimize=Debug -Doptimize-inference-runtime=ReleaseSafe
```

`-Ddeveloper=false` selects the closed deployment profile. It rejects Debug for
any optimization setting, always emits a stripped static PIE, omits
in-process crash diagnostics and runtime unwind tables, and makes the sub-1-MiB
size gate part of the ordinary install step. ReleaseFast remains opt-in rather
than the default production policy.

The compact deployment size-optimizes Voiced and stdlib, while keeping the
inference runtime speed-optimized:

```bash
zig build -Ddeveloper=false \
  -Doptimize=ReleaseSmall \
  -Doptimize-inference-runtime=ReleaseFast
```

The deployment build checks its installed daemon automatically. The named step
also works from a developer build by compiling a deployment-equivalent stripped
daemon with the selected non-Debug optimization modes:

```bash
zig build size-check \
  -Doptimize=ReleaseSmall \
  -Doptimize-inference-runtime=ReleaseFast \
  -Doptimize-stdlib=ReleaseSmall
```

The named step reports the effective deployment flags and rejects Debug
application, inference, or stdlib modes because their unoptimized diagnostic
code is not a deployment-size target.

The developer profile changes build-time diagnostics, not normal CLI output or
operational logs. In a deployment build, panics print a best-effort message to
stderr and abort with SIGABRT; memory faults use the OS signal handling instead
of Zig's rich fault handler. Neither path prints an in-process stack trace.
Zig's native link-time stripping does not emit a separate debug companion.

Before using the deployment profile, verify core collection for the actual
service; an abort does not guarantee a saved core. Ubuntu may use Apport rather
than systemd-coredump. Source-level analysis requires an exact unstripped build
companion, which this profile does not yet emit. Cores can contain audio,
transcripts, and other process memory: restrict access and retention. The build
does not change the host's collector configuration.

The `Binary Size Optimizations` section in `src/main.zig` owns the daemon's Zig
root configuration. Unused `std.Io` networking is disabled in every build mode.
Raw Unix sockets and PipeWire remain available, and the separate model-setup tool
keeps networking.

## Zig linter

Build the native linter and pass it one file or directory:

```bash
zig build zig-lint
./zig-out/bin/zig-lint src
```

For a directory inside a Git repository, `zig-lint` reads each regular,
non-symlinked `.gitignore` of at most 120 KiB from the nearest `.git` boundary
through the requested scan root and included descendants. Without a Git boundary,
ignore lookup begins at the scan root. An ignored scan root produces no files.
Other ignored directories are pruned before traversal, so a descendant rule
cannot revive anything below an excluded parent.

By default, the linter also excludes these exact descendant directory names at
every depth, independently of `.gitignore` rules:

```text
.git/
.hg/
.svn/
.jj/
.zig-cache/
zig-cache/
zig-out/
zig-pkg/
.cache/
node_modules/
.venv/
__fixtures__/
```

The native API's `LintOptions.excluded_dir_names` replaces this list; it
does not extend it. Pass an empty slice to disable directory-name exclusions,
or combine custom names with
`LintOptions.excluded_dir_names_default`. `respect_gitignore` separately
controls whether directory scans read `.gitignore` files.

Gitignore matching is pattern-based and does not inspect Git's index or status.
A tracked path that matches an ignore pattern is therefore excluded from a
directory scan. An explicit file argument is checked regardless of its extension,
surrounding ignore files, or excluded-directory configuration.

Report paths default to paths relative to the process working directory.
`LintOptions.report_path_format` can instead make diagnostic and fixed-file paths
relative to the nearest enclosing Git root or absolute. Git-root-relative output
recognizes both `.git` directories and worktree or submodule `.git` files, and
falls back to working-directory-relative output when no Git root exists.

`LintOptions.target_zig_version` tells built-in and plugin rules which Zig
language and standard-library contract to target and defaults to the toolchain
that built the linter. It does not switch the linter's parser or formatter.
For cached directory scans, changing it establishes a distinct lint-cache
identity. See the
[source language contract](docs/plugins.md#source-language-contract) for the
complete rule and plugin semantics.

### Native lint plugins

Plugins are trusted native code and are loaded only from explicit paths. See
[`docs/plugins.md`](docs/plugins.md) for registration, lifecycle, ABI
compatibility, Linux host requirements, and a complete Zig plugin.

## Run

```bash
./zig-out/bin/voiced serve

# Run commands from another terminal:
./zig-out/bin/voiced record -t
./zig-out/bin/voiced stop
./zig-out/bin/voiced status
./zig-out/bin/voiced cancel
./zig-out/bin/voiced kill
```

The service opens no microphone or model at startup. A recording starts capture
and model loading concurrently. Small.en and four inference threads are the
defaults. Service options are:

| Option | Meaning |
| --- | --- |
| `--log-level <critical\|error\|warn\|info\|debug>` | Diagnostic threshold; default `info` |
| `--log-target <auto\|journal\|stderr>` | Diagnostic destination; default `auto` selects stderr in a terminal and journal otherwise |
| `--model <name>` | `whisper.base.en`, `whisper.small.en`, or `whisper.medium.en` |
| `--model-encoder-threads <count>` | Positive inference pool size; one OS thread per worker |
| `--model-decoder-threads <count>` | Positive decoder limit within the same pool; omitted uses the whole pool |
| `--model-encoder-padding-seconds <5\|10\|30>` | Encoder silence appended after each audio chunk; default 10 seconds, with total input capped at 30 seconds |
| `--recording-seconds-max <1-65535>` | Maximum recording duration; default 3,600 seconds |
| `--model-idle-seconds-max <seconds>` | Idle retention interval; default 300, zero disables retention |
| `--microphone-serial <serial>` | Select exactly one source by stable physical Device identity |
| `--microphone-node <node-name>` | Select an explicit PipeWire source; mutually exclusive with Device serial |
| `--clipboard-backend <auto\|wayland\|x11>` | Select the automatic hierarchy (default), Wayland only, or X11 only |
| `--transcript-output <mode>` | `desktop` copies, pastes, and saves (default); `clipboard` copies and saves; `stdout` prints without desktop delivery or saving |
| `--notification-mode <errors\|off>` | Show desktop notifications for operational failures only; default `errors`. Transcript `stdout` disables them |
| `--paste-shortcut <chord>` | `ctrl+shift+v` (default), `ctrl+v`, or `shift+insert`; used in desktop mode |
| `--paste-settle-ms <0-65535>` | Wait after clipboard acquisition before pasting; default 10 ms |
| `--paste-key-gap-ms <0-65535>` | Gap between the four paste key-event frames; default 4 ms |
| `--paste-observation-ms <0-65535>` | Wait for a post-shortcut clipboard transfer before saving; default 250 ms, zero disables observation and its warning |

All service settings also accept snake_case keys in
`$XDG_CONFIG_HOME/voiced/config` (or `~/.config/voiced/config`). For example:

```ini
# Example tuned worker counts; the built-in default remains four.
log_level=info
log_target=auto
model=whisper.small.en
model_encoder_threads=16
model_decoder_threads=8
model_encoder_padding_seconds=10
model_idle_seconds_max=300
recording_seconds_max=3600
clipboard_backend=auto
transcript_output=desktop
paste_shortcut=ctrl+shift+v
paste_settle_ms=10
paste_key_gap_ms=4
paste_observation_ms=250
# microphone_serial=ABC123
```

Use `voiced serve --config PATH` to select another file. A missing default file
uses built-in defaults; an explicitly selected file must exist. The file is read
once at startup and limited to 64 KiB. Restart the service after edits.

Each line is `key=value`, split at the first `=`. Blank lines and lines starting
with `#` after whitespace are ignored. Surrounding whitespace is trimmed; values
are literal, without quotes, escaping, interpolation, or inline comments. A `#`
inside a value is preserved. Unknown keys, duplicate settings, and invalid values
are errors; file syntax and value errors include the filename and line number.

CLI settings override file settings. A CLI microphone choice replaces the file's choice;
specifying both node and serial within one input is an error. Decoder
threads must not exceed encoder threads after overrides are applied. Both phases
share one pool; workers outside the decoder subset park during decoding.
Counts may exceed the CPU count, but excessive threads can increase contention
and scheduling overhead and slow transcription down.
`--config` and command actions such as `record --toggle` are not file settings.

Desktop mode opens one virtual keyboard through `/dev/uinput` and keeps it
across dictations. It needs the user's existing uinput permissions; Voiced does
not change device permissions or run as root. Missing access is logged and
delivery continues with clipboard only. Choose `ctrl+v` for applications using
that paste shortcut. The shortcut goes to the application focused at delivery
time; physically held modifier keys can affect it.

After clipboard/paste finishes, desktop and clipboard modes save the final UTF-8
text to `$XDG_STATE_HOME/voiced/transcript.txt`, defaulting to
`~/.local/state/voiced/transcript.txt`.
The file is replaced atomically, with no added newline; rejected, empty, and
cancelled recordings preserve the previous file. Clipboard/paste failures still
attempt saving. Save errors are logged without undoing desktop delivery.

The destination path is resolved and allocated once at service startup. Saving
reuses that path and the existing transcript buffer, with no per-save path
allocations or environment lookups. It uses a direct temporary-file write and
rename, without fsync. It happens after delivery and logs its elapsed time. This
avoids a save worker, but slow storage can block commands and power-loss durability
is not guaranteed. Diagnostic stdout mode does not update the file.

Failed model-worker transcriptions log the exact stage and Zig error name. When decoding
started, logs also include chunk duration, token count and limit, encoder positions,
model probabilities, and phase timings. Audio and transcript contents are never logged.

For a diagnostic run that does not copy or press keys:

```bash
./zig-out/bin/voiced serve --transcript-output stdout
```

To use a 10-second encoder silence tail, start the service with
`voiced serve --model-encoder-padding-seconds 10`. This changes Whisper's
trailing padding, not microphone buffering or the recording limit.

Run `voiced --help`, `voiced help serve`, or `voiced record --help` for help
and examples. Help writes to stdout and does not contact the daemon. A bare
`voiced` prints a short introduction. `-h` and `--help` take precedence over
other arguments before the `--` end-of-options marker.

Service values accept both `--model-encoder-threads 8` and `--model-encoder-threads=8`. Each option may
appear only once, even if repeated values are identical; `record -t --toggle`
is also a duplicate. Source flags are mutually exclusive. Invalid commands,
unknown flags, missing or invalid values, duplicates, and conflicts produce a
specific message on stderr with a help hint. Exit status is 0 for success or
help, 1 for operational failures, and 2 for command-line errors.

Without a source option, each recording resolves the current PipeWire default.
`--microphone-serial` matches a physical Device's `device.serial`, while
`--microphone-node` matches one source node's exact `node.name`; the options are
mutually exclusive. A serial is ambiguous when one physical device exposes
multiple source nodes, in which case selecting a node disambiguates it. A
configured source that is absent or ambiguous fails the recording rather than
silently falling back to another microphone. Source removal, replacement, or
timeline discontinuity during capture stops the stream and retains its valid
published prefix for transcription.

List devices and source node names with `wpctl status --name`. A source name
under **Audio > Sources** is a `--microphone-node` value. To obtain a physical
device's serial, use its numeric ID under **Audio > Devices** with `pw-cli`:

```bash
pw-cli info 55 | grep 'device.serial' # Replace 55 with the displayed device ID.
```

Copy only the value inside quotes. PipeWire's numeric IDs can change and are not
valid Voiced microphone identifiers.

Capture requests realtime processing. Missing scheduler promotion or memory
locking does not invalidate capture; deadlines still contain an unresponsive
worker. The shared exchange is touched once before threads start and locked
when capture begins, when permitted.

## Lifecycle and output

The supervisor is the only owner of session transitions, absolute deadlines,
slot release, and final transcript acceptance. Workers publish typed completion
before the main loop can reuse their audio or text. Coalesced wakeups carry no
independent ownership state.

Commands use one fixed binary request/reply over a Unix `SOCK_SEQPACKET`
connection at `$XDG_RUNTIME_DIR/voiced/control.sock`. The directory is mode
`0700`, the socket `0600`. The server allows 16 clients with a one-second
connection deadline; the CLI has three-second socket I/O timeouts.

Successful `record`, `stop`, `cancel`, and `kill` commands print no stdout and
write one brief human acknowledgement to stderr. These messages are not a
versioned machine interface. `status` writes one grep-friendly `key=value` field
per line to stdout and no acknowledgement, for example:

```text
phase=idle
recording_id=127
recording_elapsed_seconds=unavailable
model=whisper.small.en
model_state=loaded
model_idle_seconds_remaining=241
model_idle_seconds_max=300
daemon_uptime_seconds=15237
```

`phase` is the recording workflow: `idle`, `capturing`, `stopping`,
`transcribing`, or `delivering`. `model` is the configured Whisper model this
daemon loads. `model_state` is independently `unloaded`, `loading`, `loaded`, or
`unloading`.
`recording_id` identifies the current recording, or the latest recording while
idle, and matches `VOICED_RECORDING_ID` in journal records. It resets to zero
when the daemon starts. Elapsed recording time is available only during capture;
the model idle countdown is available only while an idle loaded model awaits its
unload deadline. `daemon_uptime_seconds` confirms whether the service restarted.
Run `voiced status --help` for every field and enum value.

Toggles during stopping, transcription, or delivery are ignored and still exit
successfully. Errors go to stderr with a nonzero exit code. An acknowledgement
means the supervisor handled the command, not that recording, transcription,
or shutdown has finished. After a timeout or lost reply the outcome is unknown;
the CLI never automatically retries, particularly not a toggle.

### Control wire ABI, version 1

`src/service/control.zig` owns the layouts. Both ends send the bytes of
`extern struct` values directly, with no field serializer. Builds require little
endian and assert every field offset and total size. Wire enums are
non-exhaustive so every incoming byte is representable; receivers validate IDs
before dispatch or display. Each send is one complete record. Receivers use
`MSG_TRUNC` to reject oversized records with apparently valid prefixes.

| Request offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | ASCII `VCDQ` |
| 4 | 2 | Version: 1 |
| 6 | 1 | Command: record=2, stop=3, cancel=4, status=5, kill=6; other values invalid |
| 7 | 1 | Flags: bit 0 is toggle, valid only for record; other bits zero |

| Reply offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | ASCII `VCDR` |
| 4 | 2 | Version: 1 |
| 6 | 1 | Result: accepted=0, ignored=1, invalid_request=2, unsupported_version=3 |
| 7 | 1 | Request command echoed; zero if the request size was invalid |
| 8 | 8 | Recording ID |
| 16 | 8 | Model idle seconds maximum |
| 24 | 8 | Daemon uptime seconds |
| 32 | 8 | Recording elapsed seconds, or `2^64-1` when unavailable |
| 40 | 8 | Model idle seconds remaining, or `2^64-1` when unavailable |
| 48 | 1 | Phase: idle=1, capturing=2, stopping=3, transcribing=4, delivering=5 |
| 49 | 1 | Model state: unloaded=1, loading=2, loaded=3, unloading=4 |
| 50 | 1 | Model kind: whisper.base.en=1, whisper.small.en=2, whisper.medium.en=3 |
| 51 | 13 | Reserved: zero |

Requests are exactly 8 bytes and replies are exactly 64 bytes. No other record
sizes or layouts are accepted. Accepted and ignored replies carry a status
snapshot, including for actions whose stdout carries no output. Rejected
requests do not execute and carry no status. Empty records or disconnected
clients may be closed without a reply. Unknown IDs, flags, reserved bytes, or
versions are rejected. No pointers, implicit padding, or uninitialized bytes
cross the socket. An incompatible stream listener is treated as occupied rather
than unlinked as stale. Diagnostic `tokens.txt` and `metadata.txt` files use
their own text format independently.

### Recording and delivery

The model runtime remains resident across successful recordings until its idle
deadline. It owns mapped weights, vocabulary, compute threads, activation
workspace, and decoder caches. Readiness follows initialization. Cancellation
discards the session and unloads the runtime after inference acknowledges its
stop. Unloading joins the compute pool before freeing its borrowed storage;
the coordinator thread remains available. Normal service shutdown stops and
joins both persistent threads before removing the control socket.

Accepted nonempty text is borrowed directly by the selected native clipboard
client. The default `clipboard_backend=auto` tries Wayland ext-data-control,
Wayland wlr-data-control, screen zero of a local X11 `DISPLAY`, then core
Wayland. Wayland capability discovery does not map a surface. Core mode is
reached only after X11 is absent or cannot initialize; it then temporarily maps
a transparent surface and waits for its destruction to reach the compositor
before reporting acquisition. `clipboard_backend=wayland` stays on Wayland and
chooses ext, wlr, then core; `clipboard_backend=x11` stays on X11. An explicit
selection never falls through to the other backend.

Desktop and clipboard output complete backend connection, authentication, and
protocol setup before `service_started`. If no permitted candidate becomes
ready within its two-second setup deadline, startup fails. Stdout output does
not initialize a clipboard. After an established connection is lost, an active
delivery fails without retry; the next delivery reconnects using the same
policy. Auto reevaluates the complete hierarchy, while explicit policies retry
only their configured backend. Selection publication has a separate two-second
deadline. The retained Wayland connection reuses its transparent pixel buffer.

After acquisition, the supervisor waits `paste_settle_ms` (default 10 ms) and
sends the configured chord through native Linux input events, with
`paste_key_gap_ms` (default 4 ms) between its four frames. Both settings accept
0 through 65535 milliseconds; zero removes that intentional delay. They apply
only to desktop output. The default scheduled delay is 10 + 3 × 4 = 22 ms;
clipboard acquisition, scheduling, and application response add their own time.
Lower delays are not a guarantee of readiness in every application. To restore
the previous margins, use `paste_settle_ms=50` and `paste_key_gap_ms=8`.

The keyboard retains its one-second initial device enumeration allowance.
The paste deadline includes all three configured key gaps plus 200 ms for
scheduling delays and nonblocking writes. These are timer-driven stages; the
event loop keeps serving commands. `paste_outcome=sent` means the kernel accepted
the key releases, not that an application inserted the text. An uncertain paste
is never retried. A paste error preserves the clipboard.

After sending the shortcut, the supervisor waits up to
`paste_observation_ms` (default 250 ms) for a text transfer that began after the
first key frame. A completed transfer records
`paste_observation=clipboard_transfer_completed`; this means the clipboard owner
finished serving a post-shortcut text request, not that an application inserted
the text. If no such transfer completes, Voiced emits a `paste_unconfirmed`
warning and records `paste_observation=not_observed`. Clipboard caching or an
unusually slow reader can produce that warning after a successful paste, so it
neither changes the recording outcome nor triggers a notification or retry.
Transcript saving waits until the transfer completes or this observation window
expires, allowing the event loop to serve the paste reader before synchronous
storage. The service remains in its delivering phase during the window, so it
ignores new recording commands for at most that additional interval. Setting
`paste_observation_ms=0` records `paste_observation=disabled`, suppresses the
unconfirmed warning, and saves as soon as the shortcut has been sent.

The clipboard owner survives later recordings and model expiry. Two transcript
buffers are reserved at startup. A buffer remains immutable while offered or
borrowed by a paste transfer; availability is derived from those references.
Eight transfers may coexist, each with a two-second deadline. When both buffers
are borrowed, the next recording waits for one to become reusable. Selection
loss retires ownership without copying again. Cancellation during delivery stops
further output but cannot undo a copy or paste already sent.

`--transcript-output stdout` retains the synchronous diagnostic sink: final text followed
by a newline. Operational messages go to the journal. Discarded sessions and empty
no-speech results emit no transcription. A capture failure can still deliver a
successfully transcribed valid prefix while reporting the capture error.

Output adds no C bindings, clipboard helper process, input helper daemon, or
service thread. Wayland and X11 clipboard protocols, PipeWire capture, desktop
notifications and journal logging are implemented directly in Zig. Desktop
services and permission to use uinput are still required. Repository-level
systemd installation remains separate work.

## Native clipboard

The service does not invoke `wl-copy`, `xclip`, or `xsel`.

`src/clipboard.zig` applies the configured backend policy and owns fallback
between clients. Auto uses Wayland data control when ext or wlr is advertised,
then X11 when available, and core Wayland last. A candidate that is simply not
advertised is a debug decision; a failed candidate followed by another is a
warning. Explicit selections and an exhausted auto hierarchy are startup
errors. On Wayland, `src/clipboard/wayland.zig` owns discovery, text generations
and transfers while `src/clipboard/wayland_wire.zig` handles the native socket
and descriptor transport. Core mode uses an xdg-shell surface and GNOME's GTK
surface extension when available. The connection reserves 64 KiB for incoming
frames and 4 KiB for outgoing requests. Acquisition waits for the fallback
surface's destruction to reach the compositor, but cannot guarantee the eventual
focus target of a globally injected shortcut.

On X11, `src/clipboard/x11.zig` owns a hidden InputOnly selection window and
`src/clipboard/x11_wire.zig` handles the local Unix socket, bounded `.Xauthority` parsing,
and MIT-MAGIC-COOKIE-1 setup. It serves `TARGETS`, `TIMESTAMP`, `UTF8_STRING`,
`TEXT`, UTF-8 plain-text aliases, and bounded `INCR` streams. `MULTIPLE`, legacy
`STRING`, nonzero screens, remote TCP displays, and other `DISPLAY` transports
are intentionally unsupported. Both backends use no C bindings, dynamic
libraries, worker processes or extra threads. Text remains
borrowed while offered or being transferred; two generations and eight transfers
bound retention, and stalled transfers expire after two seconds.

## Desktop notifications

`src/notification.zig` owns error presentation, replacement, dismissal, and
per-recording suppression. `src/dbus.zig` handles the wire format and nonblocking
socket I/O. Neither uses libsystemd, libdbus, an external notification command,
or an additional thread. A session D-Bus and notification server are still
required desktop services.

Initialization resolves the first supported Unix endpoint in
`DBUS_SESSION_BUS_ADDRESS`, falling back to `$XDG_RUNTIME_DIR/bus` when unset.
Unix filesystem and abstract addresses and EXTERNAL authentication are supported;
TCP, autolaunch, and descriptor passing are intentionally unsupported. Outgoing
frames are little endian; replies in either byte order are accepted.

The client reserves 16 KiB for input and 4 KiB for output, one outstanding call,
and one coalesced pending operation. Frames above the input limit are rejected.
Authentication and calls have one-second deadlines; disconnected buses are
retried after five seconds using the address resolved at startup. Notification
IDs retain their issuing server's unique name to avoid replacing or closing an
unrelated popup after a server restart. Failures retain protocol error names and
messages or the transport error, phase, and native errno in logs. These failures
do not stop recording or delivery.

## Logging
## Logging

Service and worker diagnostics use `src/logging.zig`, including foreground
`voiced serve`. The command-line interface keeps these stream and exit-status
contracts:

| Output | Destination |
| --- | --- |
| `-h`, `--help`, and bare usage | stdout, exit 0 |
| Invalid flags or configuration | stderr, exit 2 |
| Failures before logging initializes | stderr, exit 1 |
| Daemon diagnostics after initialization | Resolved `journal` or `stderr` target |
| Successful `record`, `stop`, `cancel`, and `kill` acknowledgements | stderr |
| `status` and explicit transcript output | stdout |
| Panic and crash diagnostics | stderr |

Help remains redirectable, for example `voiced serve --help > serve-help.txt`.
Command responses and explicit transcript output keep stdout free of operational
diagnostics.

Set `log_level=info` in the config or `--log-level info` on the command line.
Levels are `critical`, `error`, `warn`, `info`, and `debug`; the threshold includes
all more severe events. The default `auto` target selects stderr when stderr is
a terminal and native journal records otherwise. Force either destination with
`log_target=journal`, `log_target=stderr`, or the corresponding command-line
option:

```bash
voiced serve --log-target stderr --log-level debug
```

The stderr target writes `level: event key=value` lines to the inherited
standard-error descriptor. Its writes are synchronous, so an undrained pipe or
slow terminal can delay the service; the journal target remains nonblocking.
Worker launches explicitly carry the effective level. Changing the config
requires a restart. At `warn` or above, info-level readiness and
effective-configuration messages are intentionally filtered.

```bash
journalctl --user -u voiced -f -o short-precise
journalctl --user -u voiced -p warning
journalctl --user -u voiced VOICED_EVENT=recording_finished
journalctl --user -u voiced VOICED_COMPONENT=capture
journalctl --user -u voiced VOICED_RECORDING_ID=21
# Foreground serve has no voiced.service unit:
journalctl --user -t voiced -f
```

### Recommended systemd user service

A user service should explicitly capture stdout and stderr even when the daemon
uses native journal records. Invalid configuration, logging initialization
failure, and panic diagnostics can occur on stderr before or outside normal
logging. Replace the executable path when Voiced is installed elsewhere.

```ini
[Unit]
Description=Voice dictation daemon
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.local/bin/voiced serve --log-target journal
Restart=on-failure
RestartSec=1
RestartPreventExitStatus=2
TimeoutStopSec=10
KillMode=mixed
RuntimeDirectory=voiced
RuntimeDirectoryMode=0700
StandardOutput=journal
StandardError=journal
SyslogIdentifier=voiced
Environment=XDG_RUNTIME_DIR=%t

[Install]
WantedBy=graphical-session.target
```

Apply the unit and inspect startup failures with:

```bash
systemctl --user daemon-reload
systemctl --user enable --now voiced.service
systemctl --user status voiced.service
journalctl --user -u voiced.service -b -e
```

With `--log-target journal`, initialized daemon failures carry native
`VOICED_EVENT` and `VOICED_COMPONENT` fields. Failures before logging initializes
reach the same journal through `StandardError=journal`. With `--log-target
stderr`, systemd still captures the readable lines but cannot add the native
`VOICED_*` fields. If the executable itself is missing or cannot be executed,
systemd records that manager-level failure because Voiced never starts.

With the journal target, every service event is a single native datagram with
real `PRIORITY`, `SYSLOG_IDENTIFIER=voiced`, `VOICED_COMPONENT`, and
`VOICED_EVENT`. Recording
events also carry `VOICED_RECORDING_ID`; the logger renders the same context once
in `MESSAGE`, so call sites cannot make the visible and native IDs disagree.
Model-cache events without a recording can instead be correlated by their
journal PID. Components are logical owners such as `capture`, `transcription`,
`clipboard`, `paste`, `storage`, and `supervisor`. The call site need not be in
that component's file: the supervisor logs a received typed capture report as a
capture event. `service_started` is emitted after clipboard initialization and
both workers start. Its single canonical record carries
`clipboard_backend_policy` plus the actual `clipboard_backend` and
`clipboard_mode` when clipboard output is active. Candidate negotiation remains
in debug or warning component events rather than adding successful info events.
Call sites pass the event name separately from ordered typed fields; arbitrary
text uses escaped string fields rather than interpolated message fragments.

Human-readable messages begin with the lower-case event name and use ordered
`key=value` fields. Event names, field names, and unquoted enum values are
lower-case snake_case. Time fields name their unit; `size` always means bytes.
Free text is quoted with escaped quotes, newlines, and control bytes. Newlines
in error details stay inside one binary-encoded `MESSAGE` and cannot create
extra journal fields. `MESSAGE` remains a diagnostic interface rather than a
versioned machine protocol; use the native fields for filtering.

Records use fixed stack storage, at most 4 KiB per datagram or stderr line;
oversized messages end with `[truncated]`. No audio callback or inference kernel logs. Disabled
events skip formatting, and expensive argument preparation must be guarded with
`logging.enabled()`. There is no logging queue, thread, or event-loop
registration. A full or unavailable journal drops the event without blocking
or falling back to stderr. `VOICED_DROPPED` on the next successfully submitted
event counts local submission losses; it does not claim that journald persisted
an accepted event. All threads share the one nonblocking journal socket
initialized at startup. Path-based sends allow subsequent messages to reach
journald after it restarts without reconnect state.

Build the daemon with:

```bash
agent-run 'zig build -Doptimize=ReleaseSafe install'
```

## Desktop error notifications
## Desktop error notifications

`notification_mode=errors` (the default) reports failed capture/transcription,
unreliable speech, incomplete recordings, and failed clipboard, paste, or save
operations. Recording, transcription progress, ordinary silence, cancellation,
and successful delivery produce no popups. `notification_mode=off` disables the
connection; `transcript_output=stdout` always disables it.

Each service owns its `Error` tagged union and returns `Result { ok, err }`
(`Result(T)` for operations with different success values). Capture errors
retain valid audio and the complete native report, including the exact failed
stage and cause. Transcription reports preserve the stage and exact Zig error
name. Paste errors include the syscall errno, ioctl request/argument or
chord/frame progress. Save errors retain the failing operation, write progress,
and any temporary-file cleanup error. These operational results do not allocate.

The supervisor consumes those types at each component boundary. After logging
the complete payload, it retains a compact component-tagged `Problem` containing
the semantic code needed by lifecycle, the terminal record, and notifications;
large fixed diagnostic buffers do not become permanent lifecycle state. The
outer union tag directly supplies `problem_component`. The supervisor maps that
problem forward to a concise notification only at the presentation boundary;
notification categories are never reverse-mapped to reconstruct information
they discarded. Delivery retains the upstream recording problem separately from
a later clipboard or paste problem, so an output error cannot overwrite the
original cause; output outcome fields and component errors preserve the later
problem. The presentation distinguishes missing/ambiguous microphones,
lost connections, source changes, audio stalls, model loading/transcription
errors, and clipboard/paste/storage errors. It logs full diagnostics at the
owning component boundary and preserves its component and semantic code through
the terminal recording outcome. Mic ambiguity logs list the observed candidate
sources; use `microphone_node` **instead of** `microphone_serial` to select one
input.

Clipboard diagnostics retain the selected backend, typed transport/protocol
errors, and native errno. Wayland reports compositor objects, codes and messages;
X11 reports authority/setup failures, response sequences, server opcodes and bad
values. Capture includes the observed candidate catalog on selection errors;
the fixed 4 KiB diagnostic can truncate unusually large catalogs.
For the full cause and affected recording, run:

```bash
journalctl --user -u voiced -b -n 50 --no-pager
```

The supervisor speaks native D-Bus on the user session bus through its existing
epoll and timerfd loop. There is no notification subprocess, library, or thread.
One outstanding request and one pending operation coalesce bursts; replies have
a one-second timeout. Repeated identical problem/outcome pairs are suppressed
within a recording.
Each new recording resets suppression; ignored commands do not. The next error
can replace the existing popup, and a successful recording closes it. Dismissal
invalidates the held ID; notification server restarts reset IDs and redisplay an
unresolved problem. Replacement and closure address the original server's
unique bus name, preventing ID reuse races.

Clipboard/paste failure messages distinguish successful transcript saving from
failed saving. When copying fails but saving succeeds, the popup includes the
saved `transcript.txt` path, shortened beneath the home directory with a `~/`
prefix. Popup bodies never contain the transcript. Notification failures are
logged and leave recording and output working. An invalid or unsupported
session-bus address disables notifications until voiced restarts. An unavailable
or disconnected endpoint is retried every five seconds; the first failure and a
lost established connection are warnings, while repeated retry failures are
debug records. A notification server can appear or restart on a live bus without
restarting voiced. Shutdown attempts
to close a known popup without waiting or flushing the bus. A timed-out Notify
whose reply never arrives has no usable ID; its popup follows desktop expiration.

## Recording metrics

`recording_id` counts accepted recordings within one service lifetime; `chunk_id`
counts their chunks from zero. Failure and cancellation consume an ID. Restarting
the service resets the sequence. Journal PID and invocation metadata distinguish
service lifetimes.

At `info`, each recording that reaches an orderly lifecycle result emits one
terminal `recording_finished` record. A critical supervisor or worker deadline
can exit the daemon before that result exists. A successful desktop shortcut is
`paste_sent`, clipboard-only output is `copied`, and stdout output is `written`.
These outcomes describe what Voiced completed; they do not claim that a target
application inserted text. Ordinary no-speech and cancellation are also `info`;
a usable degraded or partial result is `warning`; a recording with no usable
result is `error`. `problem_component` and `problem_code` appear only when a
problem affected the result. Clipboard, paste, and save outcomes appear when the
corresponding desktop operation was attempted. For example:

```text
recording_finished recording_id=21 outcome=paste_sent transcription_audio_duration_seconds=8.412 transcription_compute_duration_ms=311.240 transcription_chunks_count=1 transcript_size=126 transcription_compute_speed_ratio=27.02 stop_origin=command recording_finalize_duration_ms=344.112 clipboard_outcome=succeeded clipboard_backend=wayland clipboard_mode=core clipboard_acquire_duration_ms=119.371 paste_outcome=sent paste_shortcut=ctrl+shift+v paste_observation_ms=250 paste_settle_duration_ms=10.028 paste_duration_ms=22.268 paste_observation=clipboard_transfer_completed paste_observation_elapsed_ms=18.442 save_outcome=succeeded
recording_finished recording_id=22 outcome=failed transcription_audio_duration_seconds=0.000 transcription_compute_duration_ms=0.000 transcription_chunks_count=0 transcript_size=0 stop_origin=command recording_finalize_duration_ms=1001.337 problem_component=capture problem_code=start_timed_out
```

At `debug`, component events expose phase boundaries and performance evidence:

```text
recording_started recording_id=21
capture_started recording_id=21 capture_start_duration_ms=84.192
capture_finished recording_id=21 outcome=stopped audio_duration_seconds=8.412 ...
transcription_chunk_finished recording_id=21 chunk_id=0 audio_duration_seconds=8.412 ...
clipboard_candidate_skipped candidate=wayland_data_control reason=protocol_not_advertised fallback=x11
clipboard_ready clipboard_backend_policy=auto clipboard_backend=x11 clipboard_mode=x11
clipboard_acquired recording_id=21 transcript_size=126 ...
paste_sent recording_id=21 paste_duration_ms=22.268 ...
clipboard_text_transferred recording_id=21 transcript_size=126 clipboard_transfer_duration_ms=0.041
transcript_saved recording_id=21 transcript_size=126 transcript_save_duration_ms=0.107
```

Component warnings and errors retain the local typed evidence under the same
recording ID. A failed capture promotes its complete final report from debug to
error. Decoder and chunk limits are warnings when bounded text remains usable;
hard transcription errors remain errors. Components do not carry debug text through a worker mailbox: coordinators
log their own debug facts, while hot audio callbacks and inference kernels never
log. The supervisor consumes typed reports and owns the terminal result.

Field names put the subject before the measurement. Time values use fixed units
in their names; byte quantities use `size` without another byte suffix:

| Field | Meaning |
|---|---|
| `audio_duration_seconds` | Actual audio duration, excluding encoder silence padding |
| `audio_samples_count` | Samples in this chunk |
| `transcription_audio_duration_seconds` | Audio consumed by completed transcription chunks |
| `features_duration_ms` | Measured feature extraction work |
| `inference_duration_ms` | Measured encoder, cross-KV, and decoder work |
| `transcription_compute_duration_ms` | Sum of feature and inference work for consumed results |
| `transcription_compute_speed_ratio` | Audio duration divided by measured computation duration; 12.5 means 12.5 times realtime |
| `transcript_size` | UTF-8 bytes: raw chunk text, or accumulated trimmed text at completion/delivery |
| `transcription_tokens_count`, `transcription_tokens_count_max` | Generated token count and its inclusive maximum |

Recording computation totals include consumed no-speech chunks. They exclude
model preparation, queueing, failed attempts, text assembly, and desktop
delivery; they are not end-to-end latency. Capture-end audio duration covers all
captured samples, which can exceed the processed prefix after a failure. A speed
with zero audio or computation duration is `unavailable`. Discarded recordings
have `transcript_size=0`. Transcript contents are never diagnostics.

Durations measure one named operation; elapsed fields measure from a named
reference event. There is no duration relative to the previous log line.

| Timing | Start → end |
|---|---|
| `capture_start_duration_ms` | Accepted command received → first callback observed by supervisor |
| `model_file_load_duration_ms` | Packed file open → mapping and complete validation finish |
| `model_prepare_duration_ms` | Worker starts model loading → workspace and runtime ready |
| `clipboard_acquire_duration_ms` | Desktop delivery begins → supervisor observes clipboard acquisition |
| `paste_settle_duration_ms` | Clipboard acquisition observed → first shortcut key frame |
| `paste_duration_ms` | Clipboard acquisition observed → shortcut submission complete, including settle/key waits |
| `paste_observation_elapsed_ms` | First shortcut key frame → matching transfer completion or observation expiry |
| `clipboard_transfer_duration_ms` | Clipboard text request received → owner finishes serving its bytes |
| `transcript_save_duration_ms` | Save begins → atomic publication finishes, or save returns an error |
| `recording_finalize_duration_ms` | Stop reference → this milestone |

Packed-file loading and runtime initialization have additional duration fields
at `debug`. Lifecycle and chunk
measurements are debug details; the terminal recording summary repeats the
output measurements needed to diagnose clipboard and paste delivery at `info`.
Synchronous operations use local timestamps; the supervisor retains the
recording request, delivery boundary, and paste observation timestamps. No
callback or per-token timers were added.

With `stop_origin=command`, the stop reference is the supervisor
receiving the accepted stop/toggle command. Repeated stops do not reset it.
With `stop_origin=capture_end`, it is the supervisor observing the
capture thread's final report; work before that observation is outside the elapsed
measurement.
Without either reference, origin and elapsed are `unavailable`. These timings
exclude keybinding/CLI startup and application rendering after the paste shortcut.
Each new recording resets the reference.

Requested cancellation is `info`. Service errors retain their typed diagnostics
at `error`; a missed cancellation or unload deadline is `critical` and exits the
daemon. Thread crashes follow the configured panic/core-dump behavior. No-speech
evidence is included in the chunk event instead of a duplicate rejection event.

Expected startup refusals such as an already running daemon are `error`.
Unrecoverable supervisor operation errors are `critical`. Operational event-loop
syscall failures retain operation and native errno before returning through that
exit path; cleanup errors are logged independently. Programming assertions and
native panic traces retain Zig's emergency stderr path.

## Audio and inference ownership

The native client negotiates Float32 DSP ports at the graph's actual rate,
selects one source, averages its channels, and resamples to mono 16 kHz with a
fixed-memory low-pass filter. Supported graph rates are 8–192 kHz, with at most
eight channels and 100 ms of audio per graph cycle. Complete blocks are
validated, non-finite samples rejected, and finite overdrive clamped before
publication into three shared slots. Mixing reads borrowed planar graph buffers,
including wrapped spans, and the resampler writes directly into an unpublished
slot suffix. A failed conversion leaves the committed prefix unchanged. Audio
processing performs no allocation, logging, model work, or blocking I/O.
Inference borrows each sealed slot without
quantization or a second waveform allocation. The fixed PCM exchange occupies
about 5.49 MiB.

`pipewire_protocol.zig` owns framing, SPA PODs and descriptor transfer;
`pipewire_native.zig` owns discovery, source identity, graph mappings and DSP
cycles. The 128-entry catalog stores kind-specific fields and string references;
a reusable 64 KiB string pool covers every entry at its maximum string lengths.
Protocol queues reserve 64 KiB input and 32 KiB output. `pipewire.zig` owns
recording policy and shared-slot publication through one worker poll loop.
`audio_resampler.zig` retains the conversion phase and filter history.
`realtime.zig` requests realtime scheduling directly, falling back to RTKit over
native D-Bus with a 500 ms deadline. Scheduling and memory locking remain best
effort and their actual results are reported.

The compatibility floor is PipeWire 0.3.48 (ClientNode version 4). Tests use
private 0.3.48, 1.0.5 and 1.6.8 servers. Newer servers negotiate ClientNode
version 5; advertised and selected interface versions are distinct from the
server release and are retained in diagnostics. Source removal or identity
change ends that recording; a new recording resolves the configured source
again. Capture connects to `PIPEWIRE_REMOTE` (default `pipewire-0`) under
`PIPEWIRE_RUNTIME_DIR`, falling back to `XDG_RUNTIME_DIR`. An absolute remote
path is accepted. These environment values are resolved once at service startup.

`audio_policy.zig` owns the recording policy:

- Physical slot capacity and forced boundary: 30 seconds.
- Natural boundary: at least 20 seconds of audio, followed by 300 ms quiet.
- Initial background calibration: 300 ms.

Chunks are independent, without PCM overlap or previous-text prompting.
Outside capacity-stop cases, activity and model confidence suppress normal
no-speech results. A disagreement stops the recording and rejects the disputed
chunk. Voiced retains any earlier accepted chunks as a partial transcript with
a warning, or produces no output when no accepted prefix exists.

Each transcript buffer is sized once at 64 UTF-8 bytes per configured recording
second plus one full 4 KiB mailbox result. The one-hour default reserves about
229 KiB per buffer: one for stdout, two (about 458 KiB) for clipboard/desktop.
Unused payload pages are not eagerly initialized. Reaching this bound, the 4 KiB
chunk-text bound, or the decoder's 446-token bound stops further
recording/transcription and delivers the available text. The final prefix ends
at a complete UTF-8 character; words or sentences may be incomplete. Desktop
mode copies, sends the paste shortcut, then saves;
clipboard mode copies and saves. Publication, transfers and saving borrow the
same completed buffer; no buffer grows or is allocated per recording.

A notification distinguishes recording size, chunk size, and decoder token
limits. Decoder-limit text is retained even when confidence is low: it may repeat
or contain inaccuracies, so the notification asks you to check it. Both chunk
limits are reported if reached together. A fatal restart during inference can lose the current recording. Explicit
cancellation suppresses pending output and preserves the previous saved
transcript. Start a new recording to continue; unprocessed audio after the
cutoff is not resumed automatically.

The supervisor requires the first callback within three seconds and progress
at least every two seconds. Each inference has a ten-second deadline; model
startup has fifteen seconds. Stop, cancellation, model unload, and final thread
shutdown each have a one-second acknowledgement deadline. Capture alone closes
its PipeWire stream. A missed stop/unload deadline exits the whole daemon before
cleanup can invalidate a worker borrow; systemd restarts the service. The current
recording is not retried across a fatal restart.

## Packed models and runtime

Each installed `.voiced` file contains the model kind, native packed weights,
decoded vocabulary bytes, token offsets, and a checksum over the complete file.
The transcription worker validates and maps this file
read-only; there is no derived model cache or first-recording conversion. The
loaded `inference.Model` owns the mapping, so its weights and vocabulary always
share one backing store.

The caller keeps the model, runtime, and tensor arena alive at stable addresses
until the workers join:

```zig
var model = try packed_model.load(io, models_directory, model_file_name, expected_kind);
defer model.deinit();

var runtime: inference.Runtime = undefined;
try runtime.init(io, &model, memory, policy);
defer runtime.deinit();
```

Transcription allocates nothing and returns untrimmed text in the supplied
buffer. Vocabulary lookup copies predecoded token bytes directly from the mapped
model; one final UTF-8 check validates the assembled transcript. The supervisor
trims only the assembled final transcript. Model unload releases the mapping and
runtime storage; service mailboxes and audio/transcript buffers are reused. Zig
0.16.0 uses its debug allocator in libc-free ReleaseSafe builds, so freed small
allocations can retain bucket pages between reloads without retaining a loaded
model.

[The inference guide](docs/inference.md) explains the native Whisper implementation.

The process-level integration harness was removed. Its control, logging,
notification, capture, delivery, recovery, and cancellation scenarios remain in
Git history for future porting. The standalone notification verifier above and
runtime corpus checks remain available. Always use private runtime/state paths
for automated service checks.
