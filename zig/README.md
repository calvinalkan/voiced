# Voiced Zig daemon

The native executable contains the daemon and its CLI client. One process owns
three persistent threads: the supervisor, capture, and transcription coordinator.
The resident runtime adds the configured inference pool while a model is loaded.

```text
main.zig
├── serve → supervisor.runService          main-thread epoll loop
│            ├── control_socket.zig        public commands and status
│            ├── capture.zig               persistent capture thread
│            │   ├── pipewire.zig          stream ownership and graph cycles
│            │   └── audio_exchange.zig    three reusable Float32 slots
│            ├── transcription.zig         persistent model coordinator thread
│            │   ├── model_cache.zig       mapped weights
│            │   └── ../runtime/root.zig   resident inference thread pool
│            ├── worker.zig                typed job/result mailboxes and wakeups
│            ├── clipboard.zig             Wayland/X11 clipboard selection
│            │   ├── clipboard_wayland.zig native Wayland ownership and transfers
│            │   └── clipboard_x11.zig     native X11 ownership and transfers
│            └── paste_keyboard.zig        native uinput keyboard
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

All executable targets explicitly use static linkage. The daemon is a static
PIE by default: address randomization remains enabled, with no ELF interpreter
or shared-library dependencies. Native Zig clients speak
PipeWire and D-Bus directly; neither libc nor client development headers are
required. Inference uses host-targeted
AVX-VNNI kernels; this is not a portable baseline binary. There is
no external inference engine or C++ bridge. The checkpoint importer reads the
published CTranslate2 serialization format, not CTranslate2 code or libraries.

```bash
sudo apt install pipewire
cd zig
zig build setup-models
zig build
```

Setup atomically installs the pinned Systran `base.en` and `small.en` checkpoints.
It verifies `model.bin` and `vocabulary.txt` against their sizes and BLAKE3-256 pins,
reusing valid installed payloads. `zig build setup` is an alias for model setup;
ordinary builds neither download models nor run inference.

Plain `zig build` defaults to ReleaseSafe for both
application and inference code, retaining lifecycle assertions and Zig runtime
safety checks. Use `-Doptimize=Debug` for an unoptimized checked build;
ReleaseFast is not the production policy.

For an opt-in compact build, `-Doptimize=ReleaseSmall` compiles the application
and control code with ReleaseSmall and the inference module with ReleaseFast.
This profile disables Zig runtime safety checks; it does not replace the
ReleaseSafe production policy. Debug, ReleaseSafe, and ReleaseFast continue to
apply their selected mode to both application and inference code.

```bash
zig build -Doptimize=ReleaseSmall -Dstrip=true
```

`-Dstrip=true` asks Zig to omit debug information and symbols from `voiced` at
link time. Omit it to retain an unstripped executable for debugging. One build
invocation emits the selected form; use separate prefixes when both forms are
required. `-Dcrash-diagnostics=true` (the default) independently keeps in-process
panic stack tracing and Zig's mode-dependent fault handler.

To remove in-process symbolization without disabling ReleaseSafe checks:

```bash
zig build -Doptimize=ReleaseSafe -Dcrash-diagnostics=false -Dstrip=true
```

The matching size gate checks Zig's natively stripped static PIE and fails unless
it remains strictly below 1 MiB:

```bash
zig build size-check -Doptimize=ReleaseSafe -Dcrash-diagnostics=false -Dstrip=true
```

The step rejects other optimization, crash-diagnostic, stripping, and PIE
settings so a passing result always represents the documented profile.

This daemon-only option is independent of stripping and optimization mode. Normal
CLI output and operational logs remain unchanged. Panics print a best-effort
message to stderr and abort with SIGABRT; memory faults use the OS signal handling
instead of Zig's rich fault handler. Neither path prints an in-process stack
trace. Omit `-Dstrip=true` when external debug information is required; Zig's
native link-time stripping does not emit a separate debug companion.

Before deploying this mode, verify core collection for the actual service; an
abort does not guarantee a saved core. Ubuntu may use Apport rather than
systemd-coredump. Deploy the unstripped form when source-level core analysis is
required. Cores can contain audio, transcripts, and other process memory:
restrict access and retention. The build does not change the host's collector
configuration.

The `Binary Size Optimizations` section in `src/main.zig` owns the daemon's Zig
root configuration. Unused `std.Io` networking is disabled in every build mode.
Raw Unix sockets and PipeWire remain available, and the separate model-setup tool
keeps networking.

## Run

```bash
# Use a separate instance while evaluating the native daemon.
export VOICED_INSTANCE=test
./zig-out/bin/voiced serve

# In another terminal with the same VOICED_INSTANCE:
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
| `--model <name>` | `Systran/faster-whisper-small.en` or `Systran/faster-whisper-base.en` |
| `--model-encoder-threads <1-32>` | Inference worker count |
| `--model-decoder-threads <1-32>` | Decoder subset of the encoder pool; omitted uses the whole pool |
| `--model-encoder-padding-seconds <5\|10\|30>` | Encoder silence appended after each audio chunk; default 10 seconds, with total input capped at 30 seconds |
| `--recording-seconds-max <1-65535>` | Maximum recording duration; default 3,600 seconds |
| `--model-idle-seconds-max <seconds>` | Idle retention interval; default 300, zero disables retention |
| `--microphone-serial <serial>` | Select exactly one source by stable physical Device identity |
| `--microphone-node <node-name>` | Select an explicit PipeWire source; mutually exclusive with Device serial |
| `--transcript-output <mode>` | `desktop` copies and pastes (default); `clipboard` only copies; `stdout` writes diagnostic text |
| `--notification-mode <errors\|off>` | Error popups; default `errors`. Diagnostic `stdout` always disables notifications |
| `--paste-shortcut <chord>` | `ctrl+shift+v` (default), `ctrl+v`, or `shift+insert`; used in desktop mode |
| `--paste-settle-ms <0-65535>` | Wait after clipboard acquisition before pasting; default 10 ms |
| `--paste-key-gap-ms <0-65535>` | Gap between the four paste key-event frames; default 4 ms |

All service settings also accept snake_case keys in
`$XDG_CONFIG_HOME/voiced/config` (or `~/.config/voiced/config`). For example:

```ini
# Example tuned worker counts; the built-in default remains four.
model=Systran/faster-whisper-small.en
model_encoder_threads=16
model_decoder_threads=8
model_encoder_padding_seconds=10
model_idle_seconds_max=300
recording_seconds_max=3600
transcript_output=desktop
paste_shortcut=ctrl+shift+v
paste_settle_ms=10
paste_key_gap_ms=4
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

CLI settings override file settings. A CLI microphone choice replaces the file's
choice; specifying both node and serial within one input is an error. Decoder
threads must not exceed encoder threads after overrides are applied. Both phases
share one pool; workers outside the decoder subset park during decoding.
`--config` and command actions such as `record --toggle` are not file settings.

Desktop mode opens one virtual keyboard through `/dev/uinput` and keeps it
across dictations. It needs the user's existing uinput permissions; Voiced does
not change device permissions or run as root. Missing access is logged and
delivery continues with clipboard only. Choose `ctrl+v` for applications using
that paste shortcut. The shortcut goes to the application focused at delivery
time; physically held modifier keys can affect it.

After clipboard/paste finishes, desktop and clipboard modes save the final UTF-8
text to `$XDG_STATE_HOME/voiced/transcript.txt`, defaulting to
`~/.local/state/voiced/transcript.txt`. Named instances use `voiced-<instance>`.
The file is replaced atomically, with no added newline; rejected, empty, and
cancelled recordings preserve the previous file. Clipboard/paste failures still
attempt saving. Save errors are logged without undoing desktop delivery.

The destination path is resolved and allocated once at service startup. Saving
reuses that path and the existing transcript buffer, with no per-save path
allocations or environment lookups. It uses a direct temporary-file write and
rename, without fsync. It happens after delivery and logs its elapsed time. This
avoids a save worker, but slow storage can block commands and power-loss durability
is not guaranteed. Diagnostic stdout mode does not update the file.

Failed model-worker transcriptions automatically retain one private diagnostic
bundle at `$XDG_STATE_HOME/voiced/last-failed/` (default
`~/.local/state/voiced/last-failed/`; named instances use `voiced-<instance>`):

- `audio.wav`: the exact failed chunk, mono 16 kHz IEEE Float32, before padding;
- `generated.txt`: generated bytes, including a partial text conversion;
- `tokens.txt`: all generated token IDs, as space-separated decimal u32 values
  with a final newline; a newline alone means no tokens;
- `metadata.txt`: the original error, chunk/sample counts, decoder evidence,
  phase timings, model revision/checksums, compiler/build mode, threads, padding,
  prompt, suppression lists, and token limit.

Metadata text format version 3 uses one `key=value` line per field, with dotted
keys such as `evidence.decoder_ns`. Integers are decimal, booleans `true`/`false`,
float evidence retains non-finite values, and token lists use spaces. String
values have no surrounding quotes and are not trimmed. Backslash, double quote,
newline, carriage return, and tab use `\\\\`, `\\\"`, `\\n`, `\\r`, and `\\t`;
other non-printable or non-ASCII UTF-8 bytes use `\\xHH`. Split each line at its
first `=`; equals signs and leading/trailing spaces in values are literal.
`end_available=false` with an empty `end` means null; availability flags still
distinguish unknown decoder evidence from actual zero values. Files end with a
newline, and the offline text reader bounds each metadata/token file to 1 MiB.

All fingerprints use BLAKE3-256, including `source_blake3` and
`packed_image_blake3`. Only the current capture format is supported. The separate
offline replay executable and comparison script emit JSON result reports;
failed-capture publication uses text.

Directories are mode `0700`, files `0600`. Each failed attempt atomically replaces
the previous complete bundle. An interrupted save can leave one additional
`last-failed.pending` bundle, reused on the next failure; the maximum audio per
bundle is 30 seconds (1.83 MiB). Successful recordings preserve the last failure.
Transcript contents and token IDs stay out of journal logs. Failure logs include
the chunk, audio duration, token count/limit, actual encoder positions,
no-speech/average log probabilities, and separate phase timings when available.
Unknown decoding evidence is explicitly marked unavailable in the saved metadata.

The transcription thread writes the diagnostic capture while borrowing its
sealed audio and decoder buffers, then returns the result and original error.
It journals the original diagnostic before disk work so a stalled save cannot
hide the cause. Paths are resolved once at initialization;
successful inference adds no disk work or per-chunk allocation. Diagnostic saving
is best effort under the existing worker deadline: filesystem problems are logged
independently, and a daemon that exits during saving cannot complete this bundle. These are
runtime/feature/text-conversion errors, not normally completed results rejected
by the supervisor's speech-confidence policy. A missing model has no decoded
chunk to capture. Saving never updates the accepted transcript or clipboard.

The optional reference comparison uses its own Python environment; the daemon
has no Python dependency. Set up these comparison tools from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy ctranslate2 faster-whisper blake3
```

Build and replay against already installed models (the replay downloads none):

```bash
(cd zig && zig build replay)
.venv/bin/python zig/scripts/replay-transcription.py
```

The replay copies a consistent snapshot to a new private temporary directory,
then compares Zig and CTranslate2 INT8 with 10- and 30-second trailing padding,
matching greedy prompt, suppression, token limit, and normalized-zero padding.
A capture made with 5-second padding also gets that original setting replayed.
The reference uses independent NumPy feature extraction and CTranslate2 kernels;
its one thread count applies to both encoder and decoder. The summary reports
first differing token positions and token-limit exhaustion, with text kept in
private result files. It checks pinned source/packed-image identity and reports
compiler/build-mode differences; it does not preserve the old executable.

Use an explicit capture path as the positional argument, or compare a fixture:

```bash
.venv/bin/python zig/scripts/replay-transcription.py --audio test-fixtures/hello_world.wav
```

For a diagnostic run that does not copy or press keys:

```bash
VOICED_INSTANCE=test ./zig-out/bin/voiced serve --transcript-output stdout
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

Without a source option, each recording resolves the current default source.
An absent or ambiguous configured device fails rather than opening another
microphone. The observer independently verifies the linked source before
accepting samples. Source removal, replacement, or timeline discontinuity stops
capture and retains its valid published prefix for transcription.

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
connection at `$XDG_RUNTIME_DIR/voiced/control.sock`, or
`voiced-<instance>/control.sock` for a named instance. The directory is mode
`0700`, the socket `0600`. The server allows 16 clients with a one-second
connection deadline; the CLI has three-second socket I/O timeouts.

Successful `record`, `stop`, `cancel`, and `kill` commands are silent.
`status` prints one text line to stdout, for example:

```text
phase=idle model=warm session_id=3 model_keep_warm_seconds=300
```

Status exposes `idle`, `capturing`, `stopping`, `transcribing`, or `delivering`,
plus whether the model is absent, loading, warm, or unloading.
Toggles during stopping, transcription, or delivery are ignored and still exit
successfully. Errors go to stderr with a nonzero exit code. An acknowledgement
means the supervisor handled the command, not that recording, transcription,
or shutdown has finished. After a timeout or lost reply the outcome is unknown;
the CLI never automatically retries, particularly not a toggle.

### Control wire ABI, version 1

`control_socket.zig` owns the layouts. Both ends send the bytes of `extern struct`
values directly, with no field serializer. Builds require little endian and
assert every field offset and total size. Wire enums are non-exhaustive so every
incoming byte is representable; receivers validate IDs before dispatch or display.
Each send is one complete record. Receivers require the exact record size and
use `MSG_TRUNC` to reject oversized records with apparently valid prefixes.

| Request offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | ASCII `VCDQ` |
| 4 | 2 | Version: 1 |
| 6 | 1 | Command: record=2, stop=3, cancel=4, status=5, kill=6; other values invalid |
| 7 | 1 | Flags: bit 0 is toggle, valid only for record; other bits zero |

| Reply offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | ASCII `VCDR` |
| 4 | 2 | Server protocol version: 1 |
| 6 | 1 | Result: accepted=0, ignored=1, invalid_request=2, unsupported_version=3 |
| 7 | 1 | Request command echoed; zero if the request size was invalid |
| 8 | 8 | Session ID |
| 16 | 8 | Model keep-warm seconds |
| 24 | 1 | Phase: idle=1, capturing=2, stopping=3, transcribing=4, delivering=5 |
| 25 | 1 | Model: absent=1, legacy restarting=2, loading=3, warm=4, unloading=5 |
| 26 | 6 | Reserved: zero |

Requests are 8 bytes; all replies are 32 bytes. Accepted and ignored replies carry
a status snapshot, including for actions whose CLI output is silent. Rejected
requests do not execute; their status fields are zero (unavailable). Empty records
or disconnected clients may be closed without a reply. Unknown IDs, flags,
reserved bytes, or incompatible versions are not interpreted as valid commands
or status. No pointers, implicit padding, or uninitialized bytes cross the socket.

Restart the daemon when upgrading from the old stream/JSON protocol. There is no
compatibility fallback. An existing incompatible stream listener is treated as
occupied, never unlinked as stale. Changes to field layout or ID meanings require
a protocol version change. Diagnostic `tokens.txt` and `metadata.txt` files use
their own text format, independent of this control protocol.

### Recording and delivery

The model runtime remains resident across successful recordings until its idle
deadline. It owns mapped weights, vocabulary, compute threads, activation
workspace, and decoder caches. Readiness follows initialization. Cancellation
discards the session and unloads the runtime after inference acknowledges its
stop. Unloading joins the compute pool before freeing its borrowed storage;
the coordinator thread remains available. Normal service shutdown stops and
joins both persistent threads before removing the control socket.

Accepted nonempty text is borrowed directly by the selected native clipboard
client. Wayland is preferred when `WAYLAND_DISPLAY` is present; otherwise screen
zero of a local X11 `DISPLAY` in `:N`, `:N.0`, `unix:N`, or `unix:N.0` form is
used. Selection publication has a two-second deadline. On
Wayland, data-control protocols are preferred; the core data-device fallback
temporarily maps a transparent surface and waits for its destruction to reach
the compositor before reporting acquisition. The connection and its transparent
pixel buffer are reused.

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
event loop keeps serving commands. Completion
means the kernel accepted the key releases, not proof that an application pasted.
An uncertain paste is never retried. A paste error preserves the clipboard.

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

## Native clipboard verification

The optional `clipboard-check` target exercises the same selected native
clipboard backend used by `serve`. The service does not invoke `wl-copy`,
`xclip`, or `xsel`.

```bash
cd zig
zig build clipboard-check
./scripts/verify-clipboard.py
```

The bundled desktop verifier uses the Wayland path. It opens a dedicated GTK
text window and checks three copies/pastes, focus restoration, and restoration
of the previous plain-text clipboard. It
writes `native.log` and `report.json` under a private `/tmp/voiced-native-clipboard-*`
directory. Keep the window focused and release modifier keys. A clipboard with
non-text formats is left unchanged; copy plain text before testing. The verifier
uses Python/GTK and stock wl-clipboard for backup/restoration only. Use
`--clipboard-only` to exercise GTK paste without opening uinput, and `--fallback`
to force the temporary-surface path on a compositor with data-control support.
The bounded X11 transport checks run without a desktop using
`zig test src/x11_wire.zig -OReleaseSafe`; host X11 acceptance still requires a
manual clipboard read from an Xorg session or through XWayland.

`src/clipboard.zig` selects one backend for the session. On Wayland,
`src/clipboard_wayland.zig` owns discovery, text generations and transfers while
`src/wayland_wire.zig` handles the native socket and descriptor transport. It
prefers ext-data-control, then wlr-data-control, then core data-device with an
xdg-shell surface and GNOME's GTK surface extension when available. The
connection reserves 64 KiB for incoming frames and 4 KiB for outgoing requests.
Acquisition waits for the fallback surface's destruction to reach the compositor,
but cannot guarantee the eventual focus target of a globally injected shortcut.

On X11, `src/clipboard_x11.zig` owns a hidden InputOnly selection window and
`src/x11_wire.zig` handles the local Unix socket, bounded `.Xauthority` parsing,
and MIT-MAGIC-COOKIE-1 setup. It serves `TARGETS`, `TIMESTAMP`, `UTF8_STRING`,
`TEXT`, UTF-8 plain-text aliases, and bounded `INCR` streams. `MULTIPLE`, legacy
`STRING`, nonzero screens, remote TCP displays, and other `DISPLAY` transports
are intentionally unsupported. Both backends use no C bindings, dynamic
libraries, worker processes or extra threads. Text remains
borrowed while offered or being transferred; two generations and eight transfers
bound retention, and stalled transfers expire after two seconds.

## Desktop notifications

`src/notifications.zig` owns error presentation, replacement, dismissal, and
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

Run the checks without contacting the desktop. The Python verifier requires
`blake3` in the optional environment above to fingerprint its executable:

```bash
cd zig
zig build notification-check
agent-run 'zig test src/dbus_tests.zig -OReleaseSafe' '../.venv/bin/python scripts/verify-notifications.py --self-test'
cd ..
```

For visual verification, run `.venv/bin/python zig/scripts/verify-notifications.py`
from your desktop terminal. It uses the optional `voiced-notification-check` executable to
show two synthetic errors, replace the first with the second, and close the
popup. It never records audio or changes the clipboard, configuration, or running
service. A private directory under `/tmp/voiced-native-notifications-*` retains
`report.json`, readable client logs, and structured journal records. Acceptance
by the server is logged; visual appearance still requires a person to check.

## Journal logging

Service and worker diagnostics use `src/logging.zig`, including foreground
`voiced serve`. CLI argument/help/error messages may use stderr directly;
command responses and explicit transcript output keep their stdout contracts.
Offline replay uses the same logger with a CLI stderr sink. Zig's emergency
panic/stack-trace machinery retains its stderr path.

Set `log_level=info` in the config or `--log-level info` on the command line.
Levels are `critical`, `error`, `warn`, `info`, and `debug`; the threshold includes
all more severe events. Worker launches explicitly carry the effective level.
Changing the config requires a restart. At `warn` or above, info-level readiness
and effective-configuration messages are intentionally filtered.

```bash
journalctl --user -u voiced -f -o short-precise
journalctl --user -u voiced -p warning
# Foreground serve has no voiced.service unit:
journalctl --user -t voiced -f
```

Every service event is a single native journal datagram with real `PRIORITY`,
`SYSLOG_IDENTIFIER=voiced`, and `VOICED_COMPONENT`. Where supplied by the caller,
`VOICED_RECORDING_ORDINAL` provides the same recording identity in supervisor
and transcription-worker events, without logger-owned recording state. Model
cache events can instead be correlated by their journal PID. Newlines in error details stay inside one binary-encoded
`MESSAGE`; they cannot create extra journal fields. Human-readable messages
use snake_case, concept-first field names. Numeric values carry no unit suffix;
time fields name their unit, and `size` always means bytes. Free-text strings
are quoted with escaped quotes, newlines, and control bytes. Typed error payloads
retain their diagnostic representation; MESSAGE is not a rigid parsing API.

Records use fixed stack storage, at most 4 KiB per datagram; oversized messages
end with `[truncated]`. No audio callback or inference kernel logs. Disabled
events skip formatting, and expensive argument preparation must be guarded with
`logging.enabled()`. There is no logging queue, thread, or event-loop registration.
A full or unavailable journal drops the event without blocking or falling back
to stderr. `VOICED_DROPPED` on the next successfully submitted event counts local
submission losses; it does not claim that journald persisted an accepted event.
All threads share the one nonblocking journal socket initialized at startup. Path-based sends allow
subsequent messages to reach journald after it restarts without reconnect state.

For isolated service verification, supply a private datagram stderr socket,
which the logger duplicates instead of writing to the host journal.

## Desktop error notifications

`notification_mode=errors` (the default) reports failed capture/transcription,
unreliable speech, incomplete recordings, and failed clipboard, paste, or save
operations. Recording, transcription progress, ordinary silence, cancellation,
and successful delivery produce no popups. `notification_mode=off` disables the
connection; `transcript_output=stdout` always disables it.

Each service owns its `Error` tagged union and returns `Result { ok, err }`
(`Result(T)` for operations with different success values). Capture errors retain
valid audio and the complete native report, including the exact failed stage and cause.
Transcription reports preserve the stage and exact Zig error name. Paste errors
include the syscall errno, ioctl request/argument or chord/frame progress. Save
errors retain the failing operation, write progress, and any temporary-file
cleanup error. These operational results do not allocate.

The supervisor maps those types to concise notifications. It distinguishes
missing/ambiguous microphones, lost connections, source changes, audio stalls,
model loading/transcription errors, and clipboard/paste/storage errors. It logs
full diagnostics before retaining only the notification category and delivery
outcome. Mic ambiguity logs include the match count and candidate sources; use
`microphone_node` **instead of** `microphone_serial` to select one input.

Clipboard diagnostics retain the selected backend, typed transport/protocol
errors, and native errno. Wayland reports compositor objects, codes and messages;
X11 reports authority/setup failures, response sequences, server opcodes and bad
values. Capture's bounded
source-list presentation can omit entries; the
worker journals the complete observed candidate catalog on selection errors.
For the full cause and affected recording, run:

```bash
journalctl --user -u voiced -b -n 50 --no-pager
```

The supervisor uses sd-bus on the user session bus through its existing epoll and
timerfd loop. There is no notification subprocess or thread. One outstanding
request and one pending operation coalesce bursts; replies have a one-second
timeout. Repeated identical problem/outcome pairs are suppressed within a recording.
Each new recording resets suppression; ignored commands do not. The next error
can replace the existing popup, and a successful recording closes it.
Dismissal invalidates the held ID; notification
server restarts reset IDs and redisplay an unresolved problem. Replacement and
closure address the original server's unique bus name, preventing ID reuse races.

Clipboard/paste failure messages distinguish successful transcript saving from
failed saving. Popup bodies never contain the transcript. Notification failures
are logged and leave recording and output working. A missing or disconnected
session bus disables notifications until voiced restarts; a notification server
can appear or restart on a live bus without restarting voiced. Shutdown attempts
to close a known popup without waiting or flushing the bus. A timed-out Notify
whose reply never arrives has no usable ID; its popup follows desktop expiration.

## Recording metrics

`recording_ordinal` counts accepted recordings within one service lifetime;
`chunk_ordinal` counts their chunks from zero. Failure and cancellation consume
an ordinal. Restarting the service resets the recording sequence. Journal PID
and invocation metadata distinguish service lifetimes.

`Recording requested` marks acceptance before worker startup. `Capture started`
marks the supervisor's first observation of a PipeWire callback. That observation
waits for an event-loop wake-up; it is not the exact first-sample timestamp. `Transcription
complete` precedes clipboard acquisition, paste, and saving. `Paste shortcut sent`
confirms keyboard event submission; it cannot confirm application insertion.
`Desktop notification accepted` confirms the server accepted the request.

Field names put the subject before the measurement. Time values use fixed units
in their names; byte quantities use `size` without another byte suffix:

| Field | Meaning |
|---|---|
| `audio_duration_seconds` | Actual audio duration, excluding encoder silence padding |
| `audio_samples_count` | Samples in this chunk |
| `features_duration_ms` | Measured feature extraction work |
| `inference_duration_ms` | Measured encoder, cross-KV, and decoder work |
| `transcription_compute_duration_ms` | Sum of feature and inference work for consumed results |
| `transcription_compute_speed_ratio` | Audio duration divided by measured computation duration; 12.5 means 12.5 times realtime |
| `transcript_size` | UTF-8 bytes: raw chunk text, or accumulated trimmed text at completion/delivery |
| `tokens_count`, `tokens_count_max` | Generated token count and its inclusive maximum |

Recording computation totals include consumed no-speech chunks. They exclude
model preparation, queueing, failed attempts, text assembly, and desktop delivery;
they are not end-to-end latency.
Capture-end audio duration covers all captured samples, which can exceed the
processed prefix after a failure. A speed with zero audio or computation
duration is `unavailable`. Discarded
recordings have `transcript_size=0`. Transcript contents are never diagnostics.

Durations measure one named operation; elapsed fields measure from a named
reference event. There is no duration relative to the previous log line.

| Timing | Start → end |
|---|---|
| `capture_start_duration_ms` | Accepted command received → first callback observed by supervisor |
| `model_cache_load_duration_ms` | Cache resolution and locking → mapped image read/validation complete |
| `model_convert_duration_ms` | Pristine source resolution/read/validation → conversion complete |
| `model_prepare_duration_ms` | Worker starts model loading → vocabulary, workspace and runtime ready |
| `clipboard_acquire_duration_ms` | Desktop delivery begins → supervisor observes clipboard acquisition |
| `paste_duration_ms` | Clipboard acquisition observed → shortcut submission complete, including settle/key waits |
| `transcript_save_duration_ms` | Save begins → atomic publication finishes, or save returns an error |
| `recording_stop_elapsed_ms` | Stop reference → this milestone |

Cache lookup, publication, remapping, vocabulary loading and runtime initialization
have additional duration fields at `debug`. Main lifecycle, chunk and completion
measurements remain at `info`. Synchronous operations use local timestamps; the
supervisor retains only the recording request timestamp and one delivery boundary,
reused after clipboard acquisition. No callback or per-token timers were added.

With `recording_stop_origin=command`, the stop reference is the supervisor
receiving the accepted stop/toggle command. Repeated stops do not reset it.
With `recording_stop_origin=capture_end`, it is the supervisor observing the
capture thread's final report; work before that observation is outside the elapsed
measurement.
Without either reference, origin and elapsed are `unavailable`. These timings
exclude keybinding/CLI startup and application rendering after the paste shortcut.
Each new recording resets the reference.

Requested cancellation is `info`. Service errors retain their typed diagnostics
at `error`; a missed cancellation or unload deadline is `critical` and exits the
daemon. Thread crashes follow the configured panic/core-dump behavior. No-speech evidence is
included in the chunk event instead of a duplicate rejection event.

Expected startup refusals such as an already running instance are `error`.
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
processing performs no allocation,
logging, model work, or blocking I/O. Inference borrows each sealed slot without
quantization or a second waveform allocation. The fixed PCM exchange occupies
about 5.49 MiB.

`pipewire_protocol.zig` owns framing, SPA PODs and descriptor transfer;
`pipewire_native.zig` owns discovery, source identity, graph mappings and DSP
cycles. The 128-entry catalog stores kind-specific fields and string references;
a reusable 64 KiB string pool covers every entry at its maximum string lengths.
Protocol queues reserve 64 KiB input and 32 KiB output. `pipewire.zig` owns recording policy and shared-slot publication through
one worker poll loop. `audio_resampler.zig` retains the conversion phase and
filter history. `realtime.zig` requests realtime scheduling directly, falling
back to RTKit over native D-Bus with a 500 ms deadline. Scheduling and memory
locking remain best effort and their actual results are reported.

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
no-speech results; disagreement is an explicit failure.

Each transcript buffer is sized once at 64 UTF-8 bytes per configured recording
second plus one full 4 KiB mailbox result. The one-hour default reserves about
229 KiB per buffer: one for stdout, two (about 458 KiB) for clipboard/desktop.
Unused payload pages are not eagerly initialized. Reaching this bound, the 4 KiB chunk-text bound, or the decoder's
446-token bound stops further recording/transcription and delivers the available
text. The final prefix ends at a complete UTF-8 character; words or sentences may
be incomplete. Desktop mode copies, sends the paste shortcut, then saves;
clipboard mode copies and saves. Publication, transfers and saving borrow the
same completed buffer; no buffer grows or is allocated per recording.

A notification distinguishes recording size, chunk size, and decoder token
limits. Decoder-limit text is retained even when confidence is low: it may repeat
or contain inaccuracies, so the notification asks you to check it. Both chunk
limits are reported if reached together. Chunk/token limits still attempt the
`last-failed/` diagnostic capture before returning the prefix. A fatal restart
during inference or diagnostic saving can lose the current recording. Explicit cancellation suppresses pending output
and preserves the previous saved transcript. Start a new recording to continue;
unprocessed audio after the cutoff is not resumed automatically.

The supervisor requires the first callback within three seconds and progress
at least every two seconds. Each inference has a ten-second deadline; model
startup has fifteen seconds. Stop, cancellation, model unload, and final thread
shutdown each have a one-second acknowledgement deadline. Capture alone closes
its PipeWire stream. A missed stop/unload deadline exits the whole daemon before
cleanup can invalidate a worker borrow; systemd restarts the service. The current
recording is not retried across a fatal restart.

## Model cache and runtime

The worker verifies installed weights, converts them once, and atomically
publishes a packed image under `$XDG_CACHE_HOME/voiced/models/`. Cache identity
includes the model, pristine checksum, packing revision, and image format.
Cache envelope version 2 uses BLAKE3-256 for both the pinned source identity and
the cached payload. Older cache entries are ignored and rebuilt from verified
installed weights; existing model downloads remain usable. Private permissions, a builder lock,
atomic replacement, and file/directory synchronization prevent partial cache
publication. Unavailable caches fall back to verified pristine conversion;
invalid pristine data fails explicitly. Preparation cancellation is checked
between loading stages and while waiting for the builder lock.

The caller supplies an address-stable runtime and its tensor arena:

```zig
var runtime: inference.Runtime = undefined;
try runtime.init(io, &model, vocabulary, memory, policy);
defer runtime.deinit();
```

Do not copy or move the runtime until `deinit` joins its workers. The caller
then releases the arena and any heap storage used for the runtime itself.
Transcription allocates nothing and returns untrimmed text in the supplied
buffer. The supervisor trims only the assembled final transcript. Model reloads
allocate and release weights, vocabulary, and runtime storage; the service
mailboxes and audio/transcript buffers are reused. Zig 0.16.0 uses its debug
allocator in libc-free ReleaseSafe builds, so freed small allocations can
retain bucket pages between reloads without retaining a loaded model.

[The runtime guide](runtime/README.md) explains the native Whisper implementation.

The process-level integration harness was removed. Its control, logging,
notification, capture, delivery, recovery, and cancellation scenarios remain in
Git history for future porting. The standalone notification verifier above and
runtime corpus checks remain available. Always use `VOICED_INSTANCE=test` and
private runtime/state paths for automated service checks.
