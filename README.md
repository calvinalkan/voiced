# voiced

Voice-to-text dictation for **Linux + Wayland**. Hit a hotkey, speak, and the
transcribed text gets pasted into whatever window has focus.

Runs entirely on your machine — no cloud, no API keys, no telemetry. Works
in browsers, terminals, editors, chat apps, anywhere you can paste.

## What problem this solves

Voice dictation tools that "just work" — Wispr Flow, Superwhisper, Apple
Dictation — are all macOS-only or cloud-dependent. Linux options like
nerd-dictation or willow tend to be either incomplete, X11-only, or rough
around the UX edges.

`voiced` is the missing piece for Linux: a small persistent daemon that
records, transcribes locally, and pastes — fast enough that it disappears
into your workflow.

## Supported

- **OS:** Linux with Wayland (tested on Ubuntu 24.04+ / GNOME).
  X11 not supported (uses `wl-copy` + Wayland-aware tools).
- **CPU:** any modern x86_64 or ARM. No GPU required.
- **Languages:** English (and whatever else Whisper supports if you want
  to switch the language config).

## Requirements

| Tool | Why | How to get it |
|---|---|---|
| Python 3.12+ | runtime | apt / your distro |
| [`uv`](https://github.com/astral-sh/uv) | dep management | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `dotool` *or* `ydotool` | synthetic keyboard input | dotool: `./build-dotool.sh` (in this repo). ydotool: `apt install ydotool` |
| `wl-clipboard` | clipboard access | `apt install wl-clipboard` |
| `libnotify-bin` | desktop notifications | `apt install libnotify-bin` |
| A microphone | recording | (built into your laptop) |

## Install

```bash
git clone git@github.com:calvinalkan/voiced.git
cd voiced
uv sync
```

If you don't already have `dotool` installed (recommended over `ydotool` —
it handles your keyboard layout correctly):

```bash
./build-dotool.sh
# Follow the printed instructions for the udev rule (one-time sudo).
```

Then install the systemd user service so the daemon starts at login:

```bash
./install-service.sh
```

The service is now running. Verify:

```bash
systemctl --user status voiced
```

## Quick check

In a terminal:

```bash
voiced status     # → "Daemon running (PID: ...)"
voiced listen     # speak; auto-stops after 1s of silence; text appears
```

If text appeared in your terminal, you're done.

## Use it

```bash
voiced listen           # record + auto-stop on silence (best for short messages)
voiced record           # record manually; stop with: voiced stop
voiced record -t        # toggle: same command starts/stops
voiced stop             # explicit stop
voiced status           # is the daemon alive?
voiced kill             # stop the daemon
voiced history          # last N transcriptions
voiced history 3        # re-paste entry 3
voiced history 3 -c     # copy entry 3 to clipboard instead
```

The daemon must be running (`voiced serve` in foreground, or via the systemd
service). The CLI exchanges one request with its Unix control socket under
`$XDG_RUNTIME_DIR/voiced/`.

## Bind a hotkey

You'll want to trigger `voiced` from a keyboard shortcut, not the terminal.

### GNOME

Settings → Keyboard → View and Customize Shortcuts → Custom Shortcuts → Add:
- **Name:** Voice toggle
- **Command:** `voiced record -t`
- **Shortcut:** whatever you like (Super+Space, F8, Pause, ...)

For "press to talk" style, use `voiced listen` and pick a key that's easy
to tap and forget.

### KDE Plasma

System Settings → Shortcuts → Custom Shortcuts → New → Global Shortcut →
Command/URL → command: `voiced record -t`.

### Hyprland / Sway

```
bind = SUPER, space, exec, voiced record -t
```

### Why `record -t` (toggle)?

Toggle mode is the most flexible: press once to start, press again when
done. Works for messages of any length, doesn't care about pauses, and
doesn't auto-stop if you pause to think.

`voiced listen` is for "speak this short thing and shut up" and auto-stops
after 1 second of silence — useful for one-line commands but cuts you off
mid-sentence if you pause.

## Configure

Optional. Create `~/.config/voiced/config.json`. Recommended defaults for a
typical CPU laptop with no GPU:

```json
{
  "transcriber_engine": "whisper",
  "model": "small.en",
  "silence_duration": 1.0
}
```

That gets you ~1-2 second wait at the end of a 20s recording, very accurate
transcription, and no CPU drama while you're talking.

### All config keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `transcriber_engine` | str | `"whisper"` | `"whisper"` (recommended on CPU) or `"moonshine"` |
| `model` | str | `"base"` | model name; depends on engine — see below |
| `streaming` | bool | `false` | live streaming inference (Moonshine only). On by default = no; only enable if your CPU can keep up. |
| `whisper_vad_filter` | bool | `false` | run faster-whisper's extra VAD after voiced records audio; usually leave off because voiced already handles speech/silence detection |
| `device` | str | `"cpu"` | Whisper execution device: `"cpu"` or `"cuda"`; Moonshine requires `"cpu"` because its current API exposes no device selection |
| `input_device` | str/null | `null` | case-insensitive microphone name fragment; `null` uses the system default |
| `silence_threshold` | float | `0.02` | amplitude threshold for "is this speech" (0.0–1.0) |
| `silence_duration` | float | `0.8` | seconds of quiet before auto-stop in `listen` mode |
| `speech_start_duration` | float | `0.2` | sustained speech needed to start recording |
| `auto_enter` | bool | `false` | press Enter after pasting (great for chat apps) |
| `clipboard_copy` | bool | `true` | leave the dictation on the clipboard after pasting |
| `insertion_method` | str | `"paste"` | `"paste"` (Ctrl+Shift+V) or `"type"` (per-char keystrokes) |
| `paste_keybind` | str | `"ctrl+shift+v"` | keybind to send in paste mode |
| `typer_backend` | str | `"auto"` | `"auto"`, `"dotool"`, or `"ydotool"` |
| `keyboard_layout` | str/null | autodetected | XKB layout (e.g. `"de"`, `"us"`) — only matters in `type` insertion mode |
| `history_size` | int | `20` | keep this many recent transcriptions |
| `debug` | bool | `false` | verbose lifecycle, device, and model-worker logs |

When `input_device` is set, voiced resolves the PipeWire source before every
recording and makes that source the system default before opening the audio
stream, so USB device indexes may change safely. It refuses to record if the
name is missing or matches multiple inputs; it never silently falls back to
another microphone. Debug logging lists the available inputs, the selected
match, and selection latency.

### Available models

**Whisper** (via faster-whisper / CTranslate2 — fastest on CPU):

| Model | Size | Languages |
|---|---|---|
| `tiny`, `base`, `small`, `medium`, `large-v3` | small → huge | 99 languages |
| `tiny.en`, `base.en`, `small.en`, `medium.en` | same sizes | English only — typically more accurate on English |

**Moonshine** (via ONNX runtime):

| Model | Size | Notes |
|---|---|---|
| `tiny`, `base` | small | non-streaming |
| `tiny-streaming`, `small-streaming`, `medium-streaming` | small → large | streaming-trained variants |

### Recommended configs

**Most users (CPU laptop, English):**

```json
{ "transcriber_engine": "whisper", "model": "small.en" }
```

If `small.en` feels a bit slow, drop to `"base.en"` for ~3× speed at slightly
lower accuracy. If you want maximum accuracy and don't mind ~5-10s of
finalize wait, try `"medium.en"`.

**Fast CPU, want streaming end-of-speech latency:**

```json
{
  "transcriber_engine": "moonshine",
  "model": "small-streaming",
  "streaming": true
}
```

**Very low-power machine (Raspberry Pi, etc.):**

```json
{
  "transcriber_engine": "moonshine",
  "model": "tiny-streaming",
  "streaming": true
}
```

### Per-flag CLI overrides

Useful for one-off testing without editing config:

```bash
voiced serve --transcriber-engine whisper --model small.en --debug
voiced serve --transcriber-engine moonshine --model tiny-streaming --streaming
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                                User                                     │
│                                  │                                      │
│                  hotkey ──┐      │                                      │
│                           ▼      ▼                                      │
│             ┌──────────────────────────────────┐                        │
│             │  voiced CLI                      │                        │
│             │  serve / listen / record / stop  │                        │
│             │  status / kill / history         │                        │
│             └─────────────┬────────────────────┘                        │
│                           │                                             │
│        one JSON request on a Unix socket                                │
│        under $XDG_RUNTIME_DIR/voiced                                    │
│                           │                                             │
│                           ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │   daemon.py     (one persistent process for the user's session)    │ │
│  │                                                                    │ │
│  │   Long-lived components owned by the daemon:                       │ │
│  │                                                                    │ │
│  │   ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐   │ │
│  │   │  Audio       │  │  Transcriber     │  │  Typer             │   │ │
│  │   │  (sounddev)  │  │  (worker thread) │  │  (dotool subproc)  │   │ │
│  │   │              │  │                  │  │                    │   │ │
│  │   │  silence     │  │  Whisper or      │  │  paste via         │   │ │
│  │   │  detection   │  │  Moonshine       │  │  Ctrl+Shift+V      │   │ │
│  │   │              │  │  (buffered or    │  │  (fallback to type)│   │ │
│  │   │              │  │   streaming)     │  │                    │   │ │
│  │   └──────┬───────┘  └────────┬─────────┘  └──────────┬─────────┘   │ │
│  │          │                   │                       │             │ │
│  │          │  on_chunk ────►   │  feed/finalize ───►   │             │ │
│  │          │                                            │             │ │
│  │          ▼                                            ▼             │ │
│  │   microphone                                    focused window      │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                           │                                             │
│                           ▼                                             │
│           ~/.local/state/voiced/history.json                            │
│           (last N transcriptions, recoverable via voiced history)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How a recording flows

1. **You press the hotkey.** GNOME / KDE / your WM runs `voiced record -t`.
2. **CLI sends a command.** It opens the per-user Unix socket, sends one JSON
   request, and receives the daemon's acceptance or current phase.
3. **Daemon starts a session.** The main loop enters `capturing` and calls
   `transcriber.start_session()` (cheap — it only queues a worker message).
4. **Audio captures chunks.** Sounddevice fires a callback every ~100ms.
   The callback copies each reusable PortAudio block once into float32 storage,
   updates silence detection, and queues the same block for the model worker;
   it never runs encoder work or creates Python float objects.
5. **Worker processes chunks.**
   - In *buffered* mode: just appends to a list. CPU idle.
   - In *streaming* mode: feeds chunks live to the model encoder. Encoder
     work happens here, on the worker thread, off the audio thread.
6. **You press the hotkey again** (or stop talking, depending on mode). The
   capture owner aborts and closes the input stream; a daemon deadline restarts
   the service if PortAudio does not return.
7. **Daemon enters `transcribing`.** The worker drains any remaining queue work
   and either runs one buffered transcribe call or stops the live stream and
   reads accumulated lines. Hotkey toggles are ignored during this phase.
8. **Text gets pasted.** Goes through the clipboard via `wl-copy`, then
   the typer sends `Ctrl+Shift+V` via dotool. Fallback to per-character
   typing if paste fails.
9. **History is saved.** `~/.local/state/voiced/history.json`, last N entries.

### Why a persistent worker thread?

The naive approach — run the model directly on the audio capture callback —
silently drops audio frames whenever the model is slower than real-time.
The persistent worker keeps encoder work off the audio callback. If streaming
inference cannot keep up with real time, its queue grows during recording and
drains at the end (visible in the debug log as `queue_wait Xms + stop Yms`).

### Failure recovery

The capture thread owns the PortAudio stream and always aborts it before
closing it. The main loop separately watches progress: stream setup may take
at most three seconds, and two seconds without an audio callback requests
capture stop. If the stream aborts and closes, the daemon finalizes the
buffered speech, saves it to history, inserts the partial transcript, and stays
running. A reconnected USB microphone is opened as a new stream on the next
recording; the failed stream is never resumed.

A stop that does not finish within two more seconds means the capture thread is
still trapped in native PortAudio teardown. Every completed callback has
already queued its audio on the model worker, so the daemon still finalizes and
inserts that partial transcript before exiting with `75/TEMPFAIL`. If no usable
audio arrived, it exits immediately. Systemd then kills the blocked thread and
starts a clean daemon after one second.

A normal service stop follows a different path: the daemon waits for capture or
transcription to finish, then shuts down the model worker and typer in order.
Its internal shutdown deadline is five seconds, while systemd enforces a final
ten-second bound. Configuration errors exit with `78/CONFIG` and do not restart.

### Logs

The daemon sends all components through one ordered logger. Under systemd each
line carries a native journal priority: debug 7, info 6, warning 4, error 3, and
critical 2. Writing to stderr alone does not assign an error priority. Debug
events are emitted only when `debug` is enabled in the config or with
`voiced serve --debug`; normal transcription logs retain their short transcript
preview at info level.

```bash
journalctl --user -u voiced -f -o short-precise       # follow all logs
journalctl --user -u voiced -p warning..emerg         # warning and worse
journalctl --user -u voiced -p debug..debug           # debug only
```

A recoverable stream failure can be followed from `callback_stall` to
`capture_recovered`. If native teardown remains stuck, the journal continues
through `capture_stop_timeout`, `75/TEMPFAIL`, and the next `ready` event.

### Memory behavior

The Whisper model remains resident intentionally so every recording is ready
immediately. It accounts for most idle memory. Captured audio uses float32
chunks shared by capture and the model queue rather than Python float objects;
a 30-second recording requires about 1.8 MiB of sample storage instead of
roughly 15 MiB. Finalization may temporarily allocate contiguous model input,
but the per-callback chunks are released after the session.

### Why a clipboard paste, not just typing?

Per-character typing via uinput is rate-limited at multiple layers
(kernel, compositor, target app). Long messages get truncated or trigger
crashes (we hit Wayland EPIPE in Zed). Clipboard paste is one keystroke
regardless of message length — fast, reliable, and what every serious
dictation app does.

`Ctrl+Shift+V` instead of `Ctrl+V` because it's the universal Linux paste:
works in terminals, works as plain-paste in browsers (no rich formatting),
works in nearly every GUI app.

## Spikes

Durable native CTranslate2 optimization experiments are stored outside the
production repository:

- `/home/calvin/code/experiments/2026-09-03-zig-cpp-custom` — current best
  Whisper-small runtime, combining CTranslate2 orchestration and MKL with
  shape-specialized Zig AVX2/VNNI kernels. It includes the modified C++ and
  Zig sources, vendored build dependencies, static artifacts, binaries,
  fixtures, benchmark evidence, and rebuild instructions.
- `/home/calvin/code/experiments/2026-09-03-ctranslate2-cpp-variants` —
  reconstructed C++/MKL-only optimization lineage: packing-only, packing plus
  attention-buffer reuse, scoped packing, and the strongest measured fixed-input
  C++ path. It includes normalized patches, historical binaries, evidence, and
  a script that materializes each variant from pinned CTranslate2 upstream.

These are research snapshots and are not used by the installed `voiced`
runtime.

## Testing

```bash
make check                 # lint + typecheck + fast tests (~20s)
make test-realtime         # adds the slow ~30s real-time-paced streaming test
```

`test-fixtures/hello_world.wav` and `test-fixtures/paragraph.wav` cover the
basic pipeline and long-form behavior. The integration test starts real daemon
processes and exercises the CLI socket, transcription, ignored busy toggles,
ordered shutdown, partial-transcript preservation with and without a process
restart, and stale-socket recovery. The
`dictation-*.wav` fixtures are real Shure MV7 recordings covering ordinary
speech, technical vocabulary, and numbers; the transcriber suite checks their
distinctive phrases with Whisper `small.en`.

## Troubleshooting

**`voiced status` says "Daemon not running"** → start it with
`systemctl --user start voiced`. Inspect recent lifecycle events with
`journalctl --user -u voiced -n 50`; warnings and errors alone are available
with `journalctl --user -u voiced -p warning..emerg`.

**The configured microphone is unavailable or ambiguous** → voiced refuses to
record, logs the error, and sends a critical desktop notification. Run the
daemon with `--debug` to list the input devices seen by PipeWire, then make
`input_device` more precise or reconnect the microphone.

**Paste doesn't land in app X** → try `"insertion_method": "type"` in your config. Some sandboxed apps (Citrix, password managers) block clipboard paste.

**Transcription is gibberish on long recordings** → you might be in `streaming: true` mode with a model that's too heavy for your CPU. Set `"streaming": false` or pick a smaller model.

**Transcription suddenly becomes tiny or unrelated** → check the `[audio] done (...)`
line in `journalctl --user -u voiced -n 50`. Low `speech_chunks`, very low
`peak`, high `clipped`, or high `near_zero` means the captured mic audio is bad.
To keep a failing sample for inspection, run one dictation with
`VOICED_SAVE_AUDIO=/tmp/voiced-bad.wav voiced record -t`.

**Transcription degrades under CPU load** → look for `[audio] [warn] audio callback`
lines in the journal. `input overflow`, delayed callbacks, or audio shorter than
wall time means the OS/audio stack starved microphone capture before Whisper saw
it. A callback that disappears for two seconds triggers automatic recovery:
voiced finalizes the partial recording and stays alive if the failed stream
closes. It exits `TEMPFAIL` only when native teardown remains stuck, after first
processing the queued audio. Buffered Whisper should only spend model CPU after
recording stops.

**Typing skips characters** → you've hit the synthetic-input rate limit. Either switch to paste mode (the default) or use a slower typedelay (the daemon already does this in fallback type mode).

**Want lower CPU during recording** → make sure `streaming: false` (the default for new configs). Streaming mode runs the model continuously; buffered runs it once at the end.
