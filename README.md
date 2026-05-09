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
service) — the CLI just sends commands to it via signals.

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
| `device` | str | `"cpu"` | `"cpu"` or `"cuda"` if you have a GPU |
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
| `debug` | bool | `false` | verbose chunk-by-chunk logs |

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

**Fast CPU or GPU machine, want streaming end-of-speech latency:**

```json
{
  "transcriber_engine": "moonshine",
  "model": "small-streaming",
  "streaming": true,
  "device": "cuda"
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
│             │  voiced (bash CLI)               │                        │
│             │  serve / listen / record / stop  │                        │
│             │  status / kill / history         │                        │
│             └─────────────┬────────────────────┘                        │
│                           │                                             │
│        JSON line on /tmp/voiced-command                                 │
│        + SIGUSR1                                                        │
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
2. **CLI sends a command.** It writes a JSON line to `/tmp/voiced-command`
   and sends `SIGUSR1` to the daemon.
3. **Daemon starts a session.** `transcriber.start_session()` (cheap — just
   queues a message to the worker thread).
4. **Audio captures chunks.** Sounddevice fires a callback every ~100ms
   with audio samples. The audio thread *only enqueues* chunks (microsecond
   cost) — never runs encoder work, so it can never fall behind.
5. **Worker processes chunks.**
   - In *buffered* mode: just appends to a list. CPU idle.
   - In *streaming* mode: feeds chunks live to the model encoder. Encoder
     work happens here, on the worker thread, off the audio thread.
6. **You press the hotkey again** (or stop talking, depending on mode).
7. **Daemon calls finalize.** Worker drains any remaining queue work and
   either runs a single transcribe call (buffered) or stops the live
   stream and reads accumulated lines (streaming).
8. **Text gets pasted.** Goes through the clipboard via `wl-copy`, then
   the typer sends `Ctrl+Shift+V` via dotool. Fallback to per-character
   typing if paste fails.
9. **History is saved.** `~/.local/state/voiced/history.json`, last N entries.

### Why a persistent worker thread?

The naive approach — run the model directly on the audio capture callback —
silently drops audio frames whenever the model is slower than real-time.
The persistent worker decouples them: audio thread is never blocked, the
model can take however long it takes. If it can't keep up with real-time,
the queue grows during recording and drains at the end (visible in the
debug log as `queue_wait Xms + stop Yms`).

### Why a clipboard paste, not just typing?

Per-character typing via uinput is rate-limited at multiple layers
(kernel, compositor, target app). Long messages get truncated or trigger
crashes (we hit Wayland EPIPE in Zed). Clipboard paste is one keystroke
regardless of message length — fast, reliable, and what every serious
dictation app does.

`Ctrl+Shift+V` instead of `Ctrl+V` because it's the universal Linux paste:
works in terminals, works as plain-paste in browsers (no rich formatting),
works in nearly every GUI app.

## Testing

```bash
make check                 # lint + typecheck + fast tests (~20s)
make test-realtime         # adds the slow ~30s real-time-paced streaming test
```

`test-fixtures/hello_world.wav` and `test-fixtures/paragraph.wav` are the
reference inputs. Add your own — anything in `test-fixtures/` works the
same way.

## Troubleshooting

**`voiced status` says "Daemon not running"** → start it: `systemctl --user start voiced`. If it crashes, `journalctl --user -u voiced -n 50` for logs.

**Paste doesn't land in app X** → try `"insertion_method": "type"` in your config. Some sandboxed apps (Citrix, password managers) block clipboard paste.

**Transcription is gibberish on long recordings** → you might be in `streaming: true` mode with a model that's too heavy for your CPU. Set `"streaming": false` or pick a smaller model.

**Typing skips characters** → you've hit the synthetic-input rate limit. Either switch to paste mode (the default) or use a slower typedelay (the daemon already does this in fallback type mode).

**Want lower CPU during recording** → make sure `streaming: false` (the default for new configs). Streaming mode runs the model continuously; buffered runs it once at the end.
