# voiced

Voice-to-text for Ubuntu/Wayland. Talk into your mic, text appears wherever your cursor is
and will be pasted to your clipboard aswell.

Uses faster-whisper for transcription, runs locally, no cloud stuff. Works in any app - browsers, terminals, editors, whatever has focus gets the text.

## How it works

A daemon runs in the background doing nothing until you trigger it. Hit a hotkey, speak, and it:

1. Records from your mic (stops automatically when you go quiet, or manually)
2. Transcribes with faster-whisper
3. Types the text into your current app via `dotool/ydotool` and to your clipboard with `wl-copy`

History is saved so you can re-type or copy old transcriptions.

## Requirements

- Linux + Wayland
- Python 3.12+
- uv
- dotool or ydotool

## Install

You need Python 3.12+, [uv](https://github.com/astral-sh/uv), and either [dotool](https://git.sr.ht/~geb/dotool) or ydotool for typing.

```bash
git clone https://github.com/user/voiced.git
cd voiced
uv sync

# If you don't have dotool:
./build-dotool.sh
```

## Use it

```bash
# Start daemon
voiced serve &

# Dictate (auto-stops on silence)
voiced listen

# Or manual recording
voiced record          # stop with: voiced stop
voiced record -t       # toggle mode - press again to stop
```

Bind `voiced listen` or `voiced record -t` to a hotkey in your desktop environment.

## History

```bash
voiced history         # list recent transcriptions
voiced history 3       # re-type entry 3
voiced history 3 -c    # copy entry 3 to clipboard
```

## Config

Optional. Create `~/.config/voiced/config.json`:

```json
{
    "model": "base",
    "device": "cpu",
    "silence_threshold": 0.02,
    "silence_duration": 0.8,
    "auto_enter": false
}
```

Models: `tiny`, `base`, `small`, `medium`, `large-v3` (bigger = slower but more accurate)

Or pass flags: `voiced serve --model small --debug`

## Run as a service

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/voiced.service << 'EOF'
[Unit]
Description=Voice dictation daemon

[Service]
ExecStart=%h/path/to/voiced serve
Restart=on-failure

[Install]
WantedBy=default.target
EOF

systemctl --user enable --now voiced
journalctl --user -u voiced -f  # logs
```
