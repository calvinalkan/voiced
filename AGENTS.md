# voiced

Voice dictation daemon for Ubuntu/Wayland. Records audio, transcribes with Whisper, and types text directly into any application.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              User                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  voiced (bash)                                                      │
│  ─────────────                                                      │
│  CLI interface: start, listen, record, stop, history, shutdown      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ /tmp/voiced-command       │
                    │ {"cmd": "listen"}         │
                    │ + SIGUSR1 signal          │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  daemon.py                                                          │
│  ─────────                                                          │
│  Main loop: receives signals, dispatches commands                   │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐            │
│  │   Audio     │───▶│ Transcriber  │───▶│    Typer    │            │
│  │ (record)    │    │  (whisper)   │    │  (dotool)   │            │
│  └─────────────┘    └──────────────┘    └─────────────┘            │
│         │                                      │                    │
│         │ sounddevice          faster-whisper  │ dotool/ydotool     │
│         ▼                                      ▼                    │
│    microphone                             any application           │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
          ~/.local/state/voiced/history.json
```

## Usage

```bash
voiced listen         # Start listening (auto-stop on silence)
voiced record         # Manual record (stop with 'voiced stop')
voiced record -t      # Toggle recording on/off
voiced stop           # Stop recording
voiced history        # Show transcription history
voiced status         # Check daemon status
voiced serve          # Start daemon (foreground)
voiced kill           # Stop daemon
```

## Commands

```bash
make              # Run all checks (lint + typecheck)
make lint         # Run ruff linter
make format       # Auto-format code
make fix          # Auto-fix linter issues
make typecheck    # Run type checkers (ty + basedpyright)
```

## Testing

Test the full pipeline without a microphone. WAV files in `./test-fixtures/` can be used as input.

**Important:** Always set `VOICED_INSTANCE=test` when testing to avoid conflicts with the main daemon (e.g., running via systemd). This uses separate PID/command files (`/tmp/voiced-test.pid`, `/tmp/voiced-test-command`).

```bash
# Set test instance for all commands
export VOICED_INSTANCE=test

# Start daemon in a tmux session
tmux new-session -d -s voiced-test 'VOICED_INSTANCE=test voiced serve --debug'

# Test with audio file, write result to file (no typing, no clipboard)
VOICED_TEST_INPUT=./test-fixtures/hello_world.wav VOICED_TEST_OUTPUT=/tmp/result.txt voiced listen

# Check results
cat /tmp/result.txt                              # transcription output
tmux capture-pane -t voiced-test -p              # print daemon logs from tmux

# Attach to session for live debugging
tmux attach -t voiced-test

# Kill the session when done
tmux kill-session -t voiced-test

# Create new test fixture from real recording (a user will have to speak into the microphone)
VOICED_SAVE_AUDIO=./test-fixtures/new-test.wav voiced listen
```

| Variable | Description |
|----------|-------------|
| `VOICED_INSTANCE` | Instance name for separate PID/command files (e.g., `test`) |
| `VOICED_TEST_INPUT` | Read audio from WAV file instead of microphone |
| `VOICED_TEST_OUTPUT` | Write transcription to file instead of typing |
| `VOICED_SAVE_AUDIO` | Save recorded audio to WAV file |

All variables work with both `voiced listen` and `voiced record`.
