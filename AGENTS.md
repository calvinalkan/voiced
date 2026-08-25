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
│  voiced CLI                                                        │
│  ──────────                                                        │
│  CLI interface: serve, listen, record, stop, status, history, kill  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ Unix control socket        │
                    │ {"cmd": "listen"}         │
                    │ under $XDG_RUNTIME_DIR     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  daemon.py                                                          │
│  ─────────                                                          │
│  Main loop: owns lifecycle, socket commands, and worker deadlines   │
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

## Lifecycle invariants

- The daemon main loop is the only owner of `OperationPhase` transitions,
  command dispatch, stop deadlines, and shutdown order. Capture and
  transcription threads report completion through `worker_events`; they must
  not publish lifecycle state directly.
- The capture thread exclusively owns its PortAudio stream. It aborts and
  closes the stream itself. Do not call stream operations from the watchdog or
  pretend a blocked Python thread was cancelled by resetting daemon state.
- Stream setup has a three-second deadline. A missing callback for two seconds
  requests stop. If abort/close returns, finalize and insert queued audio, stay
  alive, and open a fresh stream on the next recording. If the capture owner
  remains in native teardown, finalize and insert queued audio before exiting
  `os.EX_TEMPFAIL`; systemd terminates the blocked thread and restarts.
- Shutdown first stops and joins capture, then stops the transcriber, then the
  typer. Never enqueue `transcriber.shutdown()` while capture may still call
  `feed()` or a transcription thread may still call `finalize()`.
- `idle`, `capturing`, `stopping`, and `transcribing` are externally observable
  through `voiced status`. Toggles during `stopping` or `transcribing` are
  intentionally ignored and logged at debug priority.

## Runtime and logging

The control protocol is one newline-terminated JSON request and response per
Unix connection. The default socket is
`$XDG_RUNTIME_DIR/voiced/control.sock`; named instances use
`$XDG_RUNTIME_DIR/voiced-<instance>/control.sock`. Keep the CLI syntax stable,
but do not restore PID files, command files, signals, or compatibility code for
the removed `/tmp` protocol.

Use `voiced_logging.log()` for daemon and worker output rather than `print()` or
stderr routing. The logger serializes threads and maps `debug`, `info`, `warn`,
`error`, and `critical` to journal priorities 7, 6, 4, 3, and 2. Debug calls are
filtered unless debug mode is enabled. Unexpected exceptions retain traceback
lines at error priority; expected microphone/configuration failures use one
concise event.

The model remains resident. Audio capture stores float32 chunks and passes the
same arrays to the transcriber queue. Do not convert callback blocks to Python
float lists or run model work on the callback thread.

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

Test daemon behavior through process-level integration scenarios, not isolated
unit tests. Extend `test.sh` with a real daemon, CLI request, and observable
process/socket/output result for lifecycle, protocol, recovery, or shutdown
changes. `test_transcriber.py` remains the existing model-worker suite.

WAV files in `./test-fixtures/` exercise the full pipeline without a microphone.

**Important:** Always set `VOICED_INSTANCE=test` when testing to avoid conflicts with the main daemon (e.g., running via systemd). This creates a separate control socket under `$XDG_RUNTIME_DIR/voiced-test/`.

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
| `VOICED_INSTANCE` | Instance name for a separate runtime directory and socket (e.g., `test`) |
| `VOICED_TEST_INPUT` | Read audio from WAV file instead of microphone |
| `VOICED_TEST_OUTPUT` | Write transcription to file instead of typing |
| `VOICED_SAVE_AUDIO` | Save recorded audio to WAV file |
| `VOICED_TEST_CAPTURE_FAULT` | With a named test instance, use `stall` or `hang` to exercise capture recovery |

The fixture and output variables work with both `voiced listen` and `voiced record`.
Fault injection exists only for the process-level integration suite.
