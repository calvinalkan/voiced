# voiced

Voiced is a native Zig dictation daemon for Linux Wayland and X11 desktops. The current usage and
implementation documentation is in `zig/README.md` and `zig/inference/README.md`.
The Python application has been removed; remaining Python code supports tests
and offline reference comparisons.

## Ownership

- `zig/src/supervisor.zig` owns lifecycle transitions, public commands, worker
  deadlines, and desktop delivery through one event loop.
- The persistent capture thread owns its PipeWire connection per capture.
  It validates borrowed graph buffers and resamples directly into unpublished
  shared Float32 slots, with no allocation, blocking I/O, model work, or logging
  in the capture callback.
- The supervisor owns one native Wayland or X11 clipboard client. Published text
  stays immutable until ownership and all outstanding transfers release its buffer.
- The persistent transcription thread borrows sealed audio and returns bounded
  typed results. Reuse storage only after the previous owners acknowledge completion.
  Cancellation is cooperative; an overdue stop exits the whole daemon for restart.
- The runtime and its arena remain at stable addresses until workers are joined.
- Configuration and environment-derived paths are resolved at initialization.
- Operational diagnostics use `zig/src/logging.zig`. CLI output retains its
  stdout/stderr contract. Keep transcript contents out of operational logs.

## Checks

Run checks through `agent-run`; read the saved log when a command fails.
From the repository root:

```bash
agent-run 'cd zig && zig build -Doptimize=ReleaseSafe install replay'
agent-run 'zig fmt --check zig/src zig/inference zig/scripts/setup.zig zig/build.zig zig/build.zig.zon'
```

Offline replay uses the optional reference environment documented in the native
README. Runtime corpus commands are documented in `zig/audio-fixtures/README.md`.
The process-level harness was removed; its scenarios can be ported from Git
history when requested.

## Test isolation

Always set `VOICED_INSTANCE=test` for service tests. Use private runtime/state
paths, PipeWire graphs, and desktop fixtures. Never exercise the user's live
clipboard, keyboard, microphone, or running service as an automated check.
Test lifecycle behavior through process-level scenarios and observable output.
Do not add new test files unless the user requests them. Do not restore
production test-scenario switches or rewrite the service around the old Python
tests.
