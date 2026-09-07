# voiced

Voiced is a native Zig dictation daemon for Linux/Wayland. The current usage and
implementation documentation is in `zig/README.md` and `zig/runtime/README.md`.
The Python application has been removed; remaining Python code supports tests
and offline reference comparisons.

## Ownership

- `zig/src/supervisor.zig` owns lifecycle transitions, public commands, worker
  deadlines, and desktop delivery through one event loop.
- The isolated audio worker owns its PipeWire stream. Capture callbacks perform
  bounded validation/copying into shared Float32 slots, with no allocation,
  blocking I/O, model work, or logging.
- The isolated model worker borrows sealed audio and publishes bounded results.
  Reuse shared storage only after the previous owners have finished or exited.
- The runtime and its arena remain at stable addresses until workers are joined.
- Configuration and environment-derived paths are resolved at initialization.
- Operational diagnostics use `zig/src/logging.zig`. CLI output retains its
  stdout/stderr contract. Keep transcript contents out of operational logs.

## Checks

Run checks through `agent-run`; read the saved log when a command fails.
From the repository root:

```bash
agent-run 'cd zig && zig build -Doptimize=ReleaseSafe install replay'
agent-run 'zig fmt --check zig/src zig/runtime zig/scripts/setup.zig zig/build.zig zig/build.zig.zon'
VOICED_INSTANCE=test agent-run './test.sh --zig-control' './test.sh --zig-logging' './test.sh --zig-output'
VOICED_INSTANCE=test VOICED_ZIG_MODEL_TESTS=1 agent-run './test.sh --zig-output'
VOICED_INSTANCE=test agent-run './test.sh --zig-replay'
```

The replay test uses the optional reference environment documented in the native
README. Runtime corpus commands are documented in `zig/audio-fixtures/README.md`.
The old `test.sh --zig` branch targets removed experimental commands; it is not
part of the current checks.

## Test isolation

Always set `VOICED_INSTANCE=test` for service tests. Use private runtime/state
paths, PipeWire graphs, and desktop fixtures. Never exercise the user's live
clipboard, keyboard, microphone, or running service as an automated check.
Test lifecycle behavior through process-level scenarios and observable output.
Do not add new test files unless the user requests them. The existing harness
needs a separate redesign; do not restore production test-scenario switches or
rewrite the service around the old Python tests.
