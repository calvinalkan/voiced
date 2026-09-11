# voiced

Voiced is a native Zig dictation daemon for Linux Wayland and X11. It captures
through PipeWire, transcribes locally with a custom high-performance Whisper
runtime, publishes through a native clipboard client, and pastes through uinput.

```text
CLI/control socket -> supervisor -> capture/transcription -> clipboard/paste
```

## Code map

- `build.zig`: build profiles, release, setup, tests, and developer tools.
- `src/main.zig`: executable root and binary-size configuration.
- `src/service/supervisor.zig`: lifecycle state machine and coordination.
- `src/capture/`: PipeWire capture, realtime processing, and resampling.
- `src/inference/`: Whisper runtime and separately compiled LLVM object.
- `src/clipboard/`: Wayland/X11 ownership and transfers.
- `src/root_test.zig`: test aggregator, including repository lint.
- `tools/zig_lint/`: self-contained ZigLint package, CLI, rules, and tests.

Use `README.md` for operation and configuration, `docs/inference.md` for
inference design, and `__fixtures__/librispeech/README.md` for corpus controls.

## Build

Discover the authoritative steps and options with:

```bash
zig build --help
```

Common commands:

```bash
zig build                         # Debug app/stdlib, ReleaseSafe LLVM inference
zig build release                 # fixed stripped compact release
zig build setup                   # install the default model
zig build setup -- --model all    # install every supported model
```

`release` rejects profile `-D` options and arguments after `--`. Normal builds
retain configurable diagnostics and optimization options; `-Dllvm` overrides
Zig's application-backend selection without affecting LLVM inference.

## Checks

Run checks through `agent-run`; read its failure log before continuing.

```bash
agent-run 'zig build -Doptimize=ReleaseSafe install'
agent-run 'zig build release'
agent-run 'zig build test'
```

When changing ZigLint itself, also run its internal package tests:

```bash
agent-run 'cd tools/zig_lint && zig build test'
```

`zig build test` is the complete repository check. It runs Debug application
tests with Zig's selected backend, ReleaseSafe inference tests against the
production LLVM object and exported ABI, repository ZigLint, and canonical
formatting. Tests never rewrite source files. Pass a case-sensitive test-name
substring after `--` for a focused run:

```bash
zig build test -- inference.root_test
zig build test -- "repository source passes lint"
zig build test -Ddeveloper=false -- "object boundary"
```

## Lint and fixes

```bash
zig build zig-lint
./zig-out/bin/zig-lint src
./zig-out/bin/zig-lint --fix src
```

`--fix` atomically writes compatible rule fixes and canonical formatting, then
reports remaining diagnostics. Re-run the repository lint test afterward.
ZigLint documentation lives under `tools/zig_lint/`.

## Debugging

Discover runtime commands and inspect the daemon before reading implementation:

```bash
./zig-out/bin/voiced --help
./zig-out/bin/voiced help serve
./zig-out/bin/voiced status
systemctl --user status voiced.service
```

Follow or filter native journal records:

```bash
journalctl --user -u voiced -f -o short-precise
journalctl --user -u voiced -p warning
journalctl --user -u voiced VOICED_COMPONENT=capture
journalctl --user -u voiced VOICED_RECORDING_ID=21
```

For foreground debugging, use
`voiced serve --log-target stderr --log-level debug`. To send foreground logs to
the journal instead, select `--log-target journal` and follow them with
`journalctl --user -t voiced -f`.

## Core design constraints

- Keep the deployment artifact a single zero-dependency, fully static executable
  with no runtime shared-library dependencies.
- Integrate through native PipeWire, Wayland, X11, D-Bus, control-socket, and
  uinput protocols; do not shell out to desktop CLIs or helper processes.
- Treat binary size as a design constraint and keep the implementation light.
  `zig build release` enforces its limit, while
  `zig build test:executable-size` checks a selected non-Debug profile. Use
  `bloaty zig-out/bin/voiced` to investigate growth and `bloaty --help` to
  discover analysis options.
- The PipeWire process callback runs on its real-time audio thread. Keep it
  allocation-free, lock-free, and free of blocking I/O, logging, and model work
  so capture remains responsive under CPU load.
