# voiced

Local voice dictation for Linux and Wayland. Record from a PipeWire microphone,
transcribe with the native Zig Whisper runtime, and paste into the focused
application using wl-copy and a persistent uinput keyboard.

The daemon, CLI, and worker roles are built from [`zig/`](zig/README.md). That
README covers dependencies, model setup, configuration, commands, logging, and
saved transcripts. The [runtime guide](zig/runtime/README.md) explains inference
and memory ownership.

From the repository root, with Zig 0.16.0 and the native dependencies installed:

```bash
(cd zig && zig build setup-models && zig build)
./zig/zig-out/bin/voiced serve
```

In another terminal:

```bash
./zig/zig-out/bin/voiced record -t
./zig/zig-out/bin/voiced stop
./zig/zig-out/bin/voiced status
```

Inference currently targets the build host's AVX-VNNI CPU capabilities. The
supported models are Base.en and Small.en. Desktop paste requires access to
`/dev/uinput`; clipboard-only output is also available.

The Python daemon and its installer have been removed. Python is used only by
the existing integration harness and optional offline reference comparison.

Tests currently use two audio sets:

- `test-fixtures/hello_world.wav`: the short microphone-to-output and replay
  fixture used by `test.sh`.
- [`zig/audio-fixtures/`](zig/audio-fixtures/README.md): the 32-clip LibriSpeech
  corpus with reference text and provenance for runtime quality measurements.

See the [native README](zig/README.md) for the existing test commands. The test
harness still contains an obsolete `--zig` branch and source-patching fixtures;
replacing that harness is separate from removing the Python application.
