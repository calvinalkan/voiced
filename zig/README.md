# Voiced Zig prototypes

The selected ASR backend compiles pinned CTranslate2, cpu_features, and spdlog
sources through Zig's bundled C++ toolchain. It links pinned static Intel oneMKL
and OpenMP archives into the executable; only the Linux platform runtime remains
dynamic.

From this directory, install the verified Intel archives required by the build
and the CTranslate2 `small.en` model required at runtime:

```bash
zig build setup
```

They can also be installed independently:

```bash
zig build setup-native
zig build setup-models
```

Then build the fixture binary:

```bash
zig build -Doptimize=ReleaseFast
```

Run the binary directly with the installed model and a PCM WAV:

```bash
./zig-out/bin/ctranslate2-fixture \
  --model ~/.local/share/voiced/models/faster-whisper-small.en \
  --audio ../test-fixtures/hello_world.wav \
  --threads 4 \
  --runs 3
```

`--threads` defaults to 4. `--runs` defaults to 3 and must be an odd number
between 1 and 9. `setup-models` installs under `$XDG_DATA_HOME/voiced/models`
when `XDG_DATA_HOME` is set; otherwise it uses `~/.local/share/voiced/models`.

`-Dmkl-prefix=/path/to/prefix` overrides the downloaded
`zig-pkg/mkl` installation. The prefix must contain `include/`,
`opt/compiler/include/`, and the required static archives under `lib/`.
