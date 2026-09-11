# ZigLint

ZigLint is a native Zig source linter with built-in structural rules, canonical
formatting checks, persistent directory-scan caching, and trusted native rule
plugins.

## Build and test

Run these commands from this directory:

```bash
zig build
zig build test
./zig-out/bin/zig-lint --help
```

Inspect a file or directory, or atomically apply compatible rule fixes and
canonical formatting before reporting remaining diagnostics:

```bash
./zig-out/bin/zig-lint src
./zig-out/bin/zig-lint --fix src
```

The package exports the `zig-lint` executable and the `zig_lint` module. A
consumer can call `zig_lint.lint` directly and inspect the returned
`zig_lint.LintReport` without launching the CLI.

Native plugin authors should follow the versioned contract in
[`docs/plugins.md`](docs/plugins.md).
