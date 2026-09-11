//! Voiced builds static PIE executables, using LLVM by default.
//!
//! `-Doptimize` selects the application mode and defaults to ReleaseSafe.
//! `-Doptimize-inference-runtime` and `-Doptimize-stdlib` can override that
//! mode for the separately compiled inference implementation and standard
//! library; without an override, each follows the application. Zig's separate `--release` option
//! is rejected so these modes have one unambiguous interface.
//! `-Ddeveloper` defaults to true and controls the artifact's diagnostics:
//!
//! - Developer builds retain symbols and in-process crash diagnostics. All
//!   optimization settings may use Debug, ReleaseSafe, ReleaseFast, or
//!   ReleaseSmall.
//! - Non-developer builds are a closed deployment profile: all optimization
//!   settings must be non-Debug, the daemon is stripped, in-process crash
//!   diagnostics are omitted, and the installed daemon must stay below 1 MiB.
//! - `size-check` accepts either developer setting, builds the corresponding
//!   deployment-equivalent daemon, and rejects Debug optimization modes.
//!
//! Representative combinations:
//!
//! - `zig build`: developer ReleaseSafe application, inference, and stdlib.
//! - `zig build -Doptimize=Debug -Doptimize-inference-runtime=ReleaseSafe`:
//!   debuggable application with optimized, checked inference.
//! - A compact deployment combines `-Ddeveloper=false`,
//!   `-Doptimize=ReleaseSmall`, and
//!   `-Doptimize-inference-runtime=ReleaseFast`.
//!
//! The inference implementation is always a separate LLVM object. `-Dllvm=false`
//! selects Zig's development backend only for application code. Thread creation
//! stays in the application: Zig startup initializes the TLS layout there, not
//! in a separately compiled stdlib copy inside an object. PIE and the
//! developer/deployment diagnostic profiles remain unconditional.

const std = @import("std");

pub fn build(b: *std.Build) void {
    if (b.release_mode != .off) {
        std.debug.print(
            "voiced: we don't support --release.\n" ++
                "Use -Doptimize=ReleaseSafe, -Doptimize=ReleaseFast, or -Doptimize=ReleaseSmall.\n",
            .{},
        );
        std.process.exit(2);
    }

    const developer = b.option(
        bool,
        "developer",
        "Retain symbols and in-process crash diagnostics (default: true)",
    ) orelse true;

    const optimize_default: std.builtin.OptimizeMode = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseSafe)",
    ) orelse .ReleaseSafe;

    const optimize_inference = b.option(
        std.builtin.OptimizeMode,
        "optimize-inference-runtime",
        "Override the inference runtime optimization mode (default: -Doptimize)",
    ) orelse optimize_default;

    // PERFORMANCE: Our Zig 0.16 AVX2 transcription probes ran much slower with
    // the self-hosted x86 backend than with LLVM. On an i7-13700HX, Base.en with
    // hello_world.wav, two encoder workers, one decoder worker, and 5-second
    // padding took 178 ms with LLVM versus 6,284 ms with the original self-hosted
    // variant (35x). One variant tuned after assembly inspection took 2,271 ms
    // (13x). These are individual historical runs from 2026-09-09; they are not
    // corpus averages, and the tuned variant is not established as the best.
    //
    // The separate object lets us use the self-hosted backend for fast Debug
    // application builds while LLVM still optimizes inference. A separate Zig
    // module alone cannot do this: backend selection applies to the entire
    // compilation. The object also lets application-only edits reuse compiled
    // inference when its sources, dependencies, and build settings are unchanged.
    // Recheck transcription performance and rebuild timings before removing
    // this boundary.
    const llvm = b.option(bool, "llvm", "Use LLVM for application code (default: true)") orelse true;

    // ReleaseFast and ReleaseSmall disable compiler-generated runtime safety
    // inside stdlib; explicit validation and error handling remain. This is a
    // reasonable deployment policy: Zig's official x86-64 Linux release is
    // built in ReleaseFast, including the compiler's in-process HTTP/TLS
    // package fetcher. Prefer ReleaseSmall here when size is the objective;
    // model downloads are network-bound and gain little from faster TLS.
    //
    // https://github.com/ziglang/zig/blob/master/ci/x86_64-linux-release.sh
    // https://github.com/ziglang/zig/blob/master/src/main.zig
    const optimize_stdlib = b.option(
        std.builtin.OptimizeMode,
        "optimize-stdlib",
        "Override the Zig standard library optimization mode (default: -Doptimize)",
    ) orelse optimize_default;

    const deployment_modes_are_valid = optimize_default != .Debug and
        optimize_inference != .Debug and optimize_stdlib != .Debug;

    const invalid_deployment = if (!developer and !deployment_modes_are_valid)
        b.addFail("-Ddeveloper=false requires non-Debug application, inference, and stdlib optimization modes")
    else
        null;

    // Zig normally compiles stdlib in the root module's mode. Naming it as an
    // explicit module gives deployments the same independent policy already
    // used for inference while preserving one shared stdlib implementation.
    // The entry/CLI root remains application code and follows `-Doptimize`.
    const stdlib_source = b.graph.zig_lib_directory.join(b.allocator, &.{ "std", "std.zig" }) catch @panic("OOM");
    const stdlib = b.createModule(.{
        .root_source_file = .{ .cwd_relative = stdlib_source },
        .target = b.graph.host,
        .optimize = optimize_stdlib,
    });

    // With a ReleaseSmall application/stdlib and ReleaseFast inference, the
    // separate LLVM object gives inference its own ReleaseFast compilation
    // root. This changes LLVM's optimization context: our Zig 0.16 disassembly
    // showed more helper inlining and loop unrolling than direct LLVM inference.
    // On 2026-09-10, the stripped deployment grew by 23.4 KiB (2.9%), almost all
    // machine code; encoder attention alone grew from 3,931 to 7,723 bytes.
    // This is profile-dependent code generation, not a fixed ABI overhead:
    // with application, stdlib, and inference all ReleaseSmall, the separate
    // object added only 280 bytes. Recheck these tradeoffs after compiler or
    // kernel changes rather than assuming isolation always improves codegen.
    const inference_object = inferenceObject(b, optimize_inference, stdlib, developer);

    // ── Voiced ──

    const build_options = b.addOptions();
    build_options.addOption(bool, "developer", developer);

    const voiced = b.addExecutable(.{
        .name = "voiced",
        .use_llvm = llvm,
        // Static PIE retains address randomization without an ELF interpreter
        // or shared libraries. Its ELF type is still DYN for self-relocation.
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = optimize_default,
            // Deployment panics do not walk the stack, so their runtime unwind
            // tables and embedded debugging information have no in-process use.
            .unwind_tables = if (developer) null else .none,
            .strip = !developer,
        }),
    });

    voiced.pie = true;
    if (invalid_deployment) |failure| voiced.step.dependOn(&failure.step);

    voiced.root_module.addOptions("build_options", build_options);
    voiced.root_module.addImport("std", stdlib);
    voiced.root_module.addObject(inference_object);

    b.installArtifact(voiced);

    // ── Size Check ──

    const size_check = b.step("size-check", "Check that deployment voiced stays below 1 MiB");
    if (!deployment_modes_are_valid) {
        const unsupported_profile = b.addFail("size-check requires non-Debug application, inference, and stdlib optimization modes");
        size_check.dependOn(&unsupported_profile.step);
    } else {
        const size_check_binary = if (developer) binary: {
            const deployment_build_options = b.addOptions();
            deployment_build_options.addOption(bool, "developer", false);

            // This target mirrors `voiced` except for the closed deployment
            // profile. Keep its root options and imports synchronized with the
            // installed target so the size gate measures the same program.
            const deployment_voiced = b.addExecutable(.{
                .name = "voiced-size-check",
                .use_llvm = llvm,
                .linkage = .static,
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/main.zig"),
                    .target = b.graph.host,
                    .optimize = optimize_default,
                    .unwind_tables = .none,
                    .strip = true,
                }),
            });
            deployment_voiced.pie = true;
            deployment_voiced.root_module.addOptions("build_options", deployment_build_options);
            deployment_voiced.root_module.addImport("std", stdlib);
            // The LLVM object also owns unwind/debug settings. Rebuild its
            // module graph for deployment so this measures the real profile.
            const deployment_inference_object = inferenceObject(b, optimize_inference, stdlib, false);
            deployment_voiced.root_module.addObject(deployment_inference_object);
            break :binary deployment_voiced.getEmittedBin();
        } else voiced.getEmittedBin();

        const binary_size_check = BinarySizeCheck.create(b, size_check_binary, 1024 * 1024, optimize_default, optimize_inference, optimize_stdlib);
        size_check.dependOn(&binary_size_check.step);
        if (!developer) b.getInstallStep().dependOn(&binary_size_check.step);
    }

    // ── Setup Models ──
    //
    // Declaring setup does not run it or download data in the ordinary build.

    const run_setup = b.addRunArtifact(voiced);
    run_setup.addArg("setup");

    run_setup.addArgs(b.args orelse
        &.{});

    run_setup.setCwd(b.path(""));

    b.step("setup", "Download and verify selected Whisper checkpoints").dependOn(&run_setup.step);

    // ── Linter ──

    const linter_module = b.createModule(.{
        .root_source_file = b.path("zig/linter/root.zig"),
        .target = b.graph.host,
        .optimize = optimize_default,
    });
    linter_module.addImport("std", stdlib);

    const zig_lint = b.addExecutable(.{
        .name = "zig-lint",
        // Native rule plugins use the platform dynamic loader. Keep the daemon
        // static, but give this developer tool a libc-backed `dlopen` host.
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/zig_lint.zig"),
            .target = b.graph.host,
            .optimize = optimize_default,
            .link_libc = true,
        }),
    });

    zig_lint.pie = true;
    if (invalid_deployment) |failure| zig_lint.step.dependOn(&failure.step);
    zig_lint.root_module.addImport("std", stdlib);
    zig_lint.root_module.addImport("linter", linter_module);

    const install_zig_lint = b.addInstallArtifact(zig_lint, .{});
    b.step("zig-lint", "Build the Zig linter CLI").dependOn(&install_zig_lint.step);

    const linter_tests = b.addTest(.{
        .root_module = b.createModule(.{
            // Temporary: the linter is under active development and remains
            // in its original directory until that work is ready to move.
            .root_source_file = b.path("zig/linter/root.zig"),
            .target = b.graph.host,
            .optimize = optimize_default,
        }),
    });
    linter_tests.pie = true;
    if (invalid_deployment) |failure| linter_tests.step.dependOn(&failure.step);
    linter_tests.root_module.addImport("std", stdlib);

    const run_linter_tests = b.addRunArtifact(linter_tests);
    run_linter_tests.setCwd(b.path("."));
    // Fixtures are discovered at runtime, not tracked through @embedFile.
    // Always execute the tests, including after fixture additions or deletions.
    run_linter_tests.has_side_effects = true;

    const tests = b.step("test", "Run tests");
    tests.dependOn(&run_linter_tests.step);
}

// Build the same inference object for the installed program and its size gate.
fn inferenceObject(
    b: *std.Build,
    inference_optimize: std.builtin.OptimizeMode,
    stdlib: *std.Build.Module,
    developer: bool,
) *std.Build.Step.Compile {
    const object_root = b.createModule(.{
        .root_source_file = b.path("src/inference/root.object.zig"),
        .target = b.graph.host,
        .optimize = inference_optimize,
        .pic = true,
        .unwind_tables = if (developer) null else .none,
        .strip = !developer,
    });
    object_root.addImport("std", stdlib);

    return b.addObject(.{
        .name = "voiced-inference",
        .root_module = object_root,
        .use_llvm = true,
    });
}

const BinarySizeCheck = struct {
    step: std.Build.Step,
    binary: std.Build.LazyPath,
    limit_bytes: u64,
    application_optimize: std.builtin.OptimizeMode,
    inference_optimize: std.builtin.OptimizeMode,
    stdlib_optimize: std.builtin.OptimizeMode,

    fn create(
        owner: *std.Build,
        binary: std.Build.LazyPath,
        limit_bytes: u64,
        application_optimize: std.builtin.OptimizeMode,
        inference_optimize: std.builtin.OptimizeMode,
        stdlib_optimize: std.builtin.OptimizeMode,
    ) *BinarySizeCheck {
        const check = owner.allocator.create(BinarySizeCheck) catch @panic("OOM");

        check.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "check voiced deployment binary size",
                .owner = owner,
                .makeFn = make,
            }),
            .binary = binary.dupe(owner),
            .limit_bytes = limit_bytes,
            .application_optimize = application_optimize,
            .inference_optimize = inference_optimize,
            .stdlib_optimize = stdlib_optimize,
        };

        check.binary.addStepDependencies(&check.step);

        return check;
    }

    fn make(step: *std.Build.Step, options: std.Build.Step.MakeOptions) !void {
        _ = options;

        const check: *BinarySizeCheck = @fieldParentPtr("step", step);
        const owner = step.owner;
        const binary_path = check.binary.getPath2(owner, step);

        const stat = std.Io.Dir.cwd().statFile(owner.graph.io, binary_path, .{}) catch |err| {
            return step.fail("could not inspect stripped binary '{s}': {s}", .{ binary_path, @errorName(err) });
        };
        const bytes_per_kibibyte: f64 = 1024;
        const bytes_per_mebibyte: f64 = 1024 * 1024;
        const binary_size_kib = @as(f64, @floatFromInt(stat.size)) / bytes_per_kibibyte;
        const limit_mib = @as(f64, @floatFromInt(check.limit_bytes)) / bytes_per_mebibyte;

        if (stat.size >= check.limit_bytes) {
            const excess_size_kib = @as(f64, @floatFromInt(stat.size - check.limit_bytes)) / bytes_per_kibibyte;
            return step.fail("voiced binary size: {d:.1} KiB / {d:.1} MiB limit ({d:.1} KiB over) - developer=false, application={s}, inference={s}, stdlib={s}", .{
                binary_size_kib,
                limit_mib,
                excess_size_kib,
                @tagName(check.application_optimize),
                @tagName(check.inference_optimize),
                @tagName(check.stdlib_optimize),
            });
        }

        const available_size_kib = @as(f64, @floatFromInt(check.limit_bytes - stat.size)) / bytes_per_kibibyte;

        std.debug.print("voiced binary size: {d:.1} KiB / {d:.1} MiB limit ({d:.1} KiB headroom) - developer=false, application={s}, inference={s}, stdlib={s}\n", .{
            binary_size_kib,
            limit_mib,
            available_size_kib,
            @tagName(check.application_optimize),
            @tagName(check.inference_optimize),
            @tagName(check.stdlib_optimize),
        });
    }
};
