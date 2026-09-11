//! Voiced builds static PIE executables. Zig selects the application backend
//! unless `-Dllvm` overrides it; inference always uses LLVM.
//!
//! `-Doptimize` selects the application mode and defaults to Debug. The stdlib
//! follows the application unless `-Doptimize-stdlib` overrides it. The
//! separately compiled inference implementation defaults independently to
//! ReleaseSafe and accepts `-Doptimize-inference-runtime`. Zig's separate
//! `--release` option is rejected so these modes have one unambiguous interface.
//! `-Ddeveloper` defaults to true and controls the artifact's diagnostics:
//!
//! - Developer builds retain symbols and in-process crash diagnostics. All
//!   optimization settings may use Debug, ReleaseSafe, ReleaseFast, or
//!   ReleaseSmall.
//! - Non-developer builds are a closed deployment profile: all optimization
//!   settings must be non-Debug, the daemon is stripped, in-process crash
//!   diagnostics are omitted, and the installed daemon must stay below 1 MiB.
//! - `test:executable-size` accepts either developer setting, builds the
//!   corresponding deployment-equivalent daemon, and rejects Debug modes.
//!
//! Representative combinations:
//!
//! - `zig build`: developer Debug application and stdlib with ReleaseSafe LLVM
//!   inference; Zig selects the application backend.
//! - `zig build -Doptimize=ReleaseSafe`: optimized, checked application,
//!   inference, and stdlib.
//! - `zig build release` installs the fixed compact deployment profile:
//!   ReleaseSmall application and stdlib, ReleaseFast inference, LLVM, and no
//!   developer diagnostics.
//!
//! The inference implementation is always a separate LLVM object. `-Dllvm`
//! overrides Zig's backend selection only for application code. Thread creation
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
        "Set the application optimization mode (default: Debug)",
    ) orelse .Debug;

    const optimize_inference = b.option(
        std.builtin.OptimizeMode,
        "optimize-inference-runtime",
        "Set the inference runtime optimization mode (default: ReleaseSafe)",
    ) orelse .ReleaseSafe;

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
    const llvm = b.option(bool, "llvm", "Override Zig's application code-generation backend");

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
    const stdlib = addStdlibModule(b, optimize_stdlib);

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
    const inference_object = addInferenceObject(b, optimize_inference, stdlib, developer);

    // ── Voiced ──

    const voiced = addVoicedExecutable(b, .{
        .name = "voiced",
        .llvm = llvm,
        .optimize = optimize_default,
        .stdlib = stdlib,
        .inference_object = inference_object,
        .developer = developer,
    });

    if (invalid_deployment) |failure| {
        voiced.step.dependOn(&failure.step);
    }

    b.installArtifact(voiced);

    // ── Release ──

    const release_application_optimize: std.builtin.OptimizeMode = .ReleaseSmall;
    const release_inference_optimize: std.builtin.OptimizeMode = .ReleaseFast;
    const release_stdlib_optimize: std.builtin.OptimizeMode = .ReleaseSmall;
    const release_stdlib = addStdlibModule(b, release_stdlib_optimize);
    const release_inference_object = addInferenceObject(b, release_inference_optimize, release_stdlib, false);

    const release_executable = addVoicedExecutable(b, .{
        .name = "voiced",
        .llvm = true,
        .optimize = release_application_optimize,
        .stdlib = release_stdlib,
        .inference_object = release_inference_object,
        .developer = false,
    });

    const release_size_check = BinarySizeCheck.create(
        b,
        release_executable.getEmittedBin(),
        1024 * 1024,
        release_application_optimize,
        release_inference_optimize,
        release_stdlib_optimize,
    );

    const install_release = b.addInstallArtifact(release_executable, .{});

    install_release.step.dependOn(&release_size_check.step);

    const release = b.step("release", "Build and install the fixed compact deployment profile");

    const release_override: ?[]const u8 = override: {
        for ([_][]const u8{
            "developer",
            "optimize",
            "optimize-inference-runtime",
            "optimize-stdlib",
            "llvm",
        }) |option_name| {
            if (b.user_input_options.contains(option_name)) {
                break :override b.fmt("-D{s}", .{option_name});
            }
        }

        if (b.args) |args| {
            if (args.len == 0) {
                break :override "--";
            }

            break :override b.fmt("argument after --: '{s}'", .{args[0]});
        }

        break :override null;
    };

    if (release_override) |override| {
        const invalid_release = b.addFail(b.fmt(
            "release has a fixed profile; it does not accept {s}",
            .{override},
        ));

        release.dependOn(&invalid_release.step);
    } else {
        release.dependOn(&install_release.step);
    }

    // ── Executable Size Test ──

    const executable_size_test = b.step("test:executable-size", "Test that the selected executable profile stays below 1 MiB");

    if (!deployment_modes_are_valid) {
        const unsupported_profile = b.addFail("test:executable-size requires non-Debug application, inference, and stdlib optimization modes");

        executable_size_test.dependOn(&unsupported_profile.step);
    } else {
        const executable = if (developer) executable: {
            // Rebuild both sides without developer diagnostics so this target
            // measures the selected optimization profile as it would deploy.
            const deployment_inference_object = addInferenceObject(b, optimize_inference, stdlib, false);

            break :executable addVoicedExecutable(b, .{
                .name = "voiced-executable-size-test",
                .llvm = llvm,
                .optimize = optimize_default,
                .stdlib = stdlib,
                .inference_object = deployment_inference_object,
                .developer = false,
            });
        } else voiced;

        const executable_size_check = BinarySizeCheck.create(b, executable.getEmittedBin(), 1024 * 1024, optimize_default, optimize_inference, optimize_stdlib);

        executable_size_test.dependOn(&executable_size_check.step);

        if (!developer) {
            b.getInstallStep().dependOn(&executable_size_check.step);
        }
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

    // ── ZigLint ──

    const zig_lint_dependency = b.dependency("zig_lint", .{
        .target = b.graph.host,
        .optimize = optimize_default,
    });

    const install_zig_lint = b.addInstallArtifact(zig_lint_dependency.artifact("zig-lint"), .{});

    b.step("zig-lint", "Build the ZigLint CLI").dependOn(&install_zig_lint.step);

    // ── Tests ──

    const application_tests = b.addTest(.{
        .name = "voiced-application-tests",
        .use_llvm = llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root_test.zig"),
            .target = b.graph.host,
            .optimize = optimize_default,
        }),
        .filters = b.args orelse
            &.{},
    });

    application_tests.pie = true;

    application_tests.root_module.addImport("std", stdlib);
    application_tests.root_module.addImport("zig_lint", zig_lint_dependency.module("zig_lint"));

    const run_application_tests = b.addRunArtifact(application_tests);

    run_application_tests.setCwd(b.path("."));

    // Packed-model inputs and repository source are discovered at runtime.
    run_application_tests.has_side_effects = true;

    const inference_tests = b.addTest(.{
        .name = "voiced-inference-tests",
        .use_llvm = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/inference_test.zig"),
            .target = b.graph.host,
            .optimize = optimize_inference,
        }),
        .filters = b.args orelse
            &.{},
    });

    inference_tests.pie = true;

    inference_tests.root_module.addImport("std", stdlib);
    inference_tests.root_module.addObject(inference_object);

    const run_inference_tests = b.addRunArtifact(inference_tests);

    run_inference_tests.setCwd(b.path("."));

    // Models and audio corpus inputs are discovered at runtime.
    run_inference_tests.has_side_effects = true;

    const tests = b.step("test", "Run repository tests");

    tests.dependOn(&run_application_tests.step);
    tests.dependOn(&run_inference_tests.step);
}

fn addStdlibModule(b: *std.Build, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const stdlib_source = b.graph.zig_lib_directory.join(b.allocator, &.{ "std", "std.zig" }) catch
        @panic("OOM");

    return b.createModule(.{
        .root_source_file = .{ .cwd_relative = stdlib_source },
        .target = b.graph.host,
        .optimize = optimize,
    });
}

fn addVoicedExecutable(b: *std.Build, options: struct {
    name: []const u8,
    llvm: ?bool,
    optimize: std.builtin.OptimizeMode,
    stdlib: *std.Build.Module,
    inference_object: *std.Build.Step.Compile,
    developer: bool,
}) *std.Build.Step.Compile {
    const build_options = b.addOptions();

    build_options.addOption(bool, "developer", options.developer);

    const executable = b.addExecutable(.{
        .name = options.name,
        .use_llvm = options.llvm,
        // Static PIE retains address randomization without an ELF interpreter
        // or shared libraries. Its ELF type is still DYN for self-relocation.
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = options.optimize,
            // Deployment panics do not walk the stack, so their runtime unwind
            // tables and embedded debugging information have no in-process use.
            .unwind_tables = if (options.developer) null else .none,
            .strip = !options.developer,
        }),
    });

    executable.pie = true;

    executable.root_module.addOptions("build_options", build_options);
    executable.root_module.addImport("std", options.stdlib);
    executable.root_module.addObject(options.inference_object);

    return executable;
}

// Build the same inference object for the installed program and its size test.
fn addInferenceObject(
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
        const check = owner.allocator.create(BinarySizeCheck) catch
            @panic("OOM");

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
