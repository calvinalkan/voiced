//! Voiced builds static PIE executables through Zig's default LLVM backend.
//!
//! `-Doptimize` selects the application mode and defaults to ReleaseSafe.
//! `-Doptimize-inference-runtime` can override that mode for the complete
//! inference module; without the override, inference follows the application.
//! `-Ddeveloper` defaults to true and controls the artifact's diagnostics:
//!
//! - Developer builds retain symbols and in-process crash diagnostics. Both
//!   optimization settings may use Debug, ReleaseSafe, ReleaseFast, or
//!   ReleaseSmall.
//! - Non-developer builds are a closed deployment profile: both optimization
//!   settings must be non-Debug, the daemon is stripped, in-process crash
//!   diagnostics are omitted, and the installed daemon must stay below 1 MiB.
//! - `size-check` accepts either developer setting, builds the corresponding
//!   deployment-equivalent daemon, and rejects Debug optimization modes.
//!
//! Representative combinations:
//!
//! - `zig build`: developer ReleaseSafe application and inference.
//! - `zig build -Doptimize=Debug -Doptimize-inference-runtime=ReleaseSafe`:
//!   debuggable application with optimized, checked inference.
//! - `zig build -Ddeveloper=false`: ReleaseSafe deployment.
//! - A compact deployment combines `-Ddeveloper=false`,
//!   `-Doptimize=ReleaseSmall`, and
//!   `-Doptimize-inference-runtime=ReleaseFast`.
//!
//! PIE is unconditional. The build exposes no backend, stripping, or crash-
//! diagnostic switches; those are implementation and profile properties.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const developer = b.option(
        bool,
        "developer",
        "Retain symbols and in-process crash diagnostics (default: true)",
    ) orelse true;
    const optimize: std.builtin.OptimizeMode = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseSafe)",
    ) orelse switch (b.release_mode) {
        .off, .any, .safe => .ReleaseSafe,
        .fast => .ReleaseFast,
        .small => .ReleaseSmall,
    };
    const inference_optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize-inference-runtime",
        "Override the inference runtime optimization mode (default: -Doptimize)",
    ) orelse optimize;
    const deployment_modes_are_valid = optimize != .Debug and inference_optimize != .Debug;
    const invalid_deployment = if (!developer and !deployment_modes_are_valid)
        b.addFail("-Ddeveloper=false requires non-Debug application and inference optimization modes")
    else
        null;

    const models_module = b.createModule(.{ .root_source_file = b.path("src/models.zig") });

    const inference = b.createModule(.{
        .root_source_file = b.path("inference/root.zig"),
        .target = b.graph.host,
        .optimize = inference_optimize,
    });

    // ── Voiced ──

    const build_options = b.addOptions();
    build_options.addOption(bool, "developer", developer);

    const voiced = b.addExecutable(.{
        .name = "voiced",
        // Static PIE retains address randomization without an ELF interpreter
        // or shared libraries. Its ELF type is still DYN for self-relocation.
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = optimize,
            // Deployment panics do not walk the stack, so their runtime unwind
            // tables and embedded debugging information have no in-process use.
            .unwind_tables = if (developer) null else .none,
            .strip = !developer,
        }),
    });

    voiced.pie = true;
    if (invalid_deployment) |failure| voiced.step.dependOn(&failure.step);

    voiced.root_module.addOptions("build_options", build_options);
    voiced.root_module.addImport("models", models_module);
    voiced.root_module.addImport("inference", inference);

    b.installArtifact(voiced);

    // ── Size Check ──

    const size_check = b.step("size-check", "Check that deployment voiced stays below 1 MiB");
    if (!deployment_modes_are_valid) {
        const unsupported_profile = b.addFail("size-check requires non-Debug application and inference optimization modes");
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
                .linkage = .static,
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/main.zig"),
                    .target = b.graph.host,
                    .optimize = optimize,
                    .unwind_tables = .none,
                    .strip = true,
                }),
            });
            deployment_voiced.pie = true;
            deployment_voiced.root_module.addOptions("build_options", deployment_build_options);
            deployment_voiced.root_module.addImport("models", models_module);
            deployment_voiced.root_module.addImport("inference", inference);
            break :binary deployment_voiced.getEmittedBin();
        } else voiced.getEmittedBin();

        const binary_size_check = BinarySizeCheck.create(b, size_check_binary, 1024 * 1024, optimize, inference_optimize);
        size_check.dependOn(&binary_size_check.step);
        if (!developer) b.getInstallStep().dependOn(&binary_size_check.step);
    }

    // ── Notification Check ──

    const notification_check = b.addExecutable(.{
        .name = "voiced-notification-check",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/notification_check.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });

    notification_check.pie = true;
    if (invalid_deployment) |failure| notification_check.step.dependOn(&failure.step);

    const install_notification_check = b.addInstallArtifact(notification_check, .{});
    b.step("notification-check", "Build the isolated notification verification driver").dependOn(&install_notification_check.step);

    // ── Clipboard Check ──

    const clipboard_check = b.addExecutable(.{
        .name = "voiced-clipboard-check",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/clipboard_check.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });

    clipboard_check.pie = true;
    if (invalid_deployment) |failure| clipboard_check.step.dependOn(&failure.step);

    const install_clipboard_check = b.addInstallArtifact(clipboard_check, .{});
    b.step("clipboard-check", "Build the native desktop clipboard verifier").dependOn(&install_clipboard_check.step);

    // ── Replay ──

    const replay = b.addExecutable(.{
        .name = "voiced-replay",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/replay.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });

    replay.pie = true;
    if (invalid_deployment) |failure| replay.step.dependOn(&failure.step);

    replay.root_module.addImport("models", models_module);
    replay.root_module.addImport("inference", inference);

    const install_replay = b.addInstallArtifact(replay, .{});
    b.step("replay", "Build the offline failed-transcription decoder").dependOn(&install_replay.step);

    // ── Setup Models ──
    //
    // Declaring setup does not run it or download data in the ordinary build.

    const setup_tool = b.addExecutable(.{
        .name = "setup",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/setup.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });

    setup_tool.root_module.addImport("models", models_module);

    setup_tool.pie = true;

    const run_setup = b.addRunArtifact(setup_tool);
    run_setup.setCwd(b.path(""));

    const setup_models = b.step("setup-models", "Download and verify the named Whisper checkpoints");
    setup_models.dependOn(&run_setup.step);

    b.step("setup", "Download and verify the named Whisper checkpoints").dependOn(setup_models);

    // ── Linter ──

    const linter_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("linter/root.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    linter_tests.pie = true;
    if (invalid_deployment) |failure| linter_tests.step.dependOn(&failure.step);

    const run_linter_tests = b.addRunArtifact(linter_tests);
    b.step("test", "Run tests").dependOn(&run_linter_tests.step);
}

const BinarySizeCheck = struct {
    step: std.Build.Step,
    binary: std.Build.LazyPath,
    limit_bytes: u64,
    application_optimize: std.builtin.OptimizeMode,
    inference_optimize: std.builtin.OptimizeMode,

    fn create(owner: *std.Build, binary: std.Build.LazyPath, limit_bytes: u64, application_optimize: std.builtin.OptimizeMode, inference_optimize: std.builtin.OptimizeMode) *BinarySizeCheck {
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
            return step.fail("voiced binary size: {d:.1} KiB / {d:.1} MiB limit ({d:.1} KiB over) - developer=false, application={s}, inference={s}", .{
                binary_size_kib,
                limit_mib,
                excess_size_kib,
                @tagName(check.application_optimize),
                @tagName(check.inference_optimize),
            });
        }

        const available_size_kib = @as(f64, @floatFromInt(check.limit_bytes - stat.size)) / bytes_per_kibibyte;
        std.debug.print("voiced binary size: {d:.1} KiB / {d:.1} MiB limit ({d:.1} KiB headroom) - developer=false, application={s}, inference={s}\n", .{
            binary_size_kib,
            limit_mib,
            available_size_kib,
            @tagName(check.application_optimize),
            @tagName(check.inference_optimize),
        });
    }
};
