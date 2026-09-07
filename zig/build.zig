const std = @import("std");

pub fn build(b: *std.Build) void {
    // `zig build` does not forward `-fPIE`; use `-Dpie` (same idea as `-Doptimize`).
    const pie = b.option(bool, "pie", "Build position-independent executables (default: true)") orelse true;
    add_default_build_command(b, pie);

    const setup_tool = b.addExecutable(.{
        .name = "setup",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/setup.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    setup_tool.root_module.addImport("models", b.createModule(.{
        .root_source_file = b.path("src/models.zig"),
    }));
    // LLVM optimizes BLAKE3 verification over the complete checkpoint files;
    // the unoptimized backend makes even an existing installation slow to check.
    // Declaring setup does not run it or download data in the ordinary build.
    setup_tool.use_llvm = true;
    setup_tool.pie = pie;

    const run_setup = b.addRunArtifact(setup_tool);
    run_setup.setCwd(b.path(""));

    const setup_models = b.step("setup-models", "Download and verify the named Whisper checkpoints");
    setup_models.dependOn(&run_setup.step);

    b.step("setup", "Download and verify the named Whisper checkpoints").dependOn(setup_models);
}

fn add_default_build_command(b: *std.Build, pie: bool) void {
    const optimize: std.builtin.OptimizeMode = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseSafe)",
    ) orelse switch (b.release_mode) {
        .off, .any, .safe => .ReleaseSafe,
        .fast => .ReleaseFast,
        .small => .ReleaseSmall,
    };

    const crash_diagnostics = b.option(
        bool,
        "crash-diagnostics",
        "Keep in-process panic/fault stack traces in voiced (default: true)",
    ) orelse true;
    const strip_binary = b.option(
        bool,
        "strip",
        "Strip debug information and symbols from voiced (default: false)",
    ) orelse false;
    const build_options = b.addOptions();
    build_options.addOption(bool, "crash_diagnostics", crash_diagnostics);

    // Keep the inference kernels speed-optimized in the compact application
    // build. Debug and ReleaseSafe still apply to every module.
    const inference_optimize = if (optimize == .ReleaseSmall) .ReleaseFast else optimize;

    const models_module = b.createModule(.{ .root_source_file = b.path("src/models.zig") });

    const inference = b.createModule(.{
        .root_source_file = b.path("runtime/root.zig"),
        .target = b.graph.host,
        .optimize = inference_optimize,
    });

    const voiced = b.addExecutable(.{
        .name = "voiced",
        // Static PIE retains address randomization without an ELF interpreter
        // or shared libraries. Its ELF type is still DYN for self-relocation.
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = optimize,
            // Without in-process stack walking, omit its runtime unwind tables.
            // Debug information and frame-pointer policy remain unchanged.
            .unwind_tables = if (crash_diagnostics) null else .none,
            // Retain debug data by default so its artifact can symbolize cores.
            // A shipping build can instead ask Zig to emit only the stripped ELF.
            .strip = strip_binary,
        }),
    });

    // LLVM optimizes the host's AVX-VNNI inference kernels. This executable is
    // host-targeted, not a portable CPU-dispatched baseline.
    voiced.use_llvm = true;
    voiced.pie = pie;
    voiced.root_module.addOptions("build_options", build_options);
    voiced.root_module.addImport("models", models_module);
    voiced.root_module.addImport("inference", inference);
    b.installArtifact(voiced);

    const size_check = b.step("size-check", "Check that stripped ReleaseSafe voiced stays below 1 MiB");
    if (optimize != .ReleaseSafe or crash_diagnostics or !pie or !strip_binary) {
        const unsupported_profile = b.addFail("size-check requires -Doptimize=ReleaseSafe -Dcrash-diagnostics=false -Dstrip=true and PIE");
        size_check.dependOn(&unsupported_profile.step);
    } else {
        const binary_size_check = BinarySizeCheck.create(b, voiced.getEmittedBin(), 1024 * 1024);
        size_check.dependOn(&binary_size_check.step);
    }

    const notification_check = b.addExecutable(.{
        .name = "voiced-notification-check",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/notification_check.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });

    const install_notification_check = b.addInstallArtifact(notification_check, .{});
    b.step("notification-check", "Build the isolated notification verification driver").dependOn(&install_notification_check.step);

    const clipboard_check = b.addExecutable(.{
        .name = "voiced-clipboard-check",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/clipboard_check.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    clipboard_check.pie = pie;
    const install_clipboard_check = b.addInstallArtifact(clipboard_check, .{});
    b.step("clipboard-check", "Build the native desktop clipboard verifier").dependOn(&install_clipboard_check.step);

    const replay = b.addExecutable(.{
        .name = "voiced-replay",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/replay.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
        .use_llvm = true,
    });
    replay.pie = pie;
    replay.root_module.addImport("models", models_module);
    replay.root_module.addImport("inference", inference);
    const install_replay = b.addInstallArtifact(replay, .{});
    b.step("replay", "Build the offline failed-transcription decoder").dependOn(&install_replay.step);
}

const BinarySizeCheck = struct {
    step: std.Build.Step,
    binary: std.Build.LazyPath,
    limit_bytes: u64,

    fn create(owner: *std.Build, binary: std.Build.LazyPath, limit_bytes: u64) *BinarySizeCheck {
        const check = owner.allocator.create(BinarySizeCheck) catch @panic("OOM");
        check.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = owner.fmt("check {s} size", .{binary.getDisplayName()}),
                .owner = owner,
                .makeFn = make,
            }),
            .binary = binary.dupe(owner),
            .limit_bytes = limit_bytes,
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
        if (stat.size >= check.limit_bytes) {
            return step.fail("stripped voiced binary is {d} bytes; it must remain below {d} bytes", .{ stat.size, check.limit_bytes });
        }
    }
};
