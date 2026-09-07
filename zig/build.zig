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
    // LLVM optimizes SHA-256 verification over the complete checkpoint files;
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
            // Retain debug data for symbolization; strip a separate shipping
            // copy rather than losing it to ReleaseSmall's default stripping.
            .strip = false,
        }),
    });
    // LLVM optimizes the host's AVX-VNNI inference kernels. This executable is
    // host-targeted, not a portable CPU-dispatched baseline.
    voiced.use_llvm = true;
    voiced.pie = pie;
    voiced.root_module.addImport("models", models_module);
    voiced.root_module.addImport("inference", inference);
    b.installArtifact(voiced);

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
