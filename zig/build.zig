const std = @import("std");

pub fn build(b: *std.Build) void {
    // `zig build` does not forward `-fPIE`; use `-Dpie` (same idea as `-Doptimize`).
    const pie = b.option(bool, "pie", "Build position-independent executables (default: true)") orelse true;
    add_default_build_command(b, pie);
    add_update_compile_flags_command(b);

    const setup_tool = b.addExecutable(.{
        .name = "setup",
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
    configurePipeWireArtifact(b, voiced);
    voiced.root_module.linkSystemLibrary("systemd", .{ .use_pkg_config = .no });
    b.installArtifact(voiced);

    const replay = b.addExecutable(.{
        .name = "voiced-replay",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/replay.zig"),
            .target = b.graph.host,
            .optimize = optimize,
            .link_libc = true,
        }),
        .use_llvm = true,
    });
    replay.pie = pie;
    replay.root_module.addImport("models", models_module);
    replay.root_module.addImport("inference", inference);
    const install_replay = b.addInstallArtifact(replay, .{});
    b.step("replay", "Build the offline failed-transcription decoder").dependOn(&install_replay.step);
}

const c_flags = [_][]const u8{
    "-std=gnu17",
    "-Weverything",
    "-Werror",
    "-pedantic-errors",
};

fn configurePipeWireArtifact(b: *std.Build, artifact: *std.Build.Step.Compile) void {
    // Explicit roots serve ZLS, translate-c, and compilation consistently;
    // no pkg-config discovery or separate system C compiler is required.
    artifact.root_module.addIncludePath(b.path("src"));
    artifact.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/pipewire-0.3" });
    artifact.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/spa-0.2" });
    artifact.root_module.addCMacro("_REENTRANT", "1");
    artifact.root_module.addCSourceFile(.{
        .file = b.path("src/audio_pipewire.c"),
        .flags = &c_flags,
    });
    artifact.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
    artifact.root_module.linkSystemLibrary("pipewire-0.3", .{ .use_pkg_config = .no });
    artifact.root_module.link_libc = true;
}

fn add_update_compile_flags_command(b: *std.Build) void {
    const flags = std.mem.join(b.allocator, "\n", &c_flags) catch @panic("OOM");
    const contents = b.fmt(
        "{s}\n" ++
            "-Isrc\n" ++
            "-isystem\n/usr/include/pipewire-0.3\n" ++
            "-isystem\n/usr/include/spa-0.2\n" ++
            "-D_REENTRANT=1\n",
        .{flags},
    );
    const generated = b.addWriteFiles().add("compile_flags.txt", contents);
    const update = b.addUpdateSourceFiles();
    update.addCopyFileToSource(generated, "compile_flags.txt");
    b.step("update-compile-flags", "Update clangd flags for the PipeWire C boundary").dependOn(&update.step);
}
