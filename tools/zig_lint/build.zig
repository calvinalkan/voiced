const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseSafe)",
    ) orelse
        .ReleaseSafe;

    _ = b.addModule("zig_lint", .{
        .root_source_file = b.path("root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const executable = b.addExecutable(.{
        .name = "zig-lint",
        // Native rule plugins use the platform dynamic loader.
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true, // Need libc for dlopen
        }),
    });

    executable.pie = true;

    b.installArtifact(executable);

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("root.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .filters = b.args orelse
            &.{},
    });

    tests.pie = true;

    const run_tests = b.addRunArtifact(tests);

    run_tests.setCwd(b.path("."));

    // Fixtures are discovered at runtime rather than tracked build inputs.
    run_tests.has_side_effects = true;

    b.step("test", "Run ZigLint tests").dependOn(&run_tests.step);
}
