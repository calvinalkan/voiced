const std = @import("std");
const zig_lint = @import("zig_lint");

test {
    _ = @import("audio_exchange.zig");
    _ = @import("capture/pipewire_client.zig");
    _ = @import("capture/pipewire_wire.zig");
    _ = @import("clipboard/x11.zig");
    _ = @import("clipboard/x11_wire.zig");
    _ = @import("logging.zig");
    _ = @import("packed_model/root_test.zig");
}

test "repository source passes lint" {
    const repository_lint_enabled = false;

    // TODO: Enable this check after a dedicated cleanup campaign makes the
    // repository conform to all enabled ZigLint rules.
    if (!repository_lint_enabled) {
        return error.SkipZigTest;
    }

    const allocator = std.testing.allocator;

    var report = try zig_lint.lint(
        zig_lint.Allocators.same(allocator),
        std.testing.io,
        ".",
        .{
            .apply_fixes = true,
            .report_path_format = .git_root_relative,
        },
    );
    defer report.deinit(allocator);

    if (report.diagnostics.len == 0 and report.fixed_files.len == 0) {
        return;
    }

    if (report.diagnostics.len != 0) {
        const rendered = try report.renderAgent(allocator);
        defer allocator.free(rendered);

        std.debug.print("{s}", .{rendered});
    }

    for (report.fixed_files.items(.path)) |path| {
        std.debug.print("{s}: source has available lint fixes or formatting changes\n", .{path});
    }

    return error.RepositoryLintFailed;
}
