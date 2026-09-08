const std = @import("std");
const Diagnostics = @import("../Diagnostics.zig");
const FileContext = @import("../FileContext.zig");

/// Forbids the use of `.?` to unwrap an optional value
/// and instead requires explicit null handling or unreachable fallback.
pub fn lint(
    _: std.mem.Allocator,
    gpa: std.mem.Allocator,
    path: []const u8,
    file: *const FileContext,
    diagnostics: *Diagnostics,
) std.mem.Allocator.Error!void {
    const ast = file.ast;

    for (ast.nodes.items(.tag), ast.nodes.items(.data)) |tag, data| {
        if (tag != .unwrap_optional) {
            continue;
        }

        try diagnostics.add_at_token(
            gpa,
            path,
            file,
            data.node_and_token[1],
            "explicit_optional_unwrap",
            "`.?` hides an unchecked optional unwrap. Handle null with capture or `orelse`; when null is impossible, use an explicit braced `orelse { unreachable; }` fallback.",
        );
    }
}
