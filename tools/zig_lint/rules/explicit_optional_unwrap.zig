const Rule = @import("../Rule.zig");
const LintContext = Rule.Context;

/// Forbids the use of `.?` to unwrap an optional value
/// and instead requires explicit null handling or unreachable fallback.
pub fn lint(context: *LintContext) Rule.Error!void {
    const ast = context.ast;

    for (ast.nodes.items(.tag), ast.nodes.items(.data)) |tag, data| {
        if (tag != .unwrap_optional) {
            continue;
        }

        try context.report(.{
            .token = data.node_and_token[1],
            .message = "`.?` unwraps an optional without handling `null`",
            .help = "use a capture or `orelse`",
            .note = "use `orelse { unreachable; }` only when `null` is impossible",
        });
    }
}
