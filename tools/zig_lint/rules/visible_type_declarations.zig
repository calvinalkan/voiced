const std = @import("std");
const Ast = std.zig.Ast;
const Rule = @import("../Rule.zig");
const LintContext = Rule.Context;

/// Give every directly declared type its own paragraph at file and container
/// scope. Fields within a type remain grouped as one aggregate declaration.
pub fn lint(context: *LintContext) Rule.Error!void {
    try lintContainer(context, .root);
}

fn lintContainer(context: *LintContext, node: Ast.Node.Index) Rule.Error!void {
    const ast = context.ast;

    var container_buffer: [2]Ast.Node.Index = undefined;
    const container = ast.fullContainerDecl(&container_buffer, node) orelse {
        return;
    };

    var previous_member: ?Ast.Node.Index = null;
    var previous_is_type = false;

    for (container.ast.members) |member| {
        const initializer = typeInitializer(ast, member);
        const member_is_type = initializer != null;

        if (previous_member) |previous| {
            if ((previous_is_type or member_is_type) and
                !context.hasEmptyLineBetween(ast.lastToken(previous), ast.firstToken(member)))
            {
                const message = if (member_is_type)
                    "type declaration must start a new paragraph"
                else
                    "type declaration must end its paragraph";

                const help = if (member_is_type)
                    "insert a blank line before this type declaration"
                else
                    "insert a blank line after the preceding type declaration";

                try context.report(.{
                    .token = ast.firstToken(member),
                    .message = message,
                    .help = help,
                });
            }
        }

        if (initializer) |type_node| {
            try lintContainer(context, type_node);
        }

        previous_member = member;
        previous_is_type = member_is_type;
    }
}

fn typeInitializer(ast: Ast, node: Ast.Node.Index) ?Ast.Node.Index {
    const variable = ast.fullVarDecl(node) orelse {
        return null;
    };

    const initializer = variable.ast.init_node.unwrap() orelse {
        return null;
    };

    return if (LintContext.nodeIsDirectType(ast, initializer)) initializer else null;
}
