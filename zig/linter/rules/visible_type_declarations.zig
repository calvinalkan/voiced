const std = @import("std");
const Ast = std.zig.Ast;
const Diagnostics = @import("../Diagnostics.zig");
const FileContext = @import("../FileContext.zig");

/// Give every directly declared type its own paragraph at file and container
/// scope. Fields within a type remain grouped as one aggregate declaration.
pub fn lint(
    _: std.mem.Allocator,
    report: *Diagnostics.Report,
) std.mem.Allocator.Error!void {
    try lintContainer(report, .root);
}

const rule_name = "visible_type_declarations";

fn lintContainer(report: *Diagnostics.Report, node: Ast.Node.Index) std.mem.Allocator.Error!void {
    const file = report.file;
    const ast = file.ast;

    var container_buffer: [2]Ast.Node.Index = undefined;
    const container = ast.fullContainerDecl(&container_buffer, node) orelse {
        return;
    };

    const members = container.ast.members;

    if (members.len >= 2) {
        for (members[1..], 1..) |right, index| {
            const left = members[index - 1];
            const left_is_type = typeInitializer(ast, left) != null;
            const right_is_type = typeInitializer(ast, right) != null;

            if ((!left_is_type and !right_is_type) or
                file.hasEmptyLineBetween(ast.lastToken(left), ast.firstToken(right)))
            {
                continue;
            }

            const description: Diagnostics.Description = if (right_is_type)
                .{
                    .message = "type declaration must start a new paragraph",
                    .help = "insert a blank line before this type declaration",
                }
            else
                .{
                    .message = "type declaration must end its paragraph",
                    .help = "insert a blank line after the preceding type declaration",
                };

            try report.add(.{
                .token = ast.firstToken(right),
                .rule_name = rule_name,
                .message = description.message,
                .help = description.help,
            });
        }
    }

    for (members) |member| {
        const initializer = typeInitializer(ast, member) orelse {
            continue;
        };

        var child_buffer: [2]Ast.Node.Index = undefined;
        if (ast.fullContainerDecl(&child_buffer, initializer) != null) {
            try lintContainer(report, initializer);
        }
    }
}

fn typeInitializer(ast: Ast, node: Ast.Node.Index) ?Ast.Node.Index {
    const variable = ast.fullVarDecl(node) orelse {
        return null;
    };

    const initializer = variable.ast.init_node.unwrap() orelse {
        return null;
    };

    return if (isDirectType(ast, initializer)) initializer else null;
}

fn isDirectType(ast: Ast, node: Ast.Node.Index) bool {
    var container_buffer: [2]Ast.Node.Index = undefined;
    if (ast.fullContainerDecl(&container_buffer, node) != null) {
        return true;
    }

    return switch (ast.nodeTag(node)) {
        .error_set_decl,
        .fn_proto_simple,
        .fn_proto_multi,
        .fn_proto_one,
        .fn_proto,
        => true,
        else => false,
    };
}
