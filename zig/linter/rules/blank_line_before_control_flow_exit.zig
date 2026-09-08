const std = @import("std");
const Ast = std.zig.Ast;
const Diagnostics = @import("../Diagnostics.zig");
const FileContext = @import("../FileContext.zig");

/// A direct `return` / `break` / `continue` / `unreachable` statement starts
/// a new paragraph when another statement precedes it in the same block.
pub fn lint(
    _: std.mem.Allocator,
    diagnostic_allocator: std.mem.Allocator,
    path: []const u8,
    file: *const FileContext,
    diagnostics: *Diagnostics,
) std.mem.Allocator.Error!void {
    const ast = file.ast;

    // ── Check Block Siblings ──
    //
    // Starting at index one is the structural proof that an earlier statement
    // exists in this exact block. Physical lines alone cannot provide it:
    //
    //   {
    //       return;  <- first statement; the opening brace is not earlier work
    //       work();
    //   }
    //
    // Direct control-flow bodies, fallback expressions, and switch arms are
    // not block siblings, so they naturally need no separate classification.
    var block_buffer: [2]Ast.Node.Index = undefined;

    for (ast.nodes.items(.tag), 0..) |_, index_usize| {
        const block: Ast.Node.Index = @enumFromInt(index_usize);
        const statements = ast.blockStatements(&block_buffer, block) orelse {
            continue;
        };
        if (statements.len < 2) {
            continue;
        }

        for (statements[1..], 1..) |statement, statement_index| {
            const tag = ast.nodeTag(statement);
            const is_exit = switch (tag) {
                .@"return", .@"break", .@"continue", .unreachable_literal => true,
                else => false,
            };
            if (!is_exit) {
                continue;
            }

            const previous_statement = statements[statement_index - 1];
            if (file.hasEmptyLineBetween(ast.lastToken(previous_statement), ast.firstToken(statement))) {
                // Comments attached to the exit may occupy the intervening
                // lines; an actual empty line must separate the two siblings:
                //
                //   work();
                //
                //   // Explain the exit.
                //   return;
                continue;
            }

            try diagnostics.add_at_token(
                diagnostic_allocator,
                path,
                file,
                ast.nodeMainToken(statement),
                "blank_line_before_control_flow_exit",
                messageFor(tag),
            );
        }
    }
}

fn messageFor(tag: Ast.Node.Tag) []const u8 {
    return switch (tag) {
        .@"return" => "`return` follows another statement in the same block without an empty line. Insert an empty line before `return`; if comments describe this exit, put the empty line before those comments. No empty line is required when no statement precedes `return` in its block.",
        .@"break" => "`break` follows another statement in the same block without an empty line. Insert an empty line before `break`; if comments describe this exit, put the empty line before those comments. No empty line is required when no statement precedes `break` in its block.",
        .@"continue" => "`continue` follows another statement in the same block without an empty line. Insert an empty line before `continue`; if comments describe this exit, put the empty line before those comments. No empty line is required when no statement precedes `continue` in its block.",
        .unreachable_literal => "`unreachable` follows another statement in the same block without an empty line. Insert an empty line before `unreachable`; if comments describe this exit, put the empty line before those comments. No empty line is required when no statement precedes `unreachable` in its block.",
        else => unreachable,
    };
}
