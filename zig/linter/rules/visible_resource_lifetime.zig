const std = @import("std");
const Ast = std.zig.Ast;
const Diagnostics = @import("../Diagnostics.zig");
const FileContext = @import("../FileContext.zig");

/// Keep setup and consecutive `defer` / `errdefer` registrations together,
/// with empty lines separating them from other statements. Block boundaries
/// provide separation. This checks layout, not ownership.
pub fn lint(
    _: std.mem.Allocator,
    diagnostic_allocator: std.mem.Allocator,
    path: []const u8,
    file: *const FileContext,
    diagnostics: *Diagnostics,
) std.mem.Allocator.Error!void {
    const ast = file.ast;

    for (ast.nodes.items(.tag), 0..) |_, index_usize| {
        const node: Ast.Node.Index = @enumFromInt(index_usize);
        var block_buffer: [2]Ast.Node.Index = undefined;
        const statements = ast.blockStatements(&block_buffer, node) orelse {
            continue;
        };

        var index: usize = 0;

        while (index < statements.len) {
            if (!isCleanup(ast.nodeTag(statements[index]))) {
                index += 1;

                continue;
            }

            // A group contains one setup statement and all following registrations:
            //
            //   acquire();          <- `start`
            //   defer release();    <- `first_cleanup`
            //   errdefer rollback();
            //
            //   work();             <- `index` after the scan
            //
            // At block start there may be no setup statement at all.
            const first_cleanup = index;
            var start = index -| 1;
            if (start > 0 and initializesUndefinedVariable(ast, statements[start - 1], statements[start])) {
                start -= 1;
            }

            while (index < statements.len and isCleanup(ast.nodeTag(statements[index]))) {
                index += 1;
            }

            // A preceding cleanup group already checks its trailing gap. Do
            // not report that same missing separator again for the next pair:
            //
            //   acquireA();
            //   defer releaseA();
            //   acquireB();        <- reported by the first group only
            //   defer releaseB();
            if (start > 0 and !isCleanup(ast.nodeTag(statements[start - 1])) and
                !hasEmptyLineBetween(file, statements[start - 1], statements[start]))
            {
                try diagnostics.add_at_token(
                    diagnostic_allocator,
                    path,
                    file,
                    ast.firstToken(statements[start]),
                    "visible_resource_lifetime",
                    "Resource setup follows other work without an empty line. Insert an empty line before the setup and its cleanup registrations.",
                );
            }

            for (start + 1..index) |member| {
                if (hasEmptyLineBetween(file, statements[member - 1], statements[member])) {
                    try diagnostics.add_at_token(
                        diagnostic_allocator,
                        path,
                        file,
                        ast.firstToken(statements[member]),
                        "visible_resource_lifetime",
                        "An empty line separates resource setup and cleanup registrations. Keep them together; comments may remain between them.",
                    );
                }
            }

            for (statements[first_cleanup..index]) |cleanup| {
                const token = ast.nodeMainToken(cleanup);
                if (token > 0 and file.tokensOnSameLine(token - 1, token)) {
                    try diagnostics.add_at_token(
                        diagnostic_allocator,
                        path,
                        file,
                        token,
                        "visible_resource_lifetime",
                        "Cleanup registration must start on its own line. Put `defer` or `errdefer` below the setup statement.",
                    );
                }
            }

            if (index < statements.len) {
                const next_statement = statements[index];
                const next_is_exit = switch (ast.nodeTag(next_statement)) {
                    .@"return", .@"break", .@"continue", .unreachable_literal => true,
                    else => false,
                };

                // The dedicated exit-paragraph rule prescribes this same blank
                // line. Let it own direct exits rather than emitting duplicate
                // diagnostics at one token for one edit.
                if (!next_is_exit and !hasEmptyLineBetween(file, statements[index - 1], next_statement)) {
                    try diagnostics.add_at_token(
                        diagnostic_allocator,
                        path,
                        file,
                        ast.firstToken(next_statement),
                        "visible_resource_lifetime",
                        "Other work follows cleanup registrations without an empty line. Insert an empty line after the cleanup group.",
                    );
                }
            }
        }
    }
}

fn isCleanup(tag: Ast.Node.Tag) bool {
    return tag == .@"defer" or tag == .@"errdefer";
}

fn initializesUndefinedVariable(ast: Ast, declaration: Ast.Node.Index, initialization: Ast.Node.Index) bool {
    // Only this two-statement setup extends the ordinary one-statement pair:
    //
    //   var resource: Resource = undefined;  <- `declaration`
    //   try resource.init();                <- `initialization`; `try` is optional
    //   defer resource.deinit();
    //
    // Matching the receiver prevents unrelated calls from absorbing a variable
    // declaration. No inference about what `init` or the cleanup owns is made.
    const variable = ast.fullVarDecl(declaration) orelse {
        return false;
    };

    if (ast.tokenTag(variable.ast.mut_token) != .keyword_var) {
        return false;
    }

    const initializer = variable.ast.init_node.unwrap() orelse {
        return false;
    };

    if (ast.nodeTag(initializer) != .identifier or
        !std.mem.eql(u8, ast.tokenSlice(ast.nodeMainToken(initializer)), "undefined"))
    {
        return false;
    }

    const call_node = if (ast.nodeTag(initialization) == .@"try")
        ast.nodeData(initialization).node
    else
        initialization;

    var call_buffer: [1]Ast.Node.Index = undefined;
    const call = ast.fullCall(&call_buffer, call_node) orelse {
        return false;
    };

    if (ast.nodeTag(call.ast.fn_expr) != .field_access) {
        return false;
    }

    const field = ast.nodeData(call.ast.fn_expr).node_and_token;
    if (ast.nodeTag(field[0]) != .identifier or !std.mem.eql(u8, ast.tokenSlice(field[1]), "init")) {
        return false;
    }

    return std.mem.eql(
        u8,
        ast.tokenSlice(variable.ast.mut_token + 1),
        ast.tokenSlice(ast.nodeMainToken(field[0])),
    );
}

fn hasEmptyLineBetween(file: *const FileContext, left: Ast.Node.Index, right: Ast.Node.Index) bool {
    // Inspect complete physical lines, not just the distance between tokens:
    //
    //   defer release();  <- last line of `left`
    //   // Next phase.    <- a comment is not an empty separator
    //   work();           <- first line of `right`
    //
    // Starting after the whole left statement also excludes blank lines inside
    // multiline cleanup bodies. Semicolons and trailing comments stay on that
    // statement's last line and cannot become separators.
    return file.hasEmptyLineBetween(file.ast.lastToken(left), file.ast.firstToken(right));
}
