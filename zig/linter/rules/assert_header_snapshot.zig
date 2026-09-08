const std = @import("std");
const Ast = std.zig.Ast;
const Rule = @import("../Rule.zig");
const LintContext = Rule.Context;
const identifierText = LintContext.identifierText;

/// Snapshot `std.debug.assert` in the file header as `const assert = std.debug.assert;`
/// and call `assert`. Qualified spellings and file-root aliases of that function are hits.
pub fn lint(context: *LintContext) Rule.Error!void {
    const file_allocator = context.scratch_allocator;
    const ast = context.ast;

    // ── Header Snapshot ──
    //
    // The header is the leading private root `const`s. It ends at the first
    // `pub`, `var`, function, test, comptime, or type body:
    //
    //   const std = @import("std");
    //   const assert = std.debug.assert;
    //
    //   pub fn start() void {}
    //   const assert = std.debug.assert;  <- not in the header
    var allowed_init: ?Ast.Node.Index = null;

    for (ast.rootDecls()) |decl| {
        const variable = ast.fullVarDecl(decl) orelse {
            break;
        };

        if (variable.visib_token != null or
            variable.extern_export_token != null or
            variable.threadlocal_token != null or
            variable.comptime_token != null or
            ast.tokenTag(variable.ast.mut_token) != .keyword_const)
        {
            break;
        }

        const init = variable.ast.init_node.unwrap() orelse {
            break;
        };

        if (LintContext.nodeIsDirectType(ast, init) or ast.nodeTag(init) == .fn_decl) {
            break;
        }

        if (variable.ast.type_node != .none or
            ast.tokenTag(variable.ast.mut_token + 1) != .identifier or
            !std.mem.eql(u8, identifierText(ast, variable.ast.mut_token + 1), "assert") or
            ast.nodeTag(init) != .field_access)
        {
            continue;
        }

        const debug_access, const assert_token = ast.nodeData(init).node_and_token;

        if (!std.mem.eql(u8, identifierText(ast, assert_token), "assert") or
            ast.nodeTag(debug_access) != .field_access)
        {
            continue;
        }

        const std_ident, const debug_token = ast.nodeData(debug_access).node_and_token;

        if (!std.mem.eql(u8, identifierText(ast, debug_token), "debug") or
            ast.nodeTag(std_ident) != .identifier or
            !std.mem.eql(u8, identifierText(ast, ast.nodeMainToken(std_ident)), "std"))
        {
            continue;
        }

        allowed_init = init;

        break;
    }

    // ── File-Root Aliases ──
    //
    // Index container-level `const`s once and resolve each path at most once:
    //
    //   const debug = std.debug;
    //   debug.assert(true);
    //
    // A function-local `const debug = std.debug` is not followed.
    var aliases: std.ArrayList(Alias) = .empty;
    defer aliases.deinit(file_allocator);

    for (ast.rootDecls()) |decl| {
        const variable = ast.fullVarDecl(decl) orelse {
            continue;
        };

        if (ast.tokenTag(variable.ast.mut_token) != .keyword_const or
            ast.tokenTag(variable.ast.mut_token + 1) != .identifier)
        {
            continue;
        }

        const init = variable.ast.init_node.unwrap() orelse {
            continue;
        };

        try aliases.append(file_allocator, .{
            .name = identifierText(ast, variable.ast.mut_token + 1),
            .init = init,
        });
    }

    // ── Qualified Uses ──

    for (ast.nodes.items(.tag), 0..) |tag, index_usize| {
        if (tag != .field_access) {
            continue;
        }

        const node: Ast.Node.Index = @enumFromInt(index_usize);
        const assert_token = ast.nodeData(node).node_and_token[1];

        if (!std.mem.eql(u8, identifierText(ast, assert_token), "assert")) {
            continue;
        }

        if (allowed_init == node) {
            continue;
        }

        if (pathOf(ast, node, aliases.items) != .std_debug_assert) {
            continue;
        }

        try context.report(.{
            .token = assert_token,
            .message = "assertions must use the file-header `assert` snapshot",
            .help = "define `const assert = std.debug.assert;` in the file header and call `assert`",
        });
    }
}

const Path = enum {
    none,
    std,
    std_debug,
    std_debug_assert,
};

const Alias = struct {
    name: []const u8,
    init: Ast.Node.Index,
    path: Path = .none,
    state: enum { idle, visiting, done } = .idle,
};

fn pathOf(ast: Ast, node: Ast.Node.Index, aliases: []Alias) Path {
    switch (ast.nodeTag(node)) {
        .grouped_expression => return pathOf(ast, ast.nodeData(node).node_and_token[0], aliases),

        .identifier => {
            const name = identifierText(ast, ast.nodeMainToken(node));
            if (std.mem.eql(u8, name, "std")) {
                return .std;
            }

            for (aliases) |*alias| {
                if (!std.mem.eql(u8, alias.name, name)) {
                    continue;
                }

                switch (alias.state) {
                    .done => return alias.path,
                    .visiting => return .none,

                    .idle => {
                        alias.state = .visiting;
                        alias.path = pathOf(ast, alias.init, aliases);
                        alias.state = .done;

                        return alias.path;
                    },
                }
            }

            return .none;
        },

        .field_access => {
            const lhs, const field_token = ast.nodeData(node).node_and_token;
            const field = identifierText(ast, field_token);

            return switch (pathOf(ast, lhs, aliases)) {
                .std => if (std.mem.eql(u8, field, "debug")) .std_debug else .none,
                .std_debug => if (std.mem.eql(u8, field, "assert")) .std_debug_assert else .none,
                .none, .std_debug_assert => .none,
            };
        },

        .builtin_call_two,
        .builtin_call_two_comma,
        .builtin_call,
        .builtin_call_comma,
        => {
            if (!std.mem.eql(u8, ast.tokenSlice(ast.nodeMainToken(node)), "@import")) {
                return .none;
            }

            var buffer: [2]Ast.Node.Index = undefined;
            const params = ast.builtinCallParams(&buffer, node) orelse {
                return .none;
            };

            if (params.len != 1 or ast.nodeTag(params[0]) != .string_literal) {
                return .none;
            }

            if (!std.mem.eql(u8, ast.tokenSlice(ast.nodeMainToken(params[0])), "\"std\"")) {
                return .none;
            }

            return .std;
        },

        else => return .none,
    }
}
