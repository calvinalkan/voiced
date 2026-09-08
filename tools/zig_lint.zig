//! Minimal command-line driver for the native Zig linter. It accepts one file
//! or directory plus explicitly selected trusted plugins, writes agent-formatted
//! diagnostics to stdout, and exits with status 1 when the report contains
//! diagnostics. Directory scans apply `.gitignore` files from the nearest Git
//! boundary through the scan root and included descendants, plus the default
//! exact directory exclusions.

const std = @import("std");
const linter = @import("linter");

pub fn main(init: std.process.Init) u8 {
    const arguments = init.minimal.args.toSlice(init.gpa) catch |err| {
        std.debug.print("zig-lint: could not read arguments ({s}).\n", .{@errorName(err)});

        return 1;
    };
    defer init.gpa.free(arguments);

    if (arguments.len == 2 and
        (std.mem.eql(u8, arguments[1], "-h") or std.mem.eql(u8, arguments[1], "--help")))
    {
        std.Io.File.stdout().writeStreamingAll(init.io, usage) catch |err| {
            std.debug.print("zig-lint: could not write help ({s}).\n", .{@errorName(err)});

            return 1;
        };

        return 0;
    }

    var plugin_handles: std.ArrayList(linter.Plugin.Handle) = .empty;
    defer plugin_handles.deinit(init.gpa);
    defer {
        for (plugin_handles.items) |handle| {
            _ = linter.Plugin.unregister(init.io, handle) catch false;
        }
    }

    var path: ?[]const u8 = null;
    var argument_index: usize = 1;

    while (argument_index < arguments.len) : (argument_index += 1) {
        const argument = arguments[argument_index];

        if (std.mem.eql(u8, argument, "--plugin")) {
            argument_index += 1;
            if (argument_index == arguments.len) {
                std.debug.print("zig-lint: --plugin requires a shared-library path.\n{s}", .{usage});

                return 2;
            }

            const plugin_path = arguments[argument_index];
            const handle = linter.Plugin.register(init.io, plugin_path) catch |err| {
                std.debug.print("zig-lint: could not load plugin '{s}' ({s}).\n", .{ plugin_path, @errorName(err) });

                return 1;
            };

            plugin_handles.append(init.gpa, handle) catch {
                _ = linter.Plugin.unregister(init.io, handle) catch false;
                std.debug.print("zig-lint: could not retain plugin selection (OutOfMemory).\n", .{});

                return 1;
            };

            continue;
        }

        if (argument.len != 0 and argument[0] == '-' or path != null) {
            std.debug.print("{s}", .{usage});

            return 2;
        }

        path = argument;
    }

    const input_path = path orelse {
        std.debug.print("{s}", .{usage});

        return 2;
    };

    var report = linter.lint(
        linter.Allocators.same(init.gpa),
        init.io,
        input_path,
        .{ .plugins = plugin_handles.items },
    ) catch |err| {
        std.debug.print("zig-lint: could not lint '{s}' ({s}).\n", .{ input_path, @errorName(err) });

        return 1;
    };
    defer report.deinit(init.gpa);

    const rendered = report.renderAgent(init.gpa) catch |err| {
        std.debug.print("zig-lint: could not render diagnostics ({s}).\n", .{@errorName(err)});

        return 1;
    };
    defer init.gpa.free(rendered);

    std.Io.File.stdout().writeStreamingAll(init.io, rendered) catch |err| {
        std.debug.print("zig-lint: could not write diagnostics ({s}).\n", .{@errorName(err)});

        return 1;
    };

    return if (report.diagnostics.len == 0) 0 else 1;
}

const usage =
    \\Usage: zig-lint [--plugin <shared-library>]... <file-or-directory>
    \\
    \\Loads only explicitly named trusted plugins; --plugin may be repeated.
    \\Writes agent-formatted diagnostics to stdout.
    \\Directory scans apply .gitignore files from the nearest Git root downward.
    \\The following descendant directories are excluded by default at every depth:
    \\  .git, .hg, .svn, .jj, .zig-cache, zig-cache, zig-out, zig-pkg,
    \\  .cache, node_modules, .venv, __fixtures__
++ "\n";
