//! Minimal command-line driver for the native Zig linter. It accepts one file
//! or directory plus explicitly selected trusted plugins, optionally writes
//! compatible fixes atomically, emits agent-formatted diagnostics to stdout,
//! and exits with status 1 when diagnostics remain. Directory scans apply
//! `.gitignore` files from the nearest Git boundary through the scan root and
//! included descendants, plus the default
//! exact directory exclusions.

const std = @import("std");
const zig_lint = @import("root.zig");

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

    var plugin_handles: std.ArrayList(zig_lint.Plugin.Handle) = .empty;
    defer plugin_handles.deinit(init.gpa);
    defer {
        for (plugin_handles.items) |handle| {
            _ = zig_lint.Plugin.unregister(init.io, handle) catch false;
        }
    }

    var path: ?[]const u8 = null;
    var apply_fixes = false;
    var argument_index: usize = 1;

    while (argument_index < arguments.len) : (argument_index += 1) {
        const argument = arguments[argument_index];
        if (std.mem.eql(u8, argument, "--fix")) {
            if (apply_fixes) {
                std.debug.print("zig-lint: --fix may be passed only once.\n{s}", .{usage});

                return 2;
            }

            apply_fixes = true;

            continue;
        }

        if (std.mem.eql(u8, argument, "--plugin")) {
            argument_index += 1;

            if (argument_index == arguments.len) {
                std.debug.print("zig-lint: --plugin requires a shared-library path.\n{s}", .{usage});

                return 2;
            }

            const plugin_path = arguments[argument_index];

            const handle = zig_lint.Plugin.register(init.io, plugin_path) catch |err| {
                std.debug.print("zig-lint: could not load plugin '{s}' ({s}).\n", .{ plugin_path, @errorName(err) });

                return 1;
            };

            plugin_handles.append(init.gpa, handle) catch {
                _ = zig_lint.Plugin.unregister(init.io, handle) catch false;

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

    var report = zig_lint.lint(
        zig_lint.Allocators.same(init.gpa),
        init.io,
        input_path,
        .{
            .apply_fixes = apply_fixes,
            .plugins = plugin_handles.items,
        },
    ) catch |err| {
        std.debug.print("zig-lint: could not lint '{s}' ({s}).\n", .{ input_path, @errorName(err) });

        return 1;
    };
    defer report.deinit(init.gpa);

    if (apply_fixes) {
        for (0..report.fixed_files.len) |fixed_file_index| {
            const fixed_file = report.fixed_files.get(fixed_file_index);

            replaceSourceFile(init.io, fixed_file.path, fixed_file.text) catch |err| {
                std.debug.print("zig-lint: could not fix '{s}' ({s}).\n", .{ fixed_file.path, @errorName(err) });

                return 1;
            };

            std.debug.print("zig-lint: fixed {s}\n", .{fixed_file.path});
        }

        if (report.fixed_files.len != 0) {
            const verified_report = zig_lint.lint(
                zig_lint.Allocators.same(init.gpa),
                init.io,
                input_path,
                .{ .plugins = plugin_handles.items },
            ) catch |err| {
                std.debug.print("zig-lint: could not verify fixes for '{s}' ({s}).\n", .{ input_path, @errorName(err) });

                return 1;
            };

            report.deinit(init.gpa);

            report = verified_report;
        }
    }

    const rendered = report.renderAgent(init.gpa) catch |err| {
        std.debug.print("zig-lint: could not render diagnostics ({s}).\n", .{@errorName(err)});

        return 1;
    };
    defer init.gpa.free(rendered);

    std.Io.File.stdout().writeStreamingAll(init.io, rendered) catch |err| {
        std.debug.print("zig-lint: could not write diagnostics ({s}).\n", .{@errorName(err)});

        return 1;
    };

    return if (report.total_diagnostic_count == 0) 0 else 1;
}

fn replaceSourceFile(io: std.Io, path: []const u8, text: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const stat = try cwd.statFile(io, path, .{ .follow_symlinks = false });

    if (stat.kind != .file) {
        return error.InvalidFixTarget;
    }

    var atomic_file = try cwd.createFileAtomic(io, path, .{
        .permissions = stat.permissions,
        .replace = true,
    });
    defer atomic_file.deinit(io);

    try atomic_file.file.writeStreamingAll(io, text);
    try atomic_file.file.sync(io);
    try atomic_file.replace(io);
}

const usage =
    \\Usage: zig-lint [--fix] [--plugin <shared-library>]... <file-or-directory>
    \\
    \\--fix atomically writes compatible rule fixes and canonical formatting,
    \\then reports any diagnostics that remain.
    \\Loads only explicitly named trusted plugins; --plugin may be repeated.
    \\Writes agent-formatted diagnostics to stdout.
    \\Directory scans apply .gitignore files from the nearest Git root downward.
    \\The following descendant directories are excluded by default at every depth:
    \\  .git, .hg, .svn, .jj, .zig-cache, zig-cache, zig-out, zig-pkg,
    \\  .cache, node_modules, .venv, __fixtures__
++ "\n";
