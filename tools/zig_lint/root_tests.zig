const std = @import("std");
const builtin = @import("builtin");
const api = @import("root.zig");

const Allocators = api.Allocators;
const LintOptions = api.LintOptions;
const LintReport = api.LintReport;

const invalid_zig = "const value = ;\n";

test "lint options target the linter toolchain by default" {
    try std.testing.expectEqualDeep(
        builtin.zig_version,
        (LintOptions{}).target_zig_version,
    );
}

// ─── Repository Fixtures ────────────────────────────────────────────────────
//
// Each `.input.zig` file has a required human-diagnostic snapshot and an
// optional canonical-source snapshot. Check mode is read-only; the explicit
// update environment variable atomically replaces changed snapshots.

test "lint and fix fixtures" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;

    // ── Select Check Or Update Mode ──
    //
    // Normal runs only compare fixtures. The exact value `1` explicitly permits
    // snapshot writes, preventing an accidental nonempty value from updating them.
    //
    // ZIG_LINTER_UPDATE_FIXTURES=1 zig build test -- "lint and fix fixtures"

    var discovery_arena = std.heap.ArenaAllocator.init(gpa);
    defer discovery_arena.deinit();

    const update_value = std.testing.environ.getAlloc(
        discovery_arena.allocator(),
        "ZIG_LINTER_UPDATE_FIXTURES",
    ) catch |err|
        switch (err) {
            error.EnvironmentVariableMissing => "",
            else => return err,
        };

    const mode: FixtureMode = if (std.mem.eql(u8, update_value, "1")) .update else .check;

    // ── Discover Fixture Inputs ──
    //
    // Recursive discovery lets each rule own nested scenario groups. Sorting
    // makes the first failing fixture independent of filesystem iteration order.

    var dir = try std.Io.Dir.cwd().openDir(
        io,
        "__fixtures__",
        .{ .iterate = true },
    );
    defer dir.close(io);

    const input_paths = try discoverFixtureInputPaths(io, discovery_arena.allocator(), dir);

    // ── Check Or Update Every Fixture ──

    try std.testing.expect(input_paths.len != 0);

    for (input_paths) |input_path| {
        try lintAndFixFixture(io, gpa, dir, input_path, mode);
    }
}

const FixtureMode = enum { check, update };

fn discoverFixtureInputPaths(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
) ![][]const u8 {
    var input_paths: std.ArrayList([]const u8) = .empty;
    defer input_paths.deinit(allocator);

    // Every rule owns a fixture subdirectory; recursive discovery also permits
    // focused scenario groups beneath a rule without changing this harness.
    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".input.zig")) {
            continue;
        }

        if (std.fs.path.dirname(entry.path) == null) {
            return error.UngroupedFixture;
        }

        // Walker paths expire at the next iteration.
        try input_paths.append(allocator, try allocator.dupe(u8, entry.path));
    }

    const owned_input_paths = try input_paths.toOwnedSlice(allocator);

    // Filesystem order must not choose which fixture fails first.
    std.mem.sort([]const u8, owned_input_paths, {}, struct {
        fn lessThan(_: void, left: []const u8, right: []const u8) bool {
            return std.mem.lessThan(u8, left, right);
        }
    }.lessThan);

    return owned_input_paths;
}

fn lintAndFixFixture(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    input_path: []const u8,
    mode: FixtureMode,
) !void {
    const stem = input_path[0 .. input_path.len - ".input.zig".len];

    var phase: []const u8 = "load";
    errdefer std.debug.print("fixture {s}: {s} failed\n", .{ stem, phase });

    // ── Lint And Fix ──

    const source = try dir.readFileAlloc(io, input_path, allocator, .unlimited);
    defer allocator.free(source);

    phase = "lint";

    const filesystem_input_path = try std.fs.path.join(allocator, &.{
        "__fixtures__",
        input_path,
    });
    defer allocator.free(filesystem_input_path);

    var report: LintReport = try api.lint(
        Allocators.same(allocator),
        io,
        filesystem_input_path,
        .{ .apply_fixes = true },
    );
    defer report.deinit(allocator);

    const rendered = try report.renderHuman(allocator);
    defer allocator.free(rendered);

    phase = "fix";

    try std.testing.expect(report.fixed_files.len <= 1);

    const fixed_text: ?[:0]const u8 = if (report.fixed_files.len == 1)
        report.fixed_files.get(0).text
    else
        null;

    const fixed_source: []const u8 = fixed_text orelse source;

    if (fixed_text) |text| {
        phase = "parse fixed source";

        var fixed_ast = try std.zig.Ast.parse(allocator, text, .zig);
        defer fixed_ast.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 0), fixed_ast.errors.len);

        phase = "check Zig fmt stability";

        const formatted_fixed = try fixed_ast.renderAlloc(allocator);
        defer allocator.free(formatted_fixed);

        try std.testing.expectEqualStrings(text, formatted_fixed);
    }

    const lint_path = try std.fmt.allocPrint(allocator, "{s}.lint", .{stem});
    defer allocator.free(lint_path);

    const fix_path = try std.fmt.allocPrint(allocator, "{s}.fix.zig", .{stem});
    defer allocator.free(fix_path);

    switch (mode) {
        .check => {
            // ── Load Expectations ──

            phase = "load";

            const expected_lint = dir.readFileAlloc(io, lint_path, allocator, .unlimited) catch |err|
                switch (err) {
                    error.FileNotFound => return error.MissingFixtureLint,
                    else => return err,
                };
            defer allocator.free(expected_lint);

            const expected_fix = dir.readFileAlloc(io, fix_path, allocator, .unlimited) catch |err|
                switch (err) {
                    error.FileNotFound => null,
                    else => return err,
                };
            defer if (expected_fix) |bytes| {
                allocator.free(bytes);
            };

            // ── Check Diagnostics ──

            phase = "lint";

            try std.testing.expectEqualStrings(expected_lint, rendered);

            // ── Check Fixed Source ──

            phase = "fix";

            try std.testing.expectEqualStrings(expected_fix orelse source, fixed_source);
        },

        .update => {
            phase = "update lint";

            try updateFixtureFile(io, allocator, dir, lint_path, rendered);

            phase = "update fix";

            if (!std.mem.eql(u8, source, fixed_source)) {
                try updateFixtureFile(io, allocator, dir, fix_path, fixed_source);
            } else {
                dir.deleteFile(io, fix_path) catch |err| {
                    switch (err) {
                        error.FileNotFound => {},
                        else => return err,
                    }
                };
            }
        },
    }
}

fn updateFixtureFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    path: []const u8,
    contents: []const u8,
) !void {
    const previous = dir.readFileAlloc(io, path, allocator, .unlimited) catch |err|
        switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
    defer if (previous) |bytes| {
        allocator.free(bytes);
    };

    if (previous) |bytes| {
        if (std.mem.eql(u8, bytes, contents)) {
            return;
        }
    }

    // Replace each snapshot atomically so a failed write cannot truncate it.
    var output = try dir.createFileAtomic(io, path, .{ .replace = true });
    defer output.deinit(io);

    try output.file.writeStreamingAll(io, contents);
    try output.replace(io);

    std.debug.print("updated fixture {s}\n", .{path});
}

// ─── Report Paths And Explicit Inputs ───────────────────────────────────────

test "report path formats apply to explicit files and directory results" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Worktree ──
    //
    // The regular `.git` file models a linked worktree boundary above the scan
    // root. One file produces a diagnostic; the other produces canonical text.
    //
    //   project/                       Git root
    //   ├── .git                       worktree marker
    //   └── apps/tool/                 scan root
    //       ├── invalid.zig            diagnostic
    //       └── fixed.zig              fixed file

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "project/apps/tool");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "project/.git", .data = "gitdir: elsewhere\n" });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "project/apps/tool/invalid.zig", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "project/apps/tool/fixed.zig", .data = "const value=1;\n" });

    // ── Derive Expected Coordinates ──
    //
    // Each table row names the exact coordinate expected for the same two files.

    const cwd_path = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    defer allocator.free(cwd_path);

    const scan_path = try tmp_dir.dir.realPathFileAlloc(io, "project/apps/tool", allocator);
    defer allocator.free(scan_path);

    const invalid_path = try tmp_dir.dir.realPathFileAlloc(
        io,
        "project/apps/tool/invalid.zig",
        allocator,
    );
    defer allocator.free(invalid_path);

    const fixed_path = try tmp_dir.dir.realPathFileAlloc(
        io,
        "project/apps/tool/fixed.zig",
        allocator,
    );
    defer allocator.free(fixed_path);

    const cwd_relative_invalid_path = try std.fs.path.relative(
        allocator,
        cwd_path,
        null,
        cwd_path,
        invalid_path,
    );
    defer allocator.free(cwd_relative_invalid_path);

    const cwd_relative_fixed_path = try std.fs.path.relative(
        allocator,
        cwd_path,
        null,
        cwd_path,
        fixed_path,
    );
    defer allocator.free(cwd_relative_fixed_path);

    const cases = [_]struct {
        format: api.ReportPathFormat,
        invalid_path: []const u8,
        fixed_path: []const u8,
    }{
        .{
            .format = .cwd_relative,
            .invalid_path = cwd_relative_invalid_path,
            .fixed_path = cwd_relative_fixed_path,
        },
        .{
            .format = .git_root_relative,
            .invalid_path = "apps/tool/invalid.zig",
            .fixed_path = "apps/tool/fixed.zig",
        },
        .{
            .format = .absolute,
            .invalid_path = invalid_path,
            .fixed_path = fixed_path,
        },
    };

    // ── Compare Directory And Explicit Inputs ──
    //
    // Directory linting applies the format to diagnostics and fixed files.
    // Explicit linting must report the same invalid file in identical coordinates.

    try std.testing.expectEqual(.cwd_relative, (LintOptions{}).report_path_format);

    for (cases) |case| {
        var dir_report = try api.lint(
            Allocators.same(allocator),
            io,
            scan_path,
            .{
                .apply_fixes = true,
                .report_path_format = case.format,
            },
        );
        defer dir_report.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 1), dir_report.diagnostics.len);

        try std.testing.expectEqualStrings(
            case.invalid_path,
            dir_report.diagnostics[0].path,
        );

        try std.testing.expectEqual(@as(usize, 1), dir_report.fixed_files.len);

        try std.testing.expectEqualStrings(
            case.fixed_path,
            dir_report.fixed_files.get(0).path,
        );

        var file_report = try api.lint(
            Allocators.same(allocator),
            io,
            invalid_path,
            .{ .report_path_format = case.format },
        );
        defer file_report.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 1), file_report.diagnostics.len);

        try std.testing.expectEqualStrings(
            case.invalid_path,
            file_report.diagnostics[0].path,
        );
    }
}

test "git-root-relative report paths fall back to the working directory" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Source Outside A Git Repository ──
    //
    // `std.testing.tmpDir` lives beneath this repository and would find its
    // `.git` marker. Borrow its random name for an isolated fixture beneath
    // `/tmp`, whose ancestors contain no Git root.
    //
    //   process working directory     repository containing this test
    //   /tmp/
    //   └── zig-linter-<random>/      no `.git` marker
    //       └── source/
    //           └── invalid.zig

    var unique_name_source = std.testing.tmpDir(.{});
    defer unique_name_source.cleanup();

    const fixture_path = try std.fmt.allocPrint(
        allocator,
        "/tmp/zig-linter-{s}",
        .{&unique_name_source.sub_path},
    );
    defer allocator.free(fixture_path);
    defer std.Io.Dir.cwd().deleteTree(io, fixture_path) catch {};

    const scan_path = try std.fs.path.join(allocator, &.{ fixture_path, "source" });
    defer allocator.free(scan_path);

    const file_path = try std.fs.path.join(allocator, &.{ scan_path, "invalid.zig" });
    defer allocator.free(file_path);

    try std.Io.Dir.cwd().createDirPath(io, scan_path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file_path, .data = invalid_zig });

    // ── Derive The Fallback Path ──
    //
    // With no Git root, `git_root_relative` must produce the same path as
    // `cwd_relative`: the route from the process working directory to the file.

    const cwd_path = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    defer allocator.free(cwd_path);

    const cwd_relative_file_path = try std.fs.path.relative(
        allocator,
        cwd_path,
        null, // env map
        cwd_path,
        file_path,
    );
    defer allocator.free(cwd_relative_file_path);

    // ── Render Diagnostic ──
    //
    // The complete human output makes the selected path coordinate visible.

    const actual_human = try lintAndRenderHuman(allocator, scan_path, .{
        .report_path_format = .git_root_relative,
    });
    defer allocator.free(actual_human);

    const expected_human = try std.fmt.allocPrint(allocator,
        \\error[parse]: expected expression, found ';'
        \\ --> {s}:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
    , .{cwd_relative_file_path});
    defer allocator.free(expected_human);

    try std.testing.expectEqualStrings(expected_human, actual_human);
}

test "explicit files bypass directory exclusions and gitignore" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const options: LintOptions = .{ .report_path_format = .git_root_relative };

    // ── Set Up Repository ──
    //
    // Each directory-only filter excludes a different invalid file. The default
    // exclusion prunes `node_modules`, while the root Gitignore prunes `ignored`.
    //
    //   source/
    //   ├── .git/
    //   ├── .gitignore             "ignored/"
    //   ├── node_modules/          default directory exclusion
    //   │   └── invalid.zig
    //   └── ignored/               Gitignore exclusion
    //       └── invalid.zig

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.createDirPath(io, "source/node_modules");
    try tmp_dir.dir.createDirPath(io, "source/ignored");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/.gitignore", .data = "ignored/\n" });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/node_modules/invalid.zig", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/ignored/invalid.zig", .data = invalid_zig });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    const dir_excluded_path = try tmp_dir.dir.realPathFileAlloc(
        io,
        "source/node_modules/invalid.zig",
        allocator,
    );
    defer allocator.free(dir_excluded_path);

    const gitignored_path = try tmp_dir.dir.realPathFileAlloc(
        io,
        "source/ignored/invalid.zig",
        allocator,
    );
    defer allocator.free(gitignored_path);

    // ── Scan Repository ──
    //
    // Both filters apply to a directory scan, so neither parse error appears.

    const dir_human = try lintAndRenderHuman(allocator, root, options);
    defer allocator.free(dir_human);

    try std.testing.expectEqualStrings("", dir_human);

    // ── Lint Explicit Files ──
    //
    // Selecting either file directly bypasses the filter that hid it during
    // the directory scan. Both parse errors therefore appear.

    const dir_excluded_human = try lintAndRenderHuman(
        allocator,
        dir_excluded_path,
        options,
    );
    defer allocator.free(dir_excluded_human);

    const expected_dir_excluded_human =
        \\error[parse]: expected expression, found ';'
        \\ --> node_modules/invalid.zig:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
    ;

    try std.testing.expectEqualStrings(
        expected_dir_excluded_human,
        dir_excluded_human,
    );

    const gitignored_human = try lintAndRenderHuman(allocator, gitignored_path, options);
    defer allocator.free(gitignored_human);

    const expected_gitignored_human =
        \\error[parse]: expected expression, found ';'
        \\ --> ignored/invalid.zig:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
    ;

    try std.testing.expectEqualStrings(expected_gitignored_human, gitignored_human);
}

// ─── Fixes ──────────────────────────────────────────────────────────────────

test "formatting runs without lint fixes" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Unformatted Source ──
    //
    // The input is valid Zig and violates no lint rule. Only spacing differs
    // from canonical Zig formatting:
    //
    //   input:  const value=1;
    //   output: const value = 1;

    const unformatted_zig = "const value=1;\n";
    const formatted_zig = "const value = 1;\n";

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "unformatted.zig",
        .data = unformatted_zig,
    });

    const path = try tmp_dir.dir.realPathFileAlloc(io, "unformatted.zig", allocator);
    defer allocator.free(path);

    // ── Lint And Format ──
    //
    // `apply_fixes` requests both rule-authored edits and Zig formatting.

    var report: LintReport = try api.lint(
        Allocators.same(allocator),
        io,
        path,
        .{ .apply_fixes = true },
    );
    defer report.deinit(allocator);

    // ── Check Report ──
    //
    // Formatting alone publishes one fixed file without creating a diagnostic
    // or incrementing the counts reserved for rule-authored fixes.

    const human = try report.renderHuman(allocator);
    defer allocator.free(human);

    try std.testing.expectEqualStrings("", human);
    try std.testing.expectEqual(@as(usize, 1), report.fixed_files.len);
    try std.testing.expectEqual(@as(usize, 0), report.applied_fix_count);
    try std.testing.expectEqual(@as(usize, 0), report.skipped_fix_count);
    try std.testing.expectEqualStrings(formatted_zig, report.fixed_files.get(0).text);
}

test "directory fixes are report-owned and never written" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Two Unformatted Files ──
    //
    // Both files need only Zig formatting. `a.zig` changes `const a=1;` to
    // `const a = 1;`, while `b.zig` makes the corresponding change for `b`.

    const source_a = "const a=1;\n";
    const source_b = "const b=2;\n";

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = source_a });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = source_b });

    const path = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(path);

    // ── Produce Report With Separate Scratch Storage ──
    //
    // Resetting scratch immediately after `lint` proves that every returned path
    // and source byte belongs to the report allocator.

    var scratch_arena = std.heap.ArenaAllocator.init(allocator);
    defer scratch_arena.deinit();

    var report: LintReport = try api.lint(
        .{
            .scratch = scratch_arena.allocator(),
            .report = allocator,
        },
        io,
        path,
        .{
            .apply_fixes = true,
            .report_path_format = .git_root_relative,
        },
    );
    defer report.deinit(allocator);

    _ = scratch_arena.reset(.retain_capacity);

    // ── Check Returned Report ──
    //
    // The report remains readable after scratch reset and keeps fixed files in
    // path order. Formatting produces no lint diagnostics.

    const human = try report.renderHuman(allocator);
    defer allocator.free(human);

    try std.testing.expectEqualStrings("", human);
    try std.testing.expectEqual(@as(usize, 2), report.fixed_files.len);

    const fixed_paths: []const []const u8 = report.fixed_files.items(.path);
    const fixed_texts: []const [:0]const u8 = report.fixed_files.items(.text);

    try std.testing.expectEqualStrings("a.zig", fixed_paths[0]);
    try std.testing.expectEqualStrings("b.zig", fixed_paths[1]);
    try std.testing.expectEqualStrings("const a = 1;\n", fixed_texts[0]);
    try std.testing.expectEqualStrings("const b = 2;\n", fixed_texts[1]);

    // ── Check Filesystem ──
    //
    // Returning fixed text must not modify either source file.

    const unchanged_a = try tmp_dir.dir.readFileAlloc(io, "source/a.zig", allocator, .unlimited);
    defer allocator.free(unchanged_a);

    const unchanged_b = try tmp_dir.dir.readFileAlloc(io, "source/b.zig", allocator, .unlimited);
    defer allocator.free(unchanged_b);

    try std.testing.expectEqualStrings(source_a, unchanged_a);
    try std.testing.expectEqualStrings(source_b, unchanged_b);
}

// ─── Filesystem Selection ───────────────────────────────────────────────────

test "file size limit is inclusive and checked before parsing" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Explicit File ──
    //
    // The same file exercises zero, small, default, and above-default limits.
    // Spaces remain valid Zig at every accepted size.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.writeFile(io, .{ .sub_path = "limit.zig", .data = "" });

    const path = try tmp_dir.dir.realPathFileAlloc(io, "limit.zig", allocator);
    defer allocator.free(path);

    const default_limit = (LintOptions{}).file_size_max;
    const limits = [_]u32{ 0, 64, default_limit, default_limit + 32 };

    // ── Check Both Sides Of Each Limit ──
    //
    // A file one byte above the limit fails before report-path allocation; the
    // failing allocators make that ordering observable. Truncating the same
    // source to the exact limit succeeds and renders no diagnostic.

    for (limits) |file_size_max| {
        const source = try allocator.alloc(u8, file_size_max + 1);
        defer allocator.free(source);

        @memset(source, ' ');

        try tmp_dir.dir.writeFile(io, .{ .sub_path = "limit.zig", .data = source });

        try std.testing.expectError(error.FileTooLarge, api.lint(
            .{
                .scratch = std.testing.failing_allocator,
                .report = std.testing.failing_allocator,
            },
            io,
            path,
            .{ .file_size_max = file_size_max },
        ));

        try tmp_dir.dir.writeFile(io, .{
            .sub_path = "limit.zig",
            .data = source[0..file_size_max],
        });

        const human = try lintAndRenderHuman(allocator, path, .{
            .file_size_max = file_size_max,
        });
        defer allocator.free(human);

        try std.testing.expectEqualStrings("", human);
    }
}

test "directory scan reports an oversized file and continues" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const file_size_max = invalid_zig.len;
    const oversized_zig = " " ** (file_size_max + 1);

    // ── Set Up One Eligible And One Oversized File ──
    //
    // The normal source fits exactly. The second file exceeds the same limit
    // by one byte, so a directory scan must report it without opening or parsing
    // its contents.
    //
    //   source/
    //   ├── .git/
    //   ├── included.zig      16 bytes, parsed normally
    //   └── oversized.zig     17 bytes, reported before parsing

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/included.zig", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/oversized.zig", .data = oversized_zig });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    // ── Scan Directory ──
    //
    // Continuing after the size diagnostic exposes the parse error from the
    // fitting file in the same human report.

    try lintAndExpectHuman(root, .{
        .file_size_max = file_size_max,
        .report_path_format = .git_root_relative,
    },
        \\error[parse]: expected expression, found ';'
        \\ --> included.zig:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
        \\error[file_size]: file exceeds the 16-byte size limit
        \\ --> oversized.zig:1:1
        \\  = help: split the file or increase `file_size_max`
        \\
    );
}

test "directory scan follows regular file aliases and skips ignored and non-regular paths" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Build The Traversal Cases ──
    //
    // Three regular files are eligible for linting. Every other `.zig` path is
    // pruned, skipped by kind, or excluded by its non-Zig basename:
    //
    //   source/.git/                     repository marker
    //   source/error.zig                 parse error
    //   source/regular_alias.zig         alias of the same parse error
    //   source/ignored.txt               wrong extension
    //   source/.gitignore                ignored symlink to external rules
    //   source/fifo_scope/.gitignore     ignored FIFO that must not be opened
    //   source/fifo_scope/error.zig      parse error
    //   source/__fixtures__/ignored.zig  pruned directory
    //   source/dir_alias.zig             directory symlink
    //   source/device_alias.zig          character-device symlink

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.createDirPath(io, "source/__fixtures__");
    try tmp_dir.dir.createDirPath(io, "source/fifo_scope");
    try tmp_dir.dir.createDirPath(io, "other");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/error.zig", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/ignored.txt", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/__fixtures__/ignored.zig", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/fifo_scope/error.zig", .data = invalid_zig });
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "other/ignored.zig", .data = invalid_zig });

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "other/.gitignore",
        .data = "error.zig\nregular_alias.zig\n",
    });

    try tmp_dir.dir.symLink(io, "../other/.gitignore", "source/.gitignore", .{});
    try tmp_dir.dir.symLink(io, "error.zig", "source/regular_alias.zig", .{});

    const fifo_result = std.os.linux.mknodat(
        tmp_dir.dir.handle,
        "source/fifo_scope/.gitignore",
        std.os.linux.S.IFIFO | 0o600,
        0,
    );

    try std.testing.expectEqual(.SUCCESS, std.posix.errno(fifo_result));
    try tmp_dir.dir.symLink(io, "../other", "source/dir_alias.zig", .{});
    try tmp_dir.dir.symLink(io, "/dev/zero", "source/device_alias.zig", .{});

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    // ── Scan Eligible Regular Files ──
    //
    // Following the `.gitignore` symlink would suppress two eligible files,
    // opening the FIFO would block, and reading `/dev/zero` would exceed the
    // exact source-size limit.
    try lintAndExpectGitRootDiagnosticPaths(root, .{
        .file_size_max = invalid_zig.len,
    }, &.{
        "error.zig",
        "fifo_scope/error.zig",
        "regular_alias.zig",
    });
}

// ─── Gitignore And Directory Exclusions ─────────────────────────────────────

test "directory scan applies root and nested gitignore rules before traversal" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Construct The Repository ──
    //
    // Every source file contains invalid Zig. The resulting diagnostic paths
    // reveal exactly which files survived rule evaluation and directory pruning.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const dirs = [_][]const u8{
        "source/.git",
        "source/sub",
        "source/src",
        "source/build",
        "source/output",
        "source/cache/nested",
    };

    for (dirs) |dir| {
        try tmp_dir.dir.createDirPath(io, dir);
    }

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "source/.gitignore",
        .data =
        \\*.generated.zig
        \\/root-only.zig
        \\build/
        \\!build/keep.zig
        \\output/*
        \\!output/keep.zig
        \\cache/**
        \\!.git/invalid.zig
        ,
    });

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "source/src/.gitignore",
        .data = "!keep.generated.zig\n",
    });

    const source_paths = [_][]const u8{
        "source/bad.zig",
        "source/drop.generated.zig",
        "source/root-only.zig",
        "source/sub/root-only.zig",
        "source/src/drop.generated.zig",
        "source/src/keep.generated.zig",
        "source/build/keep.zig",
        "source/output/drop.zig",
        "source/output/keep.zig",
        "source/cache/direct.zig",
        "source/cache/nested/deep.zig",
        "source/.git/invalid.zig",
    };

    for (source_paths) |source_path| {
        try tmp_dir.dir.writeFile(io, .{ .sub_path = source_path, .data = invalid_zig });
    }

    // ── Verify The Surviving Files ──
    //
    // `build/` demonstrates prune semantics: the later negation cannot revive
    // a file below an excluded parent. The default `.git` directory-name
    // exclusion is independent of repository-authored negation.

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    try lintAndExpectGitRootDiagnosticPaths(root, .{}, &.{
        "bad.zig",
        "output/keep.zig",
        "src/keep.generated.zig",
        "sub/root-only.zig",
    });
}

test "nested directory scan inherits gitignore rules from repository root" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Repository And Inherited Rules ──
    //
    // The requested `tool` directory has one rule at every active level. A
    // separate root rule excludes `blocked` before a scan can enter it.
    //
    //   repository/
    //   ├── .git/
    //   ├── .gitignore                 root-ignored.zig; blocked/
    //   └── apps/
    //       ├── .gitignore             parent-ignored.zig
    //       ├── blocked/               ignored requested root
    //       │   └── error.zig
    //       └── tool/                  included requested root
    //           ├── .gitignore         local-ignored.zig
    //           ├── included.zig
    //           ├── local-ignored.zig
    //           ├── parent-ignored.zig
    //           └── root-ignored.zig

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "repository/.git");
    try tmp_dir.dir.createDirPath(io, "repository/apps/blocked");
    try tmp_dir.dir.createDirPath(io, "repository/apps/tool");

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "repository/.gitignore",
        .data = "/apps/tool/root-ignored.zig\n/apps/blocked/\n",
    });

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "repository/apps/.gitignore",
        .data = "tool/parent-ignored.zig\n",
    });

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "repository/apps/tool/.gitignore",
        .data = "local-ignored.zig\n",
    });

    const source_paths = [_][]const u8{
        "repository/apps/blocked/error.zig",
        "repository/apps/tool/included.zig",
        "repository/apps/tool/local-ignored.zig",
        "repository/apps/tool/parent-ignored.zig",
        "repository/apps/tool/root-ignored.zig",
    };

    for (source_paths) |source_path| {
        try tmp_dir.dir.writeFile(io, .{ .sub_path = source_path, .data = invalid_zig });
    }

    // ── Scan Included Nested Root ──
    //
    // Rules loaded from the repository root, `apps`, and `tool` each remove one
    // source file before the remaining file produces its parse diagnostic.

    const tool_root = try tmp_dir.dir.realPathFileAlloc(io, "repository/apps/tool", allocator);
    defer allocator.free(tool_root);

    try lintAndExpectGitRootDiagnosticPaths(tool_root, .{}, &.{
        "apps/tool/included.zig",
    });

    // ── Scan Ignored Nested Root ──
    //
    // The repository rule excludes `blocked` itself, so its descendant is never
    // visited and the complete report remains empty.

    const blocked_root = try tmp_dir.dir.realPathFileAlloc(io, "repository/apps/blocked", allocator);
    defer allocator.free(blocked_root);

    try lintAndExpectGitRootDiagnosticPaths(blocked_root, .{}, &.{});

    // Disabling Gitignore restores direct discovery beneath the same scan root.
    try lintAndExpectGitRootDiagnosticPaths(blocked_root, .{
        .respect_gitignore = false,
    }, &.{
        "apps/blocked/error.zig",
    });
}

test "directory options replace default exclusions and control gitignore" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Set Up Independent Exclusions ──
    //
    // Each mechanism hides a different invalid file under the default options.
    // `custom` remains as the one visible control path.
    //
    //   source/
    //   ├── .git/error.zig             default exclusion
    //   ├── ignored/error.zig          root Gitignore
    //   ├── node_modules/error.zig     default exclusion
    //   └── custom/error.zig           included control

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const dirs = [_][]const u8{
        "source/.git",
        "source/custom",
        "source/ignored",
        "source/node_modules",
    };

    for (dirs) |dir| {
        try tmp_dir.dir.createDirPath(io, dir);
    }

    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/.gitignore", .data = "ignored/\n" });

    const source_paths = [_][]const u8{
        "source/.git/error.zig",
        "source/custom/error.zig",
        "source/ignored/error.zig",
        "source/node_modules/error.zig",
    };

    for (source_paths) |source_path| {
        try tmp_dir.dir.writeFile(io, .{ .sub_path = source_path, .data = invalid_zig });
    }

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    // ── Compare Option Sets ──

    // Defaults prune `.git` and `node_modules`; `.gitignore` prunes `ignored`.
    try lintAndExpectGitRootDiagnosticPaths(root, .{}, &.{
        "custom/error.zig",
    });

    // Replacement removes every default and makes only `custom` authoritative.
    try lintAndExpectGitRootDiagnosticPaths(root, .{
        .excluded_dir_names = &.{"custom"},
    }, &.{
        ".git/error.zig",
        "node_modules/error.zig",
    });

    // Gitignore control does not disable the independent default exclusions.
    try lintAndExpectGitRootDiagnosticPaths(root, .{
        .respect_gitignore = false,
    }, &.{
        "custom/error.zig",
        "ignored/error.zig",
    });

    // Empty replacement and disabled gitignore expose every source file.
    try lintAndExpectGitRootDiagnosticPaths(root, .{
        .excluded_dir_names = &.{},
        .respect_gitignore = false,
    }, &.{
        ".git/error.zig",
        "custom/error.zig",
        "ignored/error.zig",
        "node_modules/error.zig",
    });
}

test "directory scan handles gitignore syntax regressions through the public API" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Define Isolated Syntax Cases ──
    //
    // Each row becomes one directory containing its own `.gitignore` and invalid
    // source file. Isolation prevents one pattern from changing another decision.

    const cases = [_]struct {
        scope: []const u8,
        patterns: []const u8,
        source_path: []const u8,
    }{
        .{ .scope = "malformed-class", .patterns = "file[.zig\n", .source_path = "file[.zig" },
        .{ .scope = "dangling-escape", .patterns = "file.zig\\\n", .source_path = "file.zig" },
        .{ .scope = "escaped-class", .patterns = "file\\[.zig\n", .source_path = "file[.zig" },
        .{ .scope = "posix-class", .patterns = "file[[:digit:]].zig\n", .source_path = "file7.zig" },
        .{ .scope = "all-stars", .patterns = "a/***/file.zig\n", .source_path = "a/b/file.zig" },
        .{ .scope = "escaped-separator", .patterns = "a\\/b/file.zig\n", .source_path = "a/b/file.zig" },
    };

    // ── Build Repository ──

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");

    for (cases) |case| {
        const scope = try std.fmt.allocPrint(allocator, "source/{s}", .{case.scope});
        defer allocator.free(scope);

        const ignore_path = try std.fmt.allocPrint(allocator, "{s}/.gitignore", .{scope});
        defer allocator.free(ignore_path);

        const source_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ scope, case.source_path });
        defer allocator.free(source_path);

        try tmp_dir.dir.createDirPath(io, std.fs.path.dirname(source_path) orelse {
            unreachable;
        });

        try tmp_dir.dir.writeFile(io, .{ .sub_path = ignore_path, .data = case.patterns });
        try tmp_dir.dir.writeFile(io, .{ .sub_path = source_path, .data = invalid_zig });
    }

    // ── Check Surviving Source Files ──
    //
    // Malformed classes and dangling escapes match nothing. The escaped class,
    // POSIX class, recursive stars, and escaped separator each suppress their
    // source file, leaving exactly these two diagnostics.

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    try lintAndExpectGitRootDiagnosticPaths(root, .{}, &.{
        "dangling-escape/file.zig",
        "malformed-class/file[.zig",
    });
}

test "directory scan rejects an oversized gitignore" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const gitignore_size_max = 120 * 1024;

    // ── Set Up Oversized Gitignore ──
    //
    // One byte beyond the documented 120-KiB limit makes the root ignore file
    // invalid before traversal begins.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");

    var ignore_file = try tmp_dir.dir.createFile(io, "source/.gitignore", .{});
    defer ignore_file.close(io);

    try ignore_file.setLength(io, gitignore_size_max + 1);

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    // ── Reject Scan ──

    try std.testing.expectError(
        error.StreamTooLong,
        api.lint(Allocators.same(allocator), io, root, .{}),
    );
}

// ─── Persistent Caching ─────────────────────────────────────────────────────

test "directory cache follows its documented metadata heuristic" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const clean_zig = "const value =1;\n";

    const invalid_human =
        \\error[parse]: expected expression, found ';'
        \\ --> a.zig:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
    ;

    // ── Populate Cache From Clean Source ──
    //
    // The first scan parses `a.zig`, emits no diagnostic, and records its path,
    // inode, size, and modification time. The cache lives outside the scan root.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = clean_zig });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    const file_path = try tmp_dir.dir.realPathFileAlloc(io, "source/a.zig", allocator);
    defer allocator.free(file_path);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    const fresh_cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "fresh-cache" });
    defer allocator.free(fresh_cache_base);

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, "");

    // ── Change Content Without Changing Cache Identity ──
    //
    // Replace the clean source with same-length invalid bytes and restore its
    // modification time. The write preserves its inode, leaving every cache-key
    // field unchanged even though the source now contains a parse error.

    try replaceFilePreservingCacheIdentity(
        io,
        tmp_dir.dir,
        "source/a.zig",
        invalid_zig,
    );

    // ── Compare Existing And Fresh Cache Directories ──
    //
    // The original cache trusts the unchanged metadata and emits nothing. A
    // different, fresh base directory must miss and expose the parse error.

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, "");

    // An explicit file never consults or updates persistent cache state. It
    // observes the invalid source, while the complete directory scope retains
    // its independently reusable metadata-identified clean fact.
    try lintAndExpectHuman(file_path, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, invalid_human);

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, "");

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = fresh_cache_base },
        .report_path_format = .git_root_relative,
    }, invalid_human);

    // ── Invalidate Cache With A Size Change ──
    //
    // Removing the final newline changes the size. The cached scan must miss,
    // parse the invalid source, and return the same visible diagnostic.

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "source/a.zig",
        .data = invalid_zig[0 .. invalid_zig.len - 1],
    });

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, invalid_human);
}

test "ignored scan root commits an empty cache generation" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const clean_zig = "const value =1;\n";

    const invalid_human =
        \\error[parse]: expected expression, found ';'
        \\ --> project/a.zig:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
    ;

    // ── Populate Caches For An Included Root ──

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "repo/.git");
    try tmp_dir.dir.createDirPath(io, "repo/project");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "repo/project/a.zig", .data = clean_zig });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "repo/project", allocator);
    defer allocator.free(root);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, "");

    // ── Exclude The Complete Scan Root ──
    //
    // An ignored root is a successful scan containing no selected files, so it
    // commits an empty lint and syntax generation rather than retaining entries
    // from the earlier included scan.

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "repo/.gitignore",
        .data = "project/\n",
    });

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, "");

    // ── Include The Root With Metadata-Preserving Invalid Source ──
    //
    // If either old cache entry survived the ignored scan, its matching source
    // identity would hide this parse error.

    try tmp_dir.dir.deleteFile(io, "repo/.gitignore");

    try replaceFilePreservingCacheIdentity(
        io,
        tmp_dir.dir,
        "repo/project/a.zig",
        invalid_zig,
    );

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, invalid_human);
}

test "directory cache verifies linting and Zig formatting independently" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const unformatted_zig = "const value=1;\n";
    const formatted_zig = "const value = 1;\n";

    // ── Set Up Unformatted Cached Source ──
    //
    // The source passes every lint rule but differs from canonical Zig format.
    // Cache files remain outside the scanned repository.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = unformatted_zig });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    // ── Verify Linting Only ──
    //
    // The first pass records clean linting but does not request or verify format.

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, "");

    // ── Request Formatting ──
    //
    // The cache may skip lint rules, but it must still format the source because
    // the first pass did not verify canonical text.

    var first_fix = try api.lint(Allocators.same(allocator), io, root, .{
        .apply_fixes = true,
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    });
    defer first_fix.deinit(allocator);

    const first_human = try first_fix.renderHuman(allocator);
    defer allocator.free(first_human);

    try std.testing.expectEqualStrings("", first_human);
    try std.testing.expectEqual(@as(usize, 1), first_fix.fixed_files.len);
    try std.testing.expectEqualStrings("a.zig", first_fix.fixed_files.get(0).path);
    try std.testing.expectEqualStrings(formatted_zig, first_fix.fixed_files.get(0).text);

    // ── Repeat Without Publishing ──
    //
    // Returning fixed text does not modify the input. The unchanged unformatted
    // file therefore produces the same fixed output on the next pass.

    var repeated_fix = try api.lint(Allocators.same(allocator), io, root, .{
        .apply_fixes = true,
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    });
    defer repeated_fix.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), repeated_fix.fixed_files.len);
    try std.testing.expectEqualStrings("a.zig", repeated_fix.fixed_files.get(0).path);
    try std.testing.expectEqualStrings(formatted_zig, repeated_fix.fixed_files.get(0).text);

    // ── Publish And Verify Canonical Source ──
    //
    // Once the caller writes the returned text, the next pass verifies format
    // and returns no diagnostic or fixed file.

    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = formatted_zig });

    var verified = try api.lint(Allocators.same(allocator), io, root, .{
        .apply_fixes = true,
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    });
    defer verified.deinit(allocator);

    const verified_human = try verified.renderHuman(allocator);
    defer allocator.free(verified_human);

    try std.testing.expectEqualStrings("", verified_human);
    try std.testing.expectEqual(@as(usize, 0), verified.fixed_files.len);
}

test "plugin selection bypasses unkeyed lint-clean cache state" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // ── Populate A Plugin-Free Clean Fact ──

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");

    try tmp_dir.dir.writeFile(io, .{
        .sub_path = "source/a.zig",
        .data = "const value = 1;\n",
    });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    var initial_report = try api.lint(Allocators.same(allocator), io, root, .{
        .cache = .{ .base_dir = cache_base },
    });
    defer initial_report.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), initial_report.diagnostics.len);

    // ── Require Plugin Validation On The Same Source Identity ──
    //
    // Until plugin artifacts participate in lint-cache identity, selecting any
    // plugin must bypass the clean fact above. The absent handle is therefore
    // observed rather than hidden by a cache hit.

    const absent_plugin: api.Plugin.Handle = @enumFromInt(std.math.maxInt(u64));

    try std.testing.expectError(
        error.PluginNotRegistered,
        api.lint(Allocators.same(allocator), io, root, .{
            .cache = .{ .base_dir = cache_base },
            .plugins = &.{absent_plugin},
        }),
    );
}

test "directory token cache follows its documented metadata heuristic" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const clean_zig = "const value =1;\n";

    const invalid_human =
        \\error[parse]: expected expression, found ';'
        \\ --> a.zig:1:15
        \\  |
        \\1 | const value = ;
        \\  |               ^
        \\  |
        \\  = help: fix this syntax error before running the linter
        \\
    ;

    // ── Cache Invalid Source And Tokens ──
    //
    // A syntax diagnostic prevents a clean-lint record, but the completed scan
    // still saves the source snapshot and tokenizer output.

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.createDirPath(io, "source/.git");
    try tmp_dir.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = invalid_zig });

    const root = try tmp_dir.dir.realPathFileAlloc(io, "source", allocator);
    defer allocator.free(root);

    const tmp_path = try tmp_dir.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmp_path);

    const cache_base = try std.fs.path.join(allocator, &.{ tmp_path, "cache" });
    defer allocator.free(cache_base);

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, invalid_human);

    // ── Change Content Without Changing Token Identity ──
    //
    // Replace the source with same-length valid Zig and restore its modification
    // time. The inode, size, and mtime still identify the cached invalid snapshot.

    try replaceFilePreservingCacheIdentity(
        io,
        tmp_dir.dir,
        "source/a.zig",
        clean_zig,
    );

    // ── Compare Cached And Uncached Scans ──
    //
    // The cached scan parses its stored invalid snapshot. The uncached scan reads
    // the replacement and emits nothing because that source is valid.

    try lintAndExpectHuman(root, .{
        .cache = .{ .base_dir = cache_base },
        .report_path_format = .git_root_relative,
    }, invalid_human);

    try lintAndExpectHuman(root, .{
        .report_path_format = .git_root_relative,
    }, "");
}

// ─── Example-Test Operations ────────────────────────────────────────────────

fn replaceFilePreservingCacheIdentity(
    io: std.Io,
    dir: std.Io.Dir,
    path: []const u8,
    replacement: []const u8,
) !void {
    const before = try dir.statFile(io, path, .{});

    try std.testing.expectEqual(before.size, replacement.len);
    try dir.writeFile(io, .{ .sub_path = path, .data = replacement });

    {
        var file = try dir.openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);

        try file.setTimestamps(io, .{ .modify_timestamp = .{ .new = before.mtime } });
    }

    const after = try dir.statFile(io, path, .{});

    try std.testing.expectEqual(before.inode, after.inode);
    try std.testing.expectEqual(before.size, after.size);
    try std.testing.expectEqual(before.mtime.nanoseconds, after.mtime.nanoseconds);
}

fn lintAndRenderHuman(
    allocator: std.mem.Allocator,
    path: []const u8,
    options: LintOptions,
) ![]u8 {
    var report = try api.lint(
        Allocators.same(allocator),
        std.testing.io,
        path,
        options,
    );
    defer report.deinit(allocator);

    return report.renderHuman(allocator);
}

fn lintAndExpectHuman(
    path: []const u8,
    options: LintOptions,
    expected: []const u8,
) !void {
    const allocator = std.testing.allocator;

    const actual = try lintAndRenderHuman(allocator, path, options);
    defer allocator.free(actual);

    try std.testing.expectEqualStrings(expected, actual);
}

fn lintAndExpectGitRootDiagnosticPaths(
    path: []const u8,
    options: LintOptions,
    expected_paths: []const []const u8,
) !void {
    const allocator = std.testing.allocator;

    var git_root_options = options;

    git_root_options.report_path_format = .git_root_relative;

    var report = try api.lint(
        Allocators.same(allocator),
        std.testing.io,
        path,
        git_root_options,
    );
    defer report.deinit(allocator);

    try std.testing.expectEqual(expected_paths.len, report.diagnostics.len);

    for (expected_paths, report.diagnostics) |expected, diagnostic| {
        try std.testing.expectEqualStrings(expected, diagnostic.path);
    }
}

// ─── Generated Gitignore Oracle Tests ──────────────────────────────────────
//
// One test-runner seed generates a complete repository rather than isolated
// matcher cases. Files and `.gitignore` scopes may occur at the root or at any
// generated depth, so parent rules, sibling paths, and nested rules interact:
//
//   seed
//    └─ repository/
//       ├─ .gitignore
//       ├─ <random>-0.zig
//       ├─ <random>-0/
//       │  ├─ .gitignore
//       │  └─ <random>-1.zig
//       └─ <random>-1/<random>-2/<random>-2.zig
//
// Every generated source file contains a deliberate syntax error. One batched
// `git check-ignore --no-index` invocation supplies the oracle: Git prints
// ignored source paths and omits included ones. A public linter scan from every
// generated directory must do the inverse by reporting only included sources.

test "generated repository agrees with git check-ignore" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const corpus_allocator = arena.allocator();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // ── Establish An Isolated Git Repository ──
    //
    // Git supplies only the ignore oracle. The test skips when the executable is
    // unavailable and otherwise keeps its index and configuration inside `tmp_dir`.

    const init_result = std.process.run(corpus_allocator, io, .{
        .argv = &.{ "git", "init", "--quiet" },
        .cwd = .{ .dir = tmp_dir.dir },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
    }) catch |err|
        switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };

    if (init_result.term != .exited or init_result.term.exited != 0) {
        std.debug.print("git init failed: {s}\n", .{init_result.stderr});

        return error.GitInitFailed;
    }

    const seed = std.testing.random_seed;
    errdefer std.debug.print(
        "ignore differential failed with seed 0x{x}\n" ++
            "replay: zig build --seed {d} test\n",
        .{ seed, seed },
    );

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    // ── Generate One Directory Tree ──
    //
    // Index zero is the repository root. Every later directory chooses an
    // existing parent, while the depth cap keeps paths accepted by both tools.

    const Dir = struct {
        path: []const u8,
        depth: usize,
    };

    const dir_depth_max = 6;
    const dirs_count = random.intRangeAtMost(usize, 24, 64);

    var dirs: std.ArrayList(Dir) = .empty;
    defer dirs.deinit(allocator);

    try dirs.ensureTotalCapacity(allocator, dirs_count + 1);
    dirs.appendAssumeCapacity(.{ .path = "", .depth = 0 });

    for (0..dirs_count) |dir_index| {
        var parent = dirs.items[random.uintLessThan(usize, dirs.items.len)];

        while (parent.depth == dir_depth_max) {
            parent = dirs.items[random.uintLessThan(usize, dirs.items.len)];
        }

        const basename = try generatePathName(corpus_allocator, random, dir_index, "");
        const path = try joinRelativePath(corpus_allocator, parent.path, basename);

        try tmp_dir.dir.createDirPath(io, path);
        dirs.appendAssumeCapacity(.{ .path = path, .depth = parent.depth + 1 });
    }

    // ── Distribute Invalid Zig Sources ──
    //
    // Every source independently chooses any generated directory, including the
    // repository root. No location is required to contain a source file.

    const source_paths_count = random.intRangeAtMost(usize, 1500, 3000);

    var source_paths: std.ArrayList([]const u8) = .empty;
    defer source_paths.deinit(allocator);

    try source_paths.ensureTotalCapacity(allocator, source_paths_count);

    for (0..source_paths_count) |source_index| {
        const dir = dirs.items[random.uintLessThan(usize, dirs.items.len)];

        const basename = try generatePathName(corpus_allocator, random, source_index, ".zig");
        const path = try joinRelativePath(corpus_allocator, dir.path, basename);

        try tmp_dir.dir.writeFile(io, .{
            .sub_path = path,
            .data = invalid_zig,
        });

        source_paths.appendAssumeCapacity(path);
    }

    // ── Distribute Interacting Gitignore Rules ──
    //
    // Every directory independently receives a `.gitignore` with one-in-three
    // probability. The repository root follows the same rule as every descendant,
    // so a corpus may have root rules, nested rules only, or no ignore file.

    var gitignore_files: std.ArrayList(GeneratedGitignoreFile) = .empty;
    defer gitignore_files.deinit(allocator);

    try gitignore_files.ensureTotalCapacity(allocator, dirs.items.len);

    for (dirs.items) |dir| {
        const writes_gitignore = random.uintLessThan(u8, 3) == 0;
        if (!writes_gitignore) {
            continue;
        }

        // Generate every rule in this directory's own coordinate system.
        const contents = try generateGitignoreContents(
            corpus_allocator,
            random,
            dir.path,
            source_paths.items,
        );

        const path = try joinRelativePath(corpus_allocator, dir.path, ".gitignore");

        try tmp_dir.dir.writeFile(io, .{
            .sub_path = path,
            .data = contents,
        });

        // Retain the exact generated file so a mismatch can print every
        // `.gitignore` on the failing source's ancestor chain.
        gitignore_files.appendAssumeCapacity(.{
            .path = path,
            .contents = contents,
        });
    }

    // ── Collect Git's Decisions In One Process ──
    //
    // Source paths determine expected diagnostics. Non-root directory paths
    // reveal whether inherited rules exclude a requested scan root. One batched
    // invocation avoids starting Git separately for every path or scan.

    var git_arguments: std.ArrayList([]const u8) = .empty;
    defer git_arguments.deinit(allocator);

    try git_arguments.ensureTotalCapacity(
        allocator,
        source_paths.items.len + dirs.items.len + 6,
    );

    git_arguments.appendSliceAssumeCapacity(&.{
        "git",
        "-c",
        "core.quotePath=false",
        "check-ignore",
        "--no-index",
        "--",
    });

    for (source_paths.items) |source_path| {
        git_arguments.appendAssumeCapacity(source_path);
    }

    // Rules inside the repository cannot ignore the repository root itself, so
    // only descendants need an explicit Git decision.
    for (dirs.items[1..]) |dir| {
        git_arguments.appendAssumeCapacity(dir.path);
    }

    const check_result = try std.process.run(corpus_allocator, io, .{
        .argv = git_arguments.items,
        .cwd = .{ .dir = tmp_dir.dir },
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(4096),
    });

    // Exit 1 means none of the generated paths are ignored, which is a valid
    // corpus outcome.
    if (check_result.term != .exited or check_result.term.exited > 1) {
        std.debug.print("git check-ignore failed: {s}\n", .{check_result.stderr});

        return error.GitCheckIgnoreFailed;
    }

    var git_ignored_paths = std.StringHashMap(void).init(allocator);
    defer git_ignored_paths.deinit();

    var output_paths = std.mem.splitScalar(u8, check_result.stdout, '\n');

    while (output_paths.next()) |output_path| {
        if (output_path.len != 0) {
            try git_ignored_paths.put(output_path, {});
        }
    }

    // ── Compare Every Directory Scan ──
    //
    // Each scan sees only source files beneath its requested root. An ignored
    // scan root produces no diagnostics. Otherwise, ignored sources remain absent
    // and each included source produces its deliberate parse diagnostic.

    var linter_diagnostic_paths = std.StringHashMap(void).init(allocator);
    defer linter_diagnostic_paths.deinit();

    for (dirs.items) |dir| {
        linter_diagnostic_paths.clearRetainingCapacity();

        const scan_sub_path = if (dir.path.len == 0) "." else dir.path;

        const scan_path = try tmp_dir.dir.realPathFileAlloc(io, scan_sub_path, allocator);
        defer allocator.free(scan_path);

        // Git-root-relative reporting keeps diagnostics from every scan in the
        // same coordinate system as `git check-ignore` output.
        var report = try api.lint(api.Allocators.same(allocator), io, scan_path, .{
            .diagnostics_count_max = @intCast(source_paths.items.len),
            .report_path_format = .git_root_relative,
        });
        defer report.deinit(allocator);

        for (report.diagnostics) |diagnostic| {
            try linter_diagnostic_paths.put(diagnostic.path, {});
        }

        // The repository root uses an empty path and was not sent to Git because
        // rules inside a repository cannot exclude that repository itself.
        const scan_root_is_repository_root = dir.path.len == 0;

        const git_excludes_scan_root = if (scan_root_is_repository_root)
            false
        else
            git_ignored_paths.contains(dir.path);

        var expected_diagnostics_count: usize = 0;

        for (source_paths.items, 0..) |source_path, source_index| {
            if (pathRelativeToDir(source_path, dir.path) == null) {
                // A scan cannot report sources outside its own subtree.
                continue;
            }

            // Excluding the requested root prunes its complete subtree. For an
            // included root, Git's source-path decision is authoritative.
            const git_excludes_source = git_excludes_scan_root or
                git_ignored_paths.contains(source_path);

            const linter_excludes_source = !linter_diagnostic_paths.contains(source_path);

            if (!git_excludes_source) {
                expected_diagnostics_count += 1;
            }

            if (linter_excludes_source != git_excludes_source) {
                printIgnoreDifferentialMismatch(
                    scan_sub_path,
                    source_index,
                    source_path,
                    linter_excludes_source,
                    git_excludes_source,
                    git_excludes_scan_root,
                    gitignore_files.items,
                );

                return error.IgnoreDifferentialMismatch;
            }
        }

        if (expected_diagnostics_count != report.diagnostics.len) {
            std.debug.print("ignore differential count mismatch: scan=\"{f}\"\n", .{
                std.zig.fmtString(scan_sub_path),
            });
        }

        try std.testing.expectEqual(expected_diagnostics_count, report.diagnostics.len);
    }
}

const GeneratedGitignoreFile = struct {
    path: []const u8,
    contents: []const u8,
};

fn printIgnoreDifferentialMismatch(
    scan_path: []const u8,
    source_index: usize,
    source_path: []const u8,
    linter_excludes_source: bool,
    git_excludes_source: bool,
    git_excludes_scan_root: bool,
    gitignore_files: []const GeneratedGitignoreFile,
) void {
    std.debug.print(
        "ignore differential mismatch: scan=\"{f}\" source={d} path=\"{f}\"\n" ++
            "linter excludes: {} git excludes: {} scan root ignored: {}\n" ++
            "source ancestor .gitignore files:\n",
        .{
            std.zig.fmtString(scan_path),
            source_index,
            std.zig.fmtString(source_path),
            linter_excludes_source,
            git_excludes_source,
            git_excludes_scan_root,
        },
    );

    for (gitignore_files) |gitignore_file| {
        const gitignore_dir_path = std.fs.path.dirname(gitignore_file.path) orelse "";
        if (pathRelativeToDir(source_path, gitignore_dir_path) == null) {
            continue;
        }

        std.debug.print(
            "  \"{f}\": \"{f}\"\n",
            .{
                std.zig.fmtString(gitignore_file.path),
                std.zig.fmtString(gitignore_file.contents),
            },
        );
    }
}

fn generateGitignoreContents(
    allocator: std.mem.Allocator,
    random: std.Random,
    gitignore_dir_path: []const u8,
    source_paths: []const []const u8,
) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();

    const rules_count = random.intRangeAtMost(usize, 3, 12);

    // Source-derived rules exercise matching against real generated paths. A
    // directory with no source beneath it still exercises pattern parsing.
    for (0..rules_count) |_| {
        if (randomSourcePathWithinDir(
            random,
            gitignore_dir_path,
            source_paths,
        )) |relative_source_path| {
            try appendGeneratedRule(&output.writer, random, relative_source_path);
        } else {
            try appendRandomPatternLine(&output.writer, random);
        }
    }

    return output.toOwnedSlice();
}

fn appendGeneratedRule(
    writer: *std.Io.Writer,
    random: std.Random,
    source_path: []const u8,
) !void {
    const basename = std.fs.path.basename(source_path);

    const RuleKind = enum {
        exact_basename,
        exact_path,
        anchored_path,
        recursive_basename,
        single_byte_wildcard,
        alpha_class,
        posix_alpha_class,
        dir,
        negation,
        ignore_then_include,
        include_then_ignore,
        comment,
        trailing_spaces,
        unclosed_class,
        dangling_escape,
        recursive_star_run,
        escaped_separators,
        arbitrary_bytes,
    };

    switch (random.enumValue(RuleKind)) {
        .exact_basename => try appendEscapedLiteral(writer, basename),
        .exact_path => try appendEscapedLiteral(writer, source_path),
        .anchored_path => {
            try writer.writeByte('/');
            try appendEscapedLiteral(writer, source_path);
        },

        .recursive_basename => {
            try writer.writeAll("**/");
            try appendEscapedLiteral(writer, basename);
        },

        .single_byte_wildcard => {
            try writer.writeByte('?');
            try appendEscapedLiteral(writer, basename[1..]);
        },

        .alpha_class => {
            try writer.writeAll("[a-z]");
            try appendEscapedLiteral(writer, basename[1..]);
        },

        .posix_alpha_class => {
            try writer.writeAll("[[:alpha:]]");
            try appendEscapedLiteral(writer, basename[1..]);
        },
        .dir => if (std.mem.indexOfScalar(u8, source_path, '/')) |slash| {
            try appendEscapedLiteral(writer, source_path[0..slash]);
            try writer.writeByte('/');
        } else {
            return appendRandomPatternLine(writer, random);
        },
        .negation => {
            try writer.writeByte('!');
            try appendEscapedLiteral(writer, source_path);
        },

        .ignore_then_include => {
            try appendEscapedLiteral(writer, source_path);
            try writer.writeAll("\n!");
            try appendEscapedLiteral(writer, source_path);
        },

        .include_then_ignore => {
            try writer.writeByte('!');
            try appendEscapedLiteral(writer, source_path);
            try writer.writeByte('\n');
            try appendEscapedLiteral(writer, source_path);
        },

        .comment => {
            try writer.writeByte('#');
            try appendRandomPatternBytes(writer, random, random.intRangeAtMost(usize, 1, 24));
        },

        .trailing_spaces => {
            try appendEscapedLiteral(writer, basename);
            try writer.splatByteAll(' ', random.intRangeAtMost(usize, 1, 4));
        },

        .unclosed_class => {
            try writer.writeByte('[');
            try appendRandomPatternBytes(writer, random, random.uintLessThan(usize, 10));
        },

        .dangling_escape => {
            try appendEscapedLiteral(writer, basename);
            try writer.writeByte('\\');
        },

        .recursive_star_run => {
            try writer.splatByteAll('*', random.intRangeAtMost(usize, 2, 5));
            try writer.writeByte('/');
            try appendEscapedLiteral(writer, basename);
        },

        .escaped_separators => for (source_path) |byte| {
            if (byte == '/') {
                try writer.writeAll("\\/");
            } else {
                try appendEscapedLiteral(writer, &.{byte});
            }
        },
        .arbitrary_bytes => return appendRandomPatternLine(writer, random),
    }

    try writer.writeByte('\n');
}

fn appendRandomPatternLine(writer: *std.Io.Writer, random: std.Random) !void {
    try appendRandomPatternBytes(writer, random, random.uintLessThan(usize, 41));
    try writer.writeByte('\n');
}

fn appendRandomPatternBytes(writer: *std.Io.Writer, random: std.Random, count: usize) !void {
    const alphabet = "abcXYZ019*?[]!#\\/-_ .:\t\r\x80\xfe\xff";

    for (0..count) |_| {
        try writer.writeByte(alphabet[random.uintLessThan(usize, alphabet.len)]);
    }
}

fn appendEscapedLiteral(writer: *std.Io.Writer, bytes: []const u8) !void {
    for (bytes, 0..) |byte, byte_index| {
        const requires_escape = byte == '\\' or
            byte == '*' or
            byte == '?' or
            byte == '[' or
            byte == ' ' or
            (byte_index == 0 and (byte == '!' or byte == '#'));

        if (requires_escape) {
            try writer.writeByte('\\');
        }

        try writer.writeByte(byte);
    }
}

fn randomSourcePathWithinDir(
    random: std.Random,
    dir_path: []const u8,
    source_paths: []const []const u8,
) ?[]const u8 {
    const start = random.uintLessThan(usize, source_paths.len);

    for (0..source_paths.len) |offset| {
        const source_path = source_paths[(start + offset) % source_paths.len];
        if (pathRelativeToDir(source_path, dir_path)) |relative_path| {
            return relative_path;
        }
    }

    return null;
}

fn pathRelativeToDir(path: []const u8, dir_path: []const u8) ?[]const u8 {
    if (dir_path.len == 0) {
        return path;
    }

    if (path.len <= dir_path.len or
        !std.mem.startsWith(u8, path, dir_path) or
        path[dir_path.len] != '/')
    {
        return null;
    }

    return path[dir_path.len + 1 ..];
}

fn generatePathName(
    allocator: std.mem.Allocator,
    random: std.Random,
    index: usize,
    suffix: []const u8,
) ![]const u8 {
    // Concentrate generation on ordinary bytes, Gitignore metacharacters,
    // whitespace, and invalid UTF-8 rather than making interesting bytes rare.
    const alphabet = "abcXYZ019-_ .[]!#*?\x80\xfe\xff";

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();

    const random_bytes_count = random.intRangeAtMost(usize, 1, 12);

    for (0..random_bytes_count) |_| {
        try output.writer.writeByte(alphabet[random.uintLessThan(usize, alphabet.len)]);
    }

    // The random bytes lead the basename so root-sensitive pattern behavior is
    // not fixed. The ordinal suffix only prevents generated path collisions.
    try output.writer.print("-{d}{s}", .{ index, suffix });

    return output.toOwnedSlice();
}

fn joinRelativePath(
    allocator: std.mem.Allocator,
    dir: []const u8,
    basename: []const u8,
) ![]const u8 {
    if (dir.len == 0) {
        return allocator.dupe(u8, basename);
    }

    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, basename });
}
