//! Orchestrates one file or directory lint operation.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Ast = std.zig.Ast;

const AstTokenCache = @import("AstTokenCache.zig");
const BuiltinRules = @import("BuiltinRules.zig");
const Fixes = @import("Fixes.zig");
const Gitignore = @import("Gitignore.zig");
const LintCache = @import("LintCache.zig");
const LintContext = @import("LintContext.zig");
const Plugin = @import("Plugin.zig");
const Report = @import("LintReport.zig");

/// `Allocators` separates temporary linting memory from returned report memory.
///
/// The linter releases every reference to `scratch` before returning and never
/// resets or deinitializes either caller-owned allocator. Every slice in the
/// returned report uses `report` and remains valid until the caller deinitializes
/// the report or releases that allocator. The linter reserves bounded diagnostic
/// storage from `report` before parsing source and uses coarse allocations for
/// complete fixed-file output.
pub const Allocators = struct {
    scratch: Allocator,
    report: Allocator,

    /// `same` gives scratch data and the returned report the same allocation
    /// lifetime. Resetting an arena passed here invalidates the report.
    pub fn same(allocator: Allocator) Allocators {
        return .{ .scratch = allocator, .report = allocator };
    }
};

/// `ReportPathFormat` selects the coordinate system for diagnostic and fixed-file
/// paths without changing traversal, ignore matching, or cache identity.
pub const ReportPathFormat = enum {
    /// Paths are relative to the process working directory when `lint` begins.
    cwd_relative,

    /// Paths are relative to the nearest enclosing `.git` directory or file.
    /// When no Git root exists, paths are relative to the process working
    /// directory.
    git_root_relative,

    /// Paths are canonical absolute filesystem paths.
    absolute,
};

const file_size_max_default = 1 * 1024 * 1024; // 1 MiB
const gitignore_file_size_max = 120 * 1024; // 120 KiB

/// `LintOptions` controls one file or directory lint operation.
pub const LintOptions = struct {
    /// Exact directory basenames excluded by a default directory scan. Callers
    /// may concatenate this array with their own names before invoking `lint`.
    pub const excluded_dir_names_default = [_][]const u8{
        ".git",
        ".hg",
        ".svn",
        ".jj",
        ".zig-cache",
        "zig-cache",
        "zig-out",
        "zig-pkg",
        ".cache",
        "node_modules",
        ".venv",
        "__fixtures__",
    };

    /// `file_size_max` is an inclusive source-byte limit. An oversized explicit
    /// file fails with `error.FileTooLarge`; an oversized file found beneath a
    /// directory produces a diagnostic and does not stop the scan.
    file_size_max: u32 = file_size_max_default,

    /// `diagnostics_count_max` limits retained findings to the deterministic
    /// top-K report order. The report still counts every omitted finding.
    diagnostics_count_max: u32 = 500,

    /// `diagnostic_text_size_max` limits the combined path, rule name, message,
    /// help, note, and source-line bytes retained for one finding. A finding
    /// that exceeds this inclusive maximum is counted but omitted completely.
    diagnostic_text_size_max: u32 = 1024,

    /// The Zig release whose language and standard-library contracts rules
    /// should target. Built-in and plugin rules receive this exact value. It
    /// does not select the linter's parser or formatter, both of which remain
    /// those of the toolchain that built the linter.
    ///
    /// The default targets the toolchain that built the linter. Prerelease and
    /// build metadata are preserved and participate in lint-cache identity;
    /// their backing slices must remain valid until `lint` returns.
    target_zig_version: std.SemanticVersion = builtin.zig_version,

    /// When true, the report contains canonical text for every file changed by
    /// compatible rule edits or Zig formatting. The linter never writes source
    /// files, and diagnostics continue to describe the input text.
    apply_fixes: bool = false,

    /// `report_path_format` applies uniformly to diagnostics and fixed files.
    report_path_format: ReportPathFormat = .cwd_relative,

    /// Exact directory basenames excluded at every descendant depth. Supplying
    /// this field replaces the defaults rather than extending them; pass an
    /// empty slice to disable directory-name exclusions. These exclusions are
    /// independent of `.gitignore` and cannot be overridden by its negations.
    excluded_dir_names: []const []const u8 = &excluded_dir_names_default,

    /// When true, directory scans inside a Git repository apply regular,
    /// non-symlinked `.gitignore` files from the nearest `.git` boundary through
    /// the scan root and included descendants. An ignored scan root produces an
    /// empty report. Without a Git boundary, the scan root starts ignore lookup.
    respect_gitignore: bool = true,

    /// Registered native plugins selected explicitly for this operation. An
    /// empty slice runs no plugins. The registry serializes plugin callbacks,
    /// prevents unload while they run, and rejects absent or duplicate handles.
    plugins: []const Plugin.Handle = &.{},

    /// Null disables persistent caching. Explicit file inputs never consult or
    /// update persistent cache state.
    cache: ?Cache = null,

    /// `Cache` configures persistent storage used to accelerate linting.
    pub const Cache = struct {
        /// `base_dir` names a trusted writable directory dedicated to private
        /// linter cache state. The linter manages everything beneath it and may
        /// create multiple files or subdirectories, replace formats, or discard
        /// entries without notice. Callers must treat its contents as private
        /// and unstable and must not depend on their names or representation.
        ///
        /// Cache state is scoped to this directory; the linter never consults a
        /// different base directory. Selecting a fresh base directory therefore
        /// guarantees a full cache miss. The linter does not consult `HOME` or
        /// XDG directories to choose one implicitly.
        base_dir: []const u8,
    };
};

/// `lint` checks one filesystem path and returns caller-owned immutable output.
///
/// A regular file is always checked regardless of its extension or cache state;
/// directory-name exclusions and `.gitignore` files do not apply to it. A
/// directory is traversed recursively and checks regular `.zig` files. By
/// default it applies regular, non-symlinked `.gitignore` files of at
/// most 120 KiB from the nearest Git boundary through the scan root and included
/// descendants. Without a Git boundary, ignore lookup begins at the scan root.
/// It also prunes the exact basenames in
/// `LintOptions.excluded_dir_names_default`. Directory-name exclusions
/// are applied independently before `.gitignore` rules. Ignore matching does not
/// inspect Git's index, so a tracked path that matches a pattern is excluded
/// from a directory scan. During a directory walk, a symlink is linted only
/// when its target is a regular `.zig` file.
///
/// Traversal and source-read failures fail the operation. Cache corruption and
/// cache-specific I/O failures become misses; a failed scan does not publish
/// partial cache state.
pub fn lint(
    allocators: Allocators,
    io: std.Io,
    path: []const u8,
    options: LintOptions,
) !Report.LintReport {
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (stat.kind != .file and stat.kind != .directory) {
        return error.UnsupportedFileType;
    }

    if (stat.kind == .file and stat.size > options.file_size_max) {
        return error.FileTooLarge;
    }

    var builder = try Report.Builder.init(allocators.scratch, allocators.report, .{
        .diagnostics_count_max = options.diagnostics_count_max,
        .diagnostic_text_size_max = options.diagnostic_text_size_max,
    });
    defer builder.deinit();

    const absolute_input_path = try std.Io.Dir.cwd().realPathFileAlloc(
        io,
        path,
        allocators.scratch,
    );
    defer allocators.scratch.free(absolute_input_path);

    const git_root_is_needed = options.report_path_format == .git_root_relative or
        (stat.kind == .directory and options.respect_gitignore);

    const git_root = if (git_root_is_needed)
        try findGitRoot(io, absolute_input_path, stat.kind)
    else
        null;

    var report_paths = try ReportPaths.init(
        allocators.scratch,
        io,
        absolute_input_path,
        options.report_path_format,
        git_root,
    );
    defer report_paths.deinit(allocators.scratch);

    switch (stat.kind) {
        .file => try lintFilePath(
            allocators.scratch,
            io,
            absolute_input_path,
            report_paths.input_path,
            options,
            &builder,
        ),
        .directory => try lintDir(
            allocators.scratch,
            io,
            absolute_input_path,
            if (options.respect_gitignore) git_root else null,
            options,
            &report_paths,
            &builder,
        ),
        else => unreachable,
    }

    return builder.finish();
}

const ReportPaths = struct {
    input_path: []u8,
    entry_path_buffer: std.ArrayList(u8) = .empty,

    fn init(
        allocator: Allocator,
        io: std.Io,
        absolute_input_path: []const u8,
        format: ReportPathFormat,
        git_root: ?[]const u8,
    ) !ReportPaths {
        switch (format) {
            .absolute => return .{
                .input_path = try allocator.dupe(u8, absolute_input_path),
            },
            .git_root_relative => if (git_root) |root| {
                return .{
                    .input_path = try allocator.dupe(
                        u8,
                        relativePathFromAncestor(absolute_input_path, root),
                    ),
                };
            },
            .cwd_relative => {},
        }

        const cwd_path = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
        defer allocator.free(cwd_path);

        const input_path = try std.fs.path.relative(
            allocator,
            cwd_path,
            null,
            cwd_path,
            absolute_input_path,
        );

        return .{ .input_path = input_path };
    }

    fn deinit(paths: *ReportPaths, allocator: Allocator) void {
        paths.entry_path_buffer.deinit(allocator);
        allocator.free(paths.input_path);

        paths.* = undefined;
    }

    fn entryPath(
        paths: *ReportPaths,
        allocator: Allocator,
        scan_relative_path: []const u8,
    ) Allocator.Error![]const u8 {
        const input_path = paths.input_path;
        if (input_path.len == 0) {
            return scan_relative_path;
        }

        paths.entry_path_buffer.clearRetainingCapacity();

        const separator_is_needed = !std.fs.path.isSep(input_path[input_path.len - 1]);

        try paths.entry_path_buffer.ensureTotalCapacity(
            allocator,
            input_path.len +
                @intFromBool(separator_is_needed) +
                scan_relative_path.len,
        );

        paths.entry_path_buffer.appendSliceAssumeCapacity(input_path);

        if (separator_is_needed) {
            paths.entry_path_buffer.appendAssumeCapacity(std.fs.path.sep);
        }

        paths.entry_path_buffer.appendSliceAssumeCapacity(scan_relative_path);

        return paths.entry_path_buffer.items;
    }
};

fn findGitRoot(
    io: std.Io,
    absolute_input_path: []const u8,
    input_kind: std.Io.File.Kind,
) !?[]const u8 {
    var ancestor = if (input_kind == .directory)
        absolute_input_path
    else
        std.fs.path.dirname(absolute_input_path) orelse {
            return null;
        };

    var marker_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    while (true) {
        const marker_path = if (std.mem.eql(u8, ancestor, std.fs.path.sep_str))
            try std.fmt.bufPrint(&marker_path_buffer, "{s}.git", .{ancestor})
        else
            try std.fmt.bufPrint(&marker_path_buffer, "{s}" ++ std.fs.path.sep_str ++ ".git", .{ancestor});

        const marker_stat = std.Io.Dir.cwd().statFile(
            io,
            marker_path,
            .{ .follow_symlinks = false },
        ) catch |err|
            switch (err) {
                error.FileNotFound => null,
                else => return err,
            };

        if (marker_stat) |stat| {
            if (stat.kind == .directory or stat.kind == .file) {
                // A primary worktree normally has a `.git` directory. Linked
                // worktrees and modern submodules use a `.git` file that points
                // to their Git directory instead.
                return ancestor;
            }
        }

        const parent = std.fs.path.dirname(ancestor) orelse {
            return null;
        };

        if (parent.len == ancestor.len) {
            return null;
        }

        ancestor = parent;
    }
}

fn relativePathFromAncestor(path: []const u8, ancestor: []const u8) []const u8 {
    assert(ancestor.len != 0);
    assert(std.mem.startsWith(u8, path, ancestor));

    if (path.len == ancestor.len) {
        return "";
    }

    if (std.fs.path.isSep(ancestor[ancestor.len - 1])) {
        return path[ancestor.len..];
    }

    assert(std.fs.path.isSep(path[ancestor.len]));

    return path[ancestor.len + 1 ..];
}

// ─── File Linting ────────────────────────────────────────────────────────────

const FileResult = struct {
    lint_clean: bool,
    zig_fmt_clean: bool,
};

fn lintFilePath(
    scratch_allocator: Allocator,
    io: std.Io,
    path: []const u8,
    report_path: []const u8,
    options: LintOptions,
    builder: *Report.Builder,
) !void {
    const opened = try openRegularFile(io, std.Io.Dir.cwd(), path, .{}) orelse {
        return error.UnsupportedFileType;
    };

    var source_file = opened.file;
    defer source_file.close(io);

    if (opened.stat.size > options.file_size_max) {
        return error.FileTooLarge;
    }

    var file_arena = std.heap.ArenaAllocator.init(scratch_allocator);
    defer file_arena.deinit();

    const file_allocator = file_arena.allocator();
    const buffer_size = @as(usize, options.file_size_max) + 1;
    const buffer = try file_allocator.alloc(u8, buffer_size);

    var reader = source_file.reader(io, &.{});

    const read_size = reader.interface.readSliceShort(buffer) catch |err|
        switch (err) {
            error.ReadFailed => return reader.err orelse {
                unreachable;
            },
        };

    if (read_size == buffer.len) {
        return error.FileTooLarge;
    }

    buffer[read_size] = 0;

    const source = buffer[0..read_size :0];

    var ast = try Ast.parse(file_allocator, source, .zig);
    defer ast.deinit(file_allocator);

    _ = try lintAst(
        file_allocator,
        io,
        builder,
        report_path,
        ast,
        options,
        .{},
    );
}

fn lintAst(
    file_allocator: Allocator,
    io: std.Io,
    builder: *Report.Builder,
    path: []const u8,
    ast: Ast,
    options: LintOptions,
    previously_verified: LintCache.Verified,
) !FileResult {
    if (ast.errors.len != 0) {
        if (!previously_verified.lint_clean) {
            const parse_error = ast.errors[0];
            const token = parse_error.token + @intFromBool(parse_error.token_is_prev);
            const location = ast.tokenLocation(0, token);
            const start_offset = ast.tokenStart(token);

            var parse_message: std.Io.Writer.Allocating = .init(file_allocator);
            defer parse_message.deinit();

            ast.renderError(parse_error, &parse_message.writer) catch {
                return error.OutOfMemory;
            };

            const message = try parse_message.toOwnedSlice();
            defer file_allocator.free(message);

            builder.addDiagnostic(.{
                .path = path,
                .rule_name = "parse",
                .message = message,
                .help = "fix this syntax error before running the linter",
                .location = .{
                    .line = location.line + 1,
                    .column = location.column + 1,
                    .range = .{
                        .start_offset = start_offset,
                        .end_offset = @intCast(start_offset + ast.tokenSlice(token).len),
                    },
                },
                .source_line = ast.source[location.line_start..location.line_end],
            });
        }

        return .{
            .lint_clean = previously_verified.lint_clean,
            .zig_fmt_clean = previously_verified.zig_fmt_clean,
        };
    }

    var fixes = Fixes.init(file_allocator);
    defer fixes.deinit();

    fixes.reset(ast.source);

    const total_diagnostic_count_before = builder.totalDiagnosticCount();

    if (!previously_verified.lint_clean) {
        var context = try LintContext.init(
            file_allocator,
            path,
            ast,
            options.target_zig_version,
            builder,
            if (options.apply_fixes)
                &fixes
            else
                null,
        );
        defer context.deinit();

        for (BuiltinRules.all) |definition| {
            try context.runRule(definition);
        }

        try Plugin.runConfigured(io, options.plugins, &context);
    }

    const lint_clean = previously_verified.lint_clean or
        builder.totalDiagnosticCount() == total_diagnostic_count_before;

    var zig_fmt_clean = previously_verified.zig_fmt_clean;

    if (options.apply_fixes and !zig_fmt_clean) {
        const applied = try fixes.applyAndFormat(ast, options.file_size_max);
        defer if (applied.updated_source) |updated_source| {
            file_allocator.free(updated_source);
        };

        if (applied.updated_source) |updated_source| {
            try builder.addFixedFile(
                path,
                updated_source,
                applied.applied_fix_count,
                applied.skipped_fix_count,
            );
        } else {
            builder.addFixCounts(applied.applied_fix_count, applied.skipped_fix_count);

            zig_fmt_clean = true;
        }
    }

    return .{
        .lint_clean = lint_clean,
        .zig_fmt_clean = zig_fmt_clean,
    };
}

// ─── Directory Linting ───────────────────────────────────────────────────────

fn lintDir(
    scratch_allocator: Allocator,
    io: std.Io,
    absolute_scan_path: []const u8,
    git_root: ?[]const u8,
    options: LintOptions,
    report_paths: *ReportPaths,
    builder: *Report.Builder,
) !void {
    // ── Prepare Scan ──

    var dir_handle = try std.Io.Dir.cwd().openDir(io, absolute_scan_path, .{ .iterate = true });
    defer dir_handle.close(io);

    var lint_cache: ?LintCache = if (options.cache) |configuration|
        LintCache.init(
            scratch_allocator,
            io,
            dir_handle,
            configuration.base_dir,
            options.target_zig_version,
        ) catch null
    else
        null;
    defer if (lint_cache) |*cache| {
        cache.deinit(scratch_allocator);
    };

    // Open the syntax cache before traversal so every selected path can mark its
    // entry seen, including files skipped by lint-clean facts. A completed scan
    // can then prune deleted, renamed, and excluded entries. This deliberately
    // loads the syntax pack on a fully lint-cached scan; mixed scans need it
    // anyway, and avoiding persistent stale entries keeps the cache bounded.
    var syntax_cache: ?AstTokenCache = if (options.cache) |configuration|
        AstTokenCache.init(
            scratch_allocator,
            io,
            dir_handle,
            configuration.base_dir,
        ) catch null
    else
        null;
    defer if (syntax_cache) |*cache| {
        cache.deinit(scratch_allocator);
    };

    // Plugin artifacts are not part of lint-cache identity. A plugin scan still
    // opens and commits the index so a successful complete scan discards old
    // unkeyed facts, but those facts never suppress or describe plugin work.
    const lint_cache_facts_are_usable = options.plugins.len == 0;

    // ── Resolve Inherited Exclusions ──

    var gitignore: Gitignore = .{};
    defer gitignore.deinit(scratch_allocator);

    const gitignore_scan_dir_path = if (git_root) |root|
        relativePathFromAncestor(absolute_scan_path, root)
    else
        "";

    if (git_root) |root| {
        if (gitignore_scan_dir_path.len != 0) {
            const scan_root_is_ignored = try loadAncestorGitignores(
                scratch_allocator,
                io,
                root,
                gitignore_scan_dir_path,
                &gitignore,
            );

            if (scan_root_is_ignored) {
                // The complete selected set is empty. Commit it so files from
                // an earlier included scan cannot survive this generation.
                commitDirectoryCaches(
                    scratch_allocator,
                    io,
                    &lint_cache,
                    &syntax_cache,
                );

                return;
            }
        }
    }

    const root_has_gitignore = options.respect_gitignore and try loadGitignoreFile(
        scratch_allocator,
        io,
        &gitignore,
        dir_handle,
        ".gitignore",
        gitignore_scan_dir_path,
    );

    var dirs_have_gitignore: std.ArrayList(bool) = .empty;
    defer dirs_have_gitignore.deinit(scratch_allocator);

    try dirs_have_gitignore.append(scratch_allocator, root_has_gitignore);

    var gitignore_entry_path_buffer: std.ArrayList(u8) = .empty;
    defer gitignore_entry_path_buffer.deinit(scratch_allocator);

    var walker = try dir_handle.walkSelectively(scratch_allocator);
    defer walker.deinit();

    // Allocate only on the first miss. An entirely verified scan needs neither
    // this configurable read buffer nor its ReleaseSafe allocation/free poison fills.
    var file_buffer: ?[]u8 = null;
    defer if (file_buffer) |buffer| {
        scratch_allocator.free(buffer);
    };

    var file_arena = std.heap.ArenaAllocator.init(scratch_allocator);
    defer file_arena.deinit();

    // ── Check Files ──

    while (try walker.next(io)) |entry| {
        // ── Restore The Entry's Parent Scope ──

        while (dirs_have_gitignore.items.len > entry.depth()) {
            const dir_had_gitignore = dirs_have_gitignore.pop() orelse {
                unreachable;
            };

            if (dir_had_gitignore) {
                gitignore.pop();
            }
        }

        assert(dirs_have_gitignore.items.len == entry.depth());

        // ── Resolve Unknown Filesystem Types ──

        var entry_kind = entry.kind;

        if (entry_kind == .unknown) {
            entry_kind = (try entry.dir.statFile(
                io,
                entry.basename,
                .{ .follow_symlinks = false },
            )).kind;
        }

        // ── Apply Directory And Repository Exclusions Before I/O ──

        if (entry_kind == .directory and dirNameIsExcluded(
            options.excluded_dir_names,
            entry.basename,
        )) {
            continue;
        }

        const gitignore_entry_path = if (options.respect_gitignore and gitignore_scan_dir_path.len != 0) path: {
            gitignore_entry_path_buffer.clearRetainingCapacity();

            try gitignore_entry_path_buffer.ensureTotalCapacity(
                scratch_allocator,
                gitignore_scan_dir_path.len + 1 + entry.path.len,
            );

            gitignore_entry_path_buffer.appendSliceAssumeCapacity(gitignore_scan_dir_path);
            gitignore_entry_path_buffer.appendAssumeCapacity('/');
            gitignore_entry_path_buffer.appendSliceAssumeCapacity(entry.path);

            break :path gitignore_entry_path_buffer.items;
        } else entry.path;

        if (options.respect_gitignore and gitignore.pathIsIgnored(
            gitignore_entry_path,
            entry_kind == .directory,
        )) {
            continue;
        }

        // ── Enter Included Directory Scope ──

        if (entry_kind == .directory) {
            var gitignore_path_buffer: [std.Io.Dir.max_name_bytes + "/.gitignore".len]u8 = undefined;
            const gitignore_path = try std.fmt.bufPrint(
                &gitignore_path_buffer,
                "{s}/.gitignore",
                .{entry.basename},
            );

            try dirs_have_gitignore.ensureUnusedCapacity(scratch_allocator, 1);

            const child_has_gitignore = options.respect_gitignore and try loadGitignoreFile(
                scratch_allocator,
                io,
                &gitignore,
                entry.dir,
                gitignore_path,
                gitignore_entry_path,
            );

            var dir_entry = entry;

            dir_entry.kind = .directory;

            walker.enter(io, dir_entry) catch |err| {
                if (child_has_gitignore) {
                    gitignore.pop();
                }

                return err;
            };

            dirs_have_gitignore.appendAssumeCapacity(child_has_gitignore);

            continue;
        }

        // ── Select Eligible Zig Files ──

        if (!std.mem.endsWith(u8, entry.basename, ".zig")) {
            continue;
        }

        switch (entry_kind) {
            .file, .sym_link => {},
            else => continue,
        }

        const opened = try openRegularFile(io, entry.dir, entry.basename, .{}) orelse {
            continue;
        };

        var source_file = opened.file;
        defer source_file.close(io);

        const cached_syntax: ?AstTokenCache.Entry = if (syntax_cache) |*cache|
            cache.lookup(entry.path, opened.stat, options.file_size_max)
        else
            null;

        const previously_verified = if (lint_cache_facts_are_usable)
            if (lint_cache) |*cache|
                cache.lookup(entry.path, opened.stat, options.file_size_max) orelse
                    LintCache.Verified{}
            else
                LintCache.Verified{}
        else
            LintCache.Verified{};

        if (previously_verified.lint_clean and
            (!options.apply_fixes or previously_verified.zig_fmt_clean))
        {
            continue;
        }

        const report_path = try report_paths.entryPath(scratch_allocator, entry.path);

        var linted_source_size: usize = undefined;
        var result: FileResult = undefined;
        if (cached_syntax) |token_entry| {
            _ = file_arena.reset(.retain_capacity);

            const file_allocator = file_arena.allocator();

            var ast = try token_entry.parse(file_allocator);
            defer ast.deinit(file_allocator);

            result = try lintAst(
                file_allocator,
                io,
                builder,
                report_path,
                ast,
                options,
                previously_verified,
            );
            linted_source_size = token_entry.source.len;
        } else {
            const buffer = file_buffer orelse buffer: {
                const buffer_size = @as(usize, options.file_size_max) + 1;
                const allocated = try scratch_allocator.alloc(u8, buffer_size);

                file_buffer = allocated;

                break :buffer allocated;
            };

            var reader = source_file.reader(io, &.{});

            const read_size = reader.interface.readSliceShort(buffer) catch |err|
                switch (err) {
                    error.ReadFailed => return reader.err orelse {
                        unreachable;
                    },
                };

            if (read_size == buffer.len) {
                const message = try std.fmt.allocPrint(
                    scratch_allocator,
                    "file exceeds the {d}-byte size limit",
                    .{options.file_size_max},
                );
                defer scratch_allocator.free(message);

                builder.addDiagnostic(.{
                    .path = report_path,
                    .rule_name = "file_size",
                    .message = message,
                    .help = "split the file or increase `file_size_max`",
                    .location = .{
                        .line = 1,
                        .column = 1,
                        .range = .{ .start_offset = 0, .end_offset = 0 },
                    },
                });

                continue;
            }

            buffer[read_size] = 0;

            const source = buffer[0..read_size :0];

            _ = file_arena.reset(.retain_capacity);

            const file_allocator = file_arena.allocator();

            var ast = try Ast.parse(file_allocator, source, .zig);
            defer ast.deinit(file_allocator);

            if (syntax_cache) |*cache| {
                if (opened.stat.size == read_size) {
                    cache.put(
                        scratch_allocator,
                        entry.path,
                        opened.stat,
                        source,
                        ast.tokens,
                    ) catch {};
                }
            }

            result = try lintAst(
                file_allocator,
                io,
                builder,
                report_path,
                ast,
                options,
                previously_verified,
            );
            linted_source_size = read_size;
        }

        if (lint_cache_facts_are_usable) {
            if (lint_cache) |*cache| {
                if (!result.lint_clean or opened.stat.size != linted_source_size) {
                    continue;
                }

                cache.put(scratch_allocator, entry.path, opened.stat, .{
                    .lint_clean = true,
                    .zig_fmt_clean = result.zig_fmt_clean,
                }) catch {};
            }
        }
    }

    // Reaching here means the complete walk succeeded. Failed scans leave the
    // previous disk files intact, and each record is revalidated on the next run.
    commitDirectoryCaches(
        scratch_allocator,
        io,
        &lint_cache,
        &syntax_cache,
    );
}

fn commitDirectoryCaches(
    allocator: Allocator,
    io: std.Io,
    lint_cache: *?LintCache,
    syntax_cache: *?AstTokenCache,
) void {
    if (lint_cache.*) |*cache| {
        cache.save(allocator, io) catch {};
    }

    if (syntax_cache.*) |*cache| {
        cache.save(allocator, io) catch {};
    }
}

// `std.Io.Dir.openFile` performs a blocking read-only open. A FIFO named with a
// `.zig` suffix would wait for a writer before the linter could inspect its
// type. `O_NONBLOCK` makes opening every candidate safe; after `fstat` confirms
// a regular file, the flag does not change regular-file reads.
fn openRegularFile(
    io: std.Io,
    dir: std.Io.Dir,
    sub_path: []const u8,
    options: struct { follow_symlinks: bool = true },
) !?struct { file: std.Io.File, stat: std.Io.File.Stat } {
    const handle = try std.posix.openat(dir.handle, sub_path, .{
        .ACCMODE = .RDONLY,
        .NONBLOCK = true,
        .NOCTTY = true,
        .NOFOLLOW = !options.follow_symlinks,
        .CLOEXEC = true,
    }, 0);

    var file: std.Io.File = .{
        .handle = handle,
        .flags = .{ .nonblocking = true },
    };
    errdefer file.close(io);

    const stat = try file.stat(io);
    if (stat.kind != .file) {
        file.close(io);

        return null;
    }

    return .{ .file = file, .stat = stat };
}

// The scan walker starts at the requested root and cannot discover `.gitignore`
// files above it. Descend only that root's ancestor chain, evaluating each
// directory before loading its own file because an ignored directory is a prune
// boundary that descendant rules cannot reverse. Returns whether one of those
// rules excludes the requested scan root.
fn loadAncestorGitignores(
    allocator: Allocator,
    io: std.Io,
    git_root: []const u8,
    git_root_relative_scan_path: []const u8,
    gitignore: *Gitignore,
) !bool {
    assert(git_root_relative_scan_path.len != 0);

    var ancestor_dir = try std.Io.Dir.cwd().openDir(io, git_root, .{});
    defer ancestor_dir.close(io);

    _ = try loadGitignoreFile(
        allocator,
        io,
        gitignore,
        ancestor_dir,
        ".gitignore",
        "",
    );

    var components = std.mem.splitScalar(u8, git_root_relative_scan_path, '/');
    var ancestor_path_size: usize = 0;

    while (components.next()) |component| {
        ancestor_path_size += @intFromBool(ancestor_path_size != 0) + component.len;

        const ancestor_path = git_root_relative_scan_path[0..ancestor_path_size];
        if (gitignore.pathIsIgnored(ancestor_path, true)) {
            return true;
        }

        if (ancestor_path_size == git_root_relative_scan_path.len) {
            return false;
        }

        const child_dir = try ancestor_dir.openDir(io, component, .{});

        ancestor_dir.close(io);

        ancestor_dir = child_dir;

        _ = try loadGitignoreFile(
            allocator,
            io,
            gitignore,
            ancestor_dir,
            ".gitignore",
            ancestor_path,
        );
    }

    return false;
}

fn dirNameIsExcluded(excluded_names: []const []const u8, name: []const u8) bool {
    for (excluded_names) |excluded_name| {
        if (std.mem.eql(u8, name, excluded_name)) {
            return true;
        }
    }

    return false;
}

// `loadGitignoreFile` reads one regular, non-symlink file through `allocator`
// and compiles it into `gitignore`. It returns false when the path is absent or
// not an eligible file; the `Gitignore` copy outlives the temporary read buffer.
fn loadGitignoreFile(
    allocator: Allocator,
    io: std.Io,
    gitignore: *Gitignore,
    dir: std.Io.Dir,
    sub_path: []const u8,
    matching_dir_path: []const u8,
) !bool {
    const opened = openRegularFile(
        io,
        dir,
        sub_path,
        .{ .follow_symlinks = false },
    ) catch |err|
        switch (err) {
            error.FileNotFound, error.IsDir, error.SymLinkLoop => return false,
            else => return err,
        } orelse {
        return false;
    };

    var file = opened.file;
    defer file.close(io);

    var reader = file.reader(io, &.{});

    const contents = reader.interface.allocRemaining(
        allocator,
        .limited(gitignore_file_size_max),
    ) catch |err|
        switch (err) {
            error.ReadFailed => return reader.err orelse {
                unreachable;
            },
            error.OutOfMemory, error.StreamTooLong => |failure| return failure,
        };
    defer allocator.free(contents);

    try gitignore.push(allocator, matching_dir_path, contents);

    return true;
}
