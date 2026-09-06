//! Latest accepted transcript, replaced after desktop delivery. Direct writes
//! avoid a worker and a second transcript buffer; slow storage can block the
//! supervisor here. Atomic replacement is not a power-loss durability promise.
const std = @import("std");

pub const Error = union(enum) {
    open_directory: std.Io.Dir.CreateDirPathOpenError,
    stat_directory: std.os.linux.E,
    unsafe_directory: struct { uid: u32, expected_uid: u32, mode: u16, uid_available: bool, mode_available: bool },
    permissions: std.Io.Dir.SetPermissionsError,
    create_temporary: std.Io.Dir.CreateFileAtomicError,
    write: struct { cause: std.Io.File.Writer.Error, bytes_written: usize, bytes_total: usize, cleanup: ?CleanupError },
    replace: struct { cause: std.Io.File.Atomic.ReplaceError, cleanup: ?CleanupError },
};

pub const CleanupError = struct { cause: std.Io.Dir.DeleteFileError, temporary_name: [16]u8 };

pub const Result = union(enum) { ok: void, err: Error };

/// Errors refer to directory_path/transcript.txt; the caller owns that stable
/// path. std.Io errors retain the library's exact error set, not invented errno.
pub fn save(io: std.Io, directory_path: []const u8, text: []const u8) Result {
    std.debug.assert(text.len > 0);
    const directory = std.Io.Dir.cwd().createDirPathOpen(io, directory_path, .{
        .permissions = .fromMode(0o700),
        .open_options = .{ .iterate = true, .follow_symlinks = false },
    }) catch |err| return .{ .err = .{ .open_directory = err } };
    defer directory.close(io);
    const linux = std.os.linux;
    var stat: linux.Statx = undefined;
    const errno = linux.errno(linux.statx(directory.handle, "", linux.AT.EMPTY_PATH, .BASIC_STATS, &stat));
    if (errno != .SUCCESS) return .{ .err = .{ .stat_directory = errno } };
    if (!stat.mask.UID or !stat.mask.MODE or stat.uid != linux.geteuid())
        return .{ .err = .{ .unsafe_directory = .{ .uid = stat.uid, .expected_uid = linux.geteuid(), .mode = stat.mode, .uid_available = stat.mask.UID, .mode_available = stat.mask.MODE } } };
    if (stat.mode & 0o777 != 0o700) directory.setPermissions(io, .fromMode(0o700)) catch |err| return .{ .err = .{ .permissions = err } };

    var output = directory.createFileAtomic(io, "transcript.txt", .{ .replace = true, .permissions = .fromMode(0o600) }) catch |err| return .{ .err = .{ .create_temporary = err } };
    defer output.deinit(io);
    var written: usize = 0;
    while (written < text.len) {
        written += output.file.writeStreaming(io, &.{}, &.{text[written..]}, 1) catch |err|
            return .{ .err = .{ .write = .{ .cause = err, .bytes_written = written, .bytes_total = text.len, .cleanup = discardTemporary(io, &output) } } };
    }
    output.replace(io) catch |err| return .{ .err = .{ .replace = .{ .cause = err, .cleanup = discardTemporary(io, &output) } } };
    return .{ .ok = {} };
}

// Atomic.deinit silently ignores unlink errors. Capture that independent
// cleanup result before deinit closes handles; never overwrite the write/rename
// error, and keep the temporary basename when removal failed.
fn discardTemporary(io: std.Io, output: *std.Io.File.Atomic) ?CleanupError {
    if (!output.file_exists) return null;
    const name = std.fmt.hex(output.file_basename_hex);
    output.file_exists = false;
    output.dir.deleteFile(io, &name) catch |err| return .{ .cause = err, .temporary_name = name };
    return null;
}

/// Resolve once at service startup. The caller owns the returned directory path.
/// The control socket has already validated the instance name.
pub fn allocDirectoryPath(
    allocator: std.mem.Allocator,
    state_home: ?[]const u8,
    home: ?[]const u8,
    instance: []const u8,
) ![]u8 {
    const use_state_home = state_home != null and std.fs.path.isAbsolute(state_home.?);
    const root = if (use_state_home) state_home.? else home orelse return error.HomeNotSet;
    if (!std.fs.path.isAbsolute(root)) return error.StateHomeNotAbsolute;
    var name_buffer: [47]u8 = undefined;
    const name = if (instance.len == 0) "voiced" else try std.fmt.bufPrint(&name_buffer, "voiced-{s}", .{instance});
    return std.fs.path.join(allocator, &.{ root, if (use_state_home) "" else ".local/state", name });
}
