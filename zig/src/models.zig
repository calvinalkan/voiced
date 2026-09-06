//! Names the runtime models that Voiced installs and may load. The model name
//! preserves both its publisher and repository variant so configuration never
//! turns an arbitrary directory into an implicit model identity.

const std = @import("std");

pub const Model = enum(u8) {
    systran_base_en = 1,
    systran_small_en = 2,

    /// `parse` accepts only canonical publisher/repository names, not paths.
    pub fn parse(text: []const u8) ?Model {
        inline for (std.meta.tags(Model)) |model| {
            if (std.mem.eql(u8, text, model.name())) {
                return model;
            }
        }
        return null;
    }

    pub fn name(model: Model) []const u8 {
        return model.metadata().name;
    }

    pub fn metadata(model: Model) Metadata {
        return switch (model) {
            .systran_base_en => .{
                .name = "Systran/faster-whisper-base.en",
                .revision = "3d3d5dee26484f91867d81cb899cfcf72b96be6c",
                .weights = .{ .size = 145_216_508, .sha256 = "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef" },
            },
            .systran_small_en => .{
                .name = "Systran/faster-whisper-small.en",
                .revision = "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
                .weights = .{ .size = 483_545_366, .sha256 = "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a" },
            },
        };
    }
};

pub const Metadata = struct {
    name: []const u8,
    revision: []const u8,
    weights: struct { size: usize, sha256: []const u8 },
};

pub const default: Model = .systran_small_en;
pub const vocabulary = .{
    .size = 422_309,
    .sha256 = "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
};

/// `allocInstalledDirectoryPath` resolves one named model under the user's XDG
/// data directory. The caller owns the returned path and
/// must free it with `init.gpa`.
pub fn allocInstalledDirectoryPath(
    init: std.process.Init,
    model: Model,
) ![]u8 {
    const data_home_path = if (init.environ_map.get("XDG_DATA_HOME")) |path|
        path
    else
        init.environ_map.get("HOME") orelse {
            return error.HomeNotSet;
        };
    if (!std.fs.path.isAbsolute(data_home_path)) {
        return error.DataHomeNotAbsolute;
    }

    const path_parts: []const []const u8 = if (init.environ_map.get("XDG_DATA_HOME") != null)
        &.{ data_home_path, "voiced/models", model.name() }
    else
        &.{ data_home_path, ".local/share/voiced/models", model.name() };

    return std.fs.path.join(init.gpa, path_parts);
}
