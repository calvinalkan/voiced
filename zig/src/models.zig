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
                .weights = .{ .size = 145_216_508, .blake3 = "46fa7ff77f6613205ae186b6b763e74b5e0f0ee19ce008f96e85094afaa17f4d" },
            },
            .systran_small_en => .{
                .name = "Systran/faster-whisper-small.en",
                .revision = "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
                .weights = .{ .size = 483_545_366, .blake3 = "6f8da5f2d48b1133b5bc150c5f59a3d3861209637ee15ecf55a09e704fc0254d" },
            },
        };
    }
};

pub const Metadata = struct {
    name: []const u8,
    revision: []const u8,
    weights: struct { size: usize, blake3: []const u8 },
};

pub const default: Model = .systran_small_en;
pub const vocabulary = .{
    .size = 422_309,
    .blake3 = "5ba2618f5d7940b9cebc94299dcc42f056848f660e602ace121ac29af488cb15",
};

/// Resolve the environment-owned model root once before starting workers.
pub fn allocInstalledRootPath(init: std.process.Init) ![]u8 {
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
        &.{ data_home_path, "voiced/models" }
    else
        &.{ data_home_path, ".local/share/voiced/models" };

    return std.fs.path.join(init.gpa, path_parts);
}

pub fn allocInstalledDirectoryPath(allocator: std.mem.Allocator, root: []const u8, model: Model) ![]u8 {
    return std.fs.path.join(allocator, &.{ root, model.name() });
}
