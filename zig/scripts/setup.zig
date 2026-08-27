//! Installs every downloaded input used by the CTranslate2 prototype.
//!
//! One package model describes both direct model files and ZIP-compatible Intel
//! wheels. The command selects a package table and publication directory:
//!
//!   setup native  ──> zig-pkg/mkl/
//!   setup models  ──> $XDG_DATA_HOME/voiced/models/faster-whisper-small.en/
//!   setup all     ──> both installations
//!
//! Every installation is assembled under a sibling `.tmp` directory. Direct
//! files are retained as downloaded; ZIP payloads retain only their declared
//! paths. A generated manifest records package versions and artifact hashes
//! before the complete directory is published.

const std = @import("std");

const assert = std.debug.assert;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

// One immutable remote file and the identity required before publication.
const Artifact = struct {
    display_name: []const u8,
    url: []const u8,
    file_name: []const u8,
    expected_size: u64,
    expected_sha256: []const u8,
};

const Package = struct {
    manifest_name: []const u8,
    version: []const u8,
    payloads: []const union(enum) {
        // Direct files already have their final names and download into the
        // unpublished installation without extraction.
        file: Artifact,

        // ZIP archives extract in scratch space, then move only
        // the paths consumed by the build into the unpublished installation.
        zip: struct {
            artifact: Artifact,
            moves: []const union(enum) {
                file: struct {
                    source_path: []const u8,
                    destination_path: []const u8,
                },
                directory: struct {
                    source_path: []const u8,
                    destination_path: []const u8,
                    required_file_path: []const u8,
                },
            },
        },
    },
};

const SetupCommand = enum {
    native,
    models,
    all,
};

pub fn main(init: std.process.Init) !void {
    const command = try parseCommand(init.minimal.args, init.gpa);

    var http_client: std.http.Client = .{
        .allocator = init.gpa,
        .io = init.io,
    };
    defer http_client.deinit();

    switch (command) {
        .native => try setupNative(init.io, init.gpa, &http_client),
        .models => try setupModels(init, &http_client),
        .all => {
            try setupNative(init.io, init.gpa, &http_client);
            try setupModels(init, &http_client);
        },
    }
}

fn parseCommand(process_arguments: std.process.Args, gpa: Allocator) !SetupCommand {
    var iterator = try std.process.Args.Iterator.initAllocator(process_arguments, gpa);
    defer iterator.deinit();

    assert(iterator.skip());

    const command: ?SetupCommand = if (iterator.next()) |command_text|
        if (std.mem.eql(u8, command_text, "native"))
            .native
        else if (std.mem.eql(u8, command_text, "models"))
            .models
        else if (std.mem.eql(u8, command_text, "all"))
            .all
        else
            null
    else
        null;

    if (command == null or iterator.next() != null) {
        std.debug.print("usage: setup <native|models|all>\n", .{});

        return error.InvalidArguments;
    }

    return command.?;
}

fn setupNative(
    io: Io,
    gpa: Allocator,
    http_client: *std.http.Client,
) !void {
    // ── Describe Pinned Native Packages ──
    //
    // Each Intel package contains one wheel. The path tables normalize Python's
    // installer layout into the include/lib/opt prefix consumed by build.zig.
    const native_packages = [_]Package{
        .{
            .manifest_name = "mkl-static",
            .version = "2025.3.1",
            .payloads = &.{.{
                .zip = .{
                    .artifact = .{
                        .display_name = "Intel oneMKL static 2025.3.1",
                        .url = "https://files.pythonhosted.org/packages/6c/ca/" ++
                            "3f030ea6a9690f7b9918961d1a8cb9854ae2b716c4b40745d3f8d555cfa5/" ++
                            "mkl_static-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl",
                        .file_name = "mkl_static-2025.3.1.whl",
                        .expected_size = 218_574_677,
                        .expected_sha256 = "a8f13291243aa78bae7c2d38092ae34c99cc133fcd426beca2421cc44517f355",
                    },
                    // CTranslate2 links the ILP64 interface, threaded
                    // implementation, and core. Other MKL variants are discarded.
                    .moves = &.{
                        .{ .file = .{
                            .source_path = "mkl_static-2025.3.1.data/data/lib/" ++
                                "libmkl_intel_ilp64.a",
                            .destination_path = "lib/libmkl_intel_ilp64.a",
                        } },
                        .{ .file = .{
                            .source_path = "mkl_static-2025.3.1.data/data/lib/" ++
                                "libmkl_intel_thread.a",
                            .destination_path = "lib/libmkl_intel_thread.a",
                        } },
                        .{ .file = .{
                            .source_path = "mkl_static-2025.3.1.data/data/lib/libmkl_core.a",
                            .destination_path = "lib/libmkl_core.a",
                        } },
                    },
                },
            }},
        },
        .{
            .manifest_name = "mkl-include",
            .version = "2025.3.1",
            .payloads = &.{.{
                .zip = .{
                    .artifact = .{
                        .display_name = "Intel oneMKL headers 2025.3.1",
                        .url = "https://files.pythonhosted.org/packages/a6/43/" ++
                            "7c107c3b05da5590797d34a48666b235b05265398feb48984a14dd1b458b/" ++
                            "mkl_include-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl",
                        .file_name = "mkl_include-2025.3.1.whl",
                        .expected_size = 1_263_883,
                        .expected_sha256 = "b81b08dfd64901a398bd1a751783e9a4de53d9e2ba2e019438dab9e3f8c7e1ea",
                    },
                    // Preserve the complete tree because mkl.h resolves other
                    // Intel headers from it internally.
                    .moves = &.{.{ .directory = .{
                        .source_path = "mkl_include-2025.3.1.data/data/include",
                        .destination_path = "include",
                        .required_file_path = "include/mkl.h",
                    } }},
                },
            }},
        },
        .{
            .manifest_name = "intel-openmp",
            .version = "2025.3.3",
            .payloads = &.{.{
                .zip = .{
                    .artifact = .{
                        .display_name = "Intel OpenMP static 2025.3.3",
                        .url = "https://files.pythonhosted.org/packages/3f/12/" ++
                            "5e07caf359d5bedd67b40e84e2faec6b5a3ad5da27ce151860fe6b270935/" ++
                            "intel_openmp-2025.3.3-py2.py3-none-manylinux_2_28_x86_64.whl",
                        .file_name = "intel_openmp-2025.3.3.whl",
                        .expected_size = 74_335_293,
                        .expected_sha256 = "0b20a49b041afc8c1ec9b5746f006c2aa1d2475a9fe6d29ed551dba7cb84d9ab",
                    },
                    // Threaded MKL needs Intel's OpenMP archive and compiler
                    // headers. Preserve the complete opt tree around those headers.
                    .moves = &.{
                        .{ .file = .{
                            .source_path = "intel_openmp-2025.3.3.data/data/lib/libiomp5.a",
                            .destination_path = "lib/libiomp5.a",
                        } },
                        .{ .directory = .{
                            .source_path = "intel_openmp-2025.3.3.data/data/opt",
                            .destination_path = "opt",
                            .required_file_path = "opt/compiler/include/omp.h",
                        } },
                    },
                },
            }},
        },
    };

    try installPackages(
        io,
        gpa,
        http_client,
        &native_packages,
        "Native dependencies",
        "zig-pkg/mkl",
        "voiced-native-dependencies.txt",
    );
}

fn setupModels(
    init: std.process.Init,
    http_client: *std.http.Client,
) !void {
    // ── Describe Pinned CTranslate2 Model ──
    //
    // Systran publishes an already-converted English Whisper model. These four
    // files form the directory consumed by CTranslate2's ModelLoader; they are
    // not whisper.cpp GGML model files.
    const model_revision = "d1d751a5f8271d482d14ca55d9e2deeebbae577f";
    const model_url =
        "https://huggingface.co/Systran/faster-whisper-small.en/resolve/" ++ model_revision ++ "/";

    const model_packages = [_]Package{.{
        .manifest_name = "faster-whisper-small.en",
        .version = model_revision,
        .payloads = &.{
            .{ .file = .{
                .display_name = "CTranslate2 Whisper small.en configuration",
                .url = model_url ++ "config.json",
                .file_name = "config.json",
                .expected_size = 2_657,
                .expected_sha256 = "666a9605530ac1f61fa8177f3702b4dacec9966749e42610839fcc32661d5fae",
            } },
            .{ .file = .{
                .display_name = "CTranslate2 Whisper small.en weights",
                .url = model_url ++ "model.bin",
                .file_name = "model.bin",
                .expected_size = 483_545_366,
                .expected_sha256 = "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a",
            } },
            .{ .file = .{
                .display_name = "CTranslate2 Whisper small.en tokenizer",
                .url = model_url ++ "tokenizer.json",
                .file_name = "tokenizer.json",
                .expected_size = 2_128_466,
                .expected_sha256 = "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
            } },
            .{ .file = .{
                .display_name = "CTranslate2 Whisper small.en vocabulary",
                .url = model_url ++ "vocabulary.txt",
                .file_name = "vocabulary.txt",
                .expected_size = 422_309,
                .expected_sha256 = "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
            } },
        },
    }};

    // ── Resolve User Model Directory ──
    //
    // Runtime model data follows the XDG data convention instead of living
    // beside build inputs in the repository.

    const env = init.environ_map;

    const models_dir_path = if (env.get("XDG_DATA_HOME")) |xdg_data_home_path|
        try std.fs.path.join(init.gpa, &.{ xdg_data_home_path, "voiced/models" })
    else models_dir_path: {
        const home_path = env.get("HOME") orelse return error.HomeNotSet;
        break :models_dir_path try std.fs.path.join(
            init.gpa,
            &.{ home_path, ".local/share/voiced/models" },
        );
    };
    defer init.gpa.free(models_dir_path);

    if (!std.fs.path.isAbsolute(models_dir_path)) {
        return error.DataHomeNotAbsolute;
    }

    const model_dir_path = try std.fs.path.join(
        init.gpa,
        &.{ models_dir_path, "faster-whisper-small.en" },
    );
    defer init.gpa.free(model_dir_path);

    try installPackages(
        init.io,
        init.gpa,
        http_client,
        &model_packages,
        "CTranslate2 Whisper small.en model",
        model_dir_path,
        "voiced-model.txt",
    );
}

fn installPackages(
    io: Io,
    gpa: Allocator,
    http_client: *std.http.Client,
    packages: []const Package,
    installation_name: []const u8,
    installation_dir_path: []const u8,
    manifest_file_name: []const u8,
) !void {
    assert(packages.len > 0);
    assert(installation_name.len > 0);
    assert(installation_dir_path.len > 0);
    assert(manifest_file_name.len > 0);

    // ── Accept An Existing Complete Installation ──
    //
    // The manifest comes from the same table that drives downloads. Required
    // file checks reject a matching marker beside an incomplete installation.
    const manifest = try packageManifest(gpa, packages);
    defer gpa.free(manifest);

    const cwd = Io.Dir.cwd();
    if (try existingInstallationIsComplete(
        io,
        gpa,
        cwd,
        installation_dir_path,
        manifest_file_name,
        manifest,
        packages,
    )) {
        std.debug.print(
            "{s} already installed in {s}\n",
            .{ installation_name, installation_dir_path },
        );
        return;
    }

    // ── Prepare An Unpublished Installation ──
    //
    // Recreate a sibling temporary directory. `errdefer` removes it on any
    // failed download, verification, extraction, move, or manifest write.
    const parent_dir_path = std.fs.path.dirname(installation_dir_path) orelse ".";
    const tmp_dir_path = try std.fmt.allocPrint(gpa, "{s}.tmp", .{installation_dir_path});
    defer gpa.free(tmp_dir_path);

    try cwd.createDirPath(io, parent_dir_path);
    try cwd.deleteTree(io, tmp_dir_path);
    try cwd.createDirPath(io, tmp_dir_path);
    errdefer cwd.deleteTree(io, tmp_dir_path) catch {};

    {
        var tmp_dir = try cwd.openDir(io, tmp_dir_path, .{});
        defer tmp_dir.close(io);

        // ── Materialize Package Payloads ──
        //
        // Direct files publish into the temporary installation immediately.
        // ZIP payloads use isolated scratch space and retain only listed paths.
        for (packages) |package| {
            for (package.payloads) |payload| {
                switch (payload) {
                    .file => |artifact| try downloadAndVerifyArtifact(
                        io,
                        http_client,
                        artifact,
                        tmp_dir,
                    ),
                    .zip => |archive| try downloadExtractAndMoveArchive(
                        io,
                        http_client,
                        archive.artifact,
                        archive.moves,
                        tmp_dir,
                    ),
                }
            }
        }

        // Writing the marker last means it never identifies a partially
        // assembled temporary directory as complete.
        try tmp_dir.writeFile(io, .{
            .sub_path = manifest_file_name,
            .data = manifest,
        });
    }

    // ── Publish Complete Installation ──
    try cwd.deleteTree(io, installation_dir_path);
    try cwd.rename(tmp_dir_path, cwd, installation_dir_path, io);

    std.debug.print(
        "Installed {s} in {s}\n",
        .{ installation_name, installation_dir_path },
    );
}

fn packageManifest(gpa: Allocator, packages: []const Package) ![]u8 {
    const lines = try gpa.alloc([]const u8, packages.len);
    defer gpa.free(lines);

    var lines_initialized_count: usize = 0;
    defer {
        for (lines[0..lines_initialized_count]) |line| {
            gpa.free(line);
        }
    }

    for (packages, 0..) |package, package_index| {
        const fields = try gpa.alloc([]const u8, package.payloads.len + 2);
        defer gpa.free(fields);

        fields[0] = package.manifest_name;
        fields[1] = package.version;

        for (package.payloads, 0..) |payload, payload_index| {
            fields[payload_index + 2] = switch (payload) {
                .file => |artifact| artifact.expected_sha256,
                .zip => |archive| archive.artifact.expected_sha256,
            };
        }

        const line_body = try std.mem.join(gpa, " ", fields);
        if (package_index + 1 == packages.len) {
            lines[package_index] = line_body;
        } else {
            defer gpa.free(line_body);
            lines[package_index] = try std.fmt.allocPrint(gpa, "{s}\n", .{line_body});
        }
        lines_initialized_count += 1;
    }

    return std.mem.concat(gpa, u8, lines);
}

fn existingInstallationIsComplete(
    io: Io,
    gpa: Allocator,
    cwd: Io.Dir,
    installation_dir_path: []const u8,
    manifest_file_name: []const u8,
    expected_manifest: []const u8,
    packages: []const Package,
) !bool {
    const manifest_path = try std.fs.path.join(
        gpa,
        &.{ installation_dir_path, manifest_file_name },
    );
    defer gpa.free(manifest_path);

    const actual_manifest = cwd.readFileAlloc(
        io,
        manifest_path,
        gpa,
        .limited(expected_manifest.len + 1),
    ) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer gpa.free(actual_manifest);

    if (!std.mem.eql(u8, actual_manifest, expected_manifest)) {
        return false;
    }

    for (packages) |package| {
        for (package.payloads) |payload| {
            switch (payload) {
                .file => |artifact| {
                    if (!try fileExists(
                        io,
                        gpa,
                        cwd,
                        installation_dir_path,
                        artifact.file_name,
                    )) return false;
                },
                .zip => |archive| {
                    for (archive.moves) |path_move| {
                        const required_file_path = switch (path_move) {
                            .file => |file_move| file_move.destination_path,
                            .directory => |directory_move| directory_move.required_file_path,
                        };

                        if (!try fileExists(
                            io,
                            gpa,
                            cwd,
                            installation_dir_path,
                            required_file_path,
                        )) return false;
                    }
                },
            }
        }
    }

    return true;
}

fn fileExists(
    io: Io,
    gpa: Allocator,
    cwd: Io.Dir,
    installation_dir_path: []const u8,
    required_file_path: []const u8,
) !bool {
    const path = try std.fs.path.join(
        gpa,
        &.{ installation_dir_path, required_file_path },
    );
    defer gpa.free(path);

    const file = cwd.openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    file.close(io);

    return true;
}

fn downloadExtractAndMoveArchive(
    io: Io,
    http_client: *std.http.Client,
    artifact: Artifact,
    moves: anytype,
    tmp_dir: Io.Dir,
) !void {
    const extracted_archive_dir_path = ".archive";
    try tmp_dir.deleteTree(io, extracted_archive_dir_path);
    try tmp_dir.createDirPath(io, extracted_archive_dir_path);

    {
        var extracted_archive_dir = try tmp_dir.openDir(
            io,
            extracted_archive_dir_path,
            .{},
        );
        defer extracted_archive_dir.close(io);

        try downloadAndVerifyArtifact(
            io,
            http_client,
            artifact,
            extracted_archive_dir,
        );

        var archive_file = try extracted_archive_dir.openFile(io, artifact.file_name, .{});
        defer archive_file.close(io);

        var archive_reader_buffer: [64 * 1024]u8 = undefined;
        var archive_reader = archive_file.reader(io, &archive_reader_buffer);
        try std.zip.extract(extracted_archive_dir, &archive_reader, .{});

        for (moves) |path_move| {
            const source_path, const destination_path = switch (path_move) {
                .file => |file_move| .{
                    file_move.source_path,
                    file_move.destination_path,
                },
                .directory => |directory_move| .{
                    directory_move.source_path,
                    directory_move.destination_path,
                },
            };

            if (std.fs.path.dirname(destination_path)) |destination_parent_path| {
                try tmp_dir.createDirPath(io, destination_parent_path);
            }

            try extracted_archive_dir.rename(
                source_path,
                tmp_dir,
                destination_path,
                io,
            );
        }
    }

    try tmp_dir.deleteTree(io, extracted_archive_dir_path);
}

// Download one artifact into a caller-owned directory. The HTTP body streams
// through SHA-256 into an unpublished file; only the exact pinned size and
// digest allow its final name to become visible.
fn downloadAndVerifyArtifact(
    io: Io,
    http_client: *std.http.Client,
    artifact: Artifact,
    destination_dir: Io.Dir,
) !void {
    assert(artifact.display_name.len > 0);
    assert(artifact.url.len > 0);
    assert(artifact.file_name.len > 0);
    assert(artifact.expected_size > 0);
    assert(artifact.expected_sha256.len == Sha256.digest_length * 2);

    std.debug.print("Downloading {s}", .{artifact.display_name});

    var download_line_is_open = true;
    defer {
        // Progress rewrites one unterminated terminal line. An early error must
        // end that line before the shell prints its next prompt.
        if (download_line_is_open) {
            std.debug.print("\n", .{});
        }
    }

    // ── Request Artifact ──
    //
    // Hugging Face and Python package hosting both redirect to storage
    // backends. Disable content encoding so the pins describe bytes on disk.
    var request = try http_client.request(
        .GET,
        try std.Uri.parse(artifact.url),
        .{
            .redirect_behavior = .init(3),
            .headers = .{
                .user_agent = .{ .override = "voiced-setup" },
                .accept_encoding = .omit,
            },
        },
    );
    defer request.deinit();
    try request.sendBodiless();

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var response = try request.receiveHead(&redirect_buffer);
    if (response.head.status != .ok) {
        std.debug.print(
            "\r\x1b[2Kerror: HTTP request failed\n" ++
                "  output: {s}\n" ++
                "  url: {s}\n" ++
                "  status: {d}\n",
            .{
                artifact.file_name,
                artifact.url,
                @intFromEnum(response.head.status),
            },
        );
        download_line_is_open = false;
        return error.UnexpectedHttpStatus;
    }

    // ── Stream Into Unpublished File ──
    //
    // The atomic file lives in the destination directory and may be unnamed on
    // Linux. Every exit before `link` removes it without exposing partial data.
    var artifact_file = try destination_dir.createFileAtomic(
        io,
        artifact.file_name,
        .{},
    );
    defer artifact_file.deinit(io);

    var response_transfer_buffer: [64]u8 = undefined;
    var download_buffer: [64 * 1024]u8 = undefined;
    const response_body = response.reader(&response_transfer_buffer);

    var artifact_writer = artifact_file.file.writerStreaming(io, &.{});
    var sha256_writer = artifact_writer.interface.hashed(Sha256.init(.{}), &.{});

    const expected_mib_tenths = artifact.expected_size * 10 / (1024 * 1024);
    var downloaded_size: u64 = 0;
    var progress_percent_reported: u8 = 0;

    while (true) {
        const read_size = response_body.readSliceShort(&download_buffer) catch {
            return response.bodyErr().?;
        };
        if (read_size == 0) break;

        try sha256_writer.writer.writeAll(download_buffer[0..read_size]);
        downloaded_size += read_size;

        const progress_percent: u8 = @intCast(@min(
            downloaded_size * 100 / artifact.expected_size,
            100,
        ));
        if (progress_percent > progress_percent_reported) {
            const downloaded_mib_tenths = downloaded_size * 10 / (1024 * 1024);
            std.debug.print(
                "\r\x1b[2KDownloading {s} - {d}.{d}/{d}.{d} MiB ({d}%)",
                .{
                    artifact.display_name,
                    downloaded_mib_tenths / 10,
                    downloaded_mib_tenths % 10,
                    expected_mib_tenths / 10,
                    expected_mib_tenths % 10,
                    progress_percent,
                },
            );
            progress_percent_reported = progress_percent;
        }
    }
    assert(downloaded_size == artifact_writer.pos);

    // ── Verify Complete Artifact ──
    if (artifact_writer.pos != artifact.expected_size) {
        std.debug.print(
            "\r\x1b[2Kerror: artifact size mismatch\n" ++
                "  output: {s}\n" ++
                "  expected: {d} bytes\n" ++
                "  actual: {d} bytes\n",
            .{
                artifact.file_name,
                artifact.expected_size,
                artifact_writer.pos,
            },
        );
        download_line_is_open = false;
        return error.DownloadSizeMismatch;
    }

    var actual_sha256: [Sha256.digest_length]u8 = undefined;
    sha256_writer.hasher.final(&actual_sha256);
    const actual_sha256_hex = std.fmt.bytesToHex(actual_sha256, .lower);
    if (!std.mem.eql(u8, &actual_sha256_hex, artifact.expected_sha256)) {
        std.debug.print(
            "\r\x1b[2Kerror: artifact SHA-256 mismatch\n" ++
                "  output: {s}\n" ++
                "  expected: {s}\n" ++
                "  actual: {s}\n",
            .{
                artifact.file_name,
                artifact.expected_sha256,
                &actual_sha256_hex,
            },
        );
        download_line_is_open = false;
        return error.DownloadHashMismatch;
    }

    // ── Publish Verified Artifact ──
    try artifact_file.file.sync(io);
    try artifact_file.link(io);

    std.debug.print(
        "\r\x1b[2KDownloaded {s} as {s}\n",
        .{ artifact.display_name, artifact.file_name },
    );
    download_line_is_open = false;
}
