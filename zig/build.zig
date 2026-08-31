const std = @import("std");
const assert = std.debug.assert;

pub fn build(b: *std.Build) void {
    // `zig build`
    add_default_build_command(b);

    // `zig build update-compile-flags`
    add_update_compile_flags_command(b);

    const setup_tool = add_setup_tool(b);

    // `zig build setup-native`
    add_setup_native_command(b, setup_tool);

    // `zig build setup-models`
    add_setup_models_command(b, setup_tool);

    // `zig build setup`
    add_setup_all_command(b, setup_tool);
}

const RepositoryCppCompilePolicy = struct {
    const standard_flag = "-std=c++17";
    const warning_flags = [_][]const u8{
        "-Weverything",
        "-Werror",
        "-pedantic-errors",
        "-Wno-c++98-compat",
        "-Wno-c++98-compat-pedantic",
        "-Wno-pre-c++14-compat",
        "-Wno-pre-c++17-compat",
    };
    const flags = [_][]const u8{standard_flag} ++ warning_flags;
};

// This builds CTranslate2 for CPU inference only. CUDA and cuDNN are not
// compiled, so the binary does not need GPU drivers. Zig compiles the pinned
// CTranslate2 sources directly and statically links oneMKL and Intel OpenMP.
//
// CMake is not invoked. The source list and compiler settings below match the
// pinned CTranslate2 CPU build.
//
// Build graph:
//
// ┌────────────────────────────────────────────────────────────────────────────┐
// │                                BUILD INPUTS                                │
// ├────────────────────────┬─────────────────────────┬─────────────────────────┤
// │ build.zig.zon          │ src/                    │ zig-pkg/mkl/            │
// ├────────────────────────┼─────────────────────────┼─────────────────────────┤
// │ CTranslate2 + spdlog   │ Zig model spike         │ oneMKL archives         │
// │ cpu_features C         │ C++ boundary            │ Intel OpenMP archive    │
// └────────────────────────┴─────────────────────────┴─────────────────────────┘
//                                       │
//                                       ▼
// ┌────────────────────────────────────────────────────────────────────────────┐
// │                             COMPILED ARTIFACTS                             │
// ├───────────────────────────┬───────────────────────┬────────────────────────┤
// │ voiced-ctranslate2.a      │ C boundary object     │ AVX kernel objects     │
// └───────────────────────────┴───────────────────────┴────────────────────────┘
//                                       │
//                                       │  static link with oneMKL + Intel OpenMP
//                                       ▼
//                        ┌──────────────────────────────┐
//                        │         model-spike          │
//                        └──────────────────────────────┘
//
// `zig build` only compiles and installs the binary. `setup-native` downloads
// MKL/OpenMP build inputs, `setup-models` downloads the runtime models, and
// `setup` performs both explicitly. `update-compile-flags` writes clangd's
// checked-in C++ flags to compile_flags.txt so Zed can pick them up for the LSP.
fn add_default_build_command(b: *std.Build) void {
    // ── Resolve Build Inputs ──
    //
    // `build.zig.zon` supplies the CTranslate2 and cpu_features source trees
    // plus spdlog's headers. `setup-native` places oneMKL and Intel OpenMP under
    // the default prefix; `-Dmkl-prefix` selects an existing alternate prefix.
    // Every target artifact uses one optimization mode and an Ubuntu 22.04
    // x86-64 baseline so the final executable never requires an AVX extension.

    // Plain `zig build` remains a Debug development build. Production builds
    // explicitly select ReleaseSafe so trusted process, exchange, and callback
    // contracts remain active; ReleaseFast is reserved for comparative
    // benchmarks where removing those assertions is intentional.
    const optimize = b.standardOptimizeOption(.{});

    const mkl_prefix = b.option(
        []const u8,
        "mkl-prefix",
        "Intel oneMKL and OpenMP installation prefix",
    ) orelse "zig-pkg/mkl";
    assert(mkl_prefix.len > 0);

    const ctranslate2 = b.dependency("ctranslate2", .{});
    const cpu_features = b.dependency("cpu_features", .{});
    const spdlog = b.dependency("spdlog", .{});

    const baseline_target = resolve_linux_x86_target(b, &.{.sse4_1});

    // The prefix layout comes from Intel's archives: MKL and OpenMP expose
    // separate include directories and a shared `lib` directory used at link.
    const mkl_include: std.Build.LazyPath = .{
        .cwd_relative = b.pathJoin(&.{ mkl_prefix, "include" }),
    };
    const openmp_include: std.Build.LazyPath = .{
        .cwd_relative = b.pathJoin(&.{ mkl_prefix, "opt/compiler/include" }),
    };

    // ── Compile Baseline CTranslate2 ──
    //
    // The main static archive contains code that may run on every supported
    // host: CTranslate2's generic implementation, its SSE4.1 kernels, and the
    // CPU detection code added in the next stage. Faster kernels remain outside
    // this archive so loading the executable is safe before runtime dispatch.

    const ctranslate2_library = b.addLibrary(.{
        .name = "voiced-ctranslate2",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = baseline_target,
            .optimize = optimize,
        }),
    });

    // Direct compilation of upstream `.cc` files requires CTranslate2's public,
    // private, and third-party headers. cpu_features supplies dispatch headers,
    // spdlog supplies logging headers, and MKL/OpenMP supply the CPU backend API.
    ctranslate2_library.root_module.addSystemIncludePath(ctranslate2.path("include"));
    ctranslate2_library.root_module.addIncludePath(ctranslate2.path("src"));
    ctranslate2_library.root_module.addIncludePath(ctranslate2.path("third_party"));
    ctranslate2_library.root_module.addSystemIncludePath(cpu_features.path("include"));
    ctranslate2_library.root_module.addSystemIncludePath(cpu_features.path("include/internal"));
    ctranslate2_library.root_module.addSystemIncludePath(spdlog.path("include"));
    ctranslate2_library.root_module.addSystemIncludePath(mkl_include);
    ctranslate2_library.root_module.addSystemIncludePath(openmp_include);
    ctranslate2_library.root_module.link_libc = true;
    ctranslate2_library.root_module.link_libcpp = true;

    // These definitions select the same CPU-only configuration as the pinned
    // CMake target. Upstream sources intentionally do not receive voiced's
    // `-Weverything -Werror` policy; only repository-owned C++ receives it.
    const ctranslate2_cpp_flags = [_][]const u8{
        "-std=c++17",
        "-DCT2_WITH_CPU_DISPATCH",
        "-DCT2_WITH_MKL",
        "-DCT2_X86_BUILD",
        "-DMKL_ILP64",
        "-DSTACK_LINE_READER_BUFFER_SIZE=1024",
        "-fopenmp",

        // ReleaseSafe instruments C and C++ undefined behavior as well as Zig.
        // CTranslate2 4.6.2 contains unchecked nullable reference paths that its
        // normal optimized build relies on never taking; the instrumentation
        // nevertheless evaluates those paths inside native worker threads and
        // aborts valid Whisper inference. Keep ReleaseSafe for repository-owned
        // Zig and boundary code, but compile this pinned upstream library with
        // the native release assumptions under which it is supported.
        "-fno-sanitize=undefined",
    };

    // This list is the CPU-only `SOURCES` list from the pinned CTranslate2
    // CMake target. `src/cpu/kernels.cc` supplies its SSE4.1 implementation;
    // the dispatched AVX variants are compiled as separate objects below.
    const ctranslate2_source_paths = [_][]const u8{
        "src/allocator.cc",
        "src/batch_reader.cc",
        "src/buffered_translation_wrapper.cc",
        "src/cpu/allocator.cc",
        "src/cpu/backend.cc",
        "src/cpu/cpu_info.cc",
        "src/cpu/cpu_isa.cc",
        "src/cpu/kernels.cc",
        "src/cpu/parallel.cc",
        "src/cpu/primitives.cc",
        "src/decoding.cc",
        "src/decoding_utils.cc",
        "src/devices.cc",
        "src/dtw.cc",
        "src/encoder.cc",
        "src/env.cc",
        "src/filesystem.cc",
        "src/generator.cc",
        "src/layers/attention_layer.cc",
        "src/layers/attention.cc",
        "src/layers/flash_attention.cc",
        "src/layers/common.cc",
        "src/layers/decoder.cc",
        "src/layers/transformer.cc",
        "src/layers/wav2vec2.cc",
        "src/layers/wav2vec2bert.cc",
        "src/layers/whisper.cc",
        "src/logging.cc",
        "src/models/language_model.cc",
        "src/models/model.cc",
        "src/models/model_factory.cc",
        "src/models/model_reader.cc",
        "src/models/sequence_to_sequence.cc",
        "src/models/transformer.cc",
        "src/models/wav2vec2.cc",
        "src/models/wav2vec2bert.cc",
        "src/models/whisper.cc",
        "src/ops/activation.cc",
        "src/ops/add.cc",
        "src/ops/alibi_add.cc",
        "src/ops/alibi_add_cpu.cc",
        "src/ops/bias_add.cc",
        "src/ops/bias_add_cpu.cc",
        "src/ops/concat.cc",
        "src/ops/concat_split_slide_cpu.cc",
        "src/ops/conv1d.cc",
        "src/ops/conv1d_cpu.cc",
        "src/ops/cos.cc",
        "src/ops/dequantize.cc",
        "src/ops/dequantize_cpu.cc",
        "src/ops/flash_attention.cc",
        "src/ops/flash_attention_cpu.cc",
        "src/ops/gather.cc",
        "src/ops/gather_cpu.cc",
        "src/ops/gelu.cc",
        "src/ops/gemm.cc",
        "src/ops/gumbel_max.cc",
        "src/ops/gumbel_max_cpu.cc",
        "src/ops/layer_norm.cc",
        "src/ops/layer_norm_cpu.cc",
        "src/ops/log.cc",
        "src/ops/matmul.cc",
        "src/ops/mean.cc",
        "src/ops/mean_cpu.cc",
        "src/ops/median_filter.cc",
        "src/ops/min_max.cc",
        "src/ops/mul.cc",
        "src/ops/multinomial.cc",
        "src/ops/multinomial_cpu.cc",
        "src/ops/quantize.cc",
        "src/ops/quantize_cpu.cc",
        "src/ops/relu.cc",
        "src/ops/rms_norm.cc",
        "src/ops/rms_norm_cpu.cc",
        "src/ops/rotary.cc",
        "src/ops/rotary_cpu.cc",
        "src/ops/sin.cc",
        "src/ops/softmax.cc",
        "src/ops/softmax_cpu.cc",
        "src/ops/split.cc",
        "src/ops/slide.cc",
        "src/ops/sub.cc",
        "src/ops/sigmoid.cc",
        "src/ops/swish.cc",
        "src/ops/tanh.cc",
        "src/ops/tile.cc",
        "src/ops/tile_cpu.cc",
        "src/ops/topk.cc",
        "src/ops/topk_cpu.cc",
        "src/ops/topp_mask.cc",
        "src/ops/topp_mask_cpu.cc",
        "src/ops/transpose.cc",
        "src/ops/nccl_ops.cc",
        "src/ops/nccl_ops_cpu.cc",
        "src/ops/awq/dequantize.cc",
        "src/ops/awq/dequantize_cpu.cc",
        "src/ops/awq/gemm.cc",
        "src/ops/awq/gemm_cpu.cc",
        "src/ops/awq/gemv.cc",
        "src/ops/awq/gemv_cpu.cc",
        "src/ops/sum.cc",
        "src/padder.cc",
        "src/profiler.cc",
        "src/random.cc",
        "src/sampling.cc",
        "src/scoring.cc",
        "src/storage_view.cc",
        "src/thread_pool.cc",
        "src/translator.cc",
        "src/types.cc",
        "src/utils.cc",
        "src/vocabulary.cc",
        "src/vocabulary_map.cc",
    };
    ctranslate2_library.root_module.addCSourceFiles(.{
        .root = ctranslate2.path(""),
        .files = &ctranslate2_source_paths,
        .flags = &ctranslate2_cpp_flags,
    });

    // ── Add Runtime CPU Detection ──
    //
    // CTranslate2 uses cpu_features to detect SSE4.1, AVX, AVX2, and AVX-512.
    // It uses that result to choose a compatible kernel and to decide whether an
    // Intel CPU may use MKL by default. Detection runs in baseline code before
    // CTranslate2 calls any AVX-specific object.

    const cpu_features_c_flags = [_][]const u8{
        "-std=c99",
        "-DCT2_X86_BUILD",
        "-DSTACK_LINE_READER_BUFFER_SIZE=1024",

        // cpu_features compares its four-byte vendor constants through pointers
        // that may originate at an unaligned string-literal address. x86 permits
        // the load, but Zig's ReleaseSafe alignment instrumentation traps before
        // feature detection can run. Disable only that check for this pinned
        // upstream C source; voiced's Zig assertions and runtime safety remain
        // active, and the detector still runs before any dispatched AVX kernel.
        "-fno-sanitize=alignment",
    };
    // This is cpu_features' portable implementation set. Platform macros make
    // only the Linux x86 implementation provide this target's feature probes;
    // the common filesystem, reader, and string sources support that probe.
    const cpu_features_source_paths = [_][]const u8{
        "src/impl_aarch64_linux_or_android.c",
        "src/impl_arm_linux_or_android.c",
        "src/impl_mips_linux_or_android.c",
        "src/impl_ppc_linux.c",
        "src/impl_x86_freebsd.c",
        "src/impl_x86_linux_or_android.c",
        "src/impl_x86_macos.c",
        "src/impl_x86_windows.c",
        "src/filesystem.c",
        "src/stack_line_reader.c",
        "src/string_view.c",
    };
    ctranslate2_library.root_module.addCSourceFiles(.{
        .root = cpu_features.path(""),
        .files = &cpu_features_source_paths,
        .flags = &cpu_features_c_flags,
    });

    // ── Compile The C Boundary ──
    //
    // The model spike imports only `ctranslate2_bridge.h`. This object owns the
    // C++ model objects, translates exceptions into the C error representation,
    // and keeps CTranslate2 templates and standard-library types out of Zig.
    // Keeping it separate applies voiced's strict warning policy to the boundary
    // without applying that policy to the upstream CTranslate2 implementation.
    const c_boundary = b.addObject(.{
        .name = "voiced-ctranslate2-c-boundary",
        .root_module = b.createModule(.{
            .target = baseline_target,
            .optimize = optimize,
        }),
    });
    c_boundary.root_module.addSystemIncludePath(ctranslate2.path("include"));
    c_boundary.root_module.addIncludePath(b.path("src"));
    c_boundary.root_module.link_libc = true;
    c_boundary.root_module.link_libcpp = true;
    c_boundary.root_module.addCSourceFile(.{
        .file = b.path("src/ctranslate2_bridge.cpp"),
        .flags = &RepositoryCppCompilePolicy.flags,
    });

    // ── Compile Dispatched CPU Kernels ──
    //
    // The baseline archive already contains the SSE4.1 form of `kernels.cc`.
    // These three objects compile the same implementation with progressively
    // stronger target features. CTranslate2 reaches one only after the CPU
    // detector above reports that every instruction required by it is available.
    //
    // Zig archives objects by basename. Separate generated names prevent the
    // three builds of `kernels.cc` from replacing one another in the archive.
    const generated_kernel_sources = b.addWriteFiles();
    const upstream_kernel_source = ctranslate2.path("src/cpu/kernels.cc");

    const avx_kernel = b.addObject(.{
        .name = "voiced-ctranslate2-kernel-avx",
        .root_module = b.createModule(.{
            .target = resolve_linux_x86_target(b, &.{ .sse4_1, .avx }),
            .optimize = optimize,
        }),
    });
    const avx2_kernel = b.addObject(.{
        .name = "voiced-ctranslate2-kernel-avx2",
        .root_module = b.createModule(.{
            .target = resolve_linux_x86_target(b, &.{ .sse4_1, .avx2, .fma }),
            .optimize = optimize,
        }),
    });
    const avx512_kernel = b.addObject(.{
        .name = "voiced-ctranslate2-kernel-avx512",
        .root_module = b.createModule(.{
            .target = resolve_linux_x86_target(
                b,
                &.{ .sse4_1, .avx512f, .avx512cd, .avx512vl, .avx512bw, .avx512dq },
            ),
            .optimize = optimize,
        }),
    });

    // Target features differ, but all three variants must see the same upstream
    // headers and MKL/OpenMP configuration as the baseline implementation.
    const dispatched_kernels = [_]*std.Build.Step.Compile{
        avx_kernel,
        avx2_kernel,
        avx512_kernel,
    };
    for (dispatched_kernels) |kernel| {
        kernel.root_module.addSystemIncludePath(ctranslate2.path("include"));
        kernel.root_module.addIncludePath(ctranslate2.path("src"));
        kernel.root_module.addIncludePath(ctranslate2.path("third_party"));
        kernel.root_module.addSystemIncludePath(mkl_include);
        kernel.root_module.addSystemIncludePath(openmp_include);
        kernel.root_module.link_libc = true;
        kernel.root_module.link_libcpp = true;
    }

    avx_kernel.root_module.addCSourceFile(.{
        .file = generated_kernel_sources.addCopyFile(
            upstream_kernel_source,
            "kernels_avx.cc",
        ),
        .flags = &ctranslate2_cpp_flags,
    });
    avx2_kernel.root_module.addCSourceFile(.{
        .file = generated_kernel_sources.addCopyFile(
            upstream_kernel_source,
            "kernels_avx2.cc",
        ),
        .flags = &ctranslate2_cpp_flags,
    });
    avx512_kernel.root_module.addCSourceFile(.{
        .file = generated_kernel_sources.addCopyFile(
            upstream_kernel_source,
            "kernels_avx512.cc",
        ),
        .flags = &ctranslate2_cpp_flags,
    });

    // ── Link And Install The Model Spike ──
    //
    // The Zig root computes log-Mel features and calls the C boundary. Linking
    // the baseline archive plus every dispatched kernel makes runtime selection
    // self-contained; the executable itself retains the baseline target because
    // no specialized object is entered until after the CPU capability check.

    const model_spike = b.addExecutable(.{
        .name = "model-spike",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/model_spike.zig"),
            .target = baseline_target,
            .optimize = optimize,
        }),
    });
    model_spike.root_module.addIncludePath(b.path("src"));
    model_spike.root_module.linkLibrary(ctranslate2_library);
    model_spike.root_module.addObjectFile(c_boundary.getEmittedBin());
    model_spike.root_module.addObjectFile(avx_kernel.getEmittedBin());
    model_spike.root_module.addObjectFile(avx2_kernel.getEmittedBin());
    model_spike.root_module.addObjectFile(avx512_kernel.getEmittedBin());

    // Intel's static link order lets each later archive resolve references from
    // the interface, threading, and core archives that precede it.
    const mkl_libraries: std.Build.LazyPath = .{
        .cwd_relative = b.pathJoin(&.{ mkl_prefix, "lib" }),
    };
    model_spike.root_module.addObjectFile(mkl_libraries.path(b, "libmkl_intel_ilp64.a"));
    model_spike.root_module.addObjectFile(mkl_libraries.path(b, "libmkl_intel_thread.a"));
    model_spike.root_module.addObjectFile(mkl_libraries.path(b, "libmkl_core.a"));
    model_spike.root_module.addObjectFile(mkl_libraries.path(b, "libiomp5.a"));
    model_spike.root_module.linkSystemLibrary("dl", .{});
    model_spike.root_module.linkSystemLibrary("m", .{});
    model_spike.root_module.linkSystemLibrary("pthread", .{});
    model_spike.root_module.link_libc = true;
    model_spike.root_module.link_libcpp = true;

    // ── Compile And Install The Audio Process ──
    //
    // The spike imports the five supervisor-side operations from
    // `audio_process.zig`. `start` launches the sibling `audio-process`
    // executable, whose root enters the private worker side of the same module.
    // Both artifacts contain `pipewire.zig` and its narrow C boundary, so each
    // receives the identical host PipeWire configuration below.
    const audio_process = b.addExecutable(.{
        .name = "audio-process",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/audio_process.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    configurePipeWireArtifact(b, audio_process);

    const audio_spike = b.addExecutable(.{
        .name = "audio-spike",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/audio_spike.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    configurePipeWireArtifact(b, audio_spike);

    // `voiced supervisor-spike` is one executable with supervisor, fake audio,
    // and fake transcription roles. The internal roles are process entries, not
    // user commands; their deterministic work establishes the final process and
    // shared-memory contract before native model inference is attached.
    const voiced = b.addExecutable(.{
        .name = "voiced",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    voiced.root_module.addImport("models", b.createModule(.{
        .root_source_file = b.path("src/models.zig"),
    }));
    configurePipeWireArtifact(b, voiced);

    // The production role launcher contains the same resident model runtime as
    // `model-spike`. The transcription child reaches these objects only after
    // the supervisor selects its private model role, while the audio child
    // continues to use only the PipeWire portion of the executable.
    voiced.root_module.linkLibrary(ctranslate2_library);
    voiced.root_module.addObjectFile(c_boundary.getEmittedBin());
    voiced.root_module.addObjectFile(avx_kernel.getEmittedBin());
    voiced.root_module.addObjectFile(avx2_kernel.getEmittedBin());
    voiced.root_module.addObjectFile(avx512_kernel.getEmittedBin());
    voiced.root_module.addObjectFile(mkl_libraries.path(b, "libmkl_intel_ilp64.a"));
    voiced.root_module.addObjectFile(mkl_libraries.path(b, "libmkl_intel_thread.a"));
    voiced.root_module.addObjectFile(mkl_libraries.path(b, "libmkl_core.a"));
    voiced.root_module.addObjectFile(mkl_libraries.path(b, "libiomp5.a"));
    voiced.root_module.linkSystemLibrary("dl", .{});
    voiced.root_module.linkSystemLibrary("m", .{});
    voiced.root_module.linkSystemLibrary("pthread", .{});
    voiced.root_module.link_libcpp = true;

    // Installation copies the executables to `zig-out/bin`. No artifact is
    // attached to a run step, so a normal build never records or transcribes.
    b.installArtifact(model_spike);
    b.installArtifact(audio_process);
    b.installArtifact(audio_spike);
    b.installArtifact(voiced);
}

fn configurePipeWireArtifact(
    b: *std.Build,
    artifact: *std.Build.Step.Compile,
) void {
    // This host-linked boundary deliberately does not use pkg-config discovery.
    // Explicit roots serve ZLS, translate-c, and compilation consistently, while
    // the repository-owned C file retains the strict warning policy.
    artifact.root_module.addIncludePath(b.path("src"));
    artifact.root_module.addSystemIncludePath(.{
        .cwd_relative = "/usr/include/pipewire-0.3",
    });
    artifact.root_module.addSystemIncludePath(.{
        .cwd_relative = "/usr/include/spa-0.2",
    });
    artifact.root_module.addCMacro("_REENTRANT", "1");
    artifact.root_module.addCSourceFile(.{
        .file = b.path("src/audio_pipewire.c"),
        .flags = &.{
            "-std=gnu17",
            "-Weverything",
            "-Werror",
            "-pedantic-errors",
        },
    });
    artifact.root_module.addLibraryPath(.{
        .cwd_relative = "/usr/lib/x86_64-linux-gnu",
    });
    artifact.root_module.linkSystemLibrary("pipewire-0.3", .{
        .use_pkg_config = .no,
    });
    artifact.root_module.link_libc = true;
}

fn resolve_linux_x86_target(
    b: *std.Build,
    features: []const std.Target.x86.Feature,
) std.Build.ResolvedTarget {
    assert(features.len > 0);

    // Generic code stays at SSE4.1 while dispatched kernel objects add their
    // own features. All variants retain the Ubuntu 22.04 glibc baseline.
    return b.resolveTargetQuery(.{
        .cpu_arch = .x86_64,
        .cpu_model = .baseline,
        .cpu_features_add = std.Target.x86.featureSet(features),
        .os_tag = .linux,
        .glibc_version = .{ .major = 2, .minor = 35, .patch = 0 },
        .abi = .gnu,
    });
}

fn add_update_compile_flags_command(b: *std.Build) void {
    // `b.step` exposes its name as a command after `zig build`. The dependency
    // chain attached below generates the file contents, then copies them to the
    // repository only when they differ:
    //
    //   zig build update-compile-flags
    //     └─> update compile_flags.txt
    //           └─> generate compile_flags.txt
    const update_compile_flags_command = b.step(
        "update-compile-flags",
        "Update compile_flags.txt from native compiler settings in build.zig",
    );

    const ctranslate2 = b.dependency("ctranslate2", .{});

    // Zig supplies include paths directly to compilation, but clangd reads them
    // from `compile_flags.txt`. Keep the fetched dependency path relative so the
    // file remains valid from any checkout; PipeWire's system roots are the
    // fixed paths supplied by Ubuntu's libpipewire-0.3-dev package.
    const ctranslate2_include_for_clangd = std.fs.path.relative(
        b.allocator,
        ".",
        null,
        b.build_root.path orelse ".",
        ctranslate2.path("include").getPath(b),
    ) catch @panic("OOM");

    const repository_cpp_warning_flags_text = std.mem.join(
        b.allocator,
        "\n",
        &RepositoryCppCompilePolicy.warning_flags,
    ) catch @panic("OOM");

    const compile_flags_text = b.fmt(
        "{s}\n" ++
            "-Isrc\n" ++
            "-isystem\n{s}\n" ++
            "-isystem\n/usr/include/pipewire-0.3\n" ++
            "-isystem\n/usr/include/spa-0.2\n" ++
            "-D_REENTRANT=1\n" ++
            "{s}\n",
        .{
            RepositoryCppCompilePolicy.standard_flag,
            ctranslate2_include_for_clangd,
            repository_cpp_warning_flags_text,
        },
    );

    const generated_compile_flags = b.addWriteFiles().add(
        "compile_flags.txt",
        compile_flags_text,
    );

    const update_compile_flags = b.addUpdateSourceFiles();
    update_compile_flags.addCopyFileToSource(
        generated_compile_flags,
        "compile_flags.txt",
    );

    update_compile_flags_command.dependOn(&update_compile_flags.step);
}

fn add_setup_tool(b: *std.Build) *std.Build.Step.Compile {
    // Native libraries and runtime models use one table-driven host tool. Its
    // three build commands below select package tables with a subcommand; merely
    // declaring the executable does not compile or run it during `zig build`.
    const setup_tool = b.addExecutable(.{
        .name = "setup",
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/setup.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    setup_tool.root_module.addImport("models", b.createModule(.{
        .root_source_file = b.path("src/models.zig"),
    }));

    // This pure-Zig host tool does not need LLVM code generation. The native
    // backend avoids a long cold compile before the first dependency download.
    setup_tool.use_llvm = false;

    return setup_tool;
}

fn add_setup_native_command(
    b: *std.Build,
    setup_tool: *std.Build.Step.Compile,
) void {
    // Intel's ZIP-compatible wheels require extraction and path normalization
    // under `zig-pkg/mkl`; ordinary builds remain disconnected and offline.
    const run_setup_native = b.addRunArtifact(setup_tool);
    run_setup_native.addArg("native");
    run_setup_native.setCwd(b.path(""));

    const setup_native = b.step(
        "setup-native",
        "Download and verify static Intel oneMKL and OpenMP archives",
    );
    setup_native.dependOn(&run_setup_native.step);
}

fn add_setup_models_command(
    b: *std.Build,
    setup_tool: *std.Build.Step.Compile,
) void {
    // Models are runtime data under the user's XDG data directory, never build
    // inputs or implicit network dependencies of the installed binary.
    const run_setup_models = b.addRunArtifact(setup_tool);
    run_setup_models.addArg("models");
    run_setup_models.setCwd(b.path(""));

    const setup_models = b.step(
        "setup-models",
        "Download and verify the named CTranslate2 Whisper models",
    );
    setup_models.dependOn(&run_setup_models.step);
}

fn add_setup_all_command(
    b: *std.Build,
    setup_tool: *std.Build.Step.Compile,
) void {
    // One process and HTTP client install both independently published package
    // sets while preserving the explicit, offline default build.
    const run_setup_all = b.addRunArtifact(setup_tool);
    run_setup_all.addArg("all");
    run_setup_all.setCwd(b.path(""));

    const setup_all = b.step(
        "setup",
        "Download and verify native dependencies and CTranslate2 models",
    );
    setup_all.dependOn(&run_setup_all.step);
}
