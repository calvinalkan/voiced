//! Owns one PipeWire input stream and writes its audio into a fixed shared
//! exchange. PipeWire state and format callbacks run on the main-loop thread;
//! with realtime processing enabled, `processAudio` runs concurrently on
//! PipeWire's data thread. The realtime callback performs only bounded
//! validation, copies, atomic publication, buffer return, and one nonblocking
//! eventfd signal. The main-loop thread owns diagnostics and stream teardown.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const assert = std.debug.assert;

// These calls use Zig's direct Linux syscall wrappers even though this binary
// links libc. Their return values must be decoded with `linux.errno`; using
// `std.posix.errno` would instead consult libc errno, which direct syscalls do
// not set.
const linux = std.os.linux;
const AudioExchange = audio_exchange.AudioExchange;
const AudioSlot = audio_exchange.AudioSlot;

const pipewire = @cImport({
    @cInclude("audio_pipewire.h");
});

const callback_error_message_capacity = 512;
pub const setup_error_message_capacity = 4096;
pub const pipewire_version_capacity =
    pipewire.VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY;
pub const source_identity_text_capacity =
    pipewire.VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY;
pub const source_catalog_capacity =
    pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY;

pub const TimelineValidation = enum(u32) {
    header_only,
    full,
};

/// One supervisor command ends an active recording. `stop` retains every
/// complete callback block, including the final partially filled exchange slot;
/// `cancel` abandons that private final slot and tells the supervisor to discard
/// every earlier publication belonging to the session.
pub const ControlCommand = enum(u32) {
    stop,
    cancel,
};

/// Fixed control-socket record shared by the supervisor and audio worker.
/// Reserved bytes make accidental protocol drift an asserted internal defect.
pub const ControlPacket = extern struct {
    command: u32,
    reserved: u32,
};

pub const Launch = struct {
    exchange: *AudioExchange,
    control_socket: std.posix.fd_t,
    target: ?[:0]const u8,
    configured_device_serial: ?[:0]const u8,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    process_realtime: bool,
};

/// Classifies why capture stopped so the supervisor can choose policy without
/// parsing diagnostic text. `completed`, `stopped`, and `cancelled` are expected
/// terminal states and therefore carry no `RuntimeError`; every other value does.
pub const Outcome = enum(u32) {
    completed,
    stopped,
    cancelled,
    control_error,
    pipeline_full,
    format_parse_error,
    unexpected_format,
    buffer_configuration_error,
    timeline_error,
    timeline_discontinuity,
    stream_error,
    stream_disconnected,
    source_disconnected,
    source_changed,
    source_observation_error,
    invalid_buffer,
    corrupted_buffer,
    buffer_return_error,
};

/// Returns the first terminal capture cause and all stable measurements after
/// the stream has stopped. `runtime_error` explains failure outcomes and remains
/// empty for completion, normal stop, or cancellation. `teardown_error`
/// independently reports a later disconnect error: teardown must never erase
/// the event that actually stopped capture or discard audio retained before it.
pub const Report = struct {
    outcome: Outcome,
    runtime_error: RuntimeError,
    error_message: [callback_error_message_capacity]u8,
    error_message_size: u16,
    teardown_error: RuntimeError,
    teardown_error_message: [callback_error_message_capacity]u8,
    teardown_error_message_size: u16,

    timeline_validation: TimelineValidation,
    pipewire_headers_version: [pipewire_version_capacity]u8,
    pipewire_headers_version_size: u8,
    pipewire_library_version: [pipewire_version_capacity]u8,
    pipewire_library_version_size: u8,
    pipewire_server_version: [pipewire_version_capacity]u8,
    pipewire_server_version_size: u8,

    source_identity: SourceIdentity,
    negotiated_sample_rate_hz: u32,
    negotiated_channels_count: u32,
    // Accepted callback samples include a cancelled final private slot. The
    // published count includes only samples release-published to consumers; the
    // two counts are equal for every outcome except cancellation.
    samples_count: u32,
    published_samples_count: u32,
    slot_publications_count: u32,
    shared_memory_is_locked: bool,
    shared_memory_lock_error_code: i32,
    shared_memory_lock_limit_bytes: u64,

    main_loop_thread_id: i32,
    callback_thread_id: i32,
    callback_scheduler_policy: i32,
    callback_scheduler_priority: i32,

    callbacks_count: u64,
    missing_buffers_count: u32,
    clipped_samples_count: u32,
    header_metadata_buffers_count: u32,
    header_gap_buffers_count: u32,
    header_gap_samples_count: u32,
    block_samples_count_min: u32,
    block_samples_count_max: u32,
    callback_duration_ns_max: u64,
    callback_gap_ns_max: u64,
};

pub const SetupErrorStage = enum(u32) {
    none,
    file_descriptor_limits_query,
    file_descriptor_limit,
    memory_lock_limits_query,
    main_loop_create,
    callback_event_create,
    callback_event_register,
    control_socket_register,
    properties_create,
    property_config,
    property_media_type,
    property_media_category,
    property_media_role,
    property_target,
    stream_create,
    source_resolution,
    source_resolution_main_loop_create,
    source_resolution_context_create,
    source_resolution_core_connect,
    source_resolution_core_observer_register,
    source_resolution_registry_create,
    source_resolution_registry_observer_register,
    source_resolution_node_bind,
    source_resolution_device_bind,
    source_resolution_sync,
    source_resolution_main_loop,
    source_observer_create,
    source_registry_create,
    source_observer_register,
    server_observer_register,
    stream_connect,
};

/// Defines how to interpret an error's numeric code. Native domains retain the
/// exact Zig error value, errno, or negative PipeWire result. `spa_buffer` and
/// `voiced_audio` use the stable enums below; `boundary_validation` uses the C
/// boundary's stable enum; `resource_limit` carries the observed soft limit.
/// PipeWire state callbacks provide text but no numeric result, so
/// `pipewire_callback` legitimately uses code zero.
pub const ErrorDomain = enum(u32) {
    none,
    zig_error,
    linux_errno,
    libc_errno,
    pipewire_result,
    pipewire_callback,
    boundary_validation,
    resource_limit,
    spa_buffer,
    voiced_audio,
};

/// Names the capture operation that observed a runtime or teardown error.
pub const RuntimeErrorStage = enum(u32) {
    none,
    main_loop,
    callback_event,
    format_negotiation,
    buffer_negotiation,
    stream_state,
    source_identity,
    supervisor_control,
    stream_disconnect,
    buffer_metadata,
    buffer_data,
    sample_validation,
    timeline_continuity,
    exchange_publication,
    buffer_return,
};

/// Stable codes for Voiced policy errors that have no native code.
pub const SourceResolutionErrorCode = enum(u32) {
    none,
    configured_device_not_found,
    configured_device_ambiguous,
};

pub const AudioErrorCode = enum(u32) {
    none,
    callback_event_source_error,
    callback_event_short_read,
    format_parameter_removed,
    unexpected_format,
    exchange_full,
    control_socket_source_error,
    control_socket_closed,
    source_link_removed,
    source_node_removed,
    source_device_removed,
    source_changed,
};

/// Machine-readable coordinates for one audio error. An empty coordinate is
/// exactly `{ none, none, 0 }`; a present error has a non-none stage and domain.
pub const RuntimeError = struct {
    stage: RuntimeErrorStage,
    domain: ErrorDomain,
    code: i64,
};

/// Identifies the first concrete PipeWire source linked during one recording.
/// Global IDs and object serials diagnose that graph instance; `device_serial`
/// is the stable hardware identity expected to survive unplug and replug.
pub const SourceIdentity = struct {
    is_resolved: bool,
    node_id: u32,
    node_object_serial: u64,
    device_id: u32,
    device_object_serial: u64,
    node_name: [source_identity_text_capacity]u8,
    node_name_size: u16,
    node_description: [source_identity_text_capacity]u8,
    node_description_size: u16,
    device_serial: [source_identity_text_capacity]u8,
    device_serial_size: u16,
    device_description: [source_identity_text_capacity]u8,
    device_description_size: u16,
};

/// Describes an error returned before a runtime `Report` can be constructed.
pub const SetupError = struct {
    stage: SetupErrorStage,
    domain: ErrorDomain,
    code: i64,
    message: [setup_error_message_capacity]u8,
    message_size: u16,
    pipewire_version: [pipewire_version_capacity]u8,
    pipewire_version_size: u8,
};

const Audio = struct {
    main_loop: *pipewire.pw_main_loop,
    stream: ?*pipewire.pw_stream,
    callback_event_fd: std.posix.fd_t,
    callback_event_source: ?*pipewire.spa_source,
    control_socket: std.posix.fd_t,
    control_socket_source: ?*pipewire.spa_source,
    server_observer: pipewire.voiced_audio_pipewire_server_observer,
    source_observer: ?*pipewire.voiced_audio_pipewire_source_observer,
    source_identity: pipewire.voiced_audio_pipewire_source_identity,

    exchange: *AudioExchange,
    target: ?[:0]const u8,
    recording_samples_target: u32,
    slot_samples_boundary: u32,
    process_realtime: bool,
    timeline_validation: TimelineValidation,

    disconnect_is_expected: bool,
    source_is_linked: *std.atomic.Value(bool),
    negotiated_format_is_accepted: *std.atomic.Value(bool),
    negotiated_sample_rate_hz: u32,
    negotiated_channels_count: u32,

    error_message: [callback_error_message_capacity]u8,
    error_message_size: u16,
    runtime_error: RuntimeError,
    teardown_error_message: [callback_error_message_capacity]u8,
    teardown_error_message_size: u16,
    teardown_error: RuntimeError,

    realtime: RealtimeCapture,
};

const RealtimeCapture = struct {
    stream: ?*pipewire.pw_stream,
    exchange: *AudioExchange,
    source_is_linked: *std.atomic.Value(bool),
    negotiated_format_is_accepted: *std.atomic.Value(bool),
    callback_event_fd: std.posix.fd_t,
    timeline_validation: TimelineValidation,

    recording_samples_target: u32,
    slot_samples_boundary: u32,
    active_slot_index: ?u8,
    active_slot_samples_count: u32,
    next_publication_ordinal: u32,
    slot_publications_count: u32,
    samples_count: u32,
    published_samples_count: u32,

    terminal_outcome: std.atomic.Value(TerminalOutcome),
    runtime_error: RuntimeError,
    buffer_error: BufferError,
    timeline_buffer_ticks_previous: ?u64,
    timeline_graph_rate_num_previous: u32,
    timeline_graph_rate_denom_previous: u32,
    timeline_failure: TimelineFailure,
    queue_buffer_result: i32,

    main_loop_thread_id: i32,
    callback_thread_id: i32,
    callback_scheduler_policy: i32,
    callback_scheduler_priority: i32,

    callbacks_count: u64,
    missing_buffers_count: u32,
    clipped_samples_count: u32,
    header_metadata_buffers_count: u32,
    header_gap_buffers_count: u32,
    header_gap_samples_count: u32,
    block_samples_count_min: u32,
    block_samples_count_max: u32,
    callback_started_ns_previous: u64,
    callback_duration_ns_max: u64,
    callback_gap_ns_max: u64,
};

const TerminalOutcome = enum(u8) {
    none,
    completed,
    stopped,
    cancelled,
    control_error,
    pipeline_full,
    format_parse_error,
    unexpected_format,
    buffer_configuration_error,
    timeline_error,
    timeline_discontinuity,
    stream_error,
    stream_disconnected,
    source_disconnected,
    source_changed,
    source_observation_error,
    invalid_buffer,
    corrupted_buffer,
    buffer_return_error,
};

/// Stable `voiced_audio` codes for a malformed or discontinuous PipeWire graph
/// timeline. The runtime stage disambiguates these from other Voiced codes.
pub const TimelineErrorCode = enum(u32) {
    none,
    invalid_rate,
    invalid_graph_time,
    buffer_from_future,
    tick_underflow,
    rate_changed,
    ticks_regressed,
    discontinuity,
};

/// Stable `spa_buffer` codes for external metadata, geometry, and sample data
/// rejected on PipeWire's realtime callback thread.
pub const BufferErrorCode = enum(u32) {
    none,
    missing_spa_buffer,
    metadata_count_exceeds_realtime_limit,
    missing_metadata_array,
    duplicate_header_metadata,
    undersized_header_metadata,
    missing_header_metadata_data,
    misaligned_header_metadata_data,
    unsupported_header_flags,
    header_corrupted,
    timeline_discontinuity,
    unexpected_data_planes,
    missing_data_planes,
    missing_chunk,
    chunk_corrupted,
    missing_mapped_data,
    unreadable_data,
    empty_data_capacity,
    invalid_stride,
    incomplete_sample,
    misaligned_sample_offset,
    non_finite_sample,
    block_exceeds_realtime_limit,
};

const TimelineFailure = struct {
    query_result: i32,
    code: TimelineErrorCode,
    graph_now_ns: i64,
    graph_rate_num: u32,
    graph_rate_denom: u32,
    graph_ticks: u64,
    buffer_cycle_ns: u64,
    buffer_ticks_previous: u64,
    buffer_ticks_current: u64,
    block_samples_count: u32,
};

const BufferError = struct {
    reason: BufferErrorCode,
    data_planes_count: u32,
    data_flags: u32,
    data_max_size: u32,
    metadata_count: u32,
    metadata_size: u32,
    header_flags: u32,
    header_sequence: u64,
    header_presentation_timestamp_ns: i64,
    chunk_offset: u32,
    chunk_size: u32,
    chunk_stride: i32,
    invalid_sample_index: u32,
};

const BorrowedAudioBlock = struct {
    first_bytes: []const u8,
    second_bytes: []const u8,
    samples_count: u32,
    is_silence: bool,
    header_marked_gap: bool,
};

/// `run` initializes one stream, captures complete PipeWire callback blocks into
/// the supplied exchange, and returns only after callbacks have stopped. The
/// caller must initialize the exchange, keep its mapping and control socket alive
/// for the complete call, and send at most one `ControlPacket`. Reaching
/// `recording_samples_target` may exceed the target by one callback block.
/// Normal stop publishes the final partially filled slot; cancel abandons it.
pub fn run(launch: Launch, setup_error: *SetupError) !Report {
    assert(setup_error.stage == .none);
    assert(setup_error.domain == .none);
    assert(setup_error.code == 0);
    assert(setup_error.message_size == 0);
    assert(setup_error.pipewire_version_size == 0);
    assert(launch.exchange.version == audio_exchange.format_version);
    assert(launch.exchange.generation > 0);
    assert(launch.control_socket >= 0);
    assert(launch.target == null or launch.configured_device_serial == null);
    if (launch.target) |target| {
        assert(target.len > 0);
        assert(target.len < source_identity_text_capacity);
    }
    if (launch.configured_device_serial) |device_serial| {
        assert(device_serial.len > 0);
        assert(device_serial.len < source_identity_text_capacity);
    }
    assert(launch.recording_samples_target > 0);
    assert(launch.recording_samples_target <= 90 * audio_exchange.sample_rate_hz);
    assert(launch.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    assert(launch.slot_samples_boundary <= audio_exchange.slot_samples_capacity);
    assert(launch.exchange.reserved == 0);
    assert(launch.exchange.audio_callbacks_count == 0);
    assert(launch.exchange.audio_samples_count == 0);
    assert(launch.exchange.reserved_2 == 0);

    for (&launch.exchange.slots) |*slot| {
        assert(slot.state == audio_exchange.slot_state_available);
        assert(slot.samples_count == 0);
        assert(slot.generation == 0);
        assert(slot.publication_ordinal == 0);
        assert(slot.reserved == 0);
    }

    // PipeWire creates loop pollers, eventfds, a protocol socket, local memfds,
    // and imported buffer descriptors during setup. On PipeWire 1.0.5, limits
    // in the teens fail at different internal stages; some asynchronous errors
    // never become terminal stream events. Requiring a soft limit of sixty-four
    // is a conservative policy for this otherwise descriptor-clean worker, not
    // a PipeWire API minimum, and does not replace the setup deadline.
    const pipewire_file_descriptor_limit_min: u64 = 64;

    const file_descriptor_limits = std.posix.getrlimit(.NOFILE) catch |limit_error| {
        recordSetupError(
            setup_error,
            .file_descriptor_limits_query,
            .zig_error,
            @intFromError(limit_error),
            &.{},
            "getrlimit(RLIMIT_NOFILE) returned {s}",
            .{@errorName(limit_error)},
        );
        return limit_error;
    };

    if (file_descriptor_limits.cur < pipewire_file_descriptor_limit_min) {
        recordSetupError(
            setup_error,
            .file_descriptor_limit,
            .resource_limit,
            @intCast(file_descriptor_limits.cur),
            &.{},
            "RLIMIT_NOFILE soft limit is {d}; Voiced requires at least {d}",
            .{ file_descriptor_limits.cur, pipewire_file_descriptor_limit_min },
        );
        return error.PipeWireFileDescriptorLimitTooLow;
    }

    const memory_lock_limits = std.posix.getrlimit(.MEMLOCK) catch |limit_error| {
        recordSetupError(
            setup_error,
            .memory_lock_limits_query,
            .zig_error,
            @intFromError(limit_error),
            &.{},
            "getrlimit(RLIMIT_MEMLOCK) returned {s}",
            .{@errorName(limit_error)},
        );
        return limit_error;
    };

    // Touch every shared page before PipeWire can enter the realtime callback.
    // `mlock` remains best effort: an unlocked exchange is still safe, but the
    // kernel may reclaim its pages under pressure and delay a later callback.
    // Retain both errno and the inherited soft limit so the supervisor can warn
    // without guessing whether the mapping exceeded policy or permission was
    // denied for another reason.
    const exchange_bytes = std.mem.asBytes(launch.exchange);
    var page_offset: usize = 0;
    while (page_offset < exchange_bytes.len) : (page_offset += std.heap.page_size_min) {
        const page_byte: *volatile u8 = @ptrCast(&exchange_bytes[page_offset]);
        page_byte.* = page_byte.*;
    }
    const memory_lock_result = linux.mlock(exchange_bytes.ptr, exchange_bytes.len);
    const memory_lock_errno = linux.errno(memory_lock_result);
    const shared_memory_is_locked = memory_lock_errno == .SUCCESS;
    const shared_memory_lock_error_code: i32 = if (shared_memory_is_locked)
        0
    else
        @intFromEnum(memory_lock_errno);
    defer if (shared_memory_is_locked) {
        // The mapping disappears with this isolated worker regardless. A rare
        // cleanup error cannot be repaired here and must not erase an already
        // completed audio report by turning teardown into an assertion crash.
        _ = linux.munlock(exchange_bytes.ptr, exchange_bytes.len);
    };

    pipewire.pw_init(null, null);
    defer pipewire.pw_deinit();

    // The C boundary intersects the headers used to build this executable with
    // the library loaded on this machine. PipeWire 1.0.5 provides the exact
    // cycle timestamp needed for full per-buffer continuity checks; older
    // installations continue in Header-only mode without touching that newer
    // struct field. Keep both versions in every successful report so an issue
    // does not mistake build capability for runtime capability.
    var native_environment: pipewire.voiced_audio_pipewire_environment = undefined;
    pipewire.voiced_audio_pipewire_environment_read(&native_environment);
    assert(native_environment.headers_version_size > 0);
    assert(native_environment.headers_version_size <=
        native_environment.headers_version.len);
    assert(native_environment.library_version_size > 0);
    assert(native_environment.library_version_size <=
        native_environment.library_version.len);
    const timeline_validation: TimelineValidation =
        switch (native_environment.timeline_validation) {
            pipewire.VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_HEADER_ONLY => .header_only,
            pipewire.VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_FULL => .full,
            else => unreachable,
        };

    // Every synchronous C wrapper clears this output before doing native work,
    // leaves it empty on success, and captures the native code and owned message
    // before returning an error. Reusing one value makes that boundary contract
    // visible without separating a failed call from a later errno lookup.
    var native_error: pipewire.voiced_audio_pipewire_error = undefined;

    // A configured physical microphone is named by stable Device serial, not by
    // a PipeWire graph ID or node name. Inventory the current graph before the
    // capture stream exists, join each source Node to its Device, and resolve
    // exactly one current node name. Failure therefore never opens the default
    // microphone, and every new recording naturally re-resolves after hotplug.
    var configured_source_resolution = std.mem.zeroes(
        pipewire.voiced_audio_pipewire_source_resolution,
    );
    const capture_target: ?[:0]const u8 = if (launch.configured_device_serial) |device_serial| configured: {
        if (!pipewire.voiced_audio_pipewire_source_resolve_device_serial(
            device_serial.ptr,
            &configured_source_resolution,
            &native_error,
        )) {
            const error_stage: SetupErrorStage = switch (native_error.stage) {
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_MAIN_LOOP_CREATE => .source_resolution_main_loop_create,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CONTEXT_CREATE => .source_resolution_context_create,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CORE_CONNECT => .source_resolution_core_connect,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CORE_OBSERVER_REGISTER => .source_resolution_core_observer_register,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_REGISTRY_CREATE => .source_resolution_registry_create,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_REGISTRY_OBSERVER_REGISTER => .source_resolution_registry_observer_register,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_NODE_BIND => .source_resolution_node_bind,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_DEVICE_BIND => .source_resolution_device_bind,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_SYNC => .source_resolution_sync,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_MAIN_LOOP_RUN => .source_resolution_main_loop,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION => .source_resolution,
                else => unreachable,
            };
            const error_domain: ErrorDomain = switch (native_error.domain) {
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO => .libc_errno,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT => .pipewire_result,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_CALLBACK => .pipewire_callback,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION => .boundary_validation,
                else => unreachable,
            };
            assert(native_error.message_size <= native_error.message.len);
            recordSetupError(
                setup_error,
                error_stage,
                error_domain,
                native_error.code,
                native_error.message[0..native_error.message_size],
                "configured microphone discovery failed during {s}",
                .{@tagName(error_stage)},
            );
            return error.PipeWireConfiguredSourceResolutionFailed;
        }
        assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
        assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
        assert(native_error.code == 0);
        assert(native_error.message_size == 0);
        assert(configured_source_resolution.sources_count <= source_catalog_capacity);
        assert(configured_source_resolution.matches_count <=
            configured_source_resolution.sources_count);
        assert(configured_source_resolution.reserved == 0);

        var available_sources: [3072]u8 = undefined;
        var available_sources_size: usize = 0;
        var available_sources_listed: u32 = 0;
        const source_lines_capacity = available_sources.len - 96;
        for (configured_source_resolution.sources[0..configured_source_resolution.sources_count]) |source| {
            assert(source.is_resolved == 1);
            assert(source.node_name_size <= source.node_name.len);
            assert(source.node_description_size <= source.node_description.len);
            assert(source.device_serial_size <= source.device_serial.len);
            assert(source.device_description_size <= source.device_description.len);
            const source_name = if (source.node_description_size > 0)
                source.node_description[0..source.node_description_size]
            else if (source.node_name_size > 0)
                source.node_name[0..source.node_name_size]
            else
                "unnamed PipeWire source";
            const source_device_serial = if (source.device_serial_size > 0)
                source.device_serial[0..source.device_serial_size]
            else
                "not provided";
            const line = std.fmt.bufPrint(
                available_sources[available_sources_size..source_lines_capacity],
                "\n  - {s} (node={s}, device.serial={s})",
                .{
                    source_name,
                    source.node_name[0..source.node_name_size],
                    source_device_serial,
                },
            ) catch break;
            available_sources_size += line.len;
            available_sources_listed += 1;
        }
        if (available_sources_listed < configured_source_resolution.sources_count) {
            const omitted = std.fmt.bufPrint(
                available_sources[available_sources_size..],
                "\n  - ... {d} additional sources omitted",
                .{configured_source_resolution.sources_count - available_sources_listed},
            ) catch unreachable;
            available_sources_size += omitted.len;
        }
        if (available_sources_size == 0) {
            const none = std.fmt.bufPrint(
                &available_sources,
                " none",
                .{},
            ) catch unreachable;
            available_sources_size = none.len;
        }

        switch (configured_source_resolution.status) {
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_NOT_FOUND => {
                assert(configured_source_resolution.matches_count == 0);
                recordSetupError(
                    setup_error,
                    .source_resolution,
                    .voiced_audio,
                    @intFromEnum(SourceResolutionErrorCode.configured_device_not_found),
                    &.{},
                    "no available PipeWire source belongs to configured device.serial={s}; available sources:{s}",
                    .{
                        device_serial,
                        available_sources[0..available_sources_size],
                    },
                );
                return error.ConfiguredMicrophoneNotFound;
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_AMBIGUOUS => {
                assert(configured_source_resolution.matches_count > 1);
                recordSetupError(
                    setup_error,
                    .source_resolution,
                    .voiced_audio,
                    @intFromEnum(SourceResolutionErrorCode.configured_device_ambiguous),
                    &.{},
                    "configured device.serial={s} exposes {d} available PipeWire sources; available sources:{s}",
                    .{
                        device_serial,
                        configured_source_resolution.matches_count,
                        available_sources[0..available_sources_size],
                    },
                );
                return error.ConfiguredMicrophoneAmbiguous;
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_RESOLVED => {},
            else => unreachable,
        }

        const resolved_source = &configured_source_resolution.resolved_source;
        assert(configured_source_resolution.matches_count == 1);
        assert(resolved_source.is_resolved == 1);
        assert(resolved_source.node_name_size > 0);
        assert(resolved_source.node_name_size < resolved_source.node_name.len);
        assert(resolved_source.node_name[resolved_source.node_name_size] == 0);
        assert(resolved_source.device_serial_size == device_serial.len);
        assert(std.mem.eql(
            u8,
            resolved_source.device_serial[0..resolved_source.device_serial_size],
            device_serial,
        ));
        break :configured resolved_source.node_name[0..resolved_source.node_name_size :0];
    } else launch.target;

    const main_loop_optional = pipewire.voiced_audio_pipewire_main_loop_create(
        &native_error,
    );
    if (main_loop_optional == null) {
        assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_CREATE);
        assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO);
        assert(native_error.code >= 0);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            .main_loop_create,
            .libc_errno,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "pw_main_loop_new returned null",
            .{},
        );

        return error.PipeWireMainLoopCreationFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    const main_loop = main_loop_optional.?;
    defer pipewire.pw_main_loop_destroy(main_loop);

    const callback_event_result = linux.eventfd(
        0,
        linux.EFD.CLOEXEC | linux.EFD.NONBLOCK,
    );

    const callback_event_errno = linux.errno(callback_event_result);

    if (callback_event_errno != .SUCCESS) {
        const error_code: i32 = @intFromEnum(callback_event_errno);
        recordSetupError(
            setup_error,
            .callback_event_create,
            .linux_errno,
            error_code,
            &.{},
            "eventfd could not create the realtime callback wake descriptor: errno={s}",
            .{@tagName(callback_event_errno)},
        );
        return error.CallbackEventCreationFailed;
    }
    const callback_event_fd: std.posix.fd_t = @intCast(callback_event_result);
    defer closeFileDescriptor(callback_event_fd);

    var source_is_linked = std.atomic.Value(bool).init(false);
    var negotiated_format_is_accepted = std.atomic.Value(bool).init(false);
    var audio: Audio = .{
        .main_loop = main_loop,
        .stream = null,
        .callback_event_fd = callback_event_fd,
        .callback_event_source = null,
        .control_socket = launch.control_socket,
        .control_socket_source = null,
        .server_observer = std.mem.zeroes(
            pipewire.voiced_audio_pipewire_server_observer,
        ),
        .source_observer = null,
        .source_identity = std.mem.zeroes(
            pipewire.voiced_audio_pipewire_source_identity,
        ),
        .exchange = launch.exchange,
        .target = capture_target,
        .recording_samples_target = launch.recording_samples_target,
        .slot_samples_boundary = launch.slot_samples_boundary,
        .process_realtime = launch.process_realtime,
        .timeline_validation = timeline_validation,
        .disconnect_is_expected = false,
        .source_is_linked = &source_is_linked,
        .negotiated_format_is_accepted = &negotiated_format_is_accepted,
        .negotiated_sample_rate_hz = 0,
        .negotiated_channels_count = 0,
        .error_message = @splat(0),
        .error_message_size = 0,
        .runtime_error = .{
            .stage = .none,
            .domain = .none,
            .code = 0,
        },
        .teardown_error_message = @splat(0),
        .teardown_error_message_size = 0,
        .teardown_error = .{
            .stage = .none,
            .domain = .none,
            .code = 0,
        },
        .realtime = .{
            .stream = null,
            .exchange = launch.exchange,
            .source_is_linked = &source_is_linked,
            .negotiated_format_is_accepted = &negotiated_format_is_accepted,
            .callback_event_fd = callback_event_fd,
            .timeline_validation = timeline_validation,
            .recording_samples_target = launch.recording_samples_target,
            .slot_samples_boundary = launch.slot_samples_boundary,
            .active_slot_index = null,
            .active_slot_samples_count = 0,
            .next_publication_ordinal = 0,
            .slot_publications_count = 0,
            .samples_count = 0,
            .published_samples_count = 0,
            .terminal_outcome = .init(.none),
            .runtime_error = .{
                .stage = .none,
                .domain = .none,
                .code = 0,
            },
            .buffer_error = .{
                .reason = .none,
                .data_planes_count = 0,
                .data_flags = 0,
                .data_max_size = 0,
                .metadata_count = 0,
                .metadata_size = 0,
                .header_flags = 0,
                .header_sequence = 0,
                .header_presentation_timestamp_ns = 0,
                .chunk_offset = 0,
                .chunk_size = 0,
                .chunk_stride = 0,
                .invalid_sample_index = 0,
            },
            .timeline_buffer_ticks_previous = null,
            .timeline_graph_rate_num_previous = 0,
            .timeline_graph_rate_denom_previous = 0,
            .timeline_failure = std.mem.zeroes(TimelineFailure),
            .queue_buffer_result = 0,
            .main_loop_thread_id = linux.gettid(),
            .callback_thread_id = 0,
            .callback_scheduler_policy = -1,
            .callback_scheduler_priority = -1,
            .callbacks_count = 0,
            .missing_buffers_count = 0,
            .clipped_samples_count = 0,
            .header_metadata_buffers_count = 0,
            .header_gap_buffers_count = 0,
            .header_gap_samples_count = 0,
            .block_samples_count_min = std.math.maxInt(u32),
            .block_samples_count_max = 0,
            .callback_started_ns_previous = 0,
            .callback_duration_ns_max = 0,
            .callback_gap_ns_max = 0,
        },
    };

    // The eventfd is the only callback-to-main-loop wake path. The realtime
    // thread writes one fixed integer after publishing a terminal outcome; this
    // source drains that integer and stops the loop on its owning thread.
    const pipewire_loop = pipewire.pw_main_loop_get_loop(main_loop);
    const callback_event_source_optional =
        pipewire.voiced_audio_pipewire_callback_event_register(
            pipewire_loop,
            callback_event_fd,
            pipewire.SPA_IO_IN | pipewire.SPA_IO_ERR | pipewire.SPA_IO_HUP,
            callbackEventReceived,
            &audio,
            &native_error,
        );
    if (callback_event_source_optional == null) {
        assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CALLBACK_EVENT_REGISTER);
        assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO);
        assert(native_error.code >= 0);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            .callback_event_register,
            .libc_errno,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "pw_loop_add_io returned null for the callback eventfd",
            .{},
        );
        return error.CallbackEventRegistrationFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    const callback_event_source = callback_event_source_optional.?;
    audio.callback_event_source = callback_event_source;
    defer pipewire.voiced_audio_pipewire_loop_io_unregister(
        pipewire_loop,
        callback_event_source,
    );

    // The worker receives its launch packet before entering `run`, then leaves
    // the same SOCK_SEQPACKET endpoint with this loop. A queued stop or cancel
    // therefore wakes the PipeWire owner thread without polling from the
    // realtime callback or introducing a second control thread.
    const control_socket_source_optional =
        pipewire.voiced_audio_pipewire_control_socket_register(
            pipewire_loop,
            launch.control_socket,
            pipewire.SPA_IO_IN | pipewire.SPA_IO_ERR | pipewire.SPA_IO_HUP,
            controlSocketEventReceived,
            &audio,
            &native_error,
        );
    if (control_socket_source_optional == null) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CONTROL_SOCKET_REGISTER);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO);
        assert(native_error.code >= 0);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            .control_socket_register,
            .libc_errno,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "pw_loop_add_io could not register the supervisor control socket",
            .{},
        );
        return error.ControlSocketRegistrationFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    const control_socket_source = control_socket_source_optional.?;
    audio.control_socket_source = control_socket_source;
    defer pipewire.voiced_audio_pipewire_loop_io_unregister(
        pipewire_loop,
        control_socket_source,
    );

    // C owns stream property construction as one semantic operation. Its error
    // stage still identifies the exact failed property or constructor without
    // exposing each `pw_properties_*` call as a separate Zig-facing API.
    var stream_callbacks: pipewire.voiced_audio_pipewire_stream_callbacks = .{
        .context = &audio,
        .state_changed = streamStateChanged,
        .format_changed = streamFormatChanged,
        .process = processAudio,
    };

    // This cleanup is declared before stream destruction so every callback has
    // stopped before an otherwise unpublished slot is returned to the exchange.
    // Normal terminal paths publish their final non-empty slot explicitly.
    defer abandonUnpublishedActiveSlot(&audio.realtime);

    const stream_optional = pipewire.voiced_audio_pipewire_capture_stream_create(
        pipewire_loop,
        if (capture_target) |target| target.ptr else null,
        launch.process_realtime,
        &stream_callbacks,
        &native_error,
    );
    if (stream_optional == null) {
        const error_stage: SetupErrorStage = switch (native_error.stage) {
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTIES_CREATE => .properties_create,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_CONFIG => .property_config,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_TYPE => .property_media_type,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_CATEGORY => .property_media_category,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_ROLE => .property_media_role,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_TARGET => .property_target,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CREATE => .stream_create,
            else => unreachable,
        };
        const error_domain: ErrorDomain = switch (native_error.domain) {
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO => .libc_errno,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT => .pipewire_result,
            else => unreachable,
        };
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            error_stage,
            error_domain,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "capture stream creation failed during {s}",
            .{@tagName(error_stage)},
        );
        return error.PipeWireCaptureStreamCreationFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);
    const stream = stream_optional.?;
    audio.stream = stream;
    audio.realtime.stream = stream;
    assert(audio.stream == stream);
    assert(audio.realtime.stream == stream);
    var stream_is_alive = true;
    defer if (stream_is_alive) pipewire.pw_stream_destroy(stream);

    // Claim the first physical slot before connecting. PipeWire may start its
    // data thread as soon as the stream becomes active, so the callback must
    // never observe an exchange without one producer-owned destination.
    // The worker starts from the exchange asserted above, so the initial slot
    // claim cannot encounter consumer-owned audio. Later claims may fail
    // normally when all three slots are still published to the supervisor.
    assert(beginNextAvailableSlot(&audio.realtime));

    // WARNING: a zero result means only that PipeWire accepted this asynchronous
    // connection request. It does not prove that a target was linked, a Format
    // was negotiated, the stream reached `STREAMING`, or a process callback will
    // ever arrive. Descriptor exhaustion and an unresponsive hardware source
    // have both produced successful calls followed by no progress. The
    // supervising process must therefore enforce separate setup and callback-
    // progress deadlines; this worker must never treat this return as readiness.
    const connect_result = pipewire.voiced_audio_pipewire_capture_stream_connect(
        stream,
        audio_exchange.sample_rate_hz,
        audio_exchange.callback_samples_count_max,
        launch.process_realtime,
        &native_error,
    );
    if (connect_result < 0) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CONNECT);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT);
        assert(native_error.code == connect_result);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            .stream_connect,
            .pipewire_result,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "pw_stream_connect rejected the asynchronous input request",
            .{},
        );
        return error.PipeWireStreamConnectionFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    // A PipeWire target is only a routing request. Real WirePlumber graphs can
    // replace the requested microphone's Link with one from the new default
    // source after unplug, even while `node.dont-reconnect` remains true. The
    // stream exposes its private core only after connect, but no graph callback
    // can run until the loop starts below. Register in that window so the first
    // incoming Link locks one source and opens the realtime copy gate.
    var source_callbacks: pipewire.voiced_audio_pipewire_source_callbacks = .{
        .context = &audio,
        .event = sourceEventReceived,
    };
    const source_observer_optional =
        pipewire.voiced_audio_pipewire_capture_source_observer_create(
            stream,
            if (launch.configured_device_serial) |device_serial|
                device_serial.ptr
            else
                null,
            &source_callbacks,
            &native_error,
        );
    if (source_observer_optional == null) {
        const error_stage: SetupErrorStage = switch (native_error.stage) {
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_CREATE => .source_observer_create,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_REGISTRY_CREATE => .source_registry_create,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_REGISTER => .source_observer_register,
            else => unreachable,
        };
        const error_domain: ErrorDomain = switch (native_error.domain) {
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO => .libc_errno,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT => .pipewire_result,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION => .boundary_validation,
            else => unreachable,
        };
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            error_stage,
            error_domain,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "capture source observation failed during {s}",
            .{@tagName(error_stage)},
        );
        return error.PipeWireSourceObserverCreationFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);
    const source_observer = source_observer_optional.?;
    audio.source_observer = source_observer;
    var source_observer_is_alive = true;
    defer if (source_observer_is_alive) {
        pipewire.voiced_audio_pipewire_capture_source_observer_destroy(
            source_observer,
        );
    };

    // The linked client-library version says which local ABI is loaded; it does
    // not identify the remote PipeWire daemon. Connecting creates the stream's
    // private core, but no asynchronous core event can run before the loop below.
    // Register now so that first event retains the server version in fixed
    // worker-owned storage for the final report.
    const server_observer_result =
        pipewire.voiced_audio_pipewire_capture_server_observer_register(
            stream,
            &audio.server_observer,
            &native_error,
        );
    if (server_observer_result < 0) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SERVER_OBSERVER_REGISTER);
        const error_domain: ErrorDomain = switch (native_error.domain) {
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT => .pipewire_result,
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION => .boundary_validation,
            else => unreachable,
        };
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordSetupError(
            setup_error,
            .server_observer_register,
            error_domain,
            native_error.code,
            native_error.message[0..native_error.message_size],
            "the capture stream could not observe its PipeWire server",
            .{},
        );
        return error.PipeWireServerObserverRegistrationFailed;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);
    assert(audio.server_observer.is_registered == 1);
    var server_observer_is_registered = true;
    defer if (server_observer_is_registered) {
        pipewire.voiced_audio_pipewire_capture_server_observer_unregister(
            &audio.server_observer,
        );
    };

    // `pw_main_loop_run` has no capture deadline. It returns only after one of
    // our callbacks quits the loop or PipeWire itself fails the loop. A source
    // can remain nominally connected without either event, so process-level
    // supervision is part of this operation's correctness contract.
    const loop_result = pipewire.voiced_audio_pipewire_main_loop_run(
        main_loop,
        &native_error,
    );
    if (loop_result < 0) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_RUN);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT);
        assert(native_error.code == loop_result);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        if (claimMainLoopOutcome(&audio, .stream_error)) {
            recordRuntimeError(
                &audio.runtime_error,
                .main_loop,
                .pipewire_result,
                native_error.code,
            );
            writeErrorMessage(
                &audio,
                "pw_main_loop_run terminated with result {d}: {s}",
                .{
                    native_error.code,
                    native_error.message[0..native_error.message_size],
                },
            );
        }
    } else {
        assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
        assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
        assert(native_error.code == 0);
        assert(native_error.message_size == 0);
    }
    assert(audio.realtime.terminal_outcome.load(.acquire) != .none);

    // Preserve the first-linked source before unregistering graph callbacks.
    // A false result with an empty error is valid when connection failed before
    // any Link appeared. A resolved snapshot remains the original source even
    // when its Link or Node was the event that just ended this recording.
    const source_was_resolved =
        pipewire.voiced_audio_pipewire_capture_source_snapshot(
            source_observer,
            &audio.source_identity,
            &native_error,
        );
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);
    assert(source_was_resolved == (audio.source_identity.is_resolved == 1));
    assert(audio.source_identity.reserved == 0);
    if (source_was_resolved) {
        assert(audio.source_identity.node_id != pipewire.PW_ID_ANY);
    } else {
        assert(audio.source_identity.node_id == pipewire.PW_ID_ANY);
    }

    // Close the copy gate before removing the registry observer. The terminal
    // latch already makes later process callbacks return borrowed buffers only;
    // this release also states that no source remains authorized for teardown.
    audio.source_is_linked.store(false, .release);
    assert(source_observer_is_alive);
    pipewire.voiced_audio_pipewire_capture_source_observer_destroy(
        source_observer,
    );
    source_observer_is_alive = false;
    audio.source_observer = null;
    assert(!audio.source_is_linked.load(.acquire));

    audio.disconnect_is_expected = true;
    const disconnect_result =
        pipewire.voiced_audio_pipewire_capture_stream_disconnect(
            stream,
            &native_error,
        );

    // A failed disconnect must not discard an already captured prefix, and it
    // must not let us inspect callback-owned fields while the data thread could
    // still be running. Destroy the stream synchronously on every path. The
    // report retains a disconnect error separately because capture already has
    // a first-wins terminal outcome that teardown must not overwrite. The source
    // observer is already gone so expected disconnect cannot masquerade as a
    // microphone removal; remove the core observer before destroying its core.
    assert(stream_is_alive);
    assert(!source_observer_is_alive);
    assert(audio.source_observer == null);
    assert(server_observer_is_registered);
    pipewire.voiced_audio_pipewire_capture_server_observer_unregister(
        &audio.server_observer,
    );
    server_observer_is_registered = false;
    assert(audio.server_observer.is_registered == 0);
    pipewire.pw_stream_destroy(stream);
    stream_is_alive = false;
    assert(!stream_is_alive);
    audio.stream = null;
    audio.realtime.stream = null;
    assert(audio.stream == null);
    assert(audio.realtime.stream == null);

    // Stream destruction is the synchronization boundary with PipeWire's data
    // thread: after it returns, no callback can still be writing the active
    // slot. Normal stop and every terminal capture result retain that complete
    // callback-block prefix. Cancel alone abandons the private final slot; any
    // slots published earlier remain valid, but the supervisor must discard the
    // whole cancelled session rather than producing user-visible output.
    const terminal_outcome = audio.realtime.terminal_outcome.load(.acquire);
    assert(terminal_outcome != .none);
    if (terminal_outcome == .cancelled) {
        abandonUnpublishedActiveSlot(&audio.realtime);
    } else {
        publishActiveSlotIfNonEmpty(&audio.realtime);
    }

    if (disconnect_result < 0) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_DISCONNECT);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT);
        assert(native_error.code == disconnect_result);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        recordRuntimeError(
            &audio.teardown_error,
            .stream_disconnect,
            .pipewire_result,
            native_error.code,
        );
        writeTeardownErrorMessage(
            &audio,
            "pw_stream_disconnect failed with result {d}: {s}; the stream was destroyed",
            .{
                disconnect_result,
                native_error.message[0..native_error.message_size],
            },
        );
    } else {
        assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
        assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
        assert(native_error.code == 0);
        assert(native_error.message_size == 0);
    }

    // Every thread competes for one atomic first-wins terminal cause. Reading
    // the value captured after stream destruction preserves whichever callback
    // actually ended capture first.
    const outcome: Outcome = switch (terminal_outcome) {
        .none => unreachable,
        .completed => .completed,
        .stopped => .stopped,
        .cancelled => .cancelled,
        .control_error => .control_error,
        .pipeline_full => .pipeline_full,
        .format_parse_error => .format_parse_error,
        .unexpected_format => .unexpected_format,
        .buffer_configuration_error => .buffer_configuration_error,
        .timeline_error => .timeline_error,
        .timeline_discontinuity => .timeline_discontinuity,
        .stream_error => .stream_error,
        .stream_disconnected => .stream_disconnected,
        .source_disconnected => .source_disconnected,
        .source_changed => .source_changed,
        .source_observation_error => .source_observation_error,
        .invalid_buffer => .invalid_buffer,
        .corrupted_buffer => .corrupted_buffer,
        .buffer_return_error => .buffer_return_error,
    };
    const runtime_error = switch (outcome) {
        .completed, .stopped, .cancelled => RuntimeError{
            .stage = .none,
            .domain = .none,
            .code = 0,
        },
        .control_error,
        .format_parse_error,
        .unexpected_format,
        .buffer_configuration_error,
        .stream_error,
        .stream_disconnected,
        .source_disconnected,
        .source_changed,
        .source_observation_error,
        => audio.runtime_error,
        .pipeline_full,
        .timeline_error,
        .timeline_discontinuity,
        .invalid_buffer,
        .corrupted_buffer,
        .buffer_return_error,
        => audio.realtime.runtime_error,
    };

    if (audio.error_message_size == 0) {
        switch (outcome) {
            .pipeline_full => writeErrorMessage(
                &audio,
                "all audio exchange slots remained published to the consumer",
                .{},
            ),
            .invalid_buffer => writeErrorMessage(
                &audio,
                "buffer validation failed ({s}): metas={d}, meta_size={d}, " ++
                    "header_flags=0x{x}, header_seq={d}, header_pts={d}, " ++
                    "planes={d}, data_flags=0x{x}, max_size={d}, " ++
                    "chunk=offset {d}, size {d}, stride {d}, invalid_sample={d}",
                .{
                    @tagName(audio.realtime.buffer_error.reason),
                    audio.realtime.buffer_error.metadata_count,
                    audio.realtime.buffer_error.metadata_size,
                    audio.realtime.buffer_error.header_flags,
                    audio.realtime.buffer_error.header_sequence,
                    audio.realtime.buffer_error.header_presentation_timestamp_ns,
                    audio.realtime.buffer_error.data_planes_count,
                    audio.realtime.buffer_error.data_flags,
                    audio.realtime.buffer_error.data_max_size,
                    audio.realtime.buffer_error.chunk_offset,
                    audio.realtime.buffer_error.chunk_size,
                    audio.realtime.buffer_error.chunk_stride,
                    audio.realtime.buffer_error.invalid_sample_index,
                },
            ),
            .corrupted_buffer => writeErrorMessage(
                &audio,
                "PipeWire marked audio corrupted: header_flags=0x{x}, " ++
                    "chunk_offset={d}, chunk_size={d}",
                .{
                    audio.realtime.buffer_error.header_flags,
                    audio.realtime.buffer_error.chunk_offset,
                    audio.realtime.buffer_error.chunk_size,
                },
            ),
            .timeline_error => {
                const failure = audio.realtime.timeline_failure;
                if (failure.query_result < 0) {
                    writeErrorMessage(
                        &audio,
                        "pw_stream_get_time_n failed with result {d}",
                        .{failure.query_result},
                    );
                } else {
                    writeErrorMessage(
                        &audio,
                        "PipeWire returned an unusable buffer timeline ({s}): " ++
                            "graph_now={d}, rate={d}/{d}, graph_ticks={d}, " ++
                            "buffer_time={d}, previous_buffer_ticks={d}, " ++
                            "current_buffer_ticks={d}, block_samples={d}",
                        .{
                            @tagName(failure.code),
                            failure.graph_now_ns,
                            failure.graph_rate_num,
                            failure.graph_rate_denom,
                            failure.graph_ticks,
                            failure.buffer_cycle_ns,
                            failure.buffer_ticks_previous,
                            failure.buffer_ticks_current,
                            failure.block_samples_count,
                        },
                    );
                }
            },
            .timeline_discontinuity => {
                const timeline_failure = audio.realtime.timeline_failure;
                if (timeline_failure.code == .discontinuity) {
                    writeErrorMessage(
                        &audio,
                        "PipeWire buffer timeline skipped or duplicated audio: " ++
                            "rate={d}/{d}, previous_buffer_ticks={d}, " ++
                            "current_buffer_ticks={d}, tick_delta={d}, " ++
                            "block_samples={d}",
                        .{
                            timeline_failure.graph_rate_num,
                            timeline_failure.graph_rate_denom,
                            timeline_failure.buffer_ticks_previous,
                            timeline_failure.buffer_ticks_current,
                            timeline_failure.buffer_ticks_current -
                                timeline_failure.buffer_ticks_previous,
                            timeline_failure.block_samples_count,
                        },
                    );
                } else {
                    writeErrorMessage(
                        &audio,
                        "PipeWire marked an audio timeline discontinuity: " ++
                            "header_flags=0x{x}, sequence={d}, pts={d}",
                        .{
                            audio.realtime.buffer_error.header_flags,
                            audio.realtime.buffer_error.header_sequence,
                            audio.realtime.buffer_error.header_presentation_timestamp_ns,
                        },
                    );
                }
            },
            .buffer_return_error => writeErrorMessage(
                &audio,
                "pw_stream_queue_buffer failed with result {d}",
                .{audio.realtime.queue_buffer_result},
            ),
            else => {},
        }
    }

    const report: Report = .{
        .outcome = outcome,
        .runtime_error = runtime_error,
        .error_message = audio.error_message,
        .error_message_size = audio.error_message_size,
        .teardown_error = audio.teardown_error,
        .teardown_error_message = audio.teardown_error_message,
        .teardown_error_message_size = audio.teardown_error_message_size,
        .timeline_validation = timeline_validation,
        .pipewire_headers_version = native_environment.headers_version,
        .pipewire_headers_version_size = @intCast(
            native_environment.headers_version_size,
        ),
        .pipewire_library_version = native_environment.library_version,
        .pipewire_library_version_size = @intCast(
            native_environment.library_version_size,
        ),
        .pipewire_server_version = audio.server_observer.version,
        .pipewire_server_version_size = @intCast(
            audio.server_observer.version_size,
        ),
        .source_identity = .{
            .is_resolved = audio.source_identity.is_resolved == 1,
            .node_id = audio.source_identity.node_id,
            .node_object_serial = audio.source_identity.node_object_serial,
            .device_id = audio.source_identity.device_id,
            .device_object_serial = audio.source_identity.device_object_serial,
            .node_name = audio.source_identity.node_name,
            .node_name_size = @intCast(audio.source_identity.node_name_size),
            .node_description = audio.source_identity.node_description,
            .node_description_size = @intCast(
                audio.source_identity.node_description_size,
            ),
            .device_serial = audio.source_identity.device_serial,
            .device_serial_size = @intCast(
                audio.source_identity.device_serial_size,
            ),
            .device_description = audio.source_identity.device_description,
            .device_description_size = @intCast(
                audio.source_identity.device_description_size,
            ),
        },
        .negotiated_sample_rate_hz = audio.negotiated_sample_rate_hz,
        .negotiated_channels_count = audio.negotiated_channels_count,
        .samples_count = audio.realtime.samples_count,
        .published_samples_count = audio.realtime.published_samples_count,
        .slot_publications_count = audio.realtime.slot_publications_count,
        .shared_memory_is_locked = shared_memory_is_locked,
        .shared_memory_lock_error_code = shared_memory_lock_error_code,
        .shared_memory_lock_limit_bytes = memory_lock_limits.cur,
        .main_loop_thread_id = audio.realtime.main_loop_thread_id,
        .callback_thread_id = audio.realtime.callback_thread_id,
        .callback_scheduler_policy = audio.realtime.callback_scheduler_policy,
        .callback_scheduler_priority = audio.realtime.callback_scheduler_priority,
        .callbacks_count = audio.realtime.callbacks_count,
        .missing_buffers_count = audio.realtime.missing_buffers_count,
        .clipped_samples_count = audio.realtime.clipped_samples_count,
        .header_metadata_buffers_count = audio.realtime.header_metadata_buffers_count,
        .header_gap_buffers_count = audio.realtime.header_gap_buffers_count,
        .header_gap_samples_count = audio.realtime.header_gap_samples_count,
        .block_samples_count_min = if (audio.realtime.block_samples_count_max == 0)
            0
        else
            audio.realtime.block_samples_count_min,
        .block_samples_count_max = audio.realtime.block_samples_count_max,
        .callback_duration_ns_max = audio.realtime.callback_duration_ns_max,
        .callback_gap_ns_max = audio.realtime.callback_gap_ns_max,
    };
    assert(setup_error.stage == .none);
    assert(setup_error.domain == .none);
    assert(setup_error.code == 0);
    assert(setup_error.message_size == 0);
    assert(setup_error.pipewire_version_size == 0);
    assert(report.error_message_size <= report.error_message.len);
    assert(report.teardown_error_message_size <= report.teardown_error_message.len);
    assert(report.pipewire_headers_version_size > 0);
    assert(report.pipewire_headers_version_size <=
        report.pipewire_headers_version.len);
    assert(report.pipewire_library_version_size > 0);
    assert(report.pipewire_library_version_size <=
        report.pipewire_library_version.len);
    assert(report.pipewire_server_version_size <=
        report.pipewire_server_version.len);
    assert(report.source_identity.node_name_size <=
        report.source_identity.node_name.len);
    assert(report.source_identity.node_description_size <=
        report.source_identity.node_description.len);
    assert(report.source_identity.device_serial_size <=
        report.source_identity.device_serial.len);
    assert(report.source_identity.device_description_size <=
        report.source_identity.device_description.len);
    if (report.source_identity.is_resolved) {
        assert(report.source_identity.node_id != pipewire.PW_ID_ANY);
    } else {
        assert(report.source_identity.node_id == pipewire.PW_ID_ANY);
    }
    if (launch.configured_device_serial) |device_serial| {
        if (report.samples_count > 0) {
            assert(report.source_identity.is_resolved);
            assert(report.source_identity.device_serial_size == device_serial.len);
            assert(std.mem.eql(
                u8,
                report.source_identity.device_serial[0..report.source_identity.device_serial_size],
                device_serial,
            ));
        }
    }
    assert(report.shared_memory_lock_limit_bytes == memory_lock_limits.cur);
    if (report.shared_memory_is_locked) {
        assert(report.shared_memory_lock_error_code == 0);
    } else {
        assert(report.shared_memory_lock_error_code > 0);
    }
    switch (report.timeline_validation) {
        .header_only => {
            assert(audio.realtime.timeline_buffer_ticks_previous == null);
            assert(audio.realtime.timeline_graph_rate_num_previous == 0);
            assert(audio.realtime.timeline_graph_rate_denom_previous == 0);
        },
        .full => {},
    }
    switch (report.outcome) {
        .completed, .stopped, .cancelled => {
            assert(report.runtime_error.stage == .none);
            assert(report.runtime_error.domain == .none);
            assert(report.runtime_error.code == 0);
            assert(report.error_message_size == 0);
        },
        .control_error => {
            assert(report.runtime_error.stage == .supervisor_control);
            assert(report.runtime_error.domain == .linux_errno or
                report.runtime_error.domain == .voiced_audio);
            assert(report.runtime_error.code > 0);
            assert(report.error_message_size > 0);
        },
        .pipeline_full => {
            assert(report.runtime_error.stage == .exchange_publication);
            assert(report.runtime_error.domain == .voiced_audio);
            assert(report.runtime_error.code == @intFromEnum(AudioErrorCode.exchange_full));
            assert(report.error_message_size > 0);
        },
        .format_parse_error => {
            assert(report.runtime_error.stage == .format_negotiation);
            switch (report.runtime_error.domain) {
                .voiced_audio => assert(report.runtime_error.code ==
                    @intFromEnum(AudioErrorCode.format_parameter_removed)),
                .boundary_validation => assert(report.runtime_error.code > 0),
                .pipewire_result => assert(report.runtime_error.code < 0),
                else => unreachable,
            }
            assert(report.error_message_size > 0);
        },
        .unexpected_format => {
            assert(report.runtime_error.stage == .format_negotiation);
            assert(report.runtime_error.domain == .voiced_audio);
            assert(report.runtime_error.code == @intFromEnum(AudioErrorCode.unexpected_format));
            assert(report.error_message_size > 0);
        },
        .buffer_configuration_error => {
            assert(report.runtime_error.stage == .buffer_negotiation);
            assert(report.runtime_error.domain == .pipewire_result);
            assert(report.runtime_error.code < 0);
            assert(report.error_message_size > 0);
        },
        .timeline_error => {
            assert(report.runtime_error.stage == .timeline_continuity);
            switch (report.runtime_error.domain) {
                .pipewire_result => assert(report.runtime_error.code < 0),
                .voiced_audio => {
                    const timeline_error_code: TimelineErrorCode = @enumFromInt(
                        @as(u32, @intCast(report.runtime_error.code)),
                    );
                    switch (timeline_error_code) {
                        .invalid_rate,
                        .invalid_graph_time,
                        .buffer_from_future,
                        .tick_underflow,
                        .rate_changed,
                        .ticks_regressed,
                        => {},
                        .none, .discontinuity => unreachable,
                    }
                },
                else => unreachable,
            }
            assert(report.error_message_size > 0);
        },
        .timeline_discontinuity => {
            assert(report.runtime_error.stage == .timeline_continuity);
            switch (report.runtime_error.domain) {
                .spa_buffer => assert(report.runtime_error.code ==
                    @intFromEnum(BufferErrorCode.timeline_discontinuity)),
                .voiced_audio => assert(report.runtime_error.code ==
                    @intFromEnum(TimelineErrorCode.discontinuity)),
                else => unreachable,
            }
            assert(report.error_message_size > 0);
        },
        .stream_error => {
            switch (report.runtime_error.stage) {
                .main_loop => {
                    assert(report.runtime_error.domain == .pipewire_result);
                    assert(report.runtime_error.code < 0);
                },
                .callback_event => {
                    assert(report.runtime_error.domain == .linux_errno or
                        report.runtime_error.domain == .voiced_audio);
                    assert(report.runtime_error.code > 0);
                },
                .stream_state => {
                    assert(report.runtime_error.domain == .pipewire_callback);
                    assert(report.runtime_error.code == 0);
                },
                else => unreachable,
            }
            assert(report.error_message_size > 0);
        },
        .stream_disconnected => {
            assert(report.runtime_error.stage == .stream_state);
            assert(report.runtime_error.domain == .pipewire_callback);
            assert(report.runtime_error.code == 0);
            assert(report.error_message_size > 0);
        },
        .source_disconnected => {
            assert(report.runtime_error.stage == .source_identity);
            assert(report.runtime_error.domain == .voiced_audio);
            assert(report.runtime_error.code ==
                @intFromEnum(AudioErrorCode.source_link_removed) or
                report.runtime_error.code ==
                    @intFromEnum(AudioErrorCode.source_node_removed) or
                report.runtime_error.code ==
                    @intFromEnum(AudioErrorCode.source_device_removed));
            assert(report.source_identity.is_resolved);
            assert(report.error_message_size > 0);
        },
        .source_changed => {
            assert(report.runtime_error.stage == .source_identity);
            assert(report.runtime_error.domain == .voiced_audio);
            assert(report.runtime_error.code ==
                @intFromEnum(AudioErrorCode.source_changed));
            assert(report.source_identity.is_resolved);
            assert(report.error_message_size > 0);
        },
        .source_observation_error => {
            assert(report.runtime_error.stage == .source_identity);
            assert(report.runtime_error.domain == .boundary_validation or
                report.runtime_error.domain == .pipewire_result);
            if (report.runtime_error.domain == .boundary_validation) {
                assert(report.runtime_error.code > 0);
            } else {
                assert(report.runtime_error.code < 0);
            }
            assert(report.error_message_size > 0);
        },
        .invalid_buffer => {
            assert(report.runtime_error.stage == .buffer_metadata or
                report.runtime_error.stage == .buffer_data or
                report.runtime_error.stage == .sample_validation);
            assert(report.runtime_error.domain == .spa_buffer);
            assert(report.runtime_error.code > 0);
            assert(report.error_message_size > 0);
        },
        .corrupted_buffer => {
            assert(report.runtime_error.stage == .buffer_metadata or
                report.runtime_error.stage == .buffer_data);
            assert(report.runtime_error.domain == .spa_buffer);
            assert(report.runtime_error.code ==
                @intFromEnum(BufferErrorCode.header_corrupted) or
                report.runtime_error.code ==
                    @intFromEnum(BufferErrorCode.chunk_corrupted));
            assert(report.error_message_size > 0);
        },
        .buffer_return_error => {
            assert(report.runtime_error.stage == .buffer_return);
            assert(report.runtime_error.domain == .pipewire_result);
            assert(report.runtime_error.code < 0);
            assert(report.error_message_size > 0);
        },
    }
    if (report.teardown_error.stage == .none) {
        assert(report.teardown_error.domain == .none);
        assert(report.teardown_error.code == 0);
        assert(report.teardown_error_message_size == 0);
    } else {
        assert(report.teardown_error.stage == .stream_disconnect);
        assert(report.teardown_error.domain == .pipewire_result);
        assert(report.teardown_error.code < 0);
        assert(report.teardown_error_message_size > 0);
    }
    assert(report.main_loop_thread_id > 0);
    assert(report.samples_count <=
        launch.recording_samples_target + audio_exchange.callback_samples_count_max);
    assert(report.published_samples_count <= report.samples_count);
    assert(report.slot_publications_count <= report.published_samples_count);
    assert(report.missing_buffers_count <= report.callbacks_count);
    assert(report.clipped_samples_count <= report.samples_count);
    assert(report.header_metadata_buffers_count <= report.callbacks_count);
    assert(report.header_gap_buffers_count <= report.header_metadata_buffers_count);
    assert(report.header_gap_samples_count <= report.samples_count);
    assert(report.callbacks_count ==
        audio_exchange.acquireAudioCallbacksCount(launch.exchange));
    assert(report.samples_count ==
        audio_exchange.acquireAudioSamplesCount(launch.exchange));
    if (report.callbacks_count == 0) {
        assert(report.callback_thread_id == 0);
    } else {
        assert(report.callback_thread_id > 0);
    }
    if (report.samples_count == 0) {
        assert(report.published_samples_count == 0);
        assert(report.slot_publications_count == 0);
        assert(report.clipped_samples_count == 0);
        assert(report.header_gap_samples_count == 0);
        assert(report.block_samples_count_min == 0);
        assert(report.block_samples_count_max == 0);
    } else {
        assert(report.source_identity.is_resolved);
        assert(report.negotiated_sample_rate_hz == audio_exchange.sample_rate_hz);
        assert(report.negotiated_channels_count == audio_exchange.channels_count);
        assert(report.block_samples_count_min > 0);
        assert(report.block_samples_count_min <= report.block_samples_count_max);
        assert(report.block_samples_count_max <= audio_exchange.callback_samples_count_max);
        assert(report.block_samples_count_max <= launch.slot_samples_boundary);
    }
    switch (report.outcome) {
        .completed => {
            assert(report.source_identity.is_resolved);
            assert(report.samples_count >= launch.recording_samples_target);
            assert(report.published_samples_count == report.samples_count);
        },
        .stopped => assert(report.published_samples_count == report.samples_count),
        .cancelled => assert(report.samples_count - report.published_samples_count <=
            launch.slot_samples_boundary),
        else => assert(report.published_samples_count == report.samples_count),
    }
    if (report.published_samples_count == 0) {
        assert(report.slot_publications_count == 0);
    } else {
        assert(report.slot_publications_count > 0);
    }
    return report;
}

fn controlSocketEventReceived(
    context: ?*anyopaque,
    file_descriptor: c_int,
    mask: u32,
) callconv(.c) void {
    assert(context != null);
    const audio: *Audio = @ptrCast(@alignCast(context.?));
    assert(file_descriptor == audio.control_socket);
    assert(mask & (pipewire.SPA_IO_IN | pipewire.SPA_IO_ERR | pipewire.SPA_IO_HUP) != 0);

    // Read the command before interpreting HUP: SOCK_SEQPACKET may report both
    // when a final complete record was queued immediately before peer closure.
    // `MSG_TRUNC` exposes the record's actual size, while `DONTWAIT` prevents a
    // spurious readiness notification from ever blocking the PipeWire loop.
    if (mask & pipewire.SPA_IO_IN != 0) {
        var packet: ControlPacket = undefined;
        while (true) {
            const receive_result = linux.recvfrom(
                file_descriptor,
                std.mem.asBytes(&packet).ptr,
                @sizeOf(ControlPacket),
                linux.MSG.TRUNC | linux.MSG.DONTWAIT,
                null,
                null,
            );
            switch (linux.errno(receive_result)) {
                .SUCCESS => {
                    if (receive_result == 0) {
                        if (claimMainLoopOutcome(audio, .control_error)) {
                            recordRuntimeError(
                                &audio.runtime_error,
                                .supervisor_control,
                                .voiced_audio,
                                @intFromEnum(AudioErrorCode.control_socket_closed),
                            );
                            writeErrorMessage(
                                audio,
                                "supervisor control socket closed without a stop or cancel command",
                                .{},
                            );
                        }
                        _ = pipewire.pw_main_loop_quit(audio.main_loop);
                        return;
                    }

                    // Both processes run the same executable and this packet is
                    // trusted internal state. A wrong size, reserved value, or
                    // command is a protocol defect—not hostile input to recover.
                    assert(receive_result == @sizeOf(ControlPacket));
                    assert(packet.reserved == 0);
                    const requested_outcome: Outcome = switch (packet.command) {
                        @intFromEnum(ControlCommand.stop) => .stopped,
                        @intFromEnum(ControlCommand.cancel) => .cancelled,
                        else => unreachable,
                    };

                    // Completion, a PipeWire failure, and supervisor control can
                    // arrive together. The atomic terminal latch preserves the
                    // first event. Only a winning control command closes the
                    // source gate and quits the owner loop itself; a losing one
                    // leaves the already-latched callback event to do so.
                    if (claimMainLoopOutcome(audio, requested_outcome)) {
                        audio.source_is_linked.store(false, .release);
                        _ = pipewire.pw_main_loop_quit(audio.main_loop);
                    }
                    return;
                },
                .INTR => continue,
                .AGAIN => break,
                else => |receive_errno| {
                    if (claimMainLoopOutcome(audio, .control_error)) {
                        recordRuntimeError(
                            &audio.runtime_error,
                            .supervisor_control,
                            .linux_errno,
                            @intFromEnum(receive_errno),
                        );
                        writeErrorMessage(
                            audio,
                            "supervisor control receive failed: errno={s}",
                            .{@tagName(receive_errno)},
                        );
                    }
                    _ = pipewire.pw_main_loop_quit(audio.main_loop);
                    return;
                },
            }
        }
    }

    if (mask & (pipewire.SPA_IO_ERR | pipewire.SPA_IO_HUP) != 0) {
        if (claimMainLoopOutcome(audio, .control_error)) {
            recordRuntimeError(
                &audio.runtime_error,
                .supervisor_control,
                .voiced_audio,
                @intFromEnum(AudioErrorCode.control_socket_source_error),
            );
            writeErrorMessage(
                audio,
                "supervisor control socket failed: fd={d}, mask=0x{x}",
                .{ file_descriptor, mask },
            );
        }
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
    }
}

fn callbackEventReceived(
    context: ?*anyopaque,
    file_descriptor: c_int,
    mask: u32,
) callconv(.c) void {
    assert(context != null);
    const audio: *Audio = @ptrCast(@alignCast(context.?));

    // The descriptor is bound to this callback when the source is registered;
    // PipeWire may report operating HUP/ERR events, but it may not substitute a
    // different descriptor for Voiced's callback context.
    assert(file_descriptor == audio.callback_event_fd);
    if (mask & (pipewire.SPA_IO_ERR | pipewire.SPA_IO_HUP) != 0) {
        if (claimMainLoopOutcome(audio, .stream_error)) {
            recordRuntimeError(
                &audio.runtime_error,
                .callback_event,
                .voiced_audio,
                @intFromEnum(AudioErrorCode.callback_event_source_error),
            );
            writeErrorMessage(
                audio,
                "callback event source failed: fd={d}, mask=0x{x}",
                .{ file_descriptor, mask },
            );
        }
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
        return;
    }
    assert(mask & pipewire.SPA_IO_IN != 0);

    var signals_count: u64 = 0;
    const signal_bytes = std.mem.asBytes(&signals_count);
    while (true) {
        const read_result = linux.read(file_descriptor, signal_bytes.ptr, signal_bytes.len);
        const read_errno = linux.errno(read_result);
        if (read_errno == .INTR) continue;
        if (read_errno != .SUCCESS or read_result != signal_bytes.len) {
            if (claimMainLoopOutcome(audio, .stream_error)) {
                if (read_errno != .SUCCESS) {
                    recordRuntimeError(
                        &audio.runtime_error,
                        .callback_event,
                        .linux_errno,
                        @intFromEnum(read_errno),
                    );
                } else {
                    recordRuntimeError(
                        &audio.runtime_error,
                        .callback_event,
                        .voiced_audio,
                        @intFromEnum(AudioErrorCode.callback_event_short_read),
                    );
                }
                writeErrorMessage(
                    audio,
                    "callback event read failed: errno={s}, raw_result=0x{x}, expected={d}",
                    .{ @tagName(read_errno), read_result, signal_bytes.len },
                );
            }
        }
        break;
    }

    _ = pipewire.pw_main_loop_quit(audio.main_loop);
}

fn sourceEventReceived(
    context: ?*anyopaque,
    event_value: u32,
    previous_source_node_id: u32,
    current_source_node_id: u32,
    native_error: [*c]const pipewire.voiced_audio_pipewire_error,
) callconv(.c) void {
    assert(context != null);
    assert(native_error != null);
    assert(native_error.*.message_size <= native_error.*.message.len);
    const audio: *Audio = @ptrCast(@alignCast(context.?));

    // The first incoming Link is the authorization boundary for sample copying,
    // not merely a diagnostic discovered after capture. Registry observation
    // was installed before connect, so release-opening this gate means C has
    // locked the stream node to one concrete source node. If another terminal
    // callback won first, leave the gate closed and retain that earlier cause.
    if (event_value == pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINKED) {
        assert(native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
        assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
        assert(native_error.*.code == 0);
        assert(native_error.*.message_size == 0);
        assert(previous_source_node_id == pipewire.PW_ID_ANY);
        assert(current_source_node_id != pipewire.PW_ID_ANY);
        if (audio.realtime.terminal_outcome.load(.acquire) == .none) {
            assert(!audio.source_is_linked.load(.acquire));
            audio.source_is_linked.store(true, .release);
            assert(audio.source_is_linked.load(.acquire));
        }
        return;
    }

    // A Link removal may be followed immediately by a new default-source Link.
    // Close the realtime gate before recording the event so no callback that
    // starts afterward can copy from the replacement. The current callback may
    // still finish a buffer borrowed from the original graph; stream teardown
    // and the first-wins terminal latch contain that bounded overlap.
    audio.source_is_linked.store(false, .release);
    const outcome: Outcome = switch (event_value) {
        pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINK_REMOVED,
        pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_NODE_REMOVED,
        pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_DEVICE_REMOVED,
        => .source_disconnected,
        pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_CHANGED => .source_changed,
        pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_OBSERVATION_ERROR => .source_observation_error,
        pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINKED => unreachable,
        else => unreachable,
    };
    if (claimMainLoopOutcome(audio, outcome)) {
        switch (event_value) {
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINK_REMOVED => {
                assert(native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
                assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
                assert(previous_source_node_id != pipewire.PW_ID_ANY);
                assert(current_source_node_id == pipewire.PW_ID_ANY);
                recordRuntimeError(
                    &audio.runtime_error,
                    .source_identity,
                    .voiced_audio,
                    @intFromEnum(AudioErrorCode.source_link_removed),
                );
                writeErrorMessage(
                    audio,
                    "the Link from PipeWire source node {d} disappeared",
                    .{previous_source_node_id},
                );
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_NODE_REMOVED => {
                assert(native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
                assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
                assert(previous_source_node_id != pipewire.PW_ID_ANY);
                assert(current_source_node_id == pipewire.PW_ID_ANY);
                recordRuntimeError(
                    &audio.runtime_error,
                    .source_identity,
                    .voiced_audio,
                    @intFromEnum(AudioErrorCode.source_node_removed),
                );
                writeErrorMessage(
                    audio,
                    "PipeWire source node {d} disappeared",
                    .{previous_source_node_id},
                );
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_DEVICE_REMOVED => {
                assert(native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
                assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
                assert(previous_source_node_id != pipewire.PW_ID_ANY);
                assert(current_source_node_id == pipewire.PW_ID_ANY);
                recordRuntimeError(
                    &audio.runtime_error,
                    .source_identity,
                    .voiced_audio,
                    @intFromEnum(AudioErrorCode.source_device_removed),
                );
                writeErrorMessage(
                    audio,
                    "the Device behind PipeWire source node {d} disappeared",
                    .{previous_source_node_id},
                );
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_CHANGED => {
                assert(native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
                assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
                assert(previous_source_node_id != pipewire.PW_ID_ANY);
                assert(current_source_node_id != pipewire.PW_ID_ANY);
                assert(current_source_node_id != previous_source_node_id);
                recordRuntimeError(
                    &audio.runtime_error,
                    .source_identity,
                    .voiced_audio,
                    @intFromEnum(AudioErrorCode.source_changed),
                );
                writeErrorMessage(
                    audio,
                    "PipeWire replaced source node {d} with source node {d}",
                    .{ previous_source_node_id, current_source_node_id },
                );
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_OBSERVATION_ERROR => {
                assert(native_error.*.stage ==
                    pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVATION);
                const error_domain: ErrorDomain = switch (native_error.*.domain) {
                    pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT => .pipewire_result,
                    pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION => .boundary_validation,
                    else => unreachable,
                };
                if (error_domain == .pipewire_result) {
                    assert(native_error.*.code < 0);
                } else {
                    assert(native_error.*.code > 0);
                }
                assert(native_error.*.message_size > 0);
                recordRuntimeError(
                    &audio.runtime_error,
                    .source_identity,
                    error_domain,
                    native_error.*.code,
                );
                writeErrorMessage(
                    audio,
                    "PipeWire source observation failed: {s}",
                    .{native_error.*.message[0..native_error.*.message_size]},
                );
            },
            pipewire.VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINKED => unreachable,
            else => unreachable,
        }
    }

    assert(!audio.source_is_linked.load(.acquire));
    assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
    _ = pipewire.pw_main_loop_quit(audio.main_loop);
}

fn streamStateChanged(
    context: ?*anyopaque,
    old_state: pipewire.pw_stream_state,
    new_state: pipewire.pw_stream_state,
    native_error: [*c]const pipewire.voiced_audio_pipewire_error,
) callconv(.c) void {
    assert(context != null);
    assert(native_error != null);
    assert(native_error.*.code == 0);
    assert(native_error.*.message_size <= native_error.*.message.len);

    if (native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_STATE) {
        assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_CALLBACK);
    } else {
        assert(native_error.*.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
        assert(native_error.*.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
        assert(native_error.*.message_size == 0);
    }
    const audio: *Audio = @ptrCast(@alignCast(context.?));

    // `CONNECTING`, `PAUSED`, and even `STREAMING` describe PipeWire's graph
    // state; none proves that a Format or process buffer will arrive. They are
    // deliberately nonterminal here and must be paired with the supervisor's
    // setup/progress deadlines rather than promoted to a ready acknowledgement.
    if (new_state == pipewire.PW_STREAM_STATE_ERROR) {
        audio.source_is_linked.store(false, .release);
        audio.negotiated_format_is_accepted.store(false, .release);
        if (claimMainLoopOutcome(audio, .stream_error)) {
            recordRuntimeError(
                &audio.runtime_error,
                .stream_state,
                .pipewire_callback,
                native_error.*.code,
            );
            if (native_error.*.message_size == 0) {
                writeErrorMessage(audio, "PipeWire entered ERROR without a diagnostic", .{});
            } else {
                copyNativeErrorMessage(audio, native_error);
            }
        }
        assert(!audio.negotiated_format_is_accepted.load(.acquire));
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
        return;
    }

    if (new_state == pipewire.PW_STREAM_STATE_UNCONNECTED and
        old_state != pipewire.PW_STREAM_STATE_UNCONNECTED and
        !audio.disconnect_is_expected)
    {
        audio.source_is_linked.store(false, .release);
        audio.negotiated_format_is_accepted.store(false, .release);
        if (claimMainLoopOutcome(audio, .stream_disconnected)) {
            recordRuntimeError(
                &audio.runtime_error,
                .stream_state,
                .pipewire_callback,
                native_error.*.code,
            );
            if (native_error.*.message_size == 0) {
                writeErrorMessage(
                    audio,
                    "PipeWire became UNCONNECTED without a diagnostic",
                    .{},
                );
            } else {
                copyNativeErrorMessage(audio, native_error);
            }
        }
        assert(!audio.negotiated_format_is_accepted.load(.acquire));
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
    }
}

fn streamFormatChanged(
    context: ?*anyopaque,
    parameter_id: u32,
    parameter: ?*const pipewire.spa_pod,
) callconv(.c) void {
    assert(context != null);
    const audio: *Audio = @ptrCast(@alignCast(context.?));

    if (parameter_id != pipewire.SPA_PARAM_Format) {
        return;
    }

    // `pw_stream_disconnect` clears the stream's Format by delivering a null
    // parameter. Capture has already stopped and the accepted format remains
    // the one that governed every published sample, so this teardown event must
    // not replace the realtime callback's completed outcome with a format error.
    if (audio.disconnect_is_expected) {
        return;
    }

    // Every runtime Format event replaces the authorization established by the
    // previous one. Close that gate before inspecting the new pod; only a fully
    // parsed float32/16 kHz/mono format below may reopen it. A callback already
    // in flight still owns a buffer negotiated under the preceding format, but
    // subsequent callbacks cannot copy through stale acceptance.
    const previous_format_was_accepted =
        audio.negotiated_format_is_accepted.load(.acquire);
    audio.negotiated_format_is_accepted.store(false, .release);

    if (parameter == null) {
        // PipeWire 0.3.48 begins initial negotiation with a null Format event
        // before publishing the selected pod. No capture contract exists yet
        // for that event to revoke, so remain closed and wait for the pod. If
        // none arrives, the supervisor's setup deadline reports the stalled
        // negotiation. A null event after acceptance is different: the stream
        // has removed the format governing live buffers, so capture must stop.
        if (!previous_format_was_accepted) {
            assert(!audio.negotiated_format_is_accepted.load(.acquire));
            return;
        }

        if (claimMainLoopOutcome(audio, .format_parse_error)) {
            recordRuntimeError(
                &audio.runtime_error,
                .format_negotiation,
                .voiced_audio,
                @intFromEnum(AudioErrorCode.format_parameter_removed),
            );
            writeErrorMessage(audio, "PipeWire removed the negotiated Format parameter", .{});
        }
        assert(!audio.negotiated_format_is_accepted.load(.acquire));
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
        return;
    }

    var negotiated_format: pipewire.spa_audio_info_raw = undefined;
    var native_error: pipewire.voiced_audio_pipewire_error = undefined;
    if (!pipewire.voiced_audio_pipewire_parse_negotiated_format(
        parameter,
        &negotiated_format,
        &native_error,
    )) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_FORMAT_PARSE);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION or
            native_error.domain ==
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        if (claimMainLoopOutcome(audio, .format_parse_error)) {
            const error_domain: ErrorDomain = switch (native_error.domain) {
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION => .boundary_validation,
                pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT => .pipewire_result,
                else => unreachable,
            };
            recordRuntimeError(
                &audio.runtime_error,
                .format_negotiation,
                error_domain,
                native_error.code,
            );
            writeErrorMessage(
                audio,
                "PipeWire returned an invalid raw-audio Format pod: {s}",
                .{native_error.message[0..native_error.message_size]},
            );
        }
        assert(!audio.negotiated_format_is_accepted.load(.acquire));
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
        return;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    if (negotiated_format.format != pipewire.SPA_AUDIO_FORMAT_F32 or
        negotiated_format.rate != audio_exchange.sample_rate_hz or
        negotiated_format.channels != audio_exchange.channels_count or
        negotiated_format.position[0] != pipewire.SPA_AUDIO_CHANNEL_MONO)
    {
        if (claimMainLoopOutcome(audio, .unexpected_format)) {
            recordRuntimeError(
                &audio.runtime_error,
                .format_negotiation,
                .voiced_audio,
                @intFromEnum(AudioErrorCode.unexpected_format),
            );
            writeErrorMessage(
                audio,
                "PipeWire negotiated format={d}, rate={d}, channels={d}, position={d}; " ++
                    "Voiced requires native float32, 16000 Hz, one MONO-positioned channel",
                .{
                    negotiated_format.format,
                    negotiated_format.rate,
                    negotiated_format.channels,
                    negotiated_format.position[0],
                },
            );
        }
        assert(!audio.negotiated_format_is_accepted.load(.acquire));
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
        return;
    }

    // Buffer negotiation follows format negotiation. Ask PipeWire to attach a
    // complete SPA Header before opening the format gate. When supplied, it
    // carries authoritative gap, corruption, and discontinuity flags. PipeWire
    // may omit this requested metadata, so per-buffer graph-clock validation in
    // the process callback independently detects lost timeline intervals. This
    // update runs synchronously inside the asynchronous Format callback and uses
    // the same fixed error-output contract as setup operations.
    assert(audio.stream != null);
    const buffer_configuration_result =
        pipewire.voiced_audio_pipewire_capture_stream_configure_buffers(
            audio.stream.?,
            audio_exchange.callback_samples_count_max,
            &native_error,
        );
    if (buffer_configuration_result < 0) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_BUFFERS_CONFIGURE);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT);
        assert(native_error.code == buffer_configuration_result);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        if (claimMainLoopOutcome(audio, .buffer_configuration_error)) {
            recordRuntimeError(
                &audio.runtime_error,
                .buffer_negotiation,
                .pipewire_result,
                native_error.code,
            );
            writeErrorMessage(
                audio,
                "PipeWire rejected bounded audio-buffer and SPA Header negotiation " ++
                    "with result {d}: {s}",
                .{
                    buffer_configuration_result,
                    native_error.message[0..native_error.message_size],
                },
            );
        }
        assert(!audio.negotiated_format_is_accepted.load(.acquire));
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
        _ = pipewire.pw_main_loop_quit(audio.main_loop);
        return;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    // Publish the complete accepted format last. The realtime callback's
    // acquire-load prevents it from observing acceptance before the format and
    // metadata requirements above are complete.
    audio.negotiated_sample_rate_hz = negotiated_format.rate;
    audio.negotiated_channels_count = negotiated_format.channels;
    audio.negotiated_format_is_accepted.store(true, .release);
    assert(audio.negotiated_sample_rate_hz == audio_exchange.sample_rate_hz);
    assert(audio.negotiated_channels_count == audio_exchange.channels_count);
    assert(audio.negotiated_format_is_accepted.load(.acquire));
}

fn processAudio(context: ?*anyopaque) callconv(.c) void {
    assert(context != null);
    const audio: *Audio = @ptrCast(@alignCast(context.?));
    const realtime = &audio.realtime;
    assert(realtime.exchange.version == audio_exchange.format_version);
    assert(realtime.exchange.generation > 0);
    assert(realtime.recording_samples_target > 0);
    assert(realtime.slot_samples_boundary >= audio_exchange.callback_samples_count_max);
    const callback_started_ns = monotonicNanoseconds();
    defer {
        const callback_duration_ns = monotonicNanoseconds() -| callback_started_ns;
        realtime.callback_duration_ns_max = @max(
            realtime.callback_duration_ns_max,
            callback_duration_ns,
        );
    }

    if (realtime.callback_thread_id == 0) {
        realtime.callback_thread_id = linux.gettid();
        const scheduler_policy_result = linux.sched_getscheduler(0);
        if (linux.errno(scheduler_policy_result) == .SUCCESS) {
            // PipeWire's realtime module adds `SCHED_RESET_ON_FORK` to the
            // scheduler result. That bit controls inheritance rather than
            // scheduling behavior, so retain only the mode for stable reporting
            // and policy decisions.
            const scheduler: linux.SCHED = @bitCast(
                @as(i32, @intCast(scheduler_policy_result)),
            );
            realtime.callback_scheduler_policy = @intFromEnum(scheduler.mode);
        }

        var scheduler_parameters: linux.sched_param = undefined;
        const scheduler_parameters_result = linux.sched_getparam(0, &scheduler_parameters);
        if (linux.errno(scheduler_parameters_result) == .SUCCESS) {
            realtime.callback_scheduler_priority = scheduler_parameters.priority;
        }
    }

    if (realtime.callback_started_ns_previous > 0) {
        realtime.callback_gap_ns_max = @max(
            realtime.callback_gap_ns_max,
            callback_started_ns -| realtime.callback_started_ns_previous,
        );
    }
    realtime.callback_started_ns_previous = callback_started_ns;
    realtime.callbacks_count += 1;

    // Publish liveness before dequeueing. Even a callback that finds no buffer
    // proves that PipeWire's data path is scheduling this worker; sample access
    // remains governed separately by each slot's publication state.
    audio_exchange.publishAudioCallbacksCount(
        realtime.exchange,
        realtime.callbacks_count,
    );

    assert(realtime.stream != null);
    const stream = realtime.stream.?;
    const pipewire_buffer = pipewire.pw_stream_dequeue_buffer(stream) orelse {
        realtime.missing_buffers_count += 1;
        return;
    };

    // Publishing a terminal result wakes the main-loop thread, but it cannot
    // synchronously prevent PipeWire from scheduling another process callback.
    // Completion and pipeline pressure both publish the active slot and leave
    // no producer-owned destination, so processing another block would reach an
    // impossible-state trap. Once a result is latched, only return any buffer
    // PipeWire still lends us; the first terminal callback remains authoritative.
    if (realtime.terminal_outcome.load(.acquire) != .none) {
        _ = pipewire.pw_stream_queue_buffer(stream, pipewire_buffer);
        return;
    }

    // Registry observation opens this gate only after identifying the first
    // concrete source Link and closes it before reporting removal or replacement.
    // Return any buffer borrowed outside that authorization window: PipeWire
    // buffers contain samples but no source identity, so copying first and
    // diagnosing the graph later could silently mix microphones.
    if (!realtime.source_is_linked.load(.acquire)) {
        const queue_result = pipewire.pw_stream_queue_buffer(stream, pipewire_buffer);
        if (queue_result < 0) {
            realtime.queue_buffer_result = queue_result;
            publishRealtimeOutcome(realtime, .buffer_return_error);
        }
        return;
    }

    // A Format callback closes this gate before rejecting or replacing the
    // negotiated layout. PipeWire can still lend one buffer while that main-
    // loop callback and this data-thread callback overlap. Return it untouched:
    // treating the short authorization gap as malformed audio would turn a
    // harmless repeated Format event into a terminal capture buffer_error.
    if (!realtime.negotiated_format_is_accepted.load(.acquire)) {
        const queue_result = pipewire.pw_stream_queue_buffer(stream, pipewire_buffer);
        if (queue_result < 0) {
            realtime.queue_buffer_result = queue_result;
            publishRealtimeOutcome(realtime, .buffer_return_error);
        }
        return;
    }

    var terminal_outcome: TerminalOutcome = .none;
    const block = validatePipeWireBuffer(realtime, pipewire_buffer) catch |buffer_error| block: {
        terminal_outcome = switch (buffer_error) {
            error.InvalidBuffer => .invalid_buffer,
            error.CorruptedBuffer => .corrupted_buffer,
            error.TimelineDiscontinuity => .timeline_discontinuity,
        };
        break :block null;
    };

    if (block) |audio_block| {
        if (audio_block.samples_count > 0) {
            switch (realtime.timeline_validation) {
                .header_only => {
                    assert(realtime.timeline_buffer_ticks_previous == null);
                    assert(realtime.timeline_graph_rate_num_previous == 0);
                    assert(realtime.timeline_graph_rate_denom_previous == 0);
                },
                .full => validateBufferTimeline(
                    realtime,
                    pipewire_buffer,
                    audio_block.samples_count,
                ) catch |timeline_error| {
                    terminal_outcome = switch (timeline_error) {
                        error.TimelineError => .timeline_error,
                        error.TimelineDiscontinuity => .timeline_discontinuity,
                    };
                },
            }
            if (terminal_outcome == .none) {
                terminal_outcome = publishCompleteBlock(realtime, audio_block);
            }
        }
    }

    // A borrowed buffer must return to PipeWire before the callback wakes the
    // main loop. Waking first would allow stream teardown to race this queue.
    const queue_result = pipewire.pw_stream_queue_buffer(stream, pipewire_buffer);
    if (queue_result < 0) {
        realtime.queue_buffer_result = queue_result;
        terminal_outcome = .buffer_return_error;
    }

    if (terminal_outcome != .none) {
        // The terminal publisher must win the shared latch before exposing its
        // final slot. In particular, a concurrent cancel may already own the
        // outcome and requires this callback's private partial slot to remain
        // unpublished. `publishRealtimeOutcome` performs that atomic decision,
        // publishes only for a winning callback, and wakes the main loop last.
        publishRealtimeOutcome(realtime, terminal_outcome);
    }
}

fn validatePipeWireBuffer(
    realtime: *RealtimeCapture,
    pipewire_buffer: *pipewire.pw_buffer,
) error{ InvalidBuffer, CorruptedBuffer, TimelineDiscontinuity }!?BorrowedAudioBlock {
    assert(realtime.stream != null);
    assert(realtime.negotiated_format_is_accepted.load(.acquire));
    assert(realtime.slot_samples_boundary >= audio_exchange.callback_samples_count_max);

    var buffer_error = BufferError{
        .reason = .none,
        .data_planes_count = 0,
        .data_flags = 0,
        .data_max_size = 0,
        .metadata_count = 0,
        .metadata_size = 0,
        .header_flags = 0,
        .header_sequence = 0,
        .header_presentation_timestamp_ns = 0,
        .chunk_offset = 0,
        .chunk_size = 0,
        .chunk_stride = 0,
        .invalid_sample_index = 0,
    };

    const buffer = pipewire_buffer.*.buffer orelse {
        buffer_error.reason = .missing_spa_buffer;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    };

    // ParamMeta is a request, not a guarantee. PipeWire 1.0.5 accepted Voiced's
    // Header proposal yet omitted metadata on a valid audio-converter stream, so
    // absence cannot make otherwise usable microphone audio fail. When a Header
    // is present, however, its array, size, alignment, and flags remain external
    // input and are validated before any field controls capture policy. The
    // report counts accepted Headers so this capability gap stays observable.
    const metadata_count_max: u32 = 16;
    buffer_error.metadata_count = buffer.*.n_metas;
    if (buffer.*.n_metas > metadata_count_max) {
        buffer_error.reason = .metadata_count_exceeds_realtime_limit;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }
    if (buffer.*.n_metas > 0 and buffer.*.metas == null) {
        buffer_error.reason = .missing_metadata_array;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    var header_optional: ?*const pipewire.spa_meta_header = null;
    for (0..buffer.*.n_metas) |metadata_index| {
        const metadata = &buffer.*.metas[metadata_index];
        if (metadata.*.type != pipewire.SPA_META_Header) continue;
        if (header_optional != null) {
            buffer_error.reason = .duplicate_header_metadata;
            realtime.buffer_error = buffer_error;
            return error.InvalidBuffer;
        }

        buffer_error.metadata_size = metadata.*.size;
        if (metadata.*.size < @sizeOf(pipewire.spa_meta_header)) {
            buffer_error.reason = .undersized_header_metadata;
            realtime.buffer_error = buffer_error;
            return error.InvalidBuffer;
        }
        if (metadata.*.data == null) {
            buffer_error.reason = .missing_header_metadata_data;
            realtime.buffer_error = buffer_error;
            return error.InvalidBuffer;
        }
        if (@intFromPtr(metadata.*.data) % @alignOf(pipewire.spa_meta_header) != 0) {
            buffer_error.reason = .misaligned_header_metadata_data;
            realtime.buffer_error = buffer_error;
            return error.InvalidBuffer;
        }
        header_optional = @ptrCast(@alignCast(metadata.*.data.?));
    }

    var header_marks_silence = false;
    if (header_optional) |header| {
        buffer_error.header_flags = header.*.flags;
        buffer_error.header_sequence = header.*.seq;
        buffer_error.header_presentation_timestamp_ns = header.*.pts;

        // Known flags that matter to encoded media (MARKER, HEADER, and
        // DELTA_UNIT) are harmless for raw float32 audio. Unknown future bits
        // cannot silently acquire policy; GAP, CORRUPTED, and DISCONT are the
        // three flags with explicit behavior below.
        const header_flags_known: u32 = pipewire.SPA_META_HEADER_FLAG_DISCONT |
            pipewire.SPA_META_HEADER_FLAG_CORRUPTED |
            pipewire.SPA_META_HEADER_FLAG_MARKER |
            pipewire.SPA_META_HEADER_FLAG_HEADER |
            pipewire.SPA_META_HEADER_FLAG_GAP |
            pipewire.SPA_META_HEADER_FLAG_DELTA_UNIT;
        if (header.*.flags & ~header_flags_known != 0) {
            buffer_error.reason = .unsupported_header_flags;
            realtime.buffer_error = buffer_error;
            return error.InvalidBuffer;
        }

        realtime.header_metadata_buffers_count += 1;
        realtime.buffer_error = buffer_error;
        if (header.*.flags & pipewire.SPA_META_HEADER_FLAG_CORRUPTED != 0) {
            buffer_error.reason = .header_corrupted;
            realtime.buffer_error = buffer_error;
            return error.CorruptedBuffer;
        }
        if (header.*.flags & pipewire.SPA_META_HEADER_FLAG_DISCONT != 0) {
            // DISCONT identifies an unknown interval, but does not say how many
            // samples disappeared. Inventing zeroes would create a plausible but
            // false timeline, so retain the valid prefix and stop this capture.
            buffer_error.reason = .timeline_discontinuity;
            realtime.buffer_error = buffer_error;
            return error.TimelineDiscontinuity;
        }
        header_marks_silence =
            header.*.flags & pipewire.SPA_META_HEADER_FLAG_GAP != 0;
    }

    buffer_error.data_planes_count = buffer.*.n_datas;
    if (buffer.*.n_datas != 1) {
        buffer_error.reason = .unexpected_data_planes;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }
    if (buffer.*.datas == null) {
        buffer_error.reason = .missing_data_planes;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    const data = &buffer.*.datas[0];
    buffer_error.data_flags = data.*.flags;
    buffer_error.data_max_size = data.*.maxsize;
    if (data.*.chunk == null) {
        buffer_error.reason = .missing_chunk;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    const chunk = data.*.chunk.?;
    buffer_error.chunk_offset = chunk.*.offset;
    buffer_error.chunk_size = chunk.*.size;
    buffer_error.chunk_stride = chunk.*.stride;

    if (chunk.*.flags & pipewire.SPA_CHUNK_FLAG_CORRUPTED != 0) {
        buffer_error.reason = .chunk_corrupted;
        realtime.buffer_error = buffer_error;
        return error.CorruptedBuffer;
    }
    if (data.*.maxsize == 0) {
        buffer_error.reason = .empty_data_capacity;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }
    // PipeWire 0.3.48's audio adapter leaves the per-chunk stride at zero even
    // after accepting our four-byte ParamBuffers stride. That zero means the
    // producer did not restate a dynamic stride; the negotiated packed mono F32
    // format still defines one contiguous four-byte sample per frame. Accept
    // either representation, but reject any contradictory nonzero stride.
    if (chunk.*.stride != 0 and chunk.*.stride != @sizeOf(f32)) {
        buffer_error.reason = .invalid_stride;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    // SPA defines the readable region as `offset mod maxsize` and clamps its
    // size to `maxsize`. The region may cross the allocation's end, so the
    // callback returns two borrowed byte slices rather than manufacturing one
    // out-of-bounds slice.
    const available_size = @min(chunk.*.size, data.*.maxsize);
    if (available_size % @sizeOf(f32) != 0) {
        buffer_error.reason = .incomplete_sample;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }
    if (chunk.*.offset % @alignOf(f32) != 0) {
        buffer_error.reason = .misaligned_sample_offset;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    const samples_count = available_size / @sizeOf(f32);
    if (samples_count > audio_exchange.callback_samples_count_max) {
        buffer_error.reason = .block_exceeds_realtime_limit;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    // Both SPA Header GAP and chunk EMPTY mean that this block's represented
    // duration contains media-neutral audio. Unlike DISCONT, the chunk still
    // states exactly how many samples occupy that interval, so writing the same
    // number of zeroes preserves the recording timeline without reading a
    // payload that PipeWire is permitted to omit. Corruption deliberately took
    // precedence over both silence signals.
    if (header_marks_silence or
        chunk.*.flags & pipewire.VOICED_AUDIO_PIPEWIRE_CHUNK_FLAG_EMPTY != 0)
    {
        realtime.buffer_error = buffer_error;
        return .{
            .first_bytes = &.{},
            .second_bytes = &.{},
            .samples_count = @intCast(samples_count),
            .is_silence = true,
            .header_marked_gap = header_marks_silence,
        };
    }
    if (data.*.data == null) {
        buffer_error.reason = .missing_mapped_data;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }
    if (data.*.flags & pipewire.SPA_DATA_FLAG_READABLE == 0) {
        buffer_error.reason = .unreadable_data;
        realtime.buffer_error = buffer_error;
        return error.InvalidBuffer;
    }

    const source_offset = chunk.*.offset % data.*.maxsize;
    const first_size = @min(available_size, data.*.maxsize - source_offset);
    const second_size = available_size - first_size;
    const source: [*]const u8 = @ptrCast(data.*.data.?);

    // Retain the successful structural metadata too. Content validation runs
    // after the bytes reach aligned owned memory; if it finds NaN or infinity,
    // its diagnostic must describe the actual PipeWire plane and chunk rather
    // than the zero-initialized error record from process startup.
    realtime.buffer_error = buffer_error;
    return .{
        .first_bytes = source[source_offset..][0..first_size],
        .second_bytes = source[0..second_size],
        .samples_count = @intCast(samples_count),
        .is_silence = false,
        .header_marked_gap = false,
    };
}

fn validateBufferTimeline(
    realtime: *RealtimeCapture,
    pipewire_buffer: *const pipewire.pw_buffer,
    block_samples_count: u32,
) error{ TimelineError, TimelineDiscontinuity }!void {
    assert(realtime.stream != null);
    assert(realtime.timeline_validation == .full);
    assert(block_samples_count > 0);
    assert(block_samples_count <= audio_exchange.callback_samples_count_max);
    assert(realtime.timeline_failure.query_result == 0);
    assert(realtime.timeline_failure.code == .none);
    if (realtime.timeline_buffer_ticks_previous == null) {
        assert(realtime.timeline_graph_rate_num_previous == 0);
        assert(realtime.timeline_graph_rate_denom_previous == 0);
    } else {
        assert(realtime.timeline_graph_rate_num_previous > 0);
        assert(realtime.timeline_graph_rate_denom_previous > 0);
    }

    // SPA Header metadata is optional in practice. In full mode, the C boundary
    // safely reads the versioned PipeWire fields that bind this dequeued buffer
    // to one graph cycle. Their documented relationship lets an overloaded
    // main-loop consumer recover the buffer's own tick even when newer graph
    // cycles and buffers are already queued behind it.
    var failure: TimelineFailure = std.mem.zeroes(TimelineFailure);
    failure.block_samples_count = block_samples_count;

    var timeline: pipewire.voiced_audio_pipewire_timeline = undefined;
    var native_error: pipewire.voiced_audio_pipewire_error = undefined;
    if (!pipewire.voiced_audio_pipewire_capture_buffer_timeline(
        realtime.stream.?,
        pipewire_buffer,
        pipewire.VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_FULL,
        &timeline,
        &native_error,
    )) {
        assert(native_error.stage ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_TIMELINE_QUERY);
        assert(native_error.domain ==
            pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT);
        assert(native_error.code < 0);
        assert(native_error.message_size > 0);
        assert(native_error.message_size <= native_error.message.len);
        failure.query_result = native_error.code;
        assert(failure.code == .none);
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }
    assert(native_error.stage == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE);
    assert(native_error.domain == pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE);
    assert(native_error.code == 0);
    assert(native_error.message_size == 0);

    failure.graph_now_ns = timeline.graph_now_ns;
    failure.graph_rate_num = timeline.graph_rate_num;
    failure.graph_rate_denom = timeline.graph_rate_denom;
    failure.graph_ticks = timeline.graph_ticks;
    failure.buffer_cycle_ns = timeline.buffer_cycle_ns;
    if (timeline.graph_rate_num == 0 or timeline.graph_rate_denom == 0) {
        failure.code = .invalid_rate;
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }
    if (timeline.graph_now_ns < 0) {
        failure.code = .invalid_graph_time;
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }

    const graph_now_ns: u64 = @intCast(timeline.graph_now_ns);
    if (timeline.buffer_cycle_ns > graph_now_ns) {
        failure.code = .buffer_from_future;
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }

    // Convert the nanoseconds between this buffer and the latest graph report
    // back into graph ticks using the formula documented for `pw_time`. Round to
    // the nearest tick: hardware clock timestamps can drift by fractions of a
    // graph sample even though the graph tick counter itself remains integral.
    const elapsed_ns = graph_now_ns - timeline.buffer_cycle_ns;
    const elapsed_ticks_numerator =
        @as(u128, elapsed_ns) * @as(u128, timeline.graph_rate_denom);
    const elapsed_ticks_denominator =
        @as(u128, timeline.graph_rate_num) * @as(u128, std.time.ns_per_s);
    const elapsed_ticks = (elapsed_ticks_numerator + elapsed_ticks_denominator / 2) /
        elapsed_ticks_denominator;
    if (elapsed_ticks > timeline.graph_ticks) {
        failure.code = .tick_underflow;
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }
    const buffer_ticks = timeline.graph_ticks - @as(u64, @intCast(elapsed_ticks));
    failure.buffer_ticks_current = buffer_ticks;

    const previous_buffer_ticks = realtime.timeline_buffer_ticks_previous orelse {
        realtime.timeline_buffer_ticks_previous = buffer_ticks;
        realtime.timeline_graph_rate_num_previous = timeline.graph_rate_num;
        realtime.timeline_graph_rate_denom_previous = timeline.graph_rate_denom;

        assert(realtime.timeline_buffer_ticks_previous == buffer_ticks);
        assert(realtime.timeline_graph_rate_num_previous > 0);
        assert(realtime.timeline_graph_rate_denom_previous > 0);
        assert(realtime.timeline_failure.query_result == 0);
        assert(realtime.timeline_failure.code == .none);
        return;
    };
    failure.buffer_ticks_previous = previous_buffer_ticks;
    assert(realtime.timeline_graph_rate_num_previous > 0);
    assert(realtime.timeline_graph_rate_denom_previous > 0);

    if (timeline.graph_rate_num != realtime.timeline_graph_rate_num_previous or
        timeline.graph_rate_denom != realtime.timeline_graph_rate_denom_previous)
    {
        failure.code = .rate_changed;
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }
    if (buffer_ticks < previous_buffer_ticks) {
        failure.code = .ticks_regressed;
        realtime.timeline_failure = failure;
        return error.TimelineError;
    }

    // Compare durations without floating point. The left side is the elapsed
    // graph duration scaled by Voiced's 16 kHz output rate; the right side is
    // this block's duration scaled by the graph rate denominator. A resampler
    // must round integral output samples, so one output sample is the complete
    // tolerance. Baseline runs and stable 256-, 1024-, and 2048-frame graph
    // quanta on PipeWire 1.0.5 produced residuals of -2 through +1 graph ticks.
    // Quantum transitions, SIGSTOP, and CPU pressure that actually lost audio
    // exceeded this bound by at least 344 graph ticks.
    const buffer_ticks_delta = buffer_ticks - previous_buffer_ticks;
    const graph_duration_scaled = @as(u128, buffer_ticks_delta) *
        @as(u128, timeline.graph_rate_num) *
        @as(u128, audio_exchange.sample_rate_hz);
    const block_duration_scaled = @as(u128, block_samples_count) *
        @as(u128, timeline.graph_rate_denom);
    const duration_difference = if (graph_duration_scaled >= block_duration_scaled)
        graph_duration_scaled - block_duration_scaled
    else
        block_duration_scaled - graph_duration_scaled;
    const one_output_sample_tolerance = @as(u128, timeline.graph_rate_denom);
    if (duration_difference > one_output_sample_tolerance) {
        failure.code = .discontinuity;
        realtime.timeline_failure = failure;
        return error.TimelineDiscontinuity;
    }

    realtime.timeline_buffer_ticks_previous = buffer_ticks;
    realtime.timeline_graph_rate_num_previous = timeline.graph_rate_num;
    realtime.timeline_graph_rate_denom_previous = timeline.graph_rate_denom;

    assert(realtime.timeline_buffer_ticks_previous.? >= previous_buffer_ticks);
    assert(realtime.timeline_graph_rate_num_previous == timeline.graph_rate_num);
    assert(realtime.timeline_graph_rate_denom_previous == timeline.graph_rate_denom);
    assert(realtime.timeline_failure.query_result == 0);
    assert(realtime.timeline_failure.code == .none);
}

fn publishCompleteBlock(
    realtime: *RealtimeCapture,
    block: BorrowedAudioBlock,
) TerminalOutcome {
    if (block.is_silence) {
        assert(block.first_bytes.len == 0);
        assert(block.second_bytes.len == 0);
    } else {
        assert(block.first_bytes.len + block.second_bytes.len == block.samples_count * @sizeOf(f32));
    }
    assert(block.samples_count > 0);
    assert(block.samples_count <= audio_exchange.callback_samples_count_max);
    assert(block.samples_count <= realtime.slot_samples_boundary);
    assert(realtime.active_slot_samples_count <= realtime.slot_samples_boundary);
    assert(realtime.samples_count <
        realtime.recording_samples_target + audio_exchange.callback_samples_count_max);

    // Slots contain complete callback blocks only. If this block does not fit,
    // publish the preceding prefix and claim any slot already released by the
    // consumer. Failing that claim is real pipeline pressure: overwriting a
    // published slot would race the model process and corrupt retained audio.
    if (block.samples_count >
        realtime.slot_samples_boundary - realtime.active_slot_samples_count)
    {
        publishActiveSlotIfNonEmpty(realtime);
        if (!beginNextAvailableSlot(realtime)) {
            return .pipeline_full;
        }
    }

    assert(realtime.active_slot_index != null);
    const active_slot_index = realtime.active_slot_index.?;
    assert(active_slot_index < audio_exchange.slots_count);
    const slot: *AudioSlot = &realtime.exchange.slots[active_slot_index];
    const destination_samples =
        slot.samples[realtime.active_slot_samples_count..][0..block.samples_count];
    const destination_bytes = std.mem.sliceAsBytes(destination_samples);

    if (block.is_silence) {
        @memset(destination_bytes, 0);
    } else {
        @memcpy(destination_bytes[0..block.first_bytes.len], block.first_bytes);
        @memcpy(destination_bytes[block.first_bytes.len..], block.second_bytes);

        // Float32 permits NaN, infinity, and enormous finite values, but
        // PipeWire's normalized F32 audio contract is -1.0 through +1.0.
        // Whisper squares FFT magnitudes, so merely requiring finiteness would
        // still let a maximum finite sample overflow and poison later features.
        // Reject non-finite input, but clamp ordinary overdriven audio and count
        // each clamp for diagnostics instead of terminating a real recording.
        // This bounded pass happens in aligned owned memory before any part of
        // the current block becomes visible to another process.
        var block_clipped_samples_count: u32 = 0;
        for (destination_samples, 0..) |*sample, sample_index| {
            if (!std.math.isFinite(sample.*)) {
                realtime.buffer_error.reason = .non_finite_sample;
                realtime.buffer_error.invalid_sample_index = @intCast(sample_index);
                return .invalid_buffer;
            }
            if (sample.* > 1.0) {
                sample.* = 1.0;
                block_clipped_samples_count += 1;
            } else if (sample.* < -1.0) {
                sample.* = -1.0;
                block_clipped_samples_count += 1;
            }
        }
        realtime.clipped_samples_count += block_clipped_samples_count;
    }

    // Count a Header gap only after its zero block has entered an owned slot.
    // Validation alone is insufficient: pipeline pressure can reject a valid
    // borrowed block before publication, and diagnostics must describe retained
    // audio rather than samples that were returned to PipeWire untouched.
    if (block.header_marked_gap) {
        assert(block.is_silence);
        realtime.header_gap_buffers_count += 1;
        realtime.header_gap_samples_count += block.samples_count;
    }

    realtime.active_slot_samples_count += block.samples_count;
    realtime.samples_count += block.samples_count;
    assert(realtime.active_slot_samples_count <= realtime.slot_samples_boundary);
    assert(realtime.samples_count <=
        realtime.recording_samples_target + audio_exchange.callback_samples_count_max);
    audio_exchange.publishAudioSamplesCount(
        realtime.exchange,
        realtime.samples_count,
    );

    realtime.block_samples_count_min = @min(
        realtime.block_samples_count_min,
        block.samples_count,
    );
    realtime.block_samples_count_max = @max(
        realtime.block_samples_count_max,
        block.samples_count,
    );

    if (realtime.samples_count >= realtime.recording_samples_target) {
        return .completed;
    }
    return .none;
}

fn beginNextAvailableSlot(realtime: *RealtimeCapture) bool {
    assert(realtime.exchange.generation > 0);
    assert(realtime.active_slot_index == null);
    assert(realtime.active_slot_samples_count == 0);
    assert(realtime.next_publication_ordinal <= realtime.slot_publications_count + 1);

    // Prefer cyclic physical reuse so a healthy consumer spreads writes across
    // all slots, but accept any released slot. Scanning the fixed three entries
    // keeps acquisition work strictly bounded on the realtime thread.
    const preferred_slot_index =
        realtime.next_publication_ordinal % audio_exchange.slots_count;
    for (0..audio_exchange.slots_count) |slot_offset| {
        const slot_index: u8 = @intCast(
            (preferred_slot_index + slot_offset) % audio_exchange.slots_count,
        );
        if (!audio_exchange.tryBeginWrite(
            &realtime.exchange.slots[slot_index],
            realtime.exchange.generation,
            realtime.next_publication_ordinal,
        )) {
            continue;
        }

        realtime.active_slot_index = slot_index;
        realtime.next_publication_ordinal += 1;

        assert(realtime.active_slot_index != null);
        assert(realtime.active_slot_samples_count == 0);
        assert(realtime.next_publication_ordinal == realtime.slot_publications_count + 1);
        return true;
    }

    return false;
}

fn publishActiveSlotIfNonEmpty(realtime: *RealtimeCapture) void {
    assert(realtime.active_slot_samples_count <= realtime.slot_samples_boundary);
    if (realtime.active_slot_index == null) {
        assert(realtime.active_slot_samples_count == 0);
        return;
    }
    const active_slot_index = realtime.active_slot_index.?;
    if (realtime.active_slot_samples_count == 0) return;

    const published_samples_count = realtime.active_slot_samples_count;
    audio_exchange.publishWrittenSlot(
        &realtime.exchange.slots[active_slot_index],
        published_samples_count,
    );
    realtime.active_slot_index = null;
    realtime.active_slot_samples_count = 0;
    realtime.published_samples_count += published_samples_count;
    realtime.slot_publications_count += 1;

    assert(realtime.active_slot_index == null);
    assert(realtime.active_slot_samples_count == 0);
    assert(realtime.published_samples_count <= realtime.samples_count);
    assert(realtime.slot_publications_count == realtime.next_publication_ordinal);
}

fn abandonUnpublishedActiveSlot(realtime: *RealtimeCapture) void {
    if (realtime.active_slot_index == null) {
        assert(realtime.active_slot_samples_count == 0);
        return;
    }
    const active_slot_index = realtime.active_slot_index.?;
    assert(realtime.active_slot_samples_count <= realtime.slot_samples_boundary);

    // Samples may have been copied into this slot and counted as callback
    // progress, but the slot metadata remains unpublished. Returning ownership
    // therefore exposes none of those private bytes to a consumer. Cancel uses
    // this distinction to retain diagnostics while discarding its final prefix.
    audio_exchange.abandonEmptyWrite(&realtime.exchange.slots[active_slot_index]);
    realtime.active_slot_index = null;
    realtime.active_slot_samples_count = 0;

    assert(realtime.active_slot_index == null);
    assert(realtime.active_slot_samples_count == 0);
    assert(realtime.published_samples_count <= realtime.samples_count);
}

fn publishRealtimeOutcome(
    realtime: *RealtimeCapture,
    outcome: TerminalOutcome,
) void {
    assert(outcome != .none);
    assert(realtime.runtime_error.stage == .none);
    assert(realtime.runtime_error.domain == .none);
    assert(realtime.runtime_error.code == 0);

    // The realtime callback cannot format text or allocate, so it publishes one
    // bounded numeric error beside the terminal outcome. The main-loop thread
    // turns callback-owned buffer details into prose only after stream teardown.
    switch (outcome) {
        .completed => {},
        .pipeline_full => recordRuntimeError(
            &realtime.runtime_error,
            .exchange_publication,
            .voiced_audio,
            @intFromEnum(AudioErrorCode.exchange_full),
        ),
        .invalid_buffer => {
            const error_stage: RuntimeErrorStage = switch (realtime.buffer_error.reason) {
                .none,
                .header_corrupted,
                .timeline_discontinuity,
                .chunk_corrupted,
                => unreachable,
                .metadata_count_exceeds_realtime_limit,
                .missing_metadata_array,
                .duplicate_header_metadata,
                .undersized_header_metadata,
                .missing_header_metadata_data,
                .misaligned_header_metadata_data,
                .unsupported_header_flags,
                => .buffer_metadata,
                .missing_spa_buffer,
                .unexpected_data_planes,
                .missing_data_planes,
                .missing_chunk,
                .missing_mapped_data,
                .unreadable_data,
                .empty_data_capacity,
                .invalid_stride,
                .incomplete_sample,
                .misaligned_sample_offset,
                .block_exceeds_realtime_limit,
                => .buffer_data,
                .non_finite_sample => .sample_validation,
            };
            recordRuntimeError(
                &realtime.runtime_error,
                error_stage,
                .spa_buffer,
                @intFromEnum(realtime.buffer_error.reason),
            );
        },
        .corrupted_buffer => {
            const error_stage: RuntimeErrorStage = switch (realtime.buffer_error.reason) {
                .header_corrupted => .buffer_metadata,
                .chunk_corrupted => .buffer_data,
                else => unreachable,
            };
            recordRuntimeError(
                &realtime.runtime_error,
                error_stage,
                .spa_buffer,
                @intFromEnum(realtime.buffer_error.reason),
            );
        },
        .timeline_error => {
            const failure = realtime.timeline_failure;
            if (failure.query_result < 0) {
                assert(failure.code == .none);
                recordRuntimeError(
                    &realtime.runtime_error,
                    .timeline_continuity,
                    .pipewire_result,
                    failure.query_result,
                );
            } else {
                assert(failure.code != .none);
                assert(failure.code != .discontinuity);
                recordRuntimeError(
                    &realtime.runtime_error,
                    .timeline_continuity,
                    .voiced_audio,
                    @intFromEnum(failure.code),
                );
            }
        },
        .timeline_discontinuity => {
            if (realtime.timeline_failure.code == .discontinuity) {
                recordRuntimeError(
                    &realtime.runtime_error,
                    .timeline_continuity,
                    .voiced_audio,
                    @intFromEnum(TimelineErrorCode.discontinuity),
                );
            } else {
                assert(realtime.buffer_error.reason == .timeline_discontinuity);
                recordRuntimeError(
                    &realtime.runtime_error,
                    .timeline_continuity,
                    .spa_buffer,
                    @intFromEnum(realtime.buffer_error.reason),
                );
            }
        },
        .buffer_return_error => {
            assert(realtime.queue_buffer_result < 0);
            recordRuntimeError(
                &realtime.runtime_error,
                .buffer_return,
                .pipewire_result,
                realtime.queue_buffer_result,
            );
        },
        .stopped,
        .cancelled,
        .control_error,
        .format_parse_error,
        .unexpected_format,
        .buffer_configuration_error,
        .stream_error,
        .stream_disconnected,
        .source_disconnected,
        .source_changed,
        .source_observation_error,
        => unreachable,
        .none => unreachable,
    }

    const outcome_previous = realtime.terminal_outcome.cmpxchgStrong(
        .none,
        outcome,
        .release,
        .monotonic,
    );
    if (outcome_previous != null) {
        assert(outcome_previous.? != .none);
        assert(realtime.terminal_outcome.load(.acquire) != .none);
        return;
    }
    assert(realtime.terminal_outcome.load(.acquire) == outcome);
    if (outcome == .completed) {
        assert(realtime.runtime_error.stage == .none);
        assert(realtime.runtime_error.domain == .none);
        assert(realtime.runtime_error.code == 0);
    } else {
        assert(realtime.runtime_error.stage != .none);
        assert(realtime.runtime_error.domain != .none);
    }

    // Winning the latch establishes this callback as the terminal owner. Publish
    // its complete callback-block prefix before signaling the main loop. A
    // callback that lost to cancel returned above and leaves the active slot
    // private for teardown to abandon after stream destruction.
    publishActiveSlotIfNonEmpty(realtime);

    var signal: u64 = 1;
    const signal_bytes = std.mem.asBytes(&signal);
    while (true) {
        const write_result = linux.write(
            realtime.callback_event_fd,
            signal_bytes.ptr,
            signal_bytes.len,
        );
        switch (linux.errno(write_result)) {
            .SUCCESS => return,
            // A signal can interrupt the syscall before eventfd observes it;
            // retrying one fixed nonblocking write preserves the already latched
            // outcome without adding unbounded data work to this callback.
            .INTR => continue,
            // EAGAIN means the event counter was already saturated and therefore
            // readable. The main-loop source will still wake and inspect the
            // terminal outcome published before this attempted write.
            .AGAIN => return,
            // Any other error leaves no reliable callback-to-loop wake path.
            // Trap only this isolated worker so supervisor deadlines and crash
            // classification contain the failed native audio process.
            else => @trap(),
        }
    }
}

fn claimMainLoopOutcome(audio: *Audio, outcome: Outcome) bool {
    // PipeWire dispatches state, format, graph, and eventfd callbacks on the loop
    // thread while processAudio may finish concurrently on the data thread.
    // All terminal paths compete through this one atomic latch, so a teardown
    // notification cannot rewrite an earlier completion or error merely
    // because its callback ran last.
    const terminal_outcome: TerminalOutcome = switch (outcome) {
        .completed => .completed,
        .stopped => .stopped,
        .cancelled => .cancelled,
        .control_error => .control_error,
        .pipeline_full => .pipeline_full,
        .format_parse_error => .format_parse_error,
        .unexpected_format => .unexpected_format,
        .buffer_configuration_error => .buffer_configuration_error,
        .timeline_error => .timeline_error,
        .timeline_discontinuity => .timeline_discontinuity,
        .stream_error => .stream_error,
        .stream_disconnected => .stream_disconnected,
        .source_disconnected => .source_disconnected,
        .source_changed => .source_changed,
        .source_observation_error => .source_observation_error,
        .invalid_buffer => .invalid_buffer,
        .corrupted_buffer => .corrupted_buffer,
        .buffer_return_error => .buffer_return_error,
    };
    const outcome_previous = audio.realtime.terminal_outcome.cmpxchgStrong(
        .none,
        terminal_outcome,
        .release,
        .monotonic,
    );
    const outcome_was_claimed = outcome_previous == null;
    if (outcome_was_claimed) {
        assert(audio.realtime.terminal_outcome.load(.acquire) == terminal_outcome);
    } else {
        assert(outcome_previous.? != .none);
        assert(audio.realtime.terminal_outcome.load(.acquire) != .none);
    }
    return outcome_was_claimed;
}

fn recordRuntimeError(
    runtime_error: *RuntimeError,
    stage: RuntimeErrorStage,
    domain: ErrorDomain,
    code: i64,
) void {
    assert(runtime_error.stage == .none);
    assert(runtime_error.domain == .none);
    assert(runtime_error.code == 0);
    assert(stage != .none);
    assert(domain != .none);

    runtime_error.* = .{
        .stage = stage,
        .domain = domain,
        .code = code,
    };

    assert(runtime_error.stage == stage);
    assert(runtime_error.domain == domain);
    assert(runtime_error.code == code);
}

fn recordSetupError(
    setup_error: *SetupError,
    stage: SetupErrorStage,
    domain: ErrorDomain,
    code: i64,
    native_message: []const u8,
    comptime detail_format: []const u8,
    detail_arguments: anytype,
) void {
    assert(setup_error.stage == .none);
    assert(setup_error.domain == .none);
    assert(stage != .none);
    assert(domain != .none);
    assert(native_message.len <= pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_MESSAGE_CAPACITY);

    var detail_buffer: [setup_error_message_capacity]u8 = undefined;
    const detail = std.fmt.bufPrint(
        &detail_buffer,
        detail_format,
        detail_arguments,
    ) catch @panic("audio setup detail exceeded its fixed buffer");
    const message = if (native_message.len == 0)
        std.fmt.bufPrint(
            &setup_error.message,
            "{s}",
            .{detail},
        ) catch @panic("audio setup diagnostic exceeded its fixed buffer")
    else
        std.fmt.bufPrint(
            &setup_error.message,
            "{s}; native code {d}: {s}",
            .{ detail, code, native_message },
        ) catch @panic("audio setup diagnostic exceeded its fixed buffer");

    const pipewire_version_pointer = pipewire.pw_get_library_version();
    assert(pipewire_version_pointer != null);
    const pipewire_version = std.mem.span(pipewire_version_pointer);
    const pipewire_version_size = @min(
        pipewire_version.len,
        setup_error.pipewire_version.len,
    );
    @memcpy(
        setup_error.pipewire_version[0..pipewire_version_size],
        pipewire_version[0..pipewire_version_size],
    );

    setup_error.stage = stage;
    setup_error.domain = domain;
    setup_error.code = code;
    setup_error.message_size = @intCast(message.len);
    setup_error.pipewire_version_size = @intCast(pipewire_version_size);

    assert(setup_error.message_size > 0);
    assert(setup_error.message_size <= setup_error.message.len);
    assert(setup_error.pipewire_version_size > 0);
    assert(setup_error.pipewire_version_size <= setup_error.pipewire_version.len);
}

fn copyNativeErrorMessage(
    audio: *Audio,
    native_error: [*c]const pipewire.voiced_audio_pipewire_error,
) void {
    // The C callback adapter has already copied PipeWire's borrowed string into
    // this bounded value. Zig copies it once more because the adapter's value is
    // stack-local to the callback, while `Audio` must retain the diagnostic until
    // stream teardown completes and the worker constructs its final report.
    assert(native_error != null);
    assert(native_error.*.stage ==
        pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_STATE);
    assert(native_error.*.domain ==
        pipewire.VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_CALLBACK);
    assert(native_error.*.code == 0);
    assert(native_error.*.message_size > 0);
    assert(native_error.*.message_size <= native_error.*.message.len);
    assert(native_error.*.message_size <= audio.error_message.len);

    const message_size = native_error.*.message_size;
    @memcpy(
        audio.error_message[0..message_size],
        native_error.*.message[0..message_size],
    );
    audio.error_message_size = @intCast(message_size);
    assert(audio.error_message_size > 0);
    assert(audio.error_message_size <= audio.error_message.len);
}

fn writeErrorMessage(
    audio: *Audio,
    comptime format: []const u8,
    arguments: anytype,
) void {
    const message = std.fmt.bufPrint(
        &audio.error_message,
        format,
        arguments,
    ) catch @panic("audio diagnostic exceeded its fixed buffer");
    audio.error_message_size = @intCast(message.len);
    assert(audio.error_message_size > 0);
    assert(audio.error_message_size <= audio.error_message.len);
}

fn writeTeardownErrorMessage(
    audio: *Audio,
    comptime format: []const u8,
    arguments: anytype,
) void {
    const message = std.fmt.bufPrint(
        &audio.teardown_error_message,
        format,
        arguments,
    ) catch @panic("audio teardown diagnostic exceeded its fixed buffer");
    audio.teardown_error_message_size = @intCast(message.len);
    assert(audio.teardown_error_message_size > 0);
    assert(audio.teardown_error_message_size <= audio.teardown_error_message.len);
}

fn monotonicNanoseconds() u64 {
    var timestamp: linux.timespec = undefined;
    const result = linux.clock_gettime(.MONOTONIC, &timestamp);
    assert(linux.errno(result) == .SUCCESS);
    assert(timestamp.sec >= 0);
    assert(timestamp.nsec >= 0);

    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

fn closeFileDescriptor(file_descriptor: std.posix.fd_t) void {
    assert(file_descriptor >= 0);
    const result = linux.close(file_descriptor);
    assert(linux.errno(result) == .SUCCESS);
}

pub fn schedulerPolicyName(policy: i32) []const u8 {
    const name = switch (policy) {
        @intFromEnum(linux.SCHED.Mode.FIFO) => "FIFO",
        @intFromEnum(linux.SCHED.Mode.RR) => "round-robin",
        @intFromEnum(linux.SCHED.Mode.NORMAL) => "normal",
        else => "other",
    };
    assert(name.len > 0);
    return name;
}

pub fn schedulerPolicyIsRealtime(policy: i32) bool {
    return policy == @intFromEnum(linux.SCHED.Mode.FIFO) or
        policy == @intFromEnum(linux.SCHED.Mode.RR);
}

pub fn linuxErrorNameFromCode(error_code: i32) []const u8 {
    assert(error_code > 0);
    return @tagName(@as(linux.E, @enumFromInt(error_code)));
}
