//! Process-owned diagnostics. Service records use either one bounded
//! nonblocking journal datagram or one explicit stderr line; recording and
//! inference hot paths must not call this API. Initialize before starting
//! threads, and close only after they have joined.
const std = @import("std");
const linux = std.os.linux;
const decimal = @import("decimal.zig");

pub const Level = enum(u8) { critical = 2, err = 3, warn = 4, info = 6, debug = 7 };
pub const Target = enum { auto, journal, stderr };
pub const Context = struct { recording_id: ?u64 = null };
const record_bytes_max = 4096;
const message_bytes_max = 3584;
const Sink = union(enum) { disabled, journal: struct { fd: i32, connected: bool }, stderr };
var sink: Sink = .disabled;
var threshold: Level = .info;
var dropped: std.atomic.Value(u64) = .init(0);

/// The auto target selects stderr when stderr is a terminal and journal
/// otherwise. The journal target sends to journald's pathname on each call, so
/// a restart requires no reconnect state. A connected datagram on stderr
/// supplies an isolated journal receiver for tests. The stderr target writes one
/// bounded human-readable line to the inherited descriptor; a pipe or slow
/// terminal can therefore block its calling thread and should be actively drained.
pub fn init(io: std.Io, target: Target, configured_level: Level) linux.E {
    std.debug.assert(sink == .disabled);

    threshold = configured_level;

    const resolved_target: Target = if (target == .auto)
        if (std.Io.File.stderr().isTty(io) catch false) .stderr else .journal
    else
        target;

    if (resolved_target == .stderr) {
        sink = .stderr;

        return .SUCCESS;
    }

    var socket_type: i32 = 0;
    var length: linux.socklen_t = @sizeOf(i32);

    if (linux.errno(linux.getsockopt(2, linux.SOL.SOCKET, linux.SO.TYPE, @ptrCast(&socket_type), &length)) == .SUCCESS and socket_type == linux.SOCK.DGRAM) {
        const result = linux.fcntl(2, linux.F.DUPFD_CLOEXEC, 3);
        if (linux.errno(result) != .SUCCESS) {
            return linux.errno(result);
        }

        sink = .{ .journal = .{ .fd = @intCast(result), .connected = true } };
    } else {
        const result = linux.socket(linux.AF.UNIX, linux.SOCK.DGRAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0);
        if (linux.errno(result) != .SUCCESS) {
            return linux.errno(result);
        }

        sink = .{ .journal = .{ .fd = @intCast(result), .connected = false } };
    }

    return .SUCCESS;
}

/// Offline CLI tools may use stderr; they do not own service deadlines. Help,
/// argument errors, and command results need not go through the logger.
pub fn initCli(configured_level: Level) void {
    std.debug.assert(sink == .disabled);

    threshold = configured_level;
    sink = .stderr;
}

pub fn deinit() void {
    if (sink == .journal) {
        _ = linux.close(sink.journal.fd);
    }

    sink = .disabled;

    dropped.store(0, .monotonic);
}

pub fn activeTarget() Target {
    return switch (sink) {
        .journal => .journal,
        .stderr => .stderr,
        .disabled => unreachable,
    };
}

pub fn parseLevel(text: []const u8) ?Level {
    if (std.mem.eql(u8, text, "error")) {
        return .err;
    }

    if (std.mem.eql(u8, text, "warn")) {
        return .warn;
    }

    if (std.mem.eql(u8, text, "critical")) {
        return .critical;
    }

    if (std.mem.eql(u8, text, "info")) {
        return .info;
    }

    if (std.mem.eql(u8, text, "debug")) {
        return .debug;
    }

    return null;
}

pub fn parseTarget(text: []const u8) ?Target {
    return std.meta.stringToEnum(Target, text);
}

pub fn levelName(value: Level) []const u8 {
    return if (value == .err)
        "error"
    else
        @tagName(value);
}

/// Returns a `{f}` formatter for a named Linux errno, with the same text as
/// `{t}` and no width or alignment. Like `{t}`, unnamed enum values are invalid.
/// Name lookup happens only when formatted; writer errors propagate unchanged.
pub fn fmtErrno(value: linux.E) ErrnoFormat {
    return .{ .value = value };
}

const ErrnoFormat = struct {
    value: linux.E,

    // PERFORMANCE: Separate name selection from writing, and share both on
    // diagnostic paths. Formatting linux.E directly with `{t}` lets LLVM copy
    // writer logic into its many tag branches. The separate lookup returns one
    // slice to one writer instead, at the cost of a call. Keep conversion lazy
    // so a filtered log never performs the lookup.
    pub noinline fn format(self: ErrnoFormat, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.writeAll(errnoName(self.value));
    }

    // PERFORMANCE: Compile-time offsets into one string replace the many
    // address/length selection branches generated for @tagName(linux.E).
    // This saved about 3.2 KiB in the stripped x86_64 Zig 0.16 release binary
    // (application/stdlib ReleaseSmall, inference ReleaseFast).
    const names = blk: {
        var bytes: []const u8 = "";

        for (std.meta.fields(linux.E)) |field| {
            bytes = bytes ++ field.name;
        }

        break :blk bytes;
    };

    const names_index = blk: {
        @setEvalBranchQuota(10000);

        const fields = std.meta.fields(linux.E);

        var maximum: usize = 0;

        for (fields) |field| {
            maximum = @max(maximum, field.value);
        }

        // Enum values may have gaps. Unnamed entries retain size zero.
        var entries: [maximum + 1]struct {
            offset: u16 = 0,
            size: u8 = 0,
        } = @splat(.{});

        var offset: usize = 0;

        for (fields) |field| {
            entries[field.value] = .{
                .offset = @intCast(offset),
                .size = @intCast(field.name.len),
            };
            offset += field.name.len;
        }

        break :blk entries;
    };

    noinline fn errnoName(value: linux.E) []const u8 {
        const index = @intFromEnum(value);
        std.debug.assert(index < names_index.len);

        const entry = names_index[index];
        std.debug.assert(entry.size > 0);

        return names[entry.offset..][0..entry.size];
    }
};

/// Event emission filters before formatting. Guard expensive argument
/// preparation at the call site with enabled(); Zig evaluates arguments first.
pub fn enabled(value: Level) bool {
    return sink != .disabled and @intFromEnum(value) <= @intFromEnum(threshold);
}

const Field = union(enum) {
    /// Writes a lower-case ASCII identifier without quotes.
    name: []const u8,
    /// Writes arbitrary bytes inside quotes with Zig-style escapes.
    str: []const u8,
    /// Writes slices consecutively as one quoted, escaped string.
    str_joined: []const []const u8,
    /// Writes a trusted, nonempty ASCII value without quotes. The value must not
    /// contain whitespace, controls, quotes, or `=`.
    verbatim: []const u8,
    u: u64,
    i: i64,
    octal: u64,
    f: struct { value: f64, digits: u3 },
    /// Preserves f32 decimal rounding; widening to f64 can change the last digit.
    f32: struct { value: f32, digits: u3 },
    b: bool,
    errno: linux.E,
};

pub fn float(value: f64, digits: u3) @FieldType(Field, "f") {
    return .{ .value = value, .digits = digits };
}

pub fn float32(value: f32, digits: u3) @FieldType(Field, "f32") {
    return .{ .value = value, .digits = digits };
}

const Entry = struct { []const u8, Field };
const FieldKind = std.meta.Tag(Field);
const FieldDefinition = struct { kind: FieldKind };

fn defineField(comptime kind: FieldKind) FieldDefinition {
    return .{ .kind = kind };
}

// One process-wide vocabulary binds every key to one semantic representation.
// Event schemas select from this catalog; they cannot redefine a shared key.
// Keep names aligned with public configuration where they describe the same
// value, and prefer unit-bearing/count-bearing names for measured quantities.
const field_catalog = .{
    // Lower-case identifiers: states, policies, outcomes, stages, and codes.
    .action = defineField(.name),
    .activity = defineField(.name),
    .cause_domain = defineField(.name),
    .chunk_limit = defineField(.name),
    .clipboard_backend = defineField(.name),
    .clipboard_backend_policy = defineField(.name),
    .clipboard_candidate = defineField(.name),
    .clipboard_error_kind = defineField(.name),
    .clipboard_fallback = defineField(.name),
    .clipboard_feature = defineField(.name),
    .clipboard_mode = defineField(.name),
    .clipboard_outcome = defineField(.name),
    .clipboard_phase = defineField(.name),
    .clipboard_skip_reason = defineField(.name),
    .command = defineField(.name),
    .disposition = defineField(.name),
    .inference_outcome = defineField(.name),
    .log_level = defineField(.name),
    .log_target = defineField(.name),
    .memory_lock_outcome = defineField(.name),
    .notification_mode = defineField(.name),
    .operation = defineField(.name),
    .outcome = defineField(.name),
    .paste_observation = defineField(.name),
    .paste_outcome = defineField(.name),
    .phase = defineField(.name),
    .problem_code = defineField(.name),
    .problem_component = defineField(.name),
    .reason = defineField(.name),
    .save_outcome = defineField(.name),
    .scheduler_policy = defineField(.name),
    .source = defineField(.name),
    .stage = defineField(.name),
    .stop_origin = defineField(.name),
    .transcript_output = defineField(.name),
    .transport = defineField(.name),
    .worker_storage = defineField(.name),

    // Arbitrary or externally supplied text is always quoted and escaped.
    .bus_error = defineField(.str),
    .cleanup_error = defineField(.str),
    .clipboard_error = defineField(.str),
    .detail = defineField(.str),
    .directory_path = defineField(.str),
    .@"error" = defineField(.str),
    .file_name = defineField(.str),
    .microphone = defineField(.str),
    .microphone_description = defineField(.str),
    .microphone_device_description = defineField(.str),
    .microphone_node = defineField(.str),
    .microphone_node_name = defineField(.str),
    .microphone_serial = defineField(.str),
    .model = defineField(.str),
    .name = defineField(.str),
    .owner_next = defineField(.str),
    .owner_previous = defineField(.str),
    .path = defineField(.str),
    .pipewire_server_version = defineField(.str),
    .response = defineField(.str),
    .temporary_name = defineField(.str),
    .wayland_server_error_message = defineField(.str),
    .x11_setup_message = defineField(.str),

    // Trusted symbolic values that require punctuation not accepted by `.name`.
    .paste_shortcut = defineField(.verbatim),

    // Nonnegative counts, sizes, identifiers, protocol values, and policy limits.
    .activity_active_run_samples_count_max = defineField(.u),
    .activity_active_samples_count = defineField(.u),
    .activity_changes_count = defineField(.u),
    .activity_observed_samples_count = defineField(.u),
    .activity_quiet_run_samples_count_max = defineField(.u),
    .activity_quiet_samples_count = defineField(.u),
    .activity_samples_count = defineField(.u),
    .activity_unknown_samples_count = defineField(.u),
    .argument = defineField(.u),
    .audio_samples_captured_count = defineField(.u),
    .audio_samples_count = defineField(.u),
    .audio_samples_published_count = defineField(.u),
    .audio_slots_published_count = defineField(.u),
    .bustype = defineField(.u),
    .bytes_total = defineField(.u),
    .bytes_written = defineField(.u),
    .callback_clipped_samples_count = defineField(.u),
    .callback_header_gap_buffers_count = defineField(.u),
    .callback_header_gap_samples_count = defineField(.u),
    .callback_header_metadata_buffers_count = defineField(.u),
    .callback_missing_buffers_count = defineField(.u),
    .callback_samples_count_max = defineField(.u),
    .callback_samples_count_min = defineField(.u),
    .callbacks_count = defineField(.u),
    .chunk_id = defineField(.u),
    .chunk_retained_size = defineField(.u),
    .chunk_transcript_size = defineField(.u),
    .clipboard_transfers_completed_count = defineField(.u),
    .clipboard_transfers_expired_count = defineField(.u),
    .clipboard_transfers_rejected_count = defineField(.u),
    .deadline_ns = defineField(.u),
    .encoder_positions_count = defineField(.u),
    .expected_size = defineField(.u),
    .ff_effects_max = defineField(.u),
    .frame = defineField(.u),
    .frame_bytes_sent = defineField(.u),
    .graph_channels_count = defineField(.u),
    .graph_rate_hz = defineField(.u),
    .memory_lock_size_max = defineField(.u),
    .model_decoder_threads = defineField(.u),
    .model_encoder_padding_seconds = defineField(.u),
    .model_encoder_threads = defineField(.u),
    .model_idle_seconds_max = defineField(.u),
    .model_runtime_size = defineField(.u),
    .notification_close_reason = defineField(.u),
    .notification_id = defineField(.u),
    .observed_ns = defineField(.u),
    .paste_key_gap_ms = defineField(.u),
    .paste_observation_ms = defineField(.u),
    .paste_settle_ms = defineField(.u),
    .pipewire_client_node_version_advertised = defineField(.u),
    .pipewire_client_node_version_selected = defineField(.u),
    .pipewire_device_id = defineField(.u),
    .pipewire_device_serial = defineField(.u),
    .pipewire_node_id = defineField(.u),
    .pipewire_node_serial = defineField(.u),
    .product = defineField(.u),
    .recording_seconds_max = defineField(.u),
    .request = defineField(.u),
    .retry_duration_seconds = defineField(.u),
    .serial = defineField(.u),
    .syscall_result = defineField(.u),
    .transcript_committed_size = defineField(.u),
    .transcript_size = defineField(.u),
    .transcript_size_max = defineField(.u),
    .transcription_chunks_count = defineField(.u),
    .transcription_tokens_count = defineField(.u),
    .transcription_tokens_count_max = defineField(.u),
    .uid = defineField(.u),
    .uid_expected = defineField(.u),
    .vendor = defineField(.u),
    .version = defineField(.u),
    .wayland_object = defineField(.u),
    .wayland_opcode = defineField(.u),
    .wayland_server_error_code = defineField(.u),
    .write_size = defineField(.u),
    .x11_bad_value = defineField(.u),
    .x11_major_opcode = defineField(.u),
    .x11_minor_opcode = defineField(.u),
    .x11_response_type = defineField(.u),
    .x11_sequence = defineField(.u),
    .x11_server_error_code = defineField(.u),
    .x11_setup_status = defineField(.u),

    // Signed operating-system values whose negative sentinel is meaningful.
    .callback_thread_id = defineField(.i),
    .capture_thread_id = defineField(.i),
    .cause_code = defineField(.i),
    .descriptor = defineField(.i),
    .scheduler_priority = defineField(.i),

    // Permission bits are rendered in base eight.
    .mode = defineField(.octal),

    // Measured durations, ratios, and continuous values with call-site precision.
    .audio_duration_seconds = defineField(.f),
    .callback_duration_ms_max = defineField(.f),
    .callback_gap_ms_max = defineField(.f),
    .capture_start_duration_ms = defineField(.f),
    .clipboard_acquire_duration_ms = defineField(.f),
    .clipboard_transfer_duration_ms = defineField(.f),
    .cross_key_values_duration_ms = defineField(.f),
    .decoder_duration_ms = defineField(.f),
    .encoder_duration_ms = defineField(.f),
    .features_duration_ms = defineField(.f),
    .inference_duration_ms = defineField(.f),
    .model_load_duration_ms = defineField(.f),
    .model_prepare_duration_ms = defineField(.f),
    .model_runtime_init_duration_ms = defineField(.f),
    .paste_duration_ms = defineField(.f),
    .paste_observation_elapsed_ms = defineField(.f),
    .paste_settle_duration_ms = defineField(.f),
    .recording_finalize_duration_ms = defineField(.f),
    .transcript_save_duration_ms = defineField(.f),
    .transcription_audio_duration_seconds = defineField(.f),
    .transcription_compute_duration_ms = defineField(.f),
    .transcription_compute_speed_ratio = defineField(.f),

    // Inference and capture values retain f32 rounding semantics.
    .activity_active_threshold_rms = defineField(.f32),
    .activity_noise_floor_rms = defineField(.f32),
    .activity_quiet_threshold_rms = defineField(.f32),
    .average_log_probability = defineField(.f32),
    .no_speech_probability = defineField(.f32),
    .no_speech_probability_active_min = defineField(.f32),
    .no_speech_probability_inactive_min = defineField(.f32),

    .activity_observed = defineField(.b),
    .mode_available = defineField(.b),
    .recording_size_exceeded = defineField(.b),
    .text_empty = defineField(.b),
    .uid_available = defineField(.b),
    .wayland_server_error_message_truncated = defineField(.b),
    .x11_setup_message_truncated = defineField(.b),

    // Linux errno values use their symbolic enum spelling.
    .system_error = defineField(.errno),
};

fn canonicalFieldKind(comptime name: []const u8) FieldKind {
    @setEvalBranchQuota(20_000);

    if (!@hasField(@TypeOf(field_catalog), name)) {
        @compileError("unknown logging field; add it to logging.field_catalog: " ++ name);
    }

    return @field(field_catalog, name).kind;
}

fn fieldValueType(comptime kind: FieldKind) type {
    return @FieldType(Field, @tagName(kind));
}

/// Defines a bounded event schema by selecting keys from the process-wide field
/// catalog. Misspelled keys, duplicate keys, fields outside the event schema,
/// and values with the wrong canonical representation fail at compile time.
/// Optional fields are added to a mutable set; emitted fields retain schema order.
pub fn FieldSet(comptime names: anytype) type {
    const Names = @TypeOf(names);

    comptime {
        // Large schemas validate every identifier byte and can exceed Zig's
        // default compile-time branch quota.
        @setEvalBranchQuota(20_000);

        const info = @typeInfo(Names);

        if (info != .@"struct" or !info.@"struct".is_tuple) {
            @compileError("logging field schema must be a tuple of field names");
        }

        if (names.len > @bitSizeOf(u64)) {
            @compileError("logging field schema exceeds the presence mask");
        }

        for (0..names.len) |left| {
            const name = @tagName(names[left]);

            requireIdentifier("logging field name", name);

            _ = canonicalFieldKind(name);

            for (left + 1..names.len) |right| {
                if (std.mem.eql(u8, name, @tagName(names[right]))) {
                    @compileError("duplicate logging field in event schema: " ++ name);
                }
            }
        }
    }

    return struct {
        const Self = @This();

        storage: [names.len]Entry = undefined,
        present_mask: u64 = 0,
        compacted_count: ?u7 = null,

        pub inline fn init(values: anytype) Self {
            var fields: Self = .{};

            fields.addAll(values);

            return fields;
        }

        pub inline fn addAll(fields: *Self, values: anytype) void {
            inline for (std.meta.fields(@TypeOf(values))) |value| {
                fields.addNamed(value.name, @field(values, value.name));
            }
        }

        pub inline fn add(fields: *Self, comptime name: @EnumLiteral(), value: fieldValueType(canonicalFieldKind(@tagName(name)))) void {
            fields.addNamed(@tagName(name), value);
        }

        inline fn addNamed(fields: *Self, comptime name: []const u8, value: fieldValueType(canonicalFieldKind(name))) void {
            putField(
                fields.storage[0..],
                &fields.present_mask,
                fields.compacted_count,
                fieldIndex(name),
                name,
                @unionInit(Field, @tagName(canonicalFieldKind(name)), value),
            );
        }

        fn fieldIndex(comptime name: []const u8) comptime_int {
            @setEvalBranchQuota(10_000);

            for (names, 0..) |allowed, index| {
                if (std.mem.eql(u8, name, @tagName(allowed))) {
                    return index;
                }
            }

            @compileError("logging field is not allowed by this event schema: " ++ name);
        }

        inline fn entries(fields: *Self) []const Entry {
            if (fields.compacted_count) |count| {
                return fields.storage[0..count];
            }

            const count = compactFields(&fields.storage, fields.present_mask);

            fields.compacted_count = count;

            return fields.storage[0..count];
        }
    };
}

// PERFORMANCE: Schema operations are forced inline so compile-time names and
// value variants disappear at call sites. Keep mask mutation and compaction
// non-generic and out-of-line so their safety checks and loops are shared by
// every schema rather than copied for every field and event.
noinline fn putField(storage: []Entry, present_mask: *u64, compacted_count: ?u7, index: usize, name: []const u8, value: Field) void {
    std.debug.assert(compacted_count == null);

    const mask = @as(u64, 1) << @intCast(index);
    std.debug.assert(present_mask.* & mask == 0);

    storage[index] = .{ name, value };
    present_mask.* |= mask;
}

noinline fn compactFields(storage: []Entry, present_mask: u64) u7 {
    var count: u7 = 0;

    for (0..storage.len) |index| {
        if (present_mask & (@as(u64, 1) << @intCast(index)) == 0) {
            continue;
        }

        storage[count] = storage[index];
        count += 1;
    }

    return count;
}

pub const NoFields = FieldSet(.{});

pub fn scoped(comptime component: @EnumLiteral()) type {
    comptime {
        if (@tagName(component).len > 64) {
            @compileError("logging component exceeds 64 bytes");
        }

        requireIdentifier("logging component", @tagName(component));
    }

    return struct {
        /// Binds one canonical event name to its complete field schema. Event
        /// names, keys, and `.name` values are nonempty lower-case ASCII
        /// identifiers (`[a-z_][a-z0-9_]*`). Integers are decimal unless marked
        /// `.octal`; every field uses its catalog-selected representation. Inputs are
        /// borrowed only during emission. Guard expensive value preparation at
        /// the call site with `enabled()` because arguments are evaluated first.
        pub fn Event(comptime event: @EnumLiteral(), comptime Fields: type) type {
            comptime {
                if (@tagName(event).len > 64) {
                    @compileError("logging event exceeds 64 bytes");
                }

                requireIdentifier("logging event", @tagName(event));
            }

            return struct {
                pub inline fn emit(severity: Level, context: Context, values: anytype) void {
                    if (!enabled(severity)) {
                        return;
                    }

                    var fields = Fields.init(values);

                    emitFields(severity, context, &fields);
                }

                pub inline fn emitFields(severity: Level, context: Context, fields: *Fields) void {
                    emitEvent(severity, @tagName(component), @tagName(event), context, fields.entries());
                }
            };
        }
    };
}

// PERFORMANCE: Keep this loop non-generic and out-of-line. Specializing on
// field shapes or inlining into callers duplicates formatting across events.
// The tagged entries and extra call trade stack space and cold-path dispatch
// for shared code; no allocation or capture/inference hot-path work is added.
noinline fn emitEvent(severity: Level, component: []const u8, event: []const u8, context: Context, fields: []const Entry) void {
    if (!enabled(severity)) {
        return;
    }

    std.debug.assert(event.len <= 64);
    assertIdentifier(event);

    var message_buffer: [message_bytes_max]u8 = undefined;
    var message = std.Io.Writer.fixed(&message_buffer);

    const truncated = writeMessage(&message, event, context, fields);

    sendRecord(severity, component, event, context, &message, truncated);
}

fn writeMessage(message: *std.Io.Writer, event: []const u8, context: Context, fields: []const Entry) bool {
    message.writeAll(event) catch {
        return true;
    };

    if (context.recording_id) |recording_id| {
        message.print(" recording_id={d}", .{recording_id}) catch {
            return true;
        };
    }

    for (fields) |entry| {
        const key, const field = entry;

        assertIdentifier(key);

        message.writeByte(' ') catch {
            return true;
        };

        message.writeAll(key) catch {
            return true;
        };

        message.writeByte('=') catch {
            return true;
        };

        switch (field) {
            .name => |value| {
                assertIdentifier(value);

                message.writeAll(value) catch {
                    return true;
                };
            },

            .str => |value| {
                message.writeByte('"') catch {
                    return true;
                };

                std.zig.stringEscape(value, message) catch {
                    return true;
                };

                message.writeByte('"') catch {
                    return true;
                };
            },

            .str_joined => |values| {
                message.writeByte('"') catch {
                    return true;
                };

                for (values) |value| std.zig.stringEscape(value, message) catch {
                    return true;
                };

                message.writeByte('"') catch {
                    return true;
                };
            },

            .verbatim => |value| {
                assertVerbatim(value);

                message.writeAll(value) catch {
                    return true;
                };
            },
            .u => |value| message.printInt(value, 10, .lower, .{}) catch {
                return true;
            },
            .i => |value| message.printInt(value, 10, .lower, .{}) catch {
                return true;
            },
            .octal => |value| message.printInt(value, 8, .lower, .{}) catch {
                return true;
            },
            .f => |value| decimal.fmt(value.value, value.digits).format(message) catch {
                return true;
            },
            .f32 => |value| decimal.fmt(value.value, value.digits).format(message) catch {
                return true;
            },
            .b => |value| message.writeAll(if (value) "true" else "false") catch {
                return true;
            },
            .errno => |value| fmtErrno(value).format(message) catch {
                return true;
            },
        }
    }

    return false;
}

test "event emits escaped fields in one journal datagram" {
    try std.testing.expect(sink == .disabled);

    var sockets: [2]i32 = undefined;
    try std.testing.expectEqual(.SUCCESS, linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.DGRAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0, &sockets)));
    defer _ = linux.close(sockets[1]);

    const previous_threshold = threshold;

    sink = .{ .journal = .{ .fd = sockets[0], .connected = true } };

    threshold = .debug;
    defer {
        deinit();

        threshold = previous_threshold;
    }

    const EscapedMessage = scoped(.logging).Event(.escaped_message, FieldSet(.{
        .outcome,
        .detail,
    }));

    EscapedMessage.emit(.warn, .{ .recording_id = 7 }, .{
        .detail = "line \"one\"\npriority=0\t\x01",
        .outcome = "accepted",
    });

    var record: [record_bytes_max]u8 = undefined;
    const received = linux.read(sockets[1], &record, record.len);

    try std.testing.expectEqual(.SUCCESS, linux.errno(received));

    const record_size: usize = received;

    const journal_prefix =
        "PRIORITY=4\n" ++
        "SYSLOG_IDENTIFIER=voiced\n" ++
        "VOICED_COMPONENT=logging\n" ++
        "VOICED_EVENT=escaped_message\n" ++
        "VOICED_RECORDING_ID=7\n" ++
        "MESSAGE\n";

    const expected_message = "escaped_message recording_id=7 outcome=accepted detail=\"line \\\"one\\\"\\npriority=0\\t\\x01\"";
    const message_size_offset = journal_prefix.len;
    const message_offset = message_size_offset + @sizeOf(u64);
    const message_end = message_offset + expected_message.len;

    try std.testing.expectEqual(message_end + 1, record_size);
    try std.testing.expectEqualStrings(journal_prefix, record[0..journal_prefix.len]);
    try std.testing.expectEqual(@as(u64, expected_message.len), std.mem.readInt(u64, record[message_size_offset..message_offset], .little));
    try std.testing.expectEqualStrings(expected_message, record[message_offset..message_end]);
    try std.testing.expectEqual(@as(u8, '\n'), record[message_end]);
}

fn requireIdentifier(comptime label: []const u8, comptime value: []const u8) void {
    if (value.len == 0) {
        @compileError(label ++ " must not be empty");
    }

    for (value, 0..) |byte, position| {
        if (!std.ascii.isLower(byte) and byte != '_' and
            !(position > 0 and std.ascii.isDigit(byte)))
        {
            @compileError(label ++ " must use lower-case snake_case");
        }
    }
}

fn assertIdentifier(value: []const u8) void {
    std.debug.assert(value.len > 0);

    for (value, 0..) |byte, position| {
        std.debug.assert(std.ascii.isLower(byte) or byte == '_' or
            (position > 0 and std.ascii.isDigit(byte)));
    }
}

fn assertVerbatim(value: []const u8) void {
    std.debug.assert(value.len > 0);

    for (value) |byte| {
        std.debug.assert(byte > ' ' and byte < 0x7f and byte != '"' and byte != '=');
    }
}

// Keep delivery non-generic and out-of-line so journal framing and syscalls are
// not copied into every message-format specialization. The extra call is on the
// logging cold path, not capture/inference hot paths. Measured with Zig 0.16 on
// x86-64 ReleaseSafe: ~151 KiB saved, a ~10% smaller whole stripped daemon.
noinline fn sendRecord(severity: Level, component: []const u8, event: []const u8, context: Context, message: *std.Io.Writer, truncated: bool) void {
    // PERFORMANCE: Finalize here, not in generic emit: otherwise every message
    // format can get another copy of trimming and truncation code. Measured
    // 2026-09-07 with stock Zig 0.16.0/LLVM, host x86-64, ReleaseSafe application
    // and inference, static PIE, crash diagnostics disabled, GNU strip --strip-all:
    // this change alone reduced 1,161,416 to 1,147,032 bytes (14,384 saved).
    // Together with decimal.zig, the applied worktree went from 1,323,960 to
    // 1,301,832 bytes (22,128 saved). The isolated baseline also had initializer
    // optimizations; do not add savings across baselines or compiler versions.
    // The combined prototype passed control, logging and real-model output
    // integration, including truncation, metadata injection and receiver drops.
    // Only emitEvent's fixed writer enters. Append the marker before trimming so a
    // truncated message retains embedded/trailing newlines before the marker.
    if (truncated) {
        const marker = " [truncated]";
        const end = @min(message.end, message.buffer.len - marker.len);

        @memcpy(message.buffer[end..][0..marker.len], marker);

        message.end = end + marker.len;
    }

    const text = std.mem.trimEnd(u8, message.buffered(), "\n");

    if (sink == .stderr) {
        var line_buffer: [record_bytes_max]u8 = undefined;
        const line = std.fmt.bufPrint(&line_buffer, "{s}: {s}\n", .{ levelName(severity), text }) catch {
            unreachable;
        };

        var remaining = line;

        while (remaining.len != 0) {
            const result = linux.write(2, remaining.ptr, remaining.len);

            switch (linux.errno(result)) {
                .SUCCESS => {
                    if (result == 0) {
                        return;
                    }

                    remaining = remaining[result..];
                },
                .INTR => continue,
                else => return,
            }
        }

        return;
    }

    const lost = dropped.swap(0, .monotonic);

    var buffer: [record_bytes_max]u8 = undefined;
    var record = std.Io.Writer.fixed(&buffer);

    record.print("PRIORITY={d}\nSYSLOG_IDENTIFIER=voiced\nVOICED_COMPONENT={s}\nVOICED_EVENT={s}\n", .{ @intFromEnum(severity), component, event }) catch {
        unreachable;
    };

    if (context.recording_id) |value| record.print("VOICED_RECORDING_ID={d}\n", .{value}) catch {
        unreachable;
    };

    if (lost != 0) record.print("VOICED_DROPPED={d}\n", .{lost}) catch unreachable;

    // Always use the native binary MESSAGE encoding. Newlines in native error
    // details remain one message and cannot inject journal metadata fields.
    record.writeAll("MESSAGE\n") catch {
        unreachable;
    };

    record.writeInt(u64, text.len, .little) catch {
        unreachable;
    };

    record.writeAll(text) catch {
        unreachable;
    };

    record.writeByte('\n') catch {
        unreachable;
    };

    const sent = switch (sink) {
        .journal => |journal| blk: {
            var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
            const path = "/run/systemd/journal/socket";

            @memcpy(address.path[0..path.len], path);

            const bytes = record.buffered();

            const result = linux.sendto(journal.fd, bytes.ptr, bytes.len, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL, if (journal.connected)
                null
            else
                @ptrCast(&address), if (journal.connected)
                0
            else
                @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1));

            break :blk linux.errno(result) == .SUCCESS and result == bytes.len;
        },
        .disabled => false,
        .stderr => unreachable,
    };

    if (!sent) {
        countDropped(lost +| 1);
    }
}

fn countDropped(count: u64) void {
    var previous = dropped.load(.monotonic);

    while (dropped.cmpxchgWeak(previous, previous +| count, .monotonic, .monotonic)) |actual| {
        previous = actual;
    }
}
