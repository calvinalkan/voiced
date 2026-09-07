//! Startup configuration. Both input syntaxes feed the same typed option setter;
//! the startup arena retains file-backed microphone names for the service lifetime.
const std = @import("std");
const logging = @import("logging.zig");
const supervisor = @import("supervisor.zig");
const models = @import("models");
const capture_module = @import("capture.zig");
const paste_keyboard = @import("paste_keyboard.zig");

pub fn load(init: std.process.Init, arguments: []const [:0]const u8, diagnostic: *Diagnostic) !supervisor.ServiceOptions {
    const allocator = init.arena.allocator();
    var explicit_path: ?[]const u8 = null;
    var cli: Arguments = .{ .values = arguments };
    while (try cli.next(diagnostic)) |entry| {
        if (entry.option == .config) {
            if (explicit_path != null) return error.DuplicateOption;
            diagnostic.expected = "a nonempty configuration path";
            if (entry.value.len == 0) return error.InvalidValue;
            explicit_path = entry.value;
        }
    }

    const path = explicit_path orelse path: {
        if (init.environ_map.get("XDG_CONFIG_HOME")) |directory| {
            if (std.fs.path.isAbsolute(directory))
                break :path try std.fs.path.join(allocator, &.{ directory, "voiced/config" });
        }
        const home = init.environ_map.get("HOME") orelse break :path null;
        if (!std.fs.path.isAbsolute(home)) break :path null;
        break :path try std.fs.path.join(allocator, &.{ home, ".config/voiced/config" });
    };

    var options: supervisor.ServiceOptions = .{};
    if (path) |file_path| {
        diagnostic.path = file_path;
        const text = std.Io.Dir.cwd().readFileAlloc(init.io, file_path, allocator, .limited(64 * 1024)) catch |err| text: {
            if (err == error.FileNotFound and explicit_path == null) break :text "";
            return err;
        };
        var lines = std.mem.splitScalar(u8, text, '\n');
        var seen: Seen = .{};
        while (lines.next()) |raw_line| {
            diagnostic.line += 1;
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or line[0] == '#') continue;
            diagnostic.argument = line;
            const equals = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidConfigLine;
            const name = std.mem.trim(u8, line[0..equals], " \t");
            const value = std.mem.trim(u8, line[equals + 1 ..], " \t");
            diagnostic.argument = name;
            diagnostic.value = value;
            const option = std.meta.stringToEnum(Option, name) orelse return error.UnknownOption;
            if (option == .config) return error.UnknownOption;
            try set(allocator, &options, &seen, option, value, diagnostic);
        }
    }

    // A new input layer may replace a file setting, including microphone choice.
    // Repetition within either layer is an error, even if the values agree.
    diagnostic.path = "";
    diagnostic.line = 0;
    cli.index = 0;
    var seen: Seen = .{};
    while (try cli.next(diagnostic)) |entry| {
        if (entry.option != .config)
            try set(allocator, &options, &seen, entry.option, entry.value, diagnostic);
    }

    const model = options.capture.transcription;
    if (model.decoder_threads_count) |count| {
        if (count > model.inference_threads_count) {
            diagnostic.argument = "model_decoder_threads";
            diagnostic.value = try std.fmt.allocPrint(allocator, "{d}", .{count});
            diagnostic.expected = "a count no greater than model_encoder_threads";
            return error.InvalidValue;
        }
    }
    return options;
}

pub const Diagnostic = struct {
    command: []const u8 = "",
    argument: []const u8 = "",
    value: []const u8 = "",
    expected: []const u8 = "",
    other_option: []const u8 = "",
    path: []const u8 = "",
    line: usize = 0,
};

const Option = enum {
    config,
    log_level,
    model,
    model_encoder_threads,
    model_decoder_threads,
    model_encoder_padding_seconds,
    model_idle_seconds_max,
    recording_seconds_max,
    microphone_node,
    microphone_serial,
    transcript_output,
    notification_mode,
    paste_shortcut,
    paste_settle_ms,
    paste_key_gap_ms,
};

const Seen = std.enums.EnumSet(Option);

fn set(allocator: std.mem.Allocator, options: *supervisor.ServiceOptions, seen: *Seen, option: Option, value: []const u8, diagnostic: *Diagnostic) !void {
    if (seen.contains(option)) return error.DuplicateOption;
    seen.insert(option);
    switch (option) {
        .config => unreachable,
        .log_level => {
            diagnostic.expected = "critical, error, warn, info, or debug";
            options.log_level = logging.parseLevel(value) orelse return error.InvalidValue;
        },
        .model => {
            diagnostic.expected = "'Systran/faster-whisper-base.en' or 'Systran/faster-whisper-small.en'";
            options.capture.transcription.model = models.Model.parse(value) orelse return error.InvalidValue;
        },
        .model_encoder_threads, .model_decoder_threads => {
            diagnostic.expected = "an integer from 1 to 32 CPU workers";
            const count = std.fmt.parseInt(u32, value, 10) catch return error.InvalidValue;
            if (count == 0 or count > 32) return error.InvalidValue;
            if (option == .model_encoder_threads)
                options.capture.transcription.inference_threads_count = count
            else
                options.capture.transcription.decoder_threads_count = count;
        },
        .model_encoder_padding_seconds => {
            diagnostic.expected = "5, 10, or 30 seconds";
            const seconds = std.fmt.parseInt(u8, value, 10) catch return error.InvalidValue;
            options.capture.transcription.encoder_trailing_padding = switch (seconds) {
                5 => .seconds_5,
                10 => .seconds_10,
                30 => .seconds_30,
                else => return error.InvalidValue,
            };
        },
        .model_idle_seconds_max => {
            diagnostic.expected = "an integer from 0 to 4294967295 seconds (0 disables idle retention)";
            options.model_keep_warm_seconds = std.fmt.parseInt(u32, value, 10) catch return error.InvalidValue;
        },
        .recording_seconds_max => {
            diagnostic.expected = "an integer from 1 to 65535 seconds";
            const seconds = std.fmt.parseInt(u16, value, 10) catch return error.InvalidValue;
            if (seconds == 0) return error.InvalidValue;
            options.capture.recording_seconds = seconds;
        },
        .microphone_node, .microphone_serial => {
            const other: Option = if (option == .microphone_node) .microphone_serial else .microphone_node;
            if (seen.contains(other)) {
                diagnostic.other_option = @tagName(other);
                return error.ConflictingOptions;
            }
            diagnostic.expected = "a nonempty source name or serial of at most 255 bytes, without NUL";
            if (value.len == 0 or value.len >= capture_module.target_name_bytes_capacity or std.mem.indexOfScalar(u8, value, 0) != null)
                return error.InvalidValue;
            const terminated = try allocator.dupeZ(u8, value);
            options.capture.source = if (option == .microphone_node) .{ .node_name = terminated } else .{ .device_serial = terminated };
        },
        .transcript_output => {
            diagnostic.expected = "'desktop', 'clipboard', or 'stdout'";
            options.output = std.meta.stringToEnum(@FieldType(supervisor.ServiceOptions, "output"), value) orelse return error.InvalidValue;
        },
        .notification_mode => {
            diagnostic.expected = "'errors' or 'off'";
            options.notification_mode = std.meta.stringToEnum(@FieldType(supervisor.ServiceOptions, "notification_mode"), value) orelse return error.InvalidValue;
        },
        .paste_settle_ms, .paste_key_gap_ms => {
            diagnostic.expected = "an integer from 0 to 65535 milliseconds";
            const milliseconds = std.fmt.parseInt(u16, value, 10) catch return error.InvalidValue;
            if (option == .paste_settle_ms)
                options.paste_settle_ms = milliseconds
            else
                options.paste_key_gap_ms = milliseconds;
        },
        .paste_shortcut => {
            diagnostic.expected = "'ctrl+shift+v', 'ctrl+v', or 'shift+insert'";
            options.paste_key = std.meta.stringToEnum(paste_keyboard.Chord, value) orelse return error.InvalidValue;
        },
    }
}

const Arguments = struct {
    values: []const [:0]const u8,
    index: usize = 0,

    fn next(arguments: *Arguments, diagnostic: *Diagnostic) !?struct { option: Option, value: []const u8 } {
        if (arguments.index == arguments.values.len) return null;
        const argument = arguments.values[arguments.index];
        arguments.index += 1;
        diagnostic.argument = argument;
        if (std.mem.eql(u8, argument, "--")) {
            if (arguments.index != arguments.values.len) {
                diagnostic.argument = arguments.values[arguments.index];
                return error.UnexpectedArgument;
            }
            return null;
        }
        if (!std.mem.startsWith(u8, argument, "-")) return error.UnexpectedArgument;
        const equals = std.mem.indexOfScalar(u8, argument, '=');
        const name = argument[0 .. equals orelse argument.len];
        diagnostic.argument = name;
        const option = cliOption(name) orelse return error.UnknownOption;
        const value = if (equals) |position| argument[position + 1 ..] else value: {
            if (arguments.index == arguments.values.len or std.mem.startsWith(u8, arguments.values[arguments.index], "--"))
                return error.MissingValue;
            const value = arguments.values[arguments.index];
            arguments.index += 1;
            break :value value;
        };
        diagnostic.value = value;
        return .{ .option = option, .value = value };
    }
};

fn cliOption(name: []const u8) ?Option {
    if (!std.mem.startsWith(u8, name, "--")) return null;
    var normalized: [64]u8 = undefined;
    const key = name[2..];
    if (key.len > normalized.len) return null;
    for (key, 0..) |byte, index| {
        if (byte == '_') return null;
        normalized[index] = if (byte == '-') '_' else byte;
    }
    return std.meta.stringToEnum(Option, normalized[0..key.len]);
}
