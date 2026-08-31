const std = @import("std");
const audio_process = @import("audio_process.zig");
const models = @import("models");
const supervisor = @import("supervisor.zig");
const transcription_process = @import("transcription_process.zig");
const assert = std.debug.assert;

pub fn main(init: std.process.Init) !void {
    var arguments = try std.process.Args.Iterator.initAllocator(
        init.minimal.args,
        init.gpa,
    );
    defer arguments.deinit();
    assert(arguments.skip());

    const command = arguments.next() orelse {
        printUsage();
        return error.InvalidArguments;
    };

    // Internal roles are not user commands. The supervisor passes only the
    // inherited control socket and its PID in argv; each role receives its
    // role-specific shared descriptors in the first seqpacket launch message.
    if (std.mem.eql(u8, command, "--internal-role")) {
        const role = arguments.next() orelse return error.InvalidInternalRole;
        const socket_text = arguments.next() orelse return error.InvalidInternalRole;
        const supervisor_pid_text = arguments.next() orelse
            return error.InvalidInternalRole;
        if (arguments.next() != null) return error.InvalidInternalRole;

        const socket = try std.fmt.parseInt(std.posix.fd_t, socket_text, 10);
        const supervisor_pid = try std.fmt.parseInt(
            std.os.linux.pid_t,
            supervisor_pid_text,
            10,
        );
        if (std.mem.eql(u8, role, "audio-fake")) {
            try audio_process.FakeWorker.run(socket, supervisor_pid);
            return;
        }
        if (std.mem.eql(u8, role, "audio-pipewire")) {
            try audio_process.PipeWireWorker.run(socket, supervisor_pid);
            return;
        }
        if (std.mem.eql(u8, role, "transcription-fake")) {
            try transcription_process.runFakeWorker(socket, supervisor_pid);
            return;
        }
        if (std.mem.eql(u8, role, "transcription-model")) {
            try transcription_process.runModelWorker(
                init,
                socket,
                supervisor_pid,
            );
            return;
        }
        return error.InvalidInternalRole;
    }

    if (!std.mem.eql(u8, command, "supervisor-spike")) {
        printUsage();
        return error.InvalidArguments;
    }

    const scenario_text = arguments.next() orelse "normal";
    if (std.mem.eql(u8, scenario_text, "pipewire")) {
        const options = parsePipeWireOptions(&arguments) catch |err| {
            printUsage();
            return err;
        };
        _ = try supervisor.runPipeWireSession(init, options);
        return;
    }

    if (arguments.next() != null) {
        printUsage();
        return error.InvalidArguments;
    }
    const scenario = std.meta.stringToEnum(
        supervisor.FakeScenario,
        scenario_text,
    ) orelse {
        printUsage();
        return error.InvalidArguments;
    };
    _ = try supervisor.runFakeSession(init, scenario);
}

fn parsePipeWireOptions(
    arguments: *std.process.Args.Iterator,
) !supervisor.PipeWireOptions {
    var recording_seconds: ?u16 = null;
    var slot_seconds: ?u8 = null;
    var source: ?audio_process.Source = null;
    var automatic_stop: audio_process.AutomaticStop = .disabled;
    var process_realtime: ?bool = null;
    var selected_model: ?models.Model = null;
    var inference_threads_count: ?u32 = null;
    var model_residency: ?supervisor.ModelResidency = null;

    while (arguments.next()) |option| {
        if (std.mem.eql(u8, option, "--seconds")) {
            if (recording_seconds != null) return error.InvalidArguments;
            const seconds_text = arguments.next() orelse return error.InvalidArguments;
            const seconds = std.fmt.parseInt(u32, seconds_text, 10) catch
                return error.InvalidArguments;
            if (seconds == 0 or
                seconds > supervisor.recording_duration_seconds_limit)
            {
                return error.InvalidArguments;
            }
            recording_seconds = @intCast(seconds);
        } else if (std.mem.eql(u8, option, "--slot-seconds")) {
            if (slot_seconds != null) return error.InvalidArguments;
            const seconds_text = arguments.next() orelse return error.InvalidArguments;
            const seconds = std.fmt.parseInt(u8, seconds_text, 10) catch
                return error.InvalidArguments;
            if (seconds == 0 or
                seconds > audio_process.slot_duration_seconds_max)
            {
                return error.InvalidArguments;
            }
            slot_seconds = seconds;
        } else if (std.mem.eql(u8, option, "--target")) {
            if (source != null) return error.InvalidArguments;
            const node_name = arguments.next() orelse return error.InvalidArguments;
            if (node_name.len == 0 or
                node_name.len >= audio_process.target_name_bytes_capacity)
            {
                return error.InvalidArguments;
            }
            source = .{ .node_name = node_name };
        } else if (std.mem.eql(u8, option, "--device-serial")) {
            if (source != null) return error.InvalidArguments;
            const device_serial = arguments.next() orelse return error.InvalidArguments;
            if (device_serial.len == 0 or
                device_serial.len >= audio_process.target_name_bytes_capacity)
            {
                return error.InvalidArguments;
            }
            source = .{ .device_serial = device_serial };
        } else if (std.mem.eql(u8, option, "--listen")) {
            if (automatic_stop != .disabled) return error.InvalidArguments;
            automatic_stop = .after_quiet;
        } else if (std.mem.eql(u8, option, "--main-loop")) {
            if (process_realtime != null) return error.InvalidArguments;
            process_realtime = false;
        } else if (std.mem.eql(u8, option, "--model")) {
            if (selected_model != null) return error.InvalidArguments;
            const name = arguments.next() orelse return error.InvalidArguments;
            selected_model = models.Model.parse(name) orelse
                return error.InvalidArguments;
        } else if (std.mem.eql(u8, option, "--threads")) {
            if (inference_threads_count != null) return error.InvalidArguments;
            const count_text = arguments.next() orelse return error.InvalidArguments;
            const count = std.fmt.parseInt(u32, count_text, 10) catch
                return error.InvalidArguments;
            if (count == 0 or
                count > transcription_process.inference_threads_count_max)
            {
                return error.InvalidArguments;
            }
            inference_threads_count = count;
        } else if (std.mem.eql(u8, option, "--model-residency")) {
            if (model_residency != null) return error.InvalidArguments;
            const residency_text = arguments.next() orelse
                return error.InvalidArguments;
            model_residency = std.meta.stringToEnum(
                supervisor.ModelResidency,
                residency_text,
            ) orelse return error.InvalidArguments;
        } else {
            return error.InvalidArguments;
        }
    }

    const defaults: supervisor.PipeWireOptions = .{};
    if ((inference_threads_count != null or model_residency != null) and
        selected_model == null)
    {
        return error.InvalidArguments;
    }
    const transcription: supervisor.TranscriptionBackend = if (selected_model) |model|
        .{ .model = .{
            .model = model,
            .inference_threads_count = inference_threads_count orelse 4,
            .residency = model_residency orelse .session,
        } }
    else
        .fake;
    const options: supervisor.PipeWireOptions = .{
        .source = source orelse defaults.source,
        .recording_seconds = recording_seconds orelse defaults.recording_seconds,
        .slot_seconds = slot_seconds orelse defaults.slot_seconds,
        .automatic_stop = automatic_stop,
        .process_realtime = process_realtime orelse defaults.process_realtime,
        .transcription = transcription,
    };
    assert(options.recording_seconds > 0);
    assert(options.recording_seconds <= supervisor.recording_duration_seconds_limit);
    assert(options.slot_seconds > 0);
    assert(options.slot_seconds <= audio_process.slot_duration_seconds_max);
    switch (options.source) {
        .default => {},
        .node_name, .device_serial => |configured_source| {
            assert(configured_source.len > 0);
            assert(configured_source.len < audio_process.target_name_bytes_capacity);
        },
    }
    switch (options.transcription) {
        .fake => {},
        .model => |model| {
            assert(model.inference_threads_count > 0);
            assert(model.inference_threads_count <=
                transcription_process.inference_threads_count_max);
        },
    }
    return options;
}

fn printUsage() void {
    std.debug.print(
        "usage:\n" ++
            "  voiced supervisor-spike " ++
            "<normal|burst_publications|slow_transcription|" ++
            "transcription_crash_before_result|" ++
            "transcription_crash_after_result|" ++
            "repeated_transcription_crash|transcription_hang>\n" ++
            "  voiced supervisor-spike pipewire " ++
            "[--seconds <1-65535>] [--slot-seconds <1-30>] [--listen] " ++
            "[--target <node> | --device-serial <serial>] [--main-loop] " ++
            "[--model <Systran/faster-whisper-base.en|" ++
            "Systran/faster-whisper-small.en> [--threads <1-32>] " ++
            "[--model-residency <session|service>]]\n",
        .{},
    );
}
