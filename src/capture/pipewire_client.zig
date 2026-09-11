//! Native DSP capture client. The control connection owns graph discovery,
//! source identity, mappings and link lifetime. process() consumes one graph
//! cycle without allocating, formatting diagnostics or waiting on descriptors.
const PipeWireClient = @This();

const std = @import("std");
const linux = std.os.linux;
const protocol = @import("pipewire_wire.zig");
const endian = @import("builtin").cpu.arch.endian();
const invalid = std.math.maxInt(u32);
const channels_max = 8;
const buffers_max = 16;
pub const samples_max = 16384;
pub const Error = protocol.Error || error{ CatalogFull, CatalogTextFull, SourceNotFound, SourceAmbiguous, SourceDisconnected, SourceChanged, SourcePortsUnavailable, UnsupportedVersion, UnsupportedFormat, UnsupportedIO, InvalidBuffer, CorruptedBuffer, TimelineDiscontinuity, GraphError, WakeFailed, TooManyChannels, TooManyBuffers, TooManyMemories, TooManyPeers, UnsupportedDefaultSourceMetadata, DefaultSourceNameInvalidUtf8, DefaultSourceNameTooLong };
pub const Source = union(enum) { default, node_name: []const u8, device_serial: []const u8 };
pub const Text = struct {
    bytes: [256]u8 = @splat(0),
    size: u16 = 0,
    pub fn get(self: *const Text) []const u8 {
        return self.bytes[0..self.size];
    }
    fn set(self: *Text, value: []const u8) Error!void {
        if (value.len > self.bytes.len) {
            return error.MessageTooLarge;
        }

        @memcpy(self.bytes[0..value.len], value);

        self.size = @intCast(value.len);
    }
};
const ObjectKind = enum { node, device, port };
pub const Object = struct {
    kind: ObjectKind,
    id: u32,
    serial: u64 = 0,
    proxy: u32 = invalid,
    parent: u32 = invalid,
    port: u32 = invalid,
    is_source: bool = false,
    is_output: bool = false,
    priority: i32 = 0,
    name: Text = .{},
    description: Text = .{},
    device_serial: Text = .{},
    channel: Text = .{},
};
pub const Identity = struct { node: Object, device: ?Object };
// Borrowed only between process() and finishCycle(). No sample allocation or
// copy is needed to turn a planar/wrapped graph buffer into resampler input.
pub const Block = struct {
    channels: [channels_max]union(enum) {
        silence,
        samples: struct { first: []align(1) const f32, second: []align(1) const f32 },
    },
    channels_count: usize,
    samples_count: usize,
    rate: u32,
    position: u64,
    header_present: bool,
    silence: bool,
    client: *PipeWireClient,

    pub fn sample(self: *const Block, index: usize) Error!f32 {
        var sum: f32 = 0;

        for (self.channels[0..self.channels_count], 0..) |channel, channel_index| {
            const value = switch (channel) {
                .silence => 0,
                .samples => |spans| if (index < spans.first.len)
                    spans.first[index]
                else
                    spans.second[index - spans.first.len],
            };

            if (!std.math.isFinite(value)) {
                // Buffers remain borrowed until finishCycle. Recover details
                // from the failing channel instead of retaining eight copies
                // or accidentally reporting the last channel validated.
                const port = self.client.ports[channel_index];
                const id = load(u32, port.io.?, 4);
                const buffer = port.buffers[id];
                const chunk = buffer.chunk.?;
                const diagnostic = &self.client.diagnostic;

                diagnostic.channel = channel_index;
                diagnostic.buffer_id = id;
                diagnostic.chunk_offset = load(u32, chunk, 0);
                diagnostic.chunk_size = load(u32, chunk, 4);
                diagnostic.chunk_stride = load(i32, chunk, 8);
                diagnostic.chunk_flags = load(u32, chunk, 12);
                diagnostic.header_flags = if (buffer.header) |h|
                    load(u32, h, 0)
                else
                    0;
                diagnostic.header_sequence = if (buffer.header) |h|
                    load(u64, h, 24)
                else
                    0;
                diagnostic.header_pts_ns = if (buffer.header) |h|
                    load(i64, h, 8)
                else
                    0;
                diagnostic.invalid_sample_index = index;

                return error.InvalidBuffer;
            }

            sum = if (channel_index == 0)
                value
            else
                sum + value;
        }

        return if (self.channels_count == 1)
            sum
        else
            sum / @as(f32, @floatFromInt(self.channels_count));
    }
};

connection: protocol.Connection = .{},
source: Source,
catalog: [128]?StoredObject = @splat(null),
text: TextStorage = .{},
selected: ?Identity = null,
node_id: u32 = invalid,
// Proxy IDs occupy the server's ordered object map. Allocate densely;
// reserving an arbitrary ID for the stream breaks later binds. Core,
// PipeWireClient and Registry already occupy 0, 1 and 2.
proxy_next: u32 = 3,
node_proxy: u32 = invalid,
metadata_proxy: u32 = invalid,
default_source: Text = .{},
server_version: Text = .{},
client_node_version: u32 = 0,
stream_created: bool = false,
links_created: bool = false,
running: bool = false,
format_serial: u1 = 0,
ports: [channels_max]Port = @splat(.{}),
ports_count: usize = 0,
memories: [128]?Memory = @splat(null),
peers: [32]?Peer = @splat(null),
activation: ?[]u8 = null,
position: ?[]u8 = null,
wake_fd: linux.fd_t = -1,
completion_fd: linux.fd_t = -1,
previous: ?struct { position: u64, duration: u64, rate: u32, clock: u32 } = null,
diagnostic: struct {
    wake_count: u64 = 0,
    graph_rate_num: u32 = 0,
    graph_rate_hz: u32 = 0,
    graph_position: u64 = 0,
    graph_duration: u64 = 0,
    channel: usize = 0,
    buffer_id: u32 = 0,
    chunk_offset: u32 = 0,
    chunk_size: u32 = 0,
    chunk_stride: i32 = 0,
    chunk_flags: u32 = 0,
    header_flags: u32 = 0,
    header_sequence: u64 = 0,
    header_pts_ns: i64 = 0,
    invalid_sample_index: ?usize = null,
} = .{},
error_code: i64 = 0,
error_object: u32 = 0,
error_sequence: u32 = 0,
error_message: [4096]u8 = @splat(0),
error_message_size: usize = 0,

pub fn init(self: *PipeWireClient, path: []const u8) Error!void {
    try self.connection.connect(path);
    try self.ints(0, 1, &.{3});

    var w: protocol.Writer = .{};

    _ = try w.begin(14);

    try w.dictionary(&.{ .{ "application.name", "Voiced" }, .{ "media.category", "Capture" } });
    try w.finish(0);
    try self.connection.send(1, 2, w.data());
    try self.ints(0, 5, &.{ 3, 2 });
    try self.ints(0, 2, &.{ 0, 100 });
}

pub fn deinit(self: *PipeWireClient) void {
    self.connection.deinit();

    if (self.wake_fd >= 0) {
        _ = linux.close(self.wake_fd);
    }

    if (self.completion_fd >= 0) {
        _ = linux.close(self.completion_fd);
    }

    for (&self.peers) |*entry| if (entry.*) |peer| {
        _ = linux.close(peer.fd);
        entry.* = null;
    };

    for (&self.memories) |*entry| if (entry.*) |memory| {
        std.posix.munmap(memory.bytes);

        entry.* = null;
    };
}

/// Consume at most one complete control message. The caller alternates
/// control work with graph events and its own stop/deadline handling.
pub fn dispatch(self: *PipeWireClient) Error!bool {
    const message = try self.connection.receive() orelse {
        return false;
    };
    defer self.connection.consume();
    defer self.releaseUnusedMemory();

    var r = message.body;

    if (message.object == self.node_proxy) {
        try self.nodeEvent(message.opcode, &r);

        return true;
    }

    switch (message.object) {
        0 => switch (message.opcode) {
            0 => {
                _ = try r.int();
                _ = try r.int();
                _ = try r.string();
                _ = try r.string();

                try self.server_version.set(try r.string());
            },

            1 => {
                const id = try r.int();
                const seq = try r.int();

                if (id == 0 and seq == 100) {
                    try self.ints(0, 2, &.{ 0, 101 });
                }

                if (id == 0 and seq == 101 and !self.stream_created) {
                    try self.createStream();
                }
            },
            2 => try self.ints(0, 3, &.{ try r.int(), try r.int() }),
            3 => {
                self.error_object = try r.int();
                self.error_sequence = try r.int();
                self.error_code = @as(i32, @bitCast(try r.int()));

                const text = try r.string();

                self.error_message_size = @min(text.len, self.error_message.len);

                @memcpy(self.error_message[0..self.error_message_size], text[0..self.error_message_size]);

                return error.GraphError;
            },

            5, 8 => {
                const proxy = try r.int();
                const global_id = try r.int();

                if (proxy == self.node_proxy) {
                    self.node_id = global_id;
                }

                for (self.ports[0..self.ports_count]) |*port| if (port.link_proxy == proxy) {
                    port.link_global = global_id;
                };
            },

            6 => {
                const id = try r.int();
                const kind = try r.id();
                const fd = try self.connection.descriptor(try r.long(18));
                const flags = try r.int();

                if (id == invalid or self.registeredMemory(id) != null) {
                    return error.InvalidMemory;
                }

                const entry = for (&self.memories) |*entry| {
                    if (entry.* == null) {
                        break entry;
                    }
                } else return error.TooManyMemories;

                if (kind != 2 or flags & 1 == 0) {
                    return error.InvalidMemory;
                }

                var stat: linux.Statx = undefined;
                self.connection.errno = linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &stat));

                if (self.connection.errno != .SUCCESS or !stat.mask.SIZE or stat.size == 0 or stat.size > 64 * 1024 * 1024) {
                    return error.InvalidMemory;
                }

                const address = linux.mmap(null, @intCast(stat.size), .{ .READ = true, .WRITE = flags & 2 != 0 }, .{ .TYPE = .SHARED }, fd, 0);

                self.connection.errno = linux.errno(address);

                if (self.connection.errno != .SUCCESS) {
                    return error.MappingFailed;
                }

                const mapping: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(address);

                entry.* = .{ .id = id, .bytes = mapping[0..@intCast(stat.size)], .writable = flags & 2 != 0 };
            },

            7 => {
                const id = try r.int();
                const memory = self.registeredMemory(id) orelse return error.InvalidMemory;

                // RemoveMem withdraws an ID, not existing borrowed views.
                // PipeWire 0.3.48 sends it before all buffer/IO withdrawals.
                // Keep that mapping until its last view disappears; the ID
                // may meanwhile be reused for a different mapping.
                memory.id = null;
            },

            4 => {},
            else => return error.InvalidMessage,
        },
        2 => switch (message.opcode) {
            0 => try self.global(&r),
            1 => {
                const id = try r.int();

                if (self.selected) |selected| {
                    if (id == selected.node.id or (selected.device != null and id == selected.device.?.id)) {
                        return error.SourceDisconnected;
                    }
                }

                for (self.ports[0..self.ports_count]) |port| if (id == port.source_global or id == port.link_global) return error.SourceDisconnected;

                for (&self.catalog) |*entry| if (entry.* != null and entry.*.?.id == id) {
                    self.text.releaseObject(entry.*.?);

                    entry.* = null;

                    break;
                };
            },
            else => return error.InvalidMessage,
        },
        else => {
            if (message.object == self.metadata_proxy and message.opcode == 0) {
                _ = try r.int();

                const key = try r.next();

                _ = try r.next();

                const value = try r.next();

                if (key.kind == 8 and std.mem.eql(u8, std.mem.trimEnd(u8, key.body, "\x00"), "default.audio.source") and value.kind == 8 and value.body.len > 0 and self.source == .default) {
                    self.default_source = try parseDefaultSource(value.body[0 .. value.body.len - 1]);
                }

                return true;
            }

            for (&self.catalog) |*entry| {
                if (entry.* == null or entry.*.?.proxy != message.object) {
                    continue;
                }

                if (message.opcode != 0) {
                    return error.InvalidMessage;
                }

                var expanded = entry.*.?.expand(&self.text);
                const object = &expanded;

                if (try r.int() != object.id) {
                    return error.InvalidMessage;
                }

                if (object.kind == .node) {
                    _ = try r.int();
                    _ = try r.int();
                }

                _ = try r.long(5);

                if (object.kind == .node) {
                    _ = try r.int();
                    _ = try r.int();
                    _ = try r.id();
                    _ = try r.next();
                }

                try properties(object, try r.structure());

                entry.* = try self.text.replace(entry.*, expanded);

                if (self.selected) |selected| {
                    if (object.id == selected.node.id and (object.serial != selected.node.serial or object.parent != selected.node.parent or !std.mem.eql(u8, object.name.get(), selected.node.name.get()))) {
                        return error.SourceChanged;
                    }

                    if (selected.device) |device| if (object.id == device.id and (object.serial != device.serial or !std.mem.eql(u8, object.device_serial.get(), device.device_serial.get()))) return error.SourceChanged;
                }

                break;
            }

            for (self.ports[0..self.ports_count]) |port| if (message.object == port.link_proxy and message.opcode == 0) {
                const global_id = try r.int();

                _ = global_id;

                if (try r.int() != self.selected.?.node.id or try r.int() != port.source_global or try r.int() != self.node_id) {
                    return error.SourceChanged;
                }

                _ = try r.int();
                _ = try r.long(5);

                const state: i32 = @bitCast(try r.int());
                if (state == -2 or state == -1) {
                    return error.SourceDisconnected;
                }
            };
        },
    }

    if (self.stream_created and !self.links_created) {
        try self.createLinks();
    }

    return true;
}

pub fn process(self: *PipeWireClient) Error!?Block {
    var count: u64 = 0;
    const result = linux.read(self.wake_fd, std.mem.asBytes(&count).ptr, 8);

    if (linux.errno(result) == .AGAIN or linux.errno(result) == .INTR) {
        return null;
    }

    if (result != 8 or count == 0) {
        self.connection.errno = linux.errno(result);

        return error.WakeFailed;
    }

    self.diagnostic = .{ .wake_count = count };

    const activation = self.activation orelse {
        return error.InvalidMemory;
    };

    atomicStore(activation, 0, 2);
    store(u64, activation, 40, now());

    var handed_off = false;
    defer if (!handed_off) self.finishCycle();

    if (!self.running) {
        return null;
    }

    const clock = self.position orelse {
        return error.InvalidMemory;
    };

    const rate_num = load(u32, clock, 80);
    const rate = load(u32, clock, 84);
    const position = load(u64, clock, 88);
    const duration = load(u64, clock, 96);
    const clock_id = load(u32, clock, 4);

    self.diagnostic.graph_rate_num = rate_num;
    self.diagnostic.graph_rate_hz = rate;
    self.diagnostic.graph_position = position;
    self.diagnostic.graph_duration = duration;

    if (rate_num != 1 or rate < 8000 or rate > 192000 or duration == 0 or duration > samples_max) {
        return error.UnsupportedFormat;
    }

    if (self.previous) |previous| {
        if (count != 1 or (previous.clock == clock_id and previous.rate == rate and position != previous.position +% previous.duration)) {
            return error.TimelineDiscontinuity;
        }
    }

    var block: Block = undefined;
    var header_present = false;

    var all_silent = true;

    for (self.ports[0..self.ports_count], 0..) |port, channel| {
        self.diagnostic.channel = channel;

        // Startup can wake us before format/IO/buffers are ready. Once
        // samples have been accepted, losing any of them is a discontinuity;
        // silently waiting would hide missing audio behind live callbacks.
        if (!port.negotiated) {
            if (self.previous != null) {
                return error.UnsupportedFormat;
            }

            return null;
        }

        const io = port.io orelse {
            if (self.previous != null) {
                return error.TimelineDiscontinuity;
            }

            return null;
        };

        const status = atomicLoad(io, 0);

        if (status != 2) {
            if (self.previous != null) {
                return error.TimelineDiscontinuity;
            }

            return null;
        }

        const id = load(u32, io, 4);

        self.diagnostic.buffer_id = id;

        if (id >= port.buffers_count) {
            return error.InvalidBuffer;
        }

        const buffer = port.buffers[id];

        const chunk = buffer.chunk orelse {
            return error.InvalidBuffer;
        };

        const data = buffer.data orelse {
            return error.InvalidBuffer;
        };

        const offset = load(u32, chunk, 0) % data.len;
        const size = load(u32, chunk, 4);
        const stride = load(i32, chunk, 8);
        const flags = load(u32, chunk, 12);

        self.diagnostic.chunk_offset = load(u32, chunk, 0);
        self.diagnostic.chunk_size = size;
        self.diagnostic.chunk_stride = stride;
        self.diagnostic.chunk_flags = flags;

        // 0.3.48 uses stride 0 for contiguous DSP floats. Accept that
        // implicit stride alongside explicit sizeof(f32), never arbitrary
        // strides that would change the mapped sample geometry.
        if (size > data.len or size % 4 != 0 or offset % 4 != 0 or (stride != 0 and stride != 4) or duration > data.len / 4) {
            return error.InvalidBuffer;
        }

        if (flags & 1 != 0) {
            return error.CorruptedBuffer;
        }

        if (flags & ~@as(u32, 3) != 0) {
            return error.InvalidBuffer;
        }

        var silent = flags & 2 != 0;

        if (buffer.header) |header| {
            header_present = true;

            const header_flags = load(u32, header, 0);

            self.diagnostic.header_flags = header_flags;
            self.diagnostic.header_sequence = load(u64, header, 24);
            self.diagnostic.header_pts_ns = load(i64, header, 8);

            if (header_flags & 2 != 0) {
                return error.CorruptedBuffer;
            }

            if (header_flags & 1 != 0 and self.previous != null) {
                return error.TimelineDiscontinuity;
            }

            if (header_flags & ~@as(u32, 0x11) != 0) {
                return error.InvalidBuffer;
            }

            silent = silent or header_flags & 16 != 0;
        }

        // DSP cycle length comes from the graph clock. Older sources
        // leave chunk.size at its initial quantum when the graph quantum
        // changes; bound this cycle against the mapped capacity instead.
        const n: usize = @intCast(duration);

        all_silent = all_silent and silent;

        const first_bytes = @min(n * 4, data.len - offset);

        block.channels[channel] = if (silent)
            .silence
        else
            .{ .samples = .{
                .first = std.mem.bytesAsSlice(f32, data[offset..][0..first_bytes]),
                .second = std.mem.bytesAsSlice(f32, data[0 .. n * 4 - first_bytes]),
            } };
    }

    if (self.ports_count == 0) {
        return null;
    }

    self.previous = .{ .position = position, .duration = duration, .rate = rate, .clock = clock_id };
    block.channels_count = self.ports_count;
    block.samples_count = @intCast(duration);
    block.rate = rate;
    block.position = position;
    block.header_present = header_present;
    block.silence = all_silent;
    block.client = self;
    handed_off = true;

    return block;
}

// Release on every path, including conversion errors. Delaying this until
// the caller consumes Block prevents PipeWire from recycling borrowed PCM.
pub fn finishCycle(self: *PipeWireClient) void {
    const activation = self.activation.?;

    for (self.ports[0..self.ports_count]) |port| if (port.io) |io| atomicStore(io, 0, 1);

    atomicStore(activation, 8, 1);
    store(u64, activation, 48, now());
    atomicStore(activation, 0, 3);

    for (self.peers) |entry| if (entry) |peer| {
        const pending_atomic: *i32 = @ptrCast(@alignCast(peer.activation[16..].ptr));

        if (@atomicRmw(i32, pending_atomic, .Sub, 1, .acq_rel) == 1) {
            store(u64, peer.activation, 32, now());
            atomicStore(peer.activation, 0, 1);

            const one: u64 = 1;

            _ = linux.write(peer.fd, std.mem.asBytes(&one).ptr, 8);
        }
    };
}

/// Publish the current graph format for stream observers. Call only when
/// the negotiated rate changes, outside process()'s buffer ownership.
pub fn publishFormat(self: *PipeWireClient, rate: u32) Error!void {
    // GNOME observes recording through pipewire-pulse. That bridge ignores
    // streams without a node-level audio Format even when their DSP ports
    // are linked and carrying audio. Describe the planar graph input here;
    // port negotiation and Voiced's later mono/16 kHz conversion stay local.
    var w: protocol.Writer = .{};

    _ = try w.begin(14);

    try w.int(3); // UPDATE_PARAMS | UPDATE_INFO
    try w.int(2);

    const format = try w.begin(15);

    try w.word(0x40003); // SPA_TYPE_OBJECT_Format
    try w.word(4); // SPA_PARAM_Format
    try w.property(1, 3, 1); // audio
    try w.property(2, 3, 1); // raw
    try w.property(0x10001, 3, 0x206); // F32P
    try w.property(0x10003, 4, rate);
    try w.property(0x10004, 4, @intCast(self.ports_count));
    try w.word(0x10005); // audio.position
    try w.word(0);

    const positions = try w.begin(13); // Array of channel IDs

    try w.word(4);
    try w.word(3);

    const channel_names = [_][]const u8{
        "UNK", "NA",   "MONO", "FL",  "FR",  "FC",   "LFE",  "SL",  "SR",  "FLC", "FRC", "RC",   "RL",  "RR",
        "TC",  "TFL",  "TFC",  "TFR", "TRL", "TRC",  "TRR",  "RLC", "RRC", "FLW", "FRW", "LFE2", "FLH", "FCH",
        "FRH", "TFLC", "TFRC", "TSL", "TSR", "LLFE", "RLFE", "BC",  "BLC", "BRC",
    };

    for (self.ports[0..self.ports_count]) |port| {
        const name = port.channel.get();

        const channel: u32 = for (channel_names, 0..) |known, id| {
            if (std.mem.eql(u8, name, known)) {
                break @intCast(id);
            }
        } else if (std.mem.startsWith(u8, name, "AUX")) auxiliary: {
            const id = std.fmt.parseInt(u8, name[3..], 10) catch {
                break :auxiliary 0;
            };

            break :auxiliary if (id < 64)
                @as(u32, 0x1000) + id
            else
                0;
        } else 0;

        try w.word(channel);
    }

    try w.finish(positions);
    try w.finish(format);

    // The stream applies no additional gain/mute before mixing. Without
    // channelVolumes the PulseAudio bridge reports its zero-initialized
    // volume as 0%, even though capture is audible. These are read-only.
    const props = try w.begin(15);

    try w.word(0x40002); // SPA_TYPE_OBJECT_Props
    try w.word(2); // SPA_PARAM_Props
    try w.property(0x10004, 2, 0); // mute = false
    try w.word(0x10008); // channelVolumes
    try w.word(0);

    const volumes = try w.begin(13);

    try w.word(4);
    try w.word(6); // Float

    for (0..self.ports_count) |_| {
        try w.word(@bitCast(@as(f32, 1)));
    }

    try w.finish(volumes);
    try w.finish(props);

    const info = try w.begin(14);

    try w.int(@intCast(self.ports_count));
    try w.int(0);
    try w.long(4); // CHANGE_MASK_PARAMS; no property dictionary
    try w.long(1); // RT
    try w.int(0);
    try w.int(2);
    try w.id(4); // Format is observable, not configurable by a mixer.
    // Parameter observers only refresh when the flags change. Toggle the
    // protocol's SERIAL bit so later graph-rate changes stay observable.
    try w.int(2 | @as(u32, self.format_serial));
    try w.id(2); // Props
    try w.int(2); // READ
    try w.finish(info);
    try w.finish(0);
    try self.connection.send(self.node_proxy, 2, w.data());

    self.format_serial ^= 1;
}

fn global(self: *PipeWireClient, r: *protocol.Reader) Error!void {
    const id = try r.int();

    _ = try r.int();

    const interface = try r.string();
    const version = try r.int();
    var props = try r.structure();

    if (std.mem.eql(u8, interface, "PipeWire:Interface:Factory")) {
        const count = try props.int();
        var factory: []const u8 = "";
        var factory_version: u32 = 0;

        for (0..try bounded(count, 1024)) |_| {
            const key = try props.string();
            const value = try props.string();

            if (std.mem.eql(u8, key, "factory.name")) {
                factory = value;
            }

            if (std.mem.eql(u8, key, "factory.type.version")) {
                factory_version = std.fmt.parseInt(u32, value, 10) catch {
                    return error.InvalidMessage;
                };
            }
        }

        if (std.mem.eql(u8, factory, "client-node")) {
            self.client_node_version = factory_version;
        }

        return;
    }

    if (std.mem.eql(u8, interface, "PipeWire:Interface:Metadata")) {
        const count = try props.int();

        for (0..try bounded(count, 1024)) |_| {
            const key = try props.string();
            const value = try props.string();

            if (std.mem.eql(u8, key, "metadata.name") and std.mem.eql(u8, value, "default")) {
                self.metadata_proxy = self.proxy_next;
                self.proxy_next += 1;

                try self.bind(id, interface, @min(version, 3), self.metadata_proxy);
            }
        }

        return;
    }

    var object: Object = .{ .id = id, .kind = if (std.mem.eql(u8, interface, "PipeWire:Interface:Node"))
        .node
    else if (std.mem.eql(u8, interface, "PipeWire:Interface:Device"))
        .device
    else if (std.mem.eql(u8, interface, "PipeWire:Interface:Port"))
        .port
    else
        return };

    try properties(&object, props);

    if (object.kind == .node or object.kind == .device) {
        object.proxy = self.proxy_next;
        self.proxy_next += 1;

        try self.bind(id, interface, @min(version, 3), object.proxy);
    }

    for (&self.catalog) |*entry| if (entry.* == null) {
        entry.* = try self.text.replace(null, object);

        return;
    };

    return error.CatalogFull;
}

fn createStream(self: *PipeWireClient) Error!void {
    if (self.client_node_version < 4) {
        return error.UnsupportedVersion;
    }

    var selected: ?Object = null;
    var matches: usize = 0;

    for (self.catalog) |entry| {
        const stored = entry orelse {
            continue;
        };

        if (stored.data != .node or !stored.data.node.is_source) {
            continue;
        }

        const object = stored.expand(&self.text);
        const device = self.find(object.parent);

        const matches_source = switch (self.source) {
            .default => self.default_source.size == 0 or std.mem.eql(u8, object.name.get(), self.default_source.get()),
            .node_name => |name| std.mem.eql(u8, name, object.name.get()),
            .device_serial => |serial| device != null and std.mem.eql(u8, serial, device.?.device_serial.get()),
        };

        if (!matches_source) {
            continue;
        }

        matches += 1;

        if (selected == null or (self.source == .default and object.priority > selected.?.priority)) {
            selected = object;
        }
    }

    if (selected == null) {
        return error.SourceNotFound;
    }

    if (matches > 1 and self.source != .default) {
        return error.SourceAmbiguous;
    }

    const node = selected.?;

    self.selected = .{ .node = node, .device = self.find(node.parent) };

    for (self.catalog) |entry| {
        const stored = entry orelse {
            continue;
        };

        if (stored.data != .port or stored.data.port.parent != node.id or !stored.data.port.is_output) {
            continue;
        }

        const object = stored.expand(&self.text);

        if (self.ports_count == self.ports.len) {
            return error.TooManyChannels;
        }

        self.ports[self.ports_count] = .{ .source_global = object.id, .source_port = object.port, .channel = object.channel };
        self.ports_count += 1;
    }

    if (self.ports_count == 0) {
        return error.SourcePortsUnavailable;
    }

    self.node_proxy = self.proxy_next;
    self.proxy_next += 1;

    var w: protocol.Writer = .{};

    _ = try w.begin(14);

    try w.string("client-node");
    try w.string("PipeWire:Interface:ClientNode");
    // 0.3.48 advertises v4, 1.0.5 v5, and 1.6.8 v6. Request only
    // the implemented interface: a newer server release is not permission
    // to interpret a newer wire/shared-memory contract as the old one.
    try w.int(@min(self.client_node_version, 5));
    try w.dictionary(&.{ .{ "node.name", "voiced-capture" }, .{ "node.description", "Voiced capture" }, .{ "media.class", "Stream/Input/Audio" }, .{ "node.autoconnect", "false" }, .{ "node.want-driver", "true" }, .{ "node.dont-reconnect", "true" } });
    try w.int(self.node_proxy);
    try w.finish(0);
    try self.connection.send(0, 6, w.data());

    w.size = 0;
    _ = try w.begin(14);

    try w.int(2);
    try w.int(0);

    const info = try w.begin(14);

    try w.int(@intCast(self.ports_count));
    try w.int(0);
    // An absent property dictionary must not carry CHANGE_MASK_PROPS;
    // older servers dereference it when that bit is advertised.
    try w.long(5);
    try w.long(1);
    try w.int(0);
    try w.int(0);
    try w.finish(info);
    try w.finish(0);
    try self.connection.send(self.node_proxy, 2, w.data());

    for (0..self.ports_count) |index| {
        try self.portUpdate(@intCast(index), null);
    }

    self.stream_created = true;
}

fn createLinks(self: *PipeWireClient) Error!void {
    if (self.node_id == invalid) {
        return;
    }

    for (self.ports[0..self.ports_count], 0..) |_, index| {
        var found = false;

        for (self.catalog) |entry| if (entry) |stored| {
            if (stored.data == .port and stored.data.port.parent == self.node_id and stored.data.port.number == index) {
                found = true;

                break;
            }
        };

        if (!found) {
            return;
        }
    }

    for (self.ports[0..self.ports_count], 0..) |*port, index| {
        var ids: [4][24]u8 = undefined;
        const source_node = std.fmt.bufPrint(&ids[0], "{d}", .{self.selected.?.node.id}) catch {
            unreachable;
        };

        const source_port = std.fmt.bufPrint(&ids[1], "{d}", .{port.source_global}) catch {
            unreachable;
        };

        const target_node = std.fmt.bufPrint(&ids[2], "{d}", .{self.node_id}) catch {
            unreachable;
        };

        const target_port = std.fmt.bufPrint(&ids[3], "{d}", .{index}) catch {
            unreachable;
        };

        port.link_proxy = self.proxy_next;
        self.proxy_next += 1;

        var w: protocol.Writer = .{};

        _ = try w.begin(14);

        try w.string("link-factory");
        try w.string("PipeWire:Interface:Link");
        try w.int(3);
        try w.dictionary(&.{ .{ "link.output.node", source_node }, .{ "link.output.port", source_port }, .{ "link.input.node", target_node }, .{ "link.input.port", target_port } });
        try w.int(port.link_proxy);
        try w.finish(0);
        try self.connection.send(0, 6, w.data());
    }

    self.links_created = true;
}

fn nodeEvent(self: *PipeWireClient, opcode: u8, r: *protocol.Reader) Error!void {
    switch (opcode) {
        0 => {
            if (self.activation != null) {
                return error.InvalidMessage;
            }

            self.wake_fd = try self.eventDescriptor(try r.long(18));
            self.completion_fd = try self.eventDescriptor(try r.long(18));
            self.activation = try self.region(try r.int(), try r.int(), try r.int(), 64, true);

            var w: protocol.Writer = .{};

            _ = try w.begin(14);

            try w.scalar(2, 1);
            try w.finish(0);
            try self.connection.send(self.node_proxy, 4, w.data());
        },

        1 => {
            const id = try r.id();

            _ = try r.int();
            _ = try r.next();

            if (id != 2) {
                return error.UnsupportedFormat;
            }
        },

        2 => {
            const id = try r.id();
            const mem = try r.int();
            const offset = try r.int();
            const size = try r.int();

            // SPA_IO_Position is supplied at a server-selected offset.
            // Its placement inside activation memory changed across these
            // releases; only the clock prefix we read is layout-stable.
            // SPA_IO_Clock (3) is not the driving graph's position clock.
            if (id == 7) {
                self.position = if (mem == invalid)
                    null
                else
                    try self.region(mem, offset, size, 104, false);
            } else if (id != 3) return error.UnsupportedIO;
        },

        4 => {
            const command = try r.next();
            if (command.kind != 15 or command.body.len < 8 or load(u32, command.body, 0) != 0x30002) {
                return error.InvalidMessage;
            }

            switch (load(u32, command.body, 4)) {
                0, 1 => self.running = false,
                2 => self.running = true,
                else => return error.UnsupportedIO,
            }
        },

        7 => {
            const direction = try r.int();
            const port = try r.int();
            const id = try r.id();

            _ = try r.int();

            const format = try r.next();

            if (direction != 0 or port >= self.ports_count or id != 4) {
                return error.UnsupportedFormat;
            }

            if (format.kind == 1) {
                self.ports[port].negotiated = false;

                try self.portUpdate(port, null);

                return;
            }

            if (format.kind != 15 or format.body.len < 8 or load(u32, format.body, 0) != 0x40003 or load(u32, format.body, 4) != 4) {
                return error.UnsupportedFormat;
            }

            var props: protocol.Reader = .{ .bytes = format.body[8..] };
            var fields: u3 = 0;

            while (props.bytes.len > 0) {
                if (props.bytes.len < 8) {
                    return error.InvalidMessage;
                }

                const key = load(u32, props.bytes, 0);

                props.bytes = props.bytes[8..];

                const value = try props.next();

                switch (key) {
                    1 => {
                        if (try value.scalar(3) != 1) {
                            return error.UnsupportedFormat;
                        }

                        fields |= 1;
                    },

                    2 => {
                        if (try value.scalar(3) != 2) {
                            return error.UnsupportedFormat;
                        }

                        fields |= 2;
                    },

                    0x10001 => {
                        if (try value.scalar(3) != 0x206) {
                            return error.UnsupportedFormat;
                        }

                        fields |= 4;
                    },
                    else => return error.UnsupportedFormat,
                }
            }

            if (fields != 7) {
                return error.UnsupportedFormat;
            }

            self.ports[port].negotiated = true;

            try self.portUpdate(port, format.encoded);
        },
        8 => try self.useBuffers(r),
        9 => {
            if (try r.int() != 0) {
                return error.UnsupportedIO;
            }

            const port = try r.int();
            const mix = try r.int();
            const id = try r.id();
            const mem = try r.int();
            const offset = try r.int();
            const size = try r.int();

            if (port >= self.ports_count or mix != 0 or id != 1) {
                return error.UnsupportedIO;
            }

            self.ports[port].io = if (mem == invalid)
                null
            else
                try self.region(mem, offset, size, 8, true);
        },

        10 => {
            const node = try r.int();
            const fd_index = try r.long(18);
            const mem = try r.int();
            const offset = try r.int();
            const size = try r.int();

            if (mem == invalid) {
                for (&self.peers) |*entry| if (entry.* != null and entry.*.?.id == node) {
                    _ = linux.close(entry.*.?.fd);
                    entry.* = null;

                    return;
                };

                return;
            }

            const activation = try self.region(mem, offset, size, 64, true);

            const fd = try self.eventDescriptor(fd_index);
            errdefer _ = linux.close(fd);

            // The server can repeat the local activation during setup.
            for (&self.peers) |*entry| if (entry.* != null and entry.*.?.id == node) {
                _ = linux.close(entry.*.?.fd);
                entry.* = .{ .id = node, .fd = fd, .activation = activation };

                return;
            };

            for (&self.peers) |*entry| if (entry.* == null) {
                entry.* = .{ .id = node, .fd = fd, .activation = activation };

                return;
            };

            return error.TooManyPeers;
        },

        11 => {
            if (try r.int() != 0) {
                return error.UnsupportedIO;
            }

            const port = try r.int();
            const mix = try r.int();
            const peer = try r.int();

            if (port >= self.ports_count or mix != 0) {
                return error.UnsupportedIO;
            }

            if (peer != self.ports[port].source_global and peer != invalid) {
                return error.SourceChanged;
            }
        },
        else => return error.UnsupportedIO,
    }
}

fn useBuffers(self: *PipeWireClient, r: *protocol.Reader) Error!void {
    if (try r.int() != 0) {
        return error.InvalidBuffer;
    }

    const index = try r.int();
    const mix = try r.int();
    const flags = try r.int();
    const count = try r.int();

    if (index >= self.ports_count or mix != 0 or flags != 0) {
        return error.InvalidBuffer;
    }

    if (count > buffers_max) {
        return error.TooManyBuffers;
    }

    const port = &self.ports[index];

    port.buffers_count = 0;
    port.buffers = @splat(.{});

    for (port.buffers[0..count]) |*buffer| {
        const mem = try r.int();
        const offset = try r.int();
        const size = try r.int();
        const metas = try r.int();
        const region_bytes = try self.region(mem, offset, size, 16, false);
        var cursor: usize = 0;

        for (0..try bounded(metas, 16)) |_| {
            const kind = try r.id();
            const meta_size = try r.int();

            if (meta_size > region_bytes.len - cursor) {
                return error.InvalidBuffer;
            }

            if (kind == 1) {
                if (buffer.header != null or meta_size < 32) {
                    return error.InvalidBuffer;
                }

                buffer.header = region_bytes[cursor..][0..meta_size];
            }

            const padded = std.mem.alignForward(usize, meta_size, 8);
            if (padded > region_bytes.len - cursor) {
                return error.InvalidBuffer;
            }

            cursor += padded;
        }

        if (try r.int() != 1 or region_bytes.len - cursor < 16) {
            return error.InvalidBuffer;
        }

        buffer.chunk = region_bytes[cursor..][0..16];

        const kind = try r.id();
        const data_id = try r.int();
        const data_flags = try r.int();
        const mapoffset = try r.int();
        const capacity = try r.int();

        if (capacity == 0 or capacity % 4 != 0 or capacity > samples_max * 4 or data_flags & 1 == 0) {
            return error.InvalidBuffer;
        }

        if (kind == 1) {
            if (data_id > region_bytes.len or capacity > region_bytes.len - data_id) {
                return error.InvalidBuffer;
            }

            buffer.data = region_bytes[data_id..][0..capacity];
        } else if (kind == 4) {
            buffer.data = try self.region(data_id, mapoffset, capacity, capacity, false);
        } else return error.InvalidBuffer;
    }

    port.buffers_count = count;
}

fn portUpdate(self: *PipeWireClient, index: u32, format: ?[]const u8) Error!void {
    var w: protocol.Writer = .{};

    _ = try w.begin(14);

    try w.int(0);
    try w.int(index);
    try w.int(3);
    try w.int(if (format == null) 4 else 5);

    var start = try w.begin(15);

    try w.word(0x40003);
    try w.word(3);
    try w.property(1, 3, 1);
    try w.property(2, 3, 2);
    try w.property(0x10001, 3, 0x206);
    try w.finish(start);

    if (format) |bytes| {
        try w.raw(bytes);
    }

    start = try w.begin(15);

    try w.word(0x40004);
    try w.word(5);
    try w.property(1, 4, 2);
    try w.property(2, 4, 1);
    try w.property(3, 4, samples_max * 4);
    try w.property(4, 4, 4);
    try w.finish(start);

    start = try w.begin(15);

    try w.word(0x40006);
    try w.word(7);
    try w.property(1, 3, 1);
    try w.property(2, 4, 8);
    try w.finish(start);

    start = try w.begin(15);

    try w.word(0x40005);
    try w.word(6);
    try w.property(1, 3, 1);
    try w.property(2, 4, 32);
    try w.finish(start);

    start = try w.begin(14);

    try w.long(15);
    try w.long(128);
    try w.int(0);
    try w.int(1);
    try w.int(3);
    try w.string("port.name");

    var name: [32]u8 = undefined;
    try w.string(std.fmt.bufPrint(&name, "input_{d}", .{index}) catch {
        unreachable;
    });

    try w.string("audio.channel");
    try w.string(self.ports[index].channel.get());
    try w.string("format.dsp");
    try w.string("32 bit float mono audio");
    try w.int(5);

    for ([_][2]u32{ .{ 3, 2 }, .{ 4, if (format == null) 4 else 6 }, .{ 5, 2 }, .{ 7, 2 }, .{ 6, 2 } }) |param| {
        try w.id(param[0]);
        try w.int(param[1]);
    }

    try w.finish(start);
    try w.finish(0);
    try self.connection.send(self.node_proxy, 3, w.data());
}

fn eventDescriptor(self: *PipeWireClient, index: i64) Error!linux.fd_t {
    const fd = try self.connection.duplicateDescriptor(index);
    errdefer _ = linux.close(fd);

    const flags = linux.fcntl(fd, linux.F.GETFL, 0);

    self.connection.errno = linux.errno(flags);

    if (self.connection.errno != .SUCCESS) {
        return error.DescriptorFailed;
    }

    self.connection.errno = linux.errno(linux.fcntl(fd, linux.F.SETFL, flags | @as(u32, @bitCast(linux.O{ .NONBLOCK = true }))));

    if (self.connection.errno != .SUCCESS) {
        return error.DescriptorFailed;
    }

    return fd;
}

fn region(self: *PipeWireClient, id: u32, offset: u32, size: u32, minimum: usize, writable: bool) Error![]u8 {
    const memory = self.registeredMemory(id) orelse {
        return error.InvalidMemory;
    };

    if (offset % 8 != 0 or size < minimum or offset > memory.bytes.len or size > memory.bytes.len - offset or (writable and !memory.writable)) {
        return error.InvalidMemory;
    }

    return memory.bytes[offset..][0..size];
}
fn registeredMemory(self: *PipeWireClient, id: u32) ?*Memory {
    for (&self.memories) |*entry| if (entry.*) |*memory| {
        if (memory.id != null and memory.id.? == id) {
            return memory;
        }
    };

    return null;
}
fn releaseUnusedMemory(self: *PipeWireClient) void {
    for (&self.memories) |*entry| if (entry.*) |memory| {
        if (memory.id == null and !self.references(memory.bytes)) {
            std.posix.munmap(memory.bytes);

            entry.* = null;
        }
    };
}
fn references(self: *const PipeWireClient, bytes: []u8) bool {
    if (inside(self.activation, bytes) or inside(self.position, bytes)) {
        return true;
    }

    for (self.peers) |peer| if (peer != null and inside(peer.?.activation, bytes)) return true;

    for (self.ports[0..self.ports_count]) |port| {
        if (inside(port.io, bytes)) {
            return true;
        }

        for (port.buffers[0..port.buffers_count]) |buffer| if (inside(buffer.header, bytes) or inside(buffer.chunk, bytes) or inside(buffer.data, bytes)) return true;
    }

    return false;
}
fn find(self: *const PipeWireClient, id: u32) ?Object {
    if (id == invalid) {
        return null;
    }

    for (self.catalog) |entry| if (entry != null and entry.?.id == id) return entry.?.expand(&self.text);

    return null;
}
fn bind(self: *PipeWireClient, id: u32, interface: []const u8, version: u32, proxy: u32) Error!void {
    var w: protocol.Writer = .{};

    _ = try w.begin(14);

    try w.int(id);
    try w.string(interface);
    try w.int(version);
    try w.int(proxy);
    try w.finish(0);
    try self.connection.send(2, 1, w.data());
}
fn ints(self: *PipeWireClient, object: u32, opcode: u8, values: []const u32) Error!void {
    var w: protocol.Writer = .{};

    _ = try w.begin(14);

    for (values) |value| {
        try w.int(value);
    }

    try w.finish(0);
    try self.connection.send(object, opcode, w.data());
}

fn parseDefaultSource(input: []const u8) Error!Text {
    const prefix = "{\"name\":\"";
    const suffix = "\"}";

    // PERFORMANCE: Voiced needs only the node name from PipeWire's canonical
    // `{"name":"..."}` metadata. This fixed envelope does not justify pulling
    // a general JSON parser into the resident daemon. Reject noncanonical data
    // rather than accepting a partial interpretation that could select the
    // wrong microphone.
    if (input.len < prefix.len + suffix.len or !std.mem.startsWith(u8, input, prefix) or !std.mem.endsWith(u8, input, suffix)) {
        return error.UnsupportedDefaultSourceMetadata;
    }

    const name = input[prefix.len .. input.len - suffix.len];
    var source: Text = .{};

    if (name.len > source.bytes.len) {
        return error.DefaultSourceNameTooLong;
    }

    if (!std.unicode.utf8ValidateSlice(name)) {
        return error.DefaultSourceNameInvalidUtf8;
    }

    for (name) |byte| if (byte < ' ' or byte == '"' or byte == '\\') return error.UnsupportedDefaultSourceMetadata;

    source.set(name) catch {
        unreachable;
    };

    return source;
}

// The catalog stores only the fields meaningful for each interface. Public
// Object/Identity values are transient expanded views used by selection and
// diagnostics; their fixed strings are never repeated in every catalog slot.
const BlockIndex = enum(u16) { _ };
const TextRef = struct { first: BlockIndex = @enumFromInt(0), size: u16 = 0 };
const StoredObject = struct {
    id: u32,
    serial: u64,
    proxy: u32,
    data: union(ObjectKind) {
        node: struct { parent: u32, priority: i32, is_source: bool, name: TextRef, description: TextRef },
        device: struct { serial: TextRef, description: TextRef },
        port: struct { parent: u32, number: u32, is_output: bool, channel: TextRef },
    },

    fn references(self: StoredObject) [2]TextRef {
        return switch (self.data) {
            .node => |n| .{ n.name, n.description },
            .device => |d| .{ d.serial, d.description },
            .port => |p| .{ p.channel, .{} },
        };
    }
    pub fn expand(self: StoredObject, text: *const TextStorage) Object {
        var object: Object = .{ .id = self.id, .serial = self.serial, .proxy = self.proxy, .kind = std.meta.activeTag(self.data) };

        switch (self.data) {
            .node => |n| {
                object.parent = n.parent;
                object.priority = n.priority;
                object.is_source = n.is_source;
                object.name = text.read(n.name);
                object.description = text.read(n.description);
            },

            .device => |d| {
                object.device_serial = text.read(d.serial);
                object.description = text.read(d.description);
            },

            .port => |p| {
                object.parent = p.parent;
                object.port = p.number;
                object.is_output = p.is_output;
                object.channel = text.read(p.channel);
            },
        }

        return object;
    }
};
const TextStorage = struct {
    // 128 objects × at most two meaningful strings × 256 bytes. This
    // covers the full existing limits, including replacement at full capacity.
    bytes: [1024][64]u8 = undefined,
    next: [1024]BlockIndex = undefined,
    free: std.StaticBitSet(1024) = .initFull(),

    fn replace(self: *TextStorage, previous: ?StoredObject, object: Object) Error!StoredObject {
        const strings: [2][]const u8 = switch (object.kind) {
            .node => .{ object.name.get(), object.description.get() },
            .device => .{ object.device_serial.get(), object.description.get() },
            .port => .{ object.channel.get(), "" },
        };

        var available = self.free.count();

        if (previous) |old| for (old.references()) |ref| {
            available += blocks(ref.size);
        };

        if (blocks(strings[0].len) + blocks(strings[1].len) > available) return error.CatalogTextFull;

        // Check before releasing old text: an exhausted pool preserves the
        // previous catalog for the error report. No allocation can fail below.
        if (previous) |old| {
            self.releaseObject(old);
        }

        const a = self.store(strings[0]);
        const b = self.store(strings[1]);

        return .{ .id = object.id, .serial = object.serial, .proxy = object.proxy, .data = switch (object.kind) {
            .node => .{ .node = .{ .parent = object.parent, .priority = object.priority, .is_source = object.is_source, .name = a, .description = b } },
            .device => .{ .device = .{ .serial = a, .description = b } },
            .port => .{ .port = .{ .parent = object.parent, .number = object.port, .is_output = object.is_output, .channel = a } },
        } };
    }
    fn store(self: *TextStorage, text: []const u8) TextRef {
        var ref: TextRef = .{ .size = @intCast(text.len) };
        var link = &ref.first;
        var offset: usize = 0;

        while (offset < text.len) {
            const index = self.free.findFirstSet().?;

            self.free.unset(index);

            link.* = @enumFromInt(index);

            const count = @min(64, text.len - offset);

            @memcpy(self.bytes[index][0..count], text[offset..][0..count]);

            offset += count;
            link = &self.next[index];
        }

        return ref;
    }
    fn releaseObject(self: *TextStorage, object: StoredObject) void {
        for (object.references()) |ref| {
            var index = ref.first;

            for (0..blocks(ref.size)) |ordinal| {
                const i = @intFromEnum(index);
                std.debug.assert(!self.free.isSet(i));

                self.free.set(i);

                if (ordinal + 1 < blocks(ref.size)) {
                    index = self.next[i];
                }
            }
        }
    }
    fn read(self: *const TextStorage, ref: TextRef) Text {
        var result: Text = .{ .size = ref.size };
        var offset: usize = 0;
        var index = ref.first;

        while (offset < ref.size) {
            const i = @intFromEnum(index);
            const count = @min(64, ref.size - offset);

            @memcpy(result.bytes[offset..][0..count], self.bytes[i][0..count]);

            offset += count;

            if (offset < ref.size) {
                index = self.next[i];
            }
        }

        return result;
    }
    fn blocks(bytes: usize) usize {
        return (bytes + 63) / 64;
    }
};

const Memory = struct { id: ?u32, bytes: []align(std.heap.page_size_min) u8, writable: bool };
const Peer = struct { id: u32, fd: linux.fd_t, activation: []u8 };
const Buffer = struct { header: ?[]const u8 = null, chunk: ?[]const u8 = null, data: ?[]const u8 = null };
const Port = struct {
    source_global: u32 = invalid,
    source_port: u32 = invalid,
    link_proxy: u32 = invalid,
    link_global: u32 = invalid,
    channel: Text = .{},
    negotiated: bool = false,
    io: ?[]u8 = null,
    buffers: [buffers_max]Buffer = @splat(.{}),
    buffers_count: usize = 0,
};
fn properties(object: *Object, input: protocol.Reader) Error!void {
    var r = input;
    const count = try r.int();

    for (0..try bounded(count, 1024)) |_| {
        const key = try r.string();
        const value = try r.string();

        if (std.mem.eql(u8, key, "object.serial")) {
            object.serial = std.fmt.parseInt(u64, value, 10) catch {
                return error.InvalidMessage;
            };
        } else if (std.mem.eql(u8, key, "node.name")) {
            try object.name.set(value);
        } else if (std.mem.eql(u8, key, "node.description") or std.mem.eql(u8, key, "device.description")) {
            try object.description.set(value);
        } else if (std.mem.eql(u8, key, "device.serial")) {
            try object.device_serial.set(value);
        } else if (std.mem.eql(u8, key, "audio.channel")) {
            try object.channel.set(value);
        } else if (std.mem.eql(u8, key, "device.id") or (object.kind == .port and std.mem.eql(u8, key, "node.id"))) {
            object.parent = std.fmt.parseInt(u32, value, 10) catch {
                return error.InvalidMessage;
            };
        } else if (std.mem.eql(u8, key, "port.id")) {
            object.port = std.fmt.parseInt(u32, value, 10) catch {
                return error.InvalidMessage;
            };
        } else if (std.mem.eql(u8, key, "port.direction")) {
            object.is_output = std.mem.eql(u8, value, "out");
        } else if (std.mem.eql(u8, key, "media.class")) {
            object.is_source = std.mem.eql(u8, value, "Audio/Source") or std.mem.eql(u8, value, "Audio/Source/Virtual");
        } else if (std.mem.eql(u8, key, "priority.session")) {
            object.priority = std.fmt.parseInt(i32, value, 10) catch {
                return error.InvalidMessage;
            };
        }
    }

    try r.end();
}
fn bounded(value: usize, maximum: usize) Error!usize {
    if (value > maximum) {
        return error.InvalidMessage;
    }

    return value;
}
fn inside(part: ?[]const u8, bytes: []const u8) bool {
    const memory_region = part orelse {
        return false;
    };

    return @intFromPtr(memory_region.ptr) >= @intFromPtr(bytes.ptr) and @intFromPtr(memory_region.ptr) < @intFromPtr(bytes.ptr) + bytes.len;
}
fn load(comptime T: type, bytes: []const u8, offset: usize) T {
    return std.mem.bytesToValue(T, bytes[offset..][0..@sizeOf(T)]);
}
fn store(comptime T: type, bytes: []u8, offset: usize, value: T) void {
    @memcpy(bytes[offset..][0..@sizeOf(T)], std.mem.asBytes(&value));
}
// These words live at protocol-defined offsets in PipeWire-supplied mappings,
// not in Voiced-owned `std.atomic.Value` storage. Raw builtins preserve the
// external shared-memory ABI.
fn atomicLoad(bytes: []u8, offset: usize) u32 {
    const word_atomic: *u32 = @ptrCast(@alignCast(bytes[offset..].ptr));

    return @atomicLoad(u32, word_atomic, .acquire);
}
fn atomicStore(bytes: []u8, offset: usize, value: u32) void {
    const word_atomic: *u32 = @ptrCast(@alignCast(bytes[offset..].ptr));

    @atomicStore(u32, word_atomic, value, .release);
}
pub fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);

    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "default source metadata accepts the canonical envelope" {
    for ([_]struct { input: []const u8, expected: []const u8 }{
        .{ .input = "{\"name\":\"alsa_input.test\"}", .expected = "alsa_input.test" },
        .{ .input = "{\"name\":\"microphone 🎙\"}", .expected = "microphone 🎙" },
        .{ .input = "{\"name\":\"\"}", .expected = "" },
    }) |case| {
        const parsed = try parseDefaultSource(case.input);

        try std.testing.expectEqualStrings(case.expected, parsed.get());
    }
}

test "default source metadata rejects noncanonical envelopes" {
    for ([_][]const u8{
        "{\"name\":\"}",
        " {\"name\":\"source\"}",
        "{\"name\":\"source\"} ",
        "{ \"name\": \"source\" }",
        "{\"name\":\"source\",\"other\":true}",
        "{\"name\":\"mic\\\\path\"}",
        "{\"na\\u006de\":\"source\"}",
        "{\"name\":\"source\\u0020name\"}",
        "{\"name\":\"source\x1f\"}",
    }) |input| {
        try std.testing.expectError(error.UnsupportedDefaultSourceMetadata, parseDefaultSource(input));
    }

    try std.testing.expectError(error.DefaultSourceNameInvalidUtf8, parseDefaultSource("{\"name\":\"\xff\"}"));

    const prefix = "{\"name\":\"";
    const suffix = "\"}";

    var oversized: [prefix.len + 257 + suffix.len]u8 = undefined;
    @memcpy(oversized[0..prefix.len], prefix);

    @memset(oversized[prefix.len..][0..257], 'a');
    @memcpy(oversized[oversized.len - suffix.len ..], suffix);
    try std.testing.expectError(error.DefaultSourceNameTooLong, parseDefaultSource(&oversized));
}

test "catalog text reuses full capacity on replacement and removal" {
    var pool: TextStorage = .{};

    var entries: [128]StoredObject = undefined;
    var object: Object = .{ .kind = .node, .id = 1 };

    try object.name.set(&(@as([256]u8, @splat('n'))));
    try object.description.set(&(@as([256]u8, @splat('d'))));

    for (&entries, 0..) |*entry, index| {
        object.id = @intCast(index);
        entry.* = try pool.replace(null, object);
    }

    try std.testing.expectEqual(@as(usize, 0), pool.free.count());

    for (0..4096) |iteration| {
        const slot = iteration % entries.len;

        object.id = @intCast(slot);
        object.name.bytes[0] = @truncate(iteration);
        entries[slot] = try pool.replace(entries[slot], object);

        const expanded = entries[slot].expand(&pool);

        try std.testing.expectEqualSlices(u8, object.name.get(), expanded.name.get());
    }

    try std.testing.expectError(error.CatalogTextFull, pool.replace(null, object));

    for (entries) |entry| {
        pool.releaseObject(entry);
    }

    try std.testing.expectEqual(@as(usize, 1024), pool.free.count());
}

test "eight-channel setup and link request batches fit the output queue" {
    var client: PipeWireClient = .{ .source = .default, .client_node_version = 5 };
    var source: Object = .{ .kind = .node, .id = 10, .is_source = true };

    try source.name.set(&(@as([256]u8, @splat('n'))));

    client.catalog[0] = try client.text.replace(null, source);

    for (0..8) |channel| {
        var port: Object = .{ .kind = .port, .id = @intCast(20 + channel), .parent = 10, .port = @intCast(channel), .is_output = true };

        try port.channel.set("AUX0");

        client.catalog[1 + channel] = try client.text.replace(null, port);
    }

    try client.createStream();

    const setup_bytes = client.connection.output_size;

    client.node_id = 50;

    for (0..8) |channel| {
        client.catalog[9 + channel] = try client.text.replace(null, .{ .kind = .port, .id = @intCast(60 + channel), .parent = 50, .port = @intCast(channel) });
    }

    // Include both batches together, without an intervening flush.
    try client.createLinks();
    try client.publishFormat(192000);
    std.debug.print("request bytes: setup={d}, setup+links+format={d}, capacity={d}\n", .{ setup_bytes, client.connection.output_size, client.connection.output.len });
    try std.testing.expect(client.links_created);
}

test "borrowed samples mix wrapped spans and silence and retain failing channel details" {
    var client: PipeWireClient = .{ .source = .default };
    var io: [8]u8 = @splat(0);
    var chunk: [16]u8 = @splat(0);

    store(u32, &chunk, 0, 12);
    store(u32, &chunk, 4, 16);
    store(i32, &chunk, 8, 4);

    client.ports[0].io = &io;
    client.ports[0].buffers[0].chunk = &chunk;

    var first = [_]f32{ 2, 4 };
    const second = [_]f32{ 6, 8 };

    var block: Block = undefined;
    block.client = &client;

    block.channels_count = 2;
    block.samples_count = 4;
    block.channels[0] = .{ .samples = .{ .first = &first, .second = &second } };
    block.channels[1] = .silence;

    for (0..4) |index| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(index + 1)), try block.sample(index));
    }

    first[1] = std.math.nan(f32);
    client.diagnostic.channel = 1;
    client.diagnostic.header_sequence = 99;

    try std.testing.expectError(error.InvalidBuffer, block.sample(1));
    try std.testing.expectEqual(@as(usize, 0), client.diagnostic.channel);
    try std.testing.expectEqual(@as(?usize, 1), client.diagnostic.invalid_sample_index);
    try std.testing.expectEqual(@as(u32, 12), client.diagnostic.chunk_offset);
    try std.testing.expectEqual(@as(u64, 0), client.diagnostic.header_sequence);
}
