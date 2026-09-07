//! Native DSP capture client. The control connection owns graph discovery,
//! source identity, mappings and link lifetime. process() consumes one graph
//! cycle without allocating, formatting diagnostics or waiting on descriptors.
const std = @import("std");
const linux = std.os.linux;
const protocol = @import("pipewire_protocol.zig");
const endian = @import("builtin").cpu.arch.endian();
const invalid = std.math.maxInt(u32);
const channels_max = 8;
const buffers_max = 16;
pub const samples_max = 16384;
pub const Error = protocol.Error || error{ CatalogFull, SourceNotFound, SourceAmbiguous, SourceDisconnected, SourceChanged, SourcePortsUnavailable, UnsupportedVersion, UnsupportedFormat, UnsupportedIO, InvalidBuffer, CorruptedBuffer, TimelineDiscontinuity, GraphError, WakeFailed, TooManyChannels, TooManyBuffers, TooManyMemories, TooManyPeers };
pub const Source = union(enum) { default, node_name: []const u8, device_serial: []const u8 };
pub const Text = struct {
    bytes: [256]u8 = @splat(0),
    size: u16 = 0,
    pub fn get(self: *const Text) []const u8 {
        return self.bytes[0..self.size];
    }
    fn set(self: *Text, value: []const u8) Error!void {
        if (value.len > self.bytes.len) return error.MessageTooLarge;
        @memcpy(self.bytes[0..value.len], value);
        self.size = @intCast(value.len);
    }
};
pub const Object = struct {
    kind: enum { node, device, port },
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
pub const Block = struct { samples: []const f32, rate: u32, position: u64, header_present: bool, silence: bool };

pub const Client = struct {
    connection: protocol.Connection = .{},
    source: Source,
    catalog: [128]?Object = @splat(null),
    selected: ?Identity = null,
    node_id: u32 = invalid,
    // Proxy IDs occupy the server's ordered object map. Allocate densely;
    // reserving an arbitrary ID for the stream breaks later binds. Core,
    // Client and Registry already occupy 0, 1 and 2.
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
    samples: [samples_max]f32 = undefined,
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

    pub fn init(self: *Client, path: []const u8) Error!void {
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

    pub fn deinit(self: *Client) void {
        self.connection.deinit();
        if (self.wake_fd >= 0) _ = linux.close(self.wake_fd);
        if (self.completion_fd >= 0) _ = linux.close(self.completion_fd);
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
    pub fn dispatch(self: *Client) Error!bool {
        const message = try self.connection.receive() orelse return false;
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
                    if (id == 0 and seq == 100) try self.ints(0, 2, &.{ 0, 101 });
                    if (id == 0 and seq == 101 and !self.stream_created) try self.createStream();
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
                    if (proxy == self.node_proxy) self.node_id = global_id;
                    for (self.ports[0..self.ports_count]) |*port| if (port.link_proxy == proxy) {
                        port.link_global = global_id;
                    };
                },
                6 => {
                    const id = try r.int();
                    const kind = try r.id();
                    const fd = try self.connection.descriptor(try r.long(18));
                    const flags = try r.int();
                    if (id == invalid or self.registeredMemory(id) != null) return error.InvalidMemory;
                    const entry = for (&self.memories) |*entry| {
                        if (entry.* == null) break entry;
                    } else return error.TooManyMemories;
                    if (kind != 2 or flags & 1 == 0) return error.InvalidMemory;
                    var stat: linux.Statx = undefined;
                    self.connection.errno = linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &stat));
                    if (self.connection.errno != .SUCCESS or !stat.mask.SIZE or stat.size == 0 or stat.size > 64 * 1024 * 1024) return error.InvalidMemory;
                    const address = linux.mmap(null, @intCast(stat.size), .{ .READ = true, .WRITE = flags & 2 != 0 }, .{ .TYPE = .SHARED }, fd, 0);
                    self.connection.errno = linux.errno(address);
                    if (self.connection.errno != .SUCCESS) return error.MappingFailed;
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
                        if (id == selected.node.id or (selected.device != null and id == selected.device.?.id)) return error.SourceDisconnected;
                    }
                    for (self.ports[0..self.ports_count]) |port| if (id == port.source_global or id == port.link_global) return error.SourceDisconnected;
                    for (&self.catalog) |*entry| if (entry.* != null and entry.*.?.id == id) {
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
                    if (key.kind == 8 and std.mem.eql(u8, std.mem.trimEnd(u8, key.body, "\x00"), "default.audio.source") and value.kind == 8 and value.body.len > 0) {
                        const parsed = std.json.parseFromSlice(struct { name: []const u8 }, std.heap.page_allocator, value.body[0 .. value.body.len - 1], .{ .ignore_unknown_fields = true }) catch return error.InvalidMessage;
                        defer parsed.deinit();
                        try self.default_source.set(parsed.value.name);
                    }
                    return true;
                }
                for (&self.catalog) |*entry| {
                    if (entry.* == null or entry.*.?.proxy != message.object) continue;
                    if (message.opcode != 0) return error.InvalidMessage;
                    var object = &entry.*.?;
                    if (try r.int() != object.id) return error.InvalidMessage;
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
                    if (self.selected) |selected| {
                        if (object.id == selected.node.id and (object.serial != selected.node.serial or object.parent != selected.node.parent or !std.mem.eql(u8, object.name.get(), selected.node.name.get()))) return error.SourceChanged;
                        if (selected.device) |device| if (object.id == device.id and (object.serial != device.serial or !std.mem.eql(u8, object.device_serial.get(), device.device_serial.get()))) return error.SourceChanged;
                    }
                    break;
                }
                for (self.ports[0..self.ports_count]) |port| if (message.object == port.link_proxy and message.opcode == 0) {
                    const global_id = try r.int();
                    _ = global_id;
                    if (try r.int() != self.selected.?.node.id or try r.int() != port.source_global or try r.int() != self.node_id) return error.SourceChanged;
                    _ = try r.int();
                    _ = try r.long(5);
                    const state: i32 = @bitCast(try r.int());
                    if (state == -2 or state == -1) return error.SourceDisconnected;
                };
            },
        }
        if (self.stream_created and !self.links_created) try self.createLinks();
        return true;
    }

    pub fn process(self: *Client) Error!?Block {
        var count: u64 = 0;
        const result = linux.read(self.wake_fd, std.mem.asBytes(&count).ptr, 8);
        if (linux.errno(result) == .AGAIN or linux.errno(result) == .INTR) return null;
        if (result != 8 or count == 0) {
            self.connection.errno = linux.errno(result);
            return error.WakeFailed;
        }
        self.diagnostic = .{ .wake_count = count };
        const activation = self.activation orelse return error.InvalidMemory;
        atomicStore(activation, 0, 2);
        store(u64, activation, 40, now());
        // Completion must run even when validation rejects a cycle: otherwise
        // the graph's driver waits for a client that already stopped reading.
        defer {
            for (self.ports[0..self.ports_count]) |port| if (port.io) |io| atomicStore(io, 0, 1);
            atomicStore(activation, 8, 1);
            store(u64, activation, 48, now());
            atomicStore(activation, 0, 3);
            for (self.peers) |entry| if (entry) |peer| {
                const pending: *i32 = @ptrCast(@alignCast(peer.activation[16..].ptr));
                if (@atomicRmw(i32, pending, .Sub, 1, .acq_rel) == 1) {
                    store(u64, peer.activation, 32, now());
                    atomicStore(peer.activation, 0, 1);
                    const one: u64 = 1;
                    _ = linux.write(peer.fd, std.mem.asBytes(&one).ptr, 8);
                }
            };
        }
        if (!self.running) return null;
        const clock = self.position orelse return error.InvalidMemory;
        const rate_num = load(u32, clock, 80);
        const rate = load(u32, clock, 84);
        const position = load(u64, clock, 88);
        const duration = load(u64, clock, 96);
        const clock_id = load(u32, clock, 4);
        self.diagnostic.graph_rate_num = rate_num;
        self.diagnostic.graph_rate_hz = rate;
        self.diagnostic.graph_position = position;
        self.diagnostic.graph_duration = duration;
        if (rate_num != 1 or rate < 8000 or rate > 192000 or duration == 0 or duration > self.samples.len) return error.UnsupportedFormat;
        if (self.previous) |previous| {
            if (count != 1 or (previous.clock == clock_id and previous.rate == rate and position != previous.position +% previous.duration)) return error.TimelineDiscontinuity;
        }
        var samples_count: ?usize = null;
        var header_present = false;
        var all_silent = true;
        for (self.ports[0..self.ports_count], 0..) |port, channel| {
            self.diagnostic.channel = channel;
            // Startup can wake us before format/IO/buffers are ready. Once
            // samples have been accepted, losing any of them is a discontinuity;
            // silently waiting would hide missing audio behind live callbacks.
            if (!port.negotiated) {
                if (self.previous != null) return error.UnsupportedFormat;
                return null;
            }
            const io = port.io orelse {
                if (self.previous != null) return error.TimelineDiscontinuity;
                return null;
            };
            const status = atomicLoad(io, 0);
            if (status != 2) {
                if (self.previous != null) return error.TimelineDiscontinuity;
                return null;
            }
            const id = load(u32, io, 4);
            self.diagnostic.buffer_id = id;
            if (id >= port.buffers_count) return error.InvalidBuffer;
            const buffer = port.buffers[id];
            const chunk = buffer.chunk orelse return error.InvalidBuffer;
            const data = buffer.data orelse return error.InvalidBuffer;
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
            if (size > data.len or size % 4 != 0 or offset % 4 != 0 or (stride != 0 and stride != 4) or duration > data.len / 4) return error.InvalidBuffer;
            if (flags & 1 != 0) return error.CorruptedBuffer;
            if (flags & ~@as(u32, 3) != 0) return error.InvalidBuffer;
            var silent = flags & 2 != 0;
            if (buffer.header) |header| {
                header_present = true;
                const header_flags = load(u32, header, 0);
                self.diagnostic.header_flags = header_flags;
                self.diagnostic.header_sequence = load(u64, header, 24);
                self.diagnostic.header_pts_ns = load(i64, header, 8);
                if (header_flags & 2 != 0) return error.CorruptedBuffer;
                if (header_flags & 1 != 0 and self.previous != null) return error.TimelineDiscontinuity;
                if (header_flags & ~@as(u32, 0x11) != 0) return error.InvalidBuffer;
                silent = silent or header_flags & 16 != 0;
            }
            // DSP cycle length comes from the graph clock. Older sources
            // leave chunk.size at its initial quantum when the graph quantum
            // changes; bound this cycle against the mapped capacity instead.
            const n: usize = @intCast(duration);
            if (samples_count != null and samples_count.? != n) return error.InvalidBuffer;
            samples_count = n;
            all_silent = all_silent and silent;
            for (self.samples[0..n], 0..) |*dest, index| {
                const value = if (silent) 0 else load(f32, data, (offset + index * 4) % data.len);
                if (!std.math.isFinite(value)) {
                    self.diagnostic.invalid_sample_index = index;
                    return error.InvalidBuffer;
                }
                if (channel == 0) dest.* = value else dest.* += value;
            }
        }
        const n = samples_count orelse return null;
        if (self.ports_count > 1) for (self.samples[0..n]) |*value| {
            value.* /= @floatFromInt(self.ports_count);
        };
        self.previous = .{ .position = position, .duration = duration, .rate = rate, .clock = clock_id };
        return .{ .samples = self.samples[0..n], .rate = rate, .position = position, .header_present = header_present, .silence = all_silent };
    }

    /// Publish the current graph format for stream observers. Call only when
    /// the negotiated rate changes, outside process()'s buffer ownership.
    pub fn publishFormat(self: *Client, rate: u32) Error!void {
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
                if (std.mem.eql(u8, name, known)) break @intCast(id);
            } else if (std.mem.startsWith(u8, name, "AUX")) auxiliary: {
                const id = std.fmt.parseInt(u8, name[3..], 10) catch break :auxiliary 0;
                break :auxiliary if (id < 64) @as(u32, 0x1000) + id else 0;
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
        for (0..self.ports_count) |_| try w.word(@bitCast(@as(f32, 1)));
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

    fn global(self: *Client, r: *protocol.Reader) Error!void {
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
                if (std.mem.eql(u8, key, "factory.name")) factory = value;
                if (std.mem.eql(u8, key, "factory.type.version")) factory_version = std.fmt.parseInt(u32, value, 10) catch return error.InvalidMessage;
            }
            if (std.mem.eql(u8, factory, "client-node")) self.client_node_version = factory_version;
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
        var object: Object = .{ .id = id, .kind = if (std.mem.eql(u8, interface, "PipeWire:Interface:Node")) .node else if (std.mem.eql(u8, interface, "PipeWire:Interface:Device")) .device else if (std.mem.eql(u8, interface, "PipeWire:Interface:Port")) .port else return };
        try properties(&object, props);
        if (object.kind == .node or object.kind == .device) {
            object.proxy = self.proxy_next;
            self.proxy_next += 1;
            try self.bind(id, interface, @min(version, 3), object.proxy);
        }
        for (&self.catalog) |*entry| if (entry.* == null) {
            entry.* = object;
            return;
        };
        return error.CatalogFull;
    }

    fn createStream(self: *Client) Error!void {
        if (self.client_node_version < 4) return error.UnsupportedVersion;
        var selected: ?Object = null;
        var matches: usize = 0;
        for (self.catalog) |entry| {
            const object = entry orelse continue;
            if (object.kind != .node or !object.is_source) continue;
            const device = self.find(object.parent);
            const matches_source = switch (self.source) {
                .default => self.default_source.size == 0 or std.mem.eql(u8, object.name.get(), self.default_source.get()),
                .node_name => |name| std.mem.eql(u8, name, object.name.get()),
                .device_serial => |serial| device != null and std.mem.eql(u8, serial, device.?.device_serial.get()),
            };
            if (!matches_source) continue;
            matches += 1;
            if (selected == null or (self.source == .default and object.priority > selected.?.priority)) selected = object;
        }
        if (selected == null) return error.SourceNotFound;
        if (matches > 1 and self.source != .default) return error.SourceAmbiguous;
        const node = selected.?;
        self.selected = .{ .node = node, .device = self.find(node.parent) };
        for (self.catalog) |entry| {
            const object = entry orelse continue;
            if (object.kind != .port or object.parent != node.id or !object.is_output) continue;
            if (self.ports_count == self.ports.len) return error.TooManyChannels;
            self.ports[self.ports_count] = .{ .source_global = object.id, .source_port = object.port, .channel = object.channel };
            self.ports_count += 1;
        }
        if (self.ports_count == 0) return error.SourcePortsUnavailable;
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
        for (0..self.ports_count) |index| try self.portUpdate(@intCast(index), null);
        self.stream_created = true;
    }

    fn createLinks(self: *Client) Error!void {
        if (self.node_id == invalid) return;
        for (self.ports[0..self.ports_count], 0..) |_, index| {
            var found = false;
            for (self.catalog) |entry| if (entry) |object| {
                if (object.kind == .port and object.parent == self.node_id and object.port == index) {
                    found = true;
                    break;
                }
            };
            if (!found) return;
        }
        for (self.ports[0..self.ports_count], 0..) |*port, index| {
            var ids: [4][24]u8 = undefined;
            const source_node = std.fmt.bufPrint(&ids[0], "{d}", .{self.selected.?.node.id}) catch unreachable;
            const source_port = std.fmt.bufPrint(&ids[1], "{d}", .{port.source_global}) catch unreachable;
            const target_node = std.fmt.bufPrint(&ids[2], "{d}", .{self.node_id}) catch unreachable;
            const target_port = std.fmt.bufPrint(&ids[3], "{d}", .{index}) catch unreachable;
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

    fn nodeEvent(self: *Client, opcode: u8, r: *protocol.Reader) Error!void {
        switch (opcode) {
            0 => {
                if (self.activation != null) return error.InvalidMessage;
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
                if (id != 2) return error.UnsupportedFormat;
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
                if (id == 7) self.position = if (mem == invalid) null else try self.region(mem, offset, size, 104, false) else if (id != 3) return error.UnsupportedIO;
            },
            4 => {
                const command = try r.next();
                if (command.kind != 15 or command.body.len < 8 or load(u32, command.body, 0) != 0x30002) return error.InvalidMessage;
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
                if (direction != 0 or port >= self.ports_count or id != 4) return error.UnsupportedFormat;
                if (format.kind == 1) {
                    self.ports[port].negotiated = false;
                    try self.portUpdate(port, null);
                    return;
                }
                if (format.kind != 15 or format.body.len < 8 or load(u32, format.body, 0) != 0x40003 or load(u32, format.body, 4) != 4) return error.UnsupportedFormat;
                var props: protocol.Reader = .{ .bytes = format.body[8..] };
                var fields: u3 = 0;
                while (props.bytes.len > 0) {
                    if (props.bytes.len < 8) return error.InvalidMessage;
                    const key = load(u32, props.bytes, 0);
                    props.bytes = props.bytes[8..];
                    const value = try props.next();
                    switch (key) {
                        1 => {
                            if (try value.scalar(3) != 1) return error.UnsupportedFormat;
                            fields |= 1;
                        },
                        2 => {
                            if (try value.scalar(3) != 2) return error.UnsupportedFormat;
                            fields |= 2;
                        },
                        0x10001 => {
                            if (try value.scalar(3) != 0x206) return error.UnsupportedFormat;
                            fields |= 4;
                        },
                        else => return error.UnsupportedFormat,
                    }
                }
                if (fields != 7) return error.UnsupportedFormat;
                self.ports[port].negotiated = true;
                try self.portUpdate(port, format.encoded);
            },
            8 => try self.useBuffers(r),
            9 => {
                if (try r.int() != 0) return error.UnsupportedIO;
                const port = try r.int();
                const mix = try r.int();
                const id = try r.id();
                const mem = try r.int();
                const offset = try r.int();
                const size = try r.int();
                if (port >= self.ports_count or mix != 0 or id != 1) return error.UnsupportedIO;
                self.ports[port].io = if (mem == invalid) null else try self.region(mem, offset, size, 8, true);
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
                if (try r.int() != 0) return error.UnsupportedIO;
                const port = try r.int();
                const mix = try r.int();
                const peer = try r.int();
                if (port >= self.ports_count or mix != 0) return error.UnsupportedIO;
                if (peer != self.ports[port].source_global and peer != invalid) return error.SourceChanged;
            },
            else => return error.UnsupportedIO,
        }
    }

    fn useBuffers(self: *Client, r: *protocol.Reader) Error!void {
        if (try r.int() != 0) return error.InvalidBuffer;
        const index = try r.int();
        const mix = try r.int();
        const flags = try r.int();
        const count = try r.int();
        if (index >= self.ports_count or mix != 0 or flags != 0) return error.InvalidBuffer;
        if (count > buffers_max) return error.TooManyBuffers;
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
                if (meta_size > region_bytes.len - cursor) return error.InvalidBuffer;
                if (kind == 1) {
                    if (buffer.header != null or meta_size < 32) return error.InvalidBuffer;
                    buffer.header = region_bytes[cursor..][0..meta_size];
                }
                const padded = std.mem.alignForward(usize, meta_size, 8);
                if (padded > region_bytes.len - cursor) return error.InvalidBuffer;
                cursor += padded;
            }
            if (try r.int() != 1 or region_bytes.len - cursor < 16) return error.InvalidBuffer;
            buffer.chunk = region_bytes[cursor..][0..16];
            const kind = try r.id();
            const data_id = try r.int();
            const data_flags = try r.int();
            const mapoffset = try r.int();
            const capacity = try r.int();
            if (capacity == 0 or capacity % 4 != 0 or capacity > samples_max * 4 or data_flags & 1 == 0) return error.InvalidBuffer;
            if (kind == 1) {
                if (data_id > region_bytes.len or capacity > region_bytes.len - data_id) return error.InvalidBuffer;
                buffer.data = region_bytes[data_id..][0..capacity];
            } else if (kind == 4) {
                buffer.data = try self.region(data_id, mapoffset, capacity, capacity, false);
            } else return error.InvalidBuffer;
        }
        port.buffers_count = count;
    }

    fn portUpdate(self: *Client, index: u32, format: ?[]const u8) Error!void {
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
        if (format) |bytes| try w.raw(bytes);
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
        try w.string(std.fmt.bufPrint(&name, "input_{d}", .{index}) catch unreachable);
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

    fn eventDescriptor(self: *Client, index: i64) Error!linux.fd_t {
        const fd = try self.connection.duplicateDescriptor(index);
        errdefer _ = linux.close(fd);
        const flags = linux.fcntl(fd, linux.F.GETFL, 0);
        self.connection.errno = linux.errno(flags);
        if (self.connection.errno != .SUCCESS) return error.DescriptorFailed;
        self.connection.errno = linux.errno(linux.fcntl(fd, linux.F.SETFL, flags | @as(u32, @bitCast(linux.O{ .NONBLOCK = true }))));
        if (self.connection.errno != .SUCCESS) return error.DescriptorFailed;
        return fd;
    }

    fn region(self: *Client, id: u32, offset: u32, size: u32, minimum: usize, writable: bool) Error![]u8 {
        const memory = self.registeredMemory(id) orelse return error.InvalidMemory;
        if (offset % 8 != 0 or size < minimum or offset > memory.bytes.len or size > memory.bytes.len - offset or (writable and !memory.writable)) return error.InvalidMemory;
        return memory.bytes[offset..][0..size];
    }
    fn registeredMemory(self: *Client, id: u32) ?*Memory {
        for (&self.memories) |*entry| if (entry.*) |*memory| {
            if (memory.id != null and memory.id.? == id) return memory;
        };
        return null;
    }
    fn releaseUnusedMemory(self: *Client) void {
        for (&self.memories) |*entry| if (entry.*) |memory| {
            if (memory.id == null and !self.references(memory.bytes)) {
                std.posix.munmap(memory.bytes);
                entry.* = null;
            }
        };
    }
    fn references(self: *const Client, bytes: []u8) bool {
        if (inside(self.activation, bytes) or inside(self.position, bytes)) return true;
        for (self.peers) |peer| if (peer != null and inside(peer.?.activation, bytes)) return true;
        for (self.ports[0..self.ports_count]) |port| {
            if (inside(port.io, bytes)) return true;
            for (port.buffers[0..port.buffers_count]) |buffer| if (inside(buffer.header, bytes) or inside(buffer.chunk, bytes) or inside(buffer.data, bytes)) return true;
        }
        return false;
    }
    fn find(self: *const Client, id: u32) ?Object {
        if (id == invalid) return null;
        for (self.catalog) |entry| if (entry != null and entry.?.id == id) return entry;
        return null;
    }
    fn bind(self: *Client, id: u32, interface: []const u8, version: u32, proxy: u32) Error!void {
        var w: protocol.Writer = .{};
        _ = try w.begin(14);
        try w.int(id);
        try w.string(interface);
        try w.int(version);
        try w.int(proxy);
        try w.finish(0);
        try self.connection.send(2, 1, w.data());
    }
    fn ints(self: *Client, object: u32, opcode: u8, values: []const u32) Error!void {
        var w: protocol.Writer = .{};
        _ = try w.begin(14);
        for (values) |value| try w.int(value);
        try w.finish(0);
        try self.connection.send(object, opcode, w.data());
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
            object.serial = std.fmt.parseInt(u64, value, 10) catch return error.InvalidMessage;
        } else if (std.mem.eql(u8, key, "node.name")) {
            try object.name.set(value);
        } else if (std.mem.eql(u8, key, "node.description") or std.mem.eql(u8, key, "device.description")) {
            try object.description.set(value);
        } else if (std.mem.eql(u8, key, "device.serial")) {
            try object.device_serial.set(value);
        } else if (std.mem.eql(u8, key, "audio.channel")) {
            try object.channel.set(value);
        } else if (std.mem.eql(u8, key, "device.id") or (object.kind == .port and std.mem.eql(u8, key, "node.id"))) {
            object.parent = std.fmt.parseInt(u32, value, 10) catch return error.InvalidMessage;
        } else if (std.mem.eql(u8, key, "port.id")) {
            object.port = std.fmt.parseInt(u32, value, 10) catch return error.InvalidMessage;
        } else if (std.mem.eql(u8, key, "port.direction")) {
            object.is_output = std.mem.eql(u8, value, "out");
        } else if (std.mem.eql(u8, key, "media.class")) {
            object.is_source = std.mem.eql(u8, value, "Audio/Source") or std.mem.eql(u8, value, "Audio/Source/Virtual");
        } else if (std.mem.eql(u8, key, "priority.session")) {
            object.priority = std.fmt.parseInt(i32, value, 10) catch return error.InvalidMessage;
        }
    }
    try r.end();
}
fn bounded(value: usize, maximum: usize) Error!usize {
    if (value > maximum) return error.InvalidMessage;
    return value;
}
fn inside(part: ?[]const u8, bytes: []const u8) bool {
    const region = part orelse return false;
    return @intFromPtr(region.ptr) >= @intFromPtr(bytes.ptr) and @intFromPtr(region.ptr) < @intFromPtr(bytes.ptr) + bytes.len;
}
fn load(comptime T: type, bytes: []const u8, offset: usize) T {
    return std.mem.bytesToValue(T, bytes[offset..][0..@sizeOf(T)]);
}
fn store(comptime T: type, bytes: []u8, offset: usize, value: T) void {
    @memcpy(bytes[offset..][0..@sizeOf(T)], std.mem.asBytes(&value));
}
fn atomicLoad(bytes: []u8, offset: usize) u32 {
    const ptr: *u32 = @ptrCast(@alignCast(bytes[offset..].ptr));
    return @atomicLoad(u32, ptr, .acquire);
}
fn atomicStore(bytes: []u8, offset: usize, value: u32) void {
    const ptr: *u32 = @ptrCast(@alignCast(bytes[offset..].ptr));
    @atomicStore(u32, ptr, value, .release);
}
pub fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
