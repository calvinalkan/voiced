pub const LoadOptions = struct {
    source: LoadSource,
};

const LoadSource = union(enum) {
    path: []const u8,
    stdin,
};

const UnrelatedLoadState = struct {
    pub fn reset() void {}
};

pub fn load(options: LoadOptions) void {
    _ = options;
}

pub const SaveOptions = struct {};

fn refreshCache() void {}

pub fn save(options: SaveOptions) void {
    _ = options;
}

pub const FirstPeerOptions = struct {};

pub const SecondPeerOptions = struct {};

pub fn firstPeer(options: FirstPeerOptions) void {
    _ = options;
}

pub fn secondPeer(options: SecondPeerOptions) void {
    _ = options;
}

pub const DecodeOptions = struct {
    format: DecodeFormat,
};

pub fn decode(options: DecodeOptions) void {
    _ = options;
}

const DecodeFormat = enum { wave, raw };

pub const OpenOptions = struct {
    source: OpenSource,
};

const OpenSource = union(enum) {
    file: OpenFile,
    stdin,
};

const OpenFile = struct {
    path: []const u8,
};

pub const OpenResult = union(enum) {
    opened,
    failed: OpenFailure,
};

const OpenFailure = struct {
    message: []const u8,
};

pub fn open(options: OpenOptions) OpenResult {
    _ = options;

    return .opened;
}

const SharedPath = struct {};

const IndependentState = struct {};

pub fn readShared(path: SharedPath) void {
    _ = path;
}

pub fn writeShared(path: SharedPath) void {
    _ = path;
}

pub const Packet = struct {
    header: PacketHeader,
};

const PacketHeader = extern struct {
    size: u32,
};

const packet_header_size = @sizeOf(PacketHeader);

comptime {
    _ = packet_header_size;
}

pub fn decodePacket(packet: Packet) void {
    _ = packet;
}

pub const PublicOptions = struct {
    detail: PublicDetail,
};

pub const PublicDetail = struct {};

pub fn usePublicContract(options: PublicOptions) void {
    _ = options;
}

pub const DiamondOptions = struct {
    left: DiamondLeft,
    right: DiamondRight,
};

const DiamondLeft = struct {
    shared: DiamondShared,
};

const DiamondRight = struct {
    shared: DiamondShared,
};

const DiamondShared = struct {};

pub fn useDiamond(options: DiamondOptions) void {
    _ = options;
}

pub const Model = struct {
    pub fn parse() Model {
        return .{};
    }
};

const ModelPeer = struct {
    pub fn reset() void {}
};

pub fn useModel(model: Model) void {
    _ = model;
}

pub const ExecuteOptions = struct {
    detail: ExecuteDetail,
};

pub fn execute(options: ExecuteOptions) void {
    _ = options;
}

const ExecuteDetail = struct {};

fn inspectExecuteDetail(detail: ExecuteDetail) void {
    _ = detail;
}

const Stateful = struct {
    value: u8,

    pub const Options = struct {};

    const Peer = struct {
        pub fn reset() void {}
    };

    pub fn run(self: *Stateful, options: Options) void {
        _ = self;
        _ = options;
    }
};

const Namespace = struct {
    pub const RunOptions = struct {
        detail: RunDetail,
    };

    const RunDetail = struct {};

    const UnrelatedRunState = struct {
        pub fn reset() void {}
    };

    pub fn run(options: RunOptions) void {
        _ = options;
    }
};

pub const BackendOptions = struct {
    backend: Backend,
};

pub fn runBackend(options: BackendOptions) void {
    _ = options;
}

const Backend = struct {
    pub fn reset() void {}
};
