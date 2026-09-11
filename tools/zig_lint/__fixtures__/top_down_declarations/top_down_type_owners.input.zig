const SyntaxNode = struct {
    children: []const SyntaxNode,
};

pub const SyntaxTree = struct {
    root: SyntaxNode,
};

pub const Graph = struct {
    vertices: []const Vertex,
};

const Vertex = struct {};

const Shared = struct {};

pub const LeftOwner = struct {
    shared: Shared,
};

pub const RightOwner = struct {
    shared: Shared,
};

const PrivatePart = struct {};

const PrivateOwner = struct {
    part: PrivatePart,
};

const CycleLeft = struct {
    right: *CycleRight,
};

const CycleRight = struct {
    left: *CycleLeft,
};

const TransitiveLeaf = struct {};

const PrivateMiddle = struct {
    leaf: TransitiveLeaf,
};

pub const PublicRoot = struct {
    middle: PrivateMiddle,
};

const WireHeader = extern struct {
    size: u32,
};

pub const Packet = struct {
    header: WireHeader,
};

const PackedHeader = packed struct {
    size: u32,
};

pub const PackedPacket = struct {
    header: PackedHeader,
};

const IndependentlyUsedPart = struct {};

fn consumeIndependentPart(part: IndependentlyUsedPart) void {
    _ = part;
}

pub const IndependentOwner = struct {
    part: IndependentlyUsedPart,
};

const Namespace = struct {
    const Entry = struct {};

    pub const Table = struct {
        entries: []const Entry,
    };

    pub const ValidOwner = struct {
        part: ValidPart,
    };

    const ValidPart = struct {};
};

const ComputedLength = enum(u8) {
    four = 4,
};

pub const ComputedPacket = struct {
    bytes: [@intFromEnum(ComputedLength.four)]u8,
};
