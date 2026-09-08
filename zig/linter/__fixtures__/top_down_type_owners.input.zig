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
