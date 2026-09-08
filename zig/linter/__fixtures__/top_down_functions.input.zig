fn loadImpl() void {}

pub fn load() void {
    loadImpl();
}

pub fn save() void {
    saveImpl();
}

fn saveImpl() void {
    performSave();
}

fn sharedImpl() void {}

pub fn firstSharedCaller() void {
    sharedImpl();
}

pub fn secondSharedCaller() void {
    sharedImpl();
}

pub fn publicBoundary() void {}

pub fn publicBoundaryCaller() void {
    publicBoundary();
}

fn recursiveBoundary() void {
    recursiveBoundary();
}

const Service = struct {
    fn startImpl() void {}

    pub fn start() void {
        startImpl();
    }
};

const Worker = struct {
    fn stopImpl() void {}

    pub fn stop() void {
        Self.stopImpl();
    }
};

const AmbiguousService = struct {
    fn refreshImpl() void {}

    pub fn refresh(receiver: anytype) void {
        receiver.refreshImpl();
    }
};

fn callbackImpl() void {}

pub fn registerCallback() void {
    register(callbackImpl);
}

fn performSave() void {}
fn register(_: anytype) void {}
