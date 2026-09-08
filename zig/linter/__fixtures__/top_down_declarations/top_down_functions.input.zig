fn loadImpl() void {}

pub fn load() void {
    loadImpl();
}

fn repeatImpl() void {}

pub fn repeat() void {
    repeatImpl();
    repeatImpl();
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

fn even(value: u8) bool {
    return value == 0 or odd(value - 1);
}

fn odd(value: u8) bool {
    return value != 0 and even(value - 1);
}

fn nestedHelper() void {}

pub fn ownsNestedDeclaration() void {
    const Local = struct {
        fn nestedHelper() void {}
    };

    _ = Local;
}

fn deepLeaf() void {}

fn deepPhase() void {
    deepLeaf();
}

pub fn deepRoot() void {
    deepPhase();
}

const PrototypeShortcut = struct {
    fn helper() void;

    pub fn entry() void {
        helper();
    }
};

const ExternShortcut = struct {
    fn helper() void {}

    extern fn entry() void;
};

const ShortExactName = struct {
    fn h() void {}

    pub fn entry() void {
        h();
    }
};

const ShortPrefixName = struct {
    fn h() void {}

    pub fn entry() void {
        house();
    }
};

const LongPrefixName = struct {
    fn helper() void {}

    pub fn entry() void {
        helperOther();
    }
};

const MixedPrefixName = struct {
    fn eMTarget() void {}

    pub fn entry() void {
        fnOther();
        eMTarget();
    }
};

const QuotedCall = struct {
    fn @"quoted name"() void {}

    pub fn entry() void {
        @"quoted name"();
    }
};

const CallbackReference = struct {
    fn helper() void {}

    pub fn entry() void {
        register(helper);
        helper();
    }
};

const QualifiedReference = struct {
    fn helper() void {}

    pub fn entry() void {
        obj.helper();
        helper();
    }
};

const EntryReference = struct {
    fn helper() void {}

    pub fn entry() void {
        helper();
    }

    const callback = entry;
};

const CallerCycle = struct {
    fn helper() void {}

    pub fn entry() void {
        helper();
        entry();
    }
};
