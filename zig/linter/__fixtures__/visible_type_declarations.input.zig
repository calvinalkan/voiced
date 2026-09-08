const First = struct {};
const Second = enum { value };

const Third = union { value: u8 };
fn followsType() void {}

fn precedesType() void {}
const Fourth = error{Failed};

const CommentedFirst = struct {};
// This comment remains attached to CommentedSecond.
const CommentedSecond = struct {};

const ValidFirst = struct {};

const ValidSecond = struct {};

fn betweenContainers() void {}

const Namespace = struct {
    const NestedFirst = struct {};
    const NestedSecond = enum { value };

    const NestedValidFirst = struct {};

    const NestedValidSecond = struct {};
};

const Callback = fn () void;
const CallbackContext = struct {};
