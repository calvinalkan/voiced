comptime {
    // Direct `zig test` does not evaluate build.zig or link production's LLVM
    // inference object. Compile the same exports into the test executable so
    // the facade's `extern` declarations resolve for every source test. This
    // validates behavior through the private ABI, not the physical object
    // boundary. A future process-level service harness must build and launch a
    // production-shaped binary to cover that boundary and full lifecycle.
    _ = @import("inference/root.object.zig");
}

test {
    _ = @import("audio_exchange.zig");
    _ = @import("capture/pipewire_client.zig");
    _ = @import("capture/pipewire_wire.zig");
    _ = @import("clipboard/x11.zig");
    _ = @import("clipboard/x11_wire.zig");
    _ = @import("inference/attention.zig");
    _ = @import("inference/Runtime.object.zig");
    _ = @import("inference/linear.zig");
    _ = @import("inference/log_mel.zig");
    _ = @import("inference/root_test.zig");
    _ = @import("logging.zig");
    _ = @import("packed_model/root_test.zig");
}
