//! Public inference API backed by a separately compiled LLVM object.
//! Models own immutable storage; runtimes borrow a model and a shared pool.

const abi = @import("abi.zig");

comptime {
    // The linked object can retain panic references even for model-only users.
    _ = abi;
}

pub const Model = @import("Model.zig");
pub const Runtime = @import("Runtime.zig").Runtime;
pub const TranscriptionResult = @import("TranscriptionResult.zig").TranscriptionResult;
pub const WorkerPool = @import("WorkerPool.zig");
pub const VnniWeight = @import("vnni_weight.zig");
pub const EncoderTrailingPadding = Runtime.EncoderPadding;
