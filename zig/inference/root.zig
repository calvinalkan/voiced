//! Public API for the allocation-free pure Zig Whisper inference runtime.
//! Models own or borrow immutable packed weights; runtimes bind fixed-capacity
//! caller memory; each transcription selects its logical trailing-silence tail.

const model = @import("model.zig");
const runtime = @import("runtime.zig");

pub const Model = model.Model;
pub const ModelKind = model.ModelKind;
pub const ModelLoadError = model.ModelLoadError;
pub const ModelSpecification = model.ModelSpecification;
pub const packed_model_image_alignment = model.packed_model_image_alignment;
pub const packed_model_image_format_version = model.packed_model_image_format_version;
pub const packed_model_cache_version = model.packed_model_cache_version;

pub const EncoderTrailingPadding = runtime.EncoderTrailingPadding;
pub const Policy = runtime.Policy;
pub const Runtime = runtime.Runtime;
pub const RuntimeError = runtime.RuntimeError;
pub const Timings = runtime.Timings;
pub const Token = runtime.Token;
pub const TranscribeOptions = runtime.TranscribeOptions;
pub const Transcription = runtime.Transcription;
pub const runtime_memory_alignment = runtime.runtime_memory_alignment;

pub const suppressed_tokens = runtime.suppressed_tokens;
pub const suppressed_first_tokens = runtime.suppressed_first_tokens;
