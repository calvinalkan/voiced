const std = @import("std");

/// Returns a `{f}` formatter with the same decimal output as `{d}` for f32/f64.
/// `precision` counts fractional digits; null preserves full precision. No width
/// or alignment is applied. Output exceeding Writer's float-buffer capacity
/// uses its `(float)` fallback. Writer errors still propagate.
pub fn fmt(value: anytype, precision: ?usize) Format {
    const T = @TypeOf(value);
    if (T != f32 and T != f64) @compileError("decimal.fmt accepts f32 or f64");
    return .{
        // Preserve the original type's bits: widening f32 here would change
        // its shortest decimal representation before precision rounding.
        .bits = @as(@Int(.unsigned, @bitSizeOf(T)), @bitCast(value)),
        .mantissa_bits = std.math.floatMantissaBits(T),
        .exponent_bits = std.math.floatExponentBits(T),
        .precision = precision,
    };
}

pub const Format = struct {
    bits: u64,
    mantissa_bits: u6,
    exponent_bits: u5,
    precision: ?usize,

    // PERFORMANCE: Keep conversion out-of-line and use Ryu's small tables.
    // std.fmt.float.render selects full power-of-five tables in ReleaseSafe;
    // even one ordinary float-formatting call can retain those shared tables.
    // This backend computes coefficients from compact tables, trading extra
    // arithmetic on diagnostic paths for a smaller executable. Do not use it
    // in capture/inference hot paths or change their optimization policy.
    //
    // Keep std's conversion and rounding: integer-only formatting would change
    // rounding boundaries and lose full-precision failed-transcription evidence.
    // Measured 2026-09-07 with stock Zig 0.16.0/LLVM, host x86-64, ReleaseSafe
    // application and inference, static PIE, crash diagnostics disabled, and
    // GNU strip --strip-all: this change alone reduced 1,161,416 to 1,153,000 bytes
    // (8,416 saved). The 5,216- and 5,472-byte full tables disappeared; .rodata
    // shrank 10,000 bytes while .text grew 1,904. Runtime cost was not benchmarked.
    // 8,092,160 byte-for-byte comparisons matched the full-table formatter:
    // random f32/f64 bits, every exponent with boundary mantissas and both signs,
    // full/2/3/6-place output, NaNs, infinities, signed zero and buffer fallback.
    // Recheck linked size and output equivalence when upgrading Zig; these are
    // historical measurements, not guaranteed or independently additive savings.
    pub noinline fn format(self: Format, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const converted = std.fmt.float.binaryToDecimal(
            u64,
            self.bits,
            self.mantissa_bits,
            self.exponent_bits,
            false,
            &std.fmt.float.Backend64_TablesSmall,
        );

        var buffer: [std.fmt.float.bufferSize(.decimal, f64)]u8 = undefined;
        const text = std.fmt.float.formatDecimal(u64, &buffer, converted, self.precision) catch "(float)";

        try writer.writeAll(text);
    }
};
