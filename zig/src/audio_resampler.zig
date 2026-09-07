//! Streaming mono conversion to Whisper's 16 kHz input. A windowed-sinc
//! low-pass filter prevents aliases before downsampling. Coefficients are
//! prepared only when the graph rate changes; processing has fixed storage
//! and retains phase/history across arbitrary callback boundaries.
const std = @import("std");
pub const Error = error{ UnsupportedRate, OutputFull, InvalidSample };
pub const output_rate = 16000;
const phases = 64;
const taps_max = 384;

pub const Resampler = struct {
    input_rate: u32 = 0,
    taps: usize = 0,
    coefficients: [phases + 1][taps_max]f32 = undefined,
    history: [taps_max]f32 = @splat(0),
    history_cursor: usize = 0,
    input_ordinal: u64 = 0,
    output_time: u64 = 0,

    pub fn configure(self: *Resampler, rate: u32) Error!void {
        if (rate == self.input_rate) return;
        if (rate < 8000 or rate > 192000) return error.UnsupportedRate;
        self.input_rate = rate;
        self.taps = 32 * @as(usize, @intCast((rate + output_rate - 1) / output_rate));
        self.history = @splat(0);
        self.history_cursor = 0;
        self.input_ordinal = 0;
        self.output_time = 0;
        if (rate == output_rate) return;
        const cutoff: f64 = 0.94 * @min(1.0, @as(f64, output_rate) / @as(f64, @floatFromInt(rate)));
        const half: f64 = @as(f64, @floatFromInt(self.taps)) / 2;
        for (&self.coefficients, 0..) |*row, phase| {
            const fraction: f64 = @as(f64, @floatFromInt(phase)) / phases;
            var sum: f64 = 0;
            for (row[0..self.taps], 0..) |*coefficient, tap| {
                const x = @as(f64, @floatFromInt(tap)) + fraction - half;
                const argument = std.math.pi * cutoff * x;
                const sinc = if (@abs(argument) < 1e-12) cutoff else cutoff * @sin(argument) / argument;
                const window = 0.42 + 0.5 * @cos(std.math.pi * x / half) + 0.08 * @cos(2 * std.math.pi * x / half);
                const value = sinc * window;
                coefficient.* = @floatCast(value);
                sum += value;
            }
            for (row[0..self.taps]) |*coefficient| coefficient.* /= @floatCast(sum);
        }
    }

    /// A fixed filter delay is retained in the stream, as in the previous
    /// PipeWire adapter. Sample counts follow the rate ratio; the caller does
    /// not append an artificial filter tail to a stopped recording.
    pub fn process(self: *Resampler, input: []const f32, output: []f32) Error![]f32 {
        std.debug.assert(self.input_rate != 0);
        if (self.input_rate == output_rate) {
            if (input.len > output.len) return error.OutputFull;
            for (input) |sample| if (!std.math.isFinite(sample)) return error.InvalidSample;
            @memcpy(output[0..input.len], input);
            return output[0..input.len];
        }
        var used: usize = 0;
        for (input) |sample| {
            if (!std.math.isFinite(sample)) return error.InvalidSample;
            self.history[self.history_cursor] = sample;
            while (self.output_time / output_rate == self.input_ordinal) {
                if (used == output.len) return error.OutputFull;
                const fraction: f32 = @as(f32, @floatFromInt(self.output_time % output_rate)) * phases / output_rate;
                const phase: usize = @intFromFloat(fraction);
                const mix = fraction - @as(f32, @floatFromInt(phase));
                var value: f32 = 0;
                var cursor = self.history_cursor;
                for (0..self.taps) |tap| {
                    const a = self.coefficients[phase][tap];
                    const b = self.coefficients[phase + 1][tap];
                    value += self.history[cursor] * (a + (b - a) * mix);
                    cursor = if (cursor == 0) self.taps - 1 else cursor - 1;
                }
                output[used] = value;
                used += 1;
                self.output_time += self.input_rate;
            }
            self.history_cursor += 1;
            if (self.history_cursor == self.taps) self.history_cursor = 0;
            self.input_ordinal += 1;
        }
        return output[0..used];
    }
};
