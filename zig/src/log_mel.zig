//! Computes the exact 80-band log-Mel representation expected by the original
//! English Whisper models. Hot dot products use explicit vectors so feature
//! extraction does not depend on the compiler discovering their SIMD shape.

const std = @import("std");
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const Simd = @Vector(simd_lanes_count, f32);

const sample_rate_hz: u32 = 16_000;
const fft_samples_count: u32 = 400;
const fft_bins_count: u32 = fft_samples_count / 2 + 1;
const dft_coefficients_count = fft_bins_count * fft_samples_count;
const hop_samples_count: u32 = 160;
const mel_bins_count: u32 = 80;
const mel_filters_count = mel_bins_count * fft_bins_count;
const simd_lanes_count: u32 = 8;

comptime {
    assert(fft_samples_count % simd_lanes_count == 0);
    assert((fft_bins_count - 1) % simd_lanes_count == 0);
}

pub const Features = struct {
    values: []f32,
    frames_count: u32,
};

pub const Extractor = struct {
    allocator: Allocator,
    hann_window: []f32,
    dft_cosines: []f32,
    dft_sines: []f32,
    mel_filters: []f32,

    /// `init` allocates immutable coefficient tables for repeated extraction.
    /// The caller must keep `extractor` at a stable address until `deinit`.
    pub fn init(extractor: *Extractor, allocator: Allocator) !void {
        const hann_window = try allocator.alloc(f32, fft_samples_count);
        errdefer allocator.free(hann_window);

        const dft_cosines = try allocator.alloc(f32, dft_coefficients_count);
        errdefer allocator.free(dft_cosines);

        const dft_sines = try allocator.alloc(f32, dft_coefficients_count);
        errdefer allocator.free(dft_sines);

        const mel_filters = try allocator.alloc(f32, mel_filters_count);
        errdefer allocator.free(mel_filters);

        // NumPy's periodic Hann window is `hanning(401)[:-1]` for a 400-point DFT.
        for (hann_window, 0..) |*coefficient, sample_index| {
            const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(sample_index)) / fft_samples_count;
            coefficient.* = @floatCast(0.5 - 0.5 * @cos(angle));
        }

        for (0..fft_bins_count) |fft_bin_index| {
            for (0..fft_samples_count) |sample_index| {
                const coefficient_index = fft_bin_index * fft_samples_count + sample_index;
                const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(fft_bin_index * sample_index)) / fft_samples_count;

                dft_cosines[coefficient_index] = @floatCast(@cos(angle));
                dft_sines[coefficient_index] = @floatCast(@sin(angle));
            }
        }

        calculate_mel_filters(mel_filters);

        extractor.* = .{
            .allocator = allocator,
            .hann_window = hann_window,
            .dft_cosines = dft_cosines,
            .dft_sines = dft_sines,
            .mel_filters = mel_filters,
        };
    }

    pub fn deinit(extractor: *Extractor) void {
        assert(extractor.hann_window.len == fft_samples_count);
        assert(extractor.dft_cosines.len == fft_bins_count * fft_samples_count);
        assert(extractor.dft_sines.len == fft_bins_count * fft_samples_count);
        assert(extractor.mel_filters.len == mel_bins_count * fft_bins_count);

        extractor.allocator.free(extractor.mel_filters);
        extractor.allocator.free(extractor.dft_sines);
        extractor.allocator.free(extractor.dft_cosines);
        extractor.allocator.free(extractor.hann_window);

        extractor.* = undefined;
    }

    /// `calculate` returns row-major `[80, frames_count]` features. The caller
    /// owns `features.values` and must free it with the extractor's allocator.
    pub fn calculate(extractor: *const Extractor, samples: []const f32) !Features {
        if (samples.len < fft_samples_count / 2 + 1) {
            return error.AudioTooShort;
        }
        if (samples.len > 30 * sample_rate_hz) {
            return error.AudioDurationExceedsLimit;
        }

        const content_frames_count: u32 = @intCast(@divFloor(samples.len, hop_samples_count));
        const encoder_frames_count: u32 = 3000;
        const content_features_count = mel_bins_count * content_frames_count;
        const encoder_features_count = mel_bins_count * encoder_frames_count;

        // ── Center And Pad Waveform ──

        const center_padding_samples_count = fft_samples_count / 2;
        const tail_padding_samples_count = hop_samples_count;
        const centered_samples_count = samples.len + 2 * center_padding_samples_count + tail_padding_samples_count;

        const centered_samples = try extractor.allocator.alloc(f32, centered_samples_count);
        defer extractor.allocator.free(centered_samples);

        for (0..center_padding_samples_count) |padding_index| {
            centered_samples[padding_index] = samples[center_padding_samples_count - padding_index];
        }

        @memcpy(
            centered_samples[center_padding_samples_count..][0..samples.len],
            samples,
        );

        const tail_padding_offset = center_padding_samples_count + samples.len;
        const tail_padding = centered_samples[tail_padding_offset..][0..tail_padding_samples_count];

        @memset(tail_padding, 0.0);

        for (0..center_padding_samples_count) |padding_index| {
            const source_index = samples.len + tail_padding_samples_count - 2 - padding_index;
            const target_index = center_padding_samples_count + samples.len +
                tail_padding_samples_count + padding_index;
            centered_samples[target_index] = if (source_index < samples.len)
                samples[source_index]
            else
                0.0;
        }

        // ── Power Spectrum ──

        const power_spectra_count = content_frames_count * fft_bins_count;
        const power_spectra = try extractor.allocator.alloc(f32, power_spectra_count);
        defer extractor.allocator.free(power_spectra);

        calculate_power_spectra(
            power_spectra,
            centered_samples,
            content_frames_count,
            extractor.hann_window,
            extractor.dft_cosines,
            extractor.dft_sines,
        );

        // ── Mel Projection And Log Scaling ──

        const content_features = try extractor.allocator.alloc(f32, content_features_count);
        defer extractor.allocator.free(content_features);

        calculate_mel_features(
            content_features,
            power_spectra,
            content_frames_count,
            extractor.mel_filters,
        );

        var feature_max: f32 = -std.math.inf(f32);
        for (content_features) |*feature| {
            feature.* = @log10(@max(feature.*, 1.0e-10));
            feature_max = @max(feature_max, feature.*);
        }

        const feature_min = feature_max - 8.0;
        normalize_features(content_features, feature_min);

        // CTranslate2's Whisper encoder requires the full 30-second extent.
        // Padding raw audio would affect log scaling, so pad normalized Mel rows.
        const encoder_features = try extractor.allocator.alloc(f32, encoder_features_count);
        errdefer extractor.allocator.free(encoder_features);
        @memset(encoder_features, 0.0);

        for (0..mel_bins_count) |mel_bin_index| {
            const content_row_offset = mel_bin_index * content_frames_count;
            const encoder_row_offset = mel_bin_index * encoder_frames_count;
            const content_row = content_features[content_row_offset..][0..content_frames_count];
            const encoder_row = encoder_features[encoder_row_offset..][0..content_frames_count];

            @memcpy(encoder_row, content_row);
        }

        assert(encoder_features.len == @as(usize, mel_bins_count) * encoder_frames_count);
        assert(content_frames_count > 0);

        return .{
            .values = encoder_features,
            .frames_count = encoder_frames_count,
        };
    }
};

fn calculate_power_spectra(
    power_spectra: []f32,
    centered_samples: []const f32,
    frames_count: u32,
    hann_window: []const f32,
    dft_cosines: []const f32,
    dft_sines: []const f32,
) void {
    assert(power_spectra.len == @as(usize, frames_count) * fft_bins_count);
    assert(centered_samples.len >= @as(usize, frames_count - 1) * hop_samples_count +
        fft_samples_count);
    assert(hann_window.len == fft_samples_count);
    assert(dft_cosines.len == fft_bins_count * fft_samples_count);
    assert(dft_sines.len == dft_cosines.len);

    // PERFORMANCE: Each frequency dot product consumes eight adjacent samples
    // per iteration. Replacing `Simd` with scalar accumulation makes the exact
    // 400-point DFT dominate short-utterance latency.
    for (0..frames_count) |frame_index| {
        const frame_offset = frame_index * hop_samples_count;
        const frame_samples = centered_samples[frame_offset..][0..fft_samples_count];

        for (0..fft_bins_count) |fft_bin_index| {
            const coefficient_offset = fft_bin_index * fft_samples_count;
            const cosines = dft_cosines[coefficient_offset..][0..fft_samples_count];
            const sines = dft_sines[coefficient_offset..][0..fft_samples_count];
            var real_parts: Simd = @splat(0.0);
            var imaginary_parts: Simd = @splat(0.0);

            var sample_index: usize = 0;
            while (sample_index < fft_samples_count) : (sample_index += simd_lanes_count) {
                const sample_vector: Simd = frame_samples[sample_index..][0..simd_lanes_count].*;
                const window_vector: Simd = hann_window[sample_index..][0..simd_lanes_count].*;
                const windowed_samples = sample_vector * window_vector;
                const cosine_vector: Simd = cosines[sample_index..][0..simd_lanes_count].*;
                const sine_vector: Simd = sines[sample_index..][0..simd_lanes_count].*;

                real_parts += windowed_samples * cosine_vector;
                imaginary_parts -= windowed_samples * sine_vector;
            }

            const real_part = @reduce(.Add, real_parts);
            const imaginary_part = @reduce(.Add, imaginary_parts);
            power_spectra[frame_index * fft_bins_count + fft_bin_index] =
                real_part * real_part + imaginary_part * imaginary_part;
        }
    }
}

fn calculate_mel_features(
    features: []f32,
    power_spectra: []const f32,
    frames_count: u32,
    mel_filters: []const f32,
) void {
    assert(features.len == @as(usize, mel_bins_count) * frames_count);
    assert(power_spectra.len == @as(usize, frames_count) * fft_bins_count);
    assert(mel_filters.len == mel_bins_count * fft_bins_count);

    // PERFORMANCE: The filter matrix is reused for every frame. Eight-wide
    // products keep each 201-bin projection contiguous; only the Nyquist bin
    // remains scalar.
    for (0..mel_bins_count) |mel_bin_index| {
        const mel_filter = mel_filters[mel_bin_index * fft_bins_count ..][0..fft_bins_count];

        for (0..frames_count) |frame_index| {
            const spectrum = power_spectra[frame_index * fft_bins_count ..][0..fft_bins_count];
            var sums: Simd = @splat(0.0);

            var fft_bin_index: usize = 0;
            while (fft_bin_index < fft_bins_count - 1) : (fft_bin_index += simd_lanes_count) {
                const spectrum_vector: Simd = spectrum[fft_bin_index..][0..simd_lanes_count].*;
                const filter_vector: Simd = mel_filter[fft_bin_index..][0..simd_lanes_count].*;
                sums += spectrum_vector * filter_vector;
            }

            var sum = @reduce(.Add, sums);
            while (fft_bin_index < fft_bins_count) : (fft_bin_index += 1) {
                sum += spectrum[fft_bin_index] * mel_filter[fft_bin_index];
            }

            features[mel_bin_index * frames_count + frame_index] = sum;
        }
    }
}

fn normalize_features(features: []f32, feature_min: f32) void {
    assert(features.len > 0);
    assert(std.math.isFinite(feature_min));

    var feature_index: usize = 0;
    const feature_min_vector: Simd = @splat(feature_min);
    const feature_add_vector: Simd = @splat(4.0);
    const feature_scale_vector: Simd = @splat(0.25);

    while (feature_index + simd_lanes_count <= features.len) : (feature_index += simd_lanes_count) {
        const feature_vector: Simd = features[feature_index..][0..simd_lanes_count].*;
        const normalized = (@max(feature_vector, feature_min_vector) + feature_add_vector) *
            feature_scale_vector;
        features[feature_index..][0..simd_lanes_count].* = normalized;
    }

    while (feature_index < features.len) : (feature_index += 1) {
        features[feature_index] = (@max(features[feature_index], feature_min) + 4.0) / 4.0;
    }
}

fn calculate_mel_filters(filters: []f32) void {
    const mel_points_count = mel_bins_count + 2;
    var frequencies_hz: [mel_points_count]f64 = undefined;

    assert(filters.len == mel_bins_count * fft_bins_count);
    assert(mel_points_count == frequencies_hz.len);

    for (&frequencies_hz, 0..) |*frequency_hz, mel_point_index| {
        const mel = 45.245640471924965 *
            @as(f64, @floatFromInt(mel_point_index)) / (mel_points_count - 1);
        const frequency_hz_linear = (200.0 / 3.0) * mel;
        const mel_log_start = 1000.0 / (200.0 / 3.0);

        frequency_hz.* = if (mel >= mel_log_start)
            1000.0 * @exp(@log(6.4) / 27.0 * (mel - mel_log_start))
        else
            frequency_hz_linear;
    }

    for (0..mel_bins_count) |mel_bin_index| {
        const frequency_lower_hz = frequencies_hz[mel_bin_index];
        const frequency_center_hz = frequencies_hz[mel_bin_index + 1];
        const frequency_upper_hz = frequencies_hz[mel_bin_index + 2];
        const energy_scale = 2.0 / (frequency_upper_hz - frequency_lower_hz);

        for (0..fft_bins_count) |fft_bin_index| {
            const frequency_hz = @as(f64, @floatFromInt(fft_bin_index)) *
                sample_rate_hz / fft_samples_count;
            const lower_weight = (frequency_hz - frequency_lower_hz) /
                (frequency_center_hz - frequency_lower_hz);
            const upper_weight = (frequency_upper_hz - frequency_hz) /
                (frequency_upper_hz - frequency_center_hz);
            const weight = @max(0.0, @min(lower_weight, upper_weight)) * energy_scale;

            filters[mel_bin_index * fft_bins_count + fft_bin_index] = @floatCast(weight);
        }
    }
}
