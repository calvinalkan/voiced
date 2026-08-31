//! Classifies normalized mono audio as active or quiet relative to the observed
//! background level. This is deliberately not a voice detector: speech, music,
//! and sustained noise can all be active. Capture policy uses these observations
//! to find likely pauses without giving this realtime state machine authority to
//! publish slots, stop PipeWire, or accept transcript text.
//!
//! One `Detector` belongs to one capture session. `observe` retains no sample
//! memory and performs no allocation, locking, atomics, I/O, or system calls, so
//! the PipeWire data thread can call it after converting each validated block to
//! the signed 16-bit PCM stored in the shared exchange.

const std = @import("std");
const assert = std.debug.assert;

pub const Detector = struct {
    sample_rate_hz: u32,
    unknown_samples_count: u32,
    quiet_samples_count: u32,
    active_samples_count: u32,
    activity: Activity,
    activity_samples_count: u32,
    activity_changes_count: u32,
    active_run_samples_count_max: u32,
    quiet_run_samples_count_max: u32,
    noise_floor_rms: f32,

    pub const Activity = enum {
        // The detector has not yet observed enough audio to estimate the
        // session's background level.
        unknown,
        quiet,
        active,
    };

    pub const Observation = struct {
        activity: Activity,
        activity_samples_count: u32,
        block_rms: f32,
        noise_floor_rms: f32,
        threshold_rms: f32,
    };

    pub const Report = struct {
        activity: Activity,
        activity_samples_count: u32,
        observed_samples_count: u32,
        unknown_samples_count: u32,
        quiet_samples_count: u32,
        active_samples_count: u32,
        activity_changes_count: u32,
        active_run_samples_count_max: u32,
        quiet_run_samples_count_max: u32,
        noise_floor_rms: f32,
        quiet_threshold_rms: f32,
        active_threshold_rms: f32,
    };

    /// `init` creates the activity state for one capture session. The first
    /// 300 ms calibrate the background and remain `unknown`; callers that use
    /// activity to detect speech onset must retain that interval in a pre-roll.
    pub fn init(sample_rate_hz: u32) Detector {
        assert(sample_rate_hz >= 8_000);
        assert(sample_rate_hz <= 192_000);

        const calibration_samples_count_target = sample_rate_hz * 300 / 1_000;
        assert(calibration_samples_count_target > 0);

        return .{
            .sample_rate_hz = sample_rate_hz,
            .unknown_samples_count = 0,
            .quiet_samples_count = 0,
            .active_samples_count = 0,
            .activity = .unknown,
            .activity_samples_count = 0,
            .activity_changes_count = 0,
            .active_run_samples_count_max = 0,
            .quiet_run_samples_count_max = 0,
            .noise_floor_rms = 0,
        };
    }

    /// `observe` consumes the next ordered signed 16-bit mono block and returns
    /// the current activity run. `samples` remains owned by the caller.
    pub fn observe(detector: *Detector, samples: []const i16) Observation {
        assert(samples.len > 0);
        assert(samples.len <= std.math.maxInt(u32));
        assert(detector.sample_rate_hz >= 8_000);
        assert(detector.sample_rate_hz <= 192_000);

        const observed_samples_count =
            detector.unknown_samples_count +
            detector.quiet_samples_count +
            detector.active_samples_count;

        assert(observed_samples_count <= std.math.maxInt(u32) - samples.len);

        const calibration_samples_count_target = detector.sample_rate_hz * 300 / 1_000;
        assert(calibration_samples_count_target > 0);

        var samples_offset: usize = 0;

        // ── Calibrate The Background ──
        // Weight each block contribution by its sample count so PipeWire quantum
        // changes cannot bias the floor toward whichever callback happened to
        // contain fewer samples.
        if (detector.unknown_samples_count < calibration_samples_count_target) {
            const calibration_samples_count_remaining =
                calibration_samples_count_target - detector.unknown_samples_count;
            const calibration_samples_count = @min(
                samples.len,
                calibration_samples_count_remaining,
            );
            const calibration_rms = calculateRms(samples[0..calibration_samples_count]);
            const unknown_samples_count_after =
                detector.unknown_samples_count + calibration_samples_count;

            detector.noise_floor_rms = @floatCast(
                (@as(f64, detector.noise_floor_rms) *
                    @as(f64, @floatFromInt(detector.unknown_samples_count)) +
                    @as(f64, calibration_rms) *
                        @as(f64, @floatFromInt(calibration_samples_count))) /
                    @as(f64, @floatFromInt(unknown_samples_count_after)),
            );
            detector.unknown_samples_count = @intCast(unknown_samples_count_after);
            detector.activity_samples_count = detector.unknown_samples_count;
            samples_offset = calibration_samples_count;

            if (samples_offset == samples.len) {
                assert(detector.activity == .unknown);
                assert(detector.quiet_samples_count == 0);
                assert(detector.active_samples_count == 0);
                return .{
                    .activity = .unknown,
                    .activity_samples_count = detector.activity_samples_count,
                    .block_rms = calibration_rms,
                    .noise_floor_rms = detector.noise_floor_rms,
                    .threshold_rms = 0,
                };
            }
        }

        // ── Classify The Remaining Block ──
        // Entering activity requires the larger threshold. Once active, the
        // lower threshold supplies hysteresis so ordinary variation around one
        // boundary does not alternate the state on every callback.
        const classified_samples = samples[samples_offset..];
        const block_rms = calculateRms(classified_samples);
        const thresholds = calculateThresholds(detector.noise_floor_rms);
        const threshold_rms = if (detector.activity == .active)
            thresholds.quiet_rms
        else
            thresholds.active_rms;
        const activity: Activity = if (block_rms > threshold_rms) .active else .quiet;
        const classified_samples_count: u32 = @intCast(classified_samples.len);

        if (activity == detector.activity) {
            detector.activity_samples_count += classified_samples_count;
        } else {
            if (detector.activity != .unknown) {
                detector.activity_changes_count += 1;
            }
            detector.activity = activity;
            detector.activity_samples_count = classified_samples_count;
        }

        switch (activity) {
            .unknown => unreachable,
            .quiet => {
                detector.quiet_samples_count += classified_samples_count;
                detector.quiet_run_samples_count_max = @max(
                    detector.quiet_run_samples_count_max,
                    detector.activity_samples_count,
                );

                // A quieter observation follows a falling background quickly;
                // a louder observation raises the floor over several seconds.
                // Classified activity never trains the floor upward.
                const adaptation_samples_count = if (block_rms < detector.noise_floor_rms)
                    detector.sample_rate_hz / 5
                else
                    detector.sample_rate_hz * 4;
                const adaptation_fraction = @min(
                    1.0,
                    @as(f32, @floatFromInt(classified_samples_count)) /
                        @as(f32, @floatFromInt(adaptation_samples_count)),
                );
                detector.noise_floor_rms +=
                    adaptation_fraction * (block_rms - detector.noise_floor_rms);
            },
            .active => {
                detector.active_samples_count += classified_samples_count;
                detector.active_run_samples_count_max = @max(
                    detector.active_run_samples_count_max,
                    detector.activity_samples_count,
                );
            },
        }

        const observed_samples_count_after = detector.unknown_samples_count +
            detector.quiet_samples_count + detector.active_samples_count;
        assert(observed_samples_count_after == observed_samples_count + samples.len);
        assert(detector.activity != .unknown);
        assert(detector.activity_samples_count > 0);
        assert(detector.noise_floor_rms >= 0);
        return .{
            .activity = detector.activity,
            .activity_samples_count = detector.activity_samples_count,
            .block_rms = block_rms,
            .noise_floor_rms = detector.noise_floor_rms,
            .threshold_rms = threshold_rms,
        };
    }

    /// `report` returns stable session measurements after the caller has
    /// synchronized with the final `observe` call. It does not mutate or reset
    /// the detector.
    pub fn report(detector: *const Detector) Report {
        const observed_samples_count = detector.unknown_samples_count +
            detector.quiet_samples_count + detector.active_samples_count;
        assert(detector.noise_floor_rms >= 0);

        const thresholds = calculateThresholds(detector.noise_floor_rms);
        const report_value: Report = .{
            .activity = detector.activity,
            .activity_samples_count = detector.activity_samples_count,
            .observed_samples_count = observed_samples_count,
            .unknown_samples_count = detector.unknown_samples_count,
            .quiet_samples_count = detector.quiet_samples_count,
            .active_samples_count = detector.active_samples_count,
            .activity_changes_count = detector.activity_changes_count,
            .active_run_samples_count_max = detector.active_run_samples_count_max,
            .quiet_run_samples_count_max = detector.quiet_run_samples_count_max,
            .noise_floor_rms = detector.noise_floor_rms,
            .quiet_threshold_rms = thresholds.quiet_rms,
            .active_threshold_rms = thresholds.active_rms,
        };

        assert(report_value.observed_samples_count == observed_samples_count);
        assert(report_value.active_samples_count <= report_value.observed_samples_count);
        assert(report_value.quiet_samples_count <= report_value.observed_samples_count);
        assert(report_value.quiet_threshold_rms <= report_value.active_threshold_rms);
        return report_value;
    }
};

fn calculateThresholds(noise_floor_rms: f32) struct {
    quiet_rms: f32,
    active_rms: f32,
} {
    assert(std.math.isFinite(noise_floor_rms));
    assert(noise_floor_rms >= 0);

    // Relative thresholds follow microphone gain and background changes.
    // Absolute floors prevent low-level startup and room transients from
    // repeatedly becoming activity when the calibrated room is very quiet.
    const quiet_rms = @max(0.003, noise_floor_rms * 1.8);
    const active_rms = @max(0.008, noise_floor_rms * 3.0);

    assert(quiet_rms <= active_rms);

    return .{
        .quiet_rms = quiet_rms,
        .active_rms = active_rms,
    };
}

fn calculateRms(samples: []const i16) f32 {
    assert(samples.len > 0);

    var squares_sum: f64 = 0;
    for (samples) |sample| {
        const normalized = @as(f64, @floatFromInt(sample)) / 32768.0;
        squares_sum += normalized * normalized;
    }

    const rms: f32 = @floatCast(@sqrt(
        squares_sum / @as(f64, @floatFromInt(samples.len)),
    ));
    assert(std.math.isFinite(rms));
    assert(rms >= 0);
    assert(rms <= 1.0);

    return rms;
}
