//! A persistent capture thread borrows the recording slots until it returns a
//! complete typed report. Cancellation wakes its PipeWire loop independently of
//! the job mailbox, including during connection setup and realtime negotiation.
const Capture = @This();

const std = @import("std");
const AudioExchange = @import("audio_exchange.zig");
const pipewire = @import("capture/pipewire.zig");
const worker = @import("worker.zig");
const assert = std.debug.assert;

const target_name_capacity = 256;

const ControlCommand = pipewire.ControlCommand;
pub const Source = pipewire.Source;

pub const sample_rate_hz = AudioExchange.sample_rate_hz;
pub const target_name_bytes_capacity = target_name_capacity;
pub const schedulerPolicyName = pipewire.schedulerPolicyName;
pub const schedulerPolicyIsRealtime = pipewire.schedulerPolicyIsRealtime;
pub const CaptureReport = pipewire.Report;
pub const RuntimeFailure = pipewire.RuntimeFailure;
pub const FailureCause = pipewire.FailureCause;
pub const SourceIdentity = pipewire.SourceIdentity;
pub const Error = pipewire.Error;
const Result = pipewire.Result;

pub const Job = struct {
    recording_id: u64,
    source: Source,
    recording_samples_target: u32,
};

mailbox: worker.Mailbox(Job, Result),
exchange: *AudioExchange,
environment: pipewire.Environment,
stop: std.atomic.Value(ControlCommand) = .init(.none),

pub fn start(self: *Capture, job: Job) void {
    // Reset before publishing: a subsequent cancel must never be erased by
    // the capture thread while it is taking the job.
    self.stop.store(.none, .release);
    self.mailbox.submit(job);
}

pub fn requestStop(self: *Capture, command: ControlCommand) void {
    assert(command != .none);
    self.stop.store(command, .release);
    worker.wake(self.mailbox.wake_fd);
}

pub fn run(self: *Capture) void {
    worker.name("voiced-capture");
    defer self.mailbox.finish();
    while (self.mailbox.next()) |job| {
        const result = pipewire.run(.{
            .recording_id = job.recording_id,
            .exchange = self.exchange,
            .publication_event_fd = self.mailbox.notification_fd,
            .control_event_fd = self.mailbox.wake_fd,
            .control = &self.stop,
            .source = job.source,
            .recording_samples_target = job.recording_samples_target,
            .environment = self.environment,
        });
        self.mailbox.complete(result);
    }
}
