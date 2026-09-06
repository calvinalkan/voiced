//! Transfers fixed launch records and exactly two role-specific Linux
//! descriptors over a Unix `SOCK_SEQPACKET` connection. Descriptor integers are
//! process-local; `SCM_RIGHTS` asks the kernel to install descriptors for the
//! same objects in the receiving process. Receipt returns one owned pair only
//! after the complete packet and close-on-exec contract have been validated.

const std = @import("std");
const assert = std.debug.assert;
const linux = std.os.linux;

pub const DescriptorPair = struct {
    values: [2]std.posix.fd_t,

    pub fn deinit(pair: *DescriptorPair) void {
        for (pair.values) |descriptor| closeDescriptor(descriptor);
        pair.* = undefined;
    }
};

/// `send` transmits one complete fixed record and two descriptors as one
/// seqpacket. The sender retains ownership of its originals.
pub fn send(
    socket: std.posix.fd_t,
    record: anytype,
    descriptors: *const [2]std.posix.fd_t,
) !void {
    return sendRecordBytes(socket, std.mem.asBytes(record), descriptors);
}

// Only conversion to a fixed byte span depends on the launch-record type.
// Share the syscall/control-message work instead of inlining it into each type;
// the extra call occurs at worker launch, not during audio or inference work.
noinline fn sendRecordBytes(socket: std.posix.fd_t, record: []const u8, descriptors: *const [2]std.posix.fd_t) !void {
    assert(socket >= 0);

    for (descriptors) |descriptor| assert(descriptor >= 0);

    const descriptors_size = @sizeOf(@TypeOf(descriptors.*));

    var control_buffer: [controlSpace(descriptors_size)]u8 align(@alignOf(usize)) = @splat(0);

    const header: *linux.cmsghdr = @ptrCast(@alignCast(&control_buffer));
    header.* = .{
        .len = controlLength(descriptors_size),
        .level = linux.SOL.SOCKET,
        .type = linux.SCM.RIGHTS,
    };
    @memcpy(
        control_buffer[controlAlign(@sizeOf(linux.cmsghdr))..][0..descriptors_size],
        std.mem.asBytes(descriptors),
    );

    var vectors = [_]std.posix.iovec_const{.{
        .base = record.ptr,
        .len = record.len,
    }};

    const message: linux.msghdr_const = .{
        .name = null,
        .namelen = 0,
        .iov = &vectors,
        .iovlen = vectors.len,
        .control = &control_buffer,
        .controllen = control_buffer.len,
        .flags = 0,
    };

    const send_result = linux.sendmsg(socket, &message, linux.MSG.NOSIGNAL);
    if (linux.errno(send_result) != .SUCCESS) {
        return error.DescriptorHandoffSendFailed;
    }
    if (send_result != record.len) {
        return error.DescriptorHandoffRecordTruncated;
    }
}

/// `receive` reads one complete launch record and returns exactly two owned
/// descriptors. If the kernel installs rights but any later packet validation
/// fails, this function closes those descriptors before returning the error.
pub fn receive(socket: std.posix.fd_t, record: anytype) !DescriptorPair {
    return receiveRecordBytes(socket, std.mem.asBytes(record));
}

// Keep packet validation and descriptor rollback in one shared operation.
// The typed wrapper supplies exactly the record's size, not spare capacity.
noinline fn receiveRecordBytes(socket: std.posix.fd_t, record: []u8) !DescriptorPair {
    assert(socket >= 0);

    const descriptors_count_expected = 2;
    const descriptors_count_max = 8;
    const descriptors_size = @sizeOf(std.posix.fd_t) * descriptors_count_expected;
    // Receive more SCM_RIGHTS slots than we accept so extras can be closed
    // instead of truncated out of this process with no handle to reclaim.
    var control_buffer: [controlSpace(@sizeOf(std.posix.fd_t) * descriptors_count_max)]u8 align(@alignOf(usize)) = undefined;
    var vectors = [_]std.posix.iovec{.{
        .base = record.ptr,
        .len = record.len,
    }};
    var message: linux.msghdr = .{
        .name = null,
        .namelen = 0,
        .iov = &vectors,
        .iovlen = vectors.len,
        .control = &control_buffer,
        .controllen = control_buffer.len,
        .flags = 0,
    };

    const receive_result = linux.recvmsg(
        socket,
        &message,
        linux.MSG.CMSG_CLOEXEC | linux.MSG.TRUNC,
    );
    if (linux.errno(receive_result) != .SUCCESS) {
        return error.DescriptorHandoffReceiveFailed;
    }
    if (receive_result == 0) {
        return error.DescriptorHandoffSocketClosed;
    }

    // SCM_RIGHTS descriptors become owned by this process during `recvmsg`, not
    // when the packet is later accepted. Recover the expected pair first and
    // arm cleanup before checking record length or truncation flags.
    if (message.controllen < @sizeOf(linux.cmsghdr)) {
        return error.DescriptorHandoffUnexpectedControlSize;
    }
    const header: *const linux.cmsghdr = @ptrCast(@alignCast(
        control_buffer[0..@sizeOf(linux.cmsghdr)].ptr,
    ));

    var installed_descriptors: [descriptors_count_max]std.posix.fd_t = undefined;
    var installed_descriptors_count: usize = 0;
    var announced_descriptors_count: usize = 0;
    if (header.level == linux.SOL.SOCKET and
        header.type == linux.SCM.RIGHTS and
        header.len >= controlLength(0) and
        header.len <= message.controllen)
    {
        const installed_bytes_count = header.len - controlAlign(@sizeOf(linux.cmsghdr));
        if (installed_bytes_count % @sizeOf(std.posix.fd_t) == 0) {
            announced_descriptors_count = installed_bytes_count / @sizeOf(std.posix.fd_t);
            installed_descriptors_count = @min(announced_descriptors_count, installed_descriptors.len);
            @memcpy(
                std.mem.sliceAsBytes(installed_descriptors[0..installed_descriptors_count]),
                control_buffer[controlAlign(@sizeOf(linux.cmsghdr))..][0 .. installed_descriptors_count * @sizeOf(std.posix.fd_t)],
            );
        }
    }
    var installed_descriptors_are_owned = true;
    errdefer if (installed_descriptors_are_owned) {
        for (installed_descriptors[0..installed_descriptors_count]) |descriptor| {
            closeDescriptor(descriptor);
        }
    };

    if (header.len != controlLength(descriptors_size) or
        header.level != linux.SOL.SOCKET or
        header.type != linux.SCM.RIGHTS or
        message.controllen != controlSpace(descriptors_size) or
        announced_descriptors_count != descriptors_count_expected or
        installed_descriptors_count != descriptors_count_expected)
    {
        return error.DescriptorHandoffUnexpectedControl;
    }

    const pair: DescriptorPair = .{ .values = installed_descriptors[0..descriptors_count_expected].* };
    if (receive_result != record.len) {
        return error.DescriptorHandoffRecordTruncated;
    }
    if (message.flags & (linux.MSG.CTRUNC | linux.MSG.TRUNC) != 0) {
        return error.DescriptorHandoffControlTruncated;
    }

    for (pair.values) |descriptor| {
        if (descriptor < 0) {
            return error.DescriptorHandoffInvalidDescriptor;
        }
        const descriptor_flags = linux.fcntl(descriptor, linux.F.GETFD, 0);
        if (linux.errno(descriptor_flags) != .SUCCESS) {
            return error.DescriptorHandoffDescriptorStatusFailed;
        }
        if (descriptor_flags & linux.FD_CLOEXEC == 0) {
            return error.DescriptorHandoffDescriptorNotCloseOnExec;
        }
    }

    installed_descriptors_are_owned = false;
    return pair;
}

fn controlAlign(size: usize) usize {
    return std.mem.alignForward(usize, size, @alignOf(usize));
}

fn controlLength(data_size: usize) usize {
    return controlAlign(@sizeOf(linux.cmsghdr)) + data_size;
}

fn controlSpace(data_size: usize) usize {
    return controlAlign(@sizeOf(linux.cmsghdr)) + controlAlign(data_size);
}

fn closeDescriptor(descriptor: std.posix.fd_t) void {
    assert(descriptor >= 0);
    const result = linux.close(descriptor);
    assert(linux.errno(result) == .SUCCESS);
}
