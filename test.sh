#!/bin/bash
# Process-level integration coverage for CLI, daemon, model worker, and recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Service tests receive native journal datagrams privately, including from workers.
if [[ "${1:-}" == --zig* ]]; then
    VOICED_JOURNAL_TEST_SUPPORT=$(mktemp -d /tmp/voiced-journal-support.XXXXXX)
    export PYTHONPATH="$VOICED_JOURNAL_TEST_SUPPORT${PYTHONPATH:+:$PYTHONPATH}"
    trap 'rm -rf "$VOICED_JOURNAL_TEST_SUPPORT"' EXIT
    cat > "$VOICED_JOURNAL_TEST_SUPPORT/journal_fixture.py" <<'PY'
import atexit
import json
import socket
import struct
import threading


def decode_record(data):
    fields = {}
    while data:
        line, data = data.split(b"\n", 1)
        if b"=" in line:
            key, value = line.split(b"=", 1)
        else:
            key = line
            size, = struct.unpack_from("<Q", data)
            value = data[8:8 + size]
            assert len(value) == size and data[8 + size:9 + size] == b"\n"
            data = data[9 + size:]
        assert key not in fields
        fields[key.decode()] = value.decode(errors="replace")
    return fields


receivers = []


class journal_log:
    # The with-block usually only spans Popen. Keep draining after it ends,
    # until suite teardown; already-running children still own their fd 2.
    def __init__(self, path):
        self.reader, self.sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.reader.settimeout(0.05)
        self.output = path.open("w")
        self.metadata = path.with_suffix(".journal.jsonl").open("w")
        self.records = []
        self.stopping = False
        self.thread = threading.Thread(target=self.drain, daemon=True)
        receivers.append(self)
        self.thread.start()

    def fileno(self):
        return self.sender.fileno()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sender.close()

    def drain(self):
        while True:
            try:
                data = self.reader.recv(8192)
            except socket.timeout:
                if self.stopping:
                    break
                continue
            if data.startswith(b"PRIORITY="):
                record = decode_record(data)
                self.records.append(record)
                self.metadata.write(json.dumps(record) + "\n")
                self.metadata.flush()
                names = {"2": "critical", "3": "error", "4": "warning", "6": "info", "7": "debug"}
                self.output.write(names[record["PRIORITY"]] + ": " + record["MESSAGE"] + "\n")
            else:
                self.output.write(data.decode(errors="replace"))
            self.output.flush()
        self.reader.close()
        self.output.close()
        self.metadata.close()


@atexit.register
def close_receivers():
    for receiver in receivers:
        receiver.stopping = True
    for receiver in receivers:
        receiver.thread.join(timeout=1)
PY
fi

if [[ "${1:-}" == "--zig-control" ]]; then
    export VOICED_INSTANCE=test
    python3 - <<'PY'
from journal_fixture import journal_log
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time

binary = str(Path("zig/zig-out/bin/voiced").resolve())
request = struct.Struct("<4sHBB")
reply = struct.Struct("<4sHBBQQBB6s")
assert request.size == 8 and reply.size == 32

with tempfile.TemporaryDirectory(prefix="voiced-control-") as temporary:
    root = Path(temporary)
    env = dict(os.environ, VOICED_INSTANCE="test", XDG_RUNTIME_DIR=str(root),
               PIPEWIRE_RUNTIME_DIR=str(root), XDG_CONFIG_HOME=str(root / "config"),
               XDG_STATE_HOME=str(root / "state"), XDG_CACHE_HOME=str(root / "cache"),
               DBUS_SESSION_BUS_ADDRESS="unix:path=" + str(root / "missing-bus"))
    path = root / "voiced-test/control.sock"

    def cli(*args):
        return subprocess.run([binary, *args], env=env, capture_output=True, timeout=5)

    def exchange(data):
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.settimeout(3)
            client.connect(str(path))
            assert client.send(data) == len(data)
            data = client.recv(4096)
            assert len(data) == reply.size, data
            fields = reply.unpack(data)
            assert fields[0:2] == (b"VCDR", 1) and fields[-1] == bytes(6), fields
            return fields

    # A stale sequenced-packet pathname is recoverable without bypassing the lock.
    path.parent.mkdir(mode=0o700)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as stale:
        stale.bind(str(path))
    with journal_log(root / "daemon.log") as log:
        daemon = subprocess.Popen([binary, "serve", "--transcript-output", "stdout"], env=env,
                                  stdout=subprocess.DEVNULL, stderr=log)
        try:
            deadline = time.monotonic() + 5
            while True:
                assert daemon.poll() is None
                result = cli("status")
                if result.returncode == 0: break
                assert time.monotonic() < deadline, result.stderr
                time.sleep(.01)
            assert result.stdout.startswith(b"phase=idle model=absent session_id=")
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
            status = request.pack(b"VCDQ", 1, 5, 0)
            snapshot = exchange(status)
            assert snapshot[2:4] == (0, 5) and snapshot[6:8] == (1, 1), snapshot
            for packet, expected in (
                (request.pack(b"BAD!", 1, 5, 0), 2),
                (request.pack(b"VCDQ", 2, 5, 0), 3),
                (request.pack(b"VCDQ", 1, 0, 0), 2),
                (request.pack(b"VCDQ", 1, 255, 0), 2),
                (request.pack(b"VCDQ", 1, 5, 1), 2),
                (request.pack(b"VCDQ", 1, 2, 128), 2),
                (status[:3], 2), (status + b"x", 2), (status * 2, 2),
                (b'{"cmd":"status"}\n', 2),
            ):
                rejected = exchange(packet)
                assert rejected[2] == expected and rejected[4:8] == (0, 0, 0, 0), rejected
            # Connections have deadlines even when the peer sends no record.
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as idle:
                idle.settimeout(3)
                idle.connect(str(path))
                assert cli("status").returncode == 0
                assert idle.recv(1) == b""
            # A disconnected reader must not terminate the service via SIGPIPE.
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as departed:
                departed.connect(str(path))
                departed.send(status)
            assert cli("status").returncode == 0
            with journal_log(root / "duplicate.log") as refusal:
                duplicate = subprocess.run([binary, "serve", "--transcript-output", "stdout"], env=env,
                                           stdout=subprocess.DEVNULL, stderr=refusal, timeout=3)
                assert duplicate.returncode == 1 and path.exists()
            # Queue both requests before one epoll batch: the first schedules
            # capture, the second must report ignored without scheduling again.
            import signal
            peers = []
            os.kill(daemon.pid, signal.SIGSTOP)
            try:
                for _ in range(2):
                    peer = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                    peers.append(peer)
                    peer.settimeout(3)
                    peer.connect(str(path))
                    packet = request.pack(b"VCDQ", 1, 2, 0)
                    assert peer.send(packet) == len(packet)
                os.kill(daemon.pid, signal.SIGCONT)
                outcomes = [reply.unpack(peer.recv(4096)) for peer in peers]
                assert [row[2] for row in outcomes] == [0, 1], outcomes
                assert all(row[3] == 2 and row[6] == 2 for row in outcomes), outcomes
            finally:
                os.kill(daemon.pid, signal.SIGCONT)
                for peer in peers: peer.close()
            assert cli("cancel").returncode == 0
            for args in (("stop",), ("cancel",), ("record",), ("stop",), ("listen",),
                         ("cancel",), ("record", "-t"), ("cancel",)):
                result = cli(*args)
                assert result.returncode == 0 and result.stdout == b"", (args, result)
                assert cli("status").returncode == 0
            result = cli("kill")
            assert result.returncode == 0 and result.stdout == b"", result
            assert daemon.wait(timeout=5) == 0
            assert not path.exists()
        finally:
            if daemon.poll() is None:
                daemon.kill()
                daemon.wait()

    # A live older stream listener is incompatible, not stale; never unlink it.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as old:
        old.bind(str(path))
        old.listen(1)
        inode = path.stat().st_ino
        result = cli("status")
        assert result.returncode == 1 and b"incompatible control protocol" in result.stderr, result
        with journal_log(root / "old-listener.log") as refusal:
            duplicate = subprocess.run([binary, "serve", "--transcript-output", "stdout"], env=env,
                                       stdout=subprocess.DEVNULL, stderr=refusal, timeout=3)
            assert duplicate.returncode == 1 and path.stat().st_ino == inode
    path.unlink()

    # Exercise the real CLI against malformed replies, including unknown enums.
    valid = reply.pack(b"VCDR", 1, 0, 5, 17, 30, 1, 4, bytes(6))
    cases = [(valid, 0, b"phase=idle model=warm session_id=17 model_keep_warm_seconds=30\n")]
    for offset, value in ((0, 0), (4, 2), (6, 1), (6, 2), (6, 255), (7, 2), (24, 0), (24, 255), (25, 255), (26, 1)):
        bad = bytearray(valid)
        bad[offset] = value
        cases.append((bytes(bad), 1, b""))
    cases += [(valid[:-1], 1, b""), (valid + b"x", 1, b""), (None, 1, b"")]
    for outcome in (2, 3):
        cases.append((reply.pack(b"VCDR", 1, outcome, 5, 0, 0, 0, 0, bytes(6)), 1, b""))
    for packet, exit_code, stdout in cases:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as fake:
            fake.bind(str(path))
            fake.listen(1)
            fake.settimeout(3)
            client = subprocess.Popen([binary, "status"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                peer, _ = fake.accept()
                with peer:
                    peer.settimeout(3)
                    assert peer.recv(1024) == request.pack(b"VCDQ", 1, 5, 0)
                    if packet is not None: assert peer.send(packet) == len(packet)
                output, errors = client.communicate(timeout=4)
                assert client.returncode == exit_code and output == stdout, (packet, output, errors)
            finally:
                if client.poll() is None: client.kill()
                client.wait()
                path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as fake:
        fake.bind(str(path)); fake.listen(1); fake.settimeout(3)
        client = subprocess.Popen([binary, "record", "-t"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            peer, _ = fake.accept()
            with peer:
                peer.settimeout(3)
                assert peer.recv(1024) == request.pack(b"VCDQ", 1, 2, 1)
                ignored = reply.pack(b"VCDR", 1, 1, 2, 17, 30, 5, 4, bytes(6))
                assert peer.send(ignored) == len(ignored)
            output, errors = client.communicate(timeout=4)
            assert client.returncode == 0 and output == b"", errors
        finally:
            if client.poll() is None: client.kill()
            client.wait()
            path.unlink()
print("Fixed-record control integration passed.")
PY
    exit 0
fi

if [[ "${1:-}" == "--zig-logging" ]]; then
    export VOICED_INSTANCE=test
    python3 - <<'PY'
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import tempfile
from journal_fixture import decode_record, journal_log

with tempfile.TemporaryDirectory(prefix="voiced-logging-test-") as temporary:
    root = Path(temporary)
    source = root / "driver.zig"
    source.write_text(r'''
const std = @import("std");
const logging = @import("logging");
const log = logging.scoped(.fixture);
const linux = std.os.linux;
const NeverFormat = struct {
    pub fn format(_: NeverFormat, _: *std.Io.Writer) std.Io.Writer.Error!void {
        @panic("disabled debug was formatted");
    }
};
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (std.mem.eql(u8, args[1], "epoll-error")) {
        // Inject one OS error into the real supervisor without production hooks.
        const Filter = extern struct { code: u16, jt: u8 = 0, jf: u8 = 0, k: u32 };
        const filters = [_]Filter{
            .{ .code = 0x20, .k = 0 }, // Load seccomp_data.nr.
            .{ .code = 0x15, .jf = 1, .k = @intFromEnum(linux.SYS.epoll_pwait) },
            .{ .code = 0x06, .k = 0x00050000 | @as(u32, @intFromEnum(linux.E.IO)) },
            .{ .code = 0x06, .k = 0x7fff0000 },
        };
        const Program = extern struct { len: u16, filter: [*]const Filter };
        const program: Program = .{ .len = filters.len, .filter = &filters };
        std.debug.assert(linux.errno(linux.prctl(@intFromEnum(linux.PR.SET_NO_NEW_PRIVS), 1, 0, 0, 0)) == .SUCCESS);
        std.debug.assert(linux.errno(linux.prctl(@intFromEnum(linux.PR.SET_SECCOMP), 2, @intFromPtr(&program), 0, 0)) == .SUCCESS);
        return std.process.replace(init.io, .{ .argv = args[2..] });
    }
    const flooding = std.mem.eql(u8, args[1], "flood");
    std.debug.assert(logging.init(if (flooding) .info else .debug) == .SUCCESS);
    defer logging.deinit();
    if (flooding) {
        log.debug(.{}, "{f}", .{NeverFormat{}});
        if (logging.enabled(.debug)) @panic("disabled expensive debug preparation");
        for (0..100000) |i| log.info(.{}, "event {d}", .{i});
        std.debug.assert(linux.write(1, "x", 1) == 1);
        var byte: [1]u8 = undefined;
        std.debug.assert(linux.read(0, &byte, 1) == 1);
        log.info(.{}, "recovered", .{});
        return;
    }
    log.critical(.{}, "critical", .{});
    log.err(.{ .recording_ordinal = 17 }, "line one\nPRIORITY=0\nMESSAGE=still the original message", .{});
    log.warn(.{}, "warning", .{});
    log.info(.{}, "info", .{});
    log.debug(.{}, "debug", .{});
    log.info(.{}, "microphone_description=\"{f}\"", .{std.zig.fmtString("USB, \"mic\"\nnext")});
    const oversized: [10000]u8 = @splat('x');
    log.err(.{}, "large:{s}", .{oversized});
}
''')
    driver = root / "driver"
    subprocess.run(["zig", "build-exe", "-OReleaseSafe", "--dep", "logging", "-Mroot=" + str(source),
                    "-Mlogging=" + str(Path("zig/src/logging.zig").resolve()), "-femit-bin=" + str(driver)],
                   check=True, timeout=60)
    reader, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    reader.settimeout(2)
    run = subprocess.Popen([str(driver), "normal"], stderr=sender)
    records = []
    for _ in range(7):
        packet = reader.recv(8192)
        assert len(packet) <= 4096
        records.append(decode_record(packet))
    assert run.wait(timeout=2) == 0
    assert [r["PRIORITY"] for r in records] == ["2", "3", "4", "6", "7", "6", "3"]
    assert all(r["SYSLOG_IDENTIFIER"] == "voiced" and r["VOICED_COMPONENT"] == "fixture" for r in records)
    assert records[1]["VOICED_RECORDING_ORDINAL"] == "17" and "VOICED_SESSION" not in records[1]
    assert records[1]["MESSAGE"] == "line one\nPRIORITY=0\nMESSAGE=still the original message"
    assert records[-2]["MESSAGE"] == r'microphone_description="USB, \"mic\"\nnext"'
    assert records[-1]["MESSAGE"].startswith("large:xxx") and records[-1]["MESSAGE"].endswith(" [truncated]")
    reader.close()
    sender.close()

    reader, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    run = subprocess.Popen([str(driver), "flood"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sender)
    assert select.select([run.stdout], [], [], 3)[0], "logger blocked on a full receiver"
    assert run.stdout.read(1) == b"x"
    reader.setblocking(False)
    while True:
        try:
            decode_record(reader.recv(8192))
        except BlockingIOError:
            break
    run.stdin.write(b"x")
    run.stdin.flush()
    reader.settimeout(2)
    recovered = decode_record(reader.recv(8192))
    assert recovered["MESSAGE"] == "recovered" and int(recovered["VOICED_DROPPED"]) > 0
    assert run.wait(timeout=2) == 0
    run.stdin.close()
    run.stdout.close()
    reader.close()
    sender.close()

    # Validate the public config/CLI path and worker-level propagation through
    # a real failed microphone attempt, with no host audio or notifications.
    binary = str(Path("zig/zig-out/bin/voiced").resolve())
    config = root / "config"
    config.write_text("log_level=error\ntranscript_output=stdout\n")
    env = dict(os.environ, XDG_RUNTIME_DIR=str(root), PIPEWIRE_RUNTIME_DIR=str(root), XDG_STATE_HOME=str(root / "state"))
    with journal_log(root / "service.log") as output:
        daemon = subprocess.Popen([binary, "serve", "--config", str(config), "--log-level", "debug"],
                                  env=env, stdout=subprocess.DEVNULL, stderr=output)
        try:
            import time
            deadline = time.monotonic() + 3
            while not (root / "voiced-test/control.sock").exists():
                assert daemon.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            with journal_log(root / "refused.log") as refusal:
                duplicate = subprocess.run([binary, "serve", "--config", str(config)], env=env, stderr=refusal, timeout=3)
                assert duplicate.returncode == 1
                deadline = time.monotonic() + 2
                while not refusal.records:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert any(row["PRIORITY"] == "3" and "Service startup refused: error=DaemonAlreadyRunning" in row["MESSAGE"] for row in refusal.records), refusal.records
                assert not any(row["PRIORITY"] == "2" for row in refusal.records), refusal.records
            for command in ("record", "stop", "stop"):
                response = subprocess.run([binary, command], env=env, capture_output=True, check=True, timeout=3)
                assert response.stdout == b""
            deadline = time.monotonic() + 5
            while dict(field.split("=", 1) for field in subprocess.check_output([binary, "status"], env=env, text=True).split())["phase"] != "idle":
                assert time.monotonic() < deadline
                time.sleep(0.01)
            subprocess.run([binary, "kill"], env=env, check=True, capture_output=True, timeout=3)
            assert daemon.wait(timeout=3) == 0
            assert "log_level=debug" in (root / "service.log").read_text()
            assert any(record["PRIORITY"] == "7" and "Worker starting:" in record["MESSAGE"] and "log_level=debug" in record["MESSAGE"] for record in output.records), output.records
        finally:
            if daemon.poll() is None:
                daemon.kill()
                daemon.wait()
    # At error level readiness/config info is silent, but status still works.
    with journal_log(root / "quiet.log") as output:
        daemon = subprocess.Popen([binary, "serve", "--config", str(config)], env=env, stderr=output)
        try:
            deadline = time.monotonic() + 3
            while not (root / "voiced-test/control.sock").exists():
                assert daemon.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            subprocess.run([binary, "kill"], env=env, check=True, capture_output=True, timeout=3)
            assert daemon.wait(timeout=3) == 0
            assert not output.records
        finally:
            if daemon.poll() is None:
                daemon.kill()
                daemon.wait()
    with journal_log(root / "syscall.log") as output:
        failed = subprocess.run([str(driver), "epoll-error", binary, "serve", "--config", str(config)], env=env, stderr=output, timeout=5)
        assert failed.returncode == 1  # Returned error, not a trap/signal.
        deadline = time.monotonic() + 2
        while len(output.records) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert any(row["PRIORITY"] == "3" and "operation=epoll_wait, errno=IO" in row["MESSAGE"] for row in output.records), output.records
        assert any(row["PRIORITY"] == "2" and "Service stopped: error=SupervisorSystemCallFailed" in row["MESSAGE"] for row in output.records), output.records
    invalid = subprocess.run([binary, "serve", "--log-level", "verbose"], env=env, capture_output=True)
    assert invalid.returncode == 2 and b"critical, error, warn, info, or debug" in invalid.stderr
    print("Journal priorities, multiline framing, truncation, filtering, backpressure, and CLI configuration passed")
PY
    exit 0
fi

if [[ "${1:-}" == "--zig-replay" ]]; then
    export VOICED_INSTANCE=test
    .venv/bin/python - <<'PY'
import array
from journal_fixture import journal_log
import importlib.util
import json
import mmap
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import tempfile

os.umask(0o077)
spec = importlib.util.spec_from_file_location("replay", "zig/scripts/replay-transcription.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
binary = str(Path("zig/zig-out/bin/voiced-replay").resolve())
service = str(Path("zig/zig-out/bin/voiced").resolve())
with tempfile.TemporaryDirectory(prefix="voiced-replay-test-") as temporary:
    root = Path(temporary)
    samples = replay.read_audio(Path("test-fixtures/hello_world.wav"))
    raw = root / "audio.f32"
    raw.write_bytes(samples.astype("<f4").tobytes())
    store = root / "state"
    def decode(limit, name, expect_success=True):
        completed = subprocess.run([binary, "Systran/faster-whisper-base.en", "8", "4", "10", str(limit),
                                    str(raw), str(root / f"{name}.json"), str(store)],
                                   capture_output=True, timeout=30)
        assert (completed.returncode == 0) == expect_success, completed.stderr.decode()
        assert b"panic" not in completed.stderr
        return json.loads((root / f"{name}.json").read_text())
    first = decode(1, "first")
    capture = store / "last-failed"
    assert first["metadata"]["error_name"] == "GeneratedTokenLimitExceeded"
    assert first["metadata"]["evidence"]["tokens"] == 1
    assert first["metadata"]["evidence"]["encoder_positions"] == 680
    assert replay.read_audio(capture / "audio.wav").tobytes() == raw.read_bytes()
    assert stat.S_IMODE(store.stat().st_mode) == 0o700
    assert stat.S_IMODE(capture.stat().st_mode) == 0o700
    assert set(p.name for p in capture.iterdir()) == {"audio.wav", "generated.txt", "tokens.txt", "metadata.txt"}
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in capture.iterdir())
    parsed_metadata = replay.read_metadata(capture / "metadata.txt")
    assert parsed_metadata == first["metadata"], {key: (parsed_metadata[key], value) for key, value in first["metadata"].items() if parsed_metadata[key] != value}
    assert replay.read_tokens(capture / "tokens.txt") == first["tokens"]
    assert (capture / "tokens.txt").read_text() == " ".join(map(str, first["tokens"])) + "\n"
    metadata_text = (capture / "metadata.txt").read_text()
    probe = root / "metadata.txt"
    escaped = r'  quote=\" path=\\ cafe=caf\xc3\xa9\nnext\r\ttail  '
    probe.write_text(metadata_text.replace("stage=offline_replay\n", "stage=" + escaped + "\n"))
    assert replay.read_metadata(probe)["stage"] == '  quote=" path=\\ cafe=café\nnext\r\ttail  '
    for invalid in (
        metadata_text + "session_id=0\n", metadata_text + "unknown=0\n", metadata_text[:-1],
        metadata_text.replace("format_version=2", "format_version=3"),
        metadata_text.replace("session_id=0", "session_id=-1"),
        metadata_text.replace("contains_activity=false", "contains_activity=no"),
        metadata_text.replace("stage=offline_replay", r"stage=\q"),
        metadata_text.replace("stage=offline_replay", r"stage=\x0"),
    ):
        probe.write_text(invalid)
        try:
            replay.read_metadata(probe)
            raise AssertionError("malformed metadata accepted")
        except ValueError:
            pass
    token_probe = root / "tokens.txt"
    token_probe.write_text("0 4294967295\n")
    assert replay.read_tokens(token_probe) == [0, 4294967295]
    for invalid in ("-1\n", "4294967296\n", "1.0\n", "12"):
        token_probe.write_text(invalid)
        try:
            replay.read_tokens(token_probe)
            raise AssertionError("malformed token list accepted")
        except ValueError:
            pass
    # Replay still reads old captures; the next save removes their JSON names.
    legacy_metadata = dict(first["metadata"], format_version=1)
    (capture / "metadata.json").write_text(json.dumps(legacy_metadata) + "\n")
    (capture / "tokens.json").write_text(json.dumps(first["tokens"]) + "\n")
    (capture / "metadata.txt").unlink()
    (capture / "tokens.txt").unlink()
    assert replay.read_metadata(capture / "metadata.json") == legacy_metadata
    assert replay.read_tokens(capture / "tokens.json") == first["tokens"]
    legacy = root / "legacy"
    legacy.mkdir(mode=0o700)
    for file in capture.iterdir():
        (legacy / file.name).write_bytes(file.read_bytes())
    store.chmod(0o777)
    second = decode(2, "second")
    assert stat.S_IMODE(store.stat().st_mode) == 0o700
    assert replay.read_tokens(capture / "tokens.txt") == second["tokens"]
    assert replay.read_metadata(capture / "metadata.txt") == second["metadata"]
    assert not (capture / "metadata.json").exists() and not (capture / "tokens.json").exists()
    assert (capture / "generated.txt").read_text() == second["text"]
    assert not (store / "last-failed.pending").exists()
    before = {p.name: p.read_bytes() for p in capture.iterdir()}
    pending = store / "last-failed.pending"
    pending.mkdir()
    (pending / "audio.wav").mkdir()  # A staging error must preserve the previous generation.
    decode(1, "failed-save", expect_success=False)
    assert before == {p.name: p.read_bytes() for p in capture.iterdir()}
    (pending / "audio.wav").rmdir()
    outside = root / "untouched"
    outside.write_text("leave this file alone")
    (pending / "generated.txt").symlink_to(outside)
    (pending / "metadata.json").symlink_to(outside)
    (pending / "tokens.json").write_text("interrupted legacy generation")
    (pending / "metadata.txt").symlink_to(outside)
    (pending / "tokens.txt").symlink_to(outside)
    decode(2, "after-interruption")
    assert outside.read_text() == "leave this file alone"
    before = {p.name: p.read_bytes() for p in capture.iterdir()}
    success = decode(446, "success")
    assert success["metadata"]["end"] == "end_of_text"
    assert before == {p.name: p.read_bytes() for p in capture.iterdir()}
    for source, name in ((capture, "comparison"), (legacy, "legacy-comparison")):
        subprocess.run([str(Path(".venv/bin/python").absolute()), "zig/scripts/replay-transcription.py",
                        str(source), "--output", str(root / name)], check=True, timeout=60)
        comparison = json.loads((root / name / "comparison.json").read_text())
        assert comparison["comparisons"]["captured/replayed"]["tokens_equal"]
        assert all(run["token_limit_reached"] for run in comparison["runs"].values())

    # Exercise the real resident worker protocol with a sealed, too-short chunk.
    # Memfds and a private socket replace capture; no microphone or desktop API.
    audio_fd, transcript_fd = os.memfd_create("audio-test"), os.memfd_create("transcript-test")
    os.ftruncate(audio_fd, 32 + 3 * (16 + 480000 * 4))
    os.ftruncate(transcript_fd, 48 + 4096)
    with mmap.mmap(audio_fd, 0) as audio, mmap.mmap(transcript_fd, 0) as transcript:
        struct.pack_into("<IIQ", audio, 0, 8, 1, 42)
        struct.pack_into("<IIII", audio, 32, 1, 7, 1, 0)
        struct.pack_into("<f", audio, 48, 0.1234567)
        struct.pack_into("<IIQ", transcript, 0, 4, 0, 42)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent.settimeout(30)
        worker_log = journal_log(root / "worker.log")
        worker = subprocess.Popen([service, "--internal-role", "transcription-model", str(child.fileno()), str(os.getpid()), "info"],
                                  pass_fds=(child.fileno(),), stdout=subprocess.PIPE, stderr=worker_log,
                                  env=dict(os.environ, XDG_STATE_HOME=str(root / "worker-state")))
        child.close()
        try:
            parent.sendmsg([struct.pack("<HBBIQII", 9, 1, 1, 8, 42, 4, 0)],
                           [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [audio_fd, transcript_fd]))])
            ready = parent.recv(4096)
            assert len(ready) == 1136 and struct.unpack_from("<H", ready)[0] == 0
            parent.send(struct.pack("<HBB", 0, 0, 0))
            error = parent.recv(4096)
            assert struct.unpack_from("<HHH", error) == (3, 1, len("AudioTooShort"))
            assert error[40:40 + len("AudioTooShort")] == b"AudioTooShort"
            assert struct.unpack_from("<IIIIII", error, 1064) == (1, 0, 7, 1, 0, 446)
            stdout, stderr = worker.communicate(timeout=30)
            assert worker.returncode == 0, (root / "worker.log").read_text()
            assert b"Failed transcription saved" in (root / "worker.log").read_bytes()
            assert struct.unpack_from("<I", transcript, 16)[0] == 0, "failure committed text"
            saved = root / "worker-state/voiced-test/last-failed"
            metadata = replay.read_metadata(saved / "metadata.txt")
            assert metadata["session_id"] == 42 and metadata["evidence"]["chunk"] == 7
            assert metadata["model_encoder_padding_seconds"] == 10
            assert metadata["evidence"]["decoding_available"] == 0
            assert (saved / "audio.wav").read_bytes()[56:] == audio[48:52]
            assert replay.read_tokens(saved / "tokens.txt") == []
            assert (saved / "tokens.txt").read_bytes() == b"\n"
            assert metadata["end"] is None
            assert (saved / "generated.txt").read_bytes() == b""
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            parent.close()
    os.close(audio_fd)
    os.close(transcript_fd)
    print("Failed-capture replacement, permissions, exact replay, and worker diagnostics passed")
PY
    exit 0
fi

if [[ "${1:-}" == "--zig-output" ]]; then
    export VOICED_INSTANCE=test
    python3 - <<'PY'
import contextlib
from journal_fixture import journal_log
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import signal
import socket
import subprocess
import tempfile
import time

spec = importlib.util.spec_from_file_location("replay", "zig/scripts/replay-transcription.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
binary = str(Path("zig/zig-out/bin/voiced").resolve())
experiment = Path("/tmp/experiments/voiced-output-integration")
experiment.mkdir(parents=True, exist_ok=True)
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0

with tempfile.TemporaryDirectory(prefix="run-", dir=experiment) as temporary:
    root = Path(temporary)
    helpers = root / "bin"
    helpers.mkdir()
    # ELF identity is part of readiness. This fixture never accesses a desktop.
    source = r'''
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(void) {
    for (int inherited = 3; inherited < 256; ++inherited)
        if (fcntl(inherited, F_GETFD) >= 0) return 25;
    if (chdir(getenv("VOICED_COPY_FIXTURE")) != 0) return 20;
    char mode[32] = {0};
    int fd = open("mode", O_RDONLY);
    if (fd < 0 || read(fd, mode, sizeof(mode)-1) < 0) return 21;
    close(fd);
    if (!strcmp(mode, "hang")) {
        signal(SIGTERM, SIG_IGN);
        for (;;) pause();
    }
    fd = open("text", O_CREAT|O_TRUNC|O_WRONLY, 0600);
    char bytes[4096];
    ssize_t n;
    while ((n = read(0, bytes, sizeof(bytes))) > 0)
        if (write(fd, bytes, n) != n) return 22;
    close(fd);
    if (!strcmp(mode, "cancel")) return 0;
    if (!strcmp(mode, "fail")) {
        dprintf(2, "synthetic wl-copy diagnostic\n");
        memset(bytes, 'x', sizeof(bytes));
        for (int i = 0; i < 3; ++i) if (write(2, bytes, sizeof(bytes)) < 0) return 24;
        return 7;
    }
    pid_t child = fork();
    if (child < 0) return 23;
    if (child > 0) return 0;
    fd = open("owner", O_CREAT|O_TRUNC|O_WRONLY, 0600);
    dprintf(fd, "%d", getpid());
    close(fd);
    if (!strcmp(mode, "descendants")) {
        signal(SIGTERM, SIG_IGN);
        if (fork() == 0) for (;;) pause();
    }
    for (;;) pause();
}
'''
    subprocess.run(["cc", "-Wall", "-Wextra", "-Werror", "-x", "c", "-", "-o", str(helpers / "wl-copy")],
                   input=source, text=True, check=True, timeout=20)
    env = dict(os.environ, VOICED_INSTANCE="test", XDG_RUNTIME_DIR=str(root),
               XDG_CONFIG_HOME=str(root / "user-config"),
               XDG_STATE_HOME=str(root / "state"),
               VOICED_COPY_FIXTURE=str(root),
               DBUS_SESSION_BUS_ADDRESS="unix:path=" + str(root / "missing-bus"),
               WAYLAND_DISPLAY="unavailable-voiced-test", GDK_BACKEND="wayland", NO_AT_BRIDGE="1")
    def sandbox_preexec(overlay, ignored_sigchld=False):
        overlay_bytes = os.fsencode(str(overlay))
        libc = ctypes.CDLL(None, use_errno=True)
        uid, gid = os.getuid(), os.getgid()
        def preexec():
            if ignored_sigchld:
                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            if libc.unshare(0x10000000 | 0x00020000) != 0:
                os._exit(127)
            try:
                with open("/proc/self/setgroups", "w") as mapping:
                    mapping.write("deny")
                with open("/proc/self/uid_map", "w") as mapping:
                    mapping.write(f"{uid} {uid} 1")
                with open("/proc/self/gid_map", "w") as mapping:
                    mapping.write(f"{gid} {gid} 1")
            except OSError:
                os._exit(127)
            if libc.mount(overlay_bytes, b"/usr/bin/wl-copy", None, 4096, None) != 0:
                os._exit(127)
        return preexec
    for key in ("DISPLAY", "WAYLAND_SOCKET", "SWAYSOCK"):
        env.pop(key, None)
    log_path = root / "daemon.log"
    saved_transcript = root / "state/voiced-test/transcript.txt"

    def wait_until(predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError(log_path.read_text() if log_path.exists() else "condition timed out")

    def children(pid):
        path = Path(f"/proc/{pid}/task/{pid}/children")
        return {int(child) for child in path.read_text().split()} if path.exists() else set()

    def descendants(pid):
        result = children(pid)
        for child in tuple(result):
            result |= descendants(child)
        return result

    @contextlib.contextmanager
    def guardian(mode, ignored_sigchld=False, wl_copy=None):
        if wl_copy is None:
            wl_copy = helpers / "wl-copy"
        (root / "mode").write_text(mode)
        parent, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent.settimeout(3)
        with journal_log(log_path) as log:
            process = subprocess.Popen([binary, "--internal-role", "clipboard", str(worker.fileno()), str(os.getpid()), "info"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, env=env,
                pass_fds=(worker.fileno(),),
                preexec_fn=sandbox_preexec(wl_copy, ignored_sigchld))
        worker.close()
        try:
            parent.send(b"\1")
            text = "Voiced: café — 日本語\nsecond line".encode()
            process.stdin.write(text)
            process.stdin.close()
            yield process, parent, text
        finally:
            parent.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=4)
            assert not children(process.pid)

    class ClipboardError(ctypes.Structure):
        _fields_ = [("kind", ctypes.c_uint8), ("errno", ctypes.c_uint16),
                    ("wait_status_present", ctypes.c_uint8), ("wait_status", ctypes.c_uint32),
                    ("name_size", ctypes.c_uint8), ("operation_size", ctypes.c_uint8),
                    ("stderr_size", ctypes.c_uint16), ("stderr_omitted", ctypes.c_uint64),
                    ("name", ctypes.c_char * 64), ("operation", ctypes.c_char * 64),
                    ("stderr", ctypes.c_char * 2048)]

    def clipboard_error(control):
        packet = control.recv(1 + ctypes.sizeof(ClipboardError))
        assert len(packet) == 1 + ctypes.sizeof(ClipboardError) and packet[0] == 3, packet
        return ClipboardError.from_buffer_copy(packet[1:])

    print("=== Guardian readiness, exact UTF-8 and bounded descendants ===")
    for mode in ("normal", "descendants"):
        with guardian(mode) as (process, control, text):
            assert control.recv(2) == b"\1", log_path.read_text()
            assert (root / "text").read_bytes() == text
            if mode == "descendants":
                wait_until(lambda: len(descendants(process.pid)) == 2)
            owned = descendants(process.pid)
            assert owned
            control.send(b"\2")
            assert process.wait(timeout=3) == 0, log_path.read_text()
            assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
    with guardian("normal", ignored_sigchld=True) as (process, control, _):
        assert control.recv(2) == b"\1"
        owned = descendants(process.pid)
        control.close()
        assert process.wait(timeout=3) == 0
        assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
    for mode in ("fail", "cancel"):
        with guardian(mode) as (process, control, _):
            detail = clipboard_error(control)
            assert detail.wait_status_present == 1
            assert detail.wait_status == (7 << 8 if mode == "fail" else 0)
            if mode == "fail":
                assert detail.kind == 1 and detail.name == b"WlCopyLauncherFailed"
                assert detail.stderr.startswith(b"synthetic wl-copy diagnostic\n")
                assert detail.stderr_size == 2048
                assert detail.stderr_size + detail.stderr_omitted == len(b"synthetic wl-copy diagnostic\n") + 3 * 4096
            else:
                assert detail.kind == 2 and detail.name == b"NoClipboardOwner"
            assert process.wait(timeout=3) == 1
    with guardian("hang") as (process, control, _):
        wait_until(lambda: bool(children(process.pid)))
        owned = descendants(process.pid)
        process.terminate()
        assert process.wait(timeout=3) == 0, log_path.read_text()
        assert all(not Path(f"/proc/{pid}").exists() for pid in owned)

    with guardian("normal", wl_copy=Path("/dev/null")) as (process, control, _):
        detail = clipboard_error(control)
        assert detail.kind == 0 and detail.name == b"WlCopyNotFound"
        assert detail.wait_status_present == 0
        assert process.wait(timeout=3) == 1
        assert "WlCopyNotFound" in log_path.read_text()

    print("=== Exec failure preserves native errno before child exit ===")
    bogus = root / "wl-copy-bogus"
    bogus.write_text("invalid executable format\n")
    bogus.chmod(0o700)
    with guardian("normal", wl_copy=bogus) as (process, control, _):
        detail = clipboard_error(control)
        assert detail.name == b"WlCopyExecFailed" and detail.operation == b"execve", (detail.name, detail.operation, detail.errno, log_path.read_text())
        assert detail.errno == 8, (detail.errno, log_path.read_text())  # ENOEXEC
        assert detail.wait_status_present == 0  # exec failed; the child has not exited yet
        assert process.wait(timeout=3) == 1, log_path.read_text()
        assert "errno=8" in log_path.read_text(), log_path.read_text()

    print("=== Parent death cleans up with the transcript writer still open ===")
    (root / "mode").write_text("normal")
    parent, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    input_read, input_write = os.pipe()
    pid_read, pid_write = os.pipe()
    launcher = os.fork()
    if launcher == 0:
        parent.close()
        os.close(input_write)
        os.close(pid_read)
        with journal_log(log_path) as log:
            process = subprocess.Popen([binary, "--internal-role", "clipboard", str(worker.fileno()), str(os.getpid()), "info"],
                stdin=input_read, stdout=subprocess.DEVNULL, stderr=log, env=env, pass_fds=(worker.fileno(),),
                preexec_fn=sandbox_preexec(helpers / "wl-copy"))
        os.write(pid_write, str(process.pid).encode())
        os.close(pid_write)
        while True:
            signal.pause()
    worker.close()
    os.close(input_read)
    os.close(pid_write)
    orphan = int(os.read(pid_read, 32))
    os.close(pid_read)
    launcher_reaped = False
    orphan_reaped = False
    try:
        parent.send(b"\1")
        wait_until(lambda: bool(children(orphan)))
        owned = descendants(orphan)
        os.kill(launcher, signal.SIGKILL)
        os.waitpid(launcher, 0)
        launcher_reaped = True
        wait_until(lambda: os.waitpid(orphan, os.WNOHANG)[0] == orphan)
        orphan_reaped = True
        assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
    finally:
        parent.close()
        os.close(input_write)
        if not launcher_reaped:
            os.kill(launcher, signal.SIGKILL)
            os.waitpid(launcher, 0)
        if not orphan_reaped:
            if os.getsid(orphan) == orphan:
                os.killpg(orphan, signal.SIGKILL)
            else:
                os.kill(orphan, signal.SIGKILL)
            os.waitpid(orphan, 0)

    print("=== Asynchronous native notifications on a private D-Bus ===")
    notification_log = root / "notifications.jsonl"
    server_source = r'''
import json, os, sys
from gi.repository import Gio, GLib
connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
interface = Gio.DBusNodeInfo.new_for_xml(''' + '"""' + r'''
<node><interface name="org.freedesktop.Notifications">
<method name="Notify">
<arg type="s" direction="in"/><arg type="u" direction="in"/>
<arg type="s" direction="in"/><arg type="s" direction="in"/>
<arg type="s" direction="in"/><arg type="as" direction="in"/>
<arg type="a{sv}" direction="in"/><arg type="i" direction="in"/>
<arg type="u" direction="out"/>
</method><method name="CloseNotification"><arg type="u" direction="in"/></method>
<signal name="NotificationClosed"><arg type="u"/><arg type="u"/></signal>
</interface></node>
''' + '"""' + r''').interfaces[0]
mode = "normal"
next_id = 1
held = []
def log(value):
    with open(sys.argv[1], "a") as f: f.write(json.dumps(value) + "\n")
def closed(n):
    connection.emit_signal(None, "/org/freedesktop/Notifications", "org.freedesktop.Notifications",
                           "NotificationClosed", GLib.Variant("(uu)", (n, 2)))
def method(conn, sender, path, iface, name, parameters, invocation):
    global next_id
    args = parameters.unpack()
    if name == "Notify":
        n = args[1] or next_id
        next_id = max(next_id, n + 1)
        log(dict(method=name, id=n, replaces=args[1], title=args[3], body=args[4], sender=sender, owner=connection.get_unique_name()))
        if mode == "hold": held.append((invocation, n)); return
        if mode == "error":
            invocation.return_dbus_error("org.freedesktop.Notifications.Error.Failed", "fixture failure"); return
        invocation.return_value(GLib.Variant("(u)", (0 if mode == "invalid" else n,)))
    else:
        log(dict(method=name, id=args[0]))
        closed(args[0])
        invocation.return_value(None)
connection.register_object("/org/freedesktop/Notifications", interface, method, None, None)
def own():
    return Gio.bus_own_name_on_connection(connection, "org.freedesktop.Notifications",
        Gio.BusNameOwnerFlags.NONE, lambda *_: log(dict(event="ready")), None)
owner = own()
def command(fd, condition):
    global owner, mode, connection
    line = sys.stdin.readline().strip()
    if not line: return False
    if line == "release": Gio.bus_unown_name(owner)
    elif line == "acquire": owner = own()
    elif line == "reconnect":
        Gio.bus_unown_name(owner)
        connection.close_sync(None)
        connection = Gio.DBusConnection.new_for_address_sync(os.environ["DBUS_SESSION_BUS_ADDRESS"],
            Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
            None, None)
        connection.register_object("/org/freedesktop/Notifications", interface, method, None, None)
        owner = own()
    elif line == "reply":
        for invocation, n in held: invocation.return_value(GLib.Variant("(u)", (n,)))
        held.clear()
    elif line.startswith("dismiss "): closed(int(line.split()[1]))
    else: mode = line
    log(dict(command=line))
    return True
GLib.io_add_watch(sys.stdin, GLib.IO_IN, command)
GLib.MainLoop().run()
'''
    server_script = root / "notification-server.py"
    server_script.write_text(server_source)

    def notification_events():
        return [json.loads(line) for line in notification_log.read_text().splitlines()] if notification_log.exists() else []

    def notifications_sent():
        return [event for event in notification_events() if event.get("method") == "Notify"]

    def server_command(server, command):
        previous = len(notification_events())
        server.stdin.write(command + "\n")
        server.stdin.flush()
        wait_until(lambda: any(event.get("command") == command for event in notification_events()[previous:]))

    @contextlib.contextmanager
    def notification_server():
        notification_log.write_text("")
        with (root / "notification-server.log").open("w") as log:
            server = subprocess.Popen(["/usr/bin/python3", str(server_script), str(notification_log)],
                env=env, stdin=subprocess.PIPE, stdout=log, stderr=log, text=True)
        try:
            wait_until(lambda: any(e.get("event") == "ready" for e in notification_events()) or server.poll() is not None)
            assert server.poll() is None, (root / "notification-server.log").read_text()
            yield server
        finally:
            server.terminate()
            server.wait(timeout=3)
            server.stdin.close()

    print("=== Paste syscall errors preserve chord progress ===")
    paste_source = root / "paste-errors.zig"
    paste_source.write_text(r'''
const std = @import("std");
const paste = @import("paste");
const linux = std.os.linux;
const assert = std.debug.assert;
pub fn main() void {
    var keyboard: paste.Keyboard = .{ .fd = -1, .usable_after_ns = 0 };
    keyboard.beginPaste(.@"ctrl+v", 0, 0);
    const failed = keyboard.advance(0).err.write;
    assert(failed.errno == .BADF);
    assert(failed.progress.chord == .@"ctrl+v" and failed.progress.frame == 0 and failed.progress.frame_bytes_sent == 0);
    assert(keyboard.pending != null); // uncertainty remains until the caller cleans up

    var pipe: [2]i32 = undefined;
    assert(linux.errno(linux.pipe2(&pipe, .{ .NONBLOCK = true, .CLOEXEC = true })) == .SUCCESS);
    defer _ = linux.close(pipe[0]);
    defer _ = linux.close(pipe[1]);
    keyboard = .{ .fd = pipe[1], .usable_after_ns = 0 };
    keyboard.beginPaste(.@"ctrl+shift+v", 0, 0);
    assert(!keyboard.advance(0).ok); // first frame accepted
    var bytes: [4096]u8 = @splat(0);
    while (linux.errno(linux.write(pipe[1], &bytes, bytes.len)) == .SUCCESS) {}
    // Fill remaining capacity with whole input-sized writes as well.
    while (linux.errno(linux.write(pipe[1], &bytes, 1)) == .SUCCESS) {}
    assert(!keyboard.advance(0).ok); // EAGAIN must not change progress
    const timeout = keyboard.advance(200 * std.time.ns_per_ms).err.timed_out;
    assert(timeout.progress.frame == 1 and timeout.progress.frame_bytes_sent == 0);
    assert(timeout.progress.chord == .@"ctrl+shift+v");
    assert(timeout.deadline_ns == 200 * std.time.ns_per_ms and timeout.observed_ns == timeout.deadline_ns);
}
''')
    paste_driver = root / "paste-errors"
    subprocess.run(["zig", "build-exe", "-O", "ReleaseSafe", "--dep", "paste", "-Mroot=" + str(paste_source),
                    "-Mpaste=" + str(Path("zig/src/paste_keyboard.zig").resolve()), "-femit-bin=" + str(paste_driver)],
                   check=True, capture_output=True, timeout=40)
    subprocess.run([str(paste_driver)], check=True, capture_output=True, timeout=3)

    driver_binary = root / "notification-driver"
    subprocess.run(["zig", "build-exe", "-OReleaseSafe", "zig/src/notification_check.zig",
                    "-femit-bin=" + str(driver_binary)], check=True, timeout=90)

    @contextlib.contextmanager
    def notification_driver():
        with journal_log(log_path) as log:
            driver = subprocess.Popen([str(driver_binary)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        try:
            yield driver
        finally:
            driver.terminate()
            driver.wait(timeout=3)
            driver.stdin.close()
            driver.stdout.close()

    def driver_command(driver, command):
        import select
        driver.stdin.write(command.encode())
        driver.stdin.flush()
        assert select.select([driver.stdout], [], [], 0.5)[0], "notification I/O blocked control"
        assert os.read(driver.stdout.fileno(), 3) == b"ok\n", log_path.read_text()

    bus = subprocess.Popen(["dbus-daemon", "--session", "--nofork", "--print-address"], stdout=subprocess.PIPE, text=True)
    env["DBUS_SESSION_BUS_ADDRESS"] = bus.stdout.readline().strip()
    try:
        with notification_server() as server, notification_driver() as driver:
            driver_command(driver, "a")
            wait_until(lambda: len(notifications_sent()) == 1)
            wait_until(lambda: "Desktop notification accepted:" in log_path.read_text())
            first = notifications_sent()[0]
            assert first["replaces"] == 0 and "microphone" in first["title"]
            driver_command(driver, "a")
            time.sleep(0.05)
            assert len(notifications_sent()) == 1
            driver_command(driver, "b")
            wait_until(lambda: len(notifications_sent()) == 2)
            assert notifications_sent()[-1]["replaces"] == first["id"]
            server_command(server, "dismiss " + str(first["id"]))
            time.sleep(0.05)
            driver_command(driver, "c")
            wait_until(lambda: len(notifications_sent()) == 3)
            assert notifications_sent()[-1]["replaces"] == 0
            driver_command(driver, "r")
            wait_until(lambda: any(e.get("method") == "CloseNotification" for e in notification_events()))

            server_command(server, "hold")
            driver_command(driver, "a")
            wait_until(lambda: len(notifications_sent()) == 4)
            driver_command(driver, "b")
            driver_command(driver, "c")
            time.sleep(0.05)
            assert len(notifications_sent()) == 4
            server_command(server, "normal")
            server_command(server, "reply")
            wait_until(lambda: len(notifications_sent()) == 5)
            assert "save failed" in notifications_sent()[-1]["title"]
            assert notifications_sent()[-1]["replaces"] == notifications_sent()[-2]["id"]

            # Recovery while Notify is outstanding must close its late ID.
            server_command(server, "hold")
            driver_command(driver, "a")
            wait_until(lambda: len(notifications_sent()) == 6)
            driver_command(driver, "r")
            closes = sum(e.get("method") == "CloseNotification" for e in notification_events())
            server_command(server, "reply")
            wait_until(lambda: sum(e.get("method") == "CloseNotification" for e in notification_events()) > closes)

            driver_command(driver, "b")
            wait_until(lambda: len(notifications_sent()) == 7)
            for _ in range(20): driver_command(driver, "b")
            wait_until(lambda: "org.freedesktop.DBus.Error.NoReply" in log_path.read_text(), timeout=3)
            assert len(notifications_sent()) == 7
            server_command(server, "reply")  # An expired reply cannot restore its ID.
            server_command(server, "normal")
            driver_command(driver, "c")
            wait_until(lambda: len(notifications_sent()) == 8)
            assert notifications_sent()[-1]["replaces"] == 0

            old_owner = notifications_sent()[-1]["owner"]
            server_command(server, "reconnect")
            wait_until(lambda: len(notifications_sent()) == 9)
            assert notifications_sent()[-1]["replaces"] == 0
            assert notifications_sent()[-1]["owner"] != old_owner
            server_command(server, "error")
            driver_command(driver, "a")
            wait_until(lambda: "org.freedesktop.Notifications.Error.Failed" in log_path.read_text())
            driver_command(driver, "a")
            time.sleep(0.05)
            assert len(notifications_sent()) == 10
            server_command(server, "invalid")
            driver_command(driver, "b")
            wait_until(lambda: "invalid reply" in log_path.read_text())
            # Each recording can report the same error, replacing the old ID.
            server_command(server, "normal")
            prior = len(notifications_sent())
            driver_command(driver, "n")
            driver_command(driver, "b")
            wait_until(lambda: len(notifications_sent()) == prior + 1)
            first = notifications_sent()[-1]
            driver_command(driver, "n")
            time.sleep(0.05)
            assert len(notifications_sent()) == prior + 1  # Reset itself is silent.
            driver_command(driver, "b")
            wait_until(lambda: len(notifications_sent()) == prior + 2)
            assert notifications_sent()[-1]["replaces"] == first["id"]
            for _ in range(10): driver_command(driver, "b")
            time.sleep(0.05)
            assert len(notifications_sent()) == prior + 2
            assert not children(driver.pid)
            assert len(list(Path(f"/proc/{driver.pid}/task").iterdir())) == 1
            bus.terminate()
            bus.wait(timeout=3)
            wait_until(lambda: "Desktop notifications disconnected:" in log_path.read_text())
            driver_command(driver, "b")
    finally:
        if bus.poll() is None:
            bus.terminate()
            bus.wait(timeout=3)
        bus.stdout.close()
        env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=" + str(root / "missing-bus")

    # A bus socket that accepts connections but never authenticates must not
    # block the caller or cause pending error requests to grow without bound.
    stalled_bus = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stalled_bus.bind(str(root / "stalled-bus"))
    stalled_bus.listen(1)
    stalled_bus.settimeout(3)
    env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=" + str(root / "stalled-bus")
    try:
        with notification_driver() as driver:
            peer, _ = stalled_bus.accept()
            try:
                for _ in range(30):
                    for command in "abcr": driver_command(driver, command)
                assert not children(driver.pid)
            finally:
                peer.close()
    finally:
        stalled_bus.close()
        env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=" + str(root / "missing-bus")

    print("=== Production options and idle control responsiveness ===")

    def cli(*args):
        result = subprocess.run([binary, *args], env=env, capture_output=True, text=True, timeout=3)
        assert result.returncode == 0, (args, result.stderr, log_path.read_text())
        if args != ("status",):
            assert result.stdout == "", (args, result.stdout)
            return None
        status = dict(field.split("=", 1) for field in result.stdout.split())
        for field in ("session_id", "model_keep_warm_seconds"):
            status[field] = int(status[field])
        return status

    @contextlib.contextmanager
    def service(*options):
        with journal_log(log_path) as log:
            daemon = subprocess.Popen([binary, "serve", *options], env=env, stdout=log, stderr=log,
                                       preexec_fn=sandbox_preexec(helpers / "wl-copy"))
        try:
            wait_until(lambda: "Supervisor service ready" in log_path.read_text() or daemon.poll() is not None)
            assert daemon.poll() is None, log_path.read_text()
            yield daemon
            owned = descendants(daemon.pid)
            cli("kill")
            assert daemon.wait(timeout=5) == 0, log_path.read_text()
            assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
            assert log_path.read_text().count("Effective config:") == 1
        except BaseException:
            print(log_path.read_text())
            raise
        finally:
            if daemon.poll() is None:
                daemon.terminate()
                daemon.wait(timeout=5)

    for args in (("--transcript-output", "invalid"), ("--paste-shortcut", "invalid"),
                 ("--transcript-output", "stdout", "--transcript-output", "stdout")):
        result = subprocess.run([binary, "serve", *args], env=env, capture_output=True, timeout=3)
        assert result.returncode == 2, result
    with service("--transcript-output", "stdout") as daemon:
        assert cli("status")["phase"] == "idle"
        startup = log_path.read_text()
        for setting in ("model=Systran/faster-whisper-small.en", "model_encoder_threads=4",
                        "model_decoder_threads=4", "model_encoder_padding_seconds=10",
                        "model_idle_seconds_max=300", "recording_seconds_max=3600",
                        "transcript_output=stdout", "paste_shortcut=ctrl+shift+v",
                        "paste_settle_ms=10", "paste_key_gap_ms=4", 'microphone="default"'):
            assert setting in startup, startup
        assert not children(daemon.pid)
        cli("stop")
        assert cli("status")["phase"] == "idle"

    print("=== Native file configuration and CLI precedence ===")
    settings = root / "settings.conf"
    def invalid_config(text, expected, line=1):
        settings.write_text(text)
        result = subprocess.run([binary, "serve", "--config", str(settings)],
                                env=env, capture_output=True, text=True, timeout=3)
        assert result.returncode == 2, result
        assert expected in result.stderr, result.stderr
        if line:
            assert f"{settings}:{line}:" in result.stderr, result.stderr
        assert not (root / "voiced-test/control.sock").exists()

    invalid_config("# comment\nmodel_encoder_threads 8\n", "expected key=value", 2)
    invalid_config("threads=8\n", "unknown option")
    invalid_config("model-encoder-threads=8\n", "unknown option")
    invalid_config("config=other.conf\n", "unknown option")
    invalid_config("model_encoder_threads=8\nmodel_encoder_threads=8\n", "more than once", 2)
    invalid_config("microphone_node=one\nmicrophone_serial=two\n", "cannot be used together", 2)
    invalid_config("microphone_node=bad\0name\n", "invalid value")
    for key, value in {
        "model": "unknown", "model_encoder_threads": "0", "model_decoder_threads": "33",
        "model_encoder_padding_seconds": "7", "model_idle_seconds_max": "-1",
        "recording_seconds_max": "0", "microphone_node": "", "microphone_serial": "",
        "transcript_output": "unknown", "paste_shortcut": "unknown", "notification_mode": "unknown",
    }.items():
        invalid_config(f"{key}={value}\n", "invalid value")
        flag = "--" + key.replace("_", "-")
        result = subprocess.run([binary, "serve", flag, value], env=env,
                                capture_output=True, text=True, timeout=3)
        assert result.returncode == 2 and "invalid value" in result.stderr, result
    invalid_config("model_encoder_threads=4\nmodel_decoder_threads=8\n", "no greater", line=0)
    # Cross-setting constraints are checked after overrides, independent of order.
    with service("--config", str(settings), "--model-encoder-threads=8", "--transcript-output=stdout"):
        assert cli("status")["phase"] == "idle"
        startup = log_path.read_text()
        assert "model_encoder_threads=8, model_decoder_threads=8" in startup, startup
    for key in ("paste_settle_ms", "paste_key_gap_ms"):
        flag = "--" + key.replace("_", "-")
        for value in ("-1", "65536", "4ms", "1.5", ""):
            invalid_config(f"{key}={value}\n", "invalid value")
            result = subprocess.run([binary, "serve", flag + "=" + value], env=env,
                                    capture_output=True, text=True, timeout=3)
            assert result.returncode == 2 and "invalid value" in result.stderr, result
        invalid_config(f"{key}=4\n{key}=4\n", "more than once", 2)
        result = subprocess.run([binary, "serve", flag, "4", flag + "=4"], env=env,
                                capture_output=True, text=True, timeout=3)
        assert result.returncode == 2 and "more than once" in result.stderr, result
    settings.write_text("paste_settle_ms=50\npaste_key_gap_ms=8\ntranscript_output=stdout\n")
    with service("--config", str(settings)):
        startup = log_path.read_text()
        assert "paste_settle_ms=50, paste_key_gap_ms=8" in startup, startup
    with service("--config", str(settings), "--paste-settle-ms=0", "--paste-key-gap-ms", "65535"):
        startup = log_path.read_text()
        assert "paste_settle_ms=0, paste_key_gap_ms=65535" in startup, startup
    with service("--config", str(settings), "--paste-settle-ms", "65535", "--paste-key-gap-ms=0"):
        startup = log_path.read_text()
        assert "paste_settle_ms=65535, paste_key_gap_ms=0" in startup, startup

    settings.write_text("notification_mode=off\ntranscript_output=clipboard\n")
    with service("--config", str(settings)):
        assert "notification_mode=off" in log_path.read_text()
        assert "Desktop notifications unavailable" not in log_path.read_text()
    with service("--config", str(settings), "--notification-mode=errors"):
        assert "notification_mode=errors" in log_path.read_text()
        assert "Desktop notifications unavailable" in log_path.read_text()
        assert cli("status")["phase"] == "idle"
    with service("--config", str(settings), "--notification-mode=errors", "--transcript-output=stdout"):
        assert "notification_mode=off" in log_path.read_text()
        assert "Desktop notifications unavailable" not in log_path.read_text()
    invalid_config("notification_mode=off\nnotification_mode=errors\n", "more than once", 2)

    result = subprocess.run([binary, "serve", "--config", str(root / "absent")],
                            env=env, capture_output=True, text=True, timeout=3)
    assert result.returncode == 2 and "FileNotFound" in result.stderr, result
    settings.write_text("invalid config")
    result = subprocess.run([binary, "serve", "--config", str(settings), "--help"],
                            env=env, capture_output=True, text=True, timeout=3)
    assert result.returncode == 0 and "--model-decoder-threads" in result.stdout, result
    assert "--paste-settle-ms" in result.stdout and "default: 10 ms" in result.stdout, result
    assert "--paste-key-gap-ms" in result.stdout and "default: 4 ms" in result.stdout, result
    for flag in ("--threads", "--seconds", "--output", "--target", "--paste-key",
                 "--device-serial", "--encoder-trailing-padding-seconds", "--model-keep-warm-seconds"):
        result = subprocess.run([binary, "serve", flag, "8"], env=env,
                                capture_output=True, text=True, timeout=3)
        assert result.returncode == 2 and "unknown option" in result.stderr, result
    settings.write_bytes(b"  # comment\r\n\r\n model_encoder_threads = 8 \r\nmodel_decoder_threads=2\r\n"
                         b"microphone_node = literal # and = name \r\ntranscript_output=stdout")
    with service("--config", str(settings), "--microphone-serial", "override"):
        assert cli("status")["phase"] == "idle"
        startup = log_path.read_text()
        assert "model_encoder_threads=8, model_decoder_threads=2" in startup, startup
        assert 'microphone_serial="override"' in startup and "literal #" not in startup, startup
    # Default discovery and explicit-path replacement are observable at startup.
    default_config = Path(env["XDG_CONFIG_HOME"]) / "voiced/config"
    default_config.parent.mkdir(parents=True)
    default_config.write_text("unknown=1")
    result = subprocess.run([binary, "serve"], env=env, capture_output=True, text=True, timeout=3)
    assert result.returncode == 2 and str(default_config) in result.stderr, result
    with service("--config", str(settings)):
        assert cli("status")["phase"] == "idle"
    default_config.write_text("transcript_output=stdout\n")
    with service():
        assert cli("status")["phase"] == "idle"
    default_config.unlink()

    if os.environ.get("VOICED_ZIG_MODEL_TESTS") == "1":
        print("=== Private microphone graph, real inference and service delivery ===")
        saved_transcript.parent.mkdir(parents=True)
        saved_transcript.parent.chmod(0o775)
        config = root / "config"
        config.mkdir()
        pipewire_config = Path("/usr/share/pipewire/pipewire.conf").read_text().replace("context.objects = [", '''context.objects = [
    { factory = adapter args = { factory.name = support.null-audio-sink
      node.name = voiced-test-source node.description = "Voiced fixture source"
      media.class = "Audio/Source/Virtual" audio.position = [ MONO ]
      monitor.passthrough = true } }''')
        # The private graph has no realtime scheduling. Use an 85 ms cycle at
        # 48 kHz for long multi-chunk runs, below capture's 100 ms block bound.
        pipewire_config = pipewire_config.replace("context.properties = {", """context.properties = {
    default.clock.rate = 48000
    default.clock.quantum = 4096
    default.clock.min-quantum = 4096
    default.clock.max-quantum = 4096""")
        (config / "pipewire-test.conf").write_text(pipewire_config)
        wireplumber_config = Path("/usr/share/wireplumber/wireplumber.conf").read_text()
        wireplumber_config = "\n".join(line for line in wireplumber_config.splitlines()
            if not any(module in line for module in ("alsa.lua", "bluez.lua", "bluetooth.lua", "v4l2.lua")))
        (config / "wireplumber-test.conf").write_text(wireplumber_config)
        env.update(PIPEWIRE_RUNTIME_DIR=str(root), XDG_CACHE_HOME=str(root / "cache"),
                   XDG_CONFIG_HOME=str(root / "config-home"), XDG_STATE_HOME=str(root / "state"), GIO_USE_VFS="local")
        processes = []
        notification_stack = contextlib.ExitStack()
        try:
            bus = subprocess.Popen(["dbus-daemon", "--session", "--nofork", "--print-address"], stdout=subprocess.PIPE, text=True)
            processes.append(bus)
            address = bus.stdout.readline().strip()
            assert address
            env.update(DBUS_SESSION_BUS_ADDRESS=address, DBUS_SYSTEM_BUS_ADDRESS=address)
            desktop_server = notification_stack.enter_context(notification_server())
            with (root / "pipewire.log").open("w") as log:
                processes.append(subprocess.Popen(["pipewire", "-c", "pipewire-test.conf"],
                    env=dict(env, PIPEWIRE_CONFIG_DIR=str(config)), stdout=log, stderr=log))
            wait_until(lambda: (root / "pipewire-0").exists())
            with (root / "wireplumber.log").open("w") as log:
                processes.append(subprocess.Popen(["wireplumber", "-c", str(config / "wireplumber-test.conf")],
                    env=env, stdout=log, stderr=log))
            time.sleep(1)
            assert all(process.poll() is None for process in processes)

            stop_requests = {}
            def record_fixture(two_chunks=False, stop_command=("stop",), automatic=False, pause_model=False):
                cli("listen" if automatic else "record")
                wait_until(lambda: cli("status")["model"] == "warm", timeout=15)
                recording = cli("status")["session_id"]
                if pause_model:
                    model_pid = next(pid for pid in children(daemon.pid) if b"transcription-model" in Path(f"/proc/{pid}/cmdline").read_bytes())
                    os.kill(model_pid, signal.SIGSTOP)
                time.sleep(0.5)
                for play_index in range(2 if two_chunks else 1):
                    if play_index:
                        # Wait for the first natural boundary, then put actual
                        # speech in the final chunk instead of pure silence.
                        wait_until(lambda: f"Transcription chunk: recording_ordinal={recording}, chunk_ordinal=0," in log_path.read_text(), timeout=30)
                    playback = subprocess.Popen(["pw-cat", "--playback", "--target", "0", "--properties",
                        "node.name=voiced-fixture-playback", "test-fixtures/hello_world.wav"],
                        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    try:
                        for _ in range(100):
                            outputs = subprocess.run(["pw-link", "-o"], env=env, capture_output=True, text=True, timeout=2).stdout.splitlines()
                            inputs = subprocess.run(["pw-link", "-i"], env=env, capture_output=True, text=True, timeout=2).stdout.splitlines()
                            source = [p.strip() for p in outputs if p.strip().startswith("voiced-fixture-playback:")]
                            target = [p.strip() for p in inputs if p.strip().startswith("voiced-test-source:")]
                            if source and target:
                                break
                            time.sleep(0.01)
                        assert len(source) == len(target) == 1, (outputs, inputs)
                        subprocess.run(["pw-link", source[0], target[0]], env=env, check=True, capture_output=True, timeout=2)
                        assert playback.wait(timeout=10) == 0, playback.stderr.read()
                    finally:
                        if playback.poll() is None:
                            playback.kill()
                            playback.wait()
                if not automatic:
                    stop_requests[recording] = time.monotonic_ns()
                    cli(*stop_command)
                    cli("stop")  # A repeated stop must not restart the timer.

            def assert_recording_metrics(recording, expected_chunks, stop_origin="command"):
                logs = log_path.read_text()
                def events(label):
                    prefix = f"{label}: recording_ordinal={recording}, "
                    return [dict(field.split("=", 1) for field in line.split(f"{label}: ", 1)[1].split(", "))
                            for line in logs.splitlines() if prefix in line]

                chunks = events("Transcription chunk")
                assert len(chunks) == expected_chunks, logs
                assert [int(chunk["chunk_ordinal"]) for chunk in chunks] == list(range(expected_chunks)), logs
                samples = processing_ms = accepted_output_size = 0
                for chunk in chunks:
                    audio_seconds = int(chunk["audio_samples_count"]) / 16000
                    elapsed_ms = float(chunk["transcription_compute_duration_ms"])
                    assert abs(float(chunk["audio_duration_seconds"]) - audio_seconds) <= 0.00051, chunk
                    assert abs(elapsed_ms - float(chunk["features_duration_ms"]) -
                               float(chunk["inference_duration_ms"])) <= 0.002, chunk
                    assert abs(float(chunk["transcription_compute_speed_ratio"]) - audio_seconds * 1000 / elapsed_ms) < 0.02, chunk
                    samples += int(chunk["audio_samples_count"])
                    processing_ms += elapsed_ms
                    if chunk["disposition"] == "accepted":
                        accepted_output_size += int(chunk["transcript_size"])
                summaries = events("Transcription complete")
                assert len(summaries) == 1, logs
                summary = summaries[0]
                assert int(summary["chunks_count"]) == expected_chunks, summary
                assert int(summary["chunks_accepted_count"]) + int(summary["chunks_no_speech_count"]) == expected_chunks, summary
                assert abs(float(summary["audio_duration_seconds"]) - samples / 16000) <= 0.00051, summary
                assert abs(float(summary["transcription_compute_duration_ms"]) - processing_ms) <= 0.002, summary
                assert abs(float(summary["transcription_compute_speed_ratio"]) - samples / 16000 * 1000 / processing_ms) < 0.02, summary
                copied = (root / "text").read_bytes()
                assert saved_transcript.read_bytes() == copied
                assert saved_transcript.stat().st_mode & 0o777 == 0o600
                assert saved_transcript.parent.stat().st_mode & 0o777 == 0o700
                assert len(events("Transcript saved")) == 1, logs
                assert logs.index(f"Clipboard acquired: recording_ordinal={recording},") < logs.index(f"Transcript saved: recording_ordinal={recording},"), logs
                assert int(summary["transcript_size"]) == len(copied) <= accepted_output_size, summary
                acquisitions = events("Clipboard acquired")
                assert len(acquisitions) == 1, logs
                acquired = acquisitions[0]
                assert 0 <= float(acquired["clipboard_acquire_duration_ms"]) <= float(acquired["recording_stop_elapsed_ms"]), acquired
                capture_started = events("Capture started")
                assert len(capture_started) == 1 and float(capture_started[0]["capture_start_duration_ms"]) >= 0, logs
                saved = events("Transcript saved")[0]
                assert float(saved["transcript_save_duration_ms"]) >= 0, saved
                native = [json.loads(line) for line in log_path.with_suffix(".journal.jsonl").read_text().splitlines()]
                assert any(row.get("VOICED_RECORDING_ORDINAL") == str(recording) and row["MESSAGE"].startswith("Clipboard acquired:") for row in native), native
                assert acquired["transcript_size"] == str(len(copied)), acquired
                assert summary["recording_stop_origin"] == acquired["recording_stop_origin"] == stop_origin, logs
                completed_ms = float(summary["recording_stop_elapsed_ms"])
                acquired_ms = float(acquired["recording_stop_elapsed_ms"])
                assert 0 <= completed_ms <= acquired_ms, logs
                if stop_origin == "command":
                    assert acquired_ms <= (time.monotonic_ns() - stop_requests[recording]) / 1e6, logs
                    assert len(events("Recording stop requested")) == 1, logs
                else:
                    assert not events("Recording stop requested"), logs
                    assert len(events("Recording stop observed")) == 1, logs
                assert copied.strip() and copied.decode().strip() not in logs, logs
                assert f"Capture ended: recording_ordinal={recording}, " in logs, logs

            settings.write_text("model=Systran/faster-whisper-base.en\n"
                                "model_encoder_threads=8\nmodel_decoder_threads=2\n"
                                "model_encoder_padding_seconds=10\nmodel_idle_seconds_max=30\n"
                                "recording_seconds_max=20\ntranscript_output=stdout\n"
                                "paste_shortcut=ctrl+v\nmicrophone_node=voiced-test-source\n")
            options = ("--config", str(settings), "--transcript-output", "clipboard")
            with service(*options) as daemon:
                def guardian_pid():
                    return next(pid for pid in children(daemon.pid)
                                if b"clipboard" in Path(f"/proc/{pid}/cmdline").read_bytes())

                (root / "mode").write_text("normal")
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert "hello" in (root / "text").read_text().lower(), log_path.read_text()
                assert_recording_metrics(1, 1)
                first = guardian_pid()
                owned = descendants(first)
                first_fds = len(list(Path(f"/proc/{daemon.pid}/fd").iterdir()))
                saved_inode = saved_transcript.stat().st_ino
                (root / "mode").write_text("cancel")
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert guardian_pid() == first
                assert all(Path(f"/proc/{pid}").exists() for pid in owned)
                assert saved_transcript.read_bytes() == (root / "text").read_bytes()
                assert saved_transcript.stat().st_ino != saved_inode
                assert "Transcript saved: recording_ordinal=2," in log_path.read_text()
                wait_until(lambda: "Desktop notification accepted: problem=clipboard_failed" in log_path.read_text())
                assert notifications_sent()[-1]["title"] == "Voiced: copy failed"
                assert "was saved" in notifications_sent()[-1]["body"]
                (root / "mode").write_text("descendants")
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert_recording_metrics(3, 1)
                wait_until(lambda: any(e.get("method") == "CloseNotification" for e in notification_events()))
                second = guardian_pid()
                assert second != first and not Path(f"/proc/{first}").exists()
                assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
                assert len(list(Path(f"/proc/{daemon.pid}/fd").iterdir())) == first_fds
                owned = descendants(second)
                os.kill(second, signal.SIGKILL)
                wait_until(lambda: not Path(f"/proc/{second}").exists())
                native = [json.loads(line) for line in log_path.with_suffix(".journal.jsonl").read_text().splitlines()]
                assert any(row["PRIORITY"] == "3" and "Worker exited: role=clipboard" in row["MESSAGE"] and f"pid={second}," in row["MESSAGE"] and "signal=9" in row["MESSAGE"] for row in native), native
                assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
                assert cli("status")["model"] == "warm"
                (root / "mode").write_text("hang")
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "delivering", timeout=15)
                recording = cli("status")["session_id"]
                cli("record", "-t")
                assert cli("status")["phase"] == "delivering"
                assert cli("status")["session_id"] == recording
                pending = guardian_pid()
                wait_until(lambda: cli("status")["phase"] == "idle")
                assert not Path(f"/proc/{pending}").exists()
                assert "stage = .acquisition" in log_path.read_text() and "bytes_sent" in log_path.read_text()
                assert "Transcript saved: recording_ordinal=4," in log_path.read_text()
                wait_until(lambda: len(notifications_sent()) == 2)
                saved_inode = saved_transcript.stat().st_ino
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "delivering", timeout=15)
                cli("cancel")
                wait_until(lambda: cli("status")["phase"] == "idle")
                assert len(children(daemon.pid)) == 1
                assert saved_transcript.stat().st_ino == saved_inode
                # A frozen guardian cannot perform cooperative cleanup. Its
                # whole session is killed while its numeric PID stays reserved.
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "delivering", timeout=15)
                frozen = guardian_pid()
                wait_until(lambda: bool(children(frozen)))
                owned = descendants(frozen)
                os.kill(frozen, signal.SIGSTOP)
                cli("cancel")
                wait_until(lambda: cli("status")["phase"] == "idle")
                assert not Path(f"/proc/{frozen}").exists()
                assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
                assert saved_transcript.stat().st_ino == saved_inode
                assert sorted(p.name for p in saved_transcript.parent.iterdir()) == ["transcript.txt"]
                assert len(notifications_sent()) == 2  # Cancellation never creates a popup.
                assert [line.split("recording_ordinal=", 1)[1] for line in log_path.read_text().splitlines()
                        if "Recording requested: " in line] == [str(i) for i in range(1, 7)]

            print("=== Failed save preserves clipboard and cleans temporary file ===")
            previous = saved_transcript.with_name("previous.txt")
            saved_transcript.rename(previous)
            saved_transcript.mkdir()
            (saved_transcript / "marker").write_text("preserve")
            (root / "mode").write_text("normal")
            with service(*options):
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                logs = log_path.read_text()
                assert "hello" in (root / "text").read_text().lower(), logs
                assert "Transcript save error: recording_ordinal=1," in logs, logs
                wait_until(lambda: "Desktop notification accepted: problem=transcript_save_failed" in log_path.read_text())
                assert notifications_sent()[-1]["title"] == "Voiced: transcript save failed"
                assert logs.index("Clipboard acquired:") < logs.index("Transcript save error:"), logs
                assert (saved_transcript / "marker").read_text() == "preserve"
                assert sorted(p.name for p in saved_transcript.parent.iterdir()) == ["previous.txt", "transcript.txt"]
            (saved_transcript / "marker").unlink()
            saved_transcript.rmdir()
            previous.rename(saved_transcript)

            print("=== Diagnostic stdout does not replace saved transcript ===")
            saved_inode = saved_transcript.stat().st_ino
            with service("--config", str(settings)):
                record_fixture()
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert "Transcript written: recording_ordinal=1," in log_path.read_text()
                assert "Transcript saved:" not in log_path.read_text()
                assert saved_transcript.stat().st_ino == saved_inode

            print("=== Multi-chunk metrics and recording ordinal reset ===")
            (root / "mode").write_text("normal")
            with service(*options, "--recording-seconds-max", "60"):
                record_fixture(two_chunks=True)
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert_recording_metrics(1, 2)
                record_fixture(stop_command=("record", "-t"))
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert_recording_metrics(2, 1)
                record_fixture(automatic=True)
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert_recording_metrics(3, 1, stop_origin="capture_end")

            print("=== Cancellation severity and explicit transcription retry ===")
            with service(*options, "--log-level=debug") as daemon:
                cli("record")
                wait_until(lambda: "Capture started: recording_ordinal=1," in log_path.read_text())
                cli("cancel")
                wait_until(lambda: cli("status")["phase"] == "idle")
                native = [json.loads(line) for line in log_path.with_suffix(".journal.jsonl").read_text().splitlines()]
                assert any(row["PRIORITY"] == "6" and "Recording discarded: recording_ordinal=1, reason=user_cancelled" in row["MESSAGE"] for row in native), native
                assert any(row["PRIORITY"] == "7" and "Worker exited:" in row["MESSAGE"] and "expected=true" in row["MESSAGE"] for row in native), native
                record_fixture(pause_model=True)
                model_pid = next(pid for pid in children(daemon.pid) if b"transcription-model" in Path(f"/proc/{pid}/cmdline").read_bytes())
                os.kill(model_pid, signal.SIGKILL)
                wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                assert_recording_metrics(2, 1)
                native = [json.loads(line) for line in log_path.with_suffix(".journal.jsonl").read_text().splitlines()]
                assert any(row["PRIORITY"] == "4" and "Transcription retry: recording_ordinal=2, chunk_ordinal=0" in row["MESSAGE"] and "attempt=2" in row["MESSAGE"] for row in native), native
                assert any(row["PRIORITY"] == "3" and "Worker exited: role=transcription" in row["MESSAGE"] and "signal=9" in row["MESSAGE"] for row in native), native
                assert "model_cache_load_duration_ms=" in log_path.read_text()
                assert "model_runtime_init_duration_ms=" in log_path.read_text()


            print("=== Capacity stops deliver partial text, save it, and warn ===")
            # Compile reduced limits into a private copy, leaving shipped limits
            # and CLI untouched. Only device creation is replaced: real keyboard
            # event serialization goes to a file, never the host's /dev/uinput.
            fixture = root / "limit-build"
            shutil.copytree("zig/src", fixture / "src")
            for name in ("build.zig", "build.zig.zon"):
                shutil.copy2(Path("zig") / name, fixture / name)
            for name in ("runtime", "scripts"):
                (fixture / name).symlink_to((Path("zig") / name).resolve(), target_is_directory=True)
            def replace_source(name, old, new):
                path = fixture / "src" / name
                source = path.read_text()
                assert source.count(old) == 1, (name, old)
                path.write_text(source.replace(old, new))
            replace_source("transcription_process.zig", "    const policy: inference.Policy = .{", """
    const fixture_mode = init.environ_map.get("VOICED_LIMIT_FIXTURE") orelse "";
    const fixture_tokens = std.mem.startsWith(u8, fixture_mode, "tokens") or std.mem.eql(u8, fixture_mode, "both");
    const fixture_chunk = std.mem.startsWith(u8, fixture_mode, "chunk") or std.mem.eql(u8, fixture_mode, "both");
    const policy: inference.Policy = .{""")
            replace_source("transcription_process.zig", ".generated_tokens_count_max = 446,",
                           ".generated_tokens_count_max = if (fixture_tokens) 2 else 446,")
            replace_source("transcription_process.zig", "const generated_text = runtime.transcribe(samples, &transcript.bytes,",
                           "var generated_text = runtime.transcribe(samples, transcript.bytes[0..if (fixture_chunk) @as(usize, 5) else transcript.bytes.len],")
            replace_source("transcription_process.zig", "        if (decoded) |value| {", r'''
        if (std.mem.endsWith(u8, fixture_mode, "unicode")) {
            const unicode = " ab\xe2\x82\xacxy";
            @memcpy(transcript.bytes[0..unicode.len], unicode);
            decoded.?.text = transcript.bytes[0..if (fixture_chunk) @as(usize, 5) else unicode.len];
            if (generated_text != null) generated_text = decoded;
        }
        // Even disagreement/repetition must retain text for a limited result.
        if (fixture_tokens or fixture_chunk) decoded.?.no_speech_probability = 0.99;
        if (decoded) |value| {
''')
            replace_source("transcription_process.zig", "        if (error_message) |message| {", """
        if (std.mem.eql(u8, fixture_mode, "tokens_exit")) std.process.exit(9);
        if (std.mem.eql(u8, fixture_mode, "tokens_stall")) _ = linux.kill(linux.getpid(), linux.SIG.STOP);
        if (error_message) |message| {""")
            replace_source("supervisor.zig", "const transcript_bytes_capacity = sessionTranscriptBytesCapacity(configuration);",
                'const transcript_bytes_capacity = if (std.mem.startsWith(u8, init.environ_map.get("VOICED_LIMIT_FIXTURE") orelse "", "recording")) @as(usize, 5) else sessionTranscriptBytesCapacity(configuration);')
            keys = root / "paste-events"
            keyboard = fixture / "src/paste_keyboard.zig"
            source = keyboard.read_text()
            start = source.index("    pub fn open(")
            end = source.index("    pub fn beginPaste(", start)
            keyboard.write_text(source[:start] + """
    pub fn open(now_ns: u64) Result(Keyboard) {
        const result = linux.open(""" + json.dumps(str(keys)) + """, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, 0o600);
        if (linux.errno(result) != .SUCCESS) return .{ .err = .{ .open = linux.errno(result) } };
        return .{ .ok = .{ .fd = @intCast(result), .usable_after_ns = now_ns } };
    }

""" + source[end:])
            replace_source("paste_keyboard.zig", "linux.errno(linux.ioctl(self.fd, linux.IOCTL.IO('U', 2), 0))",
                           "linux.E.SUCCESS")
            subprocess.run(["zig", "build", "-Doptimize=ReleaseSafe"], cwd=fixture, check=True, timeout=180)
            production_binary = binary
            binary = str(fixture / "zig-out/bin/voiced")
            try:
                cases = (
                    ("recording", "recording_size", "size limit", None),
                    ("chunk", "chunk_size", "chunk text limit", "OutputTooSmall"),
                    ("tokens", "decoder_tokens", "decoder token limit", "GeneratedTokenLimitExceeded"),
                    ("both", "chunk_size_and_decoder_tokens", "text and token limits", "OutputTooSmall"),
                    ("recording_unicode", "recording_size", "size limit", None),
                    ("chunk_unicode", "chunk_size", "chunk text limit", "OutputTooSmall"),
                    ("tokens_exit", "decoder_tokens", "decoder token limit", None),
                    ("tokens_stall", "decoder_tokens", "decoder token limit", None),
                )
                for mode, outcome, title, error_name in cases:
                    env["VOICED_LIMIT_FIXTURE"] = mode
                    (root / "mode").write_text("normal")
                    prior = len(notifications_sent())
                    with service("--config", str(settings), "--transcript-output=desktop") as daemon:
                        record_fixture()
                        wait_until(lambda: cli("status")["phase"] == "idle", timeout=18)
                        logs = log_path.read_text()
                        assert f"outcome={outcome}," in logs, logs
                        assert "Recording discarded:" not in logs and "Transcription retry:" not in logs, logs
                        copied = (root / "text").read_bytes()
                        assert copied and copied == saved_transcript.read_bytes(), (mode, copied, logs)
                        copied.decode("utf-8")
                        if mode.endswith("unicode"): assert copied == b"ab", copied
                        assert logs.index("Clipboard acquired:") < logs.index("Paste shortcut sent:") < logs.index("Transcript saved:"), logs
                        assert logs.count("Paste shortcut sent:") == 1, logs
                        events = [struct.unpack("llHHi", keys.read_bytes()[i:i+24])[2:]
                                  for i in range(0, keys.stat().st_size, 24)]
                        assert [event for event in events if event[0] == 1] == [(1, 29, 1), (1, 47, 1), (1, 47, 0), (1, 29, 0)], events
                        wait_until(lambda: len(notifications_sent()) > prior)
                        message = notifications_sent()[-1]
                        assert title in message["title"] and "delivered and saved" in message["body"], message
                        if "tokens" in mode or mode == "both": assert "repetition" in message["body"], message
                        if error_name:
                            metadata = replay.read_metadata(saved_transcript.parent / "last-failed/metadata.txt")
                            assert metadata["error_name"] == error_name, metadata
                            assert (saved_transcript.parent / "last-failed/audio.wav").stat().st_size > 44
                        if mode == "tokens_stall": assert "Limited transcription recovered at deadline:" in logs, logs
                # Cancellation during delivery still preserves the prior save.
                env["VOICED_LIMIT_FIXTURE"] = "tokens"
                (root / "mode").write_text("hang")
                prior = len(notifications_sent())
                saved_inode = saved_transcript.stat().st_ino
                with service("--config", str(settings), "--transcript-output=desktop") as daemon:
                    record_fixture()
                    wait_until(lambda: cli("status")["phase"] == "delivering", timeout=15)
                    cli("cancel")
                    wait_until(lambda: cli("status")["phase"] == "idle")
                    assert saved_transcript.stat().st_ino == saved_inode
                    assert keys.stat().st_size == 0
                    assert len(notifications_sent()) == prior
            finally:
                binary = production_binary
                env.pop("VOICED_LIMIT_FIXTURE", None)
                (root / "mode").write_text("normal")

            print("=== Microphone failure, stalled notifications and repeat suppression ===")
            prior = len(notifications_sent())
            saved_inode = saved_transcript.stat().st_ino
            server_command(desktop_server, "hold")
            with service(*options, "--microphone-node", "voiced-test-missing-source"):
                cli("record")
                wait_until(lambda: len(notifications_sent()) == prior + 1, timeout=10)
                assert notifications_sent()[-1]["title"] == "Voiced: configured mic not found"
                for _ in range(10):
                    started = time.monotonic()
                    assert cli("status")["phase"] == "idle"
                    assert time.monotonic() - started < 0.5
                wait_until(lambda: "org.freedesktop.DBus.Error.NoReply" in log_path.read_text(), timeout=3)
                cli("record")
                wait_until(lambda: "Recording discarded: recording_ordinal=2," in log_path.read_text(), timeout=10)
                wait_until(lambda: len(notifications_sent()) == prior + 2)
                assert notifications_sent()[-1]["title"] == "Voiced: configured mic not found"
                assert saved_transcript.stat().st_ino == saved_inode
            server_command(desktop_server, "reply")
            server_command(desktop_server, "normal")
            with service(*options, "--notification-mode=off", "--microphone-node", "voiced-test-missing-source"):
                cli("record")
                wait_until(lambda: "Recording discarded: recording_ordinal=1," in log_path.read_text(), timeout=10)
                assert "Desktop notification" not in log_path.read_text()
                assert len(notifications_sent()) == prior + 2
            with service(*options, "--microphone-serial", "voiced-test-missing-serial"):
                before = len(notifications_sent())
                cli("record")
                wait_until(lambda: "Recording discarded: recording_ordinal=1," in log_path.read_text(), timeout=10)
                wait_until(lambda: len(notifications_sent()) > before)
                logs = log_path.read_text()
                assert "kind=source_not_found" in logs and "SourceNotFound" in logs, logs
                assert "voiced-test-missing-serial" in logs and "available sources:" in logs, logs
                assert notifications_sent()[-1]["title"] == "Voiced: configured mic not found"
        finally:
            notification_stack.close()
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    print("Native clipboard/output integration passed.")
PY
    exit 0
fi

if [[ "${1:-}" == "--zig" ]]; then
    export VOICED_INSTANCE=test
    python3 - <<'PY'
import contextlib
from journal_fixture import journal_log
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import tempfile
import time

binary = str(Path("zig/zig-out/bin/voiced").resolve())
experiment = Path("/tmp/experiments/voiced-native-integration")
experiment.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(prefix="run-", dir=experiment) as temporary:
    root = Path(temporary)
    env = dict(os.environ, VOICED_INSTANCE="test", XDG_RUNTIME_DIR=str(root))
    socket_path = root / "voiced-test/control.sock"
    log_path = root / "daemon.log"

    def cli(*args):
        result = subprocess.run([binary, *args], env=env, capture_output=True, text=True, timeout=4)
        assert result.returncode == 0, (args, result.stderr, log_path.read_text())
        return json.loads(result.stdout)

    def wait_until(predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError(log_path.read_text())

    def children(daemon):
        return {int(pid) for pid in Path(f"/proc/{daemon.pid}/task/{daemon.pid}/children").read_text().split()}

    @contextlib.contextmanager
    def service(seconds, scenario="normal", model=None, expected_error=None):
        options = ["--test-scenario", scenario] if model is None else ["--model", model, "--target", "voiced-test-source", "--seconds", "20"]
        with journal_log(log_path) as log:
            daemon = subprocess.Popen([binary, "serve", *options,
                                       "--model-keep-warm-seconds", str(seconds)], env=env, stdout=log, stderr=log)
            try:
                wait_until(lambda: "Supervisor service ready" in log_path.read_text() or daemon.poll() is not None)
                assert daemon.poll() is None, log_path.read_text()
                yield daemon
                if expected_error is None:
                    cli("kill")
                    assert daemon.wait(timeout=3) == 0, log_path.read_text()
                else:
                    assert daemon.wait(timeout=4) > 0, log_path.read_text()
                    assert expected_error in log_path.read_text(), log_path.read_text()
                    assert "panic" not in log_path.read_text(), log_path.read_text()
                assert not socket_path.exists()
            except BaseException:
                print(log_path.read_text())
                raise
            finally:
                if daemon.poll() is None:
                    daemon.terminate()
                    try:
                        daemon.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        daemon.kill()
                        daemon.wait()

    print("=== Removed residency controls cannot start a worker ===")
    for command in (("serve",), ("supervisor-spike", "pipewire")):
        for residency in ("session", "service"):
            rejected = subprocess.run([binary, *command, "--model", "Systran/faster-whisper-base.en",
                                       "--model-residency", residency], env=env, capture_output=True, timeout=3)
            assert rejected.returncode != 0 and b"InvalidArguments" in rejected.stderr, rejected
            assert not socket_path.exists()

    print("=== Native socket framing, duplicate exclusion, idle reuse and expiry ===")
    with service(1) as daemon:
        assert cli("status")["phase"] == "idle"
        assert cli("status")["model"] == "absent"
        duplicate = subprocess.run([binary, "serve", "--test-scenario", "normal"], env=env, capture_output=True, timeout=3)
        assert duplicate.returncode != 0
        assert "DaemonAlreadyRunning" in duplicate.stderr.decode()
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(2)
            client.connect(str(socket_path))
            client.sendall(b'{"cmd":')
            time.sleep(0.05)
            client.sendall(b'"status"}\n')
            assert json.loads(client.recv(1024))["phase"] == "idle"
        for invalid in (b'{"cmd":"unknown"}\n', b'{}\n', b'{"cmd":"status","toggle":true}\n', b'x' * 4096):
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(2)
                client.connect(str(socket_path))
                client.sendall(invalid)
                assert not json.loads(client.recv(1024))["ok"]
        with socket.socket(socket.AF_UNIX) as slow:
            slow.settimeout(2)
            slow.connect(str(socket_path))
            assert cli("status")["phase"] == "idle"
            assert slow.recv(1) == b""
        cli("record")
        wait_until(lambda: cli("status")["phase"] == "idle")
        assert cli("status")["model"] == "warm"
        retained, = children(daemon)
        time.sleep(0.65)
        cli("record")
        wait_until(lambda: cli("status")["phase"] == "idle")
        assert children(daemon) == {retained}
        time.sleep(0.5)
        assert children(daemon) == {retained}, "old idle deadline unloaded the reused model"
        wait_until(lambda: cli("status")["model"] == "absent")
        assert not children(daemon)
        cli("record")
        wait_until(lambda: cli("status")["phase"] == "idle")
        replacement, = children(daemon)
        assert replacement != retained
        os.kill(replacement, signal.SIGKILL)
        wait_until(lambda: cli("status")["model"] == "absent")
        cli("record")
        wait_until(lambda: cli("status")["phase"] == "idle")
        assert cli("status")["model"] == "warm"
        assert log_path.read_text().count("Supervisor session complete") == 4

    print("=== Expiry escalation and a recording queued during unload ===")
    with service(1) as daemon:
        cli("record")
        wait_until(lambda: cli("status")["phase"] == "idle")
        frozen, = children(daemon)
        os.kill(frozen, signal.SIGSTOP)
        wait_until(lambda: cli("status")["model"] == "unloading")
        assert cli("record")["phase"] == "capturing"
        wait_until(lambda: cli("status")["phase"] == "idle")
        replacement, = children(daemon)
        assert replacement != frozen
        assert not Path(f"/proc/{frozen}").exists()
        assert log_path.read_text().count("Supervisor session complete") == 2

    print("=== Stale sockets, zero retention, stop, cancellation, retry and bounded failure ===")
    with socket.socket(socket.AF_UNIX) as stale:
        stale.bind(str(socket_path))
    with service(0) as daemon:
        for _ in range(3):
            cli("record")
            wait_until(lambda: cli("status")["phase"] == "idle")
            assert not children(daemon)
        assert log_path.read_text().count("Supervisor session complete") == 3
    with service(1, "slow_transcription") as daemon:
        cli("record")
        wait_until(lambda: cli("status")["phase"] == "transcribing")
        cli("stop")
        assert cli("record", "-t")["ignored"], log_path.read_text()
        wait_until(lambda: cli("status")["phase"] == "idle")
        cli("record")
        cli("cancel")
        wait_until(lambda: cli("status")["phase"] == "idle")
        assert not children(daemon)
        assert "Reason: user_cancelled" in log_path.read_text()
    for scenario, outcome in (("transcription_crash_before_result", "complete"),
                              ("transcription_crash_after_result", "complete"),
                              ("repeated_transcription_crash", "discarded"),
                              ("transcription_hang", "discarded")):
        with service(1, scenario):
            for _ in range(2):
                cli("record")
                wait_until(lambda: cli("status")["phase"] == "idle")
            assert log_path.read_text().count("Supervisor session " + outcome) == 2, log_path.read_text()
    print("=== Unexpected loop failure reaps both workers without double-closing descriptors ===")
    with service(1, "transcription_hang", expected_error="ControlAcceptFailed") as daemon:
        cli("record")
        wait_until(lambda: len(children(daemon)) == 2)
        os.kill(daemon.pid, signal.SIGSTOP)
        owned_workers = children(daemon)
        try:
            assert len(owned_workers) == 2
            for pid in owned_workers:
                os.kill(pid, signal.SIGSTOP)
            # Lower only this test daemon's soft limit. Existing descriptors
            # remain usable, but the next accept fails with EMFILE and unwinds
            # the supervisor while both child processes still need cleanup.
            _, hard_limit = resource.prlimit(daemon.pid, resource.RLIMIT_NOFILE)
            resource.prlimit(daemon.pid, resource.RLIMIT_NOFILE, (0, hard_limit))
            with socket.socket(socket.AF_UNIX) as client:
                client.connect(str(socket_path))
        finally:
            os.kill(daemon.pid, signal.SIGCONT)
        assert daemon.wait(timeout=4) > 0, log_path.read_text()
        assert all(not Path(f"/proc/{pid}").exists() for pid in owned_workers)

    if os.environ.get("VOICED_ZIG_MODEL_TESTS") == "1":
        print("=== Private PipeWire, real Float32 inference, complete-runtime retention ===")
        # No desktop graph, session bus, microphone, or installed model is
        # modified. Only the deliberately created virtual source can be used.
        config = root / "config"
        config.mkdir()
        pipewire_config = Path("/usr/share/pipewire/pipewire.conf").read_text()
        assert pipewire_config.count("context.objects = [") == 1
        pipewire_config = pipewire_config.replace("context.objects = [", '''context.objects = [
    { factory = adapter args = { factory.name = support.null-audio-sink
      node.name = voiced-test-source node.description = "Voiced fixture source"
      media.class = "Audio/Source/Virtual" audio.position = [ MONO ]
      monitor.passthrough = true } }''')
        (config / "pipewire-test.conf").write_text(pipewire_config)
        wireplumber_config = Path("/usr/share/wireplumber/wireplumber.conf").read_text()
        wireplumber_config = "\n".join(line for line in wireplumber_config.splitlines()
                                      if not any(module in line for module in ("alsa.lua", "bluez.lua", "bluetooth.lua", "v4l2.lua")))
        (config / "wireplumber-test.conf").write_text(wireplumber_config)
        env.update(PIPEWIRE_RUNTIME_DIR=str(root), XDG_CACHE_HOME=str(root / "cache"),
                   XDG_CONFIG_HOME=str(root / "config-home"), XDG_STATE_HOME=str(root / "state"), GIO_USE_VFS="local")
        processes = []
        try:
            bus = subprocess.Popen(["dbus-daemon", "--session", "--nofork", "--print-address"], stdout=subprocess.PIPE, text=True)
            processes.append(bus)
            address = bus.stdout.readline().strip()
            assert address
            env.update(DBUS_SESSION_BUS_ADDRESS=address, DBUS_SYSTEM_BUS_ADDRESS=address)
            with (root / "pipewire.log").open("w") as log:
                processes.append(subprocess.Popen(["pipewire", "-c", "pipewire-test.conf"],
                    env=dict(env, PIPEWIRE_CONFIG_DIR=str(config)), stdout=log, stderr=log))
            wait_until(lambda: (root / "pipewire-0").exists())
            with (root / "wireplumber.log").open("w") as log:
                processes.append(subprocess.Popen(["wireplumber", "-c", str(config / "wireplumber-test.conf")], env=env, stdout=log, stderr=log))
            time.sleep(1)
            assert all(process.poll() is None for process in processes)

            def model_pid(daemon):
                for pid in children(daemon):
                    if b"transcription-model" in Path(f"/proc/{pid}/cmdline").read_bytes():
                        return pid
                raise AssertionError("model process is absent")

            def rss_kib(pid):
                return int(next(line.split()[1] for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("VmRSS:")))

            for variant in ("base.en", "small.en"):
                with service(2, model="Systran/faster-whisper-" + variant) as daemon:
                    retained = None
                    first_rss = None
                    for recording in range(2):
                        cli("record")
                        wait_until(lambda: cli("status")["model"] == "warm", timeout=15)
                        current = model_pid(daemon)
                        if retained is not None:
                            assert current == retained, "warm runtime was recreated"
                        retained = current
                        time.sleep(0.5)  # Allow source selection and noise-floor calibration.
                        playback = subprocess.Popen(["pw-cat", "--playback", "--target", "0", "--properties", "node.name=voiced-fixture-playback", "test-fixtures/hello_world.wav"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        try:
                            for _ in range(100):
                                outputs = subprocess.run(["pw-link", "-o"], env=env, capture_output=True, text=True, timeout=2).stdout.splitlines()
                                inputs = subprocess.run(["pw-link", "-i"], env=env, capture_output=True, text=True, timeout=2).stdout.splitlines()
                                source_ports = [port.strip() for port in outputs if port.strip().startswith("voiced-fixture-playback:")]
                                target_ports = [port.strip() for port in inputs if port.strip().startswith("voiced-test-source:")]
                                if source_ports and target_ports:
                                    break
                                time.sleep(0.01)
                            assert len(source_ports) == len(target_ports) == 1, (outputs, inputs)
                            linked = subprocess.run(["pw-link", source_ports[0], target_ports[0]], env=env, capture_output=True, timeout=2)
                            assert linked.returncode == 0, linked.stderr
                            assert playback.wait(timeout=10) == 0, playback.stderr.read()
                        finally:
                            if playback.poll() is None:
                                playback.kill()
                                playback.wait()
                        cli("stop")
                        wait_until(lambda: cli("status")["phase"] == "idle", timeout=15)
                        assert cli("status")["model"] == "warm", log_path.read_text()
                        assert children(daemon) == {retained}
                        rss = rss_kib(retained)
                        if first_rss is None:
                            first_rss = rss
                        assert rss < first_rss + 8 * 1024, "per-recording memory growth"
                        assert log_path.read_text().lower().count("transcript: hello") == recording + 1, log_path.read_text()
                        assert "Warm-up:" not in log_path.read_text()
                        assert "Model residency:" not in log_path.read_text()
                    print(f"{variant}: retained worker RSS {rss / 1024:.1f} MiB; two fixture recordings reused PID {retained}")
                    wait_until(lambda: cli("status")["model"] == "absent", timeout=4)
                    assert not children(daemon)
                    cli("record")
                    wait_until(lambda: cli("status")["model"] == "warm", timeout=15)
                    assert model_pid(daemon) != retained
                    assert "cache hit:" in log_path.read_text(), log_path.read_text()
                    cli("cancel")
                    wait_until(lambda: cli("status")["phase"] == "idle")
                    assert not children(daemon)
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    print("Native process/socket integration passed.")
PY
    exit 0
fi

echo "Usage: $0 --zig-logging | --zig-output | --zig-replay" >&2
exit 2
