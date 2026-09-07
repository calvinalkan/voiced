#!/usr/bin/env python3
"""Exercise the native notification client and retain private diagnostic logs.

Default: show, replace, and close two synthetic desktop error notifications.
--self-test: use scripted private Unix sockets; never contact the desktop bus.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import tempfile
import threading
import time

BUS = "org.freedesktop.DBus"
SERVICE = "org.freedesktop.Notifications"
PATH = "/org/freedesktop/Notifications"


class Driver:
    def __init__(self, binary, root, address=None, runtime_directory=None):
        self.root = root
        root.mkdir(mode=0o700)
        self.log_path = root / "client.log"
        self.logs = []
        self.receiver, child_log = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.receiver.settimeout(0.1)
        self.done = threading.Event()
        env = dict(os.environ, VOICED_INSTANCE="test")
        if address is not None:
            env["DBUS_SESSION_BUS_ADDRESS"] = address
        if runtime_directory is not None:
            env.pop("DBUS_SESSION_BUS_ADDRESS", None)
            env["XDG_RUNTIME_DIR"] = str(runtime_directory)
        self.process = subprocess.Popen([str(binary)], env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=child_log)
        child_log.close()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        with self.log_path.open("w") as output, (self.root / "journal.jsonl").open("w") as journal:
            while True:
                try:
                    packet = self.receiver.recv(65536)
                except socket.timeout:
                    if self.done.is_set():
                        break
                    continue
                raw = packet
                fields = {}
                while packet:
                    line, separator, rest = packet.partition(b"\n")
                    if not separator:
                        break
                    if b"=" in line:
                        key, value = line.split(b"=", 1)
                        packet = rest
                    else:
                        if len(rest) < 8:
                            break
                        size = int.from_bytes(rest[:8], "little")
                        key, value = line, rest[8:8 + size]
                        packet = rest[9 + size:]
                    fields[key.decode(errors="replace")] = value.decode(errors="replace")
                message = fields.get("MESSAGE", raw.decode(errors="replace"))
                self.logs.append(message)
                output.write(message + "\n")
                output.flush()
                journal.write(json.dumps(dict(time=time.time(), **fields)) + "\n")
                journal.flush()

    def command(self, command):
        self.process.stdin.write(command.encode())
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 0.5)[0]:
            raise AssertionError("Notification work blocked the control channel")
        assert os.read(self.process.stdout.fileno(), 3) == b"ok\n", self.log_path

    def wait_log(self, text, timeout=3, after=0):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if any(text in line for line in self.logs[after:]):
                return
            if self.process.poll() is not None:
                raise AssertionError(f"Client exited {self.process.returncode}; see {self.log_path}")
            time.sleep(0.01)
        raise AssertionError(f"Missing log {text!r}; see {self.log_path}")

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(b"q")
            self.process.stdin.flush()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.done.set()
        self.thread.join(timeout=2)
        self.receiver.close()
        assert self.process.returncode == 0, (self.process.returncode, self.log_path)


@contextlib.contextmanager
def driver(binary, root, **kwargs):
    client = Driver(binary, root, **kwargs)
    try:
        yield client
    finally:
        client.close()


def encoded(signature, values, endian="<", offset=0):
    data = bytearray()
    for code, value in zip(signature, values, strict=True):
        alignment = 4 if code in "suoi" else 1
        data.extend(b"\0" * (-(offset + len(data)) % alignment))
        if code in "so":
            value = value.encode()
            data.extend(struct.pack(endian + "I", len(value)) + value + b"\0")
        elif code == "g":
            value = value.encode()
            data.extend(bytes([len(value)]) + value + b"\0")
        elif code in "ui":
            data.extend(struct.pack(endian + ("I" if code == "u" else "i"), value))
        else:
            raise AssertionError(code)
    return bytes(data)


def frame(kind, serial, fields, signature="", values=(), endian="<"):
    header = bytearray()
    if signature:
        fields = [*fields, (8, "g", signature)]
    for code, field_signature, value in fields:
        header.extend(b"\0" * (-len(header) % 8))
        header.append(code)
        header.extend(encoded("g", [field_signature]))
        header.extend(encoded(field_signature, [value], endian, len(header)))
    body = encoded(signature, values, endian)
    fixed = bytes([ord("l" if endian == "<" else "B"), kind, 0, 1])
    fixed += struct.pack(endian + "III", len(body), serial, len(header))
    return fixed + header + b"\0" * (-len(header) % 8) + body


def read_exact(peer, count):
    result = bytearray()
    while len(result) < count:
        block = peer.recv(count - len(result))
        assert block, "Unexpected EOF"
        result.extend(block)
    return bytes(result)


def receive(peer):
    fixed = read_exact(peer, 16)
    assert fixed[:4] == b"l\x01\0\x01", fixed
    size, serial, fields = struct.unpack("<III", fixed[4:])
    tail = read_exact(peer, ((fields + 7) & ~7) + size)
    return serial, tail[:fields], tail[(fields + 7) & ~7:]


def auth(peer):
    expected = b"\0AUTH EXTERNAL " + str(os.getuid()).encode().hex().encode() + b"\r\n"
    assert read_exact(peer, len(expected)) == expected


def reply(peer, serial, signature="", values=(), sender=BUS, endian="<", fragment=False):
    packet = frame(2, 50 + serial, [(5, "u", serial), (7, "s", sender)], signature, values, endian)
    if fragment:
        for byte in packet:
            peer.sendall(bytes([byte]))
            time.sleep(0.0005)
    else:
        peer.sendall(packet)


def setup(peer, fragment=False):
    auth(peer)
    text = b"OK 0123456789abcdef0123456789abcdef\r\n"
    for byte in text if fragment else [text]:
        peer.sendall(bytes([byte]) if isinstance(byte, int) else byte)
    assert read_exact(peer, 7) == b"BEGIN\r\n"
    serial, header, body = receive(peer)
    assert b"Hello" in header and not body
    reply(peer, serial, "s", [":1.42"], endian=">" if fragment else "<", fragment=fragment)
    for match in [b"NotificationClosed", b"NameOwnerChanged"]:
        serial, header, body = receive(peer)
        assert b"AddMatch" in header and match in body
        reply(peer, serial, endian=">" if fragment else "<", fragment=fragment)


def notification(peer, client, command="a", replaces=0, endian="<", fragment=False):
    client.command(command)
    serial, header, body = receive(peer)
    assert b"Notify\0" in header
    string_length = struct.unpack_from("<I", body)[0]
    offset = (4 + string_length + 1 + 3) & ~3
    assert struct.unpack_from("<I", body, offset)[0] == replaces, body
    reply(peer, serial, "u", [7], sender=":1.7", endian=endian, fragment=fragment)
    return serial


def private_tests(binary, root):
    completed = []

    @contextlib.contextmanager
    def scenario(name, fallback=False):
        case = root / name
        case.mkdir()
        sockpath = case / ("bus" if fallback else "socket")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(sockpath))
            listener.listen(8)
            listener.settimeout(8)
            options = {"runtime_directory": case} if fallback else {"address": "unix:path=" + str(sockpath)}
            with driver(binary, case / "driver", **options) as client:
                peer, _ = listener.accept()
                peer.settimeout(3)
                with peer:
                    yield client, peer, listener
        completed.append(name)
        print("PASS:", name, flush=True)

    with scenario("fragmented-big-endian") as (client, peer, _):
        setup(peer, fragment=True)
        notification(peer, client, endian=">", fragment=True)
        client.wait_log("Desktop notification accepted")
        # A different application cannot dismiss our ID with a forged signal.
        peer.sendall(frame(4, 100, [(7, "s", ":1.99"), (1, "o", PATH),
                                   (2, "s", SERVICE), (3, "s", "NotificationClosed")], "uu", [7, 2]))
        notification(peer, client, "b", replaces=7)
        client.wait_log("problem=transcription_failed")
        peer.sendall(frame(4, 101, [(7, "s", ":1.7"), (1, "o", PATH),
                                   (2, "s", SERVICE), (3, "s", "NotificationClosed")], "uu", [7, 2]))
        client.wait_log("Desktop notification closed:")
        notification(peer, client, "c", replaces=0)
        client.wait_log("problem=transcript_save_failed")
        client.command("r")
        serial, header, body = receive(peer)
        assert b"CloseNotification" in header and body == struct.pack("<I", 7)
        reply(peer, serial, sender=":1.7")
        client.wait_log("close accepted")

    with scenario("rejected-authentication") as (client, peer, _):
        auth(peer)
        peer.sendall(b"REJECTED EXTERNAL\r\n")
        client.wait_log("AuthenticationRejected")
        client.command("a")

    with scenario("authentication-timeout") as (client, peer, _):
        auth(peer)
        client.wait_log("AuthenticationOrSetupTimedOut")
        client.command("a")

    for name, packet, error in [
        ("oversize-frame", b"l\x02\0\x01" + struct.pack("<III", 0xFFFFFFFF, 2, 0), "MessageTooLarge"),
        ("invalid-frame", b"x\x02\0\x01" + struct.pack("<III", 0, 2, 0), "InvalidMessage"),
        ("invalid-utf8", frame(4, 101, [(7, "s", ":1.7"), (1, "o", PATH),
                                      (2, "s", SERVICE), (3, "s", "NotificationClosed")], "uu", [7, 2]).replace(b":1.7", b":1.\xff"), "InvalidMessage"),
    ]:
        with scenario(name) as (client, peer, _):
            setup(peer)
            client.wait_log("bus ready")
            peer.sendall(packet)
            client.wait_log(error)
            client.command("a")

    with scenario("unexpected-message-flood") as (client, peer, _):
        setup(peer)
        # More than one dispatch budget, delivered in a single socket write.
        packet = frame(4, 101, [(7, "s", ":1.99"), (1, "o", "/other"),
                                (2, "s", "org.example.Other"), (3, "s", "Changed")])
        peer.sendall(packet * 80)
        notification(peer, client)
        client.wait_log("Desktop notification accepted")

    with scenario("bus-reconnect") as (client, peer, listener):
        setup(peer)
        notification(peer, client)
        client.wait_log("Desktop notification accepted")
        before = len(client.logs)
        peer.shutdown(socket.SHUT_RDWR)
        peer.close()
        client.wait_log("Desktop notifications disconnected", after=before)
        client.command("b")
        second, _ = listener.accept()
        with second:
            second.settimeout(3)
            setup(second)
            serial, header, body = receive(second)
            assert b"Notify" in header
            assert b"transcription failed" in body
            offset = (4 + struct.unpack_from("<I", body)[0] + 1 + 3) & ~3
            assert struct.unpack_from("<I", body, offset)[0] == 0
            reply(second, serial, "u", [8], sender=":1.8")
            client.wait_log("problem=transcription_failed")

    # Fallback filesystem paths must not be interpreted as escaped addresses.
    with scenario("runtime,%directory", fallback=True) as (client, peer, _):
        setup(peer)
        notification(peer, client)
        client.wait_log("Desktop notification accepted")

    with driver(binary, root / "unsupported-address", address="tcp:host=localhost,port=1") as client:
        client.wait_log("UnsupportedAddress")
        client.command("a")
    completed.append("unsupported-address")
    return completed


def desktop_test(binary, root):
    print("This check shows two synthetic Voiced error messages, then closes them.")
    print("It does not use your microphone, clipboard, configuration, or running service.")
    with driver(binary, root / "desktop") as client:
        closed = False
        try:
            client.wait_log("bus ready", timeout=5)
            client.command("a")
            client.wait_log("problem=microphone_failed")
            print("1/3: Microphone error accepted. Leaving it visible for four seconds.", flush=True)
            time.sleep(4)
            client.command("b")
            client.wait_log("problem=transcription_failed")
            print("2/3: Transcription error accepted as a replacement. Waiting four seconds.", flush=True)
            time.sleep(4)
            client.command("r")
            client.wait_log("close accepted")
            closed = True
            print("3/3: Close request accepted. The test popup should disappear.", flush=True)
        finally:
            if not closed and client.process.poll() is None:
                with contextlib.suppress(Exception):
                    client.command("r")
                    client.wait_log("close accepted", timeout=1.5)
    return ["show", "replace", "close"]


def main():
    from blake3 import blake3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "zig-out/bin/voiced-notification-check")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("Build the check first: cd zig && zig build notification-check")
    root = Path(tempfile.mkdtemp(prefix="voiced-native-notifications-"))
    report = dict(binary=str(args.binary.resolve()), binary_blake3=blake3(args.binary.read_bytes()).hexdigest(), mode="private" if args.self_test else "desktop", ok=False)
    print("Logs:", root, flush=True)
    try:
        report["checks"] = private_tests(args.binary, root) if args.self_test else desktop_test(args.binary, root)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Report:", root / "report.json", flush=True)


if __name__ == "__main__":
    main()
