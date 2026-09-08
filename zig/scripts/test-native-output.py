#!/usr/bin/env python3
"""Private Wayland protocol and optional PipeWire→model→clipboard integration.

Protocol checks run directly. Model checks still depend on the removed
integration harness's journal_fixture module.
The fake compositor serves only this test; no desktop or uinput is accessed.
"""
import array
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

REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / 'zig/zig-out/bin/voiced'
CHECK = BINARY.with_name('voiced-clipboard-check')


def words(*values):
    return struct.pack('<' + 'I' * len(values), *values)


def string(value):
    data = value.encode() + b'\0'
    return words(len(data)) + data + b'\0' * (-len(data) % 4)


class Compositor:
    def __init__(self, root):
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(root / 'wayland-test'))
        self.listener.listen()
        self.connection = None
        self.source = None
        self.error = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def send(self, object_id, opcode, body=b'', fd=None):
        packet = words(object_id, (len(body) + 8) << 16 | opcode) + body
        with self.lock:
            if fd is None:
                self.connection.sendall(packet)
            else:
                sent = self.connection.sendmsg([packet], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))])
                if sent < len(packet):
                    self.connection.sendall(packet[sent:])

    def read(self, count):
        result = b''
        while len(result) < count:
            part = self.connection.recv(count - len(result))
            if not part:
                raise EOFError
            result += part
        return result

    def run(self):
        try:
            self.connection, _ = self.listener.accept()
            kinds = {1: 'display'}
            while True:
                object_id, header = struct.unpack('<II', self.read(8))
                opcode, size = header & 65535, header >> 16
                assert size >= 8 and size % 4 == 0
                body = self.read(size - 8)
                kind = kinds[object_id]
                if kind == 'display' and opcode == 1:
                    registry, = struct.unpack('<I', body)
                    kinds[registry] = 'registry'
                    self.send(registry, 0, words(1) + string('wl_seat') + words(7))
                    self.send(registry, 0, words(2) + string('ext_data_control_manager_v1') + words(1))
                elif kind == 'display' and opcode == 0:
                    callback, = struct.unpack('<I', body)
                    self.send(callback, 0, words(0))
                    self.send(1, 1, words(callback))
                elif kind == 'registry' and opcode == 0:
                    name, size = struct.unpack_from('<II', body)
                    interface = body[8:8 + size - 1].decode()
                    version, new_id = struct.unpack_from('<II', body, 8 + (size + 3) // 4 * 4)
                    kinds[new_id] = 'seat' if name == 1 else 'manager'
                    if name == 1:
                        self.send(new_id, 0, words(2))
                    else:
                        assert interface == 'ext_data_control_manager_v1' and version == 1
                elif kind == 'manager' and opcode == 1:
                    device, _ = struct.unpack('<II', body)
                    kinds[device] = 'device'
                    self.send(device, 1, words(0))
                elif kind == 'manager' and opcode == 0:
                    new_id, = struct.unpack('<I', body)
                    kinds[new_id] = 'source'
                elif kind == 'source' and opcode == 0:
                    pass  # MIME offer; requests below exercise UTF-8 delivery.
                elif kind == 'source' and opcode == 1:
                    self.send(1, 1, words(object_id))
                    del kinds[object_id]
                elif kind == 'device' and opcode == 0:
                    new_id, = struct.unpack('<I', body)
                    if self.source:
                        self.send(self.source, 1)
                    self.source = new_id
                else:
                    raise AssertionError((kind, opcode))
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.error = error

    def text(self):
        read_fd, write_fd = os.pipe()
        try:
            self.send(self.source, 0, string('text/plain;charset=utf-8'), write_fd)
            os.close(write_fd)
            write_fd = -1
            output = bytearray()
            while True:
                assert select.select([read_fd], [], [], 3)[0], 'clipboard transfer stalled'
                part = os.read(read_fd, 4096)
                if not part:
                    return output.decode()
                output.extend(part)
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)

    def close(self):
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        self.listener.close()
        self.thread.join(timeout=2)
        assert self.error is None, self.error


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def event(process, expected):
    assert select.select([process.stdout], [], [], 4)[0], expected
    value = json.loads(process.stdout.readline())
    assert value['event'] == expected, value
    return value


def protocol_check(root, env):
    server = Compositor(root)
    first = 'Native clipboard café — 日本語 🎤\n' * 100
    second = 'Replacement transcript\nSecond line.\n'
    (root / 'a').write_text(first)
    (root / 'b').write_text(second)
    # The verifier poisons Client storage before init. Failed initialization
    # must still leave every owned-descriptor field safe for deferred cleanup.
    failed = subprocess.run([str(CHECK), str(root / 'a'), str(root / 'b')],
                            env=dict(env, WAYLAND_DISPLAY='missing-wayland-test'),
                            input='', capture_output=True, text=True, timeout=4)
    assert failed.returncode == 1, (failed.returncode, failed.stderr)
    assert failed.stdout.strip(), failed.stderr
    assert json.loads(failed.stdout) == {'event': 'error', 'kind': 'transport'}, failed.stdout
    with (root / 'clipboard.log').open('w') as log:
        process = subprocess.Popen([str(CHECK), str(root / 'a'), str(root / 'b')], env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0)
    try:
        assert event(process, 'ready')['mode'] == 'ext'
        for ordinal in range(24):
            key, text = (b'a', first) if ordinal % 2 == 0 else (b'b', second)
            process.stdin.write(key)
            event(process, 'acquired')
            for _ in range(3):
                assert server.text() == text
        # Complete malformed framing is a typed error; it must not crash.
        with server.lock:
            server.connection.sendall(words(1, 4 << 16))
        assert event(process, 'error')['kind'] == 'transport'
        assert process.wait(timeout=3) == 1
    finally:
        stop(process)
        server.close()
    print('Native clipboard: poisoned-storage init/failed-init cleanup, 24 replacements, 72 exact reads, malformed-frame rejection passed.')


def model_check(root, env):
    from journal_fixture import journal_log
    processes = []
    server = Compositor(root)
    config = Path('/usr/share/pipewire/pipewire.conf').read_text().replace('context.objects = [', '''context.objects = [
 { factory = adapter args = { factory.name = support.null-audio-sink
   node.name = voiced-test-source node.description = "Voiced fixture source"
   media.class = "Audio/Source/Virtual" audio.position = [ MONO ] monitor.passthrough = true } }
''').replace('context.properties = {', 'context.properties = { default.clock.rate = 48000 default.clock.quantum = 1024 default.clock.min-quantum = 1024 default.clock.max-quantum = 1024')
    (root / 'pipewire.conf').write_text(config)
    wp = Path('/usr/share/wireplumber/wireplumber.conf').read_text()
    (root / 'wireplumber.conf').write_text('\n'.join(line for line in wp.splitlines() if not any(name in line for name in ('alsa.lua', 'bluez.lua', 'bluetooth.lua', 'v4l2.lua'))))
    env = dict(env, PIPEWIRE_RUNTIME_DIR=str(root), XDG_CACHE_HOME=str(root / 'cache'), GIO_USE_VFS='local')
    log_path = root / 'service.log'

    def cli(*args):
        result = subprocess.run([str(BINARY), *args], env=env, capture_output=True, text=True, timeout=3)
        assert result.returncode == 0, (result.stderr, log_path.read_text())
        return result.stdout

    def wait_for(predicate, seconds=20):
        deadline = time.monotonic() + seconds
        while not predicate():
            assert time.monotonic() < deadline, log_path.read_text()
            assert service.poll() is None, log_path.read_text()
            time.sleep(0.02)

    def children():
        return [int(value) for value in Path(f'/proc/{service.pid}/task/{service.pid}/children').read_text().split()]

    def capture_tid():
        return next((int(path.parent.name) for path in Path(f'/proc/{service.pid}/task').glob('*/comm') if path.read_text().strip() == 'voiced-capture'), None)

    try:
        bus = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--print-address'], stdout=subprocess.PIPE, text=True)
        processes.append(bus)
        env['DBUS_SESSION_BUS_ADDRESS'] = env['DBUS_SYSTEM_BUS_ADDRESS'] = bus.stdout.readline().strip()
        for args, label, extra in [(['pipewire', '-c', 'pipewire.conf'], 'pipewire', {'PIPEWIRE_CONFIG_DIR': str(root)}), (['wireplumber', '-c', str(root / 'wireplumber.conf')], 'wireplumber', {})]:
            with (root / (label + '.log')).open('w') as log:
                processes.append(subprocess.Popen(args, env=dict(env, **extra), stdout=log, stderr=log))
            time.sleep(0.6)
        with journal_log(log_path) as log:
            service = subprocess.Popen([str(BINARY), 'serve', '--log-level', 'debug', '--transcript-output', 'clipboard', '--notification-mode', 'off', '--model', 'Systran/faster-whisper-base.en', '--model-encoder-threads', '8', '--model-decoder-threads', '2', '--microphone-node', 'voiced-test-source'], env=env, stdout=subprocess.DEVNULL, stderr=log)
        processes.append(service)
        wait_for(lambda: 'service_started' in log_path.read_text())
        wait_for(lambda: capture_tid() is not None)
        pid = capture_tid()
        idle_descriptors = len(list(Path(f'/proc/{service.pid}/fd').iterdir()))
        saved = root / 'state/voiced-test/transcript.txt'
        for ordinal in range(1, 5):
            if ordinal == 3:
                # Refuse atomic replacement after successful clipboard delivery.
                # Preserve the previous file separately, then restore it so the
                # next recording proves recovery without a daemon restart.
                previous = saved.with_name('previous-transcript.txt')
                previous_text = saved.read_text()
                saved.rename(previous)
                saved.mkdir()
            cli('record')
            wait_for(lambda: f'capture_started recording_id={ordinal} ' in log_path.read_text())
            playback = subprocess.Popen(['pw-cat', '--playback', '--target', '0', '--properties', 'node.name=voiced-fixture-playback', str(REPO / 'test-fixtures/hello_world.wav')], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            processes.append(playback)
            ports = []
            for _ in range(100):
                ports = [port.strip() for port in subprocess.check_output(['pw-link', '-o'], env=env, text=True).splitlines() if port.strip().startswith('voiced-fixture-playback:')]
                if ports:
                    break
                time.sleep(0.02)
            assert len(ports) == 1, ports
            subprocess.run(['pw-link', ports[0], 'voiced-test-source:input_MONO'], env=env, check=True, capture_output=True)
            assert playback.wait(timeout=8) == 0, playback.stderr.read()
            cli('stop')
            # Observe completion without waking the supervisor with status
            # requests: ready clipboard work must advance on its own.
            event = 'transcript_save_failed' if ordinal == 3 else 'recording_finished'
            wait_for(lambda: f'{event} recording_id={ordinal} ' in log_path.read_text())
            assert 'phase=idle' in cli('status')
            assert capture_tid() == pid
            text = server.text()
            assert 'hello' in text.lower(), text
            if ordinal == 3:
                wait_for(lambda: any(row.get('VOICED_EVENT') == 'transcript_save_failed' for row in log.records))
                failure = next(row for row in log.records if row.get('VOICED_EVENT') == 'transcript_save_failed')
                assert failure['PRIORITY'] == '3' and failure['VOICED_COMPONENT'] == 'storage' and failure['VOICED_RECORDING_ID'] == '3', failure
                assert f'directory_path="{saved.parent}"' in failure['MESSAGE'], failure
                for field in ('file_name="transcript.txt"', 'problem_code=replace', 'error="IsDir"', 'transcript_save_duration_ms='):
                    assert field in failure['MESSAGE'], failure
                assert 'cleanup_error=' not in failure['MESSAGE'], failure
                assert {entry.name for entry in saved.parent.iterdir()} == {'transcript.txt', 'previous-transcript.txt'}
                assert previous.read_text() == previous_text
                saved.rmdir()
                previous.rename(saved)
            else:
                assert saved.read_text() == text
            # Model mappings close their source descriptors; each capture must
            # return to the same shared descriptor budget and spawn no process.
            assert len(list(Path(f'/proc/{service.pid}/fd').iterdir())) == idle_descriptors
            assert not children()
            if ordinal == 1:
                # The next delivery reconstructs Client in the same union
                # storage after a real disconnect, not fresh process memory.
                server.close()
                wait_for(lambda: any(row.get('VOICED_EVENT') == 'clipboard_failed' for row in log.records))
                failure = next(row for row in log.records if row.get('VOICED_EVENT') == 'clipboard_failed')
                assert failure['VOICED_COMPONENT'] == 'clipboard' and failure['VOICED_RECORDING_ID'] == '1', failure
                for field in ('kind="transport"', 'error="', 'system_error=', 'object=', 'opcode='):
                    assert field in failure['MESSAGE'], failure
                (root / 'wayland-test').unlink()
                server = Compositor(root)
        assert server.text() == text
        cli('record')
        wait_for(lambda: 'capture_started recording_id=5 ' in log_path.read_text())
        cli('cancel')
        wait_for(lambda: 'recording_finished recording_id=5 outcome=cancelled' in log_path.read_text())
        assert 'phase=idle' in cli('status')
        assert saved.read_text() == text
        cli('kill')
        assert service.wait(timeout=4) == 0
        print('Native capture/model/clipboard/save: repeated delivery, clipboard reconnect, save-error diagnostics/recovery, thread reuse and cancellation passed.')
    finally:
        for process in reversed(processes):
            stop(process)
        server.close()


def main():
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix='voiced-native-output-'))
    print(f'Native output logs: {root}', flush=True)
    subprocess.run(['zig', 'build', 'clipboard-check', '-Doptimize=ReleaseSafe'], cwd=REPO / 'zig', check=True)
    for name, check in [('protocol', protocol_check), ('model', model_check)]:
        if name == 'model' and os.environ.get('VOICED_ZIG_MODEL_TESTS') != '1':
            continue
        directory = root / name
        directory.mkdir()
        env = dict(os.environ, VOICED_INSTANCE='test', XDG_RUNTIME_DIR=str(directory),
                   XDG_CONFIG_HOME=str(directory / 'config'), XDG_STATE_HOME=str(directory / 'state'),
                   WAYLAND_DISPLAY='wayland-test')
        for key in ('WAYLAND_SOCKET', 'DISPLAY', 'SWAYSOCK', 'DBUS_SESSION_BUS_ADDRESS'):
            env.pop(key, None)
        check(directory, env)


if __name__ == '__main__':
    main()
