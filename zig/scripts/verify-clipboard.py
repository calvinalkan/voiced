#!/usr/bin/env python3
"""Verify the native clipboard spike in a dedicated Wayland GTK window.

The production service, microphone and saved transcript are not used. The
clipboard is temporarily replaced; existing plain text is kept only in memory
and restored if no other application has since copied something different.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import tempfile
import time
import uuid

BINARY = Path(__file__).resolve().parents[1] / 'zig-out/bin/voiced-clipboard-check'
TEXT_TYPES = {'text/plain', 'text/plain;charset=utf-8', 'text/plain;charset=UTF-8',
              'UTF8_STRING', 'TEXT', 'STRING', 'TARGETS', 'TIMESTAMP', 'MULTIPLE', 'SAVE_TARGETS'}


def command(argv, env, data=None):
    # The verifier uses stock tools only for backing up/restoring the previous
    # selection. Publication and the test's transfer path use the Zig client.
    # wl-copy's background owner retains stderr. A diagnostic file avoids
    # waiting for that owner's lifetime when the launcher has already exited.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors,
                                   start_new_session=True)
        try:
            stdout, _ = process.communicate(data, timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuntimeError(f'{argv[0]} timed out') from None
        if len(stdout) > 4 * 1024 * 1024:
            raise RuntimeError('Clipboard exceeds the verifier limit')
        errors.seek(0)
        return process.returncode, stdout, errors.read(8192)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clipboard-only', action='store_true',
                        help='Use GTK paste directly; do not open uinput or inject keys')
    parser.add_argument('--fallback', action='store_true',
                        help='Exercise the temporary-surface path even when data-control is available')
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix='voiced-native-clipboard-'))
    print(f'Logs and report: {root}', flush=True)
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'result': 'failed', 'clipboard_restoration': 'not_needed',
              'keyboard_injection': not args.clipboard_only, 'rounds': [],
              'session': {key: os.environ.get(key) for key in
                          ('XDG_CURRENT_DESKTOP', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE')}}
    env = dict(os.environ, VOICED_INSTANCE='test', GDK_BACKEND='wayland')
    native = None
    backup = None
    empty = False
    published = False
    expected = []
    try:
        if not BINARY.is_file():
            raise RuntimeError('Build first: cd zig && zig build clipboard-check')
        if env.get('WAYLAND_SOCKET'):
            raise RuntimeError('Run from a regular desktop terminal without WAYLAND_SOCKET')
        if any(shutil.which(tool) is None for tool in ('wl-copy', 'wl-paste')):
            raise RuntimeError('The verifier needs wl-copy/wl-paste to back up and restore your clipboard')
        code, targets, _ = command(['wl-paste', '--list-types'], env)
        if code:
            # Confirm empty using wl-copy's peer; an arbitrary connection error
            # must never be mistaken for an empty clipboard.
            code, _, detail = command(['wl-paste', '--no-newline'], env)
            if code and b'Nothing is copied' in detail:
                empty = True
            else:
                raise RuntimeError('Cannot safely determine the previous clipboard contents')
        else:
            formats = set(targets.decode().splitlines())
            if formats - TEXT_TYPES:
                report['result'] = 'skipped'
                raise RuntimeError('Clipboard includes non-text formats. Copy some plain text and run again; nothing was replaced.')
            code, backup, detail = command(['wl-paste', '--no-newline'], env)
            if code:
                raise RuntimeError('Could not preserve the previous clipboard text')
            backup.decode('utf-8')
        if not args.clipboard_only:
            fd = os.open('/dev/uinput', os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            os.close(fd)
        os.environ['GDK_BACKEND'] = 'wayland'
        import gi
        gi.require_version('Gtk', '3.0')
        gi.require_version('Gdk', '3.0')
        from gi.repository import Gtk, Gdk, GLib
        if not Gtk.init_check([])[0]:
            raise RuntimeError('GTK could not connect to Wayland')
        marker = uuid.uuid4().hex[:12]
        expected = [f'Voiced native clipboard {marker}: café — 日本語 🎤\n',
                    f'Second transcript {marker}.\nTwo lines, same clipboard owner.\n']
        for name, text in zip(('a.txt', 'b.txt'), expected):
            (root / name).write_text(text)
        log = (root / 'native.log').open('wb')
        native = subprocess.Popen([str(BINARY), *(['--fallback'] if args.fallback else []),
                                   *([] if args.clipboard_only else ['--paste']),
                                   str(root / 'a.txt'), str(root / 'b.txt')],
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log, bufsize=0)
        log.close()
        os.set_blocking(native.stdout.fileno(), False)
        window = Gtk.Window(title='Voiced native clipboard verification')
        window.set_default_size(740, 280)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_border_width(16)
        label = Gtk.Label(label='Keep this window focused and release modifier keys.\nThe test copies and pastes three times. Escape cancels.')
        view = Gtk.TextView()
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        box.pack_start(label, False, False, 0)
        box.pack_start(view, True, True, 0)
        window.add(box)
        window.show_all()
        view.grab_focus()
        state = {'phase': 'ready', 'deadline': time.monotonic() + 15,
                 'ready': False, 'round': 0, 'buffer': b'', 'error': None,
                 'started': 0.0, 'focus_lost': False, 'done': False,
                 'not_before': time.monotonic() + 1.1}
        rounds = [('a', expected[0]), ('b', expected[1]), ('a', expected[0])]
        def finish(error=None):
            if state['done']:
                return
            state['done'] = True
            state['error'] = error
            Gtk.main_quit()
        def send(text):
            native.stdin.write(text.encode())
            native.stdin.flush()
        def focused():
            return window.is_active() and view.has_focus()
        def key_press(_view, event):
            if event.keyval == Gdk.KEY_Escape:
                finish('Cancelled')
                return True
            return False
        def lost_focus(*_):
            if state['phase'] == 'acquiring':
                state['focus_lost'] = True
            if state['phase'] == 'pasting':
                finish('Focus changed while pasting; no retry was attempted')
        def tick():
            nonlocal published
            if state['done']:
                return False
            try:
                now = time.monotonic()
                if now > state['deadline']:
                    raise RuntimeError('Timed out during ' + state['phase'])
                if native.poll() is not None:
                    raise RuntimeError('Native client exited; see native.log')
                while select.select([native.stdout], [], [], 0)[0]:
                    data = os.read(native.stdout.fileno(), 4096)
                    if not data:
                        break
                    state['buffer'] += data
                while b'\n' in state['buffer']:
                    line, state['buffer'] = state['buffer'].split(b'\n', 1)
                    event = json.loads(line)
                    if event['event'] == 'error':
                        raise RuntimeError('Native client reported ' + event['kind'])
                    if event['event'] == 'ready':
                        report['mode'] = event['mode']
                        state['ready'] = True
                    if event['event'] == 'acquired':
                        state['phase'] = 'focus'
                        state['acquired'] = now
                        state['deadline'] = now + 3
                if state['phase'] == 'ready' and state['ready'] and focused() and now >= state['not_before']:
                    modifiers = Gdk.Keymap.get_for_display(window.get_display()).get_modifier_state()
                    if modifiers & (Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.SHIFT_MASK | Gdk.ModifierType.MOD1_MASK | Gdk.ModifierType.SUPER_MASK):
                        return True
                    view.get_buffer().set_text('')
                    state.update(phase='acquiring', started=now, deadline=now + 4, focus_lost=False)
                    published = True
                    send(rounds[state['round']][0])
                elif state['phase'] == 'focus' and focused() and now - state['acquired'] > .15:
                    state['phase'] = 'pasting'
                    state['deadline'] = now + 4
                    if args.clipboard_only:
                        view.emit('paste-clipboard')
                    else:
                        send('p')
                elif state['phase'] == 'pasting':
                    buffer = view.get_buffer()
                    actual = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
                    if actual == rounds[state['round']][1]:
                        report['rounds'].append({'round': state['round'] + 1,
                                                 'focus_restored': True,
                                                 'focus_loss_observed': state['focus_lost'],
                                                 'text_matches': True,
                                                 'duration_ms': round((now - state['started']) * 1000, 2)})
                        state['round'] += 1
                        if state['round'] == len(rounds):
                            finish()
                        else:
                            state.update(phase='ready', deadline=now + 5)
                return True
            except Exception as exc:
                finish(str(exc))
                return False
        view.connect('key-press-event', key_press)
        window.connect('focus-out-event', lost_focus)
        window.connect('delete-event', lambda *_: finish('Window closed') or True)
        GLib.timeout_add(20, tick)
        signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(finish, 'Interrupted'))
        print('Opening the test window. Keep it focused and release modifier keys.', flush=True)
        Gtk.main()
        window.destroy()
        if state['error']:
            raise RuntimeError(state['error'])
        report['result'] = 'passed'
        print('PASS: Three exact pastes; focus returned after each publication.', flush=True)
    except Exception as exc:
        report['message'] = str(exc)
        print(f'{report["result"].upper()}: {exc}', flush=True)
    finally:
        # Stop optional keyboard injection before any restoration changes focus.
        if native and native.poll() is None:
            native.terminate()
            try:
                native.wait(timeout=3)
            except subprocess.TimeoutExpired:
                native.kill()
                native.wait()
        if published:
            try:
                code, current, detail = command(['wl-paste', '--no-newline'], env)
                # A clipboard manager may retain the test text after owner exit.
                # If selection is empty or still ours, restore; preserve a newer
                # user copy. This check/replacement is necessarily best effort.
                if code != 0 and b'Nothing is copied' not in detail:
                    raise RuntimeError('Could not determine clipboard ownership for restoration')
                if code != 0 or current in [text.encode() for text in expected]:
                    code, _, _ = command(['wl-copy', '--clear'] if empty else ['wl-copy', '--type', 'text/plain;charset=utf-8'], env, backup)
                    report['clipboard_restoration'] = 'restored' if code == 0 else 'failed'
                else:
                    report['clipboard_restoration'] = 'newer_selection_preserved'
            except Exception as exc:
                report['clipboard_restoration'] = 'failed'
                report['restoration_error'] = str(exc)
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Clipboard restoration: ' + report['clipboard_restoration'], flush=True)
        print('Report: ' + str(root / 'report.json'), flush=True)
    return 0 if report['result'] == 'passed' and report['clipboard_restoration'] != 'failed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
