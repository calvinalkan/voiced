#!/bin/bash
# Process-level integration coverage for CLI, daemon, model worker, and recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export VOICED_INSTANCE="integration-$BASHPID"
RUNTIME_DIRECTORY="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/voiced-$VOICED_INSTANCE"
DAEMON_LOG="/tmp/voiced-$VOICED_INSTANCE.log"
RESULT_FILE="/tmp/voiced-$VOICED_INSTANCE-result.txt"
BUSY_RESULT_FILE="/tmp/voiced-$VOICED_INSTANCE-busy-result.txt"
DUPLICATE_LOG="/tmp/voiced-$VOICED_INSTANCE-duplicate.log"
DAEMON_PID=""

cleanup() {
    if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill -TERM "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    rm -rf "$RUNTIME_DIRECTORY"
    rm -f "$DAEMON_LOG" "$RESULT_FILE" "$BUSY_RESULT_FILE" "$DUPLICATE_LOG"
}
trap cleanup EXIT

start_daemon() {
    : > "$DAEMON_LOG"
    ./voiced serve \
        --debug \
        --transcriber-engine whisper \
        --model base \
        --no-streaming \
        >"$DAEMON_LOG" 2>&1 &
    DAEMON_PID=$!

    for _ in {1..50}; do
        if ./voiced status | grep -q "Daemon running"; then
            return
        fi
        if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
            echo "FAIL: daemon exited during startup"
            cat "$DAEMON_LOG"
            exit 1
        fi
        sleep 0.1
    done

    echo "FAIL: daemon did not open its control socket"
    cat "$DAEMON_LOG"
    exit 1
}

wait_for_file() {
    local path="$1"
    for _ in {1..100}; do
        [[ -f "$path" ]] && return
        sleep 0.1
    done
    echo "FAIL: timed out waiting for $path"
    cat "$DAEMON_LOG"
    exit 1
}

normalize_transcript() {
    tr '[:upper:]' '[:lower:]' | sed -E 's/[[:punct:]]//g; s/[[:space:]]+/ /g; s/^ //; s/ $//'
}

echo "=== Invalid explicit config exits CONFIG without loading a model ==="
set +e
./voiced serve --config "/tmp/voiced-$VOICED_INSTANCE-missing.json" >"$DUPLICATE_LOG" 2>&1
CONFIG_STATUS=$?
set -e
if [[ "$CONFIG_STATUS" -ne 78 ]] || grep -q "loading model" "$DUPLICATE_LOG"; then
    echo "FAIL: invalid explicit config did not fail early with 78/CONFIG"
    cat "$DUPLICATE_LOG"
    exit 1
fi

echo "=== Unix socket + full transcription pipeline ==="
start_daemon

set +e
./voiced serve --transcriber-engine whisper --model base --no-streaming \
    >"$DUPLICATE_LOG" 2>&1
DUPLICATE_STATUS=$?
set -e
if [[ "$DUPLICATE_STATUS" -ne 69 ]] || grep -q "loading model" "$DUPLICATE_LOG"; then
    echo "FAIL: duplicate daemon did not reject before model loading"
    cat "$DUPLICATE_LOG"
    exit 1
fi

python3 - "$RUNTIME_DIRECTORY/control.sock" <<'PY'
import socket
import sys
import time

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect(sys.argv[1])
    time.sleep(0.7)
PY
./voiced status | grep -q "phase: idle"
grep -q "invalid request: timed out" "$DAEMON_LOG"

rm -f "$RESULT_FILE"
VOICED_TEST_INPUT="$SCRIPT_DIR/test-fixtures/hello_world.wav" \
VOICED_TEST_OUTPUT="$RESULT_FILE" \
./voiced listen
wait_for_file "$RESULT_FILE"

RESULT=$(<"$RESULT_FILE")
EXPECTED=$(<"$SCRIPT_DIR/test-fixtures/hello_world.expected")
NORMALIZED_RESULT=$(printf '%s' "$RESULT" | normalize_transcript)
NORMALIZED_EXPECTED=$(printf '%s' "$EXPECTED" | normalize_transcript)
if [[ "$NORMALIZED_RESULT" != "$NORMALIZED_EXPECTED" ]]; then
    echo "FAIL: transcription does not match"
    diff <(printf '%s\n' "$EXPECTED") <(printf '%s\n' "$RESULT") || true
    exit 1
fi
./voiced status | grep -q "phase: idle"

echo "=== Toggle ignored while transcribing ==="
rm -f "$BUSY_RESULT_FILE"
VOICED_TEST_INPUT="$SCRIPT_DIR/test-fixtures/paragraph.wav" \
VOICED_TEST_OUTPUT="$BUSY_RESULT_FILE" \
./voiced listen

TRANSCRIBING=false
for _ in {1..100}; do
    if ./voiced status | grep -q "phase: transcribing"; then
        TRANSCRIBING=true
        break
    fi
    sleep 0.02
done
if [[ "$TRANSCRIBING" != true ]]; then
    echo "FAIL: did not observe transcribing phase"
    cat "$DAEMON_LOG"
    exit 1
fi
./voiced record -t
wait_for_file "$BUSY_RESULT_FILE"
grep -q "ignored toggle phase=transcribing" "$DAEMON_LOG"
./voiced status | grep -q "phase: idle"

echo "=== Ordered clean shutdown ==="
./voiced kill >/dev/null
wait "$DAEMON_PID"
DAEMON_PID=""
grep -q "event=shutdown_complete" "$DAEMON_LOG"
[[ ! -S "$RUNTIME_DIRECTORY/control.sock" ]]

echo "=== Callback stall preserves speech and keeps a closed stream recoverable ==="
start_daemon
rm -f "$RESULT_FILE"
VOICED_TEST_CAPTURE_FAULT=stall \
VOICED_TEST_INPUT="$SCRIPT_DIR/test-fixtures/hello_world.wav" \
VOICED_TEST_OUTPUT="$RESULT_FILE" \
    ./voiced record
wait_for_file "$RESULT_FILE"
./voiced status | grep -q "phase: idle"
grep -q "event=callback_stall" "$DAEMON_LOG"
grep -q "event=recovery_transcription.*reason=callback_stall" "$DAEMON_LOG"
grep -q "event=capture_recovered.*action=kept-running" "$DAEMON_LOG"
if grep -q "event=recovery_restart" "$DAEMON_LOG"; then
    echo "FAIL: a closed failed stream unnecessarily restarted the daemon"
    exit 1
fi
RESULT="$(<"$RESULT_FILE")"
NORMALIZED_RESULT="$(printf '%s' "$RESULT" | normalize_transcript)"
if [[ "$NORMALIZED_RESULT" != "$NORMALIZED_EXPECTED" ]]; then
    echo "FAIL: callback recovery did not preserve buffered speech"
    echo "Expected: $EXPECTED"
    echo "Got:      $RESULT"
    exit 1
fi
./voiced kill >/dev/null
wait "$DAEMON_PID"
DAEMON_PID=""

echo "=== Callback stall preserves buffered speech before TEMPFAIL ==="
start_daemon
rm -f "$RESULT_FILE"
VOICED_TEST_CAPTURE_FAULT=hang \
VOICED_TEST_INPUT="$SCRIPT_DIR/test-fixtures/hello_world.wav" \
VOICED_TEST_OUTPUT="$RESULT_FILE" \
    ./voiced record

set +e
wait "$DAEMON_PID"
DAEMON_STATUS=$?
set -e
DAEMON_PID=""
if [[ "$DAEMON_STATUS" -ne 75 ]]; then
    echo "FAIL: stuck capture exited $DAEMON_STATUS, expected 75/TEMPFAIL"
    cat "$DAEMON_LOG"
    exit 1
fi
grep -q "event=callback_stall" "$DAEMON_LOG"
grep -q "event=capture_stop_requested.*reason=callback-stall" "$DAEMON_LOG"
grep -q "event=recovery_transcription.*reason=capture_stop_timeout" "$DAEMON_LOG"
grep -q "event=recovery_restart reason=capture_stop_timeout" "$DAEMON_LOG"
RESULT="$(<"$RESULT_FILE")"
NORMALIZED_RESULT="$(printf '%s' "$RESULT" | normalize_transcript)"
if [[ "$NORMALIZED_RESULT" != "$NORMALIZED_EXPECTED" ]]; then
    echo "FAIL: callback recovery did not preserve buffered speech"
    echo "Expected: $EXPECTED"
    echo "Got:      $RESULT"
    exit 1
fi

echo "=== SIGKILL leaves a stale socket that the next daemon replaces ==="
start_daemon
kill -KILL "$DAEMON_PID"
set +e
wait "$DAEMON_PID"
KILLED_STATUS=$?
set -e
DAEMON_PID=""
if [[ "$KILLED_STATUS" -ne 137 || ! -S "$RUNTIME_DIRECTORY/control.sock" ]]; then
    echo "FAIL: SIGKILL did not leave the expected stale socket"
    exit 1
fi

start_daemon
./voiced status | grep -q "phase: idle"
./voiced kill >/dev/null
wait "$DAEMON_PID"
DAEMON_PID=""

echo "PASS: integration pipeline, lifecycle, and recovery"
