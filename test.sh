#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use a separate instance for testing to avoid conflicts with running daemon
export VOICED_INSTANCE="test"

# Cleanup function
cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    rm -f /tmp/voiced-test-result.txt
}
trap cleanup EXIT

# Stop any existing test daemon first
voiced kill 2>/dev/null || true
sleep 1

echo "=== Starting daemon ==="
voiced serve --debug --model base &
DAEMON_PID=$!
sleep 3

# Check daemon is running
if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "FAIL: Daemon failed to start"
    exit 1
fi
echo "Daemon running (pid: $DAEMON_PID)"

echo ""
echo "=== Running transcription test ==="
rm -f /tmp/voiced-test-result.txt
VOICED_TEST_INPUT="$SCRIPT_DIR/test-fixtures/hello_world.wav" \
VOICED_TEST_OUTPUT=/tmp/voiced-test-result.txt \
voiced listen

# Check exit code
if [ $? -ne 0 ]; then
    echo "FAIL: voiced listen returned non-zero exit code"
    exit 1
fi

# Wait for output file (transcription happens async)
echo "Waiting for transcription..."
for i in {1..30}; do
    if [ -f /tmp/voiced-test-result.txt ]; then
        break
    fi
    sleep 0.5
done

# Check output file exists
if [ ! -f /tmp/voiced-test-result.txt ]; then
    echo "FAIL: Output file not created (timeout)"
    exit 1
fi

# Check transcription result
RESULT=$(cat /tmp/voiced-test-result.txt)
EXPECTED=$(cat "$SCRIPT_DIR/test-fixtures/hello_world.expected")

echo ""
echo "=== Checking result ==="
echo "Expected: $EXPECTED"
echo "Got:      $RESULT"

if [ "$RESULT" = "$EXPECTED" ]; then
    echo ""
    echo "PASS: Transcription matches expected output"
    exit 0
else
    echo ""
    echo "FAIL: Transcription does not match"
    diff <(echo "$EXPECTED") <(echo "$RESULT") || true
    exit 1
fi
