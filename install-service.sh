#!/bin/bash
#
# Install voiced as a systemd user service
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/voiced.service"

echo "Installing voiced service..."
echo "  Source: $SCRIPT_DIR/voiced"
echo "  Service: $SERVICE_FILE"
echo

# Check voiced script exists
if [[ ! -x "$SCRIPT_DIR/voiced" ]]; then
    echo "Error: $SCRIPT_DIR/voiced not found or not executable" >&2
    exit 1
fi

# Check venv exists
if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "Error: .venv not found. Run 'uv sync' first." >&2
    exit 1
fi

# Create systemd user directory
mkdir -p "$SERVICE_DIR"

# Create service file
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Voice dictation daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$SCRIPT_DIR/voiced serve
Restart=on-failure
RestartSec=1
RestartPreventExitStatus=69 78
TimeoutStopSec=10
KillMode=control-group
RuntimeDirectory=voiced
RuntimeDirectoryMode=0700
StandardOutput=journal
StandardError=journal
SyslogIdentifier=voiced

# Environment for Wayland/audio
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=XDG_RUNTIME_DIR=/run/user/%U

[Install]
WantedBy=default.target
EOF

echo "Created service file."

# Reload systemd
systemctl --user daemon-reload
echo "Reloaded systemd."

# Enable and start service
systemctl --user enable voiced
echo "Enabled service."

systemctl --user restart voiced

ready=false
for _ in {1..240}; do
    if "$SCRIPT_DIR/voiced" status 2>/dev/null | grep -q "Daemon running"; then
        ready=true
        break
    fi
    if systemctl --user is-failed --quiet voiced; then
        break
    fi
    sleep 0.25
done
if [[ "$ready" != true ]]; then
    echo "Error: voiced did not become ready. Recent logs:" >&2
    journalctl --user -u voiced -n 20 --no-pager >&2
    exit 1
fi
echo "Started service and opened its control socket."

echo
echo "Done! Service installed and running."
echo
echo "Useful commands:"
echo "  systemctl --user status voiced    # check status"
echo "  journalctl --user -u voiced -f    # view logs"
echo "  systemctl --user restart voiced   # restart"
echo "  systemctl --user stop voiced      # stop"
echo "  systemctl --user disable voiced   # disable autostart"
