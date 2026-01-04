#!/bin/bash
#
# Build and install dotool - layout-aware keyboard input tool
#
set -euo pipefail

echo "=== Building dotool ==="
echo ""

# Check dependencies
if ! pkg-config --exists xkbcommon 2>/dev/null; then
    echo "Missing libxkbcommon-dev. Install with:"
    echo "  sudo apt install libxkbcommon-dev"
    exit 1
fi

if ! which go >/dev/null 2>&1; then
    echo "Missing go. Install with:"
    echo "  sudo apt install golang"
    exit 1
fi

# Clone/update
DOTOOL_DIR="/tmp/dotool-build"
rm -rf "$DOTOOL_DIR"
git clone https://git.sr.ht/~geb/dotool "$DOTOOL_DIR"
cd "$DOTOOL_DIR"

# Build
echo "Building..."
GOCACHE=/tmp/go-cache GOMODCACHE=/tmp/go-mod go build

# Install to ~/.local/bin
mkdir -p ~/.local/bin
cp "$DOTOOL_DIR/dotool" ~/.local/bin/
echo ""
echo "Installed to: ~/.local/bin/dotool"
echo ""
echo "For udev rules (required for non-root usage):"
echo "  sudo cp $DOTOOL_DIR/80-dotool.rules /etc/udev/rules.d/"
echo "  sudo udevadm control --reload && sudo udevadm trigger"
echo ""
echo "Test with:"
echo "  echo 'Hello äöü ß' | dotool type"
