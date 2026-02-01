#!/bin/bash
# Install protoc for NanoDeploy (etcd-client dependency).
# Run: bash scripts/setup-protoc.sh
# Or: apt-get install -y protobuf-compiler  # Debian/Ubuntu

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROTOC="${SCRIPT_DIR}/.bin/protoc"

if command -v protoc &>/dev/null; then
    echo "protoc already installed: $(protoc --version)"
    exit 0
fi

if [[ -x "$LOCAL_PROTOC" ]]; then
    echo "Using local protoc: $LOCAL_PROTOC"
    echo "Add to your shell: export PATH=\"${SCRIPT_DIR}/.bin:\$PATH\""
    echo "Or: export PROTOC=\"$LOCAL_PROTOC\""
    exit 0
fi

echo "Installing protoc..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update && sudo apt-get install -y protobuf-compiler
elif command -v brew &>/dev/null; then
    brew install protobuf
elif command -v conda &>/dev/null; then
    conda install -y protobuf
else
    # Download prebuilt from GitHub
    mkdir -p "${SCRIPT_DIR}/.bin"
    ARCH=$(uname -m)
    [[ "$ARCH" == "x86_64" ]] && ARCH="x86_64" || true
    [[ "$ARCH" == "aarch64" ]] && ARCH="aarch_64" || true
    URL="https://github.com/protocolbuffers/protobuf/releases/download/v28.3/protoc-28.3-linux-${ARCH}.zip"
    echo "Downloading $URL ..."
    (cd /tmp && curl -sL -o protoc.zip "$URL" && unzip -o protoc.zip && cp bin/protoc "${SCRIPT_DIR}/.bin/" && chmod +x "${SCRIPT_DIR}/.bin/protoc")
    echo "Done. Add: export PATH=\"${SCRIPT_DIR}/.bin:\$PATH\""
fi
echo "protoc: $(protoc --version 2>/dev/null || "${LOCAL_PROTOC}" --version)"
