#!/usr/bin/env bash

# Run the reduced Figure 3 Attention grid for a quick trend check.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_attention.sh" "$@" \
  --total-tokens "8192,65536,524288,1048576"
