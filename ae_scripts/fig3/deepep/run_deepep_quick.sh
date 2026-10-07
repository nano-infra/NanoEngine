#!/usr/bin/env bash

# Run the reduced Figure 3 DeepEP grid for a quick trend check.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_deepep.sh" "$@" \
  --batch-sizes "1 4 8 32 56 80 104 128 152 176 200 224 256"
