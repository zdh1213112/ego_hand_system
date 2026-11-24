#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${EGO_NATIVE_PYTHON:-/home/p3/miniforge3/envs/ego-hand/bin/python}" "$TOOL_DIR/server.py" "$@"
