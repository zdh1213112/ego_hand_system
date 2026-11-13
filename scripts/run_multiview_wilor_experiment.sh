#!/usr/bin/env bash
# Orbbec EGO4 -> shared MANO -> final four-view video and selected results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_hand_inference.sh" "$@"
