#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EXPECTED_SHA256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEST="${PROJECT_DIR}/models/hand_landmarker.task"
REPLACE=0

[[ "${1:-}" == "--replace" ]] && REPLACE=1
if [[ -e "$DEST" && "$REPLACE" -ne 1 ]]; then
  actual="$(sha256sum "$DEST" | awk '{print $1}')"
  if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
    echo "MediaPipe model already installed and verified: $DEST"
    exit 0
  fi
  echo "error: existing model checksum differs; use --replace deliberately" >&2
  exit 1
fi

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl --fail --location --retry 3 --output "$tmp" "$URL"
actual="$(sha256sum "$tmp" | awk '{print $1}')"
[[ "$actual" == "$EXPECTED_SHA256" ]] || {
  echo "error: MediaPipe model SHA-256 mismatch" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $actual" >&2
  exit 1
}
mkdir -p "$(dirname "$DEST")"
cp "$tmp" "$DEST"
echo "Installed and verified MediaPipe model: $DEST"

