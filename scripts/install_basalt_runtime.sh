#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VERSION="0.1.7"
ARCHIVE_NAME="basalt-0.1.7-x86_64-unknown-linux-gnu.tar.gz"
DEFAULT_URL="https://gitlab.com/VladyslavUsenko/basalt/-/releases/${VERSION}/downloads/${ARCHIVE_NAME}"
EXPECTED_SHA256="8ab56b2ab27315a9c64cb8ae9c99d0d4d894e2cc87bc9841d8defe30b86e6d24"
DEST="${PROJECT_DIR}/third_party/basalt_runtime"
ARCHIVE=""
REPLACE=0

if [[ -x "${DEST}/bin/basalt_vio" && -f "${DEST}/lib/libbasalt.so" && -f "${DEST}/VERSION" ]]; then
  if grep -Fq "sha256=${EXPECTED_SHA256}" "${DEST}/VERSION"; then
    echo "Basalt runtime already installed and verified: ${DEST}"
    exit 0
  fi
fi

usage() {
  cat <<EOF
Install the official Basalt ${VERSION} Linux x86_64 runtime into:
  ${DEST}

Usage:
  $0                         download official release and verify SHA-256
  $0 --archive /path/file    install an already downloaded archive
  $0 --replace               replace an existing runtime after verification

Environment:
  BASALT_DOWNLOAD_URL        override the release URL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) [[ $# -ge 2 ]] || { echo "--archive requires a path" >&2; exit 2; }; ARCHIVE="$2"; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: this runtime is for Linux x86_64; build Basalt from the submodule on this platform." >&2
  exit 1
fi

if [[ -z "$ARCHIVE" ]]; then
  command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  ARCHIVE="${TMP_DIR}/${ARCHIVE_NAME}"
  URL="${BASALT_DOWNLOAD_URL:-${DEFAULT_URL}}"
  echo "Downloading ${URL}"
  curl --fail --location --retry 3 --output "$ARCHIVE" "$URL"
else
  ARCHIVE="$(realpath "$ARCHIVE")"
  [[ -f "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }
fi

ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "error: Basalt archive SHA-256 mismatch" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  exit 1
fi
echo "SHA-256 verified: $ACTUAL_SHA256"

if [[ -e "$DEST" && -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "$REPLACE" -ne 1 ]]; then
  echo "error: runtime directory is not empty: $DEST" >&2
  echo "use --replace only when replacing a verified runtime" >&2
  exit 1
fi

TMP_EXTRACT="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR:-}" "$TMP_EXTRACT"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP_EXTRACT"
RELEASE="$TMP_EXTRACT/release"
[[ -x "$RELEASE/bin/basalt_vio" && -f "$RELEASE/lib/libbasalt.so" ]] || {
  echo "error: invalid Basalt archive structure" >&2
  exit 1
}

rm -rf "$DEST"
mkdir -p "$DEST/bin" "$DEST/lib"
cp "$RELEASE/bin/basalt_vio" "$DEST/bin/basalt_vio"
cp "$RELEASE/lib/libbasalt.so" "$DEST/lib/libbasalt.so"
chmod +x "$DEST/bin/basalt_vio"
cat > "$DEST/VERSION" <<EOF
Basalt ${VERSION}
artifact=${ARCHIVE_NAME}
sha256=${EXPECTED_SHA256}
EOF
cat > "$DEST/README.md" <<EOF
# Basalt runtime (local, not committed)

Basalt ${VERSION} Linux x86_64 runtime installed by scripts/install_basalt_runtime.sh.

Artifact: ${ARCHIVE_NAME}
SHA-256: ${EXPECTED_SHA256}

The source is the pinned Git submodule at third_party/basalt.
EOF

echo "Installed Basalt runtime to $DEST"
"${PROJECT_DIR}/scripts/check_third_party.py" --require-basalt
