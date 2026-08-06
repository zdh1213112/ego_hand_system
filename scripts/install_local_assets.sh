#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE=""
REPLACE=0

usage() {
  cat <<EOF
Install a private/local EGO asset archive into:
  ${PROJECT_DIR}

The archive must contain paths beginning with:
  models/hand_landmarker.task
  models/mano/MANO_LEFT.pkl
  models/mano/MANO_RIGHT.pkl
  third_party/orbbec_sdk/
  third_party/basalt_runtime/

Usage:
  $0 --archive /path/ego_hand_assets.tar.gz
  $0 --archive /path/ego_hand_assets.tar.gz --replace
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

[[ -n "$ARCHIVE" ]] || { usage >&2; exit 2; }
ARCHIVE="$(realpath "$ARCHIVE")"
[[ -f "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }

LISTING="$(mktemp)"
trap 'rm -f "$LISTING"' EXIT
if ! tar -tzf "$ARCHIVE" > "$LISTING"; then
  echo "error: archive is not a readable gzip tar archive" >&2
  exit 1
fi

if awk 'index($0, "../") || $0 ~ /^\// { bad=1 } END { exit bad }' "$LISTING"; then
  :
else
  echo "error: archive contains an unsafe absolute or parent path" >&2
  exit 1
fi

required=(
  models/hand_landmarker.task
  models/mano/MANO_LEFT.pkl
  models/mano/MANO_RIGHT.pkl
)
for path in "${required[@]}"; do
  if ! grep -Fxq "$path" "$LISTING"; then
    echo "error: archive is missing $path" >&2
    exit 1
  fi
done

if [[ "$REPLACE" -ne 1 ]]; then
  for path in "${required[@]}"; do
    if [[ -e "$PROJECT_DIR/$path" ]]; then
      echo "error: target already exists: $PROJECT_DIR/$path" >&2
      echo "use --replace only for an archive you trust" >&2
      exit 1
    fi
  done
fi

mkdir -p "$PROJECT_DIR/models/mano" "$PROJECT_DIR/third_party"
tar -xzf "$ARCHIVE" -C "$PROJECT_DIR"
echo "Installed private/local assets into $PROJECT_DIR"
"${PROJECT_DIR}/scripts/check_third_party.py" --require-mano --require-live --require-basalt
