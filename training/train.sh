#!/usr/bin/env bash
set -euo pipefail

training="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'EOF'
Usage: bash training/train.sh {wilor|fah|gem-x} [options]

  wilor   Train with v4 raw-image hand crops by default. For KB4-rectified hand crops,
          use --config training/configs/glove/nokov_v4_bbox_pinhole.yaml.
  fah     Fine-tune or evaluate FAH-S. See fah --help for V0–V5 and continuation.
  gem-x   Fine-tune or evaluate GEM-X EgoHand. See gem-x --help for required assets.

Set EGO_TRAIN_PYTHON to the Python executable for the selected model.
EOF
}

case "${1:-}" in
  ""|-h|--help) usage; exit 0 ;;
  wilor)
    shift
    if [[ $# == 0 || ($# == 2 && $1 == --mode && $2 == joint_lora) ||
          ($# == 1 && $1 == --mode=joint_lora) ]]; then
      set -- --mode joint_lora --config "$training/configs/glove/nokov_v4_joint_lora.yaml"
    fi
    set -- "$training/train_glove.py" "$@"
    ;;
  fah|gem-x) set -- "$training/train_hand.py" "$@" ;;
  *) usage >&2; exit 2 ;;
esac

if [[ -n "${EGO_TRAIN_PYTHON:-}" ]]; then
  exec "$EGO_TRAIN_PYTHON" -B "$@"
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == wilor_training ]]; then
  exec "$CONDA_PREFIX/bin/python" -B "$@"
fi
exec conda run --no-capture-output -n wilor_training python -B "$@"
