#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE="$ROOT/scripts/run_multiview_wilor_label_export.sh"
MERGER="$ROOT/scripts/merge_wilor_training_datasets.py"
CHECKER="$ROOT/scripts/check_wilor_training_dataset.py"
RENDERER="$ROOT/scripts/render_wilor_training_dataset.py"

EXPERIMENT_ROOT=""
OUTPUT_ROOT=""
LAYOUT="separate"
EXPERIMENT_PATTERN="*"
FUSION_NAME="fusion_multiview"
LABEL_DIR_NAME="wilor_labels_camera2_complete21"
CONDA_ENV="ego-hand"
DEVICE="cuda"
VIEW_FILTER="complete21"
MAX_SAMPLES=0
SAMPLE_STRIDE=0
CONTINUE_ON_ERROR=1
DRY_RUN=0
RENDER_VISUALIZATION=1
VISUALIZATION_SAMPLES=12
VISUALIZATION_SEED=42
MANO_SOURCE="${GLOVE_MANO_SOURCE:-$ROOT/third_party/MANO}"
MANO_MODEL_DIR="${GLOVE_MANO_MODEL_DIR:-$ROOT/models/mano}"
REFERENCE_NPY="/home/zdh/tool/npy_decoder/000865.npy"
EXPORT_CAMERAS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_multiview_wilor_label_batch.sh \
    --experiment-root /path/to/multiview_batch_output \
    --layout separate|merged \
    [--output-root /path/to/label_output]

Output layouts:
  separate  Each experiment writes its own label directory. Without --output-root:
            <experiment>/<label-dir-name>/
            With --output-root: <output-root>/<experiment-name>/
  merged    Per-recording workspaces go to <output-root>/runs/<experiment-name>/
            and the final combined training set goes to <output-root>/dataset/

Options:
  --experiment-root DIR
  --experiment-pattern GLOB       Experiment directory pattern (default *)
  --output-root DIR
  --layout separate|merged        Default separate
  --fusion-name NAME              Fusion subdirectory (default fusion_multiview)
  --label-dir-name NAME           In-place label directory name
  --conda-env NAME                Default ego-hand
  --device cuda|cpu|auto          Default cuda
  --export-camera CAMERA          Repeat for multiple views (default camera2+camera3)
  --view-filter legacy|complete21 Default complete21
  --max-samples N                 Default 0
  --sample-stride N               Default 0 (motion adaptive)
  --render-visualization 0|1      Default 1
  --visualization-samples N       Default 12
  --visualization-seed N          Default 42
  --continue-on-error 0|1         Default 1
  --mano-source DIR
  --mano-model-dir DIR
  --reference-npy FILE
  --dry-run
EOF
}

while (($#)); do
  case "$1" in
    --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
    --experiment-pattern) EXPERIMENT_PATTERN="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --layout) LAYOUT="$2"; shift 2 ;;
    --fusion-name) FUSION_NAME="$2"; shift 2 ;;
    --label-dir-name) LABEL_DIR_NAME="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --export-camera) EXPORT_CAMERAS+=("$2"); shift 2 ;;
    --view-filter) VIEW_FILTER="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --sample-stride) SAMPLE_STRIDE="$2"; shift 2 ;;
    --render-visualization) RENDER_VISUALIZATION="$2"; shift 2 ;;
    --visualization-samples) VISUALIZATION_SAMPLES="$2"; shift 2 ;;
    --visualization-seed) VISUALIZATION_SEED="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR="$2"; shift 2 ;;
    --mano-source) MANO_SOURCE="$2"; shift 2 ;;
    --mano-model-dir) MANO_MODEL_DIR="$2"; shift 2 ;;
    --reference-npy) REFERENCE_NPY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$EXPERIMENT_ROOT" && -d "$EXPERIMENT_ROOT" ]] || {
  echo "Missing --experiment-root: $EXPERIMENT_ROOT" >&2; exit 2;
}
[[ "$LAYOUT" == "separate" || "$LAYOUT" == "merged" ]] || {
  echo "--layout must be separate or merged" >&2; exit 2;
}
[[ "$VIEW_FILTER" == "legacy" || "$VIEW_FILTER" == "complete21" ]] || {
  echo "--view-filter must be legacy or complete21" >&2; exit 2;
}
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ && "$SAMPLE_STRIDE" =~ ^[0-9]+$ ]] || {
  echo "max-samples/sample-stride must be non-negative integers" >&2; exit 2;
}
[[ "$RENDER_VISUALIZATION" =~ ^[01]$ && "$CONTINUE_ON_ERROR" =~ ^[01]$ ]] || {
  echo "render-visualization/continue-on-error must be 0 or 1" >&2; exit 2;
}
[[ "$VISUALIZATION_SAMPLES" =~ ^[1-9][0-9]*$ && "$VISUALIZATION_SEED" =~ ^[0-9]+$ ]] || {
  echo "invalid visualization samples/seed" >&2; exit 2;
}
if [[ "$LAYOUT" == "merged" && -z "$OUTPUT_ROOT" ]]; then
  echo "--layout merged requires --output-root" >&2
  exit 2
fi

EXPERIMENT_ROOT="$(cd "$EXPERIMENT_ROOT" && pwd)"
if [[ -n "$OUTPUT_ROOT" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
fi

EXPERIMENTS=()
if [[ -f "$EXPERIMENT_ROOT/normalized_multiview/manifest.json" ]]; then
  if [[ -s "$EXPERIMENT_ROOT/$FUSION_NAME/accepted.jsonl" && -f "$EXPERIMENT_ROOT/$FUSION_NAME/summary.json" ]]; then
    EXPERIMENTS+=("$EXPERIMENT_ROOT")
  fi
else
  while IFS= read -r -d '' candidate; do
    [[ -f "$candidate/normalized_multiview/manifest.json" ]] || continue
    [[ -s "$candidate/$FUSION_NAME/accepted.jsonl" && -f "$candidate/$FUSION_NAME/summary.json" ]] || continue
    EXPERIMENTS+=("$candidate")
  done < <(find "$EXPERIMENT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$EXPERIMENT_PATTERN" -print0 | sort -z)
fi
if ((${#EXPERIMENTS[@]} == 0)); then
  echo "No completed multiview experiments found in $EXPERIMENT_ROOT" >&2
  exit 2
fi

if [[ -n "$OUTPUT_ROOT" ]]; then
  STATUS_JSONL="$OUTPUT_ROOT/label_batch_status.jsonl"
  SUMMARY_JSON="$OUTPUT_ROOT/label_batch_summary.json"
else
  STATUS_JSONL="$EXPERIMENT_ROOT/label_batch_status.jsonl"
  SUMMARY_JSON="$EXPERIMENT_ROOT/label_batch_summary.json"
fi
: > "$STATUS_JSONL"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" \
    env PYTHONPATH="$ROOT/scripts" MPLCONFIGDIR="/tmp/ego-hand-matplotlib" python "$@"
}

append_status() {
  python3 - "$STATUS_JSONL" "$1" "$2" "$3" "$4" <<'PY'
import json
from pathlib import Path
import sys
path, experiment, output, status, exit_code = sys.argv[1:]
with Path(path).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "experiment": experiment, "output": output,
        "status": status, "exit_code": int(exit_code),
    }, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

DATASETS=()
FAILED=0
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  NAME="$(basename "$EXPERIMENT")"
  if [[ "$LAYOUT" == "merged" ]]; then
    LABEL_OUTPUT="$OUTPUT_ROOT/runs/$NAME"
  elif [[ -n "$OUTPUT_ROOT" ]]; then
    LABEL_OUTPUT="$OUTPUT_ROOT/$NAME"
  else
    LABEL_OUTPUT="$EXPERIMENT/$LABEL_DIR_NAME"
  fi
  FUSION="$EXPERIMENT/$FUSION_NAME"
  CMD=(
    "$SINGLE" --experiment "$EXPERIMENT" --fusion "$FUSION"
    --output "$LABEL_OUTPUT" --conda-env "$CONDA_ENV" --device "$DEVICE"
    --view-filter "$VIEW_FILTER" --max-samples "$MAX_SAMPLES"
    --sample-stride "$SAMPLE_STRIDE"
    --visualization-samples "$VISUALIZATION_SAMPLES"
    --visualization-seed "$VISUALIZATION_SEED"
    --mano-source "$MANO_SOURCE" --mano-model-dir "$MANO_MODEL_DIR"
    --reference-npy "$REFERENCE_NPY"
  )
  for camera in "${EXPORT_CAMERAS[@]}"; do CMD+=(--export-camera "$camera"); done
  if [[ "$LAYOUT" == "merged" ]]; then
    CMD+=(--render-visualization 0)
  else
    CMD+=(--render-visualization "$RENDER_VISUALIZATION")
  fi

  echo "[label-batch] $NAME -> $LABEL_OUTPUT"
  if ((DRY_RUN)); then
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    append_status "$EXPERIMENT" "$LABEL_OUTPUT" "dry_run" 0
    continue
  fi
  set +e
  "${CMD[@]}"
  EXIT_CODE=$?
  set -e
  if ((EXIT_CODE == 0)); then
    append_status "$EXPERIMENT" "$LABEL_OUTPUT" "success" 0
    DATASETS+=("$LABEL_OUTPUT/dataset")
  else
    append_status "$EXPERIMENT" "$LABEL_OUTPUT" "failed" "$EXIT_CODE"
    FAILED=1
    if ((CONTINUE_ON_ERROR == 0)); then break; fi
  fi
done

if [[ "$LAYOUT" == "merged" && "$DRY_RUN" == 0 && "$FAILED" == 0 && ${#DATASETS[@]} -gt 0 ]]; then
  MERGED_DATASET="$OUTPUT_ROOT/dataset"
  echo "[label-batch] merge ${#DATASETS[@]} datasets -> $MERGED_DATASET"
  run_python "$MERGER" --inputs "${DATASETS[@]}" --output "$MERGED_DATASET"
  CHECK_ARGS=(
    "$CHECKER" "$MERGED_DATASET" --mano-source "$MANO_SOURCE"
    --mano-model-dir "$MANO_MODEL_DIR"
  )
  [[ -f "$REFERENCE_NPY" ]] && CHECK_ARGS+=(--reference "$REFERENCE_NPY")
  run_python "${CHECK_ARGS[@]}"
  if ((RENDER_VISUALIZATION)); then
    RENDER_ARGS=(
      "$RENDERER" --dataset "$MERGED_DATASET"
      --output "$MERGED_DATASET/visualization"
      --samples "$VISUALIZATION_SAMPLES" --seed "$VISUALIZATION_SEED"
    )
    ((${#EXPORT_CAMERAS[@]})) && RENDER_ARGS+=(--cameras "${EXPORT_CAMERAS[@]}")
    run_python "${RENDER_ARGS[@]}"
  fi
fi

python3 - "$STATUS_JSONL" "$SUMMARY_JSON" "$LAYOUT" <<'PY'
import json
from pathlib import Path
import sys
status_path, summary_path, layout = sys.argv[1:]
rows = [json.loads(line) for line in Path(status_path).read_text(encoding="utf-8").splitlines() if line.strip()]
summary = {
    "stage": "batch_multiview_wilor_label_export", "layout": layout,
    "total": len(rows),
    "success": sum(row["status"] == "success" for row in rows),
    "failed": sum(row["status"] == "failed" for row in rows),
    "dry_run": sum(row["status"] == "dry_run" for row in rows),
    "records": rows,
}
Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("total", "success", "failed", "dry_run")}, ensure_ascii=False))
PY

((FAILED == 0))
