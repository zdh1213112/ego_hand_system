#!/usr/bin/env bash
set -Eeuo pipefail

# Batch wrapper around run_multiview_wilor_experiment.sh. Each MCAP gets an
# independent output directory, so per-recording resume/config checks remain active.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE="$ROOT/scripts/run_multiview_wilor_experiment.sh"
INPUT_DIR=""
OUTPUT_ROOT=""
PATTERN="*.mcap"
RECURSIVE=0
CONTINUE_ON_ERROR=1
DRY_RUN=0
CONDA_ENV=""
DEVICE=""
GPU_PROFILE=""
REFERENCE_CAMERA=""
MAX_FRAMES=""
BATCH_SIZE=""
FRAME_BATCH_SIZE=""
PREPROCESS_WORKERS=""
MAX_DETECTIONS_PER_CLASS=""
COMPILE_BACKBONE=""
FUSION_WORKERS=""
NO_VIDEO=0
DETECTOR_HANDEDNESS=""
GLOVE_MARKER_ASSIST=""
MARKER_SATURATION_MAX=""
MARKER_VALUE_MIN=""
MARKER_MIN_MATCHES=""
MARKER_MIN_FINGER_GROUPS=""
MARKER_SEARCH_PADDING_PX=""
MARKER_SEED_DISTANCE_PX=""
MARKER_MATCH_DISTANCE_PX=""
MARKER_MAX_SHIFT_PX=""
MARKER_BLEND=""
CAMERAS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_multiview_wilor_batch.sh \
    --input-dir /path/to/mcaps \
    --output-root /path/to/batch_output \
    [options]

Options:
  --input-dir DIR                 Folder containing MCAP files
  --output-root DIR               One subdirectory per MCAP
  --pattern GLOB                  File pattern (default *.mcap)
  --recursive                     Search input-dir recursively
  --continue-on-error 0|1         Continue after a failed MCAP (default 1)
  --dry-run                       Print commands without running them
  --conda-env NAME                Forward to single-recording script
  --device cuda|cpu|auto          Forward to single-recording script
  --gpu-profile compatible|rtx5090d
  --cameras CAMERA ...            Forward camera list until the next --option
  --reference-camera CAMERA
  --max-frames N
  --batch-size N
  --frame-batch-size N
  --preprocess-workers N
  --max-detections-per-class N
  --compile-backbone 0|1
  --fusion-workers N
  --glove-marker-assist 0|1       Detect reflective glove balls in RGB frames
  --nokov-wilor-assist 0|1        Compatibility alias for glove-marker-assist
  --marker-saturation-max N       HSV saturation threshold (default 100)
  --marker-value-min N            HSV brightness threshold (default 160)
  --marker-min-matches N          Minimum one-to-one matches (default 5)
  --marker-min-finger-groups N    Minimum covered fingers (default 3)
  --marker-search-padding-px PX
  --marker-seed-distance-px PX
  --marker-match-distance-px PX
  --marker-max-shift-px PX
  --marker-blend VALUE            0 keeps shifted WiLoR, 1 uses marker centers
  --detector-handedness strict|ignore|adaptive
  --no-video
EOF
}

while (($#)); do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --recursive) RECURSIVE=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --gpu-profile) GPU_PROFILE="$2"; shift 2 ;;
    --cameras)
      CAMERAS=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do CAMERAS+=("$1"); shift; done
      ;;
    --reference-camera) REFERENCE_CAMERA="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --frame-batch-size) FRAME_BATCH_SIZE="$2"; shift 2 ;;
    --preprocess-workers) PREPROCESS_WORKERS="$2"; shift 2 ;;
    --max-detections-per-class) MAX_DETECTIONS_PER_CLASS="$2"; shift 2 ;;
    --compile-backbone) COMPILE_BACKBONE="$2"; shift 2 ;;
    --fusion-workers) FUSION_WORKERS="$2"; shift 2 ;;
    --glove-marker-assist|--nokov-wilor-assist) GLOVE_MARKER_ASSIST="$2"; shift 2 ;;
    --marker-saturation-max) MARKER_SATURATION_MAX="$2"; shift 2 ;;
    --marker-value-min) MARKER_VALUE_MIN="$2"; shift 2 ;;
    --marker-min-matches) MARKER_MIN_MATCHES="$2"; shift 2 ;;
    --marker-min-finger-groups) MARKER_MIN_FINGER_GROUPS="$2"; shift 2 ;;
    --marker-search-padding-px) MARKER_SEARCH_PADDING_PX="$2"; shift 2 ;;
    --marker-seed-distance-px) MARKER_SEED_DISTANCE_PX="$2"; shift 2 ;;
    --marker-match-distance-px) MARKER_MATCH_DISTANCE_PX="$2"; shift 2 ;;
    --marker-max-shift-px) MARKER_MAX_SHIFT_PX="$2"; shift 2 ;;
    --marker-blend) MARKER_BLEND="$2"; shift 2 ;;
    --detector-handedness) DETECTOR_HANDEDNESS="$2"; shift 2 ;;
    --no-video) NO_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$INPUT_DIR" && -d "$INPUT_DIR" ]] || { echo "Missing --input-dir: $INPUT_DIR" >&2; exit 2; }
[[ -n "$OUTPUT_ROOT" ]] || { usage >&2; exit 2; }
[[ "$RECURSIVE" =~ ^[01]$ && "$CONTINUE_ON_ERROR" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || {
  echo "recursive/continue-on-error/dry-run must be 0 or 1" >&2; exit 2;
}
[[ -x "$SINGLE" ]] || { echo "Single-recording script is not executable: $SINGLE" >&2; exit 2; }
case "$DETECTOR_HANDEDNESS" in
  ""|strict|ignore|adaptive) ;;
  *) echo "--detector-handedness must be strict, ignore, or adaptive: $DETECTOR_HANDEDNESS" >&2; exit 2 ;;
esac
case "$GLOVE_MARKER_ASSIST" in
  ""|0|1) ;;
  *) echo "--glove-marker-assist must be 0 or 1: $GLOVE_MARKER_ASSIST" >&2; exit 2 ;;
esac

INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
STATUS_JSONL="$OUTPUT_ROOT/batch_status.jsonl"
: > "$STATUS_JSONL"

mapfile -d '' MCAPS < <(
  if ((RECURSIVE)); then
    find "$INPUT_DIR" -type f -name "$PATTERN" -print0 | sort -z
  else
    find "$INPUT_DIR" -maxdepth 1 -type f -name "$PATTERN" -print0 | sort -z
  fi
)
if ((${#MCAPS[@]} == 0)); then
  echo "No MCAP files found in $INPUT_DIR (pattern: $PATTERN)" >&2
  exit 2
fi

declare -A OUTPUT_SEEN=()
for MCAP in "${MCAPS[@]}"; do
  NAME="$(basename "$MCAP")"
  STEM="${NAME%.*}"
  if ((RECURSIVE)); then
    RELATIVE="${MCAP#"$INPUT_DIR"/}"
    STEM="${RELATIVE%.*}"
    STEM="${STEM//\//__}"
  fi
  [[ -n "$STEM" ]] || { echo "Cannot derive output name from $MCAP" >&2; exit 2; }
  [[ -z "${OUTPUT_SEEN[$STEM]:-}" ]] || {
    echo "Output name collision for MCAP files: $STEM" >&2; exit 2;
  }
  OUTPUT_SEEN[$STEM]="$MCAP"
done

append_json_status() {
  local mcap="$1" output="$2" status="$3" exit_code="$4" started="$5" finished="$6" log="$7"
  python3 - "$STATUS_JSONL" "$mcap" "$output" "$status" "$exit_code" "$started" "$finished" "$log" <<'PY'
import json
from pathlib import Path
import sys
path, mcap, output, status, exit_code, started, finished, log = sys.argv[1:]
row = {
    "mcap": mcap, "output": output, "status": status,
    "exit_code": int(exit_code), "started_epoch": int(started),
    "finished_epoch": int(finished), "elapsed_seconds": int(finished) - int(started),
    "log": log,
}
with Path(path).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

for MCAP in "${MCAPS[@]}"; do
  NAME="$(basename "$MCAP")"
  STEM="${NAME%.*}"
  if ((RECURSIVE)); then
    RELATIVE="${MCAP#"$INPUT_DIR"/}"
    STEM="${RELATIVE%.*}"
    STEM="${STEM//\//__}"
  fi
  RECORD_OUTPUT="$OUTPUT_ROOT/$STEM"
  LOG_PATH="$OUTPUT_ROOT/${STEM}.log"
  STARTED="$(date +%s)"
  CMD=("$SINGLE" --mcap "$MCAP" --output "$RECORD_OUTPUT")
  [[ -n "$CONDA_ENV" ]] && CMD+=(--conda-env "$CONDA_ENV")
  [[ -n "$DEVICE" ]] && CMD+=(--device "$DEVICE")
  [[ -n "$GPU_PROFILE" ]] && CMD+=(--gpu-profile "$GPU_PROFILE")
  [[ -n "$REFERENCE_CAMERA" ]] && CMD+=(--reference-camera "$REFERENCE_CAMERA")
  [[ -n "$MAX_FRAMES" ]] && CMD+=(--max-frames "$MAX_FRAMES")
  [[ -n "$BATCH_SIZE" ]] && CMD+=(--batch-size "$BATCH_SIZE")
  [[ -n "$FRAME_BATCH_SIZE" ]] && CMD+=(--frame-batch-size "$FRAME_BATCH_SIZE")
  [[ -n "$PREPROCESS_WORKERS" ]] && CMD+=(--preprocess-workers "$PREPROCESS_WORKERS")
  [[ -n "$MAX_DETECTIONS_PER_CLASS" ]] && CMD+=(--max-detections-per-class "$MAX_DETECTIONS_PER_CLASS")
  [[ -n "$COMPILE_BACKBONE" ]] && CMD+=(--compile-backbone "$COMPILE_BACKBONE")
  [[ -n "$FUSION_WORKERS" ]] && CMD+=(--fusion-workers "$FUSION_WORKERS")
  [[ -n "$GLOVE_MARKER_ASSIST" ]] && CMD+=(--glove-marker-assist "$GLOVE_MARKER_ASSIST")
  [[ -n "$MARKER_SATURATION_MAX" ]] && CMD+=(--marker-saturation-max "$MARKER_SATURATION_MAX")
  [[ -n "$MARKER_VALUE_MIN" ]] && CMD+=(--marker-value-min "$MARKER_VALUE_MIN")
  [[ -n "$MARKER_MIN_MATCHES" ]] && CMD+=(--marker-min-matches "$MARKER_MIN_MATCHES")
  [[ -n "$MARKER_MIN_FINGER_GROUPS" ]] && CMD+=(--marker-min-finger-groups "$MARKER_MIN_FINGER_GROUPS")
  [[ -n "$MARKER_SEARCH_PADDING_PX" ]] && CMD+=(--marker-search-padding-px "$MARKER_SEARCH_PADDING_PX")
  [[ -n "$MARKER_SEED_DISTANCE_PX" ]] && CMD+=(--marker-seed-distance-px "$MARKER_SEED_DISTANCE_PX")
  [[ -n "$MARKER_MATCH_DISTANCE_PX" ]] && CMD+=(--marker-match-distance-px "$MARKER_MATCH_DISTANCE_PX")
  [[ -n "$MARKER_MAX_SHIFT_PX" ]] && CMD+=(--marker-max-shift-px "$MARKER_MAX_SHIFT_PX")
  [[ -n "$MARKER_BLEND" ]] && CMD+=(--marker-blend "$MARKER_BLEND")
  [[ -n "$DETECTOR_HANDEDNESS" ]] && CMD+=(--detector-handedness "$DETECTOR_HANDEDNESS")
  ((${#CAMERAS[@]})) && CMD+=(--cameras "${CAMERAS[@]}")
  ((NO_VIDEO)) && CMD+=(--no-video)

  echo "[batch] ${NAME} -> ${RECORD_OUTPUT}"
  if ((DRY_RUN)); then
    printf '  '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    FINISHED="$(date +%s)"
    append_json_status "$MCAP" "$RECORD_OUTPUT" "dry_run" 0 "$STARTED" "$FINISHED" "$LOG_PATH"
    continue
  fi

  set +e
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e
  FINISHED="$(date +%s)"
  if ((EXIT_CODE == 0)); then
    append_json_status "$MCAP" "$RECORD_OUTPUT" "success" 0 "$STARTED" "$FINISHED" "$LOG_PATH"
    echo "[batch] success: ${NAME}"
  else
    append_json_status "$MCAP" "$RECORD_OUTPUT" "failed" "$EXIT_CODE" "$STARTED" "$FINISHED" "$LOG_PATH"
    echo "[batch] failed: ${NAME} (exit ${EXIT_CODE})" >&2
    if ((CONTINUE_ON_ERROR == 0)); then
      break
    fi
  fi
done

python3 - "$STATUS_JSONL" "$OUTPUT_ROOT/batch_summary.json" "$INPUT_DIR" "$PATTERN" <<'PY'
import json
from pathlib import Path
import sys
status_path, summary_path, input_dir, pattern = sys.argv[1:]
rows = [json.loads(line) for line in Path(status_path).read_text(encoding="utf-8").splitlines() if line.strip()]
summary = {
    "stage": "batch_multiview_wilor_experiment",
    "input_dir": input_dir,
    "pattern": pattern,
    "total": len(rows),
    "success": sum(row["status"] == "success" for row in rows),
    "failed": sum(row["status"] == "failed" for row in rows),
    "dry_run": sum(row["status"] == "dry_run" for row in rows),
    "records": rows,
}
Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("total", "success", "failed", "dry_run")}, ensure_ascii=False))
PY

if python3 - "$STATUS_JSONL" <<'PY'
import json
from pathlib import Path
import sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
raise SystemExit(0 if all(row["status"] != "failed" for row in rows) else 1)
PY
then
  exit 0
else
  exit 1
fi
