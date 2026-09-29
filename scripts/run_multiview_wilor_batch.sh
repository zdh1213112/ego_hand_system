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
PIPELINE=""
CONDA_ENV=""
DEVICE=""
DETECTOR="${EGO_WILOR_DETECTOR:-}"
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
MARKER_GLOBAL_MIN_MATCHES=""
MARKER_GLOBAL_MIN_FINGER_GROUPS=""
MARKER_SEARCH_PADDING_PX=""
MARKER_BBOX_PADDING_PX=""
MARKER_SEED_DISTANCE_PX=""
MARKER_MATCH_DISTANCE_PX=""
MARKER_MAX_SHIFT_PX=""
MARKER_MAX_APPLIED_SHIFT_PX=""
MARKER_BLEND=""
MARKER_MAX_LOCAL_ADJUSTMENT_PX=""
SAM3_GLOVE_ASSIST=""
SAM3_CONDA_ENV=""
SAM3_TOOL_ROOT=""
SAM3_MODEL=""
SAM3_PROMPT=""
SAM3_DEVICE=""
SAM3_CONF=""
SAM3_MIN_SCORE=""
SAM3_MIN_BOX_IOU=""
SAM3_MIN_JOINT_SUPPORT=""
SAM3_HARD_JOINT_SUPPORT=""
SAM3_TEXT_ONLY=""
SAM3_REFINE_2D=""
SAM3_MAX_2D_CORRECTION_PX=""
SAM3_2D_CORRECTION_BLEND=""
SAM3_MAX_2D_BONE_CHANGE_RATIO=""
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
  --pipeline 0|1                 Overlap next GPU inference with prior CPU postprocess
  --conda-env NAME                Forward to single-recording script
  --device cuda|cpu|auto          Forward to single-recording script
  --detector FILE                 WiLoR detector checkpoint
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
  --marker-min-matches N          Evidence threshold (default 3)
  --marker-min-finger-groups N    Evidence finger threshold (default 2)
  --marker-global-min-matches N   Whole-hand correction threshold (default 8)
  --marker-global-min-finger-groups N (default 4)
  --marker-search-padding-px PX
  --marker-bbox-padding-px PX
  --marker-seed-distance-px PX
  --marker-match-distance-px PX
  --marker-max-shift-px PX
  --marker-max-applied-shift-px PX
  --marker-blend VALUE            Conservative local marker pull (default 0.15)
  --marker-max-local-adjustment-px PX
  --sam3-glove-assist 0|1         Refine 2D joints with SAM3 and gate fusion views
  --sam3-conda-env NAME           SAM3 environment (default annotator)
  --sam3-tool-root DIR
  --sam3-model FILE
  --sam3-prompt TEXT
  --sam3-device cuda|cpu
  --sam3-conf VALUE
  --sam3-min-score VALUE
  --sam3-min-box-iou VALUE
  --sam3-min-joint-support VALUE
  --sam3-hard-joint-support VALUE
  --sam3-text-only 0|1
  --sam3-refine-2d 0|1
  --sam3-max-2d-correction-px PX
  --sam3-2d-correction-blend VALUE
  --sam3-max-2d-bone-change-ratio VALUE
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
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --detector) DETECTOR="$2"; shift 2 ;;
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
    --marker-global-min-matches) MARKER_GLOBAL_MIN_MATCHES="$2"; shift 2 ;;
    --marker-global-min-finger-groups) MARKER_GLOBAL_MIN_FINGER_GROUPS="$2"; shift 2 ;;
    --marker-search-padding-px) MARKER_SEARCH_PADDING_PX="$2"; shift 2 ;;
    --marker-bbox-padding-px) MARKER_BBOX_PADDING_PX="$2"; shift 2 ;;
    --marker-seed-distance-px) MARKER_SEED_DISTANCE_PX="$2"; shift 2 ;;
    --marker-match-distance-px) MARKER_MATCH_DISTANCE_PX="$2"; shift 2 ;;
    --marker-max-shift-px) MARKER_MAX_SHIFT_PX="$2"; shift 2 ;;
    --marker-max-applied-shift-px) MARKER_MAX_APPLIED_SHIFT_PX="$2"; shift 2 ;;
    --marker-blend) MARKER_BLEND="$2"; shift 2 ;;
    --marker-max-local-adjustment-px) MARKER_MAX_LOCAL_ADJUSTMENT_PX="$2"; shift 2 ;;
    --sam3-glove-assist|--sam3-assist) SAM3_GLOVE_ASSIST="$2"; shift 2 ;;
    --sam3-conda-env) SAM3_CONDA_ENV="$2"; shift 2 ;;
    --sam3-tool-root) SAM3_TOOL_ROOT="$2"; shift 2 ;;
    --sam3-model) SAM3_MODEL="$2"; shift 2 ;;
    --sam3-prompt) SAM3_PROMPT="$2"; shift 2 ;;
    --sam3-device) SAM3_DEVICE="$2"; shift 2 ;;
    --sam3-conf) SAM3_CONF="$2"; shift 2 ;;
    --sam3-min-score) SAM3_MIN_SCORE="$2"; shift 2 ;;
    --sam3-min-box-iou) SAM3_MIN_BOX_IOU="$2"; shift 2 ;;
    --sam3-min-joint-support) SAM3_MIN_JOINT_SUPPORT="$2"; shift 2 ;;
    --sam3-hard-joint-support) SAM3_HARD_JOINT_SUPPORT="$2"; shift 2 ;;
    --sam3-text-only) SAM3_TEXT_ONLY="$2"; shift 2 ;;
    --sam3-refine-2d) SAM3_REFINE_2D="$2"; shift 2 ;;
    --sam3-max-2d-correction-px) SAM3_MAX_2D_CORRECTION_PX="$2"; shift 2 ;;
    --sam3-2d-correction-blend) SAM3_2D_CORRECTION_BLEND="$2"; shift 2 ;;
    --sam3-max-2d-bone-change-ratio) SAM3_MAX_2D_BONE_CHANGE_RATIO="$2"; shift 2 ;;
    --detector-handedness) DETECTOR_HANDEDNESS="$2"; shift 2 ;;
    --no-video) NO_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PIPELINE" ]]; then
  [[ "$GPU_PROFILE" == "rtx5090d" && "$SAM3_GLOVE_ASSIST" != "1" ]] && PIPELINE=1 || PIPELINE=0
fi
if [[ "$SAM3_GLOVE_ASSIST" == "1" && "$PIPELINE" == "1" ]]; then
  echo "[batch] SAM3 uses the GPU during postprocess; disabling --pipeline 1 to avoid GPU contention"
  PIPELINE=0
fi

[[ -n "$INPUT_DIR" && -d "$INPUT_DIR" ]] || { echo "Missing --input-dir: $INPUT_DIR" >&2; exit 2; }
[[ -n "$OUTPUT_ROOT" ]] || { usage >&2; exit 2; }
[[ "$RECURSIVE" =~ ^[01]$ && "$CONTINUE_ON_ERROR" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ && "$PIPELINE" =~ ^[01]$ ]] || {
  echo "recursive/continue-on-error/dry-run/pipeline must be 0 or 1" >&2; exit 2;
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
case "$SAM3_GLOVE_ASSIST" in
  ""|0|1) ;;
  *) echo "--sam3-glove-assist must be 0 or 1: $SAM3_GLOVE_ASSIST" >&2; exit 2 ;;
esac
case "$SAM3_TEXT_ONLY" in
  ""|0|1) ;;
  *) echo "--sam3-text-only must be 0 or 1: $SAM3_TEXT_ONLY" >&2; exit 2 ;;
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

record_completion() {
  local mcap="$1" output="$2" name="$3" exit_code="$4" started="$5" log="$6"
  local finished status
  finished="$(date +%s)"
  if ((exit_code == 0)); then
    status="success"
    echo "[batch] success: ${name}"
  else
    status="failed"
    echo "[batch] failed: ${name} (exit ${exit_code})" >&2
  fi
  append_json_status \
    "$mcap" "$output" "$status" "$exit_code" "$started" "$finished" "$log"
}

PENDING_PID=""
PENDING_MCAP=""
PENDING_OUTPUT=""
PENDING_NAME=""
PENDING_STARTED=""
PENDING_LOG=""
LAST_POST_FAILED=0

finish_pending_postprocess() {
  local exit_code
  LAST_POST_FAILED=0
  [[ -n "$PENDING_PID" ]] || return 0
  set +e
  wait "$PENDING_PID"
  exit_code=$?
  set -e
  record_completion \
    "$PENDING_MCAP" "$PENDING_OUTPUT" "$PENDING_NAME" "$exit_code" \
    "$PENDING_STARTED" "$PENDING_LOG"
  ((exit_code != 0)) && LAST_POST_FAILED=1
  PENDING_PID=""
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
  [[ -n "$DETECTOR" ]] && CMD+=(--detector "$DETECTOR")
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
  [[ -n "$MARKER_GLOBAL_MIN_MATCHES" ]] && CMD+=(--marker-global-min-matches "$MARKER_GLOBAL_MIN_MATCHES")
  [[ -n "$MARKER_GLOBAL_MIN_FINGER_GROUPS" ]] && CMD+=(--marker-global-min-finger-groups "$MARKER_GLOBAL_MIN_FINGER_GROUPS")
  [[ -n "$MARKER_SEARCH_PADDING_PX" ]] && CMD+=(--marker-search-padding-px "$MARKER_SEARCH_PADDING_PX")
  [[ -n "$MARKER_BBOX_PADDING_PX" ]] && CMD+=(--marker-bbox-padding-px "$MARKER_BBOX_PADDING_PX")
  [[ -n "$MARKER_SEED_DISTANCE_PX" ]] && CMD+=(--marker-seed-distance-px "$MARKER_SEED_DISTANCE_PX")
  [[ -n "$MARKER_MATCH_DISTANCE_PX" ]] && CMD+=(--marker-match-distance-px "$MARKER_MATCH_DISTANCE_PX")
  [[ -n "$MARKER_MAX_SHIFT_PX" ]] && CMD+=(--marker-max-shift-px "$MARKER_MAX_SHIFT_PX")
  [[ -n "$MARKER_MAX_APPLIED_SHIFT_PX" ]] && CMD+=(--marker-max-applied-shift-px "$MARKER_MAX_APPLIED_SHIFT_PX")
  [[ -n "$MARKER_BLEND" ]] && CMD+=(--marker-blend "$MARKER_BLEND")
  [[ -n "$MARKER_MAX_LOCAL_ADJUSTMENT_PX" ]] && CMD+=(--marker-max-local-adjustment-px "$MARKER_MAX_LOCAL_ADJUSTMENT_PX")
  [[ -n "$SAM3_GLOVE_ASSIST" ]] && CMD+=(--sam3-glove-assist "$SAM3_GLOVE_ASSIST")
  [[ -n "$SAM3_CONDA_ENV" ]] && CMD+=(--sam3-conda-env "$SAM3_CONDA_ENV")
  [[ -n "$SAM3_TOOL_ROOT" ]] && CMD+=(--sam3-tool-root "$SAM3_TOOL_ROOT")
  [[ -n "$SAM3_MODEL" ]] && CMD+=(--sam3-model "$SAM3_MODEL")
  [[ -n "$SAM3_PROMPT" ]] && CMD+=(--sam3-prompt "$SAM3_PROMPT")
  [[ -n "$SAM3_DEVICE" ]] && CMD+=(--sam3-device "$SAM3_DEVICE")
  [[ -n "$SAM3_CONF" ]] && CMD+=(--sam3-conf "$SAM3_CONF")
  [[ -n "$SAM3_MIN_SCORE" ]] && CMD+=(--sam3-min-score "$SAM3_MIN_SCORE")
  [[ -n "$SAM3_MIN_BOX_IOU" ]] && CMD+=(--sam3-min-box-iou "$SAM3_MIN_BOX_IOU")
  [[ -n "$SAM3_MIN_JOINT_SUPPORT" ]] && CMD+=(--sam3-min-joint-support "$SAM3_MIN_JOINT_SUPPORT")
  [[ -n "$SAM3_HARD_JOINT_SUPPORT" ]] && CMD+=(--sam3-hard-joint-support "$SAM3_HARD_JOINT_SUPPORT")
  [[ -n "$SAM3_TEXT_ONLY" ]] && CMD+=(--sam3-text-only "$SAM3_TEXT_ONLY")
  [[ -n "$SAM3_REFINE_2D" ]] && CMD+=(--sam3-refine-2d "$SAM3_REFINE_2D")
  [[ -n "$SAM3_MAX_2D_CORRECTION_PX" ]] && CMD+=(--sam3-max-2d-correction-px "$SAM3_MAX_2D_CORRECTION_PX")
  [[ -n "$SAM3_2D_CORRECTION_BLEND" ]] && CMD+=(--sam3-2d-correction-blend "$SAM3_2D_CORRECTION_BLEND")
  [[ -n "$SAM3_MAX_2D_BONE_CHANGE_RATIO" ]] && CMD+=(--sam3-max-2d-bone-change-ratio "$SAM3_MAX_2D_BONE_CHANGE_RATIO")
  [[ -n "$DETECTOR_HANDEDNESS" ]] && CMD+=(--detector-handedness "$DETECTOR_HANDEDNESS")
  ((${#CAMERAS[@]})) && CMD+=(--cameras "${CAMERAS[@]}")
  ((NO_VIDEO)) && CMD+=(--no-video)

  echo "[batch] ${NAME} -> ${RECORD_OUTPUT}"
  if ((DRY_RUN)); then
    if ((PIPELINE)); then
      printf '  '
      printf '%q ' "${CMD[@]}" --stop-after-inference
      printf '\n  '
      printf '%q ' "${CMD[@]}"
      printf '\n'
    else
      printf '  '
      printf '%q ' "${CMD[@]}"
      printf '\n'
    fi
    FINISHED="$(date +%s)"
    append_json_status "$MCAP" "$RECORD_OUTPUT" "dry_run" 0 "$STARTED" "$FINISHED" "$LOG_PATH"
    continue
  fi

  if ((PIPELINE)); then
    echo "[batch] GPU inference phase: ${NAME}"
    set +e
    "${CMD[@]}" --stop-after-inference 2>&1 | tee "$LOG_PATH"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    # The previous CPU/NVENC phase has been hidden behind this GPU phase.
    finish_pending_postprocess
    if ((EXIT_CODE != 0)); then
      record_completion \
        "$MCAP" "$RECORD_OUTPUT" "$NAME" "$EXIT_CODE" "$STARTED" "$LOG_PATH"
      if ((CONTINUE_ON_ERROR == 0)); then
        break
      fi
      continue
    fi

    if ((LAST_POST_FAILED && CONTINUE_ON_ERROR == 0)); then
      # This inference was already in flight when the previous failure became
      # visible. Complete its postprocess, then honor stop-on-error.
      echo "[batch] CPU/NVENC postprocess phase: ${NAME}"
      set +e
      "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
      EXIT_CODE=${PIPESTATUS[0]}
      set -e
      record_completion \
        "$MCAP" "$RECORD_OUTPUT" "$NAME" "$EXIT_CODE" "$STARTED" "$LOG_PATH"
      break
    fi

    echo "[batch] launch CPU/NVENC postprocess: ${NAME}"
    (
      set +e
      "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
      phase_exit=${PIPESTATUS[0]}
      exit "$phase_exit"
    ) &
    PENDING_PID=$!
    PENDING_MCAP="$MCAP"
    PENDING_OUTPUT="$RECORD_OUTPUT"
    PENDING_NAME="$NAME"
    PENDING_STARTED="$STARTED"
    PENDING_LOG="$LOG_PATH"
    continue
  fi

  set +e
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e
  record_completion \
    "$MCAP" "$RECORD_OUTPUT" "$NAME" "$EXIT_CODE" "$STARTED" "$LOG_PATH"
  if ((EXIT_CODE != 0 && CONTINUE_ON_ERROR == 0)); then
    break
  fi
done

finish_pending_postprocess

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
