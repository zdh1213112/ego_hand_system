#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
    cat <<'EOF'
Run the offline stereo hand pipeline for Orbbec EGO sessions or GEN DAS EGO MCAPs.

Required environment variables:
  EGO_SOURCE=orbbec|gen
  EGO_OUTPUT=output/<experiment-name>

Orbbec input:
  EGO_SESSION=recordings/Orbbec_Ego_<serial>_<time>

GEN input:
  EGO_MCAP=/path/to/recording.mcap
  EGO_LEFT_CAMERA=camera2       # optional
  EGO_RIGHT_CAMERA=camera3      # optional

Usage:
  ./scripts/run_offline.sh [check|prepare|stereo|stabilize|fit|render|wilor|all]

The positional stage overrides EGO_STAGE. The default stage is "all".

Useful optional variables:
  EGO_DEVICE=auto|cuda|cpu      MANO device; default: auto
  EGO_MAX_PAIRS=0               limit downstream pairs; 0 means all
  EGO_MAX_FRAMES=0              normalization frame limit; 0 means all (GEN only)
  EGO_NO_VIDEO=0|1              skip diagnostic videos; final CSVs remain
  EGO_CONDA_ENV=ego-hand        Conda environment used when not already active
  EGO_PYTHON=/path/to/python    override Python/Conda selection
  EGO_NORMALIZED_DATASET=...    reuse/relocate the normalized dataset
  EGO_RECTIFIED_DATASET=...     reuse/relocate the rectified dataset
  EGO_HAND_ROUTE=mediapipe|wilor|parallel  processing route; default: mediapipe
  EGO_WILOR_CAMERAS=left|right|both         WiLoR cameras; default: both
  EGO_WILOR_DEVICE_LEFT=...                left WiLoR device; default: EGO_DEVICE
  EGO_WILOR_DEVICE_RIGHT=...               right WiLoR device; default: EGO_DEVICE
  EGO_WILOR_FRAME_STRIDE=1                 WiLoR frame stride
  EGO_WILOR_BATCH_SIZE=16                  WiLoR inference batch size
  EGO_WILOR_FAST=0|1                       enable WiLoR fast CUDA mode
  EGO_WILOR_SAVE_VERTICES=0|1              include 778 vertices in WiLoR JSONL

Examples:
  export EGO_SOURCE=orbbec
  export EGO_SESSION=recordings/Orbbec_Ego_AZER764008C_20260806_110653
  export EGO_OUTPUT=output/recording_20260806_110653_run2
  ./scripts/run_offline.sh check
  ./scripts/run_offline.sh all

  export EGO_SOURCE=gen
  export EGO_MCAP=/data/DAS-Ego_example.mcap
  export EGO_OUTPUT=output/gen_das_example
  ./scripts/run_offline.sh all
EOF
}

log() {
    printf '[offline] %s\n' "$*"
}

warn() {
    printf '[offline] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[offline] ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "missing directory: $1"
}

require_single_match() {
    local description="$1"
    shift
    local matches=("$@")
    ((${#matches[@]} == 1)) || die "expected exactly one ${description}; found ${#matches[@]}"
}

validate_non_negative_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}

stage_output_ready() {
    local label="$1"
    local directory="$2"
    local marker="$3"
    if [[ -f "${marker}" ]]; then
        log "skip ${label}: already complete (${marker})"
        return 0
    fi
    if [[ -e "${directory}" ]]; then
        die "${label} output exists but is incomplete: ${directory}. Move it away or choose a new EGO_OUTPUT."
    fi
    return 1
}

select_python() {
    if [[ -n "${EGO_PYTHON:-}" ]]; then
        PYTHON_RUN=(env PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib "${EGO_PYTHON}")
        PYTHON_DESCRIPTION="${EGO_PYTHON}"
    elif [[ "${CONDA_DEFAULT_ENV:-}" == "${EGO_CONDA_ENV}" ]]; then
        PYTHON_RUN=(env PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib python)
        PYTHON_DESCRIPTION="python (active conda env: ${EGO_CONDA_ENV})"
    elif command -v conda >/dev/null 2>&1; then
        PYTHON_RUN=(conda run --no-capture-output -n "${EGO_CONDA_ENV}" env PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib python)
        PYTHON_DESCRIPTION="conda run -n ${EGO_CONDA_ENV} python"
    else
        PYTHON_RUN=(env PYTHONNOUSERSITE=1 PYTHONPATH= MPLCONFIGDIR=/tmp/ego-hand-matplotlib python3)
        PYTHON_DESCRIPTION="python3 (conda not found)"
    fi
}

run_python() {
    "${PYTHON_RUN[@]}" "$@"
}

print_config() {
    log "source: ${EGO_SOURCE}"
    if [[ "${EGO_SOURCE}" == "orbbec" ]]; then
        log "session: ${EGO_SESSION}"
    else
        log "mcap: ${EGO_MCAP}"
        log "GEN cameras: ${EGO_LEFT_CAMERA}/${EGO_RIGHT_CAMERA}"
        log "normalized dataset: ${NORMALIZED_DATASET}"
        log "rectified dataset: ${RECTIFIED_DATASET}"
    fi
    log "output root: ${EGO_OUTPUT}"
    log "stage: ${EGO_STAGE}"
    log "python: ${PYTHON_DESCRIPTION}"
    log "MANO device: ${EGO_DEVICE}"
    log "hand route: ${EGO_HAND_ROUTE}"
    log "pair limit: ${EGO_MAX_PAIRS} (0 = all)"
    log "diagnostic videos: $([[ "${EGO_NO_VIDEO}" == "1" ]] && printf disabled || printf enabled)"
}

validate_orbbec_session() {
    require_dir "${EGO_SESSION}"
    shopt -s nullglob
    local left_videos=("${EGO_SESSION}"/*_camera_left.mp4)
    local right_videos=("${EGO_SESSION}"/*_camera_right.mp4)
    local left_pts=("${EGO_SESSION}"/*_camera_left_pts.csv)
    local right_pts=("${EGO_SESSION}"/*_camera_right_pts.csv)
    local calibrations=("${EGO_SESSION}"/*_calibration_camera.yaml)
    shopt -u nullglob
    require_single_match "Orbbec left video" "${left_videos[@]}"
    require_single_match "Orbbec right video" "${right_videos[@]}"
    require_single_match "Orbbec left timestamp CSV" "${left_pts[@]}"
    require_single_match "Orbbec right timestamp CSV" "${right_pts[@]}"
    require_single_match "Orbbec camera calibration YAML" "${calibrations[@]}"
}

validate_common_assets() {
    require_file "${EGO_MODEL}"
    require_dir "${EGO_MANO_SOURCE}"
    require_file "${EGO_MANO_SOURCE}/mano/model.py"
    require_file "${EGO_MANO_MODELS}/MANO_RIGHT.pkl"
}

validate_wilor_assets() {
    require_file "${EGO_WILOR_CHECKPOINT}"
    require_file "${EGO_WILOR_DETECTOR}"
    require_file "${EGO_WILOR_CONFIG}"
    require_file "${EGO_WILOR_MANO_DIR}/MANO_RIGHT.pkl"
}

prepare_orbbec() {
    local output="${EGO_OUTPUT}/session_check"
    local marker="${output}/stereo_rectified.jpg"
    if stage_output_ready "Orbbec session check" "${output}" "${marker}"; then
        return
    fi
    if [[ ! -x "${PROJECT_DIR}/build/ego_session_inspect" ]]; then
        warn "build/ego_session_inspect is unavailable; skipping the optional session preview"
        warn "build it with: cmake -S . -B build && cmake --build build -j\"\$(nproc)\""
        return
    fi
    log "checking Orbbec session"
    "${PROJECT_DIR}/build/ego_session_inspect" \
        --session "${EGO_SESSION}" \
        --output "${output}"
}

normalize_orbbec() {
    if stage_output_ready "Orbbec normalization" "${NORMALIZED_DATASET}" "${NORMALIZED_DATASET}/manifest.json"; then
        return
    fi
    log "normalizing Orbbec session"
    run_python "${PROJECT_DIR}/scripts/normalize_orbbec_session.py" \
        --session "${EGO_SESSION}" \
        --output "${NORMALIZED_DATASET}"
}

rectify_dataset() {
    if stage_output_ready "stereo rectification" "${RECTIFIED_DATASET}" "${RECTIFIED_DATASET}/manifest.json"; then
        return
    fi
    local rectify_limit=()
    if ((EGO_MAX_PAIRS > 0)); then
        rectify_limit=(--max-pairs "${EGO_MAX_PAIRS}")
    fi
    log "rectifying stereo dataset"
    run_python "${PROJECT_DIR}/scripts/rectify_stereo_dataset.py" \
        --input "${NORMALIZED_DATASET}" \
        --output "${RECTIFIED_DATASET}" \
        --camera-model auto \
        --focal-scale 1.0 \
        --video-codec h264 \
        --crf 18 \
        "${rectify_limit[@]}"
}

prepare_gen() {
    if ! stage_output_ready "GEN normalization" "${NORMALIZED_DATASET}" "${NORMALIZED_DATASET}/manifest.json"; then
        log "checking GEN Python dependencies and MCAP decoding"
        run_python "${PROJECT_DIR}/scripts/check_gen_environment.py" \
            --mcap "${EGO_MCAP}" \
            --camera "${EGO_LEFT_CAMERA}" \
            --other-camera "${EGO_RIGHT_CAMERA}" \
            --device "${EGO_DEVICE}"
        local normalize_limit=()
        if ((EGO_MAX_FRAMES > 0)); then
            normalize_limit=(--max-frames "${EGO_MAX_FRAMES}")
        fi
        log "normalizing GEN MCAP"
        run_python "${PROJECT_DIR}/scripts/normalize_recording.py" \
            --input "${EGO_MCAP}" \
            --output "${NORMALIZED_DATASET}" \
            --left-camera "${EGO_LEFT_CAMERA}" \
            --right-camera "${EGO_RIGHT_CAMERA}" \
            "${normalize_limit[@]}"
    fi

    rectify_dataset
}

run_prepare() {
    if [[ "${EGO_SOURCE}" == "orbbec" ]]; then
        prepare_orbbec
        normalize_orbbec
        rectify_dataset
    else
        prepare_gen
    fi
}

run_stereo() {
    local output="${EGO_OUTPUT}/mediapipe_stereo"
    local marker="${output}/summary.json"
    if stage_output_ready "stereo MediaPipe" "${output}" "${marker}"; then
        return
    fi
    require_file "${RECTIFIED_DATASET}/manifest.json"
    local source_args=(--rectified-dataset "${RECTIFIED_DATASET}")
    local pair_limit=()
    if ((EGO_MAX_PAIRS > 0)); then
        pair_limit=(--max-pairs "${EGO_MAX_PAIRS}")
    fi
    local video_args=()
    if [[ "${EGO_NO_VIDEO}" == "1" ]]; then
        video_args=(--no-video)
    fi
    log "running stereo MediaPipe and triangulation"
    run_python "${PROJECT_DIR}/scripts/mediapipe_stereo_triangulate.py" \
        "${source_args[@]}" \
        --model "${EGO_MODEL}" \
        --output "${output}" \
        --track-max-missed 75 \
        --track-max-distance-px 280 \
        --track-reacquire-distance-px 700 \
        "${pair_limit[@]}" \
        "${video_args[@]}"
}

run_stabilize() {
    local input="${EGO_OUTPUT}/mediapipe_stereo/stereo_landmarks_3d.csv"
    local output="${EGO_OUTPUT}/mano_preparation"
    local marker="${output}/mano_input.npz"
    require_file "${input}"
    if stage_output_ready "3D stabilization" "${output}" "${marker}"; then
        return
    fi
    local video_args=()
    if [[ "${EGO_NO_VIDEO}" == "1" ]]; then
        video_args=(--no-video)
    fi
    log "stabilizing stereo 3D landmarks"
    run_python "${PROJECT_DIR}/scripts/stabilize_hand_3d.py" \
        --input "${input}" \
        --output "${output}" \
        --pixel-outlier-window 4 \
        --pixel-outlier-distance 0.45 \
        --pixel-scale-ratio 1.8 \
        "${video_args[@]}"
    run_python "${PROJECT_DIR}/scripts/check_mano_assets.py" \
        --input "${marker}" \
        --input-only
}

run_fit() {
    local input="${EGO_OUTPUT}/mano_preparation/mano_input.npz"
    local initial_output="${EGO_OUTPUT}/mano_fit_right_canonical_initial_rigid"
    local final_output="${EGO_OUTPUT}/mano_fit_right_canonical_final"
    require_file "${input}"
    run_python "${PROJECT_DIR}/scripts/check_mano_assets.py" \
        --mano-source "${EGO_MANO_SOURCE}" \
        --model-dir "${EGO_MANO_MODELS}" \
        --input "${input}"

    local pair_limit=()
    if ((EGO_MAX_PAIRS > 0)); then
        pair_limit=(--max-pairs "${EGO_MAX_PAIRS}")
    fi

    if ! stage_output_ready "initial MANO fit" "${initial_output}" "${initial_output}/summary.json"; then
        log "running initial MANO fit"
        run_python "${PROJECT_DIR}/scripts/fit_mano_sequence.py" \
            --input "${input}" \
            --mano-source "${EGO_MANO_SOURCE}" \
            --model-dir "${EGO_MANO_MODELS}" \
            --mano-convention wilor_right_canonical_v1 \
            --output "${initial_output}" \
            --shape-iterations 300 \
            --pose-iterations 140 \
            --pose-window 32 \
            --pose-overlap 12 \
            --learning-rate 0.006 \
            --w-3d 1.0 \
            --w-2d 0.35 \
            --w-pinch 0.35 \
            --pinch-threshold-m 0.025 \
            --w-contact-tips 0.75 \
            --contact-tip-threshold-m 0.035 \
            --min-fit-observed-points 12 \
            --max-unobserved-gap 5 \
            --w-pose 0.0025 \
            --w-temporal 0.05 \
            --w-rigid-temporal 0.02 \
            --w-acceleration 0.015 \
            --boundary-weight 0.15 \
            --max-orient-step-deg 40 \
            --max-translation-step-m 0.04 \
            --max-pose-step 2.5 \
            --rigid-initialization \
            --no-image-rigid-alignment \
            --device "${EGO_DEVICE}" \
            --no-video \
            "${pair_limit[@]}"
    fi

    if ! stage_output_ready "final MANO fit" "${final_output}" "${final_output}/summary.json"; then
        log "running final MANO refinement"
        run_python "${PROJECT_DIR}/scripts/fit_mano_sequence.py" \
            --input "${input}" \
            --mano-source "${EGO_MANO_SOURCE}" \
            --model-dir "${EGO_MANO_MODELS}" \
            --mano-convention wilor_right_canonical_v1 \
            --output "${final_output}" \
            --initial-output "${initial_output}" \
            --shape-iterations 0 \
            --pose-iterations 100 \
            --pose-window 48 \
            --pose-overlap 16 \
            --learning-rate 0.001 \
            --w-3d 1.0 \
            --w-2d 0.30 \
            --w-pinch 0.35 \
            --pinch-threshold-m 0.025 \
            --w-contact-tips 0.75 \
            --contact-tip-threshold-m 0.035 \
            --min-fit-observed-points 12 \
            --max-unobserved-gap 5 \
            --w-pose 0.0025 \
            --w-temporal 0.08 \
            --w-rigid-temporal 0.03 \
            --w-acceleration 0.025 \
            --boundary-weight 0.20 \
            --max-orient-step-deg 35 \
            --max-translation-step-m 0.035 \
            --max-pose-step 1.8 \
            --no-image-rigid-alignment \
            --device "${EGO_DEVICE}" \
            --no-video \
            "${pair_limit[@]}"
    fi
}

run_render() {
    local mano_fit="${EGO_OUTPUT}/mano_fit_right_canonical_final"
    local stereo_frames="${EGO_OUTPUT}/mediapipe_stereo/stereo_frames.csv"
    local output="${EGO_OUTPUT}/mano_overlay_optimized"
    local marker="${output}/summary.json"
    require_file "${mano_fit}/summary.json"
    require_file "${stereo_frames}"
    if stage_output_ready "MANO overlay" "${output}" "${marker}"; then
        return
    fi
    require_file "${NORMALIZED_DATASET}/manifest.json"
    require_file "${RECTIFIED_DATASET}/manifest.json"
    local source_args=(
        --normalized-dataset "${NORMALIZED_DATASET}"
        --rectified-dataset "${RECTIFIED_DATASET}"
    )
    local pair_limit=()
    if ((EGO_MAX_PAIRS > 0)); then
        pair_limit=(--max-pairs "${EGO_MAX_PAIRS}")
    fi
    local video_args=()
    if [[ "${EGO_NO_VIDEO}" == "1" ]]; then
        video_args=(--no-video)
    fi
    log "rendering MANO overlay and exporting 21-DOF/6D CSVs"
    run_python "${PROJECT_DIR}/scripts/render_mano_overlay_angles.py" \
        "${source_args[@]}" \
        --mano-fit "${mano_fit}" \
        --mano-source "${EGO_MANO_SOURCE}" \
        --model-dir "${EGO_MANO_MODELS}" \
        --stereo-frames "${stereo_frames}" \
        --output "${output}" \
        "${pair_limit[@]}" \
        "${video_args[@]}"
}

run_wilor() {
    local output="${EGO_OUTPUT}/wilor_stereo"
    local marker="${output}/summary.json"
    if [[ -f "${marker}" ]]; then
        log "skip WiLoR stereo: already complete (${marker})"
        return
    fi
    if [[ -e "${output}" ]]; then
        local failed_backup="${output}.failed-$(date +%Y%m%d-%H%M%S)"
        warn "WiLoR output is incomplete; preserving it at ${failed_backup} before retry"
        mv -- "${output}" "${failed_backup}"
    fi
    require_file "${RECTIFIED_DATASET}/manifest.json"
    validate_wilor_assets
    mkdir -p "${output}"
    local cameras=()
    case "${EGO_WILOR_CAMERAS}" in
        left) cameras=(left) ;;
        right) cameras=(right) ;;
        both) cameras=(left right) ;;
        *) die "EGO_WILOR_CAMERAS must be left, right, or both" ;;
    esac
    local pids=() camera device camera_output optional_args
    for camera in "${cameras[@]}"; do
        if [[ "${camera}" == "left" ]]; then device="${EGO_WILOR_DEVICE_LEFT}"; else device="${EGO_WILOR_DEVICE_RIGHT}"; fi
        camera_output="${output}/${camera}"
        optional_args=()
        if ((EGO_MAX_PAIRS > 0)); then optional_args+=(--max-pairs "${EGO_MAX_PAIRS}"); fi
        if [[ "${EGO_NO_VIDEO}" == "1" ]]; then optional_args+=(--no-video); fi
        if [[ "${EGO_WILOR_FAST}" == "1" ]]; then optional_args+=(--fast); fi
        if [[ "${EGO_WILOR_SAVE_VERTICES}" == "1" ]]; then optional_args+=(--save-vertices); fi
        run_python "${PROJECT_DIR}/scripts/wilor_inference.py" \
            --rectified-dataset "${RECTIFIED_DATASET}" \
            --output "${camera_output}" \
            --camera "${camera}" \
            --checkpoint "${EGO_WILOR_CHECKPOINT}" \
            --model-config "${EGO_WILOR_CONFIG}" \
            --detector "${EGO_WILOR_DETECTOR}" \
            --mano-model-dir "${EGO_MANO_MODELS}" \
            --device "${device}" \
            --batch-size "${EGO_WILOR_BATCH_SIZE}" \
            --frame-stride "${EGO_WILOR_FRAME_STRIDE}" \
            "${optional_args[@]}" \
            >"${output}/${camera}.log" 2>&1 &
        pids+=("$!")
    done
    local status=0 pid
    for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
    if ((status != 0)); then
        die "WiLoR inference failed; inspect ${output}/*.log"
    fi
    if [[ "${EGO_NO_VIDEO}" != "1" ]]; then
        for camera in "${cameras[@]}"; do
            run_python "${PROJECT_DIR}/scripts/render_wilor_predictions.py" \
                --rectified-dataset "${RECTIFIED_DATASET}" \
                --predictions "${output}/${camera}/predictions.jsonl" \
                --camera "${camera}" \
                --output "${output}/${camera}/wilor_annotated.mp4"
        done
    fi
    "${PYTHON_RUN[@]}" - "${output}" "${EGO_WILOR_CAMERAS}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cameras = sys.argv[2].split()
if sys.argv[2] == 'both': cameras = ['left', 'right']
summary = {"schema_version": 1, "stage": "wilor_stereo", "cameras": cameras,
           "left": str(root / 'left' / 'summary.json') if 'left' in cameras else None,
           "right": str(root / 'right' / 'summary.json') if 'right' in cameras else None}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
PY
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (($# > 1)); then
    usage >&2
    exit 2
fi

EGO_SOURCE="${EGO_SOURCE:-}"
EGO_OUTPUT="${EGO_OUTPUT:-}"
EGO_STAGE="${1:-${EGO_STAGE:-all}}"
EGO_SESSION="${EGO_SESSION:-}"
EGO_MCAP="${EGO_MCAP:-}"
EGO_LEFT_CAMERA="${EGO_LEFT_CAMERA:-camera2}"
EGO_RIGHT_CAMERA="${EGO_RIGHT_CAMERA:-camera3}"
EGO_MODEL="${EGO_MODEL:-${PROJECT_DIR}/models/hand_landmarker.task}"
EGO_MANO_SOURCE="${EGO_MANO_SOURCE:-${PROJECT_DIR}/third_party/MANO}"
EGO_MANO_MODELS="${EGO_MANO_MODELS:-${PROJECT_DIR}/models/mano}"
EGO_DEVICE="${EGO_DEVICE:-auto}"
EGO_MAX_PAIRS="${EGO_MAX_PAIRS:-0}"
EGO_MAX_FRAMES="${EGO_MAX_FRAMES:-0}"
EGO_NO_VIDEO="${EGO_NO_VIDEO:-0}"
EGO_CONDA_ENV="${EGO_CONDA_ENV:-ego-hand}"
EGO_HAND_ROUTE="${EGO_HAND_ROUTE:-mediapipe}"
EGO_WILOR_CAMERAS="${EGO_WILOR_CAMERAS:-both}"
EGO_WILOR_DEVICE_LEFT="${EGO_WILOR_DEVICE_LEFT:-${EGO_DEVICE}}"
EGO_WILOR_DEVICE_RIGHT="${EGO_WILOR_DEVICE_RIGHT:-${EGO_DEVICE}}"
EGO_WILOR_FRAME_STRIDE="${EGO_WILOR_FRAME_STRIDE:-1}"
EGO_WILOR_BATCH_SIZE="${EGO_WILOR_BATCH_SIZE:-16}"
EGO_WILOR_FAST="${EGO_WILOR_FAST:-0}"
EGO_WILOR_SAVE_VERTICES="${EGO_WILOR_SAVE_VERTICES:-0}"
EGO_WILOR_CHECKPOINT="${EGO_WILOR_CHECKPOINT:-${PROJECT_DIR}/models/wilor/wilor_final.ckpt}"
EGO_WILOR_DETECTOR="${EGO_WILOR_DETECTOR:-${PROJECT_DIR}/models/wilor/detector.pt}"
EGO_WILOR_CONFIG="${EGO_WILOR_CONFIG:-${PROJECT_DIR}/models/wilor/model_config.yaml}"
EGO_WILOR_MANO_DIR="${EGO_WILOR_MANO_DIR:-${EGO_MANO_MODELS}}"

[[ "${EGO_SOURCE}" == "orbbec" || "${EGO_SOURCE}" == "gen" ]] || \
    die "set EGO_SOURCE to 'orbbec' or 'gen'"
[[ -n "${EGO_OUTPUT}" ]] || die "set EGO_OUTPUT to a new experiment directory"
[[ "${EGO_OUTPUT}" != "/" && "${EGO_OUTPUT}" != "." && "${EGO_OUTPUT}" != "${PROJECT_DIR}" ]] || \
    die "unsafe EGO_OUTPUT: ${EGO_OUTPUT}"
[[ "${EGO_DEVICE}" == "auto" || "${EGO_DEVICE}" == "cuda" || "${EGO_DEVICE}" == "cpu" ]] || \
    die "EGO_DEVICE must be auto, cuda, or cpu"
[[ "${EGO_NO_VIDEO}" == "0" || "${EGO_NO_VIDEO}" == "1" ]] || \
    die "EGO_NO_VIDEO must be 0 or 1"
[[ "${EGO_HAND_ROUTE}" == "mediapipe" || "${EGO_HAND_ROUTE}" == "wilor" || "${EGO_HAND_ROUTE}" == "parallel" ]] || \
    die "EGO_HAND_ROUTE must be mediapipe, wilor, or parallel"
[[ "${EGO_WILOR_FAST}" == "0" || "${EGO_WILOR_FAST}" == "1" ]] || die "EGO_WILOR_FAST must be 0 or 1"
[[ "${EGO_WILOR_SAVE_VERTICES}" == "0" || "${EGO_WILOR_SAVE_VERTICES}" == "1" ]] || die "EGO_WILOR_SAVE_VERTICES must be 0 or 1"
validate_non_negative_integer EGO_MAX_PAIRS "${EGO_MAX_PAIRS}"
validate_non_negative_integer EGO_MAX_FRAMES "${EGO_MAX_FRAMES}"
validate_non_negative_integer EGO_WILOR_FRAME_STRIDE "${EGO_WILOR_FRAME_STRIDE}"
[[ "${EGO_WILOR_FRAME_STRIDE}" != "0" ]] || die "EGO_WILOR_FRAME_STRIDE must be at least 1"
validate_non_negative_integer EGO_WILOR_BATCH_SIZE "${EGO_WILOR_BATCH_SIZE}"
[[ "${EGO_WILOR_BATCH_SIZE}" != "0" ]] || die "EGO_WILOR_BATCH_SIZE must be at least 1"

case "${EGO_STAGE}" in
    check|prepare|stereo|stabilize|fit|render|wilor|all) ;;
    *) die "unknown stage '${EGO_STAGE}'; use check, prepare, stereo, stabilize, fit, render, wilor, or all" ;;
esac

if [[ "${EGO_SOURCE}" == "orbbec" ]]; then
    [[ -n "${EGO_SESSION}" ]] || die "set EGO_SESSION for EGO_SOURCE=orbbec"
    EGO_SESSION="$(realpath -m "${EGO_SESSION}")"
    validate_orbbec_session
else
    [[ -n "${EGO_MCAP}" ]] || die "set EGO_MCAP for EGO_SOURCE=gen"
    [[ "${EGO_LEFT_CAMERA}" != "${EGO_RIGHT_CAMERA}" ]] || die "GEN left/right cameras must differ"
    EGO_MCAP="$(realpath -m "${EGO_MCAP}")"
    require_file "${EGO_MCAP}"
fi

EGO_OUTPUT="$(realpath -m "${EGO_OUTPUT}")"
[[ "${EGO_OUTPUT}" != "/" && "${EGO_OUTPUT}" != "${PROJECT_DIR}" ]] || \
    die "unsafe resolved EGO_OUTPUT: ${EGO_OUTPUT}"
NORMALIZED_DATASET="$(realpath -m "${EGO_NORMALIZED_DATASET:-${EGO_OUTPUT}/normalized}")"
RECTIFIED_DATASET="$(realpath -m "${EGO_RECTIFIED_DATASET:-${EGO_OUTPUT}/rectified}")"
EGO_MODEL="$(realpath -m "${EGO_MODEL}")"
EGO_MANO_SOURCE="$(realpath -m "${EGO_MANO_SOURCE}")"
EGO_MANO_MODELS="$(realpath -m "${EGO_MANO_MODELS}")"
select_python

case "${EGO_STAGE}" in
    check|all|stereo|wilor)
        if [[ "${EGO_HAND_ROUTE}" == "wilor" && "${EGO_STAGE}" != "stereo" ]]; then
            :
        else
        require_file "${EGO_MODEL}"
        fi
        ;;
esac
case "${EGO_STAGE}" in
    check|all|fit|render)
        if [[ "${EGO_HAND_ROUTE}" == "mediapipe" || "${EGO_HAND_ROUTE}" == "parallel" || "${EGO_STAGE}" != "check" && "${EGO_STAGE}" != "all" ]]; then
            validate_common_assets
        fi
        ;;
esac

print_config

case "${EGO_STAGE}" in
    check)
        if [[ "${EGO_HAND_ROUTE}" == "wilor" || "${EGO_HAND_ROUTE}" == "parallel" ]]; then
            validate_wilor_assets
        fi
        if [[ "${EGO_SOURCE}" == "gen" ]]; then
            log "checking GEN runtime dependencies and one decoded stereo frame"
            run_python "${PROJECT_DIR}/scripts/check_gen_environment.py" \
                --mcap "${EGO_MCAP}" \
                --camera "${EGO_LEFT_CAMERA}" \
                --other-camera "${EGO_RIGHT_CAMERA}" \
                --device "${EGO_DEVICE}"
        fi
        log "configuration and required inputs are ready"
        ;;
    prepare)
        run_prepare
        ;;
    stereo)
        run_stereo
        ;;
    stabilize)
        run_stabilize
        ;;
    fit)
        run_fit
        ;;
    render)
        run_render
        ;;
    wilor)
        run_prepare
        run_wilor
        ;;
    all)
        run_prepare
        if [[ "${EGO_HAND_ROUTE}" == "mediapipe" || "${EGO_HAND_ROUTE}" == "parallel" ]]; then
            run_stereo
            run_stabilize
            run_fit
            run_render
        fi
        if [[ "${EGO_HAND_ROUTE}" == "wilor" || "${EGO_HAND_ROUTE}" == "parallel" ]]; then
            run_wilor
        fi
        log "pipeline complete: ${EGO_OUTPUT}"
        ;;
esac
