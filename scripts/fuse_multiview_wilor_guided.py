#!/usr/bin/env python3
"""Fuse WiLoR with a reliable stereo anchor and independently matched side views."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from camera_models import project_points  # noqa: E402
from ego_data.calibration import CameraCalibration  # noqa: E402
from fuse_multiview_wilor import (  # noqa: E402
    _errors, _groups, _load_jsonl, _ray_in_base, _serialize_hand,
    _triangulate_hand, regularized_ray_intersection,
)
from fusion_anatomy_refinement import (  # noqa: E402
    FusionAnatomyConfig,
    refine_accepted_rows,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cameras", nargs="+",
        help="camera subset used for fusion (default: all cameras in the dataset)",
    )
    parser.add_argument("--anchor-cameras", nargs=2, default=("camera2", "camera3"))
    parser.add_argument(
        "--anchor-border-penalty", type=float, default=0.0,
        help=(
            "cost added for every anchor-hand box touching the image border; "
            "useful when edge-truncated hands create weak depth anchors"
        ),
    )
    parser.add_argument("--anchor-border-margin-px", type=float, default=3.0)
    parser.add_argument(
        "--tip-three-view-penalty", type=float, default=0.0,
        help=(
            "cost per missing inlier view below three for each reconstructed "
            "fingertip; zero preserves legacy selection"
        ),
    )
    parser.add_argument("--max-anchor-detections", type=int, default=3)
    parser.add_argument("--max-side-detections", type=int, default=4)
    parser.add_argument(
        "--detector-handedness", choices=("strict", "ignore", "adaptive"), default="strict",
        help=(
            "strict keeps detector.pt identity; ignore lets geometry choose; "
            "adaptive tries strict first and falls back to ignore for a rejected frame"
        ),
    )
    parser.add_argument(
        "--require-marker-evidence",
        choices=("auto", "0", "1"),
        default="auto",
        help=(
            "reject a physical-hand branch unless at least one selected view has "
            "topology-valid marker evidence; auto enables this for marker-assisted inputs"
        ),
    )
    parser.add_argument(
        "--independent-hand-fusion", type=int, choices=(0, 1), default=1,
        help=(
            "fuse each hand independently before identity assignment; this "
            "prevents a weak hand from hard-binding the other hand"
        ),
    )
    parser.add_argument(
        "--depth-regularization", type=int, choices=(0, 1), default=1,
        help=(
            "regularize weak two-view depth branches with a bidirectional "
            "temporal ray prior (default: 1)"
        ),
    )
    parser.add_argument("--depth-regularization-radius", type=int, default=8)
    parser.add_argument("--depth-regularization-trigger-m", type=float, default=0.055)
    parser.add_argument("--depth-regularization-max-2d-motion-px", type=float, default=45.0)
    parser.add_argument("--depth-regularization-blend", type=float, default=0.85)
    parser.add_argument(
        "--temporal-depth-guard", type=int, choices=(0, 1), default=1,
        help=(
            "repair short, isolated 3-D depth branches when shared-camera "
            "2-D tracks remain continuous (default: 1)"
        ),
    )
    parser.add_argument("--temporal-depth-guard-radius", type=int, default=8)
    parser.add_argument("--temporal-depth-guard-trigger-m", type=float, default=0.055)
    parser.add_argument(
        "--temporal-depth-guard-max-2d-motion-px", type=float, default=75.0,
    )
    parser.add_argument(
        "--temporal-depth-guard-max-step-m", type=float, default=0.115,
    )
    parser.add_argument(
        "--temporal-depth-guard-max-bracket-speed-m", type=float, default=0.120,
    )
    parser.add_argument(
        "--temporal-depth-guard-max-shift-m", type=float, default=0.160,
    )
    parser.add_argument(
        "--temporal-depth-guard-max-run-length", type=int, default=3,
    )
    parser.add_argument(
        "--temporal-depth-guard-max-reprojection-regression-px", type=float,
        default=12.0,
    )
    parser.add_argument(
        "--temporal-depth-guard-max-reprojection-p95-regression-px", type=float,
        default=24.0,
    )
    parser.add_argument(
        "--temporal-depth-guard-min-shared-cameras", type=int, default=2,
    )
    parser.add_argument(
        "--depth-preferred-ray-angle-deg", type=float, default=8.0,
        help=(
            "preferred palm ray angle; weaker angles remain legal but receive "
            "a strong depth-branch penalty"
        ),
    )
    parser.add_argument(
        "--depth-min-palm-angle-deg", type=float, default=4.5,
        help=(
            "hard lower bound for a two-view palm ray angle; branches below "
            "this are not safe depth observations"
        ),
    )
    parser.add_argument(
        "--depth-min-palm-three-view", type=int, default=2,
        help=(
            "minimum palm joints with at least three inlier views before a "
            "multiview branch is considered well conditioned"
        ),
    )
    parser.add_argument(
        "--depth-max-pair-spread-m", type=float, default=0.075,
        help="pair-intersection spread above which a depth branch is strongly penalized",
    )
    parser.add_argument(
        "--depth-unsafe-pair-spread-m", type=float, default=0.090,
        help=(
            "pair-intersection spread above which a branch is unsafe for a "
            "complete hand/temporal alternate (default: 0.09 m)"
        ),
    )
    parser.add_argument(
        "--depth-condition-weight", type=float, default=1.0,
        help="multiplier for the depth-conditioning cost terms",
    )
    parser.add_argument(
        "--legacy-continuity-passes", type=int, choices=(0, 1), default=0,
        help=(
            "run the older multiple continuity/Viterbi passes; disabled by "
            "default so branch and identity decisions stay in one chain"
        ),
    )
    parser.add_argument("--association-threshold-px", type=float, default=55.0)
    parser.add_argument("--anchor-threshold-px", type=float, default=60.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=20.0)
    parser.add_argument(
        "--duplicate-bbox-iou", type=float, default=0.65,
        help="IoU above which overlapping detector boxes are considered duplicates",
    )
    parser.add_argument(
        "--duplicate-joint-distance-px", type=float, default=18.0,
        help="median 2-D joint distance used to identify duplicate hypotheses",
    )
    parser.add_argument(
        "--duplicate-overlap-cameras", type=int, default=2,
        help="shared camera views required to reject a duplicate two-hand assignment",
    )
    parser.add_argument(
        "--duplicate-center-distance-m", type=float, default=0.055,
        help=(
            "3-D palm-centre distance below which two independent branches "
            "must be proved distinct in at least two shared cameras"
        ),
    )
    parser.add_argument(
        "--duplicate-center-min-distinct-cameras", type=int, default=2,
        help=(
            "minimum shared cameras with distinct detector observations "
            "needed for two near-coincident 3-D hands"
        ),
    )
    parser.add_argument(
        "--min-full-hand-support", type=float, default=2.5,
        help=(
            "minimum median inlier-view support for each side when a complete "
            "two-hand result is accepted"
        ),
    )
    parser.add_argument(
        "--full-hand-support-gap", type=float, default=1.5,
        help=(
            "support gap that marks one side of an otherwise complete result "
            "as independently weak"
        ),
    )
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--max-reprojection-median-px", type=float, default=15.0)
    parser.add_argument("--max-reprojection-p95-px", type=float, default=40.0)
    parser.add_argument("--temporal-recovery-gap", type=int, default=3)
    parser.add_argument("--temporal-association-threshold-px", type=float, default=90.0)
    parser.add_argument("--max-temporal-wrist-step-m", type=float, default=0.12)
    parser.add_argument(
        "--identity-partial-switch-penalty", type=float, default=1.50,
        help=(
            "penalty for changing the physical side of a plausible one-hand "
            "partial observation"
        ),
    )
    parser.add_argument(
        "--identity-partial-switch-max-cost", type=float, default=3.0,
        help=(
            "only penalize a partial-side switch when the observation is "
            "compatible with the same 3-D track"
        ),
    )
    parser.add_argument(
        "--identity-2d-soft-motion-px", type=float, default=80.0,
        help=(
            "shared-camera palm motion per frame considered a normal identity "
            "link; 2-D evidence is independent of reconstructed depth"
        ),
    )
    parser.add_argument(
        "--identity-2d-hard-motion-px", type=float, default=125.0,
        help=(
            "shared-camera palm motion per frame that breaks a partial identity "
            "link when another track explains the observation"
        ),
    )
    parser.add_argument(
        "--identity-2d-max-gap", type=int, default=8,
        help="maximum missing-frame gap for 2-D identity reconnects",
    )
    parser.add_argument(
        "--identity-2d-hard-break-penalty", type=float, default=24.0,
        help=(
            "cost for reconnecting a partial state across a hard 2-D track "
            "break; explicit missing/re-entry is preferred"
        ),
    )
    parser.add_argument(
        "--identity-2d-anchor-radius", type=int, default=24,
        help="bracketing complete-row radius used to repair partial side labels",
    )
    parser.add_argument(
        "--identity-fill-max-gap", type=int, default=3,
        help="maximum short detector dropout (missing frames) to interpolate per hand",
    )
    parser.add_argument(
        "--identity-fill-max-2d-motion-px", type=float, default=80.0,
        help="maximum bracketed image motion per frame for short-gap hand filling",
    )
    parser.add_argument(
        "--partial-continuity-radius", type=int, default=6,
        help="temporal neighbourhood used to re-score one-hand partial rows",
    )
    parser.add_argument(
        "--partial-continuity-min-improvement", type=float, default=5.0,
        help="minimum combined cost improvement required to replace a partial branch",
    )
    parser.add_argument(
        "--partial-continuity-transition-weight", type=float, default=24.0,
        help="weight for one-hand 3-D transition continuity during branch reselection",
    )
    parser.add_argument("--anatomy-refinement", type=int, choices=(0, 1), default=1)
    parser.add_argument("--anatomy-outlier-window", type=int, default=4)
    parser.add_argument("--anatomy-outlier-distance-m", type=float, default=0.10)
    parser.add_argument("--anatomy-bone-relative", type=float, default=0.45)
    parser.add_argument("--anatomy-smoothing-radius", type=int, default=2)
    parser.add_argument("--anatomy-shape-strength", type=float, default=0.55)
    parser.add_argument("--anatomy-reliable-adjustment-blend", type=float, default=0.35)
    parser.add_argument("--anatomy-max-reliable-adjustment-m", type=float, default=0.020)
    parser.add_argument("--anatomy-max-reprojection-regression-px", type=float, default=15.0)
    parser.add_argument("--anatomy-max-reprojection-shift-px", type=float, default=35.0)
    parser.add_argument("--anatomy-temporal-radius", type=int, default=4)
    parser.add_argument("--anatomy-temporal-palm-residual-m", type=float, default=0.070)
    parser.add_argument("--anatomy-temporal-local-residual", type=float, default=0.90)
    parser.add_argument(
        "--gesture-smoothing", type=int, choices=(0, 1), default=1,
        help="apply conservative zero-phase smoothing to small joint jitter",
    )
    parser.add_argument("--gesture-smoothing-radius", type=int, default=2)
    parser.add_argument("--gesture-smoothing-blend", type=float, default=0.30)
    parser.add_argument("--gesture-smoothing-max-motion-m", type=float, default=0.080)
    parser.add_argument("--gesture-smoothing-max-residual-m", type=float, default=0.045)
    parser.add_argument("--gesture-smoothing-max-adjustment-m", type=float, default=0.022)
    parser.add_argument("--tracking-radius", type=int, default=6)
    parser.add_argument("--tracking-min-neighbors-per-side", type=int, default=2)
    parser.add_argument("--tracking-min-good-joints", type=int, default=18)
    parser.add_argument("--tracking-min-reference-support", type=float, default=3.0)
    parser.add_argument("--tracking-max-reference-corrections", type=int, default=8)
    parser.add_argument(
        "--tracking-apply-support", type=float, default=3.0,
        help=(
            "median inlier-view support below which confidence tracking may "
            "repair a hand; three views are treated as normal for cameras 1-4"
        ),
    )
    parser.add_argument("--tracking-max-reprojection-median-px", type=float, default=12.0)
    parser.add_argument("--tracking-max-reprojection-p95-px", type=float, default=90.0)
    parser.add_argument("--tracking-palm-residual-m", type=float, default=0.060)
    parser.add_argument("--tracking-shape-residual", type=float, default=0.75)
    parser.add_argument("--tracking-blend", type=float, default=0.75)
    parser.add_argument("--tracking-max-adjustment-m", type=float, default=0.10)
    parser.add_argument(
        "--tracking-low-redundancy-max-adjustment-m", type=float, default=0.035,
        help="maximum fallback correction for a two/partial-view hand",
    )
    parser.add_argument(
        "--tracking-allow-low-redundancy-rigid", type=int, choices=(0, 1), default=0,
        help="allow a temporal tracker to translate a low-redundancy hand",
    )
    parser.add_argument(
        "--continuity-reselection", type=int, choices=(0, 1), default=1,
        help=(
            "re-score accepted geometric hypotheses with an offline trajectory "
            "continuity cost (default: 1)"
        ),
    )
    parser.add_argument(
        "--continuity-weight", type=float, default=20000.0,
        help="weight for squared excess palm displacement in hypothesis reselection",
    )
    parser.add_argument(
        "--continuity-max-step-m", type=float, default=0.11,
        help="per-frame palm displacement before continuity penalty applies",
    )
    parser.add_argument(
        "--continuity-wrist-max-step-m", type=float, default=0.14,
        help="per-frame wrist displacement before continuity penalty applies",
    )
    parser.add_argument(
        "--continuity-switch-penalty", type=float, default=8.0,
        help="small penalty for changing the selected anchor camera pair",
    )
    parser.add_argument(
        "--continuity-min-support", type=float, default=2.5,
        help="minimum median inlier-view support for continuity reselection",
    )
    parser.add_argument(
        "--continuity-min-palm-support", type=float, default=2.0,
        help=(
            "minimum median inlier-view support on palm anchors for an "
            "alternate continuity branch"
        ),
    )
    parser.add_argument(
        "--continuity-min-palm-points", type=int, default=3,
        help=(
            "minimum palm anchors with finite/reliably supported geometry "
            "for an alternate continuity branch"
        ),
    )
    parser.add_argument(
        "--continuity-min-branch-valid-joints", type=int, default=8,
        help=(
            "minimum valid joints per hand for a continuity alternate; "
            "finger joints may be sparse when the palm is reliable"
        ),
    )
    parser.add_argument(
        "--continuity-max-cost-delta", type=float, default=300.0,
        help=(
            "maximum per-frame geometric cost above the best hypothesis that "
            "continuity reselection may consider"
        ),
    )
    parser.add_argument(
        "--continuity-branch-cost-delta", type=float, default=1400.0,
        help=(
            "larger geometric-cost allowance for a physically valid alternate "
            "depth branch whose bidirectional bracket improves materially"
        ),
    )
    parser.add_argument(
        "--continuity-branch-min-improvement-m", type=float, default=0.025,
        help=(
            "minimum bidirectional bracket improvement required before the "
            "larger alternate-branch cost allowance is used"
        ),
    )
    parser.add_argument(
        "--continuity-max-hypotheses", type=int, default=32,
        help="maximum alternate frame hypotheses enumerated for a suspicious frame",
    )
    parser.add_argument(
        "--continuity-branch-candidates-per-observation", type=int, default=4,
        help=(
            "number of distinct 3-D branches retained for each detector "
            "observation set during continuity enumeration"
        ),
    )
    parser.add_argument(
        "--continuity-max-anchor-pairs", type=int, default=8,
        help="maximum anchor camera pairs enumerated for a suspicious frame",
    )
    parser.add_argument(
        "--continuity-target-step-m", type=float, default=0.085,
        help=(
            "flag frames whose fixed-label palm step exceeds this value so "
            "alternate depth branches can be compared"
        ),
    )
    parser.add_argument(
        "--continuity-bracket-weight", type=float, default=150000.0,
        help="weight for a candidate's bidirectional bracket residual",
    )
    parser.add_argument(
        "--continuity-bracket-residual-m", type=float, default=0.030,
        help="residual below which a bidirectional bracket is considered plausible",
    )
    parser.add_argument(
        "--continuity-target-bracket-m", type=float, default=0.075,
        help="bidirectional residual that marks a frame for branch enumeration",
    )
    parser.add_argument(
        "--track-continuity", type=int, choices=(0, 1), default=1,
        help=(
            "jointly select depth branches and physical hand tracks across "
            "full/partial observation boundaries (default: 1)"
        ),
    )
    parser.add_argument(
        "--track-continuity-boundary-radius", type=int, default=2,
        help="number of neighbours around a full/partial boundary to inspect",
    )
    parser.add_argument(
        "--track-continuity-max-targets", type=int, default=96,
        help=(
            "maximum frames sent to expensive branch enumeration in the unified "
            "continuity pass"
        ),
    )
    parser.add_argument(
        "--track-continuity-min-improvement", type=float, default=4.0,
        help="minimum cost improvement before replacing a delivered track state",
    )
    parser.add_argument(
        "--track-reconnect-max-gap", type=int, default=8,
        help=(
            "maximum missing-frame gap for hard 2-D identity/depth reconnect "
            "checks (default: 8)"
        ),
    )
    parser.add_argument(
        "--track-reconnect-max-2d-motion-px", type=float, default=55.0,
        help=(
            "per-frame shared-camera palm motion below which an observation "
            "is treated as the same image-space hand track"
        ),
    )
    parser.add_argument(
        "--track-reconnect-max-3d-step-m", type=float, default=0.095,
        help=(
            "per-frame 3-D palm step allowed when image-space motion remains "
            "continuous; larger motion is a depth-branch contradiction"
        ),
    )
    parser.add_argument(
        "--track-reconnect-depth-penalty", type=float, default=2500.0,
        help="cost assigned to a 2-D-continuous but 3-D-discontinuous link",
    )
    parser.add_argument(
        "--track-reselection-min-depth-improvement-m", type=float, default=0.025,
        help=(
            "minimum local 3-D trajectory improvement required before an "
            "alternate branch may replace the delivered observation"
        ),
    )
    parser.add_argument(
        "--final-sequence-guard", type=int, choices=(0, 1), default=1,
        help=(
            "run a final per-hand 2-D/3-D trajectory guard after anatomy "
            "refinement (default: 1)"
        ),
    )
    parser.add_argument(
        "--final-sequence-max-step-m", type=float, default=0.115,
        help="3-D palm step above which a continuous 2-D link is suspicious",
    )
    parser.add_argument(
        "--final-sequence-max-2d-motion-px", type=float, default=75.0,
        help="shared-camera 2-D palm motion allowing a depth-continuous link",
    )
    parser.add_argument(
        "--final-sequence-max-gap", type=int, default=6,
        help="maximum gap inspected by the final trajectory guard",
    )
    parser.add_argument(
        "--final-sequence-min-improvement-m", type=float, default=0.025,
        help=(
            "minimum bidirectional 3-D improvement required before the final "
            "guard replaces a depth branch"
        ),
    )
    parser.add_argument(
        "--final-sequence-delete-step-m", type=float, default=0.24,
        help=(
            "severe continuous 2-D/3-D step at which an unsafe hand may be "
            "removed when no safe branch exists"
        ),
    )
    parser.add_argument(
        "--final-sequence-max-replacement-shift-m", type=float, default=0.080,
        help=(
            "maximum palm displacement allowed for a final depth-branch "
            "replacement; larger excursions are left to track selection"
        ),
    )
    parser.add_argument(
        "--final-sequence-max-targets", type=int, default=48,
        help=(
            "maximum suspicious hand frames sent to final depth-branch "
            "validation (default: 48)"
        ),
    )
    parser.add_argument(
        "--anchor-baseline-reference-m", type=float, default=0.10,
        help="baseline below which an anchor-pair conditioning penalty is applied",
    )
    parser.add_argument(
        "--anchor-baseline-penalty", type=float, default=1400.0,
        help="linear cost per metre of anchor-baseline shortfall",
    )
    parser.add_argument(
        "--secondary-hand-min-valid-joints", type=int, default=10,
        help="minimum joints for a weak secondary hand to remain pairable",
    )
    parser.add_argument(
        "--secondary-hand-min-palm-points", type=int, default=3,
        help="minimum reliable palm joints for a weak secondary hand",
    )
    parser.add_argument(
        "--secondary-hand-min-palm-angle-deg", type=float, default=7.0,
        help=(
            "minimum palm ray angle for a weak secondary hand; below this, "
            "a branch may be retained only when both hands are weak"
        ),
    )
    parser.add_argument(
        "--secondary-hand-max-cost-gap", type=float, default=420.0,
        help=(
            "maximum quality-cost gap tolerated when pairing a weak hand with "
            "a strong hand"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="independent frame-fusion worker processes",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _complete_detections(
    groups: dict[int, dict[int, dict[str, Any]]], limit: int,
) -> list[int]:
    detections = [index for index, sides in groups.items() if 0 in sides and 1 in sides]
    detections.sort(
        key=lambda index: max(
            (_marker_evidence_rank(hand) for hand in groups[index].values()),
            default=(0, 0, 0.0),
        ),
        reverse=True,
    )
    return detections[:limit]


def _marker_evidence_rank(hand: Mapping[str, Any]) -> tuple[int, int, float]:
    marker = hand.get("marker_assist", {})
    level = 2 if marker.get("applied") else 1 if marker.get("evidence_only") else 0
    return (
        level,
        int(marker.get("matched_marker_count", 0) or 0),
        float(hand.get("confidence", 0.0)),
    )


def _selection_has_marker_evidence(selected: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(_marker_evidence_rank(hand)[0] > 0 for hand in selected.values())


def _detector_side(hand: dict[str, Any]) -> int | None:
    value = hand.get("detector_is_right")
    return None if value is None else int(value)


def _bbox_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return IoU for two detector boxes, or zero when either is missing."""
    a = np.asarray(first.get("bbox_xyxy", []), dtype=np.float64)
    b = np.asarray(second.get("bbox_xyxy", []), dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not (
        np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    ):
        return 0.0
    aw = max(0.0, float(a[2] - a[0]))
    ah = max(0.0, float(a[3] - a[1]))
    bw = max(0.0, float(b[2] - b[0]))
    bh = max(0.0, float(b[3] - b[1]))
    area_a = aw * ah
    area_b = bw * bh
    if area_a <= 1e-9 or area_b <= 1e-9:
        return 0.0
    intersection = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 1e-9 else 0.0


def _same_detector_observation(
    first: dict[str, Any],
    second: dict[str, Any],
    bbox_iou_threshold: float = 0.65,
    joint_distance_threshold_px: float = 18.0,
) -> bool:
    """Detect two hypotheses that are actually one detector observation.

    WiLoR stores both handedness hypotheses for each detector box.  During
    motion blur/NMS instability it can also emit several *different* indices
    for the same image blob.  Rejecting only an equal index is therefore not
    sufficient: those duplicate boxes can triangulate into two plausible but
    physically unrelated depths.  Identical 2-D joints are decisive; a high
    box overlap plus a small median joint distance catches the duplicated-index
    case while allowing genuinely adjacent hands whose wrists/fingers are
    visibly separated.
    """
    first_index = first.get("detection_index")
    second_index = second.get("detection_index")
    if first_index is not None and second_index is not None:
        try:
            if int(first_index) == int(second_index):
                return True
        except (TypeError, ValueError):
            pass

    first_points = np.asarray(first.get("joints_2d", []), dtype=np.float64)
    second_points = np.asarray(second.get("joints_2d", []), dtype=np.float64)
    joint_distance = float("inf")
    wrist_distance = float("inf")
    if first_points.shape == (21, 2) and second_points.shape == (21, 2):
        finite = np.isfinite(first_points).all(axis=1) & np.isfinite(second_points).all(axis=1)
        if np.count_nonzero(finite) >= 5:
            joint_distance = float(np.median(
                np.linalg.norm(first_points[finite] - second_points[finite], axis=1)
            ))
            if np.all(np.isfinite(first_points[0])) and np.all(np.isfinite(second_points[0])):
                wrist_distance = float(np.linalg.norm(first_points[0] - second_points[0]))
            if joint_distance <= max(float(joint_distance_threshold_px), 0.0):
                return True

    overlap = _bbox_iou(first, second)
    if overlap < max(float(bbox_iou_threshold), 0.0):
        return False
    if overlap >= 0.75 and np.isfinite(wrist_distance):
        first_box = np.asarray(first.get("bbox_xyxy", []), dtype=np.float64)
        second_box = np.asarray(second.get("bbox_xyxy", []), dtype=np.float64)
        if first_box.shape == (4,) and second_box.shape == (4,):
            first_span = min(
                max(0.0, float(first_box[2] - first_box[0])),
                max(0.0, float(first_box[3] - first_box[1])),
            )
            second_span = min(
                max(0.0, float(second_box[2] - second_box[0])),
                max(0.0, float(second_box[3] - second_box[1])),
            )
            # Duplicate boxes normally have almost coincident wrists even
            # when their finger hypotheses differ under motion blur.
            if wrist_distance <= max(40.0, 0.25 * min(first_span, second_span)):
                return True
    # If no reliable joints are present, a near-identical high-overlap box is
    # still almost certainly a duplicate.  With reliable joints require a
    # looser distance than the decisive exact-hypothesis test above.
    if not np.isfinite(joint_distance):
        return True
    return joint_distance <= 2.5 * max(float(joint_distance_threshold_px), 1.0)


def _duplicate_hand_observation_count(
    selected_by_side: dict[int, dict[str, dict[str, Any]]],
    bbox_iou_threshold: float = 0.65,
    joint_distance_threshold_px: float = 18.0,
) -> int:
    """Count shared camera views where two selected hands are one blob."""
    first_views = selected_by_side.get(0, {})
    second_views = selected_by_side.get(1, {})
    overlap = 0
    for camera in set(first_views) & set(second_views):
        if _same_detector_observation(
            first_views[camera], second_views[camera],
            bbox_iou_threshold, joint_distance_threshold_px,
        ):
            overlap += 1
    return overlap


def _duplicate_assignment_is_unsafe(
    selected_by_side: dict[int, dict[str, dict[str, Any]]],
    anchor_cameras: tuple[str, str] | list[str] | None = None,
    bbox_iou_threshold: float = 0.65,
    joint_distance_threshold_px: float = 18.0,
    minimum_duplicate_views: int = 2,
) -> tuple[bool, int]:
    """Decide whether a two-hand assignment is likely one image blob.

    Two genuinely different hands may overlap in one view.  Conversely, when
    the assignment is supported by only two or three views, a single duplicate
    view is already enough to make depth/identity ambiguous.  Use a stricter
    rule for sparse stereo and retain the configurable two-view rule for well
    supported multiview assignments.
    """
    duplicate_views = _duplicate_hand_observation_count(
        selected_by_side, bbox_iou_threshold, joint_distance_threshold_px
    )
    common_count = len(
        set(selected_by_side.get(0, {})) & set(selected_by_side.get(1, {}))
    )
    sparse = min(
        len(selected_by_side.get(0, {})), len(selected_by_side.get(1, {}))
    ) <= 3
    unsafe = duplicate_views >= max(int(minimum_duplicate_views), 1)
    if duplicate_views and sparse and common_count <= 3:
        unsafe = True
    # An overlap in an anchor view is especially risky when no additional
    # camera can independently disambiguate the two hypotheses.
    if anchor_cameras:
        anchor_set = set(anchor_cameras)
        anchor_overlap = len(
            anchor_set
            & (
                set(selected_by_side.get(0, {}))
                & set(selected_by_side.get(1, {}))
            )
        )
        duplicate_anchor = sum(
            int(
                camera in selected_by_side.get(0, {})
                and camera in selected_by_side.get(1, {})
                and _same_detector_observation(
                    selected_by_side[0][camera], selected_by_side[1][camera],
                    bbox_iou_threshold, joint_distance_threshold_px,
                )
            )
            for camera in anchor_set
        )
        if duplicate_anchor and anchor_overlap <= 1:
            unsafe = True
    return bool(unsafe), int(duplicate_views)


def _near_coincident_independent_pair_is_unsafe(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Reject two independent branches that are really one physical hand.

    The image duplicate check above only compares cameras used by *both*
    branches.  That leaves a subtle hole when branch A is triangulated from
    cameras 2/4 and branch B from cameras 1/3: there is no shared view, so the
    two hypotheses can pass the duplicate test even though their reconstructed
    palm centres differ by only a few centimetres.  Such a pair is normally a
    detector-side/depth permutation, not two hands.

    A near-coincident pair is allowed only when enough shared cameras contain
    genuinely different detector observations.  Falling back to one hand is
    safer than emitting two copies of the same hand; a real pair that is
    touching can still pass when two cameras independently separate the boxes.
    """
    first_result = first.get("result", {})
    second_result = second.get("result", {})
    try:
        first_center = _hypothesis_center(first_result)
    except (KeyError, TypeError, ValueError):
        first_center = np.full(3, np.nan)
    try:
        second_center = _hypothesis_center(second_result)
    except (KeyError, TypeError, ValueError):
        second_center = np.full(3, np.nan)
    if not (
        np.all(np.isfinite(first_center))
        and np.all(np.isfinite(second_center))
    ):
        return False, {
            "center_distance_m": None,
            "shared_camera_count": 0,
            "distinct_camera_count": 0,
            "required_distinct_camera_count": int(max(
                getattr(args, "duplicate_center_min_distinct_cameras", 2), 1
            )),
        }
    distance = float(np.linalg.norm(first_center - second_center))
    threshold = max(float(getattr(
        args, "duplicate_center_distance_m", 0.055
    )), 0.0)
    first_selected = first.get("selected", {})
    second_selected = second.get("selected", {})
    shared = sorted(set(first_selected) & set(second_selected))
    # Synthetic/unit-test candidates may omit image views entirely.  In that
    # case this helper has no evidence on which to impose a physical-space
    # duplicate decision, so leave the pair to the ordinary gates.  Real
    # candidates, however, always carry at least one selected camera.  If two
    # such candidates have *no shared camera* and their reconstructed centres
    # are nearly coincident, that is precisely the cross-camera duplicate hole
    # this guard is meant to close: there is no image view capable of proving
    # that they are two different detector observations.
    if distance >= threshold or not first_selected or not second_selected:
        return False, {
            "center_distance_m": distance,
            "shared_camera_count": len(shared),
            "distinct_camera_count": 0,
            "required_distinct_camera_count": int(max(
                getattr(args, "duplicate_center_min_distinct_cameras", 2), 1
            )),
        }
    distinct = 0
    for camera in shared:
        first_view = first_selected.get(camera, {})
        second_view = second_selected.get(camera, {})
        # ``_same_detector_observation`` is deliberately conservative: an
        # equal detection index or nearly identical 2-D pose is not proof of
        # two hands, while clearly separated boxes/joints are.
        if not _same_detector_observation(
            first_view,
            second_view,
            float(getattr(args, "duplicate_bbox_iou", 0.65)),
            float(getattr(args, "duplicate_joint_distance_px", 18.0)),
        ):
            distinct += 1
    required = int(max(getattr(
        args, "duplicate_center_min_distinct_cameras", 2
    ), 1))
    unsafe = distinct < required
    return unsafe, {
        "center_distance_m": distance,
        "shared_camera_count": len(shared),
        "distinct_camera_count": distinct,
        "required_distinct_camera_count": required,
    }


def _ordered_hand_pairs(
    detections: list[int], groups: dict[int, dict[int, dict[str, Any]]],
    handedness_mode: str,
) -> list[tuple[int, int]]:
    pairs = list(itertools.permutations(detections, 2))
    if handedness_mode == "strict":
        pairs = [
            pair for pair in pairs
            if _detector_side(groups[pair[0]][0]) in (None, 0)
            and _detector_side(groups[pair[1]][1]) in (None, 1)
        ]
    return pairs


def _project_base_points(
    camera: CameraCalibration, points_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = camera.T_base_camera[:3, :3]
    center = camera.T_base_camera[:3, 3]
    points_camera = (rotation.T @ (points_base - center).T).T
    return project_points(camera, points_camera)


def _candidate_error(
    anchor_points: np.ndarray, camera: CameraCalibration, hand: dict[str, Any],
    minimum_valid_joints: int = 12,
) -> float:
    projected, valid = _project_base_points(camera, anchor_points)
    observed = np.asarray(hand["joints_2d"], dtype=np.float64)
    valid &= np.isfinite(anchor_points).all(axis=1) & np.isfinite(observed).all(axis=1)
    if np.count_nonzero(valid) < max(int(minimum_valid_joints), 1):
        return float("inf")
    errors = np.linalg.norm(projected[valid] - observed[valid], axis=1)
    return float(np.median(errors))


def _match_camera(
    camera_id: str,
    groups: dict[int, dict[int, dict[str, Any]]],
    anchors: dict[int, dict[str, Any]],
    calibration: CameraCalibration,
    threshold_px: float,
    max_detections: int,
    handedness_mode: str,
    minimum_valid_joints: int = 12,
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    detections = _complete_detections(groups, max_detections)
    candidates: dict[int, list[tuple[int | None, float, dict[str, Any] | None]]] = {}
    for side in (0, 1):
        values: list[tuple[int | None, float, dict[str, Any] | None]] = [
            (None, threshold_px, None)
        ]
        for detection in detections:
            hand = groups[detection][side]
            if handedness_mode == "strict" and _detector_side(hand) not in (None, side):
                continue
            error = _candidate_error(
                anchors[side]["points"], calibration, hand, minimum_valid_joints
            )
            if error <= threshold_px:
                values.append((detection, error, hand))
        candidates[side] = values
    options = []
    for left, right in itertools.product(candidates[0], candidates[1]):
        if left[0] is not None and left[0] == right[0]:
            continue
        matched = int(left[0] is not None) + int(right[0] is not None)
        options.append((left[1] + right[1] - 0.25 * threshold_px * matched, left, right))
    if not options:
        return {}, {}
    _, left, right = min(options, key=lambda item: item[0])
    selected = {}
    errors = {}
    for side, value in ((0, left), (1, right)):
        if value[0] is not None and value[2] is not None:
            selected[side] = value[2]
            errors[side] = float(value[1])
    return selected, errors


def _single_side_options(
    groups: dict[int, dict[int, dict[str, Any]]],
    side: int,
    limit: int,
    handedness_mode: str,
) -> list[tuple[int, dict[str, Any]]]:
    """Return detection candidates for one physical hand side.

    A WiLoR row can contain two handedness hypotheses for one detector box.
    The dual-hand path needs complete detection groups, but a partial frame
    must be able to use that single box as a stereo observation.  Keep at most
    one hypothesis per detector index and prefer the detector's native side in
    strict mode.
    """
    values: list[tuple[int, dict[str, Any]]] = []
    for detection, sides in groups.items():
        hand = sides.get(side)
        if hand is None:
            continue
        detector_side = _detector_side(hand)
        if handedness_mode == "strict" and detector_side not in (None, side):
            continue
        values.append((int(detection), hand))
    values.sort(key=lambda item: _marker_evidence_rank(item[1]), reverse=True)
    return values[:limit]


def _match_camera_single_side(
    groups: dict[int, dict[int, dict[str, Any]]],
    side: int,
    anchor_points: np.ndarray,
    calibration: CameraCalibration,
    threshold_px: float,
    max_detections: int,
    handedness_mode: str,
    minimum_valid_joints: int = 12,
) -> tuple[dict[str, Any] | None, float | None]:
    """Match one physical side in a camera against a triangulated anchor."""
    options: list[tuple[float, dict[str, Any]]] = []
    for _index, hand in _single_side_options(
        groups, side, max_detections, handedness_mode
    ):
        error = _candidate_error(
            anchor_points, calibration, hand, minimum_valid_joints
        )
        if error <= threshold_px:
            options.append((error, hand))
    if not options:
        return None, None
    error, hand = min(options, key=lambda item: item[0])
    return hand, float(error)


def _anatomy_penalty(points: np.ndarray) -> float:
    lengths = []
    for start, end in HAND_EDGES:
        if np.all(np.isfinite(points[[start, end]])):
            lengths.append(float(np.linalg.norm(points[start] - points[end])))
    if len(lengths) < 12:
        return 500.0
    values = np.asarray(lengths)
    return float(
        1000.0 * np.maximum(0.006 - values, 0.0).sum()
        + 1000.0 * np.maximum(values - 0.09, 0.0).sum()
    )


def _anchor_baseline_m(
    anchor_cameras: tuple[str, str] | list[str],
    calibrations: dict[str, CameraCalibration],
) -> float:
    """Return the physical baseline of an anchor pair in metres."""
    if len(anchor_cameras) != 2 or any(
        camera not in calibrations for camera in anchor_cameras
    ):
        return 0.0
    first = np.asarray(
        calibrations[anchor_cameras[0]].T_base_camera[:3, 3], dtype=np.float64
    )
    second = np.asarray(
        calibrations[anchor_cameras[1]].T_base_camera[:3, 3], dtype=np.float64
    )
    value = float(np.linalg.norm(first - second))
    return value if np.isfinite(value) else 0.0


def _anchor_condition_penalty(
    anchor_cameras: tuple[str, str] | list[str],
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> float:
    """Penalize poorly conditioned short-baseline depth hypotheses.

    Reprojection error alone cannot distinguish two nearly parallel stereo
    solutions.  A small baseline penalty makes the wider, better-conditioned
    pair win when image-space costs are close, while still allowing a short
    pair when the wider cameras do not see the hand.
    """
    reference = max(float(getattr(args, "anchor_baseline_reference_m", 0.10)), 0.0)
    weight = max(float(getattr(args, "anchor_baseline_penalty", 1400.0)), 0.0)
    return weight * max(0.0, reference - _anchor_baseline_m(anchor_cameras, calibrations))


def _result_geometry_cost(result: dict[str, Any]) -> float:
    """Comparable image/geometry cost for one fused hand."""
    valid = int(result.get("valid_joints", 0))
    missing = max(0, 21 - valid)
    median = float(result.get("median_px", np.inf))
    p95 = float(result.get("p95_px", np.inf))
    all_median = float(result.get("all_view_median_px", median))
    all_p95 = float(result.get("all_view_p95_px", p95))
    geometry_rejected = int(result.get("geometry_rejected_joints", 0))
    return float(
        100.0 * missing
        + median
        + 0.20 * p95
        + 0.15 * all_median
        + 0.03 * all_p95
        + 25.0 * geometry_rejected
    )


def _support_quality_penalty(
    result: dict[str, Any], selected: Mapping[str, Any] | None = None,
) -> float:
    """Penalize weakly supported stereo when a better multiview solution exists.

    Reprojection error is not a sufficient depth-quality metric: a two-camera
    branch can fit the two rays perfectly while a three/four-camera branch is
    materially more stable.  Keep two-view candidates legal (they are needed
    during occlusion), but make them lose a close cost comparison.
    """
    counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
    finite = counts[np.isfinite(counts) & (counts >= 2)]
    median_support = float(np.median(finite)) if finite.size else 0.0
    view_count = len(selected) if selected is not None else 0
    support_gap = max(0.0, 3.0 - median_support)
    # The view term is deliberately smaller than the support term.  It only
    # breaks near ties and does not discard a valid two-camera observation.
    return float(45.0 * support_gap * support_gap + 18.0 * max(0, 3 - view_count))


def _depth_condition_profile(
    result: Mapping[str, Any], selected_count: int | None = None,
) -> dict[str, float | int | bool]:
    """Summarize whether a triangulated hand has a trustworthy depth branch.

    ``median_px`` and ``p95_px`` measure image agreement, but do not measure
    depth observability.  The profile intentionally uses palm anchors first:
    fingers can be occluded, while a bad palm depth branch moves the entire
    hand and is the source of the visible jumps.
    """
    if selected_count is None:
        selected_count = len(result.get("cameras", []))
    try:
        selected_count = int(selected_count)
    except (TypeError, ValueError):
        selected_count = 0
    def finite_value(name: str, default: float = float("nan")) -> float:
        try:
            value = float(result.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    palm_three = int(result.get("palm_three_view_count", 0) or 0)
    palm_two_fraction = finite_value("palm_two_view_fraction", 1.0)
    two_fraction = finite_value("two_view_joint_fraction", 1.0)
    palm_angle = finite_value("palm_ray_angle_p25_deg")
    angle = finite_value("ray_angle_p25_deg")
    palm_spread = max(
        finite_value("palm_pair_spread_p75_m", 0.0), 0.0
    )
    spread = max(finite_value("pair_spread_p75_m", 0.0), 0.0)
    valid = int(result.get("valid_joints", 0) or 0)
    return {
        "selected_view_count": selected_count,
        "valid_joint_count": valid,
        "palm_three_view_count": palm_three,
        "palm_two_view_fraction": palm_two_fraction,
        "two_view_joint_fraction": two_fraction,
        "palm_ray_angle_p25_deg": palm_angle,
        "ray_angle_p25_deg": angle,
        "palm_pair_spread_p75_m": palm_spread,
        "pair_spread_p75_m": spread,
        "low_support": bool(
            selected_count <= 2
            or palm_three < 2
            or palm_two_fraction > 0.60
        ),
    }


def _depth_condition_level(
    result: Mapping[str, Any],
    selected_count: int | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Classify the *depth* of a branch independently of image residuals.

    A low reprojection error is not evidence that a stereo solution is stable.
    In particular, pair intersections whose palm solutions disagree by a few
    centimetres can jump to a different depth branch while every selected
    pixel still fits.  The old code only called a branch ``bad`` above 10 cm,
    which left the 75--100 mm tail available to the temporal and identity
    passes.  Keep three explicit levels so callers can be conservative without
    throwing away every weak/occluded observation:

    ``normal``
        safe as a temporal reference and as a complete-hand companion;
    ``weak``
        legal for a standalone observation, but never a reference and only
        pairable when an independent continuity check agrees;
    ``unsafe``
        not eligible for a complete pair or a temporal alternate.
    """
    profile = _depth_condition_profile(result, selected_count)
    selected = int(profile["selected_view_count"])
    spread = float(profile["palm_pair_spread_p75_m"])
    angle = float(profile["palm_ray_angle_p25_deg"])
    palm_three = int(profile["palm_three_view_count"])
    palm_two = float(profile["palm_two_view_fraction"])
    soft_spread = max(
        float(getattr(args, "depth_max_pair_spread_m", 0.075))
        if args is not None else 0.075,
        1e-6,
    )
    unsafe_spread = max(
        float(getattr(args, "depth_unsafe_pair_spread_m", 0.090))
        if args is not None else 0.090,
        soft_spread,
    )
    minimum_angle = max(
        float(getattr(args, "depth_min_palm_angle_deg", 4.5))
        if args is not None else 4.5,
        0.0,
    )
    # A two-view angle below the preferred 7--8 degree region is weak even
    # when it is not mathematically near-parallel.  Three-view branches with
    # no redundant palm anchor are similarly weak, because the palm is what
    # determines the rigid hand depth.
    preferred_angle = max(
        float(getattr(args, "secondary_hand_min_palm_angle_deg", 7.0))
        if args is not None else 7.0,
        minimum_angle,
    )
    reasons: list[str] = []
    unsafe = False
    if np.isfinite(spread) and spread > unsafe_spread:
        unsafe = True
        reasons.append("depth_pair_disagreement")
    if selected <= 2 and np.isfinite(angle) and angle < minimum_angle:
        unsafe = True
        reasons.append("near_parallel_palm_rays")
    if selected >= 3 and np.isfinite(angle) and angle < minimum_angle:
        unsafe = True
        reasons.append("near_parallel_palm_rays")
    if unsafe:
        return "unsafe", tuple(sorted(set(reasons)))

    weak = False
    if np.isfinite(spread) and spread > soft_spread:
        weak = True
        reasons.append("weak_depth_pair_spread")
    if selected <= 2:
        if np.isfinite(angle) and angle < preferred_angle:
            weak = True
            reasons.append("weak_palm_ray_angle")
        elif not np.isfinite(angle):
            weak = True
            reasons.append("two_view_depth")
        # A well-separated two-view pair can be a stable observation even
        # though it has no third-ray redundancy.  Keep it ``normal`` when the
        # palm angle is comfortably above the weak threshold; the support cost
        # still makes a three/four-view branch win close comparisons.
    elif palm_three < max(
        int(getattr(args, "depth_min_palm_three_view", 2))
        if args is not None else 2,
        1,
    ) or palm_two > 0.60:
        weak = True
        reasons.append("low_palm_redundancy")
    if np.isfinite(angle) and angle < preferred_angle:
        weak = True
        reasons.append("weak_palm_ray_angle")
    return ("weak" if weak else "normal"), tuple(sorted(set(reasons)))


def _depth_condition_penalty(
    result: Mapping[str, Any],
    selected_count: int | None = None,
    args: argparse.Namespace | None = None,
) -> float:
    """Return a soft cost for an under-constrained depth hypothesis.

    The cost is deliberately soft.  A hand seen in only two cameras can be a
    legitimate side-on observation, but when a wider-baseline alternative is
    available it should lose even if both branches have similar reprojection
    error.  Hard rejection is reserved for the existing geometric gates.
    """
    profile = _depth_condition_profile(result, selected_count)
    target_angle = max(
        float(getattr(args, "depth_preferred_ray_angle_deg", 8.0))
        if args is not None else 8.0,
        0.0,
    )
    required_three = max(
        int(getattr(args, "depth_min_palm_three_view", 2))
        if args is not None else 2,
        0,
    )
    spread_limit = max(
        float(getattr(args, "depth_max_pair_spread_m", 0.075))
        if args is not None else 0.075,
        1e-6,
    )
    multiplier = max(
        float(getattr(args, "depth_condition_weight", 1.0))
        if args is not None else 1.0,
        0.0,
    )
    selected_count = int(profile["selected_view_count"])
    palm_angle = float(profile["palm_ray_angle_p25_deg"])
    angle = float(profile["ray_angle_p25_deg"])
    palm_three = int(profile["palm_three_view_count"])
    palm_two = float(profile["palm_two_view_fraction"])
    two_fraction = float(profile["two_view_joint_fraction"])
    palm_spread = float(profile["palm_pair_spread_p75_m"])
    spread = float(profile["pair_spread_p75_m"])

    cost = 0.0
    # A two-view branch has no redundancy.  Keep this term modest so it does
    # not erase a real occlusion, then let angle and temporal evidence decide.
    if selected_count <= 2:
        cost += 75.0
    else:
        cost += 28.0 * max(0, required_three - palm_three)
        cost += 55.0 * max(0.0, palm_two - 0.40)
        cost += 18.0 * max(0.0, two_fraction - 0.60)

    if np.isfinite(palm_angle):
        cost += 12.0 * max(0.0, target_angle - palm_angle) ** 2
    elif selected_count >= 2:
        cost += 65.0
    if np.isfinite(angle):
        cost += 4.0 * max(0.0, target_angle - angle) ** 2

    # Pair disagreement is a direct indication that the recovered depth is
    # branch-sensitive.  Use a quadratic excess so small calibration noise is
    # harmless while 10-cm disagreements become expensive.
    # A 75--100 mm palm disagreement is not ordinary pixel noise.  The old
    # coefficient made a 90 mm branch only a few cost units worse than a
    # stable branch, so it routinely survived the independent selector.  Make
    # the excess decisive while retaining a soft (rather than unconditional)
    # penalty below the hard unsafe threshold.
    cost += 600000.0 * max(0.0, palm_spread - spread_limit) ** 2
    cost += 160000.0 * max(0.0, spread - spread_limit) ** 2
    return float(multiplier * cost)


def _depth_condition_reasons(
    result: Mapping[str, Any],
    selected_count: int | None = None,
    args: argparse.Namespace | None = None,
) -> list[str]:
    """Explain severe (rather than merely weak) depth conditioning."""
    profile = _depth_condition_profile(result, selected_count)
    selected_count = int(profile["selected_view_count"])
    angle = float(profile["palm_ray_angle_p25_deg"])
    spread = float(profile["palm_pair_spread_p75_m"])
    reasons: list[str] = []
    # These thresholds are intentionally stricter than the soft-cost target,
    # and only identify branches that should not be used as a *secondary* hand
    # or as a temporal alternate.
    minimum_angle = max(
        float(getattr(args, "depth_min_palm_angle_deg", 4.5))
        if args is not None else 4.5,
        0.0,
    )
    if selected_count <= 2 and np.isfinite(angle) and angle < minimum_angle:
        reasons.append("near_parallel_palm_rays")
    unsafe_spread = max(
        float(getattr(args, "depth_unsafe_pair_spread_m", 0.090))
        if args is not None else 0.090,
        float(getattr(args, "depth_max_pair_spread_m", 0.075))
        if args is not None else 0.075,
    )
    if np.isfinite(spread) and spread > unsafe_spread:
        reasons.append("depth_pair_disagreement")
    if selected_count >= 3 and int(profile["palm_three_view_count"]) == 0:
        if np.isfinite(angle) and angle < 4.5:
            reasons.append("multiview_without_palm_redundancy")
    return reasons


def _marker_assist_cost(
    selected_by_side: dict[int, dict[str, dict[str, Any]]],
) -> float:
    """Use marker evidence as a mild hypothesis-selection term.

    Multiview geometry remains dominant. Strong marker corrections get the
    original mild preference; topology-valid partial evidence gets only one
    quarter of that weight and can therefore break close choices without
    moving image joints.
    """
    cost = 0.0
    for selected in selected_by_side.values():
        for hand in selected.values():
            marker = hand.get("marker_assist", {})
            if marker.get("applied"):
                weight = 1.0
            elif marker.get("evidence_only"):
                weight = 0.25
            else:
                continue
            matches = float(marker.get("matched_marker_count", 0))
            groups = float(marker.get("finger_group_count", 0))
            residual = float(marker.get("match_residual_median_px", 0.0))
            cost += weight * (
                0.15 * residual - 0.30 * matches - 0.20 * groups
            )
    return cost


def _tip_support_penalty(result: dict[str, Any], args: argparse.Namespace) -> float:
    """Prefer branches whose fingertips are confirmed by at least three views."""
    weight = max(float(getattr(args, "tip_three_view_penalty", 0.0)), 0.0)
    if weight <= 0.0:
        return 0.0
    counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
    points = np.asarray(result.get("points", []), dtype=np.float64)
    tips = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)
    if counts.shape != (21,) or points.shape != (21, 3):
        return 0.0
    finite = np.isfinite(points[tips]).all(axis=1)
    deficit = np.maximum(3.0 - counts[tips][finite], 0.0)
    return float(weight * deficit.sum())


def _anchor_border_cost(
    anchor_cameras: tuple[str, str] | list[str],
    selections: list[dict[str, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> float:
    """Penalize edge-truncated hands only when they are used as anchors."""
    weight = max(float(getattr(args, "anchor_border_penalty", 0.0)), 0.0)
    margin = max(float(getattr(args, "anchor_border_margin_px", 3.0)), 0.0)
    if weight <= 0.0:
        return 0.0
    truncated = 0
    for selected in selections:
        for camera_id in anchor_cameras:
            hand = selected.get(camera_id)
            camera = calibrations.get(camera_id)
            if hand is None or camera is None:
                continue
            box = np.asarray(hand.get("bbox_xyxy", []), dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            width, height = camera.image_size
            if (
                box[0] <= margin
                or box[1] <= margin
                or box[2] >= float(width - 1) - margin
                or box[3] >= float(height - 1) - margin
            ):
                truncated += 1
    return float(weight * truncated)


def _evaluate_anchor_assignment(
    anchor_cameras: tuple[str, str],
    anchor_pairs: tuple[tuple[int, int], tuple[int, int]],
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    all_cameras: tuple[str, ...],
    args: argparse.Namespace,
    allow_sparse: bool = False,
) -> dict[str, Any]:
    # Continuity enumeration may inspect a side-on branch whose fingers are
    # partially occluded.  Keep the production fusion gate strict, but allow
    # a smaller *palm-backed* branch into the offline selector.  The branch is
    # still rejected later unless its palm/reprojection/geometry quality is
    # sound, so this does not create synthetic hands in ordinary output.
    branch_min_valid = max(
        int(getattr(args, "continuity_min_branch_valid_joints", 8)), 1
    ) if allow_sparse else int(args.min_valid_joints)
    selected_by_side: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    for camera, pair in zip(anchor_cameras, anchor_pairs):
        selected_by_side[0][camera] = groups[camera][pair[0]][0]
        selected_by_side[1][camera] = groups[camera][pair[1]][1]
    anchor_results = {
        side: _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.anchor_threshold_px, 2,
        )
        for side in (0, 1)
    }
    if any(result["valid_joints"] < branch_min_valid for result in anchor_results.values()):
        return {"cost": float("inf")}
    association_errors: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for camera in all_cameras:
        if camera in anchor_cameras:
            continue
        matched, errors = _match_camera(
            camera, groups[camera], anchor_results, calibrations[camera],
            args.association_threshold_px, args.max_side_detections,
            args.detector_handedness, branch_min_valid,
        )
        for side, hand in matched.items():
            selected_by_side[side][camera] = hand
            association_errors[side][camera] = errors[side]
    duplicate_unsafe, duplicate_views = _duplicate_assignment_is_unsafe(
        selected_by_side,
        anchor_cameras,
        float(getattr(args, "duplicate_bbox_iou", 0.65)),
        float(getattr(args, "duplicate_joint_distance_px", 18.0)),
        int(getattr(args, "duplicate_overlap_cameras", 2)),
    )
    if duplicate_unsafe:
        # A pair of highly overlapping boxes is not evidence for two hands.
        # Returning an infinite candidate cost lets the independent per-hand
        # fallback retain whichever physical side has usable support.
        return {
            "cost": float("inf"),
            "rejection_reason": "duplicate_detector_observations",
            "duplicate_observation_view_count": int(duplicate_views),
        }
    if bool(getattr(args, "_require_marker_evidence", False)) and any(
        not _selection_has_marker_evidence(selected_by_side[side])
        for side in (0, 1)
    ):
        return {
            "cost": float("inf"),
            "rejection_reason": "missing_marker_evidence",
        }
    final_results = [
        _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px, 2,
        )
        for side in (0, 1)
    ]
    # A high-spread palm is an unstable *hand branch*, even when only one or
    # two joints triggered the low-level RANSAC gate.  Mark it here so the
    # independent solver cannot combine a strong hand with a numerically neat
    # but depth-ambiguous companion.
    extra_joint_support = sum(
        int(np.maximum(result["inlier_counts"] - 2, 0).sum())
        for result in final_results
    )
    matched_side_views = sum(len(selected_by_side[side]) - 2 for side in (0, 1))
    cost = (
        sum(_result_geometry_cost(result) for result in final_results)
        + sum(
            _support_quality_penalty(result, selected_by_side[side])
            for side, result in enumerate(final_results)
        )
        + sum(
            _depth_condition_penalty(
                result, len(selected_by_side[side]), args
            )
            for side, result in enumerate(final_results)
        )
        + sum(_anatomy_penalty(result["points"]) for result in final_results)
        + sum(_tip_support_penalty(result, args) for result in final_results)
        + _anchor_border_cost(
            anchor_cameras,
            [selected_by_side[0], selected_by_side[1]],
            calibrations,
            args,
        )
        + _marker_assist_cost(selected_by_side)
        - 1.5 * extra_joint_support
        - 4.0 * matched_side_views
        + _anchor_condition_penalty(anchor_cameras, calibrations, args)
    )
    return {
        "cost": float(cost),
        "anchor_cameras": list(anchor_cameras),
        "anchor_baseline_m": _anchor_baseline_m(anchor_cameras, calibrations),
        "selected": selected_by_side,
        "results": final_results,
        "association_errors": association_errors,
        "duplicate_observation_view_count": int(duplicate_views),
        "depth_levels": [
            _depth_condition_level(result, len(selected_by_side[side]), args)[0]
            for side, result in enumerate(final_results)
        ],
    }


def _quality_reasons(results: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    reasons = []
    for result in results:
        if result["valid_joints"] < args.min_valid_joints:
            reasons.append("too_few_valid_joints")
        palm = np.asarray(result.get("points", []), dtype=np.float64)
        if palm.shape == (21, 3) and np.count_nonzero(
            np.isfinite(palm[[0, 5, 9, 13, 17]]).all(axis=1)
        ) < 3:
            reasons.append("too_few_valid_palm_joints")
        if result["median_px"] > args.max_reprojection_median_px:
            reasons.append("reprojection_median_too_large")
        if result["p95_px"] > args.max_reprojection_p95_px:
            reasons.append("reprojection_p95_too_large")
        # Reprojection agreement alone cannot validate a nearly-parallel
        # stereo solution.  Apply the same severe depth-condition gate here
        # as in the independent/continuity selectors so legacy and fallback
        # paths cannot re-introduce an unstable branch.
        reasons.extend(_depth_condition_reasons(
            result, len(result.get("cameras", [])), args
        ))
    return sorted(set(reasons))


def _serialized_hand_quality(hand: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable per-hand quality values from a serialized row."""
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    quality = hand.get("quality", {})
    support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
    finite = (
        np.isfinite(points).all(axis=1)
        if points.shape == (21, 3) else np.zeros(21, dtype=bool)
    )
    support_values = support[finite] if support.shape == (21,) else np.empty(0)
    return {
        "valid_joint_count": int(quality.get(
            "valid_joint_count", np.count_nonzero(finite)
        )),
        "median_support": (
            float(np.median(support_values)) if len(support_values) else 0.0
        ),
        "reprojection_median_px": float(quality.get(
            "multiview_reprojection_median_px", np.inf
        )),
        "reprojection_p95_px": float(quality.get(
            "multiview_reprojection_p95_px", np.inf
        )),
        "all_view_median_px": float(quality.get(
            "all_view_reprojection_median_px", np.inf
        )),
        "geometry_rejected_joints": int(quality.get(
            "geometry_rejected_joint_count",
            hand.get("geometry_rejected_joints", 0),
        )),
    }


def _complete_row_side_diagnostics(
    row: dict[str, Any], args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool]:
    """Return side quality diagnostics and whether one side is clearly weak.

    A complete row is allowed to contain two-camera stereo for both hands.  We
    only downgrade it to independent/partial acceptance when one side is
    materially weaker than the other; this avoids deleting valid stereo during
    normal occlusion while preventing one bad side from hard-binding the good
    side to its branch.
    """
    hands = list(row.get("hands", []))
    if len(hands) != 2 or row.get("partial_hand_acceptance"):
        return [], False
    diagnostics = [_serialized_hand_quality(hand) for hand in hands]
    # Two-view stereo is a valid observation, especially for the four-camera
    # 1--4 subset where one hand is often visible in only two cameras.  The
    # previous gate treated a perfectly usable 2-vs-3 support split as a hard
    # binding failure and converted long stretches of normal dual-hand data to
    # partial rows.  Only a *material* quality failure is allowed to trigger
    # the independent-side fallback; support differences alone are a tie
    # breaker unless the gap is very large and the weak side is also sparse.
    min_support = float(getattr(args, "min_full_hand_support", 2.5))
    support_gap = float(getattr(args, "full_hand_support_gap", 1.5))
    weak = []
    for value in diagnostics:
        valid_count = int(value["valid_joint_count"])
        median_support = float(value["median_support"])
        reproj = float(value["reprojection_median_px"])
        p95 = float(value["reprojection_p95_px"])
        geometry_rejected = int(value["geometry_rejected_joints"])
        severe = (
            valid_count < int(getattr(args, "min_valid_joints", 12))
            or median_support < 1.75
            or geometry_rejected >= max(3, int(0.25 * max(valid_count, 1)))
            or reproj > float(getattr(args, "max_reprojection_median_px", 15.0))
            or p95 > 1.25 * float(
                getattr(args, "max_reprojection_p95_px", 40.0)
            )
        )
        weak.append(bool(severe))
    support_values = [value["median_support"] for value in diagnostics]
    valid_values = [value["valid_joint_count"] for value in diagnostics]
    support_only_gap = bool(
        min(support_values) <= 2.0
        and max(support_values) - min(support_values)
        >= max(support_gap, 2.5)
        and min(valid_values) < 0.85 * max(valid_values)
    )
    one_side_weak = bool(
        (weak[0] != weak[1] and max(support_values) >= 2.0)
        or (support_only_gap and support_values[0] != support_values[1])
    )
    return diagnostics, one_side_weak


def _full_row_is_reliable(row: dict[str, Any], args: argparse.Namespace) -> bool:
    diagnostics, weak = _complete_row_side_diagnostics(row, args)
    if not diagnostics or weak:
        return False
    # A complete row is only reliable when both hands have a stable depth
    # branch.  This prevents a high-spread side from hard-binding the good
    # side before identity tracking runs.
    for hand in row.get("hands", []):
        quality = hand.get("quality", {})
        level, _reasons = _depth_condition_level(
            quality, len(hand.get("camera_ids", [])), args
        )
        if level != "normal":
            return False
    return True


def _single_side_candidates(
    side: int,
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
    allow_sparse: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate usable stereo/multiview assignments for one physical side."""
    minimum_valid_joints = max(
        int(getattr(args, "continuity_min_branch_valid_joints", 8)), 1
    ) if allow_sparse else int(args.min_valid_joints)
    available = {
        camera: _single_side_options(
            groups[camera], side, args.max_anchor_detections,
            args.detector_handedness,
        )
        for camera in cameras
    }
    anchor_pairs = [
        pair for pair in itertools.combinations(cameras, 2)
        if all(available[camera] for camera in pair)
    ]
    candidates: list[dict[str, Any]] = []
    for anchor_pair in anchor_pairs:
        for first, second in itertools.product(
            available[anchor_pair[0]], available[anchor_pair[1]]
        ):
            selected: dict[str, dict[str, Any]] = {
                anchor_pair[0]: first[1], anchor_pair[1]: second[1]
            }
            errors: dict[str, float] = {}
            anchor_result = _triangulate_hand(
                side, selected, calibrations, args.anchor_threshold_px, 2
            )
            if anchor_result["valid_joints"] < minimum_valid_joints:
                continue
            errors[anchor_pair[0]] = float(anchor_result.get("median_px", np.inf))
            errors[anchor_pair[1]] = float(anchor_result.get("median_px", np.inf))
            for camera in cameras:
                if camera in selected:
                    continue
                hand, error = _match_camera_single_side(
                    groups[camera], side, anchor_result["points"],
                    calibrations[camera], args.association_threshold_px,
                    args.max_side_detections, args.detector_handedness,
                )
                if hand is not None:
                    selected[camera] = hand
                    errors[camera] = float(error)
            if (
                bool(getattr(args, "_require_marker_evidence", False))
                and not _selection_has_marker_evidence(selected)
            ):
                continue
            result = _triangulate_hand(
                side, selected, calibrations, args.ransac_threshold_px, 2
            )
            if allow_sparse:
                points = np.asarray(result.get("points", []), dtype=np.float64)
                palm_indices = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
                finite_palm = (
                    points.shape == (21, 3)
                    and np.isfinite(points[palm_indices]).all(axis=1)
                )
                counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
                reliable_palm = (
                    finite_palm
                    & (counts.shape == (21,))
                    & (
                        counts[palm_indices] >= 2
                        if counts.shape == (21,)
                        else np.zeros(len(palm_indices), dtype=bool)
                    )
                )
                support = (
                    counts[palm_indices][reliable_palm]
                    if counts.shape == (21,) else np.empty(0)
                )
                median_support = (
                    float(np.median(support)) if support.size else 0.0
                )
                if (
                    result["valid_joints"] < minimum_valid_joints
                    or int(np.count_nonzero(reliable_palm)) < int(
                        getattr(args, "continuity_min_palm_points", 3)
                    )
                    or median_support < float(
                        getattr(args, "continuity_min_palm_support", 2.0)
                    )
                    or not np.isfinite(result.get("median_px", np.inf))
                    or float(result.get("median_px", np.inf))
                    > float(getattr(args, "max_reprojection_median_px", 15.0))
                    or not np.isfinite(result.get("p95_px", np.inf))
                    or float(result.get("p95_px", np.inf))
                    > float(getattr(args, "max_reprojection_p95_px", 40.0))
                ):
                    continue
            elif _quality_reasons([result], args):
                continue
            support = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
            extra_support = int(np.maximum(support - 2, 0).sum()) if support.size else 0
            cost = (
                _result_geometry_cost(result)
                + _support_quality_penalty(result, selected)
                + _depth_condition_penalty(result, len(selected), args)
                + _anatomy_penalty(result["points"])
                + _tip_support_penalty(result, args)
                + _anchor_border_cost(
                    anchor_pair, [selected], calibrations, args
                )
                + _marker_assist_cost({side: selected})
                - 1.5 * extra_support - 4.0 * max(0, len(selected) - 2)
                + _anchor_condition_penalty(anchor_pair, calibrations, args)
            )
            if tuple(anchor_pair) != tuple(preferred_anchors):
                cost += 5.0
            candidates.append({
                "cost": float(cost),
                "side": side,
                "anchor_cameras": list(anchor_pair),
                "anchor_baseline_m": _anchor_baseline_m(anchor_pair, calibrations),
                "selected": selected,
                "result": result,
                "association_errors": errors,
            })
    return candidates


def _fuse_partial_hands(
    sync_index: int,
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Accept independently visible hands when the dual-hand path is unavailable."""
    selected_candidates: list[dict[str, Any]] = []
    chosen_anchors: list[str] = []
    for side in (0, 1):
        candidates = _single_side_candidates(
            side, groups, calibrations, cameras, preferred_anchors, args
        )
        safe_candidates = [
            candidate for candidate in candidates
            if _independent_candidate_quality(candidate, args)["single_safe"]
        ]
        if not safe_candidates:
            continue
        best = min(safe_candidates, key=lambda item: item["cost"])
        selected_candidates.append(best)
    # In handedness-ignore mode the two hypotheses belonging to one detector
    # box can otherwise be emitted as two physical hands.  If both independently
    # selected sides reuse the same box in at least two cameras, keep only the
    # geometrically better one.  The normal dual-hand solver handles distinct
    # boxes before this fallback is reached.
    duplicate_views = 0
    duplicate_suppressed = False
    pair_rejected_reason: str | None = None
    if len(selected_candidates) == 2:
        pair_allowed, pair_reason, _first_quality, _second_quality = (
            _independent_pair_decision(
                selected_candidates[0], selected_candidates[1], args
            )
        )
        if not pair_allowed:
            # Preserve the stronger safe branch as a partial row.  A fallback
            # path must obey the same weak-secondary rule as the main
            # independent solver, otherwise it can recreate hard binding.
            pair_rejected_reason = pair_reason
            selected_candidates = [max(
                selected_candidates,
                key=lambda item: (
                    int(_independent_candidate_quality(item, args)["reliable"]),
                    int(_independent_candidate_quality(item, args)["usable"]),
                    -float(item.get("cost", np.inf)),
                ),
            )]
    if len(selected_candidates) == 2:
        duplicate_unsafe, duplicate_views = _duplicate_assignment_is_unsafe(
            {
                int(candidate["side"]): candidate["selected"]
                for candidate in selected_candidates
            },
            None,
            float(getattr(args, "duplicate_bbox_iou", 0.65)),
            float(getattr(args, "duplicate_joint_distance_px", 18.0)),
            int(getattr(args, "duplicate_overlap_cameras", 2)),
        )
        if duplicate_unsafe:
            selected_candidates = [min(
                selected_candidates, key=lambda item: item["cost"]
            )]
            duplicate_suppressed = True
            pair_rejected_reason = "duplicate_detector_observations"
    hands: list[dict[str, Any]] = []
    for best in selected_candidates:
        side = int(best["side"])
        anchors = tuple(best["anchor_cameras"])
        hand = _serialize_hand(
            best["result"], best["selected"], calibrations, anchors
        )
        hand["association_error_px"] = {
            camera: float(value) for camera, value in best["association_errors"].items()
        }
        hand["fusion_mode"] = (
            "multiview" if len(best["selected"]) >= 3 else "stereo_anchor"
        )
        hand["side_selection_cost"] = float(best["cost"])
        hands.append(hand)
        chosen_anchors.extend(anchors)
    if not hands:
        return False, {"sync_index": sync_index, "reason": "no_two_camera_anchor_for_any_hand"}
    present = {int(hand["side"]) for hand in hands}
    missing = sorted({0, 1} - present)
    active_cameras = sorted({camera for hand in hands for camera in hand["camera_ids"]})
    # Keep the preferred pair when both hands selected it; otherwise record the
    # union for diagnostics without pretending that both sides share a stereo pair.
    anchor_record = list(preferred_anchors) if all(
        camera in chosen_anchors for camera in preferred_anchors
    ) else sorted(set(chosen_anchors))
    return True, {
        "sync_index": sync_index,
        "anchor_cameras": anchor_record,
        "anchor_baseline_m": (
            _anchor_baseline_m(anchor_record, calibrations)
            if len(anchor_record) == 2 else None
        ),
        "assignment_cost": float(sum(
            float(candidate["cost"]) for candidate in selected_candidates
        )),
        "partial_hand_acceptance": len(present) < 2,
        "missing_sides": missing,
        "partial_reason": (
            "duplicate_detector_observations"
            if duplicate_suppressed else (
                pair_rejected_reason or "independent_side_fusion"
            )
        ),
        "duplicate_observation_view_count": int(duplicate_views),
        "active_cameras": active_cameras,
        "active_camera_count": len(active_cameras),
        "hands": hands,
    }


def _independent_candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable key for duplicate independent-hand hypotheses."""
    centre = _hand_center_from_row({
        "joints_base_m": np.asarray(
            candidate.get("result", {}).get("points", []), dtype=np.float64
        ).tolist()
    })
    selected_key = tuple(sorted(
        (
            str(camera),
            int(hand.get("detection_index", -1)),
            int(hand.get("hypothesis_index", -1)),
            int(hand.get("is_right", candidate.get("side", 0))),
        )
        for camera, hand in candidate.get("selected", {}).items()
    ))
    rounded = tuple(np.round(centre, 4)) if np.all(np.isfinite(centre)) else ()
    return selected_key + rounded


def _independent_frame_candidates(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Enumerate one-hand branches without imposing a left/right pair.

    The old complete-row path first chose a detector-index permutation and only
    then decided whether the resulting two-hand pose was acceptable.  A weak
    detector box could therefore drag a good hand onto the wrong depth branch.
    This helper deliberately treats every hand candidate as an independent
    observation.  The caller performs a small two-candidate matching problem
    and can keep one hand when the other is absent.
    """
    groups = {
        camera: _groups(prediction_rows[camera][sync_index])
        for camera in cameras
    }
    modes = [args.detector_handedness]
    if args.detector_handedness == "adaptive":
        modes = ["strict", "ignore"]
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    # Keep a strict candidate whenever it exists, but include geometry-only
    # alternatives because detector.pt labels are known to flip on side-on
    # frames.  Sparse candidates are admitted only with the palm/reprojection
    # quality gate used by the continuity code.
    for mode in modes:
        local_args = argparse.Namespace(**vars(args))
        local_args.detector_handedness = mode
        for hypothesis_side in (0, 1):
            candidates = _single_side_candidates(
                hypothesis_side,
                groups,
                calibrations,
                cameras,
                preferred_anchors,
                local_args,
                allow_sparse=False,
            )
            sparse_candidates = _single_side_candidates(
                hypothesis_side,
                groups,
                calibrations,
                cameras,
                preferred_anchors,
                local_args,
                allow_sparse=True,
            )
            for candidate in [*candidates, *sparse_candidates]:
                if not np.isfinite(float(candidate.get("cost", np.inf))):
                    continue
                candidate = dict(candidate)
                candidate["hypothesis_side"] = int(hypothesis_side)
                candidate["detector_mode"] = mode
                # Preserve the fact that this was a sparse fallback.  It is
                # useful for diagnostics and lets pair selection prefer two
                # fully supported sides when both are available.
                candidate["sparse_branch"] = bool(
                    int(candidate.get("result", {}).get("valid_joints", 0))
                    < int(getattr(args, "min_valid_joints", 12))
                )
                key = _independent_candidate_key(candidate)
                previous = by_key.get(key)
                if previous is None or float(candidate["cost"]) < float(previous["cost"]):
                    by_key[key] = candidate
    values = list(by_key.values())
    values.sort(key=lambda item: float(item.get("cost", np.inf)))
    # Keep depth alternatives for the *same* image observation together.  A
    # global cost sort is unsafe here: a cheap branch for one detector box can
    # consume the whole 32-entry budget and evict the more expensive, but
    # temporally correct, branch for another box.  Grouping by camera-local
    # detection indices preserves the branch diversity needed by the offline
    # continuity pass while retaining a deterministic global cap.
    per_observation = max(int(getattr(
        args, "continuity_branch_candidates_per_observation", 4
    )), 1)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in values:
        observation_key = tuple(sorted(
            (
                str(camera),
                int(hand.get("detection_index", -1)),
            )
            for camera, hand in candidate.get("selected", {}).items()
        ))
        grouped.setdefault(observation_key, []).append(candidate)
    for group in grouped.values():
        group.sort(key=lambda item: float(item.get("cost", np.inf)))
    # First reserve the best branch from every observation set.  Remaining
    # slots are filled round-robin so different 3-D depths of one observation
    # survive alongside alternatives from other detector boxes.
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: float(group[0].get("cost", np.inf)),
    )
    maximum = max(int(getattr(args, "continuity_max_hypotheses", 32)), 1)
    selected_values: list[dict[str, Any]] = []
    for rank in range(per_observation):
        for group in ordered_groups:
            if rank < len(group):
                selected_values.append(group[rank])
                if len(selected_values) >= maximum:
                    break
        if len(selected_values) >= maximum:
            break
    selected_values.sort(key=lambda item: float(item.get("cost", np.inf)))
    return selected_values[:maximum]


def _independent_pair_orientation(
    first: dict[str, Any], second: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], float]:
    """Choose a deterministic initial slot order for two hand candidates."""
    first_side = int(first.get("hypothesis_side", first.get("side", 0)))
    second_side = int(second.get("hypothesis_side", second.get("side", 1)))
    if first_side != second_side and {first_side, second_side} == {0, 1}:
        if first_side == 0:
            return {0: first, 1: second}, 0.0
        return {0: second, 1: first}, 0.0
    first_centre = _hand_center_from_row({
        "joints_base_m": np.asarray(first["result"]["points"]).tolist()
    })
    second_centre = _hand_center_from_row({
        "joints_base_m": np.asarray(second["result"]["points"]).tolist()
    })
    # The base-x tie break is only used for the first frame or when both
    # hypotheses carry the same detector side.  The global trajectory pass
    # remains the authority for physical identity.
    if np.all(np.isfinite(first_centre)) and np.all(np.isfinite(second_centre)):
        if first_centre[0] <= second_centre[0]:
            return {0: first, 1: second}, 0.02
        return {0: second, 1: first}, 0.02
    return {0: first, 1: second}, 0.02


def _independent_candidate_quality(
    candidate: Mapping[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    """Classify one independent hand branch for pair selection.

    ``_single_side_candidates`` intentionally returns sparse alternatives so a
    side-on hand can be recovered by the temporal pass.  Those alternatives
    must not automatically become the second member of a normal two-hand row:
    doing so recreates the old hard-binding failure.  Keep three states here:
    reliable (safe for an ordinary pair), usable (safe only when the other
    hand is similarly weak), and unusable (single-hand fallback only).
    """
    result = candidate.get("result", {})
    selected = candidate.get("selected", {})
    points = np.asarray(result.get("points", []), dtype=np.float64)
    palm = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
    finite_palm = (
        points.shape == (21, 3)
        and np.isfinite(points[palm]).all(axis=1)
    )
    counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
    if counts.shape != (21,):
        counts = np.zeros(21, dtype=np.float64)
    reliable_palm = finite_palm & (counts[palm] >= 2)
    palm_count = int(np.count_nonzero(reliable_palm))
    palm_support = (
        float(np.median(counts[palm][reliable_palm]))
        if palm_count else 0.0
    )
    valid = int(result.get("valid_joints", 0) or 0)
    median = float(result.get("median_px", np.inf))
    p95 = float(result.get("p95_px", np.inf))
    depth_profile = _depth_condition_profile(result, len(selected))
    depth_reasons = _depth_condition_reasons(result, len(selected), args)
    depth_level, depth_level_reasons = _depth_condition_level(
        result, len(selected), args
    )
    minimum_valid = max(
        int(getattr(args, "secondary_hand_min_valid_joints", 10)), 1
    )
    minimum_palm = max(
        int(getattr(args, "secondary_hand_min_palm_points", 3)), 3
    )
    image_ok = bool(
        np.isfinite(median)
        and np.isfinite(p95)
        and median <= float(getattr(args, "max_reprojection_median_px", 15.0))
        and p95 <= float(getattr(args, "max_reprojection_p95_px", 40.0))
    )
    geometry_ok = not bool(result.get("geometry_rejection_reasons")) and (
        int(result.get("geometry_rejected_joints", 0) or 0)
        <= max(10, 21 - minimum_valid)
    )
    usable = bool(
        valid >= minimum_valid
        and palm_count >= minimum_palm
        and palm_support >= float(
            getattr(args, "continuity_min_palm_support", 2.0)
        )
        and image_ok
        and geometry_ok
        and not depth_reasons
        and depth_level != "unsafe"
    )
    # A branch with full joint count and at least three reliable palm anchors
    # is strong enough to be paired.  For >=3 cameras, demand some palm
    # redundancy; for true stereo, demand a non-degenerate angle.
    palm_three = int(depth_profile["palm_three_view_count"])
    palm_angle = float(depth_profile["palm_ray_angle_p25_deg"])
    selected_count = int(depth_profile["selected_view_count"])
    if selected_count >= 3:
        reliable = usable and depth_level == "normal" and valid >= int(getattr(args, "min_valid_joints", 12)) and (
            palm_three >= max(int(getattr(args, "depth_min_palm_three_view", 2)), 1)
            or palm_support >= 3.0
        )
    else:
        reliable = usable and depth_level == "normal" and valid >= int(getattr(args, "min_valid_joints", 12)) and (
            not np.isfinite(palm_angle)
            or palm_angle >= float(getattr(
                args, "secondary_hand_min_palm_angle_deg", 7.0
            ))
        )
    return {
        "valid": valid,
        "palm_count": palm_count,
        "palm_support": palm_support,
        "median_px": median,
        "p95_px": p95,
        "selected_count": selected_count,
        "depth_profile": depth_profile,
        "depth_reasons": depth_reasons,
        "depth_level": depth_level,
        "depth_level_reasons": list(depth_level_reasons),
        "usable": usable,
        # A single-hand output may be sparse for a side-on/occluded gesture,
        # but it must still pass the image, geometry, palm-support and severe
        # depth gates above.  Keep this explicit so callers never fall back
        # to an arbitrary finite-cost unsafe branch.
        # Unsafe depth is never emitted, even as a one-hand fallback.  A weak
        # branch may still be useful when the only visible hand is occluded,
        # but it cannot bind a second hand or seed a temporal reference.
        "single_safe": bool(usable and depth_level != "unsafe"),
        "reliable": bool(reliable),
        "sparse": bool(candidate.get("sparse_branch", False)),
        "cost": float(candidate.get("cost", np.inf)),
    }


def _candidate_detector_observation_overlap(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> int:
    """Count camera observations reused by two independent candidates.

    The duplicate-box test compares image geometry, but an exact
    ``(camera, detection_index)`` reuse is stronger evidence: WiLoR stores two
    handedness hypotheses for the same detector crop and those hypotheses may
    have different 2-D joints.  Treating them as two hands is never valid,
    even when the crops happen not to satisfy the looser IoU threshold.
    """
    first_selected = first.get("selected", {})
    second_selected = second.get("selected", {})
    overlap = 0
    for camera in set(first_selected) & set(second_selected):
        first_hand = first_selected.get(camera, {})
        second_hand = second_selected.get(camera, {})
        try:
            first_index = int(first_hand.get("detection_index", -1))
            second_index = int(second_hand.get("detection_index", -2))
        except (TypeError, ValueError):
            continue
        if first_index == second_index:
            overlap += 1
    return overlap


def _independent_pair_decision(
    first: Mapping[str, Any], second: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, Any], dict[str, Any]]:
    """Return whether two independent branches are safe to emit together."""
    first_quality = _independent_candidate_quality(first, args)
    second_quality = _independent_candidate_quality(second, args)
    exact_overlap = _candidate_detector_observation_overlap(first, second)
    if exact_overlap:
        return False, "same_detector_observation", first_quality, second_quality
    near_duplicate, _near_duplicate_diagnostics = (
        _near_coincident_independent_pair_is_unsafe(first, second, args)
    )
    if near_duplicate:
        return (
            False,
            "near_coincident_3d_hands",
            first_quality,
            second_quality,
        )
    # A branch with a materially disagreeing palm depth is not a valid member
    # of a complete two-hand row.  Keeping it beside a strong hand used to
    # satisfy the row-level contract while poisoning the identity/temporal
    # selector.  It may still be emitted as a standalone weak observation if
    # no safer hand exists.
    for quality in (first_quality, second_quality):
        if quality.get("depth_level") == "unsafe":
            return False, "weak_secondary_depth_condition", first_quality, second_quality
        if "weak_depth_pair_spread" in quality.get("depth_level_reasons", []):
            return False, "weak_secondary_depth_spread", first_quality, second_quality
    if first_quality["reliable"] and second_quality["reliable"]:
        return True, "both_reliable", first_quality, second_quality
    if not first_quality["usable"] or not second_quality["usable"]:
        return False, "weak_secondary_hand", first_quality, second_quality
    # Two under-constrained stereo branches are not made safe by appearing
    # together.  Requiring both to clear the reliable angle threshold avoids
    # the exact failure mode where two independent low-angle intersections are
    # emitted as a complete pair and then jump in depth on the next frame.
    minimum_pair_angle = max(float(getattr(
        args, "secondary_hand_min_palm_angle_deg", 7.0
    )), 0.0)
    low_angle_qualities = []
    for quality in (first_quality, second_quality):
        profile = quality["depth_profile"]
        angle = float(profile.get("palm_ray_angle_p25_deg", np.nan))
        if (
            int(quality["selected_count"]) <= 2
            and np.isfinite(angle)
            and angle < minimum_pair_angle
        ):
            low_angle_qualities.append(quality)
    if low_angle_qualities:
        return False, "weak_secondary_depth_condition", first_quality, second_quality
    # If exactly one branch is strong, the other branch must have enough depth
    # observability to be a safe companion.  A low-angle two-view branch can
    # have a perfect reprojection while being tens of centimetres away in Z;
    # keeping it in the pair is precisely the old hard-binding failure.
    if first_quality["reliable"] != second_quality["reliable"]:
        weak_quality = (
            second_quality if first_quality["reliable"] else first_quality
        )
        profile = weak_quality["depth_profile"]
        angle = float(profile.get("palm_ray_angle_p25_deg", np.nan))
        min_angle = float(getattr(
            args, "secondary_hand_min_palm_angle_deg", 7.0
        ))
        if (
            int(weak_quality["selected_count"]) <= 2
            and np.isfinite(angle)
            and angle < min_angle
        ):
            return False, "weak_secondary_depth_condition", first_quality, second_quality
        if weak_quality["sparse"]:
            return False, "sparse_secondary_hand", first_quality, second_quality
    # Two moderately weak hands may still be a real pair (for example a
    # side-on gesture seen by three cameras).  Do not accept a branch whose
    # quality cost is much worse than its partner: that is usually a duplicate
    # detector box or an unconstrained depth solution.
    gap = abs(first_quality["cost"] - second_quality["cost"])
    max_gap = max(float(getattr(args, "secondary_hand_max_cost_gap", 420.0)), 0.0)
    if gap > max_gap:
        return False, "weak_secondary_cost_gap", first_quality, second_quality
    if first_quality["sparse"] or second_quality["sparse"]:
        return False, "sparse_secondary_hand", first_quality, second_quality
    return True, "both_usable", first_quality, second_quality


def _independent_frame_fusion(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Fuse visible hands independently, then expose a row-level identity seed."""
    candidates = _independent_frame_candidates(
        sync_index, prediction_rows, calibrations, cameras,
        preferred_anchors, args,
    )
    if not candidates:
        return False, {
            "sync_index": sync_index,
            "reason": "no_independent_hand_candidate",
        }

    pair_options: list[
        tuple[
            float, dict[int, dict[str, Any]], int, str,
            dict[str, Any], dict[str, Any],
        ]
    ] = []
    rejected_pair_reasons: Counter[str] = Counter()
    rejected_pair_cost_gaps: list[float] = []
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            oriented, orientation_penalty = _independent_pair_orientation(first, second)
            near_duplicate, near_duplicate_diagnostics = (
                _near_coincident_independent_pair_is_unsafe(first, second, args)
            )
            if near_duplicate:
                rejected_pair_reasons["near_coincident_3d_hands"] += 1
                rejected_pair_cost_gaps.append(float(
                    near_duplicate_diagnostics.get("center_distance_m")
                    or 0.0
                ))
                continue
            duplicate_unsafe, duplicate_views = _duplicate_assignment_is_unsafe(
                {
                    slot: candidate.get("selected", {})
                    for slot, candidate in oriented.items()
                },
                None,
                float(getattr(args, "duplicate_bbox_iou", 0.65)),
                float(getattr(args, "duplicate_joint_distance_px", 18.0)),
                int(getattr(args, "duplicate_overlap_cameras", 2)),
            )
            if duplicate_unsafe:
                rejected_pair_reasons["duplicate_detector_observations"] += 1
                continue
            pair_allowed, pair_reason, first_quality, second_quality = (
                _independent_pair_decision(first, second, args)
            )
            if not pair_allowed:
                rejected_pair_reasons[pair_reason] += 1
                rejected_pair_cost_gaps.append(abs(
                    float(first.get("cost", np.inf))
                    - float(second.get("cost", np.inf))
                ))
                continue
            pair_cost = (
                float(first.get("cost", np.inf))
                + float(second.get("cost", np.inf))
                + orientation_penalty
            )
            pair_options.append((
                pair_cost, oriented, duplicate_views, pair_reason,
                first_quality, second_quality,
            ))

    chosen: dict[int, dict[str, Any]]
    duplicate_views = 0
    pair_margin = None
    selected_pair_reason = None
    selected_pair_quality: list[dict[str, Any]] = []
    if pair_options:
        pair_options.sort(key=lambda item: item[0])
        (
            best_cost, chosen, duplicate_views, selected_pair_reason,
            first_quality, second_quality,
        ) = pair_options[0]
        selected_pair_quality = [first_quality, second_quality]
        if len(pair_options) > 1:
            pair_margin = float(pair_options[1][0] - best_cost)
    else:
        # A visible single hand is still useful.  Do not fabricate a second
        # hand merely to satisfy a complete-row contract.
        safe_singles = [
            item for item in candidates
            if _independent_candidate_quality(item, args)["single_safe"]
        ]
        if not safe_singles:
            return False, {
                "sync_index": sync_index,
                "reason": "no_safe_independent_hand_candidate",
                "candidate_count": len(candidates),
                "rejected_pair_reason_counts": dict(rejected_pair_reasons),
            }
        ranked_singles = sorted(
            safe_singles,
            key=lambda item: (
                not _independent_candidate_quality(item, args)["reliable"],
                float(item.get("cost", np.inf)),
            ),
        )
        best = ranked_singles[0]
        chosen = {
            int(best.get("hypothesis_side", best.get("side", 0))): best,
        }
        best_cost = float(best.get("cost", np.inf))

    hands: list[dict[str, Any]] = []
    selected_slots: dict[int, dict[str, dict[str, Any]]] = {}
    for slot, candidate in sorted(chosen.items()):
        selected_slots[slot] = candidate.get("selected", {})
        anchors = tuple(candidate.get("anchor_cameras", preferred_anchors))
        hand = _serialize_hand(
            candidate["result"], candidate["selected"], calibrations, anchors,
        )
        hand["side"] = int(slot)
        hand["candidate_hypothesis_side"] = int(
            candidate.get("hypothesis_side", candidate.get("side", slot))
        )
        hand["detector_handedness_mode"] = candidate.get("detector_mode")
        hand["sparse_branch"] = bool(candidate.get("sparse_branch", False))
        hand["side_selection_cost"] = float(candidate.get("cost", np.inf))
        hand["association_error_px"] = {
            camera: float(value)
            for camera, value in candidate.get("association_errors", {}).items()
        }
        hand["fusion_mode"] = (
            "multiview" if len(candidate.get("selected", {})) >= 3
            else "stereo_anchor"
        )
        hands.append(hand)

    if not hands:
        return False, {
            "sync_index": sync_index,
            "reason": "independent_pair_rejected",
        }
    present = {int(hand["side"]) for hand in hands}
    active_cameras = sorted(set().union(*(
        set(hand.get("camera_ids", [])) for hand in hands
    )))
    common_preferred = all(
        all(camera in candidate.get("selected", {}) for candidate in chosen.values())
        for camera in preferred_anchors
    ) if len(chosen) == 2 else False
    if common_preferred:
        row_anchors = list(preferred_anchors)
    elif len(chosen) == 1:
        row_anchors = list(next(iter(chosen.values())).get(
            "anchor_cameras", preferred_anchors
        ))
    else:
        row_anchors = sorted(set().union(*(
            set(candidate.get("anchor_cameras", []))
            for candidate in chosen.values()
        )))
    mode_values = {
        str(candidate.get("detector_mode")) for candidate in chosen.values()
    }
    if len(mode_values) == 1:
        mode = next(iter(mode_values))
        handedness_mode = f"independent_{mode}"
    else:
        handedness_mode = "independent_mixed"
    row = {
        "sync_index": sync_index,
        "anchor_cameras": row_anchors,
        "anchor_baseline_m": (
            _anchor_baseline_m(row_anchors, calibrations)
            if len(row_anchors) == 2 else None
        ),
        "assignment_cost": float(best_cost),
        "assignment_margin": pair_margin,
        "active_cameras": active_cameras,
        "active_camera_count": len(active_cameras),
        "hands": sorted(hands, key=lambda hand: int(hand.get("side", 0))),
        "handedness_mode": handedness_mode,
        "independent_hand_selection": {
            "algorithm": "independent_side_candidates_depth_guarded_v2",
            "candidate_count": len(candidates),
            "selected_count": len(hands),
            "duplicate_observation_view_count": int(duplicate_views),
            "selected_pair_reason": selected_pair_reason,
            "selected_pair_quality": selected_pair_quality,
            "rejected_pair_reason_counts": dict(rejected_pair_reasons),
            "weak_candidate_cost_gap": (
                float(min(rejected_pair_cost_gaps))
                if rejected_pair_cost_gaps else None
            ),
            "pair_rejected_weak_side": bool(
                len(hands) == 1
                and any(
                    reason in rejected_pair_reasons
                    for reason in (
                        "weak_secondary_hand", "weak_secondary_cost_gap",
                        "sparse_secondary_hand",
                    )
                )
            ),
            "selected": [
                {
                    "slot": int(slot),
                    "hypothesis_side": int(candidate.get(
                        "hypothesis_side", candidate.get("side", slot)
                    )),
                    "detector_mode": candidate.get("detector_mode"),
                    "cost": float(candidate.get("cost", np.inf)),
                    "anchor_cameras": list(candidate.get("anchor_cameras", [])),
                    "selected_camera_count": len(candidate.get("selected", {})),
                }
                for slot, candidate in sorted(chosen.items())
            ],
        },
    }
    if len(present) < 2:
        row["partial_hand_acceptance"] = True
        row["missing_sides"] = sorted({0, 1} - present)
        row["partial_reason"] = (
            "weak_secondary_hand"
            if row["independent_hand_selection"]["pair_rejected_weak_side"]
            else "independent_side_fusion"
        )
    return True, row


def _independent_hypothesis_missing() -> dict[str, Any]:
    """Return the empty hand slot used by independent-track hypotheses."""
    return {
        "points": np.full((21, 3), np.nan, dtype=np.float64),
        "inlier_counts": np.zeros(21, dtype=np.int32),
        "valid_joints": 0,
        "median_px": np.inf,
        "p95_px": np.inf,
        "missing": True,
    }


def _independent_candidate_hypothesis(
    chosen: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Convert independently selected candidates to the track representation."""
    results = [_independent_hypothesis_missing(), _independent_hypothesis_missing()]
    selected: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    errors: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for slot, candidate in chosen.items():
        slot = int(slot)
        if slot not in (0, 1):
            continue
        results[slot] = candidate["result"]
        selected[slot] = candidate.get("selected", {})
        errors[slot] = candidate.get("association_errors", {})
    return {
        "cost": float(sum(
            float(candidate.get("cost", np.inf))
            for candidate in chosen.values()
        )),
        "anchor_cameras": list(next(iter(chosen.values())).get(
            "anchor_cameras", []
        )) if chosen else [],
        "anchor_baseline_m": float(next(iter(chosen.values())).get(
            "anchor_baseline_m", 0.0
        )) if chosen else 0.0,
        "results": results,
        "selected": selected,
        "association_errors": errors,
        "_independent_candidates": dict(chosen),
    }


def _independent_side_cost(
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    args: argparse.Namespace,
) -> float:
    """Return the per-hand part of the independent branch cost.

    Keeping this calculation separate lets the continuity pass build a hybrid
    hypothesis (one delivered hand plus one alternate depth branch) without
    pretending that the untouched hand was triangulated again.
    """
    points = np.asarray(result.get("points", []), dtype=np.float64)
    support = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
    extra_support = int(np.maximum(support - 2, 0).sum()) if support.size else 0
    return float(
        _result_geometry_cost(dict(result))
        + _support_quality_penalty(dict(result), selected)
        + _depth_condition_penalty(dict(result), len(selected), args)
        + _anatomy_penalty(points)
        + _marker_assist_cost({0: dict(selected)})
        - 1.5 * extra_support
        - 4.0 * max(0, len(selected) - 2)
    )


def _independent_hybrid_hypothesis(
    original: dict[str, Any],
    original_row: dict[str, Any],
    side: int,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Replace one side of a delivered row while preserving the other side.

    The previous continuity enumerator only emitted complete hypotheses made
    entirely from newly triangulated candidates.  That couples the fate of a
    good hand to a bad companion: if the pair spread of the companion is
    rejected, the good hand cannot be retained while its partner changes
    branch.  This helper creates the missing hybrid state explicitly.
    """
    side = int(side)
    if side not in (0, 1):
        return None
    by_side = {
        int(hand.get("side", -1)): hand
        for hand in original_row.get("hands", [])
        if int(hand.get("side", -1)) in (0, 1)
    }
    replaced_hand = by_side.get(side)
    other_hand = by_side.get(1 - side)
    if replaced_hand is None or other_hand is None:
        # A partial row has no untouched companion.  Ordinary single-candidate
        # hypotheses already cover that case and should remain the only path.
        return None
    if not candidate.get("selected"):
        return None
    # Rebuild the two slot arrays from the delivered serialized row.  The
    # replacement result is the only newly triangulated member; the other
    # result/2-D observations remain byte-for-byte tied to the delivered row.
    results = [
        {
            "points": np.asarray(
                by_side.get(slot, {}).get("joints_base_m", []),
                dtype=np.float64,
            ),
            "inlier_counts": np.asarray(
                by_side.get(slot, {}).get("inlier_view_counts", []),
                dtype=np.int32,
            ),
            "valid_joints": int(by_side.get(slot, {}).get(
                "quality", {}
            ).get("valid_joint_count", 0)),
            "median_px": float(by_side.get(slot, {}).get(
                "quality", {}
            ).get("multiview_reprojection_median_px", np.inf)),
            "p95_px": float(by_side.get(slot, {}).get(
                "quality", {}
            ).get("multiview_reprojection_p95_px", np.inf)),
            "missing": slot not in by_side,
        }
        for slot in (0, 1)
    ]
    selected: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    for slot in (0, 1):
        selected[slot] = {
            str(camera): dict(view)
            for camera, view in by_side[slot].get("views", {}).items()
            if isinstance(view, dict)
        }
    old_selected = dict(selected[side])
    # Carry depth metadata from the untouched serialized result so the
    # branch-quality and transition costs remain comparable to normal states.
    for slot in (0, 1):
        quality = by_side[slot].get("quality", {})
        for key in (
            "ray_angle_p25_deg", "palm_ray_angle_p25_deg",
            "palm_three_view_count", "palm_two_view_fraction",
            "two_view_joint_fraction", "pair_spread_p75_m",
            "palm_pair_spread_p75_m",
        ):
            if key in quality:
                results[slot][key] = quality[key]
    results[side] = candidate["result"]
    selected[side] = {
        str(camera): dict(view)
        for camera, view in candidate.get("selected", {}).items()
        if isinstance(view, dict)
    }
    # ``results[side]`` was just replaced; recover the old values from the
    # serialized hand for the delta calculation.
    old_points = np.asarray(replaced_hand.get("joints_base_m", []), dtype=np.float64)
    old_counts = np.asarray(replaced_hand.get("inlier_view_counts", []), dtype=np.int32)
    old_quality = replaced_hand.get("quality", {})
    old_result_for_cost = {
        "points": old_points,
        "inlier_counts": old_counts,
        "valid_joints": int(old_quality.get("valid_joint_count", 0)),
        "median_px": float(old_quality.get("multiview_reprojection_median_px", np.inf)),
        "p95_px": float(old_quality.get("multiview_reprojection_p95_px", np.inf)),
        **{
            key: old_quality[key]
            for key in (
                "ray_angle_p25_deg", "palm_ray_angle_p25_deg",
                "palm_three_view_count", "palm_two_view_fraction",
                "two_view_joint_fraction", "pair_spread_p75_m",
                "palm_pair_spread_p75_m",
            )
            if key in old_quality
        },
    }
    original_cost = float(original.get("cost", 0.0))
    old_side_cost = _independent_side_cost(
        old_result_for_cost, old_selected, args
    )
    new_side_cost = _independent_side_cost(
        candidate["result"], selected[side], args
    )
    hypothesis = {
        "cost": float(original_cost + new_side_cost - old_side_cost),
        "anchor_cameras": list(original.get("anchor_cameras", [])),
        "anchor_baseline_m": float(original.get("anchor_baseline_m", 0.0) or 0.0),
        "results": results,
        "selected": selected,
        "association_errors": {
            0: {}, 1: {},
        },
        "_hybrid": True,
        "_hybrid_replacement_side": side,
        "_hybrid_candidate": candidate,
        "_hybrid_original_hand": replaced_hand,
        "_hybrid_other_hand": other_hand,
    }
    hypothesis["association_errors"][side] = candidate.get(
        "association_errors", {}
    )
    return hypothesis


def _independent_frame_hypotheses(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
    original_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate full and partial alternatives for the unified track pass."""
    candidates = _independent_frame_candidates(
        sync_index, prediction_rows, calibrations, cameras,
        preferred_anchors, args,
    )
    if not candidates:
        return []
    hypotheses: list[dict[str, Any]] = []
    # Singles are retained explicitly; a hand disappearing must not force a
    # synthetic second slot into the track.
    for candidate in candidates:
        slot = int(candidate.get("hypothesis_side", candidate.get("side", 0)))
        if slot not in (0, 1):
            slot = 0
        hypothesis = _independent_candidate_hypothesis({slot: candidate})
        hypothesis["_partial_candidate"] = dict(candidate)
        hypotheses.append(hypothesis)
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            oriented, _penalty = _independent_pair_orientation(first, second)
            near_duplicate, _near_duplicate_diagnostics = (
                _near_coincident_independent_pair_is_unsafe(first, second, args)
            )
            if near_duplicate:
                continue
            duplicate_unsafe, duplicate_views = _duplicate_assignment_is_unsafe(
                {
                    slot: candidate.get("selected", {})
                    for slot, candidate in oriented.items()
                },
                None,
                float(getattr(args, "duplicate_bbox_iou", 0.65)),
                float(getattr(args, "duplicate_joint_distance_px", 18.0)),
                int(getattr(args, "duplicate_overlap_cameras", 2)),
            )
            if duplicate_unsafe:
                continue
            pair_allowed, pair_reason, first_quality, second_quality = (
                _independent_pair_decision(first, second, args)
            )
            if not pair_allowed:
                continue
            hypothesis = _independent_candidate_hypothesis(oriented)
            hypothesis["duplicate_observation_view_count"] = int(duplicate_views)
            hypothesis["independent_pair_reason"] = pair_reason
            hypothesis["independent_pair_quality"] = [
                first_quality, second_quality,
            ]
            hypotheses.append(hypothesis)
    # Add hybrid states when a delivered complete row is available.  These
    # states deliberately keep the other hand's geometry and detector tracks
    # unchanged, allowing a weak/depth-ambiguous side to switch branch without
    # making the good side disappear or changing the row cardinality.
    if original_row is not None and len(original_row.get("hands", [])) == 2:
        original = _row_to_hypothesis(original_row)
        if original is not None:
            original_hands = {
                int(hand.get("side", -1)): hand
                for hand in original_row.get("hands", [])
                if int(hand.get("side", -1)) in (0, 1)
            }
            for side in (0, 1):
                if side not in original_hands or (1 - side) not in original_hands:
                    continue
                for candidate in candidates:
                    quality = _independent_candidate_quality(candidate, args)
                    # Unsafe depth is never promoted.  A weak-but-not-unsafe
                    # branch is intentionally retained here; the later track
                    # gate requires strong 2-D continuity and a bidirectional
                    # 3-D improvement before it can be applied.
                    if not quality.get("single_safe"):
                        continue
                    hybrid = _independent_hybrid_hypothesis(
                        original, original_row, side, candidate,
                        args,
                    )
                    if hybrid is not None:
                        hypotheses.append(hybrid)
    # Deduplicate branches with identical 3-D centers and camera selections.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for hypothesis in hypotheses:
        centres = tuple(
            tuple(np.round(_hypothesis_center(result), 4))
            if np.all(np.isfinite(_hypothesis_center(result))) else None
            for result in hypothesis["results"]
        )
        key = (
            tuple(hypothesis.get("anchor_cameras", [])),
            centres,
            tuple(
                tuple(sorted((str(camera), int(hand.get("detection_index", -1)))
                             for camera, hand in hypothesis.get("selected", {}).get(slot, {}).items()))
                for slot in (0, 1)
            ),
        )
        previous = unique.get(key)
        if previous is None or float(hypothesis["cost"]) < float(previous["cost"]):
            unique[key] = hypothesis
    values = list(unique.values())
    values.sort(key=lambda item: float(item.get("cost", np.inf)))
    return values[:max(int(getattr(args, "continuity_max_hypotheses", 32)), 1)]


def _hand_palm_image_center(hand: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return per-camera palm centres used for the low-support depth test."""
    result: dict[str, np.ndarray] = {}
    for camera, view in hand.get("views", {}).items():
        points = np.asarray(view.get("joints_2d", []), dtype=np.float64)
        if points.shape != (21, 2):
            continue
        palm = points[[0, 5, 9, 13, 17]]
        finite = np.isfinite(palm).all(axis=1)
        if np.count_nonzero(finite) >= 3:
            result[str(camera)] = np.median(palm[finite], axis=0)
    return result


def _hand_support_summary(hand: dict[str, Any]) -> tuple[int, float, int]:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
    finite = (
        np.isfinite(points).all(axis=1)
        if points.shape == (21, 3) else np.zeros(21, dtype=bool)
    )
    values = support[finite] if support.shape == (21,) else np.empty(0)
    median = float(np.median(values)) if values.size else 0.0
    return int(np.count_nonzero(finite)), median, len(hand.get("camera_ids", []))


def _depth_reference_hand(
    ordered: list[dict[str, Any]],
    position: int,
    side: int,
    direction: int,
    radius: int,
    segment_id: int | None = None,
) -> tuple[dict[str, Any], int] | tuple[None, None]:
    """Find a reliable same-side temporal reference for depth regularization."""
    frame = int(ordered[position]["sync_index"])
    cursor = position + direction
    while 0 <= cursor < len(ordered):
        candidate_frame = int(ordered[cursor]["sync_index"])
        gap = abs(candidate_frame - frame)
        if gap > radius:
            break
        for hand in ordered[cursor].get("hands", []):
            if int(hand.get("side", -1)) != side:
                continue
            if segment_id is not None:
                try:
                    if int(hand.get("temporal_segment_id", 0)) != int(segment_id):
                        continue
                except (TypeError, ValueError):
                    continue
            points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            if points.shape != (21, 3):
                continue
            finite = np.isfinite(points).all(axis=1)
            if not np.all(np.isfinite(np.nanmedian(points[[0, 5, 9, 13, 17]], axis=0))):
                continue
            valid_count, median_support, camera_count = _hand_support_summary(hand)
            support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.float64)
            points_finite = np.isfinite(points).all(axis=1)
            palm = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
            reliable_palm = (
                points_finite[palm]
                & (support.shape == (21,))
                & np.isfinite(support[palm])
                & (support[palm] >= 2)
            )
            # A three-camera pose can legitimately have a median support of
            # two when one or two fingers are occluded.  For a temporal depth
            # reference, reliable palm anchors and the camera count matter
            # more than the all-joint median.
            if valid_count < 15 or camera_count < 3 or np.count_nonzero(reliable_palm) < 3:
                continue
            # Only well-conditioned depth may teach the temporal prior.  A
            # branch can be physically compact and have low reprojection while
            # its pair intersections disagree by centimetres; using it as a
            # reference propagates the wrong Z into every neighbouring frame.
            quality = hand.get("quality", {})
            depth_level, _depth_reasons = _depth_condition_level(
                quality, camera_count
            )
            if depth_level != "normal":
                continue
            return hand, gap
        cursor += direction
    return None, None


def _shared_image_motion(
    current: dict[str, Any],
    previous: dict[str, Any],
    following: dict[str, Any],
    gap_before: int,
    gap_after: int,
) -> tuple[float, float] | None:
    """Measure normalized 2-D palm motion around a candidate frame."""
    current_centres = _hand_palm_image_center(current)
    previous_centres = _hand_palm_image_center(previous)
    following_centres = _hand_palm_image_center(following)
    before_values = []
    after_values = []
    for camera in set(current_centres) & set(previous_centres):
        before_values.append(float(
            np.linalg.norm(current_centres[camera] - previous_centres[camera])
        ) / max(int(gap_before), 1))
    for camera in set(current_centres) & set(following_centres):
        after_values.append(float(
            np.linalg.norm(current_centres[camera] - following_centres[camera])
        ) / max(int(gap_after), 1))
    if not before_values or not after_values:
        return None
    return float(np.median(before_values)), float(np.median(after_values))


def _temporal_depth_guard_hand(
    row: Mapping[str, Any], side: int,
) -> dict[str, Any] | None:
    """Return a finite serialized hand for ``side`` when one is present."""
    for hand in row.get("hands", []):
        try:
            hand_side = int(hand.get("side", -1))
        except (TypeError, ValueError):
            continue
        if hand_side != int(side):
            continue
        if np.all(np.isfinite(_hand_center_from_row(dict(hand)))):
            return hand
    return None


def _temporal_depth_guard_neighbour(
    ordered: list[dict[str, Any]],
    position: int,
    side: int,
    direction: int,
    radius: int,
    segment_id: Any,
    excluded: set[int] | None = None,
    quality_args: argparse.Namespace | None = None,
) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    """Find the nearest same-segment hand around a depth-guard frame."""
    frame = int(ordered[position].get("sync_index", 0))
    cursor = int(position) + (1 if direction > 0 else -1)
    while 0 <= cursor < len(ordered):
        if excluded and cursor in excluded:
            cursor += 1 if direction > 0 else -1
            continue
        candidate_frame = int(ordered[cursor].get("sync_index", 0))
        if abs(candidate_frame - frame) > int(radius):
            break
        hand = _temporal_depth_guard_hand(ordered[cursor], side)
        if hand is not None:
            if segment_id is not None:
                try:
                    if int(hand.get("temporal_segment_id", 0)) != int(segment_id):
                        cursor += 1 if direction > 0 else -1
                        continue
                except (TypeError, ValueError):
                    cursor += 1 if direction > 0 else -1
                    continue
            # For a repair bracket, do not stop at the first geometrically
            # finite hand if that hand is itself a weak stereo branch.  Walk
            # farther within the bounded radius to find a stable reference.
            # Detection/suspicion callers leave this unset so they can still
            # report a local contradiction around an under-conditioned frame.
            if quality_args is not None and not _temporal_depth_guard_quality_ok(
                hand, quality_args
            ):
                cursor += 1 if direction > 0 else -1
                continue
            return cursor, hand
        cursor += 1 if direction > 0 else -1
    return None, None


def _temporal_depth_guard_quality_ok(
    hand: Mapping[str, Any], args: argparse.Namespace,
) -> bool:
    """Whether a hand is safe to use as a temporal depth reference."""
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3):
        return False
    finite = np.isfinite(points).all(axis=1)
    palm = np.asarray((0, 5, 9, 13, 17), dtype=np.int64)
    if int(np.count_nonzero(finite)) < 15 or int(np.count_nonzero(finite[palm])) < 3:
        return False
    camera_count = len(hand.get("views", {}))
    if camera_count < 2:
        return False
    quality = hand.get("quality", {})
    try:
        level, reasons = _depth_condition_level(quality, camera_count, args)
    except (TypeError, ValueError, KeyError):
        return False
    if level != "normal" or any(
        reason in {"depth_pair_disagreement", "weak_depth_pair_spread"}
        for reason in reasons
    ):
        return False
    # A well-conditioned two-view pair is a valid temporal anchor when its
    # baseline angle and reprojection support are strong.  Requiring three
    # cameras here discarded useful references at exactly the partial/full
    # boundaries where the depth branch is most likely to flip.
    if camera_count == 2:
        try:
            angle = float(quality.get("palm_ray_angle_p25_deg", np.nan))
            median = float(quality.get(
                "multiview_reprojection_median_px", np.inf
            ))
        except (TypeError, ValueError):
            return False
        preferred = max(float(getattr(
            args, "secondary_hand_min_palm_angle_deg", 7.0
        )), 8.0)
        max_median = float(getattr(
            args, "temporal_depth_guard_max_reference_reprojection_px", 15.0
        ))
        if (
            not np.isfinite(angle)
            or angle < preferred
            or not np.isfinite(median)
            or median > max_median
        ):
            return False
    return True


def _temporal_depth_guard_suspicion(
    ordered: list[dict[str, Any]],
    position: int,
    side: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Identify an isolated 3-D excursion with a continuous image track.

    The test is intentionally based on a zero-phase bracket.  A genuinely
    fast gesture has a corresponding image move (or a large bracket speed),
    whereas a wrong stereo-depth branch keeps the detector pixels nearly
    fixed and appears as a V-shaped excursion in base coordinates.
    """
    current = _temporal_depth_guard_hand(ordered[position], side)
    if current is None or current.get("temporal_boundary_before"):
        return None
    radius = max(int(getattr(args, "temporal_depth_guard_radius", 8)), 1)
    segment_id = current.get("temporal_segment_id")
    before_index, before = _temporal_depth_guard_neighbour(
        ordered, position, side, -1, radius, segment_id
    )
    after_index, after = _temporal_depth_guard_neighbour(
        ordered, position, side, 1, radius, segment_id
    )
    if before is None or after is None:
        return None
    gap_before = int(ordered[position]["sync_index"]) - int(
        ordered[before_index]["sync_index"]
    )
    gap_after = int(ordered[after_index]["sync_index"]) - int(
        ordered[position]["sync_index"]
    )
    if gap_before < 1 or gap_after < 1:
        return None
    image_before = _hand_image_track_motion(before, current, gap_before)
    image_after = _hand_image_track_motion(current, after, gap_after)
    minimum_shared = max(int(getattr(
        args, "temporal_depth_guard_min_shared_cameras", 2
    )), 1)
    if (
        not image_before.get("available")
        or not image_after.get("available")
        or int(image_before.get("shared_camera_count", 0)) < minimum_shared
        or int(image_after.get("shared_camera_count", 0)) < minimum_shared
    ):
        return None
    max_image = max(float(image_before["robust_px_per_frame"]),
                    float(image_after["robust_px_per_frame"]))
    if max_image > max(float(getattr(
        args, "temporal_depth_guard_max_2d_motion_px", 75.0
    )), 0.0):
        return None
    current_center = _hand_center_from_row(current)
    before_center = _hand_center_from_row(before)
    after_center = _hand_center_from_row(after)
    if not (
        np.all(np.isfinite(current_center))
        and np.all(np.isfinite(before_center))
        and np.all(np.isfinite(after_center))
    ):
        return None
    alpha = gap_before / float(gap_before + gap_after)
    predicted_center = (1.0 - alpha) * before_center + alpha * after_center
    residual = float(np.linalg.norm(current_center - predicted_center))
    edge_before = float(np.linalg.norm(current_center - before_center)) / gap_before
    edge_after = float(np.linalg.norm(after_center - current_center)) / gap_after
    bracket_speed = float(np.linalg.norm(after_center - before_center)) / (
        gap_before + gap_after
    )
    trigger = max(float(getattr(
        args, "temporal_depth_guard_trigger_m", 0.055
    )), 0.0)
    max_step = max(float(getattr(
        args, "temporal_depth_guard_max_step_m", 0.115
    )), 0.0)
    max_bracket_speed = max(float(getattr(
        args, "temporal_depth_guard_max_bracket_speed_m", 0.120
    )), 0.0)
    if (
        residual <= trigger
        or bracket_speed > max_bracket_speed
        or max(edge_before, edge_after) <= max_step
    ):
        return None
    return {
        "position": int(position),
        "side": int(side),
        "before_index": int(before_index),
        "after_index": int(after_index),
        "before": before,
        "after": after,
        "current": current,
        "gap_before": int(gap_before),
        "gap_after": int(gap_after),
        "predicted_center": predicted_center,
        "current_center": current_center,
        "residual_m": residual,
        "edge_before_m": edge_before,
        "edge_after_m": edge_after,
        "bracket_speed_m_per_frame": bracket_speed,
        "image_motion_px_per_frame": [
            float(image_before["robust_px_per_frame"]),
            float(image_after["robust_px_per_frame"]),
        ],
        "shared_camera_count": [
            int(image_before["shared_camera_count"]),
            int(image_after["shared_camera_count"]),
        ],
    }


def _temporal_depth_guard_reprojection_stats(
    hand: Mapping[str, Any],
    points: np.ndarray,
    calibrations: Mapping[str, CameraCalibration],
) -> dict[str, Any]:
    """Collect robust selected-view reprojection statistics for a proposal."""
    values: list[float] = []
    per_camera: dict[str, float] = {}
    for camera_id, view in hand.get("views", {}).items():
        calibration = calibrations.get(camera_id)
        observed = np.asarray(view.get("joints_2d", []), dtype=np.float64)
        if calibration is None or observed.shape != (21, 2):
            continue
        errors: list[float] = []
        for joint in range(21):
            if not (
                np.all(np.isfinite(points[joint]))
                and np.all(np.isfinite(observed[joint]))
            ):
                continue
            error = _errors(
                points[joint], [(camera_id, calibration, observed[joint])]
            )[0]
            if np.isfinite(error):
                errors.append(float(error))
                values.append(float(error))
        if errors:
            per_camera[str(camera_id)] = float(np.median(errors))
    array = np.asarray(values, dtype=np.float64)
    return {
        "finite_count": int(array.size),
        "median_px": float(np.median(array)) if array.size else float("inf"),
        "p95_px": float(np.percentile(array, 95)) if array.size else float("inf"),
        "per_camera_median_px": per_camera,
    }


def _apply_temporal_depth_guard(
    accepted: list[dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> int:
    """Repair short depth-branch runs while preserving articulated shape.

    A detector can keep the same 2-D track while stereo selects a different
    depth solution.  This pass detects short V-shaped runs, obtains stable
    references outside the run, and regularizes each joint along its observed
    camera rays toward the interpolated pose.  It never smooths a large image
    motion and accepts a proposal only when reprojection and local trajectory
    checks both pass.
    """
    if not getattr(args, "temporal_depth_guard", 1) or len(accepted) < 3:
        return 0
    ordered = sorted(accepted, key=lambda row: int(row["sync_index"]))
    radius = max(int(getattr(args, "temporal_depth_guard_radius", 8)), 1)
    max_run = max(int(getattr(
        args, "temporal_depth_guard_max_run_length", 3
    )), 1)
    suspects: dict[tuple[int, int], dict[str, Any]] = {}
    for position in range(len(ordered)):
        for side in (0, 1):
            diagnostic = _temporal_depth_guard_suspicion(
                ordered, position, side, args
            )
            if diagnostic is not None:
                suspects[(position, side)] = diagnostic
    if not suspects:
        return 0

    updates: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any]]] = []
    for side in (0, 1):
        positions = sorted(
            position for position, value_side in suspects if value_side == side
        )
        runs: list[list[int]] = []
        for position in positions:
            if (
                not runs
                or int(ordered[position]["sync_index"])
                - int(ordered[runs[-1][-1]]["sync_index"])
                > max(2, max_run)
            ):
                runs.append([position])
            elif len(runs[-1]) < max_run:
                runs[-1].append(position)
        for run in runs:
            if not run or len(run) > max_run:
                continue
            first_position = run[0]
            last_position = run[-1]
            segment_id = _temporal_depth_guard_hand(
                ordered[first_position], side
            ).get("temporal_segment_id")
            excluded = set(run)
            before_index, before = _temporal_depth_guard_neighbour(
                ordered, first_position, side, -1, radius, segment_id, excluded,
                args,
            )
            after_index, after = _temporal_depth_guard_neighbour(
                ordered, last_position, side, 1, radius, segment_id, excluded,
                args,
            )
            if before is None or after is None:
                continue
            if not _temporal_depth_guard_quality_ok(before, args) or not _temporal_depth_guard_quality_ok(after, args):
                continue
            before_frame = int(ordered[before_index]["sync_index"])
            after_frame = int(ordered[after_index]["sync_index"])
            if after_frame <= before_frame:
                continue
            before_points = np.asarray(before["joints_base_m"], dtype=np.float64)
            after_points = np.asarray(after["joints_base_m"], dtype=np.float64)
            before_center = _hand_center_from_row(before)
            after_center = _hand_center_from_row(after)
            if not (
                before_points.shape == (21, 3)
                and after_points.shape == (21, 3)
                and np.all(np.isfinite(before_center))
                and np.all(np.isfinite(after_center))
            ):
                continue
            for position in run:
                current = _temporal_depth_guard_hand(ordered[position], side)
                if current is None:
                    continue
                points = np.asarray(current["joints_base_m"], dtype=np.float64)
                if points.shape != (21, 3):
                    continue
                frame = int(ordered[position]["sync_index"])
                alpha = (frame - before_frame) / float(after_frame - before_frame)
                predicted_points = (1.0 - alpha) * before_points + alpha * after_points
                predicted_center = (1.0 - alpha) * before_center + alpha * after_center
                current_center = _hand_center_from_row(current)
                if not np.all(np.isfinite(current_center)):
                    continue
                residual_before = float(np.linalg.norm(current_center - predicted_center))
                if residual_before <= max(float(getattr(
                    args, "temporal_depth_guard_trigger_m", 0.055
                )), 0.0):
                    continue

                deltas: list[tuple[int, np.ndarray]] = []
                for joint in range(21):
                    if not (
                        np.all(np.isfinite(points[joint]))
                        and np.all(np.isfinite(predicted_points[joint]))
                    ):
                        continue
                    rays: list[tuple[np.ndarray, np.ndarray]] = []
                    for camera_id, view in current.get("views", {}).items():
                        calibration = calibrations.get(camera_id)
                        pixels = np.asarray(view.get("joints_2d", []), dtype=np.float64)
                        if (
                            calibration is None
                            or pixels.shape != (21, 2)
                            or not np.all(np.isfinite(pixels[joint]))
                        ):
                            continue
                        ray = _ray_in_base(calibration, pixels[joint])
                        if ray is not None:
                            rays.append(ray)
                    if len(rays) < 2:
                        continue
                    weight = 0.03 if len(rays) <= 3 else 0.10
                    regularized = regularized_ray_intersection(
                        rays, predicted_points[joint], weight
                    )
                    if regularized is None or not np.all(np.isfinite(regularized)):
                        continue
                    deltas.append((joint, regularized - points[joint]))
                if len(deltas) < 3:
                    continue
                delta_array = np.asarray([delta for _joint, delta in deltas])
                median_delta = np.median(delta_array, axis=0)
                deviation = np.linalg.norm(delta_array - median_delta, axis=1)
                # A few occluded fingers can have a different ray solution;
                # keep the coherent hand translation for those joints instead
                # of creating a disconnected skeleton.
                coherent_limit = max(0.060, 3.0 * float(np.median(deviation)))
                robust_deltas = {
                    joint: (delta if float(np.linalg.norm(delta - median_delta)) <= coherent_limit else median_delta)
                    for (joint, delta), value in zip(deltas, deviation)
                }
                finite_joints = np.flatnonzero(np.isfinite(points).all(axis=1))
                max_shift = max(float(getattr(
                    args, "temporal_depth_guard_max_shift_m", 0.160
                )), 0.0)
                raw_stats = _temporal_depth_guard_reprojection_stats(
                    current, points, calibrations
                )
                raw_edges = np.asarray((
                    np.linalg.norm(current_center - before_center) / max(frame - before_frame, 1),
                    np.linalg.norm(after_center - current_center) / max(after_frame - frame, 1),
                ), dtype=np.float64)
                fractions = [
                    min(max(float(getattr(args, "depth_regularization_blend", 0.85)), 0.0), 1.0),
                    0.70, 0.55, 0.40, 0.30,
                ]
                fractions = sorted({value for value in fractions if value > 0.0}, reverse=True)
                selected_points = None
                selected_diag: dict[str, Any] | None = None
                for fraction in fractions:
                    proposal = points.copy()
                    for joint in finite_joints:
                        delta = robust_deltas.get(int(joint), median_delta)
                        shift = fraction * np.asarray(delta, dtype=np.float64)
                        shift_norm = float(np.linalg.norm(shift))
                        if shift_norm > max_shift and shift_norm > 1e-12:
                            shift *= max_shift / shift_norm
                        proposal[joint] += shift
                    proposal_center = _hand_center_from_row({
                        "joints_base_m": proposal.tolist()
                    })
                    if not np.all(np.isfinite(proposal_center)):
                        continue
                    residual_after = float(np.linalg.norm(proposal_center - predicted_center))
                    improvement = residual_before - residual_after
                    if improvement < 0.020:
                        continue
                    candidate_edges = np.asarray((
                        np.linalg.norm(proposal_center - before_center) / max(frame - before_frame, 1),
                        np.linalg.norm(after_center - proposal_center) / max(after_frame - frame, 1),
                    ), dtype=np.float64)
                    if (
                        not np.all(np.isfinite(candidate_edges))
                        or np.max(candidate_edges) > np.max(raw_edges) + 0.012
                        or np.sum(candidate_edges ** 2) > np.sum(raw_edges ** 2) + 0.012 ** 2
                    ):
                        continue
                    proposal_stats = _temporal_depth_guard_reprojection_stats(
                        current, proposal, calibrations
                    )
                    if proposal_stats["finite_count"] < max(3, int(0.80 * raw_stats["finite_count"])):
                        continue
                    median_regression = proposal_stats["median_px"] - raw_stats["median_px"]
                    p95_regression = proposal_stats["p95_px"] - raw_stats["p95_px"]
                    if (
                        median_regression > float(getattr(
                            args, "temporal_depth_guard_max_reprojection_regression_px", 12.0
                        ))
                        or p95_regression > float(getattr(
                            args, "temporal_depth_guard_max_reprojection_p95_regression_px", 24.0
                        ))
                    ):
                        continue
                    selected_points = proposal
                    selected_diag = {
                        "fraction": float(fraction),
                        "residual_before_m": residual_before,
                        "residual_after_m": residual_after,
                        "improvement_m": improvement,
                        "raw_reprojection": raw_stats,
                        "proposal_reprojection": proposal_stats,
                        "median_reprojection_regression_px": median_regression,
                        "p95_reprojection_regression_px": p95_regression,
                        "corrected_joint_count": int(len(robust_deltas)),
                        "max_shift_m": float(np.max([
                            np.linalg.norm(proposal[joint] - points[joint])
                            for joint in finite_joints
                        ])) if len(finite_joints) else 0.0,
                    }
                    break
                if selected_points is None or selected_diag is None:
                    continue
                selected_diag.update({
                    "algorithm": "temporal_depth_branch_guard_v1",
                    "before_sync_index": before_frame,
                    "after_sync_index": after_frame,
                    "run_sync_indices": [int(ordered[value]["sync_index"]) for value in run],
                    "image_motion_px_per_frame": suspects[(position, side)]["image_motion_px_per_frame"],
                    "shared_camera_count": suspects[(position, side)]["shared_camera_count"],
                    "bracket_speed_m_per_frame": suspects[(position, side)]["bracket_speed_m_per_frame"],
                })
                updates.append((ordered[position], current, selected_points, selected_diag))

    changed = 0
    for row, hand, points, diagnostics in updates:
        hand["pre_temporal_depth_guard_joints_base_m"] = hand.get("joints_base_m")
        hand["joints_base_m"] = points.tolist()
        hand["temporal_depth_guard"] = diagnostics
        row.setdefault("temporal_depth_guard", {
            "algorithm": "temporal_depth_branch_guard_v1",
            "corrected_hand_count": 0,
        })
        row["temporal_depth_guard"]["corrected_hand_count"] = int(
            row["temporal_depth_guard"].get("corrected_hand_count", 0) + 1
        )
        changed += 1
    return changed


def _apply_low_support_depth_regularization(
    accepted: list[dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> int:
    """Correct weak stereo depth branches using a bounded temporal prior.

    A two-view intersection can move tens of centimetres in depth while its
    image residual remains small.  This pass acts only on a low-support frame
    whose 2-D palm motion is small and whose 3-D palm centre is an isolated
    excursion between reliable observations.  The ray equations remain in
    the objective, so the result is a depth regularization—not a blind video
    smoother—and every changed point is checked against its original pixels.
    """
    if not getattr(args, "depth_regularization", 1) or len(accepted) < 3:
        return 0
    radius = max(int(getattr(args, "depth_regularization_radius", 8)), 1)
    trigger = max(float(getattr(args, "depth_regularization_trigger_m", 0.055)), 0.0)
    max_2d = max(float(getattr(args, "depth_regularization_max_2d_motion_px", 45.0)), 0.0)
    blend = min(max(float(getattr(args, "depth_regularization_blend", 0.85)), 0.0), 1.0)
    ordered = sorted(accepted, key=lambda row: int(row["sync_index"]))
    updates: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any]]] = []
    for position, row in enumerate(ordered):
        for side in (0, 1):
            current = next(
                (hand for hand in row.get("hands", [])
                 if int(hand.get("side", -1)) == side),
                None,
            )
            if current is None:
                continue
            if current.get("temporal_boundary_before"):
                # A re-entry is a new image track.  Do not use the last hand
                # before the hard 2-D break as a depth prior for its first
                # sample, even if the geometric centers happen to be close.
                continue
            points = np.asarray(current.get("joints_base_m", []), dtype=np.float64)
            if points.shape != (21, 3):
                continue
            valid_count, median_support, camera_count = _hand_support_summary(current)
            weak = camera_count <= 2 or median_support < 2.75
            if not weak or valid_count < 12:
                continue
            previous, gap_before = _depth_reference_hand(
                ordered, position, side, -1, radius,
                current.get("temporal_segment_id"),
            )
            following, gap_after = _depth_reference_hand(
                ordered, position, side, 1, radius,
                current.get("temporal_segment_id"),
            )
            if previous is None or following is None:
                continue
            previous_points = np.asarray(previous.get("joints_base_m", []), dtype=np.float64)
            following_points = np.asarray(following.get("joints_base_m", []), dtype=np.float64)
            if previous_points.shape != (21, 3) or following_points.shape != (21, 3):
                continue
            current_palm = points[[0, 5, 9, 13, 17]]
            previous_palm = previous_points[[0, 5, 9, 13, 17]]
            following_palm = following_points[[0, 5, 9, 13, 17]]
            finite_palm = (
                np.isfinite(current_palm).all(axis=1)
                & np.isfinite(previous_palm).all(axis=1)
                & np.isfinite(following_palm).all(axis=1)
            )
            if np.count_nonzero(finite_palm) < 3:
                continue
            alpha = gap_before / float(max(gap_before + gap_after, 1))
            current_center = np.median(current_palm[finite_palm], axis=0)
            previous_center = np.median(previous_palm[finite_palm], axis=0)
            following_center = np.median(following_palm[finite_palm], axis=0)
            predicted_center = (1.0 - alpha) * previous_center + alpha * following_center
            residual = float(np.linalg.norm(current_center - predicted_center))
            bracket_speed = float(np.linalg.norm(following_center - previous_center)) / max(
                gap_before + gap_after, 1
            )
            # Do not touch a genuinely fast motion: its bracket itself must be
            # moderate, and the image evidence must not show a large move.
            if residual <= trigger or bracket_speed > 0.11:
                continue
            image_motion = _shared_image_motion(
                current, previous, following, gap_before, gap_after
            )
            if image_motion is None or max(image_motion) > max_2d:
                continue
            selected_views = current.get("views", {})
            candidate_points = points.copy()
            changed_mask = np.zeros(21, dtype=bool)
            projected_before = []
            projected_after = []
            for joint in range(21):
                if not (
                    np.all(np.isfinite(points[joint]))
                    and np.all(np.isfinite(previous_points[joint]))
                    and np.all(np.isfinite(following_points[joint]))
                ):
                    continue
                prior = (1.0 - alpha) * previous_points[joint] + alpha * following_points[joint]
                rays = []
                observations = []
                for camera_id, view in selected_views.items():
                    calibration = calibrations.get(camera_id)
                    pixel = np.asarray(view.get("joints_2d", [])[joint], dtype=np.float64)
                    if calibration is None or pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
                        continue
                    ray = _ray_in_base(calibration, pixel)
                    if ray is None:
                        continue
                    rays.append(ray)
                    observations.append((camera_id, calibration, pixel))
                if len(rays) < 2:
                    continue
                # Stronger regularization for a two-view branch; with three or
                # more rays the observations already constrain depth better.
                regularization_weight = 4.0 if len(rays) == 2 else 1.8
                regularized = regularized_ray_intersection(
                    rays, prior, regularization_weight
                )
                if regularized is None or not np.all(np.isfinite(regularized)):
                    continue
                delta = regularized - points[joint]
                norm = float(np.linalg.norm(delta))
                if norm <= 0.005:
                    continue
                if norm > 0.18:
                    delta *= 0.18 / norm
                proposal = points[joint] + blend * delta
                before_errors = _errors(points[joint], observations)
                after_errors = _errors(proposal, observations)
                finite_before = before_errors[np.isfinite(before_errors)]
                finite_after = after_errors[np.isfinite(after_errors)]
                if not len(finite_before) or not len(finite_after):
                    continue
                # A depth correction may slightly worsen selected-view pixels,
                # but never by more than a small bounded amount.
                if float(np.median(finite_after)) > float(np.median(finite_before)) + 8.0:
                    continue
                candidate_points[joint] = proposal
                changed_mask[joint] = True
                projected_before.extend(finite_before.tolist())
                projected_after.extend(finite_after.tolist())
            if np.count_nonzero(changed_mask) < 3:
                continue
            diagnostics = {
                "algorithm": "low_support_temporal_ray_regularization_v1",
                "previous_sync_index": int(ordered[position - 1]["sync_index"])
                if position > 0 else None,
                "following_sync_index": int(ordered[position + 1]["sync_index"])
                if position + 1 < len(ordered) else None,
                "residual_m": residual,
                "bracket_speed_m_per_frame": bracket_speed,
                "image_motion_px_per_frame": list(image_motion),
                "median_support": median_support,
                "camera_count": camera_count,
                "corrected_joint_count": int(np.count_nonzero(changed_mask)),
                "max_adjustment_m": float(np.max(np.linalg.norm(
                    candidate_points[changed_mask] - points[changed_mask], axis=1
                ))),
            }
            updates.append((row, current, candidate_points, diagnostics))
    changed = 0
    for row, hand, candidate_points, diagnostics in updates:
        original = hand.get("joints_base_m")
        hand["pre_depth_regularization_joints_base_m"] = original
        hand["joints_base_m"] = candidate_points.tolist()
        hand["depth_regularization"] = diagnostics
        row.setdefault("depth_regularization", {
            "algorithm": "low_support_temporal_ray_regularization_v1",
            "corrected_hand_count": 0,
        })
        row["depth_regularization"]["corrected_hand_count"] = int(
            row["depth_regularization"].get("corrected_hand_count", 0) + 1
        )
        changed += 1
    return changed


def _hand_center_from_row(hand: dict[str, Any]) -> np.ndarray:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3):
        return np.full(3, np.nan)
    palm = points[[0, 5, 9, 13, 17]]
    finite = np.isfinite(palm).all(axis=1)
    return np.median(palm[finite], axis=0) if np.any(finite) else np.full(3, np.nan)


def _stabilize_partial_side_labels(
    rows: list[dict[str, Any]], max_step_m: float = 0.18,
) -> int:
    """Keep a one-hand fallback on the same physical side through detector flips.

    The two handedness hypotheses attached to one detector box have identical
    image joints, so a detector-side flip must not become a physical identity
    switch in the fused trajectory.  Process partial rows in time order and
    assign their sole hand to the side whose most recent 3-D palm center is
    closest.  Full dual-hand rows seed/update both tracks and are never edited.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    last_center: dict[int, np.ndarray] = {}
    last_frame: dict[int, int] = {}
    changed = 0
    for row in ordered:
        frame = int(row["sync_index"])
        hands = row.get("hands", [])
        if not row.get("partial_hand_acceptance") or len(hands) != 1:
            for hand in hands:
                side = int(hand["side"])
                center = _hand_center_from_row(hand)
                if np.all(np.isfinite(center)):
                    last_center[side] = center
                    last_frame[side] = frame
            continue
        hand = hands[0]
        center = _hand_center_from_row(hand)
        if not np.all(np.isfinite(center)):
            continue
        current_side = int(hand["side"])
        scores: dict[int, float] = {}
        for side in (0, 1):
            previous = last_center.get(side)
            if previous is None or not np.all(np.isfinite(previous)):
                scores[side] = float("inf")
                continue
            gap = max(frame - last_frame[side], 1)
            distance = float(np.linalg.norm(center - previous))
            # Penalize stale tracks and reject implausibly large one-frame
            # jumps unless the opposite track is unavailable as well.
            scores[side] = distance / gap + 0.001 * gap
        finite_scores = {side: value for side, value in scores.items() if np.isfinite(value)}
        if finite_scores:
            chosen_side = min(finite_scores, key=finite_scores.get)
            if (
                chosen_side != current_side
                and finite_scores[chosen_side] <= max_step_m
                and (
                    not np.isfinite(scores.get(current_side, float("inf")))
                    or finite_scores[chosen_side] + 0.01 < scores[current_side]
                )
            ):
                hand["side"] = chosen_side
                row["missing_sides"] = sorted({0, 1} - {chosen_side})
                changed += 1
                current_side = chosen_side
        last_center[current_side] = center
        last_frame[current_side] = frame
    return changed


def _hand_wrist_from_row(hand: dict[str, Any]) -> np.ndarray:
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points[0])):
        return np.full(3, np.nan)
    return points[0]


def _hand_shape_signature(hand: dict[str, Any]) -> np.ndarray:
    """Return a scale-normalized pose signature for identity matching."""
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3):
        return np.full((21, 3), np.nan)
    centre = _hand_center_from_row(hand)
    if not np.all(np.isfinite(centre)):
        return np.full((21, 3), np.nan)
    scale = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(scale) or scale < 1e-5:
        return np.full((21, 3), np.nan)
    signature = (points - centre) / scale
    signature[~np.isfinite(signature).all(axis=1)] = np.nan
    return signature


def _hand_detector_mode(hand: dict[str, Any]) -> int | None:
    values = []
    for view in hand.get("views", {}).values():
        value = view.get("detector_is_right")
        if value is not None:
            values.append(int(value))
    if not values:
        return None
    counts = Counter(values)
    # A tie is deliberately treated as unknown.  Detector handedness is a
    # weak prior and must never decide an identity on its own.
    if len(counts) > 1 and counts.most_common()[0][1] == counts.most_common()[1][1]:
        return None
    return int(counts.most_common(1)[0][0])


def _identity_transition_cost(
    state: dict[str, Any] | None,
    hand: dict[str, Any],
    frame: int,
    side: int,
    max_step_m: float,
) -> float:
    """Cost of assigning a detected hand to an established physical track."""
    centre = _hand_center_from_row(hand)
    wrist = _hand_wrist_from_row(hand)
    if not np.all(np.isfinite(centre)):
        return float("inf")
    if not state or not np.all(np.isfinite(state.get("center", np.nan))):
        # Prefer an established track over creating a new one, while keeping
        # the first dual-hand row deterministic through the side tie-break.
        return 0.50 + 0.05 * float(int(hand.get("side", side)) != side)
    last_frame = int(state.get("frame", frame))
    gap = max(frame - last_frame, 1)
    if gap > 30:
        return 0.50 + 0.05 * float(int(hand.get("side", side)) != side)
    velocity = np.asarray(state.get("velocity", np.zeros(3)), dtype=np.float64)
    if not np.all(np.isfinite(velocity)):
        velocity = np.zeros(3, dtype=np.float64)
    predicted = np.asarray(state["center"], dtype=np.float64) + velocity * gap
    centre_distance = float(np.linalg.norm(centre - predicted))
    # The denominator is intentionally generous for fast deliberate motion;
    # the identity decision is based on relative costs of the two tracks.
    cost = centre_distance / max(0.035, 0.06 * gap + 0.25 * max_step_m)

    previous_wrist = np.asarray(state.get("wrist", np.nan), dtype=np.float64)
    if np.all(np.isfinite(previous_wrist)) and np.all(np.isfinite(wrist)):
        wrist_velocity = np.asarray(
            state.get("wrist_velocity", np.zeros(3)), dtype=np.float64
        )
        if not np.all(np.isfinite(wrist_velocity)):
            wrist_velocity = np.zeros(3, dtype=np.float64)
        predicted_wrist = previous_wrist + wrist_velocity * gap
        wrist_distance = float(np.linalg.norm(wrist - predicted_wrist))
        cost += 0.65 * wrist_distance / max(0.045, 0.08 * gap)

    previous_signature = np.asarray(state.get("signature", np.nan), dtype=np.float64)
    signature = _hand_shape_signature(hand)
    if previous_signature.shape == (21, 3) and signature.shape == (21, 3):
        finite = np.isfinite(previous_signature).all(axis=1) & np.isfinite(signature).all(axis=1)
        if np.count_nonzero(finite) >= 8:
            shape_distance = float(np.median(
                np.linalg.norm(previous_signature[finite] - signature[finite], axis=1)
            ))
            cost += 0.35 * shape_distance

    detector = _hand_detector_mode(hand)
    previous_detector = state.get("detector_mode")
    if detector is not None and previous_detector is not None and detector != previous_detector:
        # Detector flips are common in side-on views; this small term can break
        # an exact geometric tie but cannot override a plausible trajectory.
        cost += 0.12
    return float(cost)


def _stabilize_hand_identities_greedy(
    rows: list[dict[str, Any]], max_step_m: float = 0.22,
) -> int:
    """Assign every fused hand to a persistent physical side track.

    WiLoR emits a left/right hypothesis for every detector box and different
    cameras may order boxes differently.  This pass treats detector
    handedness as a weak hint, enumerates both assignments for dual-hand rows,
    and follows the 3-D palm/wrist trajectory.  It also handles partial rows;
    a visible hand therefore keeps its physical ID while the other hand is out
    of view.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    tracks: dict[int, dict[str, Any] | None] = {0: None, 1: None}
    changed = 0
    for row in ordered:
        frame = int(row["sync_index"])
        original = list(row.get("hands", []))
        # Ignore all-NaN placeholders while retaining them for the later
        # anatomy/pruning diagnostics.
        usable = [
            hand for hand in original
            if np.all(np.isfinite(_hand_center_from_row(hand)))
        ]
        if not usable:
            if original:
                row["hands"] = []
                row["partial_hand_acceptance"] = True
                row["missing_sides"] = [0, 1]
            continue
        # A normal frame contains at most two physical hands.  If duplicate
        # hypotheses leak through, retain the two highest-confidence distinct
        # tracks and leave the rest untouched rather than assigning one track
        # twice.
        if len(usable) > 2:
            ranked = sorted(
                usable,
                key=lambda hand: float(hand.get("quality", {}).get(
                    "valid_joint_count", 0
                )),
                reverse=True,
            )
            usable = ranked[:2]

        assignments: dict[int, int] = {}
        assignment_cost = float("inf")
        if len(usable) >= 2:
            for permutation in itertools.permutations(range(len(usable)), 2):
                cost = 0.0
                valid = True
                for side, index in enumerate(permutation):
                    value = _identity_transition_cost(
                        tracks[side], usable[index], frame, side, max_step_m
                    )
                    if not np.isfinite(value):
                        valid = False
                        break
                    cost += value
                if valid and cost < assignment_cost:
                    assignment_cost = cost
                    assignments = {0: permutation[0], 1: permutation[1]}
        else:
            hand = usable[0]
            values = [
                _identity_transition_cost(
                    tracks[side], hand, frame, side, max_step_m
                )
                for side in (0, 1)
            ]
            finite = [side for side, value in enumerate(values) if np.isfinite(value)]
            if finite:
                chosen = min(finite, key=lambda side: values[side])
                assignments = {chosen: 0}
                assignment_cost = values[chosen]
            else:
                assignments = {int(hand.get("side", 0)) if int(hand.get("side", 0)) in (0, 1) else 0: 0}

        if not assignments:
            continue
        assignment_records = []
        assigned_hands = []
        for side, index in assignments.items():
            hand = usable[index]
            previous_side = int(hand.get("side", side))
            if previous_side != side:
                changed += 1
            hand["side"] = int(side)
            assigned_hands.append(hand)
            assignment_records.append({
                "from_side": previous_side,
                "to_side": int(side),
                "cost": float(_identity_transition_cost(
                    tracks[side], hand, frame, side, max_step_m
                )),
            })

            centre = _hand_center_from_row(hand)
            wrist = _hand_wrist_from_row(hand)
            old = tracks[side]
            if old and int(frame - old.get("frame", frame)) <= 30:
                gap = max(frame - int(old["frame"]), 1)
                velocity = (centre - np.asarray(old["center"])) / gap
                wrist_velocity = (wrist - np.asarray(old["wrist"])) / gap
                old_velocity = np.asarray(old.get("velocity", np.zeros(3)))
                old_wrist_velocity = np.asarray(
                    old.get("wrist_velocity", np.zeros(3))
                )
                velocity = 0.65 * velocity + 0.35 * old_velocity
                wrist_velocity = 0.65 * wrist_velocity + 0.35 * old_wrist_velocity
            else:
                velocity = np.zeros(3, dtype=np.float64)
                wrist_velocity = np.zeros(3, dtype=np.float64)
            # Prevent one bad branch from poisoning the next prediction.
            velocity_norm = float(np.linalg.norm(velocity))
            if velocity_norm > 0.35:
                velocity *= 0.35 / velocity_norm
            wrist_velocity_norm = float(np.linalg.norm(wrist_velocity))
            if wrist_velocity_norm > 0.45:
                wrist_velocity *= 0.45 / wrist_velocity_norm
            detector = _hand_detector_mode(hand)
            if old and old.get("detector_mode") is not None and detector is None:
                detector = old["detector_mode"]
            tracks[side] = {
                "center": centre,
                "wrist": wrist,
                "velocity": velocity,
                "wrist_velocity": wrist_velocity,
                "signature": _hand_shape_signature(hand),
                "detector_mode": detector,
                "frame": frame,
            }

        # Do not pass all-NaN placeholders into the anatomy arrays: duplicate
        # side IDs would make the last placeholder overwrite a valid track.
        # The row is retained (and can subsequently be rejected) so the frame
        # accounting remains explicit.
        row["hands"] = sorted(assigned_hands, key=lambda hand: int(hand.get("side", 0)))
        present = {int(hand["side"]) for hand in assigned_hands}
        if len(present) < 2:
            row["partial_hand_acceptance"] = True
            row["missing_sides"] = sorted({0, 1} - present)
        else:
            row.pop("missing_sides", None)
        row["identity_tracking"] = {
            "algorithm": "trajectory_identity_assignment_v1",
            "relabelled": any(record["from_side"] != record["to_side"] for record in assignment_records),
            "assignment_cost": float(assignment_cost),
            "assignments": assignment_records,
        }
    return changed


def _hand_image_track_cost(
    previous: Mapping[str, Any], current: Mapping[str, Any], gap: int,
) -> float:
    """Return a shared-camera 2-D trajectory cost for one physical hand.

    Detector handedness may flip, and a weak stereo pair may move the 3-D hand
    to another depth while its pixels remain continuous.  Shared-camera palm,
    wrist, and box motion therefore provide identity evidence that is
    independent of the reconstructed depth.  The term is deliberately bounded
    and cannot by itself freeze a genuinely fast gesture.
    """
    gap = max(int(gap), 1)
    previous_centres = _hand_palm_image_center(dict(previous))
    current_centres = _hand_palm_image_center(dict(current))
    shared = set(previous_centres) & set(current_centres)
    if not shared:
        return 0.0
    palm_motion = [
        float(np.linalg.norm(previous_centres[camera] - current_centres[camera]))
        for camera in shared
    ]
    palm_scale = 35.0 + 32.0 * gap
    cost = 0.85 * min(float(np.median(palm_motion)) / palm_scale, 8.0)

    wrist_motion = []
    box_motion = []
    box_iou = []
    previous_views = previous.get("views", {})
    current_views = current.get("views", {})
    for camera in shared:
        previous_view = previous_views.get(camera, {})
        current_view = current_views.get(camera, {})
        previous_points = np.asarray(
            previous_view.get("joints_2d", []), dtype=np.float64
        )
        current_points = np.asarray(
            current_view.get("joints_2d", []), dtype=np.float64
        )
        if (
            previous_points.shape == (21, 2)
            and current_points.shape == (21, 2)
            and np.all(np.isfinite(previous_points[0]))
            and np.all(np.isfinite(current_points[0]))
        ):
            wrist_motion.append(float(np.linalg.norm(
                previous_points[0] - current_points[0]
            )))
        previous_box = np.asarray(
            previous_view.get("bbox_xyxy", []), dtype=np.float64
        )
        current_box = np.asarray(
            current_view.get("bbox_xyxy", []), dtype=np.float64
        )
        if (
            previous_box.shape == (4,)
            and current_box.shape == (4,)
            and np.all(np.isfinite(previous_box))
            and np.all(np.isfinite(current_box))
        ):
            previous_box_center = 0.5 * (previous_box[:2] + previous_box[2:])
            current_box_center = 0.5 * (current_box[:2] + current_box[2:])
            box_motion.append(float(np.linalg.norm(
                previous_box_center - current_box_center
            )))
            box_iou.append(_bbox_iou(previous_view, current_view))
    if wrist_motion:
        cost += 0.35 * min(
            float(np.median(wrist_motion)) / (45.0 + 38.0 * gap), 8.0
        )
    if box_motion:
        cost += 0.20 * min(
            float(np.median(box_motion)) / (55.0 + 45.0 * gap), 8.0
        )
    if box_iou and gap <= 2:
        # IoU is only a weak cue because a fast articulated hand can change its
        # box substantially.  It mainly breaks a tie between two nearby tracks.
        cost += 0.12 * max(0.0, 0.25 - float(np.median(box_iou)))
    return float(cost)


def _hand_image_track_motion(
    previous: Mapping[str, Any], current: Mapping[str, Any], gap: int,
) -> dict[str, float | int | bool]:
    """Return robust shared-camera image motion for a track link.

    This is deliberately separate from :func:`_hand_image_track_cost`.  A
    bounded soft cost is useful for ordinary identity ranking, but it cannot
    express the important logical contradiction seen in the recordings: the
    palm and wrist barely move in several shared camera images while the
    triangulated hand suddenly moves 20--40 cm in depth.  Such a link is not a
    fast gesture; it is a different stereo-depth branch.
    """
    gap = max(int(gap), 1)
    previous_centres = _hand_palm_image_center(dict(previous))
    current_centres = _hand_palm_image_center(dict(current))
    shared = sorted(set(previous_centres) & set(current_centres))
    palm = [
        float(np.linalg.norm(
            previous_centres[camera] - current_centres[camera]
        )) / gap
        for camera in shared
    ]
    wrist: list[float] = []
    box: list[float] = []
    previous_views = previous.get("views", {})
    current_views = current.get("views", {})
    for camera in shared:
        previous_view = previous_views.get(camera, {})
        current_view = current_views.get(camera, {})
        previous_points = np.asarray(
            previous_view.get("joints_2d", []), dtype=np.float64
        )
        current_points = np.asarray(
            current_view.get("joints_2d", []), dtype=np.float64
        )
        if (
            previous_points.shape == (21, 2)
            and current_points.shape == (21, 2)
            and np.all(np.isfinite(previous_points[0]))
            and np.all(np.isfinite(current_points[0]))
        ):
            wrist.append(float(np.linalg.norm(
                previous_points[0] - current_points[0]
            )) / gap)
        previous_box = np.asarray(
            previous_view.get("bbox_xyxy", []), dtype=np.float64
        )
        current_box = np.asarray(
            current_view.get("bbox_xyxy", []), dtype=np.float64
        )
        if (
            previous_box.shape == (4,)
            and current_box.shape == (4,)
            and np.all(np.isfinite(previous_box))
            and np.all(np.isfinite(current_box))
        ):
            previous_center = 0.5 * (previous_box[:2] + previous_box[2:])
            current_center = 0.5 * (current_box[:2] + current_box[2:])
            box.append(float(np.linalg.norm(
                previous_center - current_center
            )) / gap)
    palm_median = float(np.median(palm)) if palm else float("nan")
    wrist_median = float(np.median(wrist)) if wrist else float("nan")
    box_median = float(np.median(box)) if box else float("nan")
    available = [
        value for value in (palm_median, wrist_median, box_median)
        if np.isfinite(value)
    ]
    robust = float(np.median(available)) if available else float("nan")
    return {
        "shared_camera_count": len(shared),
        "palm_px_per_frame": palm_median,
        "wrist_px_per_frame": wrist_median,
        "bbox_px_per_frame": box_median,
        "robust_px_per_frame": robust,
        "available": bool(shared and np.isfinite(robust)),
    }


def _identity_2d_link_policy(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    gap: int,
    args: argparse.Namespace | None,
) -> tuple[str, dict[str, Any]]:
    """Classify an image-space link used by the identity hysteresis pass.

    A partial row is not an ordinary missing value: it is often the only
    observation of a hand while the other hand is occluded.  The old identity
    DP therefore allowed ``partial -> partial`` rows to change sides by paying
    two tiny missing-side penalties, even when the two boxes were hundreds of
    pixels apart.  This helper makes the state transition explicit:

    ``continuous``
        shared-camera motion is small enough to be the same image track;
    ``break``
        the image track has ended/re-entered and must not be reconnected as
        the same partial state;
    ``ambiguous``/``unknown``
        insufficient evidence for a hard decision.

    The returned diagnostics are serialized in the identity metadata so a
    later video audit can distinguish a genuine re-entry from a detector
    handedness flip.
    """
    gap = max(int(gap), 1)
    image = _hand_image_track_motion(previous, current, gap)
    diagnostics = dict(image)
    diagnostics["gap"] = gap
    soft = max(float(getattr(
        args, "identity_2d_soft_motion_px", 80.0
    )) if args is not None else 80.0, 0.0)
    hard = max(float(getattr(
        args, "identity_2d_hard_motion_px", 125.0
    )) if args is not None else 125.0, soft)
    max_gap = max(int(getattr(
        args, "identity_2d_max_gap", 8
    )) if args is not None else 8, 0)
    diagnostics["soft_limit_px_per_frame"] = soft
    diagnostics["hard_limit_px_per_frame"] = hard
    diagnostics["max_gap"] = max_gap
    if (
        max_gap <= 0
        or gap > max_gap
        or not bool(image.get("available"))
    ):
        return "unknown", diagnostics
    # A single shared view is useful only for an adjacent frame.  Across an
    # occlusion it is too easy for a re-entering hand to reuse an unrelated
    # detector box, so leave that case ambiguous.
    shared = int(image.get("shared_camera_count", 0))
    if shared < 2 and not (shared >= 1 and gap == 1):
        return "unknown", diagnostics
    motion = float(image.get("robust_px_per_frame", np.nan))
    if not np.isfinite(motion):
        return "unknown", diagnostics
    if motion <= soft:
        return "continuous", diagnostics
    if motion >= hard:
        return "break", diagnostics
    return "ambiguous", diagnostics


def _depth_reconnect_contradiction(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    gap: int,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Detect a same-image-track link that jumps to another 3-D branch."""
    gap = max(int(gap), 1)
    maximum_gap = max(int(getattr(args, "track_reconnect_max_gap", 8)), 0)
    image = _hand_image_track_motion(previous, current, gap)
    previous_center = _hand_center_from_row(dict(previous))
    current_center = _hand_center_from_row(dict(current))
    distance = (
        float(np.linalg.norm(current_center - previous_center))
        if np.all(np.isfinite(previous_center))
        and np.all(np.isfinite(current_center))
        else float("nan")
    )
    step = distance / gap if np.isfinite(distance) else float("nan")
    image_limit = max(float(getattr(
        args, "track_reconnect_max_2d_motion_px", 55.0
    )), 0.0)
    depth_limit = max(float(getattr(
        args, "track_reconnect_max_3d_step_m", 0.095
    )), 0.0)
    # Require two shared views for a hard decision whenever possible.  A
    # single shared camera is still useful over a one-frame gap, but is too
    # ambiguous after an occlusion to reject a real re-entry on its own.
    shared = int(image["shared_camera_count"])
    enough_image_evidence = shared >= 2 or (shared >= 1 and gap == 1)
    contradiction = bool(
        maximum_gap > 0
        and gap <= maximum_gap
        and enough_image_evidence
        and image["available"]
        and float(image["robust_px_per_frame"]) <= image_limit
        and np.isfinite(step)
        and step > depth_limit
    )
    return contradiction, {
        **image,
        "gap": gap,
        "center_distance_m": distance,
        "center_step_m_per_frame": step,
        "image_continuity_limit_px_per_frame": image_limit,
        "depth_step_limit_m_per_frame": depth_limit,
    }


def _identity_observation_cost(
    previous: dict[str, Any],
    current: dict[str, Any],
    gap: int,
    max_step_m: float,
    args: argparse.Namespace | None = None,
) -> float:
    """Cost of linking two observations to one physical hand track."""
    previous_center = _hand_center_from_row(previous)
    current_center = _hand_center_from_row(current)
    if not (
        np.all(np.isfinite(previous_center))
        and np.all(np.isfinite(current_center))
    ):
        return float("inf")
    gap = max(int(gap), 1)
    if gap > 30:
        # A long disappearance should not force a stale identity assignment.
        return 0.25
    if args is not None:
        contradictory, _diagnostics = _depth_reconnect_contradiction(
            previous, current, gap, args
        )
        if contradictory:
            # Identity and depth are different decisions.  Continuous pixels
            # are strong evidence that this is the same physical hand even
            # though its triangulation is bad.  Preserve that identity here;
            # the unified branch selector applies the decisive depth penalty
            # and either chooses another triangulation or leaves the frame to
            # the guarded temporal repair.  Penalising this link itself would
            # make the DP swap left/right IDs merely to avoid a bad Z value.
            return 0.05 + _hand_image_track_cost(previous, current, gap)
    scale = max(0.035, 0.045 * gap + 0.20 * max_step_m)
    distance = float(np.linalg.norm(current_center - previous_center))
    cost = min(distance / scale, 8.0)

    previous_wrist = _hand_wrist_from_row(previous)
    current_wrist = _hand_wrist_from_row(current)
    if np.all(np.isfinite(previous_wrist)) and np.all(np.isfinite(current_wrist)):
        wrist_scale = max(0.045, 0.065 * gap + 0.15 * max_step_m)
        cost += 0.55 * min(
            float(np.linalg.norm(current_wrist - previous_wrist)) / wrist_scale,
            8.0,
        )

    previous_signature = _hand_shape_signature(previous)
    current_signature = _hand_shape_signature(current)
    if previous_signature.shape == (21, 3) and current_signature.shape == (21, 3):
        finite = (
            np.isfinite(previous_signature).all(axis=1)
            & np.isfinite(current_signature).all(axis=1)
        )
        if np.count_nonzero(finite) >= 8:
            shape_distance = float(np.median(
                np.linalg.norm(
                    previous_signature[finite] - current_signature[finite], axis=1
                )
            ))
            cost += 0.30 * min(shape_distance, 8.0)
    cost += _hand_image_track_cost(previous, current, gap)
    return float(cost)


def _identity_state_prior(
    row: dict[str, Any], hands: list[dict[str, Any]], state: tuple[int, int],
) -> float:
    """Small detector-label prior used only to break exact trajectory ties."""
    cost = 0.0
    for side, index in enumerate(state):
        if index < 0 or index >= len(hands):
            continue
        original_side = hands[index].get("side")
        if original_side is not None:
            try:
                cost += 0.035 * float(int(original_side) != side)
            except (TypeError, ValueError):
                pass
        detector_side = _hand_detector_mode(hands[index])
        if detector_side is not None:
            # Detector handedness is deliberately weaker than trajectory
            # evidence; this term cannot override a plausible crossing.
            cost += 0.018 * float(detector_side != side)
    return cost


def _identity_transition_cost_states(
    previous_hands: list[dict[str, Any]],
    previous_state: tuple[int, int],
    current_hands: list[dict[str, Any]],
    current_state: tuple[int, int],
    gap: int,
    max_step_m: float,
    partial_switch_penalty: float = 1.50,
    partial_switch_max_cost: float = 3.0,
    args: argparse.Namespace | None = None,
) -> float:
    cost = 0.0
    for side in (0, 1):
        previous_index = previous_state[side]
        current_index = current_state[side]
        if previous_index >= 0 and current_index >= 0:
            # A partial observation is allowed to disappear and re-enter, but
            # it must not inherit the old physical side when its image track
            # has already broken.  The former DP saw only a small 3-D/missing
            # penalty here and consequently linked frame 279 (the lower hand)
            # to frame 281 (the upper hand) in the 0828/111 recording.  Make a
            # hard 2-D break more expensive than an explicit missing/new state;
            # continuous links retain the normal trajectory cost below.
            link_policy, _link_diagnostics = _identity_2d_link_policy(
                previous_hands[previous_index], current_hands[current_index],
                gap, args,
            )
            if link_policy == "break":
                partial_link = (
                    len(previous_hands) == 1 or len(current_hands) == 1
                )
                if partial_link:
                    return float(
                        max(float(getattr(
                            args, "identity_2d_hard_break_penalty", 24.0
                        )) if args is not None else 24.0, 0.0)
                    )
            value = _identity_observation_cost(
                previous_hands[previous_index], current_hands[current_index],
                gap, max_step_m, args,
            )
            if not np.isfinite(value):
                return float("inf")
            cost += value
        elif previous_index >= 0 or current_index >= 0:
            # Missing one side is allowed.  Keep the penalty modest so a
            # visible hand is not forced onto the other physical track merely
            # to avoid a gap cost.
            cost += 0.18 * min(max(int(gap), 1), 5)

    # A dual-hand row can still choose the crossed permutation above.  When a
    # previous observation and a current observation share continuous image
    # evidence but disagree only in triangulated depth, keep them on the same
    # physical side.  Otherwise detector order can convert a depth failure
    # into a left/right identity flip.
    if args is not None and gap <= max(int(getattr(
        args, "track_reconnect_max_gap", 8
    )), 0):
        for previous_side, previous_index in enumerate(previous_state):
            if previous_index < 0 or previous_index >= len(previous_hands):
                continue
            previous_hand = previous_hands[previous_index]
            for current_side, current_index in enumerate(current_state):
                if current_side == previous_side:
                    continue
                if current_index < 0 or current_index >= len(current_hands):
                    continue
                current_hand = current_hands[current_index]
                contradictory, diagnostics = _depth_reconnect_contradiction(
                    previous_hand, current_hand, gap, args
                )
                if contradictory:
                    cost += max(float(getattr(
                        args, "track_reconnect_depth_penalty", 2500.0
                    )), 0.0)

    # When both adjacent rows contain exactly one hand, swapping the side
    # state used to cost only two missing-side penalties.  That made a
    # perfectly continuous hand (for example during a short occlusion) jump
    # from physical side 1 to side 0 whenever its motion exceeded 0.36 in the
    # normalized identity cost.  Compare the observations even across the
    # state change and add hysteresis only when they are plausibly the same
    # track.  If the observations are far apart, the side switch is likely a
    # genuine hand re-entry and remains allowed.
    previous_present = [
        (side, index) for side, index in enumerate(previous_state)
        if index >= 0 and index < len(previous_hands)
    ]
    current_present = [
        (side, index) for side, index in enumerate(current_state)
        if index >= 0 and index < len(current_hands)
    ]
    if (
        len(previous_present) == 1
        and len(current_present) == 1
        and previous_present[0][0] != current_present[0][0]
        and gap <= 8
    ):
        previous_hand = previous_hands[previous_present[0][1]]
        current_hand = current_hands[current_present[0][1]]
        continuity = _identity_observation_cost(
            previous_hand, current_hand, gap, max_step_m, args
        )
        switch_limit = max(float(partial_switch_max_cost), 0.0)
        switch_penalty = max(float(partial_switch_penalty), 0.0)
        if continuity <= switch_limit:
            cost += switch_penalty + 0.25 * max(0.0, switch_limit - continuity)
    return float(cost)


def _identity_states(hands: list[dict[str, Any]]) -> list[tuple[int, int]]:
    if len(hands) >= 2:
        return [(0, 1), (1, 0)]
    if len(hands) == 1:
        return [(0, -1), (-1, 0)]
    return [(-1, -1)]


def _identity_state_from_existing_labels(
    hands: list[dict[str, Any]],
) -> tuple[int, int] | None:
    """Return the side assignment already written on a complete row.

    The 2-D pass is a *canonicalisation* pass, not a second global left/right
    solver.  Complete rows have already gone through the geometry/trajectory
    selector and are the only reliable orientation anchors.  Letting the
    image-only DP freely permute every complete row makes the two globally
    symmetric paths equally cheap; ``argmin`` then chooses whichever
    orientation happens to win over a long partial interval.  That is exactly
    how an otherwise correct stream acquired a whole-segment side swap.

    ``None`` is returned for malformed rows (duplicate/missing labels), where
    the ordinary state enumeration remains the safest fallback.
    """
    by_side: dict[int, int] = {}
    for index, hand in enumerate(hands):
        try:
            side = int(hand.get("side", -1))
        except (TypeError, ValueError):
            return None
        if side not in (0, 1) or side in by_side:
            return None
        by_side[side] = index
    if set(by_side) != {0, 1}:
        return None
    return (by_side[0], by_side[1])


def _identity_2d_transition_cost_states(
    previous_hands: list[dict[str, Any]],
    previous_state: tuple[int, int],
    current_hands: list[dict[str, Any]],
    current_state: tuple[int, int],
    gap: int,
    args: argparse.Namespace | None = None,
) -> float:
    """Cost a physical-side assignment using image tracks only.

    The 3-D trajectory is deliberately absent from this cost.  A wrong stereo
    branch can move a hand by 20--40 cm while its detector pixels remain almost
    unchanged; using that depth in the identity solver turns a depth failure
    into a left/right swap.  Conversely, when a partial observation really is
    a new/re-entering image track, reconnecting it to the old side must be
    more expensive than paying one missing-side transition.
    """
    gap = max(int(gap), 1)
    soft = max(float(getattr(
        args, "identity_2d_soft_motion_px", 80.0
    )) if args is not None else 80.0, 1.0)
    hard = max(float(getattr(
        args, "identity_2d_hard_motion_px", 125.0
    )) if args is not None else 125.0, soft)
    missing_penalty = 1.15 * min(gap, 4)
    break_penalty = 18.0
    cost = 0.0
    links: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    for side in (0, 1):
        before_index = previous_state[side]
        after_index = current_state[side]
        before_present = 0 <= before_index < len(previous_hands)
        after_present = 0 <= after_index < len(current_hands)
        if before_present and after_present:
            before = previous_hands[before_index]
            after = current_hands[after_index]
            links.append((before, after, True))
            image = _hand_image_track_motion(before, after, gap)
            if not image.get("available"):
                # No shared image evidence: keep the link possible, but do not
                # let it beat a clearly supported alternative by a zero cost.
                cost += 1.5
                continue
            motion = float(image.get("robust_px_per_frame", np.nan))
            if not np.isfinite(motion):
                cost += 1.5
            elif motion <= soft:
                cost += 0.20 * motion / soft
            elif motion < hard:
                # Smooth Huber-like ramp in the ambiguous interval.
                cost += 0.20 + 2.0 * (motion - soft) / max(hard - soft, 1.0)
            else:
                # A hard break is only decisive for a partial-to-partial link.
                # Full rows may contain a genuinely fast gesture, so retain a
                # bounded (rather than infinite) cost there.
                partial_link = (
                    len(previous_hands) == 1 or len(current_hands) == 1
                )
                cost += break_penalty if partial_link else 5.0 + min(
                    (motion - hard) / hard, 5.0
                )
        elif before_present or after_present:
            # Explicit missing/re-entry state.  This is intentionally cheaper
            # than reconnecting two partial rows whose pixels prove a track
            # break, which fixes full -> partial -> full identity poisoning.
            cost += missing_penalty

    # Small detector/original-side priors resolve exact 2-D ties only.  They
    # never override a continuous image link or a hard break.
    for side, index in enumerate(current_state):
        if not (0 <= index < len(current_hands)):
            continue
        hand = current_hands[index]
        original_side = hand.get("side")
        try:
            cost += 0.025 * float(int(original_side) != side)
        except (TypeError, ValueError):
            pass
        detector_side = _hand_detector_mode(hand)
        if detector_side is not None:
            cost += 0.010 * float(detector_side != side)

    # When both rows contain one hand, a continuous link should have strong
    # hysteresis against changing slots.  A hard image break, on the other
    # hand, is explicitly treated as a re-entry and is allowed to select the
    # other physical side without paying this penalty.
    previous_present = [
        (side, index) for side, index in enumerate(previous_state)
        if 0 <= index < len(previous_hands)
    ]
    current_present = [
        (side, index) for side, index in enumerate(current_state)
        if 0 <= index < len(current_hands)
    ]
    if len(previous_present) == 1 and len(current_present) == 1:
        previous_side, previous_index = previous_present[0]
        current_side, current_index = current_present[0]
        if previous_side != current_side:
            image = _hand_image_track_motion(
                previous_hands[previous_index], current_hands[current_index], gap
            )
            motion = float(image.get("robust_px_per_frame", np.nan))
            if image.get("available") and np.isfinite(motion) and motion <= soft:
                cost += 3.0
    return float(cost)


def _annotate_2d_temporal_segments(
    ordered: list[dict[str, Any]], args: argparse.Namespace | None = None,
) -> int:
    """Mark per-side image-track segments for downstream temporal filters.

    Identity canonicalisation and numerical smoothing are separate concerns.
    Once a shared-camera link is classified as a hard break, every later
    temporal stage must treat the new observation as a re-entry; otherwise a
    valid filter can interpolate the old hand into the new one.  Segment IDs
    are attached to hands (and a compact boundary record to rows) so the
    anatomy module can enforce the same rule without knowing detector details.
    """
    segments = [0, 0]
    last_hand: list[dict[str, Any] | None] = [None, None]
    last_frame: list[int | None] = [None, None]
    boundary_count = 0
    max_gap = max(int(getattr(
        args, "identity_2d_max_gap", 8
    )) if args is not None else 8, 0)
    for row in ordered:
        frame = int(row.get("sync_index", 0))
        boundaries: list[int] = []
        by_side = {
            int(hand.get("side", -1)): hand
            for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        for side in (0, 1):
            hand = by_side.get(side)
            if hand is None:
                continue
            previous = last_hand[side]
            previous_frame = last_frame[side]
            boundary = False
            if previous is not None and previous_frame is not None:
                gap = frame - int(previous_frame)
                policy, diagnostics = _identity_2d_link_policy(
                    previous, hand, gap, args
                )
                # A long blackout or a hard pixel jump starts a new segment.
                # Unknown links across a one-frame absence are kept together
                # only when the detector supplies enough shared evidence; an
                # unknown long-gap link is safer as a re-entry.
                if policy == "break" or gap > max_gap or (
                    policy == "unknown" and gap > 1
                ):
                    segments[side] += 1
                    boundary = True
                    boundary_count += 1
                    if "identity_reentry_2d" not in row:
                        motion = diagnostics.get("robust_px_per_frame")
                        row["identity_reentry_2d"] = {
                            "previous_sync_index": int(previous_frame),
                            "motion_px_per_frame": (
                                float(motion) if motion is not None
                                and np.isfinite(float(motion)) else None
                            ),
                            "shared_camera_count": int(
                                diagnostics.get("shared_camera_count", 0)
                            ),
                        }
            hand["temporal_segment_id"] = int(segments[side])
            if boundary:
                hand["temporal_boundary_before"] = True
                boundaries.append(side)
            else:
                hand.pop("temporal_boundary_before", None)
            last_hand[side] = hand
            last_frame[side] = frame
        if boundaries:
            row["temporal_track_boundary_before"] = sorted(set(boundaries))
        else:
            row.pop("temporal_track_boundary_before", None)
        row["temporal_segment_ids"] = {
            str(side): int(by_side[side].get("temporal_segment_id", segments[side]))
            for side in by_side
        }
    return boundary_count


def _stabilize_hand_identities_2d(
    rows: list[dict[str, Any]], args: argparse.Namespace | None = None,
) -> int:
    """Canonicalize physical hand IDs from shared-camera image trajectories.

    This is a final identity pass, separate from the geometric/depth Viterbi
    selector.  It keeps a visible hand on its image track through a partial
    interval, while treating a large 2-D discontinuity as a new/re-entry
    observation.  In particular, it prevents two unrelated one-hand rows from
    alternating between the same physical side merely because their 3-D
    triangulations happen to be close in cost.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    infos: list[tuple[dict[str, Any], list[dict[str, Any]], list[tuple[int, int]]]] = []
    complete_anchor_seen = False
    for row in ordered:
        hands = [
            hand for hand in row.get("hands", [])
            if np.all(np.isfinite(_hand_center_from_row(hand)))
        ]
        if len(hands) > 2:
            hands = sorted(
                hands,
                key=lambda hand: float(hand.get("quality", {}).get(
                    "valid_joint_count", 0
                )),
                reverse=True,
            )[:2]
        states = _identity_states(hands)
        # Anchor the first complete row to remove the global left/right
        # symmetry, then allow later complete rows to use either permutation.
        # Keeping every complete row fixed made a one-frame detector/geometry
        # swap impossible to repair (frames 188--189 and 244--246 in the
        # cameras1--4 recording).  A fixed first anchor plus the weak existing
        # label prior prevents the old whole-sequence mirror solution while
        # shared-camera 2-D trajectories can now correct local complete-row
        # flips.
        existing = _identity_state_from_existing_labels(hands) if len(hands) == 2 else None
        if existing is not None:
            if not complete_anchor_seen:
                states = [existing]
                complete_anchor_seen = True
            else:
                swapped = (existing[1], existing[0])
                states = [existing, swapped]
        infos.append((row, hands, states))
    if not infos:
        return 0

    dynamic: list[np.ndarray] = []
    back: list[np.ndarray | None] = [None]
    first_row, first_hands, first_states = infos[0]
    first_cost = np.asarray([
        _identity_state_prior(first_row, first_hands, state)
        + (0.20 * index if len(first_states) > 1 else 0.0)
        for index, state in enumerate(first_states)
    ], dtype=np.float64)
    dynamic.append(first_cost)
    for position in range(1, len(infos)):
        row, hands, states = infos[position]
        previous_row, previous_hands, previous_states = infos[position - 1]
        gap = int(row["sync_index"]) - int(previous_row["sync_index"])
        matrix = np.full((len(previous_states), len(states)), np.inf)
        for previous_index, previous_state in enumerate(previous_states):
            for current_index, current_state in enumerate(states):
                transition = _identity_2d_transition_cost_states(
                    previous_hands, previous_state, hands, current_state, gap, args
                )
                matrix[previous_index, current_index] = (
                    dynamic[-1][previous_index]
                    + transition
                    + _identity_state_prior(row, hands, current_state)
                )
        back.append(np.argmin(matrix, axis=0))
        dynamic.append(np.min(matrix, axis=0))

    selected = int(np.argmin(dynamic[-1]))
    chosen_indices = [selected]
    for links in reversed(back[1:]):
        assert links is not None
        selected = int(links[selected])
        chosen_indices.append(selected)
    chosen_indices.reverse()

    changed = 0
    for position, ((row, hands, states), selected_index) in enumerate(
        zip(infos, chosen_indices)
    ):
        state = states[selected_index]
        assigned: list[dict[str, Any]] = []
        records = []
        for side, hand_index in enumerate(state):
            if not (0 <= hand_index < len(hands)):
                continue
            hand = hands[hand_index]
            try:
                old_side = int(hand.get("side", side))
            except (TypeError, ValueError):
                old_side = side
            if old_side != side:
                changed += 1
            hand["side"] = int(side)
            assigned.append(hand)
            records.append({"from_side": old_side, "to_side": int(side)})
        row["hands"] = sorted(assigned, key=lambda hand: int(hand.get("side", 0)))
        present = {int(hand["side"]) for hand in assigned}
        if len(present) < 2:
            row["partial_hand_acceptance"] = True
            row["missing_sides"] = sorted({0, 1} - present)
        else:
            row.pop("missing_sides", None)
            row.pop("partial_hand_acceptance", None)
        if assigned:
            row["identity_tracking_2d"] = {
                "algorithm": "shared_camera_2d_track_hysteresis_v1",
                "relabelled": any(
                    record["from_side"] != record["to_side"] for record in records
                ),
                "assignment_cost": float(dynamic[position][chosen_indices[position]]),
                "assignments": records,
            }
            # Mark a partial observation that follows a hard image break as a
            # re-entry for downstream sequence diagnostics.
            if position:
                previous_row, previous_hands, previous_states = infos[position - 1]
                gap = int(row["sync_index"]) - int(previous_row["sync_index"])
                if len(previous_hands) == 1 and len(assigned) == 1:
                    image = _hand_image_track_motion(
                        previous_hands[0], assigned[0], gap
                    )
                    motion = float(image.get("robust_px_per_frame", np.nan))
                    if image.get("available") and np.isfinite(motion) and motion >= max(
                        float(getattr(args, "identity_2d_hard_motion_px", 125.0))
                        if args is not None else 125.0,
                        float(getattr(args, "identity_2d_soft_motion_px", 80.0))
                        if args is not None else 80.0,
                    ):
                        row["identity_reentry_2d"] = {
                            "previous_sync_index": int(previous_row["sync_index"]),
                            "motion_px_per_frame": motion,
                            "shared_camera_count": int(image.get("shared_camera_count", 0)),
                        }
    _annotate_2d_temporal_segments(ordered, args)
    return changed


def _repair_partial_identity_labels_2d(
    rows: list[dict[str, Any]], args: argparse.Namespace | None = None,
) -> int:
    """Repair one-hand labels using bracketing complete image-track anchors.

    A one-hand row is inherently ambiguous if it is compared only with the
    immediately preceding row: during an alternating detector dropout the
    previous row may contain the *other* hand.  Complete rows on both sides
    provide a much safer physical-side anchor, while shared-camera pixels keep
    the decision independent of a bad 3-D depth branch.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    radius = max(int(getattr(args, "identity_2d_anchor_radius", 24)) if args else 24, 2)
    soft = max(float(getattr(args, "identity_2d_soft_motion_px", 80.0)) if args else 80.0, 1.0)
    hard = max(float(getattr(args, "identity_2d_hard_motion_px", 125.0)) if args else 125.0, soft)
    anchors: dict[int, list[tuple[int, dict[str, Any]]]] = {0: [], 1: []}
    for row in ordered:
        hands = [
            hand for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
            and np.all(np.isfinite(_hand_center_from_row(hand)))
        ]
        if len(hands) < 2:
            continue
        by_side = {int(hand["side"]): hand for hand in hands}
        if set(by_side) != {0, 1}:
            continue
        frame = int(row["sync_index"])
        for side in (0, 1):
            anchors[side].append((frame, by_side[side]))

    changed = 0
    for row in ordered:
        hands = [hand for hand in row.get("hands", []) if np.all(
            np.isfinite(_hand_center_from_row(hand))
        )]
        if len(hands) != 1:
            continue
        hand = hands[0]
        frame = int(row["sync_index"])
        scores: dict[int, float] = {}
        for side in (0, 1):
            candidates: list[tuple[int, float]] = []
            for anchor_frame, anchor in anchors[side]:
                gap = abs(frame - anchor_frame)
                if gap <= 0 or gap > radius:
                    continue
                image = _hand_image_track_motion(anchor, hand, gap)
                if not image.get("available"):
                    continue
                shared = int(image.get("shared_camera_count", 0))
                if shared < 2 and gap > 1:
                    continue
                motion = float(image.get("robust_px_per_frame", np.nan))
                if not np.isfinite(motion):
                    continue
                candidates.append((gap, motion))
            if candidates:
                candidates.sort(key=lambda value: value[0])
                # Nearest anchors carry most identity information.  Including
                # a second bracket prevents one bad complete row from winning.
                nearest = np.asarray([value[1] for value in candidates[:3]], dtype=np.float64)
                scores[side] = float(np.median(nearest) + 0.20 * np.min(nearest))

        if len(scores) != 2:
            continue
        best_side = min(scores, key=scores.get)
        try:
            current_side = int(hand.get("side", best_side))
        except (TypeError, ValueError):
            current_side = best_side
        current_score = scores.get(current_side, float("inf"))
        best_score = scores[best_side]
        # Do not relabel a plausible existing assignment for a tiny tie.  A
        # clear bracketed advantage is required, especially around a true
        # two-hand crossing where image tracks can briefly approach each other.
        if best_side != current_side:
            # A current label that already has a plausible bracket must be
            # preserved.  Only repair an unmistakably bad partial label when
            # the alternate side has a clearly continuous bracket; otherwise
            # a real crossing can be mistaken for an identity error.
            if (
                current_score <= hard
                or best_score > soft
                or best_score + max(20.0, 0.20 * soft) >= current_score
            ):
                continue
            hand["side"] = int(best_side)
            changed += 1
            row.setdefault("identity_partial_repair", {})
            row["identity_partial_repair"] = {
                "algorithm": "bracketed_complete_2d_anchor_v1",
                "from_side": current_side,
                "to_side": best_side,
                "score_before": float(current_score),
                "score_after": float(best_score),
            }
    for row in ordered:
        row["hands"] = sorted(row.get("hands", []), key=lambda hand: int(hand.get("side", 0)))
    return changed


def _repair_partial_identity_by_persistent_image_track(
    rows: list[dict[str, Any]], args: argparse.Namespace | None = None,
) -> int:
    """Keep partial rows on the persistent image track through detector flips.

    This is deliberately a final conservative pass.  When a one-hand row has
    a shared-camera track that is continuous with one of the two physical
    tracks, relabel it to that side even if its 3-D branch is far away.  A
    hard image break, a true two-hand row, or an ambiguous match is left
    untouched.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    radius = max(int(getattr(args, "identity_2d_max_gap", 8)) if args else 8, 1)
    soft = max(float(getattr(args, "identity_2d_soft_motion_px", 80.0)) if args else 80.0, 1.0)
    changed = 0
    for pos, row in enumerate(ordered):
        hands = [h for h in row.get("hands", []) if np.all(np.isfinite(_hand_center_from_row(h)))]
        if len(hands) != 1:
            continue
        hand = hands[0]
        frame = int(row["sync_index"])
        evidence: dict[int, list[float]] = {0: [], 1: []}
        for side in (0, 1):
            for direction in (-1, 1):
                for cursor in range(pos + direction, len(ordered) if direction > 0 else -1, direction):
                    other_row = ordered[cursor]
                    gap = abs(int(other_row["sync_index"]) - frame)
                    if gap > radius:
                        break
                    other = next((h for h in other_row.get("hands", [])
                                  if int(h.get("side", -1)) == side
                                  and np.all(np.isfinite(_hand_center_from_row(h)))), None)
                    if other is None:
                        continue
                    image = _hand_image_track_motion(other, hand, gap)
                    if not image.get("available"):
                        continue
                    shared = int(image.get("shared_camera_count", 0))
                    if shared < 2 and gap > 1:
                        continue
                    motion = float(image.get("robust_px_per_frame", np.nan))
                    if np.isfinite(motion):
                        evidence[side].append(motion)
                    break
        if not evidence[0] or not evidence[1]:
            continue
        scores = {side: float(np.median(values)) for side, values in evidence.items()}
        best = min(scores, key=scores.get)
        current = int(hand.get("side", best))
        # A partial label is changed only when the alternative is clearly the
        # continuous track and the current side shows a hard image mismatch.
        if best == current or scores[best] > soft or scores[current] < 1.35 * soft:
            continue
        hand["side"] = int(best)
        row.setdefault("identity_partial_repair", {})
        row["identity_partial_repair"] = {
            "algorithm": "persistent_shared_camera_track_v1",
            "from_side": current,
            "to_side": best,
            "score_before": scores[current],
            "score_after": scores[best],
        }
        changed += 1
    for row in ordered:
        row["hands"] = sorted(row.get("hands", []), key=lambda hand: int(hand.get("side", 0)))
    return changed


def _fill_short_missing_hand_tracks(
    rows: list[dict[str, Any]], args: argparse.Namespace | None = None,
) -> int:
    """Fill a brief detector dropout from the same physical track.

    The source result may contain both gloves in the image while only one
    detector hypothesis survives a frame.  Alternating one-hand rows then
    look exactly like a left/right swap.  For a short gap bracketed by the
    same side, interpolate the 3-D pose and mark it synthetic.  Long gaps,
    hard 2-D re-entries, and missing brackets are never hallucinated.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    max_missing = max(int(getattr(args, "identity_fill_max_gap", 3)) if args else 3, 0)
    max_motion = max(float(getattr(args, "identity_fill_max_2d_motion_px", 80.0)) if args else 80.0, 0.0)
    filled = 0
    for position, row in enumerate(ordered):
        present_sides = {
            int(hand.get("side", -1)) for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        if not present_sides or present_sides == {0, 1}:
            continue
        frame = int(row["sync_index"])
        for side in (0, 1):
            if side in present_sides:
                continue
            before = None
            before_pos = None
            for cursor in range(position - 1, -1, -1):
                candidate = next((hand for hand in ordered[cursor].get("hands", [])
                                  if int(hand.get("side", -1)) == side
                                  and not hand.get("synthetic_track_fill")), None)
                if candidate is not None:
                    before = candidate
                    before_pos = cursor
                    break
            after = None
            after_pos = None
            for cursor in range(position + 1, len(ordered)):
                candidate = next((hand for hand in ordered[cursor].get("hands", [])
                                  if int(hand.get("side", -1)) == side
                                  and not hand.get("synthetic_track_fill")), None)
                if candidate is not None:
                    after = candidate
                    after_pos = cursor
                    break
            if before is None or after is None or before_pos is None or after_pos is None:
                continue
            before_frame = int(ordered[before_pos]["sync_index"])
            after_frame = int(ordered[after_pos]["sync_index"])
            if not (before_frame < frame < after_frame):
                continue
            missing_count = after_frame - before_frame - 1
            if missing_count < 1 or missing_count > max_missing:
                continue
            image = _hand_image_track_motion(before, after, after_frame - before_frame)
            if not image.get("available") or int(image.get("shared_camera_count", 0)) < 2:
                continue
            motion = float(image.get("robust_px_per_frame", np.nan))
            if not np.isfinite(motion) or motion > max_motion:
                continue
            previous_segment = before.get("temporal_segment_id")
            following_segment = after.get("temporal_segment_id")
            if (
                previous_segment is not None and following_segment is not None
                and int(previous_segment) != int(following_segment)
            ):
                continue
            first = np.asarray(before.get("joints_base_m", []), dtype=np.float64)
            second = np.asarray(after.get("joints_base_m", []), dtype=np.float64)
            if first.shape != (21, 3) or second.shape != (21, 3):
                continue
            alpha = (frame - before_frame) / float(after_frame - before_frame)
            points = np.full((21, 3), np.nan, dtype=np.float64)
            finite_first = np.isfinite(first).all(axis=1)
            finite_second = np.isfinite(second).all(axis=1)
            both = finite_first & finite_second
            points[both] = (1.0 - alpha) * first[both] + alpha * second[both]
            points[finite_first & ~finite_second] = first[finite_first & ~finite_second]
            points[~finite_first & finite_second] = second[~finite_first & finite_second]
            if np.count_nonzero(np.isfinite(points).all(axis=1)) < 3:
                continue
            synthetic = copy.deepcopy(before)
            synthetic["side"] = int(side)
            synthetic["joints_base_m"] = points.tolist()
            synthetic["unrefined_joints_base_m"] = points.tolist()
            synthetic["views"] = {}
            synthetic["camera_ids"] = []
            synthetic["inlier_view_counts"] = [0] * 21
            quality = dict(synthetic.get("quality", {}))
            quality["valid_joint_count"] = int(np.count_nonzero(np.isfinite(points).all(axis=1)))
            quality["synthetic_track_fill"] = True
            synthetic["quality"] = quality
            synthetic["fusion_mode"] = "temporal_track_fill"
            synthetic["synthetic_track_fill"] = True
            synthetic["synthetic_source_sync_indices"] = [before_frame, after_frame]
            if previous_segment is not None:
                synthetic["temporal_segment_id"] = int(previous_segment)
            row.setdefault("synthetic_hand_fill_sides", []).append(int(side))
            row.setdefault("original_missing_sides", []).append(int(side))
            row["partial_hand_acceptance"] = True
            row["missing_sides"] = []
            row.setdefault("hands", []).append(synthetic)
            row["hands"] = sorted(row["hands"], key=lambda hand: int(hand.get("side", 0)))
            filled += 1
    return filled


def _final_sequence_hand_center(hand: Mapping[str, Any]) -> np.ndarray:
    """Robust palm centre used by the post-anatomy sequence guard."""
    points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
    if points.shape != (21, 3):
        return np.full(3, np.nan)
    palm = points[[0, 5, 9, 13, 17]]
    finite = np.isfinite(palm).all(axis=1)
    return np.median(palm[finite], axis=0) if np.any(finite) else np.full(3, np.nan)


def _final_sequence_2d_motion(
    previous: Mapping[str, Any], current: Mapping[str, Any], gap: int,
) -> tuple[float, int]:
    """Return robust shared-camera palm motion and support count."""
    first = _hand_palm_image_center(dict(previous))
    second = _hand_palm_image_center(dict(current))
    shared = sorted(set(first) & set(second))
    if not shared:
        return float("nan"), 0
    gap = max(int(gap), 1)
    values = [float(np.linalg.norm(first[camera] - second[camera])) / gap for camera in shared]
    return float(np.median(values)), len(shared)


def _same_image_observation(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> tuple[bool, int]:
    """Return whether two serialized hands use the same detector tracks.

    Detection indices are camera-local, so the comparison is made for every
    shared camera.  A branch candidate that selects another box is a different
    hand hypothesis even if its reconstructed 3-D centre happens to sit on the
    temporal bracket; allowing it into a depth replacement would silently swap
    identities.
    """
    first_views = first.get("views", {})
    second_views = second.get("views", {})
    shared = set(first_views) & set(second_views)
    if not shared:
        return False, 0
    same = 0
    for camera in shared:
        a = first_views.get(camera, {})
        b = second_views.get(camera, {})
        try:
            if int(a.get("detection_index", -1)) == int(b.get("detection_index", -2)):
                same += 1
                continue
        except (TypeError, ValueError):
            pass
        # Some prediction producers omit detection_index.  In that case an
        # effectively identical 2-D pose is still the same observation.
        pa = np.asarray(a.get("joints_2d", []), dtype=np.float64)
        pb = np.asarray(b.get("joints_2d", []), dtype=np.float64)
        if pa.shape == (21, 2) and pb.shape == (21, 2):
            finite = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
            if np.count_nonzero(finite) >= 5 and float(np.median(
                np.linalg.norm(pa[finite] - pb[finite], axis=1)
            )) <= 12.0:
                same += 1
    return same >= max(1, min(2, len(shared))), same


def _final_sequence_neighbours(
    ordered: list[dict[str, Any]], position: int, side: int, radius: int,
    excluded_positions: set[int] | None = None,
) -> tuple[
    tuple[dict[str, Any], dict[str, Any], int] | None,
    tuple[dict[str, Any], dict[str, Any], int] | None,
]:
    """Find nearest finite same-side rows before and after a candidate."""
    frame = int(ordered[position]["sync_index"])
    found: list[tuple[dict[str, Any], dict[str, Any], int] | None] = [None, None]
    for direction, slot in ((-1, 0), (1, 1)):
        cursor = position + direction
        while 0 <= cursor < len(ordered):
            if excluded_positions and cursor in excluded_positions:
                cursor += direction
                continue
            gap = abs(int(ordered[cursor]["sync_index"]) - frame)
            if gap > radius:
                break
            hand = next((value for value in ordered[cursor].get("hands", [])
                         if int(value.get("side", -1)) == side
                         and np.all(np.isfinite(_final_sequence_hand_center(value)))), None)
            if hand is not None:
                found[slot] = (ordered[cursor], hand, gap)
                break
            cursor += direction
    return found[0], found[1]


def _refresh_row_hand_metadata(row: dict[str, Any]) -> None:
    """Synchronize row cardinality/camera fields after a per-hand edit."""
    hands = list(row.get("hands", []))
    present = {
        int(hand.get("side", -1)) for hand in hands
        if int(hand.get("side", -1)) in (0, 1)
    }
    cameras = sorted({
        str(camera)
        for hand in hands
        for camera in hand.get("camera_ids", hand.get("views", {}).keys())
    })
    row["active_cameras"] = cameras
    row["active_camera_count"] = len(cameras)
    if present == {0, 1}:
        row.pop("missing_sides", None)
        row.pop("partial_hand_acceptance", None)
    else:
        row["partial_hand_acceptance"] = True
        row["missing_sides"] = sorted({0, 1} - present)


def _final_sequence_guard(
    rows: list[dict[str, Any]],
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> dict[str, int]:
    """Validate the delivered post-anatomy trajectories one hand at a time.

    This is intentionally a *validator*, not a low-pass filter.  It only acts
    on an isolated frame whose shared-camera 2-D track is continuous but whose
    3-D palm jumps.  A safe alternate branch is regenerated from the original
    detections and accepted only if the bidirectional 3-D bracket improves;
    when no alternate is safe, the offending hand is removed while any other
    hand in the row remains intact.  Real fast gestures (large 2-D motion) are
    never flattened.
    """
    result = {
        "checked_hand_count": 0,
        "suspicious_hand_count": 0,
        "replaced_hand_count": 0,
        "removed_hand_count": 0,
        "reentry_count": 0,
    }
    if not getattr(args, "final_sequence_guard", 1) or len(rows) < 2:
        return result
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    # Rows/hands already marked unreconstructable cannot be used as temporal
    # brackets.  The normal pipeline prunes them before entering this guard;
    # keep the exclusion local as a defensive post-condition for direct
    # callers and future pipeline orderings.
    excluded_positions: set[int] = set()
    for index, candidate_row in enumerate(ordered):
        usable = False
        for candidate_hand in candidate_row.get("hands", []):
            points = np.asarray(
                candidate_hand.get("joints_base_m", []), dtype=np.float64
            )
            if points.shape != (21, 3):
                continue
            if candidate_hand.get("anatomy_refinement", {}).get(
                "dropped_unreconstructable"
            ):
                continue
            if np.count_nonzero(np.isfinite(points).all(axis=1)) >= 3:
                usable = True
                break
        if not usable:
            excluded_positions.add(index)
    radius = max(int(getattr(args, "final_sequence_max_gap", 6)), 1)
    max_step = max(float(getattr(args, "final_sequence_max_step_m", 0.115)), 0.0)
    max_2d = max(float(getattr(args, "final_sequence_max_2d_motion_px", 75.0)), 0.0)
    delete_step = max(float(getattr(args, "final_sequence_delete_step_m", 0.24)), max_step)
    min_improvement = max(float(getattr(args, "final_sequence_min_improvement_m", 0.025)), 0.0)
    max_targets = max(int(getattr(args, "final_sequence_max_targets", 48)), 1)
    # A replacement is only a conservative depth correction.  If the
    # alternate branch would move the palm by a large fraction of a hand in a
    # single frame, the evidence is indistinguishable from an identity/re-entry
    # error and the branch selector should remain the authority.  In
    # particular, do not let this post-anatomy pass turn a 30--50 mm raw pose
    # into a 100+ mm jump simply because a stale bracket looked cheaper.
    replacement_max_shift = max(
        float(getattr(args, "final_sequence_max_replacement_shift_m", 0.080)),
        0.0,
    )
    targets: list[tuple[
        float, int, int, dict[str, Any], dict[str, Any], dict[str, Any], int,
        dict[str, Any], dict[str, Any], dict[str, Any], int, int,
    ]] = []
    for position, row in enumerate(ordered):
        for side in (0, 1):
            hand = next((value for value in row.get("hands", [])
                         if int(value.get("side", -1)) == side
                         and np.all(np.isfinite(_final_sequence_hand_center(value)))), None)
            if hand is None:
                continue
            result["checked_hand_count"] += 1
            before, after = _final_sequence_neighbours(
                ordered, position, side, radius, excluded_positions
            )
            if before is None or after is None:
                continue
            previous_row, previous, gap_before = before
            following_row, following, gap_after = after
            current_center = _final_sequence_hand_center(hand)
            previous_center = _final_sequence_hand_center(previous)
            following_center = _final_sequence_hand_center(following)
            if not (np.all(np.isfinite(current_center)) and np.all(np.isfinite(previous_center))
                    and np.all(np.isfinite(following_center))):
                continue
            step_before = float(np.linalg.norm(current_center - previous_center)) / max(gap_before, 1)
            step_after = float(np.linalg.norm(following_center - current_center)) / max(gap_after, 1)
            image_before, shared_before = _final_sequence_2d_motion(previous, hand, gap_before)
            image_after, shared_after = _final_sequence_2d_motion(hand, following, gap_after)
            if (not np.isfinite(image_before) or not np.isfinite(image_after)
                    or max(image_before, image_after) > max_2d
                    or max(step_before, step_after) <= max_step):
                # A large image move is a genuine gesture/re-entry; do not
                # smooth it.  Lack of 2-D support is similarly ambiguous.
                if (np.isfinite(image_before) and np.isfinite(image_after)
                        and max(image_before, image_after) >= max(
                            float(getattr(args, "identity_2d_hard_motion_px", 125.0)),
                            max_2d,
                        )):
                    result["reentry_count"] += 1
                continue
            alpha = gap_before / float(max(gap_before + gap_after, 1))
            prediction = (1.0 - alpha) * previous_center + alpha * following_center
            bracket = float(np.linalg.norm(current_center - prediction))
            severity = max(step_before, step_after, bracket)
            targets.append((
                severity, position, side, hand, previous_row, previous,
                gap_before, following_row, following, row, gap_after,
                shared_before + shared_after,
            ))
    targets.sort(reverse=True, key=lambda value: value[0])
    targets = targets[:max_targets]
    for (
        severity, position, side, hand, previous_row, previous, gap_before,
        following_row, following, row, gap_after, _shared,
    ) in targets:
        result["suspicious_hand_count"] += 1
        original_points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
        if original_points.shape != (21, 3):
            continue
        # Re-enumerate only this side.  Candidates are built from untouched
        # detector observations, so anatomy output cannot recursively feed the
        # branch selector.
        candidates = _partial_frame_candidates(
            int(row["sync_index"]), prediction_rows, calibrations,
            cameras, preferred_anchors, args,
        )
        physical_side = int(hand.get("side", side))
        safe: list[tuple[float, dict[str, Any], dict[str, Any], float]] = []
        previous_center = _final_sequence_hand_center(previous)
        following_center = _final_sequence_hand_center(following)
        alpha = gap_before / float(max(gap_before + gap_after, 1))
        prediction = (1.0 - alpha) * previous_center + alpha * following_center
        original_residual = float(np.linalg.norm(
            _final_sequence_hand_center(hand) - prediction
        ))
        for candidate in candidates:
            if not _track_partial_candidate_safe(candidate, args):
                continue
            candidate_hand = _partial_candidate_to_hand(candidate, physical_side, calibrations)
            candidate_center = _final_sequence_hand_center(candidate_hand)
            if not np.all(np.isfinite(candidate_center)):
                continue
            # Branch enumeration can return a perfectly valid *other hand*.
            # Its 3-D centre may be closer to the temporal bracket, but
            # replacing the current side with it is an identity swap in
            # disguise (this happened at frames 4/5 and 272/273).  A depth
            # alternate must therefore explain the same image-space track as
            # the delivered observation before it can be considered.  The
            # detector labels are intentionally ignored here; shared pixels
            # are the stronger identity evidence.
            candidate_image_motion, candidate_shared = _final_sequence_2d_motion(
                hand, candidate_hand, 1
            )
            same_observation, same_observation_count = _same_image_observation(
                hand, candidate_hand
            )
            original_view_count = len(hand.get("views", {}))
            required_shared = 2 if original_view_count >= 2 else 1
            image_limit = min(
                max_2d,
                max(float(getattr(
                    args, "identity_2d_hard_motion_px", 125.0
                )), max_2d),
            )
            if (
                not np.isfinite(candidate_image_motion)
                or candidate_shared < required_shared
                or candidate_image_motion > image_limit
                or not same_observation
            ):
                continue
            candidate_residual = float(np.linalg.norm(candidate_center - prediction))
            candidate_shift = float(np.linalg.norm(candidate_center - current_center))
            if candidate_shift > replacement_max_shift:
                continue
            before_motion, _ = _final_sequence_2d_motion(previous, candidate_hand, gap_before)
            after_motion, _ = _final_sequence_2d_motion(candidate_hand, following, gap_after)
            candidate_before = float(np.linalg.norm(candidate_center - previous_center)) / max(gap_before, 1)
            candidate_after = float(np.linalg.norm(following_center - candidate_center)) / max(gap_after, 1)
            if (not np.isfinite(before_motion) or not np.isfinite(after_motion)
                    or max(before_motion, after_motion) > max_2d
                    or max(candidate_before, candidate_after) > delete_step):
                continue
            safe.append((candidate_residual, candidate, candidate_hand, float(
                original_residual - candidate_residual
            )))
        safe.sort(key=lambda item: item[0])
        if safe and safe[0][3] >= min_improvement:
            _residual, candidate, candidate_hand, improvement = safe[0]
            # A replacement is regenerated from raw detections and therefore
            # lacks post-anatomy/tracking diagnostics.  Preserve the delivered
            # hand's provenance while replacing only geometry-owned fields.
            preserved_metadata = {
                key: value for key, value in hand.items()
                if key in {
                    "identity_tracking", "identity_tracking_2d",
                    "depth_regularization", "tracking_refinement",
                }
            }
            previous_anatomy = hand.get("anatomy_refinement")
            previous_points = original_points.tolist()
            hand.clear()
            hand.update(candidate_hand)
            hand.update(preserved_metadata)
            # The raw alternate has not itself gone through anatomy fitting.
            # Keep the old diagnostics only as provenance, never as a claim
            # that they describe the newly selected coordinates.
            hand["unrefined_joints_base_m"] = previous_points
            if previous_anatomy is not None:
                hand["pre_guard_anatomy_refinement"] = previous_anatomy
            hand["final_sequence_guard"] = {
                "action": "replace_depth_branch",
                "improvement_m": improvement,
                "previous_sync_index": int(previous_row["sync_index"]),
                "following_sync_index": int(following_row["sync_index"]),
            }
            _refresh_row_hand_metadata(row)
            result["replaced_hand_count"] += 1
            continue
        # If a continuous image track makes a truly severe 3-D excursion and
        # no safe branch exists, remove only this side.  Keeping the row's
        # other hand is the important cardinality guarantee.
        if severity >= delete_step:
            hands = [value for value in row.get("hands", []) if value is not hand]
            if len(hands) != len(row.get("hands", [])):
                row["hands"] = hands
                result["removed_hand_count"] += 1
                _refresh_row_hand_metadata(row)
    return result


def _stabilize_hand_identities(
    rows: list[dict[str, Any]], max_step_m: float = 0.22,
    partial_switch_penalty: float = 1.50,
    partial_switch_max_cost: float = 3.0,
    args: argparse.Namespace | None = None,
) -> int:
    """Globally assign fused observations to persistent physical hand tracks.

    The former implementation made a greedy decision at each frame.  A single
    detector flip or depth branch could therefore poison the next velocity
    estimate and cause repeated left/right swaps.  This pass solves a small
    dynamic-programming problem over all rows, evaluating both permutations for
    dual-hand rows and both side choices for partial rows.  Detector labels are
    only a weak tie-breaker; palm, wrist and normalized shape continuity drive
    the assignment.
    """
    ordered = sorted(rows, key=lambda row: int(row["sync_index"]))
    infos: list[tuple[dict[str, Any], list[dict[str, Any]], list[tuple[int, int]]]] = []
    for row in ordered:
        original = list(row.get("hands", []))
        usable = [
            hand for hand in original
            if np.all(np.isfinite(_hand_center_from_row(hand)))
        ]
        if len(usable) > 2:
            usable = sorted(
                usable,
                key=lambda hand: float(hand.get("quality", {}).get(
                    "valid_joint_count", 0
                )),
                reverse=True,
            )[:2]
        infos.append((row, usable, _identity_states(usable)))

    if not infos:
        return 0
    dynamic_costs: list[np.ndarray] = []
    back_links: list[np.ndarray | None] = []
    first_row, first_hands, first_states = infos[0]
    # Pick a deterministic orientation for the first dual-hand observation.
    # Without this small tie-break, two equally cheap paths can backtrack to a
    # relabelled first frame and make the reported left/right semantics depend
    # on argmin's array-order detail.
    first_priors = np.asarray([
        _identity_state_prior(first_row, first_hands, state)
        for state in first_states
    ], dtype=np.float64)
    if len(first_priors) > 1:
        first_priors += 0.20 * np.arange(len(first_priors), dtype=np.float64)
    dynamic_costs.append(first_priors)
    back_links.append(None)
    for position in range(1, len(infos)):
        row, hands, states = infos[position]
        previous_row, previous_hands, previous_states = infos[position - 1]
        gap = int(row["sync_index"]) - int(previous_row["sync_index"])
        matrix = np.full((len(previous_states), len(states)), np.inf, dtype=np.float64)
        for previous_index, previous_state in enumerate(previous_states):
            for current_index, current_state in enumerate(states):
                transition = _identity_transition_cost_states(
                    previous_hands, previous_state, hands, current_state,
                    gap, max_step_m,
                    partial_switch_penalty,
                    partial_switch_max_cost,
                    args,
                )
                if np.isfinite(transition):
                    matrix[previous_index, current_index] = (
                        dynamic_costs[-1][previous_index]
                        + transition
                        + _identity_state_prior(row, hands, current_state)
                    )
        back_links.append(np.argmin(matrix, axis=0))
        dynamic_costs.append(np.min(matrix, axis=0))

    state_index = int(np.argmin(dynamic_costs[-1]))
    chosen_indices = [state_index]
    for links in reversed(back_links[1:]):
        assert links is not None
        state_index = int(links[state_index])
        chosen_indices.append(state_index)
    chosen_indices.reverse()

    changed = 0
    for position, ((row, hands, states), selected_index) in enumerate(
        zip(infos, chosen_indices)
    ):
        state = states[selected_index]
        assigned_hands = []
        assignment_records = []
        for side, hand_index in enumerate(state):
            if hand_index < 0 or hand_index >= len(hands):
                continue
            hand = hands[hand_index]
            previous_side = hand.get("side")
            try:
                previous_side_int = int(previous_side)
            except (TypeError, ValueError):
                previous_side_int = side
            if previous_side_int != side:
                changed += 1
            hand["side"] = int(side)
            assigned_hands.append(hand)
            assignment_records.append({
                "from_side": previous_side_int,
                "to_side": int(side),
            })
        row["hands"] = sorted(
            assigned_hands, key=lambda hand: int(hand.get("side", 0))
        )
        present = {int(hand["side"]) for hand in assigned_hands}
        if len(present) < 2:
            row["partial_hand_acceptance"] = True
            row["missing_sides"] = sorted({0, 1} - present)
        else:
            row.pop("missing_sides", None)
        if assigned_hands:
            row["identity_tracking"] = {
                "algorithm": "global_trajectory_identity_assignment_v2",
                "relabelled": any(
                    record["from_side"] != record["to_side"]
                    for record in assignment_records
                ),
                "assignment_cost": float(
                    dynamic_costs[position][chosen_indices[position]]
                ),
                "assignments": assignment_records,
            }
        elif row.get("hands"):
            row["hands"] = []
            row["partial_hand_acceptance"] = True
            row["missing_sides"] = [0, 1]
    return changed


def _prune_unusable_hands(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove hands that anatomy refinement could not reconstruct.

    A rejected hand is represented internally as an all-NaN 21-joint array so
    temporal passes cannot accidentally reuse it.  Before serialization it
    must also disappear from the row; otherwise renderers and downstream MANO
    export see a bbox with no valid 3-D pose and show a one-frame flash.
    """
    kept_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for row in rows:
        original_hands = list(row.get("hands", []))
        valid_hands = []
        for hand in original_hands:
            points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
            finite = (
                points.shape == (21, 3)
                and np.isfinite(points).all(axis=1)
            )
            anatomy = hand.get("anatomy_refinement", {})
            if anatomy.get("dropped_unreconstructable") or np.count_nonzero(finite) < 3:
                continue
            valid_hands.append(hand)
        if not valid_hands:
            rejected = dict(row)
            rejected["reason"] = "all_hands_unreconstructable"
            rejected["dropped_hand_count"] = len(original_hands)
            dropped_rows.append(rejected)
            continue
        if len(valid_hands) != len(original_hands):
            row["hands"] = valid_hands
            row["partial_hand_acceptance"] = True
            row["missing_sides"] = sorted(
                {0, 1} - {int(hand["side"]) for hand in valid_hands}
            )
            row["active_cameras"] = sorted({
                camera for hand in valid_hands for camera in hand.get("camera_ids", [])
            })
            row["active_camera_count"] = len(row["active_cameras"])
        kept_rows.append(row)
    return kept_rows, dropped_rows


def _recover_from_reference(
    sync_index: int,
    reference: dict[str, Any],
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    args: argparse.Namespace,
    sides: tuple[int, ...] | None = None,
) -> dict[str, Any] | None:
    requested_sides = tuple(sides or (0, 1))
    reference_points = {}
    for hand in reference.get("hands", []):
        side = int(hand.get("side", -1))
        points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
        if side in (0, 1) and points.shape == (21, 3):
            if np.count_nonzero(np.isfinite(points).all(axis=1)) >= args.min_valid_joints:
                reference_points[side] = points
    requested_sides = tuple(side for side in requested_sides if side in reference_points)
    if not requested_sides:
        return None
    selected_by_side: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    association_errors: dict[int, dict[str, float]] = {0: {}, 1: {}}
    results: dict[int, dict[str, Any]] = {}
    frame_gap = abs(sync_index - int(reference["sync_index"]))
    max_wrist_step = args.max_temporal_wrist_step_m * max(frame_gap, 1)
    for side in requested_sides:
        seed = {"points": reference_points[side]}
        for camera in cameras:
            hand, error = _match_camera_single_side(
                groups[camera], side, seed["points"], calibrations[camera],
                args.temporal_association_threshold_px,
                args.max_side_detections, args.detector_handedness,
            )
            if hand is not None:
                selected_by_side[side][camera] = hand
                association_errors[side][camera] = float(error)
        if len(selected_by_side[side]) < 2:
            continue
        result = _triangulate_hand(
            side, selected_by_side[side], calibrations,
            args.ransac_threshold_px, 2,
        )
        if _quality_reasons([result], args) or _depth_condition_reasons(
            result, len(selected_by_side[side]), args
        ):
            selected_by_side[side].clear()
            association_errors[side].clear()
            continue
        wrist = result["points"][0]
        reference_wrist = reference_points[side][0]
        if not (np.all(np.isfinite(wrist)) and np.all(np.isfinite(reference_wrist))):
            selected_by_side[side].clear()
            association_errors[side].clear()
            continue
        if np.linalg.norm(wrist - reference_wrist) > max_wrist_step:
            selected_by_side[side].clear()
            association_errors[side].clear()
            continue
        results[side] = result
    if not results:
        return None

    common = set().union(*(set(selected_by_side[side]) for side in results))
    preferred = tuple(args.anchor_cameras)
    if all(camera in common for camera in preferred):
        comparison_pair = preferred
    elif len(common) >= 2:
        comparison_pair = tuple(sorted(common)[:2])
    else:
        first_side = next(iter(results))
        available = tuple(selected_by_side[first_side])
        comparison_pair = tuple(available[:2]) if len(available) >= 2 else preferred
    hands = []
    for side in sorted(results):
        result = results[side]
        hand = _serialize_hand(
            result, selected_by_side[side], calibrations, comparison_pair
        )
        hand["association_error_px"] = association_errors[side]
        hand["fusion_mode"] = "temporal_guided_multiview"
        hands.append(hand)
    active = sorted(set().union(*(hand["camera_ids"] for hand in hands)))
    present = {int(hand["side"]) for hand in hands}
    output = {
        "sync_index": sync_index,
        "anchor_cameras": list(comparison_pair),
        "anchor_baseline_m": _anchor_baseline_m(comparison_pair, calibrations),
        "recovery": {
            "method": "nearest_confirmed_multiview",
            "reference_sync_index": int(reference["sync_index"]),
            "frame_gap": frame_gap,
        },
        "active_cameras": active,
        "active_camera_count": len(active),
        "hands": hands,
    }
    if len(present) < 2:
        output["partial_hand_acceptance"] = True
        output["missing_sides"] = sorted({0, 1} - present)
    return output


def _evaluate_anchor_pairs(
    anchor_pairs: list[tuple[str, str]],
    complete_by_camera: dict[str, list[int]],
    groups: dict[str, dict[int, dict[int, dict[str, Any]]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
    allow_sparse: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for anchor_pair in anchor_pairs:
        pair_options = [
            _ordered_hand_pairs(
                complete_by_camera[camera], groups[camera], args.detector_handedness
            )
            for camera in anchor_pair
        ]
        for detection_pairs in itertools.product(*pair_options):
            candidate = _evaluate_anchor_assignment(
                anchor_pair, detection_pairs, groups, calibrations, cameras, args,
                allow_sparse,
            )
            if np.isfinite(candidate["cost"]):
                missing_preferred = len(set(preferred_anchors) - set(anchor_pair))
                candidate["cost"] += 5.0 * missing_preferred
                results.append(candidate)
    return results


def _all_frame_hypotheses(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Return every geometrically usable assignment for one frame.

    The normal fusion path intentionally tries the preferred stereo pair first
    and only falls back when that pair is unusable.  That is a good default for
    isolated frames, but it hides alternate depth branches from a trajectory
    tracker.  The continuity pass therefore enumerates every available pair and
    lets a sequence-level cost decide between close reprojection hypotheses.
    """
    groups = {
        camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras
    }
    # Continuity is only invoked on suspicious frames.  Keeping the two most
    # confident detector boxes and prioritising the preferred anchor pair
    # avoids an exponential scan over duplicate low-confidence boxes in six
    # camera recordings.
    continuity_detection_limit = min(args.max_anchor_detections, 2)
    complete_by_camera = {
        camera: _complete_detections(groups[camera], continuity_detection_limit)
        for camera in cameras
    }
    all_anchor_pairs = [
        pair for pair in itertools.combinations(cameras, 2)
        if all(len(complete_by_camera[camera]) >= 2 for camera in pair)
    ]
    if not all_anchor_pairs:
        return []
    anchor_pairs = []
    if preferred_anchors in all_anchor_pairs:
        anchor_pairs.append(preferred_anchors)
    anchor_pairs.extend(
        pair for pair in all_anchor_pairs
        if pair != preferred_anchors
        and (pair[0] in preferred_anchors or pair[1] in preferred_anchors)
    )
    # Keep the preferred pair, the widest available pairs, and one short pair
    # as a fallback.  This covers both conditioning extremes without the
    # exponential cost of evaluating every pair/detection permutation on a
    # long recording.
    anchor_pairs.extend(pair for pair in all_anchor_pairs if pair not in anchor_pairs)
    pair_limit = max(int(getattr(args, "continuity_max_anchor_pairs", 8)), 1)
    preferred = [preferred_anchors] if preferred_anchors in anchor_pairs else []
    remaining = [pair for pair in anchor_pairs if pair != preferred_anchors]
    remaining.sort(
        key=lambda pair: _anchor_baseline_m(pair, calibrations), reverse=True
    )
    if remaining and len(remaining) >= pair_limit:
        # Retain the shortest pair as a diagnostic alternative too; it can be
        # useful when wide cameras are occluded in a side-on pose.
        shortest = min(remaining, key=lambda pair: _anchor_baseline_m(pair, calibrations))
        wide = [pair for pair in remaining if pair != shortest]
        keep = wide[: max(pair_limit - len(preferred) - 1, 0)]
        if len(keep) + len(preferred) < pair_limit:
            keep.append(shortest)
        anchor_pairs = preferred + keep[: max(pair_limit - len(preferred), 0)]
    else:
        anchor_pairs = (preferred + remaining)[:pair_limit]
    hypotheses = _evaluate_anchor_pairs(
        anchor_pairs, complete_by_camera, groups, calibrations, cameras,
        preferred_anchors, args, allow_sparse=True,
    )
    hypotheses = [
        candidate for candidate in hypotheses
        if np.isfinite(candidate.get("cost", float("inf")))
        and _hypothesis_branch_quality(candidate, args)
    ]
    hypotheses.sort(key=lambda candidate: float(candidate.get("cost", np.inf)))
    return hypotheses[:max(int(getattr(args, "continuity_max_hypotheses", 32)), 1)]


def _hypothesis_center(result: dict[str, Any]) -> np.ndarray:
    points = np.asarray(result["points"], dtype=np.float64)
    palm = points[[0, 5, 9, 13, 17]]
    finite = np.isfinite(palm).all(axis=1)
    return np.mean(palm[finite], axis=0) if np.any(finite) else np.full(3, np.nan)


def _hypothesis_transition_cost(
    previous: dict[str, Any], current: dict[str, Any], gap: int, args: argparse.Namespace,
) -> float:
    """Robust transition cost for full *and* partial hand hypotheses.

    Detector order is not a physical identity.  The previous implementation
    handled two-hand rows permutation-invariantly, but silently assigned zero
    cost to a missing side.  That made ``full -> partial -> full`` boundaries
    invisible to the depth selector.  Match the available observations first,
    then charge only a small explicit penalty for a genuinely missing track.
    """
    gap = max(int(gap), 1)

    def pair_cost(
        before_result: dict[str, Any], after_result: dict[str, Any],
        before_selected: Mapping[str, Any] | None = None,
        after_selected: Mapping[str, Any] | None = None,
    ) -> float:
        cost = 0.0
        before = _hypothesis_center(before_result)
        after = _hypothesis_center(after_result)
        if not (np.all(np.isfinite(before)) and np.all(np.isfinite(after))):
            return cost
        if before_selected and after_selected:
            contradictory, _diagnostics = _depth_reconnect_contradiction(
                {"joints_base_m": before_result.get("points"),
                 "views": dict(before_selected)},
                {"joints_base_m": after_result.get("points"),
                 "views": dict(after_selected)},
                gap,
                args,
            )
            if contradictory:
                return max(float(getattr(
                    args, "track_reconnect_depth_penalty", 2500.0
                )), 0.0)
        distance = float(np.linalg.norm(after - before))
        excess = max(0.0, distance - args.continuity_max_step_m * gap)
        # A capped robust loss prevents a genuinely fast, well-supported motion
        # from being rejected merely because it exceeds the nominal step.
        cost += args.continuity_weight * min(excess * excess, 0.25)
        wrist_before = np.asarray(before_result["points"][0])
        wrist_after = np.asarray(after_result["points"][0])
        if np.all(np.isfinite(wrist_before)) and np.all(np.isfinite(wrist_after)):
            wrist_distance = float(np.linalg.norm(wrist_after - wrist_before))
            wrist_excess = max(0.0, wrist_distance - args.continuity_wrist_max_step_m * gap)
            cost += 0.45 * args.continuity_weight * min(wrist_excess * wrist_excess, 0.25)
        # A branch can preserve the palm centre while sending a finger/depth
        # subset elsewhere.  Include a robust all-joint term, but keep it much
        # weaker than the palm term so articulation is not flattened.
        before_points = np.asarray(before_result.get("points", []), dtype=np.float64)
        after_points = np.asarray(after_result.get("points", []), dtype=np.float64)
        if before_points.shape == (21, 3) and after_points.shape == (21, 3):
            finite = np.isfinite(before_points).all(axis=1) & np.isfinite(after_points).all(axis=1)
            if np.count_nonzero(finite) >= 8:
                joint_distance = float(np.median(
                    np.linalg.norm(after_points[finite] - before_points[finite], axis=1)
                ))
                joint_excess = max(0.0, joint_distance - 0.16 * gap)
                cost += 0.05 * args.continuity_weight * min(joint_excess * joint_excess, 0.25)
        if before_selected and after_selected:
            cost += 1.10 * _hand_image_track_cost(
                {"views": dict(before_selected)},
                {"views": dict(after_selected)},
                gap,
            )
        return cost

    previous_selected_by_side = previous.get("selected", {})
    current_selected_by_side = current.get("selected", {})
    previous_results = [
        (side, result, previous_selected_by_side.get(side, {}))
        for side, result in enumerate(previous.get("results", []))
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    current_results = [
        (side, result, current_selected_by_side.get(side, {}))
        for side, result in enumerate(current.get("results", []))
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    if not previous_results and not current_results:
        cost = 0.0
    elif previous_results and current_results:
        # There are at most two physical hands.  Enumerating the tiny matching
        # problem is clearer and, importantly, also covers one-hand rows.
        best = float("inf")
        match_count = min(len(previous_results), len(current_results))
        for previous_indices in itertools.combinations(
            range(len(previous_results)), match_count
        ):
            for permutation in itertools.permutations(
                range(len(current_results)), match_count
            ):
                value = 0.0
                for index, current_index in enumerate(permutation):
                    value += pair_cost(
                        previous_results[previous_indices[index]][1],
                        current_results[current_index][1],
                        previous_results[previous_indices[index]][2],
                        current_results[current_index][2],
                    )
                value += 0.18 * min(gap, 5) * abs(
                    len(previous_results) - len(current_results)
                )
                best = min(best, value)
        cost = best
    else:
        # A hand appearing/disappearing is expected during occlusion.  Keep
        # the cost finite but non-zero so a branch with a continuous visible
        # hand wins over an arbitrary detector relabel.
        cost = 0.18 * min(gap, 5) * max(
            len(previous_results), len(current_results), 1
        )
    if tuple(previous["anchor_cameras"]) != tuple(current["anchor_cameras"]):
        cost += args.continuity_switch_penalty
        # Do not let a short-baseline branch replace a well-conditioned branch
        # for one frame merely because its reprojection is a fraction lower.
        # The term is directional: switching to a wider pair is harmless,
        # while dropping from (say) 130 mm to 40 mm is a classic depth jump.
        try:
            previous_baseline = float(previous.get("anchor_baseline_m", 0.0) or 0.0)
            current_baseline = float(current.get("anchor_baseline_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            previous_baseline = current_baseline = 0.0
        if previous_baseline > 0.0 and current_baseline > 0.0:
            baseline_drop = max(0.0, previous_baseline - current_baseline)
            cost += 1800.0 * baseline_drop * baseline_drop
    return float(cost)


def _hypothesis_support(candidate: dict[str, Any]) -> float:
    values = []
    for result in candidate["results"]:
        counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
        finite = counts[np.isfinite(counts) & (counts >= 2)]
        if finite.size:
            values.append(float(np.median(finite)))
    return float(np.mean(values)) if values else 0.0


def _hypothesis_bracket_residual(
    previous: dict[str, Any],
    current: dict[str, Any],
    following: dict[str, Any],
    gap_before: int,
    gap_after: int,
) -> float:
    """Return the best two-hand residual to a zero-phase temporal bracket.

    The first-order continuity term cannot see a ``V`` shaped excursion: a
    wrong depth branch may be close to one neighbour and still win on raw
    reprojection cost.  Comparing the candidate to the interpolation between
    both neighbours exposes that excursion while remaining permutation
    invariant to detector left/right labels.
    """
    previous_centres = [
        _hypothesis_center(result) for result in previous.get("results", [])
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    current_centres = [
        _hypothesis_center(result) for result in current.get("results", [])
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    following_centres = [
        _hypothesis_center(result) for result in following.get("results", [])
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    if not (previous_centres and current_centres and following_centres):
        return 0.0
    gap_before = max(int(gap_before), 1)
    gap_after = max(int(gap_after), 1)
    alpha = gap_before / float(gap_before + gap_after)
    best = float("inf")
    match_count = min(len(previous_centres), len(current_centres), len(following_centres))
    # A partial row has one current centre.  For a full row, enumerate both
    # permutations; for unequal cardinalities enumerate the available subset
    # so a missing hand does not erase the useful bracket for the visible one.
    for current_indices in itertools.combinations(range(len(current_centres)), match_count):
        for previous_indices in itertools.permutations(range(len(previous_centres)), match_count):
            for following_indices in itertools.permutations(range(len(following_centres)), match_count):
                residuals = []
                for index in range(match_count):
                    before = previous_centres[previous_indices[index]]
                    after = following_centres[following_indices[index]]
                    candidate = current_centres[current_indices[index]]
                    prediction = (1.0 - alpha) * before + alpha * after
                    residuals.append(float(np.linalg.norm(candidate - prediction)))
                best = min(best, float(np.sum(residuals)))
    return 0.0 if not np.isfinite(best) else best


def _hypothesis_matched_depth_residual(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Return the best summed palm-centre distance for available hands."""
    reference_centres = [
        _hypothesis_center(result) for result in reference.get("results", [])
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    candidate_centres = [
        _hypothesis_center(result) for result in candidate.get("results", [])
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    if not reference_centres or not candidate_centres:
        return float("inf")
    count = min(len(reference_centres), len(candidate_centres))
    best = float("inf")
    for reference_indices in itertools.combinations(
        range(len(reference_centres)), count
    ):
        for candidate_indices in itertools.permutations(
            range(len(candidate_centres)), count
        ):
            value = sum(
                float(np.linalg.norm(
                    reference_centres[reference_indices[index]]
                    - candidate_centres[candidate_indices[index]]
                ))
                for index in range(count)
            )
            best = min(best, value)
    return best


def _track_reselection_depth_improvement(
    original: dict[str, Any],
    candidate: dict[str, Any],
    previous: dict[str, Any] | None,
    following: dict[str, Any] | None,
    gap_before: int,
    gap_after: int,
) -> tuple[float, dict[str, float | None]]:
    """Measure whether an alternate fixes an actual local depth trajectory.

    Local Viterbi cost can differ by a few hundredths simply because detector
    hypotheses or anchor bookkeeping differ.  Replacing a hand for that tiny
    gain is harmful.  We require a measurable reduction in either the
    bidirectional bracket residual or the matched 3-D distance to an adjacent
    state before an alternate may be written.
    """
    original_bracket = candidate_bracket = None
    improvements: list[float] = []
    if previous is not None and following is not None:
        original_bracket = _hypothesis_bracket_residual(
            previous, original, following, gap_before, gap_after
        )
        candidate_bracket = _hypothesis_bracket_residual(
            previous, candidate, following, gap_before, gap_after
        )
        improvements.append(original_bracket - candidate_bracket)
    original_edge = candidate_edge = None
    edge_improvements: list[float] = []
    if previous is not None:
        original_value = _hypothesis_matched_depth_residual(previous, original)
        candidate_value = _hypothesis_matched_depth_residual(previous, candidate)
        if np.isfinite(original_value) and np.isfinite(candidate_value):
            edge_improvements.append(
                (original_value - candidate_value) / max(int(gap_before), 1)
            )
    if following is not None:
        original_value = _hypothesis_matched_depth_residual(original, following)
        candidate_value = _hypothesis_matched_depth_residual(candidate, following)
        if np.isfinite(original_value) and np.isfinite(candidate_value):
            edge_improvements.append(
                (original_value - candidate_value) / max(int(gap_after), 1)
            )
    if edge_improvements:
        # Both edges should benefit at a genuine isolated depth correction;
        # for a boundary with only one neighbour the sole edge is sufficient.
        original_edge = 0.0
        candidate_edge = -min(edge_improvements)
        improvements.append(min(edge_improvements))
    value = max(improvements) if improvements else float("-inf")
    return float(value), {
        "original_bracket_residual_m": original_bracket,
        "selected_bracket_residual_m": candidate_bracket,
        "minimum_edge_improvement_m_per_frame": (
            float(min(edge_improvements)) if edge_improvements else None
        ),
    }


def _hypothesis_bracket_penalty(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    following: dict[str, Any] | None,
    gap_before: int,
    gap_after: int,
    args: argparse.Namespace,
) -> float:
    if previous is None or following is None:
        return 0.0
    residual = _hypothesis_bracket_residual(
        previous, current, following, gap_before, gap_after
    )
    threshold = max(
        float(getattr(args, "continuity_bracket_residual_m", 0.030)), 0.0
    )
    weight = max(float(getattr(args, "continuity_bracket_weight", 150000.0)), 0.0)
    excess = max(0.0, residual - threshold)
    # Cap the robust loss so a genuinely fast, curved motion cannot dominate
    # the whole sequence; branch-sized isolated excursions still receive a
    # decisive penalty.
    return weight * min(excess * excess, 0.25)


def _hypothesis_branch_quality(
    candidate: dict[str, Any], args: argparse.Namespace,
) -> bool:
    """Return whether an alternate branch is safe enough for continuity use.

    A trajectory prior is allowed to rescue a branch that has a higher image
    cost, but it must not turn a nearly empty or anatomically impossible
    hypothesis into a synthetic hand.  This gate is intentionally weaker than
    the ordinary complete-row acceptance gate: side-on hands often lose
    several finger joints, yet their palm/depth branch can still be the only
    temporally coherent one.
    """
    return not _hypothesis_branch_quality_reasons(candidate, args)


def _hypothesis_branch_quality_reasons(
    candidate: dict[str, Any], args: argparse.Namespace,
) -> list[str]:
    """Explain why a continuity-only alternate branch is unsafe.

    The ordinary complete-row gate asks every hand for ``min_valid_joints``.
    That is intentionally too strict for a side-on glove: a bad detector
    branch can lose most finger rays while retaining a well-supported palm.
    Continuity selection is therefore gated in two stages: the whole
    hypothesis needs enough aggregate support, and each hand needs a reliable
    palm.  Finger support is allowed to be sparse, but a branch with no palm
    or with an implausible reprojection/geometry score is never synthesized.
    """
    reasons: list[str] = []
    results = list(candidate.get("results", []))
    if len(results) < 2:
        return ["too_few_hands"]

    aggregate_support = _hypothesis_support(candidate)
    minimum_support = float(getattr(args, "continuity_min_support", 2.5))
    if np.isfinite(aggregate_support) and aggregate_support < minimum_support:
        reasons.append("aggregate_support_too_low")

    minimum_valid = max(
        int(getattr(args, "continuity_min_branch_valid_joints", 8)), 1
    )
    minimum_palm_points = max(
        int(getattr(args, "continuity_min_palm_points", 3)), 3
    )
    minimum_palm_support = max(
        float(getattr(args, "continuity_min_palm_support", 2.0)), 0.0
    )
    palm_indices = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
    for side, result in enumerate(results):
        if int(result.get("valid_joints", 0)) < minimum_valid:
            reasons.append(f"side{side}_too_few_valid_joints")
        points = np.asarray(result.get("points", []), dtype=np.float64)
        if points.shape != (21, 3):
            reasons.append(f"side{side}_bad_point_shape")
            continue
        finite_points = np.isfinite(points).all(axis=1)
        finite_palm = finite_points[palm_indices]
        counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
        if counts.shape != (21,):
            counts = np.zeros(21, dtype=np.float64)
        reliable_palm = finite_palm & np.isfinite(counts[palm_indices]) & (
            counts[palm_indices] >= 2
        )
        if int(np.count_nonzero(reliable_palm)) < minimum_palm_points:
            reasons.append(f"side{side}_too_few_reliable_palm_points")
        palm_support = counts[palm_indices][reliable_palm]
        if palm_support.size and float(np.median(palm_support)) < minimum_palm_support:
            reasons.append(f"side{side}_palm_support_too_low")
        median = float(result.get("median_px", np.inf))
        p95 = float(result.get("p95_px", np.inf))
        if not np.isfinite(median) or not np.isfinite(p95):
            reasons.append(f"side{side}_nonfinite_reprojection")
            continue
        # Keep the same image-space quality ceiling as normal acceptance.  A
        # branch may be sparse, but it may not be an unconstrained reprojection
        # outlier merely because its temporal path is smooth.
        if median > float(getattr(args, "max_reprojection_median_px", 15.0)):
            reasons.append(f"side{side}_reprojection_median_too_large")
        if p95 > float(getattr(args, "max_reprojection_p95_px", 40.0)):
            reasons.append(f"side{side}_reprojection_p95_too_large")
        rejected = int(result.get("geometry_rejected_joints", 0))
        # Up to roughly half of the joints may be unavailable in a side-on
        # view.  A larger rejection count is treated as invalid rather than a
        # candidate for temporal hallucination.  Use the branch-valid count
        # here, not the ordinary complete-row threshold: 12 visible joints
        # (including 3 palm anchors) is a legitimate partial finger view.
        if rejected > max(10, 21 - minimum_valid):
            reasons.append(f"side{side}_too_many_geometry_rejections")
        for depth_reason in _depth_condition_reasons(
            result, len(candidate.get("selected", {}).get(side, {})), args
        ):
            reasons.append(f"side{side}_{depth_reason}")
        depth_level, depth_level_reasons = _depth_condition_level(
            result, len(candidate.get("selected", {}).get(side, {})), args
        )
        if depth_level == "unsafe":
            reasons.append(f"side{side}_unsafe_depth")
        if "weak_depth_pair_spread" in depth_level_reasons:
            reasons.append(f"side{side}_weak_depth_pair_spread")
    return sorted(set(reasons))


def _candidate_to_row(
    sync_index: int, candidate: dict[str, Any], calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace, margin: float | None,
) -> dict[str, Any]:
    chosen_anchors = tuple(candidate["anchor_cameras"])
    hands = []
    for side, result in enumerate(candidate["results"]):
        hand = _serialize_hand(
            result, candidate["selected"][side], calibrations, chosen_anchors
        )
        hand["association_error_px"] = candidate["association_errors"][side]
        hand["fusion_mode"] = (
            "multiview" if len(candidate["selected"][side]) >= 3 else "stereo_anchor"
        )
        hands.append(hand)
    active_cameras = sorted(set().union(*(hand["camera_ids"] for hand in hands)))
    return {
        "sync_index": sync_index,
        "anchor_cameras": list(chosen_anchors),
        "anchor_baseline_m": _anchor_baseline_m(chosen_anchors, calibrations),
        "assignment_cost": float(candidate["cost"]),
        "assignment_margin": margin,
        "active_cameras": active_cameras,
        "active_camera_count": len(active_cameras),
        "hands": hands,
    }


def _partial_candidate_to_hand(
    candidate: dict[str, Any],
    physical_side: int,
    calibrations: dict[str, CameraCalibration],
) -> dict[str, Any]:
    """Serialize one independent-side candidate without binding a label.

    ``_single_side_candidates`` uses the detector's ``is_right`` hypothesis as
    its local side index.  That index is not a physical identity.  Partial
    branch reselection therefore serializes the geometry first and writes the
    currently tracked physical side explicitly; the global identity pass can
    still change it later when the surrounding trajectory provides stronger
    evidence.
    """
    anchors = tuple(candidate["anchor_cameras"])
    hand = _serialize_hand(
        candidate["result"], candidate["selected"], calibrations, anchors
    )
    hand["side"] = int(physical_side)
    hand["association_error_px"] = {
        camera: float(value)
        for camera, value in candidate.get("association_errors", {}).items()
    }
    hand["fusion_mode"] = (
        "multiview" if len(candidate.get("selected", {})) >= 3
        else "stereo_anchor"
    )
    hand["side_selection_cost"] = float(candidate.get("cost", np.inf))
    return hand


def _partial_neighbor_hands(
    ordered: list[dict[str, Any]],
    position: int,
    direction: int,
    radius: int,
) -> tuple[list[dict[str, Any]], int] | tuple[None, None]:
    """Find the nearest non-empty hand row on one temporal side."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    frame = int(ordered[position]["sync_index"])
    radius = max(int(radius), 0)
    cursor = position + direction
    while 0 <= cursor < len(ordered):
        neighbour_frame = int(ordered[cursor]["sync_index"])
        gap = abs(neighbour_frame - frame)
        if gap > radius:
            break
        hands = [
            hand for hand in ordered[cursor].get("hands", [])
            if np.all(np.isfinite(_hand_center_from_row(hand)))
        ]
        if hands:
            return hands, gap
        cursor += direction
    return None, None


def _partial_context_cost(
    hand: dict[str, Any],
    before: list[dict[str, Any]] | None,
    gap_before: int | None,
    after: list[dict[str, Any]] | None,
    gap_after: int | None,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    """Score one partial pose against the nearest temporal observations.

    The matching is permutation-free for a one-hand row: when a neighbouring
    row contains two hands, try both and retain the pair that gives the
    smallest transition/bracket cost.  This lets a detector-side flip or a
    wrong depth branch be corrected without assuming that serialized side
    labels are already stable.
    """
    if not np.all(np.isfinite(_hand_center_from_row(hand))):
        return float("inf"), {"valid": False}
    transition_weight = max(
        float(getattr(args, "partial_continuity_transition_weight", 24.0)), 0.0
    )
    bracket_weight = max(
        float(getattr(args, "continuity_bracket_weight", 150000.0)), 0.0
    )
    bracket_threshold = max(
        float(getattr(args, "continuity_bracket_residual_m", 0.030)), 0.0
    )
    best = float("inf")
    best_detail: dict[str, Any] = {"valid": False}
    hand_center = _hand_center_from_row(hand)

    if before and after and gap_before is not None and gap_after is not None:
        gb = max(int(gap_before), 1)
        ga = max(int(gap_after), 1)
        alpha = gb / float(gb + ga)
        for previous in before:
            for following in after:
                transition_before = _identity_observation_cost(
                    previous, hand, gb, 0.22, args
                )
                transition_after = _identity_observation_cost(
                    hand, following, ga, 0.22, args
                )
                if not (
                    np.isfinite(transition_before)
                    and np.isfinite(transition_after)
                ):
                    continue
                previous_center = _hand_center_from_row(previous)
                following_center = _hand_center_from_row(following)
                prediction = (
                    (1.0 - alpha) * previous_center + alpha * following_center
                )
                residual = float(np.linalg.norm(hand_center - prediction))
                excess = max(0.0, residual - bracket_threshold)
                bracket = bracket_weight * min(excess * excess, 0.25)
                value = transition_weight * (
                    transition_before + transition_after
                ) + bracket
                if value < best:
                    best = value
                    best_detail = {
                        "valid": True,
                        "transition_before": float(transition_before),
                        "transition_after": float(transition_after),
                        "bracket_residual_m": residual,
                        "bracket_penalty": float(bracket),
                    }
    elif before and gap_before is not None:
        gap = max(int(gap_before), 1)
        for previous in before:
            transition = _identity_observation_cost(
                previous, hand, gap, 0.22, args
            )
            if np.isfinite(transition):
                value = transition_weight * transition
                if value < best:
                    best = value
                    best_detail = {
                        "valid": True,
                        "transition_before": float(transition),
                    }
    elif after and gap_after is not None:
        gap = max(int(gap_after), 1)
        for following in after:
            transition = _identity_observation_cost(
                hand, following, gap, 0.22, args
            )
            if np.isfinite(transition):
                value = transition_weight * transition
                if value < best:
                    best = value
                    best_detail = {
                        "valid": True,
                        "transition_after": float(transition),
                    }
    return best, best_detail


def _partial_frame_candidates(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Enumerate independent-side branches, including detector-flip rescue."""
    groups = {
        camera: _groups(prediction_rows[camera][sync_index])
        for camera in cameras
    }
    modes = [args.detector_handedness]
    if args.detector_handedness == "adaptive":
        modes = ["strict", "ignore"]
    candidates: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for mode in modes:
        local_args = argparse.Namespace(**vars(args))
        local_args.detector_handedness = mode
        for hypothesis_side in (0, 1):
            for candidate in _single_side_candidates(
                hypothesis_side,
                groups,
                calibrations,
                cameras,
                preferred_anchors,
                local_args,
                allow_sparse=True,
            ):
                centre = _hand_center_from_row({
                    "joints_base_m": candidate["result"]["points"].tolist()
                })
                selected_key = tuple(sorted(
                    (
                        str(camera),
                        int(hand.get("detection_index", -1)),
                        int(hand.get("hypothesis_index", -1)),
                        int(hand.get("is_right", hypothesis_side)),
                    )
                    for camera, hand in candidate.get("selected", {}).items()
                ))
                key = selected_key + tuple(
                    np.round(centre, 4).tolist()
                )
                current = seen.get(key)
                if current is None or float(candidate["cost"]) < float(current["cost"]):
                    seen[key] = candidate
    candidates.extend(seen.values())
    candidates.sort(key=lambda item: float(item.get("cost", np.inf)))
    return candidates[:max(int(getattr(args, "continuity_max_hypotheses", 32)), 1)]


def _partial_continuity_reselect(
    accepted: list[dict[str, Any]],
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> int:
    """Re-score one-hand rows with bidirectional depth/identity evidence.

    The ordinary Viterbi pass intentionally ignores partial rows because a
    two-hand hypothesis is unavailable.  That omission leaves exactly the
    short occlusion intervals where detector flips and weak stereo branches
    are most visible.  This pass never invents the missing hand: it only
    replaces the observed side when a safe candidate has a materially lower
    geometric-plus-continuity cost.
    """
    if not accepted or not getattr(args, "continuity_reselection", 1):
        return 0
    ordered = sorted(accepted, key=lambda row: int(row["sync_index"]))
    radius = max(int(getattr(args, "partial_continuity_radius", 6)), 0)
    if radius <= 0:
        return 0
    changed = 0
    minimum_improvement = max(
        float(getattr(args, "partial_continuity_min_improvement", 5.0)), 0.0
    )
    minimum_support = float(getattr(args, "continuity_min_support", 2.5))
    for position, row in enumerate(ordered):
        if not row.get("partial_hand_acceptance") or len(row.get("hands", [])) != 1:
            continue
        before, gap_before = _partial_neighbor_hands(
            ordered, position, -1, radius
        )
        after, gap_after = _partial_neighbor_hands(
            ordered, position, 1, radius
        )
        if before is None and after is None:
            continue
        original_hand = row["hands"][0]
        original_base = float(original_hand.get(
            "side_selection_cost", row.get("assignment_cost", 0.0)
        ))
        if not np.isfinite(original_base):
            original_base = 0.0
        original_context, original_detail = _partial_context_cost(
            original_hand, before, gap_before, after, gap_after, args
        )
        if not np.isfinite(original_context):
            continue
        original_score = original_base + original_context
        candidates = _partial_frame_candidates(
            int(row["sync_index"]), prediction_rows, calibrations,
            cameras, preferred_anchors, args,
        )
        if not candidates:
            continue
        physical_side = int(original_hand.get("side", 0))
        ranked: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for candidate in candidates:
            support = np.asarray(
                candidate["result"].get("inlier_counts", []), dtype=np.float64
            )
            support_values = support[support >= 2]
            median_support = (
                float(np.median(support_values)) if support_values.size else 0.0
            )
            selected_count = len(candidate.get("selected", {}))
            if median_support < minimum_support and selected_count < 3:
                continue
            candidate_hand = _partial_candidate_to_hand(
                candidate, physical_side, calibrations
            )
            context, detail = _partial_context_cost(
                candidate_hand, before, gap_before, after, gap_after, args
            )
            if not np.isfinite(context):
                continue
            score = float(candidate.get("cost", np.inf)) + context
            ranked.append((score, candidate, candidate_hand, detail))
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0])
        score, candidate, candidate_hand, detail = ranked[0]
        if not np.isfinite(score) or original_score - score < minimum_improvement:
            continue
        replacement = {
            "sync_index": int(row["sync_index"]),
            "anchor_cameras": list(candidate["anchor_cameras"]),
            "anchor_baseline_m": _anchor_baseline_m(
                candidate["anchor_cameras"], calibrations
            ),
            "assignment_cost": float(candidate.get("cost", np.inf)),
            "assignment_margin": None,
            "active_cameras": sorted(candidate_hand.get("camera_ids", [])),
            "active_camera_count": len(candidate_hand.get("camera_ids", [])),
            "hands": [candidate_hand],
            "partial_hand_acceptance": True,
            "missing_sides": sorted({0, 1} - {physical_side}),
            "partial_reason": "partial_continuity_reselection",
            "depth_branch_selection": {
                "algorithm": "partial_bidirectional_depth_branch_v1",
                "changed": True,
                "original_cost": original_base,
                "selected_cost": float(candidate.get("cost", np.inf)),
                "original_continuity_cost": float(original_context),
                "selected_continuity_cost": float(context),
                "original_context": original_detail,
                "selected_context": detail,
                "support_median": median_support,
            },
        }
        for key in (
            "handedness_mode", "strict_rejection", "fallback_attempted",
            "recovery",
        ):
            if key in row:
                replacement[key] = row[key]
        if args.detector_handedness == "adaptive":
            replacement["handedness_mode"] = "ignore_geometry"
        row.clear()
        row.update(replacement)
        changed += 1
    return changed


def _row_to_hypothesis(row: dict[str, Any]) -> dict[str, Any] | None:
    """Represent the already selected row as a cheap continuity hypothesis.

    Partial rows deliberately keep a two-slot representation as well: the
    missing hand is represented by an all-NaN result.  This allows a single
    trajectory selector to reason across ``full -> partial -> full`` intervals
    without hallucinating the absent hand.
    """
    by_side = {
        int(hand.get("side", -1)): hand
        for hand in row.get("hands", [])
        if int(hand.get("side", -1)) in (0, 1)
    }
    if not by_side:
        return None
    results = []
    selected: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    for side in (0, 1):
        hand = by_side.get(side)
        if hand is None:
            results.append({
                "points": np.full((21, 3), np.nan, dtype=np.float64),
                "inlier_counts": np.zeros(21, dtype=np.int32),
                "valid_joints": 0,
                "median_px": np.inf,
                "p95_px": np.inf,
                "missing": True,
            })
            continue
        points = np.asarray(hand.get("joints_base_m", []), dtype=np.float64)
        if points.shape != (21, 3):
            points = np.full((21, 3), np.nan, dtype=np.float64)
        quality = hand.get("quality", {})
        support = np.asarray(hand.get("inlier_view_counts", []), dtype=np.int32)
        if support.shape != (21,):
            support = np.zeros(21, dtype=np.int32)
        result_hypothesis = {
            "points": points,
            "inlier_counts": support,
            "valid_joints": int(quality.get(
                "valid_joint_count", np.count_nonzero(np.isfinite(points).all(axis=1))
            )),
            "median_px": float(quality.get("multiview_reprojection_median_px", np.inf)),
            "p95_px": float(quality.get("multiview_reprojection_p95_px", np.inf)),
            "missing": False,
        }
        for key in (
            "ray_angle_p25_deg", "palm_ray_angle_p25_deg",
            "palm_three_view_count", "palm_two_view_fraction",
            "two_view_joint_fraction", "pair_spread_p75_m",
            "palm_pair_spread_p75_m",
        ):
            if key in quality:
                result_hypothesis[key] = quality[key]
        results.append(result_hypothesis)
        # Preserve the serialized 2-D observations for the identity/depth
        # transition cost.  Earlier hypotheses discarded ``views`` here, so
        # the offline selector could only see 3-D motion and was blind to a
        # detector flip whose pixels were actually continuous.
        selected[side] = {
            str(camera): dict(view)
            for camera, view in hand.get("views", {}).items()
            if isinstance(view, dict)
        }
    cost = float(row.get("assignment_cost", 0.0))
    if not np.isfinite(cost):
        cost = float(sum(result["median_px"] for result in results))
    return {
        "_original": True,
        "cost": cost,
        "anchor_cameras": list(row.get("anchor_cameras", [])),
        "anchor_baseline_m": float(row.get("anchor_baseline_m", 0.0) or 0.0),
        "results": results,
        "selected": selected,
        "association_errors": {0: {}, 1: {}},
    }


def _continuity_target_indices(
    ordered: list[dict[str, Any]], args: argparse.Namespace,
) -> set[int]:
    """Find isolated discontinuities worth the expensive branch enumeration.

    Enumerating every frame is prohibitively expensive on long recordings and
    also makes a normal fast gesture look like a depth failure.  We therefore
    look for a *local* zero-phase contradiction first: the current two-hand
    centre must sit well away from the interpolation of its immediate temporal
    neighbours.  A fixed-label step is used only as a secondary cue for an
    identity/depth ambiguity.
    """
    targets: set[int] = set()
    hypotheses = [_row_to_hypothesis(row) for row in ordered]
    target_bracket = max(
        float(getattr(args, "continuity_target_bracket_m", 0.075)), 0.0
    )
    target_step = max(
        float(getattr(args, "continuity_target_step_m", 0.085)), 0.0
    )
    max_gap = max(int(getattr(args, "temporal_recovery_gap", 3)), 1) + 1

    boundary_radius = max(
        int(getattr(args, "track_continuity_boundary_radius", 2)), 0
    )
    partial_flags = [
        bool(row.get("partial_hand_acceptance") or len(row.get("hands", [])) < 2)
        for row in ordered
    ]
    partial_boundaries = {
        position for position, is_partial in enumerate(partial_flags)
        if is_partial and (
            position == 0
            or position + 1 == len(partial_flags)
            or not partial_flags[position - 1]
            or not partial_flags[position + 1]
        )
    }
    # A full/partial transition is exactly where the old complete-row-only
    # selector lost the physical track.  Include a small neighbourhood around
    # the *boundaries* only; enumerating every frame in a long partial run made
    # the offline selector unnecessarily expensive and encouraged overfitting.
    for position in partial_boundaries:
        for offset in range(-boundary_radius, boundary_radius + 1):
            candidate = position + offset
            if 0 <= candidate < len(ordered) and hypotheses[candidate] is not None:
                targets.add(candidate)

    for position in range(1, len(ordered) - 1):
        previous = hypotheses[position - 1]
        current = hypotheses[position]
        following = hypotheses[position + 1]
        if previous is None or current is None or following is None:
            continue
        gap_before = int(ordered[position]["sync_index"]) - int(
            ordered[position - 1]["sync_index"]
        )
        gap_after = int(ordered[position + 1]["sync_index"]) - int(
            ordered[position]["sync_index"]
        )
        if gap_before > max_gap or gap_after > max_gap:
            continue
        bracket = _hypothesis_bracket_residual(
            previous, current, following, gap_before, gap_after
        )
        fixed_steps = []
        previous_centres = [
            _hypothesis_center(result) for result in previous.get("results", [])
            if np.all(np.isfinite(_hypothesis_center(result)))
        ]
        current_centres = [
            _hypothesis_center(result) for result in current.get("results", [])
            if np.all(np.isfinite(_hypothesis_center(result)))
        ]
        for before in previous_centres:
            if not current_centres:
                continue
            fixed_steps.append(min(
                float(np.linalg.norm(after - before)) for after in current_centres
            ))
        margin = ordered[position].get("assignment_margin")
        low_margin = False
        if margin is not None:
            try:
                assignment_cost = float(ordered[position].get("assignment_cost", 0.0))
                low_margin = float(margin) < 0.05 * max(abs(assignment_cost), 1.0)
            except (TypeError, ValueError):
                low_margin = False
        suspicious = bracket > target_bracket and bool(
            (fixed_steps and max(fixed_steps) > 0.70 * target_step)
            or low_margin
            or bracket > target_bracket + 0.035
        )
        if suspicious:
            targets.add(position)

    # Add one neighbouring frame only.  It supplies an alternate bracket for
    # the Viterbi pass without multiplying the candidate count by a wide
    # context window on long recordings.
    expanded = set(targets)
    for position in targets:
        for offset in (-1, 1):
            candidate = position + offset
            if 0 <= candidate < len(ordered) and hypotheses[candidate] is not None:
                expanded.add(candidate)
    return expanded


def _continuity_reselect(
    accepted: list[dict[str, Any]],
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> int:
    """Choose a globally continuous path through close stereo hypotheses.

    This is deliberately an offline Viterbi pass.  It does not smooth the
    coordinates or alter a good fast motion; it only changes which already
    triangulated, reprojection-valid branch is retained for a frame.
    """
    if not args.continuity_reselection or len(accepted) < 2:
        return 0
    ordered = sorted(accepted, key=lambda row: int(row["sync_index"]))
    target_positions = _continuity_target_indices(ordered, args)
    options: list[list[dict[str, Any]]] = []
    for position, row in enumerate(ordered):
        original = _row_to_hypothesis(row)
        if original is None:
            options.append([])
            continue
        index = int(row["sync_index"])
        if position not in target_positions:
            original["_continuity_cost"] = float(original["cost"])
            options.append([original])
            continue
        hypotheses = _all_frame_hypotheses(
            index, prediction_rows, calibrations, cameras, preferred_anchors, args
        )
        # Continuity must not turn a genuinely weak stereo solution into the
        # winner merely because it is close to its neighbours.  Keep the
        # original per-frame choice available and use trajectory cost only for
        # candidates with adequate multi-view support.
        strong = [
            hypothesis for hypothesis in hypotheses
            if _hypothesis_support(hypothesis) >= args.continuity_min_support
        ]
        if strong:
            hypotheses = strong
        if not hypotheses:
            options.append([original])
            continue
        previous = (
            _row_to_hypothesis(ordered[position - 1])
            if position > 0 else None
        )
        following = (
            _row_to_hypothesis(ordered[position + 1])
            if position + 1 < len(ordered) else None
        )
        gap_before = (
            index - int(ordered[position - 1]["sync_index"])
            if previous is not None else 1
        )
        gap_after = (
            int(ordered[position + 1]["sync_index"]) - index
            if following is not None else 1
        )
        best_cost = min(float(hypothesis["cost"]) for hypothesis in hypotheses)
        # The geometric shortlist is normally deliberately narrow.  It must
        # not, however, discard a valid alternate depth branch solely because
        # a detector-side/depth assignment happened to obtain a lower local
        # reprojection cost.  Such branches are admitted only when the
        # bidirectional bracket is substantially better than the delivered
        # branch; otherwise the original conservative delta remains in force.
        original_bracket = None
        if previous is not None and following is not None:
            original_bracket = _hypothesis_bracket_residual(
                previous, original, following, gap_before, gap_after
            )
        normal_delta = max(float(args.continuity_max_cost_delta), 0.0)
        branch_delta = max(float(getattr(
            args, "continuity_branch_cost_delta", normal_delta
        )), normal_delta)
        branch_improvement = max(float(getattr(
            args, "continuity_branch_min_improvement_m", 0.025
        )), 0.0)
        shortlisted = []
        for hypothesis in hypotheses:
            geometric_delta = float(hypothesis["cost"]) - best_cost
            keep = geometric_delta <= normal_delta
            if (
                not keep
                and original_bracket is not None
                and _hypothesis_branch_quality(hypothesis, args)
            ):
                candidate_bracket = _hypothesis_bracket_residual(
                    previous, hypothesis, following, gap_before, gap_after
                )
                keep = (
                    geometric_delta <= branch_delta
                    and original_bracket - candidate_bracket >= branch_improvement
                )
            if keep:
                shortlisted.append(hypothesis)
        hypotheses = shortlisted
        # Keep the delivered branch available even if it falls outside the
        # geometric shortlist; continuity is a tie-breaker, never a reason to
        # erase a frame with no safe alternative.
        choices = [original] + [
            h for h in hypotheses if not h.get("_original")
        ]
        for choice in choices:
            choice["_continuity_cost"] = float(choice["cost"]) + (
                _hypothesis_bracket_penalty(
                    previous, choice, following,
                    gap_before, gap_after, args,
                )
            )
        options.append(choices)
    changed = 0
    start = 0
    while start < len(ordered):
        while start < len(ordered) and not options[start]:
            start += 1
        if start >= len(ordered):
            break
        end = start + 1
        while end < len(ordered) and options[end]:
            # Do not bridge a long observation blackout with a constant-speed
            # prior; the existing temporal recovery path handles short gaps.
            if int(ordered[end]["sync_index"]) - int(ordered[end - 1]["sync_index"]) > max(args.temporal_recovery_gap, 1) + 1:
                break
            end += 1
        costs = [np.asarray([
            float(h.get("_continuity_cost", h["cost"]))
            for h in options[start]
        ], dtype=np.float64)]
        back: list[np.ndarray] = []
        for pos in range(start + 1, end):
            previous = options[pos - 1]
            current = options[pos]
            matrix = np.empty((len(previous), len(current)), dtype=np.float64)
            for i, before in enumerate(previous):
                for j, after in enumerate(current):
                    matrix[i, j] = costs[-1][i] + _hypothesis_transition_cost(
                        before, after,
                        int(ordered[pos]["sync_index"]) - int(ordered[pos - 1]["sync_index"]),
                        args,
                    ) + float(after.get("_continuity_cost", after["cost"]))
            back.append(np.argmin(matrix, axis=0))
            costs.append(np.min(matrix, axis=0))
        state = int(np.argmin(costs[-1]))
        chosen = [state]
        for links in reversed(back):
            state = int(links[state])
            chosen.append(state)
        chosen.reverse()
        for offset, candidate_index in enumerate(chosen):
            pos = start + offset
            candidate = options[pos][candidate_index]
            current_row = ordered[pos]
            if candidate.get("_original"):
                continue
            old_anchor = tuple(current_row.get("anchor_cameras", ()))
            usable_costs = sorted(float(h["cost"]) for h in options[pos])
            margin = usable_costs[1] - usable_costs[0] if len(usable_costs) > 1 else None
            replacement = _candidate_to_row(
                int(current_row["sync_index"]), candidate, calibrations, args, margin
            )
            replacement["depth_branch_selection"] = {
                "algorithm": "local_viterbi_depth_branch_v3",
                "changed": True,
                "original_cost": current_row.get("assignment_cost"),
                "selected_cost": float(candidate.get("cost", np.inf)),
                "selected_continuity_cost": float(
                    candidate.get("_continuity_cost", candidate.get("cost", np.inf))
                ),
                "ambiguity_margin": margin,
            }
            # Preserve labels produced by the adaptive/temporal wrapper.
            for key in ("handedness_mode", "strict_rejection", "fallback_attempted"):
                if key in current_row:
                    replacement[key] = current_row[key]
            if "recovery" in current_row:
                replacement["recovery"] = current_row["recovery"]
            if old_anchor != tuple(replacement["anchor_cameras"]):
                changed += 1
            current_row.clear()
            current_row.update(replacement)
        start = end
    return changed


def _partial_candidate_hypothesis(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Convert an independent-side candidate to a two-slot track hypothesis."""
    side = int(candidate.get("side", 0))
    missing = {
        "points": np.full((21, 3), np.nan, dtype=np.float64),
        "inlier_counts": np.zeros(21, dtype=np.int32),
        "valid_joints": 0,
        "median_px": np.inf,
        "p95_px": np.inf,
        "missing": True,
    }
    result = candidate["result"]
    results = [missing, missing]
    results[side] = result
    return {
        "cost": float(candidate.get("cost", np.inf)),
        "anchor_cameras": list(candidate.get("anchor_cameras", [])),
        "anchor_baseline_m": float(candidate.get("anchor_baseline_m", 0.0)),
        "results": results,
        "selected": {side: candidate.get("selected", {})},
        "association_errors": {side: candidate.get("association_errors", {})},
        "_partial_candidate": candidate,
    }


def _track_state_from_hypothesis(
    hypothesis: dict[str, Any],
    *,
    original: bool = False,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a continuity hypothesis with the metadata needed by Viterbi."""
    state = {
        "hypothesis": hypothesis,
        "cost": float(hypothesis.get("cost", np.inf)),
        "_original": bool(original),
    }
    if row is not None:
        state["_row"] = row
    if hypothesis.get("_partial_candidate") is not None:
        state["_partial_candidate"] = hypothesis["_partial_candidate"]
    return state


def _track_partial_candidate_safe(
    candidate: dict[str, Any], args: argparse.Namespace,
) -> bool:
    """Apply the continuity quality gate to an independent-side branch."""
    result = candidate.get("result", {})
    minimum_valid = max(
        int(getattr(args, "continuity_min_branch_valid_joints", 8)), 1
    )
    if int(result.get("valid_joints", 0)) < minimum_valid:
        return False
    points = np.asarray(result.get("points", []), dtype=np.float64)
    if points.shape != (21, 3):
        return False
    palm = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
    counts = np.asarray(result.get("inlier_counts", []), dtype=np.float64)
    if counts.shape != (21,):
        return False
    finite = np.isfinite(points[palm]).all(axis=1)
    reliable = finite & (counts[palm] >= 2)
    if int(np.count_nonzero(reliable)) < max(
        int(getattr(args, "continuity_min_palm_points", 3)), 3
    ):
        return False
    support = counts[palm][reliable]
    if support.size and float(np.median(support)) < float(
        getattr(args, "continuity_min_palm_support", 2.0)
    ):
        return False
    median = float(result.get("median_px", np.inf))
    p95 = float(result.get("p95_px", np.inf))
    if _depth_condition_reasons(
        result, len(candidate.get("selected", {})), args
    ):
        return False
    depth_level, depth_reasons = _depth_condition_level(
        result, len(candidate.get("selected", {})), args
    )
    if depth_level == "unsafe":
        return False
    # Do not let the continuity pass promote a high-spread branch into a
    # persistent partial track.  A weak two-view branch with no pair spread is
    # still allowed for a genuinely occluded hand; disagreement between palm
    # pair intersections is a direct depth ambiguity and must remain
    # disposable.
    if "weak_depth_pair_spread" in depth_reasons:
        return False
    return bool(
        np.isfinite(median)
        and np.isfinite(p95)
        and median <= float(getattr(args, "max_reprojection_median_px", 15.0))
        and p95 <= float(getattr(args, "max_reprojection_p95_px", 40.0))
    )


def _hypothesis_hand_view(
    hypothesis: dict[str, Any], side: int,
) -> dict[str, Any] | None:
    """Expose one hypothesis slot through the serialized-hand interface."""
    side = int(side)
    results = hypothesis.get("results", [])
    if side not in (0, 1) or side >= len(results):
        return None
    result = results[side]
    centre = _hypothesis_center(result)
    if not np.all(np.isfinite(centre)):
        return None
    return {
        "joints_base_m": np.asarray(
            result.get("points", []), dtype=np.float64
        ),
        "views": {
            str(camera): dict(view)
            for camera, view in hypothesis.get("selected", {}).get(side, {}).items()
            if isinstance(view, dict)
        },
    }


def _hybrid_2d_track_evidence(
    candidate: dict[str, Any],
    side: int,
    previous: dict[str, Any] | None,
    following: dict[str, Any] | None,
    gap_before: int,
    gap_after: int,
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure whether a hybrid replacement follows the target image track."""
    current = _partial_candidate_to_hand(candidate, int(side), calibrations)
    continuous = 0
    ambiguous = 0
    breaks = 0
    motions: list[float] = []
    links: list[dict[str, Any]] = []
    for direction, neighbour, gap in (
        ("previous", previous, gap_before),
        ("following", following, gap_after),
    ):
        if neighbour is None:
            continue
        neighbour_hypothesis = neighbour
        neighbour_hand = _hypothesis_hand_view(neighbour_hypothesis, int(side))
        if neighbour_hand is None:
            continue
        policy, diagnostics = _identity_2d_link_policy(
            neighbour_hand, current, int(gap), args
        )
        motion = float(diagnostics.get("robust_px_per_frame", np.nan))
        if np.isfinite(motion):
            motions.append(motion)
        if policy == "continuous":
            continuous += 1
        elif policy == "ambiguous":
            ambiguous += 1
        elif policy == "break":
            breaks += 1
        links.append({
            "direction": direction,
            "policy": policy,
            "motion_px_per_frame": motion if np.isfinite(motion) else None,
            "shared_camera_count": int(diagnostics.get("shared_camera_count", 0)),
        })
    return {
        "continuous_link_count": continuous,
        "ambiguous_link_count": ambiguous,
        "break_link_count": breaks,
        "motion_px_per_frame": float(np.median(motions)) if motions else None,
        "links": links,
    }


def _side_hypothesis_bracket_residual(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    following: dict[str, Any] | None,
    side: int,
    gap_before: int,
    gap_after: int,
) -> float:
    """Return a side-aware zero-phase residual for a branch replacement."""
    if previous is None or following is None:
        return float("inf")
    before = _hypothesis_hand_view(previous, side)
    now = _hypothesis_hand_view(current, side)
    after = _hypothesis_hand_view(following, side)
    if before is None or now is None or after is None:
        return float("inf")
    before_center = _hand_center_from_row(before)
    current_center = _hand_center_from_row(now)
    after_center = _hand_center_from_row(after)
    if not (
        np.all(np.isfinite(before_center))
        and np.all(np.isfinite(current_center))
        and np.all(np.isfinite(after_center))
    ):
        return float("inf")
    gap_before = max(int(gap_before), 1)
    gap_after = max(int(gap_after), 1)
    alpha = gap_before / float(gap_before + gap_after)
    prediction = (1.0 - alpha) * before_center + alpha * after_center
    return float(np.linalg.norm(current_center - prediction))


def _hybrid_replacement_safe(
    hypothesis: dict[str, Any],
    side: int,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    """Check the non-temporal safety conditions for a one-side hybrid."""
    quality = _independent_candidate_quality(candidate, args)
    if not quality.get("single_safe"):
        return False
    result = candidate.get("result", {})
    if int(result.get("valid_joints", 0)) < max(
        int(getattr(args, "continuity_min_branch_valid_joints", 8)), 1
    ):
        return False
    # A detector-side relabel can produce two independently triangulated
    # hypotheses at essentially the same 3-D location.  Treating those as a
    # hybrid pair would duplicate one physical hand even when their boxes are
    # not identical enough for the 2-D IoU test.  Keep a small physical-space
    # separation guard; genuinely touching hands remain possible above this
    # tolerance and are still checked by the image duplicate gate.
    replacement_center = _hypothesis_center(result)
    other_result = hypothesis.get("results", [None, None])[1 - int(side)]
    other_center = _hypothesis_center(other_result)
    if np.all(np.isfinite(replacement_center)) and np.all(
        np.isfinite(other_center)
    ):
        centre_distance = float(np.linalg.norm(
            replacement_center - other_center
        ))
        # Keep the old very-tight unconditional guard, then apply the same
        # shared-camera proof rule used by ordinary independent pairs for the
        # wider near-coincident band.
        if centre_distance < 0.025:
            return False
    # Do not let a replacement reuse the other physical hand's detector box.
    other = hypothesis.get("selected", {}).get(1 - int(side), {})
    mixed = {
        0: other if int(side) == 1 else candidate.get("selected", {}),
        1: candidate.get("selected", {}) if int(side) == 1 else other,
    }
    near_duplicate, _near_duplicate_diagnostics = (
        _near_coincident_independent_pair_is_unsafe(
            {
                "result": result,
                "selected": candidate.get("selected", {}),
            },
            {
                "result": other_result,
                "selected": other,
            },
            args,
        )
    )
    if near_duplicate:
        return False
    unsafe, _count = _duplicate_assignment_is_unsafe(
        mixed,
        None,
        float(getattr(args, "duplicate_bbox_iou", 0.65)),
        float(getattr(args, "duplicate_joint_distance_px", 18.0)),
        int(getattr(args, "duplicate_overlap_cameras", 2)),
    )
    return not unsafe


def _track_continuity_targets(
    ordered: list[dict[str, Any]], args: argparse.Namespace,
) -> set[int]:
    """Find rows where a full/partial boundary can change branch or identity."""
    targets = set(_continuity_target_indices(ordered, args))
    # The legacy detector errs on the side of inclusion. Keep priorities so a
    # long partial run cannot send hundreds of ordinary frames through costly
    # branch enumeration.
    priority: dict[int, float] = {index: 10.0 for index in targets}

    def add_target(index: int, score: float) -> None:
        if 0 <= index < len(ordered) and _row_to_hypothesis(ordered[index]) is not None:
            targets.add(index)
            priority[index] = max(priority.get(index, 0.0), float(score))
    radius = max(int(getattr(args, "track_continuity_boundary_radius", 2)), 0)
    partial_flags = [
        bool(row.get("partial_hand_acceptance") or len(row.get("hands", [])) < 2)
        for row in ordered
    ]
    partial_boundaries = {
        index for index, is_partial in enumerate(partial_flags)
        if is_partial and (
            index == 0
            or index + 1 == len(partial_flags)
            or not partial_flags[index - 1]
            or not partial_flags[index + 1]
        )
    }
    for index in partial_boundaries:
        for offset in range(-radius, radius + 1):
            candidate = index + offset
            if 0 <= candidate < len(ordered):
                # The boundary itself is most important; neighbouring frames
                # are context and should lose to a genuine spike when the
                # target budget is tight.
                add_target(candidate, 90.0 - 12.0 * abs(offset))

    # Use the already canonicalized physical labels as a secondary cue.  This
    # catches a branch that is not a strict V-shaped excursion but still makes
    # one track jump when a hand re-enters after a short partial interval.
    target_step = max(float(getattr(args, "continuity_target_step_m", 0.085)), 0.0)
    for index, row in enumerate(ordered):
        if index == 0 or index + 1 >= len(ordered):
            continue
        current_by_side = {
            int(hand.get("side", -1)): hand
            for hand in row.get("hands", [])
            if int(hand.get("side", -1)) in (0, 1)
        }
        if not current_by_side:
            continue
        for side, hand in current_by_side.items():
            centre = _hand_center_from_row(hand)
            if not np.all(np.isfinite(centre)):
                continue
            distances = []
            for direction in (-1, 1):
                cursor = index + direction
                while 0 <= cursor < len(ordered):
                    neighbour = next(
                        (
                            value for value in ordered[cursor].get("hands", [])
                            if int(value.get("side", -1)) == side
                            and np.all(np.isfinite(_hand_center_from_row(value)))
                        ),
                        None,
                    )
                    if neighbour is not None:
                        gap = abs(int(ordered[cursor]["sync_index"]) - int(row["sync_index"]))
                        distances.append(
                            float(np.linalg.norm(
                                _hand_center_from_row(neighbour) - centre
                            )) / max(gap, 1)
                        )
                        break
                    cursor += direction
            if distances and max(distances) > target_step:
                add_target(index, 120.0 + 80.0 * min(
                    max(distances) / max(target_step, 1e-6), 3.0
                ))
                break

    # Explicitly inspect under-conditioned depth rows even when their centre
    # does not form a perfect V-shaped excursion.  A wrong two-view branch can
    # persist for several frames, and the old target detector only sampled the
    # turn-around frame.  Limit this to rows with a local jump or a low
    # assignment margin so normal side-on stereo does not trigger expensive
    # all-branch enumeration.
    for index, row in enumerate(ordered):
        if index == 0 or index + 1 >= len(ordered):
            continue
        suspicious_quality = False
        severe_depth = False
        ambiguous_depth = False
        for hand in row.get("hands", []):
            quality = hand.get("quality", {})
            try:
                camera_count = len(hand.get("camera_ids", []))
                palm_three = int(quality.get("palm_three_view_count", 0) or 0)
                palm_two = float(quality.get("palm_two_view_fraction", 1.0))
                palm_angle = float(quality.get("palm_ray_angle_p25_deg", np.nan))
                pair_spread = float(quality.get("palm_pair_spread_p75_m", 0.0))
            except (TypeError, ValueError):
                continue
            low_angle = np.isfinite(palm_angle) and palm_angle < 6.0
            if np.isfinite(palm_angle) and palm_angle < float(
                getattr(args, "depth_min_palm_angle_deg", 4.5)
            ):
                severe_depth = True
            if np.isfinite(pair_spread) and pair_spread > 0.09:
                ambiguous_depth = True
            low_support = (
                camera_count <= 2
                or palm_three < 2
                or palm_two > 0.60
            )
            if low_angle or low_support or pair_spread > 0.06:
                suspicious_quality = True
                break
        if not suspicious_quality:
            continue
        current_hypothesis = _row_to_hypothesis(row)
        previous_hypothesis = _row_to_hypothesis(ordered[index - 1])
        following_hypothesis = _row_to_hypothesis(ordered[index + 1])
        if not (current_hypothesis and previous_hypothesis and following_hypothesis):
            continue
        bracket = _hypothesis_bracket_residual(
            previous_hypothesis, current_hypothesis, following_hypothesis, 1, 1
        )
        local_jump = 0.0
        for side, current_hand in enumerate(row.get("hands", [])):
            centre = _hand_center_from_row(current_hand)
            if not np.all(np.isfinite(centre)):
                continue
            for neighbour in (ordered[index - 1], ordered[index + 1]):
                neighbour_hand = next(
                    (
                        value for value in neighbour.get("hands", [])
                        if int(value.get("side", -1)) == int(current_hand.get("side", side))
                        and np.all(np.isfinite(_hand_center_from_row(value)))
                    ),
                    None,
                )
                if neighbour_hand is not None:
                    local_jump = max(local_jump, float(np.linalg.norm(
                        _hand_center_from_row(neighbour_hand) - centre
                    )))
        margin = row.get("assignment_margin")
        low_margin = False
        try:
            low_margin = margin is not None and float(margin) < 35.0
        except (TypeError, ValueError):
            pass
        if severe_depth or ambiguous_depth or local_jump > 0.045 or bracket > 0.055 or low_margin:
            depth_score = 0.0
            for hand in row.get("hands", []):
                quality = hand.get("quality", {})
                try:
                    angle = float(quality.get("palm_ray_angle_p25_deg", np.nan))
                    spread = float(quality.get("palm_pair_spread_p75_m", 0.0))
                    if np.isfinite(angle) and angle < float(
                        getattr(args, "depth_min_palm_angle_deg", 4.5)
                    ):
                        # Hard depth failures are mandatory targets even when
                        # the global budget is small; otherwise a large but
                        # less certain motion elsewhere could hide them.
                        depth_score = max(depth_score, 1000.0)
                    if np.isfinite(spread) and spread > float(
                        getattr(args, "depth_max_pair_spread_m", 0.075)
                    ):
                        depth_score = max(depth_score, 500.0 if spread > 0.09 else 25.0)
                except (TypeError, ValueError):
                    pass
            score = (
                100.0
                + 80.0 * min(bracket / max(target_step, 1e-6), 3.0)
                + 45.0 * min(local_jump / max(target_step, 1e-6), 3.0)
                + (25.0 if low_margin else 0.0)
                + depth_score
            )
            if severe_depth:
                score += 1000.0
            elif ambiguous_depth:
                score += 500.0
            add_target(index, score)
    # Bound the expensive pass while retaining the most informative frames.
    maximum = max(int(getattr(args, "track_continuity_max_targets", 96)), 1)
    if len(targets) > maximum:
        ranked = sorted(
            targets,
            key=lambda index: (-priority.get(index, 0.0), int(index)),
        )
        targets = set(ranked[:maximum])
    return targets


def _track_partial_side_for_state(
    state: dict[str, Any],
    previous: dict[str, Any] | None,
    following: dict[str, Any] | None,
    fallback: int,
    args: argparse.Namespace | None = None,
) -> int:
    """Choose the physical side for a one-hand state from its neighbours."""
    hypothesis = state["hypothesis"]
    present = [
        result for result in hypothesis.get("results", [])
        if np.all(np.isfinite(_hypothesis_center(result)))
    ]
    if not present:
        return int(fallback)
    current_result = present[0]
    current = _hypothesis_center(current_result)
    hypothesis_side = next(
        (
            side for side, result in enumerate(hypothesis.get("results", []))
            if result is current_result
        ),
        int(fallback),
    )
    current_views = hypothesis.get("selected", {}).get(hypothesis_side, {})
    scores = {0: 0.0, 1: 0.0}
    evidence = {0: 0, 1: 0}
    for neighbour in (previous, following):
        if neighbour is None:
            continue
        for side, result in enumerate(neighbour["hypothesis"].get("results", [])):
            centre = _hypothesis_center(result)
            if np.all(np.isfinite(centre)):
                value = float(np.linalg.norm(current - centre))
                if args is not None:
                    neighbour_views = neighbour["hypothesis"].get(
                        "selected", {}
                    ).get(side, {})
                    contradictory, _diagnostics = _depth_reconnect_contradiction(
                        {"joints_base_m": result.get("points"),
                         "views": dict(neighbour_views)},
                        {"joints_base_m": current_result.get("points"),
                         "views": dict(current_views)},
                        1,
                        args,
                    )
                    if contradictory:
                        value += max(float(getattr(
                            args, "track_reconnect_depth_penalty", 2500.0
                        )), 0.0)
                scores[side] += value
                evidence[side] += 1
    viable = [side for side in (0, 1) if evidence[side]]
    if not viable:
        return int(fallback)
    return int(min(viable, key=lambda side: (scores[side], side != int(fallback))))


def _track_continuity_reselect(
    accepted: list[dict[str, Any]],
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> int:
    """Jointly select depth branches and identities through partial intervals.

    Every row keeps its delivered observation as a valid state.  Only rows near
    a discontinuity or a full/partial boundary receive alternate geometric
    states.  A small Viterbi pass then links the available hands, with missing
    sides represented explicitly rather than copied from the other hand.
    """
    if not getattr(args, "track_continuity", 1) or len(accepted) < 2:
        return 0
    ordered = sorted(accepted, key=lambda row: int(row["sync_index"]))
    targets = _track_continuity_targets(ordered, args)
    options: list[list[dict[str, Any]]] = []
    for position, row in enumerate(ordered):
        original = _row_to_hypothesis(row)
        if original is None:
            options.append([])
            continue
        original_state = _track_state_from_hypothesis(
            original, original=True, row=row
        )
        if position not in targets:
            original_state["_continuity_cost"] = float(original_state["cost"])
            options.append([original_state])
            continue

        alternatives: list[dict[str, Any]] = []
        index = int(row["sync_index"])
        if getattr(args, "independent_hand_fusion", 0):
            for candidate in _independent_frame_hypotheses(
                index, prediction_rows, calibrations, cameras,
                preferred_anchors, args, original_row=row,
            ):
                present = [
                    result for result in candidate.get("results", [])
                    if np.all(np.isfinite(_hypothesis_center(result)))
                ]
                if len(present) >= 2:
                    if candidate.get("_hybrid"):
                        replacement_side = int(candidate.get(
                            "_hybrid_replacement_side", -1
                        ))
                        replacement = candidate.get("_hybrid_candidate")
                        if (
                            replacement is not None
                            and replacement_side in (0, 1)
                            and _hybrid_replacement_safe(
                                candidate, replacement_side, replacement, args
                            )
                        ):
                            alternatives.append(candidate)
                    elif _hypothesis_branch_quality(candidate, args):
                        alternatives.append(candidate)
                elif candidate.get("_partial_candidate") is not None:
                    if _track_partial_candidate_safe(
                        candidate["_partial_candidate"], args
                    ):
                        alternatives.append(candidate)
        elif row.get("partial_hand_acceptance") or len(row.get("hands", [])) < 2:
            for candidate in _partial_frame_candidates(
                index, prediction_rows, calibrations, cameras,
                preferred_anchors, args,
            ):
                if _track_partial_candidate_safe(candidate, args):
                    alternatives.append(_partial_candidate_hypothesis(candidate))
        else:
            for candidate in _all_frame_hypotheses(
                index, prediction_rows, calibrations, cameras,
                preferred_anchors, args,
            ):
                if _hypothesis_branch_quality(candidate, args):
                    alternatives.append(candidate)

        previous = _row_to_hypothesis(ordered[position - 1]) if position else None
        following = (
            _row_to_hypothesis(ordered[position + 1])
            if position + 1 < len(ordered) else None
        )
        gap_before = (
            index - int(ordered[position - 1]["sync_index"])
            if position else 1
        )
        gap_after = (
            int(ordered[position + 1]["sync_index"]) - index
            if position + 1 < len(ordered) else 1
        )
        original_bracket = (
            _hypothesis_bracket_residual(
                previous, original, following, gap_before, gap_after
            )
            if previous is not None and following is not None else None
        )
        best_cost = min(
            [float(original.get("cost", np.inf))]
            + [float(value.get("cost", np.inf)) for value in alternatives]
        )
        normal_delta = max(float(getattr(args, "continuity_max_cost_delta", 300.0)), 0.0)
        branch_delta = max(float(getattr(
            args, "continuity_branch_cost_delta", normal_delta
        )), normal_delta)
        branch_improvement = max(float(getattr(
            args, "continuity_branch_min_improvement_m", 0.025
        )), 0.0)
        states = [original_state]
        seen = set()
        for candidate in alternatives:
            # A continuity alternative may rescue a missing-side boundary, but
            # it must never silently turn a delivered complete two-hand frame
            # into a one-hand frame.  The old selector did exactly that near a
            # partial interval: a cheap single-hand branch won the Viterbi
            # score, and both the good hand and its identity context vanished
            # for several neighbouring frames.  Cardinality is an observation
            # contract, not a soft trajectory preference.
            original_present_count = sum(
                int(np.all(np.isfinite(_hypothesis_center(result))))
                for result in original.get("results", [])
            )
            candidate_present_count = sum(
                int(np.all(np.isfinite(_hypothesis_center(result))))
                for result in candidate.get("results", [])
            )
            if original_present_count >= 2 and candidate_present_count < 2:
                continue
            if candidate.get("_hybrid"):
                replacement_side = int(candidate.get(
                    "_hybrid_replacement_side", -1
                ))
                replacement = candidate.get("_hybrid_candidate")
                if replacement is None or replacement_side not in (0, 1):
                    continue
                evidence = _hybrid_2d_track_evidence(
                    replacement,
                    replacement_side,
                    previous,
                    following,
                    gap_before,
                    gap_after,
                    calibrations,
                    args,
                )
                candidate["_hybrid_2d_evidence"] = evidence
                # A hybrid is meaningful only if its detector pixels follow
                # the target physical side.  In particular, do not use a
                # numerically attractive branch from the other hand merely
                # because its 3-D centre lies near the bracket.
                if (
                    int(evidence.get("continuous_link_count", 0)) < 1
                    or int(evidence.get("break_link_count", 0)) > 0
                ):
                    continue
                if previous is None or following is None:
                    continue
                original_side_residual = _side_hypothesis_bracket_residual(
                    previous, original, following, replacement_side,
                    gap_before, gap_after,
                )
                candidate_side_residual = _side_hypothesis_bracket_residual(
                    previous, candidate, following, replacement_side,
                    gap_before, gap_after,
                )
                side_improvement = (
                    original_side_residual - candidate_side_residual
                    if np.isfinite(original_side_residual)
                    and np.isfinite(candidate_side_residual)
                    else float("-inf")
                )
                candidate["_hybrid_side_bracket_improvement_m"] = float(
                    side_improvement
                )
                if side_improvement < max(
                    float(getattr(
                        args, "track_hybrid_min_bracket_improvement_m", 0.035
                    )),
                    branch_improvement,
                ):
                    continue
            geometric_delta = float(candidate.get("cost", np.inf)) - best_cost
            keep = geometric_delta <= normal_delta
            if (
                not keep
                and previous is not None
                and following is not None
                and original_bracket is not None
            ):
                candidate_bracket = _hypothesis_bracket_residual(
                    previous, candidate, following, gap_before, gap_after
                )
                keep = (
                    geometric_delta <= branch_delta
                    and original_bracket - candidate_bracket >= branch_improvement
                )
            if not keep:
                continue
            centres = tuple(
                tuple(np.round(_hypothesis_center(result), 4))
                for result in candidate.get("results", [])
                if np.all(np.isfinite(_hypothesis_center(result)))
            )
            key = (tuple(candidate.get("anchor_cameras", [])), centres)
            if key in seen:
                continue
            seen.add(key)
            state = _track_state_from_hypothesis(candidate)
            state["_continuity_cost"] = float(candidate.get("cost", np.inf)) + (
                _hypothesis_bracket_penalty(
                    previous, candidate, following, gap_before, gap_after, args
                )
            )
            states.append(state)
        original_state["_continuity_cost"] = float(original.get("cost", np.inf)) + (
            _hypothesis_bracket_penalty(
                previous, original, following, gap_before, gap_after, args
            )
        )
        options.append(states)

    changed = 0
    start = 0
    max_gap = max(int(getattr(args, "temporal_recovery_gap", 3)), 1) + 1
    while start < len(ordered):
        while start < len(ordered) and not options[start]:
            start += 1
        if start >= len(ordered):
            break
        end = start + 1
        while end < len(ordered) and options[end]:
            if int(ordered[end]["sync_index"]) - int(
                ordered[end - 1]["sync_index"]
            ) > max_gap:
                break
            end += 1
        costs = [np.asarray([
            float(state.get("_continuity_cost", state["cost"]))
            for state in options[start]
        ], dtype=np.float64)]
        back: list[np.ndarray] = []
        for position in range(start + 1, end):
            previous_states = options[position - 1]
            current_states = options[position]
            matrix = np.full(
                (len(previous_states), len(current_states)), np.inf,
                dtype=np.float64,
            )
            gap = int(ordered[position]["sync_index"]) - int(
                ordered[position - 1]["sync_index"]
            )
            for before_index, before in enumerate(previous_states):
                for after_index, after in enumerate(current_states):
                    transition = _hypothesis_transition_cost(
                        before["hypothesis"], after["hypothesis"], gap, args
                    )
                    matrix[before_index, after_index] = (
                        costs[-1][before_index]
                        + transition
                        + float(after.get("_continuity_cost", after["cost"]))
                    )
            back.append(np.argmin(matrix, axis=0))
            costs.append(np.min(matrix, axis=0))
        state_index = int(np.argmin(costs[-1]))
        chosen = [state_index]
        for links in reversed(back):
            state_index = int(links[state_index])
            chosen.append(state_index)
        chosen.reverse()

        for offset, candidate_index in enumerate(chosen):
            position = start + offset
            state = options[position][candidate_index]
            if state.get("_original"):
                continue
            row = ordered[position]
            previous_state = (
                options[position - 1][chosen[offset - 1]]
                if offset > 0 else None
            )
            following_state = (
                options[position + 1][chosen[offset + 1]]
                if offset + 1 < len(chosen) else None
            )
            # Do not replace a delivered result for a numerically negligible
            # gain.  This hysteresis is especially useful when detector labels
            # tie while the two hands cross in the image.
            original_state = next(
                value for value in options[position] if value.get("_original")
            )
            def local_score(value: dict[str, Any]) -> float:
                score = float(value.get("_continuity_cost", value["cost"]))
                if previous_state is not None:
                    score += _hypothesis_transition_cost(
                        previous_state["hypothesis"], value["hypothesis"],
                        int(row["sync_index"]) - int(
                            ordered[position - 1]["sync_index"]
                        ), args,
                    )
                if following_state is not None:
                    score += _hypothesis_transition_cost(
                        value["hypothesis"], following_state["hypothesis"],
                        int(ordered[position + 1]["sync_index"]) - int(
                            row["sync_index"]
                        ), args,
                    )
                return score
            if local_score(original_state) - local_score(state) < max(
                float(getattr(args, "track_continuity_min_improvement", 4.0)), 0.0
            ):
                continue

            # A state-cost improvement is not necessarily a depth
            # improvement.  Detector-local labels and anchor bookkeeping can
            # produce tiny cost differences for effectively the same—or even
            # a worse—3-D pose.  Require a measurable local trajectory gain
            # before replacing the delivered observation.  This specifically
            # blocks the 0.02-cost substitutions seen at frames 266/272/273.
            candidate = state["hypothesis"]
            previous_hypothesis = (
                previous_state["hypothesis"] if previous_state is not None
                else None
            )
            following_hypothesis = (
                following_state["hypothesis"] if following_state is not None
                else None
            )
            gap_before = (
                int(row["sync_index"])
                - int(ordered[position - 1]["sync_index"])
                if previous_state is not None else 1
            )
            gap_after = (
                int(ordered[position + 1]["sync_index"])
                - int(row["sync_index"])
                if following_state is not None else 1
            )
            depth_improvement, depth_diagnostics = (
                _track_reselection_depth_improvement(
                    original_state["hypothesis"],
                    candidate,
                    previous_hypothesis,
                    following_hypothesis,
                    gap_before,
                    gap_after,
                )
            )
            required_depth_improvement = max(float(getattr(
                args, "track_reselection_min_depth_improvement_m", 0.025
            )), 0.0)
            if depth_improvement < required_depth_improvement:
                continue

            old_anchor = tuple(row.get("anchor_cameras", ()))
            if candidate.get("_hybrid"):
                replacement_side = int(candidate.get(
                    "_hybrid_replacement_side", -1
                ))
                replacement_candidate = candidate.get("_hybrid_candidate")
                if replacement_side not in (0, 1) or replacement_candidate is None:
                    continue
                # Keep the untouched hand exactly as delivered.  Rebuilding
                # both hands through ``_candidate_to_row`` would re-run the
                # detector pairing and reintroduce the very hard-binding bug
                # this hybrid state is meant to avoid.
                replacement = dict(row)
                original_hands = list(row.get("hands", []))
                replacement_hand = _partial_candidate_to_hand(
                    replacement_candidate, replacement_side, calibrations
                )
                replacement["hands"] = [
                    replacement_hand
                    if int(hand.get("side", -1)) == replacement_side
                    else hand
                    for hand in original_hands
                ]
                replacement["hands"] = sorted(
                    replacement["hands"],
                    key=lambda hand: int(hand.get("side", 0)),
                )
                replacement["active_cameras"] = sorted(set().union(*(
                    set(hand.get("camera_ids", []))
                    for hand in replacement["hands"]
                )))
                replacement["active_camera_count"] = len(
                    replacement["active_cameras"]
                )
                replacement["assignment_cost"] = float(candidate.get("cost", np.inf))
                replacement["hybrid_branch_selection"] = {
                    "algorithm": "one_side_depth_branch_hybrid_v1",
                    "changed": True,
                    "replacement_side": replacement_side,
                    "original_cost": row.get("assignment_cost"),
                    "selected_cost": float(candidate.get("cost", np.inf)),
                    "depth_improvement_m": float(depth_improvement),
                    "side_bracket_improvement_m": float(candidate.get(
                        "_hybrid_side_bracket_improvement_m", np.nan
                    )),
                    "image_track_evidence": candidate.get(
                        "_hybrid_2d_evidence", {}
                    ),
                }
                # Preserve the row's physical cardinality and explicitly
                # clear stale partial markers from an originally complete row.
                if len(replacement["hands"]) >= 2:
                    replacement.pop("partial_hand_acceptance", None)
                    replacement.pop("missing_sides", None)
            elif state.get("_partial_candidate") is not None:
                fallback_side = int(row.get("hands", [{}])[0].get("side", 0))
                physical_side = _track_partial_side_for_state(
                    state, previous_state, following_state, fallback_side, args
                )
                hand = _partial_candidate_to_hand(
                    state["_partial_candidate"], physical_side, calibrations
                )
                replacement = {
                    "sync_index": int(row["sync_index"]),
                    "anchor_cameras": list(candidate.get("anchor_cameras", [])),
                    "anchor_baseline_m": _anchor_baseline_m(
                        candidate.get("anchor_cameras", []), calibrations
                    ),
                    "assignment_cost": float(candidate.get("cost", np.inf)),
                    "assignment_margin": None,
                    "active_cameras": sorted(hand.get("camera_ids", [])),
                    "active_camera_count": len(hand.get("camera_ids", [])),
                    "hands": [hand],
                    "partial_hand_acceptance": True,
                    "missing_sides": sorted({0, 1} - {physical_side}),
                    "partial_reason": "track_continuity_reselection",
                }
            else:
                usable_costs = sorted(
                    float(value["hypothesis"].get("cost", np.inf))
                    for value in options[position]
                )
                margin = usable_costs[1] - usable_costs[0] if len(usable_costs) > 1 else None
                replacement = _candidate_to_row(
                    int(row["sync_index"]), candidate, calibrations, args, margin
                )
                replacement["track_continuity_selection"] = {
                    "algorithm": "unified_full_partial_track_reconnect_guarded_v2",
                    "changed": True,
                    "original_cost": row.get("assignment_cost"),
                    "selected_cost": float(candidate.get("cost", np.inf)),
                    "depth_improvement_m": float(depth_improvement),
                    "depth_diagnostics": depth_diagnostics,
                }
            for key in (
                "handedness_mode", "strict_rejection", "fallback_attempted",
                "recovery", "strict_reference", "geometry_reference",
            ):
                if key in row:
                    replacement[key] = row[key]
            row.clear()
            row.update(replacement)
            if old_anchor != tuple(replacement.get("anchor_cameras", ())):
                changed += 1
            else:
                changed += 1
        start = end
    return changed


def _fuse_primary_frame_mode(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    groups = {
        camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras
    }
    complete_by_camera = {
        camera: _complete_detections(groups[camera], args.max_anchor_detections)
        for camera in cameras
    }
    preferred_available = all(
        len(complete_by_camera[camera]) >= 2 for camera in preferred_anchors
    )
    fallback_pairs = [
        pair for pair in itertools.combinations(cameras, 2)
        if pair != preferred_anchors
        and any(camera in preferred_anchors for camera in pair)
        and all(len(complete_by_camera[camera]) >= 2 for camera in pair)
    ]
    if not preferred_available and not fallback_pairs:
        return _fuse_partial_hands(
            sync_index, groups, calibrations, cameras, preferred_anchors, args
        )

    candidates = (
        _evaluate_anchor_pairs(
            [preferred_anchors], complete_by_camera, groups, calibrations,
            cameras, preferred_anchors, args,
        )
        if preferred_available else []
    )
    usable = [
        candidate for candidate in candidates
        if not _quality_reasons(candidate["results"], args)
    ]
    # A complete geometric assignment is only allowed when *both* hands have
    # a normal depth condition.  Reprojection error alone cannot distinguish
    # a weak two-view branch from the correct solution: both can fit their
    # pixels while landing on different depth branches.  Previously we only
    # removed ``unsafe`` candidates here and then fell back to
    # ``candidates[0]`` when every candidate was weak.  That silently
    # re-introduced the very branch this gate was intended to prevent.
    usable = [
        candidate for candidate in usable
        if all(
            _depth_condition_level(
                result, len(candidate.get("selected", {}).get(side, {})), args
            )[0] == "normal"
            for side, result in enumerate(candidate.get("results", []))
        )
    ]
    if not usable and fallback_pairs:
        fallback_candidates = _evaluate_anchor_pairs(
            fallback_pairs, complete_by_camera, groups, calibrations,
            cameras, preferred_anchors, args,
        )
        candidates.extend(fallback_candidates)
        usable = [
            candidate for candidate in fallback_candidates
            if not _quality_reasons(candidate["results"], args)
        ]
        usable = [
            candidate for candidate in usable
            if all(
                _depth_condition_level(
                    result, len(candidate.get("selected", {}).get(side, {})), args
                )[0] == "normal"
                for side, result in enumerate(candidate.get("results", []))
            )
        ]
    if not candidates:
        return _fuse_partial_hands(
            sync_index, groups, calibrations, cameras, preferred_anchors, args
        )

    # Never select an arbitrary weak/unsafe complete candidate merely because
    # all complete candidates failed the depth gate.  Let the independent
    # path keep a stable visible hand (or reject the frame) instead.
    if not usable:
        partial_accepted, partial_row = _fuse_partial_hands(
            sync_index, groups, calibrations, cameras, preferred_anchors, args
        )
        if partial_accepted:
            partial_row.setdefault("partial_reason", "complete_depth_condition")
            partial_row["complete_depth_rejected"] = True
            return True, partial_row
        return False, {
            "sync_index": sync_index,
            "reasons": ["complete_depth_condition"],
        }

    candidates.sort(key=lambda item: item["cost"])
    usable.sort(key=lambda item: item["cost"])
    best = usable[0] if usable else candidates[0]
    chosen_anchors = tuple(best["anchor_cameras"])
    quality_reasons = _quality_reasons(best["results"], args)
    if quality_reasons:
        partial_accepted, partial_row = _fuse_partial_hands(
            sync_index, groups, calibrations, cameras, preferred_anchors, args
        )
        if partial_accepted:
            return True, partial_row
        return False, {
            "sync_index": sync_index,
            "reasons": sorted(set(quality_reasons)),
        }
    margin = usable[1]["cost"] - best["cost"] if len(usable) > 1 else None
    hands = []
    for side, result in enumerate(best["results"]):
        hand = _serialize_hand(
            result, best["selected"][side], calibrations, chosen_anchors
        )
        hand["side_selection_cost"] = float(
            _result_geometry_cost(result)
            + _anatomy_penalty(result["points"])
        )
        hand["association_error_px"] = best["association_errors"][side]
        hand["fusion_mode"] = (
            "multiview" if len(best["selected"][side]) >= 3 else "stereo_anchor"
        )
        hands.append(hand)
    active_cameras = sorted(set().union(*(hand["camera_ids"] for hand in hands)))
    complete_row = {
        "sync_index": sync_index,
        "anchor_cameras": list(chosen_anchors),
        "anchor_baseline_m": _anchor_baseline_m(chosen_anchors, calibrations),
        "assignment_cost": best["cost"],
        "assignment_margin": margin,
        "active_cameras": active_cameras,
        "active_camera_count": len(active_cameras),
        "hands": hands,
    }
    side_quality, side_weak = _complete_row_side_diagnostics(complete_row, args)
    if side_weak:
        # A complete geometric assignment can technically pass the global
        # reprojection thresholds while one side is supported by a much weaker
        # stereo branch.  Re-run the independent path so that the strong side
        # is not hard-bound to the weak side's detector/depth decision.
        partial_accepted, partial_row = _fuse_partial_hands(
            sync_index, groups, calibrations, cameras, preferred_anchors, args
        )
        if partial_accepted:
            partial_row["partial_reason"] = "one_side_independently_weak"
            partial_row["complete_side_quality"] = side_quality
            return True, partial_row
        complete_row["side_quality_warning"] = side_quality
    return True, complete_row


def _compact_branch_reference(row: dict[str, Any]) -> dict[str, Any]:
    """Keep adaptive-mode diagnostics without embedding a second full frame."""
    hands = []
    for hand in row.get("hands", []):
        center = _hand_center_from_row(hand)
        hands.append({
            "side": int(hand.get("side", -1)),
            "center_base_m": (
                center.tolist() if np.all(np.isfinite(center)) else None
            ),
            "valid_joint_count": hand.get("quality", {}).get("valid_joint_count"),
            "reprojection_median_px": hand.get("quality", {}).get(
                "multiview_reprojection_median_px"
            ),
            "detector_mode": _hand_detector_mode(hand),
        })
    return {
        "accepted": bool(row.get("hands")),
        "anchor_cameras": row.get("anchor_cameras"),
        "anchor_baseline_m": row.get("anchor_baseline_m"),
        "assignment_cost": row.get("assignment_cost"),
        "partial_hand_acceptance": bool(row.get("partial_hand_acceptance")),
        "reason": row.get("reason"),
        "reasons": row.get("reasons"),
        "hands": hands,
    }


def _fuse_primary_frame(
    sync_index: int,
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    """Fuse one frame, treating detector handedness as a weak prior.

    Adaptive mode evaluates both the detector-constrained and geometry-only
    solutions.  The old implementation always returned a successful strict
    solution, even when a detector flip had forced it onto the wrong depth
    branch.  Comparing the geometric assignment costs lets the trajectory
    pass choose the physically coherent branch later.
    """
    if bool(getattr(args, "independent_hand_fusion", 0)):
        return _independent_frame_fusion(
            sync_index, prediction_rows, calibrations, cameras,
            preferred_anchors, args,
        )
    if args.detector_handedness != "adaptive":
        return _fuse_primary_frame_mode(
            sync_index, prediction_rows, calibrations, cameras,
            preferred_anchors, args,
        )
    strict_args = argparse.Namespace(**vars(args))
    strict_args.detector_handedness = "strict"
    accepted, strict_row = _fuse_primary_frame_mode(
        sync_index, prediction_rows, calibrations, cameras,
        preferred_anchors, strict_args,
    )
    ignore_args = argparse.Namespace(**vars(args))
    ignore_args.detector_handedness = "ignore"
    rescued, rescued_row = _fuse_primary_frame_mode(
        sync_index, prediction_rows, calibrations, cameras,
        preferred_anchors, ignore_args,
    )
    if accepted and rescued:
        strict_partial = bool(strict_row.get("partial_hand_acceptance"))
        ignore_partial = bool(rescued_row.get("partial_hand_acceptance"))
        if strict_partial != ignore_partial:
            # Prefer a complete row only when *both* of its hands are
            # independently well supported.  A detector-constrained complete
            # row with one weak side is the classic hard-binding failure: it
            # looks globally valid but drags the good side onto the wrong
            # identity/depth branch.  In that case retain the partial geometry
            # solution and let the missing side recover later.
            strict_reliable = _full_row_is_reliable(strict_row, args)
            ignore_reliable = _full_row_is_reliable(rescued_row, args)
            if strict_partial:
                chosen = rescued_row if ignore_reliable else strict_row
            else:
                chosen = strict_row if strict_reliable else rescued_row
            chosen["handedness_mode"] = (
                "ignore_fallback" if chosen is rescued_row and strict_partial
                else ("ignore_geometry" if chosen is rescued_row else "strict")
            )
            if chosen is rescued_row:
                chosen["strict_reference"] = _compact_branch_reference(strict_row)
            return True, chosen
        strict_cost = float(strict_row.get("assignment_cost", np.inf))
        ignore_cost = float(rescued_row.get("assignment_cost", np.inf))
        strict_hand_count = len(strict_row.get("hands", []))
        ignore_hand_count = len(rescued_row.get("hands", []))
        if ignore_hand_count != strict_hand_count:
            chosen = rescued_row if ignore_hand_count > strict_hand_count else strict_row
            chosen["handedness_mode"] = (
                "ignore_geometry" if chosen is rescued_row else "strict"
            )
            if chosen is rescued_row:
                chosen["strict_reference"] = _compact_branch_reference(strict_row)
            else:
                chosen["geometry_reference"] = _compact_branch_reference(rescued_row)
            return True, chosen
        # A one-unit margin is intentionally small relative to the reprojection
        # and anatomy terms; strict remains the deterministic tie-break.
        if np.isfinite(ignore_cost) and ignore_cost + 1.0 < strict_cost:
            rescued_row["handedness_mode"] = "ignore_geometry"
            rescued_row["strict_reference"] = _compact_branch_reference(strict_row)
            return True, rescued_row
        strict_row["handedness_mode"] = "strict"
        strict_row["geometry_reference"] = _compact_branch_reference(rescued_row)
        return True, strict_row
    if rescued:
        rescued_row["handedness_mode"] = "ignore_fallback"
        rescued_row["strict_rejection"] = _compact_branch_reference(strict_row)
        return True, rescued_row
    strict_row["handedness_mode"] = "adaptive_rejected"
    strict_row["fallback_attempted"] = True
    return False, strict_row


_FUSION_WORKER_CONTEXT: tuple[Any, ...] | None = None


def _initialize_fusion_worker(
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
) -> None:
    global _FUSION_WORKER_CONTEXT
    _FUSION_WORKER_CONTEXT = (
        prediction_rows, calibrations, cameras, preferred_anchors, args,
    )


def _fuse_primary_frame_worker(sync_index: int) -> tuple[bool, dict[str, Any]]:
    if _FUSION_WORKER_CONTEXT is None:
        raise RuntimeError("fusion worker was not initialized")
    return _fuse_primary_frame(sync_index, *_FUSION_WORKER_CONTEXT)


def _primary_results(
    frame_ids: list[int],
    prediction_rows: dict[str, dict[int, dict[str, Any]]],
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    preferred_anchors: tuple[str, str],
    args: argparse.Namespace,
):
    if args.workers == 1:
        for sync_index in frame_ids:
            yield _fuse_primary_frame(
                sync_index, prediction_rows, calibrations, cameras,
                preferred_anchors, args,
            )
        return
    start_methods = multiprocessing.get_all_start_methods()
    method = "fork" if "fork" in start_methods else start_methods[0]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context(method),
        initializer=_initialize_fusion_worker,
        initargs=(prediction_rows, calibrations, cameras, preferred_anchors, args),
    ) as executor:
        yield from executor.map(_fuse_primary_frame_worker, frame_ids, chunksize=1)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if (
        args.workers < 1
        or args.progress_interval < 1
        or args.max_frames < 0
        or args.anchor_border_penalty < 0.0
        or args.anchor_border_margin_px < 0.0
        or args.tip_three_view_penalty < 0.0
        or args.duplicate_bbox_iou < 0.0
        or args.duplicate_bbox_iou > 1.0
        or args.duplicate_joint_distance_px < 0.0
        or args.duplicate_overlap_cameras < 1
        or args.duplicate_center_distance_m < 0.0
        or args.duplicate_center_min_distinct_cameras < 1
        or args.min_full_hand_support < 0.0
        or args.full_hand_support_gap < 0.0
        or args.identity_partial_switch_penalty < 0.0
        or args.identity_partial_switch_max_cost < 0.0
        or args.identity_2d_soft_motion_px < 0.0
        or args.identity_2d_hard_motion_px < args.identity_2d_soft_motion_px
        or args.identity_2d_max_gap < 0
        or args.identity_2d_hard_break_penalty < 0.0
        or args.identity_2d_anchor_radius < 0
        or args.identity_fill_max_gap < 0
        or args.identity_fill_max_2d_motion_px < 0.0
        or args.gesture_smoothing_radius < 0
        or args.gesture_smoothing_blend < 0.0
        or args.gesture_smoothing_blend > 1.0
        or args.gesture_smoothing_max_motion_m <= 0.0
        or args.gesture_smoothing_max_residual_m <= 0.0
        or args.gesture_smoothing_max_adjustment_m <= 0.0
        or args.final_sequence_max_targets < 1
        or args.final_sequence_min_improvement_m < 0.0
        or args.final_sequence_delete_step_m < args.final_sequence_max_step_m
        or args.final_sequence_max_replacement_shift_m < 0.0
        or args.partial_continuity_radius < 0
        or args.partial_continuity_min_improvement < 0.0
        or args.partial_continuity_transition_weight < 0.0
        or args.continuity_min_palm_support < 0.0
        or args.continuity_min_palm_points < 3
        or args.continuity_min_branch_valid_joints < 1
        or args.continuity_branch_cost_delta < 0.0
        or args.continuity_branch_min_improvement_m < 0.0
        or args.continuity_branch_candidates_per_observation < 1
        or args.track_continuity_boundary_radius < 0
        or args.track_continuity_max_targets < 1
        or args.track_continuity_min_improvement < 0.0
        or args.track_reconnect_max_gap < 0
        or args.track_reconnect_max_2d_motion_px < 0.0
        or args.track_reconnect_max_3d_step_m < 0.0
        or args.track_reconnect_depth_penalty < 0.0
        or args.track_reselection_min_depth_improvement_m < 0.0
        or args.tracking_low_redundancy_max_adjustment_m < 0.0
        or args.depth_regularization_radius < 1
        or args.depth_regularization_trigger_m < 0.0
        or args.depth_regularization_max_2d_motion_px < 0.0
        or args.depth_regularization_blend < 0.0
        or args.depth_regularization_blend > 1.0
        or args.temporal_depth_guard_radius < 1
        or args.temporal_depth_guard_trigger_m < 0.0
        or args.temporal_depth_guard_max_2d_motion_px < 0.0
        or args.temporal_depth_guard_max_step_m < 0.0
        or args.temporal_depth_guard_max_bracket_speed_m < 0.0
        or args.temporal_depth_guard_max_shift_m < 0.0
        or args.temporal_depth_guard_max_run_length < 1
        or args.temporal_depth_guard_max_reprojection_regression_px < 0.0
        or args.temporal_depth_guard_max_reprojection_p95_regression_px < 0.0
        or args.temporal_depth_guard_min_shared_cameras < 1
        or args.depth_preferred_ray_angle_deg < 0.0
        or args.depth_min_palm_angle_deg < 0.0
        or args.depth_min_palm_three_view < 0
        or args.depth_max_pair_spread_m <= 0.0
        or args.depth_unsafe_pair_spread_m <= 0.0
        or args.depth_unsafe_pair_spread_m < args.depth_max_pair_spread_m
        or args.depth_condition_weight < 0.0
        or args.secondary_hand_min_valid_joints < 1
        or args.secondary_hand_min_palm_points < 3
        or args.secondary_hand_min_palm_angle_deg < 0.0
        or args.secondary_hand_max_cost_gap < 0.0
    ):
        raise ValueError(
            "invalid workers/progress/max-frames or duplicate-observation thresholds"
        )
    anatomy_config = FusionAnatomyConfig(
        outlier_window=args.anatomy_outlier_window,
        outlier_distance_m=args.anatomy_outlier_distance_m,
        bone_outlier_relative=args.anatomy_bone_relative,
        smoothing_radius=args.anatomy_smoothing_radius,
        local_shape_strength=args.anatomy_shape_strength,
        reliable_adjustment_blend=args.anatomy_reliable_adjustment_blend,
        max_reliable_adjustment_m=args.anatomy_max_reliable_adjustment_m,
        max_reprojection_regression_px=(
            args.anatomy_max_reprojection_regression_px
        ),
        max_reprojection_shift_px=args.anatomy_max_reprojection_shift_px,
        temporal_radius=args.anatomy_temporal_radius,
        temporal_palm_residual_m=args.anatomy_temporal_palm_residual_m,
        temporal_local_residual=args.anatomy_temporal_local_residual,
        tracking_radius=args.tracking_radius,
        tracking_min_neighbors_per_side=args.tracking_min_neighbors_per_side,
        tracking_min_good_joints=args.tracking_min_good_joints,
        tracking_min_reference_support=args.tracking_min_reference_support,
        tracking_max_reference_corrections=args.tracking_max_reference_corrections,
        tracking_apply_support=args.tracking_apply_support,
        tracking_max_reprojection_median_px=args.tracking_max_reprojection_median_px,
        tracking_max_reprojection_p95_px=args.tracking_max_reprojection_p95_px,
        tracking_palm_residual_m=args.tracking_palm_residual_m,
        tracking_shape_residual=args.tracking_shape_residual,
        tracking_blend=args.tracking_blend,
        tracking_max_adjustment_m=args.tracking_max_adjustment_m,
        tracking_low_redundancy_max_adjustment_m=(
            args.tracking_low_redundancy_max_adjustment_m
        ),
        gesture_smoothing_enabled=bool(args.gesture_smoothing),
        gesture_smoothing_radius=args.gesture_smoothing_radius,
        gesture_smoothing_blend=args.gesture_smoothing_blend,
        gesture_smoothing_max_motion_m=args.gesture_smoothing_max_motion_m,
        gesture_smoothing_max_residual_m=args.gesture_smoothing_max_residual_m,
        gesture_smoothing_max_adjustment_m=args.gesture_smoothing_max_adjustment_m,
        tracking_allow_low_redundancy_rigid=bool(
            args.tracking_allow_low_redundancy_rigid
        ),
    )
    anatomy_config.validate()
    fusion_started = time.perf_counter()
    dataset = args.dataset.resolve()
    prediction_root = args.predictions.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    available_cameras = tuple(manifest["camera_ids"])
    cameras = tuple(dict.fromkeys(args.cameras or available_cameras))
    if len(cameras) < 2:
        raise ValueError("fusion requires at least two cameras")
    unknown_cameras = [camera for camera in cameras if camera not in available_cameras]
    if unknown_cameras:
        raise ValueError(f"selected cameras are not present in the dataset: {unknown_cameras}")
    preferred_anchors = tuple(args.anchor_cameras)
    if len(set(preferred_anchors)) != 2 or any(camera not in cameras for camera in preferred_anchors):
        raise ValueError(f"invalid anchor cameras: {preferred_anchors}")
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    prediction_rows = {
        camera: _load_jsonl(prediction_root / camera / "predictions.jsonl")
        for camera in cameras
    }
    prediction_summary_path = prediction_root / "summary.json"
    prediction_summary = (
        json.loads(prediction_summary_path.read_text(encoding="utf-8"))
        if prediction_summary_path.is_file() else {}
    )
    # The first synchronized row may legitimately contain no hands.  The
    # producer stage is sequence-level metadata and is therefore the reliable
    # signal for enabling the marker-evidence guard in auto mode.
    marker_assisted_input = (
        "marker" in str(prediction_summary.get("stage", "")).lower()
    )
    args._require_marker_evidence = (
        marker_assisted_input
        if args.require_marker_evidence == "auto"
        else bool(int(args.require_marker_evidence))
    )
    frame_ids = sorted(set.intersection(*(set(rows) for rows in prediction_rows.values())))
    frame_ids = frame_ids[: args.max_frames or None]
    accepted = []
    rejected = []
    primary_started = time.perf_counter()
    results = _primary_results(
        frame_ids, prediction_rows, calibrations, cameras, preferred_anchors, args
    )
    for completed, (is_accepted, row) in enumerate(results, start=1):
        (accepted if is_accepted else rejected).append(row)
        if completed % args.progress_interval == 0 or completed == len(frame_ids):
            elapsed = time.perf_counter() - primary_started
            print(
                f"fusion: {completed}/{len(frame_ids)} | "
                f"{completed / max(elapsed, 1e-9):.2f} frames/s | "
                f"workers={args.workers}",
                flush=True,
            )
    primary_seconds = time.perf_counter() - primary_started
    primary_accepted_count = len(accepted)
    temporal_recovered = []
    if args.temporal_recovery_gap > 0 and accepted:
        references = sorted(accepted, key=lambda row: int(row["sync_index"]))
        remaining_rejected = []
        for rejection in rejected:
            sync_index = int(rejection["sync_index"])
            groups = {
                camera: _groups(prediction_rows[camera][sync_index]) for camera in cameras
            }
            recovered_parts = []
            # Recovery is hand-level: a visible side must not be discarded
            # because the other side has left every camera.  Use the nearest
            # reference independently for each physical side.
            for side in (0, 1):
                side_references = [
                    row for row in references
                    if any(int(hand.get("side", -1)) == side for hand in row.get("hands", []))
                ]
                if not side_references:
                    continue
                reference = min(
                    side_references,
                    key=lambda row: abs(int(row["sync_index"]) - sync_index),
                )
                if abs(int(reference["sync_index"]) - sync_index) > args.temporal_recovery_gap:
                    continue
                recovered = _recover_from_reference(
                    sync_index, reference, groups, calibrations, cameras, args,
                    sides=(side,),
                )
                if recovered is not None:
                    recovered_parts.extend(recovered.get("hands", []))
            if not recovered_parts:
                remaining_rejected.append(rejection)
            else:
                present = {int(hand["side"]) for hand in recovered_parts}
                active = sorted(set().union(*(
                    set(hand.get("camera_ids", [])) for hand in recovered_parts
                )))
                temporal_recovered.append({
                    "sync_index": sync_index,
                    "anchor_cameras": list(args.anchor_cameras),
                    "recovery": {
                        "method": "nearest_confirmed_multiview_per_hand",
                        "frame_gap": min(
                            abs(sync_index - int(row["sync_index"]))
                            for row in references
                        ),
                    },
                    "active_cameras": active,
                    "active_camera_count": len(active),
                    "hands": recovered_parts,
                    "partial_hand_acceptance": len(present) < 2,
                    "missing_sides": sorted({0, 1} - present),
                })
        accepted.extend(temporal_recovered)
        accepted.sort(key=lambda row: int(row["sync_index"]))
        rejected = remaining_rejected
    # The older partial and full Viterbi passes were designed independently and
    # can successively overwrite one another's branch/identity decisions.  The
    # unified track pass below is now the sole continuity owner by default;
    # retain the legacy passes only as an explicit compatibility switch.
    if getattr(args, "legacy_continuity_passes", 0):
        partial_continuity_reselected_count = _partial_continuity_reselect(
            accepted, prediction_rows, calibrations, cameras, preferred_anchors, args
        )
        continuity_reselected_count = _continuity_reselect(
            accepted, prediction_rows, calibrations, cameras, preferred_anchors, args
        )
    else:
        partial_continuity_reselected_count = 0
        continuity_reselected_count = 0
    accepted.sort(key=lambda row: int(row["sync_index"]))
    # Seed physical tracks before the unified selector.  This gives the
    # full/partial boundary pass a stable side context while its own geometric
    # transition cost remains permutation-invariant.
    identity_relabel_before = _stabilize_hand_identities(
        accepted,
        partial_switch_penalty=args.identity_partial_switch_penalty,
        partial_switch_max_cost=args.identity_partial_switch_max_cost,
        args=args,
    )
    track_continuity_reselected_count = _track_continuity_reselect(
        accepted, prediction_rows, calibrations, cameras, preferred_anchors, args
    )
    accepted.sort(key=lambda row: int(row["sync_index"]))
    # Canonicalize once more after branch selection.  A branch alternative may
    # carry detector-local labels, so the final pass writes the persistent
    # physical IDs used by anatomy, rendering, and MANO export.
    identity_relabel_after = _stabilize_hand_identities(
        accepted,
        partial_switch_penalty=args.identity_partial_switch_penalty,
        partial_switch_max_cost=args.identity_partial_switch_max_cost,
        args=args,
    )
    # Geometry/depth selection can leave a detector-local label on a partial
    # row even after the 3-D DP has finished.  Canonicalize once from the raw
    # shared-camera image tracks before anatomy builds temporal references;
    # this is the pass that distinguishes a detector flip from a genuine
    # hand re-entry.
    identity_relabel_2d = _stabilize_hand_identities_2d(accepted, args)
    identity_partial_repair_count = _repair_partial_identity_labels_2d(
        accepted, args
    )
    identity_partial_repair_count += _repair_partial_identity_by_persistent_image_track(
        accepted, args
    )
    _annotate_2d_temporal_segments(accepted, args)
    identity_fill_count = _fill_short_missing_hand_tracks(accepted, args)
    temporal_depth_guard_hand_count = _apply_temporal_depth_guard(
        accepted, calibrations, args
    )
    depth_regularized_hand_count = _apply_low_support_depth_regularization(
        accepted, calibrations, args
    )
    partial_side_relabel_count = sum(
        int(bool(
            row.get("partial_hand_acceptance")
            and row.get("identity_tracking", {}).get("relabelled", False)
        ))
        for row in accepted
    )
    hybrid_branch_reselected_count = sum(
        int(bool(row.get("hybrid_branch_selection", {}).get("changed")))
        for row in accepted
    )
    temporal_track_boundary_count = sum(
        len(row.get("temporal_track_boundary_before", []))
        for row in accepted
    )
    anatomy_summary = None
    if args.anatomy_refinement:
        anatomy_summary = refine_accepted_rows(
            accepted, len(cameras), anatomy_config, calibrations
        )
    # Anatomy refinement can intentionally invalidate a hand (for example an
    # all-outlier depth branch).  Prune those hands *before* the final sequence
    # guard.  The guard uses the nearest preceding/following rows as its
    # bracket; if an invalid row is left in that bracket and is only removed
    # afterwards, a replacement can be selected against a neighbour that will
    # not exist in the delivered stream.  That ordering produced a fresh jump
    # at frame 277 in the v43 recording after frame 278 was pruned.
    accepted, dropped_before_guard = _prune_unusable_hands(accepted)
    if dropped_before_guard:
        rejected.extend(dropped_before_guard)
    final_sequence_summary = _final_sequence_guard(
        accepted, prediction_rows, calibrations, cameras, preferred_anchors, args
    )
    # A raw alternate selected by the guard should normally be valid, but keep
    # the post-condition explicit in case a future candidate builder emits an
    # incomplete hand.  This second pass is intentionally after the guard and
    # does not feed its removed rows back into the guard's bracket.
    accepted, dropped_after_guard = _prune_unusable_hands(accepted)
    if dropped_after_guard:
        rejected.extend(dropped_after_guard)
    reasons = Counter()
    for rejection in rejected:
        if "reason" in rejection:
            reasons[rejection["reason"]] += 1
        else:
            reasons.update(rejection.get("reasons", []))
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in accepted),
        encoding="utf-8",
    )
    (output / "rejected.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rejected),
        encoding="utf-8",
    )
    hands = [hand for row in accepted for hand in row["hands"]]
    view_counts = [len(hand["camera_ids"]) for hand in hands]
    joint_support = [
        count for hand in hands for count in hand["inlier_view_counts"] if count >= 2
    ]
    multiview_hands = sum(hand["fusion_mode"] != "stereo_anchor" for hand in hands)
    anchor_pair_counts = Counter(
        "+".join(row["anchor_cameras"]) for row in accepted
    )
    camera_contribution_counts = Counter(
        camera
        for hand in hands for camera, view in hand["views"].items()
        if int(view.get("inlier_joint_count", 0)) > 0
    )
    detector_handedness_mismatches = sum(
        1
        for hand in hands for view in hand["views"].values()
        if int(view.get("inlier_joint_count", 0)) > 0
        and view.get("detector_is_right") is not None
        and int(view["detector_is_right"]) != int(hand["side"])
    )
    handedness_fallback_count = sum(
        row.get("handedness_mode") == "ignore_fallback" for row in accepted
    )
    marker_assisted_views = [
        view
        for hand in hands for view in hand["views"].values()
        if view.get("marker_assist", {}).get("applied")
    ]
    marker_match_counts = [
        int(view["marker_assist"]["matched_marker_count"])
        for view in marker_assisted_views
    ]
    marker_evidence_views = [
        view
        for hand in hands for view in hand["views"].values()
        if view.get("marker_assist", {}).get("applied")
        or view.get("marker_assist", {}).get("evidence_only")
    ]
    marker_evidence_only_views = [
        view
        for view in marker_evidence_views
        if view.get("marker_assist", {}).get("evidence_only")
    ]
    stereo_cross_view = [
        hand["stereo_baseline_comparison"]["cross_view_reprojection_median_px"]
        for hand in hands
        if hand["stereo_baseline_comparison"].get("cross_view_reprojection_median_px") is not None
    ]
    multiview_cross_view = [
        hand["quality"]["all_view_reprojection_median_px"]
        for hand in hands
        if hand["stereo_baseline_comparison"].get("cross_view_reprojection_median_px") is not None
    ]
    differences_mm = [
        hand["stereo_baseline_comparison"]["multiview_3d_difference_median_mm"]
        for hand in hands
        if hand["stereo_baseline_comparison"].get("multiview_3d_difference_median_mm") is not None
    ]
    stereo_cross_median = float(np.median(stereo_cross_view)) if stereo_cross_view else None
    multiview_cross_median = float(np.median(multiview_cross_view)) if multiview_cross_view else None
    fusion_seconds = time.perf_counter() - fusion_started
    summary = {
        "schema_version": 1,
        "stage": (
            "wilor_anchor_guided_multiview_fusion_with_glove_marker_assist"
            if marker_assisted_views
            else "wilor_anchor_guided_multiview_fusion"
        ),
        "camera_ids": list(cameras),
        "preferred_anchor_cameras": list(preferred_anchors),
        "selected_anchor_pair_counts": dict(anchor_pair_counts),
        "camera_contribution_hand_counts": {
            camera: camera_contribution_counts.get(camera, 0) for camera in cameras
        },
        "detector_handedness_mismatch_observation_count": detector_handedness_mismatches,
        "marker_assisted_view_count": len(marker_assisted_views),
        "marker_evidence_view_count": len(marker_evidence_views),
        "marker_evidence_only_view_count": len(marker_evidence_only_views),
        "marker_matched_count_median": (
            float(np.median(marker_match_counts)) if marker_match_counts else None
        ),
        "processed_frame_count": len(frame_ids),
        "accepted_frame_count": len(accepted),
        "primary_accepted_frame_count": primary_accepted_count,
        "temporal_recovered_frame_count": len(temporal_recovered),
        "continuity_reselected_frame_count": continuity_reselected_count,
        "track_continuity_reselected_frame_count": (
            track_continuity_reselected_count
        ),
        "hybrid_branch_reselected_frame_count": (
            hybrid_branch_reselected_count
        ),
        "temporal_track_boundary_count": temporal_track_boundary_count,
        "identity_relabel_before_track_count": identity_relabel_before,
        "partial_continuity_reselected_frame_count": (
            partial_continuity_reselected_count
        ),
        "depth_regularized_hand_count": depth_regularized_hand_count,
        "temporal_depth_guard_hand_count": temporal_depth_guard_hand_count,
        "partial_hand_frame_count": sum(
            bool(row.get("partial_hand_acceptance")) for row in accepted
        ),
        "partial_side_relabel_count": partial_side_relabel_count,
        "identity_relabel_count": identity_relabel_after,
        "identity_relabel_2d_count": identity_relabel_2d,
        "identity_partial_repair_count": identity_partial_repair_count,
        "identity_short_gap_fill_count": identity_fill_count,
        "final_sequence_guard": final_sequence_summary,
        "missing_side_counts": dict(Counter(
            int(side) for row in accepted
            for side in row.get("missing_sides", [])
        )),
        "rejected_frame_count": len(rejected),
        "accepted_rate": len(accepted) / max(len(frame_ids), 1),
        "rejected_reason_counts": dict(reasons),
        "hand_count": len(hands),
        "multiview_hand_count": multiview_hands,
        "multiview_hand_rate": multiview_hands / max(len(hands), 1),
        "selected_view_count_median": float(np.median(view_counts)) if view_counts else None,
        "valid_joint_inlier_view_count_median": (
            float(np.median(joint_support)) if joint_support else None
        ),
        "valid_joint_inlier_view_count_distribution": {
            str(count): joint_support.count(count) for count in sorted(set(joint_support))
        },
        "reprojection_median_px": (
            float(np.median([hand["quality"]["multiview_reprojection_median_px"] for hand in hands]))
            if hands else None
        ),
        "fusion_workers": args.workers,
        "primary_fusion_seconds": primary_seconds,
        "fusion_seconds": fusion_seconds,
        "fusion_frames_per_second": len(frame_ids) / max(fusion_seconds, 1e-9),
        "stereo_anchor_comparison": {
            "comparable_hand_count": len(stereo_cross_view),
            "stereo_cross_view_median_px": stereo_cross_median,
            "multiview_cross_view_median_px": multiview_cross_median,
            "cross_view_median_improvement_percent": (
                100.0 * (1.0 - multiview_cross_median / stereo_cross_median)
                if stereo_cross_median and multiview_cross_median is not None else None
            ),
            "stereo_multiview_3d_difference_median_mm": (
                float(np.median(differences_mm)) if differences_mm else None
            ),
        },
        "parameters": {
            "association_threshold_px": args.association_threshold_px,
            "anchor_threshold_px": args.anchor_threshold_px,
            "anchor_border_penalty": args.anchor_border_penalty,
            "anchor_border_margin_px": args.anchor_border_margin_px,
            "tip_three_view_penalty": args.tip_three_view_penalty,
            "ransac_threshold_px": args.ransac_threshold_px,
            "duplicate_bbox_iou": args.duplicate_bbox_iou,
            "duplicate_joint_distance_px": args.duplicate_joint_distance_px,
            "duplicate_overlap_cameras": args.duplicate_overlap_cameras,
            "duplicate_center_distance_m": args.duplicate_center_distance_m,
            "duplicate_center_min_distinct_cameras": (
                args.duplicate_center_min_distinct_cameras
            ),
            "min_full_hand_support": args.min_full_hand_support,
            "full_hand_support_gap": args.full_hand_support_gap,
            "min_valid_joints": args.min_valid_joints,
            "identity_partial_switch_penalty": args.identity_partial_switch_penalty,
            "identity_partial_switch_max_cost": args.identity_partial_switch_max_cost,
            "identity_2d_soft_motion_px": args.identity_2d_soft_motion_px,
            "identity_2d_hard_motion_px": args.identity_2d_hard_motion_px,
            "identity_2d_max_gap": args.identity_2d_max_gap,
            "identity_2d_hard_break_penalty": args.identity_2d_hard_break_penalty,
            "identity_2d_anchor_radius": args.identity_2d_anchor_radius,
            "identity_fill_max_gap": args.identity_fill_max_gap,
            "identity_fill_max_2d_motion_px": args.identity_fill_max_2d_motion_px,
            "gesture_smoothing": args.gesture_smoothing,
            "gesture_smoothing_radius": args.gesture_smoothing_radius,
            "gesture_smoothing_blend": args.gesture_smoothing_blend,
            "gesture_smoothing_max_motion_m": args.gesture_smoothing_max_motion_m,
            "gesture_smoothing_max_residual_m": args.gesture_smoothing_max_residual_m,
            "gesture_smoothing_max_adjustment_m": args.gesture_smoothing_max_adjustment_m,
            "partial_continuity_radius": args.partial_continuity_radius,
            "partial_continuity_min_improvement": args.partial_continuity_min_improvement,
            "partial_continuity_transition_weight": args.partial_continuity_transition_weight,
            "temporal_recovery_gap": args.temporal_recovery_gap,
            "temporal_association_threshold_px": args.temporal_association_threshold_px,
            "continuity_reselection": args.continuity_reselection,
            "continuity_weight": args.continuity_weight,
            "continuity_max_step_m": args.continuity_max_step_m,
            "continuity_wrist_max_step_m": args.continuity_wrist_max_step_m,
            "continuity_switch_penalty": args.continuity_switch_penalty,
            "continuity_min_support": args.continuity_min_support,
            "continuity_min_palm_support": args.continuity_min_palm_support,
            "continuity_min_palm_points": args.continuity_min_palm_points,
            "continuity_min_branch_valid_joints": (
                args.continuity_min_branch_valid_joints
            ),
            "continuity_max_cost_delta": args.continuity_max_cost_delta,
            "continuity_branch_cost_delta": args.continuity_branch_cost_delta,
            "continuity_branch_min_improvement_m": (
                args.continuity_branch_min_improvement_m
            ),
            "track_continuity": args.track_continuity,
            "track_continuity_boundary_radius": (
                args.track_continuity_boundary_radius
            ),
            "track_continuity_max_targets": args.track_continuity_max_targets,
            "track_continuity_min_improvement": (
                args.track_continuity_min_improvement
            ),
            "track_reconnect_max_gap": args.track_reconnect_max_gap,
            "track_reconnect_max_2d_motion_px": (
                args.track_reconnect_max_2d_motion_px
            ),
            "track_reconnect_max_3d_step_m": (
                args.track_reconnect_max_3d_step_m
            ),
            "track_reconnect_depth_penalty": (
                args.track_reconnect_depth_penalty
            ),
            "track_reselection_min_depth_improvement_m": (
                args.track_reselection_min_depth_improvement_m
            ),
            "continuity_max_hypotheses": args.continuity_max_hypotheses,
            "continuity_branch_candidates_per_observation": (
                args.continuity_branch_candidates_per_observation
            ),
            "continuity_max_anchor_pairs": args.continuity_max_anchor_pairs,
            "continuity_target_step_m": args.continuity_target_step_m,
            "continuity_target_bracket_m": args.continuity_target_bracket_m,
            "continuity_bracket_weight": args.continuity_bracket_weight,
            "continuity_bracket_residual_m": args.continuity_bracket_residual_m,
            "anchor_baseline_reference_m": args.anchor_baseline_reference_m,
            "anchor_baseline_penalty": args.anchor_baseline_penalty,
            "detector_handedness": args.detector_handedness,
            "require_marker_evidence": bool(args._require_marker_evidence),
            "independent_hand_fusion": args.independent_hand_fusion,
            "depth_regularization": args.depth_regularization,
            "depth_regularization_radius": args.depth_regularization_radius,
            "depth_regularization_trigger_m": args.depth_regularization_trigger_m,
            "depth_regularization_max_2d_motion_px": (
                args.depth_regularization_max_2d_motion_px
            ),
            "depth_regularization_blend": args.depth_regularization_blend,
            "temporal_depth_guard": args.temporal_depth_guard,
            "temporal_depth_guard_radius": args.temporal_depth_guard_radius,
            "temporal_depth_guard_trigger_m": args.temporal_depth_guard_trigger_m,
            "temporal_depth_guard_max_2d_motion_px": (
                args.temporal_depth_guard_max_2d_motion_px
            ),
            "temporal_depth_guard_max_step_m": args.temporal_depth_guard_max_step_m,
            "temporal_depth_guard_max_bracket_speed_m": (
                args.temporal_depth_guard_max_bracket_speed_m
            ),
            "temporal_depth_guard_max_shift_m": args.temporal_depth_guard_max_shift_m,
            "temporal_depth_guard_max_run_length": args.temporal_depth_guard_max_run_length,
            "temporal_depth_guard_max_reprojection_regression_px": (
                args.temporal_depth_guard_max_reprojection_regression_px
            ),
            "temporal_depth_guard_max_reprojection_p95_regression_px": (
                args.temporal_depth_guard_max_reprojection_p95_regression_px
            ),
            "temporal_depth_guard_min_shared_cameras": (
                args.temporal_depth_guard_min_shared_cameras
            ),
            "depth_preferred_ray_angle_deg": args.depth_preferred_ray_angle_deg,
            "depth_min_palm_angle_deg": args.depth_min_palm_angle_deg,
            "depth_min_palm_three_view": args.depth_min_palm_three_view,
            "depth_max_pair_spread_m": args.depth_max_pair_spread_m,
            "depth_condition_weight": args.depth_condition_weight,
            "secondary_hand_min_valid_joints": args.secondary_hand_min_valid_joints,
            "secondary_hand_min_palm_points": args.secondary_hand_min_palm_points,
            "secondary_hand_min_palm_angle_deg": args.secondary_hand_min_palm_angle_deg,
            "secondary_hand_max_cost_gap": args.secondary_hand_max_cost_gap,
            "final_sequence_guard": args.final_sequence_guard,
            "final_sequence_max_step_m": args.final_sequence_max_step_m,
            "final_sequence_max_2d_motion_px": args.final_sequence_max_2d_motion_px,
            "final_sequence_max_gap": args.final_sequence_max_gap,
            "final_sequence_min_improvement_m": args.final_sequence_min_improvement_m,
            "final_sequence_delete_step_m": args.final_sequence_delete_step_m,
            "final_sequence_max_replacement_shift_m": (
                args.final_sequence_max_replacement_shift_m
            ),
            "final_sequence_max_targets": args.final_sequence_max_targets,
            "legacy_continuity_passes": args.legacy_continuity_passes,
            "workers": args.workers,
        },
        "anatomy_refinement": anatomy_summary or {
            "enabled": False,
            "parameters": anatomy_config.to_dict(),
        },
    }
    if args.detector_handedness == "adaptive":
        summary["handedness_fallback_frame_count"] = handedness_fallback_count
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
