#!/usr/bin/env python3
"""Re-run anatomy refinement from the unrefined joints of a saved fusion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from ego_data.calibration import CameraCalibration
from fusion_anatomy_refinement import FusionAnatomyConfig, refine_accepted_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    fusion = args.fusion.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")

    source_summary = json.loads(
        (fusion / "summary.json").read_text(encoding="utf-8")
    )
    parameters = source_summary["anatomy_refinement"]["parameters"]
    config = FusionAnatomyConfig(**parameters)
    cameras = tuple(source_summary["camera_ids"])
    calibrations = {
        camera: CameraCalibration.load(dataset / "calibration" / f"{camera}.json")
        for camera in cameras
    }
    rows = [
        json.loads(line) for line in (fusion / "accepted.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    for row in rows:
        for hand in row.get("hands", []):
            unrefined = hand.get("unrefined_joints_base_m")
            if unrefined is None:
                raise ValueError(
                    f"sync {row.get('sync_index')} hand {hand.get('side')} has no "
                    "unrefined_joints_base_m"
                )
            hand["joints_base_m"] = copy.deepcopy(unrefined)
            hand.pop("anatomy_refinement", None)

    anatomy_summary = refine_accepted_rows(
        rows, len(cameras), config, calibrations
    )
    output.mkdir(parents=True)
    (output / "accepted.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    rejected = fusion / "rejected.jsonl"
    if rejected.is_file():
        shutil.copyfile(rejected, output / "rejected.jsonl")
    candidate_summary = copy.deepcopy(source_summary)
    candidate_summary["stage"] = (
        str(source_summary.get("stage", "fusion")) + "_anatomy_rerun"
    )
    candidate_summary["source_fusion"] = str(fusion)
    candidate_summary["anatomy_refinement"] = anatomy_summary
    (output / "summary.json").write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
