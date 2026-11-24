#!/usr/bin/env python3
"""Prepare either camera-aligned glove data or decoded OpenTouch data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wilor.datasets.glove_dataset import make_splits
from wilor.datasets.glove_validation import build_dataset_profile
from wilor.datasets.opentouch_dataset import (
    initialize_empty_bbox_annotations,
    prepare_opentouch_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-type", choices=("glove", "opentouch"), default="glove"
    )
    parser.add_argument("--root", type=Path, default=Path("datasets/dataset"))
    parser.add_argument(
        "--mano-path", type=Path, default=Path("mano_data/mano_v1_2"),
        help="MANO release/model directory used to replay label parameters",
    )
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--holdout-by-time", action="store_true",
                        help="hold out the last fraction of each recording's timeline")
    parser.add_argument("--split-by-source-group", action="store_true",
                        help="keep entire recording sessions in one split")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--projection-tolerance-px", type=float, default=1e-3)
    parser.add_argument("--mano-reconstruction-tolerance-m", type=float, default=1e-5)
    parser.add_argument(
        "--decoded-root", type=Path, default=Path("datasets/opentouch_decoded")
    )
    parser.add_argument(
        "--bbox-annotations",
        type=Path,
        default=Path("datasets/opentouch_bbox_annotations.jsonl"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("datasets/opentouch_training")
    )
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--init-empty-bboxes", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_type == "opentouch":
        decoded_root = args.decoded_root.expanduser().resolve()
        bbox_annotations = args.bbox_annotations.expanduser().resolve()
        if args.init_empty_bboxes:
            count = initialize_empty_bbox_annotations(
                decoded_root,
                bbox_annotations,
                overwrite=args.force,
            )
            print(json.dumps({
                "bbox_annotations": str(bbox_annotations),
                "samples": count,
                "usable": 0,
            }, indent=2))
            return 0
        result = prepare_opentouch_dataset(
            decoded_root,
            bbox_annotations,
            args.output_root,
            sample_fps=args.sample_fps,
            source_fps=args.source_fps,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
        )
        print(json.dumps(result, indent=2))
        return 0

    root = args.root.expanduser().resolve()
    profile = build_dataset_profile(
        root,
        args.mano_path.expanduser().resolve(),
        projection_tolerance_px=args.projection_tolerance_px,
        mano_reconstruction_tolerance_m=args.mano_reconstruction_tolerance_m,
    )
    print(json.dumps(profile, indent=2))
    if not profile["geometry_validation"]["valid"]:
        return 1
    if not args.validate_only:
        print(json.dumps(make_splits(
            root,
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
            group_by_source=args.split_by_source_group,
            holdout_by_time=args.holdout_by_time,
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
