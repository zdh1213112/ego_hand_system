#!/usr/bin/env python3
"""Merge validated WiLoR image/NPY datasets without changing label contents."""

from __future__ import annotations

import argparse
import hashlib
import os
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from wilor_frame_annotations import write_frame_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hardlink", action="store_true",
                        help="hard-link images/labels instead of copying (same filesystem only)")
    return parser.parse_args()


def _place(source: Path, target: Path, hardlink: bool) -> None:
    if hardlink:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": digest,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _common(summaries: list[dict[str, Any]], key: str) -> Any:
    values = [summary.get(key) for summary in summaries]
    return values[0] if all(value == values[0] for value in values) else None


def _source_experiment_name(dataset: Path) -> str:
    return dataset.parent.name


def main() -> int:
    args = parse_args()
    inputs = tuple(dict.fromkeys(path.resolve() for path in args.inputs))
    output = args.output.resolve()
    if len(inputs) < 1:
        raise ValueError("at least one input dataset is required")

    source_records = []
    summaries = []
    for dataset in inputs:
        required = (
            dataset / "summary.json", dataset / "index.jsonl",
            dataset / "images", dataset / "labels",
        )
        if not all(path.exists() for path in required):
            raise FileNotFoundError(f"incomplete WiLoR dataset: {dataset}")
        summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
        if summary.get("image_space") == "native_raw":
            if summary.get("schema_version") != 6 or not summary.get("camera_calibrations"):
                raise ValueError("native inputs require schema 6 and per-camera calibrations")
            if summary.get("stage") == "merged_wilor_training_dataset":
                raise ValueError("use original per-recording native datasets as merge inputs")
        rows = _load_jsonl(dataset / "index.jsonl")
        if int(summary.get("sample_count", -1)) != len(rows):
            raise ValueError(f"summary/index count mismatch: {dataset}")
        for row in rows:
            image = dataset / row["image"]
            label = dataset / row["label"]
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"missing image/label pair in {dataset}: {row}")
        source_records.append({
            "dataset": dataset,
            "summary": summary,
            "rows": rows,
            "summary_signature": _signature(dataset / "summary.json"),
            "index_signature": _signature(dataset / "index.jsonl"),
        })
        summaries.append(summary)

    merge_config = {
        "schema_version": 1,
        "sources": [
            {
                "dataset": str(record["dataset"]),
                "summary": record["summary_signature"],
                "index": record["index_signature"],
            }
            for record in source_records
        ],
    }
    config_path = output / "merge_config.json"
    if output.exists():
        if config_path.is_file() and (output / "summary.json").is_file():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous == merge_config:
                print(f"[merge] reuse completed dataset: {output}")
                return 0
        raise FileExistsError(
            f"merged output already exists with different or incomplete inputs: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        image_root = temporary / "images"
        label_root = temporary / "labels"
        image_root.mkdir()
        label_root.mkdir()
        merged_rows = []
        merged_rejected = []
        side_counts = {"left": 0, "right": 0}
        camera_counts: dict[str, int] = {}
        index = 0
        shared_images = {}
        for source_id, record in enumerate(source_records):
            dataset = record["dataset"]
            source_name = (record["summary"].get("source_group") or dataset.name) if record["summary"]["image_space"] == "native_raw" else _source_experiment_name(dataset)
            for row in record["rows"]:
                stem = f"{index:08d}"
                image_key = (source_id, row["image"])
                image_name = shared_images.get(image_key)
                if image_name is None:
                    image_name = f"{len(shared_images):08d}.jpg"
                    shared_images[image_key] = image_name
                    _place(dataset / row["image"], image_root / image_name, args.hardlink)
                _place(dataset / row["label"], label_root / f"{stem}.npy", args.hardlink)
                merged = dict(row)
                merged.update({
                    "index": index,
                    "image": f"images/{image_name}",
                    "label": f"labels/{stem}.npy",
                    "source_dataset_id": source_id,
                    "source_dataset": str(dataset),
                    "source_experiment": source_name,
                    "source_index": int(row.get("index", -1)),
                    "source_image": row["image"],
                    "source_label": row["label"],
                    "calibration_group": str(source_id),
                })
                merged_rows.append(merged)
                side_name = "right" if int(row["side"]) else "left"
                side_counts[side_name] += 1
                camera = str(row["camera"])
                camera_counts[camera] = camera_counts.get(camera, 0) + 1
                index += 1
            rejected_path = dataset / "rejected.jsonl"
            if rejected_path.is_file():
                for row in _load_jsonl(rejected_path):
                    merged_rejected.append({
                        **row, "source_dataset_id": source_id,
                        "source_dataset": str(dataset),
                        "source_experiment": source_name,
                    })

        write_frame_index(temporary / "frames.jsonl", merged_rows)
        (temporary / "index.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in merged_rows),
            encoding="utf-8",
        )
        (temporary / "rejected.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in merged_rejected),
            encoding="utf-8",
        )
        first = summaries[0]
        required_common = (
            "mano_model_by_side", "mano_pose_representation", "image_space",
            "geometry_space", "mano_parameter_space", "side_semantics", "mano_assets",
        )
        for key in required_common:
            if _common(summaries, key) is None:
                raise ValueError(f"source datasets use incompatible {key}")
        merged_summary = {
            "schema_version": 6 if first["image_space"] == "native_raw" else 4,
            "stage": "merged_wilor_training_dataset",
            "schema_reference": first.get("schema_reference", "000865.npy"),
            "sample_count": len(merged_rows),
            "image_count": len(shared_images),
            "image_layout": "shared_camera_frames",
            "annotation_policy": _common(summaries, "annotation_policy") or "available_hands",
            "label_count": len(merged_rows),
            "source_dataset_count": len(source_records),
            "source_datasets": [str(record["dataset"]) for record in source_records],
            "label_generation_by_source": {str(i): {k: record["summary"].get(k) for k in ("multiview_mano_optimization_enabled", "multiview_strategy", "refinement_safety_by_track", "fit_error_point_semantics", "personal_marker24_optimization_enabled", "joint_pose_attachment_optimization_enabled", "approved_for_training")} for i, record in enumerate(source_records)},
            "side_counts": side_counts,
            "camera_counts": camera_counts,
            "rejected_count": len(merged_rejected),
            "image_size": _common(summaries, "image_size"),
            "K": _common(summaries, "K"),
            "cameras": sorted(camera_counts),
            "view_filter": _common(summaries, "view_filter") or "mixed",
            "source": "validated per-recording WiLoR training datasets",
            "mano_model_by_side": first["mano_model_by_side"],
            "mano_pose_representation": first["mano_pose_representation"],
            "mano_model": first.get("mano_model", "MANO_RIGHT.pkl"),
            "image_space": first["image_space"],
            "geometry_space": first["geometry_space"],
            "mano_parameter_space": first["mano_parameter_space"],
            "side_semantics": first["side_semantics"],
            "mano_source_revision": _common(summaries, "mano_source_revision") or "mixed",
            "mano_assets": first["mano_assets"],
        }
        if any(s.get("personal_marker24_optimization_enabled") for s in summaries):
            merged_summary.update(personal_marker24_optimization_enabled=True, joint_pose_attachment_optimization_enabled=any(s.get('joint_pose_attachment_optimization_enabled') for s in summaries), approved_for_training=False)
        if first["image_space"] == "native_raw":
            merged_summary["camera_calibrations_by_source"] = {
                str(source_id): record["summary"]["camera_calibrations"]
                for source_id, record in enumerate(source_records)
            }
            merged_summary["projection_fields"] = first.get("projection_fields", {})
        (temporary / "summary.json").write_text(
            json.dumps(merged_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "merge_config.json").write_text(
            json.dumps(merge_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(merged_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
