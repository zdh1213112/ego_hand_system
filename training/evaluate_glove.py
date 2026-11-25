#!/usr/bin/env python3
"""Evaluate a base WiLoR checkpoint or adapter on a glove manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wilor.datasets.glove_dataset import GloveDataset
from wilor.datasets.glove_validation import sha256_file, validate_manifest, manifest_camera_model
from wilor.models.adapter import checkpoint_rescale_factor, load_adapter_weights
from wilor.training.evaluation import evaluate_glove_set
from wilor.training.glove import (
    GloveTrainingConfig,
    load_glove_model,
    resolve_device,
    seed_everything,
    set_finetune_mode,
)
from wilor.training.reporting import write_json


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WiLoR on a prepared glove validation manifest."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_models/wilor_final.ckpt",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "pretrained_models/model_config.yaml",
    )
    parser.add_argument(
        "--mano-path",
        type=Path,
        default=REPO_ROOT / "mano_data/mano_v1_2",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="optional fine-tuned adapter; omit it for the original checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--marker-input", action="append", default=[], metavar="SOURCE_EXPERIMENT=NPZ",
        help="Aligned Nokov marker archive for marker21 evaluation; repeat per recording",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument('--camera-model', choices=['auto', 'virtual', 'physical_perspective', 'physical_kb', 'kb_bbox_pinhole'], default='auto')
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one GT/prediction overlay image per RGB frame.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    root = _resolved(args.root)
    manifest = _resolved(args.manifest or root / "val.jsonl")
    checkpoint = _resolved(args.checkpoint)
    model_config = _resolved(args.model_config)
    mano_path = _resolved(args.mano_path)
    adapter = _resolved(args.adapter) if args.adapter is not None else None
    output = _resolved(args.output)

    for name, path in (
        ("manifest", manifest),
        ("checkpoint", checkpoint),
        ("model config", model_config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if adapter is not None and not adapter.is_file():
        raise FileNotFoundError(f"Missing adapter: {adapter}")

    geometry = validate_manifest(manifest, max_samples=args.max_samples)
    if not geometry["valid"]:
        raise RuntimeError(json.dumps(geometry, indent=2))

    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    marker_inputs = {}
    for entry in args.marker_input:
        if "=" not in entry:
            raise ValueError("--marker-input must be SOURCE_EXPERIMENT=NPZ")
        name, path = entry.split("=", 1)
        marker_inputs[name] = _resolved(Path(path))
    config = GloveTrainingConfig(
        root=root,
        checkpoint=checkpoint,
        model_config=model_config,
        mano_path=mano_path,
        output=output,
    )
    model = load_glove_model(config, device)
    rescale_factor = getattr(model, "crop_rescale_factor", 2.0)
    if adapter is not None:
        rescale_factor = checkpoint_rescale_factor(load_adapter_weights(adapter, model))
    camera_mode = manifest_camera_model(manifest) if args.camera_model == 'auto' else args.camera_model
    if model.physical_camera_enabled and model.physical_camera_head.camera_model == 'kb_bbox_pinhole':
        if manifest_camera_model(manifest) != 'physical_kb' or args.camera_model not in ('auto', 'kb_bbox_pinhole'):
            raise ValueError('The bbox-pinhole adapter requires native KB images and rectified crops')
        camera_mode = 'kb_bbox_pinhole'
    dataset = GloveDataset(
        manifest, max_samples=args.max_samples, marker_inputs=marker_inputs,
        camera_model=camera_mode, rescale_factor=rescale_factor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.camera_model != 'auto':
        model.physical_camera_enabled = args.camera_model != 'virtual'
        if model.physical_camera_enabled:
            model.physical_camera_head.camera_model = args.camera_model
    elif dataset.camera_model == 'physical_kb':
        if model.physical_camera_enabled and model.physical_camera_head.camera_model != 'physical_kb':
            raise ValueError('Checkpoint camera mode does not match native KB images')
        model.physical_camera_enabled = True
        model.physical_camera_head.camera_model = 'physical_kb'
    elif model.physical_camera_enabled and model.physical_camera_head.camera_model == 'physical_kb':
        raise ValueError('Native KB checkpoint requires native camera images')
    if model.marker_head_enabled and not marker_inputs:
        raise ValueError("Marker checkpoint evaluation requires --marker-input")
    set_finetune_mode(model, training=False)

    metrics = evaluate_glove_set(
        model, loader, device, output, use_amp, save_overlays=args.save_overlays
    )
    report = {
        "mode": "adapter" if adapter is not None else "base_checkpoint",
        "manifest": str(manifest),
        "samples": len(dataset),
        "device": str(device),
        "amp": use_amp,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "adapter": (
            {"path": str(adapter), "sha256": sha256_file(adapter)}
            if adapter is not None else None
        ),
        "point_type": "marker21" if model.marker_head_enabled else "mano21",
        "metrics": metrics,
    }
    write_json(output / "evaluation_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
