#!/usr/bin/env python3
"""Single YAML entry point for both WiLoR glove training stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from wilor.training import (
    load_glove_training_config,
    load_training_config,
    run_glove_training,
    run_training,
)


REPO_ROOT = Path(__file__).resolve().parent
MODE_CONFIGS = {
    "joint_lora": "configs/glove/nokov_v5_bbox_pinhole.yaml",
    "lora_stage1": "configs/glove/lora_stage1.yaml",
    "lora_stage2": "configs/glove/lora_stage2.yaml",
}
MODE_RECIPES = {
    "joint_lora": "joint_lora_finetune",
    "lora_stage1": "domain_adapt",
    "lora_stage2": "refiner_finetune",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select joint LoRA training or either stage of two-stage training."
    )
    parser.add_argument(
        "--mode",
        choices=(*MODE_CONFIGS, "two_stage"),
        help="training content; two_stage runs lora_stage1 then lora_stage2",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="config for a single training run (default: V5 local pinhole for joint LoRA)",
    )
    parser.add_argument("--stage1-config", type=Path)
    parser.add_argument("--stage2-config", type=Path)
    local = parser.add_argument_group("local pinhole export")
    boxes = local.add_mutually_exclusive_group()
    boxes.add_argument("--detector", type=Path, help="Use local YOLO weights for the export")
    boxes.add_argument("--detector-cache", type=Path, help="Use cached native-frame proposals for the export")
    local.add_argument("--prepare-only", action="store_true", help="Prepare or verify local views and exit")
    args = parser.parse_args()
    if (args.detector or args.detector_cache or args.prepare_only) and args.mode not in (None, "joint_lora"):
        parser.error("Local export options require a single local pinhole training config")
    return args


def _mode_config(path: Path, mode: str):
    config = load_training_config(path)
    expected = MODE_RECIPES[mode]
    if config.recipe_type != expected:
        raise ValueError(
            f"--mode {mode} requires recipe.type: {expected}, "
            f"got {config.recipe_type}"
        )
    if mode == "lora_stage1" and config.vit_tuning != "lora":
        raise ValueError("--mode lora_stage1 requires recipe.vit_tuning: lora")
    if mode == "lora_stage2" and config.init_weights is None and config.resume is None:
        raise ValueError(
            "--mode lora_stage2 requires paths.init_weights from Stage 1 "
            "or paths.resume"
        )
    return config


def main() -> int:
    args = parse_args()
    if args.mode == "two_stage":
        if args.config is not None:
            raise ValueError("--mode two_stage uses --stage1-config/--stage2-config")
        stage1_path = args.stage1_config or REPO_ROOT / MODE_CONFIGS["lora_stage1"]
        stage2_path = args.stage2_config or REPO_ROOT / MODE_CONFIGS["lora_stage2"]
        stage1 = _mode_config(stage1_path, "lora_stage1")
        stage2 = load_training_config(stage2_path)
        if stage2.recipe_type != "refiner_finetune":
            raise ValueError("--mode two_stage requires Stage 2 refiner_finetune")
        if stage2.resume is not None:
            raise ValueError("--mode two_stage cannot override Stage 2 resume")
        if stage1.output == stage2.output:
            raise ValueError("Stage 1 and Stage 2 must have different output paths")
        stage2 = replace(
            stage2,
            init_weights=stage1.output / "best_trainable.pt",
        )
        stage2.validate()
        first = run_training(stage1)
        second = run_training(stage2)
        print(json.dumps({"stage1": first, "stage2": second}, indent=2))
        return 0

    if args.stage1_config is not None or args.stage2_config is not None:
        raise ValueError("--stage1-config/--stage2-config require --mode two_stage")
    path = args.config or REPO_ROOT / MODE_CONFIGS[args.mode or "joint_lora"]
    raw = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version") if isinstance(raw, dict) else None
    if args.detector or args.detector_cache or args.prepare_only:
        if schema_version != 3 or "local_export" not in raw:
            raise ValueError("Local export options require a schema-v3 config with a local_export section")
    if schema_version == 3:
        config = _mode_config(path, args.mode) if args.mode else load_training_config(path)
        if args.detector or args.detector_cache:
            config.local_export = replace(
                config.local_export,
                detector=args.detector.expanduser().resolve() if args.detector else None,
                detector_cache=args.detector_cache.expanduser().resolve() if args.detector_cache else None,
            )
        if args.prepare_only:
            from wilor.datasets.local_pinhole import prepare_local_views
            report = prepare_local_views(config)
        else:
            report = run_training(config)
    elif schema_version == 2:
        if args.mode:
            raise ValueError(f"{path}: --mode {args.mode} requires a schema-v3 config")
        report = run_glove_training(load_glove_training_config(path))
    else:
        raise ValueError(
            f"{path}: expected schema_version 2 or 3, got {schema_version!r}"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
