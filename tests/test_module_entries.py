"""Verify module routing without CUDA, optional model packages or external checkouts."""
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend,script", [
    ("wilor", "train_glove.py"),
    ("fah", "train_hand.py"),
    ("gem-x", "train_hand.py"),
])
def test_training_routes_from_another_directory(tmp_path, backend, script):
    python = tmp_path / "show-arguments"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    opts = ["--config", "my config.yaml"] if backend == "wilor" else ["--dataset", "my data"]
    result = subprocess.run(
        ["bash", ROOT / "scripts/train_glove.sh", backend, *opts],
        cwd=tmp_path, env={**os.environ, "EGO_TRAIN_PYTHON": str(python)},
        text=True, capture_output=True, check=True,
    )
    selector = [] if backend == "wilor" else [backend]
    assert result.stdout.splitlines() == ["-B", str(ROOT / "training" / script), *selector, *opts]


@pytest.mark.parametrize("backend,option", [
    ("wilor", "--adapter"), ("fah", "--fah-weights"), ("gem-x", "--soma-assets"),
])
def test_inference_backend_help(tmp_path, backend, option):
    result = subprocess.run(
        ["bash", ROOT / "scripts/run_hand_inference.sh", backend, "--help"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert option in result.stdout


def test_release_uses_the_observation_adapter():
    profile = json.loads((ROOT / "configs/native_nokov_v4.json").read_text())
    signed = json.loads((ROOT / "tools/label_pipeline/assets/signed_cli_profile.json").read_text())
    weights = profile["label_pipeline"]["observation_weights"]
    assert weights == signed["label_pipeline"]["observation_weights"]
    assert Path(weights["adapter"]).parts[-2:] == ("observation_v2", "best_trainable.pt")
    assert (ROOT / "training/pretrained_models/model_config.yaml").is_file()
