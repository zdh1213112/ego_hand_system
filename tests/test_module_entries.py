"""Verify module routing without CUDA, optional model packages or external checkouts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def training_env(tmp_path):
    python = tmp_path / "show-arguments"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    return {**os.environ, "EGO_TRAIN_PYTHON": str(python)}


@pytest.mark.parametrize("opts", [[], ["--mode", "joint_lora"], ["--mode=joint_lora"]])
def test_public_wilor_default_is_v4(tmp_path, training_env, opts):
    result = subprocess.run(
        ["bash", ROOT / "training/train.sh", "wilor", *opts],
        cwd=tmp_path, env=training_env, text=True, capture_output=True, check=True,
    )
    assert result.stdout.splitlines()[-2:] == [
        "--config", str(ROOT / "training/configs/glove/nokov_v4_joint_lora.yaml")
    ]


@pytest.mark.parametrize("backend,script", [
    ("wilor", "train_glove.py"),
    ("fah", "train_hand.py"),
    ("gem-x", "train_hand.py"),
])
def test_training_routes_from_another_directory(tmp_path, training_env, backend, script):
    opts = ["--config", "my config.yaml"] if backend == "wilor" else ["--dataset", "my data"]
    result = subprocess.run(
        ["bash", ROOT / "training/train.sh", backend, *opts],
        cwd=tmp_path, env=training_env,
        text=True, capture_output=True, check=True,
    )
    selector = [] if backend == "wilor" else [backend]
    assert result.stdout.splitlines() == ["-B", str(ROOT / "training" / script), *selector, *opts]


@pytest.mark.parametrize("backend,option", [
    ("wilor", "--adapter"), ("fah", "--fah-weights"), ("gem-x", "--soma-assets"),
])
def test_inference_backend_help(tmp_path, backend, option):
    result = subprocess.run(
        ["bash", ROOT / "inference/run.sh", backend, "--help"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert option in result.stdout


def test_release_uses_the_observation_adapter():
    profile = json.loads((ROOT / "label_export/configs/native_nokov_v4.json").read_text())
    signed = json.loads((ROOT / "label_export/pipeline/assets/signed_cli_profile.json").read_text())
    weights = profile["label_pipeline"]["observation_weights"]
    assert weights == signed["label_pipeline"]["observation_weights"]
    assert Path(weights["adapter"]).parts[-2:] == ("observation_v2", "best_trainable.pt")
    assert (ROOT / "training/pretrained_models/model_config.yaml").is_file()


@pytest.mark.parametrize("entry,legacy,args", [
    ("label_export/export.sh", "scripts/export_ego_nokov_labels.sh", ["--help"]),
    ("training/train.sh", "scripts/train_glove.sh", ["--help"]),
    ("inference/run.sh", "scripts/run_hand_inference.sh", ["wilor", "--help"]),
])
def test_previous_commands_reach_the_same_module(tmp_path, entry, legacy, args):
    env = {**os.environ, "EGO_NATIVE_PYTHON": sys.executable}
    def help_text(script):
        return subprocess.run(
            ["bash", ROOT / script, *args], cwd=tmp_path, env=env,
            text=True, capture_output=True, check=True,
        ).stdout
    assert help_text(entry) == help_text(legacy)


def test_old_config_name_refers_to_the_module_profile():
    config = ROOT / "label_export/configs/native_nokov_v4.json"
    assert config.samefile(ROOT / "configs/native_nokov_v4.json")


def test_label_bundle_check_works_from_module_directory(tmp_path):
    track = b"initial fit"
    expected = {}
    for side in (0, 1):
        (tmp_path / f"track_{side}.npz").write_bytes(track)
        expected[str(side)] = hashlib.sha256(track).hexdigest()
    bundle = tmp_path / "attachments.npz"
    np.savez(bundle, schema="native_multiview_mano_v1",
             observation_source="independent_rgb_marker_detections",
             baseline_fit_sha256_json=json.dumps(expected))
    config = {"multiview_mano_optimization": {
        "enabled": True, "bundle": str(bundle), "initial_fit": str(tmp_path),
    }}
    # Direct execution of the label CLI exposes only its own directory to imports.
    subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys, json; sys.path.insert(0, sys.argv[1]); "
         "from label_generation_config import require_available_mode; "
         "require_available_mode(json.loads(sys.argv[2]))",
         str(ROOT / "label_export"), json.dumps(config)],
        cwd=tmp_path, check=True,
    )
