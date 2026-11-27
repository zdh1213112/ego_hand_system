from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

from wilor.training.config import load_training_config


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("args", "recipe"), [
    ([], "nokov_v5_bbox_pinhole"),
    (["--mode", "joint_lora"], "nokov_v5_bbox_pinhole"),
    (["--mode=joint_lora"], "nokov_v5_bbox_pinhole"),
    (["--config", "configs/glove/nokov_v4_joint_lora.yaml"], "nokov_v4_joint_lora"),
    (["--config", "configs/glove/nokov_v4_bbox_pinhole.yaml"], "nokov_v4_bbox_pinhole"),
])
def test_training_cli_selects_recipe(monkeypatch, args, recipe):
    import train_glove

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["train_glove.py", *args])
    configs = []
    monkeypatch.setattr(train_glove, "run_training", configs.append)
    assert train_glove.main() == 0
    config, = configs
    assert config.source == REPO_ROOT / "configs/glove" / f"{recipe}.yaml"
    if recipe == "nokov_v5_bbox_pinhole":
        assert config.local_export is not None
        assert config.camera_model == "physical_perspective"


def test_prepare_only_uses_default_v5_config(monkeypatch, tmp_path):
    import train_glove
    from wilor.datasets import local_pinhole

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_glove.py", "--prepare-only", "--detector-cache", "boxes.jsonl"])
    configs = []
    monkeypatch.setattr(local_pinhole, "prepare_local_views", configs.append)
    assert train_glove.main() == 0
    config, = configs
    assert config.source == REPO_ROOT / "configs/glove/nokov_v5_bbox_pinhole.yaml"
    assert config.local_export.detector_cache == tmp_path / "boxes.jsonl"


def test_schema_v3_configs_select_explicit_data_and_recipe() -> None:
    domain = load_training_config(
        REPO_ROOT / "configs/glove/domain_adapt.yaml"
    )
    full_vit = load_training_config(
        REPO_ROOT
        / "configs/glove/domain_adapt_opentouch_original_full_vit.yaml"
    )
    refiner = load_training_config(
        REPO_ROOT / "configs/glove/refiner_finetune.yaml"
    )
    assert domain.data_type == "opentouch"
    assert domain.recipe_type == "domain_adapt"
    assert domain.vit_tuning == "lora"
    assert domain.lora is not None and domain.lora.blocks == "all"
    assert full_vit.data_type == "opentouch"
    assert full_vit.recipe_type == "domain_adapt"
    assert full_vit.vit_tuning == "full"
    assert full_vit.lora is None
    assert full_vit.batch_size * full_vit.gradient_accumulation_steps == 128
    assert refiner.data_type == "glove_camera"
    assert refiner.recipe_type == "refiner_finetune"
    assert refiner.vit_tuning is None


def test_domain_recipe_rejects_camera_loss(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs/glove/domain_adapt.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["losses"]["keypoints_2d"] = 1.0
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="does not allow"):
        load_training_config(invalid)


def test_three_selected_training_contents_are_valid() -> None:
    from train_glove import MODE_CONFIGS, _mode_config

    expected = {
        "joint_lora": ("glove_camera", "joint_lora_finetune"),
        "lora_stage1": ("opentouch", "domain_adapt"),
        "lora_stage2": ("glove_camera", "refiner_finetune"),
    }
    for mode, (data_type, recipe_type) in expected.items():
        config = _mode_config(REPO_ROOT / MODE_CONFIGS[mode], mode)
        assert (config.data_type, config.recipe_type) == (data_type, recipe_type)
    assert _mode_config(
        REPO_ROOT / MODE_CONFIGS["joint_lora"], "joint_lora"
    ).lora is not None


def test_joint_lora_requires_camera_labels_and_lora(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs/glove/joint_lora_finetune.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["recipe"]["lora"] = None
    invalid = tmp_path / "missing_lora.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="requires recipe.vit_tuning"):
        load_training_config(invalid)
