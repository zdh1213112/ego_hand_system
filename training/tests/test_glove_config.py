from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wilor.training.glove import (
    load_glove_training_config,
    resolved_training_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_complete_stage2_config_preserves_separate_evaluation_batch_size() -> None:
    source = REPO_ROOT / "configs" / "glove" / "stage2_mano.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = load_glove_training_config(source)
    assert config.config_source == source.resolve()
    assert config.batch_size == raw["training"]["batch_size"]
    assert config.evaluation_batch_size == 16
    assert config.val_batches is None
    assert config.final_visualization
    assert config.losses["hand_pose"] == 0.001
    resolved = resolved_training_config(config)
    assert resolved["training"]["batch_size"] == config.batch_size
    assert resolved["validation"]["batch_size"] == 16
    assert Path(resolved["paths"]["checkpoint"]).is_absolute()


def test_training_config_rejects_unknown_fields(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs" / "glove" / "stage1_keypoints.yaml"
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    value["training"]["batch_szie"] = 4
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="batch_szie"):
        load_glove_training_config(invalid)


def test_training_config_rejects_resume_and_init_adapter(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs" / "glove" / "stage1_keypoints.yaml"
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    value["paths"]["resume"] = "output/old.pt"
    value["paths"]["init_adapter"] = "output/init.pt"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_glove_training_config(invalid)
