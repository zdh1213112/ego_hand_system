from __future__ import annotations

import json
from pathlib import Path

from wilor.training.reporting import plot_training_report, write_training_curves


def _history() -> list[dict]:
    return [
        {
            "step": step,
            "train_loss": 2.0 / step,
            "train_mpjpe_mm": 40.0 / step,
            "train_loss_components": {
                "weighted_keypoints_2d": 1.0 / step,
                "weighted_keypoints_3d": 1.0 / step,
            },
            **({
                "val_loss": 1.5 / step,
                "val_mpjpe_mm": 30.0 / step,
                "val_loss_components": {
                    "weighted_keypoints_2d": 0.7 / step,
                    "weighted_keypoints_3d": 0.8 / step,
                },
            } if step in (1, 3) else {}),
        }
        for step in range(1, 4)
    ]


def test_training_curves_write_and_rebuild_from_report(tmp_path: Path) -> None:
    direct = tmp_path / "direct.png"
    write_training_curves(_history(), direct)
    assert direct.is_file() and direct.stat().st_size > 1000

    report = tmp_path / "train_report.json"
    report.write_text(json.dumps({"history": _history()}), encoding="utf-8")
    rebuilt = tmp_path / "rebuilt.png"
    assert plot_training_report(report, rebuilt) == rebuilt.resolve()
    assert rebuilt.is_file() and rebuilt.stat().st_size > 1000
