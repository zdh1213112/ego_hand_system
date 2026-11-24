"""Durable JSON/YAML reporting and headless training-curve rendering."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


def _atomic_write(path: str | Path, payload: str) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, indent=2) + "\n")


def write_jsonl(path: str | Path, values: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(value, separators=(",", ":")) + "\n" for value in values
        ),
    )


def write_yaml(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(dict(value), sort_keys=False))


def _rolling_mean(values: Sequence[float], window: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return array
    width = min(int(window), len(array))
    result = np.empty_like(array)
    cumulative = np.cumsum(array, dtype=np.float64)
    for index in range(len(array)):
        start = max(0, index - width + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        result[index] = total / (index - start + 1)
    return result


def write_training_curves(
    history: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    """Render loss, MPJPE, and weighted component curves from report history."""
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([int(record["step"]) for record in history])
    train_loss = [float(record["train_loss"]) for record in history]
    metric_name = str(history[0].get("metric_name", "mpjpe_mm"))
    train_metric = [
        float(record.get("train_metric", record.get("train_mpjpe_mm", 0.0)))
        for record in history
    ]
    val_records = [record for record in history if "val_loss" in record]
    val_steps = [int(record["step"]) for record in val_records]

    figure, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    axes[0].plot(steps, train_loss, color="tab:blue", alpha=0.2, label="train raw")
    axes[0].plot(
        steps, _rolling_mean(train_loss), color="tab:blue", label="train mean(20)"
    )
    if val_records:
        axes[0].plot(
            val_steps,
            [float(record["val_loss"]) for record in val_records],
            color="tab:orange",
            marker="o",
            label="val",
        )
    axes[0].set(title="Total loss", xlabel="step", ylabel="loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        steps, train_metric, color="tab:green", alpha=0.2, label="train raw"
    )
    axes[1].plot(
        steps,
        _rolling_mean(train_metric),
        color="tab:green",
        label="train mean(20)",
    )
    if val_records:
        axes[1].plot(
            val_steps,
            [
                float(record.get("val_metric", record.get("val_mpjpe_mm", 0.0)))
                for record in val_records
            ],
            color="tab:red",
            marker="o",
            label="val",
        )
    axes[1].set(title=metric_name, xlabel="step", ylabel=metric_name)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    component_names = sorted({
        name
        for record in history
        for name in record.get("train_loss_components", {})
        if name.startswith("weighted_")
    })
    for name in component_names:
        values = [
            float(record.get("train_loss_components", {}).get(name, 0.0))
            for record in history
        ]
        line = axes[2].plot(
            steps, _rolling_mean(values), label=f"train {name.removeprefix('weighted_')}"
        )[0]
        matching_val = [
            (int(record["step"]), float(record["val_loss_components"][name]))
            for record in val_records
            if name in record.get("val_loss_components", {})
        ]
        if matching_val:
            axes[2].plot(
                [value[0] for value in matching_val],
                [value[1] for value in matching_val],
                linestyle="none",
                marker="o",
                color=line.get_color(),
            )
    axes[2].set(title="Weighted loss components", xlabel="step", ylabel="loss")
    axes[2].grid(alpha=0.25)
    if component_names:
        axes[2].legend(ncol=2, fontsize=8)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".png",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=160, format="png")
        os.replace(temporary, destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def plot_training_report(
    report_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    source = Path(report_path).expanduser().resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None else source.with_name("loss_curves.png")
    )
    write_training_curves(report.get("history", []), destination)
    return destination


def _main() -> int:
    parser = argparse.ArgumentParser(description="Plot a WiLoR train_report.json")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(plot_training_report(args.report, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
