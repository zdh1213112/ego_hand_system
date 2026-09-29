#!/usr/bin/env python3
"""Apply an explicit global physical-left/right side correction.

Multiview assignment can choose the two symmetric hand hypotheses in either
order when a clip starts with weak or glove-occluded detections.  This pass is
for that *global* ambiguity only: it swaps the exported ``side`` labels while
leaving every calibrated 3-D point and camera-native 2-D observation intact.
The source directory is copied to a new directory so the original fusion run
remains available for A/B comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _swap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        hands = []
        for hand in row.get("hands", []):
            try:
                side = int(hand.get("side", -1))
            except (TypeError, ValueError):
                hands.append(hand)
                continue
            if side not in (0, 1):
                hands.append(hand)
                continue
            hand["side"] = 1 - side
            identity = hand.get("final_identity")
            if isinstance(identity, dict):
                identity = dict(identity)
                identity["source_side_before_global_swap"] = identity.get(
                    "source_side", side
                )
                identity["source_side"] = 1 - int(identity.get("source_side", side))
                identity["global_physical_side_swap"] = True
                hand["final_identity"] = identity
            stability = hand.get("final_stability")
            if isinstance(stability, dict):
                stability = dict(stability)
                stability["global_physical_side_swap"] = True
                hand["final_stability"] = stability
            hands.append(hand)
        row["hands"] = sorted(
            hands, key=lambda value: int(value.get("side", 0))
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--output must be a new directory; source is preserved")
    if not (source / "accepted.jsonl").is_file():
        raise FileNotFoundError(source / "accepted.jsonl")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    shutil.copytree(source, output)

    accepted = _swap_rows(_read_jsonl(source / "accepted.jsonl"))
    _write_jsonl(output / "accepted.jsonl", accepted)

    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["physical_side_swap_applied"] = True
        summary["physical_side_convention"] = {
            "side_0": "physical_left",
            "side_1": "physical_right",
        }
        summary["physical_side_swap_source"] = str(source)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    report_path = output / "depth_branch_gate_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["physical_side_swap_applied"] = True
        report["physical_side_convention"] = {
            "side_0": "physical_left",
            "side_1": "physical_right",
        }
        report["physical_side_swap_source"] = str(source)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Swapped physical side labels: {source} -> {output}")
    print(f"Accepted rows rewritten: {len(accepted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
