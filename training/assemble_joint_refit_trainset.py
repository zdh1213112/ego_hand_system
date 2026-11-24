#!/usr/bin/env python3
"""Collect finished per-recording label exports into one training dataset.

Scans a work root for runs of the one-command pipeline (export_ego_nokov_labels.sh
--optimization-config) and merges every recording whose optimized labels passed
the dataset check. The time holdout keeps the last fraction of each recording's
timeline for validation, so adding a recording never reshuffles the others.
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def finished_exports(work_root: Path):
    for commands in sorted(work_root.glob('*/commands.json')):
        run = commands.parent
        stages = [row['stage'] for row in json.loads(commands.read_text())]
        if 'joint_refit' not in stages:
            continue
        dataset = run / 'optimized/dense/fit/final/export/merged'
        check = run / 'optimized/dense/fit/final/export/logs/check.log'
        if (dataset / 'index.jsonl').is_file() and check.is_file() and 'maximum_mano_joint_replay_error_m' in check.read_text():
            yield dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-root', type=Path, default=ROOT / 'work')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--extra', nargs='*', type=Path, default=[], help='additional export/merged directories not under the work root')
    a = ap.parse_args()
    inputs = list(finished_exports(a.work_root)) + [p.resolve() for p in a.extra]
    if not inputs:
        raise SystemExit(f'no finished joint_refit exports under {a.work_root}')
    print(f'[assemble] {len(inputs)} recordings', flush=True)
    for dataset in inputs:
        print(f'  {dataset}', flush=True)
    subprocess.run([PY, '-B', str(ROOT / 'ego_hand_system/scripts/merge_wilor_training_datasets.py'),
                    '--inputs', *map(str, inputs), '--output', str(a.output), '--hardlink'], check=True)
    subprocess.run([PY, '-B', str(ROOT / 'wilor_training/prepare_glove_dataset.py'),
                    '--root', str(a.output), '--mano-path', str(ROOT / 'wilor_training/mano_data/mano_v1_2'),
                    '--holdout-by-time', '--train-ratio', str(a.train_ratio)], check=True)
    splits = json.loads((a.output / 'splits.json').read_text())
    print(json.dumps(dict(recordings=len(inputs), samples=splits['samples'], sides=splits['side_counts']), indent=2))


if __name__ == '__main__':
    main()
