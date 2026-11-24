#!/usr/bin/env python3
"""Evaluate final shared results from run_multiview_*_experiment.sh.

Examples: --runs wilor=RUN_WILOR gem=RUN_GEM fah=RUN_FAH, or
--backend wilor --run RUN_WILOR. A run may also be the shared geometry folder.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from evaluate_hand_models import render_table, write_json
from hand_evaluation.dataset import LabeledDataset
from hand_evaluation.metrics import ERROR_NAMES, stats
from hand_evaluation.shared import SharedResult, evaluate_shared


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=('wilor', 'gem', 'fah'), default='wilor')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=Path)
    group.add_argument('--runs', nargs='+', metavar='BACKEND=RUN')
    parser.add_argument('--normalized-dataset', type=Path)
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--image-stride', type=int, default=1)
    parser.add_argument('--gt-consistency-mm', type=float, default=.1)
    parser.add_argument('--single-view-root', type=Path, help='optional previous evaluate_hand_models.py --backends output')
    parser.add_argument('--resume', action='store_true', help='re-score the same run; reject changed data or upstream results')
    args = parser.parse_args(argv)
    if not np.isfinite(args.gt_consistency_mm) or args.gt_consistency_mm <= 0:
        parser.error('gt-consistency-mm must be positive and finite')
    if args.runs and args.normalized_dataset:
        parser.error('--normalized-dataset is only supported with a single --run')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def load_errors(directory):
    with np.load(directory / 'joint_errors.npz', allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def compare_shared(output, reports):
    reference = reports[0]
    if any(r['dataset']['fingerprint'] != reference['dataset']['fingerprint'] or r['protocol'] != reference['protocol'] for r in reports):
        raise ValueError('shared comparisons require identical labeled samples and protocols')
    write_json(output / 'comparison.json', reports)
    render_table(output / 'comparison.html', reports)
    errors = {r['backend']: load_errors(output / r['backend']) for r in reports}
    first = next(iter(errors.values()))
    common = np.ones(len(first['ids']), bool)
    for data in errors.values():
        if not np.array_equal(data['ids'], first['ids']) or not np.array_equal(data['valid'], first['valid']):
            raise ValueError('shared comparison IDs or label masks differ')
        common &= np.all(np.isfinite(data['root_relative']) | ~data['valid'], axis=1)
    paired = dict(common_pose_samples=int(common.sum()), total_samples=len(common), sample_ids=first['ids'][common].tolist(), metrics={})
    for name, data in errors.items():
        paired['metrics'][name] = {key: stats(data[key][common][data['valid'][common]]) for key in ERROR_NAMES}
    write_json(output / 'comparison_common_samples.json', paired)


def compare_single(output, shared_dirs, single_root):
    result = dict(note='Paired camera/hand labels; single-view 2D model outputs vs final shared 3D projections. '
                      'Single-view calibrated translation and shared saved placement differ. '
                      'Do not attribute all differences to fusion if input/proposal protocols differ.', backends={})
    for backend, directory in shared_dirs.items():
        single_dir = single_root / backend
        first_summary = json.loads((single_dir / 'summary.json').read_text())
        shared_summary = json.loads((directory / 'summary.json').read_text())
        if first_summary['dataset']['fingerprint'] != shared_summary['dataset']['fingerprint']:
            raise ValueError(f'{backend}: single/shared label sets differ')
        first, second = load_errors(single_dir), load_errors(directory)
        if not np.array_equal(first['ids'], second['ids']) or not np.array_equal(first['valid'], second['valid']):
            raise ValueError('single/shared IDs or masks differ')
        valid = first['valid']
        common = np.all((np.isfinite(first['root_relative']) & np.isfinite(second['root_relative'])) | ~valid, axis=1)
        result['backends'][backend] = dict(total_samples=len(common), common_pose_samples=int(common.sum()),
            single_protocol=first_summary['protocol'], shared_protocol=shared_summary['protocol'],
            single_coverage=first_summary['overall']['pose_success'], shared_coverage=shared_summary['overall']['pose_success'],
            paired_metrics={name: dict(single=stats(first[name][common][valid[common]]),
                                      shared=stats(second[name][common][valid[common]])) for name in ERROR_NAMES})
    write_json(output / 'single_vs_shared_comparison.json', result)


def main():
    args = parse_args()
    dataset = LabeledDataset(args.dataset, cameras=args.cameras, image_stride=args.image_stride, max_images=args.max_images)
    if args.run:
        source = SharedResult(args.run, args.backend, args.normalized_dataset)
        evaluate_shared(args, dataset, source)
        if args.single_view_root:
            compare_single(args.output, {args.backend: args.output}, args.single_view_root)
        return
    runs = {}
    for item in args.runs:
        name, separator, path = item.partition('=')
        if not separator or name not in ('wilor', 'gem', 'fah') or name in runs or not path:
            raise ValueError(f'invalid or duplicate BACKEND=RUN: {item}')
        runs[name] = Path(path).expanduser().resolve()
    root, reports = args.output, []
    root.mkdir(parents=True, exist_ok=True)
    for name, path in runs.items():
        args.backend, args.output = name, root / name
        source = SharedResult(path, name)
        reports.append(evaluate_shared(args, dataset, source))
        del source
    compare_shared(root, reports)
    if args.single_view_root:
        compare_single(root, {name: root / name for name in runs}, args.single_view_root)


if __name__ == '__main__':
    main()
