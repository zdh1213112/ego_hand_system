#!/usr/bin/env python3
"""Export, validate and merge native KB labels from a recording list.

Example (both recordings, all qualifying samples):
  bash scripts/run_native_wilor_export.sh --output /path/to/new_output

Quick sample run:
  bash scripts/run_native_wilor_export.sh --output /path/to/new_output \
    --sample-stride 80 --max-samples 32

Outputs: recordings/<name>/, merged/, preview/, logs/, commands.json.
The input recordings have already been synchronized and fitted; this command
does not perform acquisition, synchronization or MANO refitting. A recording
may contain either or both physical hands. No train/validation split is made.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/native_ego_glove_export.json')
    parser.add_argument('--output', type=Path, required=True, help='new directory for all recordings and merged labels')
    parser.add_argument('--sample-stride', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=0, help='per-recording limit; 0 means all')
    parser.add_argument('--require-both-hands', action='store_true', help='keep only frames with valid left and right pose labels')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--preview-samples', type=int, default=4)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.max_samples < 0 or args.preview_samples < 1:
        parser.error('stride and preview samples must be positive; max samples must be nonnegative')
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    recordings = config['recordings']
    names = [r['name'] for r in recordings]
    if not names or len(set(names)) != len(names) or any(not n or Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('recording names must be unique, nonempty directory basenames')
    for item in recordings:
        for key in ('fusion', 'mano_fit', 'dataset'):
            if not Path(item[key]).is_absolute() or not Path(item[key]).exists():
                raise FileNotFoundError(f'{item["name"]}: {key}={item[key]}')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'input_config.json').write_text(json.dumps(config, indent=2) + '\n')
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONPATH', None)
    commands = []

    def run(stage, script, arguments):
        command = [sys.executable, '-B', str(ROOT / 'scripts' / script), *map(str, arguments)]
        log = output / 'logs' / f'{stage}.log'
        record = dict(stage=stage, command=command, log=str(log), returncode=None)
        commands.append(record)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(f'[{stage}] {log}', flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record['returncode'] = result.returncode
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            print(log.read_text()[-5000:], file=sys.stderr)
            raise RuntimeError(f'{stage} failed; see {log}')

    def check(stage, dataset):
        run(stage, 'check_wilor_training_dataset.py', [dataset,
            '--mano-source', config['mano_source'], '--mano-model-dir', config['mano_model_dir'],
            '--device', args.device])

    datasets = []
    for item in recordings:
        dataset = output / 'recordings' / item['name']
        run(item['name'] + '_export', 'export_native_wilor_training_dataset.py', [
            '--fusion', item['fusion'], '--mano-fit', item['mano_fit'],
            '--dataset', item['dataset'], '--reference-camera', item.get('reference_camera', 'camera0'),
            '--output', dataset, '--cameras', *item.get('cameras', ['camera0', 'camera1']),
            '--view-filter', 'complete21', '--image-space', 'native_raw',
            '--sample-stride', args.sample_stride, '--max-samples', args.max_samples,
            *(['--require-both-hands'] if args.require_both_hands else [])])
        check(item['name'] + '_check', dataset)
        datasets.append(dataset)
    merged = output / 'merged'
    run('merge', 'merge_wilor_training_datasets.py', ['--inputs', *datasets, '--output', merged])
    check('merged_check', merged)
    run('preview', 'render_wilor_training_dataset.py', ['--dataset', merged,
        '--output', output / 'preview', '--samples', args.preview_samples])
    print(f'Complete: {merged}', flush=True)
    print('Original recording IDs and per-recording camera calibrations are preserved.')


if __name__ == '__main__':
    main()
