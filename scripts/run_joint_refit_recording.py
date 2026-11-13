#!/usr/bin/env python3
"""Per-recording joint refit: calibrate shape and attachments on this recording, then fit and export every frame.

Runs only when an optimization config enables joint_refit. Each recording gets its own calibration; nothing is reused
from another recording. A hand that was not recorded is skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT.parent / 'wilor_training/tools'
CODE = ROOT / 'scripts/joint_refit'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'alignment', 'aligned21', 'baseline', 'nokov', 'placement', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--sample-stride', type=int, default=8)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--resume', action='store_true', help='reuse stages whose output already exists')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4), help='CPU processes for video stages')
    a = ap.parse_args()
    if a.sample_stride < 1:
        ap.error('sample-stride must be positive')
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=a.resume)
    fit = a.baseline / 'mano_fit_full_cam01'
    present = [h for side, h in enumerate(['left', 'right']) if (fit / f'track_{side}.npz').is_file()]
    if not present:
        raise RuntimeError('baseline fit has no hand track')
    context = dict(dataset=str(a.dataset.resolve()), alignment=str(a.alignment.resolve()), aligned21=str(a.aligned21.resolve()),
                   baseline_fit=str(fit.resolve()), placement=str(a.placement.resolve()), holdout=[10 ** 9, 10 ** 9 + 1],
                   device=a.device)
    (a.output / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    env = dict(os.environ, PYTHONNOUSERSITE='1'); env.pop('PYTHONPATH', None)
    commands = []

    def run(name, script, *args, clean='output'):
        # A stage counts as finished only after it exits 0 (logs/<name>.done). On --resume an unfinished stage's
        # partial output is removed first, because the stage tools refuse to write into an existing directory.
        command = [sys.executable, '-B', str(script), *map(str, args)]
        row = dict(stage=name, command=command, returncode=None); commands.append(row)
        marker = a.output / 'logs' / f'{name}.done'
        if a.resume and marker.is_file():
            row['returncode'] = 0; row['resumed'] = True; print(f'[{name}] finished earlier, skipped', flush=True); return
        print(f'[{name}] running, log: {a.output / "logs" / (name + ".log")}', flush=True); start = time.monotonic()
        if clean == 'output':
            clean = Path(command[command.index('--output') + 1]) if '--output' in command else None
        if clean is not None and Path(clean).exists():
            target = Path(clean).resolve()
            if a.output.resolve() not in target.parents: raise RuntimeError(f'refusing to remove {target} outside {a.output}')
            shutil.rmtree(target)
        (a.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (a.output / 'logs' / f'{name}.log').open('w') as log:
            row['returncode'] = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (a.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if row['returncode']:
            raise RuntimeError(f'{name} failed; see {a.output / "logs" / (name + ".log")}')
        marker.write_text('ok\n'); print(f'[{name}] done in {time.monotonic() - start:.0f} s', flush=True)

    (a.output / 'logs').mkdir(exist_ok=a.resume)
    bridge = a.output / 'baseline_bridge'; bridge.mkdir(exist_ok=a.resume)
    names = ['WristM', 'WristIn', 'WristOut', 'HandOffset'] + [f'Finger{f}{j}' for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] for j in range(1, 5)]
    for side, hand in enumerate(['left', 'right']):
        src = fit / f'track_{side}.npz'
        if not src.is_file(): continue
        z = np.load(src)
        np.savez_compressed(bridge / f'{hand}_pilot.npz', **{k: z[k] for k in z.files}, target24=z['observation_points'], marker_names=np.asarray(names), calibration_mask=np.zeros(len(z['pair_indices']), bool))
    run('skeleton', TOOLS / 'nokov_skeleton_audit/analyze.py', '--context', a.output / 'context.json', '--output', a.output / 'skeleton')
    run('prepare', TOOLS / 'keyframe_calibration/prepare_source.py', '--context', a.output / 'context.json', '--output', a.output / 'source', '--sample-stride', a.sample_stride)
    # Contours only for the calibration sample (the frames keyframe selection can use), in parallel.
    run('contours', CODE / 'contours_parallel.py', '--context', a.output / 'context.json', '--source', bridge, '--sample', a.output / 'source', '--output', a.output / 'contours', '--workers', a.workers)
    run('select', TOOLS / 'keyframe_calibration/select_keyframes.py', '--context', a.output / 'context.json', '--audit', a.output / 'contours', '--source', a.output / 'source', '--output', a.output / 'selection')
    run('calibrate', TOOLS / 'keyframe_calibration/calibrate.py', '--context', a.output / 'context.json', '--source', a.output / 'source', '--selection', a.output / 'selection/selection.json', '--output', a.output / 'calibration')
    run('shape', TOOLS / 'full_silhouette/fit_shape.py', '--context', a.output / 'context.json', '--selection', a.output / 'selection/selection.json', '--source', a.output / 'calibration', '--output', a.output / 'shape')
    run('attachments', CODE / 'symmetrize_attachments.py', '--source', a.output / 'shape', '--output', a.output / 'attachments')
    run('align', CODE / 'image_alignment.py', '--context', a.output / 'context.json', '--selection', a.output / 'selection/selection.json', '--source', a.output / 'attachments', '--skeleton', a.output / 'skeleton/aligned_skeleton.npz', '--output', a.output / 'aligned', '--workers', a.workers)
    run('fit', CODE / 'fit.py', '--context', a.output / 'context.json', '--selection', a.output / 'selection/selection.json', '--source', a.output / 'aligned', '--skeleton', a.output / 'skeleton/aligned_skeleton.npz', '--hands', *present, '--output', a.output / 'fit',
        # same settings as the validated joint_refit_v4/v5 run (scripts/run_joint_refit_labels.sh)
        '--anat-weight', .01, '--normal-min', -.002, '--final-silhouette', '--root-offset', 'free', '--wrist-marker-weight', 0)
    run('dense_source', CODE / 'build_dense_source.py', '--context', a.output / 'context.json', '--calibrated', a.output / 'fit/final', '--skeleton', a.output / 'skeleton/aligned_skeleton.npz', '--output', a.output / 'dense/source')
    run('dense_align', CODE / 'image_alignment.py', '--context', a.output / 'context.json', '--selection', a.output / 'selection/selection.json', '--source', a.output / 'dense/source', '--skeleton', a.output / 'skeleton/aligned_skeleton.npz', '--output', a.output / 'dense/aligned', '--workers', a.workers)
    run('dense_fit', CODE / 'dense_drive.py', '--selection', a.output / 'selection/selection.json', '--source', a.output / 'dense/aligned', '--output', a.output / 'dense/fit')
    run('export', CODE / 'export.py', '--context', a.output / 'context.json', '--source', a.output / 'dense/fit/final', clean=a.output / 'dense/fit/final/export')
    run('verify', CODE / 'verify_export.py', '--result', a.output / 'dense/fit/final', clean=None)
    print(json.dumps(dict(output=str(a.output), hands=present, labels=str(a.output / 'dense/fit/final/export/merged'))), flush=True)


if __name__ == '__main__':
    main()
