#!/usr/bin/env python3
"""Fit MANO and export native labels from a verified, aligned EGO/NOKOV take.

Example:
  python scripts/run_nokov_native_labels.py --processed-session /path/to/take \
    --hand-side left --output /path/to/new_labels

A processed session comes from run_ego_nokov_pipeline.py. This command does
not make the upstream raw-data alignment pipeline support right-hand assets.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from label_generation_config import load_label_generation_config, require_available_mode

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-session', type=Path, required=True)
    p.add_argument('--hand-side', choices=['left', 'right', 'both', 'auto'], default='both', help="auto: fit every hand observed in --aligned-nokov (left, right or both)")
    p.add_argument('--aligned-nokov', type=Path, help='aligned dual-hand NPZ: joints_base_m [2,F,21,3], valid [2,F,21]')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--sample-stride', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cameras', nargs='+', default=['camera0','camera1'], help='physical cameras for projection/export; this does not enable RGB MANO optimization')
    p.add_argument('--label-config', type=Path, help='JSON label-generation config; multiview MANO optimization defaults to false')
    a = p.parse_args()
    label_config = load_label_generation_config(a.label_config)
    require_available_mode(label_config)
    multiview = label_config["multiview_mano_optimization"]
    if a.hand_side == 'auto':
        if not a.aligned_nokov or not a.aligned_nokov.is_file():
            p.error('--hand-side auto requires --aligned-nokov')
        with np.load(a.aligned_nokov, allow_pickle=False) as data:
            seen = [bool((data['valid'][side].sum(-1) >= 12).any()) for side in (0, 1)]
        a.hand_side = 'both' if all(seen) else 'left' if seen[0] else 'right' if seen[1] else None
        if a.hand_side is None:
            p.error('no hand with >=12 observed points in --aligned-nokov')
        a.single_from_aligned = a.hand_side != 'both'
    else:
        a.single_from_aligned = False
    if multiview["enabled"] and (a.hand_side != "both" or a.cameras != ["camera0","camera1","camera2","camera3"]):
        p.error("Multiview MANO requires both hands and all four cameras in order")
    if a.sample_stride < 1 or a.max_samples < 0:
        p.error('stride must be positive and max-samples nonnegative')
    if len(a.cameras)<2 or len(set(a.cameras))!=len(a.cameras) or a.cameras[0]!='camera0' or not set(a.cameras).issubset({'camera0','camera1','camera2','camera3'}):
        p.error('choose at least two distinct physical cameras, starting with camera0')
    session = a.processed_session.expanduser().resolve()
    out = a.output.expanduser().resolve()
    dataset = session / 'ego/normalized_multiview'
    projection = session / 'alignment/cam0_marker21_pose/nokov_cam0_projection.npz'
    aligned = a.aligned_nokov.expanduser().resolve() if a.aligned_nokov else session / 'alignment/aligned_nokov21.npz'
    if a.hand_side == 'both':
        if not aligned.is_file():
            raise FileNotFoundError('Dual-hand mode requires --aligned-nokov with synchronized LEFT and RIGHT 3D observations. The legacy raw alignment entry is left-only.')
        with np.load(aligned, allow_pickle=False) as data:
            if 'alignment_status' in data and str(data['alignment_status']) != 'accepted':
                raise ValueError('Refusing unaccepted/candidate spatial alignment')
            valid = data['valid']
            if valid.ndim != 3 or valid.shape[0] != 2 or valid.shape[2] != 21 or not all((valid.sum(-1)[side] >= 12).any() for side in (0, 1)):
                raise ValueError('Both physical hand tracks must contain real observed 3D points')
    if not (dataset/'multiview_frames.csv').is_file() or (a.hand_side != 'both' and not a.single_from_aligned and not projection.is_file()):
        raise FileNotFoundError('Expected normalized_multiview and verified cam0 marker21 projection under processed-session')
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    commands = []
    def run(name, script, arguments):
        cmd = [sys.executable, '-B', str(script), *map(str, arguments)]
        entry = dict(stage=name, command=cmd, returncode=None)
        commands.append(entry)
        if a.dry_run:
            print(json.dumps(entry))
            return
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        print(f'[{name}] {out / "logs" / (name + ".log")}', flush=True)
        with (out/'logs'/(name+'.log')).open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry['returncode'] = result.returncode
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed: see {out / "logs" / (name + ".log")}')
    if not a.dry_run:
        (out/'logs').mkdir(parents=True)
        (out/'label_generation_config.json').write_text(json.dumps(label_config, indent=2)+'\n')
    group = session.name + '__' + a.hand_side
    pipeline = ROOT.parent / 'nokov_ego_hand_validation/integration/pipeline'
    if a.hand_side != 'both' and not a.single_from_aligned:
        run('align', pipeline/'make_aligned_nokov21.py', ['--projection',projection,'--dataset',dataset,
            '--output',out/'aligned_nokov21.npz','--hand-side',a.hand_side])
        aligned = out/'aligned_nokov21.npz'
    run('prepare', ROOT/'scripts/prepare_native_nokov_mano_input.py', [
        '--aligned-nokov',aligned,'--dataset',dataset,
        '--output',out/'mano_input_cam01.npz',
        '--fusion-output',out/'nokov_observations_cam01','--source-group',group,
        '--reference-camera','camera0','--cameras',*a.cameras])
    run('fit', ROOT/'scripts/fit_mano_sequence.py', [
        '--input',out/'mano_input_cam01.npz','--mano-source',ROOT/'third_party/MANO',
        '--model-dir',ROOT/'models/mano','--output',out/'mano_fit_full_cam01',
        '--mano-convention','wilor_right_canonical_v1',
        *(['--track-id',0,'--track-id',1] if a.hand_side=='both' else ['--track-id',0 if a.hand_side=='left' else 1]),
        '--shape-frames',256,'--shape-iterations',0 if multiview['enabled'] else 800,
        '--pose-iterations',multiview['pose_iterations'] if multiview['enabled'] else 500,
        *(['--initial-output',multiview['initial_fit'],'--multiview-marker-bundle',multiview['bundle'],
           '--multiview-weight',multiview['rgb_weight'],'--multiview-sigma-px',multiview['sigma_px']] if multiview['enabled'] else []),
        '--pose-window',96,'--pose-overlap',16,'--pca-components',45,'--learning-rate',.008,
        '--w-3d',1,'--w-2d',0,'--min-fit-observed-points',12,'--max-unobserved-gap',3,
        '--w-pose',.00005,'--w-shape',.0001,'--w-temporal',.002,'--w-rigid-temporal',.002,
        '--w-acceleration',.001,'--boundary-weight',.02,'--device',a.device,'--no-video'])
    config = dict(mano_source=str(ROOT/'third_party/MANO'),mano_model_dir=str(ROOT/'models/mano'),
        recordings=[dict(name=group,fusion=str(out/'nokov_observations_cam01'),
        mano_fit=str(out/'mano_fit_full_cam01'),dataset=str(dataset),
        reference_camera='camera0',cameras=a.cameras)])
    if not a.dry_run:
        (out/'native_export_config.json').write_text(json.dumps(config, indent=2)+'\n')
    run('export_check_preview', ROOT/'scripts/run_native_wilor_export.py', [
        '--config',out/'native_export_config.json','--output',out/'export',
        '--sample-stride',a.sample_stride,'--max-samples',a.max_samples,'--device',a.device,
        *(['--require-both-hands'] if a.hand_side=='both' else [])])
    if not a.dry_run:
        print(f'Labels: {out / "export/merged"}')

if __name__ == '__main__':
    main()
