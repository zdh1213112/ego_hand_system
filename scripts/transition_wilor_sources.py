#!/usr/bin/env python3
"""Apply native WiLoR source transitions to a completed, unblended shared output."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from ego_data.calibration import CameraCalibration
import fit_shared_wilor_multiview as fitter
from wilor_source_transition import apply_source_transitions, transition_policy


def frozen_input(entry):
    path = Path(entry['path'])
    if fitter.signature(path) != entry:
        raise ValueError(f'frozen fit input changed: {path}')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_recovery_input(config, summary, frames):
    """Accept only recovered wrists from a supported, bounded detector-pair fit."""
    message = 'incompatible saved recovery evidence; refit with fit_shared_wilor_multiview.py'
    recovery = config.get('native_recovery', {})
    if not isinstance(recovery, dict):
        raise ValueError(message)
    enabled = recovery.get('enabled', False)
    version = recovery.get('version')
    if type(enabled) is not bool:
        raise ValueError(message)
    if enabled and (version not in ('detector_pair_native_at_triangulated_wrist_v2', fitter.RECOVERY_VERSION)
                    or recovery.get('max_wrist_refinement_m') != fitter.MAX_WRIST_REFINEMENT_M
                    or recovery.get('wrist_priority') != 'triangulated_pair_before_bounded_refinement'
                    or recovery.get('selection') != 'max_supported_views_then_mean_max_of_wrist_and_median_px'
                    or recovery.get('require_source_support') is not True
                    or recovery.get('min_same_side_detection_views') != 2):
        raise ValueError(f'{message} (recovery version {version!r})')
    count = 0
    for frame in frames:
        for hand in frame['hands']:
            association = hand.get('association', {})
            if 'recovery' not in hand:
                if isinstance(association, dict) and str(association.get('version', '')).startswith('detector_pair_native_'):
                    raise ValueError(message)
                continue
            count += 1
            if (not enabled or not isinstance(hand['recovery'], dict) or not isinstance(association, dict)
                    or association.get('version') != version):
                raise ValueError(message)
            pair = association.get('reference_cameras')
            support = hand.get('supporting_cameras')
            if (not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(c, str) for c in pair)
                    or not isinstance(support, list) or not all(isinstance(c, str) for c in support)
                    or pair[0] == pair[1] or not set(pair) <= set(support)
                    or hand.get('pose_source_camera') not in pair):
                raise ValueError(message)
            try:
                anchor = np.asarray(association.get('wrist_base_m'))
                joints = np.asarray(hand.get('joints_base_m'))
            except (TypeError, ValueError) as error:
                raise ValueError(message) from error
            reported = association.get('wrist_refinement_m')
            if (anchor.shape != (3,) or joints.shape != (21, 3) or anchor.dtype.kind not in 'fiu'
                    or joints.dtype.kind not in 'fiu' or not np.isfinite(anchor).all() or not np.isfinite(joints[0]).all()
                    or type(reported) not in (int, float) or not np.isfinite(reported)):
                raise ValueError(message)
            measured = np.linalg.norm(joints[0] - anchor)
            # Saved joints are FP32. The independent triangulated wrist and
            # reported displacement are FP64 and need a rounding allowance.
            if not 0 <= reported <= fitter.MAX_WRIST_REFINEMENT_M + 1e-9 or abs(measured - reported) > 2e-6:
                raise ValueError(f'{message} (wrist at sync {frame["sync_index"]})')
    if summary.get('recovered_hand_count', 0) != count:
        raise ValueError(message)


def postprocess(shared, output, device):
    if output.exists():
        raise FileExistsError(f'use a new output directory: {output}')
    config = json.loads((shared / 'run_config.json').read_text())
    summary = json.loads((shared / 'summary.json').read_text())
    if config.get('pose_policy') != 'native-wrist' or config.get('native_pose_transition', {}).get('enabled'):
        raise ValueError('expected an unblended native-wrist run')
    pose_path = shared / 'shared_pose.pt'
    data = torch.load(pose_path, map_location='cpu', weights_only=False)
    frames = data['frames']
    if data['format'] != 'shared_wilor_mano_v1' or any('pose_transition' in h for f in frames for h in f['hands']):
        raise ValueError('expected unblended shared MANO geometry')
    validate_recovery_input(config, summary, frames)

    # The fitter records three input signatures per camera, in camera order:
    # calibration, associated predictions, native predictions.
    cameras = config['cameras']
    camera_inputs = config['inputs'][-3 * len(cameras):]
    calibrations = {cam: CameraCalibration.load(frozen_input(camera_inputs[3*i])) for i, cam in enumerate(cameras)}
    model_file = frozen_input(config['inputs'][2])
    wrapper = frozen_input(config['inputs'][3])
    if sha256(wrapper) != sha256(fitter.mano_wrapper_path()):
        raise ValueError(f'set EGO_WILOR_ROOT to the training checkout containing {wrapper}')
    layer = fitter.load_mano(model_file.parent, device=device)
    recovery = config.get('native_recovery', {})
    recovery_input_version = recovery.get('version')
    chronology = config.get('native_source_chronology')
    if chronology is not None and chronology != fitter.NATIVE_CHRONOLOGY_VERSION:
        raise ValueError(f'unsupported source chronology: {chronology}')
    if recovery.get('enabled') and chronology is None:
        by_sync = {f['sync_index']: f for f in frames}
        native_rows = {}
        if any('recovery' in h for f in frames for h in f['hands']):
            native_rows = {cam: fitter.read_prediction_records(frozen_input(camera_inputs[3*i+2]))
                           for i, cam in enumerate(cameras)}
        summary['native_source_chronology'] = fitter.apply_native_source_chronology(
            by_sync, native_rows, layer, calibrations, threshold_px=config['threshold_px'],
            switch_margin_px=config['native_pose_selection']['pose_switch_margin_px'], batch_size=config['batch_size'])
        config['native_source_chronology'] = fitter.NATIVE_CHRONOLOGY_VERSION
        recovery['version'] = fitter.RECOVERY_VERSION
    summary['native_pose_transition'] = apply_source_transitions(
        frames, layer, calibrations, config['threshold_px'], fitter.fit_quality, fitter.physical_geometry)
    config['native_pose_transition'] = dict(transition_policy(), enabled=True)
    config['transition_replay'] = {
        'shared_input': str(shared.resolve()), 'device': device, 'recovery_input_version': recovery_input_version,
        'input_sha256': {name: sha256(shared / name) for name in (
            'shared_pose.pt', 'accepted.jsonl', 'rejected.jsonl', 'run_config.json', 'summary.json')},
        'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in (
            'transition_wilor_sources.py', 'fit_shared_wilor_multiview.py', 'wilor_source_transition.py',
            'wilor_native_recovery.py', 'fusion_temporal_guard.py', 'gem_shared_geometry.py')},
    }
    hands = [hand for frame in frames for hand in frame['hands']]
    summary['supporting_view_counts'] = dict(Counter(len(h['supporting_cameras']) for h in hands))
    errors = [v['median_px'] for h in hands for v in h['views'].values() if v['supported']]
    summary['supporting_view_median_error_px'] = float(np.median(errors)) if errors else None
    summary['pose_source_counts'] = dict(Counter(h['pose_source'] for h in hands))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.wilor-transition-') as temporary:
        staged = Path(temporary) / 'shared'
        staged.mkdir()
        torch.save(data, staged / 'shared_pose.pt')
        with (staged / 'accepted.jsonl').open('w') as stream:
            for frame in frames:
                if frame['hands']:
                    row = dict(sync_index=frame['sync_index'], hands=[fitter.accepted_hand(h) for h in frame['hands']])
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
        shutil.copyfile(shared / 'rejected.jsonl', staged / 'rejected.jsonl')
        for name, document in [('run_config', config), ('summary', summary)]:
            (staged / (name + '.json')).write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
        staged.rename(output)
    return summary['native_pose_transition']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path, required=True, help='untransitioned shared output directory')
    parser.add_argument('--output', type=Path, required=True, help='new shared output directory')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(postprocess(args.shared, args.output, args.device), indent=2))


if __name__ == '__main__':
    main()
