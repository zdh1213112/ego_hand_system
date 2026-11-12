#!/usr/bin/env python3
"""Audit shared WiLoR MANO parameters, calibrated projections and final video."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from compare_multiview_wilor_undistortion import _fusion_directory, _json, _rows, _stats
from ego_data.calibration import CameraCalibration
from fit_shared_wilor_multiview import load_mano, physical_faces, physical_geometry
from gem_shared_geometry import project_base
from render_gem_multiview import MESH_STYLE


def replay_shared_models(data, layer, *, batch_size=64, tolerance_m=2e-5):
    """Decode every saved model independently, checking joints AND vertices."""
    if data['format'] != 'shared_wilor_mano_v1':
        raise ValueError('expected shared WiLoR MANO data')
    models = [hand for frame in data['frames'] for hand in frame['hands']]
    maxima = {'joints': 0., 'vertices': 0.}
    checked = 0
    for side in (0, 1):
        np.testing.assert_array_equal(data['topology']['right' if side else 'left'], physical_faces(layer, side))
        group = [hand for hand in models if int(hand['side']) == side]
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            parameters = {name: torch.stack([torch.as_tensor(h['parameters'][name], dtype=torch.float32)
                                            for h in batch]) for name in ('global_orient', 'hand_pose', 'betas')}
            with torch.no_grad():
                joints, vertices = physical_geometry(layer, parameters, side)
            for index, hand in enumerate(batch):
                rotation = torch.as_tensor(hand['rotation_base_from_model'], dtype=torch.float32)
                translation = torch.as_tensor(hand['translation_base_m'], dtype=torch.float32)
                scale = float(hand['model_scale'])
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2e-5, rtol=0)
                if not .9999 < float(torch.linalg.det(rotation)) < 1.0001:
                    raise ValueError('shared global rotation contains reflection or nonrigid scaling')
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError('invalid shared model scale')
                for name, local, key in (('joints', joints[index], 'joints_base_m'),
                                         ('vertices', vertices[index], 'vertices_base_m')):
                    reconstructed = local * scale @ rotation.T + translation
                    stored = torch.as_tensor(hand[key], dtype=torch.float32)
                    difference = float((reconstructed - stored).abs().max())
                    if not np.isfinite(difference) or difference >= tolerance_m:
                        raise ValueError(f"frame {hand['sync_index']} side {side}: {name} replay error {difference} m")
                    maxima[name] = max(maxima[name], difference)
                checked += 1
    if checked != len(models):
        raise ValueError('unexpected physical hand side')
    return {'regenerated_model_count': checked, 'tolerance_m': tolerance_m,
            'max_regenerated_joint_difference_m': maxima['joints'],
            'max_regenerated_vertex_difference_m': maxima['vertices'], 'topology_matches_mano': True}


def audit_projection_rows(data, accepted, calibrations, audit_rows):
    frames = {int(row['sync_index']): row for row in data['frames']}
    if len(frames) != len(data['frames']):
        raise ValueError('duplicate shared frame index')
    seen, comparisons, maximum = set(), 0, 0.
    for audit in audit_rows:
        sync = int(audit['sync_index'])
        if sync in seen or sync not in frames:
            raise ValueError(f'duplicate or unexpected projection frame {sync}')
        seen.add(sync)
        models = {int(h['side']): h for h in frames[sync]['hands']}
        if len(models) != len(frames[sync]['hands']):
            raise ValueError('duplicate shared hand side')
        accepted_hands = {int(h['side']): h for h in accepted.get(sync, {}).get('hands', [])}
        if models.keys() != accepted_hands.keys():
            raise ValueError('saved model and accepted skeleton coverage differ')
        expected_ids = {f'{sync}:{side}' for side in models}
        if audit['cameras'].keys() != calibrations.keys():
            raise ValueError('projection audit cameras differ')
        for camera, cal in calibrations.items():
            hands = audit['cameras'][camera]
            if len(hands) != len(models) or {h['shared_model_id'] for h in hands} != expected_ids:
                raise ValueError('cameras do not reference the same shared models')
            for hand in hands:
                side = int(hand['side'])
                model = models[side]
                if hand['shared_model_id'] != f'{sync}:{side}':
                    raise ValueError('projection model ID and physical side differ')
                points = np.asarray(model['joints_base_m'])
                np.testing.assert_array_equal(points, accepted_hands[side]['joints_base_m'])
                if len(model['supporting_cameras']) < 2:
                    raise ValueError('shared model has fewer than two supporting views')
                if hand['supporting_cameras'] != model['supporting_cameras']:
                    raise ValueError('supporting camera audit mismatch')
                expected, valid, _ = project_base(cal, points)
                stored = np.array([p if p is not None else [np.nan, np.nan] for p in hand['joints_2d']])
                if stored.shape != (21, 2) or not np.array_equal(np.isfinite(stored).all(1), valid):
                    raise ValueError('invalid projected joint mask')
                if valid.any():
                    error = float(np.abs(expected[valid] - stored[valid]).max())
                    if error >= 1e-7:
                        raise ValueError(f'projection differs by {error} px')
                    maximum = max(maximum, error)
                width, height = cal.image_size
                visible = valid & (expected[:, 0] >= 0) & (expected[:, 0] < width) & (expected[:, 1] >= 0) & (expected[:, 1] < height)
                if hand['visible_joint_count'] != int(visible.sum()):
                    raise ValueError('visible joint count differs')
                comparisons += 1
    if seen != frames.keys() or not accepted.keys() <= frames.keys():
        raise ValueError('projection audit does not cover all shared frames')
    return {'audited_frames': len(seen), 'checked_camera_hand_joint_projections': comparisons,
            'max_projection_difference_px': maximum, 'shared_model_ids_match_all_cameras': True}


def audit_native_parameters(predictions, cameras, layer, *, batch_size=64):
    """Report FP32 native replay separately from strict shared-model checks.

    Accelerated WiLoR evaluates MANO buffers in FP16. Replaying saved parameter
    values with original FP32 buffers can therefore differ slightly without a
    convention error. Native replay differences are reported, not hidden by
    altering the shared geometry tolerance.
    """
    all_errors, per_camera, missing = [], {}, 0
    for camera in cameras:
        errors, pending = [], {0: [], 1: []}

        def flush(side):
            batch = pending[side]
            if not batch:
                return
            params = {name: torch.as_tensor(np.stack([np.asarray(h['mano'][name]).reshape(shape)
                                                    for h in batch]), dtype=torch.float32)
                      for name, shape in [('global_orient', (1, 3, 3)), ('hand_pose', (15, 3, 3)), ('betas', (10,))]}
            with torch.no_grad():
                joints, _ = physical_geometry(layer, params, side)
            saved = torch.as_tensor([h['joints_3d_camera_relative'] for h in batch], dtype=torch.float32)
            differences = (joints - saved).abs().flatten(1).max(1).values.numpy()
            if not np.isfinite(differences).all():
                raise ValueError('nonfinite native parameter replay')
            errors.extend(map(float, differences))
            pending[side] = []

        with (Path(predictions) / camera / 'predictions.jsonl').open() as stream:
            for line in stream:
                for hand in json.loads(line)['hands']:
                    if 'mano' not in hand or 'joints_3d_camera_relative' not in hand:
                        missing += 1
                        continue
                    side = int(hand['is_right'])
                    if side not in (0, 1):
                        raise ValueError('invalid native hand side')
                    pending[side].append(hand)
                    if len(pending[side]) == batch_size:
                        flush(side)
        flush(0)
        flush(1)
        per_camera[camera] = {**_stats(errors), 'max': max(errors, default=None)}
        all_errors.extend(errors)
    return {'hypothesis_count': len(all_errors), 'missing_native_geometry_hypotheses': missing,
            'max_abs_joint_error_m': max(all_errors, default=None),
            'per_hypothesis_max_abs_joint_error_m': _stats(all_errors), 'per_camera': per_camera,
            'interpretation': 'Native MANO replay uses FP32 assets; AMP inference may have FP16 buffer rounding. These are numerical replay differences, not ground-truth errors.'}


def decode_video(video, frames, metadata):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'cannot open video {video}')
    samples = {0, len(frames) // 2, len(frames) - 1}
    for condition in (lambda f: len(f['hands']) == 2, lambda f: bool(f['hands'])):
        match = next((i for i, row in enumerate(frames) if condition(row)), None)
        if match is not None:
            samples.add(match)
    decoded, previews = 0, []
    width, height = metadata['video_size']
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (height, width):
                raise ValueError('decoded video resolution differs from metadata')
            if decoded in samples:
                path = video.parent / f'preview_{decoded:04d}.jpg'
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError(f'failed to write preview {path}')
                previews.append(str(path))
            decoded += 1
    finally:
        capture.release()
    if decoded != len(frames):
        raise ValueError(f'decoded {decoded}, expected {len(frames)} frames')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size',
        '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    if int(stream['nb_frames']) != decoded or abs(numerator / denominator - metadata['fps']) > .001:
        raise ValueError('container frame count/fps differs from metadata')
    return {'decoded_frames': decoded, 'video': str(video), 'probe': probe, 'previews': previews}


def validate_run(root, *, batch_size=64):
    root = Path(root).resolve()
    config = _json(root / 'run_config.json')
    if config.get('image_preprocessing'):
        raise ValueError('expected raw-image run')
    dataset = root / 'normalized_multiview'
    manifest = _json(dataset / 'manifest.json')
    if manifest.get('undistortion'):
        raise ValueError('expected unmodified source frames')
    # Shared fitting/rendering is independent of the legacy fusion stage.
    shared = _fusion_directory(root, config, require_complete=False)
    shared = shared.with_name(shared.name + '_shared_wilor')
    summary = _json(shared / 'summary.json')
    cameras = summary['camera_ids']
    if cameras != config['cameras']:
        raise ValueError('shared cameras differ from run configuration')
    video = shared / f'final_shared_{len(cameras)}view.mp4'
    metadata = _json(video.with_suffix('.json'))
    if (not metadata['skeleton_and_mesh_share_model'] or metadata['independent_view_fallback']
            or not metadata['raw_images_without_undistortion']
            or metadata['config']['mesh_style'] != MESH_STYLE):
        raise ValueError('shared renderer mode/style differs from expected output')
    if metadata['config']['shared_mtime_ns'] != (shared / 'shared_pose.pt').stat().st_mtime_ns:
        raise ValueError('video was rendered from a different shared model file')
    data = torch.load(shared / 'shared_pose.pt', weights_only=False, map_location='cpu', mmap=True)
    with (dataset / 'multiview_frames.csv').open() as stream:
        sync = list(csv.DictReader(stream))[:config.get('max_frames') or None]
    if [int(r['sync_index']) for r in sync] != [int(r['sync_index']) for r in data['frames']]:
        raise ValueError('shared models do not follow synchronized frame order')
    if len(sync) != summary['frame_count'] or len(sync) != metadata['frame_count']:
        raise ValueError('shared/video frame count mismatch')
    accepted = _rows(shared / 'accepted.jsonl')
    accepted_count = sum(bool(frame['hands']) for frame in data['frames'])
    hand_count = sum(len(frame['hands']) for frame in data['frames'])
    hand_histogram = Counter(len(frame['hands']) for frame in data['frames'])
    side_counts = Counter(int(hand['side']) for frame in data['frames'] for hand in frame['hands'])
    if accepted_count != summary['accepted_frame_count'] or hand_count != summary['accepted_hand_count']:
        raise ValueError('accepted counts differ from actual models')
    calibrations = {c: CameraCalibration.load(dataset / 'calibration' / f'{c}.json') for c in cameras}
    with video.with_suffix('.projections.jsonl').open() as stream:
        projections = audit_projection_rows(data, accepted, calibrations, map(json.loads, stream))
    layer = load_mano(Path(_json(shared / 'run_config.json')['mano_model_dir']))
    replay = replay_shared_models(data, layer, batch_size=batch_size)
    native = audit_native_parameters(root / 'wilor_multiview', cameras, layer, batch_size=batch_size)
    video_result = decode_video(video, data['frames'], metadata)
    report = {'accepted_frame_count': accepted_count, 'accepted_hand_count': hand_count,
              'accepted_hand_count_per_frame_histogram': {str(n): hand_histogram[n] for n in (0, 1, 2)},
              'accepted_left_hand_frames': side_counts[0], 'accepted_right_hand_frames': side_counts[1],
              'no_shared_pose_frames': len(sync) - accepted_count,
              'skeleton_and_mesh_share_model': True, 'independent_view_fallback': False,
              'raw_images_without_undistortion': True, 'mesh_style': MESH_STYLE,
              **replay, **projections, **video_result, 'native_parameter_replay': native,
              'interpretation': 'This validates parameter/geometry/projection consistency and video completeness, not true 3D pose accuracy.'}
    return shared, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='default: shared-result directory/validation.json')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    torch.set_num_threads(2)
    shared, report = validate_run(args.run, batch_size=args.batch_size)
    output = args.output or shared / 'validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
