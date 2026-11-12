#!/usr/bin/env python3
"""Compare legacy WiLoR and shared MANO on the same synchronized observations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from compare_multiview_wilor_undistortion import _fusion_directory, _inside, _json, _rows, _stats
from ego_data.calibration import CameraCalibration
from ego_data.dataset import read_csv
from gem_shared_geometry import project_base


def measure_geometry(rows, calibrations):
    """Measure saved 3D against selected image observations, not video fallbacks."""
    residuals, finite_frames = {}, set()
    hand_count = complete_hands = joint_count = 0
    for sync, row in rows.items():
        seen = set()
        for hand in row.get('hands', []):
            side = int(hand['side'])
            if side not in (0, 1) or side in seen:
                raise ValueError(f'duplicate/invalid hand side at frame {sync}')
            seen.add(side)
            points = np.asarray(hand['joints_base_m'], dtype=float)
            if points.shape != (21, 3):
                raise ValueError('expected 21 joints_base_m')
            finite = np.isfinite(points).all(1)
            hand_count += int(finite.any())
            complete_hands += int(finite.all())
            joint_count += int(finite.sum())
            if finite.any():
                finite_frames.add(sync)
            for camera, view in hand.get('views', {}).items():
                cal = calibrations[camera]
                observed = np.asarray(view.get('observed_joints_2d', view.get('joints_2d', [])), dtype=float)
                if observed.shape != (21, 2):
                    raise ValueError('expected selected 21 image observations')
                projected, valid, _ = project_base(cal, points)
                valid &= finite & _inside(observed, cal.image_size) & np.isfinite(projected).all(1)
                errors = np.linalg.norm(projected - observed, axis=1)
                for joint in np.flatnonzero(valid):
                    residuals[(sync, side, camera, int(joint))] = {
                        'error_px': float(errors[joint]),
                        'detection_index': view.get('detection_index'),
                        'observed': observed[joint],
                    }
    return {'accepted_rows': len(rows), 'frames_with_finite_3d': len(finite_frames),
            'finite_hands': hand_count, 'complete_21_joint_hands': complete_hands,
            'finite_3d_joints': joint_count,
            'selected_view_residual_px': _stats(r['error_px'] for r in residuals.values())}, residuals


def hand_frame_histograms(rows, sync_indices):
    """Separate stored hand entries from usable/complete geometry per frame."""
    histograms = {name: {str(n): 0 for n in (0, 1, 2)} for name in
                  ('stored_hand_entries', 'hands_with_any_finite_3d_joint', 'complete_21_joint_hands')}
    for sync in sync_indices:
        hands = rows.get(int(sync), {}).get('hands', [])
        masks = [np.isfinite(np.asarray(h['joints_base_m'], dtype=float)).all(1) for h in hands]
        counts = (len(hands), sum(bool(mask.any()) for mask in masks), sum(bool(mask.all()) for mask in masks))
        for histogram, count in zip(histograms.values(), counts):
            histogram[str(count)] += 1
    return histograms


def paired_residuals(before, after):
    """Pair only the same frame/side/camera/joint AND detector observation."""
    samples, excluded_detection, excluded_observation = [], 0, 0
    for key in sorted(before.keys() & after.keys()):
        old, new = before[key], after[key]
        detection = old['detection_index']
        if detection is None or new['detection_index'] is None or int(detection) != int(new['detection_index']):
            excluded_detection += 1
            continue
        # Temporal side repair can pair different handedness hypotheses from
        # the same box. Requiring equal pixels keeps the residual target fixed.
        if not np.allclose(old['observed'], new['observed'], rtol=0, atol=1e-5):
            excluded_observation += 1
            continue
        samples.append({'sync_index': key[0], 'side': key[1], 'camera': key[2], 'joint': key[3],
                        'detection_index': int(detection), 'legacy_residual_px': old['error_px'],
                        'shared_residual_px': new['error_px'], 'delta_px': new['error_px'] - old['error_px']})
    return {'legacy': _stats(s['legacy_residual_px'] for s in samples),
            'shared': _stats(s['shared_residual_px'] for s in samples),
            'delta_shared_minus_legacy': _stats(s['delta_px'] for s in samples),
            'excluded_different_or_missing_detection': excluded_detection,
            'excluded_different_observation': excluded_observation}, samples


def compare_run(root, *, legacy_video=None):
    root = Path(root).resolve()
    config = _json(root / 'run_config.json')
    if config.get('image_preprocessing'):
        raise ValueError('this comparison expects the original raw-image run')
    dataset = root / 'normalized_multiview'
    manifest = _json(dataset / 'manifest.json')
    if manifest.get('undistortion'):
        raise ValueError('expected original calibration and images')
    sync_rows = read_csv(dataset / 'multiview_frames.csv')[:config.get('max_frames') or None]
    expected = {int(row['sync_index']): row for row in sync_rows}
    if not expected or len(expected) != len(sync_rows):
        raise ValueError('empty or duplicate synchronization rows')
    cameras = config['cameras']
    calibrations = {c: CameraCalibration.load(dataset / 'calibration' / f'{c}.json') for c in cameras}
    native = root / 'wilor_multiview'
    inference = _json(native / 'summary.json')
    if inference['frame_count'] != len(expected):
        raise ValueError('inference frame count differs from synchronization')
    detections = {}
    for camera in cameras:
        rows = _rows(native / camera / 'predictions.jsonl')
        if rows.keys() != expected.keys():
            raise ValueError(f'{camera}: incomplete synchronized inference')
        counts = []
        for sync, row in rows.items():
            if (int(row['source_frame_index']) != int(expected[sync][f'{camera}_frame_index'])
                    or int(row['timestamp_ns']) != int(expected[sync][f'{camera}_timestamp_ns'])):
                raise ValueError(f'{camera}: frame/timestamp mismatch')
            counts.append(len({int(h['detection_index']) for h in row.get('hands', [])}))
        detections[camera] = {'physical_box_count': sum(counts),
                              'frames_with_detections': sum(c > 0 for c in counts),
                              'frame_detection_rate': sum(c > 0 for c in counts) / len(counts)}
    legacy = _fusion_directory(root, config)
    shared = legacy.with_name(legacy.name + '_shared_wilor')
    summary = _json(shared / 'summary.json')
    if summary['frame_count'] != len(expected):
        raise ValueError('shared fit frame count differs')
    before_rows, after_rows = _rows(legacy / 'accepted.jsonl'), _rows(shared / 'accepted.jsonl')
    if not before_rows.keys() <= expected.keys() or not after_rows.keys() <= expected.keys():
        raise ValueError('unexpected fusion sync indices')
    before, before_errors = measure_geometry(before_rows, calibrations)
    after, after_errors = measure_geometry(after_rows, calibrations)
    before['hand_count_per_frame_histograms'] = hand_frame_histograms(before_rows, expected)
    after['hand_count_per_frame_histograms'] = hand_frame_histograms(after_rows, expected)
    for metrics in (before, after):
        metrics['finite_3d_frame_rate'] = metrics['frames_with_finite_3d'] / len(expected)
    after['frames_without_shared_model'] = len(expected) - after['frames_with_finite_3d']
    after['skeleton_and_mesh_share_model'] = summary['skeleton_and_mesh_share_model']
    after['independent_view_fallback'] = summary['independent_view_fallback']
    paired, samples = paired_residuals(before_errors, after_errors)
    old_video = Path(legacy_video).resolve() if legacy_video else legacy / f'final_only_{len(cameras)}view.mp4'
    new_video = shared / f'final_shared_{len(cameras)}view.mp4'
    old_render, new_render = _json(old_video.with_suffix('.json')), _json(new_video.with_suffix('.json'))
    for meta in (old_render, new_render):
        if meta['frame_count'] != len(expected):
            raise ValueError('final video metadata frame count differs')
    before['display'] = {key: old_render.get(key) for key in
                         ('fused_camera_tiles', 'fused_camera_hand_tiles', 'native_2d_camera_hand_tiles')}
    mesh = old_render.get('mesh', {})
    matching_mesh = mesh.get('model') == 'WiLoR shared MANO'
    native_mesh = mesh.get('model') == 'WiLoR native MANO'
    if matching_mesh:
        geometry = shared / 'shared_pose.pt'
        if (Path(mesh['shared']).resolve() != shared.resolve()
                or mesh['shared_mtime_ns'] != geometry.stat().st_mtime_ns
                or mesh['shared_size'] != geometry.stat().st_size):
            raise ValueError('legacy overlay must use the same shared MANO file')
        if mesh['drawn'] != new_render['mesh_draws']:
            raise ValueError('legacy/shared mesh visibility counts differ')
        if any(mesh.get(key) != value for key, value in new_render['config']['mesh_style'].items()):
            raise ValueError('legacy/shared mesh styles differ')
    before['display']['shared_mesh'] = matching_mesh
    before['display']['native_mesh'] = native_mesh
    before['display']['skeleton_and_mesh_share_model'] = False
    after['display'] = {key: new_render[key] for key in
                       ('projected_hand_tiles', 'mesh_draws', 'independent_view_fallback')}
    report = {'schema_version': 1, 'run': str(root), 'detector': config['detector'],
              'synchronized_frames': len(expected), 'camera_ids': cameras, 'detections': detections,
              'legacy': before, 'shared_mano': after, 'paired_selected_observation_residual_px': paired,
              'videos': {'legacy': str(old_video), 'shared': str(new_video)},
              'interpretation': [
                  'Both branches reuse the same synchronized images and WiLoR detector/neural predictions.',
                  ('Legacy skeleton is unchanged; its added mesh is the same shared MANO mesh as the other video.'
                   if matching_mesh else 'Legacy skeleton with original per-camera WiLoR MANO meshes; only mesh colors/transparency match the shared video.'
                   if native_mesh else 'Legacy video is the original hybrid skeleton display; it has no shared MANO mesh.'),
                  '3D residuals compare saved geometry with selected model-generated image joints, not ground truth or native-2D video fallbacks.',
                  'Paired samples require identical frame, side, camera, joint, detection index and observed pixel; side identities still need visual review.',
                  'All-selected-view residuals can have different populations. Read them together with coverage and paired statistics.',
                  'fused_camera_tiles counts camera/frame; fused_camera_hand_tiles and native_2d_camera_hand_tiles count camera/frame/hand.',
                  'Common 3D geometry guarantees consistent reprojection, not accurate anatomy or ground-truth 3D. No 2D/3D labels are supplied.',
              ]}
    return report, samples


def _probe(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames,avg_frame_rate', '-of', 'json', str(path)]))
    stream = data['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    return int(stream['nb_frames']), numerator / denominator


def render_comparison(report, output):
    """Encode synchronized before/after grids with explicit baseline labels."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    before, after = (Path(report['videos'][name]) for name in ('legacy', 'shared'))
    n0, fps0 = _probe(before)
    n1, fps1 = _probe(after)
    if n0 != n1 or n0 != report['synchronized_frames'] or abs(fps0 - fps1) > .001:
        raise ValueError('comparison videos must have identical synchronized frames and fps')
    matching_mesh = report.get('legacy', {}).get('display', {}).get('shared_mesh', False)
    native_mesh = report.get('legacy', {}).get('display', {}).get('native_mesh', False)
    labels = [('LEGACY skeleton + SAME shared mesh' if matching_mesh else
               'LEGACY skeleton + NATIVE WiLoR mesh' if native_mesh else 'BEFORE - legacy hybrid skeleton'),
              'SHARED MANO skeleton + mesh']
    filters = []
    for index, label in enumerate(labels):
        filters.append(f'[{index}:v]setpts=PTS-STARTPTS,scale=960:600:force_original_aspect_ratio=decrease,'
                       f'pad=960:600:(ow-iw)/2:(oh-ih)/2,setsar=1,pad=960:640:0:40:color=black,'
                       f"drawtext=text='{label}':fontcolor=white:fontsize=24:x=16:y=9[v{index}]")
    filters.append('[v0][v1]hstack=inputs=2[out]')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial.mp4')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(before), '-i', str(after), '-filter_complex_threads', '1',
                    '-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                    '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], check=True)
    if _probe(temporary)[0] != n0:
        raise ValueError('comparison encoding changed synchronized frame count')
    temporary.replace(output)
    report['videos']['comparison'] = str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new JSON report (also writes paired CSV)')
    parser.add_argument('--video', type=Path, help='optional new 1920x640 comparison MP4')
    parser.add_argument('--legacy-video', type=Path, help='optional legacy skeleton video with matching shared mesh overlay')
    args = parser.parse_args()
    output = args.output.resolve()
    paired = output.with_suffix('.paired.csv')
    if output == paired or output.exists() or paired.exists():
        parser.error('use new JSON and paired CSV paths')
    report, samples = compare_run(args.run, legacy_video=args.legacy_video)
    if args.video:
        render_comparison(report, args.video)
    output.parent.mkdir(parents=True, exist_ok=True)
    with paired.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sync_index', 'side', 'camera', 'joint',
            'detection_index', 'legacy_residual_px', 'shared_residual_px', 'delta_px'])
        writer.writeheader()
        writer.writerows(samples)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
