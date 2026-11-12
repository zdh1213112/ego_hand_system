#!/usr/bin/env python3
"""Verify the legacy-skeleton overlay uses the shared video's entire mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from ego_data.calibration import CameraCalibration
from shared_mesh_overlay import mesh_projection_fingerprint


def audit_mesh(data, calibrations, audits, reference_audits):
    frames = {int(frame['sync_index']): frame for frame in data['frames']}
    seen, checked, vertices = set(), 0, 0
    for audit in audits:
        sync = int(audit['sync_index'])
        if sync in seen or sync not in frames:
            raise ValueError('duplicate or unexpected mesh frame')
        seen.add(sync)
        models = {int(hand['side']): hand for hand in frames[sync]['hands']}
        reference = reference_audits[sync]
        if audit['cameras'].keys() != calibrations.keys():
            raise ValueError('mesh cameras differ')
        for camera, cal in calibrations.items():
            hands = audit['cameras'][camera]
            reference_ids = {hand['shared_model_id'] for hand in reference['cameras'][camera]}
            if len(hands) != len(models) or {hand['shared_model_id'] for hand in hands} != reference_ids:
                raise ValueError('mesh coverage differs from shared video')
            for hand in hands:
                side = int(hand['side'])
                if hand['shared_model_id'] != f'{sync}:{side}':
                    raise ValueError('mesh side/model ID mismatch')
                model = models[side]
                # The exact transform and arrays used to generate the already
                # existing shared video, before extracting the common helper.
                xyz = ((np.asarray(model['vertices_base_m']) - cal.T_base_camera[:3, 3])
                       @ cal.T_base_camera[:3, :3])
                expected = mesh_projection_fingerprint(xyz, data['topology']['right' if side else 'left'], cal)
                if hand['mesh_projection_sha256'] != expected:
                    raise ValueError(f'mesh projection differs at {sync}/{camera}/{side}')
                checked += 1
                vertices += len(xyz)
    if seen != frames.keys():
        raise ValueError('incomplete mesh projection audit')
    return {'audited_frames': len(seen), 'checked_camera_hand_meshes': checked,
            'checked_vertex_projections': vertices, 'all_mesh_projection_fingerprints_match': True}


def decode(video, frame_count, fps, size):
    capture = cv2.VideoCapture(str(video))
    count, previews = 0, []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[1::-1] != tuple(size):
                raise ValueError('unexpected video dimensions')
            if count in {0, frame_count // 2, frame_count - 1}:
                path = video.with_name(f'{video.stem}_preview_{count:04d}.jpg')
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError(path)
                previews.append(str(path))
            count += 1
    finally:
        capture.release()
    if count != frame_count:
        raise ValueError(f'incomplete video: {count}/{frame_count}')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration,size', '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    if int(stream['nb_frames']) != count or abs(numerator / denominator - fps) > .001:
        raise ValueError('frame count/fps mismatch')
    return {'video': str(video), 'decoded_frames': count, 'probe': probe, 'previews': previews}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-video', type=Path, required=True)
    parser.add_argument('--shared-video', type=Path, required=True)
    parser.add_argument('--original-legacy-video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    legacy = read(args.legacy_video.with_suffix('.json'))
    shared = read(args.shared_video.with_suffix('.json'))
    original = read(args.original_legacy_video.with_suffix('.json'))
    for key in ('dataset', 'fusion', 'predictions', 'cameras', 'frame_count', 'fps',
                'physical_side_swap', 'trust', 'fused_camera_tiles', 'fused_camera_hand_tiles',
                'native_2d_camera_hand_tiles'):
        if legacy[key] != original[key]:
            raise ValueError(f'legacy skeleton behavior changed: {key}')
    mesh = legacy['mesh']
    shared_dir = Path(shared['config']['shared'])
    if (Path(mesh['shared']) != shared_dir or mesh['shared_mtime_ns'] != shared['config']['shared_mtime_ns']
            or mesh['shared_mtime_ns'] != (shared_dir / 'shared_pose.pt').stat().st_mtime_ns
            or mesh['shared_size'] != (shared_dir / 'shared_pose.pt').stat().st_size
            or mesh['drawn'] != shared['mesh_draws']
            or legacy['frame_count'] != shared['frame_count']
            or legacy['cameras'] != shared['config']['cameras']
            or legacy['tile_width'] != shared['config']['tile_width']
            or any(mesh[k] != v for k, v in shared['config']['mesh_style'].items())):
        raise ValueError('shared mesh file, frames, cameras, dimensions or style differ')
    data = torch.load(shared_dir / 'shared_pose.pt', map_location='cpu', weights_only=False, mmap=True)
    dataset = Path(legacy['dataset'])
    calibrations = {c: CameraCalibration.load(dataset / 'calibration' / f'{c}.json') for c in legacy['cameras']}
    with args.shared_video.with_suffix('.projections.jsonl').open() as stream:
        references = {int(row['sync_index']): row for row in map(json.loads, stream)}
    with args.legacy_video.with_suffix('.mesh_projections.jsonl').open() as stream:
        result = audit_mesh(data, calibrations, map(json.loads, stream), references)
    result.update(decode(args.legacy_video, legacy['frame_count'], legacy['fps'], shared['video_size']))
    result.update({'legacy_skeleton_selection_counts_unchanged': True,
                   'mesh_style': shared['config']['mesh_style'],
                   'mesh_shared_across_videos_and_cameras': True,
                   'legacy_skeleton_and_mesh_share_model': False,
                   'shared_skeleton_and_mesh_share_model': shared['skeleton_and_mesh_share_model']})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
