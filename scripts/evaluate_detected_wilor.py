"""Score independent camera predictions from a detector-driven WiLoR recording."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ego_hand_inference.ego_data.calibration import CameraCalibration
from evaluate_hand_models import report, write_json
from ego_hand_inference.fit_shared_wilor_multiview import load_mano, physical_geometry
from hand_evaluation.dataset import LabeledDataset, sha256
from hand_evaluation.metrics import ERROR_NAMES, bbox_iou, measure


def select_by_side(hands):
    """Single camera, one detection per side, independent of labels and other views."""
    result = {}
    for hand in hands:
        side = hand['detector_is_right']
        if side not in (0, 1) or hand['is_right'] != side:
            continue
        if side not in result or hand['confidence'] > result[side]['confidence']:
            result[side] = hand
    return result


def read_selected(run, dataset):
    normalized = run / 'normalized_multiview'
    with (normalized / 'multiview_frames.csv').open() as stream:
        sync = list(csv.DictReader(stream))
    frames = {}
    for camera in sorted({row['camera'] for row in dataset.selected}):
        actual = CameraCalibration.load(normalized / 'calibration' / f'{camera}.json')
        expected = CameraCalibration.from_dict(dataset.summary['camera_calibrations'][camera])
        if actual.model != expected.model or actual.image_size != expected.image_size:
            raise ValueError(f'{camera}: incompatible image geometry')
        for key, tol in [('K', 1e-3), ('distortion', 1e-7), ('T_base_camera', 1e-6)]:
            np.testing.assert_allclose(getattr(actual,key), getattr(expected,key), atol=tol, rtol=0)
        with (run / 'wilor_multiview' / camera / 'predictions.jsonl').open() as stream:
            for info, record in zip(sync, map(json.loads, stream), strict=True):
                if (record['sync_index'] != int(info['sync_index'])
                        or record['source_frame_index'] != int(info[f'{camera}_frame_index'])
                        or record['timestamp_ns'] != int(info[f'{camera}_timestamp_ns'])):
                    raise ValueError(f'{camera}: prediction frame alignment failed')
                key = (camera, int(info['reference_timestamp_ns']), record['source_frame_index'])
                if key in frames:
                    raise ValueError(f'duplicate camera frame {key}')
                frames[key] = select_by_side(record['hands'])
    return frames


def native_meshes(hands, layer):
    """Reconstruct MANO vertices from the predictions for this batch of labels."""
    meshes = {}
    for side in (0, 1):
        indices = [i for i, hand in enumerate(hands) if hand and hand['is_right'] == side]
        if not indices:
            continue
        shapes = {'global_orient': (1,3,3), 'hand_pose': (15,3,3), 'betas': (10,)}
        params = {key: torch.as_tensor(np.stack([np.reshape(hands[i]['mano'][key], shape)
                  for i in indices]), dtype=torch.float32) for key, shape in shapes.items()}
        with torch.no_grad():
            _, verts = physical_geometry(layer, params, side)
        for i, v in zip(indices, verts.numpy()):
            meshes[i] = v + np.asarray(hands[i]['camera_translation'])
    return meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.backend = 'wilor'
    torch.set_num_threads(2)
    started = time.monotonic()
    dataset = LabeledDataset(args.dataset)
    frames = read_selected(args.run, dataset)
    root = Path(__file__).resolve().parents[1]
    layer = load_mano(root / 'models/mano')
    model = json.loads((args.run / 'wilor_multiview/summary.json').read_text())['model']
    protocol = dict(input='Original recording, detector boxes and detector handedness',
                    assignment='highest confidence per predicted side, require IoU >= 0.5 with same-side GT for scoring only',
                    placement='calibrated translation using only predicted 2D and actual camera calibration',
                    denominator='all labeled camera/hands for coverage and PCK, valid predictions only for mean errors')
    assets = [dict(path=model[key], sha256=sha256(model[key])) for key in ('detector','checkpoint','adapter') if model[key]]
    config = dict(model=model, assets=assets, protocol=protocol, run=str(args.run.resolve()))
    args.output.mkdir()
    write_json(args.output / 'run_config.json', config)
    scores, masks, statuses = [], [], Counter()
    errors = {name: [] for name in ERROR_NAMES}
    for start in range(0, len(dataset.selected), 64):
        labels = [dataset.load(row) for row in dataset.selected[start:start+64]]
        selected, overlaps = [], []
        for label in labels:
            row = label.row
            frame = frames[row['camera'], int(row['reference_timestamp_ns']), int(row['source_frame_index'])]
            hand = frame.get(label.side)
            overlap = bbox_iou(hand['bbox_xyxy'], label.bbox) if hand else None
            overlaps.append(overlap)
            selected.append(hand if overlap is not None and overlap >= 0.5 else None)
        meshes = native_meshes(selected, layer)
        for i, (label, hand, overlap) in enumerate(zip(labels, selected, overlaps)):
            if hand is None:
                pred = dict(joints_camera_m=np.full((21,3), np.nan), joints_2d_px=np.full((21,2), np.nan))
                status = 'no_same_side_box' if overlap is None else 'box_iou_below_0.5'
            else:
                pred = dict(joints_camera_m=np.asarray(hand['joints_3d_camera_relative']) + hand['camera_translation'],
                            joints_2d_px=hand['joints_2d'], vertices_camera_m=meshes[i])
                status = 'matched'
            score, per_joint = measure(label, pred, placement='calibrated')
            score.update(match_status=status, detection_iou=overlap)
            scores.append(score)
            statuses[status] += 1
            masks.append(label.valid)
            for name in errors:
                errors[name].append(per_joint[name])
        if len(scores) % 1024 == 0:
            print(f'Scored {len(scores)}/{len(dataset.selected)} camera/hand labels', flush=True)
    report(args, dataset, config, scores, errors, masks, time.monotonic()-started, 0,
           evaluation_protocol=protocol, extra={'match_status':dict(statuses)},
           limitations=['Fitted NOKOV/MANO reference labels; coverage is for labeled camera/hand samples.'])


if __name__ == '__main__':
    main()
