#!/usr/bin/env python3
"""Compare paired predictions in native camera coordinates without image warping."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def key(row):
    return (row['source_group'], int(row['sync_index']), row['camera'], int(row['side']))


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate prediction identity: {path}')
    return result


def to_native_pixels(pixels, rect_K, rotation, camera):
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(rect_K).T @ rotation
    return cv2.fisheye.projectPoints(
        rays[:, None], np.zeros(3), np.zeros(3),
        np.asarray(camera['K'], float), np.asarray(camera['distortion'], float),
    )[0][:, 0]


def to_native_relative(points, rotation, side):
    sign = np.array([-1., 1., 1.]) if side == 0 else np.ones(3)
    return (np.asarray(points) * sign) @ rotation * sign


def mean_distance(first, second, factor=1.):
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second), axis=-1).mean() * factor)


def compare(native, rectified, rect_K, rotation, camera):
    side = int(native['side'])
    pred_px = to_native_pixels(np.asarray(rectified['pred_joints_2d']), rect_K, rotation, camera)
    gt_px = to_native_pixels(np.asarray(rectified['gt_joints_2d']), rect_K, rotation, camera)
    pred_3d = to_native_relative(rectified['pred_joints_3d_root_relative'], rotation, side)
    gt_3d = to_native_relative(rectified['gt_joints_3d_root_relative'], rotation, side)
    pred_root = np.asarray(rectified['pred_root_camera_m']) @ rotation
    gt_root = np.asarray(rectified['gt_root_camera_m']) @ rotation
    visible = np.asarray(native['visibility_2d'], bool) & np.asarray(rectified['visibility_2d'], bool)
    if not visible.any():
        raise ValueError('No common visible joints')
    target_px = np.asarray(native['gt_joints_2d'])
    target_3d = np.asarray(native['gt_joints_3d_root_relative'])
    target_root = np.asarray(native['gt_root_camera_m'])
    return dict(
        camera=native['camera'], side=side, sync_index=native['sync_index'],
        gt_pixel_agreement_px=mean_distance(gt_px, target_px),
        gt_relative_agreement_mm=mean_distance(gt_3d, target_3d, 1000),
        gt_root_agreement_mm=mean_distance(gt_root, target_root, 1000),
        native_reprojection_px=mean_distance(np.asarray(native['pred_joints_2d'])[visible], target_px[visible]),
        rectified_to_native_reprojection_px=mean_distance(pred_px[visible], target_px[visible]),
        native_mpjpe_mm=mean_distance(native['pred_joints_3d_root_relative'], target_3d, 1000),
        rectified_to_native_mpjpe_mm=mean_distance(pred_3d, target_3d, 1000),
        native_root_error_mm=mean_distance(native['pred_root_camera_m'], target_root, 1000),
        rectified_to_native_root_error_mm=mean_distance(pred_root, target_root, 1000),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['native-predictions', 'rectified-predictions', 'rectified-manifest',
                 'rectification', 'calibration-dir', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    native, rectified = read_rows(args.native_predictions), read_rows(args.rectified_predictions)
    manifest = read_rows(args.rectified_manifest)
    if native.keys() != rectified.keys() or not native.keys() <= manifest.keys():
        raise ValueError('Require exactly matching recording/frame/camera/side predictions')
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.rectification, allow_pickle=False) as archive:
        rotations = {'camera0': archive['R1'].astype(float), 'camera1': archive['R2'].astype(float)}
    cameras = {name: json.loads((args.calibration_dir/(name+'.json')).read_text()) for name in rotations}
    result = []
    for identity, row in native.items():
        label_path = Path(manifest[identity]['label'])
        if not label_path.is_absolute():
            label_path = args.rectified_manifest.parent / label_path
        rect_K = np.load(label_path, allow_pickle=True).item()['K'].astype(float)
        result.append(compare(row, rectified[identity], rect_K, rotations[row['camera']], cameras[row['camera']]))
    for field, tolerance in [('gt_pixel_agreement_px', .1), ('gt_relative_agreement_mm', .05), ('gt_root_agreement_mm', .05)]:
        if max(row[field] for row in result) > tolerance:
            raise ValueError(f'Ground truth frame conversion failed: {field}')
    fields = [name for name in result[0] if name not in ['camera', 'side', 'sync_index']]
    def aggregate(rows):
        return dict(samples=len(rows), **{field: float(np.mean([row[field] for row in rows])) for field in fields})
    report = dict(
        comparison_space='native_raw physical cameras; canonical root-relative 3D',
        inference_boxes='label boxes, each route in its own image geometry',
        sources={name: str(value.resolve()) for name, value in vars(args).items() if name != 'output'},
        metrics=aggregate(result),
        by_camera={camera: aggregate([row for row in result if row['camera'] == camera]) for camera in cameras},
        records=result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
