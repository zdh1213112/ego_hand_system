"""Metrics in millimetres; missing predictions stay in PCK/coverage denominators."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from camera_models import project_points, unproject_points


def bbox_iou(box, target):
    """Intersection over union of two xyxy boxes in original image pixels."""
    box, target = np.asarray(box), np.asarray(target)
    intersection_size = np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2])
    intersection = np.maximum(0, intersection_size).prod()
    box_area = np.maximum(0, box[2:] - box[:2]).prod()
    target_area = np.maximum(0, target[2:] - target[:2]).prod()
    union = box_area + target_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def similarity_transform(pred, truth):
    """Fit scale + proper rotation + translation. Reflections are forbidden."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    x, y = pred - pred.mean(0), truth - truth.mean(0)
    if len(pred) < 3 or np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError('degenerate Procrustes point set')
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = (u * sign) @ vt
    scale = float(np.dot(singular, sign) / np.sum(x * x))
    return scale, rotation, truth.mean(0) - scale * pred.mean(0) @ rotation


def calibrated_translation(joints, pixels, calibration):
    """Fit only translation using model pixels and calibration, never label joints.

    Both KB and pinhole models use their actual ray/project functions. A root
    subtraction improves conditioning without changing the predicted shape.
    """
    joints, pixels = np.asarray(joints, float), np.asarray(pixels, float)
    if not np.isfinite(joints).all():
        return None
    rays, valid = unproject_points(calibration, pixels)
    valid &= np.isfinite(pixels).all(1)
    if valid.sum() < 6:
        return None
    local = joints - joints[0]
    rays = rays[valid]
    projectors = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
    A = projectors.reshape(-1, 3)
    b = -np.einsum('nij,nj->ni', projectors, local[valid]).reshape(-1)
    initial, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return None

    def residual(translation):
        projected, ok = project_points(calibration, local[valid] + translation)
        error = projected - pixels[valid]
        error[~ok] = 1e4
        return np.nan_to_num(error, nan=1e4, posinf=1e4, neginf=-1e4).ravel()

    fit = least_squares(residual, initial, loss='soft_l1', f_scale=3., max_nfev=40)
    points = local + fit.x
    _, ok = project_points(calibration, points)
    in_front_of_rays = np.einsum('ij,ij->i', points[valid], rays) > 0
    if not fit.success or not ok.all() or not in_front_of_rays.all() or not np.isfinite(points).all():
        return None
    return fit.x - joints[0]


ERROR_NAMES = ('absolute', 'root_relative', 'pa', 'pixel')


def measure(label, pred, *, placement='calibrated'):
    gt, mask = label.joints, label.valid
    joints = np.asarray(pred['joints_camera_m'], float)
    pixels = np.asarray(pred['joints_2d_px'], float)
    if joints.shape != (21, 3) or pixels.shape != (21, 2):
        raise ValueError('prediction must have 21 joints and 21 pixels')
    errors = {key: np.full(21, np.nan) for key in ERROR_NAMES}
    row = dict(index=label.row['index'], image=label.row['image'], camera=label.row['camera'],
               sync_index=label.row['sync_index'], side=label.side,
               source_group=str(label.row.get('source_group', '')), valid_label_joints=int(mask.sum()),
               pose_success=False, placement_success=False, pa_success=False,
               absolute_mpjpe_mm=None, root_relative_mpjpe_mm=None, pa_mpjpe_mm=None,
               wrist_error_mm=None, mean_2d_error_px=None, absolute_mpvpe_mm=None,
               root_relative_mpvpe_mm=None, pa_mpvpe_joint_aligned_mm=None)
    valid_pixels = mask & np.isfinite(pixels).all(1)
    errors['pixel'][valid_pixels] = np.linalg.norm(pixels[valid_pixels] - label.pixels[valid_pixels], axis=1)
    if valid_pixels.any():
        row['mean_2d_error_px'] = float(errors['pixel'][valid_pixels].mean())
    # A partly nonfinite hand is a pose failure, never silently a smaller/easier hand.
    if not np.isfinite(joints).all():
        return row, errors
    row['pose_success'] = True
    if mask[0]:
        errors['root_relative'][mask] = np.linalg.norm(
            (joints - joints[0] - gt + gt[0])[mask], axis=1) * 1000
        row['root_relative_mpjpe_mm'] = float(errors['root_relative'][mask].mean())
    transform = None
    try:
        transform = similarity_transform(joints[mask], gt[mask])
    except ValueError:
        pass
    if transform is not None:
        scale, rotation, translation = transform
        aligned = scale * joints @ rotation + translation
        errors['pa'][mask] = np.linalg.norm(aligned[mask] - gt[mask], axis=1) * 1000
        row['pa_success'] = True
        row['pa_mpjpe_mm'] = float(errors['pa'][mask].mean())
    offset = np.zeros(3) if placement == 'native' else calibrated_translation(joints, pixels, label.calibration)
    if offset is not None:
        errors['absolute'][mask] = np.linalg.norm((joints + offset - gt)[mask], axis=1) * 1000
        row['placement_success'] = True
        row['absolute_mpjpe_mm'] = float(errors['absolute'][mask].mean())
        row['wrist_error_mm'] = float(errors['absolute'][0]) if mask[0] else None
    vertices = pred.get('vertices_camera_m')
    if vertices is not None:
        vertices = np.asarray(vertices, float)
        if vertices.shape != (778, 3):
            raise ValueError('only MANO-corresponding 778-vertex predictions may be scored as MPVPE')
        if np.isfinite(vertices).all():
            if offset is not None:
                row['absolute_mpvpe_mm'] = float(np.linalg.norm(vertices + offset - label.vertices, axis=1).mean() * 1000)
            if mask[0]:
                row['root_relative_mpvpe_mm'] = float(np.linalg.norm(
                    vertices - joints[0] - label.vertices + gt[0], axis=1).mean() * 1000)
            if transform is not None:
                row['pa_mpvpe_joint_aligned_mm'] = float(np.linalg.norm(
                    scale * vertices @ rotation + translation - label.vertices, axis=1).mean() * 1000)
    return row, errors


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    return dict(count=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def summarize(rows, errors, masks):
    masks = np.asarray(masks, bool)
    denominator = int(masks.sum())
    result = dict(samples=len(rows), labeled_joints=denominator)
    for field in ('pose_success', 'placement_success', 'pa_success'):
        count = sum(bool(row[field]) for row in rows)
        result[field] = dict(count=count, rate=count / len(rows) if rows else None)
    result['metrics'] = {}
    for key in rows[0] if rows else []:
        if key.endswith(('_mm', '_px')):
            result['metrics'][key] = stats([row[key] if row[key] is not None else np.nan for row in rows])
    for name in ERROR_NAMES:
        values = np.asarray(errors[name], float)
        flat = values[masks]
        thresholds = (5, 10, 20) if name == 'pixel' else (20, 30, 50)
        pck = {str(t): float(np.count_nonzero(flat <= t) / denominator) for t in thresholds}
        # Exact integral of the empirical PCK curve on [0, 50], normalized to [0,1].
        auc = float(np.where(np.isfinite(flat), np.clip(1 - flat / 50, 0, 1), 0).sum() / denominator)
        result[name] = dict(joint_error=stats(flat), pck=pck,
                            **({} if name == 'pixel' else {'auc_0_50mm': auc}),
                            per_joint=[stats(values[:, i][masks[:, i]]) for i in range(21)])
    return result
