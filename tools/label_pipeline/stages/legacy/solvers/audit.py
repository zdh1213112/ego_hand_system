# Source work/joint_shape_review_20261119/audit.py, SHA-256 ad6a153ad7594d58b84b5e2ab1883f9fa8a3a9d3e6b50b682d702b2bae852c7f

import case_settings as case

import numpy as np

ROOT = case.stage('joint_shape_review_20261119')

WORK = ROOT.parent

from common import HANDS, track_file

def angle_distance(a, b):
    return np.abs((a - b + 180) % 360 - 180)

def sdk_in_fit_space(group, hand, sdk, raw_marker, times):
    """Restore each archived image-alignment correction using the same named physical markers."""
    with np.load(track_file(group, HANDS[hand])) as z:
        np.testing.assert_array_equal(z["pair_indices"], times)
        mirror = np.array([1 if hand else -1, 1, 1])
        source = raw_marker[hand, times]
        dest = z["target24"] * mirror
        mask = z["marker_valid"] & np.isfinite(source).all(-1)
    count = mask.sum(axis=1)[:, None, None]
    assert count.min() >= 3
    mean_a = np.where(mask[..., None], source, 0).sum(axis=1, keepdims=True) / count
    mean_b = np.where(mask[..., None], dest, 0).sum(axis=1, keepdims=True) / count
    a = np.where(mask[..., None], source - mean_a, 0)
    b = np.where(mask[..., None], dest - mean_b, 0)
    u, _, v = np.linalg.svd(a.swapaxes(1, 2) @ b)
    u[:, :, -1] *= np.linalg.det(u @ v)[:, None]
    rot = u @ v
    error = np.linalg.norm((source - mean_a) @ rot + mean_b - dest, axis=-1)
    assert error[mask].max() < 2.0e-6
    return (sdk[hand, times] - mean_a) @ rot + mean_b

def saved_wilor(group, source_frames):
    observations = {}
    for p in sorted((WORK / "joint_motion_20261119" / group).glob("*wilor.npz")):
        with np.load(p) as z:
            times = z["times"]
            valid = z["valid"]
            uv = z["joints_2d"]
            conf = z["confidence"]
            np.testing.assert_array_equal(z["camera_indices"], source_frames[times])
            for i, c, h in zip(*np.nonzero(valid)):
                key = int(times[i]), int(c), int(h)
                if key not in observations or conf[i, c, h] > observations[key][1]:
                    observations[key] = (uv[i, c, h], float(conf[i, c, h]))
    return observations
