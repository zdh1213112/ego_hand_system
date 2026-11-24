# Source work/anatomy_mounts_20261119/direction.py, SHA-256 0f014d3b8313a9b0d31a3a63be7a4810aa48297bff8bbe27bcdcc3a5e2688751

import case_settings as case

import numpy as np

from audit import sdk_in_fit_space, saved_wilor, angle_distance

CHAINS = np.array([[0, *range(1 + 4 * d, 5 + 4 * d)] for d in range(5)])

TRIPLES = np.array([p[j : j + 3] for p in CHAINS for j in range(3)])

def angles(uv):
    tris = uv[..., TRIPLES, :]
    t = np.diff(tris, axis=-2)
    a, b = t[..., 0, :], t[..., 1, :]
    angle = np.rad2deg(
        np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1))
    )
    length = np.linalg.norm(t, axis=-1).min(-1)
    return angle, length
