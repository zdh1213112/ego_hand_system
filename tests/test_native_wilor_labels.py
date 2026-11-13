from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wilor_label_projection import project_label_points
from export_multiview_wilor_training_dataset import _rectify_mano_geometry
from mano_conventions import mirror_left_points, canonical_rectification_rotation
from scipy.spatial.transform import Rotation


def test_native_projection_matches_angle_polynomial():
    theta = np.array([0., .2, .8, 1.2])
    points = np.column_stack([np.sin(theta), np.zeros(4), np.cos(theta)])
    D = np.array([.01, -.002, .0003, -.00004])
    K = np.array([[400., 0., 960.], [0., 410., 600.], [0., 0., 1.]])
    label = dict(K=K, image_space="native_raw", camera_model="KB", distortion=D)
    actual = project_label_points(points, label)
    t = theta * (1 + sum(D[i] * theta ** (2*i+2) for i in range(4)))
    np.testing.assert_allclose(actual[:, 0], 960 + 400*t, atol=1e-9)
    np.testing.assert_allclose(actual[:, 1], 600, atol=1e-9)
    perspective = project_label_points(points, dict(K=K))
    assert abs(perspective[-1, 0] - actual[-1, 0]) > 100


@pytest.mark.parametrize("side", [0, 1])
def test_native_inverse_rotation_preserves_physical_camera_points(side):
    rng = np.random.default_rng(42)
    vertices = rng.normal(size=(778, 3)).astype(np.float32) * .03
    joints = vertices[:21].copy()
    trans = np.array([.1, -.1, .5], np.float32)
    R = Rotation.from_rotvec([.03, -.05, .01]).as_matrix().astype(np.float32)
    Q = canonical_rectification_rotation(R, side)
    v, j, t, _ = _rectify_mano_geometry(vertices, joints, trans, Q)
    nv, nj, nt, _ = _rectify_mano_geometry(v, j, t, Q.T)
    np.testing.assert_allclose(nv + nt, vertices + trans, atol=1e-7)
    expected = mirror_left_points(joints + trans, side) @ R.T
    np.testing.assert_allclose(mirror_left_points(j + t, side), expected, atol=1e-7)
    np.testing.assert_allclose(nj + nt, joints + trans, atol=1e-7)


def test_native_projection_rejects_missing_or_unknown_camera():
    with pytest.raises(ValueError):
        project_label_points(np.array([[0., 0., 1.]]), dict(K=np.eye(3), image_space="native_raw", camera_model="unknown"))
    with pytest.raises(ValueError):
        project_label_points(np.array([[0., 0., -1.]]), dict(K=np.eye(3)))
