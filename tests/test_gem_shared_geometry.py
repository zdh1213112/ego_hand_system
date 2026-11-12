from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ego_data.calibration import CameraCalibration
from gem_shared_geometry import (associate_hands, project_base, project_base_torch,
                                 rigid_initialization, rotation_exp)


def camera(model='KB', offset=0.):
    transform = np.eye(4)
    transform[0, 3] = offset
    intrinsics = np.array([[500., 0, 640], [0, 510, 400], [0, 0, 1]])
    distortion = {'KB': [0.08, -0.01, 0.001, 0.0001],
                  'DS': [500., 510, 640, 400, 0.3, 0.4], 'PINHOLE': []}[model]
    return CameraCalibration('camera', 'optical', model, (1280, 800), intrinsics, distortion, transform)


@pytest.mark.parametrize('model', ['KB', 'DS', 'PINHOLE'])
def test_differentiable_projection_matches_calibration(model):
    cal = camera(model, offset=.05)
    transform = cal.T_base_camera.copy()
    transform[:3, :3] = rotation_exp(torch.tensor([.15, -.1, .2], dtype=torch.float64)).numpy()
    cal = replace(cal, T_base_camera=transform)
    xyz = np.array([[.05, 0., .5], [.2, .3, .6], [-.3, .2, 1.]])
    expected, good, _ = project_base(cal, xyz)
    points = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
    actual, valid, _ = project_base_torch(cal, points)
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-6)
    np.testing.assert_array_equal(valid, good)
    actual.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_rotation_zero_has_finite_nonzero_derivatives_and_proper_rotations():
    angles = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    rotations = rotation_exp(angles)
    np.testing.assert_allclose(rotations.detach()[0], np.eye(3))
    rotations[:, 0, 1].sum().backward()
    assert torch.isfinite(angles.grad).all()
    assert angles.grad.abs().max() > 0
    result = rotation_exp(torch.tensor([[.2, -.5, .8]], dtype=torch.float64))[0]
    np.testing.assert_allclose(result @ result.T, np.eye(3), atol=1e-9)
    assert float(torch.linalg.det(result)) == pytest.approx(1.)


def hand_points():
    points = [[0, 0, .65]]
    for finger in range(5):
        for joint in range(4):
            points.append([(finger - 2) * .022, -(joint + 1) * .025, .65 + finger * .002])
    return np.array(points)


def records_for_hand(points, calibrations):
    records = {}
    for name, cal in calibrations.items():
        pixels, _, _ = project_base(cal, points)
        box = [*pixels.min(0), *pixels.max(0)]
        # Opposite hypotheses of one box must not produce two physical hands.
        hands = [{'detection_index': 0, 'hypothesis_index': side, 'is_right': side,
                  'detector_is_right': 1, 'confidence': .9, 'bbox_xyxy': box,
                  'joints_2d': pixels.tolist()} for side in (0, 1)]
        records[name] = {'source_frame_index': 0, 'hands': hands}
    return records


def test_single_hand_multiview_is_accepted_once_and_recovers_world_points():
    cals = {'a': camera(offset=-.06), 'b': camera(offset=.06), 'c': camera(offset=.12)}
    expected = hand_points()
    hands = associate_hands(records_for_hand(expected, cals), cals)
    assert len(hands) == 1
    assert hands[0]['side'] == 1
    assert len(hands[0]['selected']) == 3
    np.testing.assert_allclose(hands[0]['points'], expected, atol=1e-6)


def test_one_view_or_coincident_cameras_cannot_claim_shared_reconstruction():
    points = hand_points()
    for cals in [{'a': camera()}, {'a': camera(), 'b': camera()}]:
        assert not associate_hands(records_for_hand(points, cals), cals)


def test_rigid_fit_does_not_reflect_and_maps_all_vertices_with_same_transform():
    points = hand_points()
    rotation = rotation_exp(torch.tensor([.3, -.2, .1], dtype=torch.float64)).numpy()
    translation = np.array([.1, .2, -.1])
    target = points @ rotation.T + translation
    r, t = rigid_initialization(points, target, np.ones(21, dtype=bool))
    np.testing.assert_allclose(r, rotation, atol=1e-9)
    np.testing.assert_allclose(t, translation, atol=1e-9)
    assert np.linalg.det(r) > .99999


def test_same_world_model_has_different_pixels_but_same_reconstructed_points():
    points = hand_points()
    first, second = camera(offset=-.06), camera(offset=.06)
    a, _, _ = project_base(first, points)
    b, _, _ = project_base(second, points)
    assert not np.allclose(a, b)
    # Both camera-space transforms invert to the exact same world object.
    for cal in (first, second):
        _, _, local = project_base(cal, points)
        recovered = local @ cal.T_base_camera[:3, :3].T + cal.T_base_camera[:3, 3]
        np.testing.assert_allclose(recovered, points)
