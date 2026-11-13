from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ego_data.calibration import CameraCalibration
from mesh_rendering import draw_mesh_surface
from shared_mesh_overlay import draw_shared_mesh, mesh_camera_vertices, mesh_projection_fingerprint
from camera_models import project_points
from render_shared_wilor_multiview import draw_shared_skeleton

def calibration(model, offset):
    transform = np.eye(4)
    transform[0, 3] = offset
    return CameraCalibration('camera0', 'optical', model, (160, 100),
        [[100, 0, 80], [0, 100, 50], [0, 0, 1]],
        [0.08, -0.01, 0.001, 0.0001] if model == 'KB' else [], transform)


@pytest.mark.parametrize('model', ['KB', 'PINHOLE'])
@pytest.mark.parametrize('side', [0, 1])
def test_shared_mesh_layer_matches_original_renderer_pixels(model, side):
    cal = calibration(model, .035)
    vertices = np.array([[-.1, -.12, .6], [.13, -.1, .61], [.02, .15, .65]], np.float32)
    hand = {'side': side, 'vertices_base_m': vertices}
    topology = {'left': np.array([[0, 2, 1]]), 'right': np.array([[0, 1, 2]])}
    before = np.full((100, 160, 3), 120, np.uint8)
    after = before.copy()
    # Freeze the expression and rasterizer call from the original shared video.
    xyz = (np.asarray(hand['vertices_base_m']) - cal.T_base_camera[:3, 3]) @ cal.T_base_camera[:3, :3]
    assert draw_mesh_surface(before, xyz, topology['right' if side else 'left'], cal.K, side, calibration=cal)
    drawn, fingerprint = draw_shared_mesh(after, hand, topology, cal)
    assert drawn and len(fingerprint) == 64
    np.testing.assert_array_equal(before, after)
    moved = calibration(model, .07)
    _, moved_fingerprint = draw_shared_mesh(after.copy(), hand, topology, moved)
    assert fingerprint != moved_fingerprint
    assert fingerprint != mesh_projection_fingerprint(xyz, topology['left' if side else 'right'], cal)


def test_shared_mesh_and_skeleton_draw_visible_kb_negative_z_geometry():
    cal = CameraCalibration('wide', 'optical', 'KB', (320, 200),
        [[60, 0, 160], [0, 60, 100], [0, 0, 1]], [0, 0, 0, 0], np.eye(4))
    vertices = np.array([[1., -.15, -.05], [1., .15, -.05], [1., 0, -.25]])
    image = np.zeros((200, 320, 3), np.uint8)
    drawn, _ = draw_shared_mesh(image, {'side': 1, 'vertices_base_m': vertices},
                                {'right': [[0, 1, 2]]}, cal)
    assert drawn and image.any()
    points = np.full((21, 3), np.nan)
    points[0], points[5] = vertices[:2]
    xy, valid = project_points(cal, points)
    result = draw_shared_skeleton(image, xy, valid, 1)
    assert result['visible_joint_count'] == 2 and result['drawn_bone_count'] == 1
