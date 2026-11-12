import base64
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visualize_nokov_mano import marker_direction_payload


def fixture():
    joints = np.zeros((1, 21, 3))
    joints[0, 9] = [0, .1, 0]
    joints[0, 5] = [-.04, .08, 0]
    joints[0, 17] = [.04, .08, 0]
    return dict(handedness=np.array('right'), joints=joints,
                vertices=np.array([[[0, 0, 0], [.04, 0, 0], [0, .04, 0]]]),
                faces=np.array([[0, 1, 2]]), attachment_face_ids=np.array([0, 0]),
                markers_world_m=np.array([[[.01234, .00965, .00321], [.01, .01, .01]]]),
                predicted_markers=np.array([[[.01, .01, 0], [.01, .01, 0]]]),
                marker_valid=np.array([[True, False]]))


def unpack(data):
    return {key: np.frombuffer(base64.b64decode(value), dtype='<f4').reshape(1, -1, 3)
            for key, value in marker_direction_payload(data, [0]).items()}


class MarkerDirectionTests(unittest.TestCase):
    def test_axes_face_winding_and_unquantized_coordinates(self):
        packed = unpack(fixture())
        np.testing.assert_allclose(packed['axes'][0], np.eye(3))
        np.testing.assert_allclose(packed['normals'][0], [[0, 0, 1], [0, 0, 1]])
        delta = packed['measured'][0, 0] - packed['attached'][0, 0]
        np.testing.assert_allclose(delta * 1000, [2.34, -.35, 3.21], atol=1e-5)
        self.assertTrue(np.isnan(packed['measured'][0, 1]).all())
        self.assertTrue(np.isfinite(packed['attached'][0, 1]).all())

    def test_hand_components_and_normals_follow_rigid_motion(self):
        data = fixture()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        original = unpack(data)
        for key in ('joints', 'vertices', 'markers_world_m', 'predicted_markers'):
            data[key] = data[key] @ rotation.T + [2., -1., .8]
        moved = unpack(data)
        delta = moved['measured'][0, 0] - moved['attached'][0, 0]
        np.testing.assert_allclose(moved['axes'][0] @ delta * 1000, [2.34, -.35, 3.21], atol=1e-5)
        np.testing.assert_allclose(moved['normals'], original['normals'] @ rotation.T)
        self.assertAlmostEqual(np.linalg.det(moved['axes'][0]), 1.)

    def test_degenerate_frame_normal_and_unsupported_hand_do_not_fake_zeros(self):
        data = fixture()
        data['joints'][:] = 0
        data['vertices'][:] = 0
        packed = unpack(data)
        self.assertTrue(np.isnan(packed['axes']).all())
        self.assertTrue(np.isnan(packed['normals']).all())
        data['handedness'] = np.array('left')
        with self.assertRaisesRegex(ValueError, 'right-hand'):
            unpack(data)


if __name__ == '__main__':
    unittest.main()
