from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import wilor_native_mesh_overlay as native


def test_mesh_selection_uses_hypothesis_or_original_observation_not_display_side():
    hands = [{'detection_index': 3, 'hypothesis_index': 6 + side,
              'is_right': side, 'joints_2d': np.full((21, 2), 10 + side).tolist()}
             for side in (0, 1)]
    assert native.select_native_hand(hands, {'hypothesis_index': 7}) is hands[1]
    # Fusion can change physical side and projected joints, while its original
    # WiLoR observation still uniquely identifies the native MANO branch.
    view = {'detection_index': 3, 'joints_2d': hands[0]['joints_2d'],
            'wilor_joints_2d': hands[1]['joints_2d']}
    assert native.select_native_hand(hands, view) is hands[1]
    assert native.select_native_hand(hands, {'hypothesis_index': 99}) is None
    assert native.select_native_hand(hands, {'detection_index': 4, 'joints_2d': hands[0]['joints_2d']}) is None


def test_held_track_reuses_original_camera_frame_mesh_and_translation(monkeypatch):
    overlay = native.WilorNativeMeshOverlay.__new__(native.WilorNativeMeshOverlay)
    hand = {'hypothesis_index': 2, 'detection_index': 1, 'is_right': 0,
            'camera_translation': [.1, .2, 2.]}
    overlay.records = {'camera0': {8: {'hands': [hand]}, 9: {'hands': []}}}
    overlay.vertices = {'camera0': {(8, 2): np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])}}
    overlay.faces = {0: np.array([[0, 2, 1]])}
    overlay.K = np.eye(3)
    overlay.audits = []
    overlay.stats = {'requested': 0, 'drawn': 0, 'missing': 0, 'outside_image': 0}
    captured = {}
    def draw(image, vertices, faces, intrinsics, side):
        captured.update(vertices=vertices, faces=faces, side=side)
        return True
    monkeypatch.setattr(native, 'draw_mesh_surface', draw)
    overlay.draw(np.zeros((10, 10, 3), np.uint8), 'camera0', 9,
                 {'mesh_source_frame_index': 8, 'hypothesis_index': 2}, 1)
    np.testing.assert_allclose(captured['vertices'], overlay.vertices['camera0'][8, 2] + [.1, .2, 2.])
    np.testing.assert_array_equal(captured['faces'], overlay.faces[0])
    assert captured['side'] == 1  # Physical display color does not mirror geometry again.
    assert overlay.audits[0]['source_frame_index'] == 8
    assert overlay.stats['drawn'] == 1
