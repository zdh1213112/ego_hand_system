from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ego_data.calibration import CameraCalibration
from render_gem_multiview import draw_mesh_surface
from shared_mesh_overlay import (SharedManoMeshOverlay, draw_shared_mesh,
                                 mesh_projection_fingerprint)
from validate_matching_wilor_mesh import audit_mesh


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


def test_overlay_uses_shared_hand_coverage_and_rejects_wrong_dataset(tmp_path):
    dataset, shared = tmp_path / 'dataset', tmp_path / 'shared'
    (dataset / 'calibration').mkdir(parents=True)
    shared.mkdir()
    inputs = []
    for relative in ('manifest.json', 'multiview_frames.csv', 'calibration/camera0.json'):
        path = dataset / relative
        path.write_text('{}')
        stat = path.stat()
        inputs.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    (shared / 'run_config.json').write_text(json.dumps({'inputs': inputs}))
    hand = {'side': 0, 'vertices_base_m': np.array([[-.1, -.1, .6], [.1, -.1, .6], [0, .1, .6]])}
    torch.save({'format': 'shared_wilor_mano_v1', 'topology': {'left': [[0, 2, 1]], 'right': [[0, 1, 2]]},
                'frames': [{'sync_index': 0, 'hands': [hand]}, {'sync_index': 1, 'hands': []}]},
               shared / 'shared_pose.pt')
    overlay = SharedManoMeshOverlay(shared, dataset, ['camera0'], [0, 1])
    image = np.zeros((100, 160, 3), np.uint8)
    audits = overlay.draw(image, 0, calibration('KB', 0.))
    assert len(audits) == 1 and audits[0]['shared_model_id'] == '0:0'
    assert overlay.draw(image, 1, calibration('KB', 0.)) == []
    assert overlay.summary()['requested'] == 1
    assert overlay.summary()['independent_view_fallback'] is False
    with pytest.raises(ValueError, match='incomplete'):
        SharedManoMeshOverlay(shared, dataset, ['camera0'], [2])
    (dataset / 'calibration/camera0.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='dataset/calibration differs'):
        SharedManoMeshOverlay(shared, dataset, ['camera0'], [0])


def test_mesh_audit_detects_modified_projection_and_missing_frames():
    cal = calibration('KB', .035)
    hand = {'side': 1, 'vertices_base_m': np.array([[-.1, -.1, .6], [.1, -.1, .6], [0, .1, .6]])}
    topology = {'left': [[0, 2, 1]], 'right': [[0, 1, 2]]}
    data = {'frames': [{'sync_index': 7, 'hands': [hand]}], 'topology': topology}
    _, fingerprint = draw_shared_mesh(np.zeros((100, 160, 3), np.uint8), hand, topology, cal)
    audit = {'sync_index': 7, 'cameras': {'camera0': [
        {'side': 1, 'shared_model_id': '7:1', 'mesh_projection_sha256': fingerprint}]}}
    result = audit_mesh(data, {'camera0': cal}, [audit], {7: audit})
    assert result['checked_camera_hand_meshes'] == 1
    assert result['checked_vertex_projections'] == 3
    with pytest.raises(ValueError, match='incomplete'):
        audit_mesh(data, {'camera0': cal}, [], {7: audit})
    audit['cameras']['camera0'][0]['mesh_projection_sha256'] = 'corrupted'
    with pytest.raises(ValueError, match='mesh projection differs'):
        audit_mesh(data, {'camera0': cal}, [audit], {7: audit})
