from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ego_data.calibration import CameraCalibration
from fit_shared_wilor_multiview import load_mano, physical_faces, physical_geometry
from gem_shared_geometry import project_base, rotation_exp
from validate_shared_wilor_result import replay_shared_models, audit_projection_rows, audit_native_parameters


@pytest.fixture(scope='module')
def mano():
    if not (ROOT / 'models/mano/MANO_RIGHT.pkl').exists():
        pytest.skip('licensed MANO asset is not installed')
    torch.set_num_threads(2)
    return load_mano(ROOT / 'models/mano')


def params():
    return {'global_orient': rotation_exp(torch.tensor([[[.2, -.1, .3]]])),
            'hand_pose': rotation_exp(torch.full((1, 15, 3), .05)),
            'betas': torch.zeros(1, 10)}


def test_all_model_replay_detects_mesh_corruption_even_when_skeleton_matches(mano):
    hands = []
    rotation = rotation_exp(torch.tensor([.1, -.2, .3]))
    translation = torch.tensor([.01, -.03, .7])
    for side in (0, 1):
        with torch.no_grad():
            joints, vertices = physical_geometry(mano, params(), side)
        hands.append({'side': side, 'sync_index': 2, 'parameters': {k: v[0] for k, v in params().items()},
                      'rotation_base_from_model': rotation, 'translation_base_m': translation,
                      'model_scale': 1.1, 'joints_base_m': joints[0] * 1.1 @ rotation.T + translation,
                      'vertices_base_m': vertices[0] * 1.1 @ rotation.T + translation})
    data = {'format': 'shared_wilor_mano_v1', 'frames': [{'sync_index': 2, 'hands': hands}],
            'topology': {'left': physical_faces(mano, 0), 'right': physical_faces(mano, 1)}}
    result = replay_shared_models(data, mano, batch_size=1)
    assert result['regenerated_model_count'] == 2
    assert result['max_regenerated_vertex_difference_m'] < 1e-7
    data['frames'][0]['hands'][0]['vertices_base_m'][20, 0] += .001
    with pytest.raises(ValueError, match='vertices replay error'):
        replay_shared_models(data, mano)


def test_projection_audit_rejects_per_view_model_or_pixel_changes():
    cals = {}
    for name, offset in (('a', -.05), ('b', .05)):
        transform = np.eye(4)
        transform[0, 3] = offset
        cals[name] = CameraCalibration(name, 'optical', 'KB', (100, 100),
            [[80, 0, 50], [0, 80, 50], [0, 0, 1]], [.1, -.01, 0, 0], transform)
    points = np.zeros((21, 3), np.float32)
    points[:, 2] = .6
    hand = {'side': 1, 'joints_base_m': points, 'supporting_cameras': ['a', 'b']}
    data = {'frames': [{'sync_index': 4, 'hands': [hand]}, {'sync_index': 5, 'hands': []}]}
    accepted = {4: {'hands': [dict(hand, joints_base_m=points.tolist())]}}
    audit = [{'sync_index': 4, 'cameras': {}}, {'sync_index': 5, 'cameras': {'a': [], 'b': []}}]
    for name, cal in cals.items():
        pixels, valid, _ = project_base(cal, points)
        assert valid.all()
        audit[0]['cameras'][name] = [{'side': 1, 'shared_model_id': '4:1',
            'joints_2d': pixels.tolist(), 'visible_joint_count': 21, 'supporting_cameras': ['a', 'b']}]
    result = audit_projection_rows(data, accepted, cals, audit)
    assert result['audited_frames'] == result['checked_camera_hand_joint_projections'] == 2
    assert result['max_projection_difference_px'] == 0.
    broken = deepcopy(audit)
    broken[0]['cameras']['b'][0]['shared_model_id'] = '4:0'
    with pytest.raises(ValueError, match='same shared models'):
        audit_projection_rows(data, accepted, cals, broken)
    broken = deepcopy(audit)
    broken[0]['cameras']['b'][0]['joints_2d'][0][0] += .1
    with pytest.raises(ValueError, match='projection differs'):
        audit_projection_rows(data, accepted, cals, broken)
    with pytest.raises(ValueError, match='cover all'):
        audit_projection_rows(data, accepted, cals, audit[:1])


def test_native_parameter_replay_reports_rounding_separately(mano, tmp_path):
    hands = []
    for side in (0, 1):
        with torch.no_grad():
            joints, _ = physical_geometry(mano, params(), side)
        # Native numerical errors are measured, not used to weaken strict
        # shared model replay tolerance or asserted as ground-truth errors.
        joints[0, 0, 0] += .0001
        hands.append({'is_right': side, 'mano': {k: v[0].tolist() for k, v in params().items()},
                      'joints_3d_camera_relative': joints[0].tolist()})
    hands.append({'is_right': 1})
    camera = tmp_path / 'camera0'
    camera.mkdir()
    (camera / 'predictions.jsonl').write_text(json.dumps({'hands': hands}) + '\n')
    result = audit_native_parameters(tmp_path, ['camera0'], mano, batch_size=1)
    assert result['hypothesis_count'] == 2
    assert result['missing_native_geometry_hypotheses'] == 1
    assert result['max_abs_joint_error_m'] == pytest.approx(.0001, abs=1e-8)
