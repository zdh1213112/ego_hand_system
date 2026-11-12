from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ego_data.calibration import CameraCalibration
from fit_shared_wilor_multiview import (
    fit_batch, fit_quality, load_mano, physical_faces, physical_geometry,
    proper_rotations, replay_model,
    read_prediction_records, validate_mano_parameters,
    validate_synchronized_records,
)
from gem_shared_geometry import project_base, rotation_exp


@pytest.fixture(scope='module')
def mano():
    if not (ROOT / 'models/mano/MANO_RIGHT.pkl').is_file():
        pytest.skip('licensed MANO asset is not installed')
    torch.set_num_threads(2)
    return load_mano(ROOT / 'models/mano')


def cameras():
    output = {}
    for name, offset in [('a', -.10), ('b', .10), ('c', .20)]:
        transform = np.eye(4)
        transform[0, 3] = offset
        intrinsics = np.array([[600., 0, 640], [0, 600., 400], [0, 0, 1.]])
        output[name] = CameraCalibration(name, 'optical', 'KB', (1280, 800), intrinsics,
                                         [.08, -.01, .001, .0001], transform)
    return output


def parameters():
    pose = torch.zeros(1, 15, 3)
    pose[0, 0, 2] = .3
    pose[0, 4, 1] = -.4
    return {'global_orient': rotation_exp(torch.tensor([[[.1, -.2, .15]]])),
            'hand_pose': rotation_exp(pose), 'betas': torch.linspace(-.2, .3, 10)[None]}


@pytest.mark.parametrize('side', [0, 1])
def test_physical_geometry_matches_native_wilor_mirror_and_winding(mano, side):
    params = parameters()
    with torch.no_grad():
        canonical = mano(**params, pose2rot=False)
        joints, vertices = physical_geometry(mano, params, side)
    expected_joints, expected_vertices = canonical.joints.clone(), canonical.vertices.clone()
    if side == 0:
        expected_joints[..., 0] *= -1
        expected_vertices[..., 0] *= -1
    torch.testing.assert_close(joints, expected_joints)
    torch.testing.assert_close(vertices, expected_vertices)
    assert joints.shape == (1, 21, 3) and vertices.shape == (1, 778, 3)
    expected_faces = mano.faces if side else mano.faces[:, [0, 2, 1]]
    np.testing.assert_array_equal(physical_faces(mano, side), expected_faces)


def synthetic_candidate(mano, side):
    params = parameters()
    with torch.no_grad():
        joints, _ = physical_geometry(mano, params, side)
    rotation = rotation_exp(torch.tensor([.3, -.2, .1])).numpy()
    points = joints[0].numpy() @ rotation.T + [.01, -.03, .75]
    cals = cameras()
    selected = {}
    for name, cal in cals.items():
        pixels, valid, _ = project_base(cal, points)
        assert valid.all()
        selected[name] = {'pixels': pixels, 'hand': {'confidence': .95, 'detection_index': 0,
            'hypothesis_index': side, 'bbox_xyxy': [*pixels.min(0), *pixels.max(0)]}}
    seed = {k: v[0].numpy().copy() for k, v in params.items()}
    seed['hand_pose'][4] = rotation_exp(torch.tensor([.35, .15, -.1])).numpy() @ seed['hand_pose'][4]
    return {'side': side, 'sync_index': 0, 'seed_camera': 'a', 'seed': {'mano': seed},
            'selected': selected, 'points': points, 'valid': np.ones(21, bool)}, cals


@pytest.mark.parametrize('side', [0, 1])
def test_multiview_fitting_reduces_residual_and_saved_parameters_reproduce_both(mano, side):
    item, cals = synthetic_candidate(mano, side)
    initial, _ = fit_batch([item], item['seed']['mano']['betas'], mano, cals, device='cpu', steps=0)
    fitted, faces = fit_batch([item], item['seed']['mano']['betas'], mano, cals, device='cpu', steps=100)
    model = fitted[0]
    def image_error(value):
        return np.mean([np.linalg.norm(project_base(cal, value['joints_base_m'])[0] - item['selected'][name]['pixels'], axis=1).mean()
                        for name, cal in cals.items()])
    assert image_error(model) < image_error(initial[0]) * .5
    assert image_error(model) < .5
    with torch.no_grad():
        joints, vertices = replay_model(mano, model)
    torch.testing.assert_close(joints, model['joints_base_m'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vertices, model['vertices_base_m'], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(faces, physical_faces(mano, side))
    valid, _, support = fit_quality(model, item, cals, 30.)
    assert valid and len(support) == 3
    only_one = dict(item, selected={'a': item['selected']['a']})
    assert not fit_quality(model, only_one, cals, 30.)[0]


def test_amp_seed_projection_is_a_proper_rotation():
    original = rotation_exp(torch.tensor([[.2, -.1, .3], [.3, -.1, -.2]]))
    quantized = original.half().float()
    fixed = proper_rotations(quantized)
    torch.testing.assert_close(fixed @ fixed.transpose(-1, -2), torch.eye(3).expand(2, -1, -1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(fixed), torch.ones(2), atol=1e-6, rtol=1e-6)


def test_missing_or_invalid_native_parameters_fail_clearly():
    with pytest.raises(ValueError, match='--save-mano-params'):
        validate_mano_parameters({})
    mano = {key: value[0].numpy() for key, value in parameters().items()}
    validate_mano_parameters({'mano': mano})
    mano['betas'][0] = np.nan
    with pytest.raises(ValueError, match='invalid MANO betas'):
        validate_mano_parameters({'mano': mano})


def test_prediction_reader_rejects_ambiguous_duplicate_ids(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    row = {'sync_index': 0, 'hands': [{'hypothesis_index': 1}]}
    path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='duplicate sync_index'):
        read_prediction_records(path)
    row['hands'].append({'hypothesis_index': 1})
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='duplicate hypothesis_index'):
        read_prediction_records(path)


def test_native_seed_must_match_dataset_frame_and_timestamp():
    rows = [{'sync_index': '4', 'a_frame_index': '9', 'a_timestamp_ns': '123456'}]
    records = {4: {'source_frame_index': 9, 'timestamp_ns': 123456}}
    validate_synchronized_records(rows, records, 'a', 'test')
    for wrong, message in [({}, 'missing synchronized frame'),
                           ({4: {'source_frame_index': 8, 'timestamp_ns': 123456}}, 'source frame mismatch'),
                           ({4: {'source_frame_index': 9, 'timestamp_ns': 123457}}, 'timestamp mismatch')]:
        with pytest.raises(ValueError, match=message):
            validate_synchronized_records(rows, wrong, 'a', 'test')


@pytest.mark.parametrize('batched', [False, True])
def test_inference_parameter_opt_in_preserves_native_geometry_for_both_profiles(mano, monkeypatch, batched):
    # Replace the expensive detector/backbone, but exercise the actual
    # production serialization paths with geometry from the real MANO asset.
    import wilor_multiview_inference as inference
    cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000.), MODEL=SimpleNamespace(IMAGE_SIZE=256))
    samples = [{'img': torch.zeros(3, 16, 16), 'right': torch.tensor(float(side)),
                'personid': torch.tensor(side), 'img_size': torch.tensor([1280., 800.]),
                'box_center': torch.tensor([640., 400.]), 'box_size': torch.tensor(200.)}
               for side in (0, 1)]
    monkeypatch.setattr(inference.base, 'ViTDetDataset', lambda *args, **kwargs: samples)
    def model(batch):
        count = len(batch['right'])
        params = {key: value.expand(count, *value.shape[1:]).clone() for key, value in parameters().items()}
        result = mano(**params, pose2rot=False)
        return {'pred_cam': torch.tensor([[1., .1, -.1]]).expand(count, -1).clone(),
                'pred_mano_params': params, 'pred_keypoints_3d': result.joints, 'pred_vertices': result.vertices}
    def prepared(*args, **kwargs):
        state = {'boxes': np.array([[540., 300., 740., 500.]]), 'source_indices': [0, 0],
                 'confidences': [.9], 'detector_sides': [1], 'records': [None, None]}
        return {'states': [state], 'samples': [dict(s, frame_batch_index=0) for s in samples]}
    monkeypatch.setattr(inference, '_prepare_dual_batched', prepared)
    def run(save):
        if batched:
            return inference.infer_dual_batched([np.zeros((32, 32, 3), np.uint8)], None, model, cfg,
                torch.device('cpu'), .3, .5, 2., 2, 1, 0, save_mano_params=save)[0]
        return inference.infer_dual(np.zeros((32, 32, 3), np.uint8), None, model, cfg,
            torch.device('cpu'), .3, .5, 2., 2,
            external_boxes={'boxes': [{'bbox_xyxy': [540., 300., 740., 500.], 'confidence': .9, 'is_right': 1}]},
            save_mano_params=save)
    legacy, saved = run(False), run(True)
    assert len(legacy) == len(saved) == 2
    for before, after in zip(legacy, saved):
        assert 'mano' not in before and 'mano_vertices_camera_relative' not in after
        np.testing.assert_allclose(before['joints_2d'], after['joints_2d'], atol=1e-6)
        params = {name: torch.tensor(value)[None] for name, value in after['mano'].items()}
        with torch.no_grad():
            joints, _ = physical_geometry(mano, params, after['is_right'])
        np.testing.assert_allclose(joints[0], after['joints_3d_camera_relative'], atol=1e-7)
