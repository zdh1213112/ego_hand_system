import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS/'joint_refit')]
from bounded_articulation import components, compose, shared_times
from dorsal_ring_attachment import axial_position, intersect_triangle
from personal_marker24_config import load_joint_refit_config


def test_swing_twist_round_trip_with_distinct_ring_and_pinky_slots():
    slots = torch.tensor([0,1,2,3,4,5,9,10,11,6,7,8,12,13,14])
    bone, flex, lateral = [v.expand(1,15,3) for v in torch.eye(3)]
    axes = slots, bone, flex, lateral
    model = SimpleNamespace(hand_mean=torch.linspace(-.35,.35,45))
    angles = torch.linspace(-25.,75.,135).reshape(3,5,3,3)
    angles[0] = 0.
    pose = compose(angles, model, axes)
    torch.testing.assert_close(components(pose,model,axes),angles,atol=2e-5,rtol=1e-5)


def test_both_hands_must_have_a_common_time_for_dual_export(tmp_path):
    final = tmp_path/'final'
    final.mkdir()
    np.savez(final/'left_pilot.npz',pair_indices=[2,4])
    assert shared_times(tmp_path) == {2,4}
    np.savez(final/'right_pilot.npz',pair_indices=[1,3])
    assert not shared_times(tmp_path)
    np.savez(final/'right_pilot.npz',pair_indices=[3,4])
    assert shared_times(tmp_path) == {4}


def test_dorsal_cross_section_preserves_calibrated_axial_position():
    chain = np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
    point = np.array([1.3,.4,0])
    foot,axis,place = axial_position(point,chain)
    assert place == pytest.approx(1.3)
    tri = np.array([[1.,1,0],[2.,1,0],[1.,1,1]])
    bary = intersect_triangle((tri-foot)@axis)
    assert bary.sum() == pytest.approx(1.)
    assert ((bary@tri-foot)@axis) == pytest.approx(0.,abs=1e-10)


def test_contour_config_requires_the_bounded_pose_input(tmp_path):
    original = SCRIPTS.parent/'configs/native_nokov_articulation.json'
    assert load_joint_refit_config(original)['preserve_visible_contours']
    data = json.loads(original.read_text())
    data['joint_refit'].update(bounded_articulation=False, placement=str(SCRIPTS.parent/'configs/glove_marker_placement.json'))
    path=tmp_path/'cfg.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='requires bounded_articulation'):
        load_joint_refit_config(path)


def test_temporal_stencils_respect_missing_syncs_and_delayed_exposures():
    from temporal_articulation import consecutive_triplets

    stamps = np.array([0, 33, 66, 99, 132, 165, 198, 270, 303, 336])
    # Missing sync 3 and the long exposure interval from 6 to 7 each break the stencil.
    times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(consecutive_triplets(times, stamps), [0, 3, 6])
    assert not len(consecutive_triplets(times[:2], stamps))


def test_temporal_stage_is_opt_in_and_requires_bounded_rotations(tmp_path):
    template = SCRIPTS.parent / 'configs/native_nokov_articulation.json'
    cfg = json.loads(template.read_text())
    refit = cfg['joint_refit']
    refit['placement'] = str(SCRIPTS.parent / 'configs/glove_marker_placement.json')
    refit.pop('temporal_articulation', None)
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps(cfg))
    assert not load_joint_refit_config(path).get('temporal_articulation', False)
    refit['temporal_articulation'] = True
    path.write_text(json.dumps(cfg))
    assert load_joint_refit_config(path)['temporal_articulation']
    refit.update(bounded_articulation=False, preserve_visible_contours=False)
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='temporal_articulation requires bounded_articulation'):
        load_joint_refit_config(path)
