import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path[:0]=[str(SCRIPTS),str(SCRIPTS/'joint_refit')]
from personal_marker24_config import load_joint_refit_config
from sdk_marker_gaps import MarkerGapTargets,PROFILE_FORMAT
from sdk_directions import DirectionTargets,PROFILE_FORMAT as DIRECTION_FORMAT


@pytest.mark.parametrize('side',[0,1])
def test_only_one_missing_is_inferred_in_the_saved_hand_coordinates(tmp_path,side):
    rng=np.random.default_rng(631)
    raw=rng.normal(size=(2,3,24,3)).astype(np.float32)*.05
    estimated=raw[:,:,4:].copy()+np.array([.002,-.001,.003],np.float32)
    path=tmp_path/'gaps.npz'
    np.savez(path,format=PROFILE_FORMAT,sdk_marker_cam0_m=estimated,
             confidence=np.full((2,20),.4,np.float32),raw_marker24_cam0_m=raw)
    mirror=np.array([-1.,1.,1.] if side==0 else [1.,1.,1.])
    observed=(raw[side]*mirror).copy()
    observed[0,18]=np.nan
    observed[1,[18,19]]=np.nan
    rot=Rotation.from_rotvec([.10,-.15,.2]).as_matrix()
    shift=np.array([.006,.004,-.013])
    aligned=observed@rot.T+shift
    saved=aligned.copy()
    gaps=MarkerGapTargets(path,side,np.arange(3),observed,aligned,'cpu')
    assert gaps.rows.tolist()==[0] and gaps.markers.tolist()==[18]
    np.testing.assert_allclose(gaps.points.numpy()[0],(estimated[side,0,14]*mirror)@rot.T+shift,atol=2e-7)
    np.testing.assert_equal(aligned,saved)
    np.testing.assert_equal(np.isfinite(aligned),np.isfinite(observed))
    bad=observed.copy();bad[2,4,0]+=.01
    with pytest.raises(ValueError,match='do not match'):
        MarkerGapTargets(path,side,np.arange(3),bad,aligned,'cpu')


def test_missing_sdk_directions_do_not_contribute_or_break_gradients(tmp_path):
    raw=np.ones((2,1,24,3),np.float32)
    v=np.full((2,1,15,3),np.nan,np.float32)
    profile=tmp_path/'dir.npz'
    np.savez(profile,format=DIRECTION_FORMAT,local_directions=v,confidence=np.ones((2,15),np.float32),raw_marker24_cam0_m=raw)
    d=DirectionTargets(profile,1,np.array([0]),raw[1],'cpu')
    points=torch.randn(1,21,3,requires_grad=True)
    loss=d.penalty(points,torch.eye(3)[None]);loss.backward()
    assert loss.item()==0. and torch.equal(points.grad,torch.zeros_like(points))


def test_recording_config_preserves_legacy_and_checks_sdk_options(tmp_path):
    placement=tmp_path/'placement.json';placement.write_text('{}')
    path=tmp_path/'config.json'
    current={'enabled':True,'placement':placement.name}
    def config(**options):
        path.write_text(json.dumps({'schema_version':1,'joint_refit':dict(current,**options)}))
        return load_joint_refit_config(path)
    assert config()==dict(enabled=True,placement=str(placement),sample_stride=8)
    assert config(sdk_marker_gaps=True,min_markers_per_finger=3,joint_limit_weight=.05)['sdk_marker_gaps']
    for option in ({'sdk_marker_gaps':1},{'min_markers_per_finger':5},{'joint_limit_weight':-1}):
        with pytest.raises(ValueError):
            config(**option)
