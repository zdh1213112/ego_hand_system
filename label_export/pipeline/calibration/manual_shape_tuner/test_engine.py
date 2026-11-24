import json
from pathlib import Path
import numpy as np
import pytest
from server import Engine,EXP

@pytest.fixture(scope='module')
def engine(tmp_path_factory):return Engine(EXP/'personal_calibration_v3',tmp_path_factory.mktemp('manual-tuner'))

def request(e,side,sync=632):return dict(side=side,sync=sync,betas=e.z[side]['betas'].tolist(),translation_mm=[0,0,0],rotation_deg=[0,0,0],pose_deg=[0]*45)

@pytest.mark.parametrize('side',[0,1])
def test_initial_reconstruction_and_physical_camera_translation(engine,side):
 r=request(engine,side);g=engine.geometry(r);i=engine.maps[side][632]
 assert np.max(abs(g['vertices']-engine.z[side]['vertices'][i]))<2e-7
 before=g['vertices']*np.array([-1 if side==0 else 1,1,1]);r['translation_mm']=[5,-3,2];after=engine.geometry(r)['vertices']*np.array([-1 if side==0 else 1,1,1])
 assert np.max(abs((after-before)*1000-[5,-3,2]))<.001

@pytest.mark.parametrize('side',[0,1])
def test_camera_rotation_preserves_wrist_and_direction(engine,side):
 from scipy.spatial.transform import Rotation
 r=request(engine,side);g=engine.geometry(r);r['rotation_deg']=[8,3,-4];h=engine.geometry(r);mirror=np.array([-1 if side==0 else 1,1,1]);expected=((g['vertices']-g['joints'][0])*mirror)@Rotation.from_euler('xyz',[8,3,-4],degrees=True).as_matrix().T
 assert np.max(abs((h['vertices']-h['joints'][0])*mirror-expected))<2e-7
 assert np.max(abs(g['joints'][0]-h['joints'][0]))<2e-7

@pytest.mark.parametrize('side',[0,1])
def test_shared_shape_retains_frame_wrist(engine,side):
 r=request(engine,side);g=engine.geometry(r);r['betas'][0]+=.3;h=engine.geometry(r)
 assert np.max(abs(g['joints'][0]-h['joints'][0]))<2e-7
 assert np.linalg.norm(g['neutral_vertices']-h['neutral_vertices'])>1e-4
 assert np.array_equal(g['hand_pose'],h['hand_pose'])

@pytest.mark.parametrize('side',[0,1])
def test_pose_and_sizing_request(engine,side):
 r=request(engine,side);g=engine.geometry(r);r['pose_deg'][36]=5;h=engine.geometry(r)
 assert abs((h['hand_pose'][36]-g['hand_pose'][36])-np.deg2rad(5))<1e-7
 before=engine.metrics(r['betas']);s=engine.suggest(dict(betas=r['betas'],metric='palm_width',target_mm=float(before[0]+2)))
 assert s['metrics_mm'][0]>before[0]


def test_save_roundtrip_stale_confirmation_and_bias(engine):
 r=request(engine,1);edits={k:r[k] for k in ['translation_mm','rotation_deg','pose_deg']};frame=dict(**edits,confirmed=True,confirmed_betas=r['betas'],confirmed_edits=edits,checked_cameras=['camera0','camera1'],note='test')
 state=dict(baseline_id=engine.baseline_id,hands=[dict(betas=engine.z[0]['betas'].tolist(),frames={}),dict(betas=r['betas'].copy(),frames={'632':frame})]);saved=engine.save(state);d=json.loads(Path(saved['path']).read_text())
 assert saved['confirmed']==1 and saved['bias_diagnostics']['1']['multiview_confirmed_frames']==1
 assert saved['bias_diagnostics']['1']['provisional'] is True
 assert d['state']['hands'][1]['frames']['632']['confirmed']
 z=np.load(Path(saved['path']).parent/'1_000632.npz');assert np.max(abs(z['vertices']-engine.geometry(r)['vertices']))<1e-8
 state['hands'][1]['betas'][0]+=.1;other=engine.save(state);assert other['confirmed']==0 and not other['bias_diagnostics']

@pytest.mark.parametrize('change',[dict(betas=[float('nan')]*10),dict(sync=-1),dict(side=True),dict(rotation_deg=[100,0,0])])
def test_invalid_edits_rejected(engine,change):
 r=request(engine,1);r.update(change)
 with pytest.raises(ValueError):engine.geometry(r)
