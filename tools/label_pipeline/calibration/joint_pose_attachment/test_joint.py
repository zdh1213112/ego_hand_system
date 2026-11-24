import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_runner',HERE/'run.py');joint=importlib.util.module_from_spec(spec);spec.loader.exec_module(joint)
ROOT=HERE.parents[3];EXP=ROOT/'output/native_kb_training_20261111'

@pytest.mark.parametrize('side',[0,1])
def test_batched_projection_matches_scalar_kb_with_finite_gradients(side):
 obs=json.loads((EXP/'joint_pose_attachment_observations_v1/observations.json').read_text());records=[r for r in obs['records'] if r['side']==side][::80][:4];z=np.load(EXP/'explicit_pipeline_sampled_smoke_v2/contour_fit'/f'{["left","right"][side]}_pilot.npz');lookup={int(f):i for i,f in enumerate(z['pair_indices'])};batch=joint.ContourBatch(records,lookup,side,'cpu');v=torch.tensor(z['vertices'],requires_grad=True);pixels=batch.project(v)
 for i,r in enumerate(records):
  expected=joint.project(v[lookup[r['sync']]],r,side);torch.testing.assert_close(pixels[i],expected,atol=.0003,rtol=1e-6)
 loss=batch.loss(v);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(v.grad).all()

def test_saved_joint_result_has_shared_shape_and_changes_motion_and_attachments():
 for hand in ['left','right']:
  source=np.load(EXP/'explicit_pipeline_sampled_smoke_v2/contour_fit'/f'{hand}_pilot.npz');z=np.load(EXP/'joint_pose_attachment_all_v1'/f'{hand}_pilot.npz')
  assert np.array_equal(z['betas'],source['betas'])
  assert not np.array_equal(z['hand_pose_pca'],source['hand_pose_pca'])
  assert not np.array_equal(z['global_orient'],source['global_orient'])
  assert not np.array_equal(z['translation'],source['translation'])
  assert not np.array_equal(z['barycentric'],source['barycentric'])
  assert np.min(z['barycentric'])>=0 and np.max(abs(z['barycentric'].sum(1)-1))<1e-5
  assert np.min(z['normal_offsets_m'])>=0 and np.max(z['normal_offsets_m'])<=.004001
  assert np.max(abs(z['hand_pose_pca']-source['hand_pose_pca']))<=.120001
  assert np.max(np.linalg.norm(z['translation']-source['translation'],axis=1))<=.008001
  assert np.max(np.linalg.norm(z['global_orient']-source['global_orient'],axis=1))<=.060001
  assert np.isfinite(z['vertices']).all()

def test_opt_in_chain_materializes_joint_result(tmp_path):
 import subprocess
 project=ROOT;cfg=json.loads((project/'configs/native_personal_marker24_joint.json').read_text());cfg['personal_marker24_optimization']['placement']=str(ROOT.parent/'glove_marker_placement.json');path=tmp_path/'config.json';path.write_text(json.dumps(cfg));output=tmp_path/'out'
 r=subprocess.run([sys.executable,str(project/'scripts/run_personal_marker24_pipeline.py'),'--dataset','/planned/dataset','--alignment','/planned/alignment.json','--aligned21','/planned/aligned.npz','--baseline','/planned/baseline','--config',str(path),'--output',str(output),'--dry-run'],capture_output=True,text=True);assert r.returncode==0,r.stderr;commands=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')];by={x['stage']:x['command'] for x in commands};assert 'joint_pose_attachments' in by;assert str(output/'joint_fit') in by['materialize_optimized_fit'];assert not output.exists()
