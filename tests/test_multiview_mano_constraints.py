import json,sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multiview_mano_constraints import surface_marker_centres,MultiviewMarkerConstraints
from label_generation_config import load_label_generation_config,require_available_mode


def test_surface_marker_offset_and_gradient():
    v=torch.tensor([[[0.,0.,.5],[1.,0.,.5],[0.,1.,.5]]],requires_grad=True)
    p=surface_marker_centres(v,torch.tensor([[0,1,2]]),torch.tensor([[.2,.3,.5]]),torch.tensor([.002]))
    torch.testing.assert_close(p,torch.tensor([[[.3,.5,.502]]]))
    p.square().sum().backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0


def make_bundle():
    K=[[505.,0,939],[0,506.,592],[0,0,1]]
    cal={}
    for i in range(4):
        T=np.eye(4);T[0,3]=i*.05
        cal[f'camera{i}']=dict(K=K,distortion=[.076,-.014,.0019,.00007],T_base_camera=T.tolist())
    return dict(triangles=np.tile([[[0,1,2]]],(2,21,1)),barycentric=np.full((2,21,3),1/3),offset_m=np.full((2,21),.002),
        pixels=np.zeros((2,2,4,21,2)),confidence=np.ones((2,2,4,21)),train_frame=np.array([True,False]),camera_calibrations_json=json.dumps(cal),reference_camera='camera0')


@pytest.mark.parametrize('side',[0,1])
def test_independent_four_views_and_heldout_exclusion(side):
    b=make_bundle();m=MultiviewMarkerConstraints(b,side,np.arange(2),'cpu')
    centres=torch.tensor([[[.03,.02,.5]]*21]*2,requires_grad=True)
    ids=torch.arange(2)
    baseline=m.pixel_errors(centres,ids).detach()
    # Set observations exactly to the physical KB projections of this hand.
    m.pixels=baseline.clone();assert m.loss(centres,ids,4).item()==pytest.approx(0,abs=1e-7)
    m.pixels[1]+=1000 # heldout frame must not drive gradients
    assert m.loss(centres,ids,4).item()==pytest.approx(0,abs=1e-7)
    m.pixels[0,3]+=5 # fourth camera must drive the objective
    loss=m.loss(centres,ids,4);assert loss.item()>0
    loss.backward();assert torch.isfinite(centres.grad).all()
    assert centres.grad[0].abs().sum()>0 and centres.grad[1].abs().sum()==0
    m.confidence[0,3]=0;assert m.loss(centres,ids,4).item()==pytest.approx(0,abs=1e-7)


def test_config_defaults_do_not_read_optional_artifacts(tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps(dict(schema_version=1,multiview_mano_optimization=dict(enabled=False,bundle='/missing',initial_fit='/missing'))))
    assert load_label_generation_config(p)==load_label_generation_config()
    require_available_mode(load_label_generation_config(p))
    for value in ['false',0,None]:
        p.write_text(json.dumps(dict(schema_version=1,multiview_mano_optimization=dict(enabled=value))))
        with pytest.raises(ValueError):load_label_generation_config(p)
    p.write_text(json.dumps(dict(schema_version=1,multiview_mano_optimization=dict(enabled=True))))
    with pytest.raises(ValueError,match='bundle'):load_label_generation_config(p)


def test_pipeline_false_is_identical_and_true_passes_real_constraint_arguments(tmp_path):
    import subprocess,hashlib
    root=Path(__file__).resolve().parents[1]
    dataset=tmp_path/'take/ego/normalized_multiview';dataset.mkdir(parents=True)
    (dataset/'multiview_frames.csv').write_text('sync_index\n0\n')
    aligned=tmp_path/'aligned.npz';np.savez(aligned,valid=np.ones((2,1,21),bool),alignment_status='accepted')
    cmd=[sys.executable,'-B',str(root/'scripts/run_nokov_native_labels.py'),'--processed-session',str(tmp_path/'take'),'--aligned-nokov',str(aligned),'--output',str(tmp_path/'output'),'--cameras','camera0','camera1','camera2','camera3','--dry-run']
    omitted=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(schema_version=1,multiview_mano_optimization=dict(enabled=False))))
    disabled=subprocess.run(cmd+['--label-config',str(config)],check=True,capture_output=True,text=True).stdout
    assert omitted==disabled
    baseline=tmp_path/'baseline';baseline.mkdir();hashes={}
    for side in (0,1):
        track=baseline/f'track_{side}.npz';np.savez(track,side=side);hashes[str(side)]=hashlib.sha256(track.read_bytes()).hexdigest()
    bundle=tmp_path/'bundle.npz';np.savez(bundle,schema='native_multiview_mano_v1',observation_source='independent_rgb_marker_detections',baseline_fit_sha256_json=json.dumps(hashes))
    config.write_text(json.dumps(dict(schema_version=1,multiview_mano_optimization=dict(enabled=True,bundle='bundle.npz',initial_fit='baseline',pose_iterations=30))))
    result=subprocess.run(cmd+['--label-config',str(config)],check=True,capture_output=True,text=True)
    entries=[json.loads(x) for x in result.stdout.splitlines()];fit=next(x['command'] for x in entries if x['stage']=='fit')
    assert fit[fit.index('--shape-iterations')+1]=='0'
    assert fit[fit.index('--pose-iterations')+1]=='30'
    assert fit[fit.index('--w-2d')+1]=='0'
    assert fit[fit.index('--multiview-marker-bundle')+1]==str(bundle)
    assert not (tmp_path/'output').exists()
    (baseline/'track_1.npz').write_bytes(b'changed')
    result=subprocess.run(cmd+['--label-config',str(config)],capture_output=True,text=True)
    assert result.returncode!=0 and 'differs from attachment calibration' in result.stderr


def test_bundle_rejects_projected_observations_and_mismatched_geometry(tmp_path):
    from multiview_mano_constraints import load_bundle, sha256, CAMERAS
    b = make_bundle()
    data = dict(image_space=np.asarray('native_raw'), pair_indices=np.arange(2),
                positions_reference_camera_m=np.zeros((2,2,21,3)), valid=np.ones((2,2,21), bool),
                track_ids=np.array([0,1]), handedness=np.array(['left','right']),
                camera_calibrations_json=np.asarray(b['camera_calibrations_json']))
    source = tmp_path/'input.npz'
    np.savez(source, **data)
    b.update(schema='native_multiview_mano_v1', observation_source='independent_rgb_marker_detections',
             input_path=str(source), input_sha256=sha256(source), pair_indices=data['pair_indices'], camera_ids=np.asarray(CAMERAS))
    path = tmp_path/'bundle.npz'
    np.savez(path, **b)
    load_bundle(path, data)
    changed = dict(data, positions_reference_camera_m=data['positions_reference_camera_m']+.01)
    with pytest.raises(AssertionError):
        load_bundle(path, changed)
    b['observation_source'] = 'nokov_projected_pixels'
    np.savez(path, **b)
    with pytest.raises(ValueError, match='independent RGB'):
        load_bundle(path, data)


def test_safe_refinement_requires_rgb_and_preserves_3d_and_step_limits():
    from multiview_mano_constraints import safe_refinement_mask
    base=np.zeros((3,21,3));candidate=base.copy();candidate[:,:,0]=.0004
    target=base.copy();target[:,:,0]=.001
    bv=np.zeros((3,778,3));v=bv.copy();v[:,:,0]=.0004
    valid=np.ones((3,21),bool);before=np.full((3,4,21),5.);after=before-1
    confidence=np.ones_like(before);train=np.ones(3,bool);pairs=np.arange(3)
    args=[base,candidate,bv,v,target,valid,before,after,confidence,train,pairs]
    mask,_=safe_refinement_mask(*args);assert mask.all()
    confidence[0]=0
    mask,reasons=safe_refinement_mask(*args);assert not mask[0] and reasons['insufficient_rgb']==1
    candidate[1,:,0]=.004;v[1,:,0]=.004
    mask,reasons=safe_refinement_mask(*args);assert not mask[1] and reasons['step_limit']==1
    candidate[2,:,0]=-.0004
    mask,reasons=safe_refinement_mask(*args);assert not mask[2] and reasons['nokov_error_worse']>=1


def test_safe_refinement_rejects_temporal_correction_jump():
    from multiview_mano_constraints import safe_refinement_mask
    base=np.zeros((3,21,3));candidate=base.copy();candidate[1,:,0]=.002
    target=base.copy();target[:,:,0]=.003
    bv=np.zeros((3,778,3));v=bv.copy();v[1,:,0]=.002
    e=np.full((3,4,21),5.);weights=np.ones_like(e)
    mask,counts=safe_refinement_mask(base,candidate,bv,v,target,np.ones((3,21),bool),e,e-1,weights,np.ones(3,bool),np.arange(3))
    assert not mask[1] and counts['temporal_correction_jump']>=1
