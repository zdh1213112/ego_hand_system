import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('native_fourview_alignment',Path(__file__).resolve().parents[1]/'scripts/native_fourview_alignment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_shared_se3_recovers_four_camera_observations():
    rng=np.random.default_rng(41)
    points=rng.uniform(-.08,.08,(2,12,21,3));points[...,2]+=.6
    calibrations={}
    for i in range(4):
        T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0,(i-1)*.15,0]).as_matrix();T[0,3]=i*.04
        calibrations[f'camera{i}']=dict(K=[[505.,0,939],[0,506,592],[0,0,1]],distortion=[.076,-.014,.0019,.00007],T_base_camera=T.tolist())
    target=np.array([.01,-.015,.008,.003,-.002,.004])
    rows=[]
    for camera,cal in calibrations.items():
        for side in (0,1):
            for frame in range(12):
                pixels,_=module.project_native(points[side,frame],target,cal,calibrations['camera0'])
                rows.extend(dict(camera=camera,side=side,frame=frame,marker=j,pixel=p.tolist()) for j,p in enumerate(pixels))
    train=[r for r in rows if r['frame']%4!=0];held=[r for r in rows if r['frame']%4==0]
    recovered=module.fit_shared(train,points,calibrations)
    np.testing.assert_allclose(recovered,target,atol=1e-7)
    metrics={n:module.errors([r for r in held if r['camera']==n],points,recovered,calibrations) for n in calibrations}
    assert module.geometry_pass(metrics)
    metrics['camera3']['p95_px']=31
    assert not module.geometry_pass(metrics)
    metrics['camera3']={'count':0}
    assert not module.geometry_pass(metrics)


def test_label_entry_forwards_four_cameras_and_rejects_candidates(tmp_path):
    import subprocess,sys,json
    script=Path(__file__).resolve().parents[1]/'scripts/run_nokov_native_labels.py'
    dataset=tmp_path/'take/ego/normalized_multiview';dataset.mkdir(parents=True)
    (dataset/'multiview_frames.csv').write_text('sync_index\n0\n')
    aligned=tmp_path/'aligned.npz'
    cmd=[sys.executable,'-B',str(script),'--processed-session',str(tmp_path/'take'),'--aligned-nokov',str(aligned),'--output',str(tmp_path/'labels'),'--cameras','camera0','camera1','camera2','camera3','--dry-run']
    np.savez(aligned,valid=np.ones((2,1,21),bool),alignment_status='accepted')
    result=subprocess.run(cmd,text=True,capture_output=True,check=True)
    entries=[json.loads(line) for line in result.stdout.splitlines()]
    prepare=entries[0]['command'];assert prepare[prepare.index('--cameras')+1:]==['camera0','camera1','camera2','camera3']
    assert not (tmp_path/'labels').exists()
    np.savez(aligned,valid=np.ones((2,1,21),bool),alignment_status='needs_review')
    result=subprocess.run(cmd,text=True,capture_output=True)
    assert result.returncode!=0 and 'Refusing unaccepted/candidate' in result.stderr
