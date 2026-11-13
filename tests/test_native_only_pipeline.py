import json
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from ego_data.calibration import CameraCalibration
from prepare_native_nokov_mano_input import prepare_native
from native_camera_geometry import project_native_torch, rotate_mano_geometry, canonical_camera_rotation
from fit_mano_sequence import load_input


def test_native_projection_matches_opencv_and_has_finite_gradients():
    rng=np.random.default_rng(42)
    points=rng.normal(size=(2,21,3))*.05;points[...,2]+=.6;points[0,0]=[0,0,.6]
    K=np.array([[505.,0,939.],[0,506.,592.],[0,0,1.]])
    D=np.array([.076,-.014,.0019,.00007])
    for R,t in [(np.eye(3),np.zeros(3)),(Rotation.from_rotvec([.01,.02,-.01]).as_matrix(),np.array([-.1,0,0]))]:
        x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
        pred=project_native_torch(x,*[torch.tensor(v,dtype=torch.float64) for v in (R,t,K,D)])
        expected=cv2.fisheye.projectPoints((points@R.T+t).reshape(-1,1,3),np.zeros(3),np.zeros(3),K,D)[0].reshape(2,21,2)
        np.testing.assert_allclose(pred.detach().numpy(),expected,atol=1e-7)
        pred.sum().backward();assert torch.isfinite(x.grad).all()


def test_dual_native_prepare_does_not_call_rectification(tmp_path):
    dataset=tmp_path/"dataset";(dataset/"calibration").mkdir(parents=True)
    K=np.array([[70.,0,32.],[0,70.,24.],[0,0,1.]])
    for name,offset in [("camera0",0.),("camera1",.05)]:
        T=np.eye(4);T[0,3]=offset
        CameraCalibration(name,name,"KB",(64,48),K,np.zeros(4),T).save(dataset/"calibration"/(name+".json"))
    (dataset/"manifest.json").write_text(json.dumps(dict(dataset_type="normalized_multiview",camera_ids=["camera0","camera1"],image_size=[64,48],calibration=dict(available=True))))
    stamps=np.array([1,33333334],np.int64)
    (dataset/"multiview_frames.csv").write_text("sync_index,reference_timestamp_ns\n0,1\n1,33333334\n")
    points=np.zeros((2,2,21,3),np.float32);points[...,2]=.5;points[0,...,0]=-.02;points[1,...,0]=.03
    np.savez(tmp_path/"aligned.npz",joints_base_m=points,valid=np.ones((2,2,21),bool),reference_timestamp_ns=stamps)
    with patch.object(cv2.fisheye,"stereoRectify",side_effect=AssertionError("no rectification")), patch.object(cv2,"remap",side_effect=AssertionError("no remap")):
        prepare_native(tmp_path/"aligned.npz",dataset,tmp_path/"input.npz",tmp_path/"fusion","dual_fixture")
    arrays=load_input(tmp_path/"input.npz")
    assert not any("rectif" in key for key in arrays)
    assert arrays["positions_reference_camera_m"].shape==(2,2,21,3)
    rows=[json.loads(line) for line in (tmp_path/"fusion/accepted.jsonl").read_text().splitlines()]
    assert all({h["side"] for h in row["hands"]}=={0,1} for row in rows)
    assert not list(tmp_path.glob("*rectif*"))


def test_physical_camera_transform_preserves_both_hand_pivots():
    rng=np.random.default_rng(1);v=rng.normal(size=(778,3)).astype(np.float32)*.03;j=v[:21]
    t=np.array([.05,.02,.5]);R=Rotation.from_rotvec([.03,-.02,.01]).as_matrix()
    for side in (0,1):
        Q=canonical_camera_rotation(R,side)
        nv,nj,nt,_=rotate_mano_geometry(v,j,t,Q)
        np.testing.assert_allclose(nv+nt,(v+t)@Q.T,atol=1e-7)


def test_legacy_native_spelling_forwards_without_rectification(monkeypatch):
    import export_multiview_wilor_training_dataset as legacy
    import export_native_wilor_training_dataset as native
    captured=[]
    monkeypatch.setattr(native,"main",lambda: captured.extend(sys.argv))
    monkeypatch.setattr(sys,"argv",["export","--image-space","native_raw","--fusion","/unused/fusion",
        "--mano-fit","/unused/fit","--dataset","/unused/dataset","--output","/unused/output",
        "--view-filter","complete21","--rectification","/nonexistent/old_rectification.npz"])
    assert legacy.main()==0
    assert captured
    assert "--rectification" not in captured
    assert not any("old_rectification" in part for part in captured)
