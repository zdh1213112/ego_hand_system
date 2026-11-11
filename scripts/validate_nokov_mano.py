#!/usr/bin/env python3
"""Replay every saved MANO frame and verify geometry, attachment, and time contracts."""
import argparse
import json
from pathlib import Path
import numpy as np
from fit_mano_sequence import import_mano, run_model
from mano_conventions import physicalize_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    d=np.load(a.input/'reconstruction.npz');q=np.load(a.input/'mano_parameters.npz')
    root=Path(__file__).resolve().parents[1]
    model=import_mano(root/'third_party/MANO').load(model_path=str(root/'models/mano'),
        is_rhand=True,use_pca=True,num_pca_comps=int(q['pca_components']),batch_size=1,
        flat_hand_mean=bool(q['flat_hand_mean'])).to('cuda')
    n=len(d['frame_no']);max_vertex=max_joint=0.
    assert d['vertices'].shape==(n,778,3) and d['joints'].shape==(n,21,3)
    assert np.isfinite(d['vertices']).all() and np.isfinite(d['joints']).all()
    assert np.array_equal(d['frame_no'],q['frame_no']) and np.all(np.diff(d['frame_no'])==1)
    assert np.allclose(np.diff(d['time_s']),1/float(d['fps']))
    with torch.no_grad():
        for lo in range(0,n,256):
            hi=min(n,lo+256)
            v,j,_=run_model(model,torch.as_tensor(q['betas'],device='cuda').expand(hi-lo,-1),
                torch.as_tensor(q['global_orient'][lo:hi],device='cuda'),torch.as_tensor(q['hand_pose_pca'][lo:hi],device='cuda'),
                torch.as_tensor(q['translation'][lo:hi],device='cuda'))
            v,j,faces=physicalize_geometry(v.cpu().numpy(),j.cpu().numpy(),model.faces,str(q['handedness']))
            max_vertex=max(max_vertex,float(np.max(np.abs(v-d['vertices'][lo:hi]))))
            max_joint=max(max_joint,float(np.max(np.abs(j-d['joints'][lo:hi]))))
    assert max(max_vertex,max_joint)<1e-5
    weights=d['attachment_barycentric'];assert np.all(weights>=0) and np.allclose(weights.sum(-1),1)
    attachment_faces=np.array(model.faces)[d['attachment_face_ids']]
    predicted=(d['vertices'][:,attachment_faces]*weights[None,:,:,None]).sum(2)
    max_attachment=float(np.max(np.abs(predicted-d['predicted_markers'])))
    assert max_attachment<1e-6
    measured=d['markers_world_m'];valid=d['marker_valid']
    assert np.array_equal(np.isfinite(measured).all(-1),valid)
    residual=np.linalg.norm(predicted-measured,axis=-1)
    assert np.allclose(residual,d['marker_error_m'],atol=1e-6,equal_nan=True)
    per_frame=np.nanmedian(residual,axis=1)*1000
    motion=np.linalg.norm(np.diff(d['joints'],axis=0),axis=-1)*1000
    result=dict(status='passed',replayed_frames=n,max_vertex_replay_error_m=max_vertex,max_joint_replay_error_m=max_joint,
        max_attachment_replay_error_m=max_attachment,all_geometry_finite=True,frame_numbers_contiguous=True,
        missing_markers_excluded=True,frame_median_residual_max_mm=float(per_frame.max()),
        frames_median_residual_over_10mm=int((per_frame>10).sum()),
        joint_step_p95_mm=float(np.percentile(motion,95)),joint_step_max_mm=float(motion.max()),
        chunk_boundary_joint_step_p95_mm=float(np.percentile(motion[np.arange(255,len(motion),256)],95)))
    (a.input/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
