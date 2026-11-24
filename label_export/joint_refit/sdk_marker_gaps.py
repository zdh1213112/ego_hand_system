"""Estimate one absent optical finger marker from its recorded SDK segment and a calibrated local offset."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sdk_directions import FINGERS, head_to_camera, load_context, rigid_motion, sdk_links

PROFILE_FORMAT = 'nokov_missing_finger_marker_v1'


def make_profile(context, skeleton, selection, calibrated, output):
    with np.load(skeleton) as z:
        if z['geometry_space'].item()!='physical_camera0':
            raise ValueError('Expected physical camera0')
        raw,xyz,quat,names,parents,stamp = [z[k] for k in ('marker24_cam0_m','positions_cam0_m','sdk_quaternions_xyzw',
                                                           'names','parents','reference_timestamp_ns')]
    camera = head_to_camera(load_context(context),stamp)
    selected=json.loads(selection.read_text())['report']['hands']
    predicted=np.full(raw[:,:,4:].shape,np.nan,np.float32)
    confidence=np.zeros((2,20),np.float32)
    report={}
    for side,hand in enumerate(('left','right')):
        path=calibrated/f'{hand}_pilot.npz'
        if not path.is_file():
            continue
        parent,child=sdk_links(names[side],parents[side],hand)
        nodes=np.column_stack([parent.reshape(5,3),child.reshape(5,3)[:,-1]]).ravel()
        cal=np.array(selected[hand]['calibration_sync'],dtype=int)
        with np.load(path) as z:
            marker_names=z['marker_names'].tolist()[4:]
        if marker_names!=[f'Finger{finger}{k}' for finger in FINGERS for k in range(1,5)]:
            raise ValueError(f'{hand} marker identities do not match the SDK phalanx mapping')
        segments=[]
        for k,node in enumerate(nodes):
            q=quat[side,:,node]
            ok=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>.5)&np.isfinite(camera).all((1,2))
            mat=np.full_like(camera,np.nan)
            mat[ok]=camera[ok]@Rotation.from_quat(q[ok]).as_matrix()
            anchor=xyz[side,:,node]
            offset=np.einsum('fji,fj->fi',mat[cal],raw[side,cal,4+k]-anchor[cal])
            delta=offset[np.isfinite(offset).all(1)]
            rec=dict(marker=marker_names[k],sdk_segment=str(names[side,node]),calibration_observations=len(delta))
            segments.append(rec)
            if len(delta)<4:
                continue
            local=np.median(delta,axis=0)
            leave_one=np.array([np.linalg.norm(x-np.median(np.delete(delta,i,axis=0),axis=0)) for i,x in enumerate(delta)])
            sigma=float(np.median(leave_one))
            # An inferred centre gets at most half the weight of one directly tracked marker.
            confidence[side,k]=.5/(1.+(sigma/.004)**2)
            predicted[side,:,k]=anchor+mat@local
            rec.update(local_offset_mm=(local*1000).tolist(),leave_one_calibration_median_mm=1000*sigma,weight=float(confidence[side,k]))
        report[hand]=dict(calibration_sync=cal.tolist(),markers=segments)
    if not report:
        raise ValueError('No calibrated hands')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'marker_gaps.npz',format=PROFILE_FORMAT,sdk_marker_cam0_m=predicted,
                        confidence=confidence,raw_marker24_cam0_m=raw.astype(np.float32))
    (output/'marker_gaps.json').write_text(json.dumps(dict(source=str(skeleton.resolve()),hands=report),indent=2,allow_nan=False)+'\n')


class MarkerGapTargets:
    def __init__(self,profile,side,syncs,raw_canonical,aligned,device):
        mirror=np.array([-1.,1.,1.] if side==0 else [1.,1.,1.])
        with np.load(profile) as z:
            if z['format'].item()!=PROFILE_FORMAT:
                raise ValueError('Unexpected missing-marker profile')
            all_predictions=z['sdk_marker_cam0_m'][side]
            if np.any(syncs<0) or np.any(syncs>=len(all_predictions)):
                raise ValueError('Sync indices fall outside the SDK recording')
            pred=all_predictions[syncs]*mirror
            raw=z['raw_marker24_cam0_m'][side,syncs]*mirror
            conf=z['confidence'][side]
        seen=np.isfinite(raw_canonical).all(-1)
        if not seen.any() or not np.allclose(raw[seen],raw_canonical[seen],rtol=0,atol=1e-6):
            raise ValueError('SDK profile and measured optical markers do not match')
        observed=np.isfinite(aligned).all(-1)[:,4:].reshape(-1,5,4)
        missing=((~observed)&(observed.sum(2,keepdims=True)==3)).reshape(-1,20)
        use=missing & np.isfinite(pred).all(-1) & (conf>0)
        for i in np.flatnonzero(use.any(1)):
            good=np.isfinite(aligned[i]).all(1)&seen[i]
            x=raw_canonical[i,good]
            if len(x)<3 or np.linalg.matrix_rank(x-x.mean(0),tol=1e-6)<2:
                use[i]=False
                continue
            if not np.array_equal(x,aligned[i,good]):
                rot,delta=rigid_motion(x,aligned[i,good])
                pred[i,use[i]]=pred[i,use[i]]@rot.T+delta
        rows,cols=np.where(use)
        self.rows=torch.as_tensor(rows,device=device)
        self.markers=torch.as_tensor(cols+4,device=device)
        self.points=torch.as_tensor(pred[rows,cols],device=device,dtype=torch.float32)
        self.weights=torch.as_tensor(conf[cols],device=device,dtype=torch.float32)
        self.count=len(rows)

    def penalty(self,predicted,observed_weight_sum):
        delta=predicted[self.rows,self.markers]-self.points
        return ((torch.sqrt(1.+delta.square().sum(-1)/.004**2)-1.)*self.weights).sum()/observed_weight_sum


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('context','skeleton','selection','calibrated','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    make_profile(**vars(ap.parse_args()))
