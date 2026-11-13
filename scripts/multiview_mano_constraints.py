"""Independent four-view RGB constraints on physical surface marker centres."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from native_camera_geometry import project_native_torch

CAMERAS=('camera0','camera1','camera2','camera3')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surface_marker_centres(vertices, triangles, barycentric, offset_m):
    tri=vertices[:,triangles]
    normals=torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1)
    normals=normals/torch.linalg.vector_norm(normals,dim=-1,keepdim=True).clamp_min(1e-10)
    return (tri*barycentric[None,:,:,None]).sum(2)+normals*offset_m[None,:,None]


def load_bundle(path, data):
    with np.load(path,allow_pickle=False) as f:b={k:f[k].copy() for k in f.files}
    if str(b.get('schema',''))!='native_multiview_mano_v1' or str(b.get('observation_source',''))!='independent_rgb_marker_detections':
        raise ValueError('Require independent RGB marker detections, not NOKOV-projected pixels')
    if str(data.get('image_space',''))!='native_raw':raise ValueError('Four-view MANO requires native_raw input')
    np.testing.assert_array_equal(b['pair_indices'],data['pair_indices'])
    if str(b['input_sha256'])!=sha256(str(b['input_path'])):raise ValueError('Constraint source input changed')
    # Compare actual input contents, including physical side/point identities.
    with np.load(str(b['input_path']),allow_pickle=False) as original:
        for key in ['pair_indices','positions_reference_camera_m','valid','track_ids','handedness']:
            np.testing.assert_equal(original[key],data[key])
    if json.loads(str(b['camera_calibrations_json']))!=json.loads(str(data['camera_calibrations_json'])):
        raise ValueError('Constraint camera calibration mismatch')
    cal=json.loads(str(b['camera_calibrations_json']))
    if not set(CAMERAS).issubset(cal):raise ValueError('Four calibrated cameras required')
    if list(b['camera_ids'])!=list(CAMERAS):raise ValueError('Unexpected camera order')
    if b['triangles'].shape!=(2,21,3) or b['triangles'].dtype.kind not in 'iu' or np.any((b['triangles']<0)|(b['triangles']>=778)):
        raise ValueError('Expected two sets of 21 MANO triangle attachments')
    weights=b['barycentric'];offset=b['offset_m']
    if weights.shape!=(2,21,3) or not np.isfinite(weights).all() or np.any(weights<0) or not np.allclose(weights.sum(-1),1):raise ValueError('Invalid barycentric weights')
    if offset.shape!=(2,21) or not np.isfinite(offset).all() or np.any((offset<0)|(offset>.01)):raise ValueError('Invalid marker centre offsets')
    shape=(2,len(data['pair_indices']),4,21)
    if b['pixels'].shape!=shape+(2,) or b['confidence'].shape!=shape or b['train_frame'].shape!=(shape[1],):raise ValueError('RGB observations have incompatible shape')
    if b['train_frame'].dtype.kind!='b':raise ValueError('Expected boolean training-frame mask')
    confidence=b['confidence']
    if not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):raise ValueError('Invalid observation confidence')
    if np.any((confidence>0)&~np.isfinite(b['pixels']).all(-1)):raise ValueError('Valid observation pixels must be finite')
    for c in range(4):
        if np.count_nonzero(confidence[:,b['train_frame'],c])<8:raise ValueError(f'No sufficient independent training evidence for {CAMERAS[c]}')
    for side in (0,1):
        if np.count_nonzero(confidence[side,b['train_frame']])<8:raise ValueError('Both hands need RGB training evidence')
    return b


class MultiviewMarkerConstraints:
    def __init__(self,bundle,side,pairs,device):
        self.side=side
        self.triangles=torch.as_tensor(bundle['triangles'][side],dtype=torch.long,device=device)
        self.barycentric=torch.as_tensor(bundle['barycentric'][side],dtype=torch.float32,device=device)
        self.offset=torch.as_tensor(bundle['offset_m'][side],dtype=torch.float32,device=device)
        self.pixels=torch.as_tensor(np.nan_to_num(bundle['pixels'][side,pairs]),dtype=torch.float32,device=device)
        self.confidence=torch.as_tensor(bundle['confidence'][side,pairs],dtype=torch.float32,device=device)
        self.train=torch.as_tensor(bundle['train_frame'][pairs],dtype=torch.bool,device=device)
        self.sign=torch.tensor([-1.,1.,1.] if side==0 else [1.,1.,1.],device=device)
        cal=json.loads(str(bundle['camera_calibrations_json']));ref=cal[str(bundle['reference_camera'])]
        B=np.array(ref['T_base_camera']);self.views=[]
        for name in CAMERAS:
            c=cal[name];T=np.array(c['T_base_camera'])
            R=T[:3,:3].T@B[:3,:3];t=T[:3,:3].T@(B[:3,3]-T[:3,3])
            self.views.append({k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in dict(rotation=R,translation=t,K=c['K'],distortion=c['distortion']).items()})

    def centres(self,vertices):
        return surface_marker_centres(vertices,self.triangles,self.barycentric,self.offset)

    def pixel_errors(self,centres,ids):
        physical=centres*self.sign
        predictions=torch.stack([project_native_torch(physical,**view) for view in self.views],dim=1)
        return predictions-self.pixels[ids]

    def loss(self,centres,ids,sigma_px):
        residual=self.pixel_errors(centres,ids)/sigma_px
        confidence=self.confidence[ids]*self.train[ids,None,None]
        # Equal per-view reduction prevents the front cameras dominating merely
        # because they supply more detections. Missing/held-out points have zero weight.
        losses=[]
        for c in range(4):
            w=confidence[:,c];norm2=residual[:,c].square().sum(-1)
            robust=torch.sqrt(1+norm2)-1
            losses.append((robust*w).sum()/w.sum().clamp_min(1))
        return torch.stack(losses).mean()

    def diagnostics(self,centres):
        ids=torch.arange(len(centres),device=centres.device)
        e=torch.linalg.vector_norm(self.pixel_errors(centres,ids),dim=-1)
        out={}
        for split,frames in [('train',self.train),('heldout',~self.train)]:
            out[split]={}
            for c,name in enumerate(CAMERAS):
                x=e[:,c][(self.confidence[:,c]>0)&frames[:,None]].detach().cpu().numpy()
                out[split][name]=dict(count=len(x),median_px=float(np.median(x)) if len(x) else None,p95_px=float(np.percentile(x,95)) if len(x) else None)
        return out


def safe_refinement_mask(base_joints, joints, base_vertices, vertices, target, valid,
                         rgb_before, rgb_after, confidence, train, pairs, max_step_m=.003):
    """Accept bounded, RGB-supported refinements; otherwise retain baseline params.

    Distances here are changes/error checks against baseline, not anatomy truth.
    Rejection counts overlap. Adjacent accepted/reverted states are checked again.
    """
    displacement = joints - base_joints
    joint_step = np.linalg.norm(displacement, axis=-1).max(-1)
    mesh_step = np.linalg.norm(vertices-base_vertices, axis=-1).max(-1)
    old_error = np.where(valid, np.linalg.norm(base_joints-target, axis=-1), np.nan)
    new_error = np.where(valid, np.linalg.norm(joints-target, axis=-1), np.nan)
    weights = (confidence > 0) & train[:, None, None]
    before = np.where(weights, rgb_before, np.nan).reshape(len(joints), -1)
    after = np.where(weights, rgb_after, np.nan).reshape(len(joints), -1)
    # Explicit loops avoid all-NaN percentile warnings for unobserved frames.
    rgb_improved = np.zeros(len(joints), bool)
    geometry_ok = np.zeros(len(joints), bool)
    for i in range(len(joints)):
        if weights[i].any():
            rgb_improved[i] = np.nanmedian(after[i]) + .01 < np.nanmedian(before[i])
        if valid[i].any():
            geometry_ok[i] = (np.nanmedian(new_error[i]) <= np.nanmedian(old_error[i])+.0001
                              and np.nanpercentile(new_error[i],95) <= np.nanpercentile(old_error[i],95)+.0002)
    reasons = {
        'insufficient_rgb': (weights.sum((1,2)) < 6) | (weights.any(-1).sum(-1) < 2),
        'step_limit': ~np.isfinite(joint_step+mesh_step) | (joint_step > max_step_m) | (mesh_step > max_step_m),
        'nokov_error_worse': ~geometry_ok,
        'rgb_not_improved': ~rgb_improved,
    }
    accept = ~np.logical_or.reduce(list(reasons.values()))
    adjacent = np.diff(pairs) == 1
    temporal_rejected = np.zeros(len(joints), bool)
    while True:
        delta = np.where(accept[:,None,None], displacement, 0.)
        bad = adjacent & (np.linalg.norm(np.diff(delta,axis=0),axis=-1).max(-1) > .001)
        reject = np.zeros_like(accept);reject[:-1] |= bad;reject[1:] |= bad
        reject &= accept
        if not reject.any():
            break
        temporal_rejected |= reject;accept[reject] = False
    reasons['temporal_correction_jump'] = temporal_rejected
    return accept, {k:int(v.sum()) for k,v in reasons.items()}
