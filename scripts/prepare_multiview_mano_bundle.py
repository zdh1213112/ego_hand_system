#!/usr/bin/env python3
"""Estimate fixed dual-hand surface attachments and package independent RGB pixels.

Attachments are candidates estimated from baseline MANO, not measured anatomy.
The configured marker radius is physical; glove/model mismatch remains diagnostic.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from multiview_mano_constraints import sha256,CAMERAS


def estimate_attachments(vertices,faces,target,valid,radius):
    tri=vertices[:,faces]
    normals=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-10)
    centres=tri.mean(2)+normals*radius
    triangles=[];weights=[];errors=[]
    for marker in range(21):
        good=valid[:,marker]&np.isfinite(target[:,marker]).all(-1)
        if good.sum()<8:raise ValueError(f'Insufficient calibration frames for marker {marker}')
        distances=np.linalg.norm(centres[good]-target[good,marker,None],axis=-1)
        index=int(np.argmin(np.median(distances,axis=0)))
        corners=tri[good,index];n=normals[good,index];y=target[good,marker]
        def bary(x):
            z=np.r_[x,0.];z=np.exp(z-z.max());return z/z.sum()
        def residual(x):return ((corners*bary(x)[None,:,None]).sum(1)+n*radius-y).ravel()
        result=least_squares(residual,np.zeros(2),bounds=(-8,8),loss='soft_l1',f_scale=.006)
        triangles.append(faces[index]);weights.append(bary(result.x));errors.append(np.median(np.linalg.norm(residual(result.x).reshape(-1,3),axis=-1))*1000)
    return np.asarray(triangles),np.asarray(weights),errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','baseline-fit','rgb-correspondences','rgb-alignment','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--marker-diameter-mm',type=float,required=True)
    a=p.parse_args()
    if not 0<a.marker_diameter_mm<=20:p.error('marker diameter must be in (0,20] mm')
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    with np.load(a.input,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    if str(data['image_space'])!='native_raw':raise ValueError('require native input')
    report=json.loads(a.rgb_alignment.read_text())
    if report['status']!='accepted':raise ValueError('RGB alignment must be accepted')
    cameras=json.loads(str(data['camera_calibrations_json']))
    if cameras!=report['camera_calibrations']:raise ValueError('RGB/input calibration mismatch')
    rows=json.loads(a.rgb_correspondences.read_text())
    n=len(data['pair_indices']);lookup={int(pair):i for i,pair in enumerate(data['pair_indices'])}
    pixels=np.full((2,n,4,21,2),np.nan,np.float32);confidence=np.zeros((2,n,4,21),np.float32)
    for r in rows:
        if r['frame'] not in lookup:continue
        key=(int(r['side']),lookup[r['frame']],CAMERAS.index(r['camera']),int(r['marker']))
        if confidence[key]>0:raise ValueError('Duplicate camera/hand/frame/marker RGB observation')
        point=np.asarray(r['pixel'],float)
        if point.shape!=(2,) or not np.isfinite(point).all():raise ValueError('invalid actual RGB pixel')
        pixels[key]=point;confidence[key]=1.
    train=((data['pair_indices']//int(report['frame_stride']))%5)!=0
    triangles=[];barycentric=[];diagnostics={};baseline_hash={}
    radius=a.marker_diameter_mm/2000
    for side in (0,1):
        path=a.baseline_fit/f'track_{side}.npz';baseline_hash[str(side)]=sha256(path)
        with np.load(path,allow_pickle=False) as t:
            indices={int(v):i for i,v in enumerate(t['pair_indices'])}
            usable=np.array([i for i,pair in enumerate(data['pair_indices']) if train[i] and int(pair) in indices and data['valid'][side,i].all()])
            if len(usable)<8:raise ValueError('Need at least eight complete calibration frames')
            usable=usable[np.unique(np.linspace(0,len(usable)-1,min(128,len(usable))).astype(int))]
            ti=np.array([indices[int(data['pair_indices'][i])] for i in usable]);v=t['vertices'][ti].astype(float);faces=t['faces'].astype(int)
            target=data['positions_reference_camera_m'][side,usable].astype(float)*np.array([-1.,1.,1.] if side==0 else [1.,1.,1.])
            centred=v-t['joints'][ti,:1]
            f=centred[:,faces];volume=np.sum(f[:,:,0]*np.cross(f[:,:,1],f[:,:,2]),axis=(1,2))
            if np.median(volume)<0:faces=faces[:,[0,2,1]]
            tr,b,err=estimate_attachments(v,faces,target,data['valid'][side,usable],radius)
            triangles.append(tr);barycentric.append(b);diagnostics[str(side)]=dict(calibration_frames=data['pair_indices'][usable].tolist(),per_marker_median_mm=err)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,schema='native_multiview_mano_v1',observation_source='independent_rgb_marker_detections',
        baseline_fit_sha256_json=json.dumps(baseline_hash),input_path=str(a.input.resolve()),input_sha256=sha256(a.input),pair_indices=data['pair_indices'],camera_ids=np.array(CAMERAS),
        camera_calibrations_json=data['camera_calibrations_json'],reference_camera=data['reference_camera'],
        triangles=np.asarray(triangles),barycentric=np.asarray(barycentric),offset_m=np.full((2,21),radius),
        pixels=pixels,confidence=confidence,train_frame=train,attachment_status='estimated_from_baseline_not_independently_calibrated')
    metadata=dict(stage='multiview_mano_constraints',attachment_status='estimated_from_baseline_not_independently_calibrated',marker_diameter_mm=a.marker_diameter_mm,
        camera_ids=list(CAMERAS),train_rgb_counts=confidence[:,train].sum((0,1,3)).astype(int).tolist(),heldout_rgb_counts=confidence[:,~train].sum((0,1,3)).astype(int).tolist(),
        baseline_sha256=baseline_hash,attachment_diagnostics=diagnostics,rgb_correspondences=str(a.rgb_correspondences.resolve()),rgb_correspondences_sha256=sha256(a.rgb_correspondences),
        limitation='RGB holdout excludes refinement and attachment calibration frames; initial spatial alignment and baseline shape come from this recording. Glove thickness is not measured.')
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
