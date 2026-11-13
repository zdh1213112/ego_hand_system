#!/usr/bin/env python3
"""Align the recorded NOKOV hand marker sets (left, right or both) using native RGB marker observations.

Marker sets are found automatically (names starting with left_hand / right_hand); a hand that was not
recorded is stored as unobserved (NaN) so downstream archives keep the [left, right] layout.

Requires a previously calibrated head-camera transform as initialization.
No WiLoR prediction, image rectification or VIO is used as ground truth.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'nokov_ego_hand_validation/integration/pipeline'))
from align_nokov_cam0 import load_rigid,load_markers,interpolate_points,EDGES
from refine_dark_markers_cam0 import detect_dark_markers,fit,error_summary,read_frames,project
from calibrate_nokov_camera_multiframe import associate_with_shift_vote


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset','nokov','sync','seed-alignment','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--left-markerset',default='auto',help="'auto', 'none' or a marker-set name");p.add_argument('--right-markerset',default='auto',help="'auto', 'none' or a marker-set name")
    p.add_argument('--max-calibration-frames',type=int,default=180)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'dual_alignment.json').exists():raise FileExistsError('use a new alignment output')
    assets=json.loads((a.nokov/'asset_descriptions.json').read_text())
    expected=['WristM','WristIn','WristOut','HandOffset']+[f'Finger{finger}{j}' for finger in ['Thumb','Index','Middle','Ring','Pinky'] for j in range(1,5)]
    def pick(value,prefix):
        if value=='none':return None
        if value!='auto':return value
        found=[k for k in assets['marker_sets'] if k.startswith(prefix)]
        if len(found)>1:raise ValueError(f'several {prefix} marker sets {found}; pass the name explicitly')
        return found[0] if found else None
    markersets=[pick(a.left_markerset,'left_hand'),pick(a.right_markerset,'right_hand')];present=[n is not None for n in markersets]
    if not any(present):raise ValueError('no hand marker set in this recording')
    for name in markersets:
        if name is not None and assets['marker_sets'].get(name)!=expected:raise ValueError('unexpected physical marker order: '+name)
    if set(assets['rigid_bodies'].values())!={'orbbec_ego4'}:raise ValueError('expected only the recorded head rigid body')
    with (a.dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
    sync=json.loads(a.sync.read_text());mapping=sync['time_mapping']
    stamps=np.array([int(r['reference_timestamp_ns']) for r in rows],np.int64)
    ego=np.array([int(r['camera0_timestamp_ns']) for r in rows],np.int64)
    target=mapping['nokov_origin_timestamp_raw']+((ego-mapping['ego_origin_timestamp_ns'])*1e-9*mapping.get('a',1.)+mapping['b_s'])/mapping['nokov_seconds_per_timestamp_unit']
    rt,head_xyz,quats=load_rigid(a.nokov/'nokov_rigid_bodies.csv')
    head,valid=interpolate_points(rt,head_xyz,target)
    rotations=np.full((len(rows),3,3),np.nan);good=valid&(target>=rt[0])&(target<=rt[-1])
    rotations[good]=Slerp(rt,Rotation.from_quat(quats))(target[good]).as_matrix()
    hands=[]
    for name in markersets:
        if name is None:hands.append(np.full((len(rows),24,3),np.nan));continue
        t,world=load_markers(a.nokov/'nokov_markers.csv',name);world,_=interpolate_points(t,world,target)
        hands.append(np.einsum('fji,fkj->fki',rotations,world-head[:,None]))
    points=np.concatenate(hands,axis=1)
    seed=np.asarray(json.loads(a.seed_alignment.read_text())['T_cam0_head'],float)
    original=points@seed[:3,:3].T+seed[:3,3]
    cal=json.loads((a.dataset/'calibration/camera0.json').read_text());K=np.asarray(cal['K']);D=np.asarray(cal['distortion'])
    stride=max(1,len(rows)//a.max_calibration_frames);indices=list(range(0,len(rows),stride))
    images,_=read_frames(a.dataset,indices)
    def correspondences(correction,second=False):
        records=[]
        for frame in indices:
            if second and (frame//stride)%5==0:continue
            per_frame=[]
            for side in (0,1):
                if not present[side]:continue
                xyz=original[frame,side*24:(side+1)*24]
                finite=np.isfinite(xyz).all(-1)&(xyz[:,2]>.03);ids=np.flatnonzero(finite)
                if len(ids)<8:continue
                predicted=project(xyz[ids],correction,K,D)
                detected=detect_dark_markers(images[frame],predicted,margin=80 if second else 125)
                if not len(detected):continue
                ii,jj,distance,shift=associate_with_shift_vote(predicted,detected,11 if second else 13,30 if second else 150)
                if len(ii)<8 or (second and np.linalg.norm(shift)>20):continue
                for i,j,d in zip(ii,jj,distance):per_frame.append(dict(frame=frame,marker=int(ids[i]+side*24),side=side,pixel=detected[j].tolist(),distance=float(d)))
            used=[]
            for record in sorted(per_frame,key=lambda r:r['distance']):
                if any(np.linalg.norm(np.asarray(record['pixel'])-x)<2.5 for x in used):continue
                used.append(np.asarray(record['pixel']));records.append(record)
        return records
    records=correspondences(np.zeros(6));train=[r for r in records if (r['frame']//stride)%5!=0];held=[r for r in records if (r['frame']//stride)%5==0]
    if len(train)<100 or len(held)<50:raise RuntimeError(f'insufficient real marker correspondences: train={len(train)}, held={len(held)}')
    correction=fit(train,original,K,D,np.zeros(6));refined=correspondences(correction,True)
    if len(refined)>=100:correction=fit(refined,original,K,D,correction)
    C=np.eye(4);C[:3,:3]=Rotation.from_rotvec(correction[:3]).as_matrix();C[:3,3]=correction[3:];T=C@seed
    metrics={name:error_summary(data,original,correction,K,D) for name,data in [('train',refined or train),('heldout',held)]}
    metrics['heldout_by_side']={str(side):error_summary([x for x in held if x['side']==side],original,correction,K,D) for side in (0,1) if present[side]}
    geometry_ok=all(v.get('count',0)>=50 and v.get('median_px',1e9)<=12 for v in metrics['heldout_by_side'].values())
    sync_ok=sync.get('status')=='ok' and sync.get('confidence')=='strong'
    status='accepted' if geometry_ok and sync_ok else 'needs_review'
    summary=dict(status=status,geometry_ok=geometry_ok,sync_ok=sync_ok,time_mapping=mapping,T_cam0_head=T.tolist(),seed_alignment=str(a.seed_alignment.resolve()),
        source_nokov=str(a.nokov.resolve()),source_dataset=str(a.dataset.resolve()),marker_sets=markersets,hands_present=present,
        metrics=metrics,train_correspondences=len(train),heldout_correspondences=len(held),refined_train_correspondences=len(refined),
        calibration_method='shared SE3 correction from real native RGB marker spots; heldout initial associations excluded from fitting',
        camera_calibration_sha256=hashlib.sha256((a.dataset/'calibration/camera0.json').read_bytes()).hexdigest())
    (a.output/'dual_alignment.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'dual_marker_correspondences.json').write_text(json.dumps(records,indent=2)+'\n')
    physical=points@T[:3,:3].T+T[:3,3]
    selected=np.array([0]+list(range(4,24)))
    joints=physical.reshape(len(rows),2,24,3).transpose(1,0,2,3)[:,:,selected]
    B=np.asarray(cal['T_base_camera']);joints=joints@B[:3,:3].T+B[:3,3]
    observed=np.isfinite(joints).all(-1)
    filename='aligned_nokov21.npz' if status=='accepted' else 'candidate_aligned_nokov21.npz'
    np.savez_compressed(a.output/filename,joints_base_m=joints.astype(np.float32),valid=observed,reference_timestamp_ns=stamps,
        source_nokov=str(a.nokov.resolve()),marker_sets=np.array([n or '' for n in markersets]),hands_present=np.array(present),alignment_status=status)
    panels=[]
    for frame in indices[::max(1,len(indices)//6)][:6]:
        image=images[frame].copy()
        for side,color in [(0,(60,240,60)),(1,(0,180,255))]:
            if not present[side]:continue
            xyz=physical[frame,side*24:(side+1)*24][selected];valid=np.isfinite(xyz).all(-1)&(xyz[:,2]>.03)
            safe=np.where(valid[:,None],xyz,[0,0,1]);px=cv2.fisheye.projectPoints(safe[:,None],np.zeros(3),np.zeros(3),K,D)[0][:,0]
            valid&=(px[:,0]>=0)&(px[:,0]<image.shape[1])&(px[:,1]>=0)&(px[:,1]<image.shape[0])
            for i,j in EDGES:
                if valid[i] and valid[j]:cv2.line(image,tuple(np.rint(px[i]).astype(int)),tuple(np.rint(px[j]).astype(int)),color,3)
            for point in px[valid]:cv2.circle(image,tuple(np.rint(point).astype(int)),5,color,-1)
        cv2.putText(image,f'{status} sync {frame}',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2);panels.append(cv2.resize(image,(768,480)))
    cv2.imwrite(str(a.output/'dual_marker_preview.jpg'),np.vstack(panels))
    print(json.dumps(summary,indent=2))
    if status!='accepted':raise SystemExit(2)

if __name__=='__main__':main()
