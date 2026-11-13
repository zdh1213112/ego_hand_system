#!/usr/bin/env python3
"""Refine a shared NOKOV-camera SE3 with independent native RGB spots in four views.

Input is a verified aligned archive in [left, right] layout; a hand that was not recorded is all-NaN and skipped. This optimizes spatial alignment,
not MANO articulation. Camera intrinsics and inter-camera extrinsics stay fixed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'nokov_ego_hand_validation/integration/pipeline'))
from refine_dark_markers_cam0 import detect_dark_markers
from calibrate_nokov_camera_multiframe import associate_with_shift_vote


def project_native(points, correction, calibration, reference):
    corrected=points@Rotation.from_rotvec(correction[:3]).as_matrix().T+correction[3:]
    B=np.asarray(reference['T_base_camera']);T=np.asarray(calibration['T_base_camera'])
    base=corrected@B[:3,:3].T+B[:3,3]
    camera=(base-T[:3,3])@T[:3,:3]
    pixels=cv2.fisheye.projectPoints(camera.reshape(-1,1,3),np.zeros(3),np.zeros(3),
        np.asarray(calibration['K'],float),np.asarray(calibration['distortion'],float))[0][:,0]
    return pixels.reshape(*camera.shape[:-1],2),camera[...,2]


def fit_shared(records, points, calibrations, start=None):
    if not records:raise ValueError('no independent RGB observations')
    groups=[]
    for name,cal in calibrations.items():
        rows=[r for r in records if r['camera']==name]
        if not rows:continue
        xyz=np.array([points[r['side'],r['frame'],r['marker']] for r in rows])
        pixels=np.array([r['pixel'] for r in rows])
        groups.append((cal,xyz,pixels))
    def residual(correction):
        return np.concatenate([(project_native(xyz,correction,cal,calibrations['camera0'])[0]-pixels).ravel() for cal,xyz,pixels in groups])
    bounds=np.array([.1,.1,.1,.05,.05,.05])
    return least_squares(residual,np.zeros(6) if start is None else start,bounds=(-bounds,bounds),loss='soft_l1',f_scale=4,max_nfev=150).x


def errors(records, points, correction, calibrations):
    values=[]
    for row in records:
        px,_=project_native(points[row['side'],row['frame'],row['marker']][None],correction,calibrations[row['camera']],calibrations['camera0'])
        values.append(float(np.linalg.norm(px[0]-row['pixel'])))
    if not values:return dict(count=0)
    x=np.array(values)
    return dict(count=len(x),median_px=float(np.median(x)),p95_px=float(np.percentile(x,95)),within_20px_ratio=float(np.mean(x<=20)))


def geometry_pass(metrics):
    return all(v.get('count',0)>=20 and v.get('median_px',1e9)<=12 and v.get('p95_px',1e9)<=30 for v in metrics.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','aligned-nokov','seed-alignment','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-calibration-frames',type=int,default=180)
    a=p.parse_args()
    if a.max_calibration_frames<10:p.error('at least ten calibration frames required')
    if a.output.exists():raise FileExistsError(a.output)
    names=['camera0','camera1','camera2','camera3']
    calibrations={n:json.loads((a.dataset/'calibration'/f'{n}.json').read_text()) for n in names}
    if any(c['model']!='KB' for c in calibrations.values()):raise ValueError('all cameras must be KB')
    seed=json.loads(a.seed_alignment.read_text())
    if seed.get('status')!='accepted' or not seed.get('sync_ok'):raise ValueError('require accepted, strongly synchronized seed alignment')
    if Path(seed['source_dataset']).resolve()!=a.dataset.resolve():raise ValueError('seed must belong to this dataset')
    with (a.dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
    with np.load(a.aligned_nokov,allow_pickle=False) as archive:payload={k:archive[k] for k in archive.files}
    np.testing.assert_array_equal(payload['reference_timestamp_ns'],[int(r['reference_timestamp_ns']) for r in rows])
    if str(payload.get('alignment_status',''))!='accepted':raise ValueError('require accepted aligned archive')
    if Path(str(payload['source_nokov'])).resolve()!=Path(seed['source_nokov']).resolve():raise ValueError('seed/archive source mismatch')
    B=np.asarray(calibrations['camera0']['T_base_camera']);points=(payload['joints_base_m'].astype(float)-B[:3,3])@B[:3,:3]
    if points.shape!=(2,len(rows),21,3):raise ValueError('expected [left, right] hand slots')
    present=[bool(x) for x in payload['hands_present']] if 'hands_present' in payload else [bool(np.isfinite(points[s]).all(-1).sum(-1).max()>=8) for s in (0,1)]
    if not any(present):raise ValueError('no observed hand in aligned archive')
    stride=max(1,len(rows)//a.max_calibration_frames);indices=list(range(0,len(rows),stride))
    records=[];coverage={};previews={}
    for name in names:
        targets={int(rows[i][name+'_frame_index']):i for i in indices}
        cap=cv2.VideoCapture(str(a.dataset/'cameras'/name/'video.mp4'));visible=[0,0]
        try:
            for source in range(max(targets)+1):
                ok,image=cap.read()
                if not ok:raise RuntimeError(f'{name} ended at {source}')
                if source not in targets:continue
                frame=targets[source];h,w=image.shape[:2];candidates=[]
                if len(previews.setdefault(name,{}))<6 and (frame//stride)%15==0:previews[name][frame]=image.copy()
                # Reject synchronization rows outside the normalizer's tolerance.
                if abs(int(rows[frame][name+'_timestamp_ns'])-int(rows[frame]['camera0_timestamp_ns']))>1500000:continue
                for side in (0,1):
                    if not present[side]:continue
                    px,z=project_native(points[side,frame],np.zeros(6),calibrations[name],calibrations['camera0'])
                    valid=np.isfinite(px).all(-1)&(z>.03)&(px[:,0]>=0)&(px[:,0]<w)&(px[:,1]>=0)&(px[:,1]<h)
                    ids=np.flatnonzero(valid);visible[side]+=int(len(ids)>=8)
                    if len(ids)<8:continue
                    detected=detect_dark_markers(image,px[ids],margin=60)
                    if not len(detected):continue
                    ii,jj,distance,shift=associate_with_shift_vote(px[ids],detected,8,20)
                    if len(ii)<8 or np.linalg.norm(shift)>15:continue
                    for i,j,d in zip(ii,jj,distance):candidates.append(dict(camera=name,frame=frame,side=side,marker=int(ids[i]),pixel=detected[j].tolist(),association_distance=float(d)))
                used=[]
                for record in sorted(candidates,key=lambda x:x['association_distance']):
                    pixel=np.array(record['pixel'])
                    if any(np.linalg.norm(pixel-old)<2.5 for old in used):continue
                    used.append(pixel);records.append(record)
        finally:cap.release()
        coverage[name]=visible
        print(name,'RGB correspondences',sum(r['camera']==name for r in records),flush=True)
    train=[r for r in records if (r['frame']//stride)%5!=0];held=[r for r in records if (r['frame']//stride)%5==0]
    counts={n:sum(r['camera']==n for r in train) for n in names}
    if any(v<50 for v in counts.values()):raise RuntimeError(f'Insufficient training observations per camera: {counts}')
    correction=fit_shared(train,points,calibrations)
    by_camera={n:errors([r for r in held if r['camera']==n],points,correction,calibrations) for n in names}
    by_side={str(s):errors([r for r in held if r['side']==s],points,correction,calibrations) for s in (0,1) if present[s]}
    accepted=geometry_pass(by_camera) and all(v.get('count',0)>=50 and v.get('median_px',1e9)<=12 for v in by_side.values())
    status='accepted' if accepted else 'needs_review'
    C=np.eye(4);C[:3,:3]=Rotation.from_rotvec(correction[:3]).as_matrix();C[:3,3]=correction[3:]
    report=dict(status=status,optimization='shared NOKOV-to-camera SE3; independent native RGB spots in all four cameras; MANO articulation unchanged',camera_ids=names,
        sync_ok=True,source_dataset=str(a.dataset.resolve()),source_nokov=seed['source_nokov'],seed_alignment=str(a.seed_alignment.resolve()),aligned_input=str(a.aligned_nokov.resolve()),
        hands_present=present,correction=C.tolist(),T_cam0_head=(C@np.array(seed['T_cam0_head'])).tolist(),train_counts_by_camera=counts,heldout_by_camera=by_camera,heldout_by_side=by_side,
        heldout_before=errors(held,points,np.zeros(6),calibrations),heldout_after=errors(held,points,correction,calibrations),
        visibility_sample_counts=coverage,frame_stride=stride,heldout_rule='same frames excluded from all four refinement training views; fixed initial associations without post-fit filtering',
        seed_validation_limit='seed cam0 calibration was fitted on this recording; holdout measures this refinement, not an unseen recording or fully independent seed calibration',
        acceptance_thresholds=dict(heldout_count_per_camera=20,median_px=12,p95_px=30),camera_calibrations=calibrations)
    a.output.mkdir(parents=True)
    (a.output/'fourview_alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'rgb_correspondences.json').write_text(json.dumps(records,indent=2)+'\n')
    corrected=points@C[:3,:3].T+C[:3,3];payload['joints_base_m']=(corrected@B[:3,:3].T+B[:3,3]).astype(np.float32)
    payload['alignment_status']=np.asarray(status);payload['spatial_optimization']=np.asarray('four_native_rgb_views_shared_SE3')
    np.savez_compressed(a.output/('aligned_nokov21.npz' if accepted else 'candidate_aligned_nokov21.npz'),**payload)
    for name,frames in previews.items():
        panels=[]
        for frame,image in frames.items():
            for side,color in [(0,(70,240,70)),(1,(0,180,255))]:
                if not present[side]:continue
                px,z=project_native(points[side,frame],correction,calibrations[name],calibrations['camera0'])
                good=np.isfinite(px).all(-1)&(z>.03)&(px[:,0]>=0)&(px[:,0]<image.shape[1])&(px[:,1]>=0)&(px[:,1]<image.shape[0])
                for point in px[good]:cv2.circle(image,tuple(np.rint(point).astype(int)),5,color,2)
            cv2.putText(image,f'{name} sync={frame} {status}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2);panels.append(cv2.resize(image,(768,480)))
        if panels:cv2.imwrite(str(a.output/f'{name}_preview.jpg'),np.vstack(panels))
    print(json.dumps(report,indent=2))
    if not accepted:raise SystemExit(2)

if __name__=='__main__':main()
