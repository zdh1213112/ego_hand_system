#!/usr/bin/env python3
"""Audit dense RGB marker associations and KB ray consensus; never modify MANO."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import cv2
import numpy as np
from glove_marker_assist import associate_marker_blobs,MarkerAssistConfig,MarkerAssistError
from ego_data.calibration import CameraCalibration
from camera_models import project_points
from fuse_multiview_wilor import triangulate_ransac,_ray_in_base,_ray_condition_number
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'nokov_ego_hand_validation/integration/pipeline'))
from refine_dark_markers_cam0 import detect_dark_markers


def match_hand(predicted, detected):
    """Legacy finger topology association; physical WristM matched separately."""
    config=MarkerAssistConfig(match_distance_px=8,max_shift_px=12,seed_distance_px=12)
    matches=[]
    try:
        a=associate_marker_blobs(predicted,detected,config)
        if np.linalg.norm(a.shift_px)>12:return matches
        matches=[(int(j),np.asarray(pixel),float(d)) for j,pixel,d in zip(a.joint_indices,a.observations,a.residuals_px)]
        used=set(map(int,a.component_indices))
        if np.isfinite(predicted[0]).all():
            choices=[(float(np.linalg.norm(p-predicted[0]-a.shift_px)),i) for i,p in enumerate(detected) if i not in used]
            if choices:
                distance,i=min(choices)
                if distance<=6:matches.append((0,detected[i],distance))
    except MarkerAssistError:
        pass
    return matches


def unique_hand_matches(matches):
    """Drop ambiguous cross-hand blob ownership instead of assigning twice."""
    keep=np.ones(len(matches),bool)
    for i,a in enumerate(matches):
        for j in range(i+1,len(matches)):
            b=matches[j]
            if a[0]!=b[0] and np.linalg.norm(a[2]-b[2])<2.5:
                if abs(a[3]-b[3])<2:keep[i]=keep[j]=False
                else:keep[i if a[3]>b[3] else j]=False
    return [v for v,k in zip(matches,keep) if k]


def geometry_filter(observations,threshold=4.,condition_limit=10000.):
    if len(observations)<3:return None,'insufficient_views'
    point,errors,inlier=triangulate_ransac(observations,threshold,ray_angle_weighting=True)
    if not np.isfinite(point).all() or inlier.sum()<3:return None,'ransac_rejected'
    rays=[_ray_in_base(c,p) for (_,c,p),ok in zip(observations,inlier) if ok]
    if any(x is None for x in rays):return None,'invalid_ray'
    condition=_ray_condition_number(rays)
    angles=[np.degrees(np.arcsin(np.clip(np.linalg.norm(np.cross(a[1],b[1])),0,1))) for i,a in enumerate(rays) for b in rays[i+1:]]
    if condition>condition_limit or max(angles,default=0)<2:return None,'weak_geometry'
    for (_,c,_),ok in zip(observations,inlier):
        if ok and ((point-c.T_base_camera[:3,3])@c.T_base_camera[:3,:3])[2]<=.03:return None,'behind_camera'
    return (point,errors,inlier,condition),'accepted'


def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return dict(count=len(x),median=float(np.median(x)) if len(x) else None,p95=float(np.percentile(x,95)) if len(x) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset','aligned-nokov','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--preview-stride',type=int,default=8)
    a=p.parse_args()
    if a.max_frames<0 or a.preview_stride<1:p.error('invalid frame limits')
    if a.output.exists():raise FileExistsError(a.output)
    cv2.setNumThreads(2)
    names=[f'camera{i}' for i in range(4)]
    cal={n:CameraCalibration.load(a.dataset/'calibration'/f'{n}.json') for n in names}
    if any(c.model!='KB' for c in cal.values()):raise ValueError('Expected native KB cameras')
    with (a.dataset/'multiview_frames.csv').open() as f:allrows=list(csv.DictReader(f))
    with np.load(a.aligned_nokov,allow_pickle=False) as f:
        if str(f['alignment_status'])!='accepted':raise ValueError('Need accepted spatial alignment')
        np.testing.assert_array_equal(f['reference_timestamp_ns'],[int(r['reference_timestamp_ns']) for r in allrows])
        xyz=f['joints_base_m'].copy();valid3d=f['valid'].copy()
    count=min(len(allrows),a.max_frames or len(allrows));rows=allrows[:count];xyz=xyz[:,:count];valid3d=valid3d[:,:count]
    if xyz.shape!=(2,count,21,3):raise ValueError('Expected dual-hand marker ordering')
    a.output.mkdir(parents=True);(a.output/'preview').mkdir()
    pixels=np.full((2,count,4,21,2),np.nan,np.float32)
    distance=np.full((2,count,4,21),np.nan,np.float32)
    seed_pixels=np.full_like(pixels,np.nan);sync_valid=np.zeros((count,4),bool)
    for c,name in enumerate(names):
        C=cal[name];camera=(xyz-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3]
        projected,visible=project_points(C,camera);w,h=C.image_size
        visible &=valid3d&(camera[...,2]>.03)&(projected[...,0]>=0)&(projected[...,0]<w)&(projected[...,1]>=0)&(projected[...,1]<h)
        projected[~visible]=np.nan;seed_pixels[:,:,c]=projected
        targets={}
        for i,row in enumerate(rows):targets.setdefault(int(row[name+'_frame_index']),[]).append(i)
        cap=cv2.VideoCapture(str(a.dataset/'cameras'/name/'video.mp4'))
        try:
            for source in range(max(targets)+1):
                ok,image=cap.read()
                if not ok:raise RuntimeError(f'Video ended: {name}/{source}')
                for frame in targets.get(source,[]):
                    if abs(int(rows[frame][name+'_timestamp_ns'])-int(rows[frame]['camera0_timestamp_ns']))>1500000:continue
                    sync_valid[frame,c]=True;matches=[]
                    for side in (0,1):
                        seed=projected[side,frame]
                        if np.isfinite(seed[1:]).all(-1).sum()<3:continue
                        detected=detect_dark_markers(image,seed,margin=30)
                        if len(detected)==0:continue
                        matches.extend((side,j,point,error) for j,point,error in match_hand(seed,detected))
                    for side,j,point,error in unique_hand_matches(matches):pixels[side,frame,c,j]=point;distance[side,frame,c,j]=error
                if source%300==0:print(name,source,flush=True)
        finally:cap.release()
    raw=np.isfinite(pixels).all(-1)
    geometric=np.zeros(raw.shape,bool);triangulated=np.full((2,count,21,3),np.nan,np.float32);records=[];rejected={}
    for side in (0,1):
        for frame in range(count):
            for marker in range(21):
                ids=np.flatnonzero(raw[side,frame,:,marker])
                if not len(ids):continue
                obs=[(names[c],cal[names[c]],pixels[side,frame,c,marker]) for c in ids]
                result,reason=geometry_filter(obs);rejected[reason]=rejected.get(reason,0)+1
                if result is None:continue
                point,error,inlier,condition=result
                geometric[side,frame,ids[inlier],marker]=True;triangulated[side,frame,marker]=point
                records.append(dict(side=side,frame=frame,marker=marker,views=int(len(ids)),inlier_views=int(inlier.sum()),condition=condition,reprojection_median_px=float(np.median(error[inlier])),nokov_distance_mm=float(np.linalg.norm(point-xyz[side,frame,marker])*1000)))
    # Heuristic ranking only; not a calibrated probability. NOKOV errors are NOT a gate.
    confidence=np.where(geometric,np.exp(-np.nan_to_num(distance,nan=100)**2/(2*8**2)),0).astype(np.float32)
    np.savez_compressed(a.output/'observations.npz',schema='native_dense_rgb_audit_v1',image_space='native_raw',pair_indices=np.arange(count),camera_ids=names,pixels=pixels,raw_valid=raw,geometry_valid=geometric,confidence=confidence,triangulated_base_m=triangulated,seed_pixels=seed_pixels,sync_valid=sync_valid,reference_timestamp_ns=np.array([int(r['reference_timestamp_ns']) for r in rows]),camera_calibrations_json=json.dumps({n:c.to_dict() for n,c in cal.items()}))
    summary=dict(stage='dense_observation_audit',status='needs_visual_review',ready_for_mano=False,frames=count,mano_modified=False,association_source='actual RGB dark spots, NOKOV-seeded legacy finger topology',geometry_threshold_px=4,min_geometry_views=3,geometry_reasons=rejected,by_hand={},source_dataset=str(a.dataset.resolve()),aligned_sha256=hashlib.sha256(a.aligned_nokov.read_bytes()).hexdigest(),limitations=['NOKOV-seeded search introduces selection bias; no independently annotated pixel truth.','Geometry agreement does not prove marker identity or anatomical joint accuracy.','No temporal interpolation or MANO fitting is applied.','Confidence is a heuristic ranking, not a probability; missing is not asserted occlusion.'])
    for side in (0,1):
        good=geometric[side];rawside=raw[side]
        summary['by_hand'][str(side)]=dict(raw_observations=int(rawside.sum()),geometry_observations=int(good.sum()),raw_frames=int(rawside.any((1,2)).sum()),geometry_frames=int(good.any((1,2)).sum()),frames_with_6_points_2_views=int(((good.sum((1,2))>=6)&(good.any(-1).sum(-1)>=2)).sum()),raw_counts_by_camera=rawside.sum((0,2)).tolist(),geometry_counts_by_camera=good.sum((0,2)).tolist(),nokov_distance_mm=stats([x['nokov_distance_mm'] for x in records if x['side']==side]))
    (a.output/'geometry_records.json').write_text(json.dumps(records,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    stamps=np.array([int(r['reference_timestamp_ns']) for r in rows]);fps=1e9/np.median(np.diff(stamps)) if len(stamps)>1 else 30
    chosen=list(range(0,count,a.preview_stride));snapshots=set(chosen[i] for i in np.linspace(0,len(chosen)-1,min(6,len(chosen))).astype(int))
    for c,name in enumerate(names):
        C=cal[name];w,h=C.image_size;writer=cv2.VideoWriter(str(a.output/f'{name}_observations.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/a.preview_stride,(w,h))
        if not writer.isOpened():raise RuntimeError('Cannot create preview video')
        targets={int(rows[i][name+'_frame_index']):i for i in chosen};cap=cv2.VideoCapture(str(a.dataset/'cameras'/name/'video.mp4'))
        try:
            for source in range(max(targets)+1):
                ok,image=cap.read()
                if not ok:raise RuntimeError('Preview video ended')
                if source not in targets:continue
                f=targets[source]
                for side in (0,1):
                    for j in range(21):
                        seed=seed_pixels[side,f,c,j]
                        if np.isfinite(seed).all():cv2.circle(image,tuple(np.rint(seed).astype(int)),4,(255,160,40),1)
                        if raw[side,f,c,j]:
                            point=pixels[side,f,c,j];color=(50,240,50) if geometric[side,f,c,j] else (0,180,255)
                            cv2.line(image,tuple(np.rint(seed).astype(int)),tuple(np.rint(point).astype(int)),color,1)
                            cv2.circle(image,tuple(np.rint(point).astype(int)),3,color,-1)
                            cv2.putText(image,f'{"L" if side==0 else "R"}{j}',tuple(np.rint(point+[4,-4]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
                cv2.rectangle(image,(0,0),(w,56),(20,20,20),-1)
                cv2.putText(image,f'{name} frame={f} BLUE=NOKOV seed YELLOW=RGB match GREEN=3+ view consensus',(12,34),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                writer.write(image)
                if f in snapshots:cv2.imwrite(str(a.output/'preview'/f'{name}_{f:06d}.jpg'),image)
        finally:cap.release();writer.release()
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
