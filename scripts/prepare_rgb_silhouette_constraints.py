#!/usr/bin/env python3
"""Extract white-glove RGB contours and baseline visible mesh boundary vertices."""
import argparse,csv,json,sys
from pathlib import Path
import cv2
import numpy as np

def glove_component(patch,seed):
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    fg=((hsv[:,:,1]<90)&(hsv[:,:,2]>85)&(patch[:,:,2].astype(float)-patch[:,:,0]<45)).astype(np.uint8)
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(fg,8);best=None
    for k in range(1,count):
        if stats[k,4]<100:continue
        y,x=np.where(labels==k);score=np.min((x-seed[0])**2+(y-seed[1])**2)-.001*stats[k,4]
        if best is None or score<best[0]:best=(score,k)
    return None if best is None else labels==best[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0);p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--contour-points',type=int,default=128);p.add_argument('--boundary-points',type=int,default=96)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.root/'ego_hand_system/scripts'));sys.path.insert(0,str(Path(__file__).parent))
    from camera_models import project_points
    from mano_conventions import physicalize_geometry
    from ego_data.calibration import CameraCalibration
    from ego_data.dataset import SequentialVideoReader
    with np.load(a.fit/'track_0.npz') as z:v,j,faces=physicalize_geometry(z['vertices'],z['joints'],z['faces'],'Left')
    rows=list(csv.DictReader((a.dataset/'multiview_frames.csv').open()));F=len(rows);C=2
    contour=np.zeros((C,F,a.contour_points,2),np.float32);cv=np.zeros(contour.shape[:-1],bool)
    bv=np.zeros((C,F,a.boundary_points),bool);bi=np.zeros(bv.shape,np.int64)
    cameras=['camera0','camera1'];rot=[];trans=[];K=[];D=[];stats=[]
    c0=CameraCalibration.load(a.dataset/'calibration/camera0.json')
    ids=range(a.start_frame,min(F,a.start_frame+a.max_frames) if a.max_frames else F)
    for ci,cam in enumerate(cameras):
        c=CameraCalibration.load(a.dataset/f'calibration/{cam}.json');T=np.linalg.inv(c.T_base_camera)@c0.T_base_camera
        if c.model!='KB':raise ValueError('this extraction currently supports native KB cameras')
        rot.append(T[:3,:3]);trans.append(T[:3,3]);K.append(c.K);D.append(c.distortion.ravel())
        vp=project_points(c,v@T[:3,:3].T+T[:3,3])[0];jp=project_points(c,j@T[:3,:3].T+T[:3,3])[0]
        reader=SequentialVideoReader(a.dataset/f'cameras/{cam}/video.mp4',c.image_size)
        try:
            for i in ids:
                im=reader.read(int(rows[i][cam+'_frame_index']));points=vp[i]
                if not np.isfinite(points).all():continue
                lo=np.maximum(np.floor(points.min(0)-50).astype(int),0);hi=np.minimum(np.ceil(points.max(0)+50).astype(int),[im.shape[1],im.shape[0]])
                if np.any(hi-lo<20):continue
                patch=im[lo[1]:hi[1],lo[0]:hi[0]];mask=glove_component(patch,jp[i,[5,9,13,17]].mean(0)-lo)
                if mask is None:continue
                baseline=np.zeros(mask.shape,np.uint8)
                for poly in np.rint(points[faces]-lo).astype(np.int32):cv2.fillConvexPoly(baseline,poly,1)
                if not .35<float(mask.sum())/max(int(baseline.sum()),1)<3.:continue
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                if not contours:continue
                edge=max(contours,key=cv2.contourArea)[:,0].astype(float)
                origin=jp[i,0]-lo;direction=jp[i,9]-jp[i,0];direction/=max(np.linalg.norm(direction),1e-6)
                signed=(edge-origin)@direction
                keep=(signed>7)&(edge[:,0]>2)&(edge[:,1]>2)&(edge[:,0]<mask.shape[1]-3)&(edge[:,1]<mask.shape[0]-3)
                edge=edge[keep]
                if len(edge)<30:continue
                be=baseline-cv2.erode(baseline,np.ones((3,3),np.uint8));dt=cv2.distanceTransform(1-be,cv2.DIST_L2,3)
                pix=np.rint(points-lo).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<mask.shape[1])&(pix[:,1]>=0)&(pix[:,1]<mask.shape[0])
                use=np.flatnonzero(inside)
                use=use[(dt[pix[use,1],pix[use,0]]<2.5)&(((points[use]-jp[i,0])@direction)>7)]
                if len(use)<15:continue
                take=np.linspace(0,len(edge)-1,min(len(edge),a.contour_points)).astype(int);contour[ci,i,:len(take)]=edge[take]+lo;cv[ci,i,:len(take)]=True
                take=np.linspace(0,len(use)-1,min(len(use),a.boundary_points)).astype(int);bi[ci,i,:len(take)]=use[take];bv[ci,i,:len(take)]=True
                stats.append(dict(camera=cam,sync_index=i,contour_points=int(cv[ci,i].sum()),boundary_vertices=int(bv[ci,i].sum())))
                if i%200==0:print(cam,i,'/',F,flush=True)
        finally:reader.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,contour_px=contour,contour_valid=cv,boundary_vertex_indices=bi,boundary_valid=bv,rotation=np.asarray(rot,np.float32),translation=np.asarray(trans,np.float32),K=np.asarray(K,np.float32),distortion=np.asarray(D,np.float32),pair_indices=np.arange(F),reference_timestamp_ns=np.asarray([int(r['reference_timestamp_ns']) for r in rows],np.int64))
    summary=dict(stage='native_rgb_silhouette_constraints',source_dataset=str(a.dataset.resolve()),source_fit=str(a.fit.resolve()),frames=F,range=[a.start_frame,min(F,a.start_frame+a.max_frames) if a.max_frames else F],valid_frames_by_camera={cam:int(cv[k].any(-1).sum()) for k,cam in enumerate(cameras)},segmentation='white-glove HSV component seeded by baseline palm; excludes wrist cutoff; automatic proxy, not anatomical ground truth',records=stats)
    a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='records'},flush=True)

if __name__=='__main__':main()
