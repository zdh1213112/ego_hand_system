#!/usr/bin/env python3
"""Select conservative complete, uncontaminated mask candidates before subject calibration."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/personal_hand_calibration'))
from context import load_context
from run import CameraCalibration
from camera_models import project_points

def quality(mask,points,seeds,wrist,axis,image):
 h,w=mask.shape;axis=axis/max(np.linalg.norm(axis),1e-9);p=seeds[3:];hull=cv2.convexHull(np.rint(seeds).astype(np.int32));hm=np.zeros_like(mask);cv2.fillConvexPoly(hm,hull,1);outside_hull=cv2.distanceTransform(1-hm,cv2.DIST_L2,3);outside_mask=cv2.distanceTransform(1-mask,cv2.DIST_L2,3);q=np.rint(p).astype(int);q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1);missing=outside_mask[q[:,1],q[:,0]];tips=missing[[4,8,12,16,20]]
 points=np.array(points);uv=np.rint(points).astype(int);leak=outside_hull[uv[:,1],uv[:,0]];diag=np.linalg.norm(p.max(0)-p.min(0));margin=max(8,min(25,.07*diag));ys,xs=np.nonzero(mask);distal=(np.c_[xs,ys]-wrist)@axis>8;ys,xs=ys[distal],xs[distal];spill=float((outside_hull[ys,xs]>margin).mean()) if len(xs) else 1
 lo=np.maximum(0,np.floor(p.min(0)-10)).astype(int);hi=np.minimum([w,h],np.ceil(p.max(0)+10)).astype(int);crop=image[lo[1]:hi[1],lo[0]:hi[0]];sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if crop.size else 0
 fingertip=seeds[[7,11,15,19,23]];d=np.linalg.norm(fingertip[:,None]-fingertip[None],axis=-1);d+=np.eye(5)*1e6;sep=float(d.min())
 metrics=dict(marker_coverage_fraction=float((missing<=8).mean()),fingertip_max_outside_mask_px=float(tips.max()),contour_outside_marker_hull_p95_px=float(np.percentile(leak,95)),contour_outside_marker_hull_max_px=float(leak.max()),distal_mask_spill_fraction=spill,hull_margin_px=margin,sharpness=sharp,min_fingertip_separation_px=sep)
 # A pale gray glove edge can be missed by the original bright-white mask.
 # If both 3px and 6px outside many contour points remain gray, reject it.
 float_mask=mask.astype(np.float32);hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
 gradient=np.stack([cv2.Sobel(float_mask,cv2.CV_32F,1,0),cv2.Sobel(float_mask,cv2.CV_32F,0,1)],axis=-1)[uv[:,1],uv[:,0]]
 norm=np.linalg.norm(gradient,axis=1);gradient/=np.maximum(norm[:,None],1e-8);gray=np.ones(len(uv),bool)
 for radius in [3,6]:
  q=np.rint(points-gradient*radius).astype(int);q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1);pix=hsv[q[:,1],q[:,0]];gray&=(pix[:,1]<65)&(pix[:,2]>45)
 gray_fraction=float(gray[norm>0].mean()) if (norm>0).any() else 1.0
 metrics['gray_outside_boundary_fraction']=gray_fraction
 reasons=[]
 if gray_fraction>.12:reasons.append('gray_glove_may_continue_outside_mask')
 if np.percentile(leak,95)>margin or leak.max()>margin+10 or spill>.025:reasons.append('background_or_excessive_extent')
 if (missing<=8).mean()<.95 or tips.max()>6 or any((missing[1+4*k:1+4*k+4]<=8).sum()<3 for k in range(5)):reasons.append('incomplete_marker_or_fingertip_coverage')
 if sep<12:reasons.append('fingertip_overlap_or_occlusion')
 if sharp<40:reasons.append('blur_or_low_detail')
 return metrics,reasons

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-calibration-frames',type=int,default=16);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);c=load_context(a.context);audit=json.loads((a.audit/'data.js').read_text().removeprefix('window.CONTOUR_AUDIT=').strip().rstrip(';'));z=[np.load(a.source/f'{h}_pilot.npz') if (a.source/f'{h}_pilot.npz').is_file() else None for h in ['left','right']];maps=[{int(f):i for i,f in enumerate(t['pair_indices'])} if t is not None else {} for t in z];C0=CameraCalibration.load(c['dataset']/'calibration/camera0.json');cameras={cam:CameraCalibration.load(c['dataset']/'calibration'/f'{cam}.json') for cam in ['camera0','camera1','camera2','camera3']};scored=[];cv2.setNumThreads(2);last_image_path=None;image=None
 for r in audit['records']:
  if not r['accepted'] or not r['replay_exact'] or not r['mask']:continue
  side=r['side']
  if r['sync'] not in maps[side]: continue
  i=maps[side][r['sync']];C=cameras[r['camera']]
  def proj(p):
   p=p.copy();p[:,0]*=(-1 if side==0 else 1);p=p@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3];p=(p-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];return project_points(C,p)[0]
  seeds=proj(z[side]['target24'][i]);j=proj(z[side]['joints'][i]);mask=(cv2.imread(str(a.audit/r['mask']),cv2.IMREAD_UNCHANGED)[:,:,3]>0).astype(np.uint8);image_path=str(a.audit/r['image'])
  if image_path!=last_image_path:image=cv2.imread(image_path);last_image_path=image_path
  metrics,reasons=quality(mask,r['points'],seeds,j[0],j[[5,9,13,17]].mean(0)-j[0],image)
  scored.append(dict(camera=r['camera'],sync=r['sync'],side=side,accepted_quality=not reasons,reasons=reasons,metrics=metrics,image=str((a.audit/r['image']).resolve()),mask=str((a.audit/r['mask']).resolve()),points=r['points'],K=C.K.tolist(),D=C.distortion.tolist(),T_base_camera=C.T_base_camera.tolist(),T_base_cam0=C0.T_base_camera.tolist()))
 report=dict(stage='high_quality_candidate_selection_v3',selection_is_automatic_not_human_ground_truth=True,source_audit=str(a.audit.resolve()),holdout=c['holdout'],hands={})
 for side,hand in enumerate(['left','right']):
  if z[side] is None: report['hands'][hand]=dict(recorded=False); continue
  good=[r for r in scored if r['side']==side and r['accepted_quality']];groups={}
  for r in good:groups.setdefault(r['sync'],[]).append(r)
  calpool=sorted(f for f,rs in groups.items() if len(rs)>=2 and not c['holdout'][0]<=f<c['holdout'][1]);valid=sorted(f for f,rs in groups.items() if len(rs)>=2 and c['holdout'][0]<=f<c['holdout'][1])
  if calpool:
   ix=[maps[side][f] for f in calpool];feature=np.c_[z[side]['hand_pose_pca'][ix],z[side]['global_orient'][ix]];feature=(feature-feature.mean(0))/np.maximum(feature.std(0),.15);chosen=[int(np.argmin(np.linalg.norm(feature-np.median(feature,axis=0),axis=1)))];dist=np.full(len(calpool),np.inf)
   for _ in range(min(a.max_calibration_frames,len(calpool))-1):dist=np.minimum(dist,((feature-feature[chosen[-1]])**2).sum(1));dist[chosen]=-1;chosen.append(int(dist.argmax()))
   selected=sorted(calpool[k] for k in chosen)
  else:selected=[]
  report['hands'][hand]=dict(good_views=len(good),multiview_calibration_pool=len(calpool),calibration_sync=selected,validation_sync=valid,calibration_views=[dict(camera=r['camera'],sync=r['sync']) for r in good if r['sync'] in selected],validation_views=[dict(camera=r['camera'],sync=r['sync']) for r in good if r['sync'] in valid])
 (a.output/'selection.json').write_text(json.dumps(dict(report=report,records=scored),indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
