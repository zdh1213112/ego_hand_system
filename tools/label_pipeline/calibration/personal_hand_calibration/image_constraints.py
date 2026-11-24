#!/usr/bin/env python3
"""Conservative raw-image glove contours; deterministic proposals, not ground truth."""
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import ROOT,EXP,CameraCalibration
from camera_models import project_points
from context import load_context

def segment(image,seeds,other,wrist,axis):
 """Color extraction is image based; projected markers select component identity."""
 hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,1]<55)&(hsv[:,:,2]>95)).astype(np.uint8)
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
 scores=np.zeros(count)
 for x,y in seeds:
  x,y=round(x),round(y)
  if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
   ids=np.unique(labels[max(0,y-6):y+7,max(0,x-6):x+7]);scores[ids]+=1
 scores[0]=0
 if scores.max()<8:return None,'too_few_marker_component_matches'
 k=int(scores.argmax());m=(labels==k).astype(np.uint8)
 if stats[k,cv2.CC_STAT_AREA]<500:return None,'component_too_small'
 otherhits=0
 for x,y in other:
  x,y=round(x),round(y)
  if 0<=x<image.shape[1] and 0<=y<image.shape[0] and (labels[max(0,y-4):y+5,max(0,x-4):x+5]==k).any():otherhits+=1
 if otherhits>=3:return None,'both_hands_share_component'
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 if not contours:return None,'no_contour'
 contour=max(contours,key=cv2.contourArea).reshape(-1,2)
 axis=axis/max(np.linalg.norm(axis),1e-9)
 distal=(contour-wrist)@axis>8
 # Exclude forearm/cuff, image borders, and color-threshold-sensitive boundary pixels.
 c=contour[distal];c=c[(c[:,0]>4)&(c[:,0]<image.shape[1]-5)&(c[:,1]>4)&(c[:,1]<image.shape[0]-5)]
 if len(c)<80:return None,'insufficient_distal_contour'
 stable=np.ones(len(c),bool)
 for sat,val in [(45,105),(65,85)]:
  q=((hsv[:,:,1]<sat)&(hsv[:,:,2]>val)).astype(np.uint8);q=cv2.morphologyEx(q,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));edge=cv2.morphologyEx(q,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));distance=cv2.distanceTransform(1-edge,cv2.DIST_L2,3);stable&=distance[c[:,1],c[:,0]]<2.5
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);grad=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1));stable&=grad[c[:,1],c[:,0]]>35
 ratio=float(stable.mean());c=c[stable]
 if ratio<.65 or len(c)<80:return None,'unstable_color_boundary'
 if np.linalg.norm(c.max(0)-c.min(0))<65:return None,'hand_too_small'
 # Reject implausibly large components relative to marker extent (table/wall connection).
 if np.linalg.norm(c.max(0)-c.min(0))>2.8*np.linalg.norm(seeds.max(0)-seeds.min(0)):return None,'oversized_component'
 c=c[np.linspace(0,len(c)-1,min(160,len(c))).astype(int)]
 return dict(points=c,mask=m,stable_fraction=ratio,component_marker_hits=int(scores[k])),None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path);ap.add_argument('--source',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);(a.output/'previews').mkdir();(a.output/'images').mkdir();(a.output/'masks').mkdir()
 context=load_context(a.context);src=a.source or EXP/'personal_calibration_v2';ds=context['dataset'];tracks=[np.load(src/f'{h}_pilot.npz') if (src/f'{h}_pilot.npz').is_file() else None for h in ['left','right']]
 if not any(t is not None for t in tracks):raise ValueError('source has no hand fit')
 maps=[{int(f):i for i,f in enumerate(z['pair_indices'])} if z is not None else {} for z in tracks]
 with (ds/'multiview_frames.csv').open() as f:frames=list(csv.DictReader(f))
 C0=CameraCalibration.load(ds/'calibration/camera0.json');records=[];rejected=[]
 for cam in [f'camera{i}' for i in range(4)]:
  C=CameraCalibration.load(ds/'calibration'/f'{cam}.json');cap=cv2.VideoCapture(str(ds/'cameras'/cam/'video.mp4'))
  def proj(q,side):
   q=q.copy();q[:,0]*=(-1 if side==0 else 1);base=q@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3];xyz=(base-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];uv,_=project_points(C,xyz);return uv,xyz[:,2]
  for sync in sorted(set(maps[0])|set(maps[1])):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[sync][cam+'_frame_index']));ok,image=cap.read()
   if not ok:raise RuntimeError('Image missing')
   for side,z in enumerate(tracks):
    if z is None or sync not in maps[side]:continue
    i=maps[side][sync];seed,depth=proj(z['target24'][i],side);j,_=proj(z['joints'][i],side);other=np.empty((0,2))
    if sync in maps[1-side]:other,od=proj(tracks[1-side]['target24'][maps[1-side][sync]],1-side);other=other[(od>.03)&np.isfinite(other).all(1)]
    if (depth<.03).any() or not np.isfinite(seed).all() or ((seed<0).any(1)| (seed[:,0]>=image.shape[1]) | (seed[:,1]>=image.shape[0])).sum()>2:
     rejected.append(dict(camera=cam,sync=sync,side=side,reason='partial_or_behind_camera'));continue
    axis=j[[5,9,13,17]].mean(0)-j[0];obs,why=segment(image,seed,other,j[0],axis)
    if obs is None:rejected.append(dict(camera=cam,sync=sync,side=side,reason=why));continue
    uv,d=proj(z['vertices'][i],side)
    # Full triangle raster, not convex hull, preserves spaces between fingers.
    mm=np.zeros(image.shape[:2],np.uint8)
    for tri in np.rint(uv[z['faces']]).astype(np.int32):cv2.fillConvexPoly(mm,tri,1)
    edge=cv2.morphologyEx(mm,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));distance=cv2.distanceTransform(1-edge,cv2.DIST_L2,3);idx=np.rint(uv).astype(int);inside=(idx[:,0]>=0)&(idx[:,0]<image.shape[1])&(idx[:,1]>=0)&(idx[:,1]<image.shape[0]);ids=np.flatnonzero(inside);ids=ids[(distance[idx[ids,1],idx[ids,0]]<2.5)&((uv[ids]-j[0])@axis/max(np.linalg.norm(axis),1e-9)>8)]
    if len(ids)<25:rejected.append(dict(camera=cam,sync=sync,side=side,reason='insufficient_mesh_boundary'));continue
    stem=f'{cam}_{sync:06d}';cv2.imwrite(str(a.output/'images'/f'{stem}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,97]);rgba=np.zeros((*image.shape[:2],4),np.uint8);rgba[:,:,3]=obs['mask']*255;cv2.imwrite(str(a.output/'masks'/f'{stem}_{side}.png'),rgba)
    name=f'{cam}_{sync:06d}_{side}.jpg';preview=image.copy();cv2.polylines(preview,[obs['points'].astype(np.int32)],False,(0,255,0),2);preview[edge>0]=(0,130,255)
    lo=np.maximum(0,np.floor(seed.min(0)-60)).astype(int);hi=np.minimum(image.shape[1::-1],np.ceil(seed.max(0)+60)).astype(int);crop=preview[lo[1]:hi[1],lo[0]:hi[0]];cv2.imwrite(str(a.output/'previews'/name),crop)
    records.append(dict(camera=cam,sync=sync,side=side,accepted=True,replay_exact=True,mask=f'masks/{cam}_{sync:06d}_{side}.png',image=f'images/{cam}_{sync:06d}.jpg',calibration=bool(z['calibration_mask'][i]),points=obs['points'].tolist(),boundary_vertices=ids.tolist(),stable_fraction=obs['stable_fraction'],component_marker_hits=obs['component_marker_hits'],preview='previews/'+name,K=C.K.tolist(),D=C.distortion.tolist(),T_base_camera=C.T_base_camera.tolist(),T_base_cam0=C0.T_base_camera.tolist()))
  cap.release();print(cam,'accepted total',len(records),flush=True)
 data=dict(schema='automatic_native_contours_v1',records=records,rejected=rejected,limitations=['Color-threshold proposals are not independently verified silhouettes.','NOKOV projection chooses component identity; baseline mesh determines distal wrist exclusion and initial silhouette vertex indices.','Shared hand components are rejected; no per-pixel occlusion model.','No inferred fingertip identities are used.']);(a.output/'observations.json').write_text(json.dumps(data));(a.output/'data.js').write_text('window.CONTOUR_AUDIT='+json.dumps(dict(report={},records=records),separators=(',',':'))+';\n');print('DONE',len(records),'accepted',len(rejected),'rejected',flush=True)
if __name__=='__main__':main()
