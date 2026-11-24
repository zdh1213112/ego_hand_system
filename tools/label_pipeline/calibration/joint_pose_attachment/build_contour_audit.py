#!/usr/bin/env python3
"""Show exact optimization contour points, with verified deterministic segmentation replay."""
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/personal_hand_calibration'))
from image_constraints import segment,CameraCalibration
from context import load_context
from camera_models import project_points

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--source',type=Path,required=True,help='Marker fit used by the original image extractor, before contour refinement');ap.add_argument('--observations',type=Path,required=True);ap.add_argument('--joint-fit',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);(a.output/'images').mkdir();(a.output/'masks').mkdir();c=load_context(a.context);ds=c['dataset'];data=json.loads(a.observations.read_text());accepted={(r['camera'],r['sync'],r['side']):r for r in data['records']};rejected={(r['camera'],r['sync'],r['side']):r for r in data['rejected']};tracks=[np.load(a.source/f'{h}_pilot.npz') for h in ['left','right']];maps=[{int(f):i for i,f in enumerate(z['pair_indices'])} for z in tracks];C0=CameraCalibration.load(ds/'calibration/camera0.json');used=set()
 if a.joint_fit:
  s=json.loads((a.joint_fit/'summary.json').read_text())
  for side,h in enumerate(['left','right']):used.update((cam,sync,side) for cam,sync in s['hands'][h]['fit_keys'])
 with (ds/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
 records=[];matched=0;missing=0;images=0
 for cam in [f'camera{i}' for i in range(4)]:
  C=CameraCalibration.load(ds/'calibration'/f'{cam}.json');cap=cv2.VideoCapture(str(ds/'cameras'/cam/'video.mp4'))
  def proj(q,side):
   q=q.copy();q[:,0]*=(-1 if side==0 else 1);base=q@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3];xyz=(base-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];uv,_=project_points(C,xyz);return uv,xyz[:,2]
  syncs=sorted({key[1] for key in set(accepted)|set(rejected) if key[0]==cam})
  for sync in syncs:
   source=int(rows[sync][cam+'_frame_index']);cap.set(cv2.CAP_PROP_POS_FRAMES,source);ok,image=cap.read()
   if not ok:raise RuntimeError(f'Missing original frame: {cam}/{sync}')
   image_path=f'images/{cam}_{sync:06d}.jpg';cv2.imwrite(str(a.output/image_path),image,[cv2.IMWRITE_JPEG_QUALITY,97]);images+=1
   for side,z in enumerate(tracks):
    key=(cam,sync,side)
    if key not in accepted and key not in rejected:continue
    i=maps[side][sync];seed,depth=proj(z['target24'][i],side);j,_=proj(z['joints'][i],side);other=np.empty((0,2))
    if sync in maps[1-side]:other,od=proj(tracks[1-side]['target24'][maps[1-side][sync]],1-side);other=other[(od>.03)&np.isfinite(other).all(1)]
    valid=not ((depth<.03).any() or not np.isfinite(seed).all() or ((seed<0).any(1)|(seed[:,0]>=image.shape[1])|(seed[:,1]>=image.shape[0])).sum()>2)
    replay,why=segment(image,seed,other,j[0],j[[5,9,13,17]].mean(0)-j[0]) if valid else (None,'partial_or_behind_camera')
    original=accepted.get(key);points=original['points'] if original else [];match=bool(original is not None and replay is not None and np.array_equal(np.asarray(points),replay['points']))
    mask_path=None;boundary=[]
    if replay is not None:
     contours,_=cv2.findContours(replay['mask'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);boundary=[q.reshape(-1,2).tolist() for q in contours];rgba=np.zeros((*image.shape[:2],4),np.uint8);rgba[:,:,:3]=[90,240,100];rgba[:,:,3]=replay['mask']*255;mask_path=f'masks/{cam}_{sync:06d}_{side}.png';cv2.imwrite(str(a.output/mask_path),rgba)
    if original:matched+=int(match);missing+=int(not match)
    finite=seed[np.isfinite(seed).all(1)];box=[0,0,image.shape[1],image.shape[0]]
    pts=np.array(points) if points else finite
    if len(pts):box=[max(0,float(pts[:,0].min()-65)),max(0,float(pts[:,1].min()-65)),min(image.shape[1],float(pts[:,0].max()+65)),min(image.shape[0],float(pts[:,1].max()+65))]
    if box[2]-box[0]<20 or box[3]-box[1]<20:box=[0,0,image.shape[1],image.shape[0]]
    records.append(dict(camera=cam,sync=sync,side=side,source_frame=source,image=image_path,size=list(image.shape[1::-1]),accepted=original is not None,used_in_joint_fit=key in used,reason=None if original else rejected[key]['reason'],points=points,mask=mask_path,external_boundary=boundary,replay_exact=match,replay_reason=why,bbox=box,stable_fraction=original.get('stable_fraction') if original else None,component_marker_hits=original.get('component_marker_hits') if original else None))
  cap.release();print(cam,'records',len(records),'replay mismatches',missing,flush=True)
 report=dict(dataset=str(ds),observations=str(a.observations.resolve()),source_marker_fit=str(a.source.resolve()),joint_fit=str(a.joint_fit.resolve()) if a.joint_fit else None,accepted=len(accepted),rejected=len(rejected),used_in_joint_fit=len(used),original_images=images,replay_exact_count=matched,replay_mismatch_count=missing,limitations=['Purple points come directly from the optimization input JSON.','Green segmentation replay uses original video pixels and is marked verified only if the exact accepted point set matches.','Rejected samples are shown with their actual rejection reasons, not fabricated contours.'])
 (a.output/'data.js').write_text('window.CONTOUR_AUDIT='+json.dumps(dict(report=report,records=records),separators=(',',':'))+';\n');(a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'index.html').write_text(Path(__file__).with_name('contour_audit.html').read_text());print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
