#!/usr/bin/env python3
"""Raw NOKOV-to-native-image audit. No MANO fit and no calibration updates."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'label_export/pipeline/calibration/marker24_pilot'))
from run_pilot import raw24,CameraCalibration
from camera_models import project_points
NAMES=['WristM','WristIn','WristOut','HandOffset']+[f'Finger{f}{i}' for f in ['Thumb','Index','Middle','Ring','Pinky'] for i in range(1,5)]
FRAMES=[80,288,512,552,576,584,616,632,656,696,944,1248]
def detect(image):
 """Image-only candidate extraction over the full image, independent of projected points."""
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17));candidates=[]
 for kind,mask in [('dark',cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)>=20),('bright',(cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)>=18)&(gray>=225))]:
  count,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
  for i in range(1,count):
   x,y,w,h,area=stats[i]
   if not(3<=area<=160 and 2<=w<=22 and 2<=h<=22 and max(w,h)/min(w,h)<=2.7):continue
   patch=gray[max(0,y-5):min(gray.shape[0],y+h+5),max(0,x-5):min(gray.shape[1],x+w+5)];surround=float(np.median(patch));center=float(gray[y:y+h,x:x+w][labels[y:y+h,x:x+w]==i].mean())
   if kind=='dark' and (surround<120 or surround-center<15):continue
   if kind=='bright' and (surround<70 or center-surround<18):continue
   c=centres[i]
   if any(np.linalg.norm(c-np.asarray(p['pixel']))<2 for p in candidates):continue
   candidates.append(dict(pixel=c.tolist(),kind=kind,area=int(area),contrast=abs(center-surround)))
 return candidates

def associate(seeds,candidates):
 """One-to-one suggested association, explicitly projection-guided, never ground truth."""
 valid=np.isfinite(seeds).all(1);ids=np.flatnonzero(valid);pixels=np.asarray([c['pixel'] for c in candidates]);result=[]
 if not len(ids) or not len(pixels):return result
 d=np.linalg.norm(seeds[ids,None]-pixels[None],axis=-1);rows,cols=linear_sum_assignment(np.minimum(d,1e5))
 for row,col in zip(rows,cols):
  distance=float(d[row,col])
  if distance>12:continue
  alternatives=np.delete(d[row],col);other_markers=np.delete(d[:,col],row);gap=min(float(alternatives.min()) if len(alternatives) else 1e6,float(other_markers.min()) if len(other_markers) else 1e6)-distance
  idx=int(ids[row]);result.append(dict(side=idx//24,marker=idx%24,candidate=int(col),distance_px=distance,ambiguity_gap_px=gap,unambiguous=gap>=2,delta=(pixels[col]-seeds[idx]).tolist(),confirmed=False))
 return result

def metric(records):
 if not records:return {'count':0}
 e=np.array([r['distance_px'] for r in records]);d=np.array([r['delta'] for r in records]);return dict(count=len(e),median_px=float(np.median(e)),p95_px=float(np.percentile(e,95)),median_observed_minus_projected_px=np.median(d,axis=0).tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);(out/'images').mkdir();exp=ROOT/'output/native_kb_training_20261111';dataset=exp/'real_dual/pair_20261111_192916/ego/normalized_multiview';xyz,stamps=raw24(exp);C0=CameraCalibration.load(dataset/'calibration/camera0.json');base=xyz@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3]
 with (dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
 used=set()
 for path in [exp/'fourview_192916/alignment_180/rgb_correspondences.json',exp/'real_dual/pair_20261111_192916/alignment/dual_marker_correspondences.json']:
  if path.exists():used.update(int(r['frame']) for r in json.loads(path.read_text()))
 records=[];summary=dict(stage='raw_nokov24_image_audit',status='automatic_candidates_need_review',mano_used=False,calibration_modified=False,all_frames=FRAMES,association_gate_px=12,independent_pixel_ground_truth=False,cameras={})
 for camera in [f'camera{i}' for i in range(4)]:
  C=CameraCalibration.load(dataset/'calibration'/f'{camera}.json');cap=cv2.VideoCapture(str(dataset/'cameras'/camera/'video.mp4'));matches=[];check=[];ambiguous=0
  for sync in FRAMES:
   source=int(rows[sync][camera+'_frame_index']);cap.set(cv2.CAP_PROP_POS_FRAMES,source);ok,image=cap.read()
   if not ok:raise RuntimeError('Cannot read source image')
   points=(base[:,sync]-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];uv,_=project_points(C,points);w,h=image.shape[1],image.shape[0];valid=np.isfinite(uv).all(-1)&(points[:,:,2]>.03)&(uv[:,:,0]>=0)&(uv[:,:,0]<w)&(uv[:,:,1]>=0)&(uv[:,:,1]<h);seeds=uv.copy();seeds[~valid]=np.nan
   candidates=detect(image);suggestions=associate(seeds.reshape(-1,2),candidates);path=f'images/{camera}_{sync:06d}.jpg';cv2.imwrite(str(out/path),image,[cv2.IMWRITE_JPEG_QUALITY,97]);sync_ms=(int(rows[sync][camera+'_timestamp_ns'])-int(rows[sync]['camera0_timestamp_ns']))/1e6
   clean=lambda q:[[p.tolist() if np.isfinite(p).all() else None for p in hand] for hand in q]
   rec=dict(camera=camera,sync=sync,source_frame=source,reference_timestamp_ns=int(stamps[sync]),camera_delta_ms=sync_ms,used_in_previous_alignment=sync in used,image=path,size=[w,h],projected=clean(seeds),candidates=candidates,suggestions=suggestions);records.append(rec)
   accepted=[dict(m,sync=sync) for m in suggestions if m['unambiguous']];matches.extend(accepted);ambiguous+=sum(not m['unambiguous'] for m in suggestions)
   if sync not in used:check.extend(accepted)
  cap.release();summary['cameras'][camera]=dict(all_unconfirmed_candidates=metric(matches),not_used_in_prior_alignment_candidates=metric(check),ambiguous_count=ambiguous,by_hand={str(s):metric([m for m in matches if m['side']==s]) for s in [0,1]});print(camera,summary['cameras'][camera],flush=True)
 summary['frames_used_in_prior_alignment']=[f for f in FRAMES if f in used];summary['limitations']=['Image blob extraction uses no NOKOV/MANO input, but identity assignment uses NOKOV projection and a 12px gate, so reported residuals are selection-biased.','Bright reflection centroids and dark spot centroids may differ from projected sphere centres.','No automated match is a manually verified correspondence. Missing matches do not imply missing/occluded markers.','No change to synchronization, intrinsics, distortion, extrinsics or fitted labels.']
 data=dict(schema='native_marker_audit_v1',audit_id=hashlib.sha256((str(dataset.resolve())+'|'+json.dumps(FRAMES)).encode()).hexdigest(),dataset=str(dataset.resolve()),names=NAMES,frames=records,summary=summary);(out/'data.js').write_text('window.AUDIT_DATA='+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n');(out/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(out/'nokov24_native_projection_source.npz',cam0_m=xyz[:,FRAMES],sync=np.array(FRAMES),timestamps=stamps[FRAMES]);(out/'index.html').write_text(Path(__file__).with_name('viewer.html').read_text());print(out)
if __name__=='__main__':main()
