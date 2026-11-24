#!/usr/bin/env python3
"""Audit one pose-locked frame without changing any fit or label."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
ROOT=Path(__file__).resolve().parents[4]
def main():
 p=argparse.ArgumentParser();p.add_argument('--sync',type=int,default=632);p.add_argument('--side',type=int,choices=(0,1),default=1);a=p.parse_args();hand=['left','right'][a.side];exp=ROOT/'output/native_kb_training_20261111';pilot=exp/'marker24_pilot_pose_locked';z=np.load(pilot/f'{hand}_pilot.npz');b=np.load(exp/f'fourview_192916/native_labels/mano_fit_full_cam01/track_{a.side}.npz');i=int(np.flatnonzero(z['pair_indices']==a.sync)[0]);j=int(np.flatnonzero(b['pair_indices']==a.sync)[0]);v0=z['baseline_vertices'][i];j0=z['baseline_joints'][i];R0=Rotation.from_rotvec(b['global_orient'][j]).as_matrix();R1=Rotation.from_rotvec(z['global_orient'][i]).as_matrix();rel=R1@R0.T;rotvec=Rotation.from_matrix(rel).as_rotvec();faces=z['faces'];fi=z['attachment_faces'];bw=z['barycentric'];target=z['target24'][i];pred0=np.concatenate([z['wrist_local']@R0.T+j0[0],(v0[faces[fi]]*bw[:,:,None]).sum(1)]);pred1=z['predicted24'][i];err0=np.linalg.norm(pred0-target,axis=1)*1000;err1=np.linalg.norm(pred1-target,axis=1)*1000;weights=np.r_[np.full(3,.2),np.ones(21)];cost0=weights*(np.sqrt(1+(err0/6)**2)-1);cost1=weights*(np.sqrt(1+(err1/6)**2)-1)
 rows=[dict(name=str(n),baseline_same_attachment_mm=float(err0[k]),candidate_mm=float(err1[k]),weighted_loss_reduction=float(cost0[k]-cost1[k])) for k,n in enumerate(z['marker_names'])]
 out=pilot/f'diagnosis_{hand}_{a.sync:06d}';out.mkdir(exist_ok=True);report=dict(sync=a.sync,hand=hand,rotation_change_deg=float(np.linalg.norm(rotvec)*180/np.pi),rotation_axis_in_cam0=(rotvec/np.linalg.norm(rotvec)).tolist(),wrist_shift_cam0_mm=((z['joints'][i,0]-j0[0])*1000).tolist(),translation_parameter_delta_mm=((z['translation'][i]-b['translation'][j])*1000).tolist(),finger_pose_exactly_equal=bool(np.array_equal(z['hand_pose_pca'][i],b['hand_pose_pca'][j])),shape_exactly_equal=bool(np.array_equal(z['betas'],b['betas'])),marker_median_before_mm=float(np.median(err0)),marker_median_after_mm=float(np.median(err1)),markers=rows,groups={},camera_projection_changes={})
 if a.side==0:
  for key in ['rotation_axis_in_cam0','wrist_shift_cam0_mm','translation_parameter_delta_mm']:report[key][0]*=-1
 for name,ids in [('wrist',[0,1,2]),('palm',[3]),('thumb',list(range(4,8))),('index',list(range(8,12))),('middle',list(range(12,16))),('ring',list(range(16,20))),('pinky',list(range(20,24)))]:report['groups'][name]=dict(before_median_mm=float(np.median(err0[ids])),after_median_mm=float(np.median(err1[ids])),loss_reduction=float((cost0[ids]-cost1[ids]).sum()))
 review=pilot/'review';data=json.loads((review/'data.js').read_text()[len('window.LABEL_DATA='):-2])
 for f in [f for f in data['frames'] if f['sync']==a.sync]:
  h0=next((h for h in f['baseline_hands'] if h['side']==a.side),None);h1=next((h for h in f['hands'] if h['side']==a.side),None)
  if h0 is None or h1 is None:report['camera_projection_changes'][f['camera']]={'visible':False};continue
  uv0=np.asarray(h0['vertices'])[:,:2];uv1=np.asarray(h1['vertices'])[:,:2];jp0=np.asarray(h0['joints']);jp1=np.asarray(h1['joints']);d=np.linalg.norm(uv1-uv0,axis=1);report['camera_projection_changes'][f['camera']]=dict(visible=True,mesh_shift_median_px=float(np.median(d)),mesh_shift_p95_px=float(np.percentile(d,95)),wrist_pixel_delta=(jp1[0]-jp0[0]).tolist(),thumb_tip_pixel_delta=(jp1[4]-jp0[4]).tolist(),index_tip_pixel_delta=(jp1[8]-jp0[8]).tolist(),timestamp_delta_ms=f['sync_delta_ms'])
  im=cv2.imread(str(review/f['image']));points=np.r_[uv0,uv1];lo=np.maximum(0,np.floor(points.min(0)-35)).astype(int);hi=np.minimum(im.shape[1::-1],np.ceil(points.max(0)+35)).astype(int)
  if np.any(hi<=lo):continue
  panels=[]
  for title,uv in [('RAW',None),('BASELINE',uv0),('POSE LOCKED',uv1)]:
   lay=im.copy()
   if uv is not None:cv2.polylines(lay,np.rint(uv[faces]).astype(np.int32),True,(30,155,255) if a.side else (195,225,50),1,cv2.LINE_AA)
   blend=cv2.addWeighted(lay,.5,im,.5,0)[lo[1]:hi[1],lo[0]:hi[0]];hh,ww=blend.shape[:2];scale=min(480/ww,540/hh);resized=cv2.resize(blend,(int(ww*scale),int(hh*scale)));panel=np.zeros((580,480,3),np.uint8);panel[40:40+resized.shape[0],:resized.shape[1]]=resized;cv2.putText(panel,f'{f["camera"]} {title}',(10,27),0,.6,(255,255,255),1);panels.append(panel)
  cv2.imwrite(str(out/f'{f["camera"]}_crop.jpg'),np.concatenate(panels,axis=1))
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='markers'},indent=2));print('MOST INFLUENTIAL',sorted(rows,key=lambda x:-x['weighted_loss_reduction'])[:6]);print(out)
if __name__=='__main__':main()
