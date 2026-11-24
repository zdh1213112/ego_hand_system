#!/usr/bin/env python3
"""Opt-in 24-marker shared-attachment pilot. Never changes baseline exports."""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parent/'ego_hand_system'
sys.path[:0]=[str(PROJECT/'scripts'),str(ROOT.parent/'nokov_ego_hand_validation/integration/pipeline')]
from fit_mano_sequence import import_mano,run_model
from fit_surface_attachments_guarded import candidate_regions
from align_nokov_cam0 import load_rigid,load_markers,interpolate_points
from ego_data.calibration import CameraCalibration

def stats(a):
 a=np.asarray(a);a=a[np.isfinite(a)]
 return dict(count=int(a.size),median=float(np.median(a)) if a.size else None,p95=float(np.percentile(a,95)) if a.size else None)

def raw24(exp):
 a=json.loads((exp/'fourview_192916/alignment_180/fourview_alignment.json').read_text());seed=json.loads(Path(a['seed_alignment']).read_text());dataset=Path(a['source_dataset']);nokov=Path(a['source_nokov'])
 with (dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
 m=seed['time_mapping'];t=np.array([int(r['camera0_timestamp_ns']) for r in rows],np.int64);target=m['nokov_origin_timestamp_raw']+((t-m['ego_origin_timestamp_ns'])*1e-9*m.get('a',1)+m['b_s'])/m['nokov_seconds_per_timestamp_unit']
 rt,head,q=load_rigid(nokov/'nokov_rigid_bodies.csv');head,ok=interpolate_points(rt,head,target);rot=np.full((len(rows),3,3),np.nan);ok&=(target>=rt[0])&(target<=rt[-1]);rot[ok]=Slerp(rt,Rotation.from_quat(q))(target[ok]).as_matrix();T=np.asarray(a['T_cam0_head']);hands=[]
 for name in ['left_hand_4mm','right_hand_4mm']:
  ts,p=load_markers(nokov/'nokov_markers.csv',name);p,_=interpolate_points(ts,p,target);p=np.einsum('fji,fkj->fki',rot,p-head[:,None]);hands.append(p@T[:3,:3].T+T[:3,3])
 xyz=np.asarray(hands);expected=np.load(exp/'fourview_192916/alignment_180/aligned_nokov21.npz')['joints_base_m'];C=CameraCalibration.load(dataset/'calibration/camera0.json');base=xyz@C.T_base_camera[:3,:3].T+C.T_base_camera[:3,3]
 assert np.nanmax(abs(base[:,:,[0]+list(range(4,24))]-expected))<1e-5
 return xyz,t

def discrete(v,faces,target,regions):
 ids=[];weights=[]
 for j,choices in enumerate(regions):
  q=v[:,faces[choices]].astype(float);y=target[:,j];q0=q[:,:,0];a=q[:,:,1]-q0;b=q[:,:,2]-q0;r=y[:,None]-q0
  aa=(a*a).sum((0,2));ab=(a*b).sum((0,2));bb=(b*b).sum((0,2));ar=(a*r).sum((0,2));br=(b*r).sum((0,2));den=np.maximum(aa*bb-ab*ab,1e-25)
  u=(bb*ar-ab*br)/den;w=(aa*br-ab*ar)/den;proposals=[np.stack([1-u-w,u,w],-1)]
  for start,end in [(0,1),(1,2),(2,0)]:
   edge=q[:,:,end]-q[:,:,start];t=np.clip((edge*(y[:,None]-q[:,:,start])).sum((0,2))/np.maximum((edge*edge).sum((0,2)),1e-20),0,1);p=np.zeros((len(choices),3));p[:,start]=1-t;p[:,end]=t;proposals.append(p)
  costs=[]
  for p in proposals:
   residual=np.linalg.norm((q*p[None,:,:,None]).sum(2)-y[:,None],axis=-1);cost=np.mean(np.sqrt(residual**2+.006**2)-.006,0);cost[(p<0).any(1)]=np.inf;costs.append(cost)
  k,f=np.unravel_index(np.argmin(costs),np.asarray(costs).shape);ids.append(choices[f]);p=np.maximum(proposals[k][f],1e-6);weights.append(p/p.sum())
 return np.asarray(ids),np.asarray(weights,np.float32)

def check_dorsal_reference(vertices, faces, attachment_faces, camera_origin):
    """Check the known dorsal reference photo; never infer dorsal from a bare sign."""
    tri = vertices[faces[attachment_faces]]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rays = camera_origin - tri.mean(1)
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    facing = (normals * rays).sum(1)
    # The reference photos explicitly show the dorsal palm and finger bases.
    # Thumb visibility is tangential, so do not force its normal toward the camera.
    if facing[0] <= 0.1 or np.median(facing[[5, 9, 13, 17]]) <= 0.1:
        raise ValueError('Attachment region faces away from the confirmed dorsal photo; likely reversed hemisphere')
    return dict(palm_facing=float(facing[0]), finger_base_median_facing=float(np.median(facing[[5, 9, 13, 17]])))

def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--iterations',type=int,default=240);ap.add_argument('--surface-hemisphere',type=int,choices=(-1,1),default=-1);ap.add_argument('--freeze-finger-pose',action='store_true',help='Keep baseline finger articulation exactly fixed in both calibration and evaluation fitting');ap.add_argument('--fit-all-markers',action='store_true',help='Use all 24 markers in evaluation-segment pose fitting; errors are fitting residuals, not held-out accuracy');ap.add_argument('--wrist-weight',type=float,default=.2);ap.add_argument('--calibration-frames',type=int,default=80);ap.add_argument('--enable-marker24',action='store_true',required=True);a=ap.parse_args()
 if a.iterations<1 or a.calibration_frames<3 or not 0<=a.wrist_weight<=1:ap.error('invalid iteration/frame count or wrist weight')
 if a.freeze_finger_pose and not a.fit_all_markers:ap.error('--freeze-finger-pose requires --fit-all-markers')
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(0);device='cuda';exp=ROOT/'output/native_kb_training_20261111';placement_path=exp/'marker24_calibration_input/placement_user_confirmed.json';placement=json.loads(placement_path.read_text());xyz,stamps=raw24(exp);np.savez_compressed(a.output/'aligned24_cam0.npz',points=xyz,timestamp_ns=stamps)
 mano=import_mano(PROJECT/'third_party/MANO');from mano.lbs import batch_rodrigues
 model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
 for p in model.parameters():p.requires_grad_(False)
 faces=model.faces_tensor.long();fn=faces.cpu().numpy();summary={'strategy':'shared24_surface_and_wrist_local_pilot_v1','marker_normal_offset_mm':0,'surface_hemisphere':a.surface_hemisphere,'finger_pose_frozen':a.freeze_finger_pose,'shape_frozen':True,'wrist_weight':a.wrist_weight,'validation_caveat':'Validation poses start from the existing baseline, which used all 21 marker targets, including tips. Tip holdout applies only to refinement, not to baseline initialization.','temporal_regularization':False,'placement_sha256':hashlib.sha256(placement_path.read_bytes()).hexdigest(),'heldout_time_window':[512,704],'validation_excluded_pose_fit_markers':[] if a.fit_all_markers else [7,11,15,19,23],'fit_all_markers':a.fit_all_markers,'baseline_unchanged':True,'approved_for_full_export':False,'hands':{}}
 for side,hand in enumerate(['left','right']):
  begin=time.monotonic();z=np.load(exp/f'fourview_192916/native_labels/mano_fit_full_cam01/track_{side}.npz');pairs=z['pair_indices'];valid=np.isfinite(xyz[side,pairs]).all((1,2));pool=np.flatnonzero(valid&(pairs%8==0));calpool=pool[(pairs[pool]<512)|(pairs[pool]>=704)];cal=calpool[np.unique(np.linspace(0,len(calpool)-1,min(a.calibration_frames,len(calpool))).astype(int))];test=pool[(pairs[pool]>=512)&(pairs[pool]<704)];ids=np.r_[cal,test];ncal=len(cal);sync=pairs[ids];target_np=xyz[side,sync].copy();target_np[:,:,0]*=(-1 if side==0 else 1)
  to=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
  beta=to(z['betas'][None]);pose0=to(z['hand_pose_pca'][ids]);orient0=to(z['global_orient'][ids]);trans0=to(z['translation'][ids]);target=to(target_np)
  with torch.no_grad():v0,j0,_=run_model(model,beta.expand(len(ids),-1),orient0,pose0,trans0)
  assert float((v0-to(z['vertices'][ids])).abs().max())<1e-5
  # Neutral PCA coefficients producing zero hand articulation, preserving subject shape.
  neutral=(torch.linalg.solve(model.hand_components.T,-model.hand_mean) if model.use_pca else -model.hand_mean).reshape(1,-1)
  with torch.no_grad():rv,rj,_=run_model(model,beta,torch.zeros((1,3),device=device),neutral,torch.zeros((1,3),device=device))
  rvn,rjn=rv[0].cpu().numpy(),rj[0].cpu().numpy();regions,_,palm=candidate_regions(rvn,rjn,fn,model.lbs_weights.cpu().numpy(),a.surface_hemisphere)
  centres=rvn[fn].mean(1);normal=np.cross(rvn[fn[:,1]]-rvn[fn[:,0]],rvn[fn[:,2]]-rvn[fn[:,0]]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);palmregion=np.flatnonzero((model.lbs_weights.cpu().numpy()[fn].mean(1)[:,0]>.25)&(normal@palm*a.surface_hemisphere>.03)&(np.linalg.norm(centres-rjn[0],axis=1)<.095))
  if len(palmregion)<3:raise ValueError('empty palm region')
  joint_regions=regions
  names=['WristM','WristIn','WristOut','HandOffset']+[f'Finger{finger}{k}' for finger in ['Thumb','Index','Middle','Ring','Pinky'] for k in range(1,5)]
  region_map={'palm':palmregion}
  for finger_id,finger in enumerate(['thumb','index','middle','ring','pinky']):
   for slot,part in enumerate(['base','mcp','ip','tip'] if finger=='thumb' else ['mcp','pip','dip','tip']):region_map[finger+'_'+part]=joint_regions[1+finger_id*4+slot]
  regions=[]
  for name in names[3:]:
   region=placement['hands'][hand][name]['region']
   if region not in region_map:raise ValueError('Unsupported confirmed region in this pilot: '+region)
   regions.append(region_map[region])
  if any(placement['hands'][hand][n]['surface']!='dorsal' for n in names[3:]):raise ValueError('Pilot requires current dorsal-region configuration')
  face_ids,bary=discrete(v0[:ncal].cpu().numpy(),fn,target_np[:ncal,3:],regions)
  R0=batch_rodrigues(orient0);wrist_local=torch.median(torch.einsum('fji,fkj->fki',R0[:ncal],target[:ncal,:3]-j0[:ncal,:1]),dim=0).values
  initial_faces=face_ids.copy();initial_bary=bary.copy();initial_wrist=wrist_local.detach().clone()
  hint=placement['hands'][hand]['HandOffset']['suggestion'];reference_id=np.flatnonzero(pairs==int(hint['sync']))
  if len(reference_id)!=1:raise ValueError('Missing confirmed dorsal reference frame')
  calibration_dir=exp/'real_dual/pair_20261111_192916/ego/normalized_multiview/calibration'
  Cref=CameraCalibration.load(calibration_dir/(hint['camera']+'.json'));Czero=CameraCalibration.load(calibration_dir/'camera0.json')
  origin=(Cref.T_base_camera[:3,3]-Czero.T_base_camera[:3,3])@Czero.T_base_camera[:3,:3]
  if side==0:origin[0]*=-1
  reference_vertices=z['vertices'][reference_id[0]]
  initial_direction_check=check_dorsal_reference(reference_vertices,fn,face_ids,origin)
  def predict(v,j,o,fi,bw,wr):
   surface=(v[:,faces[fi]]*bw[None,:,:,None]).sum(2)
   wrist=torch.einsum('fij,kj->fki',batch_rodrigues(o),wr)+j[:,:1]
   return torch.cat([wrist,surface],1)
  fi=torch.tensor(face_ids,device=device);bw=to(bary)
  with torch.no_grad():initial_pred=predict(v0,j0,orient0,fi,bw,wrist_local)
  pose=pose0[:ncal].clone().requires_grad_(not a.freeze_finger_pose);ori=orient0[:ncal].clone().requires_grad_();tr=trans0[:ncal].clone().requires_grad_();logits=to(np.log(bary)).requires_grad_();wr=wrist_local.clone().requires_grad_()
  weights=torch.ones(24,device=device);weights[:3]=a.wrist_weight
  def loss(v,j,pred,tgt,po,po0,jbase,weights):
   err=torch.linalg.vector_norm(pred-tgt,dim=-1)/.006
   data=((torch.sqrt(1+err.square())-1)*weights).sum()/weights.expand_as(err).sum()
   anchor=.08*((j-jbase)/.010).square().mean();prior=.02*(po-po0).square().mean()+.001*po.square().mean()
   return data+anchor+prior
  optimizer=torch.optim.Adam([{'params':[pose,ori],'lr':.008},{'params':[tr,wr],'lr':.0005},{'params':[logits],'lr':.04}])
  for it in range(a.iterations):
   optimizer.zero_grad();v,j,_=run_model(model,beta.expand(ncal,-1),ori,pose,tr);pred=predict(v,j,ori,fi,logits.softmax(-1),wr);obj=loss(v,j,pred,target[:ncal],pose,pose0[:ncal],j0[:ncal],weights)+.02*((wr-initial_wrist)/.01).square().mean()
   if not torch.isfinite(obj):raise ValueError('nonfinite calibration')
   obj.backward();optimizer.step()
   if it in [a.iterations//3,2*a.iterations//3]:
    with torch.no_grad():vv,_,_=run_model(model,beta.expand(ncal,-1),ori,pose,tr)
    face_ids,bary=discrete(vv.cpu().numpy(),fn,target_np[:ncal,3:],regions);fi=torch.tensor(face_ids,device=device)
    with torch.no_grad():logits.copy_(to(np.log(bary)))
    optimizer.state.pop(logits,None)
   if it%60==0:print(hand,'calibrate',it,float(obj),flush=True)
  with torch.no_grad():vcal,jcal,_=run_model(model,beta.expand(ncal,-1),ori,pose,tr);final_bary=logits.softmax(-1).detach();final_wrist=wr.detach();calpred=predict(vcal,jcal,ori,fi,final_bary,final_wrist)
  # Freeze attachments on disjoint timestamps; five tips are withheld from validation pose optimization.
  nt=len(test);pt=pose0[ncal:].clone().requires_grad_(not a.freeze_finger_pose);ot=orient0[ncal:].clone().requires_grad_();tt=trans0[ncal:].clone().requires_grad_();tw=weights.clone()
  if not a.fit_all_markers:tw[[7,11,15,19,23]]=0
  opt=torch.optim.Adam([{'params':[pt,ot],'lr':.008},{'params':[tt],'lr':.0005}])
  for it in range(a.iterations):
   opt.zero_grad();v,j,_=run_model(model,beta.expand(nt,-1),ot,pt,tt);pred=predict(v,j,ot,fi,final_bary,final_wrist);obj=loss(v,j,pred,target[ncal:],pt,pose0[ncal:],j0[ncal:],tw);obj.backward();opt.step()
   if it%80==0:print(hand,'validation-pose',it,float(obj),flush=True)
  with torch.no_grad():vt,jt,_=run_model(model,beta.expand(nt,-1),ot,pt,tt);tp=predict(vt,jt,ot,fi,final_bary,final_wrist);reference_pred=predict(v0,j0,orient0,fi,final_bary,final_wrist)
  vv=torch.cat([vcal,vt]);jj=torch.cat([jcal,jt]);pp=torch.cat([calpred,tp]);err=((pp-target).norm(dim=-1)*1000).cpu().numpy();err_initial=((initial_pred-target).norm(dim=-1)*1000).cpu().numpy();err_ref=((reference_pred-target).norm(dim=-1)*1000).cpu().numpy();jshift=((jj-j0).norm(dim=-1)*1000).cpu().numpy();vshift=((vv-v0).norm(dim=-1)*1000).cpu().numpy();tip=[7,11,15,19,23]
  metrics=dict(calibration_frames=ncal,validation_frames=nt,initial_attachment_baseline_mm=stats(err_initial[ncal:]),final_attachment_baseline_mm=stats(err_ref[ncal:]),candidate_validation_mm=stats(err[ncal:]),heldout_tips_initial_baseline_mm=stats(err_initial[ncal:,tip]),heldout_tips_final_attachment_baseline_mm=stats(err_ref[ncal:,tip]),heldout_tips_candidate_mm=stats(err[ncal:,tip]),joint_shift_mm=stats(jshift[ncal:]),vertex_shift_mm=stats(vshift[ncal:]),per_marker_validation_mm={n:stats(err[ncal:,i]) for i,n in enumerate(names)},elapsed_s=time.monotonic()-begin)
  if a.fit_all_markers:
   for key in list(metrics):
    if key.startswith('heldout_tips_'):metrics[key.replace('heldout_tips_','fitted_tips_')]=metrics.pop(key)
  metrics['dorsal_reference_initial']=initial_direction_check
  metrics['dorsal_reference_final']=check_dorsal_reference(reference_vertices,fn,fi.cpu().numpy(),origin)
  summary['hands'][hand]=metrics
  np.savez_compressed(a.output/f'{hand}_pilot.npz',pair_indices=sync,calibration_mask=np.arange(len(ids))<ncal,vertices=vv.cpu().numpy(),joints=jj.cpu().numpy(),baseline_vertices=v0.cpu().numpy(),baseline_joints=j0.cpu().numpy(),target24=target_np,predicted24=pp.cpu().numpy(),marker_error_mm=err,faces=fn,attachment_faces=fi.cpu().numpy(),barycentric=final_bary.cpu().numpy(),wrist_local=final_wrist.cpu().numpy(),initial_attachment_faces=initial_faces,initial_barycentric=initial_bary,initial_wrist_local=initial_wrist.cpu().numpy(),betas=z['betas'],global_orient=torch.cat([ori.detach(),ot.detach()]).cpu().numpy(),hand_pose_pca=torch.cat([pose.detach(),pt.detach()]).cpu().numpy(),translation=torch.cat([tr.detach(),tt.detach()]).cpu().numpy(),marker_names=np.array(names),geometry_space='right_canonical',marker_normal_offset_mm=0)
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(hand,json.dumps(metrics),flush=True)
 print('DONE',a.output,flush=True)

if __name__=='__main__':main()
