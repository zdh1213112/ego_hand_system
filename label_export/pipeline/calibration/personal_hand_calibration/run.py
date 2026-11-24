#!/usr/bin/env python3
"""Opt-in shared hand shape/attachment calibration diagnostic. No production export."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'label_export/pipeline/calibration/marker24_pilot'))
from run_pilot import raw24,import_mano,run_model,candidate_regions,discrete,stats,PROJECT,check_dorsal_reference,CameraCalibration
EXP=ROOT/'output/native_kb_training_20261111'
from context import load_context,raw24_context
NAMES=['WristM','WristIn','WristOut','HandOffset']+[f'Finger{f}{i}' for f in ['Thumb','Index','Middle','Ring','Pinky'] for i in range(1,5)]

def regions_for(model,beta,placement,hand,run):
 neutral=(torch.linalg.solve(model.hand_components.T,-model.hand_mean) if model.use_pca else -model.hand_mean).reshape(1,-1)
 with torch.no_grad():v,j,_=run(model,beta[None],torch.zeros((1,3),device=beta.device),neutral,torch.zeros((1,3),device=beta.device))
 v=v[0].cpu().numpy();j=j[0].cpu().numpy();f=model.faces_tensor.cpu().numpy();weights=model.lbs_weights.cpu().numpy()
 regions,_,palm=candidate_regions(v,j,f,weights,-1)
 n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
 palm_ids=np.flatnonzero((weights[f].mean(1)[:,0]>.25)&(n@palm*(-1)>.03)&(np.linalg.norm(v[f].mean(1)-j[0],axis=1)<.095))
 mapping={'palm':palm_ids}
 for k,finger in enumerate(['thumb','index','middle','ring','pinky']):
  for s,part in enumerate(['base','mcp','ip','tip'] if k==0 else ['mcp','pip','dip','tip']):mapping[finger+'_'+part]=regions[1+4*k+s]
 result=[]
 for name in NAMES[3:]:
  p=placement['hands'][hand][name]
  if p['surface']!='dorsal':raise ValueError('Unsupported marker surface')
  r=mapping[p['region']]
  if len(r)<1:raise ValueError('Empty attachment region')
  result.append(r)
 return result

def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--enable-personal-calibration',action='store_true',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--freeze-shape',action='store_true');ap.add_argument('--context',type=Path);ap.add_argument('--sample-stride',type=int,default=8);ap.add_argument('--pose-scope',choices=['holdout','sampled_sequence'],default='holdout');ap.add_argument('--frames',type=int,default=48);ap.add_argument('--steps',type=int,default=160);ap.add_argument('--subset',choices=['all','even','odd'],default='all');a=ap.parse_args()
 if a.frames<8 or a.steps<10:ap.error('Need >=8 frames and >=10 steps')
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(0);context=load_context(a.context);device=context['device'];to=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
 placement_path=context['placement'];placement=json.loads(placement_path.read_text());xyz,ts=raw24_context(context)
 mano=import_mano(PROJECT/'third_party/MANO');from mano.lbs import batch_rodrigues
 model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
 for p in model.parameters():p.requires_grad_(False)
 faces=model.faces_tensor.long();fn=faces.cpu().numpy()
 summary=dict(strategy='personal_shape_shared_attachments_staged_v1',stage='marker_only_identifiability_diagnostic',image_constraints_used=False,approved_for_training=False,baseline_modified=False,heldout_window=context['holdout'],pose_scope=a.pose_scope,subset=a.subset,shape_frozen=a.freeze_shape,placement_sha256=hashlib.sha256(placement_path.read_bytes()).hexdigest(),limits=dict(beta_delta=1.5,normal_offset_range_mm=[0,4],translation_delta_mm=8,orientation_axis_angle_delta_norm_rad=.06,finger_axis_angle_component_delta_rad=.12),caveats=['Marker-only fit does not establish glove contour accuracy or anatomical ground truth.','Shape and attachment ambiguity is checked via calibration subsets before image-constrained fitting.','Baseline initialization already used marker observations from validation frames; validation residuals are fitting diagnostics, not independent accuracy.','Normals offsets are fitted effective offsets, not independently measured glove thickness or fixed sphere-radius correction.'],hands={})
 for side,hand in enumerate(['left','right']):
  start=time.monotonic();z=np.load(context['baseline_fit']/f'track_{side}.npz');pairs=z['pair_indices'];q=xyz[side,pairs].copy();q[:,:,0]*=(-1 if side==0 else 1)
  finite=np.isfinite(q).all((1,2));speed=np.zeros(len(q));dq=np.linalg.norm(np.diff(q,axis=0),axis=-1);speed[1:]=np.nanmax(dq,axis=1);good=finite&(speed<.05)&(pairs%a.sample_stride==0);lo,hi=context['holdout']
  pool=np.flatnonzero(good&((pairs<lo)|(pairs>=hi)))
  # Disjoint alternating temporal blocks, not alternating adjacent samples.
  if a.subset!='all':pool=pool[(pairs[pool]//128)%2==(0 if a.subset=='even' else 1)]
  if len(pool)<8:raise ValueError('Insufficient usable calibration frames')
  # Greedy farthest-point sampling of articulation and hand orientation.
  features=np.c_[z['hand_pose_pca'][pool],z['global_orient'][pool]];features=(features-features.mean(0))/np.maximum(features.std(0),.15)
  chosen=[int(np.argmin(np.linalg.norm(features-np.median(features,axis=0),axis=1)))];distance=np.full(len(pool),np.inf)
  for _ in range(min(a.frames,len(pool))-1):
   distance=np.minimum(distance,((features-features[chosen[-1]])**2).sum(1));distance[chosen]=-1;chosen.append(int(np.argmax(distance)))
  cal=np.sort(pool[chosen]);test=np.flatnonzero(good&(pairs>=lo)&(pairs<hi)) if a.pose_scope=='holdout' else np.setdiff1d(np.flatnonzero(good),cal);assert len(test)>0;assert not set(pairs[cal])&set(pairs[test]);ids=np.r_[cal,test];nc=len(cal);nt=len(test);target=to(q[ids]);beta0=to(z['betas']);p0=to(z['hand_pose_pca'][ids]);o0=to(z['global_orient'][ids]);t0=to(z['translation'][ids])
  with torch.no_grad():v0,j0,_=run_model(model,beta0[None].expand(len(ids),-1),o0,p0,t0)
  assert (v0-to(z['vertices'][ids])).abs().max()<1e-5
  regions=regions_for(model,beta0,placement,hand,run_model);face_ids,bary=discrete(v0[:nc].cpu().numpy(),fn,q[cal,3:],regions);fi=torch.tensor(face_ids,device=device)
  hint=(context.get('dorsal_reference') or {}).get(hand) or placement['hands'][hand]['HandOffset']['suggestion'];refid=int(np.flatnonzero(pairs==int(hint['sync']))[0]);caldir=context['dataset']/'calibration'
  C=CameraCalibration.load(caldir/(hint['camera']+'.json'));C0=CameraCalibration.load(caldir/'camera0.json');origin=(C.T_base_camera[:3,3]-C0.T_base_camera[:3,3])@C0.T_base_camera[:3,:3]
  if side==0:origin[0]*=-1
  dorsal_check=check_dorsal_reference(z['vertices'][refid],fn,face_ids,origin)
  R0=batch_rodrigues(o0)
  wrist0=torch.median(torch.einsum('fji,fkj->fki',R0[:nc],target[:nc,:3]-j0[:nc,:1]),dim=0).values
  logits=torch.nn.Parameter(to(np.log(np.maximum(bary,1e-6))));offset=torch.nn.Parameter(torch.zeros(21,device=device));wrdelta=torch.nn.Parameter(torch.zeros_like(wrist0));db=torch.nn.Parameter(torch.zeros_like(beta0),requires_grad=not a.freeze_shape)
  weights=torch.ones(24,device=device);weights[:3]=.2
  def attachments(v,j,o):
   tri=v[:,faces[fi]];surface=(tri*logits.softmax(-1)[None,:,:,None]).sum(2);normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
   surface=surface+normal*offset[None,:,None]
   wrist=torch.einsum('fij,kj->fki',batch_rodrigues(o),wrist0+wrdelta)+j[:,:1]
   return torch.cat([wrist,surface],1)
  def markerloss(pred,tgt):
   e=torch.linalg.vector_norm(pred-tgt,dim=-1)/.006
   return ((torch.sqrt(1+e.square())-1)*weights).mean()
  def regattach():return .02*((logits.softmax(-1)-to(bary))/.25).square().mean()+.08*(offset/.004).square().mean()+.08*(wrdelta/.004).square().mean()
  def bound_attach():
   with torch.no_grad():offset.clamp_(0,.004);wrdelta.clamp_(-.004,.004);db.clamp_(-1.5,1.5)
  with torch.no_grad():initial=attachments(v0,j0,o0).clone()
  history=[]
  # Stage A: only fixed attachment barycentrics/effective normal offsets.
  opt=torch.optim.Adam([{'params':[logits],'lr':.025},{'params':[offset,wrdelta],'lr':.00015}])
  for step in range(a.steps):
   opt.zero_grad();loss=markerloss(attachments(v0[:nc],j0[:nc],o0[:nc]),target[:nc])+regattach();loss.backward();opt.step();bound_attach()
  history.append(dict(stage='A_attachment_only',loss=float(loss.detach())))
  # Stage B: shape changes with fixed attachment parameters; pose remains baseline.
  for p in [logits,offset,wrdelta]:p.requires_grad_(False)
  opt=torch.optim.Adam([db],lr=.015)
  for step in range(0 if a.freeze_shape else a.steps):
   opt.zero_grad();v,j,_=run_model(model,(beta0+db)[None].expand(nc,-1),o0[:nc],p0[:nc],t0[:nc]);loss=markerloss(attachments(v,j,o0[:nc]),target[:nc])+.04*db.square().mean();loss.backward();opt.step();bound_attach()
  history.append(dict(stage='B_shape_only',loss=float(loss.detach())))
  # Stage C: jointly refine after separate initialization, with bounded pose updates.
  dp=torch.nn.Parameter(torch.zeros_like(p0[:nc]));do=torch.nn.Parameter(torch.zeros_like(o0[:nc]));dt=torch.nn.Parameter(torch.zeros_like(t0[:nc]))
  for p in [logits,offset,wrdelta]:p.requires_grad_(True)
  opt=torch.optim.Adam([{'params':[db],'lr':.005},{'params':[logits],'lr':.01},{'params':[offset,wrdelta,dt],'lr':.0001},{'params':[dp,do],'lr':.001}])
  def clamp_pose(dp,do,dt):
   with torch.no_grad():
    dp.clamp_(-.12,.12);do.mul_(torch.clamp(.06/torch.linalg.vector_norm(do,dim=-1,keepdim=True).clamp_min(1e-9),max=1));dt.mul_(torch.clamp(.008/torch.linalg.vector_norm(dt,dim=-1,keepdim=True).clamp_min(1e-9),max=1))
  def poseprior(dp,do,dt):return .03*(dp/.12).square().mean()+.03*(do/.06).square().mean()+.03*(dt/.008).square().mean()
  for step in range(a.steps):
   opt.zero_grad();v,j,_=run_model(model,(beta0+db)[None].expand(nc,-1),o0[:nc]+do,p0[:nc]+dp,t0[:nc]+dt);loss=markerloss(attachments(v,j,o0[:nc]+do),target[:nc])+regattach()+.04*db.square().mean()+poseprior(dp,do,dt);loss.backward();opt.step();bound_attach();clamp_pose(dp,do,dt)
  history.append(dict(stage='C_bounded_joint',loss=float(loss.detach())))
  with torch.no_grad():vcal,jcal,_=run_model(model,(beta0+db)[None].expand(nc,-1),o0[:nc]+do,p0[:nc]+dp,t0[:nc]+dt);predcal=attachments(vcal,jcal,o0[:nc]+do).clone()
  # Freeze every shared parameter before held-out pose fitting.
  for p in [db,logits,offset,wrdelta]:p.requires_grad_(False)
  betaf=(beta0+db).detach();dpt=torch.nn.Parameter(torch.zeros_like(p0[nc:]));dot=torch.nn.Parameter(torch.zeros_like(o0[nc:]));dtt=torch.nn.Parameter(torch.zeros_like(t0[nc:]));opt=torch.optim.Adam([{'params':[dpt,dot],'lr':.001},{'params':[dtt],'lr':.0001}])
  for step in range(a.steps):
   opt.zero_grad();v,j,_=run_model(model,betaf[None].expand(nt,-1),o0[nc:]+dot,p0[nc:]+dpt,t0[nc:]+dtt);loss=markerloss(attachments(v,j,o0[nc:]+dot),target[nc:])+poseprior(dpt,dot,dtt)
   # Temporal differences only across nearby samples, never across long gaps.
   gap=torch.as_tensor(np.diff(pairs[test])<=16,device=device)
   if gap.any():loss=loss+.02*((dpt[1:]-dpt[:-1])[gap]/.12).square().mean()+.02*((dtt[1:]-dtt[:-1])[gap]/.008).square().mean()
   loss.backward();opt.step();clamp_pose(dpt,dot,dtt)
  with torch.no_grad():
   vt,jt,_=run_model(model,betaf[None].expand(nt,-1),o0[nc:]+dot,p0[nc:]+dpt,t0[nc:]+dtt);predt=attachments(vt,jt,o0[nc:]+dot);v=torch.cat([vcal,vt]);j=torch.cat([jcal,jt]);pred=torch.cat([predcal,predt]);error=(pred-target).norm(dim=-1)*1000;ref=(attachments(v0,j0,o0)-target).norm(dim=-1)*1000
  metrics=dict(calibration_frames=nc,validation_frames=nt,calibration_sync=pairs[cal].tolist(),validation_sync=pairs[test].tolist(),beta_initial=beta0.tolist(),beta_final=betaf.tolist(),beta_delta_norm=float(db.norm()),normal_offset_mm=(offset*1000).tolist(),calibration_marker_residual_mm=stats(error[:nc].cpu().numpy()),validation_marker_residual_mm=stats(error[nc:].cpu().numpy()),baseline_same_final_attachments_mm=stats(ref[nc:].cpu().numpy()),validation_vertex_movement_mm=stats(((vt-v0[nc:]).norm(dim=-1)*1000).cpu().numpy()),history=history,elapsed_seconds=time.monotonic()-start)
  # Report parameter limits as diagnostics, not evidence of correctness.
  metrics['dorsal_reference_check']=dorsal_check
  metrics['normal_offset_zero_count']=int((offset<=1e-7).sum())
  metrics['limits_reached']=dict(shape=int((db.abs()>=1.499).sum()),offset=int((offset.abs()>=.00399).sum()),validation_translation=int((dtt.norm(dim=-1)>=.00799).sum()),validation_orientation=int((dot.norm(dim=-1)>=.05999).sum()))
  summary['hands'][hand]=metrics
  cpu=lambda x:x.detach().cpu().numpy()
  np.savez_compressed(a.output/f'{hand}_pilot.npz',heldout_mask=(pairs[ids]>=lo)&(pairs[ids]<hi),pair_indices=pairs[ids],calibration_mask=np.arange(len(ids))<nc,vertices=cpu(v),joints=cpu(j),baseline_vertices=cpu(v0),baseline_joints=cpu(j0),target24=cpu(target),predicted24=cpu(pred),marker_error_mm=cpu(error),faces=fn,attachment_faces=face_ids,barycentric=cpu(logits.softmax(-1)),wrist_local=cpu(wrist0+wrdelta),normal_offsets_m=cpu(offset),betas=cpu(betaf),initial_betas=cpu(beta0),global_orient=cpu(o0+torch.cat([do,dot])),hand_pose_pca=cpu(p0+torch.cat([dp,dpt])),translation=cpu(t0+torch.cat([dt,dtt])),marker_names=np.array(NAMES),geometry_space='right_canonical')
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(hand,json.dumps(metrics),flush=True)
 print('DONE',a.output,flush=True)
if __name__=='__main__':main()
