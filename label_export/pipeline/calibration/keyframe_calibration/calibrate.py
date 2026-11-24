#!/usr/bin/env python3
"""Fit shared shape/attachments from selected quality keyframes, then drive motion using NOKOV only."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'label_export/pipeline/calibration/personal_hand_calibration'))
from run import PROJECT,import_mano,run_model,regions_for,discrete,stats
from context import load_context
from refine_contours import project
from calibration_profile import placement_signature, file_sha256
spec=importlib.util.spec_from_file_location('joint_helpers',ROOT/'label_export/pipeline/calibration/joint_pose_attachment/run.py');joint=importlib.util.module_from_spec(spec);spec.loader.exec_module(joint)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=180);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);c=load_context(a.context);selection=json.loads(a.selection.read_text());placement=json.loads(c['placement'].read_text());device=c['device'];torch.set_num_threads(4);torch.manual_seed(0);to=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
 mano=import_mano(PROJECT/'third_party/MANO');from mano.lbs import batch_rodrigues
 model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
 for p in model.parameters():p.requires_grad_(False)
 faces=model.faces_tensor.long();fn=faces.cpu().numpy();size=json.loads((c['dataset']/'manifest.json').read_text())['image_size'];summary=dict(strategy='quality_keyframe_personal_calibration_v1',quality_keyframe_calibration_enabled=True,shared_parameters_frozen=True,pose_images_used=False,approved_for_training=False,selection=str(a.selection.resolve()),source=str(a.source.resolve()),shape_sharing='one_beta_vector_per_physical_hand',noncalibration_pose_uses_images=False,shared_calibration_uses_only_selected_frames=True,hands={},limitations=['Quality selection uses projected marker guidance and is not independently measured pixel truth.','Baseline initialization already used NOKOV markers from all frames.','Only automatically selected good contours enter shared calibration; difficult contact/occlusion cases are not certified.','MANO shape family and fixed anatomical face regions limit exact glove-surface fit.'])
 shared_result=dict(schema='personal_glove_calibration_v1',approved_for_training=False,geometry_space='right_canonical',provenance=dict(source_dataset=str(c['dataset'].resolve()),source_context=str(a.context.resolve()),source_fit=str(a.output.resolve()),placement_signature=placement_signature(placement),mano_right_sha256=file_sha256(PROJECT/'models/mano/MANO_RIGHT.pkl'),visual_review_status='automatic_selection_pending_visual_review'),hands={})
 for side,hand in enumerate(['left','right']):
  if not (a.source/f'{hand}_pilot.npz').is_file(): continue
  begin=time.monotonic();z=np.load(a.source/f'{hand}_pilot.npz');base=np.load(c['baseline_fit']/f'track_{side}.npz');pairs=z['pair_indices'];lookup={int(f):i for i,f in enumerate(pairs)};bi={int(f):i for i,f in enumerate(base['pair_indices'])};ids=np.array([bi[int(f)] for f in pairs]);chosen=selection['report']['hands'][hand]['calibration_sync'];cal=np.array([lookup[int(f)] for f in chosen]);other=np.setdiff1d(np.arange(len(pairs)),cal)
  if len(cal)<6:raise ValueError(f'{hand}: insufficient diverse calibration keyframes')
  trainkeys={(r['camera'],r['sync']) for r in selection['report']['hands'][hand]['calibration_views']};good=[dict(r) for r in selection['records'] if r['side']==side and r['accepted_quality']];train=[r for r in good if (r['camera'],r['sync']) in trainkeys];test=[r for r in good if r['sync'] not in chosen]
  assert all(r['sync'] not in chosen for r in test)
  # Retain calibration and noncalibration frames in a single stable order.
  beta0=to(base['betas']);o0=to(base['global_orient'][ids]);p0=to(base['hand_pose_pca'][ids]);t0=to(base['translation'][ids]);target=to(z['target24']);n=len(pairs)
  with torch.no_grad():v0,j0,_=run_model(model,beta0[None].expand(n,-1),o0,p0,t0)
  regions=regions_for(model,beta0,placement,hand,run_model);faceids,bary=discrete(v0[cal].cpu().numpy(),fn,z['target24'][cal,3:],regions);fi=torch.tensor(faceids,device=device);R0=batch_rodrigues(o0);wr0=torch.median(torch.einsum('fji,fkj->fki',R0[cal],target[cal,:3]-j0[cal,:1]),dim=0).values
  logits=torch.nn.Parameter(to(np.log(bary)));offset=torch.nn.Parameter(torch.zeros(21,device=device));wr=torch.nn.Parameter(wr0.clone());db=torch.nn.Parameter(torch.zeros_like(beta0));dp=torch.nn.Parameter(torch.zeros_like(p0));do=torch.nn.Parameter(torch.zeros_like(o0));dt=torch.nn.Parameter(torch.zeros_like(t0));weights=to([.2]*3+[1]*21)
  def geometry():return run_model(model,(beta0+db)[None].expand(n,-1),o0+do,p0+dp,t0+dt)[:2]
  def markers(v,j):
   tri=v[:,faces[fi]];normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1);surface=(tri*logits.softmax(-1)[None,:,:,None]).sum(2)+normal*offset[None,:,None];wrist=torch.einsum('fij,kj->fki',batch_rodrigues(o0+do),wr)+j[:,:1];return torch.cat([wrist,surface],1)
  def ml(v,j,indices):
   e=(markers(v,j)[indices]-target[indices]).norm(dim=-1)/.006;return ((torch.sqrt(1+e.square())-1)*weights).mean()
  def attachprior():return .04*((logits.softmax(-1)-to(bary))/.25).square().mean()+.05*(offset/.004).square().mean()+.05*((wr-wr0)/.004).square().mean()
  def poseprior(indices):return .03*(dp[indices]/.12).square().mean()+.03*(do[indices]/.06).square().mean()+.03*(dt[indices]/.008).square().mean()
  def clamp():
   with torch.no_grad():
    offset.clamp_(0,.004);wr.copy_(wr0+(wr-wr0).clamp(-.004,.004));db.clamp_(-1.5,1.5);dp.clamp_(-.12,.12);do.mul_(torch.clamp(.06/do.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1));dt.mul_(torch.clamp(.008/dt.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1))
  for r in good:
   # Original extractor's wrist cut is retained, independent of current fitting changes.
   r['_wrist_joints']=project(to(z['joints'][lookup[r['sync']]]).double(),r,side).cpu().numpy().tolist()
   r['boundary_vertices']=list(range(778))
  trainbatch=joint.ContourBatch(train,lookup,side,device);trainbatch.refresh(v0,j0,fn,size);testbatch=joint.ContourBatch(test,lookup,side,device) if test else None
  if testbatch:testbatch.refresh(v0,j0,fn,size)
  with torch.no_grad():before=dict(calibration_views=trainbatch.summary(v0),validation_views=testbatch.summary(v0) if testbatch else None,marker_mm=stats(((markers(v0,j0)-target).norm(dim=-1)[other]*1000).cpu().numpy()))
  # Stage A: attachments only; geometry and shape fixed.
  for p in [db,dp,do,dt]:p.requires_grad_(False)
  opt=torch.optim.Adam([{'params':[logits],'lr':.02},{'params':[offset,wr],'lr':.00015}])
  for step in range(a.steps):
   opt.zero_grad();loss=ml(v0,j0,cal)+attachprior();loss.backward();opt.step();clamp()
  print(hand,'A attachments',float(loss.detach()),flush=True)
  # Stage B: shared shape uses only selected calibrated views, with fixed attachments and motion.
  for p in [logits,offset,wr]:p.requires_grad_(False)
  db.requires_grad_(True);opt=torch.optim.Adam([db],lr=.01)
  for step in range(a.steps):
   opt.zero_grad();v,j=geometry();loss=ml(v,j,cal)+.5*trainbatch.loss(v)+.06*db.square().mean();loss.backward();opt.step();clamp()
  print(hand,'B shape',float(loss.detach()),flush=True)
  # Stage C: refine shared shape/attachments and only selected frames' motion.
  for p in [logits,offset,wr,dp,do,dt]:p.requires_grad_(True)
  opt=torch.optim.Adam([{'params':[db],'lr':.004},{'params':[logits],'lr':.01},{'params':[offset,wr,dt],'lr':.0001},{'params':[dp,do],'lr':.0015}])
  for step in range(a.steps):
   opt.zero_grad();v,j=geometry();loss=ml(v,j,cal)+.5*trainbatch.loss(v)+.06*db.square().mean()+attachprior()+poseprior(cal);loss.backward();opt.step();clamp()
   if step==a.steps//2:
    with torch.no_grad():trainbatch.refresh(v,j,fn,size)
  print(hand,'C keyframe calibration',float(loss.detach()),flush=True)
  # Record every shared parameter before the entire rest-of-sequence motion stage.
  before_freeze={k:p.detach().clone() for k,p in [('beta',db),('barycentric',logits),('offset',offset),('wrist',wr)]}
  for p in [db,logits,offset,wr]:p.requires_grad_(False)
  opt=torch.optim.Adam([{'params':[dp,do],'lr':.0015},{'params':[dt],'lr':.00015}]);order=np.argsort(pairs[other]);oi=other[order];close=np.diff(pairs[oi])<=16;left=torch.tensor(oi[:-1][close],device=device);right=torch.tensor(oi[1:][close],device=device)
  for step in range(a.steps):
   opt.zero_grad();v,j=geometry();loss=ml(v,j,other)+poseprior(other)
   if len(left):loss=loss+.01*((dp[right]-dp[left])/.12).square().mean()+.01*((dt[right]-dt[left])/.008).square().mean()
   loss.backward();opt.step();clamp()
  for k,p in [('beta',db),('barycentric',logits),('offset',offset),('wrist',wr)]:assert torch.equal(before_freeze[k],p.detach()),'Shared calibration changed during motion fit'
  with torch.no_grad():
   v,j=geometry();pred=markers(v,j);err=(pred-target).norm(dim=-1)*1000;after=dict(calibration_views=trainbatch.summary(v),validation_views=testbatch.summary(v) if testbatch else None,marker_mm=stats(err[other].cpu().numpy()))
  cpu=lambda t:t.detach().cpu().numpy();out={k:z[k] for k in z.files};out.update(betas=cpu(beta0+db),vertices=cpu(v),joints=cpu(j),baseline_vertices=cpu(v0),baseline_joints=cpu(j0),predicted24=cpu(pred),marker_error_mm=cpu(err),global_orient=cpu(o0+do),hand_pose_pca=cpu(p0+dp),translation=cpu(t0+dt),attachment_faces=faceids,barycentric=cpu(logits.softmax(-1)),normal_offsets_m=cpu(offset),wrist_local=cpu(wr),calibration_mask=np.isin(pairs,chosen),calibration_frame_has_image_constraints=np.isin(pairs,chosen));np.savez_compressed(a.output/f'{hand}_pilot.npz',**out)
  shared_result['hands'][hand]=dict(betas=out['betas'].tolist(),attachment_faces=faceids.tolist(),barycentric=out['barycentric'].tolist(),normal_offsets_m=out['normal_offsets_m'].tolist(),wrist_local_m=out['wrist_local'].tolist(),calibration_sync=chosen)
  summary['hands'][hand]=dict(calibration_sync=chosen,calibration_views=len(train),validation_views=len(test),noncalibration_frames=len(other),shared_calibration_frozen_during_noncalibration_motion=True,noncalibration_images_used_for_fitting=False,before=before,after=after,beta_delta=cpu(db).tolist(),limits_reached=dict(beta=int((db.abs()>=1.499).sum()),finger_parameters=int((dp.abs()>=.11999).sum()),root_translation=int((dt.norm(dim=-1)>=.00799).sum())),elapsed_seconds=time.monotonic()-begin)
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'shared_calibration.json').write_text(json.dumps(shared_result,indent=2));print(hand,'DONE',summary['hands'][hand]['elapsed_seconds'],flush=True)
 print('DONE',a.output,flush=True)
if __name__=='__main__':main()
