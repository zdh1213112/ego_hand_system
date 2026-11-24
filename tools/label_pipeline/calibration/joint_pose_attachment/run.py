#!/usr/bin/env python3
"""Joint per-frame pose/root and shared attachment refinement with fixed shared shape."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/personal_hand_calibration'))
from run import EXP,PROJECT,import_mano,run_model,stats
from context import load_context
from ablate_geometry import split_views
from refine_contours import project

class ContourBatch:
 def __init__(self,records,lookup,side,device):
  self.records=records;self.lookup=lookup;self.side=side;self.device=device;self.to=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
  self.frame=torch.tensor([lookup[r['sync']] for r in records],device=device);self.M=self.to([np.asarray(r['T_base_cam0'])[:3,:3].T@np.asarray(r['T_base_camera'])[:3,:3] for r in records]);self.b=self.to([(np.asarray(r['T_base_cam0'])[:3,3]-np.asarray(r['T_base_camera'])[:3,3])@np.asarray(r['T_base_camera'])[:3,:3] for r in records]);self.K=self.to([r['K'] for r in records]);self.D=self.to([np.asarray(r['D']).reshape(4) for r in records]);self.mirror=self.to([-1 if side==0 else 1,1,1]);self.set_indices([r['boundary_vertices'] for r in records])
  size=max(len(r['points']) for r in records);gt=np.zeros((len(records),size,2),np.float32);mask=np.zeros((len(records),size),bool)
  for i,r in enumerate(records):gt[i,:len(r['points'])]=r['points'];mask[i,:len(r['points'])]=True
  self.gt=self.to(gt);self.gt_valid=torch.tensor(mask,device=device)
  counts={x:sum(r['sync']==x for r in records) for x in set(r['sync'] for r in records)};self.weights=self.to([1/counts[r['sync']] for r in records]);self.weights/=self.weights.sum()
 def set_indices(self,lists):
  size=max(map(len,lists));ids=np.zeros((len(lists),size),np.int64);mask=np.zeros((len(lists),size),bool)
  for i,lst in enumerate(lists):ids[i,:len(lst)]=lst;mask[i,:len(lst)]=True
  self.ids=torch.tensor(ids,device=self.device);self.valid=torch.tensor(mask,device=self.device)
 def project(self,v):
  p=torch.einsum('rvj,rjk->rvk',v[self.frame]*self.mirror,self.M)+self.b[:,None];xy=p[:,:,:2]/p[:,:,2:].clamp_min(1e-4);radius=xy.norm(dim=-1).clamp_min(1e-9);theta=torch.atan(radius);t2=theta.square();d=self.D;td=theta*(1+d[:,0,None]*t2+d[:,1,None]*t2**2+d[:,2,None]*t2**3+d[:,3,None]*t2**4);uv=xy*(td/radius)[:,:,None];return uv*torch.stack([self.K[:,0,0],self.K[:,1,1]],-1)[:,None]+self.K[:,:2,2][:,None]
 def errors(self,v):
  uv=self.project(v);pred=uv[torch.arange(len(uv),device=self.device)[:,None],self.ids];distance=torch.cdist(pred,self.gt);a=distance.masked_fill(~self.gt_valid[:,None],1e6).min(-1).values;b=distance.masked_fill(~self.valid[:,:,None],1e6).min(1).values;return a,b
 def loss(self,v):
  a,b=self.errors(v);robust=lambda x:torch.sqrt(1+(x/5).square())-1
  per=.5*((robust(a)*self.valid).sum(1)/self.valid.sum(1)+(robust(b)*self.gt_valid).sum(1)/self.gt_valid.sum(1));return (per*self.weights).sum()
 def summary(self,v):
  a,b=self.errors(v);per=((a*self.valid).sum(1)+(b*self.gt_valid).sum(1))/(self.valid.sum(1)+self.gt_valid.sum(1));values=per.detach().cpu().numpy();return dict(views=len(values),mean_px=float(values.mean()),per_view=[dict(camera=r['camera'],sync=r['sync'],distance_px=float(e)) for r,e in zip(self.records,values)])
 def refresh(self,v,j,faces,size):
  width,height=size;uv=self.project(v).detach().cpu().numpy();indices=[]
  for r,p in zip(self.records,uv):
   mask=np.zeros((height,width),np.uint8)
   for tri in np.rint(p[faces]).astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
   contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);edge=np.zeros_like(mask);cv2.drawContours(edge,contours,-1,1,1);d=cv2.distanceTransform(1-edge,cv2.DIST_L2,3)
   # Keep original wrist cut fixed throughout the experiment.
   jp=np.asarray(r['_wrist_joints']);axis=jp[[5,9,13,17]].mean(0)-jp[0];axis/=max(np.linalg.norm(axis),1e-9);q=np.rint(p).astype(int);inside=(q[:,0]>=0)&(q[:,0]<width)&(q[:,1]>=0)&(q[:,1]<height);ids=np.flatnonzero(inside);ids=ids[(d[q[ids,1],q[ids,0]]<2.5)&((p[ids]-jp[0])@axis>8)]
   indices.append(ids.tolist() if len(ids)>=15 else r['boundary_vertices'])
  self.set_indices(indices)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--observations',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=160);ap.add_argument('--view-policy',choices=['leave_one_out','all'],default='leave_one_out');ap.add_argument('--no-boundary-refresh',action='store_true');a=ap.parse_args()
 if a.steps<1:ap.error('steps must be positive')
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);c=load_context(a.context);torch.set_num_threads(4);torch.manual_seed(0);device=c['device'];to=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device);size=json.loads((c['dataset']/'manifest.json').read_text())['image_size'];obs=json.loads(a.observations.read_text());mano=import_mano(PROJECT/'third_party/MANO');from mano.lbs import batch_rodrigues
 model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
 for q in model.parameters():q.requires_grad_(False)
 summary=dict(strategy='joint_pose_and_shared_attachments_v1',joint_pose_attachment_optimization_enabled=True,shape_frozen=True,approved_for_training=False,view_policy=a.view_policy,source=str(a.source.resolve()),observations=str(a.observations.resolve()),hands={},limits=dict(translation_increment_mm=8,orientation_increment_rad=.06,finger_component_increment_rad=.12,normal_offset_mm=[0,4],wrist_local_increment_mm=2),limitations=['Automatic color masks are proposals, not independent image truth.','Shape remains one fixed vector per physical hand. Attachments are shared across time and cameras.','Shared attachment gradients use calibration timestamps only. Per-frame motion uses NOKOV and fitting cameras.','Camera holdout tests incremental generalization; earlier initialization may already have used these images.','Pose limits and silhouette vertex approximation can leave residual error.','Normal offsets are effective learned offsets, no fixed 2mm subtraction.'])
 for side,hand in enumerate(['left','right']):
  begin=time.monotonic();z=np.load(a.source/f'{hand}_pilot.npz');lookup={int(f):i for i,f in enumerate(z['pair_indices'])};records=[dict(r) for r in obs['records'] if r['side']==side];fit,held,skipped=split_views(records)
  if a.view_policy=='all':fit=fit+held;held=[]
  if not fit:raise ValueError('No multiview-supported frames')
  eligible=np.array(sorted({lookup[r['sync']] for r in fit}));cal=z['calibration_mask'].astype(bool);shared=eligible[cal[eligible]];other=eligible[~cal[eligible]]
  if not len(shared):raise ValueError('No calibration-frame observations for shared attachments')
  beta=to(z['betas']);o0=to(z['global_orient']);p0=to(z['hand_pose_pca']);t0=to(z['translation']);target=to(z['target24']);n=len(o0);faces=to(z['faces']).long();fn=z['faces'];fi=to(z['attachment_faces']).long();bw0=to(z['barycentric']);off0=to(z['normal_offsets_m']);wr0=to(z['wrist_local']);do=torch.nn.Parameter(torch.zeros_like(o0));dt=torch.nn.Parameter(torch.zeros_like(t0));dp=torch.nn.Parameter(torch.zeros_like(p0));logits=torch.nn.Parameter(bw0.clamp_min(1e-8).log());off=torch.nn.Parameter(off0.clone());wr=torch.nn.Parameter(wr0.clone());weights=to([.2]*3+[1]*21);mask=torch.tensor(cal,device=device)
  def geometry():return run_model(model,beta[None].expand(n,-1),o0+do,p0+dp,t0+dt)[:2]
  def predict(v,j,training=False):
   bw=logits.softmax(-1);tri=v[:,faces[fi]];normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
   if training:
    b=torch.where(mask[:,None,None],bw[None],bw.detach()[None]);nf=torch.where(mask[:,None],off[None],off.detach()[None]);w=torch.where(mask[:,None,None],wr[None],wr.detach()[None])
   else:b=bw[None];nf=off[None];w=wr[None].expand(n,-1,-1)
   surface=(tri*b[:,:,:,None]).sum(2)+normal*nf[:,:,None];wrist=torch.einsum('fij,fkj->fki',batch_rodrigues(o0+do),w)+j[:,:1];return torch.cat([wrist,surface],1)
  with torch.no_grad():v0,j0=geometry()
  for r in records:r['_wrist_joints']=project(j0[lookup[r['sync']]].double(),r,side).cpu().numpy().tolist()
  fitbatch=ContourBatch(fit,lookup,side,device);heldbatch=ContourBatch(held,lookup,side,device) if held else None
  with torch.no_grad():before=dict(fit_views=fitbatch.summary(v0),held_views=heldbatch.summary(v0) if heldbatch else None,marker_mm=stats(((predict(v0,j0)-target).norm(dim=-1)[eligible]*1000).cpu().numpy()))
  opt=torch.optim.Adam([{'params':[do],'lr':.0015},{'params':[dp],'lr':.002},{'params':[dt,off,wr],'lr':.00015},{'params':[logits],'lr':.012}]);order=np.argsort(z['pair_indices']);validpair=np.diff(z['pair_indices'][order])<=16;active=np.isin(order,eligible);validpair&=active[1:]&active[:-1];idx0=torch.tensor(order[:-1][validpair],device=device);idx1=torch.tensor(order[1:][validpair],device=device)
  for step in range(a.steps):
   opt.zero_grad();v,j=geometry();error=(predict(v,j,True)-target).norm(dim=-1)/.006;cost=(torch.sqrt(1+error.square())-1)*weights
   ml=cost[shared].mean() if not len(other) else .5*(cost[shared].mean()+cost[other].mean());il=fitbatch.loss(v)
   prior=.03*(do[eligible]/.06).square().mean()+.03*(dt[eligible]/.008).square().mean()+.03*(dp[eligible]/.12).square().mean();aprior=.05*((logits.softmax(-1)-bw0)/.2).square().mean()+.05*((off-off0)/.002).square().mean()+.05*((wr-wr0)/.002).square().mean()
   temporal=0 if not len(idx0) else .01*((dp[idx1]-dp[idx0])/.12).square().mean()+.01*((dt[idx1]-dt[idx0])/.008).square().mean()
   loss=ml+.5*il+prior+aprior+temporal
   if not torch.isfinite(loss):raise ValueError('Non-finite objective')
   loss.backward();opt.step()
   with torch.no_grad():
    do.mul_(torch.clamp(.06/do.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1));dt.mul_(torch.clamp(.008/dt.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1));dp.clamp_(-.12,.12);off.clamp_(0,.004);wr.copy_(wr0+(wr-wr0).clamp(-.002,.002))
   if not a.no_boundary_refresh and step==a.steps//2 and a.steps>=40:
    with torch.no_grad():fitbatch.refresh(v,j,fn,size)
   if step%40==0:print(hand,step,'loss',float(loss.detach()),'marker',float(ml.detach()),'image',float(il.detach()),flush=True)
  with torch.no_grad():v,j=geometry();pred=predict(v,j);after=dict(fit_views=fitbatch.summary(v),held_views=heldbatch.summary(v) if heldbatch else None,marker_mm=stats(((pred-target).norm(dim=-1)[eligible]*1000).cpu().numpy()))
  # For fair before/after loss diagnostics, both endpoints use the same final fit-view vertex set.
  with torch.no_grad():before['fit_views']=fitbatch.summary(v0)
  cpu=lambda x:x.detach().cpu().numpy();values={k:z[k] for k in z.files};jointmask=np.isin(z['pair_indices'],[z['pair_indices'][i] for i in eligible]);values.update(vertices=cpu(v),joints=cpu(j),predicted24=cpu(pred),marker_error_mm=cpu((pred-target).norm(dim=-1)*1000),global_orient=cpu(o0+do),hand_pose_pca=cpu(p0+dp),translation=cpu(t0+dt),barycentric=cpu(logits.softmax(-1)),normal_offsets_m=cpu(off),wrist_local=cpu(wr),joint_fit_mask=jointmask,prior_vertices=z['vertices'],prior_joints=z['joints'])
  np.savez_compressed(a.output/f'{hand}_pilot.npz',**values)
  summary['hands'][hand]=dict(before=before,after=after,eligible_sync=z['pair_indices'][eligible].tolist(),attachment_fit_sync=z['pair_indices'][shared].tolist(),skipped_single_view=len(skipped),held_keys=[[r['camera'],r['sync']] for r in held],fit_keys=[[r['camera'],r['sync']] for r in fit],beta_exactly_unchanged=bool(np.array_equal(values['betas'],z['betas'])),attachment_barycentric_max_change=float(abs(values['barycentric']-z['barycentric']).max()),normal_offset_max_change_mm=float(abs(values['normal_offsets_m']-z['normal_offsets_m']).max()*1000),wrist_attachment_max_change_mm=float(abs(values['wrist_local']-z['wrist_local']).max()*1000),max_pose_change_rad=float(dp.abs().max()),max_root_translation_change_mm=float(dt.norm(dim=-1).max()*1000),limits_reached=dict(rotation=int((do.norm(dim=-1)>=.05999).sum()),translation=int((dt.norm(dim=-1)>=.00799).sum()),pose_components=int((dp.abs()>=.11999).sum())),elapsed_seconds=time.monotonic()-begin)
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(hand,'DONE',summary['hands'][hand]['elapsed_seconds'],flush=True)
 print('DONE',a.output,flush=True)
if __name__=='__main__':main()
