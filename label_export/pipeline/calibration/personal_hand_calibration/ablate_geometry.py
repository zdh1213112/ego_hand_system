#!/usr/bin/env python3
"""Controlled rigid/articulation/attachment diagnostics with held-out camera views."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import EXP,PROJECT,import_mano,run_model,stats
from refine_contours import project
from context import load_context

def split_views(records):
 groups={}
 for r in records:groups.setdefault(r['sync'],[]).append(r)
 fit=[];held=[];skipped=[]
 for sync,rs in sorted(groups.items()):
  rs=sorted(rs,key=lambda r:r['camera'])
  if len(rs)<2:skipped.extend(rs);continue
  k=(sync//8)%len(rs);held.append(rs[k]);fit.extend(r for i,r in enumerate(rs) if i!=k)
 return fit,held,skipped

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path);ap.add_argument('--mode',choices=['rigid','fingers','attachments'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,default=EXP/'personal_calibration_v3');ap.add_argument('--observations',type=Path,default=EXP/'personal_contours_external_v2/observations.json');ap.add_argument('--steps',type=int,default=100);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(0);device=load_context(a.context)['device'];to=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
 obs=json.loads(a.observations.read_text());mano=import_mano(PROJECT/'third_party/MANO');from mano.lbs import batch_rodrigues
 model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
 for p in model.parameters():p.requires_grad_(False)
 summary=dict(mode=a.mode,source=str(a.source.resolve()),observations=str(a.observations.resolve()),strategy='single_parameter_group_ablation',approved_for_training=False,hands={},limitations=['Held-out camera contours do not enter this optimization; v3 initialization used prior calibration-image evidence. This is incremental cross-view evidence, not independent ground truth.','Only frames with at least two accepted camera contours participate.','Per-frame rigid/finger parameters fit marker and non-held camera views, including the time-window frames. Shared attachments use only calibration timestamps.','Attachment-only optimization cannot change mesh geometry; it is a marker-fit negative control.','Automatic contour errors and fixed silhouette-vertex approximation remain.'])
 for side,hand in enumerate(['left','right']):
  z=np.load(a.source/f'{hand}_pilot.npz');lookup={int(f):i for i,f in enumerate(z['pair_indices'])};records=[r for r in obs['records'] if r['side']==side];fit,held,skipped=split_views(records);eligible=sorted({lookup[r['sync']] for r in fit});shared=[i for i in eligible if z['calibration_mask'][i]]
  assert shared and held and not {(r['camera'],r['sync']) for r in held}&{(r['camera'],r['sync']) for r in fit}
  beta=to(z['betas']);o=to(z['global_orient']);p=to(z['hand_pose_pca']);tr=to(z['translation']);n=len(o);target=to(z['target24']);faces=to(z['faces']).long();fi=to(z['attachment_faces']).long();bw0=to(z['barycentric']);off0=to(z['normal_offsets_m']);wr0=to(z['wrist_local']);weights=to([.2]*3+[1]*21)
  do=torch.nn.Parameter(torch.zeros_like(o),requires_grad=a.mode=='rigid');dt=torch.nn.Parameter(torch.zeros_like(tr),requires_grad=a.mode=='rigid');dp=torch.nn.Parameter(torch.zeros_like(p),requires_grad=a.mode=='fingers');logits=torch.nn.Parameter(bw0.clamp_min(1e-8).log(),requires_grad=a.mode=='attachments');off=torch.nn.Parameter(off0.clone(),requires_grad=a.mode=='attachments');wr=torch.nn.Parameter(wr0.clone(),requires_grad=a.mode=='attachments')
  def geometry():return run_model(model,beta[None].expand(n,-1),o+do,p+dp,tr+dt)[:2]
  def markers(v,j):
   tri=v[:,faces[fi]];normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1);b=logits.softmax(-1) if a.mode=='attachments' else bw0;s=(tri*b[None,:,:,None]).sum(2)+normal*off[None,:,None];w=torch.einsum('fij,kj->fki',batch_rodrigues(o+do),wr)+j[:,:1];return torch.cat([w,s],1)
  def distances(v,r):
   uv=project(v[lookup[r['sync']],r['boundary_vertices']],r,side);d=torch.cdist(uv[None],to(r['points'])[None])[0];return torch.cat([d.min(0).values,d.min(1).values])
  def evaluate(v,j):
   result={}
   for label,rs in [('fit_views',fit),('held_camera_views',held)]:
    per=[dict(camera=r['camera'],sync=r['sync'],distance_px=float(distances(v,r).mean())) for r in rs];result[label]=dict(count=len(per),mean_px=float(np.mean([q['distance_px'] for q in per])),per_view=per)
   e=(markers(v,j)-target).norm(dim=-1)*1000;result['eligible_marker_mm']=stats(e[eligible].cpu().numpy());result['time_window_marker_mm']=stats(e[~z['calibration_mask']].cpu().numpy());return result
  with torch.no_grad():v0,j0=geometry();before=evaluate(v0,j0)
  groups=({'params':[do],'lr':.002},{'params':[dt],'lr':.00025}) if a.mode=='rigid' else ([{'params':[dp],'lr':.003}] if a.mode=='fingers' else [{'params':[logits],'lr':.02},{'params':[off,wr],'lr':.0001}])
  opt=torch.optim.Adam(groups)
  ix=shared if a.mode=='attachments' else eligible
  for step in range(a.steps):
   opt.zero_grad();v,j=geometry();err=(markers(v,j)[ix]-target[ix]).norm(dim=-1)/.006;ml=((torch.sqrt(1+err.square())-1)*weights).mean()
   if a.mode=='attachments':loss=ml+.05*((logits.softmax(-1)-bw0)/.2).square().mean()+.05*((off-off0)/.002).square().mean()+.05*((wr-wr0)/.002).square().mean()
   else:
    il=torch.stack([(torch.sqrt(1+(distances(v,r)/5).square())-1).mean() for r in fit]).mean();loss=ml+.5*il+.03*(do/.06).square().mean()+.03*(dt/.008).square().mean()+.03*(dp/.12).square().mean()
   loss.backward();opt.step()
   with torch.no_grad():
    do.mul_(torch.clamp(.06/do.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1));dt.mul_(torch.clamp(.008/dt.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1));dp.clamp_(-.12,.12);off.clamp_(0,.004);wr.copy_(wr0+(wr-wr0).clamp(-.002,.002))
   if step%40==0:print(a.mode,hand,step,float(loss.detach()),flush=True)
  with torch.no_grad():v,j=geometry();pred=markers(v,j);after=evaluate(v,j)
  values={k:z[k] for k in z.files};values.update(vertices=v.cpu().numpy(),joints=j.cpu().numpy(),predicted24=pred.cpu().numpy(),marker_error_mm=((pred-target).norm(dim=-1)*1000).cpu().numpy(),global_orient=(o+do).detach().cpu().numpy(),hand_pose_pca=(p+dp).detach().cpu().numpy(),translation=(tr+dt).detach().cpu().numpy(),barycentric=(logits.softmax(-1) if a.mode=='attachments' else bw0).detach().cpu().numpy(),normal_offsets_m=off.detach().cpu().numpy(),wrist_local=wr.detach().cpu().numpy())
  np.savez_compressed(a.output/f'{hand}_pilot.npz',**values)
  comparison=[]
  for b,c in zip(before['held_camera_views']['per_view'],after['held_camera_views']['per_view']):comparison.append(dict(camera=b['camera'],sync=b['sync'],before_px=b['distance_px'],after_px=c['distance_px'],delta_px=c['distance_px']-b['distance_px']))
  summary['hands'][hand]=dict(before=before,after=after,held_camera_comparison=comparison,skipped_single_view=len(skipped),eligible_sync=[int(z['pair_indices'][i]) for i in eligible],attachment_fit_sync=[int(z['pair_indices'][i]) for i in shared] if a.mode=='attachments' else [],limit_counts=dict(translation=int((dt.norm(dim=-1)>=.00799).sum()),rotation=int((do.norm(dim=-1)>=.05999).sum()),finger_components=int((dp.abs()>=.11999).sum())),vertex_movement_mm=stats(((v-v0).norm(dim=-1)*1000).cpu().numpy()))
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(hand,'held before/after',before['held_camera_views']['mean_px'],after['held_camera_views']['mean_px'],flush=True)
 print('DONE',flush=True)
if __name__=='__main__':main()
