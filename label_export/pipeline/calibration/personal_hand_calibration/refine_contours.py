#!/usr/bin/env python3
"""Small shared-beta refinement using calibration-only raw-image boundary proposals."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import ROOT,EXP,PROJECT,import_mano,run_model,stats
from context import load_context

def project(q,rec,side):
 to=lambda x:torch.as_tensor(x,dtype=q.dtype,device=q.device)
 q=q*to([-1 if side==0 else 1,1,1]);t0=to(rec['T_base_cam0']);tc=to(rec['T_base_camera']);base=q@t0[:3,:3].T+t0[:3,3];p=(base-tc[:3,3])@tc[:3,:3];xy=p[:,:2]/p[:,2:3];r=xy.norm(dim=-1).clamp_min(1e-9);theta=torch.atan(r);d=to(rec['D']).flatten();td=theta*(1+d[0]*theta**2+d[1]*theta**4+d[2]*theta**6+d[3]*theta**8);uv=xy*(td/r)[:,None];K=to(rec['K']);return uv*torch.stack([K[0,0],K[1,1]])+K[:2,2]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path);ap.add_argument('--source',type=Path,default=EXP/'personal_calibration_v2');ap.add_argument('--observations',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=120);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(0);device=load_context(a.context)['device'];to=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
 obs=json.loads(a.observations.read_text());mano=import_mano(PROJECT/'third_party/MANO');from mano.lbs import batch_rodrigues
 model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
 for p in model.parameters():p.requires_grad_(False)
 summary=dict(strategy='personal_shape_raw_contour_refinement_v3',image_constraints_used=True,approved_for_training=False,source=str(a.source.resolve()),observations=str(a.observations.resolve()),hands={},limitations=['Automatic color contours are proposals, not independently measured truth.','Marker projections select connected components; wrist cut and initial boundary vertices are model guided.','Only shared betas change. Attachments, finger pose and global pose remain v2 values.','Image loss uses initial raster silhouette vertex indices; silhouette topology is not differentiated.','Evaluation images never enter optimization, but previous marker-based initialization used their marker targets.'])
 for side,hand in enumerate(['left','right']):
  z=np.load(a.source/f'{hand}_pilot.npz');lookup={int(f):i for i,f in enumerate(z['pair_indices'])};cal=z['calibration_mask'];records=[r for r in obs['records'] if r['side']==side];train=[r for r in records if r['calibration']];test=[r for r in records if not r['calibration']]
  assert len(train)>=8 and len(test)>=4
  for r in records:assert r['calibration']==bool(cal[lookup[r['sync']]])
  beta=to(z['betas']);delta=torch.nn.Parameter(torch.zeros_like(beta));o=to(z['global_orient']);p=to(z['hand_pose_pca']);tr=to(z['translation']);target=to(z['target24']);n=len(o);faces=to(z['faces']).long();fi=to(z['attachment_faces']).long();bw=to(z['barycentric']);normaloff=to(z['normal_offsets_m']);wr=to(z['wrist_local']);weights=to([.2]*3+[1]*21)
  def markers(v,j):
   tri=v[:,faces[fi]];normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1);s=(tri*bw[None,:,:,None]).sum(2)+normal*normaloff[None,:,None];w=torch.einsum('fij,kj->fki',batch_rodrigues(o),wr)+j[:,:1];return torch.cat([w,s],1)
  def image_losses(v,rs):
   values=[]
   for r in rs:
    uv=project(v[lookup[r['sync']],r['boundary_vertices']],r,side);gt=to(r['points']);distance=torch.cdist(uv[None],gt[None])[0]
    # Robust symmetric boundary distances, equal weight per hand-view.
    e=torch.cat([distance.min(0).values,distance.min(1).values]);values.append((torch.sqrt(1+(e/5).square())-1).mean())
   return torch.stack(values).mean()
  def evaluate(v,j):
   result={}
   for split,rs in [('calibration',train),('validation',test)]:
    per=[]
    for r in rs:
     uv=project(v[lookup[r['sync']],r['boundary_vertices']],r,side);d=torch.cdist(uv[None],to(r['points'])[None])[0];err=torch.cat([d.min(0).values,d.min(1).values]);per.append(dict(camera=r['camera'],sync=r['sync'],mean_symmetric_distance_px=float(err.mean()),median_symmetric_distance_px=float(err.median())))
    result[split]=dict(views=len(rs),per_view=per,mean_symmetric_distance_px=float(np.mean([x['mean_symmetric_distance_px'] for x in per])))
   error=(markers(v,j)-target).norm(dim=-1)*1000;result['validation_marker_mm']=stats(error[~cal].cpu().numpy());return result
  with torch.no_grad():v0,j0,_=run_model(model,beta[None].expand(n,-1),o,p,tr);before=evaluate(v0,j0)
  opt=torch.optim.Adam([delta],lr=.008)
  for step in range(a.steps):
   opt.zero_grad();v,j,_=run_model(model,(beta+delta)[None].expand(n,-1),o,p,tr);err=(markers(v,j)[cal]-target[cal]).norm(dim=-1)/.006;ml=((torch.sqrt(1+err.square())-1)*weights).mean();il=image_losses(v,train);loss=ml+.5*il+.1*delta.square().mean();loss.backward();opt.step()
   with torch.no_grad():delta.clamp_(-.5,.5)
   if step%30==0:print(hand,step,'marker',float(ml.detach()),'image',float(il.detach()),flush=True)
  with torch.no_grad():v,j,_=run_model(model,(beta+delta)[None].expand(n,-1),o,p,tr);after=evaluate(v,j);pred=markers(v,j)
  values={k:z[k] for k in z.files};values.update(betas=(beta+delta).detach().cpu().numpy(),vertices=v.cpu().numpy(),joints=j.cpu().numpy(),predicted24=pred.cpu().numpy(),marker_error_mm=((pred-target).norm(dim=-1)*1000).cpu().numpy())
  np.savez_compressed(a.output/f'{hand}_pilot.npz',**values)
  summary['hands'][hand]=dict(before=before,after=after,beta_delta=delta.detach().cpu().tolist(),beta_limit_count=int((delta.abs()>=.4999).sum()),calibration_sync=sorted(set(r['sync'] for r in train)),validation_sync=sorted(set(r['sync'] for r in test)))
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(hand,'before',before['validation']['mean_symmetric_distance_px'],'after',after['validation']['mean_symmetric_distance_px'],flush=True)
 print('DONE',flush=True)
if __name__=='__main__':main()
