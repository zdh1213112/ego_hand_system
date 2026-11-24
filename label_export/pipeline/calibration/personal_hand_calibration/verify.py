#!/usr/bin/env python3
"""Verify reconstruction, constraints and cross-subset stability; never approve labels."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import ROOT,EXP,PROJECT,import_mano,run_model,stats
from scipy.spatial.transform import Rotation

def main():
 torch.set_num_threads(4);mano=import_mano(PROJECT/'third_party/MANO');model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1)
 from mano.lbs import batch_rodrigues
 out=EXP/'personal_calibration_v2';report=dict(status='not_approved_pending_independent_image_constraints',hands={},checks=[])
 to=lambda a:torch.tensor(a,dtype=torch.float32)
 neutral=(torch.linalg.solve(model.hand_components.T,-model.hand_mean) if model.use_pca else -model.hand_mean).reshape(1,-1)
 def neutralmesh(beta):
  with torch.no_grad():v,j,_=run_model(model,to(beta)[None],torch.zeros(1,3),neutral,torch.zeros(1,3))
  return (v[0]-j[0,0]).numpy()
 for side,hand in enumerate(['left','right']):
  base=np.load(EXP/f'fourview_192916/native_labels/mano_fit_full_cam01/track_{side}.npz');lookup={int(f):i for i,f in enumerate(base['pair_indices'])};runs={}
  for name in ['personal_calibration_v2','personal_calibration_even_v2','personal_calibration_odd_v2','personal_calibration_fixed_v2']:
   z=np.load(EXP/name/f'{hand}_pilot.npz');ids=np.array([lookup[int(f)] for f in z['pair_indices']]);cal=z['calibration_mask'];assert not set(z['pair_indices'][cal])&set(z['pair_indices'][~cal]);assert not any(512<=x<704 for x in z['pair_indices'][cal]);n=len(ids)
   assert np.max(np.abs(z['betas']-z['initial_betas']))<=1.50001
   assert np.min(z['normal_offsets_m'])>=0 and np.max(z['normal_offsets_m'])<=.004001
   assert np.min(z['barycentric'])>=0 and np.max(abs(z['barycentric'].sum(1)-1))<1e-5
   assert np.max(np.linalg.norm(z['translation']-base['translation'][ids],axis=1))<=.008001
   assert np.max(np.linalg.norm(z['global_orient']-base['global_orient'][ids],axis=1))<=.060001
   assert np.max(abs(z['hand_pose_pca']-base['hand_pose_pca'][ids]))<=.120001
   with torch.no_grad():
    v,j,_=run_model(model,to(z['betas'])[None].expand(n,-1),to(z['global_orient']),to(z['hand_pose_pca']),to(z['translation']))
    tri=v[:,to(z['faces']).long()[to(z['attachment_faces']).long()]];norm=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
    surface=(tri*to(z['barycentric'])[None,:,:,None]).sum(2)+norm*to(z['normal_offsets_m'])[None,:,None]
    wrist=torch.einsum('fij,kj->fki',batch_rodrigues(to(z['global_orient'])),to(z['wrist_local']))+j[:,:1];pred=torch.cat([wrist,surface],1).numpy()
   verr=float(np.max(abs(v.numpy()-z['vertices'])));perr=float(np.max(abs(pred-z['predicted24'])));assert verr<2e-6 and perr<2e-6
   assert np.isfinite(pred).all()
   if 'fixed' in name:assert np.array_equal(z['betas'],z['initial_betas'])
   runs[name]=z
   report['checks'].append(dict(run=name,hand=hand,vertex_reconstruction_max_m=verr,marker_reconstruction_max_m=perr,passed=True))
  a=runs['personal_calibration_v2'];even=runs['personal_calibration_even_v2'];odd=runs['personal_calibration_odd_v2'];fixed=runs['personal_calibration_fixed_v2']
  assert not set(even['pair_indices'][even['calibration_mask']])&set(odd['pair_indices'][odd['calibration_mask']])
  i=int(np.flatnonzero(a['pair_indices']==632)[0]);j=lookup[632]
  rotation=Rotation.from_rotvec(a['global_orient'][i])*Rotation.from_rotvec(base['global_orient'][j]).inv()
  h=dict(neutral_shape_even_odd_difference_mm=stats(np.linalg.norm(neutralmesh(even['betas'])-neutralmesh(odd['betas']),axis=-1)*1000),neutral_shape_change_from_baseline_mm=stats(np.linalg.norm(neutralmesh(a['betas'])-neutralmesh(a['initial_betas']),axis=-1)*1000),validation_marker_residual_mm=stats(a['marker_error_mm'][~a['calibration_mask']]),fixed_shape_control_validation_residual_mm=stats(fixed['marker_error_mm'][~fixed['calibration_mask']]),frame632=dict(rotation_change_deg=float(rotation.magnitude()*180/np.pi),wrist_movement_mm=float(np.linalg.norm(a['joints'][i,0]-a['baseline_joints'][i,0])*1000),finger_parameter_max_delta_rad=float(abs(a['hand_pose_pca'][i]-base['hand_pose_pca'][j]).max())),offset_zero_markers=a['marker_names'][3:][a['normal_offsets_m']<1e-7].tolist(),offset_upper_bound_markers=a['marker_names'][3:][a['normal_offsets_m']>=.00399].tolist())
  report['hands'][hand]=h
 (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['hands'],indent=2));print('PASS: 8 runs reconstructed; splits disjoint; physical/pose limits; fixed-shape control. No image-accuracy claim.')
if __name__=='__main__':main()
