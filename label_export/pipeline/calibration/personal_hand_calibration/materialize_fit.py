"""Convert only actually optimized frames to standard export tracks with explicit provenance."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import PROJECT,import_mano,run_model
from context import load_context

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=load_context(a.context)
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);mano=import_mano(PROJECT/'third_party/MANO');model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1)
 torch.set_num_threads(2);summary=json.loads((c['baseline_fit']/'summary.json').read_text());summary.update(fit_target='personal_marker24',personal_marker24_optimization_enabled=True,personal_marker24_observation_source='raw_nokov24+automatic_native_contours',multiview_optimization_enabled=False,personal_shape_sharing='one_beta_vector_per_physical_hand',source_optimized_fit=str(a.source.resolve()),approved_for_training=False,tracks=[])
 optimized_summary=json.loads((a.source/'summary.json').read_text());summary['joint_pose_attachment_optimization_enabled']=optimized_summary.get('joint_pose_attachment_optimization_enabled',False);summary['joint_view_policy']=optimized_summary.get('view_policy');summary['baseline_summary_path']=str(c['baseline_fit']/'summary.json');summary['baseline_parameters']=summary.pop('parameters',{});summary.pop('elapsed_seconds',None);summary['parameters']={'context':str(a.context.resolve()),'optimized_source':str(a.source.resolve())}
 for side,hand in enumerate(['left','right']):
  if not (a.source/f'{hand}_pilot.npz').is_file(): continue
  b=np.load(c['baseline_fit']/f'track_{side}.npz');z=np.load(a.source/f'{hand}_pilot.npz');lookup={int(f):i for i,f in enumerate(b['pair_indices'])};ids=np.array([lookup[int(f)] for f in z['pair_indices']]);n=len(ids)
  # Explicitly enumerate per-frame fields: faces is not a frame axis even if lengths happen to match.
  frame_keys={'pair_indices','vertices','joints','global_orient','hand_pose_pca','translation','joint_error_m','observation_error_m','observation_points','left_reprojection_error_px','right_reprojection_error_px','loss','silhouette_loss','observation_quality','image_motion_px','temporal_strength','image_alignment_m','render_valid','fit_segment_id','hand_pose_axis_angle','full_pose_axis_angle'}
  values={k:(b[k][ids].copy() if k in frame_keys and b[k].ndim and b[k].shape[0]==len(b['pair_indices']) else b[k].copy()) for k in b.files}
  for k in ['pair_indices','vertices','joints','betas','global_orient','hand_pose_pca','translation']:values[k]=z[k]
  poses=[]
  with torch.no_grad():
   for start in range(0,n,192):
    ix=slice(start,min(n,start+192));count=ix.stop-start
    v,j,out=run_model(model,torch.tensor(values['betas'])[None].expand(count,-1),torch.tensor(values['global_orient'][ix]),torch.tensor(values['hand_pose_pca'][ix]),torch.tensor(values['translation'][ix]))
    if float((v-torch.tensor(values['vertices'][ix])).abs().max())>2e-6 or float((j-torch.tensor(values['joints'][ix])).abs().max())>2e-6:
     raise ValueError('Optimized MANO parameters do not reconstruct exported mesh and joints')
    poses.append(out.full_pose.numpy().reshape(count,16,3))
  full=np.concatenate(poses);values['full_pose_axis_angle']=full;values['hand_pose_axis_angle']=full[:,1:]
  select=[0]+list(range(4,24));observed=z['target24'][:,select];pred=z['predicted24'][:,select];values['observation_points']=pred;values['observation_error_m']=np.linalg.norm(pred-observed,axis=-1).astype(np.float32);values['joint_error_m']=np.linalg.norm(z['joints']-observed,axis=-1).astype(np.float32);values['fit_target']=np.array('personal_marker24')
  valid=z['marker_valid'][:,select].astype(bool) if 'marker_valid' in z else np.ones_like(values['observation_error_m'],bool)
  if 'marker_valid' in z:
   values['observation_valid']=valid
   for field in ('observation_error_m','joint_error_m'):values[field][~valid]=np.nan
  finite_visible=(~valid | np.isfinite(values['observation_error_m'])).all(1)
  values['render_valid']=values['render_valid']&np.isfinite(values['vertices']).all((1,2))&finite_visible&valid.any(1)
  if 'joint_fit_mask' in z.files:values['joint_fit_mask']=z['joint_fit_mask']
  values['personal_marker24_optimization_enabled']=np.array(True);values['approved_for_training']=np.array(False)
  np.savez_compressed(a.output/f'track_{side}.npz',**values);summary['tracks'].append(dict(track_id=side,handedness=hand.title(),frames=n,betas=z['betas'].tolist(),fit_target='personal_marker24',sampling_only_actually_optimized_frames=True))
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(a.output)
if __name__=='__main__':main()
