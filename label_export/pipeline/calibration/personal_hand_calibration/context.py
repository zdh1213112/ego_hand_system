"""Explicit data context for opt-in pipeline; legacy experiment defaults remain available."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'marker24_pilot'))
from run_pilot import ROOT,CameraCalibration,load_rigid,load_markers,interpolate_points
EXP=ROOT/'output/native_kb_training_20261111'
def load_context(path=None):
 if path is None:return dict(dataset=EXP/'real_dual/pair_20261111_192916/ego/normalized_multiview',alignment=EXP/'fourview_192916/alignment_180/fourview_alignment.json',aligned21=EXP/'fourview_192916/alignment_180/aligned_nokov21.npz',baseline_fit=EXP/'fourview_192916/native_labels/mano_fit_full_cam01',placement=EXP/'marker24_calibration_input/placement_user_confirmed.json',holdout=[512,704],dorsal_reference=None,device='cuda')
 p=Path(path).resolve();d=json.loads(p.read_text())
 for key in ['dataset','alignment','aligned21','baseline_fit','placement']:
  q=Path(d[key]);d[key]=(p.parent/q).resolve() if not q.is_absolute() else q.resolve()
 return d

def raw24_context(c):
 a=json.loads(c['alignment'].read_text());seed=json.loads(Path(a['seed_alignment']).read_text());dataset=c['dataset'];nokov=Path(a['source_nokov'])
 if Path(a['source_dataset']).resolve()!=dataset.resolve():raise ValueError('Alignment dataset differs from pipeline context')
 with (dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
 m=seed['time_mapping'];t=np.array([int(r['camera0_timestamp_ns']) for r in rows],np.int64);target=m['nokov_origin_timestamp_raw']+((t-m['ego_origin_timestamp_ns'])*1e-9*m.get('a',1)+m['b_s'])/m['nokov_seconds_per_timestamp_unit']
 rt,head,q=load_rigid(nokov/'nokov_rigid_bodies.csv');head,ok=interpolate_points(rt,head,target);rot=np.full((len(rows),3,3),np.nan);ok&=(target>=rt[0])&(target<=rt[-1]);rot[ok]=Slerp(rt,Rotation.from_quat(q))(target[ok]).as_matrix();T=np.asarray(a['T_cam0_head']);hands=[]
 for name in ['left_hand_4mm','right_hand_4mm']:
  ts,p=load_markers(nokov/'nokov_markers.csv',name)
  if len(ts):p,_=interpolate_points(ts,p,target);p=np.einsum('fji,fkj->fki',rot,p-head[:,None]);hands.append(p@T[:3,:3].T+T[:3,3])
  else:hands.append(np.full((len(target),24,3),np.nan))
 xyz=np.asarray(hands);expected=np.load(c['aligned21'])['joints_base_m'];C=CameraCalibration.load(dataset/'calibration/camera0.json');base=xyz@C.T_base_camera[:3,:3].T+C.T_base_camera[:3,3]
 if np.nanmax(abs(base[:,:,[0]+list(range(4,24))]-expected))>=1e-5:raise ValueError('Raw24 reconstruction disagrees with accepted aligned21')
 return xyz,t
