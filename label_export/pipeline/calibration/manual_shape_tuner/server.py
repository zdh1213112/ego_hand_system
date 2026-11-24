#!/usr/bin/env python3
"""Local multi-view MANO editing. Saves user intent and diagnostic biases, never production labels."""
import argparse,csv,hashlib,json,sys,threading,time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'label_export/pipeline/calibration/personal_hand_calibration'))
from run import EXP,PROJECT,import_mano,run_model,CameraCalibration
from camera_models import project_points
METRICS=['palm_width','palm_length','thumb_length','index_length','middle_length','ring_length','pinky_length']
METRIC_NAMES=['掌宽（食指根到小指根）','掌长（腕点到中指根）','拇指关节链长','食指关节链长','中指关节链长','无名指关节链长','小指关节链长']
class Engine:
 def __init__(self,source,output):
  self.source=source.resolve();self.output=output.resolve();self.output.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();torch.set_num_threads(2)
  mano=import_mano(PROJECT/'third_party/MANO');self.model=mano.load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1)
  if self.model.use_pca:raise ValueError('Editor requires direct 45D axis-angle MANO pose')
  for p in self.model.parameters():p.requires_grad_(False)
  self.z=[np.load(self.source/f'{h}_pilot.npz') for h in ['left','right']];self.maps=[{int(f):i for i,f in enumerate(z['pair_indices'])} for z in self.z];self.faces=self.z[0]['faces'];self.dataset=EXP/'real_dual/pair_20261111_192916/ego/normalized_multiview'
  with (self.dataset/'multiview_frames.csv').open() as f:self.rows=list(csv.DictReader(f))
  self.cameras={f'camera{i}':CameraCalibration.load(self.dataset/'calibration'/f'camera{i}.json') for i in range(4)};self.C0=self.cameras['camera0']
  self.baseline_id=hashlib.sha256(b''.join((self.source/f'{h}_pilot.npz').read_bytes() for h in ['left','right'])).hexdigest()
 def vec(self,x,n,limit,name):
  try:a=np.asarray(x,dtype=float)
  except (ValueError,TypeError):raise ValueError(name+' 必须是数字数组')
  if a.shape!=(n,) or not np.isfinite(a).all() or (abs(a)>limit).any():raise ValueError(name+' 超出范围或格式错误')
  return a
 def clean(self,r):
  side=r.get('side');sync=r.get('sync')
  if isinstance(side,bool) or side not in [0,1] or not isinstance(sync,int) or isinstance(sync,bool) or sync not in self.maps[side]:raise ValueError('无效的左右手或帧')
  return dict(side=side,sync=sync,betas=self.vec(r.get('betas'),10,8,'手型'),translation_mm=self.vec(r.get('translation_mm',[0]*3),3,100,'平移'),rotation_deg=self.vec(r.get('rotation_deg',[0]*3),3,90,'旋转'),pose_deg=self.vec(r.get('pose_deg',[0]*45),45,90,'手指姿态'))
 def neutral(self,beta):
  with torch.no_grad():v,j,_=run_model(self.model,torch.tensor(beta,dtype=torch.float32)[None],torch.zeros(1,3),-self.model.hand_mean[None],torch.zeros(1,3))
  return v[0].numpy()-j[0,0].numpy(),j[0].numpy()-j[0,0].numpy()
 def metrics(self,beta):
  _,j=self.neutral(beta);values=[np.linalg.norm(j[5]-j[17]),np.linalg.norm(j[9]-j[0])]
  for start in [1,5,9,13,17]:values.append(np.linalg.norm(np.diff(j[start:start+4],axis=0),axis=1).sum())
  return np.array(values)*1000
 def suggest(self,r):
  beta=self.vec(r.get('betas'),10,8,'手型');key=r.get('metric');target=float(r.get('target_mm'))
  if key not in METRICS or not np.isfinite(target) or target<10 or target>160:raise ValueError('无效尺寸目标')
  base=self.metrics(beta);jac=np.column_stack([(self.metrics(beta+np.eye(10)[i]*.02)-base)/.02 for i in range(10)]);desired=np.zeros(7);desired[METRICS.index(key)]=np.clip(target-base[METRICS.index(key)],-8,8)
  w=np.ones(7);w[METRICS.index(key)]=4;A=jac*w[:,None];delta=np.linalg.solve(A.T@A+.5*np.eye(10),A.T@(desired*w));delta*=min(1,.75/max(np.linalg.norm(delta),1e-9));new=np.clip(beta+delta,-8,8)
  return dict(betas=new.tolist(),metrics_mm=self.metrics(new).tolist(),requested_target_mm=target)
 def geometry(self,r):
  r=self.clean(r);side=r['side'];z=self.z[side];i=self.maps[side][r['sync']];beta=torch.tensor(r['betas'],dtype=torch.float32)[None];o0=z['global_orient'][i];p0=z['hand_pose_pca'][i];t0=z['translation'][i];mirror=np.diag([-1 if side==0 else 1,1,1])
  delta=Rotation.from_euler('xyz',r['rotation_deg'],degrees=True).as_matrix();rot=mirror@delta@mirror@Rotation.from_rotvec(o0).as_matrix();o=Rotation.from_matrix(rot).as_rotvec().astype(np.float32);pose=(p0+np.deg2rad(r['pose_deg'])).astype(np.float32)
  # Shape edits retain this frame's original wrist; translation sliders are physical camera0 axes.
  with torch.no_grad():
   _,tmp,_=run_model(self.model,beta,torch.tensor(o)[None],torch.tensor(pose)[None],torch.tensor(t0)[None]);tr=t0+(z['joints'][i,0]-tmp[0,0].numpy())+mirror@r['translation_mm']/1000
   v,j,_=run_model(self.model,beta,torch.tensor(o)[None],torch.tensor(pose)[None],torch.tensor(tr,dtype=torch.float32)[None])
  v=v[0].numpy();j=j[0].numpy();tri=v[self.faces[z['attachment_faces']]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
  surface=(tri*z['barycentric'][:,:,None]).sum(1)+normal*z['normal_offsets_m'][:,None];w=z['wrist_local']@rot.T+j[0];pred=np.r_[w,surface];observed=z['target24'][i];residual=observed-pred
  tangent=tri[:,1]-tri[:,0];tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12);second=np.cross(normal,tangent);local=np.r_[residual[:3]@rot,np.stack([(residual[3:]*a).sum(1) for a in [tangent,second,normal]],axis=1)]*1000
  return dict(request=r,vertices=v,joints=j,predicted=pred,observed=observed,residual_local_mm=local,global_orient=o,hand_pose=pose,translation=tr,neutral_vertices=self.neutral(r['betas'])[0],metrics=self.metrics(r['betas']))
 def project(self,q,camera,side):
  C=self.cameras[camera];p=q.copy();p[:,0]*=(-1 if side==0 else 1);base=p@self.C0.T_base_camera[:3,:3].T+self.C0.T_base_camera[:3,3];cam=(base-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];uv,_=project_points(C,cam);out=np.c_[uv,cam[:,2]]
  return [a.tolist() if np.isfinite(a).all() and a[2]>.03 else None for a in out]
 def render(self,r):
  g=self.geometry(r);q=g['request'];side=q['side'];sync=q['sync'];z=self.z[side];i=self.maps[side][sync];views=[]
  for cam in self.cameras:
   verts=self.project(g['vertices'],cam,side);baseline=self.project(z['vertices'][i],cam,side);views.append(dict(camera=cam,image=f'/image?camera={cam}&sync={sync}',vertices=verts,baseline=baseline,joints=self.project(g['joints'],cam,side),marker_observed=self.project(g['observed'],cam,side),marker_predicted=self.project(g['predicted'],cam,side),camera_delta_ms=(int(self.rows[sync][cam+'_timestamp_ns'])-int(self.rows[sync]['camera0_timestamp_ns']))/1e6))
  neutral=g['neutral_vertices'].copy();neutral[:,0]*=(-1 if side==0 else 1)
  posed=(g['vertices']-g['joints'][0])*np.array([-1 if side==0 else 1,1,1])*1000
  return dict(views=views,neutral_vertices_mm=(neutral*1000).tolist(),posed_vertices_mm=posed.tolist(),metrics_mm=g['metrics'].tolist(),actual_parameters=dict(betas=q['betas'].tolist(),global_orient=g['global_orient'].tolist(),hand_pose=g['hand_pose'].tolist(),translation_m=g['translation'].tolist()),marker_residual_median_mm=float(np.median(np.linalg.norm(g['observed']-g['predicted'],axis=1)*1000)),residual_local_mm=g['residual_local_mm'].tolist())
 @lru_cache(maxsize=32)
 def image(self,cam,sync):
  if cam not in self.cameras or sync not in set(self.maps[0])|set(self.maps[1]):raise ValueError('无效图像')
  cap=cv2.VideoCapture(str(self.dataset/'cameras'/cam/'video.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.rows[sync][cam+'_frame_index']));ok,im=cap.read();cap.release()
  if not ok:raise ValueError('源图像读取失败')
  ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);return b.tobytes()
 def bootstrap(self):
  return dict(schema='manual_shape_tuner_v1',baseline_id=self.baseline_id,faces=self.faces.tolist(),metrics=METRICS,metric_names=METRIC_NAMES,hands=[dict(side=s,betas=z['betas'].tolist(),frames=sorted(self.maps[s]),metrics_mm=self.metrics(z['betas']).tolist()) for s,z in enumerate(self.z)],joint_order=[dict(finger=f,start=start) for f,start in [('食指',0),('中指',3),('小指',6),('无名指',9),('拇指',12)]],cameras=list(self.cameras),source='PERSONAL-3',output=str(self.output))
 def save(self,state):
  if state.get('baseline_id')!=self.baseline_id:raise ValueError('保存来源不匹配')
  hands=state.get('hands')
  if not isinstance(hands,list) or len(hands)!=2:raise ValueError('需包含左右手状态')
  stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000);folder=self.output/stamp;folder.mkdir();records=[];aggregates={};clean_hands=[]
  for side,h in enumerate(hands):
   beta=self.vec(h.get('betas'),10,8,'手型');frames=h.get('frames',{})
   if not isinstance(frames,dict) or len(frames)>200:raise ValueError('帧记录格式不正确')
   clean_frames={};biases=[]
   for key,f in frames.items():
    sync=int(key);r=dict(f,side=side,sync=sync,betas=beta.tolist());g=self.geometry(r);q=g['request'];checked=f.get('checked_cameras',[])
    if not isinstance(checked,list) or any(c not in self.cameras for c in checked):raise ValueError('相机记录无效')
    valid=bool(f.get('confirmed')) and np.array_equal(np.asarray(f.get('confirmed_betas',[])),beta) and f.get('confirmed_edits')=={k:q[k].tolist() for k in ['translation_mm','rotation_deg','pose_deg']}
    record=dict(side=side,sync=sync,confirmed=valid,checked_cameras=sorted(set(checked)),note=str(f.get('note',''))[:2000],betas=beta.tolist(),translation_mm=q['translation_mm'].tolist(),rotation_deg=q['rotation_deg'].tolist(),pose_deg=q['pose_deg'].tolist(),global_orient=g['global_orient'].tolist(),hand_pose=g['hand_pose'].tolist(),translation_m=g['translation'].tolist(),marker_residual_local_mm=g['residual_local_mm'].tolist(),marker_observed_minus_predicted_cam0_mm=((g['observed']-g['predicted'])*np.array([-1 if side==0 else 1,1,1])*1000).tolist())
    records.append(record);clean_frames[key]={k:f[k] for k in ['translation_mm','rotation_deg','pose_deg','confirmed_betas','confirmed_edits','checked_cameras','note'] if k in f};clean_frames[key]['confirmed']=valid
    if valid and len(set(checked))>=2:biases.append(g['residual_local_mm'])
    np.savez_compressed(folder/f'{side}_{sync:06d}.npz',vertices=g['vertices'],joints=g['joints'],predicted24=g['predicted'],observed24=g['observed'],betas=beta,global_orient=g['global_orient'],hand_pose=g['hand_pose'],translation=g['translation'],geometry_space='right_canonical',approved_for_training=False)
   if biases:
    arr=np.asarray(biases);median=np.median(arr,axis=0);mad=np.median(abs(arr-median),axis=0);aggregates[str(side)]=dict(multiview_confirmed_frames=len(biases),provisional=True,enough_frames_for_initial_summary=len(biases)>=3,marker_names=self.z[side]['marker_names'].tolist(),median_local_mm=median.tolist(),mad_local_mm=mad.tolist())
   clean_hands.append(dict(betas=beta.tolist(),frames=clean_frames))
  payload=dict(schema='manual_shape_intent_v1',baseline_id=self.baseline_id,source=str(self.source),saved_at=stamp,approved_for_training=False,state=dict(baseline_id=self.baseline_id,hands=clean_hands),frames=records,bias_diagnostics=aggregates,limitations=['Manual visual matches are user intent anchors, not independently measured ground truth.','Surface residual axes are triangle tangent1/tangent2/outward normal; wrist axes are root-local. Residual sign is NOKOV observed minus predicted, with existing v3 effective normal offsets retained.','Residuals combine attachment, shape, pose, glove and calibration effects; they are not uniquely identified physical corrections.','At least three varied poses and multiple checked views are recommended before interpreting robust bias statistics.'])
  path=folder/'session.json';path.write_text(json.dumps(payload,ensure_ascii=False,indent=2));latest=self.output/'latest.json';tmp=self.output/'latest.tmp';tmp.write_text(json.dumps(dict(path=str(path)),ensure_ascii=False));tmp.replace(latest)
  return dict(path=str(path),download=f'/session/{stamp}',confirmed=sum(r['confirmed'] for r in records),bias_diagnostics=aggregates)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8767);ap.add_argument('--source',type=Path,default=EXP/'personal_calibration_v3');ap.add_argument('--output',type=Path,default=EXP/'manual_shape_tuning');a=ap.parse_args();engine=Engine(a.source,a.output)
 class Handler(BaseHTTPRequestHandler):
  def send(self,code,data,ctype='application/json; charset=utf-8'):
   b=json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if ctype.startswith('application/json') else data;self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
  def do_GET(self):
   try:
    u=urlparse(self.path)
    if u.path=='/':return self.send(200,Path(__file__).with_name('index.html').read_bytes(),'text/html; charset=utf-8')
    with engine.lock:
     if u.path=='/api/bootstrap':return self.send(200,engine.bootstrap())
     if u.path=='/image':
      q=parse_qs(u.query);return self.send(200,engine.image(q['camera'][0],int(q['sync'][0])),'image/jpeg')
     if u.path=='/api/latest':
      pointer=engine.output/'latest.json';return self.send(200,json.loads(Path(json.loads(pointer.read_text())['path']).read_text()) if pointer.exists() else None)
     if u.path.startswith('/session/'):
      name=u.path.removeprefix('/session/')
      if not name or any(c not in '0123456789_' for c in name):raise ValueError('无效文件')
      p=engine.output/name/'session.json';return self.send(200,json.loads(p.read_text()))
    self.send(404,{'error':'未找到'})
   except (ValueError,KeyError,FileNotFoundError,IndexError) as e:self.send(400,{'error':str(e)})
  def do_POST(self):
   try:
    origin=self.headers.get('Origin')
    if origin and origin not in [f'http://{self.headers.get("Host")}']:raise ValueError('不允许跨站请求')
    size=int(self.headers.get('Content-Length','0'))
    if not 0<size<4000000:raise ValueError('请求大小无效')
    request=json.loads(self.rfile.read(size))
    with engine.lock:
     if self.path=='/api/render':return self.send(200,engine.render(request))
     if self.path=='/api/suggest':return self.send(200,engine.suggest(request))
     if self.path=='/api/save':return self.send(200,engine.save(request))
    self.send(404,{'error':'未找到'})
   except (ValueError,KeyError,TypeError,IndexError) as e:self.send(400,{'error':str(e)})
  def log_message(self,fmt,*args):pass
 print(f'MANO 调整界面：http://127.0.0.1:{a.port}/',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
