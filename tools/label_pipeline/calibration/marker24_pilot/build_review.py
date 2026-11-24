#!/usr/bin/env python3
"""Render experimental validation frames, including incomplete camera views."""
import argparse,csv,json,sys,shutil,subprocess
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT / 'scripts'))
from ego_data.calibration import CameraCalibration
from camera_models import project_points

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline-pilot',type=Path);p.add_argument('--initial-sync',type=int);p.add_argument('--dataset',type=Path);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--compare-review',type=Path);a=p.parse_args();out=a.pilot/'review';out.mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True);exp=ROOT/'output/native_kb_training_20261111';dataset=a.dataset or exp/'real_dual/pair_20261111_192916/ego/normalized_multiview';tracks=[np.load(a.pilot/f'{h}_pilot.npz') for h in ['left','right']];maps=[{int(f):i for i,f in enumerate(z['pair_indices']) if not z['calibration_mask'][i]} for z in tracks];syncs=sorted(set(maps[0])&set(maps[1]));faces=tracks[0]['faces'];records=[];cals={f'camera{i}':CameraCalibration.load(dataset/'calibration'/f'camera{i}.json') for i in range(4)};C0=cals['camera0']
 previous=[np.load(a.baseline_pilot/f'{h}_pilot.npz') for h in ['left','right']] if a.baseline_pilot else None
 previous_maps=[{int(f):i for i,f in enumerate(z['pair_indices'])} for z in previous] if previous else None
 with (dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))
 for camera,C in cals.items():
  cap=cv2.VideoCapture(str(dataset/'cameras'/camera/'video.mp4'));video=cv2.VideoWriter(str(out/f'{camera}_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),3.75,(1920,440));count=0
  if not video.isOpened():raise RuntimeError('video writer failed')
  for sync in syncs:
   r=rows[sync];source=int(r[camera+'_frame_index']);cap.set(cv2.CAP_PROP_POS_FRAMES,source);ok,image=cap.read()
   if not ok:raise RuntimeError(f'missing image {camera} {source}')
   w,h=image.shape[1],image.shape[0];imagepath=f'images/{camera}_{sync:06d}.jpg';cv2.imwrite(str(out/imagepath),image,[cv2.IMWRITE_JPEG_QUALITY,95]);record=dict(camera=camera,sync=sync,source='marker24-validation',source_name='24-marker pilot: validation frames, NOT approved labels',source_frame=source,timestamp=int(r['reference_timestamp_ns']),camera_timestamp=int(r[camera+'_timestamp_ns']),sync_delta_ms=(int(r[camera+'_timestamp_ns'])-int(r['camera0_timestamp_ns']))/1e6,image=imagepath,size=[w,h],hands=[],baseline_hands=[])
   for side,z in enumerate(tracks):
    i=maps[side][sync]
    for prefix,dest in [('',record['hands']),('baseline_',record['baseline_hands'])]:
     if prefix and previous is not None:
      old=previous[side];old_i=previous_maps[side][sync];v=old['vertices'][old_i].copy();j=old['joints'][old_i].copy()
     else:v=z[prefix+'vertices'][i].copy();j=z[prefix+'joints'][i].copy()
     if side==0:v[:,0]*=-1;j[:,0]*=-1
     def project(q):
      base=q@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3];cam=(base-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];uv,visible=project_points(C,cam);return uv,cam[:,2]
     uv,depth=project(v);jpx,jz=project(j)
     if not np.isfinite(uv).all() or np.min(depth)<.03:continue
     inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
     if not inside.any():continue
     bbox=np.r_[uv.min(0)-20,uv.max(0)+20]
     dest.append(dict(side=side,vertices=np.round(np.c_[uv,depth],5).tolist(),joints=np.round(jpx,4).tolist(),bbox=bbox.tolist(),visible_vertex_fraction=float(inside.mean()),label=str(a.pilot/f'{"left" if side==0 else "right"}_pilot.npz'),K=C.K.tolist(),distortion=C.distortion.tolist()))
   records.append(record);panels=[]
   for title,hands in [('RAW',[]),('BASELINE',record['baseline_hands']),('MARKER24 PILOT',record['hands'])]:
    frame=image.copy();layer=frame.copy()
    for hand in hands:
     uv=np.asarray(hand['vertices'])[:,:2];polys=np.rint(uv[faces]).astype(np.int32);color=(50,225,195) if hand['side']==0 else (30,155,255);cv2.polylines(layer,polys,True,color,1,cv2.LINE_AA)
    frame=cv2.addWeighted(layer,.5,frame,.5,0);frame=cv2.resize(frame,(640,400));panel=np.zeros((440,640,3),np.uint8);panel[40:]=frame;cv2.putText(panel,f'{title} | {camera} sync {sync}',(12,26),0,.65,(240,240,240),1,cv2.LINE_AA);panels.append(panel)
   comparison=np.concatenate(panels,axis=1);video.write(comparison);count+=1
   if sync in [syncs[0],syncs[len(syncs)//2],syncs[-1]]:cv2.imwrite(str(out/f'{camera}_{sync:06d}_comparison.jpg'),comparison)
  cap.release();video.release()
  if shutil.which('ffmpeg'):
   source_video=out/f'{camera}_comparison.mp4';encoded=out/f'{camera}_comparison_h264.mp4'
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source_video),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],check=True)
   encoded.replace(source_video)
  print(camera,count,flush=True)
 if a.compare_review:
  other=json.loads((a.compare_review/'data.js').read_text()[len('window.LABEL_DATA='):-2]);lookup={(f['camera'],f['sync']):f for f in other['frames']}
  for f in records:f['holdout_hands']=lookup[(f['camera'],f['sync'])]['hands']
 data=dict(initial_sync=a.initial_sync,dataset=str(a.pilot.resolve()),projection='native_raw / KB; experimental, includes partial views',faces=faces.tolist(),frames=records,max_projection_difference_px=None,label_count=sum(len(f['hands']) for f in records))
 (out/'data.js').write_text('window.LABEL_DATA='+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
 html=(ROOT/'tools/label_pipeline/calibration/marker_placement/label_viewer.html').read_text();html=html.replace('grid-template-columns:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;').replace('<section class="panel"><h2>相同原图 + 导出标签</h2>','<section class="panel"><h2>当前基线</h2><canvas id="baseline"></canvas></section><section class="panel"><h2>24 点附着模型 · 实验结果</h2>');html=html.replace("canvases=[$('raw'),$('overlay')]","canvases=[$('raw'),$('overlay'),$('baseline')]").replace('ci<2','ci<3').replace("const hands=frames[index].hands.filter", "const hands=(ci===2?frames[index].baseline_hands:frames[index].hands).filter")
 html=html.replace('左侧原图，右侧叠加当前导出的 mesh。两侧同步缩放和平移，图片保持原始畸变。','原图 / 基线 / 新方案三栏同步缩放。此页为实验结果，未通过验收；包含未满足完整标签导出条件的局部视图。').replace('href="marker_placement/"','href="../marker_placement/"').replace('已导出标签 ${D.label_count} 份','实验可见手实例 ${D.label_count} 份').replace('当前标签与投影信息','当前实验与投影信息')
 html=html.replace('未导出的帧不会补插或显示为有效标签。','本页是采样实验对照，包含基线未导出的局部视图，不可直接作为训练标签。')
 html=html.replace('<div id="help">','<div class="bar">对照视频：'+''.join(f'<a href="{c}_comparison.mp4">{c}</a>' for c in cals)+'</div><div id="help">')
 if json.loads((a.pilot/'summary.json').read_text()).get('surface_hemisphere') == -1:
  html=html.replace('24 点附着模型 · 实验结果','24 点附着模型 · 手背方向已修正').replace('<h1>原图与 MANO Mesh 标签检查</h1>','<h1>手背方向修正版 · DORSAL-FIX-1</h1>').replace('src="data.js"','src="data.js?v=dorsal-fix-1"')
 if json.loads((a.pilot/'summary.json').read_text()).get('fit_all_markers'):
  html=html.replace('手背方向修正版 · DORSAL-FIX-1','全 24 点拟合对照 · ALL24-FIT-1').replace('24 点附着模型 · 手背方向已修正','全部 24 点参与拟合 · 方向已修正').replace('data.js?v=dorsal-fix-1','data.js?v=all24-fit-1')
 if a.compare_review:
  html=html.replace('<label>叠加 <select id="mode">','<label>实验版本 <select id="trial"><option value="all">全 24 点拟合</option><option value="holdout">旧：留出五个指尖</option></select></label><label>叠加 <select id="mode">')
  html=html.replace('全部 24 点参与拟合 · 方向已修正','<span id="trialTitle">全部 24 点参与拟合 · 方向已修正</span>')
  html=html.replace("ci===2?frames[index].baseline_hands:frames[index].hands", "ci===2?frames[index].baseline_hands:$('trial').value==='holdout'?frames[index].holdout_hands:frames[index].hands")
  html=html.replace('window.onresize=resize;', "$('trial').onchange=()=>{$('trialTitle').textContent=$('trial').value==='all'?'全部 24 点参与拟合 · 方向已修正':'旧诊断实验：五个指尖未参与拟合';draw()};window.onresize=resize;")
 if json.loads((a.pilot/'summary.json').read_text()).get('finger_pose_frozen'):
  html=html.replace('全 24 点拟合对照 · ALL24-FIT-1','基线手指姿态锁定 · POSE-LOCK-1').replace('全部 24 点参与拟合 · 方向已修正','基线手指姿态锁定 · 仅整体旋转平移').replace('data.js?v=all24-fit-1','data.js?v=pose-lock-1').replace('全 24 点拟合</option>','锁定基线手指姿态</option>').replace('旧：留出五个指尖','对照：全24点自由姿态').replace('旧诊断实验：五个指尖未参与拟合','对照：全24点自由姿态')
 if json.loads((a.pilot/'summary.json').read_text()).get('strategy','').startswith('personal'):
  html=html.replace('<h1>原图与 MANO Mesh 标签检查</h1>','<h1>共享个人手型＋24点＋原图轮廓（实验）</h1>').replace('24 点附着模型 · 实验结果','共享个人手型 · 当前优化结果')
 if a.initial_sync is not None:html=html.replace('resize();selectFrames();','resize();selectFrames(D.initial_sync);')
 if json.loads((a.pilot/'summary.json').read_text()).get('joint_pose_attachment_optimization_enabled'):
  html=html.replace('<h1>原图与 MANO Mesh 标签检查</h1>','<h1>每帧动作＋共享附着联合修正 · JOINT-PA-1</h1>').replace('当前基线','修正前：当前标签').replace('24 点附着模型 · 实验结果','修正后：动作＋附着同时优化')
  html=html.replace('MARKER24 PILOT','JOINT POSE + ATTACHMENT')
 (out/'index.html').write_text(html);(out/'review_report.json').write_text(json.dumps(dict(validation_syncs=syncs,frames=len(records),images=records[0]['size'],cameras=list(cals),approved_for_training=False),indent=2));print(out)

if __name__=='__main__':main()
