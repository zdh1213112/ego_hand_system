"""Check parameter isolation and publish all ablation results without promoting labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import EXP,stats
from refine_contours import project
from ablate_geometry import split_views

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=EXP/'personal_calibration_v3');ap.add_argument('--observations',type=Path,default=EXP/'personal_contours_external_v2/observations.json');ap.add_argument('--runs-root',type=Path,default=EXP);ap.add_argument('--output',type=Path,default=EXP/'personal_ablation_report_v2');a=ap.parse_args()
 out=a.output;out.mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True)
 obs=json.loads(a.observations.read_text());lookup={(r['camera'],r['sync'],r['side']):r for r in obs['records']};report=dict(status='diagnostic_only_not_approved',modes={},checks=[])
 tracks={};summaries={}
 for mode in ['rigid','fingers','attachments','recalibrated_fingers']:
  source=a.runs_root/f'personal_ablation_{mode}_v2';s=json.loads((source/'summary.json').read_text());summaries[mode]=s;report['modes'][mode]={}
  for side,hand in enumerate(['left','right']):
   b=np.load((a.runs_root/'personal_ablation_attachments_v2' if mode=='recalibrated_fingers' else a.source)/f'{hand}_pilot.npz');z=np.load(source/f'{hand}_pilot.npz');tracks[(mode,side)]=z;tracks[('source',side)]=np.load(a.source/f'{hand}_pilot.npz')
   assert np.array_equal(b['betas'],z['betas'])
   frozen={'rigid':['hand_pose_pca','attachment_faces','barycentric','normal_offsets_m','wrist_local'],'fingers':['global_orient','translation','attachment_faces','barycentric','normal_offsets_m','wrist_local'],'recalibrated_fingers':['global_orient','translation','attachment_faces','barycentric','normal_offsets_m','wrist_local'],'attachments':['hand_pose_pca','global_orient','translation','vertices','joints']}[mode]
   for k in frozen:assert np.allclose(b[k],z[k],atol=2e-7,rtol=0),(mode,hand,k)
   assert np.isfinite(z['vertices']).all() and np.min(z['normal_offsets_m'])>=0
   fit,held,_=split_views([r for r in obs['records'] if r['side']==side]);assert not {(r['camera'],r['sync']) for r in fit}&{(r['camera'],r['sync']) for r in held}
   h=s['hands'][hand];change=h['held_camera_comparison'];d=np.array([c['delta_px'] for c in change]);row=dict(held_views=len(d),before_px=h['before']['held_camera_views']['mean_px'],after_px=h['after']['held_camera_views']['mean_px'],improved_over_1px=int((d<-1).sum()),worsened_over_1px=int((d>1).sum()),delta_px=stats(d),marker_before_mm=h['before']['eligible_marker_mm'],marker_after_mm=h['after']['eligible_marker_mm'],limits=h['limit_counts'],worst=sorted(change,key=lambda c:-c['delta_px'])[:5],best=sorted(change,key=lambda c:c['delta_px'])[:5]);report['modes'][mode][hand]=row;report['checks'].append(dict(mode=mode,hand=hand,shape_frozen=True,parameter_isolation_passed=True,held_camera_disjoint=True))
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['modes'],indent=2),flush=True)
 # Original images for validation are already extracted; calibration examples remain in full JSON.
 image_dir=a.source/'review/images';wanted=set()
 for cam in ['camera0','camera1','camera2','camera3']:
  for sync in [576,616,624,632,640,648]:
   for side in [0,1]:
    if (cam,sync,side) in lookup:wanted.add((cam,sync,side))
 for mode in ['rigid','fingers']:
  for side,hand in enumerate(['left','right']):
   for r in report['modes'][mode][hand]['best']+report['modes'][mode][hand]['worst']:
    if (image_dir/f'{r["camera"]}_{r["sync"]:06d}.jpg').exists():wanted.add((r['camera'],r['sync'],side))
 cards=[]
 for cam,sync,side in sorted(wanted):
  path=image_dir/f'{cam}_{sync:06d}.jpg'
  if not path.exists():continue
  image=cv2.imread(str(path));r=lookup[(cam,sync,side)];uvs=[]
  for mode in ['source','rigid','fingers','recalibrated_fingers']:
   z=tracks[(mode,side)];i=int(np.flatnonzero(z['pair_indices']==sync)[0]);uvs.append(project(torch.tensor(z['vertices'][i],dtype=torch.float64),r,side).numpy())
  points=np.concatenate(uvs);lo=np.maximum(0,np.floor(points.min(0)-30)).astype(int);hi=np.minimum(image.shape[1::-1],np.ceil(points.max(0)+30)).astype(int)
  if np.any(hi<=lo):continue
  panels=[]
  for title,uv in zip(['RAW','SOURCE V3','RIGID ONLY','FINGERS ONLY','RECAL + FINGERS'],[None]+uvs):
   layer=image.copy()
   if uv is not None:cv2.polylines(layer,np.rint(uv[z['faces']]).astype(np.int32),True,(30,155,255) if side else (195,225,50),1,cv2.LINE_AA)
   crop=cv2.addWeighted(layer,.5,image,.5,0)[lo[1]:hi[1],lo[0]:hi[0]];h,w=crop.shape[:2];sc=min(450/w,510/h);res=cv2.resize(crop,(max(1,round(w*sc)),max(1,round(h*sc))));panel=np.zeros((550,450,3),np.uint8);panel[40:40+res.shape[0],:res.shape[1]]=res;cv2.putText(panel,title,(12,26),0,.6,(255,255,255),1);panels.append(panel)
  name=f'{cam}_{sync:06d}_{side}.jpg';cv2.imwrite(str(out/'images'/name),np.concatenate(panels,axis=1));cards.append(dict(camera=cam,sync=sync,side=side,image='images/'+name))
 html='''<!doctype html><meta charset="utf-8"><title>位姿与附着分项对照</title><style>body{background:#102735;color:white;font:17px system-ui;padding:20px}img{width:100%}select,button{font:inherit;padding:10px;margin:5px}td,th{padding:8px;border:1px solid #607380}a{color:#82d9ff}</style><h1>固定手型 · 分项对照（未验收）</h1><p>每帧留一路相机不参与本轮优化。左至右：原图／上一版 V3／仅整体位姿／仅手指姿态／重标定附着后调整手指。附着单独变化不会改变 mesh，不重复显示。这里没有加入新手型变化。</p><div id="table"></div><p>指标为自动外轮廓距离，不是三维真值精度。也包含退步帧，不能根据平均值直接批准标签。相机留出只能检查本轮增量，之前初始化并非完全独立。</p><select id="choice"></select><button id="prev">上一张</button><button id="next">下一张</button><button id="play">播放采样图</button><button id="full">全屏</button><p id="caption"></p><img id="image"><p>采样图每秒 2 张，可跨相机与手；并非连续原视频。<a href="report.json">全部量化结果与退步帧</a> · <a href="RESULT_zh.md">结论与复现指令</a></p><script>const cards=CARDS,data=REPORT,$=x=>document.getElementById(x);let timer=null;const names={rigid:'仅整手位姿',fingers:'仅手指姿态',attachments:'仅附着',recalibrated_fingers:'重标定附着＋手指'};let h='<table><tr><th>方案</th><th>左手：前→后 px</th><th>右手：前→后 px</th></tr>';for(const [mode,rows] of Object.entries(data.modes)){h+='<tr><td>'+names[mode]+'</td>';for(const side of ['left','right'])h+='<td>'+rows[side].before_px.toFixed(2)+' → '+rows[side].after_px.toFixed(2)+'</td>';h+='</tr>'}$('table').innerHTML=h+'</table>';cards.forEach((c,i)=>$('choice').add(new Option(c.camera+' / '+c.sync+' / '+(c.side?'右手':'左手'),i)));function draw(){const c=cards[+$('choice').value];$('image').src=c.image;$('caption').textContent=c.camera+' · sync '+c.sync+' · '+(c.side?'右手':'左手')};$('choice').onchange=draw;$('next').onclick=()=>{$('choice').selectedIndex=($('choice').selectedIndex+1)%cards.length;draw()};$('prev').onclick=()=>{$('choice').selectedIndex=($('choice').selectedIndex-1+cards.length)%cards.length;draw()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放采样图'}else{timer=setInterval(()=>$('next').click(),500);$('play').textContent='暂停'}};$('full').onclick=()=>$('image').requestFullscreen();const initial=cards.findIndex(c=>c.camera==='camera0'&&c.sync===632&&c.side===1);if(initial>=0)$('choice').selectedIndex=initial;draw();</script>'''
 (out/'index.html').write_text(html.replace('CARDS',json.dumps(cards)).replace('REPORT',json.dumps(report)));link=EXP/'baseline_mesh_viewer/personal_ablation'
 if not link.exists():link.symlink_to(out.resolve(),target_is_directory=True)
 print('CARDS',len(cards),flush=True)
if __name__=='__main__':main()
