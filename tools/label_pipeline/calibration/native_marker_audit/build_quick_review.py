#!/usr/bin/env python3
"""Build read-only, no-click diagnostic cards from existing pose-locked results."""
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
EXP=ROOT/'output/native_kb_training_20261111'
def main():
 src=EXP/'marker24_pilot_pose_locked/review'
 data=json.loads((src/'data.js').read_text().removeprefix('window.LABEL_DATA=').strip().rstrip(';'))
 out=EXP/'automatic_mesh_review';out.mkdir(exist_ok=True)
 faces=np.asarray(data['faces']);rows=[];cards=[]
 selected={616,624,632,640,648}
 for f in data['frames']:
  for side in [0,1]:
   a=next((h for h in f['baseline_hands'] if h['side']==side),None);b=next((h for h in f['hands'] if h['side']==side),None)
   if a is None or b is None:continue
   u=np.asarray(a['vertices'])[:,:2];v=np.asarray(b['vertices'])[:,:2];dist=np.linalg.norm(v-u,axis=1)
   rows.append(dict(camera=f['camera'],sync=f['sync'],side=side,median_movement_px=float(np.median(dist)),p95_movement_px=float(np.percentile(dist,95)),note='Change magnitude, not error against image ground truth'))
   if side!=1 or f['sync'] not in selected:continue
   im=cv2.imread(str(src/f['image']));assert im is not None
   pts=np.r_[u,v];lo=np.maximum(0,np.floor(pts.min(0)-40)).astype(int);hi=np.minimum(im.shape[1::-1],np.ceil(pts.max(0)+40)).astype(int)
   if np.any(hi<=lo):continue
   panels=[]
   for title,uv in [('RAW',None),('BASELINE',u),('POSE LOCKED',v)]:
    layer=im.copy()
    if uv is not None:cv2.polylines(layer,np.rint(uv[faces]).astype(np.int32),True,(30,155,255),1,cv2.LINE_AA)
    crop=cv2.addWeighted(layer,.55,im,.45,0)[lo[1]:hi[1],lo[0]:hi[0]];h,w=crop.shape[:2];s=min(450/w,510/h);res=cv2.resize(crop,(max(1,round(w*s)),max(1,round(h*s))))
    panel=np.full((550,450,3),18,np.uint8);panel[40:40+res.shape[0],:res.shape[1]]=res;cv2.putText(panel,title,(12,27),0,.65,(245,245,245),1);panels.append(panel)
   name=f'{f["camera"]}_{f["sync"]:06d}.jpg';assert cv2.imwrite(str(out/name),np.concatenate(panels,axis=1));cards.append(dict(camera=f['camera'],sync=f['sync'],image=name,movement=round(float(np.median(dist)),2)))
 report=json.loads((EXP/'marker24_pilot_pose_locked/diagnosis_right_000632/report.json').read_text())
 (out/'metrics.json').write_text(json.dumps(dict(scope='Existing sampled validation frames, not continuous video',metrics=rows,frame632=report),ensure_ascii=False,indent=2))
 html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>免逐点核对 · 自动对比</title><style>body{background:#102331;color:#eef6ff;font:17px system-ui;margin:24px}button,select{font:inherit;padding:10px;margin:5px}img{width:100%;cursor:zoom-in}p{line-height:1.7}.box{background:#1f394b;padding:18px;border-radius:12px}a{color:#80d6ff}</style><h1>免逐点核对 · 632 帧右手及前后采样帧</h1><div class="box">已自动整理原图、基线、优化结果，直接看对比即可，不需要点击 marker。<p>632 帧：手指姿态和手型参数未改变，但整手旋转变化 <b>6.93°</b>，腕部移动 <b>21.54 mm</b>。camera0 网格投影移动中位数 <b>8.34 px</b>。说明该帧的直接变化是整手刚体移动，不是手指关节再次变形。</p><p>附着模型拟合残差从 10.48 降至 3.07 mm，但视觉效果退步：拟合目标变小不保证贴合真实手。尚不能唯一确定是附着位置、手型还是时空标定导致。此处展示变化幅度，不将它当作图像真实误差。</p></div><p>相机 <select id="cam"></select> <button id="prev">上一帧</button><button id="play">播放采样帧</button><button id="next">下一帧</button><span id="label"></span></p><p>左：原图；中：基线；右：锁定手指姿态后的优化。点击图片全屏，Esc 退出。播放为每秒 2 张采样图，不是原视频连续运动。</p><img id="image"><p id="empty"></p><p>分析下一步：围绕整手位移／旋转做受控对照，检查附着约束为何推动整体偏移；本页未修改任何标签。<a href="metrics.json">完整自动变化统计</a></p><script>const cards=CARDS;const $=id=>document.getElementById(id);let i=2,t=null;for(const c of ['camera0','camera1','camera2','camera3'])$('cam').add(new Option(c,c));function items(){return cards.filter(c=>c.camera===$('cam').value)}function draw(){const a=items();i=Math.max(0,Math.min(i,a.length-1));$('image').hidden=!a.length;$('empty').textContent=a.length?'':'该相机没有这组右手的可用对比，未用其他相机替代。';if(!a.length){$('label').textContent='';return}const f=a[i];$('image').src=f.image;$('label').textContent='sync '+f.sync+' · 网格移动中位数 '+f.movement+' px'}$('cam').onchange=()=>{i=2;draw()};$('prev').onclick=()=>{i--;draw()};$('next').onclick=()=>{i=(i+1)%Math.max(1,items().length);draw()};$('play').onclick=()=>{if(t){clearInterval(t);t=null;$('play').textContent='播放采样帧'}else{t=setInterval(()=>$('next').click(),500);$('play').textContent='暂停'}};$('image').onclick=()=>{$('image').requestFullscreen().catch(()=>{})};draw();</script></html>'''
 controls=Path(__file__).with_name('quick_review_controls.html').read_text()
 (out/'index.html').write_text(html.replace('CARDS',json.dumps(cards)).replace('</html>',controls+'</html>'))
 link=EXP/'baseline_mesh_viewer/automatic_review'
 if not link.exists():link.symlink_to(out,target_is_directory=True)
 print(json.dumps(dict(cards=len(cards),compared_hand_views=len(rows),path=str(out))))
if __name__=='__main__':main()
