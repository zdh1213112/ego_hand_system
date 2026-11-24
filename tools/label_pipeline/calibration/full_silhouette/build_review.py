#!/usr/bin/env python3
"""Three synchronized original-camera panels, previous/new shape controls and full target silhouettes."""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent/'ego_hand_system/scripts'))
from ego_data.calibration import CameraCalibration
from camera_models import project_points
from render_masks import load_hand_mask


def main():
    ap=argparse.ArgumentParser()
    for name in ['previous-review','selection','shape-only','shape-pose','dataset','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output/'images').symlink_to((a.previous_review/'images').resolve(),target_is_directory=True)
    previous=json.loads((a.previous_review/'data.js').read_text().removeprefix('window.LABEL_DATA=').strip().rstrip(';'))
    selection=json.loads(a.selection.read_text())
    targets={(r['camera'],r['sync'],r['side']):r for r in selection['records'] if r['accepted_quality']}
    tracks={mode:[np.load(path/f'{h}_pilot.npz') for h in ['left','right']] for mode,path in [('pure',a.shape_only),('pose',a.shape_pose)]}
    lookup={mode:[{int(f):i for i,f in enumerate(z['pair_indices'])} for z in zs] for mode,zs in tracks.items()}
    cameras={f'camera{i}':CameraCalibration.load(a.dataset/'calibration'/f'camera{i}.json') for i in range(4)}
    c0=cameras['camera0'];records=[];cv2.setNumThreads(2)
    for original in previous['frames']:
        record=copy.deepcopy(original);record['source']='full-mask-shape';record['source_name']='全掩膜共享手型：未参与拟合的图像'
        record['baseline_hands']=original['hands'];record['hands']=[];record['pose_hands']=[];record['target_boundaries']=[]
        camera=cameras[record['camera']]
        for side in [0,1]:
            key=(record['camera'],record['sync'],side)
            if key in targets:
                mask=load_hand_mask(targets[key]['mask'])
                contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                record['target_boundaries'].append({'side':side,'lines':[q.reshape(-1,2).tolist() for q in contours]})
            for mode,dest in [('pure',record['hands']),('pose',record['pose_hands'])]:
                z=tracks[mode][side];i=lookup[mode][side][record['sync']]
                def proj(points):
                    points=points.copy();points[:,0]*=-1 if side==0 else 1
                    base=points@c0.T_base_camera[:3,:3].T+c0.T_base_camera[:3,3]
                    cam=(base-camera.T_base_camera[:3,3])@camera.T_base_camera[:3,:3]
                    return project_points(camera,cam)[0],cam[:,2]
                uv,depth=proj(z['vertices'][i]);jp,_=proj(z['joints'][i])
                if not np.isfinite(uv).all() or depth.min()<.03:continue
                w,h=record['size'];inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
                if not inside.any():continue
                dest.append({'side':side,'vertices':np.round(np.c_[uv,depth],5).tolist(),
                             'joints':np.round(jp,4).tolist(),'bbox':np.r_[uv.min(0)-20,uv.max(0)+20].tolist(),
                             'visible_vertex_fraction':float(inside.mean()),'label':str((a.shape_only if mode=='pure' else a.shape_pose)/f'{"left" if side==0 else "right"}_pilot.npz')})
        records.append(record)
    data={'initial_sync':664,'dataset':str(a.shape_only.resolve()),'projection':'native_raw / true KB; full-mask experiment',
          'faces':previous['faces'],'frames':records,'label_count':sum(len(r['hands']) for r in records),
          'max_projection_difference_px':None,'approved_for_training':False}
    (a.output/'data.js').write_text('window.LABEL_DATA='+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
    html=(ROOT/'tools/marker_placement/label_viewer.html').read_text()
    html=html.replace('grid-template-columns:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;')
    html=html.replace('<section class="panel"><h2>相同原图 + 导出标签</h2>', '<section class="panel"><h2>上一轮：优质样本＋边界点拟合</h2><canvas id="baseline"></canvas></section><section class="panel"><h2><span id="trialTitle">完整掩膜：只调整共享手型</span></h2>')
    html=html.replace("canvases=[$('raw'),$('overlay')]","canvases=[$('raw'),$('overlay'),$('baseline')]").replace('ci<2','ci<3')
    html=html.replace('真实相机原图','原图＋绿色目标轮廓')
    html=html.replace('<h1>原图与 MANO Mesh 标签检查</h1>','<h1>完整轮廓驱动的共享手型拟合 · FULL-MASK-1</h1>')
    html=html.replace('左侧原图，右侧叠加当前导出的 mesh。两侧同步缩放和平移，图片保持原始畸变。',
        '绿色是通过质量筛选的分割边界；青色／橙色是左右手 mesh。三栏同步缩放。左右手各一套共享手型，所有留出帧动作和附着关系保持上一轮数值。袖口／手臂不参与拟合；没有合格分割的帧不画绿色。实验结果，尚未批准为真值。')
    html=html.replace('<label>叠加 <select id="mode">','<label>对照 <select id="trial"><option value="pure">只调整共享手型</option><option value="pose">标定帧动作微调后得到的共享手型</option></select></label><label><input type="checkbox" id="targetMask" checked>绿色目标轮廓</label><label>叠加 <select id="mode">')
    html=html.replace("const hands=frames[index].hands.filter", "const hands=(ci===2?frames[index].baseline_hands:$('trial').value==='pose'?frames[index].pose_hands:frames[index].hands).filter")
    html=html.replace('ctx.drawImage(img,0,0);', '''ctx.drawImage(img,0,0);if($('targetMask').checked&&!hide){ctx.globalAlpha=1;ctx.strokeStyle='#7dff54';ctx.lineWidth=1.5/scale;for(const t of frames[index].target_boundaries||[]){if(!$(t.side?'right':'left').checked)continue;for(const line of t.lines){if(!line.length)continue;ctx.beginPath();ctx.moveTo(...line[0]);for(const p of line.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.stroke()}}}''')
    html=html.replace('已导出标签 ${D.label_count} 份','实验手实例 ${D.label_count} 份')
    html=html.replace("window.onresize=resize;", "$('targetMask').onchange=draw;$('trial').onchange=()=>{$('trialTitle').textContent=$('trial').value==='pure'?'完整掩膜：只调整共享手型':'完整掩膜：标定帧动作微调对照';draw()};window.onresize=resize;")
    html=html.replace('resize();selectFrames();','resize();selectFrames(D.initial_sync);')
    html=html.replace("if(!viewInitialized||!$('keepZoom')?.checked)fit();else draw();", "if(!viewInitialized){fit(1);setTimeout(()=>{if(seq===token)fit(1)},250)}else if(!$('keepZoom')?.checked)fit();else draw();")
    html=html.replace('<div id="help">','<div class="bar"><a href="../validation.json">独立验证报告</a><a href="../closeups/">四路双手近景对照视频</a></div><div id="help">')
    html=html.replace('</html>',"<script>$('markerLink')?.remove();document.title='完整掩膜共享手型对照';</script></html>")
    (a.output/'index.html').write_text(html)
    print(json.dumps({'frames':len(records),'hands':data['label_count'],'target_hand_views':sum(len(r['target_boundaries']) for r in records),'viewer':str(a.output)},indent=2))


if __name__=='__main__':main()
