#!/usr/bin/env python3
"""Three synchronized original-camera panels, previous/new shape controls and full target silhouettes."""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT / 'scripts'))
from ego_data.calibration import CameraCalibration
from camera_models import project_points


def main():
    ap=argparse.ArgumentParser()
    for name in ['previous-review','sdk-hand','sdk-finger0','dataset','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output/'images').symlink_to((a.previous_review/'images').resolve(),target_is_directory=True)
    previous=json.loads((a.previous_review/'data.js').read_text().removeprefix('window.LABEL_DATA=').strip().rstrip(';'))
    tracks={mode:[np.load(path/f'{h}_pilot.npz') for h in ['left','right']] for mode,path in [('pure',a.sdk_hand),('pose',a.sdk_finger0)]}
    lookup={mode:[{int(f):i for i,f in enumerate(z['pair_indices'])} for z in zs] for mode,zs in tracks.items()}
    cameras={f'camera{i}':CameraCalibration.load(a.dataset/'calibration'/f'camera{i}.json') for i in range(4)}
    c0=cameras['camera0'];records=[];cv2.setNumThreads(2)
    for original in previous['frames']:
        record=copy.deepcopy(original);record['source']='sdk-assumed-internal';record['source_name']='SDK 已补偿内部偏移（假设） · 拟合对照'
        record['baseline_hands']=original['hands'];record['hands']=[];record['pose_hands']=[];record['target_boundaries']=original.get('target_boundaries',[])
        camera=cameras[record['camera']]
        for side in [0,1]:
            for mode,dest in [('pure',record['hands']),('pose',record['pose_hands'])]:
                z=tracks[mode][side];i=lookup[mode][side][record['sync']]
                def proj(points):
                    points=points.copy();points[:,0]*=-1 if side==0 else 1
                    base=points@c0.T_base_camera[:3,:3].T+c0.T_base_camera[:3,3]
                    cam=(base-camera.T_base_camera[:3,3])@camera.T_base_camera[:3,:3]
                    return project_points(camera,cam)[0],cam[:,2]
                uv,depth=proj(z['vertices'][i]);jp,_=proj(z['joints'][i]);sdk,_=proj(z['sdk_target21'][i])
                if not np.isfinite(uv).all() or depth.min()<.03:continue
                w,h=record['size'];inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
                if not inside.any():continue
                dest.append({'side':side,'vertices':np.round(np.c_[uv,depth],5).tolist(),
                             'joints':np.round(jp,4).tolist(),'sdk_joints':np.round(sdk,4).tolist(),'bbox':np.r_[uv.min(0)-20,uv.max(0)+20].tolist(),
                             'visible_vertex_fraction':float(inside.mean()),'label':str((a.sdk_hand if mode=='pure' else a.sdk_finger0)/f'{"left" if side==0 else "right"}_pilot.npz')})
        records.append(record)
    data={'initial_sync':632,'dataset':str(a.sdk_hand.resolve()),'projection':'native_raw / true KB; SDK internal offsets ASSUMED',
          'faces':previous['faces'],'frames':records,'label_count':sum(len(r['hands']) for r in records),
          'max_projection_difference_px':None,'approved_for_training':False}
    (a.output/'data.js').write_text('window.LABEL_DATA='+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
    html=(ROOT/'tools/label_pipeline/calibration/marker_placement/label_viewer.html').read_text()
    html=html.replace('grid-template-columns:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;')
    html=html.replace('<section class="panel"><h2>相同原图 + 导出标签</h2>', '<section class="panel"><h2>当前：轮廓手型＋marker动作</h2><canvas id="baseline"></canvas></section><section class="panel"><h2><span id="trialTitle">SDK Finger0 公共根：动作拟合对照</span></h2>')
    html=html.replace("canvases=[$('raw'),$('overlay')]","canvases=[$('raw'),$('overlay'),$('baseline')]").replace('ci<2','ci<3')
    html=html.replace('真实相机原图','原图＋可靠分割轮廓')
    html=html.replace('<h1>原图与 MANO Mesh 标签检查</h1>','<h1>SDK骨架假定内部关节 · 标签对照</h1>')
    html=html.replace('左侧原图，右侧叠加当前导出的 mesh。两侧同步缩放和平移，图片保持原始畸变。',
        '按“SDK 已补偿 marker 到内部关节偏移”假设拟合 MANO 动作，左右手型固定，没有额外减 2mm。绿色是可靠分割，青色／橙色是左右手 mesh。可切换两种腕根映射，紫色 SDK21 目标可独立开关。该假设和节点映射尚未核实，结果保留实验状态。')
    html=html.replace('<label>叠加 <select id="mode">','<label>对照 <select id="trial"><option value="pure">SDK Hand 根</option><option value="pose" selected>SDK Finger0 公共根</option></select></label><label><input type="checkbox" id="sdkTargets">紫色 SDK21 目标</label><label><input type="checkbox" id="targetMask" checked>绿色分割轮廓</label><label>叠加 <select id="mode">')
    html=html.replace("const hands=frames[index].hands.filter", "const hands=(ci===2?frames[index].baseline_hands:$('trial').value==='pose'?frames[index].pose_hands:frames[index].hands).filter")
    html=html.replace('ctx.drawImage(img,0,0);', '''ctx.drawImage(img,0,0);if($('targetMask').checked&&!hide){ctx.globalAlpha=1;ctx.strokeStyle='#7dff54';ctx.lineWidth=1.5/scale;for(const t of frames[index].target_boundaries||[]){if(!$(t.side?'right':'left').checked)continue;for(const line of t.lines){if(!line.length)continue;ctx.beginPath();ctx.moveTo(...line[0]);for(const p of line.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.stroke()}}}''')
    sdk_draw="if($('sdkTargets').checked&&!hide){const targets=$('trial').value==='pose'?frames[index].pose_hands:frames[index].hands;ctx.strokeStyle='#ee61f2';ctx.fillStyle='#ee61f2';ctx.lineWidth=1.5/scale;const edges=[[0,1],[1,2],[2,3],[3,4],...[5,9,13,17].flatMap(k=>[[0,k],[k,k+1],[k+1,k+2],[k+2,k+3]])];for(const h of targets){if(!$(h.side?'right':'left').checked)continue;for(const [a,b] of edges){ctx.beginPath();ctx.moveTo(...h.sdk_joints[a]);ctx.lineTo(...h.sdk_joints[b]);ctx.stroke()}for(const p of h.sdk_joints){ctx.beginPath();ctx.arc(p[0],p[1],2.5/scale,0,Math.PI*2);ctx.fill()}}}"
    html=html.replace('if(!ci||hide)continue;',sdk_draw+'if(!ci||hide)continue;')
    html=html.replace('已导出标签 ${D.label_count} 份','实验手实例 ${D.label_count} 份')
    html=html.replace("window.onresize=resize;", "$('targetMask').onchange=draw;$('sdkTargets').onchange=draw;$('trial').onchange=()=>{$('trialTitle').textContent=$('trial').value==='pure'?'SDK Hand 根：动作拟合对照':'SDK Finger0 公共根：动作拟合对照';draw()};window.onresize=resize;")
    html=html.replace('resize();selectFrames();','resize();selectFrames(D.initial_sync);')
    html=html.replace("if(!viewInitialized||!$('keepZoom')?.checked)fit();else draw();", "if(!viewInitialized){fit(1);setTimeout(()=>{if(seq===token)fit(1)},250)}else if(!$('keepZoom')?.checked)fit();else draw();")
    html=html.replace('<div id="help">','<div class="bar"><a href="../comparison_metrics.json">拟合与轮廓比较报告</a><a href="../closeups_hand/">Hand根：四路近景</a><a href="../closeups_finger0/">Finger0根：四路近景</a><a href="../sdk_hand/export/viewer/">Hand根：实际导出标签</a><a href="../sdk_finger0/export/viewer/">Finger0根：实际导出标签</a></div><div id="help">')
    html=html.replace('未导出的帧不会补插或显示为有效标签。','本页为拟合对照，包含未通过完整性筛选的视图；实际标签请打开下方导出入口。')
    html=html.replace('</html>',"<script>$('markerLink')?.remove();document.title='SDK骨架假设标签对照';</script></html>")
    (a.output/'index.html').write_text(html)
    print(json.dumps({'frames':len(records),'hands':data['label_count'],'target_hand_views':sum(len(r['target_boundaries']) for r in records),'viewer':str(a.output)},indent=2))


if __name__=='__main__':main()
