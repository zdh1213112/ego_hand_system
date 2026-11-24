#!/usr/bin/env python3
"""Native camera comparison: current mesh, marker refit control, SDK directions."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / 'ego_hand_system/scripts'))
from ego_data.calibration import CameraCalibration
from camera_models import project_points


def main():
    ap = argparse.ArgumentParser()
    for name in ['previous-review', 'result', 'dataset', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output / 'images').symlink_to((a.previous_review / 'images').resolve(), target_is_directory=True)
    previous = json.loads((a.previous_review / 'data.js').read_text().removeprefix('window.LABEL_DATA=').strip().rstrip(';'))
    modes = [('pure', 'direction'), ('pose', 'marker_only')]
    tracks = {mode: [np.load(a.result / variant / f'{h}_pilot.npz') for h in ['left', 'right']] for mode, variant in modes}
    lookup = {mode: [{int(f): i for i, f in enumerate(z['pair_indices'])} for z in zs] for mode, zs in tracks.items()}
    targets = np.load(a.result / 'profile/direction_targets.npz')
    parents, children = targets['mano_parent'], targets['mano_child']
    cameras = {f'camera{i}': CameraCalibration.load(a.dataset / 'calibration' / f'camera{i}.json') for i in range(4)}
    c0 = cameras['camera0']; records = []
    for original in previous['frames']:
        r = copy.deepcopy(original); r['source'] = 'sdk-direction'; r['source_name'] = 'Finger0 腕根候选 · SDK方向辅助实验'
        r['baseline_hands'] = original['hands']; r['hands'] = []; r['pose_hands'] = []
        camera = cameras[r['camera']]
        for side in [0, 1]:
            def proj(points):
                p = points.copy(); p[:, 0] *= -1 if side == 0 else 1
                base = p @ c0.T_base_camera[:3, :3].T + c0.T_base_camera[:3, 3]
                cam = (base - camera.T_base_camera[:3, 3]) @ camera.T_base_camera[:3, :3]
                return project_points(camera, cam)[0], cam[:, 2]
            for mode, variant in modes:
                z = tracks[mode][side]; i = lookup[mode][side][r['sync']]
                uv, depth = proj(z['vertices'][i]); jp, _ = proj(z['joints'][i])
                if not np.isfinite(uv).all() or depth.min() < .03: continue
                w, h = r['size']; inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                if not inside.any(): continue
                R = Rotation.from_rotvec(z['global_orient'][i]).as_matrix()
                lengths = np.linalg.norm(z['joints'][i, children] - z['joints'][i, parents], axis=-1)
                anchors = z['joints'][i, parents]
                ends = anchors + (targets['local_directions'][side, r['sync']] @ R.T) * lengths[:, None]
                arrow0, _ = proj(anchors); arrow1, _ = proj(ends)
                root = targets['finger0_origins_cam0_m'][side, r['sync']].copy()
                # root is physical camera0: undo the canonical mirror expected by proj.
                root[0] *= -1 if side == 0 else 1
                root_uv, _ = proj(root[None])
                dest = r['hands'] if mode == 'pure' else r['pose_hands']
                dest.append(dict(side=side, vertices=np.round(np.c_[uv, depth], 5).tolist(),
                                 joints=np.round(jp, 4).tolist(), direction_arrows=np.round(np.stack([arrow0, arrow1], 1), 4).tolist(),
                                 sdk_finger0_root=np.round(root_uv[0], 4).tolist(),
                                 bbox=np.r_[uv.min(0) - 20, uv.max(0) + 20].tolist(), visible_vertex_fraction=float(inside.mean()),
                                 label=str(a.result / variant / f'{"left" if side == 0 else "right"}_pilot.npz')))
        records.append(r)
    data = dict(initial_sync=632, dataset=str(a.result.resolve()), projection='native_raw / true KB; calibrated SDK directions only',
                faces=previous['faces'], frames=records, label_count=sum(len(r['hands']) for r in records),
                max_projection_difference_px=None, approved_for_training=False)
    (a.output / 'data.js').write_text('window.LABEL_DATA=' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
    html = (ROOT / 'tools/marker_placement/label_viewer.html').read_text()
    html = html.replace('grid-template-columns:1fr 1fr;', 'grid-template-columns:1fr 1fr 1fr;')
    html = html.replace('<section class="panel"><h2>相同原图 + 导出标签</h2>', '<section class="panel"><h2>当前：共享手型＋marker动作</h2><canvas id="baseline"></canvas></section><section class="panel"><h2><span id="trialTitle">marker重拟合＋SDK方向辅助</span></h2>')
    html = html.replace("canvases=[$('raw'),$('overlay')]", "canvases=[$('raw'),$('overlay'),$('baseline')]").replace('ci<2', 'ci<3')
    html = html.replace('真实相机原图', '原图＋可靠分割轮廓')
    html = html.replace('原图与 MANO Mesh 标签检查', 'SDK方向辅助 · Finger0腕根候选')
    html = html.replace('左侧原图，右侧叠加当前导出的 mesh。两侧同步缩放和平移，图片保持原始畸变。',
                        '优先保留你选择的 Finger0 公共腕根候选。本次固定手型、marker附着关系和根平移，仅微调动作；SDK只提供校准后的骨骼方向，不硬拉SDK关节位置或骨长。右侧可切换只重拟合marker的对照。')
    html = html.replace('<label>叠加 <select id="mode">', '<label>实验 <select id="trial"><option value="pure">marker＋SDK方向辅助</option><option value="pose">仅marker重拟合（对照）</option></select></label><label><input type="checkbox" id="sdkDirections">紫色方向箭头</label><label><input type="checkbox" id="sdkRoot">紫色Finger0参考根</label><label><input type="checkbox" id="targetMask" checked>绿色分割轮廓</label><label>叠加 <select id="mode">')
    html = html.replace('const hands=frames[index].hands.filter', "const hands=(ci===2?frames[index].baseline_hands:$('trial').value==='pose'?frames[index].pose_hands:frames[index].hands).filter")
    html = html.replace('ctx.drawImage(img,0,0);', '''ctx.drawImage(img,0,0);if($('targetMask').checked&&!hide){ctx.globalAlpha=1;ctx.strokeStyle='#7dff54';ctx.lineWidth=1.5/scale;for(const t of frames[index].target_boundaries||[]){if(!$(t.side?'right':'left').checked)continue;for(const line of t.lines){if(!line.length)continue;ctx.beginPath();ctx.moveTo(...line[0]);for(const p of line.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.stroke()}}}''')
    draw_aux = """if(!hide&&ci!==2){const hs=$('trial').value==='pose'?frames[index].pose_hands:frames[index].hands;ctx.strokeStyle='#ee61f2';ctx.fillStyle='#ee61f2';ctx.lineWidth=1.5/scale;for(const h of hs){if(!$(h.side?'right':'left').checked)continue;if($('sdkRoot').checked){const p=h.sdk_finger0_root;ctx.beginPath();ctx.arc(p[0],p[1],4/scale,0,Math.PI*2);ctx.stroke()}if($('sdkDirections').checked)for(const [a,b] of h.direction_arrows){ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-5/scale*Math.cos(angle-.5),b[1]-5/scale*Math.sin(angle-.5));ctx.moveTo(...b);ctx.lineTo(b[0]-5/scale*Math.cos(angle+.5),b[1]-5/scale*Math.sin(angle+.5));ctx.stroke()}}}"""
    html = html.replace('if(!ci||hide)continue;', draw_aux + 'if(!ci||hide)continue;')
    html = html.replace('已导出标签 ${D.label_count} 份', '实验手实例 ${D.label_count} 份')
    html = html.replace('window.onresize=resize;', "for(const id of ['targetMask','sdkDirections','sdkRoot'])$(id).onchange=draw;$('trial').onchange=()=>{$('trialTitle').textContent=$('trial').value==='pure'?'marker重拟合＋SDK方向辅助':'仅marker重拟合（相同自由度对照）';draw()};window.onresize=resize;")
    html = html.replace('resize();selectFrames();', 'resize();selectFrames(D.initial_sync);')
    html = html.replace("if(!viewInitialized||!$('keepZoom')?.checked)fit();else draw();", "if(!viewInitialized){fit(1);setTimeout(()=>{if(seq===token)fit(1)},250)}else if(!$('keepZoom')?.checked)fit();else draw();")
    html = html.replace('<div id="help">', '<div class="bar"><a href="../comparison_metrics.json">完整比较报告</a><a href="../profile/mapping.json">SDK模型与方向映射</a><a href="../closeups_direction/">四路近景视频</a><a href="../direction/export/viewer/">实际导出标签</a><a href="../../sdk_skeleton_labels/review/">原来的Finger0／Hand根对照</a></div><div id="help">')
    html = html.replace('未导出的帧不会补插或显示为有效标签。', '紫色箭头只表示方向，箭头长度来自MANO，起点锚在MANO关节；不是SDK内部关节真值。紫色圆圈仅为Finger0参考根，不作为位置目标。本页包含未通过导出完整性筛选的视图；实际标签请打开导出入口。')
    html = html.replace('</html>', "<script>$('markerLink')?.remove();</script></html>")
    (a.output / 'index.html').write_text(html)
    print(json.dumps(dict(frames=len(records), hands=data['label_count'], viewer=str(a.output)), indent=2))


if __name__ == '__main__': main()
