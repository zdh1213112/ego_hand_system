#!/usr/bin/env python3
"""Render original / baseline / calibrated mesh closeups from actual projected fit data."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def panel(image, hand, faces, box, title):
    frame = image.copy()
    if hand is not None:
        vertices = np.asarray(hand['vertices'])[:, :2]
        layer = frame.copy()
        color = (195, 225, 50) if hand['side'] == 0 else (35, 155, 255)
        cv2.polylines(layer, np.rint(vertices[faces]).astype(np.int32), True, color, 1, cv2.LINE_AA)
        frame = cv2.addWeighted(layer, .65, frame, .35, 0)
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    scale = min(640 / crop.shape[1], 590 / crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale)
    result = np.zeros((630, 640, 3), np.uint8)
    h, w = crop.shape[:2]
    x, y = (640-w)//2, 40+(590-h)//2
    result[y:y+h, x:x+w] = crop
    cv2.putText(result, title, (12, 28), 0, .7, (235, 235, 235), 1, cv2.LINE_AA)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=3.75)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    data = json.loads((a.review/'data.js').read_text().removeprefix('window.LABEL_DATA=').strip().rstrip(';'))
    faces = np.asarray(data['faces'])
    cv2.setNumThreads(2)
    counts = {}
    for camera in sorted({r['camera'] for r in data['frames']}):
        rows = sorted([r for r in data['frames'] if r['camera'] == camera], key=lambda r:r['sync'])
        video = a.output/f'{camera}_hands_closeup.mp4'
        temporary = a.output/f'{camera}_temp.mp4'
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (1920, 1260))
        if not writer.isOpened():
            raise RuntimeError('video writer failed')
        for r in rows:
            image = cv2.imread(str(a.review/r['image']))
            stacks = []
            for side in [0, 1]:
                new = next((h for h in r['hands'] if h['side'] == side), None)
                old = next((h for h in r['baseline_hands'] if h['side'] == side), None)
                visible = [np.asarray(h['vertices'])[:, :2] for h in [new, old] if h is not None]
                if visible:
                    pts = np.concatenate(visible)
                    low = np.maximum([0, 0], np.floor(pts.min(0)-35)).astype(int)
                    high = np.minimum(image.shape[1::-1], np.ceil(pts.max(0)+35)).astype(int)
                    box = [*low, *high]
                    if high[0] <= low[0] or high[1] <= low[1]:
                        box = [0, 0, image.shape[1], image.shape[0]]
                else:
                    box = [0, 0, image.shape[1], image.shape[0]]
                prefix = f'{"LEFT" if side == 0 else "RIGHT"} | {camera} | {r["sync"]}'
                panels = [panel(image, h, faces, box, prefix+' | '+label)
                          for h, label in [(None, 'RAW'), (old, 'BASELINE'), (new, 'CALIBRATED')]]
                stacks.append(np.concatenate(panels, axis=1))
            frame = np.concatenate(stacks, axis=0)
            writer.write(frame)
            if r['sync'] in [576, 632, 688]:
                cv2.imwrite(str(a.output/f'{camera}_sync{r["sync"]}_comparison.jpg'), frame)
        writer.release()
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(temporary),
                        '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
        temporary.unlink()
        counts[camera] = len(rows)
        print(camera, len(rows), flush=True)
    report = {'source_review': str(a.review.resolve()), 'fit_changed': False, 'approved_for_training': False,
              'rendering': 'Original KB image; same crop for baseline and calibrated mesh, selected hand only',
              'layout': 'Top row left hand; bottom row right hand. Columns raw, baseline, calibrated.',
              'fps': a.fps, 'camera_frame_counts': counts}
    (a.output/'report.json').write_text(json.dumps(report, indent=2))
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>优质样本标定后的标签效果</title><style>body{margin:0;background:#0d1c29;color:#edf5ff;font:16px system-ui}main{max-width:1500px;margin:auto;padding:20px}h1{font-size:24px}a{color:#86d5ff}video,img{width:100%;background:#030a10;border-radius:8px}select,button{font:inherit;padding:8px;margin:8px 6px 12px 0}p{line-height:1.7}.bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px}</style><main><h1>优质样本标定后的标签效果 · 近景对照</h1><p>上排：左手；下排：右手。每排从左到右：<b>原图 → 原始基线 → 优质样本标定结果</b>。三栏使用相同裁剪；每排只叠加对应的一只手。保留原图畸变，使用真实 KB 投影。没有在生成本视频时重新优化。</p><div class="bar"><label>相机 <select id="camera"><option>camera0</option><option>camera1</option><option>camera2</option><option>camera3</option></select></label><button id="pinch">跳到对指示例 sync632</button><a href="../keyframe_calibration/">交互缩放检查</a><a href="../quality_keyframes/">优质样本与分割</a></div><video id="video" controls loop playsinline preload="metadata" poster="camera0_sync632_comparison.jpg" src="camera0_hands_closeup.mp4"></video><p>这是采样帧视频（3.75 帧/秒），不代表已经补全未拟合帧。仍有局部手型／动作误差；当前作为候选标签效果检查。<a href="labels/">查看实际导出的标签</a></p><h2>对指帧静态对照 · sync632</h2><img id="still" src="camera0_sync632_comparison.jpg"></main><script>const v=document.getElementById('video'),c=document.getElementById('camera'),im=document.getElementById('still');c.onchange=()=>{v.src=c.value+'_hands_closeup.mp4';v.poster=c.value+'_sync632_comparison.jpg';im.src=v.poster};document.getElementById('pinch').onclick=()=>{const t=__SYNC632_TIME__;const seek=()=>{v.currentTime=t;v.pause()};if(v.readyState>=1)seek();else v.addEventListener('loadedmetadata',seek,{once:true})};</script></html>'''
    camera0 = sorted([r for r in data['frames'] if r['camera'] == 'camera0'], key=lambda r:r['sync'])
    time = next((i/a.fps for i,r in enumerate(camera0) if r['sync']==632), 0)
    (a.output/'index.html').write_text(html.replace('__SYNC632_TIME__', str(time)))
    print(a.output, flush=True)


if __name__ == '__main__':
    main()
