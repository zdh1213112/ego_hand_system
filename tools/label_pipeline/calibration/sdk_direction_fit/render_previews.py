#!/usr/bin/env python3
"""Readable closeups: thin mesh wires, explicit silhouette edge and accepted target contour."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def panel(image, hand, side, faces, target, box, title):
    x1,y1,x2,y2=box;crop=image[y1:y2,x1:x2]
    scale=min(640/crop.shape[1],590/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
    output=np.zeros((630,640,3),np.uint8);h,w=crop.shape[:2];ox,oy=(640-w)//2,40+(590-h)//2
    output[oy:oy+h,ox:ox+w]=crop
    def transform(points):return np.rint((np.asarray(points)-[x1,y1])*scale+[ox,oy]).astype(np.int32)
    if hand is not None:
        points=transform(np.asarray(hand['vertices'])[:,:2]);color=(195,225,50) if side==0 else (35,155,255)
        layer=output.copy();cv2.polylines(layer,points[faces],True,color,1,cv2.LINE_AA)
        output=cv2.addWeighted(layer,.30,output,.70,0)
        mask=np.zeros(output.shape[:2],np.uint8)
        for triangle in points[faces]:cv2.fillConvexPoly(mask,triangle,1)
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output,contours,-1,color,1,cv2.LINE_AA)
    for line in target:
        cv2.polylines(output,[transform(line)],True,(84,255,125),1,cv2.LINE_AA)
    output[:40]=0
    cv2.putText(output,title,(10,27),0,.65,(240,240,240),1,cv2.LINE_AA)
    return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--root',choices=['direction','marker_only'],required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);cv2.setNumThreads(2)
    data=json.loads((a.review/'data.js').read_text().removeprefix('window.LABEL_DATA=').strip().rstrip(';'))
    faces=np.asarray(data['faces']);counts={}
    for camera in sorted({r['camera'] for r in data['frames']}):
        rows=sorted([r for r in data['frames'] if r['camera']==camera],key=lambda r:r['sync'])
        tmp=a.output/f'{camera}_temp.mp4';path=a.output/f'{camera}_comparison.mp4'
        writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),3.75,(1920,1260))
        if not writer.isOpened():raise RuntimeError('video writer unavailable')
        for r in rows:
            image=cv2.imread(str(a.review/r['image']));stacks=[]
            for side in [0,1]:
                new=next((h for h in r['hands' if a.root=='direction' else 'pose_hands'] if h['side']==side),None);old=next((h for h in r['baseline_hands'] if h['side']==side),None)
                target=next((t['lines'] for t in r['target_boundaries'] if t['side']==side),[])
                uv=[np.asarray(h['vertices'])[:,:2] for h in [new,old] if h is not None]
                if uv:
                    p=np.concatenate(uv);lo=np.maximum([0,0],np.floor(p.min(0)-30)).astype(int);hi=np.minimum(image.shape[1::-1],np.ceil(p.max(0)+30)).astype(int);box=[*lo,*hi]
                    if hi[0]<=lo[0] or hi[1]<=lo[1]:box=[0,0,image.shape[1],image.shape[0]]
                else:box=[0,0,image.shape[1],image.shape[0]]
                title=f'{"LEFT" if side==0 else "RIGHT"} | {camera} | {r["sync"]}'
                panels=[panel(image,hand,side,faces,target,box,title+' | '+label) for hand,label in [(None,'RAW'),(old,'PREVIOUS'),(new,'MARKER + SDK DIR' if a.root=='direction' else 'MARKER ONLY')]]
                stacks.append(np.concatenate(panels,axis=1))
            frame=np.concatenate(stacks,axis=0);writer.write(frame)
            if r['sync'] in [576,632,664,688]:cv2.imwrite(str(a.output/f'{camera}_sync{r["sync"]}.jpg'),frame)
        writer.release()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(tmp),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
        tmp.unlink();counts[camera]=len(rows);print(camera,len(rows),flush=True)
    rows=sorted([r for r in data['frames'] if r['camera']=='camera0'],key=lambda r:r['sync']);time=next((i/3.75 for i,r in enumerate(rows) if r['sync']==664),0)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>marker拟合与SDK方向辅助 · 近景对照</title><style>body{background:#102230;color:#eaf4ff;font:16px system-ui;margin:0}main{max-width:1600px;margin:auto;padding:20px}video,img{width:100%;background:#050b10}a{color:#8ad8ff}button,select{font:inherit;padding:8px;margin:5px}p{line-height:1.7}</style><main><h1>marker拟合与SDK方向辅助 · 四路标签对照</h1><p>上排左手、下排右手；每排从左到右：<b>原图 → 当前标签 → marker拟合／SDK方向辅助标签</b>。绿色线是可靠分割，青色／橙色为 mesh。固定个人手型、marker附着关系和根平移。当前实验：__ROOT__；SDK只提供经优质帧校准的方向辅助，没有SDK关节位置或骨长硬目标，没有额外减2mm。Finger0保留为参考腕根；没有可靠分割的帧不画绿色。</p><p><select id="camera"><option>camera0</option><option>camera1</option><option>camera2</option><option>camera3</option></select><button id="example">跳到 sync664</button> <a href="../review/">交互放大／切换方向辅助和marker对照</a> · <a href="../comparison_metrics.json">拟合与轮廓比较数据</a></p><video id="video" controls loop playsinline preload="metadata" poster="camera0_sync664.jpg" src="camera0_comparison.mp4"></video><p>采样播放 3.75 帧/秒；不是新增了逐帧推理。当前仍为实验对照结果，尚未批准为最终真值。<a href="../__ROOT__/export/viewer/">实际导出的标签</a> · <a href="../closeups___OTHER__/">切换另一拟合对照</a></p><img id="still" src="camera0_sync664.jpg"></main><script>const c=document.getElementById('camera'),v=document.getElementById('video'),im=document.getElementById('still');c.onchange=()=>{v.src=c.value+'_comparison.mp4';v.poster=c.value+'_sync664.jpg';im.src=v.poster};document.getElementById('example').onclick=()=>{const f=()=>{v.currentTime=__TIME__;v.pause()};if(v.readyState>=1)f();else v.addEventListener('loadedmetadata',f,{once:true})};</script></html>'''
    (a.output/'index.html').write_text(html.replace('__TIME__',str(time)).replace('__ROOT__',a.root).replace('__OTHER__','marker_only' if a.root=='direction' else 'direction'))
    (a.output/'report.json').write_text(json.dumps({'camera_frames':counts,'fps':3.75,'approved_for_training':False,'fit_changed_during_rendering':False},indent=2))
    print(a.output,flush=True)


if __name__=='__main__':main()
