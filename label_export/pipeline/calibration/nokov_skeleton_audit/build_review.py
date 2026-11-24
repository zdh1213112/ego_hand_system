#!/usr/bin/env python3
"""Quick original-image skeleton/marker/MANO audit, no fitted corrections."""
import argparse,json,sys
from pathlib import Path

import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT / 'scripts'))
from ego_data.calibration import CameraCalibration
from camera_models import project_points

MANO_EDGES=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+i,k+i+1) for k in [5,9,13,17] for i in range(3)]


def project(C,C0,points):
    base=points@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3]
    cam=(base-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3]
    uv,_=project_points(C,cam)
    return uv,cam[:,2]


def main():
    ap=argparse.ArgumentParser()
    for name in ['context','audit','mano','images','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=json.loads(a.context.read_text());dataset=Path(c['dataset']);z=np.load(a.audit/'aligned_skeleton.npz');mano=[np.load(a.mano/f'{h}_pilot.npz') for h in ['left','right']]
    lookup=[{int(f):i for i,f in enumerate(m['pair_indices'])} for m in mano];selected=[56,384,576,632,664,688,1128,1560]
    cameras={f'camera{i}':CameraCalibration.load(dataset/'calibration'/f'camera{i}.json') for i in range(4)};C0=cameras['camera0'];records=[];a.output.mkdir(parents=True);(a.output/'images').mkdir();(a.output/'comparisons').mkdir();cv2.setNumThreads(2)
    for camera,C in cameras.items():
        for sync in selected:
            source=a.images/f'{camera}_{sync:06d}.jpg'
            if not source.is_file() or not all(sync in m for m in lookup):continue
            image=cv2.imread(str(source));dest=a.output/'images'/source.name;dest.symlink_to(source.resolve());record={'camera':camera,'sync':sync,'image':'images/'+source.name,'hands':[]}
            for side in [0,1]:
                sk,depth=project(C,C0,z['positions_cam0_m'][side,sync]);mk,md=project(C,C0,z['marker24_cam0_m'][side,sync]);i=lookup[side][sync]
                points=mano[side]['joints'][i].copy();points[:,0]*=-1 if side==0 else 1
                mj,jd=project(C,C0,points)
                edges=[[int(p),j] for j,p in enumerate(z['parents'][side]) if p>=0]
                valid=np.isfinite(sk).all(1)&(depth>.03);mv=np.isfinite(mk).all(1)&(md>.03);jv=np.isfinite(mj).all(1)&(jd>.03)
                pts=np.r_[sk[valid],mk[mv],mj[jv]];lo=np.maximum([0,0],np.floor(pts.min(0)-35));hi=np.minimum(image.shape[1::-1],np.ceil(pts.max(0)+35));bbox=[*lo.tolist(),*hi.tolist()]
                if hi[0]<=lo[0] or hi[1]<=lo[1]:bbox=[0,0,image.shape[1],image.shape[0]]
                hand={'side':side,'skeleton':np.nan_to_num(sk).tolist(),'skeleton_valid':valid.tolist(),'markers':np.nan_to_num(mk).tolist(),'marker_valid':mv.tolist(),'mano':np.nan_to_num(mj).tolist(),'mano_valid':jv.tolist(),'edges':edges,'bbox':bbox,'names':z['names'][side].tolist()}
                record['hands'].append(hand)
            records.append(record)
            if sync in [576,632,664,688]:
                rows=[]
                for hand in record['hands']:
                    side=hand['side'];x1,y1,x2,y2=np.rint(hand['bbox']).astype(int);row=[]
                    for mode,title in [('raw','RAW'),('markers','SDK26 + MARKERS24'),('mano','SDK26 + MANO21')]:
                        frame=image.copy()
                        def lines(points,edges,valid,color):
                            p=np.rint(points).astype(int)
                            for j,k in edges:
                                if valid[j] and valid[k]:cv2.line(frame,tuple(p[j]),tuple(p[k]),color,1,cv2.LINE_AA)
                            for p,ok in zip(p,valid):
                                if ok:cv2.circle(frame,tuple(p),2,color,-1,cv2.LINE_AA)
                        if mode!='raw':lines(hand['skeleton'],hand['edges'],hand['skeleton_valid'],(240,65,240))
                        if mode=='markers':
                            for p,ok in zip(np.rint(hand['markers']).astype(int),hand['marker_valid']):
                                if ok:cv2.circle(frame,tuple(p),3,(50,255,100),1,cv2.LINE_AA)
                        if mode=='mano':lines(hand['mano'],MANO_EDGES,hand['mano_valid'],(255,210,40) if side==0 else (20,170,255))
                        crop=frame[y1:y2,x1:x2];scale=min(640/crop.shape[1],590/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale);panel=np.zeros((630,640,3),np.uint8);h,w=crop.shape[:2];panel[40+(590-h)//2:40+(590-h)//2+h,(640-w)//2:(640-w)//2+w]=crop
                        cv2.putText(panel,f'{"LEFT" if side==0 else "RIGHT"} | {camera} | {sync} | {title}',(8,27),0,.55,(240,240,240),1,cv2.LINE_AA);row.append(panel)
                    rows.append(np.concatenate(row,axis=1))
                cv2.imwrite(str(a.output/'comparisons'/f'{camera}_{sync:06d}.jpg'),np.concatenate(rows,axis=0))
    data={'records':records,'mano_edges':MANO_EDGES,'initial_sync':632};(a.output/'data.js').write_text('window.SKELETON_AUDIT='+json.dumps(data,separators=(',',':'))+';\n')
    (a.output/'index.html').write_text(Path(__file__).with_name('viewer.html').read_text());print('frames',len(records),'viewer',a.output,flush=True)


if __name__=='__main__':main()
