#!/usr/bin/env python3
"""Compare SDK-only motion fits against image evidence never used by this motion fitting stage."""
import argparse,json,sys
from pathlib import Path

import cv2,numpy as np,torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/label_pipeline/calibration/full_silhouette'))
from evaluate import score
from render_masks import load_hand_mask
sys.path.insert(0,str(ROOT/'tools/label_pipeline/calibration/personal_hand_calibration'))
from context import load_context
from refine_contours import project


def aggregate(rows):
    if not rows:return {'images':0}
    result={'images':len(rows)}
    for mode in ['current','sdk_hand','sdk_finger0']:
        result[mode+'_iou']=float(np.mean([r[mode]['iou'] for r in rows]))
        result[mode+'_boundary_px']=float(np.mean([r[mode]['boundary_mean_px'] for r in rows]))
        if mode!='current':
            d=np.array([r[mode]['iou']-r['current']['iou'] for r in rows]);result[mode+'_iou_delta_percentage_points']=float(d.mean()*100);result[mode+'_worsened_over_2pp']=int((d<-.02).sum())
    return result


def main():
    ap=argparse.ArgumentParser()
    for name in ['context','selection','source','sdk-hand','sdk-finger0','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();c=load_context(a.context);selection=json.loads(a.selection.read_text());cv2.setNumThreads(2);torch.set_num_threads(2)
    report={'approved_for_training':False,'assumption':'SDK marker-to-internal-joint offsets configured; NOT verified',
            'metric':'Native-resolution segmentation agreement, fixed previous wrist domain; not independent anatomical truth',
            'images_used_for_sdk_motion_fitting':False,'hands':{},'per_view':[]}
    for side,hand in enumerate(['left','right']):
        tracks={name:np.load(path/f'{hand}_pilot.npz') for name,path in [('current',a.source),('sdk_hand',a.sdk_hand),('sdk_finger0',a.sdk_finger0)]}
        lookup={name:{int(f):i for i,f in enumerate(z['pair_indices'])} for name,z in tracks.items()};chosen=set(selection['report']['hands'][hand]['calibration_sync']);rows=[]
        for r in selection['records']:
            if r['side']!=side or not r['accepted_quality']:continue
            before=tracks['current'];j=before['joints'][lookup['current'][r['sync']]];jp=project(torch.tensor(j,dtype=torch.float64),r,side).numpy();axis=jp[[5,9,13,17]].mean(0)-jp[0];axis/=max(np.linalg.norm(axis),1e-9)
            mask=load_hand_mask(r['mask']);yy,xx=np.mgrid[:mask.shape[0],:mask.shape[1]];domain=(xx-jp[0,0])*axis[0]+(yy-jp[0,1])*axis[1]>8
            row={'camera':r['camera'],'sync':r['sync'],'side':side,'used_in_earlier_shape_calibration':r['sync'] in chosen,'time_holdout':c['holdout'][0]<=r['sync']<c['holdout'][1]}
            for mode,z in tracks.items():
                uv=project(torch.tensor(z['vertices'][lookup[mode][r['sync']]],dtype=torch.float64),r,side).numpy();row[mode]=score(uv,z['faces'],mask,domain,jp[0],axis)
            rows.append(row)
        independent=[r for r in rows if not r['used_in_earlier_shape_calibration']]
        report['hands'][hand]={'all_accepted_masks':aggregate(rows),'images_not_used_for_shape_calibration':aggregate(independent),
                               'time_holdout_images':aggregate([r for r in independent if r['time_holdout']]),
                               'worst_hand_root':sorted(independent,key=lambda r:r['sdk_hand']['iou']-r['current']['iou'])[:5],
                               'worst_finger0_root':sorted(independent,key=lambda r:r['sdk_finger0']['iou']-r['current']['iou'])[:5]}
        report['per_view']+=rows;print(hand,json.dumps(report['hands'][hand]['images_not_used_for_shape_calibration']),flush=True)
    a.output.write_text(json.dumps(report,indent=2));print(a.output,flush=True)


if __name__=='__main__':main()
