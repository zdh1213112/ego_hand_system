#!/usr/bin/env python3
"""Independent native-resolution hard rasterization of full-mask shape controls.

All methods share the same accepted masks, speckle cleanup, fixed wrist plane and
calibration/test split. No differentiable-renderer training score is reused.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/personal_hand_calibration'))
from context import load_context
from refine_contours import project
from render_masks import load_hand_mask


def score(uv, faces, target, domain, wrist, axis):
    predicted=np.zeros_like(target)
    for triangle in np.rint(uv[faces]).astype(np.int32):
        cv2.fillConvexPoly(predicted,triangle,1)
    p, g = predicted.astype(bool)&domain, target.astype(bool)&domain
    intersection=np.logical_and(p,g).sum();union=np.logical_or(p,g).sum()
    contours_p,_=cv2.findContours(predicted,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours_g,_=cv2.findContours(target,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    def boundary(contours):
        points=np.concatenate([q.reshape(-1,2) for q in contours])
        points=points[(points-wrist)@axis>8]
        return points[np.linspace(0,len(points)-1,min(500,len(points))).astype(int)]
    a,b=boundary(contours_p),boundary(contours_g)
    distance=np.r_[cKDTree(a).query(b)[0],cKDTree(b).query(a)[0]]
    return {'iou':float(intersection/max(union,1)), 'boundary_mean_px':float(distance.mean()),
            'area_ratio':float(p.sum()/max(g.sum(),1))}


def aggregate(rows):
    if not rows:return {'views':0}
    result={'views':len(rows)}
    for mode in ['original_baseline','previous','shape_only','shape_pose']:
        for metric in ['iou','boundary_mean_px','area_ratio']:
            result[mode+'_'+metric]=float(np.mean([r[mode][metric] for r in rows]))
        if mode in ['shape_only','shape_pose']:
            delta=np.array([r[mode]['iou']-r['previous']['iou'] for r in rows])
            result[mode+'_iou_delta_percentage_points']=float(delta.mean()*100)
            result[mode+'_worsened_over_2pp']=int((delta<-.02).sum())
    return result


def main():
    ap=argparse.ArgumentParser()
    for name in ['context','selection','source','shape-only','shape-pose','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();c=load_context(a.context);selection=json.loads(a.selection.read_text())
    cv2.setNumThreads(2);torch.set_num_threads(2)
    report={'approved_for_training':False,'independent_surface_ground_truth':False,
            'metric':'Native-resolution full mask IoU and all silhouette boundaries, including large pinch holes',
            'small_hole_cleanup':'<=64 native pixels and <=16px-wide enclosed holes only',
            'wrist_domain':'fixed from previous fit, distal >8px; identical for every method',
            'hands':{}}
    all_rows=[]
    for side,hand in enumerate(['left','right']):
        before=np.load(a.source/f'{hand}_pilot.npz');original=np.load(c['baseline_fit']/f'track_{side}.npz')
        pure=np.load(a.shape_only/f'{hand}_pilot.npz');joint=np.load(a.shape_pose/f'{hand}_pilot.npz')
        tracks={'previous':before,'original_baseline':original,'shape_only':pure,'shape_pose':joint}
        lookup={mode:{int(f):i for i,f in enumerate(z['pair_indices'])} for mode,z in tracks.items()}
        chosen=set(selection['report']['hands'][hand]['calibration_sync'])
        keys={(r['camera'],r['sync']) for r in selection['report']['hands'][hand]['calibration_views']}
        rows=[]
        for r in selection['records']:
            if r['side']!=side or not r['accepted_quality']:continue
            j=before['joints'][lookup['previous'][r['sync']]]
            jp=project(torch.tensor(j,dtype=torch.float64),r,side).numpy()
            axis=jp[[5,9,13,17]].mean(0)-jp[0];axis/=max(np.linalg.norm(axis),1e-9)
            mask=load_hand_mask(r['mask']);yy,xx=np.mgrid[:mask.shape[0],:mask.shape[1]]
            domain=(xx-jp[0,0])*axis[0]+(yy-jp[0,1])*axis[1]>8
            row={'side':side,'camera':r['camera'],'sync':r['sync'],
                 'calibration':r['sync'] in chosen,'used_for_calibration':(r['camera'],r['sync']) in keys,
                 'time_holdout':c['holdout'][0]<=r['sync']<c['holdout'][1],
                 'image':r['image'],'mask':r['mask']}
            for mode,z in tracks.items():
                v=z['vertices'][lookup[mode][r['sync']]]
                uv=project(torch.tensor(v,dtype=torch.float64),r,side).numpy()
                row[mode]=score(uv,z['faces'],mask,domain,jp[0],axis)
            rows.append(row)
        test=[r for r in rows if not r['calibration']]
        result={'calibration':aggregate([r for r in rows if r['used_for_calibration']]),
                'noncalibration':aggregate(test),'time_holdout':aggregate([r for r in test if r['time_holdout']]),
                'per_camera':{cam:aggregate([r for r in test if r['camera']==cam]) for cam in sorted({r['camera'] for r in test})},
                'worst_shape_only':sorted(test,key=lambda r:r['shape_only']['iou']-r['previous']['iou'])[:5],
                'worst_shape_pose':sorted(test,key=lambda r:r['shape_pose']['iou']-r['previous']['iou'])[:5]}
        report['hands'][hand]=result;all_rows.extend(rows)
        print(hand,json.dumps({k:v for k,v in result.items() if k not in ['worst_shape_only','worst_shape_pose','per_camera']}),flush=True)
    report['per_view']=all_rows
    a.output.write_text(json.dumps(report,indent=2))
    print(a.output,flush=True)


if __name__=='__main__':main()
