#!/usr/bin/env python3
"""Prepare editable suggestions from raw 24-marker trajectories and exported images."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent / 'ego_hand_system'
sys.path[:0] = [str(PROJECT / 'scripts'), str(ROOT.parent / 'nokov_ego_hand_validation/integration/pipeline')]
from align_nokov_cam0 import load_rigid, load_markers, interpolate_points
from ego_data.calibration import CameraCalibration
from camera_models import project_points


def main():
    exp = ROOT / 'output/native_kb_training_20261111'
    alignment = json.loads((exp / 'fourview_192916/alignment_180/fourview_alignment.json').read_text())
    seed = json.loads(Path(alignment['seed_alignment']).read_text())
    dataset, nokov = Path(alignment['source_dataset']), Path(alignment['source_nokov'])
    export = exp / 'fourview_192916/native_labels/export/merged'
    records = [json.loads(x) for x in (export / 'index.jsonl').read_text().splitlines()]
    with (dataset / 'multiview_frames.csv').open() as f:
        rows = list(csv.DictReader(f))
    mapping = seed['time_mapping']
    times = np.array([int(r['camera0_timestamp_ns']) for r in rows], dtype=np.int64)
    target = mapping['nokov_origin_timestamp_raw'] + ((times-mapping['ego_origin_timestamp_ns'])*1e-9*mapping.get('a',1)+mapping['b_s'])/mapping['nokov_seconds_per_timestamp_unit']
    rt, head, quats = load_rigid(nokov / 'nokov_rigid_bodies.csv')
    head, valid = interpolate_points(rt, head, target)
    rotations = np.full((len(rows),3,3), np.nan)
    valid &= (target>=rt[0]) & (target<=rt[-1])
    rotations[valid] = Slerp(rt,Rotation.from_quat(quats))(target[valid]).as_matrix()
    T = np.asarray(alignment['T_cam0_head'])
    C0 = CameraCalibration.load(dataset / 'calibration/camera0.json')
    expected = np.load(exp / 'fourview_192916/alignment_180/aligned_nokov21.npz')['joints_base_m']
    names = ['WristM','WristIn','WristOut','HandOffset']+[f'Finger{f}{i}' for f in ['Thumb','Index','Middle','Ring','Pinky'] for i in range(1,5)]
    output = Path(__file__).parent / 'prefill'
    output.mkdir(exist_ok=True)
    state = dict(schema='ego_marker_placement_v1',purpose='anatomical_region_prior_not_metric_calibration',marker_diameter_mm=4,mount='directly_on_glove',hands={'left':{},'right':{}})
    photos = {}; report = []
    for side, hand, camera, sync in [(0,'left','camera1',616),(1,'right','camera2',584)]:
        t, world = load_markers(nokov/'nokov_markers.csv',hand+'_hand_4mm')
        world, _ = interpolate_points(t, world, target)
        xyz = np.einsum('fji,fkj->fki', rotations, world-head[:,None])
        cam0 = xyz@T[:3,:3].T+T[:3,3]
        base = cam0@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3]
        err = float(np.nanmax(np.abs(base[:,[0]+list(range(4,24))]-expected[side])))
        if err>1e-5:
            raise ValueError(f'24-marker reconstruction disagrees with existing alignment: {err}')
        C = CameraCalibration.load(dataset/'calibration'/f'{camera}.json')
        cam = (base[sync]-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3]
        uv, visible = project_points(C,cam)
        if not np.isfinite(uv).all() or np.any(cam[:,2]<=0):
            raise ValueError('Reference must contain finite front-facing marker projections')
        row = next(r for r in records if r['side']==side and r['camera']==camera and r['sync_index']==sync)
        image = cv2.imread(str(export/row['image']))
        lo = np.maximum(0,np.floor(uv.min(0)-45)).astype(int)
        hi = np.minimum(image.shape[1::-1],np.ceil(uv.max(0)+45)).astype(int)
        crop = image[lo[1]:hi[1],lo[0]:hi[0]]
        filename = f'{hand}_{camera}_sync{sync:06d}.png'
        cv2.imwrite(str(output/filename),crop)
        w,h = crop.shape[1],crop.shape[0]
        photo = dict(url='prefill/'+filename,name=filename,width=w,height=h,camera=camera,sync=sync,source_image=str(export/row['image']),crop_xyxy=[*lo.tolist(),*hi.tolist()],view='dorsal',projection='native_raw KB; no occlusion visibility claim')
        photos[hand+'_dorsal'] = photo
        for i,name in enumerate(names):
            region = ['wrist_center','unknown','unknown','palm'][i] if i<4 else 'unknown'
            if i>=4:
                finger=['thumb','index','middle','ring','pinky'][(i-4)//4]
                part=(['base','mcp','ip','tip'] if finger=='thumb' else ['mcp','pip','dip','tip'])[(i-4)%4]
                region=finger+'_'+part
            state['hands'][hand][name] = dict(status='suggested',region=region,surface='dorsal',notes='自动预填，贴点区域及表面均待确认；投影存在不代表该点可见。',
                point=dict(view='dorsal',x=float((uv[i,0]-lo[0])/w),y=float((uv[i,1]-lo[1])/h),source='photo',image=dict(name=filename,width=w,height=h)),
                suggestion=dict(method='raw_nokov24_projection',marker_index=i,marker_name=name,camera=camera,sync=sync,original_pixel=uv[i].tolist(),source_image=photo['source_image'],crop_xyxy=photo['crop_xyxy'],occlusion_checked=False,region_is_unverified=True))
        report.append(dict(hand=hand,camera=camera,sync=sync,points=24,reconstruction_max_abs_m=err,source_image=photo['source_image']))
    payload=dict(state=state,photos=photos,report=report)
    (Path(__file__).parent/'prefill.js').write_text('window.MARKER_PREFILL='+json.dumps(payload,ensure_ascii=False,allow_nan=False)+';\n')
    (output/'suggested_placement.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
