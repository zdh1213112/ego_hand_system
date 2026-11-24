#!/usr/bin/env python3
"""Read and project real NOKOV skeleton origins without fitting or changing calibration."""
import argparse,csv,json,sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'label_export/pipeline/calibration/personal_hand_calibration'))
from context import load_context,raw24_context
from run import CameraCalibration
sys.path.insert(0,str(ROOT.parent/'nokov_ego_hand_validation/integration/pipeline'))
from align_nokov_cam0 import load_rigid,interpolate_points


def summary(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return {'count':len(x),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(x.max())} if len(x) else {'count':0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--context',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=load_context(a.context);alignment=json.loads(c['alignment'].read_text());nokov=Path(alignment['source_nokov']);assets=json.loads((nokov/'asset_descriptions.json').read_text());metadata=json.loads((nokov/'capture_metadata.json').read_text())
    with (nokov/'nokov_frames.csv').open(encoding='utf-8-sig') as f:frames=list(csv.DictReader(f))
    times=np.array([int(r['device_timestamp_raw']) for r in frames],dtype=np.float64);lookup={int(t):i for i,t in enumerate(times)}
    if not np.all(np.diff(times)>0):raise ValueError('NOKOV timestamps not strictly increasing')
    xyz=np.full((2,len(times),26,3),np.nan);quat=np.full((2,len(times),26,4),np.nan);parents=[];names=[]
    skeleton_name = {}
    for side, hand in enumerate(['left', 'right']):
        found = [v['name'] for v in assets['skeletons'].values() if v['name'].startswith(hand + '_hand')]
        if len(found) > 1: raise ValueError(f'several {hand} skeletons {found}')
        skeleton_name[side] = found[0] if found else None
        if not found: names.append(['']*26); parents.append([-1]*26); continue
        desc = next(v for v in assets['skeletons'].values() if v['name'] == found[0])
        ids = sorted(map(int, desc['segments'])); assert ids == list(range(1, 27))
        names.append([desc['segments'][str(i)] for i in ids]); parents.append([int(desc['parent_ids'][str(i)]) - 1 for i in ids])
    if not any(skeleton_name.values()): raise ValueError('no hand skeleton in this recording')
    with (nokov/'nokov_skeleton_segments.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            side = next((k for k, v in skeleton_name.items() if v == row['skeleton_name']), None)
            if side is None or row['valid_numeric'] != '1': continue
            i=lookup[int(row['device_timestamp_raw'])];j=int(row['segment_id'])-1
            xyz[side,i,j]=[float(row[k])*1e-3 for k in ['x_mm','y_mm','z_mm']]
            quat[side,i,j]=[float(row[k]) for k in ['qx','qy','qz','qw']]
    markers=np.full((2,len(times),24,3),np.nan)
    with (nokov/'nokov_markers.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            side = 0 if row['markerset_name'].startswith('left_hand') else 1 if row['markerset_name'].startswith('right_hand') else None
            if side is not None and row['valid'] == '1':markers[side,lookup[int(row['device_timestamp_raw'])],int(row['marker_index'])]=[float(row[k])*1e-3 for k in ['x_mm','y_mm','z_mm']]
    # Names are only provisional correspondence candidates, not anatomical truth.
    select=np.array([0]+[2+5*k+i for k in range(5) for i in range(4)])
    selected_markers=np.array([0]+list(range(4,24)))
    report={'source_nokov':str(nokov),'source_dataset':str(c['dataset']),'fit_performed':False,'calibration_changed':False,
            'coordinate_hypothesis':'Skeleton positions are interpreted as SDK world positions; checked against simultaneous marker geometry',
            'metadata_coordinate_setting':metadata.get('skeleton_coordinate_frame'),'mapping_status':'provisional names only; End is not assumed to be a true fingertip',
            'hands':{}}
    for side,hand in enumerate(['left','right']):
        if skeleton_name[side] is None: report['hands'][hand]={'recorded':False}; continue
        error=np.linalg.norm(xyz[side][:,select]-markers[side][:,selected_markers],axis=-1)*1000
        perbone=[]
        for child,parent in enumerate(parents[side]):
            if parent<0:continue
            vector=xyz[side,:,child]-xyz[side,:,parent];length=np.linalg.norm(vector,axis=-1)*1000
            R=Rotation.from_quat(quat[side,:,parent]).as_matrix();local=np.einsum('fji,fj->fi',R,vector)*1000
            deviation=np.linalg.norm(local-np.median(local,axis=0),axis=-1)
            perbone.append({'child':names[side][child],'parent':names[side][parent],
                            'length_median_mm':float(np.median(length)),'length_std_mm':float(np.std(length)),
                            'parent_rotation_local_offset_deviation_mm':summary(deviation)})
        root=np.linalg.norm(xyz[side,:,0]-markers[side,:,0],axis=-1)*1000
        report['hands'][hand]={'recorded_frames':len(times),'segments_per_frame':26,
                              'finite_positions_fraction':float(np.isfinite(xyz[side]).all(-1).mean()),
                              'five_finger0_origins_max_separation_mm':float(np.nanmax(np.linalg.norm(xyz[side][:,[1,6,11,16,21]]-xyz[side][:,[1]],axis=-1))*1000),
                              'root_to_wrist_marker_mm':summary(root),'provisional_21_to_marker21_mm':summary(error),
                              'provisional_point_error_mm':[dict(segment=names[side][int(j)],**summary(error[:,k])) for k,j in enumerate(select)],
                              'bones':perbone}
    with (c['dataset']/'multiview_frames.csv').open() as f:sync=list(csv.DictReader(f))
    ego_times=np.array([int(r['camera0_timestamp_ns']) for r in sync],np.int64);seed=json.loads(Path(alignment['seed_alignment']).read_text());m=seed['time_mapping'];target=m['nokov_origin_timestamp_raw']+((ego_times-m['ego_origin_timestamp_ns'])*1e-9*m.get('a',1)+m['b_s'])/m['nokov_seconds_per_timestamp_unit']
    rt,head,q=load_rigid(nokov/'nokov_rigid_bodies.csv');head,valid=interpolate_points(rt,head,target);inside=valid&(target>=rt[0])&(target<=rt[-1]);rot=np.full((len(target),3,3),np.nan);rot[inside]=Slerp(rt,Rotation.from_quat(q))(target[inside]).as_matrix();T=np.asarray(alignment['T_cam0_head'])
    projected=[];qs=[]
    for side in range(2):
        if skeleton_name[side] is None: projected.append(np.full((len(target),26,3),np.nan)); qs.append(np.full((len(target),26,4),np.nan)); continue
        world,ok=interpolate_points(times,xyz[side],target);head_points=np.einsum('fji,fkj->fki',rot,world-head[:,None]);camera=head_points@T[:3,:3].T+T[:3,3];projected.append(camera)
        # Save global SDK quaternions only; do not silently convert their frame semantics.
        values=np.full((len(target),26,4),np.nan)
        for j in range(26):values[inside,j]=Slerp(times,Rotation.from_quat(quat[side,:,j]))(target[inside]).as_quat()
        qs.append(values)
    marker_cam0,_=raw24_context(c)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'aligned_skeleton.npz',positions_cam0_m=np.asarray(projected),sdk_quaternions_xyzw=np.asarray(qs),marker24_cam0_m=marker_cam0,reference_timestamp_ns=ego_times,names=np.asarray(names),parents=np.asarray(parents),provisional21_indices=select,geometry_space='physical_camera0',alignment_unchanged=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'context.json').write_text(a.context.read_text())
    for hand,d in report['hands'].items():
        if not d.get('recorded', True): print(hand,'not recorded',flush=True); continue
        print(hand,'root marker distance mm',d['root_to_wrist_marker_mm'],'candidate21 marker distance',d['provisional_21_to_marker21_mm'],flush=True)
        print('finger bone length standard deviations mm',[round(v['length_std_mm'],4) for v in d['bones']],flush=True)
    print(a.output,flush=True)


if __name__=='__main__':main()
