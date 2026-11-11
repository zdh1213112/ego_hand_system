#!/usr/bin/env python3
"""Stage 2: full VIO, sparse mapping and evaluation with frozen relations.

No time calibration, installation fitting or world registration is performed.
All poses and landmarks are transformed by previously saved matrices.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compare_basalt_ego4 import prepare_data, run_cases, run_mapping
from calibrate_ego_nokov_relations import link_binaries
from analyze_basalt_ego4_comparison import CASES, COLORS, LABELS, relative_pose_errors
from fuse_basalt_hand_trajectory import load_basalt_trajectory
from ego_data.fixed_relations import (FixedReference, check_initialization, error_summary,
                                      pose_errors, save_json, sha256, transform_poses,
                                      validate_bindings)
from visualize_vio_nokov_trajectories import metric_windows, plot_trajectories
from ego_data.map_view import write_map_view, refresh_map_view
from ego_data.time_sync import require_zero_camera_offset

ROOT=Path(__file__).resolve().parents[1]


def write_ply(path,points):
    with path.open('w') as f:
        f.write(f'ply\nformat ascii 1.0\ncomment NOKOV world; metres; frozen world transform\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for p in points:
            f.write(' '.join(f'{v:.16g}' for v in p)+'\n')


def write_poses(path,stamps,poses):
    q=Rotation.from_matrix(poses[:,:3,:3]).as_quat()
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp_ns','x_m','y_m','z_m','qw','qx','qy','qz'])
        for t,p,quat in zip(stamps,poses,q):
            w.writerow([int(t),*p[:3,3],quat[3],*quat[:3]])


def evaluate_full(output,relations,relations_file):
    info=json.loads((output/'input_validation.json').read_text())
    require_zero_camera_offset(info,str(output))
    require_zero_camera_offset(relations,str(relations_file))
    origin=int(info['first_timestamp_ns'])
    reference=FixedReference(relations,origin)
    evaluated={};view_cases={};reference_points=None;times=None;stamps=None;expected_stamps=None
    export=output/'evaluation';export.mkdir(exist_ok=True)
    for case in CASES:
        raw=load_basalt_trajectory(output/'runs'/case/'repeat_1/trajectory.csv')
        stamps=np.array(sorted(raw),dtype=np.int64);times=(stamps-origin)*1e-9
        if expected_stamps is not None and not np.array_equal(stamps,expected_stamps):
            raise ValueError('all cases must have identical image update times')
        expected_stamps=stamps.copy()
        matrices=np.array([raw[int(s)] for s in stamps])
        if len(stamps)!=info['prepared_groups'] or not np.isfinite(matrices).all():
            raise ValueError(f'incomplete VIO trajectory: {case}')
        world=relations['world_relations'][case]
        initialization=check_initialization(stamps[0],matrices[0],world['initialization'])
        fixed=transform_poses(matrices,world['T_nokov_vio'])
        gp,gr,valid=reference.query(times)
        if not valid.all():
            raise ValueError(f'NOKOV does not cover every VIO time: {case}; no extrapolation allowed')
        pe,re=pose_errors(fixed,gp,gr,valid)
        if reference_points is None:
            ground_truth=np.tile(np.eye(4),(len(stamps),1,1))
            ground_truth[:,:3,3]=gp;ground_truth[:,:3,:3]=gr.as_matrix()
            write_poses(export/'nokov_imu_reference.csv',stamps,ground_truth)
        write_poses(export/f'{case}_imu_nokov_world.csv',stamps,fixed)
        with (export/f'{case}_errors.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp_ns','time_s','vio_x_m','vio_y_m','vio_z_m','nokov_imu_x_m','nokov_imu_y_m','nokov_imu_z_m','position_error_m','orientation_error_deg'])
            for i,stamp in enumerate(stamps):
                w.writerow([int(stamp),times[i],*fixed[i,:3,3],*gp[i],pe[i],re[i]])
        evaluated[case]={'label':LABELS[case],'color':COLORS[case],
                         'initialization_check':initialization,
                         'all':error_summary(pe,re,valid),
                         'after_calibration':error_summary(pe,re,valid&(times>=relations['sensor_calibration_interval_s'][1]+.2)),
                         'rpe_0_1s_all':relative_pose_errors(times,fixed[:,:3,3],Rotation.from_matrix(fixed[:,:3,:3]),gp,gr,.1),
                         'rpe_1s_all':relative_pose_errors(times,fixed[:,:3,3],Rotation.from_matrix(fixed[:,:3,:3]),gp,gr,1.),
                         'world_transform_loaded':world['T_nokov_vio'],
                         'wall_s':json.loads((output/'runs'/case/'repeat_1/vio_timing.json').read_text())['wall_s']}
        view_cases[case]={'raw':matrices[:,:3,3].tolist(),'aligned':fixed[:,:3,3].tolist(),'position_error_m':pe.tolist(),'rotation_error_deg':re.tolist()}
        reference_points=gp
    summary={'relations_file':str(relations_file),'relations_sha256':sha256(relations_file),
             'ego_session':relations['ego_session'],'nokov_directory':relations['nokov_directory'],
             'fixed_parameters_only':True,'time_or_mount_or_world_refitted':False,
             'camera_time_policy':relations['camera_time_policy'],
             'nokov_to_imu_offset_s':relations['nokov_to_imu_offset_s'],
             'T_marker_imu':relations['T_marker_imu'],
             'T_imu_marker':np.linalg.inv(np.array(relations['T_marker_imu'])).tolist(),
             'transform_convention':'T_A_B maps B coordinates into A coordinates; T_N_I = T_N_M @ T_M_I',
             'calibration_scope':relations.get('scope','prefix-based calibration'),
             'sensor_calibration_interval_s':relations['sensor_calibration_interval_s'],
             'world_alignment_intervals_s':{case:w['fit_interval_s'] for case,w in relations['world_relations'].items()},
             'poses_per_case':len(stamps),'duration_s':float(times[-1]),
             'reference':'NOKOV head body mapped to IMU origin with frozen time and installation calibration',
             'scope':'All reference-covered poses scored; first calibration interval is also included and separately labelled',
             'cases':evaluated}
    # Reuse the progressive trajectory presentation, without calling its fitting stage.
    trajectory_summary={'comparison_directory':str(output),'repeat':1,'poses':len(stamps),'duration_s':float(times[-1]),
                        'camera_time_shift_ns_used_by_original_run':relations['camera_time_policy']['camera_to_imu_offset_ns'],
                        'nokov_to_imu_offset_s':relations['nokov_to_imu_offset_s'],
                        'registration':'Loaded from frozen relations.json; no registration performed during this full run',
                        'reference':summary['reference'],'fixed_relations':str(relations_file),
                        'world_registration_end_s':max(w['fit_interval_s'][1] for w in relations['world_relations'].values()),
                        'sensor_calibration_end_s':relations['sensor_calibration_interval_s'][1],'cases':{}}
    for case in CASES:
        world=relations['world_relations'][case];transform=np.array(world['T_nokov_vio']);v=view_cases[case]
        trajectory_summary['cases'][case]={'label':LABELS[case],'color':COLORS[case],
            'metrics':metric_windows(times,np.array(v['position_error_m']),np.array(v['rotation_error_deg']),trajectory_summary['world_registration_end_s'],trajectory_summary['sensor_calibration_end_s']),
            'world_translation_m':transform[:3,3].tolist(),'world_rotation_angle_deg':float(np.rad2deg(Rotation.from_matrix(transform[:3,:3]).magnitude())),
            'registration_samples':world['fit_poses'],'first_raw_position_m':v['raw'][0],
            'first_aligned_position_m':v['aligned'][0],'first_reference_position_m':reference_points[0].tolist()}
    trajectory_data={'summary':trajectory_summary,'cases':view_cases,'time':times.tolist(),'timestamps_ns':[str(t) for t in stamps],'reference':reference_points.tolist()}
    with (output/'metrics_all_intervals.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','interval','poses','position_rmse_cm','orientation_rmse_deg'])
        for case,entry in trajectory_summary['cases'].items():
            for interval,m in entry['metrics'].items():
                w.writerow([case,interval,m['count'],m['position_rmse_m']*100 if m['count'] else '',m['orientation_rmse_deg']])
    html=(ROOT/'scripts/templates/vio_nokov_trajectories.html').read_text()
    (output/'trajectories.html').write_text(html.replace('__TRAJECTORY_DATA__',json.dumps(trajectory_data,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('</','<\\/')))
    plot_trajectories(output,trajectory_data)
    return summary,trajectory_data,reference


def export_maps(output,relations,summary,trajectory_data,reference):
    origin=int(trajectory_data['timestamps_ns'][0]);maps={}
    for case in CASES:
        source=output/'runs'/case/'repeat_1/map_lm'
        timing=json.loads((source.parent/'mapping_lm_timing.json').read_text())
        if timing['returncode']!=0:
            raise RuntimeError(f'mapping failed: {case}')
        with (source/'sparse_landmarks.csv').open() as f:rows=list(csv.DictReader(f))
        first_seen={}
        with (source/'map_observations.csv').open() as f:
            for observation in csv.DictReader(f):
                lid=observation['landmark_id'];stamp=int(observation['timestamp_ns'])
                first_seen[lid]=min(first_seen.get(lid,stamp),stamp)
        raw=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in rows])
        transform=np.array(relations['world_relations'][case]['T_nokov_vio'])
        points=raw@transform[:3,:3].T+transform[:3,3]
        destination=output/'maps'/case;destination.mkdir(parents=True,exist_ok=True)
        write_ply(destination/'map_nokov_world.ply',points)
        with (destination/'landmarks_nokov_world.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['landmark_id','x_m','y_m','z_m','observation_count'])
            for row,p in zip(rows,points):w.writerow([row['landmark_id'],*p,row['observation_count']])
        with (source/'keyframe_imu_poses.csv').open() as f:keyframes=list(csv.DictReader(f))
        stamps=np.array([int(r['timestamp_ns']) for r in keyframes],dtype=np.int64)
        poses=np.tile(np.eye(4),(len(keyframes),1,1))
        poses[:,:3,3]=[[float(r[k]) for k in ['x_m','y_m','z_m']] for r in keyframes]
        poses[:,:3,:3]=Rotation.from_quat([[float(r[k]) for k in ['qx','qy','qz','qw']] for r in keyframes]).as_matrix()
        fixed=transform_poses(poses,transform);write_poses(destination/'keyframe_imu_nokov_world.csv',stamps,fixed)
        gp,gr,valid=reference.query((stamps-origin)*1e-9);pe,re=pose_errors(fixed,gp,gr,valid)
        source_summary=json.loads((source/'summary.json').read_text())
        summary['cases'][case]['mapping']={**source_summary,'wall_s':timing['wall_s'],
            'keyframe_pose_errors':error_summary(pe,re,valid),'external_surface_ground_truth_available':False,
            'map_world_transform_loaded':relations['world_relations'][case]['T_nokov_vio']}
        maps[case]={'points':np.round(points,5).tolist(),'observations':[int(r['observation_count']) for r in rows],
                    'first_seen_s':[(first_seen[r['landmark_id']]-origin)*1e-9 for r in rows],
                    'keyframes':np.round(fixed[:,:3,3],6).tolist(),
                    'keyframe_times_s':((stamps-origin)*1e-9).tolist(),
                    'online':trajectory_data['cases'][case]['aligned'],
                    'reference':trajectory_data['reference'],'stats':summary['cases'][case]}
    data={'summary':summary,'maps':maps,'time':trajectory_data['time']}
    write_map_view(output,data)
    plot_maps(output,data)


def plot_maps(output,data):
    import os
    os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,6),constrained_layout=True)
    clouds=[]
    for case in CASES:
        m=data['maps'][case];p=np.array(m['points']);gt=np.array(m['reference'])
        clouds.extend([p[np.linalg.norm(p-gt[0],axis=1)<=5],np.array(m['online']),gt])
    cloud=np.vstack(clouds);lo,hi=cloud.min(axis=0),cloud.max(axis=0)
    center=(lo+hi)/2;radius=max(hi-lo)*.55
    for i,case in enumerate(CASES):
        ax=fig.add_subplot(1,3,i+1,projection='3d');m=data['maps'][case];p=np.array(m['points']);track=np.array(m['online']);gt=np.array(m['reference'])
        mask=np.linalg.norm(p-gt[0],axis=1)<=5
        ax.scatter(*p[mask].T,c=p[mask,2],cmap='viridis',vmin=lo[2],vmax=hi[2],s=2,alpha=.6)
        ax.plot(*track.T,color=COLORS[case],lw=1);ax.plot(*gt.T,color='black',lw=1)
        ax.set(title=f'{LABELS[case]}\n{mask.sum()} / {len(p)} points within 5 m',xlabel='NOKOV X [m]',ylabel='Y [m]',zlabel='Z [m]')
        ax.set_box_aspect((1,1,1))
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
    fig.suptitle('Frozen transforms / NFR sparse maps (display radius only; PLY exports retain all points)')
    fig.savefig(output/'maps_3d.png',dpi=160);plt.close(fig)


def main(args):
    relation_file=args.relations.resolve();relation_hash=sha256(relation_file)
    relations=json.loads(relation_file.read_text());validate_bindings(relations)
    output=args.output.resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'choose a new full-run directory or use --resume: {output}')
        saved=output/'relations_used.json'
        if not saved.exists() or json.loads(saved.read_text())!=relations:
            raise ValueError('resume requires the identical frozen calibration package')
    else:
        output.mkdir(parents=True)
        link_binaries(output,Path(relations['build_cache']))
    args.session=Path(relations['ego_session']);args.output=output;args.image_scale=relations['image_scale'];args.max_frames=0
    if not (args.resume and (output/'input_validation.json').exists()):
        prepare_data(args)
    info=json.loads((output/'input_validation.json').read_text())
    if str(info['first_timestamp_ns'])!=relations['first_image_timestamp_ns']:
        raise ValueError('full run starts from a different image')
    for family in ['original','monado']:
        if sha256(output/f'config_{family}.json')!=relations['config_sha256'][family]:
            raise ValueError(f'VIO config differs from calibration: {family}')
    save_json(output/'relations_used.json',relations)
    save_json(output/'nokov_calibration.json',relations['sensor_calibration'])
    args.cases=list(CASES);args.repeats=1;args.threads=relations['threads'];args.save_marg=True
    run_cases(args)
    for case in CASES:
        if json.loads((output/'runs'/case/'repeat_1/vio_timing.json').read_text())['returncode']!=0:
            raise RuntimeError(f'full VIO failed: {case}')
    summary,trajectory_data,reference=evaluate_full(output,relations,relation_file)
    args.mapping_solver='lm';run_mapping(args)
    export_maps(output,relations,summary,trajectory_data,reference)
    if sha256(relation_file)!=relation_hash:
        raise ValueError('frozen relation file changed during full run')
    save_json(output/'summary.json',summary)
    with (output/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','poses','all_position_rmse_cm','all_orientation_rmse_deg','VIO_wall_s','map_points','map_reprojection_rmse_px','map_wall_s'])
        for case,r in summary['cases'].items():
            m=r['mapping'];w.writerow([case,r['all']['poses'],r['all']['position_rmse_m']*100,r['all']['orientation_rmse_deg'],r['wall_s'],m['landmarks'],m['reprojection_rmse_px'],m['wall_s']])
    print(json.dumps({'viewer':str(output/'viewer.html'),'summary':str(output/'summary.json'),'fixed_relations_sha256':relation_hash},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relations',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--resume',action='store_true',help='Reuse completed VIO/mapping in an existing output with the identical frozen package')
    p.add_argument('--refresh-viewer',action='store_true',help='Refresh display from existing results without rerunning VIO, mapping or calibration')
    args=p.parse_args()
    if args.refresh_viewer:
        refresh_map_view(args.output.resolve())
        print(json.dumps({'viewer':str(args.output.resolve()/'viewer.html'),
                          'vio_or_mapping_rerun':False,'calibration_refitted':False,'reports_refreshed':True},indent=2))
    else:
        if args.relations is None:p.error('--relations is required unless --refresh-viewer is used')
        main(args)
