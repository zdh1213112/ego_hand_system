#!/usr/bin/env python3
"""Show existing VIO/NOKOV trajectories in 3D/2D and report every valid pose.

Reuse the historical experiment's fixed time/installation calibration and
first-two-second world registration. Do not rerun VIO or refit sensor timing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_basalt_ego4_comparison import (CASES, COLORS, LABELS, aligned_poses,
                                           interpolate_reference, position_se3_alignment)
from fuse_basalt_hand_trajectory import load_basalt_trajectory
from ego_data.time_sync import require_zero_camera_offset

ROOT=Path(__file__).resolve().parents[1]


def error_statistics(position_error, rotation_error, select):
    p,r=position_error[select],rotation_error[select]
    return {'count':int(len(p)),
            'position_rmse_m':float(np.sqrt(np.mean(p*p))) if len(p) else None,
            'position_max_m':float(np.max(p)) if len(p) else None,
            'orientation_rmse_deg':float(np.sqrt(np.mean(r*r))) if len(r) else None}


def metric_windows(times,position_error,rotation_error,world_end=2.,sensor_end=20.):
    return {name:error_statistics(position_error,rotation_error,mask) for name,mask in [
        ('all',np.ones(len(times),dtype=bool)),
        ('after_world_registration',times>world_end),
        ('sensor_calibration_interval',times<sensor_end),
        ('after_sensor_calibration',times>=sensor_end),
        ('after_calibration_buffer',times>=sensor_end+.2)]}


def build_legacy(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'choose an empty output directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    comparison=args.comparison.resolve()
    info,calibration,origin,query=interpolate_reference(comparison)
    summary={'comparison_directory':str(comparison),'repeat':args.repeat,'origin_timestamp_ns':str(origin),
             'registration':'one constant 3D rotation and translation from first 2 s, applied to all poses; scale=1',
             'reference':'NOKOV head rigid body mapped to IMU origin with existing estimated installation/time calibration',
             'camera_time_shift_ns_used_by_original_run':info['camera_time_shift_ns'],
             'nokov_to_imu_offset_s':calibration['mocap_to_imu_offset_s'],
             'sensor_calibration_fit_interval_s':[.5,20.],
             'all_valid_times_evaluated':True,'metric_windows':{
                 'all':'all valid reference poses, including registration/calibration intervals',
                 'after_world_registration':'t>2 s, still includes sensor calibration samples',
                 'sensor_calibration_interval':'t<20 s, includes sensor calibration samples',
                 'after_sensor_calibration':'t>=20 s, historical main metric',
                 'after_calibration_buffer':'t>=20.2 s, skips the centred-filter boundary overlap'},'cases':{}}
    payload={'summary':summary,'cases':{},'reference':None,'time':None,'timestamps_ns':None}
    for case in CASES:
        path=comparison/'runs'/case/f'repeat_{args.repeat}'/'trajectory.csv'
        poses=load_basalt_trajectory(path)
        timestamps=np.array(sorted(poses),dtype=np.int64)
        times=(timestamps-np.int64(origin))*1e-9
        matrices=np.array([poses[int(t)] for t in timestamps])
        if len(times)!=info['prepared_groups'] or not np.isfinite(matrices).all():
            raise ValueError(f'incomplete trajectory: {path}')
        if payload['timestamps_ns'] is not None and payload['timestamps_ns']!=[str(t) for t in timestamps]:
            raise ValueError('comparison trajectories must have identical capture times')
        raw=matrices[:,:3,3]
        rotation=Rotation.from_matrix(matrices[:,:3,:3])
        gt,gt_rotation=query(times)
        aligned,aligned_rotation,r,t=aligned_poses(raw,rotation,gt,gt_rotation,times)
        pe=np.linalg.norm(aligned-gt,axis=1)
        re=np.rad2deg((gt_rotation.inv()*aligned_rotation).magnitude())
        stats=metric_windows(times,pe,re)
        entry={'trajectory':str(path),'label':LABELS[case],'color':COLORS[case],
               'metrics':stats,'world_rotation':r.as_matrix().tolist(),'world_translation_m':t.tolist(),
               'world_rotation_angle_deg':float(np.rad2deg(r.magnitude())),
               'registration_samples':int((times<=2).sum()),
               'first_raw_position_m':raw[0].tolist(),'first_aligned_position_m':aligned[0].tolist(),
               'first_reference_position_m':gt[0].tolist(),
               'global_position_se3_ate_rmse_m':float(np.sqrt(np.mean(np.linalg.norm(position_se3_alignment(raw,gt)-gt,axis=1)**2)))}
        summary['cases'][case]=entry
        payload['cases'][case]={'raw':raw.tolist(),'aligned':aligned.tolist(),'position_error_m':pe.tolist(),'rotation_error_deg':re.tolist()}
        payload['reference']=gt.tolist();payload['time']=times.tolist();payload['timestamps_ns']=[str(t) for t in timestamps]
        with (output/f'{case}_all_poses.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['timestamp_ns','time_s','raw_vio_x_m','raw_vio_y_m','raw_vio_z_m','aligned_vio_x_m','aligned_vio_y_m','aligned_vio_z_m','nokov_imu_x_m','nokov_imu_y_m','nokov_imu_z_m','position_error_m','orientation_error_deg','used_for_world_registration','in_sensor_calibration_interval','historical_holdout'])
            for i,stamp in enumerate(timestamps):
                writer.writerow([int(stamp),times[i],*raw[i],*aligned[i],*gt[i],pe[i],re[i],int(times[i]<=2),int(times[i]<20),int(times[i]>=20)])
    summary['poses']=len(payload['time']);summary['duration_s']=payload['time'][-1]
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (output/'metrics_all_intervals.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','repeat','interval','poses','position_rmse_cm','orientation_rmse_deg'])
        for case,entry in summary['cases'].items():
            for interval,m in entry['metrics'].items():
                w.writerow([case,args.repeat,interval,m['count'],m['position_rmse_m']*100 if m['count'] else '',m['orientation_rmse_deg']])
    template=(ROOT/'scripts/templates/vio_nokov_trajectories.html').read_text()
    (output/'viewer.html').write_text(template.replace('__TRAJECTORY_DATA__',json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('</','<\\/')))
    plot_trajectories(output,payload)
    print(json.dumps({'viewer':str(output/'viewer.html'),'poses':summary['poses'],'metrics':{name:c['metrics'] for name,c in summary['cases'].items()}},indent=2),flush=True)
    return summary


def build(args):
    """Load any supported saved-result layout through the common adapter."""
    if getattr(args,'layout','comparison')=='legacy':return build_legacy(args)
    from ego_data.vio_results import load_vio_result
    from ego_data.map_view import trajectory_comparison_data,write_map_view
    result=load_vio_result(args.comparison,repeat=args.repeat)
    data=trajectory_comparison_data(result,args.output)
    write_map_view(args.output,data)
    print(json.dumps({'viewer':str(args.output.resolve()/'viewer.html'),
        'source_format':result['format'],'scored_poses':len(data['time']),
        'scored_interval_s':data['summary']['scored_interval_s']},ensure_ascii=False,indent=2))
    return data['summary']


def plot_trajectories(output,data):
    os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14,11),constrained_layout=True)
    axes=[fig.add_subplot(221,projection='3d'),*[fig.add_subplot(i) for i in [222,223,224]]]
    gt=np.array(data['reference'])
    cloud=np.vstack([gt,*[np.array(c['aligned']) for c in data['cases'].values()]])
    lo,hi=cloud.min(axis=0),cloud.max(axis=0);center=(lo+hi)/2;radius=max(hi-lo)*.55
    axes[0].plot(*gt.T,color='black',lw=2,label='NOKOV / IMU reference')
    for name,c in data['cases'].items():
        axes[0].plot(*np.array(c['aligned']).T,color=COLORS[name],lw=1,label=LABELS[name])
    axes[0].set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title='3D / first-2s fixed world registration')
    axes[0].set_box_aspect((1,1,1));axes[0].legend(fontsize=8)
    for ax,(a,b),title in zip(axes[1:],[(0,1),(0,2),(1,2)],['XY top view','XZ side view','YZ side view']):
        ax.plot(gt[:,a],gt[:,b],color='black',lw=2)
        for name,c in data['cases'].items():
            p=np.array(c['aligned']);ax.plot(p[:,a],p[:,b],color=COLORS[name],lw=1)
        ax.set(title=title,xlabel='XYZ'[a]+' [m]',ylabel='XYZ'[b]+' [m]');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.25)
    fig.suptitle(f'VIO / NOKOV trajectories | repeat {data["summary"]["repeat"]} | all {len(gt)} valid poses',fontsize=14)
    fig.savefig(output/'trajectories_3d_2d.png',dpi=160);plt.close(fig)


def refresh_viewer(output):
    """Refresh presentation only; retain all source poses, metrics and exports."""
    page=output/'viewer.html'
    if '<script id="map-data"' in page.read_text():
        from ego_data.map_view import refresh_map_view
        refresh_map_view(output)
        return
    embedded=page.read_text().split('<script id="trajectory-data" type="application/json">',1)[1].split('</script>',1)[0]
    require_zero_camera_offset(json.loads(embedded)['summary'],str(output))
    template=(ROOT/'scripts/templates/vio_nokov_trajectories.html').read_text()
    page.write_text(template.replace('__TRAJECTORY_DATA__',embedded))
    print(json.dumps({'viewer':str(page),'data_and_metrics_changed':False,'default_display':'progressive from first pose'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,default=ROOT/'output/basalt_comparison_20261104_195844')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--layout',choices=['comparison','legacy'],default='comparison',help='Shared interactive algorithm comparison, or historical trajectory page')
    parser.add_argument('--refresh-viewer',action='store_true',help='Refresh an existing page UI without recalculating data or metrics')
    args=parser.parse_args()
    if args.repeat<1:
        parser.error('--repeat must be >=1')
    if args.refresh_viewer:
        refresh_viewer(args.output.resolve())
    else:
        build(args)
