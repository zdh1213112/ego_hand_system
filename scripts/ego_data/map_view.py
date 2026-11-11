"""Presentation frames for frozen VIO maps; never fit or alter an estimate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ego_data.fixed_relations import sha256
from ego_data.vio_performance import collect_vio_performance
from ego_data.time_sync import require_zero_camera_offset

ROOT=Path(__file__).resolve().parents[2]
COMPARISON_COLOURS={'reference':'#007f73','original_2cam':'#165dff','monado_2cam':'#d600a9','monado_4cam':'#f06b00'}
COMPARISON_LABELS={'original_2cam':'原版 Basalt · 双目',
                   'monado_2cam':'Monado · 双目','monado_4cam':'Monado · 四目'}


def read_poses(path):
    with Path(path).open() as f:
        rows=list(csv.DictReader(f))
    times=np.array([int(row['timestamp_ns']) for row in rows],dtype=np.int64)
    poses=np.tile(np.eye(4),(len(rows),1,1))
    poses[:,:3,3]=[[float(row[k]) for k in ['x_m','y_m','z_m']] for row in rows]
    poses[:,:3,:3]=Rotation.from_quat([[float(row[k]) for k in ['qx','qy','qz','qw']] for row in rows]).as_matrix()
    return times,poses


def change_frame(points,transform):
    return np.asarray(points)@transform[:3,:3].T+transform[:3,3]


def initial_imu_frame(first_pose):
    """Fixed I0 frame: origin and axes of the first estimated IMU pose."""
    return np.linalg.inv(np.asarray(first_pose))


def enrich_map_data(output,data):
    output=Path(output)
    data.pop('semantic',None)
    reference_stamps,reference_poses=read_poses(output/'evaluation/nokov_imu_reference.csv')
    if len(reference_stamps)!=len(data['time']):
        raise ValueError('reference/time mismatch')
    transform=initial_imu_frame(reference_poses[0])
    reference=reference_poses[:,:3,3]
    comparison={'imu0':{'online':{},'reference':np.round(change_frame(reference,transform),7).tolist(),
                         'imu_rotations':{},'T_imu0_nokov':transform.tolist()},
                'nokov':{'online':{},'reference':reference.tolist(),'imu_rotations':{}},
                'colours':COMPARISON_COLOURS,'labels':COMPARISON_LABELS,
                'reference_label':'NOKOV 真值（IMU 中心）'}
    performances={};stamps_by_case={}
    errors={'timestamps_ns':[str(int(t)) for t in reference_stamps],'cases':{}}
    for case,m in data['maps'].items():
        for key in ['semantic_ids','semantic_rgb','semantic_confidence']:
            m.pop(key,None)
        stamps,poses=read_poses(output/'evaluation'/f'{case}_imu_nokov_world.csv')
        if not np.array_equal(stamps,reference_stamps):
            raise ValueError('comparison requires identical algorithm/reference timestamps')
        if not np.allclose(poses[:,:3,3],m['online'],atol=1e-10,rtol=0):
            raise ValueError('viewer trajectory differs from saved evaluated poses')
        stamps_by_case[case]=stamps
        m['initial_imu_pose_nokov']=poses[0].tolist()
        m['imu_rotations']=np.round(poses[:,:3,:3].reshape(-1,9),9).tolist()
        m['imu0']={key:np.round(change_frame(m[key],transform),7).tolist()
                   for key in ['points','online','reference','keyframes']}
        m['imu0']['imu_rotations']=np.round(np.einsum('ij,njk->nik',transform[:3,:3],poses[:,:3,:3]).reshape(-1,9),9).tolist()
        comparison['nokov']['online'][case]=poses[:,:3,3].tolist()
        comparison['nokov']['imu_rotations'][case]=m['imu_rotations']
        comparison['imu0']['online'][case]=m['imu0']['online']
        comparison['imu0']['imu_rotations'][case]=m['imu0']['imu_rotations']
        stats=m['stats']
        position_error,orientation_error=pose_error_series(poses,reference_poses)
        saved_errors=output/'evaluation'/f'{case}_errors.csv'
        if saved_errors.exists():
            with saved_errors.open() as f:error_rows=list(csv.DictReader(f))
            if [int(r['timestamp_ns']) for r in error_rows]!=stamps.tolist():
                raise ValueError('saved per-frame error timestamps differ from comparison poses')
            saved_position=np.array([float(r['position_error_m']) for r in error_rows])
            saved_orientation=np.array([float(r['orientation_error_deg']) for r in error_rows])
            if (not np.isfinite(saved_position).all() or not np.isfinite(saved_orientation).all()
                    or not np.allclose(saved_position,position_error,atol=1e-8,rtol=0)
                    or not np.allclose(saved_orientation,orientation_error,atol=1e-6,rtol=0)):
                raise ValueError('saved per-frame errors differ from evaluated poses')
            position_error,orientation_error=saved_position,saved_orientation
        errors['cases'][case]={'position_error_m':position_error.tolist(),
                              'orientation_error_deg':orientation_error.tolist()}
        stats['accuracy']=accuracy_metrics(poses,reference_poses,(position_error,orientation_error))
        performance=collect_vio_performance(output,case)
        performances[case]=performance;stats['performance']=performance
        expected=performance.get('expected_frames')
        stats['accuracy']['pose_output_coverage_pct']=100*len(stamps)/expected if expected else None
        if case in data.get('summary',{}).get('cases',{}):
            data['summary']['cases'][case].update(accuracy=stats['accuracy'],performance=performance)
        source=output/'runs'/case/'repeat_1/map_lm/sparse_landmarks.csv'
        with source.open() as f:
            identifiers=np.array([int(row['landmark_id']) for row in csv.DictReader(f)],dtype=np.int64)
        m['landmark_ids']=identifiers.tolist()
        colour_cache=output/'maps'/case/'semantic_map.npz'
        if colour_cache.exists():
            with np.load(colour_cache,allow_pickle=False) as annotation:
                if not np.array_equal(annotation['landmark_ids'],identifiers) or str(annotation['source_map_sha256'])!=sha256(source):
                    raise ValueError(f'cached RGB belongs to different landmarks: {case}')
                m['rgb']=annotation['rgb'].tolist()
                export_common_rgb(output,case,m,transform)
    data['comparison']=comparison
    data['errors']=errors
    data['display_frames']={'default':'imu0','definition':'One shared I0 frame is the first NOKOV reference IMU pose; every algorithm, reference, point cloud and attitude uses this same fixed transform, without fitting',
                            'common_origin_source':'NOKOV reference IMU first pose',
                            'T_nokov_imu0':reference_poses[0].tolist(),'T_imu0_nokov':transform.tolist(),
                            'sensor_axes':'IMU body axes from calibration; do not assume X forward or Z gravity-up'}
    data['rgb_static_plot_available']=(output/'rgb_maps_imu0.png').exists()
    (output/'map_view_transforms.json').write_text(json.dumps(data['display_frames'],indent=2)+'\n')
    write_comparison_metrics(output,data,performances,stamps_by_case)
    return data


def pose_error_series(poses,reference_poses):
    error=np.linalg.norm(poses[:,:3,3]-reference_poses[:,:3,3],axis=1)
    angles=np.rad2deg((Rotation.from_matrix(reference_poses[:,:3,:3]).inv()*Rotation.from_matrix(poses[:,:3,:3])).magnitude())
    return error,angles


def accuracy_metrics(poses,reference_poses,errors=None):
    error,angles=pose_error_series(poses,reference_poses) if errors is None else errors
    return {'poses':len(poses),'position_rmse_m':float(np.sqrt(np.mean(error**2))),
        'position_median_m':float(np.median(error)),'position_p95_m':float(np.percentile(error,95)),
        'position_max_m':float(error.max()),'orientation_rmse_deg':float(np.sqrt(np.mean(angles**2))),
        'orientation_p95_deg':float(np.percentile(angles,95)),'orientation_max_deg':float(angles.max()),
        'final_position_error_m':float(error[-1]),'final_orientation_error_deg':float(angles[-1]),
        'reference_path_length_m':float(np.linalg.norm(np.diff(reference_poses[:,:3,3],axis=0),axis=1).sum()),
        'vio_path_length_m':float(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).sum())}


def export_common_rgb(output,case,m,transform):
    """Derived display export; original NOKOV geometry and colours stay intact."""
    directory=output/'maps'/case
    source=directory/'landmarks_nokov_world.csv'
    if not source.exists():return
    with source.open() as f:rows=list(csv.DictReader(f))
    if [int(r['landmark_id']) for r in rows]!=m['landmark_ids']:
        raise ValueError('RGB export landmark order mismatch')
    points=change_frame([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in rows],transform)
    with (directory/'map_rgb_imu0.ply').open('w') as f:
        f.write(f'ply\nformat ascii 1.0\ncomment shared first reference IMU frame; metres\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(points,m['rgb']):f.write(' '.join([*[f'{v:.16g}' for v in p],*[str(int(v)) for v in c]])+'\n')


def write_comparison_metrics(output,data,performances,stamps_by_case):
    """Additional reports from saved estimates and measured timings only."""
    (output/'performance_metrics.json').write_text(json.dumps({'cases':performances},indent=2,allow_nan=False)+'\n')
    report={**data.get('summary',{}),'display_frames':data['display_frames'],
            'cases':{case:m['stats'] for case,m in data['maps'].items()}}
    (output/'comparison_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    performance_rows=[];combined=[]
    for case,m in data['maps'].items():
        p=performances[case];b=p['backend_frame_ms'];stats=m['stats'];a=stats['accuracy']
        row={'case':case,'processed_frames':p['processed_frames'],'backend_measured_frames':b['count'],
             'backend_mean_ms':b['mean'],'backend_p50_ms':b['median'],'backend_p90_ms':b['p90'],
             'backend_p95_ms':b['p95'],'backend_p99_ms':b['p99'],'backend_max_ms':b['max'],'backend_std_ms':b['std'],
             'VIO_wall_s':p['process_wall_s'],'wall_ms_per_frame':p['wall_ms_per_frame'],
             'throughput_fps':p['throughput_fps'],'realtime_speedup':p['realtime_speedup'],'rtf':p['rtf'],
             'peak_rss_mib':p['peak_rss_mib'],'cpu_percent':p['cpu_percent'],
             'cpu_user_s':p['cpu_user_s'],'cpu_system_s':p['cpu_system_s'],
             'configured_tbb_threads':p['configured_tbb_threads']}
        performance_rows.append(row)
        rpe=lambda interval,field:stats.get(f'rpe_{interval}_all',{}).get(field,{}).get('rmse')
        rp01=rpe('0_1s','translation_m');rp1=rpe('1s','translation_m')
        combined.append({'case':case,'poses':a['poses'],'pose_output_coverage_pct':a['pose_output_coverage_pct'],
            'position_rmse_cm':a['position_rmse_m']*100,'position_median_cm':a['position_median_m']*100,
            'position_p95_cm':a['position_p95_m']*100,'position_max_cm':a['position_max_m']*100,
            'orientation_rmse_deg':a['orientation_rmse_deg'],'orientation_p95_deg':a['orientation_p95_deg'],
            'orientation_max_deg':a['orientation_max_deg'],'final_position_error_cm':a['final_position_error_m']*100,
            'final_orientation_error_deg':a['final_orientation_error_deg'],
            'rpe_0_1s_translation_rmse_cm':rp01*100 if rp01 is not None else None,
            'rpe_0_1s_rotation_rmse_deg':rpe('0_1s','rotation_deg'),
            'rpe_1s_translation_rmse_cm':rp1*100 if rp1 is not None else None,
            'rpe_1s_rotation_rmse_deg':rpe('1s','rotation_deg'),
            'reference_path_length_m':a['reference_path_length_m'],'vio_path_length_m':a['vio_path_length_m'],
            **{k:v for k,v in row.items() if k not in ['case','processed_frames']},
            'map_points':stats.get('mapping',{}).get('landmarks'),
            'map_reprojection_rmse_px':stats.get('mapping',{}).get('reprojection_rmse_px'),
            'NFR_map_wall_s':stats.get('mapping',{}).get('wall_s')})
        if p['backend_samples_match_poses'] and len(p['backend_frame_samples_ms'])==len(stamps_by_case[case]):
            directory=output/'performance';directory.mkdir(exist_ok=True)
            with (directory/f'{case}_frame_times.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['frame_index','timestamp_ns','backend_wall_ms'])
                w.writerows((i,int(t),v) for i,(t,v) in enumerate(zip(stamps_by_case[case],p['backend_frame_samples_ms'])))
    for filename,rows in [('performance_metrics.csv',performance_rows),('metrics_comparison.csv',combined)]:
        if rows:
            with (output/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_map_view(output,data):
    output=Path(output)
    require_zero_camera_offset(data.get('summary',{}),str(output))
    source=Path(data.get('summary',{}).get('source_directory',output))
    for directory in {output,source}:
        info=directory/'input_validation.json'
        if info.exists():require_zero_camera_offset(json.loads(info.read_text()),str(directory))
    if not data.get('trajectory_only'):
        data=enrich_map_data(output,data)
    plot_comparison_views(Path(output),data)
    template=(ROOT/'scripts/templates/fixed_vio_maps.html').read_text()
    template=template.replace('__MAP_VIEW_SCRIPT__',(ROOT/'scripts/templates/fixed_vio_map_view.js').read_text())
    encoded=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (Path(output)/'viewer.html').write_text(template.replace('__MAP_DATA__',encoded))


def trajectory_comparison_data(result,output):
    """Adapt normalized saved trajectories to the existing comparison UI."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    first=next(iter(result['cases'].values()))['runs'][0]
    stamps=first['timestamps'];truth=first['reference'];times=first['time']
    transform=initial_imu_frame(truth[0]);reference=truth[:,:3,3]
    comparison={'imu0':{'online':{},'reference':change_frame(reference,transform).tolist(),'imu_rotations':{},'T_imu0_nokov':transform.tolist()},
        'nokov':{'online':{},'reference':reference.tolist(),'imu_rotations':{}},
        'colours':COMPARISON_COLOURS,'labels':{c:v['label'] for c,v in result['cases'].items()},
        'reference_label':'NOKOV 真值（IMU 中心）'}
    summary={'source_directory':str(result['directory']),'source_format':result['format'],
        'camera_shift_ns':0,'time_sync_policy':result['info'].get('time_sync_policy',{}),
        'session':result['label'],'duration_s':result['info']['duration_s'],'poses_per_case':len(stamps),
        'scored_interval_s':[float(times[0]),float(times[-1])],
        'input_poses':result['info']['prepared_groups'],'calibration_end_s':result['calibration_end_s'],
        'registration_interval_s':result['registration_interval_s'],'cases':{},'sources':result['sources']}
    data={'trajectory_only':True,'summary':summary,'time':times.tolist(),'maps':{},'comparison':comparison,
        'errors':{'timestamps_ns':[str(int(t)) for t in stamps],'cases':{}},
        'display_frames':{'default':'imu0','T_nokov_imu0':truth[0].tolist(),'T_imu0_nokov':transform.tolist()},
        'exports':{'metrics':'viewer_metrics.csv','performance':'viewer_performance_metrics.csv',
                   'summary':'viewer_summary.json','cases':{}}}
    export=output/'viewer_evaluation';export.mkdir(exist_ok=True)
    metric_rows=[];performance_rows=[]
    for name,case in result['cases'].items():
        run=case['runs'][0];poses=run['poses'];stats=run['stats']
        if not np.array_equal(run['timestamps'],stamps):raise ValueError('viewer times differ between algorithms')
        online=poses[:,:3,3].tolist();rotations=poses[:,:3,:3].reshape(-1,9).tolist()
        local=change_frame(poses[:,:3,3],transform).tolist()
        local_r=np.einsum('ij,njk->nik',transform[:3,:3],poses[:,:3,:3]).reshape(-1,9).tolist()
        data['maps'][name]={'points':[],'keyframes':[],'keyframe_times_s':[],'first_seen_s':[],
            'online':online,'reference':reference.tolist(),'imu_rotations':rotations,'stats':stats,
            'imu0':{'points':[],'keyframes':[],'online':local,'reference':comparison['imu0']['reference'],'imu_rotations':local_r}}
        comparison['nokov']['online'][name]=online;comparison['nokov']['imu_rotations'][name]=rotations
        comparison['imu0']['online'][name]=local;comparison['imu0']['imu_rotations'][name]=local_r
        data['errors']['cases'][name]={'position_error_m':run['position_error_m'].tolist(),
            'orientation_error_deg':run['orientation_error_deg'].tolist()}
        summary['cases'][name]=stats
        path=export/f'{name}_errors.csv'
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp_ns','recording_time_s','x_m','y_m','z_m','qw','qx','qy','qz',
                'nokov_x_m','nokov_y_m','nokov_z_m','position_error_m','orientation_error_deg','split'])
            quaternions=Rotation.from_matrix(poses[:,:3,:3]).as_quat()
            for i,t in enumerate(stamps):
                q=quaternions[i];w.writerow([int(t),times[i],*poses[i,:3,3],q[3],*q[:3],*reference[i],
                    run['position_error_m'][i],run['orientation_error_deg'][i],
                    'holdout' if times[i]>=result['calibration_end_s'] else 'calibration'])
        data['exports']['cases'][name]={'errors':str(path.relative_to(output))}
        p=stats['performance'];b=p['backend_frame_ms']
        if p['backend_samples_match_poses']:
            with run['source'].open() as f:native=[int(row[0]) for row in csv.reader(line for line in f if not line.startswith('#')) if row]
            backend=export/f'{name}_backend_times.csv'
            with backend.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['timestamp_ns','backend_wall_ms']);w.writerows(zip(native,p['backend_frame_samples_ms']))
            data['exports']['cases'][name]['performance']=str(backend.relative_to(output))
        metric_rows.append({'case':name,'repeat':run['repeat'],**stats['all'],
            'holdout_position_rmse_m':stats['holdout']['position_rmse_m'],
            'holdout_orientation_rmse_deg':stats['holdout']['orientation_rmse_deg'],
            'input_poses':run['full_output_poses'],'reference_coverage_pct':run['reference_coverage_pct']})
        performance_rows.append({'case':name,'repeat':run['repeat'],'process_wall_s':p['process_wall_s'],
            'throughput_fps':p['throughput_fps'],'backend_mean_ms':b['mean'],'backend_p95_ms':b['p95'],
            'peak_rss_mib':p['peak_rss_mib'],'marginalization_cache_written':p['marginalization_cache_written']})
    for filename,rows in [('viewer_metrics.csv',metric_rows),('viewer_performance_metrics.csv',performance_rows)]:
        with (output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'viewer_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (output/'viewer_data.json').write_text(json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False))
    return data


def plot_comparison_views(output,data):
    """Standalone figures use exactly the same common frame as the web view."""
    import os
    os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    common=data['comparison']['imu0'];reference=np.array(common['reference'])
    labels={case:{'original_2cam':'Original / 2 cameras','monado_2cam':'Monado / 2 cameras','monado_4cam':'Monado / 4 cameras'}.get(case,case) for case in data['maps']}
    names=list(data['maps']);colours=data['comparison']['colours']
    trajectories={case:np.array(common['online'][case]) for case in names}
    cloud=np.vstack([reference,*trajectories.values()]);lo,hi=cloud.min(axis=0),cloud.max(axis=0)
    center=(lo+hi)/2;radius=max(.1,max(hi-lo))*.55
    fig=plt.figure(figsize=(13,10),constrained_layout=True)
    axes=[fig.add_subplot(221,projection='3d'),*[fig.add_subplot(i) for i in [222,223,224]]]
    axes[0].plot(*reference.T,color=colours['reference'],ls='--',lw=1.8,label='NOKOV / IMU reference')
    for case,p in trajectories.items():axes[0].plot(*p.T,color=colours[case],lw=1.1,label=labels[case])
    axes[0].set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='IMU0 X [m]',ylabel='Y [m]',zlabel='Z [m]',title='3D / shared initial reference IMU frame')
    axes[0].set_box_aspect((1,1,1));axes[0].legend(fontsize=8)
    for ax,(a,b) in zip(axes[1:],[(0,1),(0,2),(1,2)]):
        ax.plot(reference[:,a],reference[:,b],color=colours['reference'],ls='--',lw=1.6)
        for case,p in trajectories.items():ax.plot(p[:,a],p[:,b],color=colours[case],lw=1.1)
        ax.set(xlabel='XYZ'[a]+' [m]',ylabel='XYZ'[b]+' [m]',title='XYZ'[a]+'XYZ'[b]+' plane')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.25)
    fig.suptitle('VIO configurations / NOKOV reference / one fixed display transform')
    fig.savefig(output/'trajectories_comparison_3d_2d.png',dpi=160);plt.close(fig)
    if not all('rgb' in m for m in data['maps'].values()):return
    masked={};clouds=[]
    for case,m in data['maps'].items():
        points=np.array(m['imu0']['points']);mask=np.linalg.norm(points,axis=1)<=5
        masked[case]=(points,mask);clouds.extend([points[mask],reference])
    cloud=np.vstack(clouds);lo,hi=cloud.min(axis=0),cloud.max(axis=0);center=(lo+hi)/2;radius=max(.1,max(hi-lo))*.55
    fig=plt.figure(figsize=(16,6),constrained_layout=True)
    for i,(case,m) in enumerate(data['maps'].items()):
        ax=fig.add_subplot(1,3,i+1,projection='3d');points,mask=masked[case]
        ax.scatter(*points[mask].T,c=np.array(m['rgb'])[mask]/255.,s=3,alpha=.6)
        ax.plot(*reference.T,color=colours['reference'],ls='--',lw=1)
        for name,p in trajectories.items():ax.plot(*p.T,color=colours[name],lw=.9)
        ax.set(title=f'{labels[case]} map / {mask.sum()} points within 5 m',xlabel='IMU0 X [m]',ylabel='Y [m]',zlabel='Z [m]',
               xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-55)
    legend=[Line2D([0],[0],color=colours['reference'],ls='--',label='NOKOV reference')]+[Line2D([0],[0],color=colours[n],label=labels[n]) for n in names]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,-.02),ncol=4,frameon=False)
    fig.suptitle('RGB sparse maps / all trajectories in the shared first reference IMU frame')
    fig.savefig(output/'rgb_maps_imu0.png',dpi=160,bbox_inches='tight');plt.close(fig)


def refresh_map_view(output):
    page=Path(output)/'viewer.html'
    encoded=page.read_text().split('<script id="map-data" type="application/json">',1)[1].split('</script>',1)[0]
    write_map_view(output,json.loads(encoded))
