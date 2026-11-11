#!/usr/bin/env python3
"""Build an offline EGO/NOKOV synchronization viewer from original recordings.

No VIO trajectory or new temporal fit is used. All matching uses corrected
capture timestamps, integer nanoseconds, bounded nearest neighbours and explicit
missing-data flags. The preview preserves the selected image capture intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from ego_data.time_sync import (camera_offset_policy, require_zero_camera_offset,
                                nearest_indices, sampling_summary)
from prepare_basalt_dataset import (build_basalt_calibration, calibrate_imu_sample,
                                    read_imu_samples, read_timestamps)
from ego_data.calibration import CameraCalibration
from camera_models.kb import project as project_kb
from prepare_basalt_video_dataset import synchronize
from run_orbbec_headring_vio import load_vendor_yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EGO = Path('/home/lff/xense_project/orbbec_ego4/EGOViewer_v2.0.16_202610122355_77764c9_linux_x86_64/data/recordings/EGO-4B_AZLCA610091_20261104_195844')
DEFAULT_NOKOV = Path('/home/lff/xense_project/NOKOV/nokov-rigid-capture/sessions/session_20261104_195852_461949/nokov')
STREAMS = ('left', 'right', 'right_side', 'left_side')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def hand_projection_config(camera_yaml, imu_yaml, marker_calibration):
    """Use the existing calibrated chain; never fit image overlays to pixels.

    T_A_B maps B-frame coordinates into A. All translations below are metres.
    p_C = inverse(T_I_C) inverse(T_M_I) inverse(T_W_M) p_W.
    """
    camera_yaml = dict(camera_yaml)
    camera_yaml['cameras'] = sorted(camera_yaml['cameras'], key=lambda c:int(c['id'].split('_')[-1]))
    _, factory = build_basalt_calibration(camera_yaml, imu_yaml, 1., True)
    t_m_i = np.asarray(marker_calibration['T_marker_imu'], dtype=float)
    cameras = []
    for camera, t_i_c in zip(camera_yaml['cameras'], factory['T_imu_cameras']):
        cameras.append({'id':camera['id'],'model':'KB4',
                        'image_size':[camera['image_width'],camera['image_height']],
                        'intrinsics':camera['intrinsics'],
                        'distortion':[camera['distortion'][f'k{i}'] for i in range(1,5)],
                        'T_camera_marker':(np.linalg.inv(t_i_c)@np.linalg.inv(t_m_i)).tolist()})
    return {'source':'NOKOV world-space hand points: measured markers or coordinate-checked recorded skeleton joints',
            'formula':'p_C = inverse(T_I_C) * inverse(T_M_I) * inverse(T_W_M) * p_W',
            'installation':'Existing session-estimated marker-to-IMU transform; not refitted to images',
            'visibility':'Finite points with camera Z>0 and projection inside native image; occlusion not tested',
            'cameras':cameras}


def project_world_markers(points, head_position, head_quaternion, camera):
    """Reference implementation used for geometry tests and overlay verification."""
    local = Rotation.from_quat(head_quaternion).inv().apply(np.asarray(points)-head_position)
    t_c_m = np.asarray(camera['T_camera_marker'])
    optical = local@t_c_m[:3,:3].T+t_c_m[:3,3]
    intr=camera['intrinsics']
    k=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    calib=CameraCalibration(camera['id'],camera['id'],'KB',tuple(camera['image_size']),k,np.array(camera['distortion']),np.eye(4))
    uv,valid=project_kb(calib,optical)
    width,height=camera['image_size']
    valid &= (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    return uv,valid,optical


def skeleton_coordinate_check(positions, hand_markers):
    """Check this recording's joint positions against same-frame world markers.

    This is a coordinate consistency gate, not an independent skeleton accuracy
    measurement or a statement about every NOKOV global/local SDK setting.
    """
    root_distance=np.linalg.norm(positions[:,0]-hand_markers[:,0],axis=1)
    good=root_distance[np.isfinite(root_distance)]
    nearest=[]
    for joint in range(positions.shape[1]):
        distances=np.linalg.norm(positions[:,joint,None,:]-hand_markers,axis=2)
        distances[~np.isfinite(distances)]=np.inf
        nearest.extend(np.min(distances,axis=1).tolist())
    nearest=np.array(nearest);nearest=nearest[np.isfinite(nearest)]
    enough=len(good)>=30 and len(nearest)>0
    return {'available':bool(enough and np.percentile(good,95)<.15 and np.percentile(nearest,95)<.12),
            'method':'Recorded joint positions interpreted in NOKOV world only after same-frame marker consistency check; no fitted transform',
            'root_frames_checked':len(good),
            'root_to_wrist_marker_median_m':float(np.median(good)) if len(good) else None,
            'root_to_wrist_marker_p95_m':float(np.percentile(good,95)) if len(good) else None,
            'joint_nearest_marker_median_m':float(np.median(nearest)) if len(nearest) else None,
            'joint_nearest_marker_p95_m':float(np.percentile(nearest,95)) if len(nearest) else None,
            'limitation':'Marker agreement checks coordinate convention; skeleton joint accuracy and image reprojection accuracy are not independently certified.'}


def load_hand_skeleton(nokov, frames, hand_markers):
    descriptors=json.loads((nokov/'asset_descriptions.json').read_text())
    descriptor=next((s for s in descriptors.get('skeletons',{}).values() if s['name']=='left_hand'),None)
    path=nokov/'nokov_skeleton_segments.csv'
    if descriptor is None or not path.exists():
        return {'available':False,'reason':'No recorded left_hand skeleton'},None
    ids=sorted(int(k) for k in descriptor['segments'])
    index={j:i for i,j in enumerate(ids)};by_frame={str(n):i for i,n in enumerate(frames)}
    positions=np.full((len(frames),len(ids),3),np.nan)
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            i=by_frame.get(r['frame_no']);j=index.get(int(r['segment_id']))
            if i is not None and j is not None and r['skeleton_name']=='left_hand' and r['valid_numeric']=='1':
                positions[i,j]=[float(r[k])*.001 for k in ['x_mm','y_mm','z_mm']]
    check=skeleton_coordinate_check(positions,hand_markers)
    edges=[[index[int(parent)],index[int(child)]] for child,parent in descriptor['parent_ids'].items() if int(parent) in index and int(child) in index]
    return {**check,'ids':ids,'names':[descriptor['segments'][str(j)] for j in ids],'edges':edges},positions


def angular_velocity(times, rotations, half_window_s=.035):
    """Centred SO(3) derivative, expressed in the body frame at the centre.

    Noncausal ±35 ms derivative for offline checking, never extrapolated.
    """
    valid = (times >= times[0]+half_window_s) & (times <= times[-1]-half_window_s)
    centers = times[valid]
    slerp = Slerp(times, rotations)
    minus, plus, center = slerp(centers-half_window_s), slerp(centers+half_window_s), slerp(centers)
    omega = (center.inv()*minus).apply((minus.inv()*plus).as_rotvec()) / (2*half_window_s)
    return np.flatnonzero(valid), omega


def motion_comparison(query_ns, gyro, motion_ns, omega, offset_ns, max_delta_ns, validation_start_ns):
    matches = nearest_indices(query_ns, motion_ns, max_delta_ns, sample_offset_ns=offset_ns)
    rows, metrics = [], {}
    for t, g, j in zip(query_ns, gyro, matches):
        norm_error = float(np.linalg.norm(g)-np.linalg.norm(omega[j])) if j is not None else None
        rows.append((t, j, norm_error, float(np.linalg.norm(g-omega[j])) if j is not None else None))
    for label, predicate in [('all', lambda t: True), ('held_out_after_20s', lambda t: t >= validation_start_ns)]:
        valid = [row for row in rows if row[1] is not None and predicate(row[0])]
        metrics[label] = {'matched_queries': len(valid),
                          'speed_norm_rmse_rad_s': float(np.sqrt(np.mean([r[2]**2 for r in valid]))) if valid else None,
                          'vector_rmse_rad_s': float(np.sqrt(np.mean([r[3]**2 for r in valid]))) if valid else None}
    return rows, metrics


def write_preview(videos, groups, timestamps, output, width):
    """Decode source indices, then encode VFR using original camera-0 intervals."""
    cv2.setNumThreads(2)
    captures = [cv2.VideoCapture(str(p), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]) for p in videos]
    if not all(c.isOpened() for c in captures):
        raise RuntimeError('cannot open every EGO video')
    height = round(width*1200/1920)//2*2
    next_indices = [0]*len(captures)
    try:
        with tempfile.TemporaryDirectory(prefix='ego-sync-frames-') as temporary:
            temp = Path(temporary)
            entries = ['ffconcat version 1.0\n']
            for group_index, indices in enumerate(groups):
                images = []
                for camera, target in enumerate(indices):
                    image = None
                    while next_indices[camera] <= target:
                        ok, image = captures[camera].read()
                        if not ok:
                            raise RuntimeError(f'camera {camera} ended before source frame {target}')
                        next_indices[camera] += 1
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(image, (0,0), (width,27), (18,24,32), -1)
                    cv2.putText(image, f'cam{camera} / {STREAMS[camera]} / source #{target}', (8,19), cv2.FONT_HERSHEY_SIMPLEX, .48, (240,240,240), 1, cv2.LINE_AA)
                    images.append(image)
                mosaic = np.vstack((np.hstack(images[:2]), np.hstack(images[2:])))
                filename = f'{group_index:06d}.jpg'
                if not cv2.imwrite(str(temp/filename), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise RuntimeError('could not write preview frame')
                duration = (timestamps[group_index+1]-timestamps[group_index])*1e-9 if group_index+1<len(timestamps) else float(np.median(np.diff(timestamps)))*1e-9
                entries.append(f"file '{filename}'\noption framerate 1000000\nduration {duration:.9f}\n")
                if (group_index+1)%250 == 0:
                    print(f'preview {group_index+1}/{len(groups)}', flush=True)
            playlist = temp/'frames.ffconcat'
            playlist.write_text(''.join(entries))
            command = ['ffmpeg','-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(playlist),
                       '-fps_mode','vfr','-enc_time_base','1:1000000','-video_track_timescale','1000000',
                       '-c:v','libx264','-level:v','4.1','-preset','fast','-crf','23','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'ego_four_camera_preview.mp4')]
            (output/'preview_command.json').write_text(json.dumps(command, indent=2)+'\n')
            with (output/'preview_encode.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    finally:
        for capture in captures:
            capture.release()
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(output/'ego_four_camera_preview.mp4')], text=True))
    playback = np.array([float(r['best_effort_timestamp_time']) for r in probe['frames']])
    expected = (np.array(timestamps, dtype=np.int64)-timestamps[0])*1e-9
    if len(playback) != len(expected) or np.max(abs(playback-expected)) > .000002:
        raise RuntimeError('preview presentation timestamps differ from selected capture intervals')
    return playback.tolist()


def build(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'choose an empty output directory: {output}')
    if args.max_frames and args.max_frames < 2:
        raise ValueError('max-frames must be zero or at least two')
    if args.preview_camera_width < 160 or args.preview_camera_width % 2:
        raise ValueError('preview-camera-width must be even and at least 160')
    output.mkdir(parents=True, exist_ok=True)
    session = args.ego_session.resolve()
    nokov = args.nokov.resolve()
    if (nokov/'nokov').is_dir():
        nokov = nokov/'nokov'
    meta_path = next(session.glob('meta_*.json'))
    meta = json.loads(meta_path.read_text())
    files = meta['files']
    calibration = json.loads(args.calibration.read_text())
    if Path(calibration['reference_directory']).resolve() != nokov:
        raise ValueError('NOKOV calibration belongs to a different session')
    imu_yaml = load_vendor_yaml(session/files['calibration_imu'])
    projection=hand_projection_config(load_vendor_yaml(session/files['calibration_camera']),imu_yaml,calibration)
    policy = camera_offset_policy(args, imu_yaml)
    nokov_offset = round(calibration['mocap_to_imu_offset_s']*1e9) if args.nokov_offset_ns is None else args.nokov_offset_ns
    print('camera policy:', json.dumps(policy), flush=True)
    samples = read_imu_samples(session/files['imu'])
    imu_ns = [int(row[0])*1000 for row in samples]
    origin = imu_ns[0]
    calibrated = [calibrate_imu_sample(a,g,imu_yaml['imu0']) for _,a,g in samples]
    accel = np.array([a for a,_ in calibrated])
    gyro = np.array([g for _,g in calibrated])
    R_m_i = Rotation.from_matrix(calibration['R_marker_imu'])
    gyro_marker = R_m_i.apply(gyro)+np.array(calibration['estimated_gyro_bias_marker_rad_s'])
    gyro_smooth = savgol_filter(gyro_marker, 71, 3, axis=0)
    bodies = [r for r in read_csv(nokov/'nokov_rigid_bodies.csv') if r['rigid_body_name']==calibration['rigid_body'] and r['valid_numeric']=='1']
    raw_ns = [int(r['device_timestamp_raw'])*1_000_000 for r in bodies]
    if not 10_000_000 <= np.median(np.diff(raw_ns)) <= 12_000_000:
        raise ValueError('raw NOKOV steps do not support the expected millisecond unit')
    positions = np.array([[float(r[k])*.001 for k in ['x_mm','y_mm','z_mm']] for r in bodies])
    quat = np.array([[float(r[k]) for k in ['qx','qy','qz','qw']] for r in bodies])
    rotations = Rotation.from_quat(quat)
    time_n = (np.array(raw_ns, dtype=np.int64)-origin)*1e-9
    motion_indices, omega = angular_velocity(time_n, rotations)
    motion_ns = [raw_ns[j] for j in motion_indices]
    prior_input = args.calibration.parent/'input_validation.json'
    validation_origin = int(json.loads(prior_input.read_text())['first_timestamp_ns']) if prior_input.exists() else origin
    validation_start_ns = validation_origin + round(calibration['validation_interval_imu_s'][0]*1e9)
    # Retain the centre timestamp as its original integer; no floating epoch conversion.
    queries = imu_ns[::10]
    query_gyro = gyro_smooth[::10]
    comparisons, statistics = {}, {}
    for label, offset in [('before',0),('after',nokov_offset)]:
        comparisons[label], statistics[label] = motion_comparison(queries, query_gyro, motion_ns, omega, offset, args.max_nokov_delta_us*1000, validation_start_ns)
    streams = [[int(t)*1000 for t in read_timestamps(session/files[f'video_{name}_pts'])] for name in STREAMS]
    groups = [g for g in synchronize(streams, args.max_camera_delta_us*1000)
              if imu_ns[0] <= streams[0][g[0]] <= imu_ns[-1]]
    if args.max_frames:
        groups = groups[:args.max_frames]
    if len(groups)<2:
        raise ValueError('insufficient synchronized images within IMU coverage')
    camera_ns = [streams[0][g[0]] for g in groups]
    selected_ns = camera_ns
    imatch = nearest_indices(selected_ns, imu_ns, args.max_imu_delta_us*1000)
    nmatch = nearest_indices(selected_ns, raw_ns, args.max_nokov_delta_us*1000, sample_offset_ns=nokov_offset)
    nraw_match = nearest_indices(selected_ns, raw_ns, args.max_nokov_delta_us*1000)
    descriptions = json.loads((nokov/'asset_descriptions.json').read_text())
    marker_names = descriptions['marker_sets'].get('left_hand', [])
    hand = np.full((len(bodies), len(marker_names),3), np.nan)
    head_markers = np.full((len(bodies), 5,3), np.nan)
    body_lookup = {r['frame_no']:i for i,r in enumerate(bodies)}
    for filename, target, field, expected in [('nokov_markers.csv',hand,'markerset_name','left_hand'),('nokov_rigid_body_markers.csv',head_markers,'rigid_body_name',calibration['rigid_body'])]:
        for row in read_csv(nokov/filename):
            i = body_lookup.get(row['frame_no'])
            j = int(row['marker_index'])
            if i is not None and row[field]==expected and row['valid']=='1' and 0<=j<target.shape[1]:
                target[i,j] = [float(row[k])*.001 for k in ['x_mm','y_mm','z_mm']]
    skeleton,skeleton_positions=load_hand_skeleton(nokov,[r['frame_no'] for r in bodies],hand)
    # These are measured marker chains, not reconstructed skeleton joints.
    hand_edges = [[0,1],[0,2],[1,2],[0,3]]
    for start in [4,8,12,16,20]:
        hand_edges += [[0,start]]+[[j,j+1] for j in range(start,start+3)]
    with (output/'synchronized_frames.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['preview_index','camera0_source_index','camera0_raw_ns','camera0_on_imu_ns','imu_index','imu_raw_ns','imu_minus_camera_ns','nokov_frame_no','nokov_raw_ms','nokov_on_imu_ns','nokov_minus_camera_ns','uncorrected_nokov_frame_no','valid_hand_markers']+[f'cam{i}_raw_ns' for i in range(1,4)]+['imu_ax_m_s2','imu_ay_m_s2','imu_az_m_s2','imu_wx_rad_s','imu_wy_rad_s','imu_wz_rad_s','nokov_x_m','nokov_y_m','nokov_z_m','nokov_qx','nokov_qy','nokov_qz','nokov_qw'])
        for i,(g,t,im,nm,nr) in enumerate(zip(groups,selected_ns,imatch,nmatch,nraw_match)):
            writer.writerow([i,g[0],camera_ns[i],t,im if im is not None else '',imu_ns[im] if im is not None else '',imu_ns[im]-t if im is not None else '',bodies[nm]['frame_no'] if nm is not None else '',raw_ns[nm]//1_000_000 if nm is not None else '',raw_ns[nm]+nokov_offset if nm is not None else '',raw_ns[nm]+nokov_offset-t if nm is not None else '',bodies[nr]['frame_no'] if nr is not None else '',int(np.isfinite(hand[nm,:,0]).sum()) if nm is not None else 0]+[streams[c][g[c]] for c in range(1,4)]+(list(accel[im])+list(gyro[im]) if im is not None else ['']*6)+(list(positions[nm])+list(quat[nm]) if nm is not None else ['']*7))
    with (output/'motion_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['imu_query_ns','time_s','imu_marker_wx','imu_marker_wy','imu_marker_wz','before_nokov_raw_ns','after_nokov_raw_ns','before_norm_error_rad_s','after_norm_error_rad_s','before_vector_error_rad_s','after_vector_error_rad_s'])
        for i,t in enumerate(queries):
            b,a=comparisons['before'][i],comparisons['after'][i]
            writer.writerow([t,(t-origin)*1e-9,*query_gyro[i],motion_ns[b[1]] if b[1] is not None else '',motion_ns[a[1]] if a[1] is not None else '',b[2],a[2],b[3],a[3]])
    summary={'ego_session':str(session),'nokov_directory':str(nokov),'calibration_file':str(args.calibration.resolve()),
             'calibration_sha256':hashlib.sha256(args.calibration.read_bytes()).hexdigest(),'origin_imu_ns':str(origin),
             'camera_time_policy':policy,'nokov_offset_ns':nokov_offset,'nokov_offset_source':'session gyro/head fit' if args.nokov_offset_ns is None else 'explicit override',
             'frames':len(groups),'imu':sampling_summary(imu_ns),'nokov':sampling_summary(raw_ns),
             'matched_image_imu':sum(j is not None for j in imatch),'matched_image_nokov':sum(j is not None for j in nmatch),
             'before_after_motion':statistics,'motion_method':'71-sample cubic Savitzky-Golay IMU smoothing, centred ±35 ms SO(3) NOKOV derivative; 100 Hz queries, bounded nearest neighbours',
             'validation_start_s':(validation_start_ns-origin)*1e-9,
             'coordinate_method':'IMU gyro calibrated once, rotated into marker body frame using existing R_marker_imu and fitted constant bias; positions in recorded NOKOV metres',
             'validation':'Existing first-20s calibration reused without refitting. >=20s is held out for these motion checks.',
             'projection':projection,
             'skeleton_projection':skeleton,
             'limits':['Nearest timestamp distance alone does not certify physical synchronization.', 'Camera residual is unverified; +1.801936 ms is an explicit experiment only.', 'Optional skeleton overlay requires same-frame world-marker coordinate agreement.', 'Projection uses existing estimated installation extrinsics; no image-based spatial refit or pixel accuracy claim.'],
             'thresholds_us':{'camera':args.max_camera_delta_us,'imu':args.max_imu_delta_us,'nokov':args.max_nokov_delta_us}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    playback = (np.array(camera_ns,dtype=np.int64)-camera_ns[0])*1e-9
    if not args.skip_video:
        print('encoding four-camera preview with capture intervals',flush=True)
        playback=write_preview([session/files[f'video_{n}'] for n in STREAMS],groups,camera_ns,output,args.preview_camera_width)
    def rounded(array, digits=5):
        array=np.asarray(array)
        return [rounded(row,digits) for row in array] if array.ndim>1 else [round(float(v),digits) if np.isfinite(v) else None for v in array]
    payload={'summary':summary,'video':None if args.skip_video else 'ego_four_camera_preview.mp4',
             'playback':list(playback),'camera_t':[(t-origin)*1e-9 for t in camera_ns],
             'camera_raw_ns':[str(t) for t in camera_ns],'camera_source':[list(g) for g in groups],
             'camera_raw_ns_all':[[str(streams[c][g[c]]) for c in range(4)] for g in groups],
             'camera_t_all':[[(streams[c][g[c]]-origin)*1e-9 for c in range(4)] for g in groups],
             'projection':projection,
             'skeleton':{**skeleton,'positions':rounded(skeleton_positions) if skeleton.get('available') else None},
             'imu_t':[(t-origin)*1e-9 for t in imu_ns],'imu_raw_ns':[str(t) for t in imu_ns],
             'curve_t':[(t-origin)*1e-9 for t in queries],'gyro':rounded(query_gyro),
             'motion_t':[(t-origin)*1e-9 for t in motion_ns],'omega':rounded(omega),
             'nokov_t':time_n.tolist(),'nokov_raw_ns':[str(t) for t in raw_ns],
             'nokov_frames':[r['frame_no'] for r in bodies],
             'head':rounded(positions),'quaternion':rounded(quat,8),'head_markers':rounded(head_markers),
             'hand':rounded(hand),'hand_names':marker_names,'hand_edges':hand_edges}
    template=(ROOT/'scripts/templates/ego_nokov_sync.html').read_text()
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    (output/'viewer.html').write_text(template.replace('__SYNC_DATA__',data))
    plot_motion(output,payload,statistics,nokov_offset)
    print(json.dumps({'viewer':str(output/'viewer.html'),'frames':len(groups),'motion':statistics},indent=2),flush=True)
    return summary


def refresh_viewer(output):
    """Update embedded projection metadata and UI while retaining existing video."""
    page=output/'viewer.html'
    data=json.loads(page.read_text().split('<script id="sync-data" type="application/json">',1)[1].split('</script>',1)[0])
    summary=data['summary']
    require_zero_camera_offset(summary, str(output))
    summary['camera_time_policy']=camera_offset_policy()
    session=Path(summary['ego_session'])
    marker_file=Path(summary['calibration_file'])
    if hashlib.sha256(marker_file.read_bytes()).hexdigest()!=summary['calibration_sha256']:
        raise ValueError('existing session calibration changed; regenerate in a new directory')
    meta=json.loads(next(session.glob('meta_*.json')).read_text())
    files=meta['files']
    projection=hand_projection_config(load_vendor_yaml(session/files['calibration_camera']),
                                      load_vendor_yaml(session/files['calibration_imu']),json.loads(marker_file.read_text()))
    streams=[[int(t)*1000 for t in read_timestamps(session/files[f'video_{s}_pts'])] for s in STREAMS]
    origin=int(summary['origin_imu_ns'])
    data['camera_t_all']=[[(streams[c][g[c]]-origin)*1e-9 for c in range(4)] for g in data['camera_source']]
    data['camera_raw_ns_all']=[[str(streams[c][g[c]]) for c in range(4)] for g in data['camera_source']]
    data['projection']=summary['projection']=projection
    skeleton,positions=load_hand_skeleton(Path(summary['nokov_directory']),data['nokov_frames'],np.asarray(data['hand'],dtype=float))
    data['skeleton']=dict(skeleton)
    if skeleton.get('available'):
        rounded=np.round(positions,5)
        data['skeleton']['positions']=[[[float(v) if np.isfinite(v) else None for v in point] for point in row] for row in rounded]
    summary['skeleton_projection']=skeleton
    summary['limits']=[s for s in summary['limits'] if s not in ['No image projection or new spatial calibration is asserted.','Hand display uses measured markers, not skeleton segments with uncertain global/local setting.']]
    if 'Optional skeleton overlay requires same-frame world-marker coordinate agreement.' not in summary['limits']:
        summary['limits'].append('Optional skeleton overlay requires same-frame world-marker coordinate agreement.')
    note='Projection uses existing estimated installation extrinsics; no image-based spatial refit or pixel accuracy claim.'
    if note not in summary['limits']:
        summary['limits'].append(note)
    template=(ROOT/'scripts/templates/ego_nokov_sync.html').read_text()
    page.write_text(template.replace('__SYNC_DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'viewer':str(page),'projection_cameras':len(projection['cameras']),'video_reencoded':False},indent=2))


def plot_motion(output,data,metrics,offset_ns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qt=np.array(data['curve_t']); gyro=np.linalg.norm(np.array(data['gyro']),axis=1)
    mt=np.array(data['motion_t']); speed=np.linalg.norm(np.array(data['omega']),axis=1)
    fig,axes=plt.subplots(3,1,figsize=(14,9),constrained_layout=True)
    for ax,offset,title in [(axes[0],0,'Before correction'),(axes[1],offset_ns*1e-9,'After existing session correction')]:
        ax.plot(qt,gyro,color='#2563eb',lw=1,label='EGO gyro (smoothed)')
        ax.plot(mt+offset,speed,color='#059669',lw=1,label='NOKOV head angular speed')
        ax.axvspan(0,20,color='#f59e0b',alpha=.1,label='Calibration interval (existing fit)')
        ax.set(xlim=(0,qt[-1]),ylabel='Angular speed [rad/s]',title=title)
        ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
    with (output/'motion_comparison.csv').open() as f:
        rows=list(csv.DictReader(f))
    for key,label,color in [('before','Before','#e45656'),('after','After','#059669')]:
        values=[float(r[f'{key}_norm_error_rad_s']) if r[f'{key}_norm_error_rad_s'] else np.nan for r in rows]
        axes[2].plot(qt,values,color=color,lw=.8,label=label)
    axes[2].axvline(20,color='#d97706',ls='--');axes[2].set(xlim=(0,qt[-1]),xlabel='IMU capture time from first sample [s]',ylabel='Speed-norm residual [rad/s]');axes[2].legend();axes[2].grid(alpha=.2)
    fig.suptitle(f'Existing NOKOV correction: +{offset_ns/1e6:.6f} ms | camera residual not certified',fontsize=12)
    fig.savefig(output/'motion_alignment.png',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ego-session',type=Path,default=DEFAULT_EGO)
    parser.add_argument('--nokov',type=Path,default=DEFAULT_NOKOV)
    parser.add_argument('--calibration',type=Path,default=ROOT/'output/basalt_comparison_20261104_195844/nokov_calibration.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nokov-offset-ns',type=int,help='Override the existing session offset; no automatic refit')
    parser.add_argument('--max-camera-delta-us',type=int,default=100)
    parser.add_argument('--max-imu-delta-us',type=int,default=2000)
    parser.add_argument('--max-nokov-delta-us',type=int,default=8000)
    parser.add_argument('--preview-camera-width',type=int,default=480)
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--skip-video',action='store_true',help='Generate data, plots and interactive frame inspection without encoding video')
    parser.add_argument('--refresh-viewer',action='store_true',help='Refresh an existing page and projection calibration; preserve preview video and data')
    args=parser.parse_args()
    if args.refresh_viewer:
        if args.nokov_offset_ns is not None:
            parser.error('--refresh-viewer preserves existing time corrections; use page controls or a new output directory for offset changes')
        refresh_viewer(args.output.resolve())
    else:
        build(args)


if __name__=='__main__':
    main()
