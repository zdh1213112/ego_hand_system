#!/usr/bin/env python3
"""Project the saved NOKOV-world MANO reconstruction onto original EGO videos.

Time and mounting are estimated from this recording's raw IMU and head rigid
body using the existing calibration pipeline. Camera timestamps stay unchanged.
An optional source-bound, fixed image-marker rig correction may be supplied.
No new MANO fitting is performed. Geometry is interpolated at camera time.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from calibrate_nokov_ego4 import calibrate
from ego_data.time_sync import camera_offset_policy, nearest_indices
from prepare_basalt_dataset import read_imu_samples, calibrate_imu_sample, read_timestamps
from run_orbbec_headring_vio import load_vendor_yaml
from visualize_ego_nokov_sync import hand_projection_config, project_world_markers, STREAMS
from visualize_nokov_mano import EDGES


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare_calibration(ego, nokov, files, output):
    output.mkdir(parents=True,exist_ok=True)
    calibration=output/'nokov_calibration.json'
    binding={str(path.resolve()):digest(path) for path in (
        ego/files['imu'],ego/files['calibration_imu'],ego/files['calibration_camera'],
        nokov/'nokov_rigid_bodies.csv',nokov/'nokov_rigid_body_markers.csv')}
    binding_path=output/'source_bindings.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('calibration sources changed: use a new output directory')
    if calibration.exists():
        result=json.loads(calibration.read_text())
        if Path(result['reference_directory']).resolve()!=nokov.resolve():
            raise ValueError('calibration belongs to another NOKOV recording')
    else:
        imu=load_vendor_yaml(ego/files['calibration_imu'])
        (output/'input_adapter').mkdir(exist_ok=True)
        with (output/'input_adapter/imu.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp_ns','ax','ay','az','wx','wy','wz'])
            for t,a,g in read_imu_samples(ego/files['imu']):
                a,g=calibrate_imu_sample(a,g,imu['imu0']);w.writerow([t*1000,*a,*g])
        origin=read_timestamps(ego/files['video_left_pts'])[0]*1000
        (output/'input_validation.json').write_text(json.dumps(dict(first_timestamp_ns=origin,
            time_sync_policy=camera_offset_policy(),ego_session=str(ego),
            source='Current raw IMU adapter; no VIO run')))
        calibrate(output,nokov,20.)
        result=json.loads(calibration.read_text())
    binding_path.write_text(json.dumps(binding,indent=2)+'\n')
    return result


def association(camera_times, motion_times, offset_ns, tolerance_ns=8_000_000):
    """No endpoint holding or extrapolation outside motion-capture coverage."""
    matched=nearest_indices(list(map(int,camera_times)),list(map(int,motion_times)),
                            tolerance_ns,sample_offset_ns=int(offset_ns))
    lo,hi=int(motion_times[0])+offset_ns,int(motion_times[-1])+offset_ns
    return [j if lo<=int(t)<=hi else None for t,j in zip(camera_times,matched)]


def project_hand(data, index, position, quaternion, camera):
    points=np.concatenate((data['vertices'][index],data['joints'][index],data['markers_world_m'][index]))
    uv,inside,optical=project_world_markers(points,position,quaternion,camera)
    return uv,inside,optical


def interpolation_brackets(camera_times, motion_times, offset_ns):
    """Bounded camera-time sampling, with integer epoch subtraction first."""
    origin=int(motion_times[0])
    times=(np.asarray(motion_times,dtype=np.int64)-origin)*1e-9
    query=(np.asarray(camera_times,dtype=np.int64)-origin-int(offset_ns))*1e-9
    covered=(query>=times[0])&(query<=times[-1])
    right=np.clip(np.searchsorted(times,query,side='right'),1,len(times)-1)
    left=right-1
    alpha=(query-times[left])/(times[right]-times[left])
    alpha[~covered]=np.nan
    return left,right,alpha,covered


def interpolate_hand(data,left,right,alpha):
    """Do not fill missing marker endpoints or extrapolate reconstruction."""
    if not np.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('camera sample outside measured coverage')
    return {key:(1-alpha)*data[key][left]+alpha*data[key][right]
            for key in ('vertices','joints','markers_world_m','predicted_markers')}


def project_interpolated_hand(sample,position,quaternion,camera):
    points=np.concatenate([sample[key] for key in ('vertices','joints','markers_world_m')])
    return project_world_markers(points,position,quaternion,camera)


def marker_residual_summary(sample):
    errors=np.linalg.norm(sample['predicted_markers']-sample['markers_world_m'],axis=-1)*1000
    errors=errors[np.isfinite(errors)]
    return (float(errors.mean()),float(errors.max())) if len(errors) else (None,None)


def validate_alignment_sources(config,reconstruction,calibration_source):
    info=config.get('image_alignment')
    if not info:
        raise ValueError('image alignment config must include provenance bindings')
    bindings=info['source_bindings']
    for required in (reconstruction,calibration_source):
        if str(Path(required).resolve()) not in bindings:
            raise ValueError('alignment does not bind the selected reconstruction/calibration')
    for path,expected in bindings.items():
        if digest(path)!=expected:
            raise ValueError(f'alignment source changed: {path}')
    if info.get('time_offset_change_ms',0)!=0:
        raise ValueError('renderer currently requires the original inertial clock offset')


def draw_overlay(image, projection, faces, show_markers=True):
    uv,inside,optical=projection
    vertices,joints,markers=uv[:778],uv[778:799],uv[799:]
    h,w=image.shape[:2]
    face_valid=np.isfinite(vertices[faces]).all(axis=(1,2)) & (optical[:778][faces,:,][...,2]>.01).all(-1)
    safe=np.nan_to_num(vertices,nan=-100000,posinf=100000,neginf=-100000)
    pixels=np.rint(np.clip(safe,-100000,100000)).astype(np.int32)
    tri=pixels[faces]
    face_valid &= (tri[:,:,0].max(1)>=0)&(tri[:,:,0].min(1)<w)&(tri[:,:,1].max(1)>=0)&(tri[:,:,1].min(1)<h)
    face_valid &= np.max(np.ptp(tri,axis=1),axis=1)<w*.5
    visible_faces=np.flatnonzero(face_valid)
    overlay=image.copy()
    if len(visible_faces):
        depth=optical[:778][faces[visible_faces],2].mean(-1)
        for k in visible_faces[np.argsort(-depth)]:
            cv2.fillConvexPoly(overlay,tri[k],(245,178,55),cv2.LINE_AA)
        cv2.addWeighted(overlay,.28,image,.72,0,dst=image)
        cv2.polylines(image,[tri[k] for k in visible_faces[::5]],True,(200,150,62),1,cv2.LINE_AA)
    jvalid=inside[778:799]
    jp=np.rint(np.nan_to_num(joints,nan=-10000)).astype(np.int32)
    colours=[(55,160,245),(70,210,245),(150,235,75),(245,170,80),(215,130,235)]
    for a,b in EDGES:
        if jvalid[a] and jvalid[b]:
            colour=colours[min(4,(b-1)//4)]
            cv2.line(image,tuple(jp[a]),tuple(jp[b]),(30,30,30),4,cv2.LINE_AA)
            cv2.line(image,tuple(jp[a]),tuple(jp[b]),colour,2,cv2.LINE_AA)
    for j in np.flatnonzero(jvalid):
        cv2.circle(image,tuple(jp[j]),4,(30,30,30),-1,cv2.LINE_AA)
        cv2.circle(image,tuple(jp[j]),2,(80,220,255),-1,cv2.LINE_AA)
    if show_markers:
        for point in markers[inside[799:]]:
            u=tuple(np.rint(point).astype(int));cv2.circle(image,u,4,(100,255,120),1,cv2.LINE_AA)
    return image


def banner(image,stream,seconds,index,frame_no,delta_ms,covered,visible,residual=(None,None),candidate=False):
    cv2.rectangle(image,(0,0),(image.shape[1],99 if candidate else 71),(23,28,33),-1)
    cv2.putText(image,f'{stream} | MANO right hand | {seconds:06.2f} s | video #{index}',(18,29),
                cv2.FONT_HERSHEY_SIMPLEX,.72,(235,239,241),2,cv2.LINE_AA)
    label=(f'NOKOV nearest #{frame_no} | timestamp match delta {delta_ms:+.2f} ms | {visible}/21 joints in view'
           if covered else 'NO NOKOV COVERAGE - original image retained')
    if covered and residual[0] is not None:
        label+=f' | marker fit mean/max {residual[0]:.2f}/{residual[1]:.2f} mm'
    cv2.putText(image,label,(18,58),cv2.FONT_HERSHEY_SIMPLEX,.61,
                (140,235,175) if covered else (80,190,255),1,cv2.LINE_AA)
    if candidate:
        cv2.putText(image,'CANDIDATE fixed rig correction | full-sequence image alignment NOT validated | no per-frame hand adjustment',
                    (18,86),cv2.FONT_HERSHEY_SIMPLEX,.59,(80,190,255),1,cv2.LINE_AA)
    return image


class Encoder:
    def __init__(self,path,size,fps):
        self.path=path
        self.temporary=path.with_name(path.stem+'.encoding.mp4')
        self.process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
            '-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(self.temporary)],stdin=subprocess.PIPE)
    def write(self,image):self.process.stdin.write(np.ascontiguousarray(image).tobytes())
    def close(self):
        self.process.stdin.close()
        if self.process.wait()!=0:raise RuntimeError(f'encoding failed: {self.path}')


def validate_encoded_video(path, expected_frames, expected_fps):
    """Publish only complete streams, including a decodable final frame."""
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,nb_frames,duration','-of','json',str(path)]))
    stream=probe['streams'][0]
    if (int(stream.get('nb_frames',0)) != expected_frames
            or (stream['width'],stream['height']) != (1920,1200)
            or abs(float(stream['duration'])-expected_frames/expected_fps)>1/expected_fps):
        raise RuntimeError(f'Incomplete or unexpected video stream: {path}')
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,expected_frames-1)
        ok,frame=cap.read()
        if not ok or frame.shape[:2] != (1200,1920):
            raise RuntimeError(f'Final video frame cannot be decoded: {path}')
    finally:
        cap.release()
    return dict(stream,tail_decodable=True,bytes=path.stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ego-session',type=Path,required=True);p.add_argument('--nokov',type=Path,required=True)
    p.add_argument('--mano',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--calibration-source',type=Path)
    p.add_argument('--projection-config',type=Path)
    p.add_argument('--geometry-sampling',choices=('interpolated','nearest'),default='interpolated',
                   help='Use nearest only to preserve legacy overlays when restyling them')
    args=p.parse_args();cv2.setNumThreads(2)
    ego,nokov,output=args.ego_session.resolve(),args.nokov.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads(next(ego.glob('meta_*.json')).read_text());files=meta['files']
    if args.calibration_source:
        cal=json.loads(args.calibration_source.read_text())
        if Path(cal['reference_directory']).resolve()!=nokov:
            raise ValueError('calibration belongs to another recording')
        (output/'calibration').mkdir(exist_ok=True)
        (output/'calibration/nokov_calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
    else:
        cal=prepare_calibration(ego,nokov,files,output/'calibration')
    config=hand_projection_config(load_vendor_yaml(ego/files['calibration_camera']),load_vendor_yaml(ego/files['calibration_imu']),cal)
    config['installation']='Current recording raw IMU/head-rigid calibration; fixed after first-20-second fit; no image-based alignment'
    if args.projection_config:
        if not args.calibration_source:
            raise ValueError('--projection-config requires --calibration-source for provenance')
        config=json.loads(args.projection_config.read_text())
        validate_alignment_sources(config,args.mano/'reconstruction.npz',args.calibration_source)
    (output/'projection_config.json').write_text(json.dumps(config,indent=2)+'\n')
    data=dict(np.load(args.mano/'reconstruction.npz'))
    with (nokov/'nokov_rigid_bodies.csv').open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['rigid_body_name']=='orbbec_ego4' and r['valid_numeric']=='1']
    frames=np.array([int(r['frame_no']) for r in rows]);assert np.array_equal(frames,data['frame_no'])
    motion_ns=np.array([int(r['device_timestamp_raw'])*1_000_000 for r in rows],dtype=np.int64)
    position=np.array([[float(r[f'{a}_mm'])*.001 for a in 'xyz'] for r in rows])
    quaternion=np.array([[float(r[f'q{a}']) for a in 'xyzw'] for r in rows])
    camera_ns=np.array([np.array(read_timestamps(ego/files[f'video_{s}_pts']),dtype=np.int64)*1000 for s in STREAMS])
    count=camera_ns.shape[1];spread=np.ptp(camera_ns,axis=0)
    if spread.max()>1_000_000:raise ValueError('camera rows differ by >1 ms; explicit video association required')
    offset_ns=round(cal['mocap_to_imu_offset_s']*1e9)
    matches=[association(times,motion_ns,offset_ns) for times in camera_ns]
    brackets=[interpolation_brackets(times,motion_ns,offset_ns) for times in camera_ns]
    origin=int(motion_ns[0]);motion_s=(motion_ns-origin)*1e-9
    pose_slerp=Slerp(motion_s,Rotation.from_quat(quaternion))
    def sample_projection(c,i):
        if args.geometry_sampling=='nearest':
            j=matches[c][i]
            if j is None:raise ValueError('outside NOKOV coverage')
            sample={key:data[key][j] for key in ('markers_world_m','predicted_markers')}
            return project_hand(data,j,position[j],quaternion[j],config['cameras'][c]),marker_residual_summary(sample)
        left,right,alpha,covered=brackets[c]
        if not covered[i]:raise ValueError('outside NOKOV coverage')
        a=alpha[i];sample=interpolate_hand(data,left[i],right[i],a)
        p=(1-a)*position[left[i]]+a*position[right[i]]
        q=pose_slerp((int(camera_ns[c,i])-origin-offset_ns)*1e-9).as_quat()
        return project_interpolated_hand(sample,p,q,config['cameras'][c]),marker_residual_summary(sample)
    indices=np.array([[i if i is not None else -1 for i in match] for match in matches],dtype=np.int32)
    match_rows=[];uv_joints=np.full((4,count,21,2),np.nan,np.float32);visible_joints=np.zeros((4,count,21),bool)
    for c in range(4):
        for i,j in enumerate(matches[c]):
            if j is not None:
                (uv,inside,_),_=sample_projection(c,i)
                uv_joints[c,i]=uv[778:799];visible_joints[c,i]=inside[778:799]
            match_rows.append([c,STREAMS[c],i,int(camera_ns[c,i]),int(frames[j]) if j is not None else '',
                int(motion_ns[j]+offset_ns-camera_ns[c,i]) if j is not None else '',int(visible_joints[c,i].sum())])
    with (output/'frame_associations.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['camera','stream','video_index','camera_timestamp_ns','nokov_frame_no','nokov_minus_camera_ns','visible_joints']);w.writerows(match_rows)
    np.savez_compressed(output/'projected_joints.npz',joints_px=uv_joints,visible=visible_joints,
                        video_timestamp_ns=camera_ns,nokov_indices=indices,joint_names=data['joint_names'])
    captures=[cv2.VideoCapture(str(ego/files[f'video_{s}']),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]) for s in STREAMS]
    for cap in captures:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=count:
            raise ValueError('video frame count does not agree with source timestamp CSV')
    fps=count/(captures[0].get(cv2.CAP_PROP_FRAME_COUNT)/captures[0].get(cv2.CAP_PROP_FPS))
    # The source MP4s use essentially constant 30 Hz; retain their frame order.
    cadence_ns=np.diff(camera_ns[0]);cadence_error=np.max(np.abs(cadence_ns-np.median(cadence_ns)))
    if cadence_error>1_000_000:raise ValueError('source has irregular capture cadence; a VFR encoder is required')
    selected=[46,120,180,600,900,1380,1500,1800,2000,2200]
    selected=[i for i in selected if i<count]
    encoders=[]
    if not args.preview_only:
        encoders=[Encoder(output/f'mano_overlay_cam{c}_{s}.mp4',(1920,1200),fps) for c,s in enumerate(STREAMS)]
        mosaic_encoder=Encoder(output/'mano_overlay_4views.mp4',(1920,1200),fps)
    for i in (selected if args.preview_only else range(count)):
        views=[]
        for c,(stream,cap) in enumerate(zip(STREAMS,captures)):
            if args.preview_only:cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ok,image=cap.read()
            if not ok:raise RuntimeError(f'decode failed at cam {c} frame {i}')
            j=matches[c][i];covered=j is not None
            residual=(None,None)
            if covered:
                projection,residual=sample_projection(c,i)
                image=draw_overlay(image,projection,data['faces'])
            seconds=(int(camera_ns[c,i])-int(camera_ns[0,0]))*1e-9
            delta=(int(motion_ns[j])+offset_ns-int(camera_ns[c,i]))*1e-6 if covered else 0
            image=banner(image,f'cam{c} / {stream}',seconds,i,int(frames[j]) if covered else 0,delta,covered,int(visible_joints[c,i].sum()),residual,args.projection_config is not None)
            if not args.preview_only:encoders[c].write(image)
            if i in selected and c==0:cv2.imwrite(str(output/f'overlay_cam0_frame{i:04d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94])
            views.append(cv2.resize(image,(960,600),interpolation=cv2.INTER_AREA))
        mosaic=np.vstack((np.hstack(views[:2]),np.hstack(views[2:])))
        if i in selected:cv2.imwrite(str(output/f'overlay_4views_frame{i:04d}.jpg'),mosaic,[cv2.IMWRITE_JPEG_QUALITY,92])
        if not args.preview_only:mosaic_encoder.write(mosaic)
        if i%300==0:print(f'overlay video {i}/{count}',flush=True)
    for cap in captures:cap.release()
    for enc in encoders:enc.close()
    if not args.preview_only:mosaic_encoder.close()
    encoded_videos=[]
    if not args.preview_only:
        all_encoders=encoders+[mosaic_encoder]
        for enc in all_encoders:
            encoded_videos.append(dict(file=enc.path.name,**validate_encoded_video(enc.temporary,count,fps)))
        for enc in all_encoders:
            enc.temporary.replace(enc.path)
    covered=np.flatnonzero(indices[0]>=0)
    summary=dict(ego_session=str(ego),mano_reconstruction=str((args.mano/'reconstruction.npz').resolve()),
        source_mano_sha256=digest(args.mano/'reconstruction.npz'),frames=count,fps=fps,
        camera_span_s=float((camera_ns[0,-1]-camera_ns[0,0])*1e-9),
        matched_frames_by_camera=[int((i>=0).sum()) for i in indices],
        covered_video_interval_s=[float((camera_ns[0,i]-camera_ns[0,0])*1e-9) for i in [covered[0],covered[-1]]],
        unmatched_frames=int((indices[0]<0).sum()),
        camera_timestamp_spread_max_us=float(spread.max()/1000),source_interval_deviation_max_us=float(cadence_error/1000),
        nokov_to_camera_offset_ms=cal['mocap_to_imu_offset_s']*1000,association_tolerance_ms=8,
        camera_imu_offset_ms=0,projection_model='Original calibrated KB4 fisheye intrinsics; native 1920x1200 images',
        projection_chain=('p_C = T_camera_marker_old @ fixed_correction @ inverse(T_W_M(t)) @ p_W'
                          if args.projection_config else 'p_C = inverse(T_I_C) inverse(T_M_I) inverse(T_W_M(t)) p_W'),
        content='Blue semi-transparent MANO surface, sampled mesh edges, 21-joint skeleton, measured marker projection',
        mesh_style=dict(surface_rgb='#37b2f5',wire_rgb='#3e96c8',surface_opacity=.28),
        source_mano_modified=False,preview_only=args.preview_only,
        geometry_sampling=('Linear measured/MANO positions and head translation; SLERP head rotation at camera timestamp; no extrapolation'
                           if args.geometry_sampling=='interpolated' else 'Nearest measured NOKOV sample within 8 ms; no extrapolation'),
        timestamp_match_delta_semantics='Nearest NOKOV timestamp minus camera timestamp after constant offset; not synchronization accuracy',
        image_alignment_validation='alignment_validation.json' if args.projection_config else None,
        projection_config_sha256=digest(output/'projection_config.json'),
        limitations=['Mount and NOKOV clock offset estimated from this recording, not independently surveyed.',
            'No image keypoint based alignment or independent pixel reprojection accuracy was measured.',
            'Visibility checks positive depth and image bounds; real-object occlusion is not estimated.',
            'Frames outside NOKOV coverage keep the original camera image without extrapolation.'])
    if args.projection_config:
        summary['limitations']=config['image_alignment']['limitations']+[
            'Candidate fixed rig improves some samples but failed held-out frames remain; see alignment_validation.json.',
            'No per-frame image alignment, camera-specific mount fitting, or MANO refitting was used.',
            'Visibility checks depth/bounds only; object occlusion is not modeled.',
            'Outside motion coverage the original image is retained, with status banner; no extrapolation.']
    (output/'overlay_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not args.preview_only:
        validation=dict(encoded_videos=encoded_videos,projected_joints_shape=list(uv_joints.shape),
                        association_rows=len(match_rows),matched_camera_frames=int((indices>=0).sum()),
                        missing_tail_frames_per_camera=int((indices[0]<0).sum()),no_extrapolation=True,
                        source_mano_sha256=summary['source_mano_sha256'],
                        projection_config_sha256=summary['projection_config_sha256'],
                        geometry_sampling=summary['geometry_sampling'],
                        image_alignment_accepted=False if args.projection_config else None)
        (output/'overlay_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
