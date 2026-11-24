"""Replay marker export and single-hand eligibility on frozen observations using CPU."""
import csv
from collections import defaultdict
from pathlib import Path
import json
import cv2
import numpy as np

from .geometry import CameraCalibration, reference_to_camera, camera_label, reconstructed_poses, project, _all_points_inside
from .fingerprint import digest

def rows(path):
    with path.open() as stream:
        yield from map(json.loads,stream)

def arrays(path,*names):
    with np.load(path) as archive:
        return {k:archive[k] for k in (names or archive.files)}

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def prepare(group,out):
    paired=group/'paired_input'
    cams=[CameraCalibration.load(group/f'native/calibration/camera{i}.json') for i in range(4)]
    transforms=[reference_to_camera(cams[0],c) for c in cams]
    seq=arrays(paired/'sequence.npz')
    mano=seq['joints_camera0_m']
    glove=np.full(mano.shape,np.nan,np.float32)
    uv=np.full_like(seq['joints_2d_native'],np.nan)
    for s,h in enumerate(('left','right')):
        tr=arrays(group/'tracks'/f'{h}.npz','pair_indices','predicted24','marker_names','faces','attachment_faces','barycentric','attachment_local_offsets_m')
        wrist=arrays(group/'wrist'/f'{h}_outward.npz','pair_indices','joints')
        mo=arrays(group/'mounts'/f'{h}_mounts.npz')
        np.testing.assert_array_equal(tr['faces'][tr['attachment_faces'][1:]],mo['triangles'])
        np.testing.assert_array_equal(tr['barycentric'][1:],mo['barycentric'])
        np.testing.assert_array_equal(tr['attachment_local_offsets_m'][1:,2],mo['normal_offset_m'])
        t=tr['pair_indices']
        np.testing.assert_array_equal(t,wrist['pair_indices'])
        np.testing.assert_array_equal(np.flatnonzero(seq['pose_valid'][:,s]),t)
        # Project the physical hand in camera0. The stored sequence is float32.
        xyz=np.concatenate((wrist['joints'][:,:1],tr['predicted24'][:,4:]),1).astype(float)
        if s==0:xyz[:,:,0]*=-1
        glove[t,s]=xyz
        uv[t,:,s]=np.stack([project(xyz,rt,c,1) for c,rt in zip(cams,transforms)],1)
    seq.update(joints_3d_mano=mano,joints_2d_mano=seq['joints_2d_native'],joints_3d_glove=glove,joints_2d_glove=uv)
    np.savez_compressed(out/'sequence.npz',**seq)
    return {'valid_hand_poses':int(seq['pose_valid'].sum())}

def export(group,out,*,single_observer=None,signed_reference=True):
    paired=group/'paired_input';native=group/'native'
    seq=arrays(out/'sequence.npz','joints_2d_mano','joints_3d_glove','joints_2d_glove','isolated_pose','pass_surface','tip_relaxed','cuff_points_observed','pose_valid')
    cams=[CameraCalibration.load(native/f'calibration/camera{i}.json') for i in range(4)]
    rt=[reference_to_camera(cams[0],c) for c in cams]
    prev={(r['sync_index'],int(r['camera'][-1]),r['side']):r for r in rows(paired/'index.jsonl')}
    views=defaultdict(set)
    for r in rows(group/'native_views/index.jsonl'):
        views[r['sync_index'],int(r['camera'][-1])].add(r['side'])
    for r in rows(group/'native_views/rejected.jsonl'):
        if r['reason']=='both_hands_required':views[r['sync_index'],int(r['camera'][-1])].update(r['available_sides'])
    excluded=set(json.loads((group/'decisions/pose_screen.json').read_text())['excluded'])
    if single_observer is None:
        with (group/'decisions/photo.csv').open() as stream:
            blocked={(int(r['sync']),int(r['camera']),int(r['side'])):r['decision'] for r in csv.DictReader(stream)}
    tracks=[];poses=[]
    for h in ('left','right'):
        tr=arrays(group/'tracks'/f'{h}.npz','pair_indices','vertices','joints','betas','global_orient','hand_pose_pca','translation')
        tracks.append(tr);poses.append(reconstructed_poses(tr))
    with (native/'multiview_frames.csv').open() as stream:frames=list(csv.DictReader(stream))
    strict=('bbox','vertices','joints_3d','joints_2d','side','trans','K','mano','image_space','camera_model','distortion','pass_surface','isolated_pose','tip_relaxed','cuff_points_observed')
    if single_observer is None:
        for name in ('labels','labels_with_glove','images'):(out/name).mkdir(exist_ok=True)
    videos={};last={};n=image_n=0
    try:
        with ((out/'index.jsonl').open('w') if single_observer is None else open('/dev/null','w')) as records:
            for (t,c),sides in sorted(views.items()):
                qualified={}
                for s in sorted(sides):
                    if t in excluded or not seq['pass_surface'][t,s] or seq['isolated_pose'][t,s] or not seq['pose_valid'][t,s]:continue
                    old=prev.get((t,c,s));tr=tracks[s];rot,shift=rt[c];i=int(np.searchsorted(tr['pair_indices'],t))
                    if old:
                        pure=np.load(paired/old['label'],allow_pickle=True).item()
                    else:
                        if not _all_points_inside(seq['joints_2d_mano'][t,c,s],cams[c].image_size):continue
                        if (tr['vertices'][i]@(rot[2]*np.array([1 if s else -1,1,1]))+shift[2]).min()<=1e-8:continue
                        try:pure=camera_label(tr,i,poses[s],rt[c],cams[c],s)
                        except ValueError as exc:
                            if str(exc)!='No visible mesh in a source-qualified view':raise
                            continue
                    flags=dict(pass_surface=True,isolated_pose=False,tip_relaxed=bool(seq['tip_relaxed'][t,s]),cuff_points_observed=int(seq['cuff_points_observed'][t,s]))
                    pure.update(flags)
                    guv=seq['joints_2d_glove'][t,:,s]
                    gm=(seq['joints_3d_glove'][t,s]@rot.T+shift-pure['trans']).astype(np.float32)
                    rich=dict(pure,joints_3d_mano=pure['joints_3d'],joints_2d_mano=seq['joints_2d_mano'][t,c,s],joints_3d_glove=gm,joints_2d_glove=guv,joints_2d_glove_view=guv[c],glove_camera_index=c,joint_schema='mano_joints3d+glove_extra',approved_for_training=signed_reference and not bool(old))
                    if not signed_reference:rich['fit_method']='legacy'
                    qualified[s]=({k:pure[k] for k in strict},rich,old)
                single=len(qualified)==1
                if single_observer is not None:
                    if single:
                        side=next(iter(qualified))
                        single_observer(t,c,side,qualified[side][0],cams[c],rt[c])
                    continue
                if single and blocked[t,c,next(iter(qualified))]=='exclude':continue
                if not qualified:continue
                fr=frames[t];frame=int(fr[f'camera{c}_frame_index'])
                im=out/'images'/f'{image_n:06}.jpg'
                orig=next((v[2] for v in qualified.values() if v[2]),None)
                if orig:
                    if not im.exists():im.symlink_to((paired/orig['image']).resolve())
                else:
                    if c not in videos:
                        videos[c]=cv2.VideoCapture(str(native/f'cameras/camera{c}/video.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
                        last[c]=-1
                    cap=videos[c]
                    while last[c]+1<frame:
                        if not cap.grab():raise OSError(f'相机 {c} 缺帧 {frame}')
                        last[c]+=1
                    ok,img=cap.read();last[c]+=1
                    if not ok or last[c]!=frame or not cv2.imwrite(str(im),img,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError(im)
                for s,(pure,rich,_) in qualified.items():
                    label=f'labels/{n:06}.npy';extra=f'labels_with_glove/{n:06}.npy'
                    np.save(out/label,pure);np.save(out/extra,rich)
                    row=dict(index=n,source_group=group.name,sync_index=t,camera=f'camera{c}',side=s,single_hand=single,reference_timestamp_ns=int(fr['reference_timestamp_ns']),source_frame_index=frame,label=label,glove_label=extra,image=str(im.relative_to(out)),fit_method='legacy',approved_for_training=False)
                    records.write(json.dumps(row)+'\n');n+=1
                image_n+=1
    finally:
        for cap in videos.values():cap.release()
    return dict(labels=n,images=image_n)
