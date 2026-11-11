"""Frozen calibration contracts and evaluation; no fitting in this module."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from ego_data.time_sync import require_zero_camera_offset


SCHEMA='ego_nokov_fixed_relations_v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def check_transform(value,label):
    t=np.asarray(value,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-10):
        raise ValueError(f'invalid homogeneous transform: {label}')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
        raise ValueError(f'invalid rotation: {label}')
    return t


def transform_poses(poses,transform):
    """Apply one previously saved world transform to every pose, scale fixed."""
    t=check_transform(transform,'T_nokov_vio')
    return np.einsum('ij,njk->nik',t,np.asarray(poses))


def check_initialization(stamp,pose,binding):
    if int(stamp)!=int(binding['timestamp_ns']):
        raise ValueError('VIO initial timestamp differs from frozen calibration')
    expected=np.asarray(binding['T_vio_imu'])
    translation=float(np.linalg.norm(np.asarray(pose)[:3,3]-expected[:3,3]))
    angle=float((Rotation.from_matrix(expected[:3,:3]).inv()*Rotation.from_matrix(np.asarray(pose)[:3,:3])).magnitude())
    if translation>1e-6 or angle>1e-6:
        raise ValueError(f'VIO initialization differs from frozen world: {translation} m / {angle} rad; recalibrate instead of reusing this transform')
    return {'position_difference_m':translation,'rotation_difference_rad':angle,'passed':True}


def validate_bindings(relations):
    require_zero_camera_offset(relations,'frozen calibration')
    if relations['schema']!=SCHEMA:
        raise ValueError('unsupported fixed-relations schema')
    check_transform(relations['T_marker_imu'],'T_marker_imu')
    for case,entry in relations['world_relations'].items():
        check_transform(entry['T_nokov_vio'],case)
    for item in relations['bindings']['files']:
        if sha256(item['path'])!=item['sha256']:
            raise ValueError(f'frozen input or binary changed: {item["path"]}')


class FixedReference:
    def __init__(self,relations,origin_ns):
        # The saved clock mapping is the only clock mapping used here.
        directory=Path(relations['nokov_directory'])
        with (directory/'nokov_rigid_bodies.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=[r for r in csv.DictReader(f) if r['rigid_body_name']==relations['rigid_body'] and r['valid_numeric']=='1']
        raw_ns=np.array([int(r['device_timestamp_raw'])*1_000_000 for r in rows],dtype=np.int64)
        self.times=(raw_ns-np.int64(origin_ns))*1e-9+relations['nokov_to_imu_offset_s']
        if np.any(np.diff(self.times)<=0):
            raise ValueError('NOKOV timestamps must increase')
        self.position=np.array([[float(r[k])*.001 for k in ['x_mm','y_mm','z_mm']] for r in rows])
        q=np.array([[float(r[k]) for k in ['qx','qy','qz','qw']] for r in rows])
        self.slerp=Slerp(self.times,Rotation.from_quat(q))
        self.mount=check_transform(relations['T_marker_imu'],'T_marker_imu')

    def query(self,times):
        times=np.asarray(times)
        valid=(times>=self.times[0])&(times<=self.times[-1])
        positions=np.full((len(times),3),np.nan)
        quaternions=np.tile([0.,0,0,1],(len(times),1))
        if valid.any():
            r=self.slerp(times[valid])
            p=np.column_stack([np.interp(times[valid],self.times,self.position[:,k]) for k in range(3)])
            positions[valid]=p+r.apply(self.mount[:3,3])
            quaternions[valid]=(r*Rotation.from_matrix(self.mount[:3,:3])).as_quat()
        return positions,Rotation.from_quat(quaternions),valid


def pose_errors(poses,reference_position,reference_rotation,valid):
    position=np.full(len(poses),np.nan);angle=np.full(len(poses),np.nan)
    position[valid]=np.linalg.norm(poses[valid,:3,3]-reference_position[valid],axis=1)
    angle[valid]=np.rad2deg((reference_rotation[valid].inv()*Rotation.from_matrix(poses[valid,:3,:3])).magnitude())
    return position,angle


def error_summary(position,angle,mask):
    mask=np.asarray(mask)&np.isfinite(position)&np.isfinite(angle)
    p,a=position[mask],angle[mask]
    return {'poses':int(mask.sum()),'position_rmse_m':float(np.sqrt(np.mean(p*p))) if len(p) else None,
            'position_median_m':float(np.median(p)) if len(p) else None,
            'position_max_m':float(p.max()) if len(p) else None,
            'orientation_rmse_deg':float(np.sqrt(np.mean(a*a))) if len(a) else None}
