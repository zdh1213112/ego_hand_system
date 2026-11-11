#!/usr/bin/env python3
"""Stage 1: estimate and freeze sensor/world relations before full processing.

Uses this recording's IMU/NOKOV calibration interval plus a short VIO prefix.
World relations are bound to the exact recording, calibration, configuration,
binaries and VIO initialization; they are not universal device extrinsics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compare_basalt_ego4 import prepare_data, run_cases
from calibrate_nokov_ego4 import calibrate
from analyze_basalt_ego4_comparison import interpolate_reference, CASES
from fuse_basalt_hand_trajectory import load_basalt_trajectory
from ego_data.fixed_relations import SCHEMA, sha256, save_json
from visualize_ego_nokov_sync import DEFAULT_EGO, DEFAULT_NOKOV

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CACHE=ROOT/'output/basalt_comparison_20261104_195844'


def link_binaries(output,cache):
    pairs=[(output/'build_original',cache/'build_original'),
           (output/'basalt_monado_mapper_export',cache/'basalt_monado_mapper_export')]
    for destination,source in pairs:
        if not source.exists():
            raise FileNotFoundError(f'build cache missing: {source}')
        destination.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def main(args):
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'choose a new calibration directory: {output}')
    output.mkdir(parents=True)
    cache=args.build_cache.resolve()
    link_binaries(output,cache)
    args.session=args.ego_session;args.output=output;args.image_scale=.5
    args.max_frames=args.prefix_frames
    prepare_data(args)
    args.cases=list(CASES);args.repeats=1;args.save_marg=False;args.max_frames=0
    run_cases(args)
    for case in CASES:
        if json.loads((output/'runs'/case/'repeat_1/vio_timing.json').read_text())['returncode']!=0:
            raise RuntimeError(f'calibration-prefix VIO failed: {case}')
    calibrate(output,args.nokov.resolve(),args.calibration_seconds)
    info,sensors,origin,query=interpolate_reference(output)
    world={}
    for case in CASES:
        trajectory=output/'runs'/case/'repeat_1/trajectory.csv'
        poses=load_basalt_trajectory(trajectory)
        stamps=np.array(sorted(poses),dtype=np.int64);times=(stamps-origin)*1e-9
        if times[-1]<args.world_alignment_seconds:
            raise ValueError('prefix is too short for world registration')
        selected=times<=args.world_alignment_seconds
        matrices=np.array([poses[int(s)] for s in stamps])
        gp,gr=query(times[selected])
        r=(gr*Rotation.from_matrix(matrices[selected,:3,:3]).inv()).mean()
        t=np.mean(gp-r.apply(matrices[selected,:3,3]),axis=0)
        transform=np.eye(4);transform[:3,:3]=r.as_matrix();transform[:3,3]=t
        world[case]={'T_nokov_vio':transform.tolist(),'fit_interval_s':[0,args.world_alignment_seconds],
                     'fit_poses':int(selected.sum()),'source_trajectory':str(trajectory),
                     'initialization':{'timestamp_ns':str(stamps[0]),'T_vio_imu':matrices[0].tolist()}}
    paths=[Path(x['path']) for x in info['files']]
    paths += [args.nokov.resolve()/'nokov_rigid_bodies.csv' if (args.nokov.resolve()/'nokov_rigid_bodies.csv').exists() else args.nokov.resolve()/'nokov/nokov_rigid_bodies.csv',
              cache/'build_original/basalt_vio',ROOT/'third_party/basalt_monado/build/basalt_vio',
              cache/'build_original/basalt_mapper_export',cache/'basalt_monado_mapper_export']
    relations={'schema':SCHEMA,'ego_session':str(args.ego_session.resolve()),'nokov_directory':sensors['reference_directory'],
               'rigid_body':sensors['rigid_body'],'T_marker_imu':sensors['T_marker_imu'],
               'nokov_to_imu_offset_s':sensors['mocap_to_imu_offset_s'],'camera_time_policy':info['time_sync_policy'],
               'sensor_calibration':sensors,'sensor_calibration_interval_s':[.5,args.calibration_seconds],
               'world_relations':world,'image_scale':args.image_scale,'threads':args.threads,'build_cache':str(cache),
               'first_image_timestamp_ns':str(info['first_timestamp_ns']),
               'config_sha256':{name:sha256(output/f'config_{name}.json') for name in ['original','monado']},
               'bindings':{'files':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in paths]},
               'scope':'Frozen relations for this recording and identical VIO initialization; prefix-based calibration, not an independent installation survey',
               'full_run_rule':'Load only. Do not refit time, installation or world alignment in stage 2.'}
    save_json(output/'relations.json',relations)
    print(json.dumps({'relations':str(output/'relations.json'),'nokov_offset_s':relations['nokov_to_imu_offset_s'],'world_relations':list(world)},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ego-session',type=Path,default=DEFAULT_EGO)
    p.add_argument('--nokov',type=Path,default=DEFAULT_NOKOV)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--prefix-frames',type=int,default=90)
    p.add_argument('--calibration-seconds',type=float,default=20.)
    p.add_argument('--world-alignment-seconds',type=float,default=2.)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    if a.world_alignment_seconds<=0 or a.calibration_seconds<2 or a.prefix_frames<3:
        p.error('invalid calibration interval/prefix')
    main(a)
