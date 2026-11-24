# Source work/nokov_label_training_20261118/cohort.py, SHA-256 c1e56be6d717052cab4d6de61e688221d97843592ab64f299e48c45644b5e4e7

import case_settings as case

import json

import os

import subprocess

import sys

import numpy as np

HERE = case.stage('nokov_label_training_20261118')

ROOT = case.project

EGO = ROOT / 'ego_hand_system'

CODE = EGO / 'scripts/joint_refit'

SHARED = ('betas','attachment_faces','barycentric','attachment_local_offsets_m',
          'normal_offsets_m','wrist_local','root_anchor_local_m')

HANDS = ('left','right')

OPT = {case.recording: case.optimized}

def write_json(path, data):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

def run(log, args):
    with log.open('w') as stream:
        subprocess.run([sys.executable,'-Bu', *map(str,args)], stdout=stream, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), check=True)

def prepare(group):
    from prepare import prepare_recording
    opt = OPT[group]
    dest = HERE/group
    dest.mkdir(parents=True, exist_ok=True)
    calibration = dest/'calibration'
    if not (calibration/'preparation.json').exists():
        prepare_recording(opt,calibration)
    if not (calibration/'fit/summary.json').exists():
        run(dest/'fit.log',[CODE/'fit.py','--context',opt/'context.json','--selection',opt/'selection/selection.json',
            '--source',calibration/'aligned','--skeleton',opt/'skeleton/aligned_skeleton.npz','--hands',*HANDS,
            '--output',calibration/'fit','--anat-weight','.01','--normal-min','-.002','--final-silhouette',
            '--root-offset','free','--wrist-marker-weight','0','--joint-limit-weight',str(case.params['joint_limit_weight'])])
    if not (dest/'calibrated').exists():
        (dest/'calibrated').symlink_to(calibration/'fit/final',target_is_directory=True)
    (dest/'aligned').mkdir(exist_ok=True)
    # The rigid image correction depends on the measured optical points. Only shared
    # shape and attachment arrays change when a recording is recalibrated.
    for hand in HANDS:
        name=f'{hand}_pilot.npz'
        target=dest/'aligned'/name
        if target.exists():
            continue
        with np.load(opt/'dense/aligned'/name) as dense, np.load(opt/'fit/final'/name) as old, np.load(dest/'calibrated'/name) as new:
            for k in SHARED:
                np.testing.assert_array_equal(old[k],dense[k])
            values=dict(dense)
            values.update({k:new[k] for k in SHARED})
            temp=target.with_suffix('.partial.npz')
            np.savez_compressed(temp,**values)
            temp.replace(target)
    if not (dest/'gaps/marker_gaps.json').exists():
        run(dest/'gaps.log',[CODE/'sdk_marker_gaps.py','--context',opt/'context.json',
            '--skeleton',opt/'skeleton/aligned_skeleton.npz','--selection',opt/'selection/selection.json',
            '--calibrated',dest/'calibrated','--output',dest/'gaps'])
    if not (dest/'reference/summary.json').exists():
        solver = CODE/'dense_drive.py'
        if (case.root/'native/marker_name_compatibility/source.json').is_file():
            solver = case.ego/'tools/label_pipeline/recordings/dense_observations.py'
        run(dest/'dense.log',[solver,'--selection',opt/'selection/selection.json',
            '--source',dest/'aligned','--output',dest/'reference','--joint-limit-weight',str(case.params['joint_limit_weight']),
            '--min-markers-per-finger',str(case.params['min_markers_per_finger']),'--sdk-marker-gaps',dest/'gaps/marker_gaps.npz'])
