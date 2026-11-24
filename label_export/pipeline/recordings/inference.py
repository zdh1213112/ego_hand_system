"""Replay the approved 004 v2 invocation and pack its same-exposure detector observations."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import yaml

PACKAGE=Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
PROJECT = REPO.parent

def archive(run,output):
    with (run/'multiview_frames.csv').open() as stream:frames=list(csv.DictReader(stream))
    n=len(frames);times=np.arange(n)
    indices=np.asarray([[int(r[f'camera{c}_frame_index']) for c in range(4)] for r in frames])
    valid=np.zeros((n,4,2),bool);conf=np.zeros(valid.shape,np.float32)
    points=np.full((*valid.shape,21,2),np.nan,np.float32);boxes=np.full((*valid.shape,4),np.nan,np.float32)
    for c in range(4):
        with (run/'observations_2d'/f'camera{c}.jsonl').open() as stream:
            count=0
            for t,line in enumerate(stream):
                row=json.loads(line)
                assert (row['sync_index'],row['source_frame_index'],row['timestamp_ns'])==(t,indices[t,c],int(frames[t][f'camera{c}_timestamp_ns']))
                for h in row['hands']:
                    s=h['is_right']
                    if s!=h['detector_is_right'] or (valid[t,c,s] and conf[t,c,s]>=h['confidence']):continue
                    valid[t,c,s]=True;conf[t,c,s]=h['confidence'];points[t,c,s]=h['joints_2d'];boxes[t,c,s]=h['bbox_xyxy']
                count+=1
            if count!=n:raise ValueError((c,count,n))
    if not valid.any():raise ValueError('四机 WiLoR 均无有效观测')
    pending=output.with_suffix('.tmp.npz')
    np.savez_compressed(pending,times=times,camera_indices=indices,reference_timestamp_ns=[int(r['reference_timestamp_ns']) for r in frames],
                        joints_2d=points,bbox_xyxy=boxes,confidence=conf,valid=valid)
    os.replace(pending,output)
    return dict(frames=n,views_per_side=valid.sum((0,1)).tolist())

def command(ego, cfg, root):
    sys.path.insert(0, str(REPO / 'scripts'))
    from label_export_settings import workspace_path

    run = root / 'inference' / ego.name.split('_')[-1]
    options = {
        'input': ego / 'part0001',
        'calibration': next(ego.glob('*calibration_camera.yaml')),
        'output': run,
        'model-config': REPO / 'training/pretrained_models/model_config.yaml',
        'mano-model-dir': REPO / 'models/mano',
        'batch-size': 32, 'frame-batch-size': 8, 'preprocess-workers': 8,
        'shared-steps': 100, 'shared-batch-size': 32,
    }
    for name in ('checkpoint', 'adapter', 'detector'):
        options[name] = workspace_path(REPO, cfg[name])
    argv = ['bash', str(REPO / 'inference/run.sh')]
    for key, value in options.items():
        argv.extend(['--' + key, str(value)])
    return run, argv


def infer(cfg,ego):
    root=(PROJECT/cfg['output_root']).resolve()
    run,argv=command(ego,cfg,root);rec=ego.name.split('_')[-1]
    dest=root/'observations'/rec;dest.mkdir(parents=True,exist_ok=True)
    note=dest/'run_record.json';archive_path=dest/'native_wilor.npz'
    if archive_path.is_file() and note.is_file() and json.loads(note.read_text()).get('state')=='complete':return
    start=time.time()
    payload=dict(command=argv,shell_command=shlex.join(argv),state='running',start_unix=start)
    def save():note.write_text(json.dumps(payload,indent=2)+'\n')
    save()
    with (dest/'inference.log').open('a') as log:
        env = os.environ.copy()
        env.setdefault('EGO_HAND_PYTHON', sys.executable)
        proc=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    payload.update(elapsed_seconds=time.time()-start,returncode=proc.returncode);save()
    try:
        proc.check_returncode()
        payload.update(archive(run,archive_path),state='complete')
    except Exception as exc:
        payload.update(state='failed',error=str(exc))
        raise
    finally:save()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--recording',required=True)
    a=ap.parse_args();cfg=yaml.safe_load(a.config.read_text());ego,=(PROJECT/cfg['source']/'EGO-4B_20261120').glob('*_'+a.recording)
    sync=json.loads((PROJECT/cfg['output_root']/'pairing'/a.recording/'imu_nokov_sync.json').read_text());assert sync['confidence']=='strong'
    infer(cfg,ego)
