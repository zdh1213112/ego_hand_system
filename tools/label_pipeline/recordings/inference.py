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
PROJECT=PACKAGE.parents[2]

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

def command(ego,cfg,root):
    source=next((PROJECT/'ego_hand_system/output/glove_v2_best_detector3680/full').glob('group_004_*/run_config.json'))
    options=json.loads(source.read_text())['options']
    for name in ('checkpoint','adapter','detector'):
        if Path(options[name]).resolve()!=(PROJECT/cfg[name]).resolve():raise ValueError(name)
    run=root/'inference'/ego.name.split('_')[-1]
    options.update(input=str(ego/'part0001'),calibration=str(next(ego.glob('*calibration_camera.yaml'))),output=str(run))
    runner=PROJECT/'work/wilor_kb_inference_runtime/run_inference.py'
    argv=[sys.executable,'-u','-B',str(runner)]
    for k,v in options.items():argv.extend(['--'+k.replace('_','-'),str(v)])
    return run,argv

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
        proc=subprocess.run(argv,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
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
