"""Run both label methods for a recording batch, stopping at each actual user signoff."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import yaml

from recordings import pairing,inference
from recordings.wrist_auto import InsufficientWristEvidence
from stages.fingerprint import digest
from stages.raw_flow import run as raw_stage
from stages.signed import write_json

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]

def read(p):return json.loads(p.read_text())
def link(src,dst):
    if dst.is_symlink() or dst.exists():
        if not dst.samefile(src):raise FileExistsError(dst)
    else:dst.symlink_to(src,target_is_directory=src.is_dir())

def config_for(batch,r,dest):
    rec=r['id'];raw=dest/'raw_groups'/rec;raw.mkdir(parents=True,exist_ok=True)
    for src in (Path(r['ego']),Path(r['session'])):link(src,raw/src.name)
    cfg=dict(mode='from_raw',wrist_policy=batch.get('wrist_policy','adaptive_measured'),fit_method=batch['fit_method'],id=rec,raw_group=str(raw),
             native_wilor=str(dest/'observations'/rec/'native_wilor.npz'),output_root=str(dest),
             camera_calibration=r['calibration'],**{k:str((PROJECT/batch[k]).resolve()) for k in ('initial_alignment','marker_placement','mano_directory')})
    (dest/'configs').mkdir(exist_ok=True);(dest/'configs'/(rec+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False))
    return cfg

def preflight(batch,dest):
    src=PROJECT/batch['source']
    files={k:str((PROJECT/batch[k]).resolve(strict=True)) for k in ('initial_alignment','marker_placement','mano_directory','adapter','checkpoint','detector')}
    out=[]
    for ego in sorted((src/'EGO-4B_20261120').iterdir()):
        rec=ego.name.split('_')[-1];run,cmd=inference.command(ego,batch,dest);obs=dest/'observations'/rec/'native_wilor.npz'
        out.append(dict(recording=rec,mcap_exists=any(ego.glob('part0001/*.mcap')),calibration_exists=any(ego.glob('*_calibration_camera.yaml')),same_exposure_observations=obs.exists(),inference_command=cmd))
    write_json(dest/'preflight.json',dict(fit_method=batch['fit_method'],configured_inputs=files,recordings=out))
    print('输入检查',len(out),'段，已有同曝光观测',sum(r['same_exposure_observations'] for r in out),'段',flush=True)

def _stage(i,cfg,root):
    path=root/'records'/f'step{i:02}'/'run_record.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (last:=read(path)).get('state')=='complete':
        for p,sha in last['outputs'].items():
            if digest(root/p)!=sha:raise ValueError(f'上游产物发生变化 {p}')
        return
    state=dict(stage=i,fit_method=cfg['fit_method'],started=time.time(),state='running',config=cfg)
    write_json(path,state)
    try:
        verified=None
        if i==5 and cfg['fit_method']=='legacy':
            from recordings.exposure_check import check
            verified=check(root,Path(cfg['native_wilor']))
        if i==7:
            from recordings.preview import verify_review
            verify_review(root)
        if cfg['fit_method']=='unified':
            from unified.entry import stage as unified_stage
            files=unified_stage(i,cfg,root)
        elif i==4:
            from recordings.seed import execute
            from stages.raw_flow import case_file,run_phase
            execute(root,Path(cfg['marker_placement']))
            run_phase('runner.py',case_file(cfg,root),'dense')
            contour=root/'legacy_stages/nokov_label_training_20261118'/cfg['id']/'contour'
            files=[contour/'summary.json',contour/'final/export/merged/summary.json']
        else:
            files=raw_stage(i,cfg,root)
        if verified is not None:files.append(verified)
        if i==12:
            from recordings.preview import dataset_review
            files+=dataset_review(root,cfg)
        if i==6:
            from recordings.preview import wrist_review
            if cfg.get('fit_method')!='legacy' or cfg.get('wrist_policy')!='adaptive_measured':
                files+=wrist_review(root,cfg)
        if i==3 and cfg['fit_method']=='legacy':
            dual=read(root/'native/processed/alignment/dual_alignment.json')
            if dual['status']!='accepted':raise ValueError('双手空间标定未通过 '+str(dual['metrics']['heldout_by_side']))
        state.update(outputs={str(p.relative_to(root)):digest(p) for p in files},state='complete')
    except Exception as e:
        state.update(state='failed',error=str(e));raise
    finally:
        state['elapsed_s']=time.time()-state['started'];write_json(path,state)

def stage(i,cfg,root):
    import fcntl
    folder=root/'records';folder.mkdir(exist_ok=True)
    with (folder/f'step{i:02}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if i in (7,8):
            with (root.parents[1]/'label_fit_gpu.lock').open('a') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                _stage(i,cfg,root)
        else:
            _stage(i,cfg,root)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--check',action='store_true')
    ap.add_argument('--trial',action='store_true',help='运行独立的 003、005 对照');ap.add_argument('--recording');ap.add_argument('--from-step',type=int,default=1);ap.add_argument('--to-step',type=int);args=ap.parse_args()
    batch=yaml.safe_load(args.config.read_text());dest=(PROJECT/batch['output_root']).resolve();dest.mkdir(parents=True,exist_ok=True)
    batch.setdefault('fit_method','legacy')
    if args.to_step is None:
        args.to_step=12 if batch['fit_method']=='legacy' and batch.get('wrist_policy','adaptive_measured')=='adaptive_measured' else 6
    if batch['fit_method'] not in ('legacy','unified'):raise ValueError('未知拟合方法')
    if args.check:preflight(batch,dest);return
    if args.trial:
        if batch['fit_method']!='unified':raise ValueError('试算须显式选择 unified')
        from unified.entry import run_trial
        run_trial(batch,dest);return
    if batch['fit_method']=='unified':
        from unified.entry import promotion
        promotion(dest)
    if not 1<=args.from_step<=args.to_step<=12:ap.error('阶段须在 1–12 之间')
    data=read(dest/'pairing.json')
    if args.recording:data=[r for r in data if r['id']==args.recording]
    if not data:raise ValueError('未找到录制')
    for r in data:
        if not r['eligible']:continue
        cfg=config_for(batch,r,dest);root=dest/cfg['fit_method']/r['id'];root.mkdir(parents=True,exist_ok=True)
        for i in range(args.from_step,args.to_step+1):
            if cfg['fit_method']=='legacy' and i>=5 and not Path(cfg['native_wilor']).is_file():raise FileNotFoundError(cfg['native_wilor'])
            if i>1:
                prior=root/'records'/f'step{i-1:02}'/'run_record.json'
                if not prior.is_file() or read(prior).get('state')!='complete':raise ValueError('上游阶段未完成')
            try:
                stage(i,cfg,root)
            except InsufficientWristEvidence as exc:
                if args.recording:
                    raise
                print(str(exc),flush=True)
                break

if __name__=='__main__':main()
