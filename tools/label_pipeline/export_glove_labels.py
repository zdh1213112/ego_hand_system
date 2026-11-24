"""Run the frozen-label regression or the validated raw-input preparation stages."""
import argparse
import ast
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import yaml
from stages.fingerprint import digest,verify

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
STEPS=[f'step{i:02}' for i in range(1,13)]

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def absolute(value):
    return (PROJECT/value).resolve(strict=True)

def code_check():
    result={}
    for p in sorted(HERE.rglob('*.py')):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[node.module or ''] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names]
                if any('work' in n.split('.') for n in names):raise ValueError(f'外部历史脚本 import {p}:{node.lineno}')
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and '/work/' in node.value and '.py' in node.value:
                raise ValueError(f'运行时脚本路径 {p}:{node.lineno}')
        result[str(p.relative_to(HERE))]=digest(p)
    for p in HERE.rglob('*.cpp'):result[str(p.relative_to(HERE))]=digest(p)
    for line in (HERE/'assets/SHA256SUMS').read_text().splitlines():
        checksum,name=line.split()
        if digest(HERE/'assets'/name)!=checksum:raise ValueError(name)
    return result

def validate(cfg):
    if cfg.get('fit_method','legacy')!='legacy':raise ValueError('unified 尚未通过前置回归，未启用。')
    if cfg.get('parameters'):
        print('参数与签收版不同：'+json.dumps(cfg['parameters'],ensure_ascii=False),flush=True)
        if cfg['mode']=='from_signed':raise ValueError('冻结轨迹回归须沿用签收参数。')
    if cfg['mode']=='from_signed':
        root=absolute(cfg['frozen_root']);gold=absolute(cfg['reference_labels'])
        if 'group_005' not in cfg['recordings']:raise ValueError('须先核对 005 签收样本。')
        for name in cfg['recordings']:
            if Path(name).name!=name or not (root/name/'tracks').is_dir() or not (gold/name/'index.jsonl').is_file():raise ValueError(name)
        return
    if cfg['mode']!='from_raw':raise ValueError(cfg['mode'])
    missing=[k for k in ('raw_group','camera_calibration','initial_alignment','marker_placement','mano_directory','native_wilor') if not cfg.get(k) or not (PROJECT/cfg[k]).exists()]
    if missing:raise ValueError('缺少输入 '+'、'.join(missing))
    for key in ('marker_placement','native_wilor','initial_alignment','camera_calibration'):
        if not absolute(cfg[key]).is_file():raise ValueError('需要文件 '+key)
    supplied=absolute(cfg['mano_directory'])/'MANO_RIGHT.pkl'
    if not supplied.is_file():raise ValueError('MANO_RIGHT.pkl 缺失')
    installed=HERE.parents[1]/'models/mano/MANO_RIGHT.pkl'
    if not supplied.samefile(installed) and digest(supplied)!=digest(installed):
        raise ValueError('原生拟合使用 ego_hand_system/models/mano/MANO_RIGHT.pkl，配置的模型与该版本内容不同。')
    for field in ('raw_group','initial_alignment','camera_calibration'):cfg[field]=str(absolute(cfg[field]))
    if cfg.get('id','') in ('','.','..') or Path(cfg['id']).name!=cfg['id']:raise ValueError('无效的录制名称')
    for f in ('marker_placement','mano_directory','native_wilor'):cfg[f]=str(absolute(cfg[f]))
    for name,value in cfg.get('wilor_observations',{}).items():
        if name not in ('fast_wilor.npz','extra_wilor.npz','native_wilor.npz') or not absolute(value).is_file():raise ValueError(name)
    import numpy as np
    archives={Path(cfg['native_wilor']),*(absolute(v) for v in cfg.get('wilor_observations',{}).values())}
    for archive in archives:
        with np.load(archive,allow_pickle=False) as z:
            missing={'times','camera_indices','joints_2d','valid','confidence'}-set(z.files)
            if missing:raise ValueError(f'{archive.name} 缺少 WiLoR 观测字段 {sorted(missing)}')
            times=z['times']
            if not len(times) or any(np.diff(times)<=0) or z['camera_indices'].shape!=(len(times),4) or z['joints_2d'].shape!=(len(times),4,2,21,2):
                raise ValueError(f'{archive.name} 的观测帧或四机规则不符')
    raw=Path(cfg['raw_group'])
    from stages.raw import recording_files
    recording_files(raw)
    for pat in ('session*/nokov/nokov_rigid_bodies.csv','EGO*/*_calibration_camera.yaml'):
        found=list(raw.glob(pat))
        if len(found)!=1:raise ValueError(f'{raw}/{pat} 匹配 {len(found)} 份')
    original,=raw.glob('EGO*/*_calibration_camera.yaml')
    if digest(original)!=digest(Path(cfg['camera_calibration'])):raise ValueError('指定标定和原始录制不匹配')
    if not list(raw.glob('session*/nokov/*skeleton*')):raise ValueError('缺少同一期 NOKOV 手部骨架')

def run(args):
    cfg=yaml.safe_load(args.config.read_text());defaults=yaml.safe_load((HERE/'defaults.yaml').read_text())
    if cfg['mode']=='from_raw':
        cfg.setdefault('fit_method','legacy')
        cfg.setdefault('wrist_policy','manual' if cfg.get('reuse_historical_005_wrist') or cfg.get('wrist_sync') else 'adaptive_measured')
        if cfg['wrist_policy'] not in ('manual','adaptive_measured'):
            raise ValueError('未知的腕部参照方式 '+str(cfg['wrist_policy']))
    validate(cfg);code=code_check()
    inputs={'config':digest(args.config),'defaults':digest(HERE/'defaults.yaml'),'code':code}
    raw=cfg['mode']=='from_raw'
    if raw:
        folder=Path(cfg['raw_group'])
        from stages.raw import recording_files
        data=[*recording_files(folder),*folder.glob('session*/nokov/*.csv'),Path(cfg['initial_alignment']),Path(cfg['camera_calibration']),Path(cfg['marker_placement']),Path(cfg['mano_directory'])/'MANO_RIGHT.pkl',Path(cfg['native_wilor']),*(absolute(v) for v in cfg.get('wilor_observations',{}).values())]
        inputs['raw_files']={str(p.resolve()):digest(p) for p in sorted(set(data)) if p.is_file()}
    if not raw:
        frozen=absolute(cfg['frozen_root']);inputs['manifest']=digest(frozen/'MANIFEST.json')
    if args.check:
        print(json.dumps(dict(inputs=inputs,fit_method='legacy',mode=cfg['mode']),ensure_ascii=False,indent=2))
        return
    if args.publish and not raw:raise ValueError('冻结重发输出仅作签收对照。')
    print('拟合方法：legacy',flush=True)
    out=(PROJECT/cfg['output_root']).resolve()
    for protected in (HERE.parents[1],PROJECT/'wilor_training',PROJECT/'work/signed_labels_20261120',PROJECT/'work/glove_trainset_20261120_v2'):
        if out==protected or protected in out.parents or out in protected.parents:raise ValueError(f'输出目录与输入或源码重叠 {out}')
    if args.publish:
        from stages.raw_flow import publish
        publish(out/'legacy'/cfg['id'])
        return
    first=int((args.first or ('step01' if raw else 'step09'))[-2:]);last=int((args.last or 'step12')[-2:])
    if first>last or (not raw and first<9):raise ValueError('阶段范围不适用于该配置')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        waiting=False
        if raw:
            pause=out/'legacy'/cfg['id']/'records/step07/run_record.json'
            if pause.exists():
                state=json.loads(pause.read_text())
                waiting=state.get('state')=='awaiting_wrist_signoff'
        if any(p!=Path(lock.name) for p in out.iterdir()) and not (args.resume or waiting):raise FileExistsError(f'{out} 已有结果，续跑须显式 --resume')
        # Resume binds both code/config and upstream output bytes. No unchecked skip.
        groups=[cfg.get('id','recording')] if raw else ['group_005',*[g for g in cfg['recordings'] if g!='group_005']]
        verified=verify(frozen) if not raw else None
        for group in groups:
            root=out/'legacy'/group;root.mkdir(parents=True,exist_ok=True)
            chain=''
            for i in range(1 if raw else 9,last+1):
                step=STEPS[i-1];log=root/'records'/step/'run_record.json'
                if log.exists():
                    row=json.loads(log.read_text())
                    if row['inputs']!=inputs or row['upstream']!=chain:raise ValueError(f'{step} 代码或上游不同')
                    if row['state']=='complete':
                        if i==10 and not raw:
                            for line in (root/'export_sha256.csv').read_text().splitlines():
                                hashed,rel=line.split(',',1)
                                if digest(root/rel)!=hashed:raise ValueError(f'{step} 产物变化 {rel}')
                        for rel,hashed in row['outputs'].items():
                            if digest(root/rel)!=hashed:raise ValueError(f'{step} 产物变化 {rel}')
                        chain=digest(log);continue
                if i<first:raise ValueError(f'缺少已完成的 {step}，无法跳过')
                start=time.monotonic()
                rec=dict(stage=step,group=group,inputs=inputs,parameters={**defaults,**cfg.get('parameters',{})},upstream=chain,started=timestamp(),state='running',fit_method='legacy')
                save(log,rec)
                try:
                    if raw:
                        from stages.raw_flow import run as raw_stage
                        if i==7 and cfg.get('wrist_policy')=='adaptive_measured':
                            from recordings.preview import verify_review
                            verify_review(root)
                        if i==4 and cfg.get('verify_exposure_clock'):
                            from recordings.exposure_check import check_archive
                            check_archive(root,Path(cfg['native_wilor']))
                        files=raw_stage(i,cfg,root)
                        if i==12 and cfg.get('wrist_policy')=='adaptive_measured':
                            from recordings.preview import dataset_review
                            files+=dataset_review(root,cfg)
                    elif i==9:
                        from stages.signed import prepare
                        rec['frozen_inputs']=verified
                        rec['statistics']=prepare(frozen/group,root);files=[root/'sequence.npz']
                    elif i==10:
                        from stages.signed import export
                        rec['statistics']=export(frozen/group,root)
                        files=[root/'index.jsonl']
                        # Record fingerprints for all generated files, not only their index.
                        with (root/'export_sha256.csv').open('w') as ids:
                            for path in sorted(p for d in ('images','labels','labels_with_glove') for p in (root/d).iterdir()):
                                ids.write(f'{digest(path)},{path.relative_to(root)}\n')
                        files.append(root/'export_sha256.csv')
                    elif i==11:
                        from stages.regression import audit
                        result=audit(frozen/group,root,absolute(cfg['reference_labels'])/group)
                        save(root/'regression.json',result);files=[root/'regression.json']
                        n=result['expected']
                        if result['missing'] or result['extra'] or any(result['counts'].get(k)!=n for k in ('all_fields_equal','label_storage_normalized_sha256_match','image_sha256_match','glove_label_storage_normalized_sha256_match')):
                            raise ValueError(f'{group} 回归未通过，详见 {root}/regression.json')
                    else:
                        from stages.release import timeline
                        rec['statistics']=timeline(frozen/group,root);files=[root/'timeline.json']
                    rec.update(state='complete',outputs={str(p.relative_to(root)):digest(p) for p in files})
                except Exception as exc:
                    state='awaiting_wrist_signoff' if raw and i==7 and str(exc).startswith('等待腕点签收') else 'failed'
                    rec.update(state=state,failure=str(exc));raise
                finally:
                    rec.update(ended=timestamp(),elapsed_s=round(time.monotonic()-start,2));save(log,rec)
                chain=digest(log)

        if not raw and last==12:
            from stages.merge_verified import merge,verify_loaders
            merged=merge(frozen,out)
            if not (merged/'loader_check.json').exists():verify_loaders(merged,HERE.parents[1])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--publish',action='store_true')
    p.add_argument('--from',dest='first',choices=STEPS);p.add_argument('--to',dest='last',choices=STEPS)
    try:run(p.parse_args())
    except (ValueError,FileExistsError) as error:sys.exit(str(error))
