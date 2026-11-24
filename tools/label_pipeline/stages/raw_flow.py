"""Orchestrate one private recording and its two user reviews."""
import json
from pathlib import Path
import subprocess
import sys
from .fingerprint import digest
from .signed import write_json
from .geometry import EGO

STAGE=Path(__file__).with_name('legacy')

def case_file(cfg,root):
    path=root/'records/raw_case.json'
    data=dict(cfg,recording_output=str(root))
    native=Path(cfg['native_wilor'])
    if not native.is_absolute():native=EGO.parent/native
    sources=cfg.get('wilor_observations',{})
    if not sources:
        sources={'native_wilor.npz':str(native),'fast_wilor.npz':str(native)}
    data['wilor_observations']=sources
    write_json(path,data)
    return path

def run_phase(program,case,*args):
    cmd=[sys.executable,'-Bu',str(STAGE/program),str(case),*args]
    logs=case.parent/'subprocess';logs.mkdir(exist_ok=True)
    name=Path(program).stem+'_'+('_'.join(args) or 'review')
    with (logs/(name+'.log')).open('w') as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{name} 失败，详见 {f.name}')

def run(i,cfg,root):
    from .raw import run as prepare_native
    if i<=3:return prepare_native(i,cfg,root)
    case=case_file(cfg,root);g=cfg['id'];work=root/'legacy_stages'
    def phase(name):run_phase('runner.py',case,name)
    if i==4:
        if cfg.get('wrist_policy')=='adaptive_measured':
            from recordings.seed import execute
        else:
            from .legacy.seed import execute
        import yaml
        params=yaml.safe_load((STAGE.parents[1]/'defaults.yaml').read_text())
        execute(root,(EGO.parent/cfg['marker_placement']).resolve(),cfg.get('parameters',{}).get('sample_stride',params['sample_stride']))
        phase('dense')
        return [work/'nokov_label_training_20261118'/g/'contour/summary.json',work/'nokov_label_training_20261118'/g/'contour/final/export/merged/summary.json']
    if i==5:
        phase('motion');return [work/'joint_motion_20261119'/g/'final/temporal_summary.json',work/'joint_motion_20261119/fast_clips.json']
    if i==6:
        phase('pivot')
        if cfg.get('wrist_policy')=='adaptive_measured' and not cfg.get('reuse_historical_005_wrist'):
            from recordings.wrist_auto import finalize
            return [*finalize(root,cfg),*[work/'tip_pivot_20261119'/g/h/'geometry6_locked.npz' for h in ('left','right')]]
        args=['--reuse-signed-005'] if cfg.get('reuse_historical_005_wrist',False) else []
        run_phase('review.py',case,*args)
        return [root/'review/wrist_targets.json',*[work/'tip_pivot_20261119'/g/h/'geometry6_locked.npz' for h in ('left','right')]]
    if i==7:
        from .legacy.raw_export import setup
        env=setup(case)
        from .legacy.review import verify_wrist
        if not (root/'review/wrist_signoff.json').exists():raise ValueError('等待腕点签收 '+str(root/'review/WRIST_REVIEW.md'))
        accepted=verify_wrist(env)
        data=json.loads(case.read_text());data['wrist_sync']=accepted;write_json(case,data)
        phase('cuff');phase('anatomy')
        return [work/'direction_acceptance_20261119'/g/'distal_anatomy'/(h+'.npz') for h in ('left','right')]
    if i==8:
        phase('mcp');return [work/'mcp_outward_20261119'/g/'final'/(h+'.npz') for h in ('left','right')]
    if i==9:
        run_phase('raw_export.py',case,'bundle')
        return [root/'sequence.npz',root/'geometry_input'/g/'decisions/pose_screen.json']
    if i==10:
        run_phase('raw_export.py',case,'export')
        from .raw_release import fingerprint_export
        return [root/'exposure_index.jsonl',root/'geometry_input'/g/'decisions/photo.csv',fingerprint_export(root)]
    if i==11:
        from .raw_release import assemble
        assemble(root,g);return [root/'splits.json',root/'summary.json']
    if i==12:
        from .merge_verified import verify_loaders
        from .raw_release import review
        verify_loaders(root,EGO);review(root,g)
        return [root/'loader_check.json',root/'review/dataset_targets.json']

def publish(root):
    review=root/'review';sign_path=review/'dataset_signoff.json'
    if not sign_path.is_file():raise ValueError('等待数据签收 '+str(review/'DATASET_REVIEW.md'))
    sign=json.loads(sign_path.read_text());target=review/'dataset_targets.json'
    if sign.get('approved') is not True or sign.get('sha256')!=digest(target):raise ValueError('数据签收与候选版本不符')
    proof=json.loads(target.read_text())
    for filename,sha in proof.get('review_image_sha256',{}).items():
        if digest(review/filename)!=sha:raise ValueError('签收预览图变化 '+filename)
    for filename,sha in proof['dataset_sha256'].items():
        if digest(root/filename)!=sha:raise ValueError('已签收的候选发生变化 '+filename)
    import csv,os,shutil
    import numpy as np
    from .raw_release import lines
    from .signed import rows
    release=root.parent.parent/'published'/root.name
    release.parent.mkdir(parents=True,exist_ok=True)
    if release.exists():
        if json.loads((release/'release.json').read_text())['target_sha256']!=digest(target):raise FileExistsError(release)
        return
    pending=release.with_name('.'+release.name+'.pending');pending.mkdir(exist_ok=True)
    with (root/'export_sha256.csv').open() as stream:
        for expected,rel in csv.reader(stream):
            if digest(root/rel)!=expected:raise ValueError('样本文件变化 '+rel)
    for name in ('images','labels','sequence.npz','frames.jsonl','rejected.jsonl','splits.json'):
        dest=pending/name
        if not dest.exists():dest.symlink_to(os.path.relpath(root/name,pending))
    rich=pending/'labels_with_glove';rich.mkdir(exist_ok=True)
    for p in sorted((root/'labels_with_glove').glob('*.npy')):
        sample=np.load(p,allow_pickle=True).item();sample['approved_for_training']=True
        np.save(rich/p.name,sample)
    for name in ('index','index_with_glove','train','val'):
        lines(pending/(name+'.jsonl'),(dict(r,approved_for_training=True) for r in rows(root/(name+'.jsonl'))))
    summary=json.loads((root/'summary.json').read_text());summary['approved_for_training']=True
    write_json(pending/'summary.json',summary)
    write_json(pending/'release.json',dict(approved_for_training=True,fit_method='legacy',source=str(root),target_sha256=digest(target),user_signoff_sha256=digest(sign_path)))
    if (root/'dataset_profile.json').is_file():
        profile=json.loads((root/'dataset_profile.json').read_text());profile['dataset_root']=str(release)
        write_json(pending/'dataset_profile.json',profile)
    pending.rename(release)
