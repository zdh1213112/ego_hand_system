"""Explicit unified trial and new-recording fitting gate."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from stages.fingerprint import digest
from stages.signed import write_json

PACKAGE=Path(__file__).resolve().parents[1]
PROJECT=PACKAGE.parents[2]


def fit(native,observations,output,hand,exclude):
    cmd=[sys.executable,'-Bu',str(PACKAGE/'unified/solver.py'),'--native',str(native),'--observations',str(observations),
         '--output',str(output),'--hand',hand,'--exclude',*map(str,exclude)]
    subprocess.run(cmd,check=True)


def run_trial(batch,dest):
    from unified.benchmark import run
    source=(PROJECT/batch['trial']['native_source']).resolve();obs=(PROJECT/batch['trial']['fixed_observations']).resolve()
    b=dest/'benchmark';b.mkdir(exist_ok=True)
    for group in ('group_003','group_005'):
        for h in ('left','right'):
            fit(source/group/'native',obs/group,b/group/'tracks',h,[883])
    run(source,obs,b)
    subprocess.run([sys.executable,'-Bu',str(PACKAGE/'unified/compare_images.py'),'--source',str(source),'--exports',
       str(PROJECT/batch['trial']['legacy_exports']),'--fixed',str(obs),'--output',str(b),'--frames',str(PROJECT/batch['trial']['frames'])],check=True)


def promotion(root):
    metrics=root/'benchmark/metrics.json'
    if not metrics.is_file():raise ValueError('需先完成 003、005 数值及图像验收')
    results=json.loads(metrics.read_text())
    if {(v['group'],v['hand']) for v in results}!={(g,s) for g in ('group_003','group_005') for s in ('left','right')} or not all(v['passed'] for v in results):
        raise ValueError('003、005 尚未达到数值门槛，详见 '+str(root/'benchmark/COMPARISON.md'))
    sign=root/'benchmark/visual_signoff.json'
    if not sign.is_file():raise ValueError('等待用户判断 003、005 对比图')
    approval=json.loads(sign.read_text())
    if approval.get('approved') is not True or approval.get('metrics_sha256')!=digest(metrics):
        raise ValueError('003、005 视觉签收与本次数值版本不符')


def stage(i,cfg,root):
    """Stages 1–3 explicitly share accepted raw observations and never import a legacy fitted pose."""
    from recordings.marker_names import compatible_alignment
    shared=root.parents[1]/'legacy'/cfg['id'];native=root/'native';native.mkdir(exist_ok=True)
    def link(path,dst):
        if not dst.exists():dst.symlink_to(path.resolve(strict=True),target_is_directory=path.is_dir())
        elif not dst.samefile(path):raise ValueError(dst)
    if i<=3:
        record=shared/'records'/f'step{i:02}'/'run_record.json'
        state=json.loads(record.read_text())
        if state.get('state')!='complete':raise ValueError('共享上游没有完成 '+str(record))
        for path,expected in state['outputs'].items():
            if digest(shared/path)!=expected:raise ValueError('共享观测已改变 '+path)
        if i==1:link(shared/'converted',root/'converted')
        if i==2:link(shared/'native/processed',native/'processed')
        if i==3:link(shared/'native/fourview_alignment',native/'fourview_alignment')
        proof=root/'records'/f'raw_shared_{i}.json';write_json(proof,dict(source=str(record),sha256=digest(record)));return [proof]
    obs=root/'raw_observations';fmt=root/'solver_input';opt=root/'optical_observations'
    rgb=native/'processed/ego/normalized_multiview'
    if i==4:
        from unified.optics import prepare
        align=compatible_alignment(root);ctx=root/'records/observation_context.json'
        write_json(ctx,dict(dataset=str(rgb),alignment=str(align),aligned21=str(native/'fourview_alignment/aligned_nokov21.npz'),
            placement=cfg['marker_placement'],baseline_fit=str(root/'unused_baseline'),holdout=[],device='cpu'))
        if not (obs/'aligned_skeleton.npz').is_file():
            subprocess.run([sys.executable,'-B',str(PACKAGE/'recordings/sdk_observations.py'),'--context',str(ctx),'--output',str(obs)],check=True)
        fmt.mkdir(exist_ok=True);link(obs/'aligned_skeleton.npz',fmt/'aligned_skeleton.npz');link(rgb/'calibration',fmt/'calibration')
        prepare(rgb,obs/'aligned_skeleton.npz',opt)
        return [obs/'aligned_skeleton.npz',*[opt/f'camera{c}.npz' for c in range(4)],*[opt/f'{h}_registration.npz' for h in ('left','right')]]
    if i==5:
        for h in ('left','right'):fit(fmt,opt,root/'tracks',h,[])
        return [root/'tracks'/f'{h}.npz' for h in ('left','right')]
    if i==6:
        review=root/'review';review.mkdir(exist_ok=True);targets={}
        for h in ('left','right'):
            path=root/'tracks'/f'{h}.npz'
            with np.load(path) as z:
                t=z['pair_indices'];j=z['joints'];good=z['marker_valid'][:,:3].all(1)&z['pass_surface']&((t//120)%4==0)
                ids=np.flatnonzero(good);reach=np.linalg.norm(j[:,[8,12,16,20]]-j[:,:1],axis=2).mean(1)
                if not len(ids):raise ValueError('无完整袖口候选 '+h)
                k=ids[np.argmin(abs(reach[ids]-np.percentile(reach[ids],75)))]
                targets[h]=dict(sync=int(t[k]),wrist_right_canonical_m=j[k,0].tolist(),target_file=str(path),sha256=digest(path))
        write_json(review/'wrist_targets.json',targets)
        write_json(review/'wrist_signoff.json',dict(approved=False,**{h:{k:v[k] for k in ('sync','sha256')} for h,v in targets.items()}))
        return [review/'wrist_targets.json']
    raise ValueError('统一方案的新录制最终导出尚待试算验收。')
