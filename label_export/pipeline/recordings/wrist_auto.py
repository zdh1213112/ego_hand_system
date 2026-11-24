"""Apply a recording's measured wrist decision and bind it to the very same rendered labels."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PIPE))
from recordings.wrist_select import run
from recordings.preview import wrist_review, verify_review
from stages.signed import write_json
from stages.fingerprint import digest


class InsufficientWristEvidence(ValueError):
    """The recording cannot supply a measured reference after sampling more poses."""


def read(path):
    return json.loads(path.read_text())


def finalize(root,cfg,keep_previous=False):
    group=cfg['id']
    review=root/'review'
    evidence=review/'adaptive_wrist'
    decision=evidence/'decision.json'
    proof=review/'wrist_preview_targets.json'
    sign=review/'wrist_signoff.json'
    if decision.is_file() and sign.is_file() and read(sign).get('approved'):
        verify_review(root)
        return [review/'wrist_targets.json',proof,decision,review/'WRIST_REVIEW.md']
    if (root/'records/step07/run_record.json').exists():
        raise ValueError(f'{group}: downstream export already started, cannot replace reference')
    result=run(root,group,evidence)
    for extra in (56,112):
        if all(r['status']=='selected' for r in result.values()):
            break
        previous=evidence/f'candidate_scan_before_{extra}.json'
        shutil.copy2(evidence/'selection.json',previous)
        case=read(root/'records/raw_case.json')
        case['wrist_candidate_samples']=extra
        extra_case=evidence/f'case_{extra}.json'
        write_json(extra_case,case)
        from stages.raw_flow import run_phase
        run_phase('runner.py',extra_case,'pivot')
        result=run(root,group,evidence)
    failed=[h for h,r in result.items() if r['status']!='selected']
    if failed:
        write_json(evidence/'unusable.json',dict(status='insufficient_evidence',hands=failed,
                   reasons={h:result[h]['reason'] for h in failed},selection_sha256=digest(evidence/'selection.json')))
        raise InsufficientWristEvidence(f"{group}: 扩大采样后仍无可靠参照 {failed}，详见 {evidence/'selection.json'}")
    picks={h:r['sync'] for h,r in result.items()}
    old=review/'wrist_targets.json'
    rerender=not proof.is_file() or {h:r['sync'] for h,r in read(old).items()}!=picks
    archive=root/'review_versions'/'before_automatic'
    if keep_previous and not archive.exists():
        archive.mkdir(parents=True)
        for name in ('wrist_targets.json','wrist_signoff.json','wrist_preview_targets.json','WRIST_REVIEW.md'):
            if (review/name).is_file():
                shutil.copy2(review/name,archive/name)
        if (root/'records/step06/run_record.json').is_file():
            shutil.copy2(root/'records/step06/run_record.json',archive/'step06.json')
        if (review/'wrist_preview').is_dir():
            shutil.copytree(review/'wrist_preview',archive/'wrist_preview',copy_function=os.link)
    if rerender:
        # Old media may have archived hard links. Start with a clean directory before rendering.
        preview=review/'wrist_preview'
        if preview.exists():
            versions=root/'review_versions'
            versions.mkdir(exist_ok=True)
            preview.rename(versions/f'previous_preview_{time.time_ns()}')
        case=read(root/'records/raw_case.json')
        case['wrist_sync']=picks
        run_case=evidence/'case.json';write_json(run_case,case)
        subprocess.run([sys.executable,'-Bu',str(PIPE/'stages/legacy/review.py'),str(run_case)],check=True)
        wrist_review(root,cfg)
    if {h:r['sync'] for h,r in read(old).items()}!=picks:
        raise ValueError('选择的帧与签收页不一致')
    if read(review/'wrist_preview/pixel_audit.json').get('loader_max_abs_crop_px')!=0:
        raise ValueError('训练读取与画面不一致')
    paths=[old,proof,decision,review/'WRIST_REVIEW.md']
    signature=digest(evidence/'selection.json')
    meta=dict(source='automatic_measured_reference',recording=group,selection_sha256=signature,
              observation_sha256=digest(evidence/'cuff_rgb.json'),
              preview_sha256=digest(proof),targets={h:{k:r[k] for k in ('sync','sha256')} for h,r in read(old).items()})
    write_json(decision,meta)
    (evidence/'unusable.json').unlink(missing_ok=True)
    write_json(review/'wrist_signoff.json',dict(approved=True,source=meta['source'],preview_sha256=meta['preview_sha256'],
                                               selection_sha256=signature,observation_sha256=meta['observation_sha256'],**meta['targets']))
    verify_review(root)
    report=review/'WRIST_REVIEW.md'
    text=report.read_text().replace('本次需要确认黄色菱形所示的外旋参照。','黄色菱形标记程序选定的外旋参照。').replace('请按手确认参照帧，或指出希望重看的同步帧号。当前等待用户签收。','')
    report.write_text(text.rstrip()+'\n\n左右腕已按本段观测独立选定。程序接着做袖口挂点和手指拟合。\n\n'+'\n'.join(
        f"- {h} 第 {r['sync']} 帧，{r['chosen']['views']} 台相机各有至少两个同名袖口亮点，三球合计重投影中位 {r['chosen']['cuff_reprojection_px']:.2f} px，最大指尖位移 {r['chosen']['max_tip_mm']:.2f} mm。"
        for h,r in result.items())+'\n\n[全部候选及评分](adaptive_wrist/selection.json) · [自动判定记录](adaptive_wrist/decision.json)\n')
    return paths


def upgrade(root):
    import fcntl
    with (root/'records/step06.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        step=root/'records/step06/run_record.json'
        state=read(step)
        if state['state']!='complete':
            raise ValueError('需要完整第 06 步')
        cfg=state['config']
        for item,signature in state['outputs'].items():
            if digest(root/item)!=signature:
                # A partially completed review upgrade can be retried with identical pivot geometry.
                if item.startswith('review/') and (root/'review_versions/before_automatic').is_dir():
                    continue
                raise ValueError(f'{item}: 既有标签发生变化')
        files=finalize(root,cfg,keep_previous=True)
        state['outputs'].update({str(p.relative_to(root)):digest(p) for p in files})
        state['wrist_policy']='adaptive_measured'
        write_json(step,state)


if __name__=='__main__':
    upgrade(Path(sys.argv[1]).resolve())
