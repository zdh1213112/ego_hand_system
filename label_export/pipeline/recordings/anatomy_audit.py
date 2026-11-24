"""Compare flagged final poses with the same exposures in the SDK, triangulated WiLoR and earlier fits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PIPE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PIPE))
from stages.legacy.raw_export import setup
from stages.signed import rows,write_json


def events(flex):
    from anatomy_common import flags
    hyper,zig=flags(flex)
    mask=np.isfinite(flex)
    return np.concatenate([hyper,zig[...,None]],axis=-1),np.concatenate([mask,mask[:,:,1:].all(-1,keepdims=True)],axis=-1)


def run(root, out=None):
    case=setup(root/'records/raw_case.json')
    from anatomy_common import signed_flex,DIGITS,JOINTS
    g=case.recording
    used={0:set(),1:set()}
    samples=Counter()
    for row in rows(root/'index.jsonl'):
        side=int(row['side']);t=int(row['sync_index'])
        used[side].add(t);samples[side,t]+=1
    final_meta=json.loads((root/'review/dataset_targets.json').read_text())
    suspect={}
    scores={}
    for side,h in enumerate(('left','right')):
        final=root/'geometry_input'/g/'tracks'/(h+'.npz')
        with np.load(final) as z:
            times=z['pair_indices'];j=z['joints']
        keep=np.isin(times,list(used[side]))
        ts=times[keep]
        final_flags,_=events(signed_flex(j[keep].astype(float),True))
        ref=root/'legacy_stages/direction_acceptance_20261119'/g/(h+'_observations.npz')
        with np.load(ref) as z:
            ix=np.searchsorted(z['times'],ts)
            np.testing.assert_array_equal(z['times'][ix],ts)
            sensors={key:events(signed_flex(z[key][ix].astype(float),bool(side))) for key in ('sdk','native3d','soft','wide')}
        paired={}
        failed={(q['digit'],q['joint']) for q in final_meta[h]['failed_19']}
        exposure_flags={}
        for f,digit in enumerate(DIGITS):
            for q,name in enumerate((*JOINTS[f],'zig')):
                if (f,q)==(0,0):continue
                a=final_flags[:,f,q]; rec={'final':int(a.sum()),'poses':len(ts)}
                for key,(flags,valid) in sensors.items():
                    v=valid[:,f,q];b=flags[:,f,q]
                    rec[key]=dict(comparable=int(v.sum()),flagged=int((v&b).sum()),comparable_final_events=int((v&a).sum()),shared_events=int((v&a&b).sum()))
                sdk,sv=sensors['sdk'];wi,wv=sensors['native3d']
                both=a&sv[:,f,q]&wv[:,f,q]
                neither=both&~sdk[:,f,q]&~wi[:,f,q]
                if (digit,name) in failed:
                    for sync in ts[neither]:
                        exposure_flags.setdefault(int(sync),[]).append(digit+'_'+name)
                rec.update(comparable_both=int(both.sum()),neither_supports=int(neither.sum()),first_unsupported_syncs=ts[neither][:8].tolist())
                paired[digit+'_'+name]=rec
        scores[h]=paired
        suspect[h]=[dict(sync=t,items=items,exported_views=samples[side,t]) for t,items in sorted(exposure_flags.items())]
    out=out or root/'review';write_json(out/'same_exposure_anatomy.json',scores)
    from stages.fingerprint import digest
    write_json(out/'unsupported_exposures.json',dict(dataset_review_sha256=digest(root/'review/dataset_targets.json'),
        rule='failed aggregate item and per-pose event in final absent in both valid same-exposure SDK and triangulated WiLoR',hands=suspect))
    lines=[f'# {g} 最终导出手势的同曝光对照','','按手与同步帧去重，每个数字代表一个真实导出姿势。反折沿用 −15° 判据，折线沿用生产检查。WiLoR 取同曝光至少两个相机的三角化关节，NOKOV 取对应 SDK 骨架。参考关节缺失时不计入交叉核对。上游 soft 与 wide 取该段实际拟合。','',
        '| 手 | 关节 | 最终事件 / 姿势 | SDK 与 WiLoR 都可核对的事件 | 两者都未出现同类事件 | 代表帧 |','|---|---|---:|---:|---:|---|']
    for hand,data in scores.items():
        for failure in final_meta[hand]['failed_19']:
            key=failure['digit']+'_'+failure['joint'];v=data[key]
            lines.append(f"| {hand} | {key} | {v['final']} / {v['poses']} | {v['comparable_both']} | {v['neither_supports']} | {', '.join(map(str,v['first_unsupported_syncs'][:4]))} |")
    lines+=['','按手和同步帧去重，本段未通过的检查项中，两种可用参照都未出现同类事件的次数如下。每个姿势的所有导出相机均计入样本数。']
    for h,v in suspect.items():
        lines.append(f"- {h} 共 {len(v)} 个姿势，对应 {sum(x['exported_views'] for x in v)} 份相机样本。")
    lines+=['','[有分歧的具体帧、手指和视角数](unsupported_exposures.json) · [逐关节原始计数和阶段对照](same_exposure_anatomy.json)']
    (out/'ANATOMY_CHECK.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root.resolve())
