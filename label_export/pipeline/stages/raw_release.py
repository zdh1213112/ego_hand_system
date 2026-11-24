"""Index and validate a newly fitted recording without reading a signed label as input."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .signed import rows,arrays,write_json
from .merge_verified import rejection_rows
from .regression import key

def lines(path,values):
    tmp=path.with_suffix('.partial')
    with tmp.open('w') as f:
        for r in values:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(path)

def assemble(root,g):
    src=root/'geometry_input'/g
    paired=src/'paired_input'
    original={key(r):r for r in rows(paired/'index.jsonl')}
    summary=json.loads((paired/'summary.json').read_text())
    data=[];frames={};sides=Counter()
    exposure=root/'exposure_index.jsonl'
    for record in rows(exposure if exposure.is_file() else root/'index.jsonl'):
        s=record['side'];frame=record['image'];r={**original.get(key(record),{}),**record}
        r.update(physical_side=s,handedness=('left','right')[s],stored_image_space='native_raw',stored_image_horizontally_flipped=False,mano_parameter_space='right_canonical',pass_surface=True,isolated_pose=False)
        if frame not in frames:
            frames[frame]={k:r[k] for k in ('image','camera','sync_index','reference_timestamp_ns','source_frame_index','source_group')};frames[frame]['hands']=[]
        frames[frame]['hands'].append({k:r[k] for k in ('index','label','side')})
        sides[r['handedness']]+=1;data.append(r)
    if not data:raise ValueError(f'{g} 无有效样本')
    accepted={key(r) for r in data};assert len(accepted)==len(data)
    for f in frames.values():f['hand_count']=len(f['hands'])
    tmp=src/'reference';tmp.mkdir(exist_ok=True)
    # The same rejection checker uses only the camera image size from this summary.
    write_json(tmp/'summary.json',summary)
    rejected=list(rejection_rows(src,root,accepted))
    lines(root/'index.jsonl',data);lines(root/'index_with_glove.jsonl',(dict(r,label=r['glove_label']) for r in data))
    lines(root/'frames.jsonl',frames.values());lines(root/'rejected.jsonl',rejected)
    times=sorted({r['sync_index'] for r in data})
    if len(times)<2:raise ValueError('有效时间太少，无法时间隔离验证')
    import yaml
    params=yaml.safe_load((Path(__file__).resolve().parents[1]/'defaults.yaml').read_text())
    params.update(json.loads((root/'records/raw_case.json').read_text()).get('parameters',{}))
    cut=times[min(len(times)-1,max(1,round(len(times)*params['train_ratio'])))]
    index={}
    for name in ('train','val'):
        rec=[r for r in data if (r['sync_index']<cut)==(name=='train')]
        index[name]=[r['index'] for r in rec];lines(root/(name+'.jsonl'),rec)
    splits=dict(source=str(root/'index.jsonl'),source_positions=index,val_starts_at={g:cut},holdout_by_time=True,split_seed=params['split_seed'])
    write_json(root/'splits.json',splits)
    summary.update(sample_count=len(data),label_count=len(data),image_count=len(frames),side_counts=dict(sides),camera_counts=dict(Counter(r['camera'] for r in data)),rejected_count=len(rejected),annotation_policy='accepted_hand_instances',approved_for_training=False,fit_method='legacy')
    write_json(root/'summary.json',summary)

def review(root,g):
    import sys
    from .legacy.raw_export import setup
    case=setup(root/'records/raw_case.json')
    from observations import geometry
    from .legacy.review import board,sha
    from anatomy_common import signed_flex,anatomy_stats,REFERENCE
    from .legacy.solvers.evaluate import gate_anatomy
    src=root/'geometry_input'/g;dest=root/'review';dest.mkdir(exist_ok=True)
    native,cams,rt,frames=geometry(g);meta={};refs=json.loads((case.assets/'wilor_reference.json').read_text())[REFERENCE]
    with np.load(root/'sequence.npz') as z:
        data={k:z[k] for k in ('pose_valid','pass_surface','isolated_pose','joints_3d_glove')}
    for s,h in enumerate(('left','right')):
        d=arrays(src/'tracks'/(h+'.npz'),'pair_indices','joints','target24','marker_valid')
        ts=d['pair_indices'];good=data['pass_surface'][ts,s]&~data['isolated_pose'][ts,s]
        flex=signed_flex(d['joints'].astype(float),True)
        a=anatomy_stats(flex);failed=gate_anatomy(a,refs[h])
        surface=anatomy_stats(signed_flex(data['joints_3d_glove'][ts,s].astype(float),bool(s)))
        surface_failed=gate_anatomy(surface,refs[h])
        selected=np.flatnonzero(good)
        for k,i in enumerate(selected[np.linspace(0,len(selected)-1,min(8,len(selected))).round().astype(int)]):
            t=int(ts[i]);markers=np.where(d['marker_valid'][i,:,None],d['target24'][i],np.nan)
            glove=data['joints_3d_glove'][t,s]*[1 if s else -1,1,1]
            rgb=board(native,cams,rt,frames,t,s,d['joints'][i],markers,h,glove=glove)
            import cv2
            image=dest/f'{h}_{k+1:02}_sync{t}.jpg'
            if not cv2.imwrite(str(image),rgb):raise IOError(image)
        meta[h]=dict(valid=len(ts),passed=int(data['pass_surface'][ts,s].sum()),anatomy=a,surface_anatomy=surface,surface_failed_19=[x for x in surface_failed if (x['digit'],x['joint'])!=('thumb','CMC')],failed_19=[x for x in failed if (x['digit'],x['joint'])!=('thumb','CMC')])
    with (src/'decisions/photo.csv').open() as f:
        import csv
        per_side={h:Counter() for h in ('left','right')}
        for r in csv.DictReader(f):per_side[('left','right')[int(r['side'])]][r['decision']]+=1
        counts=per_side['left']+per_side['right']
    audit=root/'export_sha256.csv'
    if not audit.exists():fingerprint_export(root)
    meta.update(visibility=dict(counts),dataset_sha256={n:sha(root/n) for n in ('index.jsonl','index_with_glove.jsonl','frames.jsonl','rejected.jsonl','train.jsonl','val.jsonl','splits.json','summary.json','sequence.npz','export_sha256.csv','loader_check.json')})
    ref=json.loads((case.assets/'signed_anatomy_metrics.json').read_text())['groups']
    parts=['| 来源 | 手 | 有效姿势 | 表面通过率 | MANO 解剖通过 | MANO 反折率 | MANO 之字形率 |','|---|---|---:|---:|---:|---:|---:|']
    for h in ('left','right'):
        v=meta[h];a=v['anatomy']
        parts.append(f"| 本次 | {h} | {v['valid']} | {100*v['passed']/v['valid']:.2f}% | {19-len(v['failed_19'])}/19 | {np.mean(a['hyper_percent']):.2f}% | {np.mean(a['zig_percent']):.2f}% |")
    for group,values in ref.items():
        for h,v in values.items():
            a=v['mano'];parts.append(f"| 签收 {group} | {h} | {v['valid']} | {100*v['pass_surface']/v['valid']:.2f}% | {a['anatomy_passed']}/19 | {a['hyper_percent']:.2f}% | {a['zig_percent']:.2f}% |")
    parts+=['','| 来源 | 手 | 表面点解剖通过 | 表面连线反折率 | 表面连线之字形率 | 单手遮挡剔除 / 检查 |','|---|---|---:|---:|---:|---:|']
    for h in ('left','right'):
        v=meta[h];a=v['surface_anatomy'];occ=per_side[h]
        parts.append(f"| 本次 | {h} | {19-len(v['surface_failed_19'])}/19 | {np.mean(a['hyper_percent']):.2f}% | {np.mean(a['zig_percent']):.2f}% | {occ['exclude']} / {sum(occ.values())} |")
    for group,values in ref.items():
        for h,v in values.items():
            a=v['glove']
            parts.append(f"| 签收 {group} | {h} | {a['anatomy_passed']}/19 | {a['hyper_percent']:.2f}% | {a['zig_percent']:.2f}% | {v['single_rejected']} / {v['single_checked']} |")
    parts+=['','反折率为五指各三个位置的平均触发比例，之字形率为五指的平均触发比例。19 项检查沿用已签收的逐关节门槛，其中拇指 CMC 单独记录。']
    meta['review_image_sha256']={p.name:sha(p) for h in ('left','right') for p in sorted(dest.glob(h+'_[0-9]*.jpg'))}
    meta['signed_baselines']=ref
    write_json(dest/'dataset_targets.json',meta)
    sign=dest/'dataset_signoff.json'
    if not sign.exists() or json.loads(sign.read_text()).get('approved') is False:
        write_json(sign,dict(approved=False,sha256=sha(dest/'dataset_targets.json')))
    alias=root.parents[1]/'dataset'
    if not alias.exists():alias.symlink_to(root.relative_to(root.parents[1]),target_is_directory=True)
    (dest/'DATASET_REVIEW.md').write_text('# 整段标签签收\n\n每手按时间抽取八个已通过表面检查的姿势。每格左列原图，右列叠图。彩线为 MANO，青圈为实测球心，橙十字为手套挂点，黄菱形为手套显示腕点，紫十字为 MANO 腕关节中心。\n\n'+ '\n\n'.join(' · '.join(f'[{p.stem}]({p.name})' for p in sorted(dest.glob(h+'_[0-9]*.jpg'))) for h in ('left','right'))+'\n\n'+'\n'.join(parts)+'\n\n[解剖、遮挡和目标摘要](dataset_targets.json) · [几何校验](../geometry_check.log) · [训练读取校验](../loader_check.json)\n')

def fingerprint_export(root):
    import csv
    from .fingerprint import digest
    audit=root/'export_sha256.csv';partial=audit.with_suffix('.part')
    with partial.open('w') as f:
        w=csv.writer(f)
        for directory in ('images','labels','labels_with_glove'):
            for p in sorted((root/directory).iterdir()):w.writerow([digest(p),str(p.relative_to(root))])
    partial.replace(audit)
    return audit
