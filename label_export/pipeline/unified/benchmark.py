"""Score unchanged exposure and marker identities against both candidate tracks."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from stages.legacy.raw_export import setup
from unified.optics import project


def load(path, *fields):
    with np.load(path) as data:
        return {k: data[k] for k in fields}


def summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(count=int(len(values)), available=len(finite),
                median=float(np.median(finite)) if len(finite) else None,
                p95=float(np.percentile(finite, 95)) if len(finite) else None)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def run(source, fixed, dest):
    case = dest/'metric_case.json'
    write(case, dict(id='benchmark', recording_output=str(dest), marker_placement='ego_hand_system/configs/glove_marker_placement.json'))
    setup(case)
    from anatomy_common import signed_flex, anatomy_stats, REFERENCE
    from common import isolated_poses
    from stages.legacy.solvers.evaluate import gate_anatomy
    refs = json.loads((PACKAGE/'assets/wilor_reference.json').read_text())[REFERENCE]
    detailed = []
    for group in ('group_003', 'group_005'):
        src = source/group
        cams = [json.loads((src/f'native/calibration/camera{c}.json').read_text()) for c in range(4)]
        T = np.array([c['T_base_camera'] for c in cams])
        rt = [(x[:3,:3].T@T[0,:3,:3], x[:3,:3].T@(T[0,:3,3]-x[:3,3])) for x in T]
        for side,h in enumerate(('left','right')):
            trackfile = dest/group/'tracks'/(h+'.npz')
            if not trackfile.exists():continue
            keys = ('joints','pair_indices','predicted24','pass_surface')
            base = load(src/'tracks'/(h+'.npz'), *keys)
            new = load(trackfile, *keys, 'calibration_mask', 'attachment_local_offsets_m', 'betas')
            outward = load(src/'wrist'/(h+'_outward.npz'), 'joints', 'pair_indices')
            np.testing.assert_array_equal(base['pair_indices'],outward['pair_indices'])
            mirror = np.array([1 if side else -1,1,1])
            lookup = {int(t):i for i,t in enumerate(new['pair_indices'])}
            with np.load(fixed/group/(h+'_fixed_score.npz')) as z:
                mm = np.full(len(z['legacy_mm']),np.nan)
                px = np.full(len(z['legacy_px']),np.nan)
                for dimension,k,errors in [('3d','mm',mm),('2d','px',px)]:
                    times,markers=z['time_'+dimension],z['marker_'+dimension]
                    ii = np.array([lookup.get(t,-1) for t in times]); good=ii>=0
                    points=new['predicted24'][ii,markers]*mirror
                    if k=='mm':mm[good]=np.linalg.norm(points[good]-z['observed_cam0_m'][good],axis=1)*1000
                    else:
                        for c in range(4):
                            mask=good&(z['camera']==c)
                            px[mask]=np.linalg.norm(project(points[mask],cams[c],rt[c])-z['observed_uv'][mask],axis=1)
                scores={}
                for name,marker_id in [('all',None),('index',8),('middle',12),('ring',16),('pinky',20)]:
                    scores[name]={}
                    for dim,k,errors in [('3d','mm',mm),('2d','px',px)]:
                        mask = np.ones(len(errors),bool) if marker_id is None else z['marker_'+dim]==marker_id
                        scores[name][k]={v:summary(x[mask]) for v,x in [('legacy',z['legacy_'+k]),('unified',errors)]}
                # Missing projections count as infinity against the signed baseline for the promotion decision.
                delta=np.median(np.where(np.isfinite(px),px,np.inf))-np.median(z['legacy_px'])
            common,ib,iu=np.intersect1d(base['pair_indices'],new['pair_indices'],return_indices=True)
            stat={}
            masks = (ib,iu)
            sets=[]
            for mi,(method,tr,display) in enumerate([('legacy',base,outward['joints'][:,:1]),('unified',new,new['joints'][:,:1])]):
                stat[method]={}
                good=tr['pass_surface']&~isolated_poses(tr['pair_indices'])
                sets.append(set(tr['pair_indices'][good].tolist()))
                for schema,x in [('mano',tr['joints']),('glove',np.concatenate([display,tr['predicted24'][:,4:]],axis=1))]:
                    stat[method][schema]={}
                    for population,y in [('full',x),('common',x[masks[mi]])]:
                        a=anatomy_stats(signed_flex(y.astype(float),True))
                        failed=[v for v in gate_anatomy(a,refs[h]) if (v['digit'],v['joint'])!=('thumb','CMC')]
                        stat[method][schema][population]=dict(passed=19-len(failed),failures=failed,
                            hyper_percent=float(np.mean(a['hyper_percent'])),zig_percent=float(np.mean(a['zig_percent'])),
                            per_joint_hyper_percent=a['hyper_percent'],per_digit_zig_percent=a['zig_percent'])
            row=dict(group=group,hand=h,held=scores,held_worst_delta_px=float(delta) if np.isfinite(delta) else None,
                     anatomy=stat,coverage=dict(legacy=len(sets[0]),unified=len(sets[1]),lost=len(sets[0]-sets[1]),gained=len(sets[1]-sets[0]),
                     legacy_tracks=len(base['pair_indices']),unified_tracks=len(lookup),common_solver_exposures=len(common)),
                     calibration_count=int(new['calibration_mask'].sum()),betas=new['betas'].tolist(),
                     heights_mm=(new['attachment_local_offsets_m'][:,2]*1000).tolist())
            if group=='group_003':
                assert not new['calibration_mask'][lookup[883]]
                i=int(np.where(base['pair_indices']==883)[0][0]);j=new['joints'][lookup[883],0]*mirror
                row['wrist_883']={}
                for name,x in [('signed_MANO_J0',base['joints'][i,0]),('signed_display',outward['joints'][i,0])]:
                    p=np.array([j,x*mirror]);row['wrist_883'][name]=dict(mm=float(np.linalg.norm(p[1]-p[0])*1000),
                        px=[summary([np.linalg.norm(np.diff(project(p,cam,r),axis=0))])['median'] for cam,r in zip(cams,rt)])
            row['passed']=bool(np.isfinite(delta) and delta<=.5 and stat['unified']['mano']['full']['passed']==19 and
                all(stat['unified'][schema]['common'][k]<=stat['legacy'][schema]['common'][k]+1e-8
                    for schema in ('glove','mano') for k in ('zig_percent','hyper_percent')))
            detailed.append(row)
    write(dest/'metrics.json',detailed)
    lines=['# 003 / 005 单腕点试算','',
        '固定留出的相机和 marker 从配准开始排除。三维以校正后的 NOKOV 为参考。覆盖按逐手同步时刻，已应用表面误差与孤帧规则，尚未计入相机裁切和图像遮挡。左右手均用同一套参数。','',
        '| 录制 / 手 | 留出亮点数 | 中位 px 旧 → 新 | P95 px 旧 → 新 | 留出三维中位 mm 旧 → 新 | MANO 解剖旧 → 新 | 有效手姿旧 → 新 | 丢失 / 新增 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return '缺测' if x is None else f'{x:.2f}'
    def xy(d,k):return ' → '.join(fmt(d[m][k]) for m in ('legacy','unified'))
    for r in detailed:
        z=r['held']['all'];a=r['anatomy'];v=r['coverage']
        lines.append(f"| {r['group']} / {r['hand']} | {z['px']['unified']['available']}/{z['px']['legacy']['count']} | {xy(z['px'],'median')} | {xy(z['px'],'p95')} | {xy(z['mm'],'median')} | {a['legacy']['mano']['full']['passed']} → {a['unified']['mano']['full']['passed']} / 19 | {v['legacy']} → {v['unified']} | {v['lost']} / {v['gained']} |")
    lines += ['','## 四指 MCP 留出亮点','','| 录制 / 手 | 食指 px | 中指 px | 无名指 px | 小指 px |','|---|---:|---:|---:|---:|']
    for r in detailed:lines.append('| '+r['group']+' / '+r['hand']+' | '+' | '.join(xy(r['held'][d]['px'],'median')+' ('+str(r['held'][d]['px']['unified']['available'])+')' for d in ('index','middle','ring','pinky'))+' |')
    lines+=['','## 同时有解的时刻','','| 录制 / 手 | 内部关节反折 % 旧 → 新 | 内部关节之字形 % 旧 → 新 | 手套表面反折 % 旧 → 新 | 手套表面之字形 % 旧 → 新 |','|---|---:|---:|---:|---:|']
    for r in detailed:lines.append('| '+r['group']+' / '+r['hand']+' | '+' | '.join(' → '.join(fmt(r['anatomy'][m][s]['common'][k]) for m in ('legacy','unified')) for s in ('mano','glove') for k in ('hyper_percent','zig_percent'))+' |')
    lines+=['','## 003 第 883 帧','','| 手 | 对照 | 三维 mm | cam0 px | cam1 px | cam2 px | cam3 px |','|---|---|---:|---:|---:|---:|---:|']
    for r in detailed:
        for k,v in r.get('wrist_883',{}).items():lines.append('| '+r['hand']+' | '+k+' | '+' | '.join(fmt(x) for x in [v['mm'],*v['px']])+' |')
    lines+=['','四机数值为数学投影，是否在画幅内以对应原图为准。','',
            '[分母、P95、各项未通过的解剖检查和校准参数](metrics.json) · [逐帧四机对照](overlays/INDEX.md) · [各机曝光去留](VIEW_COVERAGE.md)','',
            '通过数值门槛的手别 '+str(sum(r['passed'] for r in detailed))+' / '+str(len(detailed))+'。']
    (dest/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return detailed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--fixed',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source.resolve(),a.fixed.resolve(),a.output.resolve())
