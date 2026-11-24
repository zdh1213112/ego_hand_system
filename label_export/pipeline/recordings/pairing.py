"""Match actual recording intervals, then test independent head gyro motion and hand streams."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from mcap.reader import make_reader
import pandas as pd
import yaml

PACKAGE=Path(__file__).resolve().parents[1]
PROJECT=PACKAGE.parents[2]
HANDS=('left','right')

def load(p):
    return json.loads(p.read_text())

def evidence(ego,sessions,dest):
    mcap,=ego.glob('part0001/*.mcap')
    rec=ego.name.rsplit('_',1)[1]
    cap=load(next(ego.glob('meta*.json')))['recording']
    with mcap.open('rb') as f:
        summary=make_reader(f).get_summary()
    stat=summary.statistics
    start,end=stat.message_start_time,stat.message_end_time
    counts={ch.topic:stat.channel_message_counts.get(i,0) for i,ch in summary.channels.items()}
    overlaps=[(min(end,m['ended_unix_ns'])-max(start,m['started_unix_ns']),s) for s,m in sessions.items()]
    hits=sorted([(n,s) for n,s in overlaps if n>0],reverse=True)
    row=dict(id=rec,ego=str(ego),mcap=str(mcap),calibration=str(next(ego.glob('*calibration_camera.yaml'))),camera_counts={k:v for k,v in counts.items() if k.startswith('/camera/')},
             ego_status=cap['status'],ego_failure=cap.get('failure_message'),ego_time_ns=[start,end],eligible=False)
    if len(hits)!=1:
        return dict(row,reason=f'实际录制时间与 {len(hits)} 个候选会话重叠，未得到唯一配对')
    ns,session=hits[0];d=session/'nokov'
    row.update(session=str(session),overlap_s=ns/1e9,nokov_time_ns=[sessions[session]['started_unix_ns'],sessions[session]['ended_unix_ns']])
    actual={}
    for table,name,idx,flag in [('markers','markerset_name','marker_index','valid'),('skeleton_segments','skeleton_name','segment_id','valid_numeric')]:
        obs={h:Counter() for h in HANDS}
        for block in pd.read_csv(d/f'nokov_{table}.csv',usecols=[name,idx,flag],chunksize=150000):
            for (owner,i),count in block.loc[block[flag]==1].groupby([name,idx]).size().items():
                for h in HANDS:
                    if h in str(owner).lower():obs[h][int(i)]+=int(count)
        actual[table]={h:len(v) for h,v in obs.items()}
    row['actual_valid_stream_ids']=actual
    out=dest/'pairing'/rec
    out.mkdir(parents=True,exist_ok=True)
    with (out/'sync.log').open('w') as log:
        ret=subprocess.run([sys.executable,'-B',str(PACKAGE/'stages/synchronize.py'),'--ego-mcap',str(mcap),'--imu-topic','/imu','--nokov-csv',str(d/'nokov_rigid_bodies.csv'),'--rigid-body','orbbec_ego4','--nokov-time-field','device_timestamp_raw','--nokov-time-scale','0.001','--output-dir',str(out),'--no-plot'],stdout=log,stderr=subprocess.STDOUT)
    result=out/'imu_nokov_sync.json'
    sync=load(result) if not ret.returncode and result.exists() else {}
    row.update(sync=sync.get('confidence','failed'),corr=sync.get('quality',{}).get('peak_correlation'),offset_s=sync.get('time_mapping',{}).get('b_s'))
    reasons=[]
    if row['sync']!='strong':reasons.append('IMU–NOKOV 未通过原生空间对齐所需的 strong 门槛')
    if any(actual['markers'][h]!=24 or actual['skeleton_segments'][h]==0 for h in HANDS):reasons.append('有效手套 marker 或手部 SDK 骨架缺失')
    row.update(eligible=not reasons,reason='；'.join(reasons) or '双手观测齐全、同步通过')
    print(rec,row['sync'],row['corr'],actual,row['eligible'],flush=True)
    return row

def run(cfg):
    root=(PROJECT/cfg['source']).resolve();dest=(PROJECT/cfg['output_root']).resolve()
    dest.mkdir(exist_ok=True,parents=True)
    sessions={s:load(s/'nokov/capture_metadata.json') for s in sorted((root/'session_20261120').iterdir()) if s.is_dir()}
    egos=sorted((root/'EGO-4B_20261120').glob('EGO-*'))
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(lambda e:evidence(e,sessions,dest),egos))
    names=[r['session'] for r in records if r['eligible']]
    if len(set(names))!=len(names):raise ValueError('同一 NOKOV 会话发生重复配对')
    (dest/'pairing.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    table=['# 新录制配对与腕点签收','','以 MCAP 实际消息时间与 NOKOV 采集时间确定候选，再对 EGO 陀螺仪和头部刚体的角速度做互相关。手套点和 SDK 骨架按文件里的有效行统计。','','| EGO | NOKOV | 原始重叠 s | 同步 / 相关系数 | 有效 marker 左 / 右 | 结果 |','|---|---|---:|---|---|---|']
    for r in records:
        marker=r.get('actual_valid_stream_ids',{}).get('markers',{});c=r.get('corr')
        table.append(f"| {r['id']} | {Path(r.get('session','')).name} | {r.get('overlap_s',0):.1f} | {r.get('sync','—')} / {c if c is None else round(c,3)} | {marker.get('left',0)} / {marker.get('right',0)} | {r['reason']} |")
    table+=['','183959 的采集器报告四机包数不同。后续按真实曝光时间匹配，单列四机覆盖及未配对源帧。','', '[各段实际时间、包数与同步详情](pairing.json)']
    (dest/'PAIRING.md').write_text('\n'.join(table)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(yaml.safe_load(a.config.read_text()))
