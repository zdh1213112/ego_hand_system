"""Session-level VIO reports from the same normalized results as the viewer."""
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from ego_data.vio_results import aggregate
from ego_data.fixed_relations import FixedReference
from pico_sync_quality import full_motion_quality,motion_window_quality


def table(headers,rows):
    esc=lambda v:html.escape(str(v))
    return '<div class="table-wrap"><table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def supported(times,query,radius=0.,max_gap=.05):
    times=np.asarray(times);query=np.asarray(query)
    lo=np.clip(np.searchsorted(times,query-radius,side='right')-1,0,len(times)-1)
    hi=np.clip(np.searchsorted(times,query+radius,side='left'),0,len(times)-1)
    bad=np.r_[0,np.cumsum(np.diff(times)>max_gap)]
    return (query-radius>=times[0])&(query+radius<=times[-1])&(bad[hi]-bad[lo]==0)


def angular_signal(times,rotations):
    grid=np.arange(times[0]+.036,times[-1]-.036,.02)
    spline=Slerp(times,rotations);h=.035
    speed=(spline(grid-h).inv()*spline(grid+h)).magnitude()/(2*h)
    speed[~supported(times,grid,h)]=np.nan
    return grid,speed


def motion_diagnostic(result,run):
    times=run['time'];pg,pw=angular_signal(times,Rotation.from_matrix(run['poses'][:,:3,:3]))
    sensor=result['sensor'];origin=result['info']['first_timestamp_ns']
    ref=FixedReference({'nokov_directory':sensor['reference_directory'],'rigid_body':sensor['rigid_body'],
        'T_marker_imu':sensor['T_marker_imu'],'nokov_to_imu_offset_s':sensor['mocap_to_imu_offset_s']},origin)
    ng,nw=angular_signal(ref.times,ref.slerp(ref.times))
    support=lambda q,r:supported(times,q,r+.036)&supported(ref.times,q,r+.036)
    whole=full_motion_quality(pg,pw,ng,nw,1.,0.,support,
        estimate_source=str(run['source']),reference_source=str(Path(sensor['reference_directory'])/'nokov_rigid_bodies.csv'))
    # The shared helper retains its legacy field for existing PICO reports.
    whole['estimate_source']=whole.pop('pico_source')
    windows=motion_window_quality(pg,pw,ng,nw,1.,0.,support)
    return {'whole':whole,'windows':windows}


def write_session_report(output,results,installation=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metric_rows=[];records=[]
    metrics={'position_rmse_m':'stats.all.position_rmse_m','position_p95_m':'stats.all.position_p95_m',
        'position_max_m':'stats.all.position_max_m','orientation_rmse_deg':'stats.all.orientation_rmse_deg',
        'orientation_p95_deg':'stats.all.orientation_p95_deg','holdout_position_rmse_m':'stats.holdout.position_rmse_m',
        'holdout_position_p95_m':'stats.holdout.position_p95_m','holdout_orientation_rmse_deg':'stats.holdout.orientation_rmse_deg',
        'rpe_1s_translation_m':'stats.rpe_1s_holdout.translation_m.rmse',
        'rpe_1s_rotation_deg':'stats.rpe_1s_holdout.rotation_deg.rmse','wall_s':'stats.performance.process_wall_s',
        'fps':'stats.performance.throughput_fps','backend_p95_ms':'stats.performance.backend_frame_ms.p95',
        'peak_rss_mib':'stats.performance.peak_rss_mib'}
    for source in results:
        record={'session':source['label'],'directory':str(source['directory']),'format':source['format'],
            'input_poses':source['info']['prepared_groups'],'duration_s':source['info']['duration_s'],
            'registration_interval_s':source['registration_interval_s'],'calibration_end_s':source['calibration_end_s'],
            'camera_shift_ns':source['info']['camera_time_shift_ns'],
            'nokov_to_imu_offset_s':source['sensor']['mocap_to_imu_offset_s'],
            'T_B_I':source['sensor']['T_marker_imu'],'cases':{},'sources':source['sources']}
        for name,case in source['cases'].items():
            runs=case['runs'];first=runs[0]
            entry={'label':case['label'],'repeats':len(runs),'scored_poses':len(first['time']),
                'holdout_poses':first['stats']['holdout']['poses'],'reference_coverage_pct':first['reference_coverage_pct'],
                'scored_interval_s':[float(first['time'][0]),float(first['time'][-1])],
                'aggregate':{key:aggregate(runs,path) for key,path in metrics.items()},
                'marginalization_cache_written':first['stats']['performance']['marginalization_cache_written'],
                'motion_time_quality':motion_diagnostic(source,first)}
            record['cases'][name]=entry
            metric_rows.append({'session':source['label'],'case':name,'repeats':len(runs),
                'input_poses':record['input_poses'],'scored_poses':entry['scored_poses'],'holdout_poses':entry['holdout_poses'],
                'reference_coverage_pct':entry['reference_coverage_pct'],
                **{key:value['median'] for key,value in entry['aggregate'].items()},
                'marginalization_cache_written':entry['marginalization_cache_written']})
        records.append(record)
    payload={'sessions':records,'installation_difference':installation,
        'protocol':'Recorded image/IMU timestamps; camera/IMU offset fixed at 0 ms. Saved estimates and saved transforms; scale=1; no sensor or world registration refitted by this report. All-overlap and post-calibration metrics are separate; aggregates are medians of repeat metrics, ranges are min/max.',
        'timing_note':'Timing includes PNG reading, optical flow, VIO and configured outputs; excludes input preparation and mapping. Check marginalization-cache settings before comparing costs.'}
    (output/'vio_effects_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (output/'vio_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    plot_sessions(output,results,records)
    fmt=lambda v,n=3:'—' if v is None else f'{v:.{n}f}'
    def value(entry,key,factor=1):
        a=entry['aggregate'][key]
        return fmt(a['median']*factor) if a['median'] is not None else '—'
    def interval(entry,key,factor=1):
        a=entry['aggregate'][key]
        return f"{value(entry,key,factor)} [{fmt(a['min']*factor)}, {fmt(a['max']*factor)}]" if a['median'] is not None else '—'
    rows=[];held=[];performance=[];sync=[]
    for session in records:
        for case,entry in session['cases'].items():
            row=[session['session'],entry['label']]
            rows.append(row+[entry['repeats'],entry['scored_poses'],fmt(entry['reference_coverage_pct'],2)+'%',
                interval(entry,'position_rmse_m',1000),value(entry,'position_p95_m',1000),value(entry,'position_max_m',1000),
                value(entry,'orientation_rmse_deg'),value(entry,'orientation_p95_deg')])
            held.append(row+[entry['holdout_poses'],interval(entry,'holdout_position_rmse_m',1000),
                value(entry,'holdout_position_p95_m',1000),interval(entry,'holdout_orientation_rmse_deg'),
                value(entry,'rpe_1s_translation_m',1000),value(entry,'rpe_1s_rotation_deg')])
            performance.append(row+[interval(entry,'wall_s'),interval(entry,'fps'),value(entry,'backend_p95_ms'),
                value(entry,'peak_rss_mib'),'写入' if entry['marginalization_cache_written'] else '未写入'])
            q=entry['motion_time_quality'];whole=q['whole'];window=q['windows']
            sync.append(row+[fmt(whole['residual_offset_ms']),whole['status'],
                f"{window['accepted_windows']} / {window['total_windows']}",fmt(window['residual_summary']['mean_abs_ms']),
                fmt(window['residual_summary']['max_abs_ms'])])
    links=[]
    for source in results:
        directory=source['directory'];viewer=directory/'viewer.html'
        if viewer.exists():links.append(f'<a href="{html.escape(os.path.relpath(viewer,output),quote=True)}">{html.escape(source["label"])} 单 session 交互轨迹</a>')
    interpretations=[]
    for session in records:
        cases=session['cases'];best=min(cases,key=lambda name:cases[name]['aggregate']['holdout_position_rmse_m']['median'])
        sentence=f"Session {session['session']}：后段位置 RMSE 最低的是 {cases[best]['label']}，为 {cases[best]['aggregate']['holdout_position_rmse_m']['median']*1000:.3f} mm。"
        if 'monado_2cam' in cases and 'monado_4cam' in cases:
            two=cases['monado_2cam']['aggregate']['holdout_position_rmse_m']['median']
            four=cases['monado_4cam']['aggregate']['holdout_position_rmse_m']['median']
            change=(four/two-1)*100
            sentence+=f" Monado 四目相对自身双目的位置 RMSE {'升高' if change>0 else '降低'} {abs(change):.1f}%。"
        interpretations.append('<li>'+html.escape(sentence)+'</li>')
    content=['<h1>Basalt VIO 与 NOKOV：逐 session 算法效果对比</h1>',
        '<p>图像与 IMU 时间修正统一固定为 0 ms，直接使用导出时间戳。复用各 session 对应零偏移实验的 VIO、NOKOV 时钟、安装外参与世界配准；本报告生成过程不运行 VIO。空间尺度固定为 1。表内为各次运行指标的中位数，方括号为最小值与最大值。单 session 查看器与曲线使用第 1 次运行。</p>',
        '<p>'+' · '.join(links)+'</p>',
        '<h2>整段参考有效样本性能</h2>',
        table(['Session','配置','重复数','参考有效位姿','参考覆盖率','位置 RMSE / mm [范围]','位置 P95 / mm','位置最大 / mm','姿态 RMSE / °','姿态 P95 / °'],rows),
        '<h2>标定后区间：主要比较指标</h2>',
        '<p>主要评价区间为各段传感器标定结束后且 NOKOV 有效的部分：'+html.escape('；'.join(f"Session {s['session']}：t≥{s['calibration_end_s']:g} s" for s in records))+'。该区间与传感器标定、前段世界配准窗口分开。整段指标包含标定样本，因此两张表应分别解读。</p>',
        table(['Session','配置','后段位姿','位置 RMSE / mm [范围]','位置 P95 / mm','姿态 RMSE / ° [范围]','1 s RPE 平移 / mm','1 s RPE 姿态 / °'],held),
        '<h2>效果对比解读</h2><ul>'+''.join(interpretations)+'</ul><p>同一段内部共享时间、安装标定和图像帧组；不同段的运动、场景观测和采样缺口各异，跨 session 差异不构成单一因素的因果结论。</p>',
        '<img src="session_errors_and_sync.png" alt="两组录制的误差和同步曲线">',
        '<h2>各 session 的轨迹对比</h2><p>每一列属于一段录制，三种算法与 NOKOV 使用同一首个有效参考 IMU 坐标系；显示坐标转换不改变误差。两段运动不同，不将它们当作同一条轨迹叠合。</p>',
        '<img src="session_trajectories.png" alt="不同录制中的三种算法轨迹">',
        '<h2>处理性能与计时范围</h2>',
        table(['Session','配置','VIO 耗时 / s [范围]','吞吐 / 帧组·s⁻¹ [范围]','后端 P95 / ms','峰值内存 / MiB','边缘化缓存'],performance),
        '<p>耗时包含图像读取、光流、后端和输出，排除编译、视频转换与建图。边缘化缓存是否写盘见表；若设置不同，应结合写盘成本解读。跨 session 的运动与场景不同，耗时差不能直接解释为算法速度变化。后端耗时也不是传感器到位姿的在线延迟。</p>',
        '<h2>坐标关系与安装一致性</h2><p>N：NOKOV 世界；V：每段 VIO 的局部世界；B：头环刚体；I：IMU。T_A_B 将 B 坐标映射到 A，矩阵平移单位 m。</p>',
        '<pre>T_N_B(t_N) · T_B_I = T_N_V · T_V_I(t_I)\nt_I = t_N + δ</pre>']
    if installation:
        content.append(table(['关系','安装平移偏差 / mm','安装旋转偏差 / °'],[['T_B_I',fmt(installation['translation_norm_mm']),fmt(installation['rotation_geodesic_deg'],4)]]))
    for session in records:
        source=next(s for s in results if s['label']==session['session'])
        first=next(iter(source['cases'].values()))['runs'][0]
        content.extend([f'<h3>Session {html.escape(session["session"])}</h3>',
            f'<p>原始输出 {session["input_poses"]} 帧，时长 {session["duration_s"]:.3f} s；配准窗口 {session["registration_interval_s"]} s；图像—IMU 时间修正固定 0 ms；NOKOV→IMU 偏移 {session["nokov_to_imu_offset_s"]*1000:.6f} ms。</p>',
            f'<p>参考覆盖 {first["time"][0]:.3f}–{first["time"][-1]:.3f} s；参考运动路程 {first["stats"]["all"]["reference_path_length_m"]:.3f} m。<a href="{html.escape(os.path.relpath(source["directory"]/"comparison_summary.json",output),quote=True)}">源评价指标与各次配准</a></p>',
            '<p>T_B_I：</p><pre>'+html.escape(np.array2string(np.array(session['T_B_I']),precision=8,suppress_small=True))+'</pre>'])
    content.extend(['<h2>对齐后剩余运动时间差</h2>',
        '<p>对比第 1 次 VIO 姿态与校时后 NOKOV 头环姿态的角速度模长，估计整段及约 8 秒窗口内的额外时间调整。正值表示匹配稍晚的 NOKOV 运动会更重合。结果没有应用到任何时钟或误差计算；它包含 VIO 滤波、预测和姿态噪声影响，不能当作硬件同步误差。运动不足、相关性低或最优值到达搜索边界的窗口单独标记。</p>',
        table(['Session','配置','整段剩余时间差 / ms','状态','有效窗口 / 全部','窗口平均错位大小 / ms','窗口最大错位大小 / ms'],sync),
        '<img src="session_time_residuals.png" alt="分段运动时间残差">',
        '<h2>证据范围</h2><ul><li>各段运动、真值覆盖范围与配准窗口不同，跨 session 数值用于描述这两次录制，不能当成相同测试路线上的控制实验。</li><li>NOKOV 头环刚体已按各段既有安装关系换算到 IMU 中心；安装关系来自本段原始传感器，并非独立测量真值。</li><li>当前沿用各源报告的有效参考范围与数值检查，没有借用 PICO 的 marker 质量门，也没有根据 VIO/NOKOV 残差删除样本。</li><li>世界坐标变换仅用于各段轨迹评价；跨 session 安装一致性只比较 T_B_I。</li><li>自动适配原版重复运行、部分真值覆盖重复运行和固定标定完整运行结果格式；读取时验证保存的配准能够复现已有精度指标。全部来源保留 SHA-256。</li></ul>',
        '<p><a href="vio_effects_summary.json">完整数值 JSON</a> · <a href="vio_metrics.csv">逐 session 指标 CSV</a></p>'])
    style='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Basalt VIO / NOKOV 逐 session 对比</title><style>body{max-width:1500px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#182637}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd6e1;padding:8px;text-align:left;white-space:nowrap}th{background:#edf2f7}.table-wrap{overflow:auto}img{max-width:100%;margin:20px 0}pre{background:#eef2f6;padding:14px;overflow:auto}h2{margin-top:36px}li{margin:10px 0}</style>'
    (output/'report.html').write_text(style+'\n'.join(content)+'</html>',encoding='utf-8')
    return payload


def plot_sessions(output,results,records):
    os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'original_2cam':'Basalt 2','monado_2cam':'Monado 2','monado_4cam':'Monado 4'}
    fig,axes=plt.subplots(3,len(results),figsize=(7*len(results),10),squeeze=False,constrained_layout=True)
    for col,source in enumerate(results):
        for case,entry in source['cases'].items():
            run=entry['runs'][0];t=run['time'];colour=entry['colour']
            axes[0,col].plot(t,run['position_error_m']*1000,c=colour,lw=1,label=labels.get(case,case))
            axes[1,col].plot(t,run['orientation_error_deg'],c=colour,lw=1)
        for row in [0,1]:
            axes[row,col].axvline(source['calibration_end_s'],c='#555555',ls='--',lw=1);axes[row,col].grid(alpha=.2)
            axes[row,col].set(xlabel='Recording time [s]',ylabel=['Position error [mm]','Orientation error [deg]'][row],title=source['label'])
        axes[0,col].legend()
        sync=source['directory']/'nokov_sync_diagnostics.npz'
        if sync.exists():
            with np.load(sync) as d:
                t=d['time_mocap_s']+source['sensor']['mocap_to_imu_offset_s']
                axes[2,col].plot(t,d['angular_speed_mocap'],label='NOKOV',lw=1)
                axes[2,col].plot(t,d['angular_speed_imu'],label='IMU / saved clock',lw=1,alpha=.7)
            axes[2,col].legend()
        axes[2,col].set(xlabel='Recording time [s]',ylabel='Raw angular speed [rad/s]');axes[2,col].grid(alpha=.2)
    fig.savefig(output/'session_errors_and_sync.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(7*len(results),10),constrained_layout=True)
    for col,source in enumerate(results):
        first=next(iter(source['cases'].values()))['runs'][0];truth=first['reference'];origin=np.linalg.inv(truth[0])
        change=lambda p:p@origin[:3,:3].T+origin[:3,3]
        gt=change(truth[:,:3,3]);trajectories={k:change(v['runs'][0]['poses'][:,:3,3]) for k,v in source['cases'].items()}
        cloud=np.vstack([gt,*trajectories.values()]);center=(cloud.min(0)+cloud.max(0))/2;radius=max(.1,np.ptp(cloud,axis=0).max())*.55
        ax=fig.add_subplot(2,len(results),col+1,projection='3d');ax.plot(*gt.T,c='#007f73',ls='--',label='NOKOV')
        xy=fig.add_subplot(2,len(results),col+1+len(results));xy.plot(*gt[:,:2].T,c='#007f73',ls='--',label='NOKOV')
        for case,p in trajectories.items():
            ax.plot(*p.T,c=source['cases'][case]['colour'],lw=1,label=labels.get(case,case));xy.plot(*p[:,:2].T,c=source['cases'][case]['colour'],lw=1)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='I0 X [m]',ylabel='Y [m]',zlabel='Z [m]',title=source['label']);ax.set_box_aspect((1,1,1));ax.legend(fontsize=8)
        xy.set(xlabel='I0 X [m]',ylabel='Y [m]',title=source['label']+' / XY',aspect='equal');xy.grid(alpha=.2)
    fig.savefig(output/'session_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(results),figsize=(7*len(results),4),squeeze=False,constrained_layout=True)
    for col,(source,record) in enumerate(zip(results,records)):
        ax=axes[0,col]
        for case,entry in record['cases'].items():
            rows=entry['motion_time_quality']['windows']['windows'];accepted=[r for r in rows if r['status']=='ok']
            ax.plot([r['center_s'] for r in accepted],[r['residual_offset_ms'] for r in accepted],marker='o',lw=1,c=source['cases'][case]['colour'],label=labels.get(case,case))
        ax.axhline(0,c='#555555',ls='--',lw=1);ax.set(xlabel='Recording time [s]',ylabel='Extra NOKOV-side lag [ms]',title=source['label']);ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'session_time_residuals.png',dpi=150);plt.close(fig)
