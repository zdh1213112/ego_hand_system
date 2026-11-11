#!/usr/bin/env python3
"""Compare two saved EGO/NOKOV calibrations without changing or rerunning VIO.

Compare the physical installation transform and clock parameters. Cross-apply
them on each recording to quantify their evaluation effect. World transforms
are used only for trajectory alignment, never as calibration-deviation metrics.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_nokov_ego4 import summarize
from compare_basalt_ego4 import dump, sha256
from ego_data.fixed_relations import FixedReference, check_transform
from fuse_basalt_hand_trajectory import load_basalt_trajectory
from ego_data.vio_results import result_file
from ego_data.time_sync import require_zero_camera_offset

LABELS={"original_2cam":"Basalt 双目","monado_2cam":"Monado 双目","monado_4cam":"Monado 四目"}
VARIANTS={"own":"本次参数","other_mount":"替换安装外参","other_clock":"替换时钟偏移","other_both":"同时替换两者"}


def transform_difference(a,b):
    a=check_transform(a,"A");b=check_transform(b,"B")
    delta=np.linalg.inv(a)@b
    rotation=Rotation.from_matrix(delta[:3,:3])
    return {"delta_inv_A_times_B":delta.tolist(),
        "translation_B_minus_A_in_parent_mm":((b[:3,3]-a[:3,3])*1000).tolist(),
        "translation_norm_mm":float(np.linalg.norm(b[:3,3]-a[:3,3])*1000),
        "rotation_geodesic_deg":float(np.rad2deg(rotation.magnitude())),
        "rotation_vector_in_A_deg":np.rad2deg(rotation.as_rotvec()).tolist()}


def camera_pose(value):
    t=np.eye(4);t[:3,:3]=Rotation.from_quat([value[k] for k in ["qx","qy","qz","qw"]]).as_matrix()
    t[:3,3]=[value[k] for k in ["px","py","pz"]]
    return t


def load_result(directory):
    directory=directory.resolve()
    info=json.loads((directory/"input_validation.json").read_text())
    require_zero_camera_offset(info,str(directory))
    sensor=json.loads((directory/"nokov_calibration.json").read_text())
    summary=json.loads((directory/"comparison_summary.json").read_text())
    calibration_path=result_file(directory,'dataset_4cam/calibration.json','dataset_2cam/calibration.json','input_adapter/calibration.json')
    calibration=json.loads(calibration_path.read_text())
    require_zero_camera_offset(calibration,str(calibration_path))
    if 'results' in summary:
        runs={c:entry.get('runs',entry.get('repeats')) for c,entry in summary['results'].items()}
    else:
        summary=json.loads((directory/'summary.json').read_text())
        runs={c:[{'directory':str(directory/'runs'/c/'repeat_1'),
                  'T_nokov_vio':summary['cases'][c]['world_transform_loaded']}] for c in summary['cases']}
    origin=int(info["first_timestamp_ns"])
    protocol=summary.get("protocol",{})
    world_interval=protocol.get("world_registration_interval_s",next(iter(summary.get('world_alignment_intervals_s',{}).values()),[0.,2.]))
    scored_interval=protocol.get("scored_interval_s",[0.,info["duration_s"]])
    return {"directory":directory,"info":info,"sensor":sensor,"summary":summary,
        "calibration":calibration,"runs":runs,"origin":origin,
        "world_interval":world_interval,"scored_interval":scored_interval,"calibration_path":calibration_path}


def cross_apply(data,other):
    own=data["sensor"];foreign=other["sensor"]
    with result_file(data['directory'],'dataset_4cam/frame_map.csv','dataset_2cam/frame_map.csv').open() as f:
        expected=np.array([int(r["timestamp_ns"]) for r in csv.DictReader(f)],dtype=np.int64)
    times=(expected-data["origin"])*1e-9
    refs={};mask=(times>=data["scored_interval"][0])&(times<=data["scored_interval"][1])
    for variant in VARIANTS:
        mount=foreign["T_marker_imu"] if variant in ["other_mount","other_both"] else own["T_marker_imu"]
        clock=foreign["mocap_to_imu_offset_s"] if variant in ["other_clock","other_both"] else own["mocap_to_imu_offset_s"]
        ref=FixedReference({"nokov_directory":own["reference_directory"],"rigid_body":own["rigid_body"],
            "T_marker_imu":mount,"nokov_to_imu_offset_s":clock},data["origin"])
        refs[variant]=ref;mask&=(times>=ref.times[0])&(times<=ref.times[-1])
    stamps=expected[mask];times=times[mask]
    registration=(times>=data["world_interval"][0])&(times<=data["world_interval"][1])
    holdout=times>=own["training_interval_imu_s"][1]
    if registration.sum()<3 or holdout.sum()<30:raise ValueError("insufficient common coverage")
    queries={}
    for name,ref in refs.items():
        p,r,valid=ref.query(times)
        if not valid.all():raise ValueError("reference extrapolation forbidden")
        queries[name]=(p,r)
    baseline_p,baseline_r=queries["own"]
    differences={};plot={"times":times,"position_mm":{},"orientation_deg":{}}
    for name,(p,r) in queries.items():
        pe=np.linalg.norm(p-baseline_p,axis=1)*1000
        re=np.rad2deg((baseline_r.inv()*r).magnitude())
        differences[name]={"all_position_mm":summarize(pe),"all_orientation_deg":summarize(re),
            "holdout_position_mm":summarize(pe[holdout]),"holdout_orientation_deg":summarize(re[holdout])}
        plot["position_mm"][name]=pe;plot["orientation_deg"][name]=re
    vio={}
    for case in data['runs']:
        vio[case]={name:[] for name in VARIANTS}
        for run in data["runs"][case]:
            # Resolve inside the requested result directory, even after moves.
            repetition=Path(run["directory"]).name
            path=data["directory"]/"runs"/case/repetition/"trajectory.csv"
            raw=load_basalt_trajectory(path)
            if not np.array_equal(np.array(sorted(raw),dtype=np.int64),expected):
                raise ValueError(f"VIO timestamps differ from prepared inputs: {path}")
            poses=np.array([raw[int(t)] for t in stamps]);p=poses[:,:3,3];r=Rotation.from_matrix(poses[:,:3,:3])
            for variant,(gp,gr) in queries.items():
                if variant=='own':
                    from ego_data.vio_results import saved_world
                    transform=saved_world(run,{})
                    wr=Rotation.from_matrix(transform[:3,:3]);wt=transform[:3,3]
                else:
                    wr=(gr[registration]*r[registration].inv()).mean()
                    wt=np.mean(gp[registration]-wr.apply(p[registration]),axis=0)
                pe=np.linalg.norm(wr.apply(p)+wt-gp,axis=1)
                re=np.rad2deg((gr.inv()*wr*r).magnitude())
                result={"repeat":repetition,"holdout_position_rmse_cm":float(np.sqrt(np.mean(pe[holdout]**2))*100),
                        "holdout_orientation_rmse_deg":float(np.sqrt(np.mean(re[holdout]**2))),
                        "all_position_rmse_cm":float(np.sqrt(np.mean(pe**2))*100),
                        "all_orientation_rmse_deg":float(np.sqrt(np.mean(re**2)))}
                vio[case][variant].append(result)
        for variant in VARIANTS:
            runs=vio[case][variant]
            vio[case][variant]={"runs":runs,
                "position_rmse_cm_median":float(np.median([r["holdout_position_rmse_cm"] for r in runs])),
                "orientation_rmse_deg_median":float(np.median([r["holdout_orientation_rmse_deg"] for r in runs]))}
        baseline=vio[case]["own"]
        for variant in VARIANTS:
            entry=vio[case][variant]
            entry["position_rmse_change_cm"]=entry["position_rmse_cm_median"]-baseline["position_rmse_cm_median"]
            entry["orientation_rmse_change_deg"]=entry["orientation_rmse_deg_median"]-baseline["orientation_rmse_deg_median"]
        # Check this diagnostic reproduces the original baseline protocol.
        if 'results' not in data['summary']:
            saved=data['summary']['cases'][case]['all']
            actual_p=np.median([r['all_position_rmse_cm'] for r in baseline['runs']])
            actual_r=np.median([r['all_orientation_rmse_deg'] for r in baseline['runs']])
            if not np.isclose(actual_p,saved['position_rmse_m']*100,atol=1e-8,rtol=0) or not np.isclose(actual_r,saved['orientation_rmse_deg'],atol=1e-8,rtol=0):
                raise ValueError(f"frozen baseline does not reproduce saved evaluation: {case}")
            continue
        saved=data["summary"]["results"][case]
        expected_p=saved["aggregate"]["holdout_position_rmse_m"]["median"]*100 if "aggregate" in saved else saved["holdout_position_rmse_m_median"]*100
        expected_r=saved["aggregate"]["holdout_orientation_rmse_deg"]["median"] if "aggregate" in saved else saved["holdout_rotation_rmse_deg_median"]
        if not np.isclose(baseline["position_rmse_cm_median"],expected_p,atol=1e-8,rtol=0) or not np.isclose(baseline["orientation_rmse_deg_median"],expected_r,atol=1e-8,rtol=0):
            raise ValueError(f"baseline does not reproduce saved evaluation: {data['directory']} / {case}")
    return {"scored_poses":len(stamps),"holdout_poses":int(holdout.sum()),
        "world_registration_interval_s":data["world_interval"],"reference_differences":differences,"vio":vio,
        "protocol":"Same VIO outputs and sample times; replace saved sensor parameters only, refit the original prefix world transform, preserve scale 1; no VIO rerun"},plot


def analyze(a,b):
    A=np.array(a["sensor"]["T_marker_imu"]);B=np.array(b["sensor"]["T_marker_imu"])
    mount=transform_difference(A,B)
    markers=[]
    for key in sorted(a["sensor"]["rigid_marker_consistency"].keys()&b["sensor"]["rigid_marker_consistency"].keys()):
        pa=np.array(a["sensor"]["rigid_marker_consistency"][key]["local_position_m"])
        pb=np.array(b["sensor"]["rigid_marker_consistency"][key]["local_position_m"])
        markers.append({"marker_index":key,"delta_B_minus_A_mm":((pb-pa)*1000).tolist(),
                        "distance_mm":float(np.linalg.norm(pb-pa)*1000)})
    cameras=[]
    for i,(ca,cb) in enumerate(zip(a["calibration"]["value0"]["T_imu_cam"],b["calibration"]["value0"]["T_imu_cam"])):
        camera_a,camera_b=camera_pose(ca),camera_pose(cb)
        cameras.append({"camera":i,"camera_to_imu_difference":transform_difference(camera_a,camera_b),
            "camera_to_marker_difference":transform_difference(A@camera_a,B@camera_b)})
    crossed_a,plot_a=cross_apply(a,b);crossed_b,plot_b=cross_apply(b,a)
    bound=[]
    for distance_m in [0,.5,1.,2.]:
        bound.append({"point_distance_from_imu_m":distance_m,
            "position_difference_bound_mm":mount["translation_norm_mm"]+2*np.sin(np.deg2rad(mount["rotation_geodesic_deg"])/2)*distance_m*1000})
    return {"definitions":{"A":str(a["directory"]),"B":str(b["directory"]),
        "session_labels":{key:Path(data['info']['session']).name.rsplit('_',1)[-1] for key,data in [('A',a),('B',b)]},
        "T_A_B":"Maps coordinates in B into A; T_marker_imu maps IMU coordinates into marker coordinates",
        "pose_chain":"T_N_I(t_IMU) = T_N_M(t_IMU - clock_offset) @ T_M_I",
        "difference":"Delta = inverse(T_A) @ T_B; signed translation subtraction is expressed in the common parent frame",
        "interpretation":"Differences between two estimates measure consistency, not absolute error against a surveyed transform"},
        "sensor_A":a["sensor"],"sensor_B":b["sensor"],"installation_difference":mount,
        "clock_difference_B_minus_A_ms":(b["sensor"]["mocap_to_imu_offset_s"]-a["sensor"]["mocap_to_imu_offset_s"])*1000,
        "extra_camera_shift_ms_A_B":[a["info"]["camera_time_shift_ns"]/1e6,b["info"]["camera_time_shift_ns"]/1e6],
        "camera_calibrations_identical":a["calibration"]==b["calibration"],
        "common_configs_identical":{family:json.loads((a["directory"]/f"config_{family}.json").read_text())==json.loads((b["directory"]/f"config_{family}.json").read_text()) for family in ["original","monado"]},
        "marker_template_comparison":markers,"camera_comparison":cameras,
        "same_clock_point_displacement_upper_bound":bound,"cross_apply":{"A":crossed_a,"B":crossed_b}},[plot_a,plot_b]


def figures(output,result,plots):
    os.environ["MPLCONFIGDIR"]=str(output/".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    colors={"other_mount":"#2563eb","other_clock":"#d97706","other_both":"#059669"}
    for column,(label,plot) in enumerate(zip([Path(result['definitions'][key]).name for key in ['A','B']],plots)):
        for row,(field,unit) in enumerate([("position_mm","Position difference [mm]"),("orientation_deg","Orientation difference [deg]")]):
            ax=axes[row,column]
            for variant,caption in [("other_mount","Other installation"),("other_clock","Other clock"),("other_both","Other installation + clock")]:
                ax.plot(plot["times"],plot[field][variant],lw=1,c=colors[variant],label=caption)
            ax.axvline(20,c="#888888",ls="--",lw=1);ax.set(title=label,xlabel="EGO recording time [s]",ylabel=unit)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Effect of cross-applying saved parameters on reconstructed IMU reference (same raw NOKOV)")
    fig.savefig(output/"reference_parameter_differences.png",dpi=150);plt.close(fig)


def relationship_summary(output,result):
    """Export the coordinate chain and physical installation deviation only."""
    mount=result["installation_difference"]
    session_pair=' / '.join(result['definitions']['session_labels'][key] for key in ['A','B'])
    overview=[{"session_pair":session_pair,
        "installation_translation_mm":mount["translation_norm_mm"],
        "installation_rotation_deg":mount["rotation_geodesic_deg"]}]
    details=[{"session_pair":session_pair,"relation":"T_B_I",
        "translation_difference_mm":mount["translation_norm_mm"],
        "rotation_difference_deg":mount["rotation_geodesic_deg"]}]
    matrices={"T_B_I":{"session_A":result["sensor_A"]["T_marker_imu"],
        "session_B":result["sensor_B"]["T_marker_imu"],"difference":mount}}
    result["coordinate_relationship_summary"]={
        "frames":{"N":"NOKOV world","V":"Independent Basalt VIO local world for each session/case",
                  "B":"NOKOV orbbec_ego4 rigid body","I":"Basalt output IMU body frame"},
        "chain":"T_N_B(t_N) @ T_B_I = T_N_V @ T_V_I(t_I); t_I=t_N+clock_offset",
        "translation_difference":"1000 * norm(t_B - t_A), with saved transform translations in metres",
        "rotation_difference":"degrees(angle(R_A.T @ R_B))",
        "scope":"One physical installation calibration per session, shared by all VIO cases. World transforms are used only for trajectory alignment and pose-error evaluation.",
        "overview":overview,"details":details,"matrices":matrices}
    with (output/"coordinate_relationship_deviations.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    detail_rows="".join(f"<tr><td>{r['session_pair']}</td><td><code>{r['relation']}</code></td><td>{r['translation_difference_mm']:.3f}</td><td>{r['rotation_difference_deg']:.4f}</td></tr>" for r in details)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EGO / NOKOV 安装外参偏差结论</title>
<style>body{{font:16px/1.8 system-ui,sans-serif;color:#172b43;background:#f5f7fa;margin:0}}main{{max-width:1100px;margin:auto;padding:28px 26px 60px;overflow-wrap:anywhere}}h1{{font-size:27px;line-height:1.4}}h2{{font-size:20px;margin-top:30px}}pre{{background:#eaf0f6;padding:18px;border:1px solid #d8e1ed;border-radius:8px;overflow:auto;font-size:14px}}.table{{overflow:auto;background:white;padding:6px;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #cdd7e4;padding:10px;text-align:right;white-space:nowrap}}th{{background:#edf2f8}}th:first-child,td:first-child{{text-align:left}}code{{font-family:ui-monospace,monospace}}a{{color:#2459a6}}.note{{color:#5a6c81}}li{{margin:5px 0}}</style><main>
<h1>EGO / NOKOV 两次 session 的安装外参偏差</h1>
<p>N：NOKOV 世界；V：各次 Basalt VIO 的原始局部世界；B：NOKOV 的 <code>orbbec_ego4</code> 头环刚体坐标；I：Basalt 输出的 IMU 坐标。<code>T_A_B</code> 把 B 中的坐标转换到 A，矩阵平移单位为 m。</p>
<pre>T_N_B(t_N) · X = Y · T_V_I(t_I)
X = T_B_I                 安装外参（I → B）
Y = T_N_V                 局部世界配准（V → N）
T_V_N = inverse(Y)        逆向世界配准（N → V）
T_V_B(t) = inverse(Y) · T_N_B(t)   随头环运动变化
t_I = t_N + δ            NOKOV 与 IMU 时钟对应关系</pre>
<p>转换关系分别为：</p><ul><li><b>安装外参 T_B_I</b>：EGO IMU 坐标 → NOKOV 头环刚体坐标，结果文件中的名称是 <code>T_marker_imu</code>。</li><li><b>世界配准 T_N_V</b>：Basalt VIO 局部世界 → NOKOV 世界。</li><li><b>逆向配准 T_V_N</b>：NOKOV 世界 → Basalt VIO 局部世界。</li></ul>
<p>Session A = <code>{html.escape(Path(result['definitions']['A']).name)}</code>，Session B = <code>{html.escape(Path(result['definitions']['B']).name)}</code>。安装外参来自各 session 的原始 IMU/NOKOV 标定，原版双目、Monado 双目和 Monado 四目共用。世界变换用于各段轨迹配准与位姿误差评价。</p>
<h2>安装外参的跨 session 偏差</h2><div class="table"><table><tr><th>Session 对</th><th>关系</th><th>安装平移偏差 / mm</th><th>安装旋转偏差 / °</th></tr>{detail_rows}</table></div>
<p>计算口径：安装平移偏差为 <code>1000 × ‖t_B − t_A‖₂</code>，安装旋转偏差为 <code>angle(R_Aᵀ R_B)</code>。</p>
<p><b>结论：</b>两次安装外参估计相差 {mount['translation_norm_mm']:.3f} mm、{mount['rotation_geodesic_deg']:.4f}°。</p>
<p class="note">两次 NOKOV→IMU 时钟偏移分别为 {result['sensor_A']['mocap_to_imu_offset_s']*1000:.3f} ms 和 {result['sensor_B']['mocap_to_imu_offset_s']*1000:.3f} ms，相差 {result['clock_difference_B_minus_A_ms']:.3f} ms。独立安装外参真值尚未提供，以上数值反映估计的一致性。</p>
<p><a href="coordinate_relationship_deviations.csv">下载安装偏差 CSV</a> · <a href="comparison.json">安装外参与分析数值 JSON</a> · <a href="comparison_report.html">完整分析与互换参数试验</a></p></main></html>'''
    (output/"coordinate_relationship_summary.html").write_text(text,encoding="utf-8")


def report(output,result):
    d=result["installation_difference"];a=result["sensor_A"];b=result["sensor_B"]
    def esc(value):return html.escape(str(value))
    def vec(value):return ", ".join(f"{v:.3f}" for v in value)
    rows=""
    for axis,index in zip("XYZ",range(3)):
        rows+=f"<tr><th>IMU 原点 / marker {axis} 轴 / mm</th><td>{np.array(a['T_marker_imu'])[index,3]*1000:.3f}</td><td>{np.array(b['T_marker_imu'])[index,3]*1000:.3f}</td><td>{d['translation_B_minus_A_in_parent_mm'][index]:+.3f}</td></tr>"
    rows+=f"<tr><th>杠杆臂长度 / mm</th><td>{a['lever_arm_norm_m']*1000:.3f}</td><td>{b['lever_arm_norm_m']*1000:.3f}</td><td>{(b['lever_arm_norm_m']-a['lever_arm_norm_m'])*1000:+.3f}</td></tr>"
    rows+=f"<tr><th>NOKOV → IMU 时钟偏移 / ms</th><td>{a['mocap_to_imu_offset_s']*1000:.3f}</td><td>{b['mocap_to_imu_offset_s']*1000:.3f}</td><td>{result['clock_difference_B_minus_A_ms']:+.3f}</td></tr>"
    refrows="";viorows=""
    for label in ["A","B"]:
        cross=result["cross_apply"][label]
        for variant in list(VARIANTS)[1:]:
            e=cross["reference_differences"][variant]
            refrows+=f"<tr><td>{label}</td><td>{VARIANTS[variant]}</td><td>{e['holdout_position_mm']['rmse']:.3f}</td><td>{e['holdout_position_mm']['max']:.3f}</td><td>{e['holdout_orientation_deg']['rmse']:.3f}</td><td>{e['holdout_orientation_deg']['max']:.3f}</td></tr>"
        for case,entries in cross["vio"].items():
            viorows+=f"<tr><td>{label}</td><td>{html.escape(LABELS.get(case,case))}</td>"+"".join(f"<td>{entries[v]['position_rmse_cm_median']:.3f} / {entries[v]['orientation_rmse_deg_median']:.3f}</td>" for v in VARIANTS)+"</tr>"
    markerrows="".join(f"<tr><td>{m['marker_index']}</td><td>{vec(m['delta_B_minus_A_mm'])}</td><td>{m['distance_mm']:.3f}</td></tr>" for m in result["marker_template_comparison"])
    camrows="".join(f"<tr><td>cam{c['camera']}</td><td>{c['camera_to_imu_difference']['translation_norm_mm']:.6f}</td><td>{c['camera_to_imu_difference']['rotation_geodesic_deg']:.6f}</td><td>{c['camera_to_marker_difference']['translation_norm_mm']:.3f}</td></tr>" for c in result["camera_comparison"])
    img="data:image/png;base64,"+base64.b64encode((output/"reference_parameter_differences.png").read_bytes()).decode()
    delta=np.array(d["delta_inv_A_times_B"]);delta[:3,3]*=1000
    rank_unchanged=True
    max_score_change_mm=0.
    for cross in result["cross_apply"].values():
        cases=list(cross['vio'])
        baseline_rank=sorted(cases,key=lambda c:cross["vio"][c]["own"]["position_rmse_cm_median"])
        for variant in VARIANTS:
            rank_unchanged &= sorted(cases,key=lambda c:cross["vio"][c][variant]["position_rmse_cm_median"])==baseline_rank
            max_score_change_mm=max(max_score_change_mm,max(abs(cross["vio"][c][variant]["position_rmse_change_cm"])*10 for c in cases))
    source=json.dumps(result['definitions'],indent=2,ensure_ascii=False)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两次 EGO/NOKOV 坐标转换一致性</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;background:#f3f6fa;color:#172b43;margin:0}}main{{max-width:1160px;margin:auto;padding:32px 24px 70px;overflow-wrap:anywhere}}h1{{line-height:1.35}}h2{{margin-top:36px;font-size:23px}}.lead{{background:#e1edf8;padding:20px;border-radius:10px}}.table{{background:white;overflow:auto;padding:8px;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:11px;border-bottom:1px solid #dfe6ee;white-space:nowrap;text-align:right}}td:first-child,th:first-child{{text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e5ebf2;padding:15px}}code{{overflow-wrap:anywhere}}img{{width:100%;margin-top:16px;border-radius:8px}}a{{color:#2459a6}}li{{margin:8px 0}}.note{{color:#5c6e84}}</style><main>
<p class="note">A = {esc(result['definitions']['session_labels']['A'])}；B = {esc(result['definitions']['session_labels']['B'])}。读取现有结果，未重跑 VIO，未修改原实验。<a href="coordinate_relationship_summary.html">按坐标链和偏差表查看结论</a></p>
<h1>两次录制获得的坐标转换关系有多大差异？</h1>
<p class="lead">安装外参 <b>T_marker_imu</b> 的旋转差为 <b>{d['rotation_geodesic_deg']:.6f}°</b>，平移向量差的模长为 <b>{d['translation_norm_mm']:.6f} mm</b>；B−A 主要来自 marker Z 轴（{d['translation_B_minus_A_in_parent_mm'][2]:+.3f} mm）。时钟偏移另相差 <b>{result['clock_difference_B_minus_A_ms']:.6f} ms</b>。这是两次估计的一致性差异，没有独立外参真值时，不能判断哪次更准，也不能直接认定安装发生了移动。</p>
<h2>1. 可以直接比较的安装外参</h2><p>约定 T_A_B 将 B 坐标映射到 A 坐标。因此 T_marker_imu 将 IMU 坐标映射到头环刚体坐标，平移列表示 IMU 原点在 marker 坐标系的位置；真值重建为 <code>T_N_I(t) = T_N_M(t − δ) × T_M_I</code>，其中 <code>t_IMU = t_NOKOV_raw_ms × 10⁻³ + δ</code>。</p>
<div class="table"><table><tr><th>参数</th><th>Session A</th><th>Session B</th><th>B − A</th></tr>{rows}</table></div>
<p>旋转用 SO(3) 最短转角比较，避免直接相减欧拉角。相对变换 Δ = inv(T_A) × T_B 为下列矩阵；旋转块无量纲，末列前三项使用 mm：</p><pre>{esc(np.array2string(delta,precision=8,suppress_small=True))}</pre>
<p>Δ 的旋转向量（在 A 的 IMU 轴系）为 [{vec(d['rotation_vector_in_A_deg'])}]°。注意矩阵末列在 A 的 IMU 轴系，与上表在 marker 轴系的平移差分量不同，但模长相同。</p>
<h2>2. 相机标定和 marker 坐标系是否发生变化？</h2><p>两组完整相机标定 JSON 完全相同，原版与 Monado 的配置也分别相同。相机→IMU 外参逐相机旋转差与平移差均为 0。把不同安装外参复合到相机→marker 后，会出现下表的相机原点差，旋转差仍为 {d['rotation_geodesic_deg']:.3f}°。</p>
<div class="table"><table><tr><th>相机</th><th>相机→IMU 平移差 / mm</th><th>相机→IMU 旋转差 / °</th><th>相机→marker 原点差 / mm</th></tr>{camrows}</table></div>
<p>各反光球在各自 marker 坐标系中的中位位置，两次的差异如下。最大 {max(m['distance_mm'] for m in result['marker_template_comparison']):.3f} mm；安装平移估计差为 {d['translation_norm_mm']:.3f} mm。这只能检查刚体坐标定义的一致性，不能独立证明 IMU 相对刚体绝对没有移动。</p>
<div class="table"><table><tr><th>marker 编号</th><th>局部中位位置差 B−A / XYZ mm</th><th>差向量模长 / mm</th></tr>{markerrows}</table></div>
<h2>3. 互换参数后，重建 IMU 真值改变多少？</h2><p>在每段录制自己的原始 NOKOV 上，分别互换安装外参、时钟偏移、或同时互换，比较与本段原参数重建的 IMU 真值。下表在 t≥20 s 的共同覆盖位姿上统计，不包含任何 VIO 预测。只换安装外参时，同一时刻原点距离恒为 {d['translation_norm_mm']:.3f} mm，姿态距离恒为 {d['rotation_geodesic_deg']:.3f}°；换时钟引入的差异则取决于头部运动速度。</p>
<div class="table"><table><tr><th>录制</th><th>互换方式</th><th>位置 RMSE / mm</th><th>最大位置差 / mm</th><th>姿态 RMSE / °</th><th>最大姿态差 / °</th></tr>{refrows}</table></div>
<img alt="互换参数对两段 IMU 真值的影响" src="{img}">
<h2>4. 这会怎样改变 VIO 精度结论？</h2><p>固定每段已生成的三次 VIO 轨迹和评分时刻，仅替换标定参数，并按原协议重新估计最初两秒的世界配准。该步骤只重算评价；没有重新运行算法，也没有在后段重新拟合。A 世界配准为 {result['cross_apply']['A']['world_registration_interval_s']} s，B 为 {result['cross_apply']['B']['world_registration_interval_s']} s。</p>
<p>单元格为 <b>后段位置 RMSE / cm · 姿态 RMSE / °</b>，均取三次运行的中位数。“本次参数”已数值复现原报告；这种重新配准会吸收一部分标定差异，因此评价指标变化不等于上一节的真值坐标差。</p>
<div class="table"><table><tr><th>录制</th><th>算法</th><th>本次参数</th><th>替换安装外参</th><th>替换时钟偏移</th><th>同时替换两者</th></tr>{viorows}</table></div>
<p>所有互换试验中，位置 RMSE 相对原评价变化的最大绝对值为 {max_score_change_mm:.3f} mm；各段三种配置的位置 RMSE 排序{'均未改变' if rank_unchanged else '存在变化'}。互换标定没有消除原报告中的厘米级位置误差；这也不能排除两组安装估计具有共同的系统误差。</p>
<h2>5. 差异的可信解释与使用建议</h2><ul>
<li>安装标定区间 A={a['training_interval_imu_s']} s、B={b['training_interval_imu_s']} s，使用原始 IMU/NOKOV，未经独立安装测量。加速度设计矩阵条件数 A={a['acceleration_design_condition']:.2f}，B={b['acceleration_design_condition']:.2f}；同样建模下，数值越大代表联合参数拟合的数值条件越差，不代表已有定量置信区间。两次运动激励、平滑微分误差及重力/加速度偏置耦合都可能影响杠杆臂估计。</li>
<li>后段陀螺仪残差 RMSE A={a['angular_velocity_residual']['validation_rad_s']['rmse']:.4f}、B={b['angular_velocity_residual']['validation_rad_s']['rmse']:.4f} rad/s；加速度残差 A={a['acceleration_residual']['validation_m_s2']['rmse']:.4f}、B={b['acceleration_residual']['validation_m_s2']['rmse']:.4f} m/s²。单凭不同运动段的残差不能判定哪次绝对安装外参更准。</li>
<li>两组输入的图像—IMU 时间修正均固定为 0 ms，直接使用导出时间戳。NOKOV→IMU 偏移保留，用于两个采集系统之间的参考对时。此处参数互换分析不运行 VIO。</li>
<li>如硬件安装保持不变，应通过独立、充分三轴激励的标定录制确定一组固定安装外参，再在两段未参与标定的数据上交叉验证；各会话保留各自校时和 VIO 世界初始化。不要平均世界配准矩阵后跨录制复用。</li>
<li>忽略时钟差，只考虑这两组安装估计，对距 IMU 原点 1 m 的任意空间点，坐标差上界为 {result['same_clock_point_displacement_upper_bound'][2]['position_difference_bound_mm']:.2f} mm（{d['translation_norm_mm']:.2f} mm 平移项 + 旋转项）。这是几何上界，不是实测误差或标定精度认证。</li></ul>
<p><a href="comparison.json">完整矩阵和数值 JSON</a> · <a href="cross_applied_vio_metrics.csv">互换参数后的 VIO 指标 CSV</a> · <a href="source_manifest.json">读取文件与哈希</a></p><details><summary>坐标约定和来源</summary><pre>{esc(source)}</pre></details></main></html>'''
    (output/"comparison_report.html").write_text(text,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a",type=Path,default=Path("output/basalt_comparison_20261104_195732"))
    p.add_argument("--b",type=Path,default=Path("output/basalt_comparison_20261104_195844"))
    p.add_argument("--output",type=Path,default=Path("output/basalt_calibration_comparison_195732_195844"))
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if output in [args.a.resolve(),args.b.resolve()]:raise ValueError("write analysis into a separate directory")
    a,b=load_result(args.a),load_result(args.b)
    paths=[]
    for data in [a,b]:
        paths += [data["directory"]/name for name in ["nokov_calibration.json","input_validation.json","comparison_summary.json","summary.json","config_original.json","config_monado.json"] if (data['directory']/name).exists()]
        paths += [data['calibration_path'],result_file(data['directory'],'dataset_4cam/frame_map.csv','dataset_2cam/frame_map.csv')]
        paths += sorted((data["directory"]/"runs").glob("*/repeat_*/trajectory.csv"))
        paths += [Path(data["sensor"]["reference_directory"])/"nokov_rigid_bodies.csv"]
    hashes={str(path):sha256(path) for path in paths}
    result,plots=analyze(a,b);relationship_summary(output,result);dump(output/"comparison.json",result);figures(output,result,plots);report(output,result)
    rows=[]
    for label,entry in result["cross_apply"].items():
        for case,variants in entry["vio"].items():
            for variant,metrics in variants.items():
                rows.append({"recording":label,"case":case,"variant":variant,
                    **{k:v for k,v in metrics.items() if k!="runs"}})
    with (output/"cross_applied_vio_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if any(sha256(Path(path))!=digest for path,digest in hashes.items()):raise ValueError("source result changed during analysis")
    dump(output/"source_manifest.json",{"source_files_unchanged":True,"files":[{"path":path,"sha256":digest} for path,digest in hashes.items()],"analysis_script_sha256":sha256(Path(__file__))})
    from ego_data.vio_results import load_vio_result
    from ego_data.vio_report import write_session_report
    write_session_report(output,[load_vio_result(args.a),load_vio_result(args.b)],result['installation_difference'])
    print(json.dumps({"installation_difference":result["installation_difference"],
        "clock_difference_B_minus_A_ms":result["clock_difference_B_minus_A_ms"],"report":str(output/"comparison_report.html"),"session_report":str(output/'report.html')},indent=2))


if __name__=="__main__":main()
