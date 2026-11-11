"""Normalize saved VIO results for viewers and session reports.

Readers handle the original repeated-run format, the partial-reference format,
and frozen-calibration full runs. Estimates, sensor calibration and saved world
registration are loaded as evidence; presentation never fits new transforms.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_basalt_ego4_comparison import relative_pose_errors
from ego_data.fixed_relations import FixedReference, check_transform, sha256
from ego_data.time_sync import require_zero_camera_offset
from ego_data.vio_performance import collect_vio_performance
from fuse_basalt_hand_trajectory import load_basalt_trajectory

LABELS={"original_2cam":"原版 Basalt · 双目","monado_2cam":"Monado · 双目","monado_4cam":"Monado · 四目"}
COLOURS={"original_2cam":"#165dff","monado_2cam":"#d600a9","monado_4cam":"#f06b00","reference":"#007f73"}


def read_json(path):
    return json.loads(Path(path).read_text())


def result_file(directory,*candidates):
    for name in candidates:
        path=Path(directory)/name
        if path.exists():return path
    raise FileNotFoundError(f'None of {candidates} exist in {directory}')


def pose_statistics(poses,reference,selected=None):
    selected=np.ones(len(poses),dtype=bool) if selected is None else np.asarray(selected,dtype=bool)
    p=poses[selected];g=reference[selected]
    if not len(p):return {"poses":0,"position_rmse_m":None,"orientation_rmse_deg":None}
    pe=np.linalg.norm(p[:,:3,3]-g[:,:3,3],axis=1)
    re=np.rad2deg((Rotation.from_matrix(g[:,:3,:3]).inv()*Rotation.from_matrix(p[:,:3,:3])).magnitude())
    return {"poses":len(p),"position_rmse_m":float(np.sqrt(np.mean(pe**2))),
        "position_median_m":float(np.median(pe)),"position_p95_m":float(np.percentile(pe,95)),
        "position_max_m":float(pe.max()),"orientation_rmse_deg":float(np.sqrt(np.mean(re**2))),
        "orientation_p95_deg":float(np.percentile(re,95)),"orientation_max_deg":float(re.max()),
        "final_position_error_m":float(pe[-1]),"final_orientation_error_deg":float(re[-1]),
        "reference_path_length_m":float(np.linalg.norm(np.diff(g[:,:3,3],axis=0),axis=1).sum()),
        "vio_path_length_m":float(np.linalg.norm(np.diff(p[:,:3,3],axis=0),axis=1).sum())}


def saved_world(run,case):
    if "T_nokov_vio" in run:return check_transform(run["T_nokov_vio"],"saved world")
    if "world_registration_rotation" in run:
        matrix=np.eye(4);matrix[:3,:3]=run["world_registration_rotation"];matrix[:3,3]=run["world_registration_translation_m"]
        return check_transform(matrix,"saved world")
    if "world_transform_loaded" in case:return check_transform(case["world_transform_loaded"],"frozen world")
    raise ValueError("Result has no saved world registration; regenerate evaluation before visualization")


def load_vio_result(directory,repeat=None):
    directory=Path(directory).resolve()
    info=read_json(directory/"input_validation.json");sensor=read_json(directory/"nokov_calibration.json")
    require_zero_camera_offset(info,str(directory))
    for name in ['dataset_4cam/calibration.json','dataset_2cam/calibration.json']:
        if (directory/name).exists():require_zero_camera_offset(read_json(directory/name),str(directory/name))
    summary_path=directory/"comparison_summary.json"
    summary=read_json(summary_path) if summary_path.exists() else {}
    if "results" in summary:
        entries=summary["results"];first=next(iter(entries.values()))
        source_format="overlap_repeats" if "runs" in first else "legacy_repeats"
    else:
        summary_path=directory/"summary.json";summary=read_json(summary_path)
        if "cases" not in summary:raise ValueError(f"Unsupported VIO result format: {directory}")
        entries=summary["cases"];source_format="frozen_full"
    origin=int(info["first_timestamp_ns"])
    reference=FixedReference({"nokov_directory":sensor["reference_directory"],"rigid_body":sensor["rigid_body"],
        "T_marker_imu":sensor["T_marker_imu"],"nokov_to_imu_offset_s":sensor["mocap_to_imu_offset_s"]},origin)
    protocol=summary.get("protocol",{});interval=protocol.get("scored_interval_s")
    sensor_end=float(sensor["training_interval_imu_s"][1])
    expected=None
    frame_map=next((directory/name for name in ['dataset_4cam/frame_map.csv','dataset_2cam/frame_map.csv'] if (directory/name).exists()),directory/'frame_map.csv')
    if frame_map.exists():
        with frame_map.open() as f:expected=np.array([int(r["timestamp_ns"]) for r in csv.DictReader(f)],dtype=np.int64)
    cases={};source_paths=[directory/"input_validation.json",directory/"nokov_calibration.json",summary_path,
                          Path(sensor["reference_directory"])/"nokov_rigid_bodies.csv"]
    shared_stamps=None
    for name,entry in entries.items():
        records=entry.get("runs",entry.get("repeats"))
        if records is None:records=[{"directory":str(directory/"runs"/name/"repeat_1")}]
        runs=[]
        for record in records:
            number=int(Path(record["directory"]).name.split("_")[-1])
            if repeat is not None and number!=repeat:continue
            path=directory/"runs"/name/f"repeat_{number}"/"trajectory.csv"
            poses=load_basalt_trajectory(path);stamps=np.array(sorted(poses),dtype=np.int64)
            with path.open() as f:count=sum(bool(line.strip()) and not line.startswith("#") for line in f)
            if count!=len(stamps) or (expected is not None and not np.array_equal(stamps,expected)):
                raise ValueError(f"Incomplete or duplicate native timestamps: {path}")
            if len(stamps)!=info["prepared_groups"]:raise ValueError(f"Incomplete trajectory: {path}")
            times=(stamps-origin)*1e-9
            keep=(times>=reference.times[0])&(times<=reference.times[-1])
            if interval is not None:keep&=(times>=interval[0])&(times<=interval[1])
            hi=np.clip(np.searchsorted(reference.times,times),1,len(reference.times)-1)
            keep&=reference.times[hi]-reference.times[hi-1]<=.05
            selected_stamps=stamps[keep];selected_times=times[keep]
            if not len(selected_stamps):raise ValueError(f"No reference overlap: {path}")
            if shared_stamps is not None and not np.array_equal(selected_stamps,shared_stamps):
                raise ValueError("Comparison requires common native visual timestamps")
            shared_stamps=selected_stamps.copy()
            raw=np.array([poses[int(t)] for t in stamps]);world=saved_world(record,entry)
            aligned=np.einsum("ij,njk->nik",world,raw[keep])
            gp,gr,valid=reference.query(selected_times)
            if not valid.all() or not np.isfinite(raw).all():raise ValueError("Invalid poses/reference; extrapolation forbidden")
            truth=np.tile(np.eye(4),(len(gp),1,1));truth[:,:3,3]=gp;truth[:,:3,:3]=gr.as_matrix()
            holdout=selected_times>=sensor_end
            all_stats=pose_statistics(aligned,truth);held_stats=pose_statistics(aligned,truth,holdout)
            saved_all=record.get("all_overlap",{}).get("position_m",{}).get("rmse",record.get("translation_error_all_m",{}).get("rmse"))
            if source_format=="frozen_full":saved_all=entry.get("all",{}).get("position_rmse_m")
            if saved_all is not None and not np.isclose(all_stats["position_rmse_m"],saved_all,atol=1e-8,rtol=0):
                raise ValueError(f"Loaded estimates do not reproduce saved metrics: {path}")
            perf=collect_vio_performance(directory,name,repeat=number)
            p=aligned[:,:3,3];r=Rotation.from_matrix(aligned[:,:3,:3])
            stats={"all":all_stats,"holdout":held_stats,"accuracy":dict(all_stats),"performance":perf,
                "rpe_1s_all":relative_pose_errors(selected_times,p,r,gp,gr,1.),
                "rpe_1s_holdout":relative_pose_errors(selected_times,p,r,gp,gr,1.,holdout),
                "rpe_0_1s_holdout":relative_pose_errors(selected_times,p,r,gp,gr,.1,holdout)}
            stats["accuracy"]["pose_output_coverage_pct"]=100*len(stamps)/info["prepared_groups"]
            pe=np.linalg.norm(p-gp,axis=1);re=np.rad2deg((gr.inv()*r).magnitude())
            runs.append({"repeat":number,"source":path,"timestamps":selected_stamps,"time":selected_times,
                "poses":aligned,"reference":truth,"position_error_m":pe,"orientation_error_deg":re,
                "stats":stats,"full_output_poses":len(stamps),"reference_coverage_pct":100*len(gp)/len(stamps)})
            source_paths.append(path)
        if not runs:raise ValueError(f"No selected repeat for {name}: {repeat}")
        cases[name]={"label":LABELS.get(name,entry.get("label",name)),"colour":COLOURS.get(name,"#7652a4"),"runs":runs}
    label=Path(info.get('session',directory.name)).name.rsplit('_',1)[-1]
    return {"directory":directory,"label":label,"format":source_format,
        "info":info,"sensor":sensor,"cases":cases,"calibration_end_s":sensor_end,
        "registration_interval_s":protocol.get("world_registration_interval_s",[0.,2.]),
        "sources":[{"path":str(p),"sha256":sha256(p)} for p in dict.fromkeys(source_paths)]}


def aggregate(runs,path):
    values=[]
    for run in runs:
        value=run
        for key in path.split("."):value=value.get(key) if isinstance(value,dict) else None
        if value is not None and np.isfinite(value):values.append(float(value))
    return {"median":float(np.median(values)) if values else None,
            "min":min(values) if values else None,"max":max(values) if values else None,"count":len(values)}
