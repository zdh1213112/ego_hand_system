#!/usr/bin/env python3
"""Prepare physical-camera NOKOV/MANO observations without image rectification."""
import argparse
import json
from pathlib import Path
import numpy as np
from ego_data.calibration import CameraCalibration
from camera_models import project_points
from native_camera_geometry import reference_to_camera
from prepare_nokov21_mano_input import _sync_timestamps, _nullable_points


def prepare_native(aligned, dataset, output, fusion_output, source_group,
                   reference_camera="camera0", cameras=("camera0", "camera1")):
    aligned, dataset, output, fusion_output = map(lambda p: Path(p).resolve(),
        (aligned, dataset, output, fusion_output))
    if output.exists() or output.with_suffix(".json").exists() or fusion_output.exists():
        raise FileExistsError("use new output paths")
    if not source_group.strip() or len(cameras) < 2 or len(set(cameras)) != len(cameras):
        raise ValueError("require source identity and at least two distinct cameras")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("dataset_type") != "normalized_multiview" or not manifest.get("calibration", {}).get("available"):
        raise ValueError("require calibrated normalized multiview data")
    if reference_camera != cameras[0] or not set(cameras).issubset(manifest["camera_ids"]):
        raise ValueError("reference must be first camera; cameras must exist in manifest")
    stamps = _sync_timestamps(dataset / "multiview_frames.csv")
    with np.load(aligned, allow_pickle=False) as a:
        points = a["joints_base_m"].astype(np.float64)
        observed = a["valid"].astype(bool)
        np.testing.assert_array_equal(stamps, a["reference_timestamp_ns"])
    if points.shape != (2, len(stamps), 21, 3) or observed.shape != points.shape[:-1]:
        raise ValueError("expected physical LEFT/RIGHT [2,F,21,3] reference")
    if np.any(observed & ~np.isfinite(points).all(-1)) or np.any(observed & (np.linalg.norm(points, axis=-1) > 10)):
        raise ValueError("observed points must be finite and in meters")
    if not (observed.sum(-1) >= 12).any():
        raise ValueError("no hand frame has 12 observed points")
    points[~observed] = np.nan
    cal = {c: CameraCalibration.load(dataset / "calibration" / f"{c}.json") for c in cameras}
    if any(c.model != "KB" for c in cal.values()):
        raise ValueError("native input supports KB cameras")
    T = cal[reference_camera].T_base_camera
    reference = (points - T[:3, 3]) @ T[:3, :3]
    native_views, payload = {}, {}
    for i, name in enumerate(cameras):
        camera = cal[name]
        R, t = reference_to_camera(cal[reference_camera], camera)
        pixels, valid = project_points(camera, reference @ R.T + t)
        w, h = camera.image_size
        valid &= observed & np.isfinite(pixels).all(-1)
        valid &= (pixels[..., 0] >= 0) & (pixels[..., 0] < w) & (pixels[..., 1] >= 0) & (pixels[..., 1] < h)
        pixels[~valid] = np.nan
        native_views[name] = (pixels.astype(np.float32), valid)
        if i < 2:
            view = "left" if i == 0 else "right"
            payload.update({f"{view}_native_px": pixels.astype(np.float32),
                f"{view}_native_valid": valid, f"native_{view}_rotation": R.astype(np.float32),
                f"native_{view}_translation": t.astype(np.float32),
                f"native_{view}_K": camera.K.astype(np.float32),
                f"native_{view}_D": camera.distortion.astype(np.float32)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload, image_space="native_raw", reference_camera=reference_camera,
        camera_calibrations_json=np.asarray(json.dumps({name: c.to_dict() for name,c in cal.items()},sort_keys=True)),
        positions_reference_camera_m=reference.astype(np.float32), valid=observed, observed=observed,
        confidence=observed.astype(np.float32), track_ids=np.array([0,1],np.int32),
        handedness=np.array(["Left","Right"]), pair_indices=np.arange(len(stamps),dtype=np.int32),
        fps=np.asarray(1e9/np.median(np.diff(stamps)) if len(stamps)>1 else 30., np.float32))
    accepted = []
    for frame in range(len(stamps)):
        hands = []
        for side in (0,1):
            if observed[side,frame].sum() < 12:
                continue
            views = {}
            for name, (px, mask) in native_views.items():
                if mask[side,frame].any():
                    views[name] = dict(inlier_joint_count=int(mask[side,frame].sum()),
                        joints_2d=_nullable_points(px[side,frame]), detector_is_right=side,
                        observation_source="nokov_projected_marker")
            hands.append(dict(side=side,joints_base_m=_nullable_points(points[side,frame]),
                observed_joint_count=int(observed[side,frame].sum()),views=views))
        if hands:
            accepted.append(dict(sync_index=frame,hands=hands))
    fusion_output.mkdir(parents=True)
    (fusion_output/"accepted.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in accepted))
    summary = dict(stage="nokov21_aligned_reference",image_space="native_raw",reference_camera=reference_camera,
        aligned_nokov=str(aligned),dataset=str(dataset),source_group=source_group,accepted_frames=len(accepted),
        camera_ids=list(cameras),pixel_target_provenance="KB projection of physical NOKOV marker centers; not independent RGB truth")
    (fusion_output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    output.with_suffix(".json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aligned-nokov",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fusion-output",type=Path,required=True)
    p.add_argument("--source-group",required=True)
    p.add_argument("--reference-camera",default="camera0")
    p.add_argument("--cameras",nargs="+",default=["camera0","camera1"])
    a=p.parse_args()
    print(json.dumps(prepare_native(a.aligned_nokov,a.dataset,a.output,a.fusion_output,
        a.source_group,a.reference_camera,a.cameras),indent=2))

if __name__ == "__main__":
    main()
