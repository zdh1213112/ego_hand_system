#!/usr/bin/env python3
"""Export calibrated, per-camera WiLoR hand joints, wrist poses and MANO meshes."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "python_deps/wilor_runtime"
if OVERLAY.is_dir():
    sys.path.insert(0, str(OVERLAY))
import cv2
import numpy as np
from ego_data.calibration import cameras_from_ego_yaml
from prepare_basalt_dataset import read_timestamps, unique_file
from wilor_camera_geometry import fit_translation, native_pixels, pinhole_view, proper_rotation
from fuse_basalt_hand_trajectory import quaternion_xyzw_from_matrix

SOURCES = {"cam_0": "left", "cam_1": "right", "cam_2": "right_side", "cam_3": "left_side"}
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]


def write_obj(path, vertices, faces):
    with path.open("w") as stream:
        stream.write("# Optical camera coordinates: X right, Y down, Z forward; metres\n")
        for v in vertices:
            stream.write("v " + " ".join(f"{x:.8f}" for x in v) + "\n")
        for f in faces:
            stream.write("f " + " ".join(str(int(x) + 1) for x in f) + "\n")


def infer_chunk(frames, detector, model, cfg, device, args):
    import torch
    from wilor_inference import ViTDetDataset, recursive_to, cam_crop_to_full, project_points
    detections = detector.predict(frames, conf=args.confidence, iou=0.5, imgsz=640,
                                   device=str(device), verbose=False)
    datasets, tasks = [], []
    for frame_index, detection in enumerate(detections):
        boxes = detection.boxes
        if boxes is None or len(boxes) == 0:
            continue
        selected = []
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidence = boxes.conf.detach().cpu().numpy()
        for side in (0, 1):
            candidates = np.flatnonzero(classes == side)
            if len(candidates):
                selected.append(int(candidates[np.argmax(confidence[candidates])]))
        if not selected:
            continue
        xyxy = boxes.xyxy.detach().float().cpu().numpy()[selected]
        sides = classes[selected].astype(np.float32)
        datasets.append(ViTDetDataset(cfg, frames[frame_index], xyxy, sides,
                                     rescale_factor=2.0, fp16=False))
        for index, side in zip(selected, sides):
            tasks.append((frame_index, int(side), float(confidence[index])))
    if not datasets:
        return [[] for _ in frames]
    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets),
                                         batch_size=args.batch_size, shuffle=False, num_workers=0)
    results = [[] for _ in frames]
    task_index = 0
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                   enabled=device.type == "cuda" and not args.fp32):
            output = model(batch)
        sides = batch["right"].float()
        pred_cam = output["pred_cam"].float().clone()
        pred_cam[:, 1] *= 2 * sides - 1
        sizes = batch["img_size"].float()
        focal = float(cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * sizes.max().item())
        assumed_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                                    sizes, focal).cpu().numpy()
        joints = output["pred_keypoints_3d"].float().cpu().numpy()
        vertices = output["pred_vertices"].float().cpu().numpy()
        mano = {key: value.float().cpu().numpy() for key, value in output["pred_mano_params"].items()}
        for index in range(len(joints)):
            frame_index, side, confidence = tasks[task_index]
            task_index += 1
            mirror = np.diag([1 if side else -1, 1, 1]).astype(np.float32)
            physical_joints = joints[index] @ mirror
            physical_vertices = vertices[index] @ mirror
            target_pixels = project_points(physical_joints, assumed_t[index], focal, sizes[index].cpu().numpy())
            rotation = mano["global_orient"][index].reshape(3,3)
            results[frame_index].append({
                "side": side, "confidence": confidence, "joints": physical_joints,
                "vertices": physical_vertices, "target_pixels": target_pixels,
                "rotation": proper_rotation(mirror @ rotation @ mirror),
                "mano_global_orient": rotation, "mano_hand_pose": mano["hand_pose"][index].reshape(15,3,3),
                "mano_betas": mano["betas"][index].reshape(10),
            })
    return results


def run_camera(camera_id, calibration, args, detector, model, cfg, device):
    output = args.output / camera_id
    output.mkdir()
    (output / "meshes").mkdir()
    source_video = unique_file(args.session, f"_camera_{SOURCES[camera_id]}.mp4")
    timestamps = read_timestamps(unique_file(args.session, f"_camera_{SOURCES[camera_id]}_pts.csv"))
    indices = list(range(0, min(len(timestamps), args.max_frames or len(timestamps)), args.frame_stride))
    n = len(indices)
    K, maps = pinhole_view(calibration, args.width, args.height, args.fov)
    calibration.save(output / "camera_calibration.json")
    (output / "undistortion.json").write_text(json.dumps({"K": K.tolist(), "resolution": [args.width,args.height],
        "rotation_native_to_virtual": np.eye(3).tolist(), "horizontal_fov_deg": args.fov}, indent=2) + "\n")
    archive = {
        "frame_index": np.asarray(indices, dtype=np.int64),
        "timestamp_us": np.asarray([timestamps[i] for i in indices], dtype=np.int64),
        "valid": np.zeros((n,2), bool), "confidence": np.zeros((n,2), np.float32),
        "joints_camera_m": np.full((n,2,21,3), np.nan, np.float32),
        "vertices_camera_m": np.full((n,2,778,3), np.nan, np.float32),
        "joints_2d_native_px": np.full((n,2,21,2), np.nan, np.float32),
        "T_camera_hand": np.full((n,2,4,4), np.nan, np.float32),
        "mano_global_orient_right_canonical": np.full((n,2,3,3), np.nan, np.float32),
        "mano_hand_pose_right_canonical": np.full((n,2,15,3,3), np.nan, np.float32),
        "mano_betas": np.full((n,2,10), np.nan, np.float32),
        "geometry_translation_camera_m": np.full((n,2,3), np.nan, np.float32),
        "placement_fit_rmse_px": np.full((n,2), np.nan, np.float32),
        "faces_right": np.asarray(model.mano.faces, np.int32),
        "faces_left": np.asarray(model.mano.faces[:, [0,2,1]], np.int32),
    }
    source = cv2.VideoCapture(str(source_video))
    writer = cv2.VideoWriter(str(output / "mesh_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                              30 / args.frame_stride, (800,600))
    if not source.isOpened() or not writer.isOpened():
        raise RuntimeError("cannot open input/output video")
    poses = (output / "hand_poses.csv").open("w", newline="")
    csv_writer = csv.writer(poses)
    csv_writer.writerow(("frame_index","timestamp_us","hand","confidence","wrist_x_m","wrist_y_m","wrist_z_m",
                         "qw","qx","qy","qz","placement_fit_rmse_px"))
    next_frame = 0
    started = time.perf_counter()
    try:
        for start in range(0,n,args.frame_batch_size):
            chunk = indices[start:start + args.frame_batch_size]
            originals, views = [], []
            for index in chunk:
                frame = None
                while next_frame <= index:
                    ok, frame = source.read()
                    if not ok:
                        raise RuntimeError(f"{source_video} ended before frame {index}")
                    next_frame += 1
                originals.append(frame)
                views.append(cv2.remap(frame, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
            predictions = infer_chunk(views, detector, model, cfg, device, args)
            for offset, (frame_index, frame, hands) in enumerate(zip(chunk, originals, predictions)):
                row = start + offset
                preview = cv2.resize(frame, (800,600))
                for hand in hands:
                    side = hand["side"]
                    translation, rmse = fit_translation(hand["joints"], hand["target_pixels"], K)
                    joints = hand["joints"] + translation
                    vertices = hand["vertices"] + translation
                    if not (np.isfinite(vertices).all() and np.isfinite(joints).all() and vertices[:,2].min() > .005):
                        continue
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3,:3] = hand["rotation"]
                    transform[:3,3] = joints[0]
                    projected = native_pixels(joints, calibration)
                    archive["valid"][row,side] = True
                    archive["confidence"][row,side] = hand["confidence"]
                    for key,value in (("joints_camera_m", joints), ("vertices_camera_m", vertices),
                        ("joints_2d_native_px", projected), ("T_camera_hand", transform),
                        ("mano_global_orient_right_canonical", hand["mano_global_orient"]),
                        ("mano_hand_pose_right_canonical", hand["mano_hand_pose"]), ("mano_betas",hand["mano_betas"]),
                        ("geometry_translation_camera_m",translation), ("placement_fit_rmse_px",rmse)):
                        archive[key][row,side] = value
                    name = "right" if side else "left"
                    quaternion = quaternion_xyzw_from_matrix(hand["rotation"])
                    csv_writer.writerow((frame_index,timestamps[frame_index],name,hand["confidence"],*joints[0],
                                         quaternion[3],*quaternion[:3],rmse))
                    faces = archive["faces_right" if side else "faces_left"]
                    if args.obj_stride and frame_index % args.obj_stride == 0:
                        write_obj(output / "meshes" / f"frame_{frame_index:06d}_{name}.obj", vertices, faces)
                    # Every displayed mesh is projected using the original fish-eye calibration.
                    projected_vertices = native_pixels(vertices, calibration) * [800/frame.shape[1],600/frame.shape[0]]
                    polygons = np.rint(np.clip(projected_vertices[faces], -2000, 3000)).astype(np.int32)
                    color = (255,200,40) if not side else (40,180,255)
                    layer = preview.copy()
                    triangles = vertices[faces]
                    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
                    shade = .45 + .55 * np.abs(normals @ np.asarray([0.,-.6,-.8]))
                    for face_index in np.argsort(-triangles[:,:,2].mean(axis=1)):
                        face_color = tuple(int(v) for v in np.asarray(color) * shade[face_index])
                        cv2.fillConvexPoly(layer, polygons[face_index], face_color)
                    preview = cv2.addWeighted(preview,.72,layer,.28,0)
                    pixels = np.rint(projected * [800/frame.shape[1],600/frame.shape[0]]).astype(int)
                    for a,b in EDGES:
                        cv2.line(preview,tuple(pixels[a]),tuple(pixels[b]),color,1,cv2.LINE_AA)
                    for pixel in pixels:
                        cv2.circle(preview,tuple(pixel),2,(255,255,255),-1)
                cv2.putText(preview,f"{camera_id} | frame {frame_index} | WiLoR camera-space mesh",(12,25),
                            cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                writer.write(preview)
            if start == 0 or (start + len(chunk)) % 40 == 0 or start + len(chunk) == n:
                count = int(archive["valid"][:start+len(chunk)].sum())
                print(f"{camera_id}: {start+len(chunk)}/{n} frames, {count} hand instances, {time.perf_counter()-started:.1f}s",flush=True)
    finally:
        source.release(); writer.release(); poses.close()
    np.savez_compressed(output / "hands_3d.npz", **archive)
    summary = {"camera": camera_id,"frames":n,"left_instances":int(archive['valid'][:,0].sum()),
               "right_instances":int(archive['valid'][:,1].sum()),"processing_s":time.perf_counter()-started,
               "coordinate_system":"native optical camera frame: X right, Y down, Z forward; metres",
               "shape":"21 joints, 778 mesh vertices, 1538 triangular faces per valid hand",
               "placement":"fixed WiLoR shape/orientation; calibrated translation fitted to WiLoR 2D projection",
               "accuracy":"monocular model estimate; no stereo depth, no external ground truth",
               "missing":"valid=false and NaN geometry; no gap filling",
               "hand_order":["left","right"]}
    (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--wilor-root",type=Path,default=ROOT.parent / "WiLoR")
    parser.add_argument("--cameras",nargs="+",choices=list(SOURCES),default=list(SOURCES))
    parser.add_argument("--device",choices=("cuda","cpu","auto"),default="cuda")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--frame-batch-size",type=int,default=4)
    parser.add_argument("--frame-stride",type=int,default=1)
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--confidence",type=float,default=.3)
    parser.add_argument("--width",type=int,default=1000)
    parser.add_argument("--height",type=int,default=750)
    parser.add_argument("--fov",type=float,default=130.)
    parser.add_argument("--obj-stride",type=int,default=30,help="OBJ every N source frames; NPZ always contains all frames")
    parser.add_argument("--fp32",action="store_true")
    args = parser.parse_args()
    args.output,args.session = args.output.resolve(),args.session.resolve()
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    if min(args.batch_size,args.frame_batch_size,args.frame_stride,args.width,args.height) < 1 or args.max_frames < 0 or args.obj_stride < 0:
        raise ValueError("invalid batch size, resolution or frame limit")
    if not 60 <= args.fov <= 150:
        raise ValueError("fov must be between 60 and 150 degrees")
    os.environ["EGO_WILOR_ROOT"] = str(args.wilor_root.resolve())
    os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT / "output/.yolo-config"))
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT / "output/.matplotlib"))
    import torch
    from wilor_inference import choose_device,load_detector,load_wilor_model
    torch.set_num_threads(4)
    device = choose_device(args.device)
    print(f"Loading WiLoR on {device}",flush=True)
    pretrained = args.wilor_root / "pretrained_models"
    model,cfg = load_wilor_model(pretrained / "wilor_final.ckpt",pretrained / "model_config.yaml",ROOT / "models/mano")
    model=model.to(device).eval()
    detector=load_detector(pretrained / "detector.pt",device)
    calibrations=cameras_from_ego_yaml(unique_file(args.session,"_calibration_camera.yaml"))
    args.output.mkdir(parents=True)
    (args.output / "run_config.json").write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2)+"\n")
    summaries=[run_camera(camera,calibrations[camera],args,detector,model,cfg,device) for camera in args.cameras]
    (args.output / "summary.json").write_text(json.dumps({"cameras":summaries,"device":str(device),
        "method":"independent per-camera WiLoR with calibrated pinhole views; no multiview fusion or VIO"},indent=2)+"\n")
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == "__main__":
    main()
