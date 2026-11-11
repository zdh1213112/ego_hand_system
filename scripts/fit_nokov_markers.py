#!/usr/bin/env python3
"""Fit a named NOKOV hand marker set in its recorded world coordinate frame.

The joint stage uses surface markers as approximate landmarks. An optional
surface stage calibrates fixed MANO vertex attachments, then fits all 24
markers. Neither stage uses XINGYING's inferred skeleton as ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from fit_mano_sequence import (
    MEDIAPIPE_NAMES, canonicalize_observations, import_mano,
    optimize_track, physical_result, run_model, robust_weighted_loss, weighted_kabsch,
)

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
MARKERS = ("WristM", "WristIn", "WristOut", "HandOffset") + tuple(
    f"Finger{finger}{i}" for finger in FINGERS for i in range(1, 5)
)


def load_markers(directory: Path, markerset: str):
    metadata = json.loads((directory / "capture_metadata.json").read_text())
    if metadata["position_unit"] != "mm":
        raise ValueError("NOKOV CSV position unit must be mm")
    with (directory / "nokov_frames.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    frames = np.array([int(r["frame_no"]) for r in rows], dtype=np.int64)
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("frame numbers must increase uniquely")
    lookup = {int(v): i for i, v in enumerate(frames)}
    names = {v: i for i, v in enumerate(MARKERS)}
    points = np.full((len(frames), len(MARKERS), 3), np.nan, dtype=np.float32)
    seen = np.zeros(points.shape[:2], dtype=bool)
    with (directory / "nokov_markers.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["markerset_name"] != markerset:
                continue
            i, j = lookup[int(row["frame_no"])], names[row["marker_name"]]
            if seen[i, j]:
                raise ValueError("duplicate marker in a frame")
            seen[i, j] = True
            p = np.array([float(row[f"{a}_mm"]) for a in "xyz"]) / 1000
            if row["valid"] == "1" and np.isfinite(p).all() and np.linalg.norm(p) > 1e-8:
                points[i, j] = p
    if not seen.all():
        raise ValueError("missing CSV rows for the selected marker set")
    fps = float(metadata["frame_rate_hz_from_description"])
    return dict(markers_world_m=points, marker_valid=np.isfinite(points).all(-1),
                frame_no=frames, time_s=(frames-frames[0])/fps, fps=np.array(fps),
                device_timestamp_raw=np.array([int(r["device_timestamp_raw"]) for r in rows]),
                receive_unix_ns=np.array([int(r["receive_unix_ns"]) for r in rows]))


def joint_targets(points):
    # The wrist centre is halfway between the two wrist-side markers.
    wrist = (points[:, 1] + points[:, 2]) / 2
    return np.concatenate((wrist[:, None], points[:, 4:]), axis=1)


def observations(target, side):
    valid = np.isfinite(target).all(-1)
    pixels = np.full((*target.shape[:2], 2), np.nan, dtype=np.float32)
    return canonicalize_observations(dict(
        positions=target, valid=valid, observed=valid, confidence=valid.astype(np.float32),
        left_px=pixels, right_px=pixels.copy(), rotation=np.eye(3, dtype=np.float32),
        p1=np.zeros((3, 4), dtype=np.float32), p2=np.zeros((3, 4), dtype=np.float32)), side)


def options(shape_iterations=350, pose_iterations=240, components=30):
    return SimpleNamespace(
        pca_components=components, min_fit_observed_points=12, max_unobserved_gap=5,
        rigid_initialization=True, shape_frames=96, shape_iterations=shape_iterations,
        pose_iterations=pose_iterations, pose_window=0, pose_overlap=0, learning_rate=0.018,
        w_3d=1., w_2d=0., w_pose=0.00004, w_shape=0.00015,
        w_temporal=0.0008, w_rigid_temporal=0.0001, w_acceleration=0.0004,
        boundary_weight=0., max_orient_step_deg=180., max_translation_step_m=1.,
        max_pose_step=0., image_rigid_alignment=False)


def calibrate_attachments(vertices, markers, faces):
    """Least-squares fixed barycentric attachments, restricted to nearby faces.

    One face and one barycentric coordinate is shared by the entire recording.
    Each candidate triangle is solved with the simplex constraint, including
    all three edges, so the attachment cannot leave the MANO surface.
    """
    face_ids, barycentrics = [], []
    for marker in range(markers.shape[1]):
        valid = np.isfinite(markers[:, marker]).all(-1)
        v, y = vertices[valid], markers[valid, marker]
        distances = np.median(np.linalg.norm(v-y[:, None], axis=-1), axis=0)
        nearest = np.argsort(distances)[:16]
        candidates = np.flatnonzero(np.isin(faces, nearest).any(-1))
        best = (np.inf, None, None)
        for face_id in candidates:
            tri = v[:, faces[face_id]]
            a = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1).reshape(-1, 2)
            b = (y-tri[:, 0]).reshape(-1)
            uv = np.linalg.lstsq(a, b, rcond=None)[0]
            choices = []
            if np.all(uv >= 0) and uv.sum() <= 1:
                choices.append(np.array([1-uv.sum(), *uv]))
            for i, j in ((0, 1), (0, 2), (1, 2)):
                edge = tri[:, j]-tri[:, i]
                t = np.clip(np.sum(edge*(y-tri[:, i]))/max(np.sum(edge**2), 1e-12), 0, 1)
                w = np.zeros(3); w[i], w[j] = 1-t, t
                choices.append(w)
            for weights in choices:
                error = np.mean(np.sum((np.einsum('fvc,v->fc', tri, weights)-y)**2, axis=-1))
                if error < best[0]:
                    best = error, int(face_id), weights
        face_ids.append(best[1]); barycentrics.append(best[2])
    return np.array(face_ids), np.array(barycentrics, dtype=np.float32)


def optimize_surface(model, markers, target, initial, attachment, device,
                     iterations=280, fit_shape=False, temporal=True):
    import torch
    from mano.lbs import batch_rodrigues
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)
    y = tensor(markers)
    valid = torch.isfinite(y).all(-1)
    target_t = tensor(target)
    target_valid = torch.isfinite(target_t).all(-1)
    pose = tensor(initial['hand_pose_pca']).clone().requires_grad_()
    orient = tensor(initial['global_orient']).clone().requires_grad_()
    transl = tensor(initial['translation']).clone().requires_grad_()
    betas = tensor(initial['betas']).reshape(1, 10).clone().requires_grad_(fit_shape)
    attachment_faces, attachment_weights = attachment
    indices = torch.as_tensor(model.faces[attachment_faces], dtype=torch.long, device=device)
    weights = tensor(attachment_weights)
    parameters = [pose, orient, transl] + ([betas] if fit_shape else [])
    optimizer = torch.optim.Adam(parameters, lr=0.006)
    for iteration in range(iterations):
        if iteration == int(iterations * .65):
            for group in optimizer.param_groups:
                group['lr'] *= .3
        optimizer.zero_grad(set_to_none=True)
        vertices, joints, _ = run_model(model, betas.expand(len(y), -1), orient, pose, transl)
        predicted = (vertices[:, indices]*weights[None, :, :, None]).sum(2)
        loss = robust_weighted_loss(predicted-y, valid.float(), .003)
        # A weak semantic tether discourages surface correspondences from
        # replacing the intended finger with a neighbouring finger.
        loss = loss + .035*robust_weighted_loss(joints-target_t, target_valid.float(), .01)
        loss = loss + .000035*pose.square().mean() + .00015*betas.square().mean()
        if temporal and len(y) > 2:
            loss = loss + .0008*(pose[1:]-pose[:-1]).square().mean()
            loss = loss + .0004*(pose[2:]-2*pose[1:-1]+pose[:-2]).square().mean()
            rotations = batch_rodrigues(orient)
            loss = loss + .0002*(rotations[1:]-rotations[:-1]).square().mean()
            loss = loss + .02*(transl[2:]-2*transl[1:-1]+transl[:-2]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        if fit_shape:
            with torch.no_grad():
                betas.clamp_(-4., 4.)
    with torch.no_grad():
        vertices, joints, out = run_model(model, betas.expand(len(y), -1), orient, pose, transl)
        predicted = (vertices[:, indices]*weights[None, :, :, None]).sum(2)
        error = torch.linalg.vector_norm(predicted-y, dim=-1)
    return {key: val.detach().cpu().numpy() for key, val in dict(
        vertices=vertices, joints=joints, predicted_markers=predicted,
        marker_error_m=error, hand_pose_pca=pose, global_orient=orient,
        translation=transl, betas=betas[0],
        hand_pose_axis_angle=out.hand_pose.reshape(-1, 15, 3),
        full_pose_axis_angle=out.full_pose.reshape(-1, 16, 3)).items()}


def initialize_surface_rigid(model, markers, pose, betas, attachment, device):
    """Estimate rigid motion per frame, independent of sparse pose seeds."""
    import torch
    from scipy.spatial.transform import Rotation
    count = len(markers)
    with torch.no_grad():
        vertices, joints, _ = run_model(model, torch.as_tensor(betas, device=device).expand(count, -1),
            torch.zeros((count,3),device=device), torch.as_tensor(pose,device=device),
            torch.zeros((count,3),device=device))
    vertices,joints=vertices.cpu().numpy(),joints.cpu().numpy()
    local=(vertices[:,model.faces[attachment[0]]]*attachment[1][None,:,:,None]).sum(2)
    palm=np.array([0,1,2,3,8,12,16,20])
    orient,transl=[],[]
    for i in range(count):
        keep=palm[np.isfinite(markers[i,palm]).all(-1)]
        if len(keep)<3:
            keep=np.flatnonzero(np.isfinite(markers[i]).all(-1))
        if len(keep)<3:
            raise ValueError('fewer than three markers; cannot initialize rigid pose')
        r=weighted_kabsch(local[i,keep],markers[i,keep],np.ones(len(keep)))
        t=markers[i,keep].mean(0)-r@(local[i,keep].mean(0)-joints[i,0])-joints[i,0]
        orient.append(Rotation.from_matrix(r).as_rotvec());transl.append(t)
    return np.asarray(orient,dtype=np.float32),np.asarray(transl,dtype=np.float32)


def fit_full_sequence(model, data, target, calibration, calibration_ids, args):
    from scipy.spatial.transform import Rotation, Slerp
    out = args.output
    markers = data['markers_world_m'].copy()
    if args.handedness == 'Left':
        markers[..., 0] *= -1
    canonical_target = observations(target, args.handedness)['positions']
    surface_file = out/'surface_calibration.npz'
    if surface_file.exists():
        surface = dict(np.load(surface_file))
        attachment = (surface.pop('attachment_face_ids'), surface.pop('attachment_barycentric'))
    else:
        surface = calibration
        for cycle in range(3):
            attachment = calibrate_attachments(surface['vertices'], markers[calibration_ids], model.faces)
            surface = optimize_surface(model, markers[calibration_ids], canonical_target[calibration_ids],
                                       surface, attachment, args.device, iterations=320,
                                       fit_shape=True, temporal=False)
            print(f'surface calibration {cycle+1}: median {np.nanmedian(surface["marker_error_m"])*1000:.2f} mm; '
                  f'p95 {np.nanpercentile(surface["marker_error_m"],95)*1000:.2f} mm', flush=True)
        np.savez_compressed(surface_file, **surface, attachment_face_ids=attachment[0],
                            attachment_barycentric=attachment[1])
    n = len(target)
    warm_pose = np.column_stack([np.interp(np.arange(n), calibration_ids, surface['hand_pose_pca'][:, k])
                                for k in range(args.components)]).astype(np.float32)
    times = np.clip(np.arange(n), calibration_ids[0], calibration_ids[-1])
    warm_orient = Slerp(calibration_ids, Rotation.from_rotvec(surface['global_orient']))(times).as_rotvec().astype(np.float32)
    # Rigid translation follows the observed wrist instead of interpolating
    # across long intervals of hand movement.
    wrist = canonical_target[:, 0].copy()
    for k in range(3):
        good = np.isfinite(wrist[:, k]); wrist[:, k] = np.interp(np.arange(n), np.flatnonzero(good), wrist[good, k])
    warm_transl = wrist.copy()
    with __import__('torch').no_grad():
        import torch
        for start in range(0, n, 256):
            end = min(start+256, n)
            _, joints, _ = run_model(model, torch.as_tensor(surface['betas'], device=args.device).expand(end-start, -1),
                torch.as_tensor(warm_orient[start:end], device=args.device),
                torch.as_tensor(warm_pose[start:end], device=args.device), torch.zeros((end-start,3),device=args.device))
            warm_transl[start:end] -= joints[:, 0].cpu().numpy()
    results = []
    chunk_dir = out/'chunks'; chunk_dir.mkdir(exist_ok=True)
    for start in range(0, n, 256):
        end = min(start+256, n)
        lo, hi = max(0, start-24), min(n, end+24)
        cache = chunk_dir/f'frames_{start:05d}_{end:05d}.npz'
        if cache.exists():
            result = dict(np.load(cache))
        else:
            initial = dict(betas=surface['betas'], hand_pose_pca=warm_pose[lo:hi],
                           global_orient=warm_orient[lo:hi], translation=warm_transl[lo:hi])
            result = optimize_surface(model, markers[lo:hi], canonical_target[lo:hi], initial,
                                       attachment, args.device, iterations=300)
            result = {k: v[start-lo:end-lo] if k != 'betas' else v for k,v in result.items()}
            np.savez_compressed(cache, **result)
        results.append(result)
        print(f'frames {end}/{n}: marker median {np.nanmedian(result["marker_error_m"])*1000:.2f} mm', flush=True)
    merged = {k: np.concatenate([r[k] for r in results]) for k in results[0] if k != 'betas'}
    merged['betas'] = surface['betas']
    refined=[]
    refinement_dir=out/'refined_chunks';refinement_dir.mkdir(exist_ok=True)
    for start in range(0,n,256):
        end=min(n,start+256);lo,hi=max(0,start-24),min(n,end+24)
        cache=refinement_dir/f'frames_{start:05d}_{end:05d}.npz'
        if cache.exists():
            result=dict(np.load(cache))
        else:
            pose=merged['hand_pose_pca'][lo:hi].copy()
            bad=np.nanmedian(merged['marker_error_m'][lo:hi],axis=1)>.008
            pose[bad]=warm_pose[lo:hi][bad]
            orient,transl=initialize_surface_rigid(model,markers[lo:hi],pose,surface['betas'],attachment,args.device)
            initial=dict(betas=surface['betas'],hand_pose_pca=pose,global_orient=orient,translation=transl)
            result=optimize_surface(model,markers[lo:hi],canonical_target[lo:hi],initial,
                                    attachment,args.device,iterations=320)
            result={k:v[start-lo:end-lo] if k!='betas' else v for k,v in result.items()}
            np.savez_compressed(cache,**result)
        refined.append(result)
        print(f'refined {end}/{n}: median {np.nanmedian(result["marker_error_m"])*1000:.2f} mm; '
              f'frame max median {np.nanmedian(result["marker_error_m"],axis=1).max()*1000:.2f} mm',flush=True)
    merged={k:np.concatenate([r[k] for r in refined]) for k in refined[0] if k!='betas'}
    merged['betas']=surface['betas']
    faces = np.array(model.faces)
    canonical = dict(merged)
    np.savez_compressed(out/'mano_parameters.npz', **{k:v for k,v in canonical.items()
                        if k in ('betas','global_orient','translation','hand_pose_pca','hand_pose_axis_angle','full_pose_axis_angle')},
                        frame_no=data['frame_no'], time_s=data['time_s'],
                        mano_convention=np.array('wilor_right_canonical'), handedness=np.array(args.handedness),
                        model=np.array('MANO_RIGHT.pkl'), pca_components=np.array(args.components), flat_hand_mean=np.array(False))
    if args.handedness == 'Left':
        for k in ('vertices', 'joints', 'predicted_markers', 'translation'):
            merged[k] = merged[k].copy(); merged[k][..., 0] *= -1
        faces = faces[:, [0, 2, 1]]
    supported = data['marker_valid'].sum(-1) >= 16
    np.savez_compressed(out/'reconstruction.npz', **merged, faces=faces,
                        frame_no=data['frame_no'], time_s=data['time_s'], fps=data['fps'],
                        marker_valid=data['marker_valid'], markers_world_m=data['markers_world_m'],
                        joint_names=np.array(MEDIAPIPE_NAMES), marker_names=np.array(MARKERS),
                        supported=supported, coordinate_frame=np.array('NOKOV world, metres'),
                        handedness=np.array(args.handedness), attachment_face_ids=attachment[0],
                        attachment_barycentric=attachment[1])
    with (out/'joints_21_world.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['frame_no','time_s','joint_id','joint_name','x_m','y_m','z_m','supported'])
        for i in range(n):
            for j, name in enumerate(MEDIAPIPE_NAMES):
                writer.writerow([int(data['frame_no'][i]), data['time_s'][i], j, name, *merged['joints'][i,j], int(supported[i])])
    with (out/'marker_residuals.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['frame_no','time_s','valid_markers','median_mm','p95_mm','max_mm'])
        for i in range(n):
            error = merged['marker_error_m'][i]*1000
            writer.writerow([int(data['frame_no'][i]), data['time_s'][i], int(data['marker_valid'][i].sum()),
                             np.nanmedian(error), np.nanpercentile(error,95), np.nanmax(error)])
    error = merged['marker_error_m']*1000
    per_marker = [dict(name=name, valid_frames=int(data['marker_valid'][:,i].sum()),
                       median_mm=float(np.nanmedian(error[:,i])),p95_mm=float(np.nanpercentile(error[:,i],95)))
                  for i,name in enumerate(MARKERS)]
    summary = dict(source=str(args.nokov.resolve()), markerset=args.markerset,
        handedness=args.handedness, handedness_provenance='explicit user confirmation',
        frames=n, fps=float(data['fps']), duration_s=float(data['time_s'][-1]),
        valid_marker_fraction=float(data['marker_valid'].mean()), supported_frames=int(supported.sum()),
        median_marker_residual_mm=float(np.nanmedian(error)),p95_marker_residual_mm=float(np.nanpercentile(error,95)),
        per_marker=per_marker, betas=merged['betas'].tolist(), calibration_ids=calibration_ids.tolist(),
        calibration_frames=len(calibration_ids), pca_components=args.components, vertices=778, joints=21, faces=len(faces),
        coordinate_frame='Recorded NOKOV world; metres; no camera registration',
        model='MANO_RIGHT.pkl',model_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'models/mano/MANO_RIGHT.pkl').read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((args.nokov/'nokov_markers.csv').read_bytes()).hexdigest(),
        method=f'21 marker-derived joint proxies for initialization; fixed face/barycentric surface attachments calibrated on 96 frames; shared shape; {args.components} PCA pose components; all frames optimized in 256-frame cores with 24-frame context; per-frame palm Kabsch rigid refinement; rotation-matrix temporal penalty',
        limitations=['Surface attachment locations are estimated from this recording, not independently measured.',
                     'Marker residuals are fitting residuals, not anatomical joint accuracy.',
                     'The glove, skin thickness and marker radius are absorbed by the approximate surface attachment model.',
                     'No video synchronization or camera overlay is asserted. All geometry is in NOKOV world coordinates.'],
        torch_version=__import__('torch').__version__, device=args.device)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_marker','calibration_ids')},ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nokov", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handedness", choices=("Left", "Right"), required=True)
    parser.add_argument("--markerset", default="right_hand_2mm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--components", type=int, default=30)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.components <= 45:
        raise ValueError('--components must be between 1 and 45')
    cache_config=dict(
        schema='nokov_surface_mano_v2', handedness=args.handedness,
        markerset=args.markerset, components=args.components,
        marker_csv_sha256=hashlib.sha256((args.nokov/'nokov_markers.csv').read_bytes()).hexdigest(),
        frame_csv_sha256=hashlib.sha256((args.nokov/'nokov_frames.csv').read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((root/'models/mano/MANO_RIGHT.pkl').read_bytes()).hexdigest())
    config_path=args.output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cache_config:
        raise ValueError('output contains a different input/model/hand/config; choose a new output directory')
    config_path.write_text(json.dumps(cache_config,indent=2)+'\n')
    data = load_markers(args.nokov, args.markerset)
    target = joint_targets(data["markers_world_m"])
    np.savez_compressed(args.output / "observations.npz", **data,
                        joint_proxy_world_m=target, marker_names=np.array(MARKERS),
                        joint_names=np.array(MEDIAPIPE_NAMES))
    mano = import_mano(root / "third_party/MANO")
    model = mano.load(model_path=str(root / "models/mano"), is_rhand=True,
                      use_pca=True, num_pca_comps=args.components, batch_size=1,
                      flat_hand_mean=False).to(args.device)
    quality = np.isfinite(target).all(-1).sum(-1)
    candidates = np.flatnonzero(quality == 21)
    if len(candidates)<3:
        raise ValueError('at least three fully observed calibration frames are required')
    ids = candidates[np.linspace(0, len(candidates)-1, min(96,len(candidates))).astype(int)]
    calibration_file = args.output / "joint_calibration.npz"
    if calibration_file.exists():
        result = dict(np.load(calibration_file))
        saved_ids = result.pop('calibration_ids')
        if not np.array_equal(saved_ids, ids):
            raise ValueError('calibration cache frame indices changed')
    else:
        start = time.monotonic()
        calibration_options = options(400, 320, args.components)
        calibration_options.w_temporal = 0.
        calibration_options.w_rigid_temporal = 0.
        calibration_options.w_acceleration = 0.
        result = optimize_track(model, observations(target[ids], args.handedness),
                                calibration_options, args.device)
        np.savez_compressed(calibration_file, **result, calibration_ids=ids)
        print("joint calibration", time.monotonic()-start,
              "median mm", np.nanmedian(result["joint_error_m"])*1000,
              "p95 mm", np.nanpercentile(result["joint_error_m"], 95)*1000,
              "betas", result["betas"], flush=True)
    physical, faces = physical_result(result, args.handedness, model.faces)
    np.savez_compressed(args.output / "joint_calibration_world.npz", **physical,
                        faces=faces, calibration_ids=ids)
    if args.calibration_only:
        return
    fit_full_sequence(model, data, target, result, ids, args)


if __name__ == "__main__":
    main()
