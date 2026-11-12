#!/usr/bin/env python3
"""Render synchronized, per-camera GEM-X SOMA meshes on pinhole video frames."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gem_multiview_inference import SOMA_TO_21

MESH_STYLE = {
    "style_version": 3,
    "colors_rgb": {"left": "#1964ff", "right": "#ff7b24"},
    "transparency": 0.55,
    "alpha": 0.45,
}
# OpenCV BGR; side 0 = left, side 1 = right. Alpha is opacity.
MESH_COLORS_BGR = {0: (255, 100, 25), 1: (36, 123, 255)}


def project_mesh(vertices, intrinsics):
    """Use the same pinhole approximation as GEM on the unchanged raw pixels."""
    xyz = np.asarray(vertices, dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-5)
    xy = np.full((len(xyz), 2), np.nan)
    homogeneous = xyz[valid] @ np.asarray(intrinsics, dtype=np.float64).T
    xy[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    valid &= np.isfinite(xy).all(axis=1) & (np.abs(xy) < 1e6).all(axis=1)
    return xy, valid


def select_mesh_instance(frame, view):
    """Match the chosen detection/hypothesis, never draw both handedness branches."""
    instances = frame["instances"]
    hypothesis = view.get("hypothesis_index")
    if hypothesis is not None:
        return next((item for item in instances if int(item["track_id"]) == int(hypothesis)), None)
    # Fusion stores the detection ID and original 2D observations, but can
    # relabel physical sides during temporal identity repair. Match the actual
    # observation, rather than assuming the physical side is the GEM branch.
    detection = view.get("detection_index")
    candidates = [item for item in instances
                  if detection is not None and int(item["track_id"]) // 2 == int(detection)]
    observed = np.asarray(view.get("wilor_joints_2d", view.get("joints_2d", [])), dtype=float)
    if not candidates or observed.shape != (21, 2) or not np.isfinite(observed).all():
        return None
    return min(candidates, key=lambda item: float(np.mean(np.linalg.norm(
        np.asarray(item["keypoints_2d"])[list(SOMA_TO_21)] - observed, axis=1))))


def draw_mesh_surface(image, vertices, faces, intrinsics, side, alpha=MESH_STYLE["alpha"], *, calibration=None):
    """Paint a translucent, depth-ordered SOMA surface and sparse mesh edges."""
    xyz = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if calibration is None:
        xy, valid = project_mesh(xyz, intrinsics)
    else:
        from camera_models import project_points
        xy, valid = project_points(calibration, xyz)
        valid &= np.isfinite(xy).all(axis=1) & (np.abs(xy) < 1e6).all(axis=1)
    faces = faces[valid[faces].all(axis=1)]
    if not len(faces):
        return False
    polygons = xy[faces]
    height, width = image.shape[:2]
    visible = ((polygons[:, :, 0].max(axis=1) >= 0)
               & (polygons[:, :, 0].min(axis=1) < width)
               & (polygons[:, :, 1].max(axis=1) >= 0)
               & (polygons[:, :, 1].min(axis=1) < height))
    faces, polygons = faces[visible], polygons[visible]
    if not len(faces):
        return False
    lower = np.maximum(np.floor(polygons.min(axis=(0, 1))).astype(int) - 2, 0)
    upper = np.minimum(np.ceil(polygons.max(axis=(0, 1))).astype(int) + 3, [width, height])
    x0, y0 = lower
    x1, y1 = upper
    roi = image[y0:y1, x0:x1]
    layer = roi.copy()
    triangles = xyz[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    facing = np.abs(normals[:, 2]) / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    palette = np.asarray(MESH_COLORS_BGR[side])
    colors = np.rint((0.55 + 0.45 * facing[:, None]) * palette).astype(int)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))[::-1]
    polygons = np.rint(polygons - lower).astype(np.int32)
    for index in order:
        cv2.fillConvexPoly(layer, polygons[index], tuple(map(int, colors[index])), cv2.LINE_AA)
    # Sparse edges make the topology visible without hiding the 21 landmarks.
    cv2.polylines(layer, list(polygons[::12]), True,
                  tuple(map(int, palette * 0.45)), 1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)
    return True


class GemMeshOverlay:
    """SOMA geometry for the exact native/fused observation chosen by final rendering."""

    def __init__(self, predictions, cameras):
        self.frames, self.topologies = {}, {}
        self.stats = {"requested": 0, "drawn": 0, "missing": 0, "outside_image": 0}
        for camera in cameras:
            data = torch.load(Path(predictions) / camera / "gem_pose.pt",
                              map_location="cpu", weights_only=False, mmap=True)
            self.frames[camera] = {int(frame["frame_id"]): frame for frame in data["frames"]}
            self.topologies[camera] = data["topology"]

    def draw(self, image, camera, frame_id, view, side):
        self.stats["requested"] += 1
        # A held native track must keep its original mesh as well. Detection
        # indices are only local to one frame and must never be reused blindly.
        source_frame = int(view.get("mesh_source_frame_index", frame_id))
        frame = self.frames[camera].get(source_frame)
        hand = select_mesh_instance(frame, view) if frame is not None else None
        if hand is None:
            self.stats["missing"] += 1
            return
        drawn = draw_mesh_surface(image, hand["vertices_camera"],
                                  self.topologies[camera][hand["topology"]],
                                  hand["intrinsics"], side)
        self.stats["drawn" if drawn else "outside_image"] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--tile-width', type=int, default=640)
    args = p.parse_args()
    if args.max_frames < 0 or args.tile_width < 64:
        p.error('invalid max-frames or tile-width')
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    cameras = manifest['camera_ids']
    if len(cameras) != 4:
        raise ValueError('this renderer requires four cameras')
    with (root / 'multiview_frames.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))[:args.max_frames or None]
    if not rows:
        raise ValueError('empty synchronization table')
    width, height = manifest['image_size']
    tw = args.tile_width
    th = round(height * tw / width)
    tw -= tw % 2
    th -= th % 2
    scale = np.array([tw / width, th / height])
    readers, frame_lookup, topologies, ks = {}, {}, {}, {}
    for camera in cameras:
        data = torch.load(args.predictions / camera / 'gem_pose.pt', map_location='cpu', weights_only=False)
        frame_lookup[camera] = {int(f['frame_id']): f for f in data['frames']}
        topologies[camera] = {key: np.asarray(value, dtype=np.int32) for key, value in data['topology'].items()}
        cal = json.loads((root/'calibration'/f'{camera}.json').read_text())
        if cal['model'] != 'PINHOLE':
            raise ValueError('renderer requires pinhole calibration')
        ks[camera] = np.asarray(cal['K'])
        cap = cv2.VideoCapture(str(root/'cameras'/camera/manifest['storage']['video_filename']))
        if not cap.isOpened():
            raise OSError(f'cannot open video for {camera}')
        readers[camera] = [cap, 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fps = float(manifest['fps'])
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (tw*2, th*2))
    if not writer.isOpened():
        raise OSError(f'cannot write {args.output}')
    try:
        for n, row in enumerate(rows):
            tiles = []
            for camera in cameras:
                target = int(row[f'{camera}_frame_index'])
                cap, pos = readers[camera]
                if target < pos:
                    raise ValueError('source frame indices must increase')
                while pos <= target:
                    ok, image = cap.read(); pos += 1
                    if not ok:
                        raise OSError(f'{camera} video ended before frame {target}')
                readers[camera][1] = pos
                frame = frame_lookup[camera].get(target)
                if frame is None:
                    raise ValueError(f'{camera}: no GEM result for frame {target}')
                image = cv2.resize(image, (tw, th))
                overlay = image.copy()
                for hand in frame['instances']:
                    xyz = np.asarray(hand['vertices_camera'], dtype=np.float64)
                    faces = topologies[camera][hand['topology']]
                    z = xyz[:,2]
                    valid = np.isfinite(xyz).all(axis=1) & (z > 1e-5)
                    xy = np.zeros((len(xyz),2), dtype=np.float64)
                    projected = xyz[valid] @ ks[camera].T
                    xy[valid] = projected[:,:2] / projected[:,2,None] * scale
                    good = valid[faces].all(axis=1)
                    tris = faces[good]
                    if len(tris) == 0:
                        continue
                    order = np.argsort(z[tris].mean(axis=1))[::-1]
                    color = MESH_COLORS_BGR[int(hand['side'] == 'right')]
                    for tri in tris[order]:
                        poly = xy[tri]
                        if np.isfinite(poly).all() and np.abs(poly).max() < 1e6:
                            cv2.fillConvexPoly(overlay, np.round(poly).astype(np.int32), color, cv2.LINE_AA)
                cv2.addWeighted(overlay, MESH_STYLE["alpha"], image, 1.0 - MESH_STYLE["alpha"], 0, image)
                cv2.putText(image, f'{camera}  frame {target}  hands {len(frame["instances"])}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)
                tiles.append(image)
            writer.write(np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))))
            if (n+1) % 100 == 0:
                print(f'[mesh] {n+1}/{len(rows)}', flush=True)
    finally:
        writer.release()
        for cap, _ in readers.values():
            cap.release()
    print(f'[mesh] saved {len(rows)} frames: {args.output}', flush=True)


if __name__ == '__main__':
    main()
