"""Overlay saved FAH meshes using their original inference intrinsics."""
import json
from pathlib import Path

import numpy as np

from render_gem_multiview import MESH_STYLE, draw_mesh_surface
from wilor_native_mesh_overlay import select_native_hand


class FahNativeMeshOverlay:
    def __init__(self, predictions, records):
        self.root = Path(predictions).resolve()
        self.records, self.meshes, self.intrinsics, self.faces = {}, {}, {}, {}
        self.calibrations = {}
        self.audits = []
        self.stats = {"requested": 0, "drawn": 0, "missing": 0, "outside_image": 0}
        for camera, rows in records.items():
            self.records[camera] = {int(row["source_frame_index"]): row for row in rows.values()}
            with np.load(self.root / camera / "meshes.npz", allow_pickle=False) as archive:
                vertices = archive["vertices_camera_m"]
                frames, hypotheses, sides = (archive[key] for key in
                    ("source_frame_indices", "hypothesis_indices", "is_right"))
                K, faces = archive["K"], archive["faces_right"]
                if (vertices.shape != (len(frames), 778, 3) or not np.isfinite(vertices).all()
                        or not (len(frames) == len(hypotheses) == len(sides))
                        or K.shape != (3, 3) or not np.isfinite(K).all()):
                    raise ValueError(f"{camera}: invalid FAH mesh archive")
                self.meshes[camera] = {}
                for frame, hypothesis, side, mesh in zip(frames, hypotheses, sides, vertices):
                    key = (int(frame), int(hypothesis))
                    if key in self.meshes[camera]:
                        raise ValueError(f"{camera}: duplicate FAH mesh {key}")
                    self.meshes[camera][key] = (mesh.copy(), int(side))
                if "kb_calibration_json" in archive:
                    from ego_hand_inference.ego_data.calibration import CameraCalibration
                    self.calibrations[camera] = CameraCalibration.from_dict(json.loads(archive["kb_calibration_json"].item()))
                self.intrinsics[camera] = K.copy()
                self.faces[camera] = {1: faces.copy(), 0: faces[:, [0, 2, 1]].copy()}
            for frame, record in self.records[camera].items():
                for hand in record["hands"]:
                    key = (frame, int(hand["hypothesis_index"]))
                    if key not in self.meshes[camera] or self.meshes[camera][key][1] != int(hand["is_right"]):
                        raise ValueError(f"{camera}: FAH mesh does not match prediction {key}")

    def draw(self, image, camera, frame_id, view, side):
        self.stats["requested"] += 1
        source = int(view.get("mesh_source_frame_index", frame_id))
        row = self.records[camera].get(source)
        hand = select_native_hand(row["hands"], view) if row else None
        audit = {"camera": camera, "frame_index": frame_id, "source_frame_index": source, "display_side": side}
        if hand is None:
            self.stats["missing"] += 1
            self.audits.append({**audit, "missing": True})
            return
        hypothesis = int(hand["hypothesis_index"])
        vertices, native_side = self.meshes[camera][source, hypothesis]
        options = {"calibration": self.calibrations[camera]} if camera in self.calibrations else {}
        drawn = draw_mesh_surface(image, vertices, self.faces[camera][native_side], self.intrinsics[camera], side, **options)
        self.stats["drawn" if drawn else "outside_image"] += 1
        self.audits.append({**audit, "hypothesis_index": hypothesis, "native_side": native_side, "mesh_drawn": bool(drawn)})

    def summary(self):
        return {"model": "FAH native MANO", "predictions": str(self.root),
                "geometry": "original FAH fitted per-camera vertices; camera coordinates in metres",
                "projection": {cam: "KB" if cam in self.calibrations else "pinhole" for cam in self.intrinsics},
                "intrinsics": {camera: K.tolist() for camera, K in self.intrinsics.items()},
                "shared_multiview_mesh": False, **MESH_STYLE, **self.stats}
