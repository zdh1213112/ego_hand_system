"""Single-camera preview with a shaded MANO hand-mesh panel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from camera_models import project_points

EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGERS = (
    ("THUMB", ((0, 1, 2), (1, 2, 3), (2, 3, 4)), ("CMC", "MCP", "IP")),
    ("INDEX", ((0, 5, 6), (5, 6, 7), (6, 7, 8)), ("MCP", "PIP", "DIP")),
    ("MIDDLE", ((0, 9, 10), (9, 10, 11), (10, 11, 12)), ("MCP", "PIP", "DIP")),
    ("RING", ((0, 13, 14), (13, 14, 15), (14, 15, 16)), ("MCP", "PIP", "DIP")),
    ("PINKY", ((0, 17, 18), (17, 18, 19), (18, 19, 20)), ("MCP", "PIP", "DIP")),
)
COLORS = ((255, 205, 65), (65, 170, 255))  # BGR: left, right
MAIN_CAMERAS = ("cam_0", "cam_1")
CAMERA_WIDTH = 896
CAMERA_HEIGHT = 560
ANGLES_WIDTH = 504
POSE_HEIGHT = 440
HEADER_HEIGHT = 58
CANVAS_WIDTH = CAMERA_WIDTH + ANGLES_WIDTH
CANVAS_HEIGHT = HEADER_HEIGHT + CAMERA_HEIGHT + POSE_HEIGHT
TITLE = "EGO-4B Live WiLoR"
WORLD_CENTER_M = np.asarray((0.0, 0.30, 0.30), dtype=np.float32)
WORLD_SCALE_PX_PER_M = 1050.0
WORLD_CENTER_PX = np.asarray((CANVAS_WIDTH / 2, 280.0), dtype=np.float32)
WORLD_RIGHT = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
WORLD_DOWN = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
MESH_LIGHT = np.asarray((-0.30, -0.55, -0.78), dtype=np.float32)
MESH_LIGHT /= np.linalg.norm(MESH_LIGHT)


def _text(image, value, position, color=(238, 242, 245), scale=0.55, thickness=1):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def geometric_bend_degrees(joints):
    """Return 5 x 3 unsigned bone angles; 0 degrees means straight."""
    result = np.full((5, 3), np.nan, dtype=np.float32)
    if joints is None:
        return result
    points = np.asarray(joints, dtype=np.float32)
    if points.shape != (21, 3):
        return result
    for finger_index, (_, chains, _) in enumerate(FINGERS):
        for joint_index, (parent, joint, child) in enumerate(chains):
            incoming = points[joint] - points[parent]
            outgoing = points[child] - points[joint]
            lengths = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            if not np.isfinite(lengths) or lengths < 1e-8:
                continue
            cosine = np.clip(float(np.dot(incoming, outgoing) / lengths), -1.0, 1.0)
            result[finger_index, joint_index] = np.degrees(np.arccos(cosine))
    return result


def project_metric_scene(points):
    """Fixed frontal projection of cam_0 metric coordinates."""
    world = np.asarray(points, dtype=np.float32)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("expected N x 3 metric points")
    relative = world - WORLD_CENTER_M
    return (WORLD_CENTER_PX + WORLD_SCALE_PX_PER_M * np.column_stack((
        relative @ WORLD_RIGHT, relative @ WORLD_DOWN,
    )))


def reproject_metric_joints(joints, camera):
    """Project the same triangulated 3D joints onto the camera image."""
    pixels = np.full((21, 2), np.nan, dtype=np.float32)
    if joints is None or camera is None:
        return pixels
    world = np.asarray(joints, dtype=np.float64)
    if world.shape != (21, 3):
        return pixels
    finite = np.isfinite(world).all(axis=1)
    if not finite.any():
        return pixels
    rotation = camera.T_base_camera[:3, :3]
    translation = camera.T_base_camera[:3, 3]
    camera_points = (world[finite] - translation) @ rotation
    projected, valid = project_points(camera, camera_points)
    pixels[np.flatnonzero(finite)[valid]] = projected[valid]
    return pixels


def _display_joints(hand):
    """Use the low-lag metric pose in every panel, including the image overlay."""
    return hand.get("joints_3d_base_m_display", hand.get("joints_3d_base_m"))


def fit_mano_mesh_to_stereo(vertices, model_joints, stereo_joints, camera):
    """Rigidly align a WiLoR MANO mesh with the current stereo hand pose.

    WiLoR and stereo have different depth orientation, so wrist translation
    alone can misplace fingertips by several centimetres. No mesh scaling or
    per-joint warping is applied.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    model_joints = np.asarray(model_joints, dtype=np.float64)
    stereo_joints = np.asarray(stereo_joints, dtype=np.float64)
    if (vertices.shape != (778, 3) or model_joints.shape != (21, 3)
            or stereo_joints.shape != (21, 3)
            or not np.isfinite(vertices).all()):
        return None, float("inf")
    valid = np.isfinite(model_joints).all(axis=1) & np.isfinite(stereo_joints).all(axis=1)
    if valid.sum() < 8 or not valid[0]:
        return None, float("inf")
    camera_rotation = camera.T_base_camera[:3, :3]
    source_joints = model_joints @ camera_rotation.T
    source_vertices = vertices @ camera_rotation.T
    source_center = source_joints[valid].mean(axis=0)
    target_center = stereo_joints[valid].mean(axis=0)
    covariance = (source_joints[valid] - source_center).T @ (
        stereo_joints[valid] - target_center
    )
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    fitted_joints = (source_joints[valid] - source_center) @ rotation + target_center
    error_m = float(np.median(np.linalg.norm(fitted_joints - stereo_joints[valid], axis=1)))
    if error_m > 0.030:
        return None, error_m
    fitted_vertices = (source_vertices - source_center) @ rotation + target_center
    return fitted_vertices.astype(np.float32), error_m


def _draw_camera(frame, hands, camera, frame_index, fused, calibration):
    tile = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
    sx = CAMERA_WIDTH / frame.shape[1]
    sy = CAMERA_HEIGHT / frame.shape[0]
    for side, hand in hands.items():
        side = int(side)
        color = COLORS[side]
        box = np.asarray(hand.get("bbox_xyxy", []), dtype=np.float32)
        if box.shape == (4,) and np.isfinite(box).all():
            p0 = (int(round(box[0] * sx)), int(round(box[1] * sy)))
            p1 = (int(round(box[2] * sx)), int(round(box[3] * sy)))
            cv2.rectangle(tile, p0, p1, color, 2, cv2.LINE_AA)
            label = f"{'R' if side else 'L'} {hand.get('confidence', 0):.2f}"
            _text(tile, label, (max(6, p0[0]), max(40, p0[1] - 8)),
                  color, 0.65, 2)
    for side in (0, 1):
        fused_hand = fused.get(side, {})
        color = COLORS[side]
        points = reproject_metric_joints(_display_joints(fused_hand), calibration)
        if not np.isfinite(points).any():
            continue
        finite = np.isfinite(points).all(axis=1)
        scaled = np.rint(np.where(finite[:, None], points, 0) * (sx, sy)).astype(np.int32)
        for first, second in EDGES:
            if finite[first] and finite[second]:
                cv2.line(tile, tuple(scaled[first]), tuple(scaled[second]),
                         color, 2, cv2.LINE_AA)
        for index, point in enumerate(scaled):
            if finite[index]:
                cv2.circle(tile, tuple(point), 4 if index == 0 else 3,
                           (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (CAMERA_WIDTH, 33), (22, 27, 33), -1)
    _text(tile, f"MAIN {camera}  frame {frame_index}  |  stereo 3D reprojection",
          (12, 24), scale=0.62)
    return tile


def _draw_angles_card(panel, side, top, joints, source, fused_valid):
    color = COLORS[side]
    cv2.rectangle(panel, (10, top), (ANGLES_WIDTH - 10, top + 226),
                  (29, 35, 44), -1)
    cv2.rectangle(panel, (10, top), (ANGLES_WIDTH - 10, top + 226),
                  (61, 71, 82), 1)
    _text(panel, "LEFT HAND" if side == 0 else "RIGHT HAND",
          (24, top + 25), color, 0.65, 2)
    _text(panel, f"{source}  |  fused {fused_valid}/21",
          (24, top + 46), (164, 180, 193), 0.45)
    angles = geometric_bend_degrees(joints)
    columns = (190, 290, 390)
    for finger_index, (finger, _, names) in enumerate(FINGERS):
        y = top + 78 + finger_index * 29
        _text(panel, finger, (24, y), color, 0.48)
        for column, (name, value) in enumerate(zip(names, angles[finger_index])):
            label = f"{name} {value:3.0f}" if np.isfinite(value) else f"{name}  --"
            _text(panel, label, (columns[column], y), scale=0.43)
            if np.isfinite(value):
                x = columns[column]
                cv2.rectangle(panel, (x, y + 5), (x + 80, y + 8), (64, 72, 82), -1)
                cv2.rectangle(panel, (x, y + 5),
                              (x + int(round(80 * min(float(value), 150.0) / 150.0)), y + 8),
                              color, -1)


def _draw_mano_mesh(panel, fused, mesh_sources, mesh_faces, cameras):
    """Show shaded MANO surfaces in one frontal view, without a bone overlay."""
    height, width = panel.shape[:2]
    panel[:] = (29, 35, 44)
    faces = np.asarray(mesh_faces, dtype=np.int32) if mesh_faces is not None else np.empty((0, 3), np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or (len(faces) and (faces.min() < 0 or faces.max() >= 778)):
        raise ValueError("MANO faces must index 778 vertices")
    triangles = []
    labels = []
    captions = []
    for side in (0, 1):
        label = "L" if side == 0 else "R"
        hand = fused.get(side, {})
        source = (mesh_sources or {}).get(side)
        joints = _display_joints(hand)
        if source is None or joints is None or not len(faces):
            captions.append(f"{label}: MANO mesh unavailable")
            continue
        camera = (cameras or {}).get(source["camera"])
        if camera is None:
            captions.append(f"{label}: camera calibration unavailable")
            continue
        vertices, error_m = fit_mano_mesh_to_stereo(
            source["vertices"], source["joints"], joints, camera,
        )
        if vertices is None:
            captions.append(f"{label}: mesh fit unavailable")
            continue
        pixels = np.rint(project_metric_scene(vertices)).astype(np.int32)
        face_points = vertices[faces]
        polygons = pixels[faces]
        normals = np.cross(face_points[:, 1] - face_points[:, 0],
                           face_points[:, 2] - face_points[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lighting = np.divide(
            np.abs(normals @ MESH_LIGHT), lengths,
            out=np.zeros_like(lengths), where=lengths > 1e-10,
        )
        brightness = 0.38 + 0.62 * lighting
        shaded = np.clip(np.asarray(COLORS[side]) * brightness[:, None] + 12,
                         0, 255).astype(np.uint8)
        visible = (
            (polygons[:, :, 0].max(axis=1) >= 0)
            & (polygons[:, :, 0].min(axis=1) < width)
            & (polygons[:, :, 1].max(axis=1) >= 55)
            & (polygons[:, :, 1].min(axis=1) < height - 35)
            & (lengths > 1e-10)
        )
        depth = face_points[:, :, 2].mean(axis=1)
        for index in np.flatnonzero(visible):
            triangles.append((float(depth[index]), polygons[index],
                              tuple(int(value) for value in shaded[index]),
                              index % 9 == 0))
        wrist = np.asarray(joints, dtype=np.float32)[0:1]
        labels.append((label, tuple(np.rint(project_metric_scene(wrist)[0]).astype(int)),
                       COLORS[side]))
        captions.append(f"{label}: MANO mesh  |  stereo fit {error_m * 1000:.1f} mm")
    # Far surfaces first, so nearer triangles cover them. This also handles
    # overlap between left and right hands without a separate depth buffer.
    triangles.sort(key=lambda item: item[0], reverse=True)
    for _, polygon, color, wire in triangles:
        cv2.fillConvexPoly(panel, polygon, color, cv2.LINE_AA)
        if wire:
            cv2.polylines(panel, [polygon], True, tuple(int(value * 0.72) for value in color),
                          1, cv2.LINE_AA)
    for label, point, color in labels:
        _text(panel, label, (point[0] - 12, point[1] + 18), color, 0.65, 2)
    cv2.rectangle(panel, (8, 8), (width - 8, 56), (29, 35, 44), -1)
    cv2.rectangle(panel, (8, height - 35), (width - 8, height - 8), (29, 35, 44), -1)
    cv2.rectangle(panel, (8, 8), (width - 8, height - 8), (61, 71, 82), 1)
    _text(panel, "MANO MESH  |  FRONT VIEW", (20, 32), scale=0.71, thickness=2)
    _text(panel, "WiLoR surface aligned to stereo joints  |  no bone overlay",
          (20, 51), (165, 181, 194), 0.48)
    for side, caption in enumerate(captions):
        _text(panel, caption, (20 + side * (width // 2), height - 16),
              COLORS[side], 0.51)


def render_preview(packet, frames, row, output_fps, skipped, camera="cam_0",
                   calibration=None, mesh_sources=None, mesh_faces=None,
                   cameras=None, show_exit_hint=True):
    """Render one result; this function runs off the inference thread."""
    if camera not in MAIN_CAMERAS:
        raise ValueError(f"preview camera must be one of {MAIN_CAMERAS}")
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (18, 22, 28), np.uint8)
    _text(canvas, "EGO-4B  WiLoR HAND POSE", (16, 30), scale=0.77, thickness=2)
    display_limit = output_fps / max(int(row.get("batch_pairs", 1)), 1)
    status = (f"3D {output_fps:.1f} Hz  |  display <= {display_limit:.1f} Hz  |  "
              f"{row['latency_ms']:.0f} ms data latency  |  "
              f"skipped {skipped}  |  side {'ON' if row['side_inferred'] else 'idle'}")
    _text(canvas, status, (16, 51), (164, 204, 217), 0.54)
    if show_exit_hint:
        _text(canvas, "q / Esc: quit", (CANVAS_WIDTH - 150, 43),
              (165, 180, 190), 0.48)
    fused = {int(hand["is_right"]): hand for hand in row["hands"]}
    canvas[HEADER_HEIGHT:HEADER_HEIGHT + CAMERA_HEIGHT, :CAMERA_WIDTH] = _draw_camera(
        frames[camera], row["camera_hands"].get(camera, {}),
        camera, packet["index"][camera], fused, calibration,
    )
    angle_panel = canvas[HEADER_HEIGHT:HEADER_HEIGHT + CAMERA_HEIGHT, CAMERA_WIDTH:]
    _text(angle_panel, "FINGER BEND ANGLES (deg, stereo 3D)", (16, 28),
          scale=0.63, thickness=2)
    _text(angle_panel, "3D bone angle  |  0 = straight", (16, 49),
          (165, 181, 194), 0.47)
    for side, top in ((0, 59), (1, 303)):
        fused_hand = fused.get(side, {})
        fused_points = _display_joints(fused_hand)
        valid = int(fused_hand.get("valid_joints", 0))
        _draw_angles_card(angle_panel, side, top, fused_points, "stereo 3D", valid)
    _draw_mano_mesh(canvas[HEADER_HEIGHT + CAMERA_HEIGHT:], fused,
                    mesh_sources, mesh_faces, cameras)
    return canvas


class LivePreview:
    """Keep one pending render and display it from the OpenCV GUI thread."""

    def __init__(self, camera="cam_0", calibration=None, cameras=None,
                 mesh_faces=None):
        if camera not in MAIN_CAMERAS:
            raise ValueError(f"preview camera must be one of {MAIN_CAMERAS}")
        self.camera = camera
        self.calibration = calibration
        self.cameras = cameras or {camera: calibration}
        self.mesh_faces = mesh_faces
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, CANVAS_WIDTH, CANVAS_HEIGHT)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ego-preview")
        self.future = None

    def submit(self, packet, frames, row, output_fps, skipped, mesh_sources=None):
        if self.future is None:
            self.future = self.executor.submit(
                render_preview, packet, frames, row, output_fps, skipped,
                self.camera, self.calibration, mesh_sources, self.mesh_faces,
                self.cameras,
            )

    def show(self):
        if self.future is not None and self.future.done():
            image = self.future.result()
            self.future = None
            cv2.imshow(TITLE, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return False
        try:
            return cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        try:
            cv2.destroyWindow(TITLE)
        except cv2.error:
            pass
