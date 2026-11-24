"""Calibrated shared hand mesh drawing."""

import cv2
import numpy as np

MESH_STYLE = {
    "style_version": 3,
    "colors_rgb": {"left": "#1964ff", "right": "#ff7b24"},
    "transparency": 0.55,
    "alpha": 0.45,
}
MESH_COLORS_BGR = {0: (255, 100, 25), 1: (36, 123, 255)}


def project_mesh(vertices, intrinsics):
    """Project a mesh through pinhole intrinsics."""
    xyz = np.asarray(vertices, dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-5)
    xy = np.full((len(xyz), 2), np.nan)
    homogeneous = xyz[valid] @ np.asarray(intrinsics, dtype=np.float64).T
    xy[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    valid &= np.isfinite(xy).all(axis=1) & (np.abs(xy) < 1e6).all(axis=1)
    return xy, valid


def draw_mesh_surface(
    image,
    vertices,
    faces,
    intrinsics,
    side,
    alpha=MESH_STYLE["alpha"],
    *,
    calibration=None,
):
    """Paint a translucent, depth-ordered MANO surface and sparse mesh edges."""
    xyz = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if calibration is None:
        xy, valid = project_mesh(xyz, intrinsics)
    else:
        from ego_hand_inference.camera_models import project_points

        xy, valid = project_points(calibration, xyz)
        valid &= np.isfinite(xy).all(axis=1) & (np.abs(xy) < 1e6).all(axis=1)
    faces = faces[valid[faces].all(axis=1)]
    if not len(faces):
        return False
    polygons = xy[faces]
    height, width = image.shape[:2]
    visible = (
        (polygons[:, :, 0].max(axis=1) >= 0)
        & (polygons[:, :, 0].min(axis=1) < width)
        & (polygons[:, :, 1].max(axis=1) >= 0)
        & (polygons[:, :, 1].min(axis=1) < height)
    )
    faces, polygons = faces[visible], polygons[visible]
    if not len(faces):
        return False
    lower = np.maximum(np.floor(polygons.min(axis=(0, 1))).astype(int) - 2, 0)
    upper = np.minimum(
        np.ceil(polygons.max(axis=(0, 1))).astype(int) + 3, [width, height]
    )
    x0, y0 = lower
    x1, y1 = upper
    roi = image[y0:y1, x0:x1]
    layer = roi.copy()
    triangles = xyz[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    facing = np.abs(normals[:, 2]) / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    palette = np.asarray(MESH_COLORS_BGR[side])
    colors = np.rint((0.55 + 0.45 * facing[:, None]) * palette).astype(int)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))[::-1]
    polygons = np.rint(polygons - lower).astype(np.int32)
    for index in order:
        cv2.fillConvexPoly(
            layer, polygons[index], tuple(map(int, colors[index])), cv2.LINE_AA
        )
    # Sparse edges make the topology visible without hiding the 21 landmarks.
    cv2.polylines(
        layer,
        list(polygons[::12]),
        True,
        tuple(map(int, palette * 0.45)),
        1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)
    return True
