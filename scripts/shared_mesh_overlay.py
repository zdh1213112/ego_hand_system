"""Project the same hand mesh through each calibrated camera."""
import hashlib
import numpy as np
from camera_models import project_points
from mesh_rendering import draw_mesh_surface

def mesh_camera_vertices(hand, calibration):
    """Keep the exact transform used by the shared skeleton/mesh renderer."""
    return ((np.asarray(hand['vertices_base_m']) - calibration.T_base_camera[:3, 3])
            @ calibration.T_base_camera[:3, :3])

def mesh_projection_fingerprint(vertices, faces, calibration):
    """Audit all vertices and topology, including vertices outside the image."""
    pixels, valid = project_points(calibration, np.asarray(vertices, dtype=np.float64))
    valid &= np.isfinite(pixels).all(axis=1) & (np.abs(pixels) < 1e6).all(axis=1)
    digest = hashlib.sha256()
    for array, dtype in ((vertices, '<f8'), (faces, '<i4'),
                         (np.where(valid[:, None], pixels, 0.), '<f8'), (valid, 'u1')):
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()

def draw_shared_mesh(image, hand, topology, calibration):
    side = int(hand['side'])
    vertices = mesh_camera_vertices(hand, calibration)
    faces = topology['right' if side else 'left']
    drawn = draw_mesh_surface(image, vertices, faces, calibration.K, side, calibration=calibration)
    return drawn, mesh_projection_fingerprint(vertices, faces, calibration)
