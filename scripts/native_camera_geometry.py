"""Physical camera transforms and native KB projection, with no rectification."""
import numpy as np


def reference_to_camera(reference, camera):
    ref = reference.T_base_camera
    target = camera.T_base_camera
    return target[:3, :3].T @ ref[:3, :3], target[:3, :3].T @ (ref[:3, 3] - target[:3, 3])


def project_native_torch(points, rotation, translation, K, distortion):
    import torch
    camera = torch.einsum("ij,bkj->bki", rotation, points) + translation
    xy = camera[..., :2]
    radius = torch.sqrt(xy.square().sum(-1) + 1e-14)
    theta = torch.atan2(radius, camera[..., 2])
    theta2 = theta.square()
    theta_d = theta * (1 + distortion[0]*theta2 + distortion[1]*theta2**2
                      + distortion[2]*theta2**3 + distortion[3]*theta2**4)
    normalized = xy * (theta_d / radius)[..., None]
    return torch.stack((K[0,0]*normalized[...,0] + K[0,2],
                        K[1,1]*normalized[...,1] + K[1,2]), dim=-1)


def canonical_camera_rotation(rotation, side):
    from mano_conventions import MIRROR_X
    return rotation if int(side) == 1 else MIRROR_X @ rotation @ MIRROR_X


def rotate_mano_geometry(vertices, joints, translation, rotation):
    # MANO global orientation rotates around the wrist pivot. Transfer its
    # pivot offset to translation so parameter replay and physical geometry agree.
    root_offset = rotation @ joints[0] - joints[0]
    return tuple(x.astype(np.float32) for x in (
        vertices @ rotation.T - root_offset,
        joints @ rotation.T - root_offset,
        rotation @ translation + root_offset, root_offset))


def rotate_mano_pose(full_pose_axis_angle, rotation):
    from scipy.spatial.transform import Rotation
    pose = np.asarray(full_pose_axis_angle).reshape(16, 3)
    matrices = Rotation.from_rotvec(pose).as_matrix()
    return (rotation @ matrices[0]).astype(np.float32)[None], matrices[1:].astype(np.float32)
