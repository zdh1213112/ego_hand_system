"""Palm-local transfer used for source selection and the final sequence fit."""
import numpy as np
from scipy.spatial.transform import Rotation

TIPS = [4, 8, 12, 16, 20]

def prepared_targets(data, endpoint, ref):
    """Rotate the palm around the moving fingertips. Transfer the residual drawn at 883 in local palm axes."""
    joints = data["joints"].astype(np.float64)
    wrists = joints[:, 0]
    rotations = Rotation.from_rotvec(data["global_orient"].astype(float))
    relative = rotations[ref].inv() * Rotation.from_rotvec(endpoint["global_orient"])
    palm = rotations.as_matrix()
    local_joints = (joints - wrists[:, None]) @ palm
    reach = np.linalg.norm(local_joints[:, TIPS[1:]], axis=-1).mean(1)
    amount = np.clip(reach / reach[ref], 0., 1.1)
    applied = Rotation.from_rotvec(amount[:, None] * relative.as_rotvec())
    new_orientation = rotations * applied
    world_delta = (new_orientation * rotations.inv()).as_matrix()
    pivot = joints[:, TIPS[1:]].mean(1)
    wrist = pivot + np.einsum("nij,nj->ni", world_delta, wrists - pivot)
    ink_residual = rotations[ref].inv().apply(endpoint["joints"][0] - wrist[ref])
    wrist += amount[:, None] * rotations.apply(np.broadcast_to(ink_residual, wrist.shape))

    # The reference pose applies at one hand orientation. As the pose or viewing
    # direction changes, give this frame's measured surface positions increasing influence.
    pose_difference = np.linalg.norm(local_joints[:, 1:] - local_joints[ref, 1:], axis=-1)
    rms_mm = np.sqrt((pose_difference ** 2).mean(1)) * 1000
    view_angle = (rotations[ref].inv() * rotations).magnitude()
    likeness = np.exp(-.5 * ((rms_mm / 24.) ** 2 + (view_angle / np.deg2rad(35.)) ** 2))
    basis = palm @ palm[ref].T
    def carry_residual(source, approved):
        offset = approved - source[ref]
        return source + likeness[:, None, None] * np.einsum("nij,kj->nki", basis, offset)
    tips = carry_residual(joints[:, TIPS], endpoint["joints"][TIPS])
    mounts = carry_residual(data["target24"], endpoint["mounted"])
    return dict(wrist=wrist, orient=new_orientation.as_rotvec(), reach=amount, similar=likeness,
                tips=tips, mounts=mounts)
