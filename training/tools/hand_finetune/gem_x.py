"""Fine-tune GEM-X EgoHand through its native SOMA rig using shared landmarks."""
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .data import project


# Wrist, then proximal-to-distal landmarks for thumb, index, middle, ring,
# pinky. SOMA has an additional palm/metacarpal joint on the four long fingers.
SOMA_TO_MANO21 = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14,
                  16, 17, 18, 19, 21, 22, 23, 24)


class GEMFineTune(nn.Module):
    size = 256
    canonical_side = 0
    filename = "egohand.ckpt"

    def __init__(self, checkpoint, soma_assets):
        super().__init__()
        from gem.pose.api import hand_config, load_hand_checkpoint
        from gem.pose.geometry.hand import HandGeometry

        self.model = load_hand_checkpoint(checkpoint)
        self.release = hand_config()
        self.geometry = HandGeometry(soma_assets)
        self.model.requires_grad_(False)
        self.model.task.head.decpose.requires_grad_(True)
        self.model.task.head.deccam.requires_grad_(True)
        self.model.kp2d_head.readout.coord.requires_grad_(True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def objective(self, batch):
        from gem.common.camera import compute_transl_full_cam
        from gem.pose.conditioning.camera_depth import depth_parameter_to_legacy_camera

        K, gt = batch["K"], batch["points"][:, :21]
        rgb = batch["rgb"].float() / 255
        # Native 256px pose crop samples output pixel (128,128) at the bbox center.
        box = K.new_tensor([self.size/2, self.size/2, self.size]).expand(len(gt), -1)
        head = self.model({"img": (rgb - self.mean) / self.std, "bbx_xys": box, "K_fullimg": K})
        xyz = self.geometry(head.parameters)["joints"][:, SOMA_TO_MANO21]
        direct_uv = head.auxiliary["kp2d"]["mu"][:, SOMA_TO_MANO21] * self.size + box[:, None, :2]
        legacy = depth_parameter_to_legacy_camera(
            head.pred_cam, box, K, self.model.task.head.camera_depth_mode,
            log_value_bounds=self.release.inference["camera_log_value_bounds"])
        translation = compute_transl_full_cam(legacy, box, K)
        uv_error = direct_uv - project(gt, K)
        loss = 10 * F.smooth_l1_loss(xyz - xyz[:, :1], gt - gt[:, :1], beta=0.01)
        loss += F.smooth_l1_loss(uv_error / self.size, torch.zeros_like(uv_error), beta=0.02)
        loss += F.smooth_l1_loss(xyz[:, 0] + translation, gt[:, 0], beta=0.02)
        return loss, xyz, uv_error

    def export(self, destination):
        state = {"pose_model." + key: value.cpu() for key, value in self.model.state_dict().items()}
        torch.save({"state_dict": state, "gem_pose_training": {"release": asdict(self.release)}}, destination)
