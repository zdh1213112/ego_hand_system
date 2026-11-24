"""Released FAH-S student and decoder-free, ray-conditioned point head."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import project


def decode_wrist(uv, xyz, log_depth, K, size):
    """FAH's absolute wrist in the canonical perspective crop, in metres."""
    # The released estimator fixes translation z from the depth head, then fits
    # translation x/y to *all* joint, bone and surface observations at that z.
    tz = log_depth.exp() * K[:, 0, 0] / size
    focal_xy = K.diagonal(dim1=1, dim2=2)[:, :2]
    rays_xy = (uv * size - K[:, None, :2, 2]) / focal_xy[:, None]
    txy = (rays_xy * (xyz[..., 2:] + tz[:, None, None]) - xyz[..., :2]).mean(dim=1)
    return xyz[:, 0] + torch.cat((txy, tz[:, None]), dim=1)


class FAHFineTune(nn.Module):
    size = 320
    canonical_side = 1
    filename = "model.pt"

    def __init__(self, checkpoint, *, wrist_loss_weight=0.0, train_adapter=False):
        super().__init__()
        if not math.isfinite(wrist_loss_weight) or wrist_loss_weight < 0:
            raise ValueError("wrist_loss_weight must be finite and nonnegative")
        from fah.models.heads.keypoint_head import HandKeypointHead, bone_points
        from fah.models.modules.camera_embed import CameraEncoder
        from fah.student import StudentEncoder

        self.bone_points = bone_points
        self.checkpoint = checkpoint
        self.wrist_loss_weight = wrist_loss_weight
        self.train_adapter = train_adapter
        release = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=False)
        weights = release["trainable"]

        def section(prefix):
            return {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}

        self.size = release["args"]["student_size"]
        self.encoder = StudentEncoder(release["student_name"], size=self.size, pretrained=False)
        self.encoder.load_state_dict(section("encoder."))
        state = section("model.keypoint_head.")
        self.camera = CameraEncoder(self.encoder.out_dim, self.encoder.patch)
        self.camera.load_state_dict(section("model.ray_cond_emb_hand."))
        self.register_buffer("no_mask", weights["model.prompt_encoder.no_mask_embed.weight"].reshape(1, -1, 1, 1))
        self.head = HandKeypointHead(
            pool=True, depth=True, vis=True, dim=state["point"].shape[1],
            dense=state["dense"].tolist(), dense_vidx=state["dense_vidx"].tolist(),
            stride=4 if "up.3.weight" in state else 8, parts=state["part_emb"].shape[0],
            part_hr="part_refine.0.weight" in state,
            part_depth=sum(k.startswith("part_conv.") and v.ndim == 4 for k, v in state.items()))
        self.head.load_state_dict(state)
        if state["dense"].tolist() != [40, len(state["dense_vidx"])]:
            raise ValueError("Expected the FAH dense head with two intermediate points per bone")
        self.head.parts_on = False
        self.encoder.requires_grad_(False)
        self.encoder.adapter.requires_grad_(train_adapter)
        self.camera.requires_grad_(False)
        for name, param in self.head.named_parameters():
            param.requires_grad_(not name.startswith(("vis.", "part")))
        u = torch.arange(self.size, dtype=torch.float32)
        self.register_buffer("pixels", torch.stack(torch.meshgrid(u, u, indexing="xy"))[None])

    def forward(self, rgb, K):
        """Return dense normalized UV, relative XYZ (m) and log normalized depth."""
        rgb = rgb.float() / 255
        focal = K[:, 0, 0]
        rays = (self.pixels - K[:, :2, 2, None, None]) / focal[:, None, None, None]
        # The frozen camera must pass the gradient from the head to the adapter.
        features = self.camera(self.encoder(rgb) + self.no_mask, rays)
        tokens = features.flatten(2).transpose(1, 2)
        # Identity crop affine. FAH's bbox center is S/2 in full-image pixels.
        cond = torch.cat((self.size/2 - K[:, :2, 2], torch.full_like(focal[:, None], self.size)), 1)
        pose = self.head.pool(torch.cat((tokens.mean(1), cond / focal[:, None]), 1))
        image = (rgb - self.encoder.mean) / self.encoder.std
        uv, xyz, _ = self.head(tokens, features.shape[-2:], pose, image)
        depth = self.head.depth(pose)[:, 0]
        return uv, xyz, depth

    def objective(self, batch):
        K, points = batch["K"], batch["points"]
        focal = K[:, 0, 0]
        uv, xyz, depth = self(batch["rgb"], K)
        joints = points[:, :21]
        target = torch.cat((joints, self.bone_points(joints), points[:, 21:][:, self.head.dense_vidx]), 1)
        target_uv = project(target, K) / self.size
        target_xyz = target - target[:, :1]
        loss = F.smooth_l1_loss(depth, (joints[:, 0, 2] * self.size / focal).log(), beta=0.05)
        for group in (slice(0, 21), slice(21, 61), slice(61, None)):
            loss = loss + F.smooth_l1_loss(uv[:, group], target_uv[:, group], beta=0.02)
            loss = loss + 10 * F.smooth_l1_loss(xyz[:, group], target_xyz[:, group], beta=0.01)
        if self.wrist_loss_weight:
            wrist = decode_wrist(uv, xyz, depth, K, self.size)
            # Undoing the crop rotation and left-hand reflection preserves the
            # wrist distance. Apply the 1 cm robust threshold to that distance.
            error = torch.linalg.vector_norm(wrist - joints[:, 0], dim=1)
            wrist_loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.01)
            loss = loss + self.wrist_loss_weight * wrist_loss
        return loss, xyz[:, :21], (uv[:, :21] - target_uv[:, :21]) * self.size

    def optimizer_groups(self, head_lr, adapter_lr):
        head = [p for p in self.head.parameters() if p.requires_grad]
        groups = [dict(name="head", params=head, lr=head_lr)]
        if self.train_adapter:
            groups.append(dict(name="adapter", params=list(self.encoder.adapter.parameters()), lr=adapter_lr))
        return groups

    def export(self, destination):
        release = torch.load(self.checkpoint, mmap=True, map_location="cpu", weights_only=False)
        weights = release["trainable"]
        weights.update({"model.keypoint_head." + k: v.cpu() for k, v in self.head.state_dict().items()})
        if self.train_adapter:
            weights.update({"encoder.adapter." + k: v.cpu() for k, v in self.encoder.adapter.state_dict().items()})
        torch.save(release, destination)
