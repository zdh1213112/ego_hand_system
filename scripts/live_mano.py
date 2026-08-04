#!/usr/bin/env python3
"""Incremental GPU MANO fitting for the EGO live stereo tracker."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np

from fit_mano_sequence import (
    import_mano,
    project_rectified,
    robust_weighted_loss,
    run_model,
    weighted_kabsch,
)
from render_mano_overlay_angles import (
    build_kinematic_axes,
    extract_kinematic_sequence,
    load_rest_joints,
)


class LiveManoFitter:
    def __init__(
        self,
        mano_source: Path,
        model_dir: Path,
        rectification: dict,
        device: str = "auto",
        iterations: int = 8,
        initial_iterations: int = 30,
        extra_iterations: int = 2,
        loss_threshold: float = 0.08,
        learning_rate: float = 0.012,
        pose_prior_weight: float = 0.006,
        temporal_weight: float = 0.12,
        rigid_blend: float = 0.65,
        angle_window: int = 5,
        profile_dir: Path | None = None,
    ):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA MANO requested but torch.cuda.is_available() is false")
        self.torch = torch
        self.device = torch.device(device)
        self.iterations = iterations
        self.initial_iterations = initial_iterations
        self.extra_iterations = extra_iterations
        self.loss_threshold = loss_threshold
        self.learning_rate = learning_rate
        self.pose_prior_weight = pose_prior_weight
        self.temporal_weight = temporal_weight
        self.rigid_blend = rigid_blend
        self.angle_window = angle_window
        self.model_dir = model_dir.resolve()
        self.mano = import_mano(mano_source.resolve())
        self.states: dict[int, dict] = {}
        self.models: dict[str, object] = {}
        self.profile_betas = self._load_profiles(profile_dir)
        self.rotation = torch.as_tensor(
            rectification["r1"], dtype=torch.float32, device=self.device
        )
        self.p1 = torch.as_tensor(
            rectification["p1"], dtype=torch.float32, device=self.device
        )
        self.p2 = torch.as_tensor(
            rectification["p2"], dtype=torch.float32, device=self.device
        )

    def _load_profiles(self, directory: Path | None) -> dict[str, np.ndarray]:
        profiles: dict[str, np.ndarray] = {}
        if directory is None or not directory.is_dir():
            return profiles
        for path in sorted(directory.glob("track_*.npz")):
            with np.load(path) as archive:
                if "handedness" not in archive.files or "betas" not in archive.files:
                    continue
                handedness = str(archive["handedness"])
                betas = np.asarray(archive["betas"], dtype=np.float32).reshape(10)
                if np.isfinite(betas).all():
                    profiles[handedness] = betas
        return profiles

    def _model(self, handedness: str):
        model = self.models.get(handedness)
        if model is None:
            model = self.mano.load(
                model_path=str(self.model_dir),
                is_rhand=handedness == "Right",
                use_pca=True,
                num_pca_comps=15,
                batch_size=1,
                flat_hand_mean=False,
            ).to(self.device)
            model.eval()
            self.models[handedness] = model
        return model

    def warmup(self) -> None:
        """Load both licensed models and initialize CUDA kernels before camera streaming."""
        torch = self.torch
        with torch.no_grad():
            for handedness in ("Right", "Left"):
                model = self._model(handedness)
                betas_np = self.profile_betas.get(
                    handedness, np.zeros(10, dtype=np.float32)
                )
                betas = torch.as_tensor(
                    betas_np[None], dtype=torch.float32, device=self.device
                )
                orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
                pose = torch.zeros((1, 15), dtype=torch.float32, device=self.device)
                transl = torch.tensor(
                    ((0.0, 0.0, 0.35),), dtype=torch.float32, device=self.device
                )
                run_model(model, betas, orient, pose, transl)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _create_state(self, track_id: int, handedness: str, match: dict) -> dict:
        torch = self.torch
        model = self._model(handedness)
        betas_np = self.profile_betas.get(handedness, np.zeros(10, dtype=np.float32))
        betas = torch.as_tensor(betas_np[None], dtype=torch.float32, device=self.device)
        pose = torch.zeros((1, 15), dtype=torch.float32, device=self.device, requires_grad=True)
        orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        transl = torch.zeros((1, 3), dtype=torch.float32, device=self.device, requires_grad=True)

        with torch.no_grad():
            _, template_tensor, _ = run_model(model, betas, orient, pose, transl)
        template = template_tensor[0].detach().cpu().numpy()
        target = np.asarray(match["filtered_points_left"], dtype=np.float64)
        valid = np.asarray(match["filtered_valid"], dtype=bool)
        confidence = np.asarray(match["depth_quality"], dtype=np.float64)
        palm = np.asarray((0, 1, 5, 9, 13, 17), dtype=np.int64)
        selected = palm[valid[palm]]
        if len(selected) < 3:
            selected = np.flatnonzero(valid)
        if len(selected) >= 3:
            rotation = weighted_kabsch(
                template[selected], target[selected], np.maximum(confidence[selected], 0.05)
            )
            rotation_vector, _ = cv2.Rodrigues(rotation)
            orient.data.copy_(torch.as_tensor(
                rotation_vector[:, 0][None], dtype=torch.float32, device=self.device
            ))
        with torch.no_grad():
            _, rotated_joints, _ = run_model(model, betas, orient, pose, transl)
            mask = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=self.device)
            weights = torch.as_tensor(
                np.maximum(confidence, 0.05), dtype=torch.float32, device=self.device
            )
            if torch.any(mask):
                translation = torch.sum(
                    (target_tensor[mask] - rotated_joints[0, mask]) * weights[mask, None], dim=0
                ) / weights[mask].sum().clamp_min(1e-6)
                transl.data.copy_(translation[None])

        optimizer = torch.optim.Adam([pose, orient, transl], lr=self.learning_rate)
        rest_joints = load_rest_joints(
            self.mano, self.model_dir,
            {"handedness": handedness, "betas": betas_np},
        )
        return {
            "track_id": track_id,
            "handedness": handedness,
            "model": model,
            "faces": np.asarray(model.faces, dtype=np.int32),
            "betas": betas,
            "pose": pose,
            "orient": orient,
            "transl": transl,
            "optimizer": optimizer,
            "previous_pose": pose.detach().clone(),
            "previous_orient": orient.detach().clone(),
            "previous_transl": transl.detach().clone(),
            "kinematic_axes": build_kinematic_axes(rest_joints),
            "updates": 0,
            "missed_updates": 0,
            "high_loss_updates": 0,
            "reset_pending": False,
            "angle_history": deque(maxlen=self.angle_window),
            "last_result": None,
        }

    def update(self, match: dict) -> dict | None:
        torch = self.torch
        track_id = int(match["track_id"])
        handedness = str(match.get("stable_handedness", match["left"]["label"]))
        if handedness not in ("Left", "Right"):
            return None
        state = self.states.get(track_id)
        if state is None or state.get("reset_pending", False):
            state = self._create_state(track_id, handedness, match)
            self.states[track_id] = state
        elif state["handedness"] != handedness:
            # stable_handedness is cumulative. If it changes, the initial one-frame
            # classification was wrong; keeping the mirrored MANO model permanently is
            # much worse than paying one reinitialization frame.
            state = self._create_state(track_id, handedness, match)
            self.states[track_id] = state

        valid_np = np.asarray(match["filtered_valid"], dtype=bool)
        if np.count_nonzero(valid_np) < 7:
            state["missed_updates"] += 1
            if state["last_result"] is None:
                return None
            predicted = dict(state["last_result"])
            predicted.update({"observed": False, "fit_ms": 0.0, "iterations": 0})
            return predicted
        target_np = np.asarray(match["filtered_points_left"], dtype=np.float32)
        confidence_np = np.asarray(match["depth_quality"], dtype=np.float32)
        confidence_np = np.maximum(confidence_np, 0.04)
        confidence_np[np.asarray(match["predicted_3d"], dtype=bool)] *= 0.18
        reacquired = state["missed_updates"] > 0
        state["missed_updates"] = 0

        target = torch.as_tensor(target_np[None], dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(valid_np[None], dtype=torch.bool, device=self.device)
        confidence = torch.as_tensor(confidence_np[None], dtype=torch.float32, device=self.device)
        left_quality_np = np.asarray(
            match.get("left_2d_quality", np.full(21, match["left"]["score"])),
            dtype=np.float32,
        )
        right_quality_np = np.asarray(
            match.get("right_2d_quality", np.full(21, match["right"]["score"])),
            dtype=np.float32,
        )
        left_quality = torch.as_tensor(
            left_quality_np[None], dtype=torch.float32, device=self.device
        )
        right_quality = torch.as_tensor(
            right_quality_np[None], dtype=torch.float32, device=self.device
        )
        left_px = torch.as_tensor(
            np.asarray(match.get("left_points", match["left"]["pixels"]), dtype=np.float32)[None],
            dtype=torch.float32, device=self.device,
        )
        right_px = torch.as_tensor(
            np.asarray(match.get("right_points", match["right"]["pixels"]), dtype=np.float32)[None],
            dtype=torch.float32, device=self.device,
        )

        model = state["model"]
        optimizer = state["optimizer"]
        pose = state["pose"]
        orient = state["orient"]
        transl = state["transl"]
        betas = state["betas"]
        initializing = state["updates"] == 0
        iterations = self.initial_iterations if initializing else self.iterations
        if reacquired:
            iterations = max(iterations, min(self.initial_iterations, 8))
        maximum_iterations = iterations + (0 if initializing else self.extra_iterations)

        # Correct the rigid palm transform before optimizing articulation. This is the
        # causal equivalent of the rigid initialization used by the good offline fit and
        # prevents wrist rotation/translation lag from being absorbed by finger pose.
        with torch.no_grad():
            _, current_joints, _ = run_model(model, betas, orient, pose, transl)
            palm = np.asarray((0, 1, 5, 9, 13, 17), dtype=np.int64)
            observed_np = valid_np & ~np.asarray(match["predicted_3d"], dtype=bool)
            selected = palm[observed_np[palm]]
            if len(selected) >= 3:
                current_np = current_joints[0].detach().cpu().numpy()
                delta_rotation = weighted_kabsch(
                    current_np[selected], target_np[selected], confidence_np[selected]
                )
                delta_vector, _ = cv2.Rodrigues(delta_rotation)
                blended_rotation, _ = cv2.Rodrigues(delta_vector * self.rigid_blend)
                current_rotation, _ = cv2.Rodrigues(
                    orient[0].detach().cpu().numpy().astype(np.float64)
                )
                updated_vector, _ = cv2.Rodrigues(blended_rotation @ current_rotation)
                orient.copy_(torch.as_tensor(
                    updated_vector[:, 0][None], dtype=torch.float32, device=self.device
                ))
                _, current_joints, _ = run_model(model, betas, orient, pose, transl)
            mask = valid[0]
            weights = confidence[0, mask]
            if torch.any(mask) and weights.sum() > 0:
                translation_delta = torch.sum(
                    (target[0, mask] - current_joints[0, mask]) * weights[:, None], dim=0
                ) / weights.sum().clamp_min(1e-6)
                transl.add_(translation_delta[None])
        started = time.perf_counter()
        final_loss = float("nan")
        performed_iterations = 0
        for _ in range(maximum_iterations):
            optimizer.zero_grad(set_to_none=True)
            vertices, joints, _ = run_model(model, betas, orient, pose, transl)
            weights3d = confidence * valid
            loss3d = robust_weighted_loss(joints - target, weights3d, 0.005)
            left_prediction = project_rectified(joints, self.rotation, self.p1)
            right_prediction = project_rectified(joints, self.rotation, self.p2)
            weights2d_left = torch.clamp(left_quality, min=0.03, max=1.0)
            weights2d_right = torch.clamp(right_quality, min=0.03, max=1.0)
            loss2d = robust_weighted_loss(
                (left_prediction - left_px) / 100.0, weights2d_left, 0.015
            )
            loss2d += robust_weighted_loss(
                (right_prediction - right_px) / 100.0, weights2d_right, 0.015
            )
            temporal = (pose - state["previous_pose"]).square().mean()
            temporal += 0.25 * (orient - state["previous_orient"]).square().mean()
            temporal += 8.0 * (transl - state["previous_transl"]).square().mean()
            pose_prior = pose.square().mean()
            loss = (
                loss3d + 0.12 * loss2d
                + self.pose_prior_weight * pose_prior
                + self.temporal_weight * temporal
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pose, orient, transl], 3.0)
            optimizer.step()
            with torch.no_grad():
                pose.clamp_(-5.0, 5.0)
            final_loss = float(loss.detach().cpu())
            performed_iterations += 1
            if performed_iterations >= iterations and final_loss <= self.loss_threshold:
                break

        with torch.no_grad():
            vertices, joints, output = run_model(model, betas, orient, pose, transl)
            axis_angle = output.hand_pose.reshape(1, 15, 3)[0].detach().cpu().numpy()
        state["previous_pose"] = pose.detach().clone()
        state["previous_orient"] = orient.detach().clone()
        state["previous_transl"] = transl.detach().clone()
        state["updates"] += 1
        angles = extract_kinematic_sequence(
            axis_angle[None], state["kinematic_axes"], handedness
        )[0]
        state["angle_history"].append(angles.copy())
        smoothed_angles = np.median(np.stack(state["angle_history"]), axis=0)
        if final_loss > 0.25:
            state["high_loss_updates"] += 1
        else:
            state["high_loss_updates"] = 0
        state["reset_pending"] = state["high_loss_updates"] >= 5
        result = {
            "track_id": track_id,
            "handedness": handedness,
            "vertices": vertices[0].detach().cpu().numpy(),
            "joints": joints[0].detach().cpu().numpy(),
            "faces": state["faces"],
            "hand_pose_axis_angle": axis_angle,
            "kinematic_raw": angles,
            "kinematic": smoothed_angles,
            "loss": final_loss,
            "fit_ms": (time.perf_counter() - started) * 1000.0,
            "iterations": performed_iterations,
            "device": str(self.device),
            "observed": True,
        }
        state["last_result"] = result
        return result

    def ordered_results(self, visible: dict[int, dict]) -> list[dict]:
        order = {"Right": 0, "Left": 1}
        return sorted(visible.values(), key=lambda result: order.get(result["handedness"], 2))
