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
from mano_conventions import (
    WILOR_RIGHT_CANONICAL,
    canonical_projection_rotation,
    mirror_left_points,
    physicalize_geometry,
)
from render_mano_overlay_angles import (
    build_kinematic_axes,
    compute_hand_end_effector_pose,
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
        max_orient_step_deg: float = 75.0,
        max_translation_step_m: float = 0.08,
        low_quality_freeze: float = 0.22,
        trajectory_length: int = 120,
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
        self.max_orient_step_deg = max_orient_step_deg
        self.max_translation_step_m = max_translation_step_m
        self.low_quality_freeze = low_quality_freeze
        self.trajectory_length = max(int(trajectory_length), 1)
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
                if (
                    "mano_convention" not in archive.files
                    or str(archive["mano_convention"]) != WILOR_RIGHT_CANONICAL
                ):
                    continue
                handedness = str(archive["handedness"])
                betas = np.asarray(archive["betas"], dtype=np.float32).reshape(10)
                if np.isfinite(betas).all():
                    profiles[handedness] = betas
        return profiles

    def _model(self, handedness: str):
        model = self.models.get(WILOR_RIGHT_CANONICAL)
        if model is None:
            model = self.mano.load(
                model_path=str(self.model_dir),
                is_rhand=True,
                use_pca=True,
                num_pca_comps=15,
                batch_size=1,
                flat_hand_mean=False,
            ).to(self.device)
            model.eval()
            self.models[WILOR_RIGHT_CANONICAL] = model
        return model

    def warmup(self) -> None:
        """Load MANO_RIGHT and initialize CUDA kernels before camera streaming."""
        torch = self.torch
        with torch.no_grad():
            model = self._model("Right")
            betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
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
        target = mirror_left_points(
            np.asarray(match["filtered_points_left"], dtype=np.float64), handedness
        )
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
            {
                "handedness": handedness, "betas": betas_np,
                "mano_convention": WILOR_RIGHT_CANONICAL,
            },
        )
        _, _, physical_faces = physicalize_geometry(
            template, template, np.asarray(model.faces, dtype=np.int32), handedness
        )
        return {
            "track_id": track_id,
            "handedness": handedness,
            "model": model,
            "faces": physical_faces,
            "mano_convention": WILOR_RIGHT_CANONICAL,
            "projection_rotation": self.torch.as_tensor(
                canonical_projection_rotation(
                    self.rotation.detach().cpu().numpy(), handedness
                ),
                dtype=self.torch.float32, device=self.device,
            ),
            "betas": betas,
            "pose": pose,
            "orient": orient,
            "transl": transl,
            "optimizer": optimizer,
            "previous_pose": pose.detach().clone(),
            "previous_orient": orient.detach().clone(),
            "previous_transl": transl.detach().clone(),
            "previous_pose_velocity": torch.zeros_like(pose),
            "previous_orient_velocity": torch.zeros_like(orient),
            "previous_transl_velocity": torch.zeros_like(transl),
            "previous_left_px": None,
            "kinematic_axes": build_kinematic_axes(rest_joints),
            "updates": 0,
            "missed_updates": 0,
            "high_loss_updates": 0,
            "reset_pending": False,
            "angle_history": deque(maxlen=self.angle_window),
            "trajectory_history": deque(maxlen=self.trajectory_length),
            "trajectory_origin_position": None,
            "trajectory_origin_rotation": None,
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
        target_np = mirror_left_points(
            np.asarray(match["filtered_points_left"], dtype=np.float32), handedness
        )
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
        current_left_px_np = left_px[0].detach().cpu().numpy()
        previous_left_px_np = state.get("previous_left_px")
        image_motion_px = 0.0
        if previous_left_px_np is not None:
            finite_motion = (
                np.isfinite(previous_left_px_np).all(axis=-1)
                & np.isfinite(current_left_px_np).all(axis=-1)
            )
            if np.any(finite_motion):
                image_motion_px = float(np.median(np.linalg.norm(
                    current_left_px_np[finite_motion] - previous_left_px_np[finite_motion], axis=1
                )))
        observed_np = valid_np & ~np.asarray(match["predicted_3d"], dtype=bool)
        observed_count = int(np.count_nonzero(observed_np))
        observation_quality = float(
            np.mean(confidence_np[observed_np]) * np.sqrt(observed_count / 21.0)
        ) if observed_count else 0.0
        observation_quality = float(np.clip(observation_quality, 0.0, 1.0))
        temporal_strength = float(np.clip(
            0.45 + 2.4 * (1.0 - observation_quality)
            + 0.45 * np.exp(-image_motion_px / 12.0),
            0.45, 3.0,
        ))
        motion_factor = 0.75 + 0.50 * min(image_motion_px / 35.0, 1.0)
        quality_factor = 0.80 + 0.20 * observation_quality
        limit_factor = float(np.clip(motion_factor * quality_factor, 0.60, 1.25))
        orient_limit_deg = self.max_orient_step_deg * limit_factor
        translation_limit_m = self.max_translation_step_m * limit_factor
        if image_motion_px < 3.0:
            orient_limit_deg = min(orient_limit_deg, 28.0)
            translation_limit_m = min(translation_limit_m, 0.035)

        model = state["model"]
        optimizer = state["optimizer"]
        pose = state["pose"]
        orient = state["orient"]
        transl = state["transl"]
        betas = state["betas"]
        prior_pose = state["previous_pose"].detach().clone()
        prior_orient = state["previous_orient"].detach().clone()
        prior_transl = state["previous_transl"].detach().clone()
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
            selected = palm[observed_np[palm]]
            if len(selected) >= 3:
                current_np = current_joints[0].detach().cpu().numpy()
                delta_rotation = weighted_kabsch(
                    current_np[selected], target_np[selected], confidence_np[selected]
                )
                delta_vector, _ = cv2.Rodrigues(delta_rotation)
                delta_angle = float(np.linalg.norm(delta_vector))
                maximum_delta = np.deg2rad(orient_limit_deg)
                if delta_angle > maximum_delta:
                    delta_vector *= maximum_delta / delta_angle
                rigid_blend = self.rigid_blend * (0.45 + 0.55 * observation_quality)
                blended_rotation, _ = cv2.Rodrigues(delta_vector * rigid_blend)
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
                translation_norm = torch.linalg.vector_norm(translation_delta)
                if bool(translation_norm > translation_limit_m):
                    translation_delta *= translation_limit_m / translation_norm.clamp_min(1e-8)
                transl.add_(translation_delta[None])
        started = time.perf_counter()
        final_loss = float("nan")
        performed_iterations = 0
        for _ in range(maximum_iterations):
            optimizer.zero_grad(set_to_none=True)
            vertices, joints, _ = run_model(model, betas, orient, pose, transl)
            weights3d = confidence * valid
            loss3d = robust_weighted_loss(joints - target, weights3d, 0.005)
            left_prediction = project_rectified(
                joints, state["projection_rotation"], self.p1
            )
            right_prediction = project_rectified(
                joints, state["projection_rotation"], self.p2
            )
            weights2d_left = torch.clamp(left_quality, min=0.03, max=1.0)
            weights2d_right = torch.clamp(right_quality, min=0.03, max=1.0)
            loss2d = robust_weighted_loss(
                (left_prediction - left_px) / 100.0, weights2d_left, 0.015
            )
            loss2d += robust_weighted_loss(
                (right_prediction - right_px) / 100.0, weights2d_right, 0.015
            )
            pose_velocity = pose - prior_pose
            orient_velocity = orient - prior_orient
            transl_velocity = transl - prior_transl
            temporal = pose_velocity.square().mean()
            temporal += 0.25 * orient_velocity.square().mean()
            temporal += 8.0 * transl_velocity.square().mean()
            acceleration = (pose_velocity - state["previous_pose_velocity"]).square().mean()
            acceleration += 0.25 * (
                orient_velocity - state["previous_orient_velocity"]
            ).square().mean()
            acceleration += 8.0 * (
                transl_velocity - state["previous_transl_velocity"]
            ).square().mean()
            pose_prior = pose.square().mean()
            loss = (
                loss3d + 0.12 * loss2d
                + self.pose_prior_weight * pose_prior
                + self.temporal_weight * temporal_strength * (temporal + 0.35 * acceleration)
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
            # Final trust region: reject optimizer solution switches which are
            # inconsistent with the actual 2D hand motion.
            pose_delta = pose - prior_pose
            pose_limit = 6.0 * np.clip(
                (0.60 + 0.40 * observation_quality)
                * (0.80 + 0.40 * min(image_motion_px / 35.0, 1.0)), 0.50, 1.20,
            )
            pose_norm = torch.linalg.vector_norm(pose_delta)
            if bool(pose_norm > pose_limit):
                pose.copy_(prior_pose + pose_delta * (pose_limit / pose_norm.clamp_min(1e-8)))
            previous_rotation, _ = cv2.Rodrigues(
                prior_orient[0].detach().cpu().numpy().astype(np.float64)
            )
            current_rotation, _ = cv2.Rodrigues(
                orient[0].detach().cpu().numpy().astype(np.float64)
            )
            relative_vector, _ = cv2.Rodrigues(current_rotation @ previous_rotation.T)
            relative_angle = float(np.linalg.norm(relative_vector))
            orient_limit = np.deg2rad(orient_limit_deg)
            if relative_angle > orient_limit:
                limited_relative, _ = cv2.Rodrigues(
                    relative_vector * (orient_limit / relative_angle)
                )
                limited_vector, _ = cv2.Rodrigues(limited_relative @ previous_rotation)
                orient.copy_(torch.as_tensor(
                    limited_vector[:, 0][None], dtype=torch.float32, device=self.device
                ))
            translation_delta = transl - prior_transl
            translation_norm = torch.linalg.vector_norm(translation_delta)
            if bool(translation_norm > translation_limit_m):
                transl.copy_(
                    prior_transl + translation_delta
                    * (translation_limit_m / translation_norm.clamp_min(1e-8))
                )
            if observation_quality < self.low_quality_freeze:
                keep = 0.20 + 0.60 * (observation_quality / max(self.low_quality_freeze, 1e-6))
                pose.copy_(prior_pose + keep * (pose - prior_pose))
            vertices, joints, output = run_model(model, betas, orient, pose, transl)
            axis_angle = output.hand_pose.reshape(1, 15, 3)[0].detach().cpu().numpy()
        state["previous_pose_velocity"] = pose.detach() - prior_pose
        state["previous_orient_velocity"] = orient.detach() - prior_orient
        state["previous_transl_velocity"] = transl.detach() - prior_transl
        state["previous_pose"] = pose.detach().clone()
        state["previous_orient"] = orient.detach().clone()
        state["previous_transl"] = transl.detach().clone()
        state["previous_left_px"] = current_left_px_np.copy()
        state["updates"] += 1
        angles = extract_kinematic_sequence(
            axis_angle[None], state["kinematic_axes"], "Right"
        )[0]
        state["angle_history"].append(angles.copy())
        smoothed_angles = np.median(np.stack(state["angle_history"]), axis=0)
        physical_vertices, physical_joints, _ = physicalize_geometry(
            vertices[0].detach().cpu().numpy(), joints[0].detach().cpu().numpy(),
            state["faces"], handedness,
        )
        end_effector_pose = compute_hand_end_effector_pose(
            physical_joints, handedness
        )
        if state["trajectory_origin_position"] is None:
            state["trajectory_origin_position"] = end_effector_pose["position_m"].copy()
            state["trajectory_origin_rotation"] = end_effector_pose["rotation_matrix"].copy()
        delta_camera = (
            end_effector_pose["position_m"] - state["trajectory_origin_position"]
        )
        delta_hand0 = state["trajectory_origin_rotation"].T @ delta_camera
        state["trajectory_history"].append(end_effector_pose["position_m"].copy())
        if final_loss > 0.25:
            state["high_loss_updates"] += 1
        else:
            state["high_loss_updates"] = 0
        state["reset_pending"] = state["high_loss_updates"] >= 5
        result = {
            "track_id": track_id,
            "handedness": handedness,
            "vertices": physical_vertices,
            "joints": physical_joints,
            "faces": state["faces"],
            "mano_convention": WILOR_RIGHT_CANONICAL,
            "hand_pose_axis_angle": axis_angle,
            "kinematic_raw": angles,
            "kinematic": smoothed_angles,
            "end_effector_position_m": end_effector_pose["position_m"],
            "end_effector_rotation_matrix": end_effector_pose["rotation_matrix"],
            "end_effector_rpy_rad": end_effector_pose["rpy_rad"],
            "end_effector_quaternion_xyzw": end_effector_pose["quaternion_xyzw"],
            "end_effector_delta_camera_m": delta_camera,
            "end_effector_delta_hand0_m": delta_hand0,
            "trajectory_positions_m": np.stack(state["trajectory_history"]),
            "loss": final_loss,
            "fit_ms": (time.perf_counter() - started) * 1000.0,
            "iterations": performed_iterations,
            "device": str(self.device),
            "observed": True,
            "observation_quality": observation_quality,
            "image_motion_px": image_motion_px,
        }
        state["last_result"] = result
        return result

    def ordered_results(self, visible: dict[int, dict]) -> list[dict]:
        order = {"Right": 0, "Left": 1}
        return sorted(visible.values(), key=lambda result: order.get(result["handedness"], 2))
