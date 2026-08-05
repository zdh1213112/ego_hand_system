#!/usr/bin/env python3
"""Real-time EGO stereo hand tracking with depth-aware identity and filtering."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import itertools
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ego-hand-matplotlib")

import cv2
import mediapipe as mp
import numpy as np
import yaml

from mediapipe_left_baseline import HAND_CONNECTIONS, camera_matrices, create_stereo_rectification
from mediapipe_stereo_triangulate import (
    FINGERTIP_INDICES,
    associate_hands,
    detect,
    draw_observation,
    make_options,
    prepare_stereo_matching_images,
    triangulate,
)
from render_mano_overlay_angles import (
    KINEMATIC_KEYS,
    TRACK_COLORS as MANO_TRACK_COLORS,
    compose_canvas as compose_mano_canvas,
    draw_end_effector_trajectory as draw_mano_trajectory,
    draw_mesh as draw_mano_mesh,
)


PACKET_HEADER = struct.Struct("<4sHHIIIIQQQQQ")
PACKET_MAGIC = b"EGO1"
TRACK_COLORS = ((40, 220, 40), (30, 170, 255), (255, 90, 190), (255, 220, 40))
PALM_INDICES = np.asarray((0, 5, 9, 13, 17), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track two hands from the Orbbec EGO live stereo stream."
    )
    parser.add_argument("--bridge", type=Path, default=Path("build/ego_live_bridge"))
    parser.add_argument(
        "--sdk-config", type=Path,
        default=Path("third_party/orbbec_sdk/OrbbecSDKConfig.xml"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--output", type=Path, default=Path("output/ego_live"))
    parser.add_argument("--calibration", type=Path,
                        help="use this calibration YAML instead of reading it from EGO flash")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-detection", type=float, default=0.35)
    parser.add_argument("--min-presence", type=float, default=0.35)
    parser.add_argument("--min-tracking", type=float, default=0.35)
    parser.add_argument("--max-sync-delta-us", type=int, default=2000)
    parser.add_argument("--max-epipolar-px", type=float, default=12.0)
    parser.add_argument("--max-reprojection-px", type=float, default=8.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--balance", type=float, default=0.0)
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument("--stereo-refine", action="store_true", dest="stereo_refine")
    refinement.add_argument("--no-stereo-refine", action="store_false", dest="stereo_refine")
    parser.set_defaults(stereo_refine=True)
    parser.add_argument("--refine-window", type=int, default=17)
    parser.add_argument("--refine-tip-window", type=int, default=25)
    parser.add_argument("--refine-max-x-shift-px", type=float, default=18.0)
    parser.add_argument("--refine-max-y-residual-px", type=float, default=3.0)
    parser.add_argument("--refine-max-fb-error-px", type=float, default=2.0)
    parser.add_argument("--refine-max-lk-error", type=float, default=32.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--process-every", type=int, default=1,
                        help="run stereo inference every N captured frame pairs")
    parser.add_argument("--capture-queue", type=int, default=1,
                        help="bounded stereo queue; 1 minimizes latency, 2 favors continuity")
    parser.add_argument("--prediction-frames", type=int, default=5)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--record", action="store_true", help="save the live annotated preview")
    mano_group = parser.add_mutually_exclusive_group()
    mano_group.add_argument("--mano", action="store_true",
                            help="enable live GPU MANO mesh and 21-DOF UI")
    mano_group.add_argument("--no-mano", action="store_false", dest="mano",
                            help="show the stereo 21-point diagnostic view instead")
    parser.set_defaults(mano=False)
    parser.add_argument("--mano-source", type=Path, default=Path("third_party/MANO"))
    parser.add_argument("--mano-model-dir", type=Path, default=Path("models/mano"))
    parser.add_argument("--mano-profile", type=Path, default=Path("output/mano_fit_refined"),
                        help="optional fitted betas profile directory")
    parser.add_argument("--mano-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mano-iterations", type=int, default=3)
    parser.add_argument("--mano-initial-iterations", type=int, default=20)
    parser.add_argument("--mano-extra-iterations", type=int, default=2,
                        help="extra refinement steps used only while live fit loss is high")
    parser.add_argument("--mano-loss-threshold", type=float, default=0.08)
    parser.add_argument("--mano-learning-rate", type=float, default=0.012)
    parser.add_argument("--mano-pose-prior", type=float, default=0.006)
    parser.add_argument("--mano-temporal-weight", type=float, default=0.12)
    parser.add_argument("--mano-rigid-blend", type=float, default=0.65)
    parser.add_argument("--mano-max-orient-step-deg", type=float, default=75.0)
    parser.add_argument("--mano-max-translation-step-m", type=float, default=0.08)
    parser.add_argument("--mano-low-quality-freeze", type=float, default=0.22)
    parser.add_argument("--mano-trajectory-length", type=int, default=120)
    parser.add_argument("--mano-trajectory-max-jump-m", type=float, default=0.12)
    parser.add_argument("--mano-angle-window", type=int, default=5)
    parser.add_argument("--mano-width", type=int, default=1920)
    parser.add_argument("--mano-height", type=int, default=1080)
    parser.add_argument("--mano-panel-width", type=int, default=650)
    parser.add_argument("--mesh-alpha", type=float, default=0.38)
    return parser.parse_args()


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError("EGO live bridge closed its output stream")
        chunks.extend(block)
    return bytes(chunks)


class EgoBridge:
    def __init__(self, executable: Path, sdk_config: Path, max_frames: int):
        command = [
            str(executable), "--sdk-config", str(sdk_config),
        ]
        if max_frames > 0:
            command.extend(("--max-frames", str(max_frames)))
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0)
        if self.process.stdout is None:
            raise RuntimeError("failed to open EGO bridge output")

    def read(self, decode: bool = True) -> dict:
        raw_header = read_exact(self.process.stdout, PACKET_HEADER.size)
        fields = PACKET_HEADER.unpack(raw_header)
        (
            magic, version, header_size, left_size, right_size, width, height,
            left_timestamp_us, right_timestamp_us, left_index, right_index, host_ns,
        ) = fields
        if magic != PACKET_MAGIC or version != 1 or header_size != PACKET_HEADER.size:
            raise RuntimeError("invalid EGO live bridge packet header")
        left_jpeg = read_exact(self.process.stdout, left_size)
        right_jpeg = read_exact(self.process.stdout, right_size)
        packet = {
            "left_jpeg": left_jpeg, "right_jpeg": right_jpeg,
            "width": width, "height": height,
            "left_timestamp_us": left_timestamp_us,
            "right_timestamp_us": right_timestamp_us,
            "left_index": left_index, "right_index": right_index,
            "host_ns": host_ns,
        }
        return self.decode(packet) if decode else packet

    @staticmethod
    def decode(packet: dict, executor: ThreadPoolExecutor | None = None) -> dict:
        def decode_one(data: bytes) -> np.ndarray | None:
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if executor is None:
            left = decode_one(packet["left_jpeg"])
            right = decode_one(packet["right_jpeg"])
        else:
            left_future = executor.submit(decode_one, packet["left_jpeg"])
            right_future = executor.submit(decode_one, packet["right_jpeg"])
            left = left_future.result()
            right = right_future.result()
        if left is None or right is None:
            raise RuntimeError("failed to decode one or both EGO MJPG frames")
        expected = (packet["width"], packet["height"])
        if left.shape[1::-1] != expected or right.shape[1::-1] != expected:
            raise RuntimeError("decoded EGO dimensions disagree with packet header")
        decoded = dict(packet)
        decoded.pop("left_jpeg")
        decoded.pop("right_jpeg")
        decoded.update({"left": left, "right": right})
        return decoded

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)


class LatestPacketReader:
    """Continuously drain the bridge into a strictly bounded low-latency queue."""

    def __init__(self, bridge: EgoBridge, capacity: int = 1):
        if capacity < 1:
            raise ValueError("capture queue capacity must be positive")
        self.bridge = bridge
        self.condition = threading.Condition()
        self.queue: deque[tuple[int, dict]] = deque()
        self.capacity = capacity
        self.sequence = 0
        self.dropped = 0
        self.stopped = False
        self.closing = False
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="ego-latest-frame", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.closing:
                packet = self.bridge.read(decode=False)
                with self.condition:
                    self.sequence += 1
                    if len(self.queue) >= self.capacity:
                        self.queue.popleft()
                        self.dropped += 1
                    self.queue.append((self.sequence, packet))
                    self.condition.notify_all()
        except Exception as error:
            if not self.closing:
                self.error = error
        finally:
            with self.condition:
                self.stopped = True
                self.condition.notify_all()

    def get(self, timeout: float = 2.0) -> tuple[int, dict] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.queue and not self.stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if not self.queue:
                if self.error is not None:
                    raise self.error
                return None
            sequence, packet = self.queue.popleft()
        return sequence, packet

    def close(self) -> None:
        self.closing = True
        self.bridge.close()
        self.thread.join(timeout=2.0)


def extract_calibration(bridge: Path, sdk_config: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(bridge), "--sdk-config", str(sdk_config),
            "--calibration-out", str(output), "--calibration-only",
        ],
        check=True,
    )


class DepthAwareTrackManager:
    """Keep hand identity using 2D motion and metric 3D/depth together."""

    def __init__(self, max_missed: int = 90, max_tracks: int = 2):
        self.next_id = 0
        self.max_missed = max_missed
        self.max_tracks = max_tracks
        self.tracks: dict[int, dict] = {}

    @staticmethod
    def _finite3(point: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(point)))

    def assign(self, matches: list[dict], timestamp_s: float) -> None:
        for track in self.tracks.values():
            track["missed"] += 1

        usable = [
            index for index, match in enumerate(matches)
            if np.all(np.isfinite(match["center_left_px"]))
        ]
        track_ids = list(self.tracks)
        assignment: dict[int, int] = {}
        used_track_ids: set[int] = set()
        if usable and track_ids:
            maximum = min(len(usable), len(track_ids))
            best_count = 0
            best_cost = float("inf")
            best_pairs: list[tuple[int, int]] = []
            for count in range(1, maximum + 1):
                for match_subset in itertools.combinations(usable, count):
                    for track_subset in itertools.combinations(track_ids, count):
                        for track_order in itertools.permutations(track_subset):
                            pairs = list(zip(match_subset, track_order))
                            cost = 0.0
                            accepted = True
                            for match_index, track_id in pairs:
                                match = matches[match_index]
                                track = self.tracks[track_id]
                                elapsed = max(timestamp_s - track["timestamp_s"], 0.0)
                                dt = float(np.clip(elapsed, 1 / 120, 0.25))
                                predicted_px = track["position_px"] + track["velocity_px"] * dt
                                pixel_distance = float(np.linalg.norm(match["center_left_px"] - predicted_px))
                                relaxed = track["missed"] > self.max_missed // 2 or elapsed > 1.0
                                if pixel_distance > (520.0 if relaxed else 320.0):
                                    accepted = False
                                    break
                                cost += pixel_distance * 0.45

                                if self._finite3(match["center"]) and self._finite3(track["position_3d"]):
                                    predicted_3d = track["position_3d"] + track["velocity_3d"] * dt
                                    distance_3d = float(np.linalg.norm(match["center"] - predicted_3d))
                                    depth_distance = abs(float(match["center"][2] - predicted_3d[2]))
                                    if not relaxed and (distance_3d > 0.35 or depth_distance > 0.25):
                                        accepted = False
                                        break
                                    cost += min(distance_3d, 0.5) * 700.0 + min(depth_distance, 0.35) * 450.0

                                label = match["left"]["label"]
                                if label != track["label"] and match["left"]["score"] > 0.80:
                                    cost += 65.0
                                cost += min(track["missed"], self.max_missed) * 0.5
                            if accepted and (count > best_count or (count == best_count and cost < best_cost)):
                                best_count = count
                                best_cost = cost
                                best_pairs = pairs
            assignment.update(best_pairs)
            used_track_ids.update(track_id for _, track_id in best_pairs)

        for match_index, match in enumerate(matches):
            if match_index not in assignment:
                available = [track_id for track_id in self.tracks if track_id not in used_track_ids]
                if len(self.tracks) < self.max_tracks:
                    track_id = self.next_id
                    self.next_id += 1
                elif available:
                    label = match["left"]["label"]
                    track_id = min(
                        available,
                        key=lambda candidate: (
                            self.tracks[candidate]["label"] != label,
                            -self.tracks[candidate]["missed"],
                        ),
                    )
                else:
                    continue
                assignment[match_index] = track_id
                used_track_ids.add(track_id)
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        "position_px": match["center_left_px"].copy(),
                        "velocity_px": np.zeros(2, dtype=np.float64),
                        "position_3d": match["center"].copy(),
                        "velocity_3d": np.zeros(3, dtype=np.float64),
                        "timestamp_s": timestamp_s,
                        "missed": 0,
                        "label": match["left"]["label"],
                        "label_votes": {"Left": 0.0, "Right": 0.0},
                    }
            track_id = assignment[match_index]
            match["track_id"] = track_id
            track = self.tracks[track_id]
            elapsed = max(timestamp_s - track["timestamp_s"], 0.0)
            dt = float(np.clip(elapsed, 1 / 120, 0.25))
            reacquired = elapsed > 0.75
            if reacquired:
                track["velocity_px"][:] = 0.0
                track["velocity_3d"][:] = 0.0
            else:
                observed_px_velocity = (match["center_left_px"] - track["position_px"]) / dt
                track["velocity_px"] = 0.65 * track["velocity_px"] + 0.35 * observed_px_velocity
            track["position_px"] = match["center_left_px"].copy()
            if self._finite3(match["center"]):
                if self._finite3(track["position_3d"]) and not reacquired:
                    observed_3d_velocity = (match["center"] - track["position_3d"]) / dt
                    track["velocity_3d"] = 0.75 * track["velocity_3d"] + 0.25 * observed_3d_velocity
                else:
                    track["velocity_3d"] = np.zeros(3, dtype=np.float64)
                track["position_3d"] = match["center"].copy()
            track["timestamp_s"] = timestamp_s
            track["missed"] = 0
            votes = track.setdefault("label_votes", {"Left": 0.0, "Right": 0.0})
            votes[match["left"]["label"]] += float(match["left"]["score"])
            track["label"] = max(votes, key=votes.get)
            match["stable_handedness"] = track["label"]


class OnlineStereoFilter:
    """Quality-weighted alpha-beta filter with short prediction through missing depth."""

    def __init__(self, prediction_frames: int):
        self.prediction_frames = prediction_frames
        self.states: dict[int, list[dict | None]] = {}
        self.bone_samples: dict[int, list[deque[float]]] = {}
        self.bone_targets: dict[int, np.ndarray] = {}

    @staticmethod
    def quality(match: dict) -> np.ndarray:
        epipolar = np.asarray(match["epipolar"], dtype=np.float64)
        reprojection = np.asarray(match["reprojection"], dtype=np.float64)
        disparity = np.asarray(match["disparity"], dtype=np.float64)
        handedness = min(float(match["left"]["score"]), float(match["right"]["score"]))
        quality = (
            np.exp(-np.square(epipolar / 6.0))
            * np.exp(-np.square(reprojection / 4.0))
            * np.clip((disparity - 2.0) / 24.0, 0.10, 1.0)
            * handedness
        )
        attempted = np.asarray(
            match.get("refinement_attempted", np.zeros(21, dtype=bool)), dtype=bool
        )
        used = np.asarray(
            match.get("refinement_used", np.zeros(21, dtype=bool)), dtype=bool
        )
        refinement_quality = np.asarray(
            match.get("refinement_quality", np.zeros(21)), dtype=np.float64
        )
        refinement_factor = np.ones(21, dtype=np.float64)
        refinement_factor[attempted & used] = (
            0.55 + 0.45 * refinement_quality[attempted & used]
        )
        refinement_factor[attempted & ~used] = 0.50
        failed_tips = FINGERTIP_INDICES[
            attempted[FINGERTIP_INDICES] & ~used[FINGERTIP_INDICES]
        ]
        refinement_factor[failed_tips] = 0.38
        quality *= refinement_factor
        quality[~match["valid"]] = 0.0
        return np.clip(quality, 0.0, 1.0)

    def update(self, match: dict, timestamp_s: float) -> None:
        track_id = int(match["track_id"])
        states = self.states.setdefault(track_id, [None] * 21)
        quality = self.quality(match)
        filtered = np.full((21, 3), np.nan, dtype=np.float64)
        valid = np.zeros(21, dtype=bool)
        predicted = np.zeros(21, dtype=bool)

        for joint in range(21):
            measurement_valid = bool(match["valid"][joint])
            measurement = match["points_left"][joint]
            state = states[joint]
            if state is None:
                if measurement_valid:
                    states[joint] = {
                        "position": measurement.copy(),
                        "velocity": np.zeros(3, dtype=np.float64),
                        "timestamp_s": timestamp_s,
                        "missed": 0,
                    }
                    filtered[joint] = measurement
                    valid[joint] = True
                continue

            elapsed = max(timestamp_s - state["timestamp_s"], 0.0)
            if elapsed > 0.5 and measurement_valid:
                state["position"] = measurement.copy()
                state["velocity"][:] = 0.0
                state["timestamp_s"] = timestamp_s
                state["missed"] = 0
                filtered[joint] = measurement
                valid[joint] = True
                continue
            dt = float(np.clip(elapsed, 1 / 120, 0.25))
            forecast = state["position"] + state["velocity"] * dt
            if measurement_valid:
                residual = measurement - forecast
                allowed_jump = 0.035 + 2.5 * dt
                if np.linalg.norm(residual) <= allowed_jump or state["missed"] >= self.prediction_frames:
                    # A causal live filter must favor fresh high-quality depth; the
                    # previous conservative gain looked smooth offline but visibly lagged.
                    alpha = 0.42 + 0.53 * float(quality[joint])
                    updated = forecast + alpha * residual
                    observed_velocity = (updated - state["position"]) / dt
                    state["velocity"] = 0.65 * state["velocity"] + 0.35 * observed_velocity
                    state["position"] = updated
                    state["missed"] = 0
                    filtered[joint] = updated
                    valid[joint] = True
                else:
                    state["position"] = forecast
                    state["missed"] += 1
                    if state["missed"] <= self.prediction_frames:
                        filtered[joint] = forecast
                        valid[joint] = True
                        predicted[joint] = True
            else:
                state["position"] = forecast
                state["velocity"] *= 0.92
                state["missed"] += 1
                if state["missed"] <= self.prediction_frames:
                    filtered[joint] = forecast
                    valid[joint] = True
                    predicted[joint] = True
            state["timestamp_s"] = timestamp_s

        samples = self.bone_samples.setdefault(
            track_id, [deque(maxlen=90) for _ in HAND_CONNECTIONS]
        )
        targets = self.bone_targets.setdefault(
            track_id, np.full(len(HAND_CONNECTIONS), np.nan, dtype=np.float64)
        )
        for edge_index, (parent, child) in enumerate(HAND_CONNECTIONS):
            if not (valid[parent] and valid[child]):
                continue
            if predicted[parent] or predicted[child]:
                continue
            if min(quality[parent], quality[child]) < 0.20:
                continue
            length = float(np.linalg.norm(filtered[child] - filtered[parent]))
            if 0.008 <= length <= 0.12:
                samples[edge_index].append(length)
                if len(samples[edge_index]) >= 5:
                    values = np.asarray(samples[edge_index], dtype=np.float64)
                    median = float(np.median(values))
                    deviation = np.abs(values - median)
                    inliers = values[deviation <= max(0.004, 0.25 * median)]
                    if len(inliers) >= 3:
                        targets[edge_index] = float(np.median(inliers))

        for _ in range(3):
            for edge_index, (parent, child) in enumerate(HAND_CONNECTIONS):
                target = targets[edge_index]
                if not np.isfinite(target) or not (valid[parent] and valid[child]):
                    continue
                vector = filtered[child] - filtered[parent]
                length = float(np.linalg.norm(vector))
                if length < 1e-8:
                    continue
                relative_error = abs(length - target) / target
                strength = 0.60 if relative_error > 0.20 else 0.35
                correction = vector / length * ((length - target) * strength)
                parent_share = 0.15 if parent == 0 else 0.40
                filtered[parent] += correction * parent_share
                filtered[child] -= correction * (1.0 - parent_share)

        for joint, state in enumerate(states):
            if state is not None and valid[joint]:
                state["position"] = filtered[joint].copy()

        match["depth_quality"] = quality
        match["filtered_points_left"] = filtered
        match["filtered_valid"] = valid
        match["predicted_3d"] = predicted
        match["bone_targets_m"] = targets.copy()


def project_left_rectified(points_left: np.ndarray, rectification: dict) -> np.ndarray:
    rectified = (rectification["r1"] @ points_left.T).T
    homogeneous = np.column_stack((rectified, np.ones(len(rectified))))
    projected = (rectification["p1"] @ homogeneous.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    pixels[~np.all(np.isfinite(points_left), axis=1)] = np.nan
    return pixels


def draw_filtered(image: np.ndarray, match: dict, rectification: dict,
                  color: tuple[int, int, int]) -> None:
    pixels = project_left_rectified(match["filtered_points_left"], rectification)
    valid = match["filtered_valid"] & np.all(np.isfinite(pixels), axis=1)
    points = np.rint(np.nan_to_num(pixels)).astype(int)
    for parent, child in HAND_CONNECTIONS:
        if valid[parent] and valid[child]:
            line_color = (150, 150, 150) if (
                match["predicted_3d"][parent] or match["predicted_3d"][child]
            ) else color
            cv2.line(image, tuple(points[parent]), tuple(points[child]), line_color, 3, cv2.LINE_AA)
    for joint in range(21):
        if valid[joint]:
            point_color = (150, 150, 150) if match["predicted_3d"][joint] else color
            cv2.circle(image, tuple(points[joint]), 4, point_color, -1, cv2.LINE_AA)


def compose_preview(left: np.ndarray, right: np.ndarray, matches: list[dict], rectification: dict,
                    stats: dict, display_width: int) -> np.ndarray:
    annotated_left = left.copy()
    annotated_right = right.copy()
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    for match in matches:
        color = TRACK_COLORS[int(match["track_id"]) % len(TRACK_COLORS)]
        valid_count = int(np.count_nonzero(match["filtered_valid"]))
        raw_count = int(np.count_nonzero(match["valid"]))
        depth = float(match["center"][2]) if np.all(np.isfinite(match["center"])) else float("nan")
        draw_filtered(annotated_left, match, rectification, color)
        draw_observation(
            annotated_right, match["right"], color,
            f"T{match['track_id']} raw={raw_count}/21",
        )
        anchor = np.rint(match["left"]["pixels"][0]).astype(int)
        label = (
            f"T{match['track_id']} {match.get('stable_handedness', match['left']['label'])} "
            f"z={depth:.3f}m {valid_count}/21"
        )
        cv2.putText(annotated_left, label, (max(5, anchor[0] - 40), max(30, anchor[1] - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        matched_left.add(int(match["left"]["index"]))
        matched_right.add(int(match["right"]["index"]))

    stereo = cv2.hconcat((annotated_left, annotated_right))
    target_height = max(1, int(stereo.shape[0] * display_width / stereo.shape[1]))
    preview = cv2.resize(stereo, (display_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(preview, (0, 0), (display_width, 68), (15, 15, 15), -1)
    status = (
        f"EGO LIVE | capture {stats['capture_fps']:.1f} FPS | inference {stats['inference_fps']:.1f} FPS | "
        f"latency {stats['latency_ms']:.1f} ms | drop {stats.get('dropped_pairs', 0)} | "
        f"sync {stats['sync_delta_us']} us | "
        f"matches {len(matches)}"
    )
    cv2.putText(preview, status, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(preview, "Color: observed stereo 3D   Gray: short depth prediction   Esc/Q: exit",
                (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 210, 255), 1, cv2.LINE_AA)
    return preview


def left_fisheye_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    cameras = calibration.get("cameras", [])
    if len(cameras) != 2:
        raise RuntimeError("expected two cameras in EGO calibration YAML")
    return camera_matrices(cameras[0])


def compose_live_mano(
    original_left: np.ndarray,
    current_results: dict[int, dict],
    all_results: dict[int, dict],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    args: argparse.Namespace,
    capture_index: int,
    left_index: int,
    stats: dict,
) -> np.ndarray:
    frame = original_left.copy()
    visible = sorted(
        current_results.values(),
        key=lambda result: float(np.nanmedian(result["vertices"][:, 2])),
        reverse=True,
    )
    for result in visible:
        color = (
            MANO_TRACK_COLORS.get(result["handedness"], (120, 220, 120))
            if result.get("observed", True) else (145, 145, 145)
        )
        draw_mano_trajectory(
            frame, result.get("trajectory_positions_m", result["end_effector_position_m"][None]),
            camera_matrix, distortion, color,
            args.mano_trajectory_length, args.mano_trajectory_max_jump_m,
        )
        draw_mano_mesh(
            frame, result["vertices"], result["joints"], result["faces"],
            camera_matrix, distortion, color, args.mesh_alpha,
            f"{result['handedness']} T{result['track_id']}",
            result["handedness"],
        )
    tracks = []
    frame_indices: dict[int, int] = {}
    order = {"Right": 0, "Left": 1}
    for track_id, result in sorted(
        all_results.items(), key=lambda item: order.get(item[1]["handedness"], 2)
    ):
        tracks.append({
            "track_id": track_id,
            "handedness": result["handedness"],
            "vertices": result["vertices"][None],
            "joints": result["joints"][None],
            "faces": result["faces"],
            "kinematic": result["kinematic"][None],
            "end_effector_position_m": result["end_effector_position_m"][None],
            "end_effector_rotation_matrix": result["end_effector_rotation_matrix"][None],
            "end_effector_rpy_rad": result["end_effector_rpy_rad"][None],
            "end_effector_quaternion_xyzw": result["end_effector_quaternion_xyzw"][None],
            "end_effector_delta_camera_m": result["end_effector_delta_camera_m"][None],
            "end_effector_delta_hand0_m": result["end_effector_delta_hand0_m"][None],
        })
        if track_id in current_results:
            frame_indices[track_id] = 0
    canvas = compose_mano_canvas(
        frame, tracks, frame_indices,
        (args.mano_width, args.mano_height), args.mano_panel_width,
        capture_index, left_index,
    )
    view_width = args.mano_width - args.mano_panel_width
    cv2.rectangle(canvas, (16, 76), (view_width - 16, 108), (10, 13, 18), -1)
    cv2.putText(
        canvas,
        f"LIVE | cap {stats.get('capture_fps', 0.0):.1f} | infer {stats['inference_fps']:.1f} FPS | "
        f"latency {stats['latency_ms']:.0f} ms | drop {stats.get('dropped_pairs', 0)} | "
        f"MANO {sum(result['fit_ms'] for result in current_results.values()):.0f} ms | "
        f"sync {stats['sync_delta_us']} us",
        (26, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 238, 242), 1, cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    args = parse_args()
    if (
        args.process_every < 1 or args.prediction_frames < 0 or args.num_hands < 1
        or args.capture_queue < 1
    ):
        raise ValueError("process-every/num-hands must be positive and prediction-frames non-negative")
    if args.mano and (
        args.mano_iterations < 1 or args.mano_initial_iterations < 1
        or args.mano_extra_iterations < 0 or args.mano_loss_threshold <= 0
        or args.mano_pose_prior < 0 or args.mano_temporal_weight < 0
        or not 0.0 <= args.mano_rigid_blend <= 1.0 or args.mano_angle_window < 1
        or args.mano_trajectory_length < 1
        or args.mano_trajectory_max_jump_m <= 0.0
    ):
        raise ValueError("invalid MANO fitting or trajectory parameters")
    bridge_path = args.bridge.resolve()
    sdk_config = args.sdk_config.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    for path in (bridge_path, sdk_config, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    if args.calibration is None:
        calibration_path = output / "ego_camera_calibration.yaml"
        extract_calibration(bridge_path, sdk_config, calibration_path)
    else:
        calibration_path = args.calibration.resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(calibration_path)
    rectification = create_stereo_rectification(calibration_path, args.balance)
    image_size = rectification["image_size"]

    mano_fitter = None
    fisheye_camera = None
    fisheye_distortion = None
    all_mano_results: dict[int, dict] = {}
    if args.mano:
        from live_mano import LiveManoFitter

        mano_source = args.mano_source.resolve()
        mano_model_dir = args.mano_model_dir.resolve()
        for path in (mano_source, mano_model_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)
        profile_dir = args.mano_profile.resolve()
        mano_fitter = LiveManoFitter(
            mano_source=mano_source,
            model_dir=mano_model_dir,
            rectification=rectification,
            device=args.mano_device,
            iterations=args.mano_iterations,
            initial_iterations=args.mano_initial_iterations,
            extra_iterations=args.mano_extra_iterations,
            loss_threshold=args.mano_loss_threshold,
            learning_rate=args.mano_learning_rate,
            pose_prior_weight=args.mano_pose_prior,
            temporal_weight=args.mano_temporal_weight,
            rigid_blend=args.mano_rigid_blend,
            max_orient_step_deg=args.mano_max_orient_step_deg,
            max_translation_step_m=args.mano_max_translation_step_m,
            low_quality_freeze=args.mano_low_quality_freeze,
            trajectory_length=args.mano_trajectory_length,
            angle_window=args.mano_angle_window,
            profile_dir=profile_dir if profile_dir.is_dir() else None,
        )
        fisheye_camera, fisheye_distortion = left_fisheye_calibration(calibration_path)

    bridge = None
    latest_reader = None
    tracker = DepthAwareTrackManager()
    online_filter = OnlineStereoFilter(args.prediction_frames)
    csv_path = output / "live_landmarks_3d.csv"
    angle_csv_path = output / "live_mano_21dof.csv"
    video_path = output / (
        "live_mano_21dof.mp4" if args.mano else "live_stereo_annotated.mp4"
    )
    fields = (
        "capture_index", "left_index", "right_index", "left_timestamp_us", "right_timestamp_us",
        "track_id", "handedness", "landmark_index", "raw_valid", "filtered_valid", "predicted",
        "depth_quality", "refinement_used", "refinement_quality",
        "x_left_camera_m", "y_left_camera_m", "z_left_camera_m",
    )
    angle_fields = [
        "capture_index", "left_index", "right_index", "left_timestamp_us",
        "right_timestamp_us", "track_id", "handedness", "observed",
        "fit_loss", "fit_ms", "iterations", "device",
    ]
    for key in KINEMATIC_KEYS:
        angle_fields.extend((f"{key}_raw", key, key.replace("_rad", "_deg")))
    angle_fields.extend([
        "hand_x_m", "hand_y_m", "hand_z_m",
        "hand_dx_camera_m", "hand_dy_camera_m", "hand_dz_camera_m",
        "hand_dx_hand0_m", "hand_dy_hand0_m", "hand_dz_hand0_m",
        "hand_roll_rad", "hand_pitch_rad", "hand_yaw_rad",
        "hand_roll_deg", "hand_pitch_deg", "hand_yaw_deg",
        "hand_qx", "hand_qy", "hand_qz", "hand_qw",
    ])
    angle_fields.extend(f"hand_r{row}{column}" for row in range(3) for column in range(3))
    writer = None
    frame_count = 0
    inference_count = 0
    dropped_count = 0
    started = time.perf_counter()
    inference_time = 0.0
    latency_samples: list[float] = []
    timing_totals = {
        "wait": 0.0, "decode": 0.0, "rectify": 0.0, "detect": 0.0,
        "refine_triangulate": 0.0, "stereo_csv": 0.0, "mano": 0.0,
        "render": 0.0, "display_record": 0.0,
    }
    base_timestamp_us = None
    last_left_ms = -1
    last_right_ms = -1

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as csv_stream, \
             angle_csv_path.open("w", encoding="utf-8", newline="") as angle_stream, \
             mp.tasks.vision.HandLandmarker.create_from_options(make_options(args, model_path)) as left_landmarker, \
             mp.tasks.vision.HandLandmarker.create_from_options(make_options(args, model_path)) as right_landmarker, \
             ThreadPoolExecutor(max_workers=2, thread_name_prefix="ego-mediapipe") as detector_pool:
            csv_writer = csv.DictWriter(csv_stream, fieldnames=fields)
            csv_writer.writeheader()
            angle_writer = csv.DictWriter(angle_stream, fieldnames=angle_fields)
            angle_writer.writeheader()
            warmup_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            warmup_left = detector_pool.submit(
                detect, left_landmarker, warmup_image, 0, image_size
            )
            warmup_right = detector_pool.submit(
                detect, right_landmarker, warmup_image, 0, image_size
            )
            warmup_left.result()
            warmup_right.result()
            last_left_ms = 0
            last_right_ms = 0
            if mano_fitter is not None:
                mano_fitter.warmup()
            bridge = EgoBridge(bridge_path, sdk_config, 0)
            latest_reader = LatestPacketReader(bridge, args.capture_queue)
            started = time.perf_counter()
            while args.max_frames <= 0 or inference_count < args.max_frames:
                stage_started = time.perf_counter()
                latest = latest_reader.get()
                timing_totals["wait"] += time.perf_counter() - stage_started
                if latest is None:
                    if latest_reader.stopped:
                        break
                    continue
                sequence, packet = latest
                dropped_count = latest_reader.dropped
                frame_count = sequence
                inference_started = time.perf_counter()
                packet = bridge.decode(packet, detector_pool)
                timing_totals["decode"] += time.perf_counter() - inference_started
                if packet["left"].shape[1::-1] != image_size or packet["right"].shape[1::-1] != image_size:
                    raise RuntimeError("live resolution differs from EGO calibration")
                sync_delta_us = int(packet["right_timestamp_us"] - packet["left_timestamp_us"])
                if abs(sync_delta_us) > args.max_sync_delta_us:
                    continue
                if (sequence - 1) % args.process_every != 0:
                    continue

                stage_started = time.perf_counter()
                left_rectified = cv2.remap(
                    packet["left"], rectification["map_left_x"], rectification["map_left_y"], cv2.INTER_LINEAR
                )
                right_rectified = cv2.remap(
                    packet["right"], rectification["map_right_x"], rectification["map_right_y"], cv2.INTER_LINEAR
                )
                timing_totals["rectify"] += time.perf_counter() - stage_started
                if base_timestamp_us is None:
                    base_timestamp_us = min(packet["left_timestamp_us"], packet["right_timestamp_us"])
                left_ms = max(int((packet["left_timestamp_us"] - base_timestamp_us) // 1000), last_left_ms + 1)
                right_ms = max(int((packet["right_timestamp_us"] - base_timestamp_us) // 1000), last_right_ms + 1)
                last_left_ms, last_right_ms = left_ms, right_ms
                stage_started = time.perf_counter()
                left_future = detector_pool.submit(
                    detect, left_landmarker, left_rectified, left_ms, image_size
                )
                right_future = detector_pool.submit(
                    detect, right_landmarker, right_rectified, right_ms, image_size
                )
                left_hands = left_future.result()
                right_hands = right_future.result()
                timing_totals["detect"] += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                matching_images = (
                    prepare_stereo_matching_images(left_rectified, right_rectified)
                    if args.stereo_refine else None
                )
                matches = [
                    triangulate(candidate, rectification, args, matching_images)
                    for candidate in associate_hands(left_hands, right_hands)
                ]
                timing_totals["refine_triangulate"] += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                timestamp_s = 0.5e-6 * (packet["left_timestamp_us"] + packet["right_timestamp_us"])
                tracker.assign(matches, timestamp_s)
                for match in matches:
                    online_filter.update(match, timestamp_s)
                    for joint in range(21):
                        point = match["filtered_points_left"][joint]
                        filtered_valid = bool(match["filtered_valid"][joint])
                        csv_writer.writerow({
                            "capture_index": frame_count - 1,
                            "left_index": packet["left_index"], "right_index": packet["right_index"],
                            "left_timestamp_us": packet["left_timestamp_us"],
                            "right_timestamp_us": packet["right_timestamp_us"],
                            "track_id": match["track_id"],
                            "handedness": match.get("stable_handedness", match["left"]["label"]),
                            "landmark_index": joint, "raw_valid": int(match["valid"][joint]),
                            "filtered_valid": int(filtered_valid),
                            "predicted": int(match["predicted_3d"][joint]),
                            "depth_quality": f"{match['depth_quality'][joint]:.6f}",
                            "refinement_used": int(match["refinement_used"][joint]),
                            "refinement_quality": f"{match['refinement_quality'][joint]:.6f}",
                            "x_left_camera_m": f"{point[0]:.9f}" if filtered_valid else "nan",
                            "y_left_camera_m": f"{point[1]:.9f}" if filtered_valid else "nan",
                            "z_left_camera_m": f"{point[2]:.9f}" if filtered_valid else "nan",
                        })
                timing_totals["stereo_csv"] += time.perf_counter() - stage_started
                current_mano_results: dict[int, dict] = {}
                stage_started = time.perf_counter()
                if mano_fitter is not None:
                    for match in matches:
                        result = mano_fitter.update(match)
                        if result is not None:
                            current_mano_results[int(match["track_id"])] = result
                            all_mano_results[int(match["track_id"])] = result
                            angle_row = {
                                "capture_index": frame_count - 1,
                                "left_index": packet["left_index"],
                                "right_index": packet["right_index"],
                                "left_timestamp_us": packet["left_timestamp_us"],
                                "right_timestamp_us": packet["right_timestamp_us"],
                                "track_id": result["track_id"],
                                "handedness": result["handedness"],
                                "observed": int(result.get("observed", True)),
                                "fit_loss": f"{result['loss']:.9f}",
                                "fit_ms": f"{result['fit_ms']:.3f}",
                                "iterations": result["iterations"],
                                "device": result["device"],
                            }
                            for key, raw_value, value in zip(
                                KINEMATIC_KEYS,
                                result.get("kinematic_raw", result["kinematic"]),
                                result["kinematic"],
                            ):
                                angle_row[f"{key}_raw"] = f"{raw_value:.9f}"
                                angle_row[key] = f"{value:.9f}"
                                angle_row[key.replace("_rad", "_deg")] = f"{np.degrees(value):.6f}"
                            position = result["end_effector_position_m"]
                            rpy = result["end_effector_rpy_rad"]
                            quaternion = result["end_effector_quaternion_xyzw"]
                            rotation_matrix = result["end_effector_rotation_matrix"]
                            delta_camera = result["end_effector_delta_camera_m"]
                            delta_hand0 = result["end_effector_delta_hand0_m"]
                            angle_row.update({
                                "hand_x_m": f"{position[0]:.9f}",
                                "hand_y_m": f"{position[1]:.9f}",
                                "hand_z_m": f"{position[2]:.9f}",
                                "hand_dx_camera_m": f"{delta_camera[0]:.9f}",
                                "hand_dy_camera_m": f"{delta_camera[1]:.9f}",
                                "hand_dz_camera_m": f"{delta_camera[2]:.9f}",
                                "hand_dx_hand0_m": f"{delta_hand0[0]:.9f}",
                                "hand_dy_hand0_m": f"{delta_hand0[1]:.9f}",
                                "hand_dz_hand0_m": f"{delta_hand0[2]:.9f}",
                                "hand_roll_rad": f"{rpy[0]:.9f}",
                                "hand_pitch_rad": f"{rpy[1]:.9f}",
                                "hand_yaw_rad": f"{rpy[2]:.9f}",
                                "hand_roll_deg": f"{np.degrees(rpy[0]):.6f}",
                                "hand_pitch_deg": f"{np.degrees(rpy[1]):.6f}",
                                "hand_yaw_deg": f"{np.degrees(rpy[2]):.6f}",
                                "hand_qx": f"{quaternion[0]:.9f}",
                                "hand_qy": f"{quaternion[1]:.9f}",
                                "hand_qz": f"{quaternion[2]:.9f}",
                                "hand_qw": f"{quaternion[3]:.9f}",
                            })
                            angle_row.update({
                                f"hand_r{row}{column}": f"{rotation_matrix[row, column]:.9f}"
                                for row in range(3) for column in range(3)
                            })
                            angle_writer.writerow(angle_row)
                timing_totals["mano"] += time.perf_counter() - stage_started
                inference_elapsed = time.perf_counter() - inference_started
                inference_time += inference_elapsed
                inference_count += 1
                elapsed = time.perf_counter() - started
                stats = {
                    "capture_fps": frame_count / elapsed if elapsed > 0 else 0.0,
                    "inference_fps": inference_count / inference_time if inference_time > 0 else 0.0,
                    "latency_ms": max(0.0, (time.monotonic_ns() - packet["host_ns"]) / 1e6),
                    "sync_delta_us": sync_delta_us,
                    "dropped_pairs": dropped_count,
                }
                latency_samples.append(stats["latency_ms"])
                stage_started = time.perf_counter()
                if mano_fitter is not None:
                    preview = compose_live_mano(
                        packet["left"], current_mano_results, all_mano_results,
                        fisheye_camera, fisheye_distortion, args,
                        frame_count - 1, int(packet["left_index"]), stats,
                    )
                else:
                    preview = compose_preview(
                        left_rectified, right_rectified, matches, rectification, stats, args.display_width
                    )
                timing_totals["render"] += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            30.0 / args.process_every, preview.shape[1::-1],
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot create {video_path}")
                    writer.write(preview)
                if not args.no_display:
                    title = "EGO Live MANO 21-DOF" if args.mano else "EGO Live Stereo Hand Tracking"
                    cv2.imshow(title, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                timing_totals["display_record"] += time.perf_counter() - stage_started
    finally:
        if latest_reader is not None:
            latest_reader.close()
        elif bridge is not None:
            bridge.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print("EGO live stereo tracking stopped")
    print(f"Captured pairs: {frame_count}")
    print(f"Inference pairs: {inference_count}")
    print(f"Dropped stale pairs: {dropped_count}")
    print(f"Elapsed: {elapsed:.2f} s")
    if elapsed > 0:
        print(f"End-to-end throughput: {inference_count / elapsed:.2f} FPS")
    if inference_time > 0:
        print(f"Detection/MANO throughput: {inference_count / inference_time:.2f} FPS")
    if latency_samples:
        print(
            f"Latency median/P95: {np.median(latency_samples):.1f}/"
            f"{np.percentile(latency_samples, 95):.1f} ms"
        )
    if inference_count:
        timings = " | ".join(
            f"{name} {seconds * 1000.0 / inference_count:.1f} ms"
            for name, seconds in timing_totals.items()
        )
        print(f"Stage averages: {timings}")
    print(f"3D CSV: {csv_path}")
    if args.mano:
        print(f"MANO 21-DOF CSV: {angle_csv_path}")
    if writer is not None:
        print(f"Annotated video: {video_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
