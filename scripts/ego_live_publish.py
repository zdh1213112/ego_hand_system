"""Optional, bounded Unix-socket output for the live WiLoR viewer.

The capture/inference thread never waits for the visualizer. If the viewer is
slower than inference, only the newest complete result is kept.
"""

from __future__ import annotations

from collections import deque
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
import orjson


CAMERAS = ("cam_0", "cam_1", "cam_2", "cam_3")
LENGTH = struct.Struct("!I")


class LivePublisher:
    def __init__(self, path, cameras=("cam_0",), depth_worker=None, *, image_width=None,
                 imu_hz=None, compact=False, preview=None, native=None, max_fps=None):
        self.path = str(path)
        self.cameras = tuple(cameras)
        self.depth_worker = depth_worker
        self.image_width = image_width
        self.imu_hz = imu_hz
        self.compact = compact
        self.preview = preview
        self.native = native
        self.max_fps = max_fps
        self.last_queued_us = None
        self.rate_skipped = 0
        self.last_imu_us = {"accel": -1, "gyro": -1}
        if image_width is not None and image_width < 160:
            raise ValueError("display image width must be >= 160")
        if imu_hz is not None and imu_hz <= 0:
            raise ValueError("display IMU rate must be positive")
        if preview is not None and native is not None:
            raise ValueError("preview image and native Rerun panels are mutually exclusive")
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or any(
            name not in CAMERAS for name in self.cameras
        ):
            raise ValueError("live cameras must be unique EGO-4B camera names")
        self.pending = deque(maxlen=1)
        self.condition = threading.Condition()
        self.closing = False
        self.dropped = 0
        self.sent = 0
        self.thread = threading.Thread(target=self._run, name="ego-live-publish", daemon=True)
        self.thread.start()

    def submit(self, row, packet, mesh_sources=None):
        # Original MJPEG bytes are sent directly; no extra image encoding on
        # the inference thread. The row is complete at this point.
        stamp = int(row["timestamp_us"]["cam_0"])
        if (self.max_fps and self.last_queued_us is not None and
                stamp - self.last_queued_us < 1_000_000 / self.max_fps):
            self.rate_skipped += 1
            return
        self.last_queued_us = stamp
        images = {name: packet["jpeg"][name] for name in self.cameras}
        with self.condition:
            keys = (("timestamp_us", "hands", "latency_ms", "sync_spread_us") if self.preview or self.native else
                    ("timestamp_us", "hands", "imu", "latency_ms", "sync_spread_us"))
            display_row = {key: row[key] for key in keys if key in row} if self.compact else dict(row)
            if self.pending:
                old_row = self.pending[-1][0]
                for kind in ("accel", "gyro"):
                    older = old_row.get("imu", {}).get(kind, [])
                    if older:
                        imu = dict(display_row.get("imu", {}))
                        imu[kind] = (older + imu.get(kind, []))[-4096:]
                        display_row["imu"] = imu
                self.dropped += 1
            display_row["display_queue_dropped"] = self.dropped
            self.pending.append((display_row, images, row if self.preview or self.native else None,
                                 mesh_sources))
            self.condition.notify()

    def _display_imu(self, row):
        if self.imu_hz is None:
            return
        interval = round(1_000_000 / self.imu_hz)
        reduced = {}
        for kind in ("accel", "gyro"):
            samples = []
            for sample in row.get("imu", {}).get(kind, []):
                try:
                    stamp = int(sample["timestamp_us"])
                    xyz = np.asarray(sample["xyz"], dtype=np.float64)
                    if xyz.shape == (3,) and np.isfinite(xyz).all():
                        samples.append((stamp, sample))
                except (KeyError, TypeError, ValueError):
                    continue
            samples.sort(key=lambda item: item[0])
            selected = []
            for stamp, sample in samples:
                if stamp - self.last_imu_us[kind] >= interval:
                    selected.append(sample)
                    self.last_imu_us[kind] = stamp
            reduced[kind] = selected
        row["imu"] = reduced

    def _display_images(self, images):
        if self.image_width is None:
            return images
        scaled = {}
        for name, jpeg in images.items():
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"failed to decode display camera {name}")
            if frame.shape[1] > self.image_width:
                height = max(1, round(frame.shape[0] * self.image_width / frame.shape[1]))
                frame = cv2.resize(frame, (self.image_width, height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    raise ValueError(f"failed to encode display camera {name}")
                jpeg = encoded.tobytes()
            scaled[name] = jpeg
        return scaled

    def _preview_image(self, row, images, source_row, mesh_sources):
        from ego_live_wilor_preview import render_preview

        camera = self.preview["camera"]
        frame = cv2.imdecode(np.frombuffer(images[camera], np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("failed to decode preview camera")
        preview_row = dict(source_row)
        preview_row["batch_pairs"] = 1
        canvas = render_preview(
            {"index": source_row["frame_index"]}, {camera: frame}, preview_row,
            self.max_fps or 0, self.dropped, camera,
            self.preview["cameras"][camera], mesh_sources,
            self.preview["faces"], self.preview["cameras"], False,
        )
        target_width = self.image_width or 900
        if canvas.shape[1] > target_width:
            canvas = cv2.resize(canvas, (target_width, round(canvas.shape[0] * target_width / canvas.shape[1])),
                                interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            raise ValueError("failed to encode live preview")
        return {"preview": encoded.tobytes()}

    def _native_payload(self, row, source_row, mesh_sources):
        """Send geometric data separately so Rerun renders each panel natively."""
        from ego_live_wilor_preview import fit_mano_mesh_to_stereo

        camera = self.native["camera"]
        calibration = self.native["cameras"][camera]
        width, height = calibration.image_size
        if self.image_width is not None and width > self.image_width:
            display_height = round(height * self.image_width / width)
            row["camera_overlay_scale"] = [self.image_width / width, display_height / height]
        else:
            row["camera_overlay_scale"] = [1.0, 1.0]
        row["native_panel"] = True
        row["camera_hands"] = {
            str(side): {
                "bbox_xyxy": hand.get("bbox_xyxy"),
                "joints_2d": hand.get("joints_2d_stabilized", hand.get("joints_2d")),
                "confidence": hand.get("confidence", 0),
            }
            for side, hand in source_row.get("camera_hands", {}).get(camera, {}).items()
        }
        fused = {int(hand["is_right"]): hand for hand in source_row.get("hands", [])}
        meshes = {}
        for side, source in (mesh_sources or {}).items():
            hand = fused.get(int(side))
            if hand is None:
                continue
            joints = hand.get("joints_3d_base_m_display", hand.get("joints_3d_base_m"))
            if joints is None:
                continue
            source_camera = self.native["cameras"].get(source["camera"])
            if source_camera is None:
                continue
            vertices, error_m = fit_mano_mesh_to_stereo(
                source["vertices"], source["joints"], joints, source_camera,
            )
            if vertices is not None:
                meshes[str(side)] = {
                    "vertices_base_m": np.round(vertices, 5).tolist(),
                    "stereo_fit_mm": round(error_m * 1000, 2),
                }
        row["mano_meshes"] = meshes
        if meshes:
            row["mano_faces"] = self.native["faces"].tolist()

    def _run(self):
        connection = None
        last_error = None
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.closing:
                        self.condition.wait()
                    if self.closing and not self.pending:
                        break
                    row, images, source_row, mesh_sources = self.pending.pop()
                self._display_imu(row)
                if self.preview:
                    images = self._preview_image(row, images, source_row, mesh_sources)
                else:
                    images = self._display_images(images)
                    if self.native:
                        self._native_payload(row, source_row, mesh_sources)
                if connection is None:
                    if self.closing:
                        break
                    try:
                        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        connection.settimeout(0.5)
                        connection.connect(self.path)
                    except OSError as error:
                        if str(error) != last_error:
                            print(f"Live visualizer unavailable: {error}", file=sys.stderr, flush=True)
                            last_error = str(error)
                        connection.close()
                        connection = None
                        time.sleep(0.2)
                        continue
                names = tuple(images)
                depth_result = self.depth_worker.take(int(row["timestamp_us"]["cam_0"])) if self.depth_worker else None
                if depth_result is not None:
                    row["depth_timestamp_us"] = depth_result[0]
                    row["depth_calibration"] = {
                        "K_rectified": self.depth_worker.K_rectified,
                        "R_rectified_from_cam0": self.depth_worker.R_rectified_from_cam0,
                    }
                metadata = orjson.dumps({
                    "version": 2,
                    "row": row,
                    "images": [{"camera": name, "size": len(images[name])} for name in names],
                    "depth": {"encoding": "png16_mm", "size": len(depth_result[1])} if depth_result else None,
                }, option=orjson.OPT_NON_STR_KEYS)
                try:
                    connection.sendall(LENGTH.pack(len(metadata)))
                    connection.sendall(metadata)
                    for name in names:
                        connection.sendall(images[name])
                    if depth_result is not None:
                        connection.sendall(depth_result[1])
                    self.sent += 1
                    last_error = None
                except OSError as error:
                    if str(error) != last_error:
                        print(f"Live visualizer disconnected: {error}", file=sys.stderr, flush=True)
                        last_error = str(error)
                    connection.close()
                    connection = None
                    if self.closing:
                        break
        finally:
            if connection is not None:
                connection.close()

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)
