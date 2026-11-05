"""Bounded reader for the EGO4 four-camera MJPEG live bridge protocol."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


HEADER = struct.Struct("<4sHHIIIQ" + "IQQ" * 4)
CAMERAS = ("cam_0", "cam_1", "cam_2", "cam_3")


def read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise EOFError("EGO4 bridge closed its output")
        data.extend(block)
    return bytes(data)


class FourBridge:
    def __init__(self, executable: Path, sdk_config: Path):
        self.process = subprocess.Popen(
            [str(executable), "--sdk-config", str(sdk_config), "--four-view"],
            stdout=subprocess.PIPE, bufsize=0,
        )
        if self.process.stdout is None:
            raise RuntimeError("failed to open EGO4 bridge output")

    def read(self) -> dict:
        fields = HEADER.unpack(read_exact(self.process.stdout, HEADER.size))
        magic, version, header_size, count, width, height, host_ns = fields[:7]
        if (magic, version, header_size, count) != (b"EGO4", 2, HEADER.size, 4):
            raise ValueError("invalid EGO4 bridge header")
        if not (0 < width <= 8192 and 0 < height <= 8192):
            raise ValueError("invalid EGO4 frame dimensions")
        entries = [fields[7 + 3 * i:10 + 3 * i] for i in range(4)]
        if any(size <= 0 or size > 32_000_000 for size, _, _ in entries):
            raise ValueError("invalid EGO4 JPEG size")
        images = {camera: read_exact(self.process.stdout, entry[0])
                  for camera, entry in zip(CAMERAS, entries)}
        return {
            "jpeg": images, "width": width, "height": height,
            "host_ns": host_ns,
            "timestamp_us": {camera: entry[1] for camera, entry in zip(CAMERAS, entries)},
            "index": {camera: entry[2] for camera, entry in zip(CAMERAS, entries)},
        }

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


def decode_camera(packet: dict, camera: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(packet["jpeg"][camera], np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (packet["height"], packet["width"]):
        raise ValueError(f"failed to decode {camera} MJPEG frame")
    return image


class LatestFourReader:
    def __init__(self, bridge: FourBridge, capacity: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.bridge = bridge
        self.capacity = capacity
        self.queue = deque()
        self.condition = threading.Condition()
        self.sequence = 0
        self.dropped = 0
        self.closing = False
        self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="ego4-reader")
        self.thread.start()

    def _run(self):
        try:
            while not self.closing:
                packet = self.bridge.read()
                with self.condition:
                    self.sequence += 1
                    if len(self.queue) == self.capacity:
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

    def get(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.queue and not self.stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.queue:
                return self.queue.popleft()
            if self.error is not None:
                raise self.error
            return None

    def close(self):
        self.closing = True
        self.bridge.close()
        self.thread.join(timeout=2)


class DecodedFourReader:
    """Decode the main stereo pair while the caller runs WiLoR on a prior pair."""

    def __init__(self, source: LatestFourReader, capacity: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.source = source
        self.capacity = capacity
        self.queue = deque()
        self.condition = threading.Condition()
        self.dropped_decoded = 0
        self.closing = False
        self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="ego4-decode")
        self.thread.start()

    @property
    def sequence(self):
        return self.source.sequence

    @property
    def dropped(self):
        return self.source.dropped + self.dropped_decoded

    def _run(self):
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                while not self.closing:
                    item = self.source.get(timeout=0.2)
                    if item is None:
                        if self.source.stopped:
                            break
                        continue
                    sequence, packet = item
                    decode_started = time.perf_counter()
                    futures = [executor.submit(decode_camera, packet, name)
                               for name in CAMERAS[:2]]
                    frames = {name: future.result()
                              for name, future in zip(CAMERAS[:2], futures)}
                    packet["decode_main_ms"] = round((time.perf_counter() - decode_started) * 1000, 2)
                    packet["decoded_ns"] = time.monotonic_ns()
                    with self.condition:
                        if len(self.queue) == self.capacity:
                            self.queue.popleft()
                            self.dropped_decoded += 1
                        self.queue.append((sequence, packet, frames))
                        self.condition.notify_all()
        except Exception as error:
            if not self.closing:
                self.error = error
        finally:
            with self.condition:
                self.stopped = True
                self.condition.notify_all()

    def get(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.queue and not self.stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.queue:
                return self.queue.popleft()
            if self.error is not None:
                raise self.error
            return None

    def close(self):
        self.closing = True
        self.source.close()
        self.thread.join(timeout=2)
