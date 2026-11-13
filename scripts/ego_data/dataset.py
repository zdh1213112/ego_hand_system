"""Sequential access to normalized Orbbec video streams."""
from pathlib import Path
import cv2
import numpy as np

class SequentialVideoReader:
    """Read strictly increasing frame indices without random video seeks."""

    def __init__(self, path: Path, expected_size: tuple[int, int]):
        self.path = path
        self.expected_size = expected_size
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.next_index = 0

    def read(self, target_index: int) -> np.ndarray:
        if target_index < self.next_index:
            raise RuntimeError(
                f"video frame indices must be read in increasing order: {target_index} < {self.next_index}"
            )
        frame = None
        while self.next_index <= target_index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"video {self.path} ended before frame {target_index}")
            self.next_index += 1
        if frame is None or frame.shape[1::-1] != self.expected_size:
            raise RuntimeError(f"decoded frame from {self.path} has an unexpected size")
        return frame

    def close(self) -> None:
        self.capture.release()
