"""Video encoding and detection drawing for shared-model videos."""

from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


class _OpenCvVideoWriter:
    def __init__(
        self,
        path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot create video: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


class _NvencVideoWriter:
    def __init__(
        self,
        path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        import av

        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(
            "h264_nvenc", rate=Fraction(fps).limit_denominator(100_000)
        )
        self.stream.width, self.stream.height = frame_size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "preset": "p2",
            "tune": "hq",
            "rc": "vbr",
            "cq": "19",
        }
        # Fail during setup, where auto mode can still safely use its fallback.
        self.stream.codec_context.open()
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        video_frame = self.av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def release(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


def _create_video_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
    codec: str,
):
    if codec in ("auto", "h264_nvenc"):
        try:
            return _NvencVideoWriter(path, fps, frame_size), "h264_nvenc"
        except Exception as error:
            path.unlink(missing_ok=True)
            if codec == "h264_nvenc":
                raise RuntimeError(f"cannot initialize h264_nvenc: {error}") from error
            print(
                f"NVENC unavailable ({error}); falling back to mp4v",
                flush=True,
            )
    return _OpenCvVideoWriter(path, fps, frame_size), "mp4v"


def _draw_final_detection(
    frame: np.ndarray,
    view: dict,
    side: int,
    joints_source: str,
) -> None:
    box = np.rint(view.get("bbox_xyxy", [])).astype(np.int32)
    if box.shape != (4,):
        return
    # Final delivery convention: physical left is cyan (FINAL L), right is
    # yellow (FINAL R).  Keep this independent of the diagnostic detector
    # palette so a label swap cannot make the final-only view ambiguous.
    # OpenCV uses BGR.  Physical left is cyan and physical right is yellow
    # in the delivery convention (the old tuples were accidentally swapped).
    color = (255, 255, 0) if side == 0 else (0, 255, 255)
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (20, 20, 20), 7, cv2.LINE_AA)
    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{'FINAL' if joints_source == 'refined' else 'RAW'} {'R' if side else 'L'}",
        (int(box[0]), max(28, int(box[1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
