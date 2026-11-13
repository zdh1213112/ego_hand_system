"""Shared Annex-B H264 parsing and lossless Matroska remuxing."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av


def h264_nal_types(payload: bytes) -> list[int]:
    """Return Annex-B NAL unit types without parsing slice payloads."""
    result = []
    index = 0
    while index + 3 < len(payload):
        if payload[index:index + 4] == b"\x00\x00\x00\x01":
            start = index + 4
            index = start
        elif payload[index:index + 3] == b"\x00\x00\x01":
            start = index + 3
            index = start
        else:
            index += 1
            continue
        if start < len(payload):
            result.append(payload[start] & 0x1F)
    return result


class H264MatroskaWriter:
    """Losslessly remux Annex-B H264 access units into Matroska."""

    def __init__(self, camera_id: str, path: Path, image_size: tuple[int, int]):
        self.camera_id = camera_id
        self.path = path
        self.container = av.open(str(path), "w", format="matroska")
        self.stream = self.container.add_stream("h264", rate=30)
        self.stream.width, self.stream.height = image_size
        self.message_count = 0
        self.packet_count = 0
        self.leading_non_keyframe_count = 0
        self._base_timestamp_ns: int | None = None
        self.rows: list[dict] = []

    def write(self, payload: bytes, timestamp_ns: int) -> None:
        source_index = self.message_count
        self.message_count += 1
        nal_types = h264_nal_types(payload)
        is_keyframe = 5 in nal_types
        if self._base_timestamp_ns is None:
            if not is_keyframe:
                self.leading_non_keyframe_count += 1
                return
            self._base_timestamp_ns = int(timestamp_ns)
        packet = av.Packet(payload)
        packet.stream = self.stream
        packet.pts = int(timestamp_ns) - self._base_timestamp_ns
        packet.dts = packet.pts
        packet.time_base = Fraction(1, 1_000_000_000)
        packet.is_keyframe = is_keyframe
        self.container.mux(packet)
        self.rows.append({
            "frame_index": self.packet_count,
            "timestamp_ns": int(timestamp_ns),
            "timestamp_us": int(timestamp_ns) // 1000,
            "source_message_index": source_index,
            "keyframe": int(is_keyframe),
        })
        self.packet_count += 1

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None

    def validate(self, expected_size: tuple[int, int]) -> int:
        def decoded_frame_count(codec_name: str) -> int:
            decoded = 0
            with av.open(str(self.path), "r") as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height) != expected_size:
                    raise RuntimeError(
                        f"{self.camera_id} remuxed size "
                        f"{(stream.width, stream.height)} differs from "
                        f"camera_info {expected_size}"
                    )
                decoder = av.CodecContext.create(codec_name, "r")
                decoder.extradata = stream.codec_context.extradata
                for packet in container.demux(stream):
                    for frame in decoder.decode(packet):
                        if (frame.width, frame.height) != expected_size:
                            raise RuntimeError(
                                f"{self.camera_id} decoded a frame with an "
                                "unexpected size"
                            )
                        decoded += 1
            return decoded

        try:
            decoded = decoded_frame_count("h264_cuvid")
            if decoded != self.packet_count:
                decoded = decoded_frame_count("h264")
        except Exception:
            decoded = decoded_frame_count("h264")
        if decoded != self.packet_count:
            raise RuntimeError(
                f"{self.camera_id} remuxed {self.packet_count} access units but decoded {decoded} frames"
            )
        return decoded

    def statistics(self, decoded_count: int) -> dict:
        return {
            "compressed_message_count": self.message_count,
            "video_packet_count": self.packet_count,
            "validated_decoded_frame_count": decoded_count,
            "leading_non_keyframe_count": self.leading_non_keyframe_count,
        }
