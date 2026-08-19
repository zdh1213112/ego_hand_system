"""Streaming GEN DAS EGO MCAP reader and H264 decoder."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
from mcap.reader import NonSeekingReader
from mcap_protobuf.decoder import DecoderFactory

from .calibration import CameraCalibration, camera_from_genrobot_message


@dataclass(frozen=True)
class DecodedFrame:
    camera_id: str
    frame_index: int
    timestamp_ns: int
    source_message_index: int
    image: object


class H264StreamDecoder:
    """Decode Annex-B packets while tolerating packets before the first keyframe."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.codec = av.CodecContext.create("h264", "r")
        self.message_count = 0
        self.frame_count = 0
        self.decode_errors = 0
        self._source_indices: dict[int, int] = {}
        self._last_timestamp_ns = -1

    def decode(self, payload: bytes, timestamp_ns: int) -> list[DecodedFrame]:
        source_index = self.message_count
        self.message_count += 1
        packet = av.Packet(payload)
        packet.pts = int(timestamp_ns)
        packet.dts = int(timestamp_ns)
        packet.time_base = Fraction(1, 1_000_000_000)
        self._source_indices[int(timestamp_ns)] = source_index
        try:
            frames = self.codec.decode(packet)
        except av.error.InvalidDataError:
            self.decode_errors += 1
            return []
        return self._convert(frames, timestamp_ns, source_index)

    def flush(self) -> list[DecodedFrame]:
        try:
            frames = self.codec.decode(None)
        except av.error.InvalidDataError:
            self.decode_errors += 1
            return []
        fallback_timestamp = max(self._last_timestamp_ns + 1, 0)
        return self._convert(frames, fallback_timestamp, max(self.message_count - 1, 0))

    def _convert(self, frames, fallback_timestamp: int, fallback_source: int) -> list[DecodedFrame]:
        output = []
        for frame in frames:
            timestamp_ns = int(frame.pts) if frame.pts is not None else int(fallback_timestamp)
            if timestamp_ns <= self._last_timestamp_ns:
                raise RuntimeError(f"{self.camera_id} decoded timestamps are not strictly increasing")
            source_index = self._source_indices.pop(timestamp_ns, fallback_source)
            image = frame.to_ndarray(format="bgr24")
            output.append(DecodedFrame(
                self.camera_id, self.frame_count, timestamp_ns, source_index, image
            ))
            self.frame_count += 1
            self._last_timestamp_ns = timestamp_ns
        return output

    def statistics(self) -> dict:
        return {
            "compressed_message_count": self.message_count,
            "decoded_frame_count": self.frame_count,
            "decode_error_count": self.decode_errors,
        }


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
    """Losslessly remux GEN Annex-B H264 access units into Matroska."""

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
        decoded = 0
        with av.open(str(self.path), "r") as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != expected_size:
                raise RuntimeError(
                    f"{self.camera_id} remuxed size {(stream.width, stream.height)} "
                    f"differs from camera_info {expected_size}"
                )
            for frame in container.decode(stream):
                if (frame.width, frame.height) != expected_size:
                    raise RuntimeError(f"{self.camera_id} decoded a frame with an unexpected size")
                decoded += 1
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


def camera_topics(camera_id: str) -> tuple[str, str]:
    if not camera_id.startswith("camera") or not camera_id[6:].isdigit():
        raise ValueError(f"camera id must look like camera2, got {camera_id!r}")
    root = f"/robot0/sensor/{camera_id}"
    return f"{root}/compressed", f"{root}/camera_info"


def _same_calibration(first: CameraCalibration, second: CameraCalibration) -> bool:
    import numpy as np
    return (
        first.camera_id == second.camera_id
        and first.frame_id == second.frame_id
        and first.model == second.model
        and first.image_size == second.image_size
        and np.allclose(first.K, second.K, atol=1e-10, rtol=0)
        and np.allclose(first.distortion, second.distortion, atol=1e-10, rtol=0)
        and np.allclose(first.T_base_camera, second.T_base_camera, atol=1e-10, rtol=0)
    )


def decode_stereo_mcap(
    path: Path,
    camera_ids: tuple[str, ...],
    on_frame: Callable[[DecodedFrame], None],
    max_frames_per_camera: int = 0,
) -> tuple[dict[str, CameraCalibration], dict[str, dict], dict[str, str]]:
    """Decode selected camera streams in one sequential pass.

    Returns calibration, per-camera decoder statistics, and the topic mapping.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    topics: dict[str, tuple[str, str]] = {camera_id: camera_topics(camera_id) for camera_id in camera_ids}
    topic_to_camera = {}
    for camera_id, (image_topic, info_topic) in topics.items():
        topic_to_camera[image_topic] = (camera_id, "image")
        topic_to_camera[info_topic] = (camera_id, "calibration")
    decoders = {camera_id: H264StreamDecoder(camera_id) for camera_id in camera_ids}
    calibrations: dict[str, CameraCalibration] = {}
    info_counts = {camera_id: 0 for camera_id in camera_ids}
    factory = DecoderFactory()
    with path.open("rb") as stream:
        reader = NonSeekingReader(stream, decoder_factories=[factory])
        for schema, channel, message in reader.iter_messages(
            topics=topic_to_camera.keys(), log_time_order=False
        ):
            camera_id, kind = topic_to_camera[channel.topic]
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                raise RuntimeError(f"no protobuf decoder for {channel.topic}")
            proto = decoder(message.data)
            if kind == "calibration":
                calibration = camera_from_genrobot_message(camera_id, proto)
                previous = calibrations.get(camera_id)
                if previous is not None and not _same_calibration(previous, calibration):
                    raise RuntimeError(f"{camera_id} calibration changes within one MCAP")
                calibrations[camera_id] = calibration
                info_counts[camera_id] += 1
                continue
            if str(proto.format).strip().lower() != "h264":
                raise ValueError(f"{camera_id} image format must be h264, got {proto.format!r}")
            if max_frames_per_camera and decoders[camera_id].frame_count >= max_frames_per_camera:
                if (
                    all(item.frame_count >= max_frames_per_camera for item in decoders.values())
                    and all(item in calibrations for item in camera_ids)
                ):
                    break
                continue
            for frame in decoders[camera_id].decode(bytes(proto.data), int(message.log_time)):
                on_frame(frame)
        if not max_frames_per_camera:
            for decoder in decoders.values():
                for frame in decoder.flush():
                    on_frame(frame)
    missing = set(camera_ids) - calibrations.keys()
    if missing:
        raise RuntimeError(f"missing camera_info for: {', '.join(sorted(missing))}")
    statistics = {}
    for camera_id, decoder in decoders.items():
        statistics[camera_id] = {**decoder.statistics(), "camera_info_message_count": info_counts[camera_id]}
    topic_manifest = {
        f"{camera_id}_image": topics[camera_id][0] for camera_id in camera_ids
    } | {
        f"{camera_id}_camera_info": topics[camera_id][1] for camera_id in camera_ids
    }
    return calibrations, statistics, topic_manifest


def remux_stereo_mcap(
    path: Path,
    camera_ids: tuple[str, ...],
    output_paths: dict[str, Path],
    max_frames_per_camera: int = 0,
) -> tuple[dict[str, CameraCalibration], dict[str, list[dict]], dict[str, dict], dict[str, str]]:
    """Remux selected GEN H264 streams without decoding or re-encoding."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    topics = {camera_id: camera_topics(camera_id) for camera_id in camera_ids}
    topic_to_camera = {}
    for camera_id, (image_topic, info_topic) in topics.items():
        topic_to_camera[image_topic] = (camera_id, "image")
        topic_to_camera[info_topic] = (camera_id, "calibration")
    calibrations: dict[str, CameraCalibration] = {}
    info_counts = {camera_id: 0 for camera_id in camera_ids}
    pending_packets: dict[str, list[tuple[bytes, int]]] = {camera_id: [] for camera_id in camera_ids}
    writers: dict[str, H264MatroskaWriter] = {}
    factory = DecoderFactory()

    def ensure_writer(camera_id: str) -> H264MatroskaWriter | None:
        if camera_id in writers:
            return writers[camera_id]
        calibration = calibrations.get(camera_id)
        if calibration is None:
            return None
        writer = H264MatroskaWriter(camera_id, output_paths[camera_id], calibration.image_size)
        writers[camera_id] = writer
        for payload, timestamp_ns in pending_packets.pop(camera_id):
            writer.write(payload, timestamp_ns)
        return writer

    try:
        with path.open("rb") as stream:
            reader = NonSeekingReader(stream, decoder_factories=[factory])
            for schema, channel, message in reader.iter_messages(
                topics=topic_to_camera.keys(), log_time_order=False
            ):
                camera_id, kind = topic_to_camera[channel.topic]
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is None:
                    raise RuntimeError(f"no protobuf decoder for {channel.topic}")
                proto = decoder(message.data)
                if kind == "calibration":
                    calibration = camera_from_genrobot_message(camera_id, proto)
                    previous = calibrations.get(camera_id)
                    if previous is not None and not _same_calibration(previous, calibration):
                        raise RuntimeError(f"{camera_id} calibration changes within one MCAP")
                    calibrations[camera_id] = calibration
                    info_counts[camera_id] += 1
                    ensure_writer(camera_id)
                    continue
                if str(proto.format).strip().lower() != "h264":
                    raise ValueError(f"{camera_id} image format must be h264, got {proto.format!r}")
                writer = ensure_writer(camera_id)
                if writer is None:
                    pending_packets[camera_id].append((bytes(proto.data), int(message.log_time)))
                    continue
                if max_frames_per_camera and writer.packet_count >= max_frames_per_camera:
                    if (
                        all(item in writers and writers[item].packet_count >= max_frames_per_camera for item in camera_ids)
                        and all(item in calibrations for item in camera_ids)
                    ):
                        break
                    continue
                writer.write(bytes(proto.data), int(message.log_time))
        missing = set(camera_ids) - calibrations.keys()
        if missing:
            raise RuntimeError(f"missing camera_info for: {', '.join(sorted(missing))}")
        for camera_id in camera_ids:
            ensure_writer(camera_id)
            if not writers[camera_id].packet_count:
                raise RuntimeError(f"{camera_id} contains no decodable H264 keyframe sequence")
    finally:
        for writer in writers.values():
            writer.close()
    statistics = {}
    rows = {}
    for camera_id, writer in writers.items():
        decoded_count = writer.validate(calibrations[camera_id].image_size)
        statistics[camera_id] = {
            **writer.statistics(decoded_count),
            "camera_info_message_count": info_counts[camera_id],
        }
        rows[camera_id] = writer.rows
    topic_manifest = {
        f"{camera_id}_image": topics[camera_id][0] for camera_id in camera_ids
    } | {
        f"{camera_id}_camera_info": topics[camera_id][1] for camera_id in camera_ids
    }
    return calibrations, rows, statistics, topic_manifest
