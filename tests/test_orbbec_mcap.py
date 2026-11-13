from fractions import Fraction
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap_protobuf.writer import Writer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_hand_inference.normalize_orbbec_multiview_mcap import DEFAULT_TOPICS, normalize, synchronize_rows
from ego_hand_inference.ego_data.calibration import CameraCalibration


def make_mcap(path, missing_camera=None):
    proto = descriptor_pb2.FileDescriptorProto(
        name="test_image.proto", package="ego_test", syntax="proto3"
    )
    msg = proto.message_type.add(name="Image")
    msg.field.add(
        name="format", number=1, type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    )
    msg.field.add(
        name="data", number=2, type=descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    image_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("ego_test.Image")
    )
    codec = av.CodecContext.create("libx264", "w")
    codec.width = 64
    codec.height = 48
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "x264-params": "keyint=1:repeat-headers=1:bframes=0",
    }
    packets = []
    for i in range(3):
        frame = av.VideoFrame.from_ndarray(
            np.full((48, 64, 3), 30 + i, dtype=np.uint8), format="bgr24"
        )
        frame.pts = i
        packets.extend(codec.encode(frame))
    packets.extend(codec.encode(None))
    with path.open("wb") as f, Writer(f) as writer:
        for i, packet in enumerate(packets):
            for index, (camera, topic) in enumerate(DEFAULT_TOPICS.items()):
                if camera == missing_camera:
                    continue
                timestamp = 1_000_000_000 + i * 33_333_333 + index * 1000
                writer.write_message(
                    topic,
                    image_type(format="h264", data=bytes(packet)),
                    log_time=timestamp,
                    publish_time=timestamp,
                )


def test_real_h264_mcap_remux_synchronizes_four_cameras_and_preserves_calibration(
    tmp_path,
):
    source = tmp_path / "recording.mcap"
    make_mcap(source)
    cal = tmp_path / "calibration"
    cal.mkdir()
    for i, camera in enumerate(DEFAULT_TOPICS):
        transform = np.eye(4)
        transform[0, 3] = i * 0.03
        CameraCalibration(
            camera,
            camera,
            "KB",
            (64, 48),
            [[40, 0, 32], [0, 40, 24], [0, 0, 1]],
            [0] * 4,
            transform,
        ).save(cal / f"{camera}.json")
    args = SimpleNamespace(
        input=source,
        output=tmp_path / "out",
        cameras=list(DEFAULT_TOPICS),
        camera_topic=[],
        reference_camera="camera0",
        max_delta_us=1500,
        max_frames=0,
        width=64,
        height=48,
        calibration_dir=cal,
        allow_truncated=False,
    )
    out = normalize(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        manifest["synchronization"]["frame_count"] == 3
        and manifest["calibration"]["available"]
    )
    for camera in DEFAULT_TOPICS:
        with av.open(str(out / "cameras" / camera / "video.mkv")) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 3 and (frames[0].width, frames[0].height) == (64, 48)
        assert (
            json.loads((out / "calibration" / f"{camera}.json").read_text())[
                "camera_id"
            ]
            == camera
        )
    broken = tmp_path / "missing.mcap"
    make_mcap(broken, missing_camera="camera3")
    args.input = broken
    args.output = tmp_path / "missing-output"
    with pytest.raises(RuntimeError, match="camera3"):
        normalize(args)
    assert not args.output.exists()


def test_sync_never_reuses_frames_or_accepts_outside_tolerance():
    values = {
        "camera0": [{"timestamp_ns": 10}, {"timestamp_ns": 11}, {"timestamp_ns": 100}],
        "camera1": [{"timestamp_ns": 10}, {"timestamp_ns": 101}],
    }
    rows = synchronize_rows(values, tuple(values), "camera0", 2)
    assert [r["camera1_frame_index"] for r in rows] == [0, 1]
    assert [r["camera0_frame_index"] for r in rows] == [0, 2]
