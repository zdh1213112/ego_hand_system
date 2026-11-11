"""Wire contract for the four-camera Orbbec bridge."""

import io
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_four_bridge import CAMERAS, DecodedFourReader, FourBridge, HEADER, IMU_SAMPLE, merge_imu, read_exact


class FourBridgeProtocolTest(unittest.TestCase):
    def test_reads_four_frames_in_calibration_order(self):
        images = [b"first", b"second", b"third", b"fourth"]
        entries = [value for i, data in enumerate(images) for value in (len(data), 1000 + i, 20 + i)]
        header = HEADER.pack(b"EGO4", 2, HEADER.size, 4, 1920, 1200, 123456, *entries)
        fake = FourBridge.__new__(FourBridge)
        fake.process = type("Process", (), {"stdout": io.BytesIO(header + b"".join(images))})()
        packet = fake.read()
        self.assertEqual(packet["jpeg"], dict(zip(CAMERAS, images)))
        self.assertEqual(packet["index"]["cam_3"], 23)
        self.assertEqual(packet["timestamp_us"]["cam_2"], 1002)
        self.assertEqual(packet["host_ns"], 123456)

    def test_rejects_corrupt_header(self):
        header = HEADER.pack(b"EGO1", 2, HEADER.size, 4, 1920, 1200, 0, *([1, 0, 0] * 4))
        fake = FourBridge.__new__(FourBridge)
        fake.process = type("Process", (), {"stdout": io.BytesIO(header + b"1234")})()
        with self.assertRaisesRegex(ValueError, "invalid EGO4 bridge header"):
            fake.read()

    def test_reads_v3_imu_after_four_images(self):
        images = [b"first", b"second", b"third", b"fourth"]
        entries = [value for i, data in enumerate(images) for value in (len(data), 1000 + i, 20 + i)]
        header = HEADER.pack(b"EGO4", 3, HEADER.size + 4, 4, 1920, 1200, 123456, *entries)
        samples = IMU_SAMPLE.pack(0, 1010, 1., 2., 3.) + IMU_SAMPLE.pack(1, 1011, 4., 5., 6.)
        fake = FourBridge.__new__(FourBridge)
        fake.process = type("Process", (), {"stdout": io.BytesIO(header + (2).to_bytes(4, "little")
                                                              + b"".join(images) + samples)})()
        packet = fake.read()
        self.assertEqual(packet["imu"]["accel"], [{"timestamp_us": 1010, "xyz": [1., 2., 3.]}])
        self.assertEqual(packet["imu"]["gyro"], [{"timestamp_us": 1011, "xyz": [4., 5., 6.]}])
        newer = {"imu": {"accel": [{"timestamp_us": 1030, "xyz": [0., 0., 0.]}]}}
        merge_imu(packet, newer)
        self.assertEqual([x["timestamp_us"] for x in newer["imu"]["accel"]], [1010, 1030])

    def test_short_read_fails(self):
        with self.assertRaises(EOFError):
            read_exact(io.BytesIO(b"x"), 2)

    def test_background_decode_preserves_sequence(self):
        image = np.zeros((16, 24, 3), np.uint8)
        encoded = cv2.imencode(".jpg", image)[1].tobytes()
        packet = {"jpeg": dict.fromkeys(CAMERAS, encoded), "width": 24, "height": 16}

        class Source:
            sequence = 7
            dropped = 0
            stopped = False

            def __init__(self):
                self.sent = False

            def get(self, timeout):
                if not self.sent:
                    self.sent = True
                    return 7, packet
                self.stopped = True
                return None

            def close(self):
                self.stopped = True

        reader = DecodedFourReader(Source())
        try:
            sequence, decoded_packet, frames = reader.get(timeout=2)
            self.assertEqual(sequence, 7)
            self.assertEqual(frames["cam_0"].shape, image.shape)
            self.assertIn("decode_main_ms", decoded_packet)
        finally:
            reader.close()


if __name__ == "__main__":
    unittest.main()
