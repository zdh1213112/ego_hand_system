import importlib.util
from pathlib import Path
import sys
import time
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ego_live_stereo.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("ego_live_stereo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def match(px, xyz, label="Left"):
    return {
        "center_left_px": np.asarray(px, dtype=np.float64),
        "center": np.asarray(xyz, dtype=np.float64),
        "left": {"label": label, "score": 0.95},
        "right": {"label": label, "score": 0.94},
    }


class DepthAwareTrackTests(unittest.TestCase):
    def test_latest_packet_reader_discards_older_packets(self):
        class FakeBridge:
            def __init__(self):
                self.index = 0

            def read(self, decode=True):
                self.index += 1
                if self.index > 3:
                    raise EOFError("done")
                return {"index": self.index}

            @staticmethod
            def decode(packet):
                return dict(packet)

            def close(self):
                pass

        reader = MODULE.LatestPacketReader(FakeBridge())
        deadline = time.monotonic() + 1.0
        while not reader.stopped and time.monotonic() < deadline:
            time.sleep(0.001)
        sequence, packet = reader.get()
        self.assertEqual(sequence, 3)
        self.assertEqual(packet["index"], 3)
        reader.close()

    def test_two_packet_queue_keeps_only_two_newest(self):
        class FakeBridge:
            def __init__(self):
                self.index = 0

            def read(self, decode=True):
                self.index += 1
                if self.index > 3:
                    raise EOFError("done")
                return {"index": self.index}

            def close(self):
                pass

        reader = MODULE.LatestPacketReader(FakeBridge(), capacity=2)
        deadline = time.monotonic() + 1.0
        while not reader.stopped and time.monotonic() < deadline:
            time.sleep(0.001)
        first_sequence, _ = reader.get()
        second_sequence, _ = reader.get()
        self.assertEqual((first_sequence, second_sequence), (2, 3))
        self.assertEqual(reader.dropped, 1)
        reader.close()

    def test_depth_keeps_ids_when_hands_cross_in_2d(self):
        tracker = MODULE.DepthAwareTrackManager()
        first = [
            match((100, 100), (-0.05, 0.0, 0.20), "Right"),
            match((300, 100), (0.05, 0.0, 0.50), "Left"),
        ]
        tracker.assign(first, 0.0)
        self.assertEqual([item["track_id"] for item in first], [0, 1])

        crossed = [
            match((280, 100), (-0.04, 0.0, 0.21), "Right"),
            match((120, 100), (0.04, 0.0, 0.49), "Left"),
        ]
        tracker.assign(crossed, 1 / 30)
        self.assertEqual([item["track_id"] for item in crossed], [0, 1])

    def test_one_difficult_hand_does_not_replace_both_ids(self):
        tracker = MODULE.DepthAwareTrackManager()
        first = [
            match((100, 100), (-0.05, 0.0, 0.20), "Right"),
            match((300, 100), (0.05, 0.0, 0.50), "Left"),
        ]
        tracker.assign(first, 0.0)
        second = [
            match((105, 102), (-0.049, 0.0, 0.201), "Right"),
            match((900, 700), (0.8, 0.8, 1.2), "Left"),
        ]
        tracker.assign(second, 1 / 30)
        self.assertEqual(second[0]["track_id"], 0)
        self.assertEqual(second[1]["track_id"], 1)
        self.assertEqual(set(tracker.tracks), {0, 1})

    def test_track_slots_are_reused_after_long_absence(self):
        tracker = MODULE.DepthAwareTrackManager(max_missed=5)
        first = [match((100, 100), (-0.05, 0.0, 0.20), "Right")]
        tracker.assign(first, 0.0)
        for frame in range(20):
            tracker.assign([], (frame + 1) / 30)
        returned = [match((500, 300), (0.10, 0.0, 0.35), "Right")]
        tracker.assign(returned, 1.0)
        self.assertEqual(returned[0]["track_id"], 0)
        self.assertEqual(set(tracker.tracks), {0})
        np.testing.assert_allclose(tracker.tracks[0]["velocity_px"], 0.0)
        np.testing.assert_allclose(tracker.tracks[0]["velocity_3d"], 0.0)

    def test_depth_spike_is_predicted_instead_of_accepted(self):
        online = MODULE.OnlineStereoFilter(prediction_frames=5)
        baseline = {
            **match((100, 100), (0.0, 0.0, 0.2)),
            "track_id": 0,
            "points_left": np.tile((0.0, 0.0, 0.2), (21, 1)).astype(np.float64),
            "valid": np.ones(21, dtype=bool),
            "epipolar": np.zeros(21),
            "reprojection": np.zeros(21),
            "disparity": np.full(21, 80.0),
        }
        online.update(baseline, 0.0)

        spike = {
            **match((100, 100), (0.0, 0.0, 1.0)),
            "track_id": 0,
            "points_left": np.tile((0.0, 0.0, 1.0), (21, 1)).astype(np.float64),
            "valid": np.ones(21, dtype=bool),
            "epipolar": np.zeros(21),
            "reprojection": np.zeros(21),
            "disparity": np.full(21, 80.0),
        }
        online.update(spike, 1 / 30)
        self.assertTrue(np.all(spike["predicted_3d"]))
        np.testing.assert_allclose(spike["filtered_points_left"][:, 2], 0.2)

    def test_online_bone_targets_are_learned(self):
        online = MODULE.OnlineStereoFilter(prediction_frames=5)
        points = np.zeros((21, 3), dtype=np.float64)
        for joint in range(1, 21):
            points[joint] = (0.01 * joint, 0.0, 0.25)
        last = None
        for frame in range(7):
            current = {
                **match((100, 100), (0.0, 0.0, 0.25)),
                "track_id": 0,
                "points_left": points.copy(),
                "valid": np.ones(21, dtype=bool),
                "epipolar": np.zeros(21),
                "reprojection": np.zeros(21),
                "disparity": np.full(21, 80.0),
            }
            online.update(current, frame / 30)
            last = current
        self.assertIsNotNone(last)
        self.assertGreaterEqual(np.count_nonzero(np.isfinite(last["bone_targets_m"])), 15)

    def test_binary_header_matches_cpp_contract(self):
        self.assertEqual(MODULE.PACKET_HEADER.size, 64)


if __name__ == "__main__":
    unittest.main()
