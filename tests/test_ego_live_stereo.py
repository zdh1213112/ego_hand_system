import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ego_live_stereo.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("ego_live_stereo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
import mediapipe_stereo_triangulate as STEREO


def match(px, xyz, label="Left"):
    return {
        "center_left_px": np.asarray(px, dtype=np.float64),
        "center": np.asarray(xyz, dtype=np.float64),
        "left": {"label": label, "score": 0.95},
        "right": {"label": label, "score": 0.94},
    }


class DepthAwareTrackTests(unittest.TestCase):
    @staticmethod
    def refinement_args():
        return SimpleNamespace(
            stereo_refine=True,
            refine_window=17,
            refine_tip_window=25,
            refine_max_x_shift_px=18.0,
            refine_max_y_residual_px=3.0,
            refine_max_fb_error_px=2.0,
            refine_max_lk_error=32.0,
        )

    def test_sparse_stereo_refinement_recovers_subpixel_correspondence(self):
        rng = np.random.default_rng(7)
        left = rng.integers(0, 256, size=(140, 220), dtype=np.uint8)
        left = MODULE.cv2.GaussianBlur(left, (5, 5), 0.8)
        disparity = 14
        right = np.zeros_like(left)
        right[:, :-disparity] = left[:, disparity:]
        left_points = np.asarray([
            (65 + (joint % 5) * 26, 40 + (joint // 5) * 20)
            for joint in range(21)
        ], dtype=np.float64)
        right_points = left_points.copy()
        right_points[:, 0] -= disparity - 1.7
        right_points[:, 1] += 1.2
        result = STEREO.refine_stereo_correspondences(
            left_points, right_points, (left, right), self.refinement_args()
        )
        self.assertGreaterEqual(np.count_nonzero(result["accepted"]), 18)
        used = result["accepted"]
        expected_x = left_points[:, 0] - disparity
        self.assertLess(np.median(np.abs(result["right_points"][used, 0] - expected_x[used])), 0.45)
        np.testing.assert_allclose(
            result["left_points"][used, 1], result["right_points"][used, 1], atol=1e-6
        )

    def test_sparse_stereo_refinement_rejects_textureless_regions(self):
        image = np.full((120, 180), 127, dtype=np.uint8)
        left_points = np.tile((90.0, 60.0), (21, 1))
        right_points = np.tile((75.0, 61.0), (21, 1))
        result = STEREO.refine_stereo_correspondences(
            left_points, right_points, (image, image), self.refinement_args()
        )
        self.assertEqual(np.count_nonzero(result["accepted"]), 0)

    def test_failed_tip_refinement_reduces_depth_quality(self):
        base = {
            **match((100, 100), (0.0, 0.0, 0.25)),
            "valid": np.ones(21, dtype=bool),
            "epipolar": np.zeros(21),
            "reprojection": np.zeros(21),
            "disparity": np.full(21, 60.0),
            "refinement_attempted": np.ones(21, dtype=bool),
            "refinement_used": np.ones(21, dtype=bool),
            "refinement_quality": np.ones(21),
        }
        good = MODULE.OnlineStereoFilter.quality(base)
        failed = dict(base)
        failed["refinement_used"] = base["refinement_used"].copy()
        failed["refinement_quality"] = base["refinement_quality"].copy()
        failed["refinement_used"][8] = False
        failed["refinement_quality"][8] = 0.0
        degraded = MODULE.OnlineStereoFilter.quality(failed)
        self.assertLess(degraded[8], good[8] * 0.5)
        self.assertAlmostEqual(degraded[9], good[9])

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


class OfflineStereoTrackTests(unittest.TestCase):
    @staticmethod
    def offline_match(px, xyz, label="Left"):
        value = match(px, xyz, label)
        pixels = np.tile(np.asarray(px, dtype=np.float64), (21, 1))
        value["left"]["pixels"] = pixels.copy()
        value["right"]["pixels"] = pixels.copy()
        return value

    def test_reacquires_identity_after_long_out_of_view_gap(self):
        tracker = STEREO.TrackManager(
            max_missed=75,
            max_distance_px=280.0,
            reacquire_distance_px=700.0,
            max_tracks=2,
        )
        first = [
            self.offline_match((120, 180), (-0.05, 0.0, 0.30), "Left"),
            self.offline_match((800, 180), (0.05, 0.0, 0.30), "Right"),
        ]
        tracker.assign(first)
        self.assertEqual([item["track_id"] for item in first], [0, 1])

        for frame in range(45):
            visible_left = [self.offline_match(
                (120 + frame, 180), (-0.05, 0.0, 0.30), "Left"
            )]
            tracker.assign(visible_left)

        reappeared = [
            self.offline_match((166, 180), (-0.05, 0.0, 0.30), "Left"),
            self.offline_match((1320, 420), (0.15, 0.0, 0.24), "Right"),
        ]
        tracker.assign(reappeared)

        self.assertEqual([item["track_id"] for item in reappeared], [0, 1])
        self.assertEqual(set(tracker.tracks), {0, 1})
        self.assertEqual(tracker.tracks[1]["label"], "Right")
        np.testing.assert_allclose(tracker.tracks[1]["velocity"], 0.0)

if __name__ == "__main__":
    unittest.main()
