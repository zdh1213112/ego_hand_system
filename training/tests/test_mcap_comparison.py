import copy
import json
from pathlib import Path
import tempfile
import unittest

from visualize_mcap_comparison import load_run, validate_alignment, validate_run_settings


class ComparisonAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.records = [{
            "frame_index": 0, "timestamp_ns": 1234,
            "hands": [{"is_right": 1, "confidence": 0.9, "bbox_xyxy": [10, 20, 100, 200]}],
        }]

    def test_accepts_identical_inputs_with_different_predictions(self):
        other = copy.deepcopy(self.records)
        other[0]["hands"][0]["joints_3d"] = "different pose"
        validate_alignment(self.records, other)

    def test_loads_predictions_after_run_directory_is_moved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            moved = root / "moved"
            original.mkdir()
            moved.mkdir()
            old_path = original / "predictions.jsonl"
            old_path.write_text(json.dumps([]), encoding="utf-8")
            local_path = moved / old_path.name
            local_path.write_text(json.dumps(self.records[0]) + "\n", encoding="utf-8")
            (moved / "run_report.json").write_text(
                json.dumps({"predictions": str(old_path)}), encoding="utf-8"
            )
            report, records = load_run(moved)
            self.assertEqual(records, self.records)
            self.assertEqual(report["predictions"], str(local_path.resolve()))
            old_path.unlink()
            self.assertEqual(load_run(moved)[1], self.records)

    def test_rejects_misaligned_frames_and_detections(self):
        changes = [
            lambda r: r[0].update(timestamp_ns=1235),
            lambda r: r[0].update(frame_index=1),
            lambda r: r[0].update(hands=[]),
            lambda r: r[0]["hands"][0].update(is_right=0),
            lambda r: r[0]["hands"][0].update(bbox_xyxy=[10, 20, 101, 200]),
        ]
        for change in changes:
            with self.subTest(change=change):
                other = copy.deepcopy(self.records)
                change(other)
                with self.assertRaises(ValueError):
                    validate_alignment(self.records, other)

    def test_rejects_duplicate_frames(self):
        records = self.records * 2
        with self.assertRaises(ValueError):
            validate_alignment(records, records)

    def test_legacy_reports_allow_missing_options_but_reject_known_mismatches(self):
        base = {"detector": "detector.pt", "checkpoint": "base.ckpt"}
        fine = {**base, "confidence": 0.3, "rescale_factor": 2.0}
        validate_run_settings(base, fine)
        with self.assertRaises(ValueError):
            validate_run_settings({**base, "confidence": 0.5}, fine)
        with self.assertRaises(ValueError):
            validate_run_settings(base, {**fine, "detector": "other.pt"})


if __name__ == "__main__":
    unittest.main()
