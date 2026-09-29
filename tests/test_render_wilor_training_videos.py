from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_wilor_training_videos import _group_rows, _safe_token, _video_name


class RenderWilorTrainingVideosTests(unittest.TestCase):
    def test_merged_rows_group_by_source_camera_and_sync(self):
        rows = [
            {"source_dataset_id": 0, "source_experiment": "recording_a", "camera": "camera2", "sync_index": 3, "side": 0},
            {"source_dataset_id": 0, "source_experiment": "recording_a", "camera": "camera2", "sync_index": 3, "side": 1},
            {"source_dataset_id": 1, "source_experiment": "recording_b", "camera": "camera2", "sync_index": 3, "side": 0},
        ]
        grouped = _group_rows(rows, Path("/tmp/merged/dataset"))
        self.assertEqual(
            set(grouped),
            {(0, "recording_a", "camera2"), (1, "recording_b", "camera2")},
        )
        self.assertEqual(len(grouped[(0, "recording_a", "camera2")][3]), 2)

    def test_single_dataset_uses_parent_as_source_name(self):
        rows = [{"camera": "camera2", "sync_index": 8, "side": 0}]
        grouped = _group_rows(rows, Path("/output/recording_x/dataset"))
        self.assertEqual(set(grouped), {(None, "recording_x", "camera2")})

    def test_video_names_are_safe_and_source_qualified(self):
        self.assertEqual(_safe_token("recording / a"), "recording_a")
        self.assertEqual(
            _video_name(7, "recording / a", "camera2"),
            "source0007_recording_a_camera2_labels.mp4",
        )


if __name__ == "__main__":
    unittest.main()
