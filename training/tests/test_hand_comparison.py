import json
from pathlib import Path
import tempfile
import unittest

import av
import cv2
import numpy as np

from visualize_hand_comparison import load_comparisons, matched_pair, parse_args, process_comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows, self.records = [], []
        for index, (side, frame) in enumerate(((0, 1), (1, 1), (1, 11))):
            row = {"index": index, "side": side, "sync_index": frame, "source_frame_index": frame,
                   "camera": "camera2", "source_dataset_id": 7, "image": "rgb.jpg", "label": f"{index}.npy"}
            physical = np.zeros((21, 3))
            physical[:, 0] = np.linspace(0, 0.1, 21)
            physical[:, 1] = np.linspace(0, -0.1, 21)
            physical += [0.02, 0.03, 0.04]
            np.save(self.root / row["label"], {"joints_3d": physical, "trans": [side * 0.2, 0.1, 0.6], "side": side})
            gt = physical - physical[:1]
            if side == 0:
                gt[:, 0] *= -1
            pred = gt.copy()
            pred[1:, 0] += 0.01
            record = {key: row[key] for key in ("side", "camera", "sync_index", "source_frame_index")}
            record.update(sample_id=str(index), source_group="source_dataset_id:7",
                          gt_joints_3d_root_relative=gt.tolist(), pred_joints_3d_root_relative=pred.tolist(),
                          pred_cam_t=[0, 0, 99], mpjpe_mm=999)
            self.rows.append(row)
            self.records.append(record)
        cv2.imwrite(str(self.root / "rgb.jpg"), np.full((100, 180, 3), [30, 80, 170], np.uint8))
        (self.root / "index.jsonl").write_text("\n".join(map(json.dumps, reversed(self.rows))))
        self.predictions = self.root / "predictions.jsonl"
        self.save_records()

    def save_records(self):
        self.predictions.write_text("\n".join(map(json.dumps, reversed(self.records))))

    def test_left_mirror_restored_and_predicted_camera_not_used(self):
        pair = matched_pair(self.root, self.rows[0], self.records[0])
        np.testing.assert_allclose(pair.gt[0], [0.02, 0.13, 0.64])
        np.testing.assert_allclose(pair.pred[0], pair.gt[0])
        np.testing.assert_allclose(pair.pred[1:] - pair.gt[1:], np.tile([-0.01, 0, 0], (20, 1)), atol=1e-10)
        self.assertAlmostEqual(pair.mpjpe_mm, 10 * 20 / 21)

    def test_saved_gt_and_identity_must_match(self):
        self.records[0]["gt_joints_3d_root_relative"][5][2] += 0.1
        with self.assertRaisesRegex(ValueError, "saved GT does not match"):
            matched_pair(self.root, self.rows[0], self.records[0])
        self.records[1]["camera"] = "camera3"
        with self.assertRaisesRegex(ValueError, "mismatched camera"):
            matched_pair(self.root, self.rows[1], self.records[1])

    def test_matching_is_by_sample_id_and_groups_both_hands(self):
        comparisons = load_comparisons(self.root, self.predictions)
        self.assertEqual([frame.index for frame, _ in comparisons], [1, 11])
        self.assertEqual([len(pairs) for _, pairs in comparisons], [2, 1])
        self.records.append(dict(self.records[0]))
        self.save_records()
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            load_comparisons(self.root, self.predictions)

    def test_video_and_one_example_have_expected_format_and_count(self):
        args = parse_args(["--run-dir", str(self.root), "--dataset-root", str(self.root),
                           "--predictions", str(self.predictions), "--width", "960", "--height", "600"])
        report = process_comparison(args)
        self.assertEqual(report["frames"], 2)
        self.assertEqual(report["hands"], 3)
        self.assertAlmostEqual(report["mean_mpjpe_mm"], 10 * 20 / 21)
        self.assertFalse(report["prediction_camera_translation_used"])
        self.assertEqual(set(report["example"]["sample_ids"]), {"0", "1"})
        self.assertEqual(cv2.imread(report["example"]["path"]).shape, (600, 960, 3))
        self.assertEqual(len(list(args.output_dir.glob("*.png"))), 1)
        self.assertFalse(list(args.output_dir.glob("*.partial.*")))
        with av.open(report["output"]) as video:
            stream = video.streams.video[0]
            self.assertEqual(stream.codec_context.name, "h264")
            self.assertEqual(stream.codec_context.pix_fmt, "yuv420p")
            self.assertEqual(stream.average_rate, 10)
            self.assertEqual(sum(1 for _ in video.decode(stream)), 2)


if __name__ == "__main__":
    unittest.main()
