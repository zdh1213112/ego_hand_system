from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sam3_glove_assist import (  # noqa: E402
    _annotate_record,
    _mask_guided_2d_refinement,
)


class Sam3GloveAssistTests(unittest.TestCase):
    def test_mask_guided_refinement_protects_wrist_and_pulls_nearby_joint(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        joints = np.tile(np.asarray([50.0, 50.0]), (21, 1))
        joints[0] = [10.0, 10.0]  # wrist/cuff is intentionally protected
        joints[8] = [20.0, 50.0]  # close enough to the mask to be corrected
        refined, info = _mask_guided_2d_refinement(mask, joints, 8, 24.0, 0.75)
        np.testing.assert_allclose(refined[0], joints[0])
        self.assertNotEqual(refined[8].tolist(), joints[8].tolist())
        self.assertEqual(info["corrected_joint_count"], 1)
        self.assertTrue(info["corrected_joint_flags"][8])

    def test_mask_guided_refinement_does_not_extrapolate_far_point(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        joints = np.tile(np.asarray([50.0, 50.0]), (21, 1))
        joints[8] = [1.0, 1.0]
        refined, info = _mask_guided_2d_refinement(mask, joints, 8, 24.0, 0.75)
        np.testing.assert_allclose(refined, joints)
        self.assertEqual(info["corrected_joint_count"], 0)

    def test_annotated_record_is_json_friendly_and_retains_raw_points(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        joints = np.tile(np.asarray([50.0, 50.0]), (21, 1))
        joints[8] = [20.0, 50.0]
        hand = {
            "detection_index": 0,
            "bbox_xyxy": [10, 10, 80, 80],
            "confidence": 0.9,
            "detector_is_right": 0,
            "is_right": 0,
            "joints_2d": joints.tolist(),
        }
        args = argparse.Namespace(
            mask_margin_px=8,
            min_score=0.25,
            min_box_iou=0.2,
            min_joint_support=0.55,
            refine_2d=1,
            max_2d_correction_px=24.0,
            two_d_correction_blend=0.75,
            max_2d_bone_change_ratio=0.55,
        )
        output, stats = _annotate_record(
            {"sync_index": 0, "hands": [hand]},
            [mask], [[10.0, 10.0, 80.0, 80.0]], [0.9], mask.shape, args,
        )
        self.assertEqual(stats["corrected_joints"], 1)
        result = output["hands"][0]
        self.assertEqual(result["wilor_joints_2d"], hand["joints_2d"])
        self.assertNotEqual(result["joints_2d"], hand["joints_2d"])
        self.assertTrue(result["sam3_assist"]["accepted"])
        # Numpy scalar booleans must not leak into JSONL output.
        import json
        json.dumps(output)


if __name__ == "__main__":
    unittest.main()
