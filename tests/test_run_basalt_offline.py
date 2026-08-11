#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_basalt_offline.py"
SPEC = importlib.util.spec_from_file_location("run_basalt_offline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunBasaltOfflineTests(unittest.TestCase):
    def test_session_output_name(self):
        session = Path("/tmp/Orbbec_Ego_AZER764008C_20260805_171119")
        self.assertEqual(
            MODULE.session_output_name(session), "recording_20260805_171119"
        )

    def test_fallback_session_output_name(self):
        self.assertEqual(MODULE.session_output_name(Path("/tmp/example")), "example")

    def test_default_hand_pose(self):
        project = Path("/project")
        session = Path("/data/Orbbec_Ego_AZER764008C_20260805_171119")
        self.assertEqual(
            MODULE.default_hand_pose(project, session),
            project / "output" / "recording_20260805_171119"
            / "mano_overlay_trajectory_tuned" / "hand_end_effector_6d.csv",
        )

    def test_default_hand_pose_falls_back_to_legacy_result(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            session = Path("/data/Orbbec_Ego_AZER764008C_20260805_171119")
            legacy = (
                project / "output" / "recording_20260805_171119"
                / "mano_overlay_trajectory" / "hand_end_effector_6d.csv"
            )
            legacy.parent.mkdir(parents=True)
            legacy.touch()
            self.assertEqual(MODULE.default_hand_pose(project, session), legacy)


if __name__ == "__main__":
    unittest.main()
