from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_DIR / "scripts" / "run_offline.sh"


class RunOfflineScriptTest(unittest.TestCase):
    def run_script(self, *arguments: str, env: dict[str, str] | None = None):
        return subprocess.run(
            ["bash", str(RUNNER), *arguments],
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def make_assets(self, root: Path) -> dict[str, str]:
        model = root / "models" / "hand_landmarker.task"
        mano_models = root / "models" / "mano"
        mano_source = root / "third_party" / "MANO"
        model.parent.mkdir(parents=True)
        mano_models.mkdir(parents=True)
        (mano_source / "mano").mkdir(parents=True)
        model.touch()
        (mano_models / "MANO_LEFT.pkl").touch()
        (mano_models / "MANO_RIGHT.pkl").touch()
        (mano_source / "mano" / "model.py").touch()
        return {
            "EGO_MODEL": str(model),
            "EGO_MANO_MODELS": str(mano_models),
            "EGO_MANO_SOURCE": str(mano_source),
        }

    def base_env(self, root: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.make_assets(root))
        env["EGO_OUTPUT"] = str(root / "output")
        env["EGO_PYTHON"] = sys.executable
        return env

    def test_help_lists_both_sources(self):
        result = self.run_script("--help", env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EGO_SOURCE=orbbec|gen", result.stdout)
        self.assertIn("EGO_SESSION", result.stdout)
        self.assertIn("EGO_MCAP", result.stdout)
        self.assertIn("EGO_HAND_ROUTE=mediapipe|wilor|parallel", result.stdout)

    def test_orbbec_check_accepts_complete_session_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "recordings" / "Orbbec_Ego_TEST"
            session.mkdir(parents=True)
            for suffix in (
                "camera_left.mp4",
                "camera_right.mp4",
                "camera_left_pts.csv",
                "camera_right_pts.csv",
                "calibration_camera.yaml",
            ):
                (session / f"Orbbec_Ego_TEST_{suffix}").touch()
            env = self.base_env(root)
            env.update({"EGO_SOURCE": "orbbec", "EGO_SESSION": str(session)})
            result = self.run_script("check", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("configuration and required inputs are ready", result.stdout)

    def test_gen_check_accepts_mcap_and_camera_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mcap = root / "recording.mcap"
            mcap.touch()
            fake_python = root / "fake-python"
            fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)
            env = self.base_env(root)
            env.update({
                "EGO_SOURCE": "gen",
                "EGO_MCAP": str(mcap),
                "EGO_LEFT_CAMERA": "camera2",
                "EGO_RIGHT_CAMERA": "camera3",
                "EGO_PYTHON": str(fake_python),
            })
            result = self.run_script("check", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GEN cameras: camera2/camera3", result.stdout)

    def test_rejects_unknown_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = os.environ.copy()
            env.update({
                "EGO_SOURCE": "unknown",
                "EGO_OUTPUT": str(Path(temporary) / "output"),
            })
            result = self.run_script("check", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("set EGO_SOURCE", result.stderr)

    def test_rejects_unknown_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mcap = root / "recording.mcap"
            mcap.touch()
            env = os.environ.copy()
            env.update({
                "EGO_SOURCE": "gen",
                "EGO_MCAP": str(mcap),
                "EGO_OUTPUT": str(root / "output"),
                "EGO_HAND_ROUTE": "bad",
            })
            result = self.run_script("check", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("EGO_HAND_ROUTE", result.stderr)


if __name__ == "__main__":
    unittest.main()
