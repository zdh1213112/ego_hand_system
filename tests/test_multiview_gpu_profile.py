from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_multiview_wilor_experiment.sh"
INFERENCE = ROOT / "scripts" / "wilor_multiview_inference.py"


class MultiviewGpuProfileTests(unittest.TestCase):
    def test_orbbec_defaults_and_resume_with_legacy_unused_camera_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = root / "detector.pt"
            detector.touch()
            session = root / "session"
            session.mkdir()
            mcap = root / "recording.mcap"
            mcap.touch()
            # Run the real configuration/resume logic in the current test
            # interpreter, without requiring a locally named Conda environment.
            bin_dir = root / "bin"
            bin_dir.mkdir()
            conda = bin_dir / "conda"
            conda.write_text('#!/bin/sh\nshift 4\nexec "$@"\n')
            conda.chmod(0o755)
            env = dict(os.environ, PATH=f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}")
            for option, source in (("--session", session), ("--mcap", mcap)):
                with self.subTest(input=option):
                    output = root / (option.removeprefix("--") + "-output")
                    for folder, marker in (("normalized_multiview", "manifest.json"), ("wilor_multiview", "summary.json")):
                        stage = output / folder
                        stage.mkdir(parents=True)
                        (stage / marker).write_text("{}\n")
                    command = [str(RUNNER), option, str(source), "--output", str(output),
                               "--detector", str(detector), "--stop-after-inference"]
                    result = subprocess.run(command, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    path = output / "run_config.json"
                    config = json.loads(path.read_text())
                    self.assertEqual(config["cameras"], [f"camera{i}" for i in range(4)])
                    self.assertEqual(config["reference_camera"], "camera0")
                    self.assertEqual(config["anchor_cameras"], ["camera0", "camera1"])
                    config["camera_confidences"].update(camera4=0.1, camera5=0.1)
                    path.write_text(json.dumps(config))
                    resumed = subprocess.run(command, env=env, capture_output=True, text=True)
                    self.assertEqual(resumed.returncode, 0, resumed.stderr)
                    self.assertIn("reuse WiLoR predictions", resumed.stdout)

    def test_runner_exposes_both_gpu_profiles(self):
        result = subprocess.run(
            [str(RUNNER), "--help"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertIn("--gpu-profile compatible|rtx5090d", result.stdout)
        self.assertIn("--frame-batch-size N", result.stdout)
        self.assertIn("--preprocess-workers N", result.stdout)
        self.assertIn("--max-detections-per-class N", result.stdout)
        self.assertIn("--compile-backbone 0|1", result.stdout)
        self.assertIn("--fusion-workers N", result.stdout)
        self.assertIn("--sam3-refine-2d 0|1", result.stdout)
        self.assertIn("--sam3-max-2d-correction-px N", result.stdout)
        self.assertIn("--sam3-2d-correction-blend N", result.stdout)

    def test_compatible_defaults_preserve_original_path(self):
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('GPU_PROFILE="compatible"', runner)
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-4}"', runner)
        self.assertIn('FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-1}"', runner)
        self.assertIn('PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-1}"', runner)
        self.assertIn(
            'MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-0}"', runner
        )
        self.assertIn('COMPILE_BACKBONE="${COMPILE_BACKBONE:-0}"', runner)
        self.assertIn('FUSION_WORKERS="${FUSION_WORKERS:-1}"', runner)
        self.assertIn('SAM3_REFINE_2D="${EGO_SAM3_REFINE_2D:-0}"', runner)

    def test_5090d_defaults_enable_cross_frame_amp_path(self):
        runner = RUNNER.read_text(encoding="utf-8")
        inference = INFERENCE.read_text(encoding="utf-8")
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-64}"', runner)
        self.assertIn('FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-16}"', runner)
        self.assertIn('PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"', runner)
        self.assertIn(
            'MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-1}"', runner
        )
        self.assertIn('COMPILE_BACKBONE="${COMPILE_BACKBONE:-1}"', runner)
        self.assertIn('FUSION_WORKERS="${FUSION_WORKERS:-8}"', runner)
        self.assertIn('args.gpu_profile == "rtx5090d"', inference)
        self.assertIn("infer_dual_batched", inference)
        self.assertIn("torch.autocast", inference)
        self.assertIn("ThreadPoolExecutor", inference)
        self.assertIn("cv2.GaussianBlur", inference)
        self.assertIn("torch.compile", inference)
        self.assertIn("_prefetched_frame_chunks", inference)
        self.assertIn("pin_memory", inference)


if __name__ == "__main__":
    unittest.main()
