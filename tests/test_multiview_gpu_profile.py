from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_multiview_wilor_experiment.sh"
INFERENCE = ROOT / "scripts" / "wilor_multiview_inference.py"


class MultiviewGpuProfileTests(unittest.TestCase):
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

    def test_5090d_defaults_enable_cross_frame_amp_path(self):
        runner = RUNNER.read_text(encoding="utf-8")
        inference = INFERENCE.read_text(encoding="utf-8")
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-16}"', runner)
        self.assertIn('FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-4}"', runner)
        self.assertIn('PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"', runner)
        self.assertIn(
            'MAX_DETECTIONS_PER_CLASS="${MAX_DETECTIONS_PER_CLASS:-1}"', runner
        )
        self.assertIn('COMPILE_BACKBONE="${COMPILE_BACKBONE:-1}"', runner)
        self.assertIn('args.gpu_profile == "rtx5090d"', inference)
        self.assertIn("infer_dual_batched", inference)
        self.assertIn("torch.autocast", inference)
        self.assertIn("ThreadPoolExecutor", inference)
        self.assertIn("cv2.GaussianBlur", inference)
        self.assertIn("torch.compile", inference)


if __name__ == "__main__":
    unittest.main()
