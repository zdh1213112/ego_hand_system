from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts" / "run_multiview_wilor_label_batch.sh"
SINGLE = ROOT / "scripts" / "run_multiview_wilor_label_export.sh"
FITTER = ROOT / "scripts" / "fit_mano_sequence.py"


class MultiviewLabelGpuProfileTests(unittest.TestCase):
    def test_label_runners_expose_5090_profile_and_overrides(self):
        for runner in (BATCH, SINGLE):
            result = subprocess.run(
                [str(runner), "--help"], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertIn("--gpu-profile", result.stdout)
            self.assertIn("compatible|rtx5090d", result.stdout)
            self.assertIn("--fit-pose-window", result.stdout)
            self.assertIn("--fit-pose-overlap", result.stdout)
            self.assertIn("--cuda-matmul-precision", result.stdout)
            self.assertIn("--check-batch-size", result.stdout)
            self.assertIn("--mano-replay-tolerance-m", result.stdout)

    def test_5090_profile_keeps_validated_fit_defaults(self):
        batch_source = BATCH.read_text(encoding="utf-8")
        self.assertIn('LABEL_WORKERS="${LABEL_WORKERS:-1}"', batch_source)
        self.assertIn("wait -n -p completed_pid", batch_source)
        for runner in (BATCH, SINGLE):
            source = runner.read_text(encoding="utf-8")
            self.assertIn('FIT_POSE_WINDOW="${FIT_POSE_WINDOW:-48}"', source)
            self.assertIn('FIT_POSE_OVERLAP="${FIT_POSE_OVERLAP:-16}"', source)
            self.assertIn(
                'CUDA_MATMUL_PRECISION="${CUDA_MATMUL_PRECISION:-highest}"', source
            )
            self.assertIn(
                'MANO_REPLAY_TOLERANCE="${MANO_REPLAY_TOLERANCE:-0.00001}"',
                source,
            )

    def test_batch_forwards_profile_to_single_export(self):
        source = BATCH.read_text(encoding="utf-8")
        self.assertIn('--gpu-profile "$GPU_PROFILE"', source)
        self.assertIn('--fit-pose-window "$FIT_POSE_WINDOW"', source)
        self.assertIn('--fit-pose-overlap "$FIT_POSE_OVERLAP"', source)
        self.assertIn('--cuda-matmul-precision "$CUDA_MATMUL_PRECISION"', source)
        self.assertIn('--check-batch-size "$CHECK_BATCH_SIZE"', source)
        self.assertIn('--mano-replay-tolerance-m "$MANO_REPLAY_TOLERANCE"', source)
        self.assertIn('row.get("status") == "success"', source)

    def test_fitter_configures_cuda_matmul_precision(self):
        source = FITTER.read_text(encoding="utf-8")
        self.assertIn("torch.set_float32_matmul_precision", source)
        self.assertIn("[mano-fit] CUDA device=", source)

    def test_parallel_batch_scheduler_collects_all_results(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            script_root = temporary / "scripts"
            script_root.mkdir()
            fake_single = script_root / "fake_single.sh"
            fake_single.write_text("#!/usr/bin/env bash\nsleep 0.05\n", encoding="utf-8")
            os.chmod(fake_single, 0o755)

            runner = script_root / BATCH.name
            source = BATCH.read_text(encoding="utf-8").replace(
                'SINGLE="$ROOT/scripts/run_multiview_wilor_label_export.sh"',
                f'SINGLE="{fake_single}"',
            )
            runner.write_text(source, encoding="utf-8")
            os.chmod(runner, 0o755)

            experiments = temporary / "experiments"
            for index in range(3):
                experiment = experiments / f"recording_{index}"
                (experiment / "normalized_multiview").mkdir(parents=True)
                (experiment / "normalized_multiview" / "manifest.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (experiment / "fusion_multiview").mkdir()
                (experiment / "fusion_multiview" / "accepted.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                (experiment / "fusion_multiview" / "summary.json").write_text(
                    "{}\n", encoding="utf-8"
                )

            output = temporary / "output"
            subprocess.run(
                [
                    str(runner), "--experiment-root", str(experiments),
                    "--output-root", str(output), "--layout", "separate",
                    "--gpu-profile", "rtx5090d", "--workers", "2",
                ],
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            summary = json.loads(
                (output / "label_batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["success"], 3)
            self.assertEqual(summary["failed"], 0)


if __name__ == "__main__":
    unittest.main()
