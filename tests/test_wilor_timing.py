from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wilor_timing import Timer, statistics
from profile_multiview_wilor import record_stage, write_report
from wilor_timing_bootstrap import instrument_source
from wilor_timing_report import build_averages

try:
    import torch
except ImportError:
    torch = None


class TimingTests(unittest.TestCase):
    def test_frame_average_and_forward_average_use_different_denominators(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            directory = root / "timing"
            directory.mkdir()
            (root / "wilor_multiview").mkdir()
            (root / "wilor_multiview/summary.json").write_text(json.dumps({
                "frame_count": 3, "camera_ids": ["camera0", "camera1"],
                "inference_seconds": 0.6,
                "cameras": {"camera0": {"frame_count": 3}, "camera1": {"frame_count": 3}},
            }))
            rows = []
            def add(name, batch, ms, status="ok", crops=4):
                rows.append(dict(name=name, camera="camera0", batch_index=batch,
                                 hypotheses=crops, warmup="True" if batch == 0 else "False",
                                 status=status, host_wall_ms=ms / 2, elapsed_ms=ms,
                                 cuda_ms=ms))
            for batch in range(3):
                add("model.total", batch, 60)
                for name, ms in [("vit.features", 30), ("mano.parameter_decoder", 5),
                                 ("mano.coarse", 2), ("refinenet.total", 15), ("mano.final", 3)]:
                    add(name, batch, ms)
            # A failed forward must not pollute averages, even if its ViT completed.
            add("model.total", 3, 900, status="error")
            add("vit.features", 3, 800)
            add("detector", "", 120)
            with (directory / "samples.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            stages = [{"stage": "fuse_multiview_wilor_guided", "wall_seconds": 0.3, "exit_code": 0}]
            result = build_averages(directory, stages, {"device": "cuda:0"}, [])
            frame_rows = {row["step"]: row for row in result["frame_averages"]}
            self.assertEqual(frame_rows["WiLoR 前向推理（分摊到每帧）"]["mean_ms"], 30)
            self.assertEqual(frame_rows["YOLO 手部检测"]["mean_ms"], 20)
            self.assertEqual(frame_rows["推理流程合计（实测，含解码、检测及后处理）"]["mean_ms"], 100)
            self.assertAlmostEqual(frame_rows["多视角融合与时序处理（摊销）"]["mean_ms"], 100)
            forward = result["wilor_forward"]
            self.assertEqual(forward["mean_ms"], 60)
            self.assertEqual(forward["mean_hand_crops_per_call"], 4)
            self.assertEqual(forward["successful_calls"], 3)
            self.assertEqual(forward["timing_method"], "CUDA Event")
            self.assertEqual(forward["modules"][0]["mean_ms"], 30)
            self.assertEqual(sum(row["mean_ms"] for row in forward["modules"][:-1]), 60)

    def test_bootstrap_compiles_current_adapter_without_writing_it(self):
        source = ROOT / "scripts/wilor_multiview_inference.py"
        before = source.read_bytes()
        code = instrument_source(before.decode(), str(source))
        self.assertIsNotNone(code)
        self.assertEqual(before, source.read_bytes())

    def test_statistics_interpolates_percentiles(self):
        stats = statistics([40, 10, 20, 30])
        self.assertEqual(stats["total_ms"], 100)
        self.assertEqual(stats["p50_ms"], 25)
        self.assertAlmostEqual(stats["p95_ms"], 38.5)
        self.assertEqual(statistics([]), {"calls": 0})

    def test_spans_keep_error_and_discard_iterator_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            timer = Timer(directory)
            self.assertEqual(list(timer.iterate("decode", [1, 2])), [1, 2])
            with self.assertRaisesRegex(ValueError, "sentinel"):
                with timer.span("failure"):
                    raise ValueError("sentinel")
            timer.write()
            report = json.loads((Path(directory) / "components.json").read_text())
            self.assertEqual(report["components"]["decode"]["host_wall_ms"]["all"]["calls"], 2)
            self.assertEqual(report["components"]["failure"]["error_calls"], 1)
            self.assertEqual(report["components"]["failure"]["host_wall_ms"]["all"]["calls"], 0)
            timer.directory = None  # Don't rewrite a deleted temporary directory at exit.

    def test_stage_failure_propagates_and_still_generates_report(self):
        with tempfile.TemporaryDirectory() as directory:
            status = record_stage(directory, "fusion.py", [sys.executable, "-c", "raise SystemExit(7)"])
            self.assertEqual(status, 7)
            report = write_report(Path(directory), 1.0, status, ["fake"])
            self.assertEqual(report["stages"][0]["exit_code"], 7)
            self.assertIn("未采集", (Path(directory) / "report.md").read_text())

    def test_stage_preserves_heredoc_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/profile_multiview_wilor.py"),
                 "--record-stage", directory, "-", "--", sys.executable, "-"],
                input="print('stdin preserved')\n", text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("stdin preserved", result.stdout)
            row = json.loads((Path(directory) / "stages.jsonl").read_text())
            self.assertEqual(row["stage"], "run_config")


@unittest.skipIf(torch is None, "requires PyTorch")
class ModelTimingTests(unittest.TestCase):
    def make_model(self):
        nn = torch.nn

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Linear(2, 2)
                self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
                self.last_norm = nn.LayerNorm(2)
                self.decpose = nn.Linear(2, 2)
                self.decshape = nn.Linear(2, 2)
                self.deccam = nn.Linear(2, 2)

            def forward(self, x):
                x = self.patch_embed(x)
                for block in self.blocks:
                    x = block(x)
                x = self.last_norm(x)
                return self.decpose(x) + self.decshape(x) + self.deccam(x)

        class Refine(nn.Module):
            def __init__(self):
                super().__init__()
                for name in ("deconv", "dec_pose", "dec_shape", "dec_cam"):
                    setattr(self, name, nn.Linear(2, 2))

            def forward(self, x):
                x = self.deconv(x)
                return self.dec_pose(x) + self.dec_shape(x) + self.dec_cam(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.mano = nn.Linear(2, 2)
                self.refine_net = Refine()

            def forward(self, batch):
                return self.mano(self.refine_net(self.mano(self.backbone(batch["img"]))))

        return Model()

    def test_hooks_preserve_outputs_and_split_warmup_and_mano_passes(self):
        model = self.make_model()
        batch = {"img": torch.ones(3, 2)}
        expected = model(batch)
        with tempfile.TemporaryDirectory() as directory:
            timer = Timer(directory, warmup_batches=1)
            timer.configure(torch.device("cpu"))
            timer.attach(model)
            for _ in range(3):
                torch.testing.assert_close(model(batch), expected)
            timer.write()
            report = json.loads((Path(directory) / "components.json").read_text())
            for name in ("model.total", "vit.features", "vit.transformer_blocks",
                         "mano.parameter_decoder", "mano.coarse", "mano.final", "refinenet.total"):
                entry = report["components"][name]
                self.assertEqual(entry["host_wall_ms"]["all"]["calls"], 3, name)
                self.assertEqual(entry["host_wall_ms"]["steady"]["calls"], 2, name)
                self.assertEqual(entry["cuda_ms"]["all"]["calls"], 0)
            self.assertTrue(all(row["hypotheses"] == 3 for row in timer.rows))
            self.assertGreaterEqual(
                report["components"]["backbone.total"]["host_wall_ms"]["all"]["total_ms"],
                report["components"]["vit.features"]["host_wall_ms"]["all"]["total_ms"],
            )
            write_report(Path(directory), 1, 0, ["fake"])
            self.assertIn("MANO 参数解码器", (Path(directory) / "wilor_forward_average.csv").read_text(encoding="utf-8-sig"))
            self.assertNotIn("P95", (Path(directory) / "report.md").read_text())
            timer.directory = None

    def test_early_model_error_does_not_mask_original_exception(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            timer = Timer(directory)
            timer.configure(torch.device("cpu"))
            timer.attach(model)
            with self.assertRaises(RuntimeError):
                model({"img": torch.ones(3, 7)})
            timer.write()
            report = json.loads((Path(directory) / "components.json").read_text())
            self.assertEqual(report["components"]["vit.features"]["error_calls"], 1)
            self.assertEqual(report["components"]["model.total"]["error_calls"], 1)
            # The next successful attempt must not inherit dangling spans.
            model({"img": torch.ones(3, 2)})
            timer.directory = None


if __name__ == "__main__":
    unittest.main()
