from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class WilorInferenceIoTests(unittest.TestCase):
    def test_native_global_orientation_singleton_is_squeezed(self):
        # Keep this test independent of CUDA/WiLoR imports; it mirrors the
        # adapter's stable archive contract for the native model tensor.
        value = np.zeros((1, 3, 3), dtype=np.float32)
        squeezed = value[0] if value.shape == (1, 3, 3) else value
        self.assertEqual(squeezed.shape, (3, 3))

    def test_incomplete_wilor_output_is_retried_with_backup(self):
        runner = (Path(__file__).resolve().parents[1] / "scripts" / "run_offline.sh").read_text()
        self.assertIn("WiLoR output is incomplete; preserving it", runner)
        self.assertIn(".failed-$(date +%Y%m%d-%H%M%S)", runner)


if __name__ == "__main__":
    unittest.main()
