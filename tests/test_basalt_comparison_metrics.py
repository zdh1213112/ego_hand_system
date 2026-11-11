"""Geometric invariants for the benchmark's registration and error metrics."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_basalt_ego4_comparison import aligned_poses, position_se3_alignment, relative_pose_errors


class ComparisonMetricsTest(unittest.TestCase):
    def fixture(self):
        t = np.arange(181) / 30
        p = np.column_stack([np.sin(t), np.cos(t), .1 * t])
        r = Rotation.from_rotvec(np.column_stack([t * .05, t * .02, t * .1]))
        world = Rotation.from_rotvec([.3, -.2, .6])
        gt = world.apply(p) + [.5, -.3, 1.2]
        return t, p, r, gt, world * r

    def test_registration_and_rpe_ignore_world_gauge(self):
        t, p, r, gp, gr = self.fixture()
        pp, rr, _, _ = aligned_poses(p, r, gp, gr, t)
        np.testing.assert_allclose(pp, gp, atol=1e-12)
        np.testing.assert_allclose((rr.inv() * gr).magnitude(), 0, atol=1e-12)
        error = relative_pose_errors(t, p, r, gp, gr, .1)
        self.assertGreater(error["count"], 170)
        self.assertLess(error["translation_m"]["max"], 1e-12)
        self.assertLess(error["rotation_deg"]["max"], 1e-10)

    def test_scale_error_is_not_fitted_away(self):
        _, p, _, gp, _ = self.fixture()
        exact = position_se3_alignment(p, gp)
        np.testing.assert_allclose(exact, gp, atol=1e-12)
        wrong_scale = position_se3_alignment(2 * p, gp)
        self.assertGreater(np.sqrt(np.mean(np.sum((wrong_scale - gp)**2, axis=1))), .5)

    def test_prefix_registration_does_not_remove_later_drift(self):
        t, p, r, gp, gr = self.fixture()
        drifting = p.copy()
        drifting[t > 3, 0] += .25
        pp, _, _, _ = aligned_poses(drifting, r, gp, gr, t)
        np.testing.assert_allclose(np.linalg.norm(pp[t > 3] - gp[t > 3], axis=1), .25, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
