"""Boundary and geometry checks for partial-reference VIO evaluation."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_basalt_nokov_overlap import overlap_mask, register_prefix


class PartialReferenceTest(unittest.TestCase):
    def test_only_reference_overlap_is_scored(self):
        t = np.arange(0., 7., .01)
        reference = np.arange(2., 5.001, .01)
        selected = t[overlap_mask(t, reference)]
        self.assertGreaterEqual(selected[0], 2.005)
        self.assertLessEqual(selected[-1], 4.995)

    def test_reference_holes_are_not_interpolated(self):
        reference = np.r_[np.arange(0., 1., .01), np.arange(2., 3., .01)]
        selected = overlap_mask(np.array([.5, 1., 1.5, 1.99, 2.5]), reference)
        np.testing.assert_array_equal(selected, [True, False, False, False, True])

    def test_registration_starts_at_first_overlapping_pose_and_preserves_drift(self):
        times = 3. + np.arange(181) / 30
        p = np.column_stack([np.sin(times), np.cos(times), .1 * times])
        r = Rotation.from_rotvec(np.column_stack([times*.05, times*.02, times*.1]))
        world = Rotation.from_rotvec([.3, -.2, .6])
        gp, gr = world.apply(p) + [.5, -.3, 1.2], world*r
        drifting = p.copy(); drifting[times > 6, 0] += .25
        aligned, ar, _, _, mask = register_prefix(drifting, r, gp, gr, times)
        self.assertEqual(int(mask.sum()), 61)
        np.testing.assert_allclose(aligned[mask], gp[mask], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(aligned[times > 6]-gp[times > 6], axis=1), .25, atol=1e-12)
        np.testing.assert_allclose((ar.inv()*gr).magnitude(), 0, atol=1e-12)

    def test_short_overlap_and_nonmonotonic_reference_rejected(self):
        with self.assertRaises(ValueError):
            overlap_mask([1., 2.], [1., 1.])
        with self.assertRaises(ValueError):
            register_prefix(np.zeros((2,3)), Rotation.identity(2), np.zeros((2,3)),
                            Rotation.identity(2), np.array([3., 3.5]))


if __name__ == "__main__":
    unittest.main()
