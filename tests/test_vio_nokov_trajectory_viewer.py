from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from visualize_vio_nokov_trajectories import aligned_poses, metric_windows


class TrajectoryViewerTest(unittest.TestCase):
    def test_one_world_transform_applies_to_all_times_and_preserves_later_drift(self):
        times=np.linspace(0,30,121)
        reference=np.column_stack([np.sin(times),np.cos(times),1.2+.05*times])
        reference_rot=Rotation.from_euler('z',times*.1)
        world_r=Rotation.from_euler('xyz',[.1,-.2,.8]);world_t=np.array([.4,-.3,1.1])
        raw=world_r.inv().apply(reference-world_t)
        raw_rot=world_r.inv()*reference_rot
        drift=np.maximum(times-2,0)[:,None]*np.array([.01,0,0])
        aligned,rotation,r,t=aligned_poses(raw+drift,raw_rot,reference,reference_rot,times)
        np.testing.assert_allclose(r.as_matrix(),world_r.as_matrix(),atol=1e-12)
        np.testing.assert_allclose(t,world_t,atol=1e-12)
        np.testing.assert_allclose(aligned-reference,world_r.apply(drift),atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(aligned[-1]-reference[-1]),.28)
        np.testing.assert_allclose((reference_rot.inv()*rotation).magnitude(),0,atol=1e-12)

    def test_world_registration_does_not_fit_scale(self):
        times=np.arange(10,dtype=float)
        reference=np.column_stack([times*.1,np.zeros(10),np.ones(10)])
        rotations=Rotation.identity(10)
        aligned,_,_,_=aligned_poses(reference*1.2,rotations,reference,rotations,times)
        self.assertAlmostEqual(np.linalg.norm(aligned[-1]-aligned[0]),1.08)
        self.assertGreater(np.linalg.norm(aligned[-1]-reference[-1]),.1)

    def test_all_reference_poses_are_counted_and_split_boundaries_are_explicit(self):
        times=np.array([0,1,2,3,19.9,20,20.1,20.2,25])
        stats=metric_windows(times,np.ones(9),np.full(9,2.))
        self.assertEqual(stats['all']['count'],9)
        self.assertEqual(stats['after_world_registration']['count'],6)
        self.assertEqual(stats['sensor_calibration_interval']['count'],5)
        self.assertEqual(stats['after_sensor_calibration']['count'],4)
        self.assertEqual(stats['after_calibration_buffer']['count'],2)
        self.assertEqual(stats['all']['position_rmse_m'],1)
        self.assertEqual(stats['all']['orientation_rmse_deg'],2)


if __name__=='__main__':
    unittest.main()
