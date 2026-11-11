from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from colorize_vio_semantic_map import UNKNOWN,fuse_probabilities,write_coloured_ply
from ego_data.map_view import change_frame,initial_imu_frame


class SemanticMapTest(unittest.TestCase):
    def test_initial_imu_is_origin_and_rotation_identity(self):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.3,-.4,.9]).as_matrix();pose[:3,3]=[.2,-.1,1.2]
        transform=initial_imu_frame(pose)
        np.testing.assert_allclose(transform@pose,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(change_frame([pose[:3,3]],transform),[[0,0,0]],atol=1e-12)

    def test_same_display_transform_preserves_map_distances_and_position_errors(self):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.2,-.5,1.3]).as_matrix();pose[:3,3]=[.1,.3,1.2]
        a=np.array([[.2,.6,1.4],[.4,-.1,1.3],[-.6,.5,1.5]])
        b=a+np.array([[.01,.02,.03],[-.03,.05,.07],[.1,0,0]])
        transform=initial_imu_frame(pose);aa,bb=change_frame(a,transform),change_frame(b,transform)
        np.testing.assert_allclose(np.linalg.norm(a-b,axis=1),np.linalg.norm(aa-bb,axis=1),atol=1e-12)
        np.testing.assert_allclose(change_frame(aa,pose),a,atol=1e-12)

    def test_confident_consistent_class_is_kept_and_ambiguous_or_unobserved_stays_unknown(self):
        total=np.array([[1.7,.3],[1.,1.],[0,0],[.9,.1],[.8,1.2]])
        weights=np.array([2.,2.,0.,1.,2.]);votes=np.array([[2.,0],[1.,1.],[0,0],[1.,0],[1.,1.]])
        views=np.array([2,2,0,1,2])
        labels,confidence,agreement=fuse_probabilities(total,weights,votes,views,.45,.6,2)
        np.testing.assert_array_equal(labels,[0,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN])
        self.assertAlmostEqual(confidence[0],.85);self.assertAlmostEqual(agreement[0],1.)
        self.assertTrue(np.isfinite(confidence).all())

    def test_ply_retains_unknown_points_and_writes_actual_rgb_and_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'map.ply'
            write_coloured_ply(path,np.array([[0,0,0],[1,2,3]]),np.array([[4,5,6],[100,110,120]]),np.array([3,UNKNOWN]),np.array([.9,.2]))
            lines=path.read_text().splitlines();self.assertIn('element vertex 2',lines)
            self.assertIn('property ushort class_id',lines)
            points=lines[lines.index('end_header')+1:]
            self.assertEqual(len(points),2);self.assertEqual(points[1].split()[3:7],['100','110','120','255'])


if __name__=='__main__':unittest.main()
