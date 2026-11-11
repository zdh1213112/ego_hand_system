from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fit_nokov_markers import calibrate_attachments, joint_targets
from visualize_nokov_mano import quantized


class NokovMarkerGeometryTests(unittest.TestCase):
    def test_joint_order_and_missing_wrist_are_preserved(self):
        points=np.arange(2*24*3,dtype=np.float32).reshape(2,24,3)/1000
        points[1,1]=np.nan
        target=joint_targets(points)
        self.assertEqual(target.shape,(2,21,3))
        np.testing.assert_allclose(target[0,0],(points[0,1]+points[0,2])/2)
        np.testing.assert_array_equal(target[:,1:],points[:,4:])
        self.assertTrue(np.isnan(target[1,0]).all())

    def test_attachment_is_fixed_under_rigid_motion_and_stays_on_surface(self):
        triangle=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
        vertices=np.stack([triangle,triangle+np.array([3,-2,1]),triangle+np.array([-.3,.7,2])])
        faces=np.array([[0,1,2]])
        bary=np.array([.2,.3,.5])
        markers=(vertices*bary[None,:,None]).sum(1)[:,None]
        face_ids,weights=calibrate_attachments(vertices,markers,faces)
        np.testing.assert_array_equal(face_ids,[0])
        np.testing.assert_allclose(weights[0],bary,atol=1e-6)
        outside=markers.copy();outside[:,:,0]+=3
        _,bounded=calibrate_attachments(vertices,outside,faces)
        self.assertTrue((bounded>=0).all())
        np.testing.assert_allclose(bounded.sum(-1),1)

    def test_viewer_quantization_does_not_turn_occlusion_into_observation(self):
        import base64
        origin=np.array([[1,2,3]],dtype=np.float32)
        points=np.array([[[1.01,2.02,3.03],[np.nan,np.nan,np.nan]]],dtype=np.float32)
        packed=np.frombuffer(base64.b64decode(quantized(points,origin)),dtype='<i2').reshape(1,2,3)
        np.testing.assert_array_equal(packed[0,0],[100,200,300])
        np.testing.assert_array_equal(packed[0,1],[-32768]*3)


if __name__=='__main__':unittest.main()
