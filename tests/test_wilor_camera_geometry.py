from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from wilor_camera_geometry import fit_translation, project_pinhole, native_pixels, pinhole_view, proper_rotation
from ego_data.calibration import CameraCalibration


class CameraGeometryTests(unittest.TestCase):
    def test_exported_rotation_is_proper_after_amp_rounding(self):
        noisy=np.asarray([[.9998,.001,0],[-.0008,1.0003,.0001],[0,-.0002,.9997]])
        rotation=proper_rotation(noisy)
        np.testing.assert_allclose(rotation @ rotation.T,np.eye(3),atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)),1,places=6)

    def test_known_camera_translation_recovered_without_changing_shape(self):
        rng = np.random.default_rng(17)
        points = rng.normal(size=(21,3)) * [0.04,0.07,0.02]
        expected = np.asarray([0.12,0.18,0.65])
        K = np.asarray([[280.,0,500],[0,280,375],[0,0,1]])
        actual,error = fit_translation(points, project_pinhole(points+expected,K), K)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        self.assertLess(error,1e-6)

    def test_virtual_view_does_not_rotate_native_optical_coordinates(self):
        K = np.asarray([[500.,0,800],[0,500,600],[0,0,1]])
        camera=CameraCalibration("cam_0","cam_0","KB",(1600,1200),K,np.zeros(4),np.eye(4))
        view,maps=pinhole_view(camera,1000,750,130)
        point=np.asarray([[.1,.15,.6]])
        uv=project_pinhole(point,view)[0]
        native=native_pixels(point,camera)[0]
        x,y=np.rint(uv).astype(int)
        mapped=np.asarray([maps[0][y,x],maps[1][y,x]])
        self.assertLess(np.linalg.norm(mapped-native),3.0)
        self.assertEqual(maps[0].shape,(750,1000))


if __name__=="__main__":
    unittest.main()
