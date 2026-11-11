from pathlib import Path
import sys,unittest
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_silhouette_constraints import project_native_kb,contour_loss

class RgbSilhouetteTests(unittest.TestCase):
    def test_projection_matches_native_fisheye_for_mirrored_left_hand(self):
        points=np.asarray([[[.02,.01,.4],[-.08,.03,.5],[.1,-.06,.6],[0,0,.3]]])
        R=cv2.Rodrigues(np.asarray([.01,-.04,.02]))[0]@np.diag([-1.,1.,1.])
        t=np.asarray([-.05,.001,.002]);K=np.asarray([[500.,0,950],[0,502,580],[0,0,1.]])
        d=np.asarray([.08,-.02,.007,-.0009]);camera=points@R.T+t
        expected=cv2.fisheye.projectPoints(camera.reshape(-1,1,3),np.zeros(3),np.zeros(3),K,d)[0].reshape(1,-1,2)
        pred,valid=project_native_kb(*[torch.tensor(x,dtype=torch.float64) for x in [points,R,t,K,d]])
        np.testing.assert_allclose(pred.numpy(),expected,atol=1e-6)
        self.assertTrue(bool(valid.all()))

    def data(self,vertices):
        R=torch.eye(3);t=torch.zeros(3);K=torch.tensor([[100.,0,50],[0,100,50],[0,0,1.]])
        d=torch.zeros(4);pixels,_=project_native_kb(vertices.detach(),R,t,K,d)
        return dict(K=K[None],rotation=R[None],translation=t[None],distortion=d[None],boundary_vertex_indices=torch.arange(4).reshape(1,1,4),boundary_valid=torch.ones(1,1,4,dtype=torch.bool),contour_px=pixels[None].clone(),contour_valid=torch.ones(1,1,4,dtype=torch.bool))

    def test_contour_loss_is_zero_for_matching_projection_and_has_useful_gradient(self):
        vertices=torch.tensor([[[-.02,-.02,.4],[.02,-.02,.4],[.02,.02,.4],[-.02,.02,.4]]],requires_grad=True)
        data=self.data(vertices);ids=torch.tensor([0])
        self.assertAlmostEqual(float(contour_loss(vertices,ids,data)),0.,places=7)
        data['contour_px']+=torch.tensor([10.,0.])
        loss=contour_loss(vertices,ids,data);loss.backward()
        self.assertGreater(float(loss),0.)
        self.assertTrue(bool(torch.isfinite(vertices.grad).all()))
        self.assertGreater(float(vertices.grad.abs().sum()),0.)

    def test_missing_contours_contribute_no_loss(self):
        vertices=torch.tensor([[[-.02,-.02,.4],[.02,-.02,.4],[.02,.02,.4],[-.02,.02,.4]]],requires_grad=True)
        data=self.data(vertices);data['contour_valid'][:]=False
        loss=contour_loss(vertices,torch.tensor([0]),data);loss.backward()
        self.assertEqual(float(loss),0.)
        self.assertTrue(bool(torch.isfinite(vertices.grad).all()))

if __name__=='__main__':unittest.main()
