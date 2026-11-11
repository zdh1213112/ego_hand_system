from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fit_mano_sequence import sample_observation_points,canonicalize_observations

class SurfaceMarkerFittingTests(unittest.TestCase):
    def test_default_preserves_joint_targets(self):
        vertices=torch.zeros(1,778,3);joints=torch.ones(1,21,3)
        self.assertIs(sample_observation_points(vertices,joints),joints)

    def test_surface_loss_updates_attachment_vertices_not_joint_targets(self):
        vertices=torch.zeros(1,778,3,requires_grad=True)
        joints=torch.ones(1,21,3,requires_grad=True)
        indices=torch.arange(21)+100
        selected=sample_observation_points(vertices,joints,indices)
        (selected-2).square().sum().backward()
        self.assertIsNone(joints.grad)
        self.assertTrue(torch.all(vertices.grad[:,indices]!=0))
        self.assertEqual(int(torch.count_nonzero(vertices.grad[:,:100])),0)
        self.assertEqual(int(torch.count_nonzero(vertices.grad[:,121:])),0)

    def test_left_canonicalization_preserves_attachment_ids(self):
        indices=np.arange(21)+100
        observations=dict(positions=np.ones((1,21,3)),rotation=np.eye(3),surface_marker_vertex_indices=indices)
        canonical=canonicalize_observations(observations,'Left')
        np.testing.assert_array_equal(canonical['surface_marker_vertex_indices'],indices)
        np.testing.assert_array_equal(canonical['positions'][...,0],-np.ones((1,21)))

if __name__=='__main__':unittest.main()
