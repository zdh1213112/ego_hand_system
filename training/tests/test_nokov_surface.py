import torch

from wilor.training.losses import measured_surface_loss


def test_surface_markers_follow_physical_wrist_and_mask_missing_sites():
    verts=torch.tensor([[[0.,0.,0.],[.02,0.,0.],[0.,.02,0.]]]).expand(2,-1,-1).clone().requires_grad_()
    root=torch.tensor([[.12,.04,.6],[.12,.04,.6]])
    triangle=torch.tensor([0,1,2]).expand(2,20,3)
    target=torch.tensor([.01,.005,.003]).expand(2,20,3).clone()
    target[0]+=torch.tensor([-.12,.04,.6])
    target[1]+=root[1]
    valid=torch.ones(2,20,dtype=torch.bool)
    valid[:,7]=False
    target[:,7]=100.
    sample=dict(side=torch.tensor([0,1]),surface_target_m=target,surface_valid=valid,surface_triangles=triangle,
                surface_barycentric=torch.tensor([.25,.5,.25]).expand(2,20,3),
                surface_offsets_m=torch.tensor([0.,0.,.003]).expand(2,20,3))
    output=dict(pred_vertices=verts,pred_keypoints_3d=torch.zeros(2,21,3),pred_root_camera_m=root)
    assert measured_surface_loss(output,sample).item()<1e-10
    target[:,5,1]+=.01
    loss=measured_surface_loss(output,sample)
    loss.backward()
    assert loss.item()>0 and torch.isfinite(verts.grad).all()
