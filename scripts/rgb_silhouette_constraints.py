"""Differentiable native-KB projection and bidirectional RGB contour loss."""
import torch


def project_native_kb(points, rotation, translation, K, distortion):
    camera=torch.einsum('ij,bkj->bki',rotation,points)+translation
    xy=camera[...,:2];r=torch.sqrt(xy.square().sum(-1)+1e-14)
    theta=torch.atan2(r,camera[...,2]);theta2=theta.square()
    td=theta*(1+distortion[0]*theta2+distortion[1]*theta2**2+distortion[2]*theta2**3+distortion[3]*theta2**4)
    normalized=xy*(td/r)[...,None]
    pixels=torch.stack([K[0,0]*normalized[...,0]+K[0,2],K[1,1]*normalized[...,1]+K[1,2]],dim=-1)
    return pixels,camera[...,2]>.03


def contour_loss(vertices,frame_ids,data):
    losses=[]
    for c in range(data['K'].shape[0]):
        indices=data['boundary_vertex_indices'][c,frame_ids]
        points=torch.gather(vertices,1,indices[...,None].expand(-1,-1,3))
        predicted,depth_valid=project_native_kb(points,data['rotation'][c],data['translation'][c],data['K'][c],data['distortion'][c])
        pv=data['boundary_valid'][c,frame_ids]&depth_valid
        target=data['contour_px'][c,frame_ids];tv=data['contour_valid'][c,frame_ids]
        pair_valid=pv[:,:,None]&tv[:,None,:]
        distance=torch.cdist(predicted,target).masked_fill(~pair_valid,1e5)
        nearest_pred=distance.min(dim=2).values;nearest_target=distance.min(dim=1).values
        frames_ok=pv.any(dim=1)&tv.any(dim=1)
        wp=pv&frames_ok[:,None];wt=tv&frames_ok[:,None]
        # A 1.5px tolerance avoids forcing a smooth naked-hand model to explain
        # image blur, glove texture and the outer radius of reflective balls.
        def penalty(d,weight):
            residual=torch.relu(d-1.5)*.001
            robust=torch.sqrt(residual.square()+.002**2)-.002
            return torch.where(weight,robust,0.).sum()/weight.sum().clamp_min(1)
        losses.append(.5*(penalty(nearest_pred,wp)+penalty(nearest_target,wt)))
    return torch.stack(losses).mean() if losses else vertices.sum()*0
