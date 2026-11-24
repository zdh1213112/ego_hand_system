#!/usr/bin/env python3
"""Meaningful geometry and gradient checks before fitting any real hand parameters."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from render_masks import dr, project_native, raster_mask, load_hand_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    context = dr.RasterizeCudaContext()
    uv0 = torch.tensor([[[18., 20.], [40., 21.], [27., 48.]]], device='cuda')
    depth = torch.ones((1, 3), device='cuda')*.5
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    boxes = torch.tensor([[0., 0., 64.]], device='cuda')
    mask = raster_mask(context, uv0, depth, faces, boxes, 64).detach().cpu().numpy()[0]
    # OpenCV fillConvexPoly expands integer edges by about half a pixel. Use
    # exact pixel-center polygon membership as the coordinate reference instead.
    yy, xx = np.mgrid[:64,:64]
    polygon=uv0[0].cpu().numpy()
    cpu=np.array([cv2.pointPolygonTest(polygon,(float(x),float(y)),False)>=0 for x,y in zip(xx.ravel(),yy.ravel())]).reshape(64,64).astype(np.uint8)
    iou = np.logical_and(mask>.5, cpu>0).sum()/np.logical_or(mask>.5, cpu>0).sum()
    assert iou > .97, f'Orientation / rasterization disagreement: {iou}'
    dx = torch.tensor(0., device='cuda', requires_grad=True)
    shift = torch.stack([dx, dx*0])
    pred = raster_mask(context, uv0+shift, depth, faces, boxes, 64)
    target = raster_mask(context, uv0+torch.tensor([6., 0.], device='cuda'), depth, faces, boxes, 64).detach()
    loss = (pred-target).square().mean()
    loss.backward()
    assert torch.isfinite(dx.grad) and dx.grad < -1e-6, f'Gradient should move right: {dx.grad}'
    identity = np.eye(4)
    record = {'T_base_cam0':identity, 'T_base_camera':identity, 'K':[[505.,0,939.],[0,506.,592.],[0,0,1]], 'D':[.076,-.014,.0019,.00007]}
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(1, 100, 3))*.1
    xyz[..., 2] += .6
    errors = []
    for side in [0, 1]:
        q = xyz.copy();q[..., 0] *= -1 if side == 0 else 1
        expected = cv2.fisheye.projectPoints(q.reshape(-1,1,3), np.zeros(3), np.zeros(3), np.array(record['K']), np.array(record['D']))[0].reshape(1,100,2)
        actual = project_native(torch.tensor(xyz, dtype=torch.float64, device='cuda'), [record], side)[0].detach().cpu().numpy()
        error = float(abs(expected-actual).max());errors.append(error)
        assert error < 1e-7, error
    # Full pixel masks must preserve finger gaps: two disjoint triangles stay disjoint.
    uv = torch.cat([uv0,uv0+torch.tensor([35.,0.],device='cuda')],1)
    two_faces = torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device='cuda')
    disconnected = raster_mask(context,uv,torch.ones((1,6),device='cuda')*.5,two_faces,torch.tensor([[0.,0.,100.]],device='cuda'),100)
    components = cv2.connectedComponents((disconnected[0].detach().cpu().numpy()>.5).astype(np.uint8))[0]-1
    assert components == 2, 'Renderer filled a gap between components'
    # A ring tests a true enclosed pinch hole, not merely separated components.
    ring_uv=torch.tensor([[[8.,8.],[56.,8.],[56.,56.],[8.,56.],[22.,22.],[42.,22.],[42.,42.],[22.,42.]]],device='cuda')
    ring_faces=torch.tensor([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],device='cuda',dtype=torch.int32)
    ring=raster_mask(context,ring_uv,torch.ones((1,8),device='cuda')*.5,ring_faces,boxes,64)
    assert float(ring[0,32,32])==0 and float(ring[0,12,32])>.99, 'Closed pinch hole was filled'
    import tempfile
    with tempfile.TemporaryDirectory(prefix='full-mask-speckle-') as temp:
        rgba=np.zeros((100,100,4),np.uint8);rgba[10:90,10:90,3]=255
        rgba[20:23,20:23,3]=0;rgba[40:65,40:65,3]=0
        path=Path(temp)/'mask.png';cv2.imwrite(str(path),rgba)
        cleaned=load_hand_mask(path)
        assert cleaned[21,21]==1 and cleaned[50,50]==0 and cleaned[0,0]==0
    report = {'triangle_opencv_iou':float(iou), 'translation_gradient':float(dx.grad),
              'kb_max_projection_error_px_by_side':errors, 'finger_gap_components':components,
              'closed_pinch_hole_preserved':True,'speckle_cleanup_preserves_large_holes':True,'checks_passed':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
