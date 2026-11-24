"""Refine bounded poses on reliable whole-hand masks while retaining the measured-marker and SDK gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bounded_articulation import (MARKER_ALLOWANCE_MM, RecordedTurns, angular_error, components, compose,
    finger_prior, import_mano, local_turns, marker_positions, measured_anchors, PROJECT, run_model)
from render_masks import FullMaskBatch


def silhouette_per_frame(batch, vertices):
    pixels = batch.render(vertices).clamp(0, 1)
    intersection = (pixels * batch.target * batch.domain).sum((1, 2))
    union = ((pixels + batch.target - pixels*batch.target) * batch.domain).sum((1, 2)).clamp_min(1)
    return intersection / union


def refine(model, fit, idx, records, side, axes, limits, sensor, reference_error):
    from mano.lbs import batch_rodrigues

    device = model.faces_tensor.device
    tensor = lambda x: torch.as_tensor(x,dtype=torch.float32,device=device)
    times = fit['pair_indices'][idx]
    lookup = {int(t):i for i,t in enumerate(times)}
    # Withhold one view whenever a time has multiple usable masks.
    training, held = [], []
    for sync in times:
        views = sorted((r for r in records if r['sync']==sync),key=lambda r:r['camera'])
        training.extend(views if len(views)==1 else views[:-1])
        if len(views)>1:
            held.append(views[-1])
    def make_masks(rows):
        return FullMaskBatch(rows,lookup,fit['joints'][idx],fit['vertices'][idx],fit['faces'],side,192,str(device))
    masks = make_masks(training)
    test_masks = make_masks(held) if held else None
    betas = tensor(fit['betas']).expand(len(idx),-1)
    directions, translation, source_pose, goal = [tensor(fit[k][idx]) for k in
                                                  ('global_orient','translation','hand_pose_pca','target24')]
    base = components(source_pose,model,axes)
    angles = torch.nn.Parameter(base.clone())
    valid = torch.as_tensor(fit['marker_valid'][idx],device=device)
    bound_error = tensor(reference_error) + MARKER_ALLOWANCE_MM
    triangles = model.faces_tensor.long()[torch.as_tensor(fit['attachment_faces'],device=device).long()]
    attachments = [tensor(fit[k]) for k in ('barycentric','attachment_local_offsets_m','wrist_local')]
    rotation = batch_rodrigues(directions)
    width = goal[:,8]-goal[:,20]
    digit = valid[:,4:].reshape(-1,5,4)
    sensor_valid = (sensor.valid[times].to(device) & valid[:,8,None,None] & valid[:,20,None,None] &
                    digit[:,:,:-2] & digit[:,:,1:-1] & digit[:,:,2:])
    targets = sensor.expected[:,times].to(device)
    threshold = sensor.limits.to(device).clamp_min(10.)
    def forward(values):
        pose = compose(values,model,axes)
        vertices,joints,_ = run_model(model,betas,directions,pose,translation)
        mounted = marker_positions(vertices,joints[:,0],rotation,triangles,*attachments)
        error = (mounted-goal).norm(dim=-1)*1000
        return pose,vertices,joints,mounted,error
    optimizer = torch.optim.Adam([angles],lr=.3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,90)
    for _ in range(90):
        optimizer.zero_grad()
        _,vertices,joints,_,error = forward(angles)
        iou,sdf,_ = masks.terms(vertices)
        extra = (error-bound_error+.025).clamp_min(0.).masked_fill(~valid,0.)
        bend,_ = local_turns(joints[:,1:].reshape(-1,5,4,3),width)
        # Angular residual as chord length avoids unstable derivatives near parallel vectors.
        chord = (bend-targets[0]).norm(dim=-1)
        limit_chord = 2*torch.sin(torch.deg2rad(threshold.clamp_max(180))/2)
        sensor_excess = (chord-limit_chord).clamp_min(0.).masked_fill(~sensor_valid,0.)
        loss = 7*iou+sdf+.3*extra.amax(1).square().mean()+3*sensor_excess.square().mean()
        loss += .035*((angles-base)/6.).square().mean()
        loss += .015*((error.square()*valid).sum(1)/valid.sum(1)).mean()
        loss.backward();optimizer.step();schedule.step()
        with torch.no_grad():
            angles.clamp_(limits[0]+.002,limits[1]-.002)
    with torch.no_grad():
        base_iou = silhouette_per_frame(masks,tensor(fit['vertices'][idx]))
        counts = torch.bincount(masks.frame,minlength=len(times)).clamp_min(1)
        def mean_per_frame(values):
            return torch.zeros_like(counts,dtype=values.dtype).scatter_add_(0,masks.frame,values)/counts
        best = base.clone()
        best_iou = mean_per_frame(base_iou)
        for amount in (.25,.5,.75,1.):
            candidate = torch.lerp(base,angles,amount)
            _,vertices,joints,_,error = forward(candidate)
            bend,good = local_turns(joints[:,1:].reshape(-1,5,4,3),width)
            disagreements = angular_error(bend[None].expand_as(targets),targets)
            unresolved = (sensor_valid & good & (disagreements>threshold).all(0)).any((1,2))
            median = torch.nanquantile(error.masked_fill(~valid,float('nan')),.5,dim=1)
            safe = ((error<=bound_error)|~valid).all(1) & ((error<=8.)|~valid).all(1) & (median<=2.5) & ~unresolved
            score = mean_per_frame(silhouette_per_frame(masks,vertices))
            use = safe & (score>best_iou+.0001)
            best[use],best_iou[use] = candidate[use],score[use]
        pose,vertices,joints,mounted,error = forward(best)
        draws = [dict(sync=r['sync'],camera=r['camera'],held_out=False,before=float(a),after=float(b))
                 for r,a,b in zip(training,base_iou,silhouette_per_frame(masks,vertices))]
        if held:
            draws.extend(dict(sync=r['sync'],camera=r['camera'],held_out=True,before=float(a),after=float(b))
                         for r,a,b in zip(held,silhouette_per_frame(test_masks,tensor(fit['vertices'][idx])),
                                          silhouette_per_frame(test_masks,vertices)))
        changed = (best!=base).any((1,2,3)).cpu().numpy()
        for key,value in dict(hand_pose_pca=pose,vertices=vertices,joints=joints,predicted24=mounted,
                              marker_error_mm=error.masked_fill(~valid,0.)).items():
            fit[key][idx[changed]] = value.cpu().numpy()[changed]
    return draws, times[changed].tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','skeleton','calibrated','selection','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=['cuda'])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'final').mkdir()
    report=json.loads((args.source.parent/'summary.json').read_text())
    reference=Path(report['source'])
    if report['strategy']!='joint_refit_v6_bounded':
        raise ValueError('A bounded input fit is required')
    model=import_mano(PROJECT/'third_party/MANO').load(model_path=str(PROJECT/'models/mano'),is_rhand=True,
        use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(args.device).eval().requires_grad_(False)
    data=json.loads(args.selection.read_text())
    for side,hand in enumerate(('left','right')):
        name=hand+'_pilot.npz'
        if not (args.source/name).is_file():continue
        with np.load(args.source/name) as z:fit=dict(z)
        times=fit['pair_indices']
        known=set(times)
        records=[r for r in data['records'] if r['side']==side and r['accepted_quality'] and r['sync'] in known]
        selected=np.array(sorted({r['sync'] for r in records}),dtype=int)
        info=report['hands'][hand]
        detail=[];changed=[]
        if len(selected):
            axes=finger_prior.rest_axes(model,torch.as_tensor(fit['betas'],device=args.device),args.device,include_thumb=True)
            limits=[torch.tensor(info[k],device=args.device) for k in ('lower_degrees','upper_degrees')]
            syncs=data['report']['hands'][hand]['calibration_sync']
            anchors=None
            if info['calibration'].get('refit_within_bounds'):
                anchors,_=measured_anchors(model,args.calibrated/name,syncs,axes,limits)
            sensor=RecordedTurns(args.skeleton,side,hand,args.calibrated/name,syncs,anchors)
            with np.load(reference/name) as z:
                indices=np.searchsorted(z['pair_indices'],times)
                np.testing.assert_array_equal(z['pair_indices'][indices],times)
                reference_error=z['marker_error_mm'][indices]
            for first in range(0,len(selected),24):
                sample=selected[first:first+24]
                idx=np.searchsorted(times,sample)
                scores,numbers=refine(model,fit,idx,records,side,axes,limits,sensor,reference_error[idx])
                detail.extend(scores);changed.extend(numbers)
        np.savez_compressed(args.output/'final'/name,**fit)
        (args.output/f'{hand}_views.json').write_text(json.dumps(detail))
        info['visible_contours']=dict(available_times=len(selected),changed_times=changed,views=len(detail),
            held_out_views=sum(r['held_out'] for r in detail))
        print(hand,len(selected),len(changed),len(detail),flush=True)
    report.update(strategy='joint_refit_v7_contours',bounded_source=str(args.source.resolve()))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
