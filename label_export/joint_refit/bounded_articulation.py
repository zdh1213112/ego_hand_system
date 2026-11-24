"""Refit MANO within finger-specific rotation limits, using measured markers and calibrated NOKOV turns."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import finger_prior
from dense_drive import marker_positions
from bend_targets import RecordedTurns, angular_error, local_turns, turn_coordinates
from run import PROJECT, import_mano, run_model

SHARED_KEYS = {'faces','betas','marker_names','attachment_faces','barycentric','attachment_local_offsets_m',
               'normal_offsets_m','wrist_local','root_anchor_local_m','geometry_space','approved_for_training'}
MARKER_ALLOWANCE_MM = .5


def components(pose, model, axes):
    return torch.stack(finger_prior.decompose(pose, model, axes), dim=-1)


def compose(angles, model, axes):
    """Reconstruct swing * twist in each shaped joint's local frame. Input units are degrees."""
    slots, bone, flex, lateral = axes
    twist, bend, side = torch.deg2rad(angles.flatten(1, 2)).unbind(-1)
    swing = bend[..., None] * flex + side[..., None] * lateral
    swing_angle = swing.norm(dim=-1, keepdim=True)
    sw = torch.cos(swing_angle / 2)
    sv = .5 * torch.sinc(swing_angle / (2*torch.pi)) * swing
    tw = torch.cos(twist[..., None] / 2)
    tv = torch.sin(twist[..., None] / 2) * bone
    qw = sw * tw - (sv * tv).sum(-1, keepdim=True)
    qv = sw * tv + tw * sv + torch.cross(sv, tv, dim=-1)
    qv = torch.where(qw < 0, -qv, qv)
    qnorm = qv.norm(dim=-1, keepdim=True)
    scale = 2 * torch.atan2(qnorm, qw.abs()) / qnorm.clamp_min(1e-8)
    vector = qv * torch.where(qnorm > 1e-8, scale, torch.full_like(scale, 2.))
    return vector[:, torch.argsort(slots)].flatten(1) - model.hand_mean


def rotation_limits(scans, model, axes):
    """Four finger hinge limits follow the existing prior. Thumb limits follow the MANO scan distribution."""
    thumb = components(scans, model, axes)[:, -1]
    thumb_limits = torch.quantile(thumb, thumb.new_tensor([.005,.995]), dim=0)
    lower = thumb.new_tensor([[-15.,-40.,-25.],[-10.,-5.,-10.],[-10.,-5.,-10.]]).expand(4,-1,-1)
    upper = thumb.new_tensor([[15.,110.,25.],[10.,115.,10.],[10.,100.,10.]]).expand(4,-1,-1)
    return (torch.cat([lower, (thumb_limits[0]-5.)[None]]),
            torch.cat([upper, (thumb_limits[1]+5.)[None]]))


def measured_anchors(model, calibration, syncs, axes, limits):
    """Refit the original image-selected calibration poses with the dense pass's feasible joint rotations."""
    from mano.lbs import batch_rodrigues

    device = model.faces_tensor.device
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    with np.load(calibration) as z:
        mask = np.isin(z['pair_indices'], syncs)
        times = z['pair_indices'][mask]
        beta = tensor(z['betas'])[None].expand(len(times), -1)
        orientation, pose, translation, goal = [tensor(z[k][mask]) for k in
                                                ('global_orient', 'hand_pose_pca', 'translation', 'target24')]
        triangles = model.faces_tensor.long()[torch.as_tensor(z['attachment_faces'], device=device).long()]
        attachment = [tensor(z[k]) for k in ('barycentric', 'attachment_local_offsets_m', 'wrist_local')]
    valid = goal.isfinite().all(-1)
    valid[:, :3] = False
    goal = goal.nan_to_num()
    original = components(pose, model, axes)
    lower, upper = limits
    angles = torch.nn.Parameter(original.clamp(lower+.002, upper-.002))
    rotation = batch_rodrigues(orientation)
    optimizer = torch.optim.Adam([angles], lr=.6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 220)
    for _ in range(220):
        optimizer.zero_grad()
        vertices, joints, _ = run_model(model, beta, orientation, compose(angles, model, axes), translation)
        mounts = marker_positions(vertices, joints[:, 0], rotation, triangles, *attachment)
        err = (mounts-goal).norm(dim=-1)*1000
        loss = ((torch.sqrt(1+(err/2.).square())-1.)*valid).sum()/valid.sum()
        loss += .02*((angles-original)/10.).square().mean()
        loss.backward()
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            angles.clamp_(lower+.002, upper-.002)
    with torch.no_grad():
        vertices, joints, _ = run_model(model, beta, orientation, compose(angles,model,axes), translation)
        mounts = marker_positions(vertices,joints[:,0],rotation,triangles,*attachment)
        residual = (mounts-goal).norm(dim=-1).masked_fill(~valid,float('nan'))*1000
    return dict(zip(times,joints.cpu().numpy())), float(torch.nanmedian(residual))


def fit_batch(model, arrays, axes, limits, triangles, attachment, sensor, start, stop, steps, recover=False):
    from mano.lbs import batch_rodrigues

    device = model.faces_tensor.device
    to = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    sl = slice(start, stop)
    times = arrays['pair_indices'][sl]
    old_pose, orient, translation = [to(arrays[k][sl]) for k in ('hand_pose_pca','global_orient','translation')]
    beta = to(arrays['betas'])[None].expand(len(times),-1)
    observed = torch.as_tensor(arrays['marker_valid'][sl], device=device)
    goal = to(arrays['target24'][sl])
    before_err = to(arrays['marker_error_mm'][sl])
    width = goal[:,8] - goal[:,20]
    rotation = batch_rodrigues(orient)
    old = components(old_pose, model, axes)
    lower, upper = limits
    first = old.clamp(lower + .002, upper - .002)
    values = torch.nn.Parameter(first.clone())
    optimizer = torch.optim.Adam([values], lr=.36)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    expected = sensor.expected[:,times].to(device)
    sensor_ok = sensor.valid[times].to(device) & observed[:,8,None,None] & observed[:,20,None,None]
    obs_per_digit = observed[:,4:].reshape(-1,5,4)
    sensor_ok &= (obs_per_digit[:,:,:-2] & obs_per_digit[:,:,1:-1] & obs_per_digit[:,:,2:])
    threshold = sensor.limits.to(device).clamp_min(10.)
    with torch.no_grad():
        old_bend, old_ok = local_turns(to(arrays['joints'][sl,1:]).reshape(-1,5,4,3), width)
        old_discrepancy = angular_error(old_bend, expected[0])
        supports = (turn_coordinates(expected[1:])-turn_coordinates(old_bend)[None])
        main = turn_coordinates(expected[0])-turn_coordinates(old_bend)
        reliable = sensor_ok & old_ok & ((supports * main).sum(-1)>0).all(0)
        targeted = reliable & (old_discrepancy > threshold)
        np.testing.assert_allclose(compose(old,model,axes).cpu().numpy(), old_pose.cpu().numpy(), atol=3e-5, rtol=0)
    neighbors = torch.as_tensor(np.flatnonzero(np.diff(times)==1),device=device)

    def forward(x):
        pose = compose(x, model, axes)
        vertices, joints, _ = run_model(model,beta,orient,pose,translation)
        predicted = marker_positions(vertices,joints[:,0],rotation,triangles,*attachment)
        residual = (predicted-goal).norm(dim=-1) * 1000
        bend, valid = local_turns(joints[:,1:].reshape(-1,5,4,3),width)
        return pose, vertices, joints, predicted, residual, bend, valid

    for _ in range(steps):
        optimizer.zero_grad()
        _, _, _, _, err, bend, _ = forward(values)
        marker_loss = ((torch.sqrt(1.+(err/4.).square())-1.)*observed).sum(1) / observed.sum(1)
        delta = (values-old)/10.
        turn_loss = torch.sqrt(1.+(bend-expected[0]).square().sum(-1)/np.deg2rad(10.)**2)-1.
        turn_loss = (turn_loss*targeted).sum((1,2)) / targeted.sum((1,2)).clamp_min(1)
        loss = (marker_loss + .14*turn_loss + .06*delta.square().mean((1,2,3))).mean()
        if recover:
            excess = (err-before_err-MARKER_ALLOWANCE_MM+.05).masked_fill(~observed,0.).clamp_min(0.)
            loss = loss + .4*excess.amax(1).square().mean()
        if len(neighbors):
            loss = loss + .03*(delta[neighbors+1]-delta[neighbors]).square().mean()
        loss.backward()
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            values.clamp_(lower+.002,upper-.002)

    with torch.no_grad():
        # Prefer feasible poses, then balance the measured marker residual and the sensor turns.
        # Include the exact old pose whenever it is already inside every limit.
        feasible_old = ((old>=lower) & (old<=upper)).all((1,2,3))
        best_cost = torch.full_like(before_err[:,0], float('inf'))
        best = first.clone()
        chosen = torch.zeros_like(best_cost)
        for alpha in (0., .25, .5, .75, 1., -1.):
            cmp_values = old if alpha < 0 else torch.lerp(first, values, alpha)
            _, _, _, _, err, bend, _ = forward(cmp_values)
            residual = angular_error(bend, expected[0])/10.
            excess = (err-before_err).masked_fill(~observed, 0.).amax(1)
            median = torch.nanquantile(err.masked_fill(~observed, float('nan')), .5, dim=1)
            worst = err.masked_fill(~observed,0.).amax(1)
            gate = (excess <= MARKER_ALLOWANCE_MM) & (median <= 2.5) & (worst <= 8.)
            cost = (err.square()*observed).sum(1)/observed.sum(1)
            cost += 2.*(residual.square()*targeted).sum((1,2))/targeted.sum((1,2)).clamp_min(1)
            cost += 1000.*(~gate) + 10.*F.relu(excess - MARKER_ALLOWANCE_MM)
            use = (cost < best_cost) & ((alpha>=0) | feasible_old)
            best[use] = cmp_values[use]
            chosen[use] = alpha
            best_cost[use] = cost[use]
        pose, vertices, joints, predicted, err, bend, good = forward(best)
        kept_errors = err.masked_fill(~observed, float('nan'))
        med = torch.nanquantile(kept_errors, .5, dim=1)
        worst = err.masked_fill(~observed, 0.).amax(1)
        increase = (err-before_err).masked_fill(~observed,0.).amax(1)
        half_discrepancies = angular_error(bend[None].expand_as(expected), expected)
        unresolved = sensor_ok & good & (half_discrepancies > threshold).all(0)
        components_after = components(pose,model,axes)
        out_of_range = ((components_after < lower-.002) | (components_after > upper+.002)).any((1,2,3))
        if out_of_range.any():
            raise ValueError('Stored MANO rotation failed the bounded parameter round trip')
        reasons = torch.stack([med>2.5, worst>8., increase>MARKER_ALLOWANCE_MM,
                               unresolved.any((1,2)), out_of_range], dim=1)
        out = dict(vertices=vertices,joints=joints,hand_pose_pca=pose,predicted24=predicted,
                   marker_error_mm=torch.nan_to_num(kept_errors))
        diagnostics = dict(reasons=reasons, per_marker_delta_mm=(err-before_err).masked_fill(~observed,0.),
            mask=observed, before_median_mm=torch.nanquantile(before_err.masked_fill(~observed,float('nan')), .5, dim=1),
            after_median_mm=med, targeted=targeted, unresolved=unresolved, sensor_valid=sensor_ok&good,
            before_direction_deg=old_discrepancy, after_direction_deg=half_discrepancies[0],
            before_angles=old, after_angles=components_after, step=chosen)
        return ({k:v.cpu().numpy() for k,v in out.items()}, {k:v.cpu().numpy() for k,v in diagnostics.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','calibrated','skeleton','selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=160)
    parser.add_argument('--steps',type=int,default=75)
    parser.add_argument('--refit-calibration-anchors', action='store_true',
                        help='refit the image-selected calibration poses within the same bounds before estimating sensor offsets')
    parser.add_argument('--recover-empty-recording', action='store_true',
                        help='repeat the fit with bounded calibration poses when the initial pass has no exportable hand times')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args = parser.parse_args()
    if args.batch_size<1 or args.steps<1:
        parser.error('batch-size and steps must be positive')
    solve(args)
    if args.recover_empty_recording and not args.refit_calibration_anchors and not shared_times(args.output):
        original = args.output
        args.output = original/'anchor_recovery'
        args.refit_calibration_anchors = True
        args.steps = max(args.steps, 140)
        solve(args)
        if shared_times(args.output):
            initial = original/'initial_pass'
            initial.mkdir()
            for file in original.iterdir():
                if file not in (initial, args.output):
                    file.rename(initial/file.name)
            for file in args.output.iterdir():
                file.rename(original/file.name)
            args.output.rmdir()


def shared_times(directory):
    times = []
    for file in sorted((directory/'final').glob('*_pilot.npz')):
        with np.load(file) as fit:
            times.append(set(map(int,fit['pair_indices'])))
    return set.intersection(*times) if times else set()


def solve(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'final').mkdir()
    model=import_mano(PROJECT/'third_party/MANO').load(model_path=str(PROJECT/'models/mano'),is_rhand=True,
        use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(args.device).eval()
    model.requires_grad_(False)
    with (PROJECT/'models/mano/MANO_RIGHT.pkl').open('rb') as file:
        data=pickle.load(file,encoding='latin1')
    scans=torch.as_tensor(np.asarray(data['hands_coeffs'])@np.asarray(data['hands_components']),
                          device=args.device,dtype=torch.float32)
    selection=json.loads(args.selection.read_text())['report']['hands']
    reasons=('median_marker_residual','worst_marker_residual','added_marker_residual','disputed_sensor_turn','angle_bound')
    report=dict(strategy='joint_refit_v6_bounded',approved_for_training=False,source=str(args.source.resolve()),
                marker_allowance_mm=MARKER_ALLOWANCE_MM, angle_order=['twist','flexion','lateral'],
                finger_order=list(finger_prior.FINGERS)+['thumb'], rejection_reasons=reasons, hands={})
    for side,hand in enumerate(('left','right')):
        name=f'{hand}_pilot.npz'
        if not (args.source/name).is_file():
            continue
        with np.load(args.source/name) as z:
            arrays=dict(z)
        times=arrays['pair_indices']
        n=len(times)
        if not n:
            np.savez_compressed(args.output/'final'/name,**arrays)
            report['hands'][hand]=dict(frames=0,kept=0,rejected={},reason='no_marker_fit_frames')
            continue
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=args.device)
        axes=finger_prior.rest_axes(model,tensor(arrays['betas']),args.device,include_thumb=True)
        limits=rotation_limits(scans,model,axes)
        syncs=selection[hand]['calibration_sync']
        anchors, anchor_error = (measured_anchors(model, args.calibrated/name, syncs, axes, limits)
                                 if args.refit_calibration_anchors else (None,None))
        sensor=RecordedTurns(args.skeleton,side,hand,args.calibrated/name,syncs,anchors)
        sensor.report.update(refit_within_bounds=args.refit_calibration_anchors, anchor_marker_median_mm=anchor_error)
        np.testing.assert_allclose(sensor.raw[times], arrays['target24_uncorrected'],rtol=0,atol=1e-6,equal_nan=True)
        tri=model.faces_tensor.long()[torch.as_tensor(arrays['attachment_faces'],device=args.device).long()]
        attach=[tensor(arrays[k]) for k in ('barycentric','attachment_local_offsets_m','wrist_local')]
        chunks, details = [], []
        for first in range(0,n,args.batch_size):
            out,diag=fit_batch(model,arrays,axes,limits,tri,attach,sensor,first,min(n,first+args.batch_size),args.steps,args.refit_calibration_anchors)
            chunks.append(out)
            details.append(diag)
        merged={k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
        diag={k:np.concatenate([r[k] for r in details]) for k in details[0]}
        keep=~diag['reasons'].any(1)
        np.savez_compressed(args.output/f'{hand}_diagnostics.npz',pair_indices=times,
            joints=merged['joints'], predicted24=merged['predicted24'], hand_pose_pca=merged['hand_pose_pca'],
            keep=keep,**diag)
        rejected=[dict(sync=int(times[i]),reasons=[reason for reason,bad in zip(reasons,diag['reasons'][i]) if bad])
                  for i in np.flatnonzero(~keep)]
        (args.output/f'{hand}_rejected.json').write_text(json.dumps(rejected))
        arrays.update(merged)
        saved={k:(v if k in SHARED_KEYS or not v.ndim or v.shape[0]!=n else v[keep]) for k,v in arrays.items()}
        np.savez_compressed(args.output/'final'/name,**saved)
        before,after=diag['before_angles'],diag['after_angles']
        summary=dict(frames=n,kept=int(keep.sum()),rejected={k:int(v) for k,v in zip(reasons,diag['reasons'].sum(0))},
            calibration=sensor.report,lower_degrees=limits[0].cpu().tolist(),upper_degrees=limits[1].cpu().tolist(),
            median_marker_before_mm=float(np.median(diag['before_median_mm'][keep])) if keep.any() else None,
            median_marker_after_mm=float(np.median(diag['after_median_mm'][keep])) if keep.any() else None,
            hinge_backwards_before=int((before[keep,:4,1:,1]<-5.).sum()),hinge_backwards_after=int((after[keep,:4,1:,1]<-5.001).sum()))
        report['hands'][hand]=summary
        print(hand,n,summary['kept'],summary['rejected'],flush=True)
    if not report['hands']:
        raise ValueError('Source has no fitted hands')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
