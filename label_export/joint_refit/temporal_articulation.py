"""Refine finger articulation over consecutive, measured frames with a fixed wrist and hand shape."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bounded_articulation import (
    MARKER_ALLOWANCE_MM,
    RecordedTurns,
    angular_error,
    components,
    compose,
    finger_prior,
    import_mano,
    local_turns,
    marker_positions,
    measured_anchors,
    run_model,
)
from run import PROJECT


def masked_mean(values, mask):
    return values.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)


def consecutive_triplets(times, stamps):
    """Return the start of each three-exposure stencil with normal sync and timestamp gaps."""
    if len(times) < 3:
        return np.empty(0, dtype=np.int64)
    nominal = np.median(np.diff(stamps))
    intervals = np.diff(stamps[times])
    connected = (np.diff(times) == 1) & (intervals > .8 * nominal) & (intervals < 1.2 * nominal)
    return np.flatnonzero(connected[:-1] & connected[1:])


def refine_chunk(model, arrays, sensor, axes, limits, reference, stamp, locked, steps, frame_slice):
    from mano.lbs import batch_rodrigues
    device = model.faces_tensor.device

    def tensor(data):
        return torch.as_tensor(data, device=device, dtype=torch.float32)

    times = arrays["pair_indices"][frame_slice]
    observed = torch.as_tensor(arrays["marker_valid"][frame_slice], device=device)
    target = tensor(arrays["target24"][frame_slice]).nan_to_num()
    pose, orient, shift = [
        tensor(arrays[k][frame_slice]) for k in ("hand_pose_pca", "global_orient", "translation")
    ]
    beta = tensor(arrays["betas"]).expand(len(pose), -1)
    rotation = batch_rodrigues(orient)
    attachments = [tensor(arrays[k]) for k in ("barycentric", "attachment_local_offsets_m", "wrist_local")]
    triangles = model.faces_tensor.long()[torch.as_tensor(arrays["attachment_faces"], device=device).long()]
    old = components(pose, model, axes)
    width = target[:, 8] - target[:, 20]
    target_turn = sensor.expected[:, times].to(device)
    obs_digit = observed[:, 4:].reshape(-1, 5, 4)
    sensor_ok = sensor.valid[times].to(device) & observed[:, 8, None, None] & observed[:, 20, None, None]
    sensor_ok &= obs_digit[:, :, :-2] & obs_digit[:, :, 1:-1] & obs_digit[:, :, 2:]
    sensor_cap = sensor.limits.to(device).clamp_min(10.)

    def forward(angles):
        hand_pose = compose(angles, model, axes)
        verts, joints, _ = run_model(model, beta, orient, hand_pose, shift)
        mounts = marker_positions(verts, joints[:, 0], rotation, triangles, *attachments)
        residual = (mounts - target) * 1000
        turns, good = local_turns(joints[:, 1:].reshape(-1, 5, 4, 3), width)
        return hand_pose, verts, joints, mounts, residual, turns, good

    # Each sample is a real exposure. A missing sync or irregular timestamp breaks the stencil.
    triplets = torch.as_tensor(consecutive_triplets(times, stamp), device=device)
    dynamic = observed[triplets] & observed[triplets + 1] & observed[triplets + 2]
    turn_dyn = sensor_ok[triplets] & sensor_ok[triplets + 1] & sensor_ok[triplets + 2]

    def acceleration(x):
        return x[triplets] - 2 * x[triplets + 1] + x[triplets + 2]

    with torch.no_grad():
        _, _, initial_joints, _, initial_residual, initial_turns, _ = forward(old)
        initial_error = initial_residual.norm(dim=-1)
        initial_mse = initial_error.square().masked_fill(~observed, 0.).sum(1) / observed.sum(1)
        reference_limit = tensor(reference[frame_slice]) + MARKER_ALLOWANCE_MM
        locked_tensor = torch.as_tensor(np.isin(times, locked), device=device)
    angles = torch.nn.Parameter(old.clone())
    optimizer = torch.optim.Adam([angles], lr=.28)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(steps):
        optimizer.zero_grad()
        _, _, joints, _, residual, turns, _ = forward(angles)
        marker_error = residual.norm(dim=-1)
        marker_mse = marker_error.square().masked_fill(~observed, 0.).sum(1) / observed.sum(1)
        loss = marker_mse.mean() + 3 * (marker_mse - initial_mse).clamp_min(0.).mean()
        excess = (marker_error -
                  torch.minimum(initial_error + .28, reference_limit)).clamp_min(0.).masked_fill(
                      ~observed, 0.)
        loss = loss + 6 * excess.amax(1).square().mean()
        # Match both position and direction changes measured on the moving glove.
        # Regularize twist and lateral bending only at the four long fingers' two hinge joints.
        turn_residual = turns - target_turn[0]
        hinge = angles[:, :4, 1:, [0, 2]]
        loss = loss + .65 * (hinge / 4.).square().mean() + .12 * ((angles - old) / 4.).square().mean()
        loss = loss + .40 * masked_mean(turn_residual.square().sum(-1) / np.deg2rad(10.)**2, sensor_ok)
        tip_move = (joints[:, [4, 8, 12, 16, 20]] - initial_joints[:, [4, 8, 12, 16, 20]]) * 1000
        loss = loss + .55 * tip_move.square().sum(-1).mean()
        if len(triplets):
            loss = loss + .85 * masked_mean(acceleration(residual).square().sum(-1), dynamic)
            loss = loss + .50 * masked_mean(
                acceleration(turn_residual).square().sum(-1) / np.deg2rad(5.)**2, turn_dyn)
            loss = loss + .28 * (acceleration(hinge) / 2.).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            angles.clamp_(limits[0] + .002, limits[1] - .002)
            angles[locked_tensor] = old[locked_tensor]

    with torch.no_grad():
        # Pick the largest supported correction under the original exporter gates.
        # Every rejected frame retains its original arrays exactly.
        fraction = old.new_zeros(len(pose))
        for alpha in (1., .75, .5, .25, .125):
            _, _, joints, _, residual, turn, good = forward(torch.lerp(old, angles, alpha))
            marker_error = residual.norm(dim=-1)
            marker_mse = marker_error.square().masked_fill(~observed, 0.).sum(1) / observed.sum(1)
            limits_ok = ((marker_error <= torch.minimum(initial_error + .28, reference_limit))
                         | ~observed).all(1)
            median = torch.nanquantile(marker_error.masked_fill(~observed, float("nan")), .5, dim=1)
            bad_turn = (angular_error(turn[None].expand_as(target_turn), target_turn) > sensor_cap).all(0)
            safe = limits_ok & (median <= 2.5) & (marker_error.masked_fill(~observed, 0.).amax(1) <= 8.)
            safe &= ~(bad_turn & sensor_ok & good).any((1, 2))
            tips = ((joints[:, [4, 8, 12, 16, 20]] - initial_joints[:, [4, 8, 12, 16, 20]]) *
                    1000).norm(dim=-1)
            safe &= ((marker_mse + .00001).sqrt() <= initial_mse.sqrt() + .035) & (tips.amax(1) <= 1.25)
            fraction[(fraction == 0) & safe & ~locked_tensor] = alpha
        best = torch.lerp(old, angles, fraction[:, None, None, None])
        hand_pose, verts, joints, mounts, residual, turns, _ = forward(best)
        np.testing.assert_allclose(joints[:, 0].cpu(), arrays['joints'][frame_slice, 0], atol=2e-6, rtol=0)
        data = dict(hand_pose_pca=hand_pose,
                    vertices=verts,
                    joints=joints,
                    predicted24=mounts,
                    marker_error_mm=residual.norm(dim=-1))
        unchanged = fraction.cpu().numpy() == 0
        for k in data:
            values = data[k].cpu().numpy()
            values[unchanged] = arrays[k][frame_slice][unchanged]
            data[k] = values
        # Save only valid residuals in the same convention as the input. Some previous fits
        # store numeric sentinels for missing markers. Missing entries don't carry observations.
        invalid = ~arrays['marker_valid'][frame_slice]
        data['marker_error_mm'][invalid] = arrays['marker_error_mm'][frame_slice][invalid]
        audit = dict(fraction=fraction,
                     angles_before=old,
                     angles_after=best,
                     turns_before=initial_turns,
                     turns_after=turns,
                     turns_target=target_turn[0],
                     turn_valid=sensor_ok)
        return data, {k: v.cpu().numpy() for k, v in audit.items()}


def refine_hand(args, model, hand):
    name = hand + "_pilot.npz"
    with np.load(args.source / name) as z:
        arrays = dict(z)
    times = arrays['pair_indices']
    with np.load(args.reference / name) as z:
        idx = np.searchsorted(z['pair_indices'], times)
        np.testing.assert_array_equal(z['pair_indices'][idx], times)
        reference = z['marker_error_mm'][idx]
    source_report = json.loads(args.report.read_text())['hands'][hand]
    device = model.faces_tensor.device
    axes = finger_prior.rest_axes(model,
                                  torch.as_tensor(arrays['betas'], device=device),
                                  device,
                                  include_thumb=True)
    limits = [torch.as_tensor(source_report[k], device=device) for k in ('lower_degrees', 'upper_degrees')]
    syncs = source_report['calibration']['syncs']
    fitted = None
    if source_report['calibration'].get('refit_within_bounds'):
        fitted, _ = measured_anchors(model, args.calibrated / name, syncs, axes, limits)
    sensor = RecordedTurns(args.skeleton, int(hand == 'right'), hand, args.calibrated / name, syncs, fitted)
    with np.load(args.skeleton) as z:
        stamp = z['reference_timestamp_ns']
    # Preserve image-constrained poses and any explicitly selected manual reference frame.
    locked = set(source_report.get('visible_contours', {}).get('changed_times', []))
    locked.update(int(t) for h, t in args.keep if h == hand)
    # The next batch always receives the original observations and initial poses in its halo.
    result, diagnostics = {}, {}
    for i in range(0, len(times), args.batch):
        stop = min(i + args.batch, len(times))
        lo, hi = max(0, i - 8), min(stop + 8, len(times))
        data, diag = refine_chunk(model, arrays, sensor, axes, limits, reference, stamp, sorted(locked),
                                  args.steps, slice(lo, hi))
        core = slice(i - lo, stop - lo)
        for dest, piece in ((result, data), (diagnostics, diag)):
            for k, values in piece.items():
                dest.setdefault(k, []).append(values[core].copy())
    for k, pieces in result.items():
        arrays[k] = np.concatenate(pieces)
    for k, pieces in diagnostics.items():
        diagnostics[k] = np.concatenate(pieces)
    for dest, pack in ((args.output / name, arrays), (args.output / (hand + '_motion.npz'),
                                                      dict(times=times, **diagnostics))):
        tmp = dest.with_suffix('.partial.npz')
        np.savez_compressed(tmp, **pack)
        tmp.replace(dest)
    print(hand, len(times), 'adjusted', int((diagnostics['fraction'] > 0).sum()), flush=True)


def write_report(output, source, steps, batch, keep):
    rows = {}
    for hand in ("left", "right"):
        file = output / (hand + "_motion.npz")
        if file.is_file():
            with np.load(file) as z:
                rows[hand] = dict(frames=len(z["times"]), adjusted=int((z["fraction"] > 0).sum()))
    source_report = next(
        (p for p in (source / "summary.json", source.parent / "summary.json") if p.is_file()), None)
    contours = bool(source_report
                    and json.loads(source_report.read_text()).get("strategy") == "joint_refit_v7_contours")
    report = dict(strategy="joint_refit_v9_temporal",
                  source=str(source.resolve()),
                  inherits_visible_contours=contours,
                  marker_allowance_mm=MARKER_ALLOWANCE_MM,
                  individual_change_limit_mm=.28,
                  tip_change_limit_mm=1.25,
                  steps=steps,
                  batch=batch,
                  hands=rows,
                  explicit_locks=keep,
                  approved_for_training=False)
    path = output / "temporal_summary.json"
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(path)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'report', 'reference', 'skeleton', 'calibrated', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--recording-summary',
                    type=Path,
                    help='write the combined per-hand acceptance report for the recording pipeline')
    ap.add_argument('--hands', nargs='+', default=['left', 'right'], choices=['left', 'right'])
    ap.add_argument('--keep', nargs=2, action='append', default=[], metavar=('HAND', 'SYNC'))
    ap.add_argument('--device',
                    choices=['cuda', 'cpu'],
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=46)
    ap.add_argument('--batch', type=int, default=224)
    args = ap.parse_args()
    report = json.loads(args.report.read_text())
    if report.get('strategy') not in ('joint_refit_v6_bounded', 'joint_refit_v7_contours'):
        ap.error('the source report must describe a bounded or contour-preserved fit')
    if args.source.resolve() == args.output.resolve() or min(args.steps, args.batch) < 1:
        ap.error('choose a separate output, positive steps and batch size')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'),
                                                           is_rhand=True,
                                                           use_pca=True,
                                                           num_pca_comps=45,
                                                           flat_hand_mean=False,
                                                           batch_size=1).to(
                                                               args.device).eval().requires_grad_(False)
    for hand in args.hands:
        refine_hand(args, model, hand)
    temporal = write_report(args.output, args.source, args.steps, args.batch,
                            [[h, int(t)] for h, t in args.keep])
    if args.recording_summary:
        report['pre_temporal_strategy'] = report['strategy']
        report.update(strategy=temporal['strategy'], source=str(args.source.resolve()))
        for hand in args.hands:
            report['hands'][hand]['temporal_articulation'] = temporal['hands'][hand]
        temporary = args.recording_summary.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.recording_summary)


if __name__ == '__main__':
    main()
