"""Per-recording correspondence between NOKOV surface-chain turns and MANO finger turns."""
import numpy as np
import torch
import torch.nn.functional as F

from sdk_directions import sdk_links


def local_turns(chain, palm_width):
    """Express each outgoing unit vector in the incoming link's flex/lateral frame."""
    segments = torch.diff(chain, dim=2)
    unit = F.normalize(segments, dim=-1)
    incoming, outgoing = unit[:, :, :-1], unit[:, :, 1:]
    width = F.normalize(palm_width, dim=-1)[:, None, None, :]
    across = width - (width * incoming).sum(-1, keepdim=True) * incoming
    lateral = F.normalize(across, dim=-1)
    flex = torch.cross(lateral, incoming, dim=-1)
    values = torch.stack([(outgoing * x).sum(-1) for x in (incoming, flex, lateral)], dim=-1)
    lengths = segments.norm(dim=-1)
    valid = (across.norm(dim=-1) > .25) & (lengths[:, :, :-1] > .001) & (lengths[:, :, 1:] > .001)
    return values, valid


def angular_error(a, b):
    return torch.rad2deg(torch.atan2(torch.cross(a, b, dim=-1).norm(dim=-1), (a * b).sum(-1)))


def turn_coordinates(directions):
    transverse = directions[..., 1:]
    angle = torch.atan2(transverse.norm(dim=-1), directions[..., 0])
    return torch.rad2deg(angle[..., None] * F.normalize(transverse, dim=-1))


def directions_from_coordinates(coords):
    radians = torch.deg2rad(coords)
    angle = radians.norm(dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle), torch.sinc(angle / torch.pi) * radians], dim=-1)


class RecordedTurns:
    """SDK and measured marker turns must agree before they can supervise a joint."""

    def __init__(self, skeleton, side, hand, calibration, syncs, anchor_joints=None):
        with np.load(skeleton) as sk:
            if sk['geometry_space'].item() != 'physical_camera0':
                raise ValueError('SDK positions must use physical camera0 coordinates')
            parent, child = sdk_links(sk['names'][side], sk['parents'][side], hand)
            nodes = np.column_stack([parent.reshape(5, 3), child.reshape(5, 3)[:, -1]]).ravel()
            mirror = np.array([-1., 1., 1.] if side == 0 else [1., 1., 1.], dtype=np.float32)
            raw = torch.as_tensor(sk['marker24_cam0_m'][side] * mirror, dtype=torch.float32)
            sdk_xyz = torch.as_tensor(sk['positions_cam0_m'][side][:, nodes] * mirror, dtype=torch.float32)
        width = raw[:, 8] - raw[:, 20]
        sdk, sdk_ok = local_turns(sdk_xyz.reshape(-1, 5, 4, 3), width)
        optical, optical_ok = local_turns(raw[:, 4:].reshape(-1, 5, 4, 3), width)
        agree = sdk_ok & optical_ok & (angular_error(sdk, optical) <= 10.)
        consensus = turn_coordinates(F.normalize(sdk + optical, dim=-1))
        with np.load(calibration) as z:
            available = z['pair_indices']
            use = np.isin(available, syncs)
            times = available[use]
            markers = z['target24'][use]
            fitted = z['joints'][use] if anchor_joints is None else np.stack([anchor_joints[int(t)] for t in times])
            reference, valid = local_turns(torch.from_numpy(fitted[:, 1:].reshape(-1, 5, 4, 3)),
                                          torch.from_numpy(markers[:, 8] - markers[:, 20]))
            np.testing.assert_allclose(z['target24_uncorrected'][use], raw[times].numpy(),
                                       atol=1e-6, rtol=0, equal_nan=True)
        valid &= agree[times]
        differences = (turn_coordinates(reference) - consensus[times]).numpy()
        offsets = np.zeros((3, 5, 2, 2), np.float32)
        limits = np.full((5, 2), np.inf, np.float32)
        counts = np.zeros((5, 2), int)
        for digit in range(5):
            for joint in range(2):
                samples = differences[valid[:, digit, joint].numpy(), digit, joint]
                counts[digit, joint] = len(samples)
                if len(samples) < 6:
                    continue
                for i, partition in enumerate((samples, samples[::2], samples[1::2])):
                    offsets[i, digit, joint] = np.median(partition, axis=0)
                leave_one = [np.linalg.norm(x - np.median(np.delete(samples, i, axis=0), axis=0))
                             for i, x in enumerate(samples)]
                limits[digit, joint] = np.percentile(leave_one, 90)
        self.expected = directions_from_coordinates(torch.nan_to_num(consensus)[None] + torch.from_numpy(offsets)[:, None])
        self.valid = agree & torch.from_numpy(np.isfinite(limits))
        self.limits = torch.from_numpy(limits)
        self.raw = raw.numpy()
        self.report = dict(syncs=times.tolist(), usable_by_digit=counts.tolist(),
                           median_offset_flex_side_deg=offsets[0].tolist(),
                           loo_p90_deg=[[float(x) if np.isfinite(x) else None for x in r] for r in limits])
