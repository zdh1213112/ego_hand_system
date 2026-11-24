# Source work/glove_trainset_20261120/assemble.py, SHA-256 8a3ed0077de549ae07c71055819722dd1394268a1dcc5d03ed4bc1b60fe1bc6a

import case_settings as case

from collections import defaultdict

import csv

import numpy as np

OUT = case.stage('glove_trainset_20261120')

WORK = OUT.parent

COMPAT = WORK / "glove_joints_20261120/compat"

HANDS = ("left", "right")

from anatomy_common import signed_flex

def pose_screen(group, records, summary):
    eligible = {(r['sync_index'], r['handedness'])
                for r in records if r['pass_surface'] and not r['isolated_pose']}
    spots = defaultdict(list)
    with (COMPAT.parent / 'metrics' / f'{group}_spots.csv').open() as fp:
        for r in csv.DictReader(fp):
            spots[int(r['sync']), r['hand']].append(r)
    poses = []
    with np.load(COMPAT / group / 'sequence.npz') as seq:
        times = seq['sync_index']
        gloves = seq['joints_3d_glove']
        training = seq['joints_3d_mano']
    for side, hand in enumerate(HANDS):
        with np.load(summary['fit_tracks'][hand]) as tr:
            t = tr['pair_indices']
            j = np.searchsorted(times, t)
            np.testing.assert_array_equal(times[j], t)
            valid = tr['marker_valid'][:, 3:]
            surface = np.linalg.norm(tr['predicted24'][:, 3:] - tr['target24'][:, 3:], axis=-1) * 1000
            residual = np.where(valid, surface, -np.inf).max(1)
        geo = signed_flex(gloves[j, side].astype(float), side == 1)
        mano = signed_flex(training[j, side].astype(float), side == 1)

        def angle_terms(angles):
            delta = (np.diff(angles, axis=0) + 180) % 360 - 180
            spikes = np.zeros_like(angles)
            consecutive = (np.diff(t)[:-1] == 1) & (np.diff(t)[1:] == 1)
            spikes[1:-1] = np.where((delta[:-1] * delta[1:] < 0) & consecutive[:, None, None],
                                    np.minimum(abs(delta[:-1]), abs(delta[1:])), 0)
            distal = angles[:, :, 1:]
            reverse = np.where(distal[:, :, 0] * distal[:, :, 1] < 0, np.abs(distal).min(2), 0)
            return np.maximum(0, -angles.min((1, 2))), reverse.max(1), spikes.max((1, 2))

        hyper, rev, spike = angle_terms(geo)
        mh, mr, ms = angle_terms(mano)
        for i, ti in enumerate(t):
            k = (int(ti), hand)
            if k not in eligible: continue
            assert np.isfinite(geo[i]).all() and np.isfinite(residual[i]) and np.isfinite(mano[i]).all()
            obs = spots.get(k, [])
            spot = max(obs, key=lambda r: float(r['raw_px'])) if obs else None
            values = dict(hyper=float(hyper[i]),
                          opposite=float(rev[i]),
                          surface_mm=float(residual[i]),
                          jump=float(spike[i]),
                          spot_px=float(spot['raw_px']) if spot else None,
                          mano_hyper=float(mh[i]),
                          mano_opposite=float(mr[i]),
                          mano_jump=float(ms[i]))
            thresholds = dict(hyper=25,
                              opposite=15,
                              surface_mm=8,
                              jump=30,
                              spot_px=15,
                              mano_hyper=25,
                              mano_opposite=15,
                              mano_jump=30)
            reasons = [k for k, limit in thresholds.items() if values[k] is not None and values[k] > limit]
            poses.append(
                dict(sync=k[0],
                     hand=hand,
                     reasons=reasons,
                     **values,
                     mano_min_flex=float(mano[i].min()),
                     spot_observations=len(obs),
                     spot_camera=int(spot['camera']) if spot else None,
                     spot_joint=int(spot['joint']) if spot else None))
    assert len(poses) == len(eligible)
    frames = {p['sync'] for p in poses}
    candidates = {p['sync'] for p in poses if p['reasons']}
    excluded = candidates if len(candidates) <= .02 * len(frames) else set()
    return dict(eligible_frames=len(frames),
                eligible_poses=len(poses),
                candidates=sorted(candidates),
                excluded=sorted(excluded),
                poses=poses)
