"""Use the recorded NOKOV phalanx directions in the MANO palm frame, with one profile per recording."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation, Slerp

TOOLS = Path(__file__).resolve().parents[2] / 'label_export/pipeline/calibration'
sys.path.insert(0, str(TOOLS / 'sdk_direction_fit'))
from build_mapping import MP_PARENT, MP_CHILD, angular_deg, proper_rotation, unit
from context import load_context, load_rigid

FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
WEIGHT = .10
PROFILE_FORMAT = 'nokov_phalanx_in_mano_palm_v1'


def rigid_motion(before, after):
    """Column-vector rotation and translation taking one observed marker constellation to the other."""
    good = np.isfinite(before).all(1) & np.isfinite(after).all(1)
    x, y = before[good].astype(float), after[good].astype(float)
    if len(x) < 3 or np.linalg.matrix_rank(x - x.mean(0), tol=1e-6) < 2:
        raise ValueError('Need three noncollinear common markers for the saved image correction')
    rot = proper_rotation((y-y.mean(0)).T @ (x-x.mean(0)))
    delta = y.mean(0) - rot @ x.mean(0)
    np.testing.assert_allclose(x @ rot.T + delta, y, rtol=0, atol=1e-6)
    return rot, delta


def sdk_links(names, parents, hand):
    names = list(names)
    prefix = hand.title() + 'Hand'
    chains = np.array([[names.index(prefix+finger+suffix) for suffix in ('1','2','3','End')] for finger in FINGERS])
    np.testing.assert_array_equal(parents[chains[:,1:]], chains[:,:-1])
    return chains[:,:-1].ravel(), chains[:,1:].ravel()


def head_to_camera(context, timestamps):
    alignment = json.loads(context['alignment'].read_text())
    seed = json.loads(Path(alignment['seed_alignment']).read_text())['time_mapping']
    rel = (timestamps - seed['ego_origin_timestamp_ns']) * 1e-9
    target = seed['nokov_origin_timestamp_raw'] + (rel * seed.get('a', 1.) + seed['b_s']) / seed['nokov_seconds_per_timestamp_unit']
    rt, _, quat = load_rigid(Path(alignment['source_nokov']) / 'nokov_rigid_bodies.csv')
    valid = (target>=rt[0]) & (target<=rt[-1])
    matrices = np.full((len(target),3,3),np.nan)
    head = Slerp(rt, Rotation.from_quat(quat))(target[valid]).as_matrix()
    matrices[valid] = np.asarray(alignment['T_cam0_head'])[:3,:3] @ head.transpose(0,2,1)
    return matrices


def make_profile(context, skeleton, selection, calibrated, output):
    with np.load(skeleton) as z:
        if z['geometry_space'].item() != 'physical_camera0':
            raise ValueError('Expected the SDK and optical markers in physical camera0')
        xyz, q, raw, stamp = [z[k] for k in ('positions_cam0_m','sdk_quaternions_xyzw','marker24_cam0_m','reference_timestamp_ns')]
        names, parents = z['names'], z['parents']
    world_to_cam = head_to_camera(load_context(context), stamp)
    selected = json.loads(selection.read_text())['report']['hands']
    dirs = np.full((2,len(stamp),15,3),np.nan,dtype=np.float32)
    confidence = np.zeros((2,15),np.float32)
    report = {}
    for side,hand in enumerate(('left','right')):
        path = calibrated / f'{hand}_pilot.npz'
        if not path.is_file():
            continue
        a,b = sdk_links(names[side],parents[side],hand)
        palm_id = names[side].tolist().index(hand.title()+'Hand')
        reflection = np.diag([-1.,1.,1.] if side==0 else [1.,1.,1.])
        q_hand = q[side,:,palm_id]
        ok = np.isfinite(world_to_cam).all((1,2)) & np.isfinite(q_hand).all(1) & (np.linalg.norm(q_hand,axis=-1)>.5)
        sdk_palm = np.full_like(world_to_cam,np.nan)
        sdk_palm[ok] = reflection @ world_to_cam[ok] @ Rotation.from_quat(q_hand[ok]).as_matrix() @ reflection
        bones = (xyz[side][:,b] - xyz[side][:,a]) @ reflection
        bones[np.linalg.norm(bones,axis=-1)<.001] = np.nan
        local = np.einsum('fji,fkj->fki',sdk_palm,unit(bones))
        requested = list(map(int,selected[hand]['calibration_sync']))
        with np.load(path) as z:
            lookup = {int(s):i for i,s in enumerate(z['pair_indices'])}
            cal = np.array([s for s in requested if s in lookup and np.isfinite(local[s]).all()],dtype=int)
            if len(cal)<3:
                raise ValueError(f'{hand} has fewer than three usable calibration frames')
            ix = [lookup[s] for s in cal]
            corrected, original = z['target24'][ix], z['target24_uncorrected'][ix]
            np.testing.assert_allclose(original,raw[side,cal] @ reflection,atol=1e-6,rtol=0,equal_nan=True)
            image_rotation = np.stack([rigid_motion(x,y)[0] for x,y in zip(original,corrected)])
            mano = Rotation.from_rotvec(z['global_orient'][ix]).as_matrix()
            joint = z['joints'][ix]
        # Local MANO directions are rigid-invariant. Undo image alignment for the palm-frame calibration only.
        y = np.einsum('fji,fkj->fki',mano,unit(joint[:,MP_CHILD]-joint[:,MP_PARENT]))
        mano_raw = image_rotation.transpose(0,2,1) @ mano
        common = proper_rotation((mano_raw.transpose(0,2,1) @ sdk_palm[cal]).mean(0))
        corrections = np.stack([proper_rotation(y[:,k].T @ local[cal,k]/len(cal)+.25*common) for k in range(15)])
        mapped = np.einsum('kij,fkj->fki',corrections,local)
        ang = np.median(angular_deg(mapped[cal],y),axis=0)
        confidence[side] = 1. / (1. + (ang/10.)**2)
        dirs[side] = mapped
        report[hand] = dict(calibration_sync=cal.tolist(),segments=[[str(names[side,k]),str(names[side,l])] for k,l in zip(a,b)],
                            calibration_median_deg=ang.tolist(),confidence=confidence[side].tolist(),
                            sdk_to_mano_palm=common.tolist(),per_segment_corrections=corrections.tolist())
    if not report:
        raise ValueError('No calibrated hands')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'directions.npz',format=PROFILE_FORMAT,local_directions=dirs,confidence=confidence,
                        raw_marker24_cam0_m=raw.astype(np.float32))
    (output/'mapping.json').write_text(json.dumps(dict(source_skeleton=str(skeleton.resolve()),
        calibration=str(calibrated.resolve()),weight=WEIGHT,hands=report),indent=2,allow_nan=False)+'\n')


class DirectionTargets:
    def __init__(self, profile, side, syncs, raw_canonical, device):
        with np.load(profile) as z:
            if z['format'].item()!=PROFILE_FORMAT:
                raise ValueError('Unexpected SDK direction profile')
            all_targets = z['local_directions'][side]
            if np.any(syncs<0) or np.any(syncs>=len(all_targets)):
                raise ValueError('Sync indices fall outside the SDK recording')
            vectors = all_targets[syncs]
            conf = z['confidence'][side]
            evidence = z['raw_marker24_cam0_m'][side,syncs] * ([-1.,1.,1.] if side==0 else [1.,1.,1.])
        good = np.isfinite(raw_canonical).all(-1)
        if not good.any() or not np.allclose(raw_canonical[good],evidence[good],rtol=0,atol=1e-6):
            raise ValueError('SDK profile and original marker samples refer to different recordings or coordinates')
        valid = np.isfinite(vectors).all(-1)
        if not np.allclose(np.linalg.norm(vectors[valid],axis=-1),1.,atol=1e-4) or not np.any(conf>0):
            raise ValueError('Invalid SDK directions or missing hand profile')
        self.rows = torch.as_tensor(np.nan_to_num(vectors),device=device,dtype=torch.float32)
        self.weights = torch.as_tensor(valid * conf,device=device,dtype=torch.float32)
        self.normalizer = self.weights.sum().clamp_min(1.)
        self.parents = torch.as_tensor(MP_PARENT,device=device)
        self.children = torch.as_tensor(MP_CHILD,device=device)

    def penalty(self, joints, root_rotation):
        unit_bone = F.normalize(joints[:,self.children]-joints[:,self.parents],dim=-1)
        local = unit_bone @ root_rotation
        sq = (local-self.rows).square().sum(-1)
        return ((torch.sqrt(1. + sq / np.deg2rad(10.)**2) - 1.) * self.weights).sum() / self.normalizer


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('context','skeleton','selection','calibrated','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    make_profile(**vars(ap.parse_args()))
