# Source work/group_003_cuff_fit_20261119/fit_cuff.py, SHA-256 63038ef6ec910da5e863e9104f0ca3f2de94ea47783e434fcd42442a6d19f934

import case_settings as case

import json

import numpy as np

from scipy.spatial.transform import Rotation

from scipy.optimize import brentq

import torch

from torch.nn import functional as F

DEST = case.stage('group_003_cuff_fit_20261119')

PARENT = DEST.parent / "group_003_both_wrists_20261119"

from fit_both import EGO, GROUP, geometry

from fit_sequence import HERE, TIPS, components, compose, finger_prior, import_mano, marker_positions, run_model, save

from observations import project

from wrist_markers import NAMES

def complete_rotations(template, points):
    centered = template - template.mean(0)
    u, singular, vt = np.linalg.svd(centered.T @ (points - points.mean(1, keepdims=True)))
    if np.any(singular[:, 1] < 1e-6):
        raise ValueError("A three-point set does not define a wrist plane")
    flip = np.tile(np.eye(3), (len(points), 1, 1))
    flip[:, 2, 2] = np.linalg.det(u @ vt)
    return (u @ flip @ vt).transpose(0, 2, 1)

def rigid_wrists(mounts, goals, reference_orientation, reference_wrist, times, stamps):
    """Keep the calibrated cuff triangle fixed. Each correspondence retains its recorded name."""
    reference = Rotation.from_rotvec(reference_orientation).as_matrix()
    rotations = reference.copy()
    wrists = reference_wrist.astype(float).copy()
    visible = np.isfinite(goals).all(-1)
    count = visible.sum(-1)
    complete = count == 3
    template = mounts.astype(float)
    x = goals[complete].astype(float)
    rotations[complete] = complete_rotations(template, x)
    # Two points fix an axis. Borrow only the unobserved roll, within a continuous segment.
    # At an isolated gap, the existing pose is the available roll reference.
    positive = (np.diff(times) == 1) & (np.diff(stamps) > 0)
    max_dt = np.median(np.diff(stamps)[positive])*1.5
    breaks = (np.diff(times)!=1) | (np.diff(stamps)<=0) | (np.diff(stamps)>max_dt)
    for block in np.split(np.arange(len(times)), np.flatnonzero(breaks)+1):
        anchors = block[complete[block]]
        for i in block[~complete[block]]:
            if anchors.size:
                near = anchors[np.argmin(np.abs(stamps[anchors]-stamps[i]))]
                if abs(stamps[i]-stamps[near]) <= 150_000_000:
                    rotations[i] = (reference[i] @ reference[near].T) @ rotations[near]
            if count[i] == 2:
                a, b = np.flatnonzero(visible[i])
                src = rotations[i] @ (template[b]-template[a])
                dst = goals[i, b]-goals[i, a]
                if np.linalg.norm(dst) < 1e-5:
                    raise ValueError("Coincident measured cuff markers")
                aligned, _ = Rotation.align_vectors([dst/np.linalg.norm(dst)], [src/np.linalg.norm(src)])
                rotations[i] = aligned.as_matrix() @ rotations[i]
    for k in (1, 2, 3):
        for mask in np.unique(visible[count == k], axis=0):
            ix = (visible == mask).all(-1)
            wrists[ix] = goals[ix][:, mask].mean(1) - np.einsum("nij,j->ni", rotations[ix], template[mask].mean(0))
    return rotations, wrists, visible

def wrist_pivot(hand, prior, cuff_r, cuff_origin, seen, cam, transform, *, anchor=883, calibration_mask=None):
    """A shared cuff-local joint offset, with the horizontal correction read from the named matches at 883."""
    times=prior["pair_indices"]
    use=seen.all(1) & ((times//120)%4==0) & (np.abs(times-anchor)>120)
    if calibration_mask is not None:
        use &= calibration_mask
    if use.sum()<20:raise ValueError("Insufficient complete cuff observations for pivot calibration")
    local=np.einsum("nji,nj->ni",cuff_r,prior["joints"][:,0]-cuff_origin)
    pivot=np.median(local[use],axis=0)
    idx=np.flatnonzero(times==anchor).item(); side=int(hand=="right")
    position=cuff_origin[idx]+cuff_r[idx]@pivot
    target_x=project(cuff_origin[idx:idx+1],transform,cam,side)[0,0]
    def delta(x):
        p=position+np.array([x,0.,0.])
        return float(project(p[None],transform,cam,side)[0,0]-target_x)
    shift=brentq(delta,-.05,.05)
    return pivot+cuff_r[idx].T@np.array([shift,0.,0.]),times[use]

def fit(hand, *, group=GROUP, prior_dir=PARENT, dest=DEST, anchor=883, calibration_mask=None, cuff_template=None):
    with np.load(prior_dir / "tracks" / (hand+".npz")) as z:
        prior = {k: z[k] for k in z.files}
    times = prior["pair_indices"]
    np.testing.assert_array_equal(prior["marker_names"][:3], NAMES)
    with np.load(HERE / group / "aligned" / (hand+"_pilot.npz")) as z:
        original_times = z["pair_indices"]
        indices = np.searchsorted(original_times, times)
        np.testing.assert_array_equal(original_times[indices], times)
        cuff = z["target24"][indices, :3]
        mask = np.isfinite(z["target24_uncorrected"][indices, :3]).all(-1)
        np.testing.assert_array_equal(mask, np.isfinite(cuff).all(-1))
    _, cameras, transforms, frames = geometry(group)
    stamps = np.array([int(frames[int(t)]["reference_timestamp_ns"]) for t in times])
    template = prior["wrist_local"] if cuff_template is None else cuff_template
    rot, origin, seen = rigid_wrists(template, cuff, prior["global_orient"], prior["joints"][:, 0], times, stamps)
    pivot,calibration = wrist_pivot(hand,prior,rot,origin,seen,cameras[0],transforms[0],anchor=anchor,calibration_mask=calibration_mask)
    local = template-pivot
    wrist = origin + np.einsum("nij,j->ni",rot,pivot)
    trans = wrist - (prior["joints"][:, 0] - prior["translation"])
    valid = prior["marker_valid"].copy()
    target = prior["target24"].copy()
    valid[:, :3] = seen
    target[:, :3] = np.nan_to_num(cuff)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    model = import_mano(EGO/"third_party/MANO").load(model_path=str(EGO/"models/mano"), is_rhand=True,
            use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device).eval().requires_grad_(False)
    beta = tensor(prior["betas"])
    axes = finger_prior.rest_axes(model, beta, device, include_thumb=True)
    cfg = json.loads((HERE/group/"bounded/summary.json").read_text())["hands"][hand]
    low, high = (tensor(cfg[k]) for k in ("lower_degrees", "upper_degrees"))
    seed = components(tensor(prior["hand_pose_pca"]), model, axes).clamp(low,high)
    tri = model.faces_tensor.long()[torch.as_tensor(prior["attachment_faces"],device=device).long()]
    attachments = [tensor(prior["barycentric"]),tensor(prior["attachment_local_offsets_m"]),tensor(local)]
    observations = tensor(target[:, 3:])
    ok = torch.as_tensor(valid[:, 3:], device=device)
    # Each camera-frame tip target is the corresponding 3D endpoint in the approved sequence.
    preferred = tensor(prior["joints"][:, 1:])
    rays = F.normalize(preferred,dim=-1)
    tips = tensor(prior["joints"][:, TIPS])
    orient, translation, rotation = map(tensor, (prior["global_orient"], trans, rot))
    cuff_pivot = tensor(wrist)
    sdk_offset = tensor(prior["root_anchor_local_m"])
    sdk_goal = tensor(prior["finger0_target"])
    from mano.lbs import batch_rodrigues
    weights = tensor([.7, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2])
    result = {k:np.empty_like(prior[k]) for k in ("vertices","joints","hand_pose_pca","predicted24","global_orient","translation")}
    step_ns = np.median(np.diff(stamps)[np.diff(times)==1])
    for start in range(0,len(times),128):
        end = min(len(times),start+128)
        lo,hi = max(0,start-5),min(len(times),end+5)
        sl = slice(lo,hi)
        links = np.flatnonzero((np.diff(times[sl])==1)&(np.diff(stamps[sl])>0)&(np.diff(stamps[sl])<1.5*step_ns))
        links = torch.as_tensor(links,device=device)
        angles = torch.nn.Parameter(seed[sl].clone())
        palm_delta = torch.nn.Parameter(torch.zeros_like(orient[sl]))
        joint_slip = torch.nn.Parameter(torch.zeros_like(translation[sl]))
        optim = torch.optim.Adam([dict(params=[angles],lr=.58),dict(params=[palm_delta],lr=.004),dict(params=[joint_slip],lr=.17)])
        steps = 105
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim,steps,eta_min=.0002)
        def forward():
            pose = compose(angles,model,axes)
            palm = orient[sl] + palm_delta
            shift=translation[sl]+joint_slip/1000
            verts,joints,_ = run_model(model,beta.expand(hi-lo,-1),palm,pose,shift)
            mounts = marker_positions(verts,cuff_pivot[sl],rotation[sl],tri,*attachments)
            return verts,joints,pose,mounts,palm,shift
        for step in range(steps):
            optim.zero_grad(set_to_none=True)
            _,joints,_,mounts,palm,_ = forward()
            e = (mounts[:,3:]-observations[sl])*1000.
            measured = (torch.sqrt(1+e.square().sum(-1)/16)-1)*32
            w = weights*ok[sl]
            loss = (measured*w).sum()/w.sum()
            ray = rays[sl]
            delta = (joints[:,1:]-preferred[sl])*1000.
            tangent = delta-(ray*delta).sum(-1,keepdim=True)*ray
            tip_delta = (joints[:,TIPS]-tips[sl])*1000.
            tip_weight = 6. if step < steps//2 else 28.
            loss += .28*tangent.square().mean() + tip_weight*tip_delta.square().mean()
            correction = angles-seed[sl]
            loss += .035*correction.square().mean() + .07*correction[:,:4,1:,[0,2]].square().mean()
            root = joints[:,0] + (batch_rodrigues(palm) @ sdk_offset)
            loss += .12*((root-sdk_goal[sl])*1000).square().mean() + .035*torch.rad2deg(palm_delta).square().mean()
            loss += 1.5*joint_slip.square().mean()
            if len(links):
                loss += .16*(correction[links+1]-correction[links]).square().mean()
                loss += .15*torch.rad2deg(palm_delta[links+1]-palm_delta[links]).square().mean()
                loss += .8*(joint_slip[links+1]-joint_slip[links]).square().mean()
            loss.backward(); optim.step(); schedule.step()
            with torch.no_grad():
                angles.clamp_(low,high)
        with torch.no_grad():
            for name,value in zip(("vertices","joints","hand_pose_pca","predicted24","global_orient","translation"),forward()):
                result[name][start:end]=value[start-lo:end-lo].cpu().numpy()
        if start%768==0 or end==len(times):
            print(hand,end,"/",len(times),flush=True)
    np.testing.assert_allclose(result["joints"][:,0]-result["translation"],prior["joints"][:,0]-prior["translation"],rtol=0,atol=2e-6)
    fitted = dict(prior, **result)
    fitted.update(wrist_local=local.astype(np.float32), cuff_pivot_camera0_canonical_m=wrist.astype(np.float32),
                  cuff_global_orient=Rotation.from_matrix(rot).as_rotvec().astype(np.float32), cuff_frame_pivot_m=pivot,
                  cuff_local_calibration_sync=calibration,
                  cuff_marker_frame=np.array("measured_rigid_cuff_coupled_to_anatomical_wrist"),
                  target24=target,marker_valid=valid,cuff_observed=seen,approved_for_training=np.array(False))
    err = np.linalg.norm(result["predicted24"]-target,axis=-1)*1000
    fitted["marker_error_mm"] = np.where(valid,err,np.nan).astype(np.float32)
    recover = result["joints"][:,0]+Rotation.from_rotvec(result["global_orient"]).apply(np.broadcast_to(prior["root_anchor_local_m"],wrist.shape))
    fitted["root_anchor_error_mm"]=(np.linalg.norm(recover-prior["finger0_target"],axis=-1)*1000).astype(np.float32)
    np.testing.assert_allclose(result["predicted24"][:,:3], wrist[:,None]+local@rot.transpose(0,2,1), atol=1e-6,rtol=0)
    (dest/"initial_fit").mkdir(parents=True,exist_ok=True)
    save(dest/"initial_fit"/(hand+".npz"),fitted)
    print(hand,"cuff mm median/p95",np.percentile(err[:,:3][seen],[50,95]).tolist(),"surface",np.nanmedian(fitted["marker_error_mm"][:,3:]), flush=True)
