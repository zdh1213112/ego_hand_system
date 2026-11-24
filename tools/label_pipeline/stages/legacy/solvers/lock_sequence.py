# Source work/group_003_cuff_fit_20261119/lock_sequence.py, SHA-256 72f40e966faacc8055e4c2b53cec20adbec48fdfc0b455743e82add71f5db5b2

import case_settings as case

import json

import numpy as np

from scipy.spatial.transform import Rotation

import torch

from torch.nn import functional as F

from fit_cuff import DEST, EGO, GROUP, HERE, PARENT, TIPS, components, compose, finger_prior, import_mano, marker_positions, run_model, save

class TipKinematics:
    """Evaluate exact MANO skinning at its five terminal vertices, with shared shape cached."""
    def __init__(self,model,beta,axes):
        from mano.lbs import blend_shapes
        self.model,self.axes=model,axes
        ids=list(model.tip_ids.values())
        assert list(model.tip_ids)==["thumb","index","middle","ring","pinky"] and not model.use_pca
        shaped=model.v_template+blend_shapes(beta[None],model.shapedirs)
        self.rest=model.J_regressor@shaped[0]
        self.tips=shaped[0,ids]
        self.blends=model.posedirs.reshape(135,-1,3)[:,ids].reshape(135,15)
        self.weights=model.lbs_weights[ids]
        self.eye=torch.eye(3,device=beta.device)

    def positions(self,q):
        from mano.lbs import batch_rodrigues, batch_rigid_transform
        n=len(q)
        angles=q[:,:45].reshape(n,5,3,3)
        full=torch.cat((torch.deg2rad(q[:,45:48]),compose(angles,self.model,self.axes)+self.model.hand_mean),1)
        r=batch_rodrigues(full.reshape(-1,3)).view(n,16,3,3)
        _,a=batch_rigid_transform(r,self.rest.expand(n,-1,-1),self.model.parents)
        posed=self.tips+((r[:,1:]-self.eye).flatten(1)@self.blends).reshape(n,5,3)
        skin=(self.weights@a.reshape(n,16,16)).reshape(n,5,4,4)
        return (skin@F.pad(posed,[0,1],value=1)[...,None])[:,:,:3,0]*1000+q[:,None,48:]

    def jacobian(self,q):
        n=len(q);eps=.055
        directions=torch.eye(51,device=q.device)[:48]*eps
        samples=torch.cat((q[:,None,:]+directions,q[:,None,:]-directions),1)
        pos=self.positions(samples.reshape(-1,51)).reshape(n,96,15)
        jac=(pos[:,:48]-pos[:,48:]).transpose(1,2)/(2*eps)
        translation=self.eye.repeat(5,1).expand(n,-1,-1)
        return torch.cat((jac,translation),dim=2)

def project_constraints(kin,q,target,low,high,mobility,steps):
    """Minimum weighted-change Gauss–Newton, with an active set for anatomical limits."""
    q=q.clone()
    identity=torch.eye(15,device=q.device)
    for _ in range(steps):
        residual=kin.positions(q)-target
        active=residual.norm(dim=-1).amax(1)>.002
        if not active.any():break
        x=q[active]; goal=target[active];base=residual[active].flatten(1)
        jac=kin.jacobian(x)
        w=mobility.expand(len(x),-1).clone()
        for _ in range(4):
            jt=w[:,:,None]*jac.transpose(1,2)
            delta=-(jt@torch.linalg.solve(jac@jt+.00009*identity,base[:,:,None])).squeeze(-1)
            block=((x<=low+.003)&(delta<0))|((x>=high-.003)&(delta>0))
            if not (block&(w>0)).any():break
            w.masked_fill_(block,0)
        factor=6./delta[:,:48].abs().amax(1,keepdim=True).clamp_min(6.)
        factor=torch.minimum(factor,2./delta[:,48:].abs().amax(1,keepdim=True).clamp_min(2.))
        delta=delta*factor
        scales=x.new_tensor([1,.5,.25,.125])
        trial=(x[:,None]+delta[:,None]*scales[None,:,None]).clamp(low,high)
        errs=(kin.positions(trial.reshape(-1,51)).reshape(-1,4,5,3)-goal[:,None]).square().sum((2,3))
        v,k=errs.min(1); best=trial[torch.arange(len(x),device=q.device),k]
        q[active]=torch.where((v<base.square().sum(1))[:,None],best,x)
    return q

def continuous_seed(before, solved, indices, timestamps, hand):
    """Retain the four non-thumb digits' outward-pose prior and damp isolated angle changes."""
    seed = solved.clone()
    mix = .68 if hand == "left" else .60
    # MANO's local digit order is index, middle, little, ring, thumb.
    seed[:, :36] = before[:, :36] + mix * (solved[:, :36] - before[:, :36])
    seed[:, 45:48] = before[:, 45:48] + mix * (solved[:, 45:48] - before[:, 45:48])
    dt = np.diff(timestamps).astype(float)
    median_dt = np.median(dt[np.diff(indices) == 1])
    continuous = (np.diff(indices) == 1) & (dt >= .8 * median_dt) & (dt <= 1.2 * median_dt)
    mid = np.flatnonzero(continuous[:-1] & continuous[1:]) + 1
    if not len(mid):
        return seed
    alpha = seed.new_tensor(dt[mid-1] / (dt[mid-1] + dt[mid]))[:, None]
    # Finger angles describe the physical pose. Smooth only the small global wrist correction
    # relative to the original moving hand so its recorded motion and the overall timing are preserved.
    local = seed.clone()
    local[:, 45:] -= before[:, 45:]
    interp = torch.lerp(local[mid-1], local[mid+1], alpha)
    correction = .62 * (interp - local[mid])
    caps = seed.new_tensor([2.5]*45 + [1.4]*3 + [.7]*3)
    seed[mid] += correction.clamp(-caps, caps)
    return seed

@torch.no_grad()
def lock(hand, continuous=False, *, group=GROUP, prior_dir=PARENT, dest=DEST):
    output = dest / "continuous" if continuous else dest
    with np.load(prior_dir/"tracks"/(hand+".npz")) as z:
        prior={k:z[k] for k in ("pair_indices","joints","global_orient","hand_pose_pca","translation")}
    with np.load(dest/("tracks" if continuous else "initial_fit")/(hand+".npz")) as z:
        fit={k:z[k] for k in z.files}
    np.testing.assert_array_equal(prior["pair_indices"],fit["pair_indices"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    model=import_mano(EGO/"third_party/MANO").load(model_path=str(EGO/"models/mano"),is_rhand=True,
          use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device).eval().requires_grad_(False)
    beta=tensor(fit["betas"]);axes=finger_prior.rest_axes(model,beta,device,include_thumb=True); kin=TipKinematics(model,beta,axes)
    def pack(z):
        ang=components(tensor(z["hand_pose_pca"]),model,axes)
        return torch.cat((ang.flatten(1),torch.rad2deg(tensor(z["global_orient"])),tensor(z["translation"])*1000),1)
    before=pack(prior); goal=pack(fit)
    if continuous:
        with np.load(dest / "sequence.npz") as seq:
            slots = np.searchsorted(seq["sync_index"], fit["pair_indices"])
            np.testing.assert_array_equal(seq["sync_index"][slots], fit["pair_indices"])
            goal = continuous_seed(before, goal, fit["pair_indices"], seq["reference_timestamp_ns"][slots], hand)
    target=tensor(prior["joints"][:,TIPS])*1000
    assert float((kin.positions(before)-target).abs().max()) < .0008
    limits=json.loads((HERE/group/"bounded/summary.json").read_text())["hands"][hand]
    low=torch.cat((tensor(limits["lower_degrees"]).flatten(),torch.full((6,),-float("inf"),device=device)))
    high=torch.cat((tensor(limits["upper_degrees"]).flatten(),torch.full((6,),float("inf"),device=device)))
    mobility=torch.ones(51,device=device)
    mw=mobility[:45].view(5,3,3);mw[:4,1:,0]=.14;mw[:4,1:,2]=.20;mw[:4,0,0]=.5
    mobility[45:48]=.6;mobility[48:]=0
    movable=mobility.clone();movable[48:]=.16
    n=len(before);locked=before.clone();fraction=np.zeros(n,np.float32)
    for start in range(0,n,80):
        stop=min(start+80,n); sl=slice(start,stop); a,b=before[sl],goal[sl]; g=target[sl]
        current=project_constraints(kin,b,g,low,high,mobility,13)
        active=(kin.positions(current)-g).norm(dim=-1).amax(1)>.004
        if active.any():
            current[active]=project_constraints(kin,current[active],g[active],low,high,movable,22)
        progress=np.ones(len(a),np.float32)
        for scale in (.65,.32,.12):
            bad=(kin.positions(current)-g).norm(dim=-1).amax(1)>.004
            if not bad.any():break
            current[bad]=project_constraints(kin,a[bad]+scale*(b[bad]-a[bad]),g[bad],low,high,movable,19)
            progress[bad.cpu().numpy()]=scale
        bad=(kin.positions(current)-g).norm(dim=-1).amax(1)>.004
        current[bad]=a[bad];progress[bad.cpu().numpy()]=0
        locked[sl]=current;fraction[sl]=progress
        if start%640==0 or stop==n:print(hand,"locked",stop,"/",n,flush=True)
    assert torch.all(locked>=low-.001) and torch.all(locked<=high+.001)
    track={k:v for k,v in fit.items()}
    track["translation"]=(locked[:,48:]/1000).cpu().numpy()
    track["global_orient"]=torch.deg2rad(locked[:,45:48]).cpu().numpy()
    track["hand_pose_pca"]=compose(locked[:,:45].reshape(n,5,3,3),model,axes).cpu().numpy()
    # The sleeve and palm are separate bodies. The wrist target from the rigid cuff participates
    # in the whole-hand solve. Preserve their final physical offset for inspection and replay.
    j0=track["translation"] + fit["joints"][:,0]-fit["translation"]
    cuff_r=Rotation.from_rotvec(fit["cuff_global_orient"]).as_matrix()
    cuff_pivot=fit["cuff_pivot_camera0_canonical_m"]
    track["joint_minus_cuff_pivot_local_m"]=np.einsum("nji,nj->ni",cuff_r,j0-cuff_pivot).astype(np.float32)
    tri=model.faces_tensor.long()[torch.as_tensor(fit["attachment_faces"],device=device).long()]
    mounts=[tensor(fit[k]) for k in ("barycentric","attachment_local_offsets_m","wrist_local")]
    for k in ("vertices","joints","predicted24"):track[k]=np.empty_like(fit[k])
    for start in range(0,n,128):
        sl=slice(start,min(n,start+128));count=len(locked[sl])
        verts,joints,_=run_model(model,beta.expand(count,-1),*[tensor(track[k][sl]) for k in ("global_orient","hand_pose_pca","translation")])
        mounted=marker_positions(verts,tensor(cuff_pivot[sl]),tensor(cuff_r[sl]),tri,*mounts)
        for k,val in (("vertices",verts),("joints",joints),("predicted24",mounted)):track[k][sl]=val.cpu().numpy()
    drift=np.linalg.norm(track["joints"][:,TIPS]-prior["joints"][:,TIPS],axis=-1)*1000
    if drift.max()>.005:raise ValueError(f"Final MANO mesh misplaced a fingertip: {drift.max()} mm")
    track["locked_tip_reference_camera0_canonical_m"]=prior["joints"][:,TIPS].copy()
    track["locked_tip_drift_mm"]=drift; track["tip_lock_seed_fraction"]=fraction
    if continuous:
        track["continuous_finger_refinement"] = np.array(True)
    track["marker_error_mm"]=np.where(fit["marker_valid"],np.linalg.norm(track["predicted24"]-fit["target24"],axis=-1)*1000,np.nan).astype(np.float32)
    root=track["joints"][:,0]+Rotation.from_rotvec(track["global_orient"]).apply(np.broadcast_to(track["root_anchor_local_m"],j0.shape))
    track["root_anchor_error_mm"]=(np.linalg.norm(root-fit["finger0_target"],axis=-1)*1000).astype(np.float32)
    (output/"tracks").mkdir(parents=True,exist_ok=True)
    save(output/"tracks"/(hand+".npz"),track)
    print(hand,"tip mm max",float(drift.max()),"entire pose fallback",int((fraction==0).sum()),
          "cuff median/p95",np.nanpercentile(track["marker_error_mm"][:,:3],[50,95]).tolist(),
          "surface median",float(np.nanmedian(track["marker_error_mm"][:,3:])),flush=True)
