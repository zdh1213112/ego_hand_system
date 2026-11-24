# Source work/group_003_right_mount_study_20261119/refit.py, SHA-256 196440420583e8da5a99779aa08f9babb412bc842d2b1d36df2f25083c1db11b

import case_settings as case

import json

import numpy as np

from scipy.spatial.transform import Rotation

import torch

from torch.nn import functional as F

from study import BASE, EGO, GROUP, HERE, PARENT, ROOT, TIPS, load, save

from fit_cuff import components, compose, finger_prior, import_mano, marker_positions, run_model

from lock_sequence import TipKinematics, project_constraints

def run(variant, *, hand="right", output_dir=None, relaxed_ulnar=False, group=GROUP, base=BASE,
        prior_dir=PARENT, mount_file=None, timestamps=None, cap_guard_mm=0.):
    hard = variant != "soft"
    if hand not in ("right", "left") or (relaxed_ulnar and (hard or hand != "right")):
        raise ValueError("The per-finger 5 mm experiment requires a soft right-hand fit")
    folder = ROOT / variant if output_dir is None else output_dir
    z=load(base/"tracks"/(hand+".npz"))
    with np.load(prior_dir/"tracks"/(hand+".npz")) as approved:
        ref=approved["joints"].copy()
        assert np.array_equal(approved["pair_indices"],z["pair_indices"])
    if mount_file is not None:
        z.update(load(mount_file))
    elif variant=="effective_control" and hand=="right":
        z["attachment_local_offsets_m"]=load(ROOT/"mount_parameters.npz")["effective_offsets_all_frames_m"].astype(np.float32)
    elif variant in ("hard","soft") and hand=="right":
        z.update(load(ROOT/"surface_mounts.npz"))
    z["normal_offsets_m"]=z["attachment_local_offsets_m"][:,2].copy()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=device)
    model=import_mano(EGO/"third_party/MANO").load(model_path=str(EGO/"models/mano"),is_rhand=True,
        use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device).eval().requires_grad_(False)
    from mano.lbs import batch_rodrigues
    beta=tensor(z["betas"]); axes=finger_prior.rest_axes(model,beta,device,include_thumb=True); tipkin=TipKinematics(model,beta,axes)
    limits=json.loads((HERE/group/"bounded/summary.json").read_text())["hands"][hand]
    low,high=[tensor(limits[k]) for k in ("lower_degrees","upper_degrees")]
    loq=torch.cat((low.flatten(),tensor([-np.inf]*6)));hiq=torch.cat((high.flatten(),tensor([np.inf]*6)))
    seed=components(tensor(z["hand_pose_pca"]),model,axes)
    ori=tensor(z["global_orient"]);trans=tensor(z["translation"]);init_joints=tensor(z["joints"])
    target=tensor(z["target24"][:,3:]);valid=tensor(z["marker_valid"][:,3:])
    tip=tensor(ref[:,TIPS])*1000
    kink = tensor([3., 3., 3., 4., 4.] if relaxed_ulnar else [case.params['tip_kink_mm']] * 5)
    cap = tensor([3.5, 3.5, 3.5, 5., 5.] if relaxed_ulnar else [case.params['tip_cap_mm']] * 5)
    cuff=tensor(z["cuff_pivot_camera0_canonical_m"]);cuff_rot=tensor(Rotation.from_rotvec(z["cuff_global_orient"]).as_matrix())
    sdk=tensor(z["finger0_target"]);sdk_local=tensor(z["root_anchor_local_m"]);cuff_local=tensor(z["wrist_local"])
    tri=model.faces_tensor.long()[torch.as_tensor(z["attachment_faces"],device=device).long()]
    bar=tensor(z["barycentric"]);offset=tensor(z["attachment_local_offsets_m"])
    times=z["pair_indices"]
    if timestamps is None:
        with np.load(base/"sequence.npz") as seq:
            ids=np.searchsorted(seq["sync_index"],times);stamps=seq["reference_timestamp_ns"][ids]
    else:
        stamps=np.asarray(timestamps)
        if stamps.shape!=times.shape:raise ValueError("One timestamp per measured pose is required")
    cuts=(np.diff(times)==1)&(np.diff(stamps)>0)
    cuts &= np.diff(stamps)<1.5*np.median(np.diff(stamps)[cuts])
    mobility=torch.ones(51,device=device);mw=mobility[:45].view(5,3,3)
    mw[:4,1:,[0,2]]=.20;mobility[45:48]=.65;mobility[48:]=.20
    allq=[]; corrections=[];fallback=0
    with torch.no_grad():
        q0=torch.cat((seed.flatten(1),torch.rad2deg(ori),trans*1000),1)
        assert float((tipkin.positions(q0)-init_joints[:,TIPS]*1000).abs().max())<.001
    for s in range(0,len(times),144):
        e=min(s+144,len(times));a=max(s-4,0);b=min(e+4,len(times));sl=slice(a,b)
        angles=torch.nn.Parameter(seed[sl].clone())
        rot_delta=torch.nn.Parameter(torch.zeros_like(ori[sl]))
        delta_mm=torch.nn.Parameter(torch.zeros_like(trans[sl]))
        opt=torch.optim.Adam([dict(params=[angles],lr=.63),dict(params=[rot_delta],lr=.50),dict(params=[delta_mm],lr=.40)])
        steps=150; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.003)
        links=torch.as_tensor(np.flatnonzero(cuts[a:b-1]),device=device)
        lag=torch.zeros_like(tip[sl]);mask=valid[sl]
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            pose=compose(angles,model,axes);palm=ori[sl]+torch.deg2rad(rot_delta)
            v,j,_=run_model(model,beta.expand(b-a,-1),palm,pose,trans[sl]+delta_mm/1000)
            mounts=marker_positions(v,cuff[sl],cuff_rot[sl],tri,bar,offset,cuff_local)
            err2=((mounts[:,3:]-target[sl])*1000).square().sum(-1)*mask
            loss=err2.sum()/mask.sum()+.18*err2.topk(2,dim=1).values.mean()
            loss+=.28*((j[:,0]-cuff[sl])*1000).square().sum(-1).mean()
            anchor=j[:,0]+(batch_rodrigues(palm)@sdk_local)
            loss+=.10*((anchor-sdk[sl])*1000).square().sum(-1).mean()
            move=(j[:,1:]-init_joints[sl,1:])*1000
            rays=F.normalize(init_joints[sl,1:],dim=-1)
            loss+=.08*(move-(move*rays).sum(-1,keepdim=True)*rays).square().sum(-1).mean()
            change=angles-seed[sl];loss+=.025*change.square().mean()+.055*change[:,:4,1:,[0,2]].square().mean()+.026*rot_delta.square().mean()
            if len(links):
                loss+=.09*(change[links+1]-change[links]).square().mean()+.10*(rot_delta[links+1]-rot_delta[links]).square().mean()+.20*(delta_mm[links+1]-delta_mm[links]).square().mean()
            diff=j[:,TIPS]*1000-tip[sl]
            if hard:
                rho=1.5 if it<50 else (4. if it<100 else 12.)
                loss+=(lag*diff+.5*rho*diff.square()).sum(-1).mean()
                if it in (34,64,94,119,137):lag=(lag+rho*diff.detach()).clamp(-90,90)
            else:
                dist=diff.norm(dim=-1)
                loss+=.22*dist.square().mean()+13.*F.relu(dist-kink).square().mean()
            loss.backward();opt.step();sched.step()
            with torch.no_grad():angles.clamp_(low,high)
        with torch.no_grad():
            keep=slice(s-a,e-a)
            q=torch.cat((angles.flatten(1),torch.rad2deg(ori[sl])+rot_delta,trans[sl]*1000+delta_mm),dim=1)[keep]
            pieces=[]
            for start in range(0,len(q),64):
                p=q[start:start+64];g=tip[s+start:s+start+len(p)];before=tipkin.positions(p)
                away=before-g;maximum=away.norm(dim=-1).amax(1)
                if hard:
                    goal=g
                    fix=maximum>.002
                else:
                    radius=(cap-.02)[:,None]
                    goal=g+away*(radius/torch.maximum(away.norm(dim=-1,keepdim=True),radius))
                    fix=(away.norm(dim=-1)>cap-cap_guard_mm).any(1)
                if fix.any():
                    p=p.clone();p[fix]=project_constraints(tipkin,p[fix],goal[fix],loq,hiq,mobility,28)
                    if hard:
                        bad=(tipkin.positions(p)-g).norm(dim=-1).amax(1)>.004
                        if bad.any():
                            refq=q0[s+start:s+start+len(p)][bad]
                            p[bad]=project_constraints(tipkin,.5*(refq+p[bad]),g[bad],loq,hiq,mobility,35)
                            bad=(tipkin.positions(p)-g).norm(dim=-1).amax(1)>.004
                            if bad.any():p[bad]=q0[s+start:s+start+len(p)][bad];fallback+=int(bad.sum())
                corrections.extend(fix.cpu().tolist());pieces.append(p)
            allq.append(torch.cat(pieces))
        if s%720==0 or e==len(times):print(variant,e,"/",len(times),flush=True)
    with torch.no_grad():
        allq=torch.cat(allq);assert torch.all(allq>=loq-.001) and torch.all(allq<=hiq+.001)
        z["hand_pose_pca"]=compose(allq[:,:45].reshape(-1,5,3,3),model,axes).cpu().numpy()
        z["global_orient"]=torch.deg2rad(allq[:,45:48]).cpu().numpy();z["translation"]=(allq[:,48:]/1000).cpu().numpy()
        for s in range(0,len(times),128):
            e=min(s+128,len(times));sl=slice(s,e)
            v,j,_=run_model(model,beta.expand(e-s,-1),*[tensor(z[key][sl]) for key in ("global_orient","hand_pose_pca","translation")])
            mounts=marker_positions(v,cuff[sl],cuff_rot[sl],tri,bar,offset,cuff_local)
            for key,arr in (("vertices",v),("joints",j),("predicted24",mounts)):z[key][sl]=arr.cpu().numpy()
    dr=np.linalg.norm(z["joints"][:,TIPS]-ref[:,TIPS],axis=-1)*1000
    assert (dr.max()<.005 if hard else np.all(dr.max(0)<cap.cpu().numpy()+.005)),dr.max(0)
    z["marker_error_mm"]=np.where(z["marker_valid"],np.linalg.norm(z["predicted24"]-z["target24"],axis=-1)*1000,np.nan)
    rotation=Rotation.from_rotvec(z["global_orient"]);root=z["joints"][:,0]+rotation.apply(np.broadcast_to(z["root_anchor_local_m"],(len(times),3)))
    z["root_anchor_error_mm"]=(np.linalg.norm(root-z["finger0_target"],axis=-1)*1000).astype(np.float32)
    z["joint_minus_cuff_pivot_local_m"]=np.einsum("nji,nj->ni",cuff_rot.cpu().numpy(),z["joints"][:,0]-cuff.cpu().numpy()).astype(np.float32)
    z["tip_reference_m"]=ref[:,TIPS];z["tip_drift_mm"]=dr;z["tip_constraint_mode"]=np.array(variant);z["projection_needed"]=np.array(corrections)
    z["fallback_to_baseline_count"]=np.array(fallback)
    if output_dir is not None:
        z["physical_hand"] = np.array(hand)
        if not hard:
            z["tip_kink_mm"] = kink.cpu().numpy()
            z["tip_cap_mm"] = cap.cpu().numpy()
    for key in ("locked_tip_reference_camera0_canonical_m","locked_tip_drift_mm","tip_lock_seed_fraction"):z.pop(key,None)
    folder.mkdir(parents=True,exist_ok=True);save(folder/(hand+".npz"),z)
    err=z["marker_error_mm"][:,3:];print(variant,"accept",np.sum((np.nanmedian(err,axis=1)<=2.5)&(np.nanmax(err,axis=1)<=8)),
        "surface",np.nanmedian(err),np.nanmax(err),"tip",np.percentile(dr,[50,95,100]),"fallback",fallback,flush=True)
