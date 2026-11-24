# Source work/joint_motion_20261119/fast_pinky.py, SHA-256 4e3887602473fc357c8ff1b6f4bb8342ab6232232a785d5c41069d3f68726ac6

import case_settings as case

import json

import cv2

import numpy as np

import torch

from render import OUT, HERE, HANDS, Recording

from paired_articulation_metrics import matched_spots

from observations import project

from fit_pivot import EGO, import_mano, finger_prior, components, compose, run_model, marker_positions, local_turns

from bounded_articulation import MARKER_ALLOWANCE_MM

def angle(turn):
    return torch.rad2deg(torch.atan2(torch.linalg.vector_norm(turn[...,1:],dim=-1),turn[...,0]))

def fit(clip=None):
    device='cuda'
    model=import_mano(EGO/'third_party/MANO').load(model_path=str(EGO/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device).requires_grad_(False)
    d=clip or json.loads((OUT/'visualizations/bend_motion.json').read_text())
    g=d['group'];hand=d['hand'];t=d['frames'];side=HANDS.index(hand);n=len(t)
    digit=d.get('digit',4);joint=d['joint'];slot=(digit-1)%5
    named=slice(4+4*digit,8+4*digit)
    tip=4*(digit+1)
    name=hand+'_pilot.npz'
    R=OUT/g/'final'
    def T(x):return torch.as_tensor(x,device=device,dtype=torch.float32)
    with np.load(R/name) as z:
        ix=np.searchsorted(z['pair_indices'],t)
        np.testing.assert_array_equal(z['pair_indices'][ix],t)
        beta=T(z['betas']).expand(n,-1)
        axes=finger_prior.rest_axes(model,beta[0],device,include_thumb=True)
        ori,trans,basepose,targets=[T(z[k][ix]) for k in ('global_orient','translation','hand_pose_pca','target24')]
        valid=torch.tensor(z['marker_valid'][ix],device=device)
        tri=model.faces_tensor.long()[torch.as_tensor(z['attachment_faces'],device=device).long()]
        attach=[T(z[k]) for k in ('barycentric','attachment_local_offsets_m','wrist_local')]
    with np.load(HERE/g/'contour/final'/name) as z:
        old_err=T(z['marker_error_mm'][ix]);old_avg=(old_err.square().masked_fill(~valid,0.).sum(1)/valid.sum(1)).sqrt()
    with np.load(HERE/g/'reference/final'/name) as z:
        ri=np.searchsorted(z['pair_indices'],t)
        cap=T(z['marker_error_mm'][ri])+MARKER_ALLOWANCE_MM
    with np.load(R/(hand+'_motion.npz')) as z:
        observed=T(z['turns_target'][ix,digit,joint]);assert z['turn_valid'][ix,digit,joint].all()
        source_bend=angle(T(z['turns_before'][ix,digit,joint])).cpu().numpy()
    cfg=json.loads((HERE/g/'bounded/summary.json').read_text())['hands'][hand]
    lo,hi=[T(cfg[k]) for k in ('lower_degrees','upper_degrees')]
    targets=targets.nan_to_num()
    initial=components(basepose,model,axes)
    from mano.lbs import batch_rodrigues
    rotation=batch_rodrigues(ori)
    def forward(a):
        pose=compose(a,model,axes)
        verts,joints,_=run_model(model,beta,ori,pose,trans)
        mount=marker_positions(verts,joints[:,0],rotation,tri,*attach)
        r=(mount-targets)*1000
        bend,_=local_turns(joints[:,1:].reshape(-1,5,4,3),targets[:,8]-targets[:,20])
        return joints,verts,pose,mount,r,angle(bend[:,digit,joint])
    with torch.no_grad():j0,_,_,m0,r0,b0=forward(initial)
    goal=angle(observed)
    # Preserve the recording's usual offset relative to the marker-mounted glove.
    typical=torch.median((b0-goal).detach())
    opt_angles=torch.nn.Parameter(initial[:,slot].clone())
    opt=torch.optim.Adam([opt_angles],lr=.32)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
    def build():
        return torch.cat([initial[:,:slot],opt_angles[:,None],initial[:,slot+1:]],dim=1)
    for _ in range(100):
        opt.zero_grad()
        j,_,_,_,r,b=forward(build())
        err=r.norm(dim=-1)
        finger=err[:,named];mask=valid[:,named]
        residual=b-goal-typical
        loss=(residual/2.).square().mean()+2.5*((residual[1:]-residual[:-1])/2.).square().mean()
        loss+=.75*((residual[2:]-2*residual[1:-1]+residual[:-2])/2.).square().mean()
        loss+=(finger.square().masked_fill(~mask,0).sum()/mask.sum())
        allowed=torch.minimum(cap[:,named],old_err[:,named]+.28)
        loss+=12*(finger-allowed+.02).clamp_min(0.).masked_fill(~mask,0.).amax(1).square().mean()
        loss+=.22*(((j[:,tip]-j0[:,tip])*1000).square().sum(-1)).mean()
        loss+=.2*((opt_angles-initial[:,slot])/4.).square().mean()
        loss+=.06*((opt_angles[:,1:,[0,2]])/4.).square().mean()
        loss.backward();opt.step();sched.step()
        with torch.no_grad():opt_angles.clamp_(lo[slot]+.002,hi[slot]-.002)
    # A common fraction across the clip avoids switching between methods on adjacent frames.
    # Use a smooth fade into the archived pose at each end to avoid introducing a discontinuity there.
    u=torch.linspace(0,1,n,device=device)
    taper=(torch.minimum(u,1-u)*5).clamp_max(1)
    taper=.5-.5*torch.cos(torch.pi*taper)
    result=[]
    data={}
    for alpha in (0.,.125,.25,.375,.5,.75,1.):
        with torch.no_grad():
            a=torch.lerp(initial,build(),alpha*taper[:,None,None,None])
            j,v,p,m,r,b=forward(a)
            error=r.norm(dim=-1)
            measured=error.masked_fill(~valid,float('nan'))
            gates=(((error<=torch.minimum(cap,old_err+.28)+1e-4)|~valid).all(1)&(torch.nanquantile(measured,.5,dim=1)<=2.5)&(measured.nan_to_num().amax(1)<=8)&(((j[:,tip]-j0[:,tip])*1000).norm(dim=-1)<=1.25))
            if clip is not None:
                frame_rms=(error.square().masked_fill(~valid,0.).sum(1)/valid.sum(1)).sqrt()
                gates &= (frame_rms <= old_avg+.0351)
            result.append(dict(alpha=alpha,feasible=bool(gates.all()),accepted_frames=int(gates.sum()),motion=float((b.diff()-goal.diff()).abs().mean()),absolute=float((b-goal).abs().mean()),pinky_marker_mm=float(torch.nanmean(measured[:,named])),hip=float((r[:,named]-r0[:,named]).norm(dim=-1).max())))
            data[alpha]=[x.cpu().numpy() for x in (j,v,p,m,b)]
    # Keep all other calibrated joints inside the same existing consensus gate.
    from fit_pivot import array_pack
    _,sensor,_=array_pack(g,hand,1,model=model,interval=(t[0],t[-1]))
    goal3=sensor.expected[:,t].to(device);sv=sensor.valid[t].to(device);valid_d=valid[:,4:].reshape(n,5,4)
    sv&=valid[:,8,None,None]&valid[:,20,None,None]&valid_d[:,:,:-2]&valid_d[:,:,1:-1]&valid_d[:,:,2:]
    from bounded_articulation import angular_error
    for record in result:
        joints=T(data[record['alpha']][0])
        bend,ok=local_turns(joints[:,1:].reshape(-1,5,4,3),targets[:,8]-targets[:,20])
        unresolved=((angular_error(bend[None].expand_as(goal3),goal3)>sensor.limits.to(device).clamp_min(10)).all(0)&sv&ok)
        record['sdk_violations']=int(unresolved.any((1,2)).sum())
    choice=min((r for r in result if r['feasible'] and not r['sdk_violations']),key=lambda r:r['motion'])
    rec=Recording(g,'fast_wilor.npz' if clip is not None else 'extra_wilor.npz')
    raw_results=[]
    try:
        for k in (0.,choice['alpha']):
            joints,_,_,mount,curve=data[k]
            native=[];spots=[]
            for c in range(4):
                pred=project(joints,rec.rt[c],rec.cal[c],side)
                measured_target=targets.cpu().numpy().copy();measured_target[~valid.cpu().numpy()]=np.nan
                opt_uv=project(measured_target[:,4:],rec.rt[c],rec.cal[c],side)
                fitted_uv=project(mount[:,4:],rec.rt[c],rec.cal[c],side)
                for i,sync in enumerate(t):
                    img=cv2.cvtColor(rec.read(sync,c),cv2.COLOR_RGB2BGR)
                    ids,pix=matched_spots(img,opt_uv[i],rec.cal[c].image_size)
                    for mk,pt in zip(ids,pix):
                        if 4*digit<=mk<4*digit+4:spots.append(float(np.linalg.norm(fitted_uv[i,mk]-pt)))
                    wi=rec.iw[sync]
                    if rec.w_valid[wi,c,side] and np.linalg.norm(pred[i].mean(0)-rec.native[wi,c,side].mean(0))<55:
                        native.extend(np.linalg.norm(pred[i,1+4*digit:5+4*digit]-rec.native[wi,c,side,1+4*digit:5+4*digit],axis=-1).tolist())
            raw_results.append(dict(alpha=k,raw_spot_px=float(np.mean(spots)),spots=len(spots),wilor_pinky_px=float(np.mean(native)),native_points=len(native)))
    finally:rec.close()
    dest=(OUT/'fast_pinky') if clip is None else (OUT/'fast_fits'/g/hand)
    dest.mkdir(parents=True,exist_ok=True)
    accepted=(choice['motion'] < float(np.mean(np.abs(np.diff(source_bend)-np.diff(goal.cpu().numpy())))) and choice['motion'] < result[0]['motion'] and raw_results[0]['spots']>=4 and raw_results[0]['native_points']>=24 and raw_results[1]['raw_spot_px']<=raw_results[0]['raw_spot_px']+.02 and raw_results[1]['wilor_pinky_px']<=raw_results[0]['wilor_pinky_px']+.05)
    info=dict(group=g,hand=hand,digit=digit,joint=joint,baseline_motion=float(np.mean(np.abs(np.diff(source_bend)-np.diff(goal.cpu().numpy())))),frames=t,selection=choice,candidates=result,independent=raw_results,accept_for_this_clip=bool(accepted))
    (dest/'summary.json').write_text(json.dumps(info,indent=1)+'\n')
    print(json.dumps(info,indent=1),flush=True)
    j,v,p,m,b=data[choice['alpha']]
    np.savez_compressed(dest/'candidate.npz',times=t,joints=j,vertices=v,hand_pose_pca=p,predicted24=m,bend_deg=b,accepted=accepted)
