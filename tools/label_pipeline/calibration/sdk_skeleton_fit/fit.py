#!/usr/bin/env python3
"""Fit MANO motion to SDK skeleton positions under an explicit, unverified internal-joint assumption."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/personal_hand_calibration'))
from context import load_context
from run import PROJECT,import_mano,run_model,stats


def mapping(names,side,root):
    prefix='LeftHand' if side==0 else 'RightHand'
    names=list(names);requested=[prefix if root=='hand' else prefix+'Thumb0']
    requested += [prefix+finger+suffix for finger in ['Thumb','Index','Middle','Ring','Pinky'] for suffix in ['1','2','3','End']]
    if len(set(names))!=len(names):raise ValueError('Duplicate SDK segment names')
    return np.asarray([names.index(name) for name in requested]),requested


def main():
    ap=argparse.ArgumentParser()
    for name in ['context','source','skeleton','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--root',choices=['hand','finger0'],required=True)
    ap.add_argument('--assume-sdk-internal-offsets',action='store_true',required=True)
    ap.add_argument('--steps',type=int,default=350)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.steps<1:ap.error('steps must be positive')
    c=load_context(a.context);sk=np.load(a.skeleton)
    if str(sk['geometry_space'])!='physical_camera0':raise ValueError('Expected physical camera0 skeleton coordinates')
    torch.set_num_threads(4);torch.manual_seed(0);device=c['device'];to=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    model=import_mano(PROJECT/'third_party/MANO').load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device)
    from mano.lbs import batch_rodrigues
    for p in model.parameters():p.requires_grad_(False)
    a.output.mkdir(parents=True)
    summary={'strategy':'sdk_skeleton_assumed_internal_joint_fit_v1','approved_for_training':False,
             'assumptions':{'sdk_marker_to_internal_joint_offsets_configured':True,'assumption_verified':False,
                            'additional_marker_radius_subtraction_mm':0,'sdk_names_to_mano_joint_semantics_verified':False},
             'root_policy':a.root,'source':str(a.source.resolve()),'source_skeleton':str(a.skeleton.resolve()),
             'shared_shape_frozen':True,'marker_attachments_frozen':True,'fit_uses_images':False,
             'sdk_quaternions_used_for_joint_rotation_targets':False,
             'hand_pose_representation':'45 axis-angle components before MANO mean; legacy key hand_pose_pca',
             'limits':{'hand_component_increment_rad':.5,'root_rotation_increment_norm_rad':.15,'translation_increment_m':.02},
             'hands':{}}
    for side,hand in enumerate(['left','right']):
        start=time.monotonic();z=np.load(a.source/f'{hand}_pilot.npz');pairs=z['pair_indices'];ids,names=mapping(sk['names'][side],side,a.root)
        if a.root=='finger0':
            common=sk['positions_cam0_m'][side][:,[1,6,11,16,21]]
            if np.nanmax(np.linalg.norm(common-common[:,:1],axis=-1))>1e-6:raise ValueError('Five Finger0 origins do not coincide; specify a model mapping')
        target=sk['positions_cam0_m'][side][pairs][:,ids].copy();target[:,:,0]*=-1 if side==0 else 1
        if not np.isfinite(target).all():raise ValueError('Skeleton target has missing samples; do not silently interpolate them')
        beta,o0,p0,t0=[to(z[k]) for k in ['betas','global_orient','hand_pose_pca','translation']];goal=to(target);n=len(pairs)
        dp,do,dt=[torch.nn.Parameter(torch.zeros_like(x)) for x in [p0,o0,t0]]
        optimizer=torch.optim.Adam([{'params':[dp,do],'lr':.004},{'params':[dt],'lr':.0003}])
        order=np.argsort(pairs);near=np.diff(pairs[order])<=16;left=torch.tensor(order[:-1][near],device=device);right=torch.tensor(order[1:][near],device=device)
        def geometry():return run_model(model,beta[None].expand(n,-1),o0+do,p0+dp,t0+dt)
        def objective():
            v,j,out=geometry();e=(j-goal).norm(dim=-1)/.008
            residual=(torch.sqrt(1+e.square())-1).mean()
            prior=.02*(dp/.5).square().mean()+.02*(do/.15).square().mean()+.02*(dt/.02).square().mean()
            temporal=0 if not len(left) else .005*((dp[right]-dp[left])/.5).square().mean()+.005*((dt[right]-dt[left])/.02).square().mean()
            return residual+prior+temporal,residual,v,j,out
        with torch.no_grad():
            _,_,v0,j0,_=objective();before=stats(((j0-goal).norm(dim=-1)*1000).cpu().numpy())
        history=[];best_loss=float('inf');best=None
        for step in range(a.steps):
            optimizer.zero_grad();loss,data,_,j,_=objective()
            if not torch.isfinite(loss):raise ValueError('Nonfinite SDK fit')
            if float(loss.detach())<best_loss:best_loss=float(loss.detach());best=[p.detach().clone() for p in [dp,do,dt]]
            loss.backward();optimizer.step()
            with torch.no_grad():
                dp.clamp_(-.5,.5);do.mul_(torch.clamp(.15/do.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1));dt.mul_(torch.clamp(.02/dt.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1))
            if step%70==0 or step==a.steps-1:
                row={'step':step,'loss':float(loss.detach()),'joint_median_mm':float((j-goal).norm(dim=-1).median().detach()*1000)};history.append(row);print(hand,a.root,json.dumps(row),flush=True)
        with torch.no_grad():
            loss,_,_,_,_=objective()
            if float(loss)<best_loss:best=[p.detach().clone() for p in [dp,do,dt]]
            for p,q in zip([dp,do,dt],best):p.copy_(q)
            loss,_,v,j,out=objective();residual=(j-goal).norm(dim=-1)*1000
            oldrot=batch_rodrigues((p0+model.hand_mean).reshape(-1,3));newrot=batch_rodrigues((p0+dp+model.hand_mean).reshape(-1,3));relative=oldrot.transpose(-1,-2)@newrot
            angular=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))*180/torch.pi
        cpu=lambda x:x.detach().cpu().numpy();values={k:z[k] for k in z.files}
        values.update(vertices=cpu(v),joints=cpu(j),baseline_vertices=z['vertices'],baseline_joints=z['joints'],
                      global_orient=cpu(o0+do),hand_pose_pca=cpu(p0+dp),translation=cpu(t0+dt),
                      sdk_target21=target.astype(np.float32),sdk_joint_error_mm=cpu(residual),sdk_segment_indices=ids,
                      sdk_segment_names=np.asarray(names),sdk_skeleton_fitting_enabled=np.array(True),
                      sdk_internal_offset_assumed=np.array(True),approved_for_training=np.array(False))
        # Marker attachment predictions are diagnostic only; they are not SDK fit targets.
        faces=model.faces_tensor.long();face_ids=torch.tensor(z['attachment_faces'],device=device).long()
        with torch.no_grad():
            tri=v[:,faces[face_ids]];normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
            surface=(tri*to(z['barycentric'])[None,:,:,None]).sum(2)+normal*to(z['normal_offsets_m'])[None,:,None]
            wrist=torch.einsum('fij,kj->fki',batch_rodrigues(o0+do),to(z['wrist_local']))+j[:,:1]
            pred=torch.cat([wrist,surface],1)
        values['predicted24']=cpu(pred);values['marker_error_mm']=cpu((pred-to(z['target24'])).norm(dim=-1)*1000)
        for key in ['betas','attachment_faces','barycentric','normal_offsets_m','wrist_local']:assert np.array_equal(values[key],z[key]),f'Shared parameter changed: {key}'
        np.savez_compressed(a.output/f'{hand}_pilot.npz',**values)
        summary['hands'][hand]={'frames':n,'mapping_to_mano21':names,'before_sdk_joint_residual_mm':before,
                               'after_sdk_joint_residual_mm':stats(cpu(residual)),'joint_angle_change_deg':stats(cpu(angular)),
                               'translation_increment_mm':stats(cpu(dt.norm(dim=-1)*1000)),
                               'limits_reached':{'hand_components':int((dp.abs()>=.4999).sum()),'translation':int((dt.norm(dim=-1)>=.01999).sum()),'root_rotation':int((do.norm(dim=-1)>=.14999).sum())},
                               'shared_parameters_exactly_unchanged':True,'history':history,'elapsed_seconds':time.monotonic()-start}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(hand,'DONE',summary['hands'][hand]['after_sdk_joint_residual_mm'],flush=True)
    print(a.output,flush=True)


if __name__=='__main__':main()
