"""Recording-shared calibration and one bounded, temporal marker/KB image fitting objective."""
from pathlib import Path
import argparse
import pickle
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import yaml

PACKAGE=Path(__file__).resolve().parents[1]
EGO=PACKAGE.parents[1]
PROJECT=EGO.parent
sys.path[:0]=[str(EGO/'scripts/joint_refit'),str(EGO/'scripts'),str(EGO/'tools/label_pipeline/calibration/personal_hand_calibration'),str(EGO/'tools/label_pipeline/calibration/full_silhouette')]
from fit_mano_sequence import import_mano,run_model
from bounded_articulation import compose,components,rotation_limits
from finger_prior import rest_axes
from run import regions_for,NAMES

def norm(x):
    return np.linalg.norm(x,axis=-1)
def unit(x):
    return x/np.maximum(norm(x)[...,None],1e-9)
def basis(c):
    x=unit(c[:,2]-c[:,1]);d=c[:,0]-(c[:,2]+c[:,1])/2
    y=unit(d-x*(d*x).sum(-1)[:,None])
    return np.stack((x,y,np.cross(x,y)),axis=-1)
def temporal_selection(t,m,sk,count,excluded):
    good=np.isfinite(m[:,4:]).all(-1).reshape(-1,5,4).sum(-1).min(-1)>=3
    good&=np.isfinite(m[:,:4]).all((1,2))&((t//120)%4!=2)&~np.isin(t,excluded)
    q=np.flatnonzero(good)
    if len(q)<8:raise ValueError('用于共享标定的独立姿势太少')
    tips=np.arange(4,21,4)
    features=np.c_[norm(sk[q][:,tips]-sk[q,:1]),norm(sk[q,4]-sk[q,8]),np.einsum('ni,ni->n',unit(sk[q,9]-sk[q,0]),basis(m[q,:3])[:,:,2])]
    complete=np.isfinite(features).all(1)
    if complete.sum()>=min(count,len(q)):q=q[complete];features=features[complete]
    features=np.nan_to_num((features-np.nanmedian(features,0))/np.maximum(np.nanstd(features,0),1e-5))
    # Farthest point sampling in posture and time prevents calibration collapsing onto a long static interval.
    features=np.c_[np.clip(features,-4,4),2*(t[q]-t[q].min())/max(1,np.ptp(t[q]))]
    distance=np.full(len(q),np.inf);selected=[];i=len(q)//2
    for _ in range(min(count,len(q))):
        selected.append(q[i]);distance=np.minimum(distance,((features-features[i])**2).sum(1));distance[i]=-1;i=int(distance.argmax())
    return np.sort(selected)

def observations(native,opt,h):
    side=int(h=='right');mirror=np.array([1 if side else -1,1,1],np.float32)
    with np.load(native/'aligned_skeleton.npz') as z:
        raw=z['marker24_cam0_m'][side];sdk=z['positions_cam0_m'][side][:,z['provisional21_indices']];stamps=z['reference_timestamp_ns']
    with np.load(opt/(h+'_registration.npz')) as z:
        rot=z['rotation'];off=z['translation_m'];held=z['held_marker'];held_camera=z['excluded_camera'];active=z['active']
    corrected=(raw@rot.transpose(0,2,1)+off[:,None])*mirror
    sdk=(sdk@rot.transpose(0,2,1)+off[:,None])*mirror
    uv=[]
    for c in range(4):
        with np.load(opt/f'camera{c}.npz') as z:uv.append(z['uv'][side])
    uv=np.stack(uv,1)
    uv=np.where((np.arange(4)[None,:,None]!=held_camera[:,None,None])[...,None],uv,np.nan)
    real=np.isfinite(corrected).all(-1)
    train=real&~held
    uv=np.where(train[:,None,:,None],uv,np.nan)
    # SDK rotations for a finger share its surface input, so exclude that entire SDK chain when scoring an omitted marker.
    hidden_digit=held[:,4:].reshape(-1,5,4).any(-1)
    sdk[:,1:]=np.where(np.repeat(~hidden_digit,4,axis=1)[...,None],sdk[:,1:],np.nan)
    valid=(real[:,4:].reshape(-1,5,4).sum(-1).min(-1)>=3)&(train[:,4:].sum(1)>=12)
    t=np.flatnonzero(valid)
    cal=[json.loads((native/f'calibration/camera{c}.json').read_text()) for c in range(4)]
    T=np.array([v['T_base_camera'] for v in cal])
    R=np.stack([a[:3,:3].T@T[0,:3,:3] for a in T]);b=np.stack([a[:3,:3].T@(T[0,:3,3]-a[:3,3]) for a in T])
    camera=(np.array([c['K'] for c in cal]),np.array([c['distortion'] for c in cal]),R,b,mirror)
    return t,stamps[t],corrected[t],train[t],sdk[t],uv[t],camera,active[t]

def project(points,camera):
    K,D,R,off,mirror=camera
    xyz=torch.einsum('nmk,ckj->ncmj',points*mirror,R.transpose(1,2))+off[None,:,None]
    xy=xyz[...,:2];radius=xy.norm(dim=-1).clamp_min(1e-7);theta=torch.atan2(radius,xyz[...,2].clamp_min(1e-5));a=theta.square()
    d=D[None,:,None]
    td=theta*(1+a*(d[...,0]+a*(d[...,1]+a*(d[...,2]+a*d[...,3]))))
    return xy*(td/radius)[...,None]*torch.stack((K[:,0,0],K[:,1,1]),-1)[None,:,None]+K[:,:2,2][None,:,None]

def masked_loss(delta,mask,scale):
    sq=delta.square().sum(-1)/scale**2
    return ((torch.sqrt(1+sq)-1)*mask).sum()/mask.sum().clamp_min(1)

def rigid_initial(rest,sdk,obs):
    ids=[0,5,9,13,17];x=rest[ids]
    y=sdk[:,ids].copy()
    proxy=obs[:,[0,8,12,16,20]].copy()
    y=np.where(np.isfinite(y),y,proxy)
    ok=np.isfinite(y).all(-1);n=ok.sum(1).clip(1)
    xm=(x[None]*ok[...,None]).sum(1)/n[:,None];ym=np.where(ok[...,None],y,0).sum(1)/n[:,None]
    a=(x[None]-xm[:,None])*ok[...,None];b=np.where(ok[...,None],y-ym[:,None],0)
    u,_,v=np.linalg.svd(a.transpose(0,2,1)@b);u[:,:,-1]*=np.sign(np.linalg.det(u@v))[:,None]
    R=(u@v).transpose(0,2,1)
    # MANO rotates around the rest wrist, so its translation places the rest wrist in the calibrated scene.
    tr=np.einsum('nij,nj->ni',R,rest[0]-xm)+ym-rest[0]
    return Rotation.from_matrix(R).as_rotvec(),tr

class Unified:
    def __init__(self,native,opt,mano,placement,h,params,exclude,device):
        self.params=params;self.h=h;self.dev=device
        t,stamps,obs,train,sdk,uv,camera,active=observations(native,opt,h)
        self.t=t;self.obs_np=obs;self.active=active;self.cal=temporal_selection(t,obs,sdk,params['calibration_count'],exclude)
        def ts(x):return torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device=device)
        self.ts=ts;self.cam=tuple(ts(v) for v in camera)
        self.secs=ts((stamps-stamps[0])*1e-9);self.support=torch.as_tensor(train,device=device);self.obs=ts(np.nan_to_num(obs))
        self.uv_valid=ts(np.isfinite(uv).all(-1));self.uv=ts(np.nan_to_num(uv))
        mask=np.isfinite(sdk).all(-1)
        self.sdk=ts(np.nan_to_num(sdk));self.sdk_mask=ts(mask)
        self.cuff_center=ts(np.nan_to_num(obs[:,:3].mean(1)));self.cuff_axes=ts(np.nan_to_num(basis(obs[:,:3])));self.cuff_mask=ts(train[:,:3].all(1))
        self.model=import_mano(EGO/'third_party/MANO').load(model_path=str(mano),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device).eval().requires_grad_(False)
        self.beta=torch.nn.Parameter(ts(np.zeros(10)));self.axes=rest_axes(self.model,self.beta.detach(),device,include_thumb=True)
        with np.load(PACKAGE/'assets/wilor_pose2906.npz') as f:prior=ts(f['local_rotvec'])-self.model.hand_mean
        with (mano/'MANO_RIGHT.pkl').open('rb') as stream:
            scan=pickle.load(stream,encoding='latin1')
        scans=ts(np.asarray(scan['hands_coeffs'])@np.asarray(scan['hands_components']))
        self.lo,self.hi=rotation_limits(scans,self.model,self.axes)
        priorq=components(prior,self.model,self.axes);self.prior_mean=priorq.mean(0);self.prior_std=priorq.std(0).clamp_min(5)
        with torch.no_grad():
            rest_v,rest_j,_=run_model(self.model,self.beta[None],ts([[0]*3]),-self.model.hand_mean[None],ts([[0]*3]))
        regions=regions_for(self.model,self.beta.detach(),json.loads(placement.read_text()),h,run_model)
        self.regions=regions;self.faces=self.model.faces_tensor.long()
        centers=rest_v[0,self.faces].mean(1).detach().cpu().numpy();rest=rest_j[0].cpu().numpy()
        goal=np.vstack([(rest[0]+rest[9])/2,rest[1:]])
        self.faces_idx=np.array([ids[norm(centers[ids]-g).argmin()] for ids,g in zip(regions,goal)])
        self.bary=torch.nn.Parameter(ts(np.zeros((21,2))));self.height=torch.nn.Parameter(ts(np.r_[2.,np.zeros(20)]))
        ori,tr=rigid_initial(rest,sdk,np.where(train[...,None],obs,np.nan))
        self.initial_o=ts(ori);self.initial_t=ts(tr)
        q=np.broadcast_to(self.prior_mean.detach().cpu().numpy(),(len(t),5,3,3)).copy()
        mids=np.stack([sdk[:,1+d*4:5+d*4] for d in (1,2,3,4,0)],1)
        chains=np.concatenate([np.broadcast_to(sdk[:,:1,None],(len(t),5,1,3)),mids],2)
        seg=unit(np.diff(chains,axis=2));deg=np.rad2deg(np.arccos(np.clip((seg[:,:,1:]*seg[:,:,:-1]).sum(-1),-1,1)))
        q[:,:,:,1]=np.where(np.isfinite(deg),deg,q[:,:,:,1]);self.initial_q=ts(q).clamp(self.lo+.01,self.hi-.01)
        wi=obs[:,:3].mean(1);wrist=rest[0]+tr
        local=np.einsum('nji,nj->ni',basis(obs[:,:3]),wrist-wi)*1000
        self.cuff=torch.nn.Parameter(ts(np.nanmedian(local[self.cal],axis=0)))
        self.cuff_prior=self.cuff.detach().clone()
        mode=params.get('freeze','none')
        self.beta.requires_grad_(mode!='betas')
        self.height.requires_grad_(mode!='heights')
    def mounts(self,v):
        faces=self.faces[self.faces_idx]
        triangle=v[:,faces];bar=F.softmax(F.pad(self.bary,(0,1)),dim=-1)
        normals=F.normalize(torch.cross(triangle[:,:,1]-triangle[:,:,0],triangle[:,:,2]-triangle[:,:,0],dim=-1),dim=-1)
        centered=F.pad(self.height[1:],(0,1));height=self.height[:1]+centered-centered.mean()
        return (triangle*bar[None,:,:,None]).sum(2)+normals*height[None,:,None]*.001,bar,height
    def geometry(self,i,q,o,tr):
        v,j,_=run_model(self.model,self.beta.expand(len(i),-1),o,compose(q,self.model,self.axes),tr*.001)
        return v,j,self.mounts(v)[0]
    def loss(self,i,q,o,tr,calibrate):
        p=self.params;v,j,m=self.geometry(i,q,o,tr)
        right=masked_loss(1000*(m-self.obs[i,3:]),self.support[i,3:],p['marker_scale_mm'])
        im=masked_loss(project(m,self.cam)-self.uv[i,:,3:],self.uv_valid[i,:,3:],p['pixel_scale_px'])
        target=self.cuff_center[i]+(self.cuff_axes[i]@self.cuff[:,None]).squeeze(-1)*.001
        wrist=masked_loss(1000*(j[:,0]-target),self.cuff_mask[i],p['cuff_scale_mm'])
        prior=((q-self.prior_mean)/self.prior_std).square().mean()
        bend=q[:,:4,1:,1];continuity=(F.relu(-bend)/5).square().mean()+((bend[:,:,1]-.67*bend[:,:,0])/30).square().mean()
        jdir=F.normalize(j[:,1:].reshape(-1,5,4,3).diff(dim=2),dim=-1)
        sk=F.normalize(self.sdk[i,1:].reshape(-1,5,4,3).diff(dim=2),dim=-1)
        sm=self.sdk_mask[i,1:].reshape(-1,5,4);dirs=masked_loss(jdir-sk,sm[:,:,1:]*sm[:,:,:-1],.25)
        loss=right+p['pixel_weight']*im+.35*wrist+p['prior_weight']*(prior+continuity)+p['sdk_direction_weight']*dirs
        if calibrate:
            _,_,height=self.mounts(v[:1]);low=height.new_full((21,),-2);high=height.new_full((21,),8);low[1:5]=-4;high[1:5]=10
            loss=loss+.08*self.beta.square().mean()+.12*((height-height.mean())/p['height_prior_mm']).square().mean()+.04*((height.mean()-2)/4).square()+2*(F.relu(low-height).square()+F.relu(height-high).square()).mean()+.04*((self.cuff-self.cuff_prior)/12).square().mean()
        elif len(i)>2:
            dt=self.secs[i].diff().clamp_min(.001)
            near=(dt<.11);vel=q.diff(dim=0)/(dt[:,None,None,None]*30)
            acc=(vel[1:]-vel[:-1])/(15*(dt[1:]+dt[:-1]))[:,None,None,None]
            loss=loss+p['temporal_weight']*(masked_loss(vel.flatten(1),near,30)+masked_loss(acc.flatten(1),near[1:]&near[:-1],16))
        return loss
    def variables(self,i):
        return [torch.nn.Parameter(x[i].detach().clone()) for x in (self.initial_q,self.initial_o,self.initial_t*1000)]
    def optimize(self,i,variables,steps,shared=False):
        q,o,tr=variables;groups=[dict(params=[q],lr=.75),dict(params=[o],lr=.009),dict(params=[tr],lr=.45)]
        if shared:groups.extend([dict(params=[self.beta],lr=.018),dict(params=[self.bary],lr=.036),dict(params=[self.height,self.cuff],lr=.12)])
        optim=torch.optim.Adam(groups);sch=torch.optim.lr_scheduler.CosineAnnealingLR(optim,steps)
        for step in range(steps):
            optim.zero_grad(set_to_none=True)
            # Shared calibration cycles through interleaved postures while the 76 physical parameters stay common.
            sub=slice(step%3,None,3) if shared else slice(None)
            loss=self.loss(i[sub],q[sub],o[sub],tr[sub],shared)
            if not torch.isfinite(loss):raise ValueError('统一目标出现非有限值')
            loss.backward();optim.step();sch.step()
            with torch.no_grad():
                q.clamp_(self.lo+.01,self.hi-.01)
                if shared:
                    relative=F.pad(self.height[1:],(0,1))
                    h=self.height[0]+relative-relative.mean()
                    low=torch.full_like(h,-2);high=torch.full_like(h,8)
                    low[1:5]=-4;high[1:5]=10
                    h=h.clamp(low,high)
                    self.height[0].copy_(h.mean());self.height[1:].copy_(h[:-1]-h[-1])
    @torch.no_grad()
    def choose_faces(self,idx,values):
        v,_,_=self.geometry(idx,*values)
        height=self.mounts(v[:1])[2]
        for m,region in enumerate(self.regions):
            tri=v[:,self.faces[region]]
            n=F.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
            err=(tri.mean(2)+n*height[m]*.001-self.obs[idx,m+3,None]).norm(dim=-1)*1000
            ok=self.support[idx,m+3]
            if ok.any():self.faces_idx[m]=region[int((torch.sqrt(1+(err[ok]/3).square())-1).mean(0).argmin())]
        self.bary.zero_()
    def fit(self,dest):
        c=torch.as_tensor(self.cal,device=self.dev);values=self.variables(c);steps=self.params['calibration_steps']
        self.optimize(c,values,steps//2,True);self.choose_faces(c,values);self.optimize(c,values,steps-steps//2,True)
        for x in (self.beta,self.bary,self.cuff,self.height):x.requires_grad_(False)
        # Seed held and unsampled poses independently; only the shared physical calibration crosses time blocks.
        size=self.params['window'];blend=self.params['overlap'];stride=size-2*blend
        result={k:[] for k in ('vertices','joints','mounted','global_orient','translation','hand_pose_pca','angles')}
        for start in range(0,len(self.t),stride):
            end=min(len(self.t),start+stride);i=np.arange(max(0,start-blend),min(len(self.t),end+blend))
            sel=slice(start-i[0],end-i[0]);var=self.variables(i);self.optimize(i,var,self.params['pose_steps'])
            with torch.no_grad():
                q,o,tr=var;v,j,m=self.geometry(i,*var)
                for k,x in zip(result,(v,j,m,o,tr*.001,compose(q,self.model,self.axes),q)):result[k].append(x[sel].detach().cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in result.items()};obs=self.obs_np;mount=arrays['mounted']
        er=norm(mount-obs[:,3:])*1000;passed=(np.nanmedian(er,1)<=2.5)&(np.nanmax(er,1)<=8)
        sample=np.zeros(len(self.t),bool);sample[self.cal]=True
        with torch.no_grad():_,bar,height=self.mounts(self.ts(arrays['vertices'][:1]))
        local=np.zeros((21,3),np.float32);local[:,2]=height.cpu().numpy()*.001
        cuff_obs=obs[:,:3];all_pred=np.concatenate([cuff_obs,mount],1)
        dst=dest/(self.h+'.npz');tmp=dst.with_suffix('.tmp.npz');dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(tmp,**arrays,fit_method='unified',approved_for_training=False,pair_indices=self.t,betas=self.beta.cpu().numpy(),faces=self.faces.cpu().numpy(),marker_valid=np.isfinite(obs).all(-1),target24=obs.astype('f'),predicted24=all_pred.astype('f'),observed_cuff_centers=cuff_obs.astype('f'),predicted24_cuff_source='observed_only',marker_names=NAMES,pass_surface=passed,attachment_faces=self.faces_idx,barycentric=bar.cpu().numpy(),attachment_local_offsets_m=local,calibration_mask=sample,cuff_offset_local_mm=self.cuff.cpu().numpy(),registration_active=self.active)
        tmp.replace(dst)
        diagnostics=dict(mode=self.params.get('freeze','none'),calibration_dof=76,
            effective_free_parameters=76-({'heights':21,'betas':10}.get(self.params.get('freeze'),0)),
            calibration_exposures=self.t[self.cal].tolist(),beta=self.beta.detach().cpu().tolist(),
            cuff_pair_median_mm=[float(np.nanmedian(norm(cuff_obs[self.cal,a]-cuff_obs[self.cal,b]))*1000)
                                 for a,b in ((0,1),(0,2),(1,2))],normal_height_mm=height.cpu().tolist(),
            independently_registered_exposures=int(self.active.sum()),fit_exposures=len(self.t))
        (dest/(self.h+'_calibration.json')).write_text(json.dumps(diagnostics,indent=2)+'\n')
        return dict(hand=self.h,poses=len(self.t),calibration=int(sample.sum()),pass_surface=int(passed.sum()),calibration_dof=76)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,required=True);ap.add_argument('--observations',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--hand',choices=('left','right'),required=True);ap.add_argument('--exclude',nargs='*',type=int,default=[]);ap.add_argument('--params',type=Path,default=Path(__file__).with_name('fit_config.yaml'));ap.add_argument('--mano',type=Path,default=EGO/'models/mano');ap.add_argument('--placement',type=Path,default=EGO/'configs/glove_marker_placement.json')
    ap.add_argument('--freeze',choices=('none','betas','heights'),default='none')
    a=ap.parse_args();p=yaml.safe_load(a.params.read_text());p['freeze']=a.freeze;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    obj=Unified(a.native,a.observations,a.mano,a.placement,a.hand,p,a.exclude,'cuda' if torch.cuda.is_available() else 'cpu')
    print('标定姿势',len(obj.cal),'整段手姿',len(obj.t),flush=True);print(json.dumps(obj.fit(a.output)),flush=True)
if __name__=='__main__':main()
