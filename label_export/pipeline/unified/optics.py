"""Measure native image peaks from raw NOKOV identities before introducing a MANO model."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
def project(x,cal,rt):
    r,t=rt;p=(x@r.T+t).reshape(-1,3)
    valid=np.isfinite(p).all(1)&(p[:,2]>1e-8)
    uv=np.full((len(p),2),np.nan)
    if valid.any():uv[valid]=cv2.fisheye.projectPoints(p[valid,None],np.zeros(3),np.zeros(3),np.array(cal['K']),np.array(cal['distortion']))[0][:,0]
    return uv
def peaks(im,p):
    H,W=im.shape[:2];lo=np.maximum(0,np.floor(p.min(0)-34)).astype(int);hi=np.minimum([W,H],np.ceil(p.max(0)+35)).astype(int)
    if (hi<=lo).any():return np.empty((0,2))
    roi=im[lo[1]:hi[1],lo[0]:hi[0]]
    g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY).astype(np.float32)
    dog=cv2.GaussianBlur(g,(0,0),1.3)-cv2.GaussianBlur(g,(0,0),4)
    yy,xx=np.nonzero((dog==cv2.dilate(dog,np.ones((5,5),np.uint8)))&(dog>6))
    return np.c_[xx,yy]+lo

def identify(x,cloud,valid,registration):
    out=np.full((24,2),np.nan)
    if len(cloud)<5 or len(registration)<5:return out
    tree=cKDTree(cloud);p=x[registration]
    neighbors=tree.query_ball_point(p,28)
    options=[cloud[idx]-q for q,idx in zip(p,neighbors) if len(idx)]
    if not options:return out
    shifts=np.unique(np.rint(np.concatenate(options)).astype(int),axis=0)
    dist=tree.query((p[None]+shifts[:,None]).reshape(-1,2))[0].reshape(-1,len(p))
    votes=(dist<2.7).sum(1)
    score=votes-.025*np.minimum(dist,5).mean(1)-.0006*np.linalg.norm(shifts,axis=1);best=score.argmax()
    if votes[best]<max(4,.38*len(p)):return out
    shift=shifts[best];x=x+shift
    idx=np.flatnonzero(valid);centers=x[idx];ref=cKDTree(centers)
    d,j=tree.query(centers,k=2);gap=ref.query(centers,k=2)[0][:,1]
    keep=(d[:,0]<7)&(d[:,1]-d[:,0]>1.2)&(gap>5)&(ref.query(cloud[j[:,0]])[1]==np.arange(len(idx)))
    out[idx[keep]]=cloud[j[keep,0]]
    return out

def fit_batch(x,uv,train,cal,rt):
    n,m,_=x.shape
    valid=np.isfinite(x).all(-1);center=np.nansum(np.where(valid[...,None],x,0),1)/np.maximum(1,valid.sum(1))[:,None]
    x=np.where(valid[...,None],x,center[:,None]);R=np.tile(np.eye(3),(n,1,1));shift=np.zeros((n,3))
    obs=np.isfinite(uv).all(-1)&train&valid[:,None,:]
    active=(obs.any(1).sum(1)>=5)&((obs.sum(2)>=3).sum(1)>=2)
    def proj(p):
        return np.stack([project(p.reshape(-1,3),camera,r).reshape(n,m,2) for camera,r in zip(cal,rt)],1)
    for it in range(5):
        p=(x-center[:,None])@np.transpose(R,(0,2,1))+center[:,None]+shift[:,None]
        pred=proj(p);error=np.nan_to_num(uv-pred)
        seeds=np.eye(3);J=[]
        for a in seeds:
            pp=(p-center[:,None]-shift[:,None])@Rotation.from_rotvec(a*.001).as_matrix().T+center[:,None]+shift[:,None]
            J.append((proj(pp)-pred)*10)
        for a in seeds:J.append((proj(p+a*.0001)-pred)*10)
        J=np.stack(J,-1);mask=obs&np.isfinite(J).all((-1,-2));J=np.nan_to_num(J)
        length=np.linalg.norm(error,axis=-1);w=(mask/(1+np.square(length/3.5)))[...,None,None]
        A=(np.swapaxes(J,-1,-2)@(J*w)).sum((1,2))+np.diag([.25]*3+[.1]*3)
        rhs=(np.swapaxes(J,-1,-2)@(error[...,None]*w)).sum((1,2))
        d=np.linalg.solve(A,rhs)[...,0];d=np.clip(d,-6,6);d[~active]=0
        R=Rotation.from_rotvec(d[:,:3]*.01).as_matrix()@R;shift+=d[:,3:]*.001
    off=center+shift-(R@center[...,None])[...,0]
    pred=proj(x@R.transpose(0,2,1)+off[:,None])
    gap=np.linalg.norm(pred-uv,axis=-1)
    return R,off,active,gap


def prepare(native,raw_file,dest):
    """Identify independent bright pixels on four physical cameras, using only measured NOKOV."""
    with (native/'multiview_frames.csv').open() as stream:frames=list(csv.DictReader(stream))
    with np.load(raw_file) as archive:raw=archive['marker24_cam0_m']
    cals=[json.loads((native/f'calibration/camera{c}.json').read_text()) for c in range(4)]
    mats=np.array([c['T_base_camera'] for c in cals]);rt=[(v[:3,:3].T@mats[0,:3,:3],v[:3,:3].T@(mats[0,:3,3]-v[:3,3])) for v in mats]
    dest.mkdir(parents=True,exist_ok=True);n=len(frames);cameras=[]
    for c in range(4):
        path=dest/f'camera{c}.npz'
        proj=project(raw.reshape(-1,3),cals[c],rt[c]).reshape(2,n,24,2)
        uv=np.full_like(proj,np.nan,dtype=np.float32);wh=np.asarray(cals[c]['image_size'])
        cap=cv2.VideoCapture(str(native/f'cameras/camera{c}/video.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);last=-1
        try:
            for t,r in enumerate(frames):
                f=int(r[f'camera{c}_frame_index'])
                while last+1<f:
                    if not cap.grab():raise IOError((t,c,f))
                    last+=1
                ok,im=cap.read();last+=1
                if not ok or last!=f:raise IOError((t,c,f))
                for s in (0,1):
                    xy=proj[s,t];valid=np.isfinite(xy).all(1)&(xy>2).all(1)&(xy<wh-2).all(1)
                    if valid.sum()>=5:uv[s,t]=identify(xy,peaks(im,xy[valid]),valid,np.flatnonzero(valid))
        finally:cap.release()
        np.savez_compressed(path,uv=uv);cameras.append(uv)
    obs=np.stack(cameras,2)
    for s,h in enumerate(('left','right')):
        rot=np.zeros((n,3,3));off=np.zeros((n,3));active=np.zeros(n,bool)
        for start in range(0,n,64):
            sl=slice(start,min(n,start+64));train=np.ones(obs[s,sl].shape[:-1],bool)
            rot[sl],off[sl],active[sl],_=fit_batch(raw[s,sl],obs[s,sl],train,cals,rt)
        np.savez_compressed(dest/(h+'_registration.npz'),rotation=rot,translation_m=off,active=active,
                            excluded_camera=np.full(n,-1),held_marker=np.zeros((n,24),bool))
