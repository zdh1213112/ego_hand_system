# Source work/group_003_right_mount_study_20261119/optical_holdout.py, SHA-256 3c2fa3e9549ddf22891b9a444df73dd14d062bef38187c3b7cc24736dc05210c

import case_settings as case

import itertools

import cv2

import numpy as np

from scipy.optimize import least_squares

from scipy.spatial import cKDTree

from scipy.spatial.transform import Rotation

from observations import blobs, project

ALIGN = np.array([5,6,8,9,10,12,13,14,16,17,18,20,21,22])  # reserve all cuff/nail points plus Thumb1/HandOffset

def rigid(a,b):
    u,_,v=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
    fix=np.diag([1,1,np.linalg.det(u@v)])
    r=(u@fix@v).T
    return r,b.mean(0)-r@a.mean(0)

def inside(uv,cal):
    return np.isfinite(uv).all(-1)&(uv>=5).all(-1)&(uv<np.asarray(cal.image_size)-5).all(-1)

def detected(raw, projections,cal):
    out=[]
    for points in projections:
        p=points[inside(points,cal)]
        if not len(p):continue
        corner=np.maximum(0,np.floor(p.min(0)-37).astype(int))
        end=np.minimum(cal.image_size,np.ceil(p.max(0)+38).astype(int))
        local=blobs(raw[corner[1]:end[1],corner[0]:end[0]])
        if len(local):out.append(local+corner)
    return np.unique(np.concatenate(out),axis=0) if out else np.empty((0,2))

def assign(uv,cloud,maxdist,margin,spacing):
    good=np.isfinite(uv).all(1);ids=np.flatnonzero(good)
    if not len(ids) or len(cloud)<2:return {}
    p=uv[good];tree=cKDTree(cloud);d,i=tree.query(p,k=2)
    points=cKDTree(p)
    gap=points.query(p,k=2)[0][:,1]
    reverse=points.query(cloud[i[:,0]])[1]
    keep=(d[:,0]<maxdist)&(d[:,1]-d[:,0]>margin)&(gap>spacing)&(reverse==np.arange(len(p)))
    return {int(ids[j]):cloud[i[j,0]].copy() for j in np.flatnonzero(keep)}

def translation_seed(p,cloud):
    if len(p)<5 or len(cloud)<5:return None
    tree=cKDTree(cloud);neighbors=tree.query_ball_point(p,26)
    offsets=[cloud[j]-a for a,j in zip(p,neighbors) if len(j)]
    if not offsets:return None
    shifts=np.unique(np.rint(np.concatenate(offsets)).astype(int),axis=0)
    d=tree.query((p[None]+shifts[:,None]).reshape(-1,2))[0].reshape(-1,len(p))
    votes=(d<2.7).sum(1)
    score=votes-.025*np.minimum(d,5).mean(1)-.0006*np.linalg.norm(shifts,axis=1)
    i=score.argmax()
    return shifts[i] if votes[i]>=max(4,.38*len(p)) else None

def align_sources(phys,cs,cals,rt,cloud):
    center=np.nanmean(phys[1][ALIGN],0); views=[]; initial=[]
    for c in cs:
        for hand in phys.values():
            x=hand[ALIGN];uv=project(x,rt[c],cals[c],1)
            ok=inside(uv,cals[c]);x,uv=x[ok],uv[ok]
            if len(x)<5:continue
            low=uv.min(0)-30;high=uv.max(0)+30
            nearby=cloud[c][((cloud[c]>=low)&(cloud[c]<=high)).all(1)]
            if len(nearby)<5:continue
            views.append((c,x,nearby))
            shift=translation_seed(uv,nearby)
            if shift is not None:
                hit=assign(uv+shift,nearby,3.,.55,4.)
                if len(hit)>=4:initial.append((c,x[list(hit)],np.array(list(hit.values()))))
    def solve(pairs,start):
        def residual(p):
            r=Rotation.from_rotvec(p[:3]).as_matrix();t=center+p[3:]/1000-r@center
            # An intermediate rigid proposal can cross a fisheye camera plane.
            # Keep the residual vector finite so the optimizer can reject that proposal.
            values=[np.nan_to_num(project(xx@r.T+t,rt[c],cals[c],1)-yx,
                                  nan=1e3,posinf=1e3,neginf=-1e3).ravel() for c,xx,yx in pairs]
            return np.r_[np.concatenate(values),p[:3]/.035,p[3:]/14.]
        p=least_squares(residual,start,loss="soft_l1",f_scale=1.8,max_nfev=28).x
        r=Rotation.from_rotvec(p[:3]).as_matrix()
        return p,r,center+p[3:]/1000-r@center
    if not initial:return None
    p,r,t=solve(initial,np.zeros(6))
    for tol in (4.,3.4):
        pairs=[]
        for c,x,b in views:
            hit=assign(project(x@r.T+t,rt[c],cals[c],1),b,tol,.7,4.)
            if len(hit)>=4:pairs.append((c,x[list(hit)],np.array(list(hit.values()))))
        if len({c for c,_,_ in pairs})<2 or sum(len(xx) for _,xx,_ in pairs)<12:return None
        p,r,t=solve(pairs,p)
    e=np.concatenate([np.linalg.norm(project(xx@r.T+t,rt[c],cals[c],1)-uv,axis=1) for c,xx,uv in pairs])
    if not np.isfinite(e).all() or np.median(e)>2.5 or np.linalg.norm(p[3:])>35 or np.linalg.norm(p[:3])>.2:return None
    return r,t,len(e),float(np.median(e))

def ray(c,uv,cals,transforms):
    cal=cals[c];r,t=transforms[c]
    xy=cv2.fisheye.undistortPoints(np.ascontiguousarray(uv,dtype=float).reshape(1,1,2),
                                   cal.K,cal.distortion).reshape(2)
    v=r.T@np.r_[xy,1.];return -r.T@t, v/np.linalg.norm(v)

def triangulate(observed,ref,cals,rt):
    best=None
    for combo in itertools.combinations(observed,2):
        origin,dirn=zip(*(ray(c,observed[c],cals,rt) for c in combo))
        if np.linalg.norm(np.cross(*dirn))<.085:continue
        a=np.eye(3)-np.asarray(dirn)[:,:,None]*np.asarray(dirn)[:,None,:]
        point=np.linalg.solve(a.sum(0),np.einsum("cij,cj->i",a,origin))
        error=np.array([np.linalg.norm(project(point[None],rt[c],cals[c],1)[0]-p) for c,p in observed.items()])
        away=np.linalg.norm(point-ref)*1000
        if not np.isfinite(error).all() or away>10 or np.max([np.linalg.norm(project(point[None],rt[c],cals[c],1)[0]-observed[c]) for c in combo])>2.5:continue
        key=(-int((error<3.).sum()),float(np.median(error))+.1*away)
        if best is None or key<best[0]:best=(key,point)
    return best[1] if best else None
