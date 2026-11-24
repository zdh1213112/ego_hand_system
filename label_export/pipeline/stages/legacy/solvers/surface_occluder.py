# Source work/glove_trainset_20261120_v2/surface_occluder.py, SHA-256 a89173b96c8db753f80ba4c85aa37de4eba7ea8cc058cfdc7480952b4cb0775f

import case_settings as case

import numpy as np

from scipy.spatial import cKDTree

from inputs import COMPAT, OPT, HANDS, read, take

from overlap import raster, unit

BONES=np.array([5,6,7,9,10,11,17,18,19,13,14,15,1,2,3])

def weighted_frames(j):
    length=unit(((j[...,5,:]+j[...,17,:])*.5)-j[...,0,:])
    lateral=unit(j[...,5,:]-j[...,17,:])
    normal=unit(np.cross(lateral,length))
    palm=np.stack((unit(np.cross(length,normal)),length,normal),axis=-1)
    bone=unit(j[...,BONES+1,:]-j[...,BONES,:])
    # Local transverse axis is shared across three bones of each finger.
    # For the radially oriented thumb use the palm longitudinal axis instead.
    seed=np.broadcast_to(palm[...,None,:,0],bone.shape).copy()
    seed[...,12:,:]=palm[...,None,:,1]
    z=unit(np.cross(seed,bone))
    x=unit(np.cross(bone,z))
    return np.concatenate((palm[...,None,:,:],np.stack((x,bone,z),axis=-1)),axis=-3)

def rigid(a,b):
    valid=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
    count=valid.sum(-1)[:,None,None]
    assert count.min()>=3
    am=np.where(valid[...,None],a,0).sum(1,keepdims=True)/count
    bm=np.where(valid[...,None],b,0).sum(1,keepdims=True)/count
    A=np.where(valid[...,None],a-am,0);B=np.where(valid[...,None],b-bm,0)
    u,_,vh=np.linalg.svd(A.swapaxes(1,2)@B)
    u[:,:,-1]*=np.linalg.det(u@vh)[:,None]
    rot=u@vh
    return rot,(bm-am@rot)

def skin_weights():
    import inspect
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    import pickle
    # Use the exact same vertex topology as the signed-off right-canonical tracks.
    import smplx  # existing MANO dependency
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for n,t in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
            if n not in np.__dict__:setattr(np,n,t)
        with ((case.project/case.config['mano_directory'])/'MANO_RIGHT.pkl').open('rb') as f:
            w=pickle.load(f,encoding='latin1')['weights']
    np.testing.assert_allclose(w.sum(1),1,atol=1e-5)
    return w

class Library:
    def __init__(self,g,s,weights,*,repair_nails=False):
        z=take(read(COMPAT/g/'summary.json')['fit_tracks'][HANDS[s]],'vertices','faces','target24','pair_indices','marker_valid','isolated_pose','pass_surface')
        d=take(OPT[g]/'skeleton/aligned_skeleton.npz','positions_cam0_m','provisional21_indices','marker24_cam0_m')
        self.j=d['positions_cam0_m'][s,:,d['provisional21_indices']].transpose(1,0,2).copy()
        self.markers=d['marker24_cam0_m'][s]
        times=z['pair_indices'];good=z['pass_surface']&~z['isolated_pose']
        self.sign=[1 if s else -1,1,1]
        dest=z['target24']*self.sign
        rr,tt=rigid(np.where(z['marker_valid'][...,None],self.markers[times],np.nan),dest)
        self.correction=dict(zip(times,zip(rr,tt)))
        # The optical marker alignment precedes the final per-hand fit and contains
        # many more valid clocks. Apply the archived transform to visibility only.
        a=take(OPT[g]/'dense/aligned'/(HANDS[s]+'_pilot.npz'),'target24','target24_uncorrected','pair_indices','image_alignment_corrected')
        extra=a['image_alignment_corrected']&~np.isin(a['pair_indices'],times)
        clock=a['pair_indices'][extra]
        src=a['target24_uncorrected'][extra]*self.sign
        dst=a['target24'][extra]*self.sign
        common=np.isfinite(src).all(-1)&np.isfinite(dst).all(-1)&(np.linalg.norm(dst,axis=-1)>.01)
        matched=np.linalg.norm(src-self.markers[clock],axis=-1)
        common&=np.isfinite(matched)&(matched<2.e-6)
        usable=common.sum(1)>=3
        if usable.any():
            raw=np.where(common[usable,:,None],src[usable],np.nan)
            er,et=rigid(raw,dst[usable])
            residual=np.where(common[usable],np.linalg.norm(raw@er+et-dst[usable],axis=-1),0).max(1)
            for ti,(r,t0),error in zip(clock[usable],zip(er,et),residual):
                if error<2.e-6:self.correction[ti]=(r,t0)
        self.faces=np.ascontiguousarray(z['faces'],np.int32)
        # An exported MANO pose may still lack a measured SDK joint. Skinning a
        # reference mesh needs all the joint frames for that same exposure.
        good &= np.isfinite(self.j[times]).all(axis=(1,2))
        self.times=times[good]
        self.mesh=((z['vertices'][good]*self.sign-tt[good])@rr[good].swapaxes(1,2)).astype('f')
        self.weights=weights
        self.bases=weighted_frames(self.j)
        self.raw_bases=self.bases
        self.raw_origins=self.j[:,np.r_[0,BONES]]
        if repair_nails:
            from missing_nail import correct
            self.j,self.nail_corrections,self.nail_holdout=correct(self.j,self.markers,self.times,self.bases)
            self.bases=weighted_frames(self.j)
        self.origins=self.j[:,np.r_[0,BONES]]
        local=(self.j-self.j[:,:1])@self.bases[:,0]
        self.features=local.reshape(len(local),-1)
        self.tree=cKDTree(self.features[self.times])
    def vertices(self,t):
        query=self.features[t]
        if not len(self.times) or not np.isfinite(query).all():return None
        n=min(384,len(self.times))
        distance,idx=self.tree.query(query,k=n)
        idx=np.atleast_1d(idx)
        valid=idx[np.abs(self.times[idx]-t)>120]
        if not len(valid):
            valid=np.flatnonzero(np.abs(self.times-t)>120)
            if not len(valid):return None
            valid=valid[np.argsort(np.linalg.norm(self.features[self.times[valid]]-query,axis=1))]
        i=valid[0];ref=self.times[i]
        # Transform local surface offsets via the current SDK joint frames.
        vert=self.mesh[i]
        local=np.einsum('vbi,bij->vbj',vert[:,None]-self.raw_origins[ref],self.raw_bases[ref])
        moved=np.einsum('vbi,bji->vbj',local,self.bases[t])+self.origins[t]
        surface=np.einsum('vb,vbi->vi',self.weights,moved)
        if t in self.correction:
            rot,tr=self.correction[t];surface=surface@rot+tr
        return surface

def depth(uv,z,faces,lo,scale,n):
    valid=np.isfinite(uv).all(1)&np.isfinite(z)
    fs=np.ascontiguousarray(faces[valid[faces].all(1)],np.int32)
    a=np.ascontiguousarray(np.column_stack(((uv-lo)*scale,z)),np.float32)
    d=np.full((n,n),np.inf,np.float32);raster(n,len(fs),fs,a,d)
    return d
