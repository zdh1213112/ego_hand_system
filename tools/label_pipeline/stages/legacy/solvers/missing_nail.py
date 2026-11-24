# Source work/glove_trainset_20261120_v2/missing_nail.py, SHA-256 1381382f48430e4cba40bb7fd843d63a40d015e8aa164683960e5abc90f3739e

import case_settings as case

import numpy as np

from scipy.spatial import cKDTree

def unit(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1.e-10)

def angle(a,b):return np.degrees(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=-1),-1,1)))

class NailPrior:
    def __init__(self,j,markers,valid_times,base_frames,digit):
        self.digit=digit
        self.j=j;self.m=markers;self.f=1+4*digit;self.m0=4+4*digit
        # Include only wrist and the three proximal SDK nodes of each digit.
        # The queried distal segment and nail never enter the neighbor search.
        local=(j-j[:,:1])@base_frames[:,0]
        base=np.array([1+4*d+k for d in range(5) for k in range(3)])
        focus=local[:,self.f:self.f+3]
        self.q=np.concatenate((.28*local[:,base].reshape(len(j),-1),focus.reshape(len(j),-1),2.0*np.diff(focus,axis=1).reshape(len(j),-1)),axis=1)
        self.mid=base_frames[:,1+([12,0,3,9,6][digit])+1]
        usable=np.isfinite(markers[valid_times,self.m0:self.m0+4]).all((1,2))
        self.clocks=valid_times[usable];self.tree=cKDTree(self.q[self.clocks])
        self.local_end=np.einsum('ti,tij->tj',j[:,self.f+3]-j[:,self.f+2],self.mid)
        self.local_nail=np.einsum('ti,tij->tj',markers[:,self.m0+3]-markers[:,self.m0+2],self.mid)
    def predict(self,t):
        if not np.isfinite(self.q[t]).all() or len(self.clocks)<12:return None
        _,idx=self.tree.query(self.q[t],k=min(len(self.clocks),256))
        candidates=self.clocks[idx];candidates=candidates[np.abs(candidates-t)>120]
        # Consecutive nearly identical captures must not overwhelm distinct examples.
        _,first=np.unique(candidates//8,return_index=True);chosen=candidates[np.sort(first)[:20]]
        if len(chosen)<7:return None
        end=unit(np.median(unit(self.local_end[chosen]),axis=0))
        tail=angle(self.local_end[chosen],end)
        direction=self.mid[t]@end
        current=self.j[t,self.f+3]-self.j[t,self.f+2]
        err=float(angle(current,direction))
        gate=max(25.,float(np.percentile(tail,95))+8.)
        nail_disp=self.mid[t]@np.median(self.local_nail[chosen],axis=0)
        nail=self.m[t,self.m0+2]+nail_disp
        return dict(direction=direction,angle=err,gate=gate,nail=nail,count=len(chosen))

def correct(j,m,valid_times,frames):
    out=j.copy();records=[];cv=[]
    for d in range(5):
        model=NailPrior(j,m,valid_times,frames,d);f=model.f;mv=model.m0
        available=np.isfinite(m[:,mv+2]).all(1)
        missing=~np.isfinite(m[:,mv+3]).all(1)
        for t in np.flatnonzero(missing&available):
            res=model.predict(t)
            if res is None or res['angle']<=res['gate']:continue
            length=np.linalg.norm(j[t,f+3]-j[t,f+2])
            if not np.isfinite(length) or length<=0:continue
            out[t,f+3]=j[t,f+2]+length*res['direction']
            records.append((t,d,res['angle'],res['gate'],res['count']))
        pool=model.clocks
        for t in pool[np.unique(np.linspace(0,len(pool)-1,min(40,len(pool)),dtype=int))]:
            res=model.predict(t)
            if res is not None:
                cv.append((t,d,np.linalg.norm(res['nail']-m[t,mv+3])*1000,res['angle'],res['angle']>res['gate']))
    return out,records,cv
