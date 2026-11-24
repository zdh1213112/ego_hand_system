# Source work/joint_motion_20261119/paired_articulation_metrics.py, SHA-256 c2c0b9c25a0c3f29648fa02ca216505ba1e69ba7f32a748372f1f714fe4660f7

import case_settings as case

import numpy as np

from scipy.spatial import cKDTree

from observations import blobs

def matched_spots(img,points,size):
    ids=np.flatnonzero(np.isfinite(points).all(1) & (points >= 0).all(1) & (points < size).all(1))
    if len(ids)<2:
        return [],[]
    obs=points[ids]
    lo=np.maximum(0,np.floor(obs.min(0)-32).astype(int))
    hi=np.minimum(size,np.ceil(obs.max(0)+33).astype(int))
    found=blobs(img[lo[1]:hi[1],lo[0]:hi[0]])+lo
    if len(found)<2:
        return [],[]
    dist,idx=cKDTree(found).query(obs,k=2)
    gap=cKDTree(obs).query(obs,k=2)[0][:,1]
    allow=(gap>6)&(dist[:,0]<3)&((dist[:,1]-dist[:,0])>2)
    return ids[allow],found[idx[allow,0]]
