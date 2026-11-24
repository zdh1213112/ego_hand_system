# Source work/joint_motion_20261119/choose_fast_clips.py, SHA-256 a797c8ee2e7211cf22df97e1915699a3a43709cf5c1af6f125997e53c75288e5
import numpy as np
import json

def choose(OUT,source,cutoff,keep):
    def bend(x):return np.rad2deg(np.arctan2(np.linalg.norm(x[...,1:],axis=-1),x[...,0]))
    clips={}
    for g in sorted(cutoff):
        if not (OUT/g/'final').is_dir():continue
        clips[g]=[]
        for hand in ('left','right'):
            with np.load(OUT/g/'final'/(hand+'_motion.npz')) as z:
                times=z['times'];v=bend(z['turns_target']);b=bend(z['turns_before']);valid=z['turn_valid']
            cfg=json.loads((source/g/'contour/summary.json').read_text())['hands'][hand]
            locks=cfg['visible_contours']['changed_times']+cfg['calibration']['syncs']+list(keep.get(hand, []))
            usable=(times<cutoff[g])&~np.isin(times,locks)
            indices=np.flatnonzero(usable)
            options=[]
            for run in np.split(indices,np.flatnonzero(np.diff(times[indices])!=1)+1):
                for start in range(0,len(run)-22,5):
                    idx=run[start:start+23]
                    for f in (3,4):
                        for j in (0,1):
                            if not valid[idx,f,j].all() or np.ptp(v[idx,f,j])<7:continue
                            score=float(np.mean(np.abs(np.diff(b[idx,f,j]-v[idx,f,j],2))))
                            options.append((score,f,j,idx))
            if options:
                score,f,j,idx=max(options,key=lambda s:s[0])
                clips[g].append(dict(group=g,hand=hand,digit=f,joint=j,frames=times[idx].tolist(),before_score=score))
    return clips
