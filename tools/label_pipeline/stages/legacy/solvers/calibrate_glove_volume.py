# Source work/glove_trainset_20261120_v2/calibrate_glove_volume.py, SHA-256 2a3a558930a67c31e056986106b71ef4e3ad10bb26e360b36196e7953b72d807

import case_settings as case

from collections import defaultdict

import cv2

import numpy as np

from scipy import ndimage as ndi

from skimage.measure import marching_cubes

from inputs import COMPAT, ROOT, HANDS, geometry, read, rows

from overlap import project_rays, unit

from surface_occluder import Library, skin_weights

DIR=ROOT/'checks/occlusion_review/volume'

SPACING=.003

BOUNDS={'cuff':((-0.061,.061),(-.125,.014),(-.044,.044)),
        'palm':((-.054,.058),(-.005,.084),(-.037,.037))}

def cuff_frame(markers):
    center=(markers[...,1,:]+markers[...,2,:])*.5
    x=unit(markers[...,1,:]-markers[...,2,:])
    y0=unit(markers[...,0,:]-center)
    z=unit(np.cross(x,y0));y=unit(np.cross(z,x))
    return center,np.stack((x,y,z),axis=-1)

def cuff_basis(median_forward):
    # Define the longitudinal direction once on separated calibration poses, and
    # transfer it rigidly with the three measured wrist markers in other poses.
    x=np.array([1.,0,0]);z=unit(np.cross(x,median_forward));y=unit(np.cross(z,x))
    return np.stack((x,y,z),axis=-1)

def coords(kind):
    ax=[np.arange(lo,hi+SPACING/2,SPACING) for lo,hi in BOUNDS[kind]]
    grid=np.stack(np.meshgrid(*ax,indexing='ij'),axis=-1)
    return ax,grid

def color_mask(im,palm_pixels):
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)
    interior=lab[palm_pixels[:,1],palm_pixels[:,0]]
    median=np.median(interior,axis=0)
    if median[0]<85 or np.linalg.norm(median[1:]-128)>34:return None
    chroma=np.linalg.norm(lab[:,:,1:]-median[1:],axis=2)
    ok=(lab[:,:,0]>max(55,median[0]*.49))&(chroma<28)
    # Close thin creases and the wrist's reflective fastening strip.
    ok=cv2.morphologyEx(ok.astype('uint8'),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8)).astype(bool)
    return ok

def calibrate(g):
    _,cams,rt,frames=geometry(g)
    ref=read(COMPAT/g/'summary.json')
    reviews=defaultdict(dict)
    for r in rows(COMPAT/g/'index.jsonl'):
        if r['pass_surface'] and not r['isolated_pose']:
            reviews[r['sync_index'],int(r['camera'][-1])][r['side']]=r
    out=DIR/g;out.mkdir(parents=True,exist_ok=True)
    for s,h in enumerate(HANDS):
        lib=Library(g,s,skin_weights())
        candidate=defaultdict(list)
        for (t,c),pair in reviews.items():
            if (len(pair)!=2 or t not in lib.correction or not np.isfinite(lib.markers[t,:3]).all()
                    or not np.isfinite(lib.j[t]).all()):continue
            candidate[t].append(c)
        good=np.array(sorted(t for t,v in candidate.items() if len(v)>=3))
        selected=good[np.unique(np.linspace(0,len(good)-1,min(24,len(good)),dtype=int))]
        base,cuff=cuff_frame(lib.markers[:,:3]);fwd=lib.bases[:,0,:,1]
        longitudinal=np.median(np.einsum('ti,tij->tj',fwd[selected],cuff[selected]),axis=0)
        basis=cuff_basis(longitudinal)
        loc={'cuff':base,'palm':lib.j[:,0]}
        bases={'cuff':cuff@basis,'palm':lib.bases[:,0]}
        geom={k:coords(k) for k in BOUNDS}
        hits={k:np.zeros(grid.shape[:3],np.int16) for k,(_,grid) in geom.items()}
        views=defaultdict(list);coverage={k:np.zeros_like(v) for k,v in hits.items()}
        for t in selected:
            xyz={k:grid.reshape(-1,3)@bases[k][t].T+loc[k][t] for k,(_,grid) in geom.items()}
            rr,tt=lib.correction[t]
            for c in candidate[t]:
                rot,off=rt[c];cal=cams[c];hand=reviews[t,c][s];opp=reviews[t,c][1-s]
                with open(COMPAT/g/hand['label'],'rb') as f:label=np.load(f,allow_pickle=True).item()
                with open(COMPAT/g/opp['label'],'rb') as f:other=np.load(f,allow_pickle=True).item()
                b=label['bbox'];bo=other['bbox']
                if (np.maximum(0,np.minimum(b[2:],bo[2:])-np.maximum(b[:2],bo[:2]))).prod()>0:continue
                rv={k:project_rays((x@rr+tt)@rot.T+off,cal)[0] for k,x in xyz.items()}
                pix=np.vstack(list(rv.values()))
                if not np.isfinite(pix).all() or (pix<2).any() or (pix>=np.array(cal.image_size)-2).any():continue
                joint=lib.j[t];tips=joint[[0,5,9,13,17]]
                inner=(.36*tips[0]+.64*tips[1:]);center=inner.mean(0)
                probes=(center+.56*(inner-center))
                mids=np.rint(project_rays((probes@rr+tt)@rot.T+off,cal)[0]).astype(int)
                lo=np.maximum(np.floor(np.vstack((pix,mids)).min(0)).astype(int)-12,0);hi=np.minimum(np.ceil(np.vstack((pix,mids)).max(0)).astype(int)+13,cal.image_size)
                # The real raw JPEG here has the same source clock and native dimensions as the video.
                img=cv2.imread(str(COMPAT/g/hand['image']))
                if img is None:raise OSError(hand['image'])
                region=img[lo[1]:hi[1],lo[0]:hi[0]]
                mask=color_mask(region,mids-lo)
                if mask is None:continue
                for k,uv in rv.items():
                    xy=np.rint(uv-lo).astype(int);ins=mask[xy[:,1],xy[:,0]].reshape(hits[k].shape)
                    hits[k]+=ins;coverage[k]+=1
                views[int(t)].append(c)
        result=dict(cuff_basis=basis,calibration_sync=np.array(sorted(views)),calibration_view_count=sum(map(len,views.values())))
        for kind,(ax,grid) in geom.items():
            n=coverage[kind].max()
            if n<12:raise ValueError((g,h,kind,'too few isolated observations',n))
            mask=(hits[kind]/n)>.80
            mask=ndi.binary_fill_holes(ndi.binary_closing(mask,iterations=1))
            cc,count=ndi.label(mask)
            if count==0:raise ValueError('Empty glove surface')
            mask=cc==(1+np.argmax(np.bincount(cc.ravel())[1:]))
            verts,faces,*_=marching_cubes(np.pad(mask.astype(float),1),level=.5,spacing=(SPACING,)*3)
            verts+=np.array([a[0] for a in ax])-SPACING
            result[kind+'_v']=verts.astype('f');result[kind+'_f']=faces.astype(np.int32)
            result[kind+'_extent_mm']=np.stack((verts.min(0),verts.max(0)))*1000
        np.savez_compressed(out/(h+'.npz'),**result)
        print(g,h,result['calibration_view_count'],'cuff',result['cuff_extent_mm'].round(1).tolist(),'palm',result['palm_extent_mm'].round(1).tolist(),flush=True)
