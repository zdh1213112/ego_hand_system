# Source work/glove_trainset_20261120_v2/overlap.py, SHA-256 1a0bcbeda8e782773717a6281aa71dfa454ab485bf357397053cac26f1803554

import case_settings as case

import ctypes

from functools import lru_cache

import numpy as np

from numpy.ctypeslib import ndpointer

N=160

raster=ctypes.CDLL(str(case.root/'records/crop_depth.so')).render

EDGES=np.array([(j,j+1) for start in (1,5,9,13,17) for j in range(start,start+3)])

RADII=np.repeat([.010,.009,.009,.0085,.0075],3)

ANGLES=np.arange(10)*2*np.pi/10

CS,SS=np.cos(ANGLES),np.sin(ANGLES)

PALM=np.array([0,1,5,9,13,17])

def topology():
    # Four 10-vertex rings enclose each capped bone with a rounded joint at either end.
    fs=[]
    for edge in range(len(EDGES)):
        start=40*edge
        for ring in range(3):
            for j in range(10):
                a=start+ring*10+j;b=start+ring*10+(j+1)%10
                fs.extend([(a,b,a+10),(a+10,b,b+10)])
        for r in (0,3):
            b=start+r*10;fs.extend((b,b+j,b+j+1) for j in range(1,9))
    start=40*len(EDGES)
    for side in (0,6):fs.extend((start+side,start+side+j,start+side+j+1) for j in range(1,5))
    for j in range(6):
        a=start+j;b=start+(j+1)%6;fs.extend([(a,b,a+6),(b,b+6,a+6)])
    return np.ascontiguousarray(fs,np.int32)

FACES=topology()

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def shell(points):
    a=points[EDGES[:,0]];b=points[EDGES[:,1]]
    axis=unit(b-a);ref=np.zeros_like(axis)
    ref[np.arange(len(EDGES)),np.argmin(np.abs(axis),axis=1)]=1
    u=unit(np.cross(axis,ref));v=np.cross(axis,u)
    disk=(u[:,None]*CS[None,:,None]+v[:,None]*SS[None,:,None])*RADII[:,None,None]
    pad=axis*RADII[:,None]*.8
    rings=np.stack([(a-pad)[:,None]+disk*.45,a[:,None]+disk,b[:,None]+disk,(b+pad)[:,None]+disk*.45],axis=1)
    # Offset the palm dorsally and ventrally in real metres.
    normal=unit(np.cross(points[5]-points[0],points[17]-points[0]))*.012
    palm=points[PALM]
    return np.vstack((rings.reshape(-1,3),palm+normal,palm-normal))

@lru_cache(maxsize=64)
def lens_limit(coeffs,max_radius):
    theta=np.linspace(0,np.pi,4097);t2=theta*theta;a,b,c,d=coeffs
    radial=theta*(1+t2*(a+t2*(b+t2*(c+t2*d))))
    stop=np.flatnonzero(np.diff(radial)<=0)
    n=int(stop[0]+1) if len(stop) else len(theta)
    return float(np.interp(max_radius+.25,radial[:n],theta[:n]))

def project_rays(xyz,cal):
    # KB uses polar angle, including rays grazing the optical plane. Projection
    # of finite physical radii stays finite there; pinhole focal_length/z does not.
    dist=np.linalg.norm(xyz,axis=1);radius=np.linalg.norm(xyz[:,:2],axis=1)
    theta=np.arctan2(radius,xyz[:,2]);t2=theta*theta;d=cal.distortion
    r=theta*(1+t2*(d[0]+t2*(d[1]+t2*(d[2]+t2*d[3]))))
    uv=(xyz[:,:2]*np.divide(r,radius,out=np.zeros_like(r),where=radius>1e-12)[:,None])@cal.K[:2,:2].T+cal.K[:2,2]
    # Rearward rays can still hit peripheral KB pixels. Keep the monotonic
    # calibrated branch until beyond the furthest sensor corner, including a margin.
    wh=np.asarray(cal.image_size);far=np.maximum(cal.K[:2,2],wh-cal.K[:2,2])/np.diag(cal.K)[:2]
    limit=lens_limit(tuple(d),float(np.linalg.norm(far)))
    dist[(theta>limit)|(dist<=1e-6)]=np.nan
    return uv,dist

def overlap(label,faces,other,cam):
    bbox=label['bbox'];mid=(bbox[:2]+bbox[2:])/2
    size=2*max(bbox[3]-bbox[1],(bbox[2]-bbox[0])*256/192)
    scale=N/size;lo=mid-size/2
    finite=np.isfinite(other).all(1)
    if not finite.any():return None,None,'missing'
    v=shell(other);uv,dist=project_rays(v,cam)
    valid=np.isfinite(uv).all(1)&np.isfinite(dist)
    if not valid.any():return 0.,0.,'outside_lens' if finite.all() else 'partial_outside_lens'
    mapped=(uv-lo)*scale;status='complete' if finite.all() else 'partial'
    if (mapped[valid].max(0)<0).any() or (mapped[valid].min(0)>N).any():return 0.,0.,status
    other_depth=np.full((N,N),np.inf,np.float32)
    triangles=FACES[valid[FACES].all(1)]
    raster(N,len(triangles),np.ascontiguousarray(triangles),np.ascontiguousarray(np.column_stack((mapped,dist)),np.float32),other_depth)
    yy,xx=np.ogrid[:N,:N];width,height=cam.image_size
    sensor=(lo[0]+(xx+.5)/scale>=0)&(lo[0]+(xx+.5)/scale<width)&(lo[1]+(yy+.5)/scale>=0)&(lo[1]+(yy+.5)/scale<height)
    mask=np.isfinite(other_depth)&sensor
    frac=float(mask.sum()/N**2)
    if not mask.any():return 0.,0.,status
    target_depth=np.full((N,N),np.inf,np.float32)
    xyz=label['vertices']+label['trans']
    target=np.column_stack(((np.asarray(label['joints_2d'])-lo)*scale,np.linalg.norm(xyz,axis=1)))
    raster(N,len(faces),faces,np.ascontiguousarray(target,np.float32),target_depth)
    own=np.isfinite(target_depth)&sensor;assert own.any()
    return frac,float(((other_depth<target_depth)&own).sum()/own.sum()),status

raster.argtypes=[ctypes.c_int,ctypes.c_int,ndpointer(np.int32,flags='C_CONTIGUOUS'),
                 ndpointer(np.float32,flags='C_CONTIGUOUS'),ndpointer(np.float32,flags='C_CONTIGUOUS')]
raster.restype=None
