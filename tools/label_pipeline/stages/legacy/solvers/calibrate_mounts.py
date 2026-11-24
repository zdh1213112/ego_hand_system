# Source work/group_003_right_mount_study_20261119/calibrate_mounts.py, SHA-256 57b993d3e932cda1a1f78e6afd4b1a2519f92d4bdb33dec0d86b63a437fbbca3

import case_settings as case

import numpy as np

from scipy.optimize import minimize

from study import units

def patch_fit(tri, target, radius_bounds):
    # Surface point = v0 + u*(v1-v0) + v*(v2-v0) + h*unit_normal; coordinates are millimetres.
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];normal=units(np.cross(e1,e2))
    design=np.stack((e1,e2,normal),-1).reshape(-1,3)
    goal=(target-tri[:,0]).reshape(-1)
    gram=design.T@design/len(tri);linear=design.T@goal/len(tri)
    def f(x):return .5*x@gram@x-linear@x
    fit=minimize(f,[.33,.33,2.],jac=lambda x:gram@x-linear,bounds=[(0,1),(0,1),radius_bounds],
                 constraints=[dict(type="ineq",fun=lambda x:1-x[0]-x[1],jac=lambda x:np.array([-1.,-1.,0.]))],
                 method="SLSQP",options=dict(maxiter=60,ftol=1.e-10))
    x=fit.x
    if np.min(x[:2]) < -1.e-6 or sum(x[:2])>1.000001:raise ValueError("Invalid barycentric weights")
    residual=(design@x-goal).reshape(-1,3)
    return np.r_[1-x[:2].sum(),x[:2]],x[2],np.linalg.norm(residual,axis=1)
