# Source work/glove_trainset_20261120_v2/check_glove_volume.py, SHA-256 3c5a2588f6a809700f9efe32acaed764fc70923dac27c859d597e52384d36358

import case_settings as case

import numpy as np

from calibrate_glove_volume import DIR, cuff_frame

def surfaces(lib,template,t,surf):
    hand_faces=lib.faces
    group=[(surf,hand_faces)]
    palm=template['palm_v']@lib.bases[t,0].T+lib.j[t,0]
    a=lib.markers[t,:3]
    if np.isfinite(a).all():
        ctr,frame=cuff_frame(a)
        cuff=template['cuff_v']@(frame@template['cuff_basis']).T+ctr
    else:cuff=None
    if t in lib.correction:
        r,b=lib.correction[t];palm=palm@r+b
        if cuff is not None:cuff=cuff@r+b
    group.append((palm,template['palm_f']))
    if cuff is not None:group.append((cuff,template['cuff_f']))
    return group
