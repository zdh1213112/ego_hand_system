#!/usr/bin/env python3
"""Regression: old palm-side selections must fail, corrected dorsal ones pass."""
import json
from pathlib import Path
import numpy as np
from run_pilot import check_dorsal_reference, ROOT, CameraCalibration
exp=ROOT/'output/native_kb_training_20261111';cal=exp/'real_dual/pair_20261111_192916/ego/normalized_multiview/calibration';results={}
for side,hand,camera,sync in [(0,'left','camera1',616),(1,'right','camera2',584)]:
    base=np.load(exp/f'fourview_192916/native_labels/mano_fit_full_cam01/track_{side}.npz');v=base['vertices'][np.flatnonzero(base['pair_indices']==sync)[0]]
    C=CameraCalibration.load(cal/(camera+'.json'));C0=CameraCalibration.load(cal/'camera0.json');origin=(C.T_base_camera[:3,3]-C0.T_base_camera[:3,3])@C0.T_base_camera[:3,:3]
    if side==0:origin[0]*=-1
    results[hand]={}
    for folder,expected in [('marker24_pilot_v1_run2',False),('marker24_pilot_hemisphere_minus',True)]:
        z=np.load(exp/f'{folder}/{hand}_pilot.npz')
        for key in ['initial_attachment_faces','attachment_faces']:
            try:
                result=check_dorsal_reference(v,z['faces'],z[key],origin)
                assert expected, 'Wrong hemisphere unexpectedly accepted'
            except ValueError:
                assert not expected, 'Correct hemisphere unexpectedly rejected'
                result={'rejected':True}
            results[hand][folder+'/'+key]=result
out=exp/'marker24_pilot_hemisphere_minus/direction_regression.json';out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
