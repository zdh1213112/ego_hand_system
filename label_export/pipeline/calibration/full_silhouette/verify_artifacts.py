#!/usr/bin/env python3
"""Check parameter isolation and record reproducible inputs for the completed experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
ap=argparse.ArgumentParser()
for name in ['context','selection','source','output']:ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[a.context,a.selection,a.source/'shared_calibration.json']
checks={}
for mode in ['shape_only','shape_pose']:
    for hand in ['left','right']:
        old=np.load(a.source/f'{hand}_pilot.npz');new=np.load(a.output/mode/f'{hand}_pilot.npz')
        assert new['betas'].shape==(10,) and np.isfinite(new['vertices']).all()
        for key in ['attachment_faces','barycentric','normal_offsets_m','wrist_local']:
            assert np.array_equal(old[key],new[key]),(mode,hand,key)
        mask=np.ones(len(old['pair_indices']),bool) if mode=='shape_only' else ~old['calibration_mask']
        for key in ['hand_pose_pca','global_orient','translation']:
            assert np.array_equal(old[key][mask],new[key][mask]),(mode,hand,key)
        checks[mode+'_'+hand]={'one_shared_beta_vector':True,'fixed_attachments':True,'required_motion_exactly_unchanged':True}
for hand in ['left','right']:inputs.append(a.source/f'{hand}_pilot.npz')
renderer=json.loads((a.output/'renderer_checks.json').read_text());assert renderer['checks_passed']
report={'approved_for_training':False,'experiment':'full_mask_shared_shape_v1','renderer_commit':'253ac4fcea7de5f396371124af597e6cc957bfae','input_sha256':{str(p.resolve()):sha(p) for p in inputs},'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'checks':checks,'mano_fit_directories_present':all((a.output/f'mano_fit_{mode}/summary.json').exists() for mode in ['shape_only','shape_pose']),'original_training_pipeline_changed':False,'wilor_training_performed':False}
(a.output/'provenance.json').write_text(json.dumps(report,indent=2));print('Parameter isolation verified; provenance saved')
