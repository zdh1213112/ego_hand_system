#!/usr/bin/env python3
"""Verify frozen shape, exact SDK target mapping and exported dataset provenance."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap=argparse.ArgumentParser()
for name in ['source','skeleton','output']:ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args();sdk=np.load(a.skeleton);checks={}
for mode in ['hand','finger0']:
    result={}
    for side,hand in enumerate(['left','right']):
        original=np.load(a.source/f'{hand}_pilot.npz');candidate=np.load(a.output/f'sdk_{mode}/{hand}_pilot.npz')
        for key in ['betas','attachment_faces','barycentric','normal_offsets_m','wrist_local']:assert np.array_equal(original[key],candidate[key]),(mode,hand,key)
        positions=sdk['positions_cam0_m'][side][candidate['pair_indices']][:,candidate['sdk_segment_indices']].copy();positions[:,:,0]*=-1 if side==0 else 1
        error=float(np.max(abs(positions-candidate['sdk_target21'])));assert error<1e-6
        result[hand]={'shape_and_attachments_exactly_unchanged':True,'target_mapping_max_error_m':error}
    data=a.output/f'sdk_{mode}/export/merged';summary=json.loads((data/'summary.json').read_text());assert summary['approved_for_training'] is False
    assert summary['assumptions']['sdk_marker_to_internal_joint_offsets_configured'] is True and summary['assumptions']['assumption_verified'] is False
    rows=[json.loads(s) for s in (data/'index.jsonl').read_text().splitlines()];groups=defaultdict(list)
    for row in rows:
        assert row['approved_for_training'] is False and row['sdk_internal_offset_verified'] is False and row['sdk_internal_offset_assumed'] is True
        sample=np.load(data/row['label'],allow_pickle=True).item();side=int(sample['side']);expected=np.load(a.source/f'{"left" if side==0 else "right"}_pilot.npz')['betas']
        assert np.array_equal(sample['mano']['betas'],expected)
        groups[row['image']].append(side)
    assert all(sorted(sides)==[0,1] for sides in groups.values())
    result['export']={'images':len(groups),'labels':len(rows),'every_image_has_both_hands':True,'assumptions_preserved':True}
    checks[mode]=result
report={'approved_for_training':False,'checks':checks,'input_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.skeleton,a.source/'summary.json']},'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'sdk_configuration_modified':False,'additional_2mm_subtraction_applied':False}
(a.output/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(checks,indent=2))
