import sys,json,importlib.util
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[4];p=root/'tools/label_pipeline/calibration/marker24_pilot/run_pilot.py';spec=importlib.util.spec_from_file_location('pilot',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(9);faces=np.array([[0,1,2],[3,4,5]]);v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,2],[1,0,2],[0,1,2]],float)[None]+rng.normal(size=(12,1,3));target=(v[:,:3]*np.array([.2,.3,.5])[None,:,None]).sum(1)[:,None];fi,b=m.discrete(v,faces,target,[np.array([0,1])]);assert fi[0]==0;np.testing.assert_allclose(b[0],[.2,.3,.5],atol=1e-6)
pilot=root/'output/native_kb_training_20261111/marker24_pilot_v1_run2';summary=json.loads((pilot/'summary.json').read_text());summary.update(wrist_weight=.2,validation_caveat='Baseline initialization itself used fingertip targets; heldout-tip exclusion applies only to the new refinement.',temporal_regularization=False)
checks={'synthetic_multiframe_attachment_recovery':'pass','hands':{}}
for hand in ['left','right']:
 z=np.load(pilot/f'{hand}_pilot.npz');train=z['calibration_mask'];frames=z['pair_indices'];assert not set(frames[train])&set(frames[~train]);assert np.all((frames[~train]>=512)&(frames[~train]<704));assert not np.any((frames[train]>=512)&(frames[train]<704));assert np.isfinite(z['vertices']).all();assert (z['barycentric']>=0).all();np.testing.assert_allclose(z['barycentric'].sum(-1),1,atol=1e-6)
 ids=np.argsort(frames[~train]);sync=frames[~train][ids];delta=(z['joints'][~train]-z['baseline_joints'][~train])[ids];rates=np.linalg.norm(np.diff(delta,axis=0),axis=-1)/((np.diff(sync)/30)[:,None])*1000
 summary['hands'][hand]['correction_change_mm_per_s']=m.stats(rates);checks['hands'][hand]={'finite':True,'split_disjoint':True,'valid_barycentric':True}
review=json.loads((pilot/'review/data.js').read_text()[len('window.LABEL_DATA='):-2]);dataset=root/'output/native_kb_training_20261111/fourview_192916/native_labels/export/merged';rows=[json.loads(s) for s in (dataset/'index.jsonl').read_text().splitlines()];lookup={(r['camera'],r['sync_index'],r['side']):r for r in rows};errs=[]
for f in review['frames']:
 for h in f['baseline_hands']:
  r=lookup.get((f['camera'],f['sync'],h['side']))
  if r:
   sample=np.load(dataset/r['label'],allow_pickle=True).item();errs.append(float(np.max(np.abs(np.asarray(h['vertices'])[:,:2]-np.asarray(sample['joints_2d'])))))
assert max(errs)<.001;checks['baseline_projection_comparison']={'samples':len(errs),'max_pixel_difference':max(errs)}
summary['decision']='experimental_only_not_accepted';summary['reason']='Marker objective improved, but fingertip checks and visual review do not establish consistent improvement; shape fixed, wrist rigidity approximate, no independent anatomical truth.'
(pilot/'summary.json').write_text(json.dumps(summary,indent=2));(pilot/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
