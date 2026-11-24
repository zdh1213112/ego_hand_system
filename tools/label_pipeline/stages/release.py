"""Check the per-hand timeline and per-recording chronological partition."""
import json
from pathlib import Path
import numpy as np
from .signed import arrays,rows,write_json
def identity(r):
    return (r['sync_index'],r['camera'],r['side'])
def timeline(group,out):
    seq=arrays(out/'sequence.npz','pose_valid','pass_surface','isolated_pose')
    stats={}
    for side,h in enumerate(('left','right')):
        tr=arrays(group/'tracks'/f'{h}.npz','pair_indices','target24','predicted24','marker_valid','isolated_pose')
        t=tr['pair_indices'];v=tr['marker_valid']
        np.testing.assert_array_equal(t,np.flatnonzero(seq['pose_valid'][:,side]))
        error=np.where(v,np.linalg.norm(tr['target24']-tr['predicted24'],axis=-1)*1000,np.nan)[:,3:]
        mask=(v[:,4:].reshape(-1,5,4).sum(2).min(1)>=3)&(np.nanmedian(error,1)<=2.5)&(np.nanmax(error,1)<=8)
        np.testing.assert_array_equal(mask,seq['pass_surface'][t,side])
        np.testing.assert_array_equal(tr['isolated_pose'],seq['isolated_pose'][t,side])
        stats[h]={'valid':len(t),'surface_pass':int(mask.sum()),'isolated':int(tr['isolated_pose'].sum())}
    samples=list(rows(out/'index.jsonl'))
    syncs=sorted({r['sync_index'] for r in samples})
    cut=syncs[min(len(syncs)-1,max(1,round(len(syncs)*.9)))]
    original_groups=json.loads((group.parent/'MANIFEST.json').read_text())['groups']
    offset=sum(1 for g in original_groups[:original_groups.index(group.name)] for _ in rows(group.parent/g/'reference/index.jsonl'))
    orig={identity(r):r['index']+offset for r in rows(group/'reference/index.jsonl')}
    splits=json.loads((group.parent/'signoff/splits.json').read_text())['source_positions']
    for name in ('train','val'):
        positions=set(splits[name]);actual={identity(r) for r in samples if (r['sync_index']<cut)==(name=='train')}
        expected={k for k,position in orig.items() if position in positions}
        if actual!=expected:raise ValueError(f'{group.name} {name} 划分差异 {len(actual^expected)}')
        stats[name]=len(actual)
    stats['cut_sync']=cut
    write_json(out/'timeline.json',stats);return stats
