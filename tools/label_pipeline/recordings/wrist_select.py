"""Select measured, tip-preserving cuff references per hand. Signed data are used only for evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from recordings.preview import decode, image_peaks, source_geometry, video
from stages.geometry import project
from stages.legacy.solvers.wrist_transfer import prepared_targets, TIPS
from stages.signed import write_json
from stages.fingerprint import digest

WORK = PIPE.parents[2] / 'work'
HANDS = ('left', 'right')


def norm(x):
    return np.linalg.norm(x, axis=-1)


def layout(root, group, historical):
    if historical:
        native = WORK/'signed_labels_20261120'/group/'native'
        stages = WORK
    else:
        native = root/'native/processed/ego/normalized_multiview'
        stages = root/'legacy_stages'
    return (native, stages/'nokov_label_training_20261118'/group/'aligned',
            stages/'tip_pivot_20261119'/group, stages/'joint_motion_20261119')


def candidate_rgb(native, aligned, pivot, out):
    cams, rt, rows = source_geometry(native)
    source_files = [aligned/(h+'_pilot.npz') for h in HANDS] + [pivot/h/'geometry6_locked.npz' for h in HANDS]
    rgb_inputs = [native/'multiview_frames.csv', Path(__file__), PIPE/'recordings/preview.py', PIPE/'unified/optics.py']
    for c in range(len(cams)):
        rgb_inputs.extend([native/f'calibration/camera{c}.json',native/f'cameras/camera{c}/video.mp4'])
    identity = {str(f):digest(f) for f in [*source_files,*rgb_inputs]}
    cache = out/'cuff_rgb.json'
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['source'] == identity:
            return saved['measurements']
    xy = {}
    for side,h in enumerate(HANDS):
        with np.load(source_files[side+2]) as z:
            times = z['times']
        with np.load(source_files[side]) as z:
            t = z['pair_indices']
            pos = np.searchsorted(t, times)
            np.testing.assert_array_equal(t[pos], times)
            xyz = z['target24'][pos]
        for i,t in enumerate(times):
            if np.isfinite(xyz[i,:3]).all():
                xy[int(t),side] = xyz[i]
    records = {h:{} for h in HANDS}
    for c, cam in enumerate(cams):
        capture = video(native,c)
        try:
            for t in sorted({t for t,s in xy}):
                frame = decode(capture,int(rows[t][f'camera{c}_frame_index']))
                for side,h in enumerate(HANDS):
                    if (t,side) not in xy:
                        continue
                    pixels = project(xy[t,side], rt[c], cam, side)
                    matches = image_peaks(frame,pixels)
                    err = norm(matches[:3]-pixels[:3])
                    records[h].setdefault(str(t),{})[str(c)] = [float(v) if np.isfinite(v) else None for v in err]
        finally:
            capture.release()
    write_json(cache,dict(source=identity,measurements=records))
    return records


def read_arrays(path, keys):
    with np.load(path) as z:
        return {k:z[k] for k in keys}


def motion_source(motion,group,hand):
    data = read_arrays(motion/group/'final'/(hand+'_pilot.npz'),('pair_indices','joints','global_orient','target24'))
    clip = motion/'fast_fits'/group/hand
    if (clip/'summary.json').is_file() and json.loads((clip/'summary.json').read_text())['accept_for_this_clip']:
        with np.load(clip/'candidate.npz') as patch:
            _,a,b = np.intersect1d(data['pair_indices'],patch['times'][1:-1],return_indices=True)
            data['joints'][a] = patch['joints'][b+1]
    return data


def cuff_axes(c):
    x = c[:,2]-c[:,1]
    y = c[:,0]-(c[:,1]+c[:,2])*.5
    ex = x / np.maximum(norm(x)[:,None],1e-12)
    y = y-(ex*y).sum(1)[:,None]*ex
    ey = y / np.maximum(norm(y)[:,None],1e-12)
    return np.stack((ex,ey,np.cross(ex,ey)),axis=1), norm(y)


def evaluate_hand(group, hand, aligned,pivot,motion, rgb, signed):
    data = motion_source(motion,group,hand)
    times = data['pair_indices']
    if len(times)<2:
        return dict(status='insufficient_evidence',reason='fewer than two motion observations',candidates=[])
    source = read_arrays(aligned/(hand+'_pilot.npz'),('pair_indices','target24'))
    matching = np.searchsorted(source['pair_indices'],times)
    np.testing.assert_array_equal(source['pair_indices'][matching],times)
    cuff = source['target24'][matching,:3]
    valid = np.isfinite(cuff).all((1,2))
    cutoff = times[min(len(times)-1,max(1,round(.9*len(times))))]
    block = times//120%4
    dev = valid & (times<cutoff) & (block!=2)
    if dev.sum()<3:
        return dict(status='insufficient_evidence',reason='fewer than three complete development cuff observations',candidates=[])
    basis, height = cuff_axes(cuff)
    edges = np.stack([norm(cuff[:,a]-cuff[:,b])*1000 for a,b in [(0,1),(1,2),(2,0)]],axis=1)
    template = np.median(edges[dev],axis=0)
    bad_edges = norm(edges-template)/np.sqrt(3)
    locked = read_arrays(pivot/hand/'geometry6_locked.npz',('times','joints','global_orient','mounted','tip_mm','fraction'))
    entries = []
    tracks = {}
    candidate_indices=np.flatnonzero(np.isin(locked['times'],times[dev]))
    reach = norm(locked['joints'][:,TIPS[1:]]-locked['joints'][:,0,None]).mean(1)*1000
    finite = np.isfinite(reach) & np.isfinite(locked['tip_mm']).all(1) & np.isfinite(locked['fraction'])
    for key in ('global_orient','mounted','joints'):
        finite &= np.isfinite(locked[key]).all(axis=tuple(range(1,locked[key].ndim)))
    candidate_indices = candidate_indices[finite[candidate_indices]]
    pool = [i for i in candidate_indices if (locked['times'][i]//120)%4==0]
    if not pool:
        pool=list(candidate_indices)
    if not pool:
        return dict(status='insufficient_evidence',reason='no measured cuff reference',candidates=[])
    q75=np.percentile(reach[pool],75)
    baseline=int(locked['times'][min(pool,key=lambda i:abs(reach[i]-q75))])
    for i in candidate_indices:
        t=int(locked['times'][i]); idx=int(np.searchsorted(times,t))
        matches=rgb.get(str(t),{})
        views=[v for v in matches.values() if sum(x is not None for x in v)>=2]
        points=[x for v in views for x in v if x is not None]
        identities={j for v in views for j,x in enumerate(v) if x is not None}
        tip=float(locked['tip_mm'][i].max())
        endpoint={k:locked[k][i] for k in ('joints','global_orient','mounted')}
        wrist=prepared_targets(data,endpoint,idx)['wrist']
        tracks[t]=wrist
        local=np.einsum('nij,nj->ni',basis,wrist-cuff.mean(1))*1000
        use=dev & (np.abs(times-t)>8)
        held=valid & (block==2) & (np.abs(times-t)>120)
        if use.sum()<3:
            continue
        spread=norm(local[use]-np.median(local[use],axis=0))
        contiguous=use[1:-1]&use[2:]&use[:-2]&(np.diff(times)[:-1]==1)&(np.diff(times)[1:]==1)
        jitter=norm(local[2:]-2*local[1:-1]+local[:-2])[contiguous]
        if not len(jitter):
            continue
        e=dict(sync=t,views=len(views),named_balls=len(identities),observations=len(points),
               calibration_cuff_count=int((valid&(block==0)&(np.abs(times-t)>120)).sum()),
               held_cuff_count=int(held.sum()),
               cuff_reprojection_px=float(np.median(points)) if points else None,
               cuff_edge_mm=float(bad_edges[idx]),cuff_height_mm=float(height[idx]*1000),
               max_tip_mm=tip,pivot_fraction=float(locked['fraction'][i]),reach_mm=float(reach[i]),
               local_spread_mm=float(np.percentile(spread,75)),jitter_mm=float(np.percentile(jitter,75)))
        e['eligible']=bool(len(views)>=2 and len(identities)==3 and e['cuff_reprojection_px']<=8.
                           and tip<=3.5 and e['cuff_height_mm']>3. and e['cuff_edge_mm']<=2.5
                           and e['pivot_fraction']>0 and e['calibration_cuff_count']>=20 and e['held_cuff_count']>0)
        entries.append(e)
    candidates=[e for e in entries if e['eligible']]
    if not candidates:
        return dict(status='insufficient_evidence',baseline=baseline,candidates=entries,
                    reason='no reference passes cuff RGB, triangle, outward rotation and fingertip gates')
    fields=['cuff_reprojection_px','cuff_edge_mm','local_spread_mm','jitter_mm','max_tip_mm']
    values=np.array([[e[k] for k in fields] for e in candidates])
    # Below one pixel or millimetre, differences are too small for a different reference by themselves.
    scale=np.maximum(np.percentile(values,75,axis=0)-np.percentile(values,25,axis=0),[1,1,1,.5,1])
    for e,v in zip(candidates,values):
        e['score']=float(np.maximum(0,(v-values.min(0))/scale).mean())
    best=min(candidates,key=lambda e:(e['score'],-e['views'],abs(e['reach_mm']-q75),e['sync']))
    classic=next((e for e in candidates if e['sync']==baseline),None)
    if classic and classic['score']<=best['score']+0.5:
        best=classic
    t=best['sync']
    result=dict(status='selected',sync=t,baseline=baseline,chosen=best,candidates=entries,
                holdout_start=int(cutoff),selection_blocks=[0,1,3],score_features=fields)
    held=valid&(block==2)&(np.abs(times-t)>120)
    if held.any():
        loc=np.einsum('nij,nj->ni',basis,tracks[t]-cuff.mean(1))*1000
        result['held_cuff_variation_mm']=float(np.median(norm(loc[held]-np.median(loc[dev],axis=0))))
    if signed:
        anchors=json.loads((signed/'wrist/anchors.json').read_text())
        published=read_arrays(signed/'wrist'/(hand+'_outward.npz'),('pair_indices','joints'))
        _,i,j=np.intersect1d(times,published['pair_indices'],return_indices=True)
        common=held[i] & (np.abs(times[i]-baseline)>120)
        if not common.any():
            raise ValueError('empty historical test block')
        result['historical']={'signed_sync':anchors[hand]['sync_index'],'held_count':int(common.sum())}
        for name,choice in [('selected',t),('classic',baseline)]:
            if choice not in tracks:continue
            delta=norm(tracks[choice][i[common]]-published['joints'][j[common],0])*1000
            result['historical'][name+'_p50_mm'],result['historical'][name+'_p95_mm']=map(float,np.percentile(delta,[50,95]))
    return result


def run(root,group,out,historical=False):
    out.mkdir(parents=True,exist_ok=True)
    native,aligned,pivot,motion=layout(root,group,historical)
    rgb=candidate_rgb(native,aligned,pivot,out)
    signed=WORK/'signed_labels_20261120'/group if historical else None
    results={h:evaluate_hand(group,h,aligned,pivot,motion,rgb[h],signed) for h in HANDS}
    write_json(out/'selection.json',dict(group=group,historical_evaluation=historical,hands=results))
    print(group,[(h,v['status'],v.get('sync'),v.get('baseline'),v.get('historical')) for h,v in results.items()],flush=True)
    return results


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--historical',action='store_true')
    arg=ap.parse_args();base=WORK/'labels_20261120';root=base/'legacy'/arg.group
    out=base/'adaptive_wrist'/'historical'/arg.group if arg.historical else root/'review/adaptive_wrist'
    run(root,arg.group,out,arg.historical)
