"""Materialize raw solver outputs, score held RGB spots and replay visibility decisions."""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

def link(source,destination):
    source=Path(source).resolve(strict=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if not destination.samefile(source):raise ValueError(destination)
    else:destination.symlink_to(source)

def compile_depth(case):
    source=Path(__file__).with_name('crop_depth.cpp')
    so=case.root/'records/crop_depth.so'
    so.parent.mkdir(parents=True,exist_ok=True)
    tmp=so.with_suffix('.partial.so')
    subprocess.run(['g++','-O3','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
    tmp.replace(so)

def setup(path):
    os.environ['LABEL_PIPELINE_CASE']=str(Path(path).resolve())
    sys.path.insert(0,str(Path(__file__).with_name('solvers')))
    import case_settings as case
    sys.path.insert(0,str(case.assets.parent))
    return case

def bundle(case):
    from common import output,HANDS
    from package import write_dataset
    from stages.signed import prepare
    group=case.recording
    src=case.root/'geometry_input'/group
    src.mkdir(exist_ok=True,parents=True)
    native=case.root/'native/processed/ego/normalized_multiview'
    link(native,src/'native')
    motion=case.stage('joint_motion_20261119')/group/'final/export/merged'
    link(motion,src/'native_views')
    tracks={}
    for h in HANDS:
        tracks[h]=case.stage('mcp_outward_20261119')/group/'final'/(h+'.npz')
        link(tracks[h],src/'tracks'/(h+'.npz'))
        link(output(group)/'outward/tracks'/(h+'.npz'),src/'wrist'/(h+'_outward.npz'))
        with np.load(tracks[h]) as z:
            mounts=src/'mounts';mounts.mkdir(exist_ok=True)
            np.savez_compressed(mounts/(h+'_mounts.npz'),marker_names=z['marker_names'][4:],topology_faces=z['faces'],triangles=z['faces'][z['attachment_faces'][1:]],barycentric=z['barycentric'][1:],normal_offset_m=z['attachment_local_offsets_m'][1:,2])
        link(mounts/(h+'_mounts.npz'),case.root/(h+'_mounts.npz'))
        for suffix in ('','_normal_m'):
            link(case.assets/(h+suffix+'.npy'),case.root/('glove_joint_regressor_'+h+suffix+'.npy'))
    paired=src/'paired_input'
    if not (paired/'summary.json').exists():
        staged=src/'.paired_partial'
        for d in ('labels','images'):(staged/d).mkdir(parents=True,exist_ok=True)
        write_dataset(group,staged,tracks=tracks,metadata={'fit_method':'legacy','approved_for_training':False})
        staged.rename(paired)
    prepare(src,case.root)
    from inputs import COMPAT
    compat=COMPAT/group;compat.mkdir(parents=True,exist_ok=True)
    for name in ('index.jsonl','images','labels','summary.json'):link(paired/name,compat/name)
    link(case.root/'sequence.npz',compat/'sequence.npz')
    return src

def screen(case,src):
    from stages.legacy.solvers import optical,assess,evaluate,assemble
    from common import HANDS,geometry,output
    from inputs import COMPAT,read,rows,dump
    g=case.recording;root=output(g);dst=src/'decisions';dst.mkdir(exist_ok=True)
    optical.run(g);_,cams,rt,_=geometry(g)
    spots=[]
    with np.load(case.root/'sequence.npz') as seq:
        glove=seq['joints_2d_glove'];mano=seq['joints_2d_mano']
    def measured(track):
        with np.load(track) as z:return {k:z[k] for k in ('pair_indices','predicted24','marker_valid')}
    for s,h in enumerate(HANDS):
        base={k:measured(root/k/(h+'.npz')) for k in ('hard','soft')}
        assess.score_optical(g,h,base,cams,rt)
        final=measured(src/'tracks'/(h+'.npz'))
        evaluate.compare_optics(g,h,{'soft35':base['soft'],'candidate':final},dst)
        grouped=defaultdict(list)
        with (dst/('held_spots_'+h+'.csv')).open() as f:
            for r in csv.DictReader(f):
                t,c,m=map(int,(r['sync'],r['camera'],r['marker']))
                if c in (1,2,3) and 4<=m<=23:grouped[t,c,m].append(r)
        with np.load(src/'tracks'/(h+'.npz')) as z:names=z['marker_names']
        for (t,c,m),rr in grouped.items():
            ij=np.searchsorted(final['pair_indices'],t)
            if final['pair_indices'][ij]!=t or not final['marker_valid'][ij,m]:raise ValueError((t,c,m))
            uv=np.array([[float(r['u']),float(r['v'])] for r in rr])
            if np.ptp(uv,axis=0).max()>.01:continue
            xy=glove[t,c,s,m-3];old=mano[t,c,s,m-3]
            err=np.linalg.norm(xy-uv[0])
            if np.isfinite(err):
                spots.append(dict(hand=h,sync=t,camera=c,marker=str(names[m]),joint=m-3,spot_u=uv[0,0],spot_v=uv[0,1],export_u=xy[0],export_v=xy[1],raw_px=err,old_px=float(np.linalg.norm(old-uv[0])),registered_px=float(np.median([float(r['candidate_px']) for r in rr]))))
    p=dst/'held_spots.csv'
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(spots[0]));w.writeheader();w.writerows(spots)
    link(p,COMPAT.parent/'metrics'/(g+'_spots.csv'))
    records=list(rows(COMPAT/g/'index.jsonl'))
    decisions=assemble.pose_screen(g,records,read(COMPAT/g/'summary.json'))
    dump(dst/'pose_screen.json',decisions)

def visibility(case,src):
    compile_depth(case)
    from inputs import DEST,HANDS,OPT,take
    from stages.signed import export
    from direction import sdk_in_fit_space
    from overlap import overlap
    import calibrate_glove_volume,recheck_visibility,check_visible_cuff
    g=case.recording;pre=DEST/g;pre.mkdir(parents=True,exist_ok=True)
    d=take(OPT[g]/'skeleton/aligned_skeleton.npz','positions_cam0_m','provisional21_indices','marker24_cam0_m')
    sk=d['positions_cam0_m'][:,:,d['provisional21_indices']].copy();original=sk.copy();mk=d['marker24_cam0_m']
    faces=[]
    for s,h in enumerate(HANDS):
        with np.load(src/'tracks'/(h+'.npz')) as z:
            t=z['pair_indices'];faces.append(np.ascontiguousarray(z['faces'],np.int32))
        sk[s,t]=sdk_in_fit_space(g,s,original,mk,t)
    substitute=np.concatenate([mk[:,:,:3].mean(2,keepdims=True),mk[:,:,4:]],axis=2)
    missing=~np.isfinite(sk).all(-1);sk[missing]=substitute[missing]
    with (pre/'occlusion.csv').open('w') as stream:
        log=csv.writer(stream);log.writerow(('sync','camera','side','other_crop_fraction','occluded_target_fraction','sdk_status','decision'))
        def collect(t,c,s,label,cal,rt):
            r,b=rt
            frac,occlusion,kind=overlap(label,faces[s],sk[1-s,t]@r.T+b,cal)
            log.writerow((t,c,s,frac,occlusion,kind,'exclude' if occlusion is not None and occlusion>case.params['occlusion_limit'] else 'retain'))
        export(src,case.root,single_observer=collect)
    # The color confirmation reads exactly the native exposure when no cached JPEG is available.
    (pre/'frames.jsonl').write_text('')
    calibrate_glove_volume.calibrate(g)
    recheck_visibility.audit(g,final=True)
    check_visible_cuff.audit(g)
    link(calibrate_glove_volume.DIR/'photo_decisions'/g/'decisions.csv',src/'decisions/photo.csv')
    summary=export(src,case.root,signed_reference=False)
    from shutil import copyfile
    copyfile(case.root/'index.jsonl',case.root/'exposure_index.jsonl')
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('phase',choices=('imports','bundle','export'));args=p.parse_args()
    from phase_record import record
    with record(args.case,'raw_'+args.phase):
        case=setup(args.case)
        if args.phase=='imports':
            compile_depth(case)
            import package,assess,evaluate,assemble,calibrate_glove_volume,recheck_visibility,check_visible_cuff
        elif args.phase=='bundle':screen(case,bundle(case))
        else:visibility(case,case.root/'geometry_input'/case.recording)
