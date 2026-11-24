# Source work/glove_trainset_20261120_v2/recheck_visibility.py, SHA-256 dfa0e7b6db53fd5088f088b458a4491f8ebd0582593f0af5a5e18f22b0412a1d

import case_settings as case

import csv,sys

from collections import Counter

import numpy as np

from inputs import COMPAT, DEST, HANDS, read, take, geometry, dump

from surface_occluder import Library, skin_weights, depth

from calibrate_glove_volume import DIR

from check_glove_volume import surfaces

from overlap import N, project_rays

from export_dataset import _mesh_bbox

OUT=DIR/'all_single_views'

def audit(g,final=False):
    out=(DIR/'all_single_final' if final else OUT)/g;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return
    rows=list(csv.DictReader((DEST/g/'occlusion.csv').open()))
    _,cams,rt,_=geometry(g);weights=skin_weights();yy,xx=np.mgrid[:N,:N]
    uv_grid=np.stack((xx,yy),-1)+.5
    source=read(COMPAT/g/'summary.json')
    target=[take(source['fit_tracks'][h],'pair_indices','vertices','faces') for h in HANDS]
    mapping=[dict(zip(z['pair_indices'],range(len(z['pair_indices'])))) for z in target]
    libs=[Library(g,s,weights,repair_nails=final) for s in range(2)]
    templates=[dict(np.load(DIR/g/(h+'.npz'))) for h in HANDS]
    if final:
        for s,l in enumerate(libs):
            for name,fields,values in [('nail_patches',('sync','digit','direction_gap_deg','local_gate_deg','support'),l.nail_corrections),('nail_holdout',('sync','digit','nail_error_mm','direction_error_deg','would_flag'),l.nail_holdout)]:
                with (out/(HANDS[s]+'_'+name+'.csv')).open('w') as fp:
                    writer=csv.writer(fp);writer.writerow(fields);writer.writerows(values)
    last_t=None;memo={};stats=Counter();fractions=[]
    cols=['sync','camera','side','old_occluded','hand_only','hand_and_cuff','hand_palm_cuff','other_crop_fraction','decision','archive_alignment','cuff_available','visibility_status']
    with (out/'fractions.csv.part').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for row in rows:
            ti,c,s=map(int,(row['sync'],row['camera'],row['side']));af=1-s
            if last_t!=ti:memo={};last_t=ti
            lib=libs[af];r,tr=rt[c];cal=cams[c]
            if af not in memo:
                surf=lib.vertices(ti)
                memo[af]=None if surf is None else surfaces(lib,templates[af],ti,surf)
            if memo[af] is None:
                stats['missing_sdk']+=1
                stats['rows']+=1
                stats['exclude']+=1
                writer.writerow(dict(sync=ti,camera=c,side=s,decision='exclude',visibility_status='unknown_occluder_geometry'))
                continue
            i=mapping[s][ti];z=target[s]
            tv=(z['vertices'][i]*[1 if s else -1,1,1])@r.T+tr
            tuv,tz=project_rays(tv,cal)
            bbox=_mesh_bbox(tuv,cal.image_size,.1,.50)
            size=2*max(bbox[3]-bbox[1],(bbox[2]-bbox[0])*256/192)
            lo=(bbox[:2]+bbox[2:])*.5-size*.5;scale=N/size
            depths=[]
            for vertices,faces in memo[af]:
                uv,d=project_rays(vertices@r.T+tr,cal)
                valid=np.isfinite(uv).all(1)&np.isfinite(d)
                if valid.any() and (uv[valid].max(0)>lo).all() and (uv[valid].min(0)<lo+size).all():
                    dep=depth(uv,d,faces,lo,scale,N)
                else:dep=np.full((N,N),np.inf,np.float32)
                depths.append(dep)
            occ=frac=skin=cuff_frac=0.
            full=np.minimum.reduce(depths)
            if np.isfinite(full).any():
                td=depth(tuv,tz,np.ascontiguousarray(z['faces'],np.int32),lo,scale,N)
                xy=lo+uv_grid/scale;inside=((xy>=0)&(xy<cal.image_size)).all(-1)
                own=np.isfinite(td)&inside;den=own.sum()
                if not den:raise ValueError((g,ti,c,s,'empty target'))
                frac=(np.isfinite(full)&inside).sum()/N**2
                occ=((full<td)&own).sum()/den
                skin=((depths[0]<td)&own).sum()/den
                combo=np.minimum(depths[0],depths[2]) if len(depths)==3 else depths[0]
                cuff_frac=((combo<td)&own).sum()/den
                if final and .24<occ<.36:
                    # Rasterize the actual sensor pixels around the 30 percent decision boundary.
                    corner=np.maximum(0,np.floor(tuv.min(0)).astype(int)-2)
                    stop=np.minimum(cal.image_size,np.ceil(tuv.max(0)).astype(int)+3)
                    length=int(max(stop-corner));hh=stop[1]-corner[1];ww=stop[0]-corner[0]
                    td=depth(tuv,tz,np.ascontiguousarray(z['faces'],np.int32),corner,1.,length)[:hh,:ww]
                    own=np.isfinite(td);den=own.sum()
                    od=[]
                    for vertices,faces in memo[af]:
                        suv,sd=project_rays(vertices@r.T+tr,cal)
                        od.append(depth(suv,sd,faces,corner,1.,length)[:hh,:ww])
                    occ=((np.minimum.reduce(od)<td)&own).sum()/den
                    skin=((od[0]<td)&own).sum()/den
                    combo=np.minimum(od[0],od[2]) if len(od)==3 else od[0]
                    cuff_frac=((combo<td)&own).sum()/den
                    stats['native_resolution_boundary_checks']+=1
            old=float(row['occluded_target_fraction']) if row['occluded_target_fraction'] else None;decision='exclude' if occ>case.params['occlusion_limit'] else 'retain'
            stats['rows']+=1;stats[decision]+=1
            if old is not None and old>case.params['occlusion_limit'] and occ<=case.params['occlusion_limit']:stats['restore']+=1
            if old is not None and old<=case.params['occlusion_limit'] and occ>case.params['occlusion_limit']:stats['remove']+=1
            if (occ>case.params['occlusion_limit'])!=(cuff_frac>case.params['occlusion_limit']):stats['palm_changes_gate']+=1
            if (occ>case.params['occlusion_limit'])!=(skin>case.params['occlusion_limit']):stats['garment_changes_gate']+=1
            if .2<=occ<=.4:stats['near_threshold_20_40']+=1
            writer.writerow(dict(sync=ti,camera=c,side=s,old_occluded=old,hand_only=skin,hand_and_cuff=cuff_frac,hand_palm_cuff=occ,other_crop_fraction=frac,decision=decision,archive_alignment=ti in lib.correction,cuff_available=len(memo[af])==3))
    (out/'fractions.csv.part').replace(out/'fractions.csv')
    dump(out/'summary.json',dict(stats));print(g,dict(stats),flush=True)
