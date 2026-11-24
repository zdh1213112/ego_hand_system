# Source work/glove_trainset_20261120_v2/check_visible_cuff.py, SHA-256 d745a85fbc0739ddb1b9fccc1d1fbabecc250e3ff9efeb31e678740949f5ec69

import case_settings as case

from collections import Counter

import csv,cv2

import numpy as np

from inputs import DEST, COMPAT, HANDS, read, rows, take, geometry, dump

from surface_occluder import Library, skin_weights, depth

from overlap import project_rays

from check_glove_volume import DIR, surfaces

from observations import read_rgb

def cloth(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    L,a,b=np.moveaxis(lab,-1,0)
    gray=(L>=48)&(a>=109)&(a<=145)&(b>=100)&(b<=140)&(img[:,:,0].astype(float)>=img[:,:,2]*.88)
    gold=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=34)&(hsv[:,:,1]>150)&(L>=55)
    near=cv2.distanceTransform((~gray).astype('uint8'),cv2.DIST_L2,3)
    # The gold tape on the glove is surrounded by white fabric. A wooden table or
    # skin alone cannot support this extra surface outside the SDK-driven hand.
    return gray|(gold&(near<=max(2.,min(img.shape[:2])*.075)))

def audit(g):
    source_dir=DIR/'all_single_final'/g;dst=DIR/'photo_decisions'/g;dst.mkdir(parents=True,exist_ok=True)
    if (dst/'summary.json').exists():return
    tt=read(COMPAT/g/'summary.json');original=DEST/g
    image={(r['sync_index'],int(r['camera'][-1])):r['image'] for r in rows(original/'frames.jsonl')}
    native,cals,rt,frame=geometry(g);weights=skin_weights();caps={};stats=Counter()
    trk=[take(tt['fit_tracks'][h],'vertices','faces','pair_indices') for h in HANDS]
    slots=[dict(zip(z['pair_indices'],range(len(z['pair_indices'])))) for z in trk]
    libs=[Library(g,s,weights,repair_nails=True) for s in (0,1)]
    templates=[dict(np.load(DIR/g/(h+'.npz'))) for h in HANDS]
    with (source_dir/'fractions.csv').open() as fi,(dst/'decisions.part').open('w') as fo:
        rd=csv.DictReader(fi);wr=csv.DictWriter(fo,fieldnames=rd.fieldnames+['occluded_target_fraction','confirmed_extra_fraction','native_garment_review']);wr.writeheader()
        for row in rd:
            if row.get('visibility_status')=='unknown_occluder_geometry':
                stats['rows']+=1
                stats['exclude']+=1
                stats['unknown_occluder_geometry']+=1
                wr.writerow(dict(row,decision='exclude',native_garment_review=0))
                continue
            t,c,s=map(int,(row['sync'],row['camera'],row['side']));before=float(row['hand_palm_cuff']);hand=float(row['hand_only'])
            estimate=before;extra=0.;checked=0
            if before-hand>.00001:
                checked=1;af=1-s;l=libs[af];r,b=rt[c];cal=cals[c];z=trk[s]
                native_uv,dist=project_rays((z['vertices'][slots[s][t]]*[1 if s else -1,1,1])@r.T+b,cal)
                lo=np.maximum(np.floor(native_uv.min(0)).astype(int)-3,0);hi=np.minimum(np.ceil(native_uv.max(0)).astype(int)+4,cal.image_size)
                size=int(max(hi-lo));w,h=hi-lo
                td=depth(native_uv,dist,np.ascontiguousarray(z['faces'],np.int32),lo,1,size)[:h,:w]
                parts=surfaces(l,templates[af],t,l.vertices(t));dep=[]
                for v,fs in parts:dep.append(depth(*project_rays(v@r.T+b,cal),fs,lo,1,size)[:h,:w])
                own=np.isfinite(td);den=own.sum();skin=(dep[0]<td)&own
                if (t,c) in image:im=cv2.imread(str(original/image[t,c]))
                else:
                    if c not in caps:caps[c]=cv2.VideoCapture(str(native/'cameras'/f'camera{c}'/'video.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
                    im=read_rgb(caps[c],int(frame[t][f'camera{c}_frame_index']))
                pic=im[lo[1]:hi[1],lo[0]:hi[0]]
                extra_mask=(np.minimum.reduce(dep[1:])<td)&own&~skin&cloth(pic)
                extra=extra_mask.sum()/den;estimate=skin.sum()/den+extra
            decision='exclude' if estimate>case.params['occlusion_limit'] else 'retain';old=float(row['old_occluded']) if row['old_occluded'] else None
            stats['rows']+=1;stats[decision]+=1;stats['original_rgb_checked']+=checked
            if decision!=row['decision']:stats['photo_changes_gate']+=1
            if decision=='retain' and old is not None and old>case.params['occlusion_limit']:stats['restore']+=1
            if decision=='exclude' and old is not None and old<=case.params['occlusion_limit']:stats['remove']+=1
            row.update(occluded_target_fraction=estimate,decision=decision,confirmed_extra_fraction=extra,native_garment_review=checked)
            wr.writerow(row)
    for cap in caps.values():cap.release()
    (dst/'decisions.part').replace(dst/'decisions.csv')
    dump(dst/'summary.json',dict(stats));print(g,dict(stats),flush=True)
