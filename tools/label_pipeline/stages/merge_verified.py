"""Assemble verified exposures and validate them with the production MANO checker and loader."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .regression import key
from .signed import rows,write_json,arrays

def rejection_rows(src,candidate,accepted):
    views={}
    for r in rows(src/'native_views/index.jsonl'):
        views.setdefault((r['sync_index'],r['camera']),set()).add(r['side'])
    for r in rows(src/'native_views/rejected.jsonl'):
        if r['reason']=='both_hands_required':views[r['sync_index'],r['camera']]=r['available_sides']
    photo={}
    with (src/'decisions/photo.csv').open() as stream:
        for r in csv.DictReader(stream):
            photo[int(r['sync']),f"camera{r['camera']}",int(r['side'])]=r
    excluded=set(json.loads((src/'decisions/pose_screen.json').read_text())['excluded'])
    seq=arrays(candidate/'sequence.npz','pose_valid','pass_surface','isolated_pose','joints_2d_mano')
    ref=json.loads((src/'reference/summary.json').read_text())
    for (t,c),sides in sorted(views.items()):
        for s in sorted(sides):
            k=(t,c,s)
            if k in accepted:continue
            extra={}
            if not seq['pose_valid'][t,s]:reason='missing_final_fit'
            elif not seq['pass_surface'][t,s]:reason='pass_surface=false'
            elif seq['isolated_pose'][t,s]:reason='isolated_pose=true'
            elif t in excluded:reason='round10_screen'
            elif k in photo and photo[k].get('visibility_status')=='unknown_occluder_geometry':
                reason='unknown_occluder_geometry'
            elif k in photo and photo[k]['decision']=='exclude':
                reason='other_hand_occludes_target_over_30pct'
                extra={k:float(photo[t,c,s][k]) for k in ('occluded_target_fraction','other_crop_fraction')}
            else:
                uv=seq['joints_2d_mano'][t,int(c[-1]),s]
                width,height=ref['camera_calibrations'][c]['image_size']
                # Upstream already tested the mesh plane/visibility on this exposure.
                if not np.all(np.isfinite(uv)&(uv>=0)&(uv<[width,height])):reason='joints_outside_native_image'
                else:
                    from .geometry import CameraCalibration,reference_to_camera
                    h=('left','right')[s]
                    cal=[CameraCalibration.load(src/f'native/calibration/camera{i}.json') for i in (0,int(c[-1]))]
                    rot,shift=reference_to_camera(*cal)
                    z=arrays(src/'tracks'/f'{h}.npz','pair_indices','vertices')
                    v=z['vertices'][np.searchsorted(z['pair_indices'],t)]
                    depth=v@(rot[2]*np.array([1 if s else -1,1,1]))+shift[2]
                    reason='mesh_crosses_camera_plane' if depth.min()<=1e-8 else 'mesh_visibility_below_threshold'
            yield dict(sync_index=t,camera=c,side=s,source_group=src.name,reason=reason,**extra)

def merge(frozen,root):
    out=root/'dataset';pending=root/'.merged_candidate'
    if out.is_dir():return out
    for name in ('images','labels','labels_with_glove'): (pending/name).mkdir(parents=True,exist_ok=True)
    def link(source,destination):
        if destination.is_symlink():
            if not destination.samefile(source):raise ValueError(destination)
        else:destination.symlink_to(os.path.relpath(source,destination.parent))
    def line(f,r):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    groups=json.loads((frozen/'MANIFEST.json').read_text())['groups']
    n=images=rejections=0
    summary=json.loads((frozen/'signoff/summary.json').read_text())
    with (pending/'index.jsonl').open('w') as index,(pending/'index_with_glove.jsonl').open('w') as rich,(pending/'frames.jsonl').open('w') as frames,(pending/'rejected.jsonl').open('w') as rejected,(pending/'train.jsonl').open('w') as train,(pending/'val.jsonl').open('w') as val:
        for s,g in enumerate(groups):
            candidate=root/'legacy'/g;source=frozen/g
            audit=json.loads((candidate/'regression.json').read_text());count=audit['expected']
            for metric in ('all_fields_equal','label_storage_normalized_sha256_match','glove_label_storage_normalized_sha256_match','image_sha256_match'):
                if audit['counts'][metric]!=count:raise ValueError((g,metric))
            cutoff=json.loads((candidate/'timeline.json').read_text())['cut_sync']
            refs={key(r):r for r in rows(source/'reference/index.jsonl')};imus={}
            for r in rows(candidate/'index.jsonl'):
                ref=refs[key(r)].copy();label=r['label'];im=r['image'];extra=r['glove_label']
                if im not in imus:
                    merged=f'images/{images:06d}.jpg';images+=1;link(candidate/im,pending/merged)
                    imus[im]=dict(image=merged,**{k:r[k] for k in ('sync_index','camera','source_frame_index','reference_timestamp_ns')},source_group=g,calibration_group=str(s),source_dataset_id=s,hands=[])
                frame=imus[im];lbl=f'labels/{n:06d}.npy';gl=f'labels_with_glove/{n:06d}.npy';link(candidate/label,pending/lbl);link(candidate/extra,pending/gl)
                ref.pop('glove_label')
                ref.update(index=n,image=frame['image'],label=lbl,source_index=r['index'],source_dataset=str(candidate),source_image=im,source_label=label,source_dataset_id=s,calibration_group=str(s),fit_method='legacy',approved_for_training=False)
                frame['hands'].append(dict(index=n,side=r['side'],label=lbl));n+=1
                line(index,ref);line(rich,dict(ref,label=gl));line(train if r['sync_index']<cutoff else val,ref)
            for fr in imus.values():line(frames,dict(fr,hand_count=len(fr['hands'])))
            for r in rejection_rows(source,candidate,refs.keys()):line(rejected,r);rejections+=1
    for k,actual in [('sample_count',n),('image_count',images),('rejected_count',rejections)]:
        if summary[k]!=actual:raise ValueError((k,summary[k],actual))
    refs=json.loads((frozen/'signoff/splits.json').read_text());refs['source']=str(out/'index.jsonl');write_json(pending/'splits.json',refs)
    summary.update(approved_for_training=False,fit_method='legacy',source_datasets=[str(root/'legacy'/g) for g in groups])
    write_json(pending/'summary.json',summary);pending.rename(out);return out

def verify_loaders(root,repo):
    import yaml
    from .fingerprint import digest
    defaults=yaml.safe_load((Path(__file__).resolve().parents[1]/'defaults.yaml').read_text())
    training_model=(repo.parent/defaults['training_mano_directory']).resolve(strict=True)
    if digest(training_model/'models/MANO_RIGHT.pkl')!=digest(repo/'models/mano/MANO_RIGHT.pkl'):
        raise ValueError('训练读取器的 MANO_RIGHT.pkl 与标签生成模型不一致')
    for name,cmd in [('geometry',[repo/'scripts/check_wilor_training_dataset.py',root,'--mano-source',repo/'third_party/MANO','--mano-model-dir',repo/'models/mano','--device','cpu','--batch-size','128']),('profile',[repo.parent/'wilor_training/prepare_glove_dataset.py','--root',root,'--mano-path',training_model,'--validate-only'])]:
        with (root/(name+'_check.log')).open('w') as stream:
            subprocess.run([sys.executable,'-Bu',*map(str,cmd)],cwd=repo if name=='geometry' else repo.parent/'wilor_training',stdout=stream,stderr=subprocess.STDOUT,check=True)
    sys.path.insert(0,str(repo.parent/'wilor_training'))
    from wilor.datasets.glove_dataset import GloveDataset
    import torch
    counts={}
    for split in ('train','val'):
        dataset=GloveDataset(root/(split+'.jsonl'),camera_model='physical_kb')
        if not len(dataset):raise ValueError(f'{split} 没有可读取的数据')
        requested=min(500,len(dataset))
        sample=np.linspace(0,len(dataset)-1,requested).round().astype(int)
        if len(set(sample))!=requested:raise ValueError(split)
        for i in sample:
            row=dataset[int(i)]
            if not torch.isfinite(torch.as_tensor(row['img'])).all():raise ValueError((split,i))
        counts[split]=dict(total=len(dataset),loaded=len(sample))
    write_json(root/'loader_check.json',counts)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('frozen',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();dst=merge(a.frozen.resolve(),a.output.resolve());verify_loaders(dst,Path(__file__).resolve().parents[3])
