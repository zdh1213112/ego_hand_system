"""Compare native annotation pixels and actual GloveDataset crop output on the same real RGB exposure."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACKAGE))
from recordings.preview import read_sample,video,decode
from stages.signed import write_json


def check(out,native_for):
    sys.path.insert(0,str(PACKAGE.parents[1]/'training'))
    import wilor.datasets.glove_dataset as glove
    from unittest.mock import patch
    GloveDataset=glove.GloveDataset
    report=out/'pixel_audit.json';proof=json.loads(report.read_text());temp=out/'loader_audit';temp.mkdir(exist_ok=True)
    rows=[]
    for i,r in enumerate(proof['samples']):
        path=(out/r['sample']).resolve();sample=read_sample(path)
        native=native_for(path);cap=video(native,sample['review_camera'])
        try:im=decode(cap,sample['source_frame_index'])
        finally:cap.release()
        jpg=temp/f'{i:02}.jpg'
        if not cv2.imwrite(str(jpg),im):raise IOError(jpg)
        rows.append(dict(index=i,image=str(jpg.resolve()),label=str(path),side=sample['review_side'],
            sync_index=sample['review_sync'],camera=f"camera{sample['review_camera']}",source_frame_index=sample['source_frame_index']))
    manifest=temp/'index.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ds=GloveDataset(manifest,camera_model='physical_kb')
    for i,row in enumerate(rows):
        label=read_sample(Path(row['label']));native_pixels=[];transforms=[]
        project=cv2.fisheye.projectPoints;crop=glove.generate_image_patch_cv2
        def record_projection(*args,**kwargs):
            result=project(*args,**kwargs)
            if result[0].shape==(21,1,2):native_pixels.append(result[0][:,0].astype(np.float32))
            return result
        def record_transform(*args,**kwargs):
            result=crop(*args,**kwargs);transforms.append(result[1].copy());return result
        with patch.object(cv2.fisheye,'projectPoints',record_projection),patch.object(glove,'generate_image_patch_cv2',record_transform):
            sample=ds[i]
        if len(native_pixels)!=1 or len(transforms)!=1:raise ValueError('训练读取器调用路径变化')
        np.testing.assert_array_equal(native_pixels[0],label['training_joints_2d_native'])
        uv=label['training_joints_2d_native'].copy()
        if row['side']==0:uv[:,0]=sample['image_size'][0].item()-uv[:,0]-1.0
        homogeneous=np.c_[uv,np.ones(21,np.float32)]
        transform=transforms[0]
        expected=np.asarray([transform@p for p in homogeneous],np.float32)/ds.image_size-.5
        actual=sample['keypoints_2d'].numpy()[:,:2]
        np.testing.assert_array_equal(expected,actual)
        proof['samples'][i]['max_real_loader_native_px']=float(abs(native_pixels[0]-label['training_joints_2d_native']).max())
        proof['samples'][i]['max_real_loader_crop_px']=float(np.max(abs(actual-expected))*ds.image_size)
    proof.update(actual_dataset_reader='wilor.datasets.glove_dataset.GloveDataset.__getitem__',actual_loader_checked=len(rows),
                 loader_max_abs_crop_px=max(x['max_real_loader_crop_px'] for x in proof['samples']))
    write_json(report,proof)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    check(a.benchmark.resolve(),lambda sample:a.source/sample.parent.parent.name/'native')
