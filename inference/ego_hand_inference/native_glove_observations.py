"""Predict native-camera joints and MANO parameters using a trained physical-KB WiLoR adapter."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from ego_data.dataset import SequentialVideoReader
from ego_hand_inference.detector_crop import calibrated_box, load_crop_calibration
from ego_hand_inference.inference_common import choose_device, load_detector, _top_detections_per_class

TRAINING_ROOT = Path(__file__).resolve().parents[2] / 'training'
sys.path.insert(0, str(TRAINING_ROOT))
from wilor.datasets.utils import crop_points_to_image, expand_to_aspect_ratio, generate_image_patch_cv2
from wilor.models import load_wilor
from wilor.utils import recursive_to


def crop_hand(image, box, side, calibration, config):
    center = (box[:2] + box[2:]) / 2
    size = float(expand_to_aspect_ratio(2*(box[2:]-box[:2]), target_aspect_ratio=[192,256]).max())
    res = config.MODEL.IMAGE_SIZE
    crop, transform = generate_image_patch_cv2(image, float(center[0]), float(center[1]), size, size,
        res, res, side==0, 1., 0, border_mode=cv2.BORDER_CONSTANT)
    rgb = crop[:,:,::-1].astype(np.float32)/255.
    rgb = (rgb-np.asarray(config.MODEL.IMAGE_MEAN,np.float32))/np.asarray(config.MODEL.IMAGE_STD,np.float32)
    return dict(img=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))),crop_transform=transform.astype(np.float32),
                crop_box_size=np.float32(size),camera_K=calibration['K'],camera_distortion=calibration['D'],
                bbox_xyxy=box,image_size=calibration['size'],side=side)


def run_frames(images, detector, model, config, calibration, threshold, args, device, pool):
    result = detector.predict(images, conf=threshold, iou=.5, verbose=False, device=str(device))
    proposals=[]
    for frame,detections in enumerate(result):
        boxes=detections.boxes
        scores=boxes.conf.cpu().numpy();sides=boxes.cls.cpu().numpy().astype(int);bounds=boxes.xyxy.cpu().numpy()
        for idx in _top_detections_per_class(scores,sides,args.max_detections_per_class):
            box=bounds[idx]
            if np.min(box[2:]-box[:2])<=1:continue
            crop_box = box
            if args.crop_coefficients is not None:
                crop_box = calibrated_box(args.crop_coefficients, box, calibration["id"], int(sides[idx]),
                                          calibration["size"])
            for side in (0,1):
                info=dict(detection_index=int(idx),hypothesis_index=int(2*idx+side),bbox_xyxy=box.tolist(),
                          confidence=float(scores[idx]),detector_is_right=int(sides[idx]),is_right=side,
                          proposal_source='detector',camera_model='physical_kb')
                if args.crop_coefficients is not None:
                    info["crop_bbox_xyxy"] = crop_box.tolist()
                proposals.append((frame,crop_box,info))
    hands=[[] for _ in images]
    def prepare(record):
        frame,box,info=record
        return crop_hand(images[frame],box,info['is_right'],calibration,config)
    for start in range(0,len(proposals),args.batch_size):
        rows=proposals[start:start+args.batch_size]
        samples=list(pool.map(prepare,rows))
        batch=recursive_to(default_collate(samples),device)
        amp=device.type=='cuda' and torch.cuda.is_bf16_supported()
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            output=model.forward_step(batch,train=False)
        cpu=lambda tensor:tensor.detach().float().cpu().numpy()
        joints=cpu(output['pred_keypoints_3d']);roots=cpu(output['pred_root_camera_m'])
        pixels=cpu(output['pred_keypoints_2d'])
        mano={k:cpu(v) for k,v in output['pred_mano_params'].items()}
        for i,(frame,_,info) in enumerate(rows):
            left=info['is_right']==0
            relative=joints[i].copy()
            if left:relative[:,0]*=-1
            uv=crop_points_to_image(pixels[i],samples[i]['crop_transform'],calibration['size'],left)
            trans=roots[i]-relative[0]
            if not np.isfinite(uv).all() or not np.isfinite(relative).all() or not np.isfinite(trans).all():
                raise ValueError('Non-finite native camera prediction')
            info.update(joints_2d=uv.tolist(),joints_3d_camera_relative=relative.tolist(),camera_translation=trans.tolist(),
                        physical_root_camera_m=roots[i].tolist(),mano={k:v[i].tolist() for k,v in mano.items()})
            hands[frame].append(info)
    return hands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','output','checkpoint','adapter','model-config','mano-model-dir','detector'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--frame-batch-size',type=int,default=8)
    parser.add_argument('--preprocess-workers',type=int,default=4)
    parser.add_argument('--max-detections-per-class',type=int,default=0)
    parser.add_argument('--compile-backbone',type=int,choices=(0,1),default=0)
    parser.add_argument('--gpu-profile',choices=('compatible','rtx5090d'),default='compatible')
    parser.add_argument('--camera-confidence',action='append',default=[])
    parser.add_argument('--detector-crop-calibration', type=Path)
    parser.add_argument('--save-mano-params',action='store_true')
    args=parser.parse_args()
    args.crop_coefficients = (load_crop_calibration(args.detector_crop_calibration, args.detector)
                              if args.detector_crop_calibration else None)
    device=choose_device(args.device)
    model,config=load_wilor(str(args.checkpoint),str(args.model_config),mano_path=args.mano_model_dir,
                            adapter_path=args.adapter,init_renderer=False)
    if not model.physical_camera_enabled or model.physical_camera_head.camera_model!='physical_kb':
        raise ValueError('The adapter must include the trained physical-KB camera head')
    model.to(device).eval()
    if args.compile_backbone:model.backbone=torch.compile(model.backbone)
    detector=load_detector(args.detector,device)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    with (args.dataset/'multiview_frames.csv').open() as f:rows=list(csv.DictReader(f))[:args.max_frames or None]
    limits={k:float(v) for k,v in (s.split('=') for s in args.camera_confidence)}
    args.output.mkdir(parents=True,exist_ok=False)
    camera_stats={}
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.preprocess_workers) as pool:
        for cam in manifest['camera_ids']:
            spec=json.loads((args.dataset/'calibration'/(cam+'.json')).read_text())
            if spec['model']!='KB':raise ValueError(f'Native physical-KB adapter requires KB calibration on {cam}')
            cal=dict(id=cam,K=np.asarray(spec['K'],np.float32),D=np.asarray(spec['distortion'],np.float32),
                     size=np.asarray(spec['image_size'],np.float32))
            folder=args.output/cam;folder.mkdir()
            reader=SequentialVideoReader(args.dataset/'cameras'/cam/manifest['storage']['video_filename'],tuple(spec['image_size']))
            counts=Counter()
            try:
                with (folder/'predictions.jsonl').open('w') as f:
                    for start in range(0,len(rows),args.frame_batch_size):
                        group=rows[start:start+args.frame_batch_size]
                        images=[reader.read(int(r[cam+'_frame_index'])) for r in group]
                        hands=run_frames(images,detector,model,config,cal,limits.get(cam,.3),args,device,pool)
                        for r,hypotheses in zip(group,hands):
                            rec=dict(sync_index=int(r['sync_index']),source_frame_index=int(r[cam+'_frame_index']),
                                     timestamp_ns=int(r[cam+'_timestamp_ns']),hands=hypotheses)
                            f.write(json.dumps(rec,separators=(',',':'))+'\n')
                            counts.update(frame_count=1,hypotheses=len(hypotheses),physical_box_count=len(hypotheses)//2)
            finally:
                reader.close()
            camera_stats[cam]=dict(counts)
            (folder/'summary.json').write_text(json.dumps(camera_stats[cam],indent=2)+'\n')
    summary=dict(dataset=str(args.dataset.resolve()),frame_count=len(rows),camera_ids=manifest['camera_ids'],
                 hypotheses_per_detection=2,save_mano_params=True,adapter=str(args.adapter.resolve()),
                 base_checkpoint=str(args.checkpoint.resolve()),camera_model='physical_kb',cameras=camera_stats)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
