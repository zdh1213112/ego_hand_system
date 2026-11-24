"""Run a fine-tuned FAH-S on synchronized video or indexed JPEGs with recorded hand boxes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256 as hash_bytes
from itertools import islice
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ego_hand_inference.ego_data.calibration import CameraCalibration
from ego_data.dataset import SequentialVideoReader
from fah_multiview_inference import fit_meshes, training_root
from hand_evaluation.dataset import sha256


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def crops(row, image, camera, size, proposal_source='detector'):
    from tools.hand_finetune.crop_geometry import PerspectiveCrop, rgb_chw
    if image is not None and image.shape[1::-1] != camera.image_size:
        raise ValueError('Image size differs from the KB calibration')
    prepared = []
    for box in row['boxes']:
        if box['detector_is_right'] not in (0, 1) or box['source'] != proposal_source:
            raise ValueError(f'Expected hand boxes and sides from {proposal_source}')
        view = PerspectiveCrop.from_box(box['bbox_xyxy'], camera.K, camera.distortion, size)
        rgb = rgb_chw(view.rectify(image), mirror=not box['detector_is_right'])
        prepared.append((view, rgb, box))
    return prepared


@torch.no_grad()
def forward(method, items):
    from tools.hand_finetune.fah_views import native_prediction, view_batch
    est = method.est
    size = est.input_size[0]
    views, rgb, boxes = zip(*items)
    batch = view_batch(rgb, views, est.device)
    est.model.keypoint_head.parts_on = False
    est.model.decoder_hand.active_depth = est.decoder_depth
    est.model.head_pose_hand.use_fast_hand = est.fast_mhr
    out = {k: v.detach().float().cpu().numpy() for k,v in est._forward(batch).items() if torch.is_tensor(v)}
    for key in ('affine_trans', 'bbox_center', 'bbox_scale'):
        out[key] = batch[key][:, 0].cpu().numpy()
    full = np.asarray([[0, 0, size, size]], dtype=np.float32)
    hands = []
    for i,(view,box) in enumerate(zip(views, boxes)):
        arrays = {k:v[i:i+1] for k,v in out.items()}
        hands.extend(est._results(arrays, full, full, [bool(box['detector_is_right'])],
                                  [box['confidence']], view.K.astype(np.float32), size, size))
    params = fit_meshes(method, hands)
    return [native_prediction(h,v,p) for h,v,p in zip(hands,views,params,strict=True)]


def infer_camera(model, camera, inputs, output, dataset, meta, pool, batch_size, model_hash):
    target = output / camera
    target.mkdir(parents=True, exist_ok=True)
    cache = inputs / (camera + '.jsonl')
    if sha256(cache) != meta['cameras'][camera]['sha256']:
        raise ValueError(f'Detector proposal fingerprint differs for {camera}')
    normalized = inputs/'normalized_multiview'
    calib = CameraCalibration.load(normalized/'calibration'/f'{camera}.json')
    manifest = json.loads((normalized/'manifest.json').read_text())
    archive = target/'meshes.npz'
    predictions = target/'predictions.jsonl'
    marker = target/'summary.json'
    if marker.is_file():
        info = json.loads(marker.read_text())
        if (info['model_load_sha256']==model_hash and info['proposal_sha256']==sha256(cache)
                and sha256(predictions)==info['sha256'] and sha256(archive)==info['mesh_sha256']):
            return info
        raise ValueError(f'Stale camera output {marker}')
    reader = None
    if meta['raw']:
        reader = SequentialVideoReader(normalized/'cameras'/camera/manifest['storage']['video_filename'],calib.image_size)

    def decode(packet):
        if packet.get('image') is None:
            return None
        data = (dataset/packet['image']).read_bytes()
        if hash_bytes(data).hexdigest()!=packet['image_sha256']:
            raise ValueError(f"Indexed image fingerprint differs for {packet['image']}")
        frame = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unreadable image {packet['image']}")
        return frame

    meshes, indices, hypotheses, sides = [],[],[],[]
    images = frames = 0
    try:
        with cache.open() as stream, predictions.with_suffix('.partial').open('w') as dst:
            while chunk := list(islice(stream, 12)):
                packets = list(map(json.loads,chunk))
                pictures = ([reader.read(p['source_frame_index']) for p in packets] if reader
                            else list(pool.map(decode,packets)))
                images += sum(x is not None for x in pictures)
                prepared = list(pool.map(lambda args: crops(*args, calib, model.method.est.input_size[0],
                                                            meta.get('proposal_source','detector')),
                                         zip(packets,pictures)))
                items = [sample for frame in prepared for sample in frame]
                results = iter([r for i in range(0,len(items),batch_size)
                                 for r in forward(model.method, items[i:i+batch_size])])
                for packet in packets:
                    hands=[]
                    for i,box in enumerate(packet.pop('boxes')):
                        pred, param = next(results)
                        side = int(box['detector_is_right'])
                        xyz, xy, verts = pred['keypoints_3d'], pred['keypoints_2d'], pred['vertices_mano']
                        if (pred['is_right'] != side or xyz.shape!=(21,3) or xy.shape!=(21,2)
                                or verts.shape!=(778,3) or not np.isfinite(verts).all()):
                            raise ValueError('FAH hand geometry or side mismatch')
                        hypothesis = 2*i+side
                        hands.append(dict(detection_index=i,hypothesis_index=hypothesis,is_right=side,
                            detector_is_right=side, bbox_xyxy=box['bbox_xyxy'], confidence=box['confidence'],
                            proposal_source=box['source'],pose_backend='fah',camera_translation=xyz[0].tolist(),
                            joints_3d_camera_relative=(xyz-xyz[0]).tolist(), joints_2d=xy.tolist(),mano=param))
                        meshes.append(verts)
                        indices.append(packet['source_frame_index'])
                        hypotheses.append(hypothesis)
                        sides.append(side)
                    dst.write(json.dumps(dict(**packet,camera_id=camera,hands=hands),separators=(',',':'),allow_nan=False)+'\n')
                frames += len(packets)
                if frames%600==0:
                    print(f'{output.parent.name}/{camera} {frames}/{meta["sync_count"]}',flush=True)
    finally:
        if reader: reader.close()
    if (frames,images,len(meshes)) != (meta['sync_count'],meta['cameras'][camera]['images'],meta['cameras'][camera]['boxes']):
        raise ValueError('Frame, image or proposal counts changed in prediction')
    np.savez(archive.with_suffix('.partial.npz'), vertices_camera_m=np.asarray(meshes,np.float32).reshape(-1,778,3),
        source_frame_indices=np.asarray(indices,np.int64),hypothesis_indices=np.asarray(hypotheses,np.int32),
        is_right=np.asarray(sides,np.int8),faces_right=np.asarray(model.faces,np.int32),
        K=calib.K,kb_calibration_json=json.dumps(calib.to_dict()))
    archive.with_suffix('.partial.npz').replace(archive)
    predictions.with_suffix('.partial').replace(predictions)
    info=dict(camera=camera,images=images,sync_rows=frames,detections=len(meshes),sha256=sha256(predictions),
              mesh_sha256=sha256(archive),proposal_sha256=sha256(cache),model_load_sha256=model_hash)
    save(marker,info)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','output','checkpoint'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--fah-root',type=Path,default=Path(__file__).resolve().parents[2]/'FAH')
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    root=args.fah_root.resolve()
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(training_root()))
    import fah
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    inputs,output=args.inputs.resolve(),args.output.resolve()
    output.mkdir(exist_ok=True,parents=True)
    sources=json.loads((inputs/'sources.json').read_text())
    proposal_source = sources.get('proposal_source','detector')
    if proposal_source == 'detector':
        if sha256(sources['detector']) != sources['detector_sha256']:
            raise ValueError('Detector fingerprint changed')
        source_asset = dict(detector=sources['detector'], detector_sha256=sources['detector_sha256'])
        if 'groups' in sources:
            directories = [inputs/group for group in sources['groups']]
        else:
            directories = [inputs/'recording', *sorted(inputs.glob('group_*'))]
    elif proposal_source == 'gt_box_physical_side':
        if sha256(Path(sources['dataset'])/'test.jsonl') != sources['dataset_manifest_sha256']:
            raise ValueError('GT dataset manifest changed')
        source_asset = dict(dataset_manifest_sha256=sources['dataset_manifest_sha256'])
        directories = [inputs/group for group in sources['groups']]
    else:
        raise ValueError(f'Unknown proposal source {proposal_source}')
    checkpoint=args.checkpoint.resolve()
    with patch('bench.methods.fah.build_hand_detector',return_value=SimpleNamespace(names={0:'left',1:'right'})):
        model=fah.FAH('fah-s',weights=str(root/'weights'),finetune_ckpt=str(checkpoint),device='cuda',
            mano_dir=str(root/'data/mano'),sam3d=str(root/'data/sam-3d-body-dinov3'),mode='image',mesh='mano',
            tips=False,parts=False,verify=False,fast_det=False,compile=False,parts_in_mask=False)
    info=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),**source_asset,
              proposal_source=proposal_source,sources_sha256=sha256(inputs/'sources.json'),
              runner_sha256=sha256(__file__),batch_size=args.batch_size,geometry='side-conditioned KB local perspective',
              native_mesh='FAH fitted and pixel-aligned vertices in physical camera frame',
              device=torch.cuda.get_device_name(),torch=str(torch.__version__),
              student=str(model.method.est.student.name),student_dtype=str(model.method.est.student.mean.dtype))
    path = output/'model_load.json'
    if path.exists() and json.loads(path.read_text())!=info:
        raise ValueError(f'Output configuration mismatch: {path}')
    save(path,info)
    with ThreadPoolExecutor(4) as pool:
        for directory in directories:
            meta=json.loads((directory/'summary.json').read_text())
            if (meta.get('proposal_source','detector') != proposal_source
                    or any(meta.get(key) != value for key,value in source_asset.items())):
                raise ValueError(f'Proposal source mismatch in {directory}')
            run=output/directory.name
            run.mkdir(exist_ok=True)
            link=run/'normalized_multiview'
            if not link.exists():
                link.symlink_to((directory/'normalized_multiview').resolve())
            native=run/'fah_multiview'
            summaries={}
            for camera in meta['cameras']:
                summaries[camera]=infer_camera(model,camera,directory,native,Path(sources['dataset']),
                    meta,pool,args.batch_size,sha256(path))
            save(native/'summary.json',dict(cameras=summaries,frame_count=meta['sync_count'],camera_ids=list(summaries),
                 model_load_sha256=sha256(path),save_mano_params=True,dataset=str(link.resolve()),
                 hypotheses_per_detection=1,pose_backend='fah',model=info))

if __name__=='__main__':
    main()
