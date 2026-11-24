"""Predict once per detector box on the exact native JPEGs indexed by a WiLoR test release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import wilor_multiview_inference as infer
from ego_data.calibration import CameraCalibration
from hand_evaluation.dataset import sha256
from wilor_physical_inference import project_physical_points


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'models/wilor'


def write_json(path, content):
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def checked_model(args):
    from wilor.models.lora import LoRALinear
    model, cfg = infer.base.load_wilor_model(args.checkpoint, args.config, args.mano, args.adapter)
    payload = torch.load(args.adapter, map_location='cpu', weights_only=False)
    state = model.state_dict()
    names = []
    for name, value in payload['state_dict'].items():
        if not torch.equal(value, state[name]) or not torch.isfinite(value).all():
            raise ValueError(f'adapter state differs after load: {name}')
        names.append(name)
    mods = payload['model_modifications']
    lora = model.backbone._wilor_lora_config.to_dict()
    assert lora == mods['lora'] and lora['blocks'] == 'all'
    expected = {f'blocks.{i}.{target}' for i in range(len(model.backbone.blocks)) for target in lora['targets']}
    installed = {k for k, m in model.backbone.named_modules() if isinstance(m, LoRALinear)}
    assert installed == expected
    for name in installed:
        m = model.backbone.get_submodule(name)
        assert m.enabled and m.lora_A.out_features == lora['rank']
        assert m.scaling == lora['alpha'] / lora['rank'] and m.dropout.p == lora['dropout']
    head = model.physical_camera_head
    assert model.physical_camera_enabled and head.camera_model == mods['physical_camera_model'] == 'physical_kb'
    assert head.native_root_prior == mods['native_root_prior']
    training = payload['training_metadata']['config']
    assert sha256(args.config) == training['model_config']['sha256']
    assert cfg.MODEL.IMAGE_SIZE == training['data']['image_size']
    paths = [args.detector, args.checkpoint, args.adapter, args.config, args.mano/'MANO_RIGHT.pkl',
             infer.base.WILOR_ROOT/'mano_data/mano_mean_params.npz']
    evidence = dict(assets={str(p.resolve()):sha256(p) for p in paths},
                    training_checkout=str(infer.base.WILOR_ROOT), format=payload['format'], recipe=payload['recipe'],
                    step=payload['step'], training_base=training['base_checkpoint'], training_manifests=training['manifests'],
                    checkpoint_modifications=mods, active_lora=lora, active_lora_modules=sorted(installed),
                    tensor_names_verified=sorted(names), camera_model=head.camera_model, native_root_prior=head.native_root_prior,
                    image_size=cfg.MODEL.IMAGE_SIZE, crop_rescale=training['data']['rescale_factor'])
    return model, cfg, evidence


def prepare_group(args, group, frames, summary):
    prior = args.sync_root/group/'normalized_multiview'
    source = read_json(prior/'manifest.json')
    cameras = source['camera_ids']
    with (prior/'multiview_frames.csv').open() as stream:
        sync = list(csv.DictReader(stream))
    by_time = {int(r['reference_timestamp_ns']):r for r in sync}
    assert len(by_time) == len(sync)
    by_camera = {c:{} for c in cameras}
    times = set()
    for row in frames:
        camera, index, stamp = row['camera'], row['sync_index'], row['reference_timestamp_ns']
        mapped = by_time[stamp]
        assert int(mapped['sync_index']) == index and int(mapped[camera+'_frame_index']) == row['source_frame_index']
        assert index not in by_camera[camera] and row['source_group'] == group
        by_camera[camera][index] = row['image']
        times.add(stamp)
    selected = [r for r in sync if int(r['reference_timestamp_ns']) in times]
    assert len(selected) == len(times)
    normalized = args.output/group/'normalized_multiview'
    (normalized/'calibration').mkdir(parents=True, exist_ok=True)
    with (normalized/'multiview_frames.csv').open('w') as stream:
        writer = csv.DictWriter(stream, selected[0])
        writer.writeheader()
        writer.writerows(selected)
    write_json(normalized/'manifest.json', dict(camera_ids=cameras, coordinate_system=source['coordinate_system'],
               image_size=source['image_size'], indexed_jpegs=str(args.dataset.resolve()), sync_source=str(prior.resolve()),
               storage='only camera images listed in frames.jsonl are observed'))
    calibration_groups = {r['calibration_group'] for r in frames}
    assert len(calibration_groups) == 1
    calibrations = {}
    for camera, data in summary['camera_calibrations_by_source'][calibration_groups.pop()].items():
        calibration = CameraCalibration.from_dict(data)
        cached = CameraCalibration.load(prior/'calibration'/f'{camera}.json')
        assert cached.model == calibration.model == 'KB' and cached.image_size == calibration.image_size
        for key in ('K', 'distortion', 'T_base_camera'):
            np.testing.assert_allclose(getattr(cached,key), getattr(calibration,key), atol=1e-8, rtol=0)
        calibration.save(normalized/'calibration'/f'{camera}.json')
        calibrations[camera] = calibration
    return selected, by_camera, calibrations


def decode(path):
    data = path.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'cannot decode {path}')
    return frame, hashlib.sha256(data).hexdigest()


def apply_kb_near_plane(hand, calibration):
    """Apply the training head's depth floor to pixels while preserving 3D placement."""
    xyz = np.asarray(hand['joints_3d_camera_relative'],np.float32) + np.asarray(hand['camera_translation'],np.float32)
    if not np.any(xyz[:,2] < .01):
        return False
    xyz[:,2] = np.maximum(xyz[:,2],.01)
    hand['joints_2d'] = project_physical_points(xyz,calibration).tolist()
    return True


def predict_camera(args, models, camera, calibration, sync, image_paths, output, scale, pool):
    model, cfg, detector, device = models
    output.mkdir(parents=True, exist_ok=True)
    path = output/'predictions.jsonl'
    confidence = .2 if camera == 'camera0' else .3
    runtime = dict(effective_hypothesis_batch_size=args.batch_size, cuda_oom_retries=0)
    with (output/'predictions.partial').open('w') as stream:
        count = images = near_plane_hands = 0
        for start in range(0, len(sync), args.frame_batch):
            chunk = sync[start:start+args.frame_batch]
            rows = [(i,image_paths[int(r['sync_index'])]) for i,r in enumerate(chunk) if int(r['sync_index']) in image_paths]
            observed = {}
            if rows:
                decoded = list(pool.map(decode, [args.dataset/name for _,name in rows]))
                frames = [pair[0] for pair in decoded]
                assert all(f.shape[:2][::-1] == calibration.image_size for f in frames)
                prepared = infer._prepare_dual_batched(frames, detector, cfg, confidence, .5, scale, 4, 0,
                                                       pool, dual_hypothesis=False)
                hands = infer._infer_prepared_batched(prepared, model, cfg, device, args.batch_size, runtime,
                                                      full_output=True, calibration=calibration)
                for (i,name), (_,digest), detections, state in zip(rows, decoded, hands, prepared['states'], strict=True):
                    assert len(detections) == len(state.get('boxes', []))
                    assert all(h['detector_is_right'] == h['is_right'] in (0,1) for h in detections)
                    near_plane_hands += sum(apply_kb_near_plane(h,calibration) for h in detections)
                    observed[i] = dict(image=name, image_sha256=digest, hands=detections)
                    count += len(detections)
                    images += 1
            for i,row in enumerate(chunk):
                packet = dict(sync_index=int(row['sync_index']), source_frame_index=int(row[camera+'_frame_index']),
                              timestamp_ns=int(row[camera+'_timestamp_ns']), **observed.get(i,dict(image=None,hands=[])))
                stream.write(json.dumps(packet,separators=(',',':'),allow_nan=False)+'\n')
            if (start // args.frame_batch) % 40 == 0:
                print(f'{output.parent.parent.name}/{camera} {start}/{len(sync)} sync rows, {images} JPEGs', flush=True)
    assert images == len(image_paths)
    (output/'predictions.partial').replace(path)
    return dict(camera=camera, images=images, detections=count, sync_rows=len(sync), detector_confidence=confidence,
                sha256=sha256(path), near_plane_hands=near_plane_hands, **runtime)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--sync-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--frame-batch', type=int, default=12)
    p.add_argument('--detector', type=Path, default=ASSETS/'2.0_glove_wilor_3680_261118.pt')
    p.add_argument('--checkpoint', type=Path, default=ASSETS/'wilor_final.ckpt')
    p.add_argument('--adapter', type=Path, default=ASSETS/'wilor_100000_best_trainable_48300_261122.pt')
    p.add_argument('--config', type=Path, default=ASSETS/'model_config.yaml')
    p.add_argument('--mano', type=Path, default=ROOT/'models/mano')
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    model_record = args.output/'model_load.json'
    if model_record.exists() or any(args.output.glob('group_*/')):
        raise FileExistsError(f'Prediction artifacts already exist in {args.output}; use a fresh output directory')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = infer.base.choose_device('cuda')
    model, cfg, evidence = checked_model(args)
    detector = infer.base.load_detector(args.detector,device)
    names = detector.names
    assert [str(names[i]).lower() for i in (0,1)] in (['left','right'], ['left_hand','right_hand'], ['left hand','right hand'])
    evidence.update(detector_classes=names, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device))
    with model_record.open('x') as stream:
        stream.write(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    models = model.to(device).eval().half(), cfg, detector, device
    summary = read_json(args.dataset/'summary.json')
    split = read_json(args.dataset/'splits.json')
    rows = read_rows(args.dataset/'frames.jsonl')
    assert len(rows) == summary['image_count'] == len({r['image'] for r in rows})
    with ThreadPoolExecutor(4) as pool:
        for info in split['recordings']:
            group = info['recording']
            frames = [r for r in rows if r['source_group'] == group]
            assert len(frames) == info['images']
            sync, by_camera, calibrations = prepare_group(args,group,frames,summary)
            assert len(sync) == info['sync_times']
            run = args.output/group/'wilor_multiview'
            summaries = {}
            for cam, paths in by_camera.items():
                summaries[cam] = predict_camera(args,models,cam,calibrations[cam],sync,paths,run/cam,
                                                evidence['crop_rescale'],pool)
                write_json(run/cam/'summary.json', summaries[cam])
            write_json(run/'summary.json', dict(cameras=summaries, frame_count=len(sync), camera_ids=list(by_camera),
                       input='exact indexed JPEG, original physical orientation', detector_iou=.5, detector_imgsz=640,
                       hypotheses_per_detection=1, precision='FP16 CUDA AMP', crop='WiLoR OpenCV anti-aliased',
                       native_2d_projection='training physical_kb with 0.01 m depth floor',
                       model_load_sha256=sha256(args.output/'model_load.json')))


if __name__ == '__main__':
    main()
