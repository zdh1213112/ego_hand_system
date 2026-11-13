#!/usr/bin/env python3
"""Export native-camera NOKOV/MANO labels using only physical camera geometry."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
from ego_data.calibration import CameraCalibration
from ego_data.dataset import SequentialVideoReader
from native_camera_geometry import (reference_to_camera, canonical_camera_rotation as canonical_rotation,
    rotate_mano_geometry, rotate_mano_pose)
from mano_conventions import WILOR_RIGHT_CANONICAL, mirror_left_points
from export_multiview_wilor_training_dataset import (
    _load_sync_rows, _load_fusion, _load_track, _sha256, _finite_error,
    _complete21_source_view, _all_points_inside, _mesh_bbox,
    _select_sync_indices, _limit_to_sync_groups)
from wilor_label_projection import project_label_points
from wilor_frame_annotations import require_both_hands, assign_image_paths, write_frame_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fusion','mano-fit','dataset','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference-camera',default='camera0')
    p.add_argument('--cameras',nargs='+',default=['camera0','camera1'])
    p.add_argument('--image-space',choices=['native_raw'],default='native_raw')
    p.add_argument('--view-filter',choices=['complete21'],default='complete21')
    p.add_argument('--require-both-hands',action='store_true')
    p.add_argument('--sample-stride',type=int,default=1)
    p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--bbox-margin',type=float,default=.10)
    p.add_argument('--min-visible-vertex-fraction',type=float,default=.50)
    p.add_argument('--max-nokov-fit-median-mm',type=float,default=10.)
    p.add_argument('--max-nokov-fit-p95-mm',type=float,default=20.)
    p.add_argument('--max-reprojection-median-px',type=float,default=25.)
    p.add_argument('--max-reprojection-p95-px',type=float,default=60.)
    p.add_argument('--jpeg-quality',type=int,default=95)
    a=p.parse_args()
    if a.sample_stride<0 or a.max_samples<0 or not 0<=a.bbox_margin<=1 or not 0<a.min_visible_vertex_fraction<=1 or not 1<=a.jpeg_quality<=100:
        p.error('invalid sampling, bounding-box or JPEG parameters')
    dataset,fit,fusion,output=[x.resolve() for x in (a.dataset,a.mano_fit,a.fusion,a.output)]
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest.get('dataset_type')!='normalized_multiview':raise ValueError('require normalized multiview data')
    camera_ids=list(dict.fromkeys(a.cameras))
    if not set(camera_ids+[a.reference_camera]).issubset(manifest['camera_ids']):raise ValueError('unknown cameras')
    cal={c:CameraCalibration.load(dataset/'calibration'/f'{c}.json') for c in set(camera_ids+[a.reference_camera])}
    size=tuple(manifest['image_size'])
    if any(c.model!='KB' or c.image_size!=size for c in cal.values()):raise ValueError('require matching-size KB cameras')
    fit_summary=json.loads((fit/'summary.json').read_text())
    fusion_summary=json.loads((fusion/'summary.json').read_text())
    multiview_fit = fit_summary.get('fit_target') == 'multiview_surface_markers'
    personal_fit = fit_summary.get('fit_target') == 'personal_marker24'
    marker_fit = multiview_fit or personal_fit
    if personal_fit and (not fit_summary.get('personal_marker24_optimization_enabled') or fit_summary.get('personal_marker24_observation_source') != 'raw_nokov24+automatic_native_contours' or fit_summary.get('approved_for_training') is not False):
        raise ValueError('Missing explicit experimental personal marker24 provenance')
    if multiview_fit and (not fit_summary.get('multiview_optimization_enabled') or fit_summary.get('multiview_observation_source') != 'independent_rgb_marker_detections'):
        raise ValueError('Missing provenance for multiview MANO fit')
    if fit_summary.get('mano_convention')!=WILOR_RIGHT_CANONICAL or fit_summary.get('fit_target','mano_joints') not in ('mano_joints','multiview_surface_markers','personal_marker24'):
        raise ValueError('require right-canonical MANO joint fit')
    if fusion_summary.get('stage')!='nokov21_aligned_reference':raise ValueError('require physical NOKOV reference observations')
    fitted_calibrations=fit_summary.get('camera_calibrations')
    if fitted_calibrations:
        for name, fitted in fitted_calibrations.items():
            actual=CameraCalibration.load(dataset/'calibration'/f'{name}.json').to_dict()
            if fitted!=actual:raise ValueError('camera calibration changed since MANO fitting')
    reference=fit_summary.get('reference_camera')
    if reference is not None and reference!=a.reference_camera:raise ValueError('fit reference camera mismatch')
    model=Path(fit_summary['model_dir'])/'MANO_RIGHT.pkl'
    if not model.is_file():raise FileNotFoundError(model)
    expected={'left':'MANO_RIGHT.pkl','right':'MANO_RIGHT.pkl'}
    if fit_summary.get('mano_model_by_side')!=expected:raise ValueError('incompatible MANO model mapping')
    rows=_load_fusion(fusion/'accepted.jsonl');sync=_load_sync_rows(dataset/'multiview_frames.csv')
    sides={int(h['side']) for r in rows for h in r['hands']}
    tracks={s:_load_track(fit/f'track_{s}.npz') for s in sides}
    lookup={s:{int(pair):i for i,pair in enumerate(t['pair_indices'])} for s,t in tracks.items()}
    for t in tracks.values():
        if str(t['mano_convention'])!=WILOR_RIGHT_CANONICAL:raise ValueError('track convention mismatch')
    pending=[];rejected=[]
    for row in rows:
        frame=int(row['sync_index'])
        if frame not in sync:raise ValueError('frame missing from synchronization table')
        for hand in row['hands']:
            side=int(hand['side']);track=tracks[side];i=lookup[side].get(frame)
            base=dict(sync_index=frame,side=side)
            def reject(reason,**kwargs):rejected.append(dict(base,reason=reason,**kwargs))
            if i is None or not bool(track['render_valid'][i]):
                reject('missing_or_invalid_mano_fit');continue
            errors=np.asarray(track['observation_error_m' if marker_fit else 'joint_error_m'][i],float)*1000
            if errors.shape!=(21,) or not np.isfinite(errors).all():
                reject('incomplete_nokov_joint_fit');continue
            median,p95=float(np.median(errors)),float(np.percentile(errors,95))
            if median>a.max_nokov_fit_median_mm or p95>a.max_nokov_fit_p95_mm:
                reject('mano_nokov_3d_fit_error',median_mm=median,p95_mm=p95);continue
            trans_ref=track['translation'][i].astype(np.float32)
            local_v=track['vertices'][i].astype(np.float32)-trans_ref
            local_j=track['joints'][i].astype(np.float32)-trans_ref
            motion=mirror_left_points(local_j+trans_ref,side)
            for camera in camera_ids:
                view=hand.get('views',{}).get(camera)
                if view is None or not _complete21_source_view(view,size):
                    reject('camera_view_not_complete21',camera=camera);continue
                R,t=reference_to_camera(cal[a.reference_camera],cal[camera])
                Q=canonical_rotation(R,side)
                v,j,translation,root_offset=rotate_mano_geometry(local_v,local_j,trans_ref,Q)
                translation+=mirror_left_points(t,side)
                v,j,translation=[mirror_left_points(x,side).astype(np.float32) for x in (v,j,translation)]
                global_pose,hand_pose=rotate_mano_pose(track['full_pose_axis_angle'][i],Q)
                info=dict(K=cal[camera].K.astype(np.float32),image_space='native_raw',
                    camera_model='KB',distortion=cal[camera].distortion.astype(np.float64))
                try:
                    px=project_label_points(v+translation,info).astype(np.float32)
                    joint_px=project_label_points(j+translation,info).astype(np.float32)
                except ValueError:
                    reject('native_projection_failed',camera=camera);continue
                target=np.asarray(view['joints_2d'],float)
                quality_px = joint_px
                if marker_fit:
                    marker_ref = mirror_left_points(track['observation_points'][i], side)
                    try:
                        quality_px = project_label_points(marker_ref @ R.T + t, info)
                    except ValueError:
                        reject('native_marker_projection_failed',camera=camera);continue
                pixel_error=np.linalg.norm(quality_px-target,axis=-1)
                med,high=_finite_error(pixel_error)
                if med>a.max_reprojection_median_px or high>a.max_reprojection_p95_px:
                    reject('native_mano_reprojection_error',camera=camera,median_px=med,p95_px=high);continue
                if not _all_points_inside(joint_px,size):
                    reject('native_mano_joints_not_complete21',camera=camera);continue
                bbox=_mesh_bbox(px,size,a.bbox_margin,a.min_visible_vertex_fraction)
                if bbox is None:
                    reject('mesh_not_sufficiently_visible',camera=camera);continue
                label=dict(bbox=bbox,vertices=v,joints_3d=j,joints_2d=px,
                    side=np.asarray(side,np.float32),trans=translation,K=info['K'],
                    mano=dict(global_orient=global_pose,hand_pose=hand_pose,betas=track['betas'].astype(np.float32)))
                label.update({k:info[k] for k in ('image_space','camera_model','distortion')})
                metadata=dict(base,camera=camera,physical_side=side,handedness='right' if side else 'left',
                    stored_image_space='native_raw',stored_image_horizontally_flipped=False,
                    mano_parameter_space='right_canonical',view_filter='complete21',
                    camera_inlier_joint_count=21,fit_reprojection_space='native_raw',
                    fit_reprojection_median_px=med,fit_reprojection_p95_px=high,
                    fit_nokov_joint_median_mm=median,fit_nokov_joint_p95_mm=p95,
                    source_group=fusion_summary['source_group'],supervision_source='nokov21_mano_fit',
                    reference_camera=a.reference_camera,source_frame_index=int(sync[frame][f'{camera}_frame_index']),
                    reference_timestamp_ns=int(sync[frame]['reference_timestamp_ns']))
                if marker_fit:
                    joint_errors = np.asarray(track['joint_error_m'][i], float) * 1000
                    metadata['fit_nokov_joint_median_mm'] = float(np.nanmedian(joint_errors))
                    metadata['fit_nokov_joint_p95_mm'] = float(np.nanpercentile(joint_errors, 95))
                    metadata['fit_nokov_marker_median_mm'] = median
                    metadata['fit_nokov_marker_p95_mm'] = p95
                metadata['fit_error_point_semantics'] = 'surface_marker_centres' if marker_fit else 'mano_joints_vs_marker_centres'
                if personal_fit:
                    metadata.update(joint_pose_attachment_optimization_enabled=bool(fit_summary.get('joint_pose_attachment_optimization_enabled',False)),joint_pose_attachment_frame_optimized=bool(track['joint_fit_mask'][i]) if 'joint_fit_mask' in track else False,personal_marker24_optimization_enabled=True, approved_for_training=False, supervision_source='nokov24_surface_and_native_contour_fit')
                metadata['multiview_mano_optimization_enabled'] = bool(fit_summary.get('multiview_optimization_enabled', False))
                metadata['multiview_strategy'] = fit_summary.get('multiview_strategy')
                if fit_summary.get('multiview_strategy') == 'baseline_anchored_joint_v2':
                    metadata['refinement_safety_source'] = str(fit/'summary.json')
                pending.append(dict(base,camera=camera,label=label,motion_points=motion,metadata=metadata))
    if a.require_both_hands:
        pending,missing=require_both_hands(pending);rejected.extend(missing)
    pending.sort(key=lambda x:(x['sync_index'],x['side'],camera_ids.index(x['camera'])))
    selected,sampling=_select_sync_indices(pending,sync,a.sample_stride)
    pending=[x for x in pending if x['sync_index'] in set(selected)]
    pending=_limit_to_sync_groups(pending,selected,a.max_samples)
    if not pending:raise RuntimeError('no complete native hand samples passed quality checks')
    assign_image_paths(pending,True)
    for i,item in enumerate(pending):
        item['metadata'].update(index=i,image=f"images/{item['image_stem']}.jpg",label=f'labels/{i:06d}.npy',
            sampling_mode=sampling['mode'],sample_stride=a.sample_stride)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+output.name+'.tmp-',dir=output.parent))
    try:
        import torch
        (temporary/'images').mkdir();(temporary/'labels').mkdir()
        for item in pending:
            label=dict(item['label']);label['joints_2d']=torch.from_numpy(label['joints_2d'])
            np.save(temporary/item['metadata']['label'],label,allow_pickle=True)
        for camera in camera_ids:
            chosen={int(x['sync_index']):x for x in pending if x['camera']==camera}
            if not chosen:continue
            reader=SequentialVideoReader(dataset/'cameras'/camera/manifest['storage']['video_filename'],size)
            try:
                for frame,item in sorted(chosen.items()):
                    image=reader.read(item['metadata']['source_frame_index'])
                    if not cv2.imwrite(str(temporary/item['metadata']['image']),image,[cv2.IMWRITE_JPEG_QUALITY,a.jpeg_quality]):
                        raise RuntimeError('cannot save native frame')
            finally:reader.close()
        metadata=[x['metadata'] for x in pending]
        write_frame_index(temporary/'frames.jsonl',metadata)
        for filename,records in [('index.jsonl',metadata),('rejected.jsonl',rejected)]:
            (temporary/filename).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
        sampling.update(exported_sync_count=len({x['sync_index'] for x in pending}),exported_sample_count=len(pending))
        summary=dict(schema_version=6,stage='native_wilor_training_dataset',sample_count=len(pending),
            image_count=len({x['image_stem'] for x in pending}),label_count=len(pending),
            image_layout='shared_camera_frames',annotation_policy='require_both_hands' if a.require_both_hands else 'available_hands',
            side_counts={name:sum(x['side']==side for x in pending) for side,name in enumerate(('left','right'))},
            camera_counts={c:sum(x['camera']==c for x in pending) for c in camera_ids},cameras=camera_ids,
            image_size=list(size),K=None,image_space='native_raw',geometry_space='physical_side',
            mano_parameter_space='right_canonical',side_semantics='physical_identity',
            mano_model_by_side=expected,mano_pose_representation='rotation_matrix_full_mean',mano_model='MANO_RIGHT.pkl',
            mano_source_revision=fit_summary.get('mano_revision','unknown'),
            mano_assets={'MANO_RIGHT.pkl':dict(path=str(model),sha256=_sha256(model),size_bytes=model.stat().st_size)},
            camera_calibrations={c:{**cal[c].to_dict(),'sha256':_sha256(dataset/'calibration'/f'{c}.json')} for c in camera_ids},
            source_dataset=str(dataset),source_fit=str(fit),source_group=fusion_summary['source_group'],
            source='physical NOKOV reference + MANO fit + native KB projection',reference_camera=a.reference_camera,
            multiview_mano_optimization_enabled=bool(fit_summary.get('multiview_optimization_enabled',False)),multiview_strategy=fit_summary.get('multiview_strategy'),refinement_safety_by_track={str(x['track_id']):x.get('refinement_safety',{}) for x in fit_summary.get('tracks',[])},fit_error_point_semantics='surface_marker_centres' if marker_fit else 'mano_joints_vs_marker_centres',
            projection_space='native_raw',uses_rectification=False,sampling=sampling,max_samples=a.max_samples,
            rejected_count=len(rejected),view_filter='complete21',quality_thresholds=dict(
                max_nokov_fit_median_mm=a.max_nokov_fit_median_mm,max_nokov_fit_p95_mm=a.max_nokov_fit_p95_mm,
                max_native_reprojection_median_px=a.max_reprojection_median_px,max_native_reprojection_p95_px=a.max_reprojection_p95_px,
                min_visible_vertex_fraction=a.min_visible_vertex_fraction))
        if personal_fit:
            summary.update(joint_pose_attachment_optimization_enabled=bool(fit_summary.get('joint_pose_attachment_optimization_enabled',False)),personal_marker24_optimization_enabled=True, approved_for_training=False, source='raw NOKOV24 + MANO surface attachment fit + native contour refinement', fit_error_point_semantics='surface_marker_centres')
        (temporary/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True);raise
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
