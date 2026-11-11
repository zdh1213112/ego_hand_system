#!/usr/bin/env python3
"""Render reviewable raw/before/after marker crops and scoped MANO residuals."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import cv2
import numpy as np
from calibrate_nokov_image_markers import Recording, evaluate, statistics, sha256


def marker_label(name):
    if name.startswith('Finger'):
        return name[6] + name[-1]
    return {'WristM':'WM','WristIn':'WI','WristOut':'WO','HandOffset':'HO'}[name]


def image_alignment_gate(errors, role):
    """Gate one camera/frame only; development points cannot certify a test."""
    errors=np.asarray(errors)
    if role=='fit':return 'fit_only'
    if role=='development':return 'development_only'
    if role not in ('validation','final_test'):return 'unknown_role'
    if not np.isfinite(errors).all():return 'invalid_observations'
    if len(errors)<6:return 'insufficient_observations'
    return 'passed_sparse_scope' if np.median(errors)<=3 and np.percentile(errors,95)<=5 else 'failed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','mano','observations','config','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validation-output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    recording=Recording(args.baseline,args.mano)
    annotation=json.loads(args.observations.read_text());obs=annotation['observations']
    config=json.loads(args.config.read_text());delta=config.get('image_alignment',{}).get('time_offset_change_ms',0.)/1000
    before,eb=evaluate(recording,obs,recording.config)
    after,ea=evaluate(recording,obs,config,delta)
    fitted,ef=evaluate(recording,obs,config,delta,'predicted_markers')
    byframe={}
    for c,f in sorted({(o['camera'],o['video_frame']) for o in obs}):
        mask=np.array([o['camera']==c and o['video_frame']==f for o in obs]);e=ea[mask]
        role=next(o['split'] for o in obs if o['camera']==c and o['video_frame']==f)
        status=image_alignment_gate(e,role)
        byframe[f'{c}:{f}']=dict(camera=c,video_frame=f,role=role,measured_reprojection_px=statistics(e),
                               status=status,sparse_gate_pass=status=='passed_sparse_scope')
    for c,f in sorted({(o['camera'],o['video_frame']) for o in obs}):
        idx=np.array([i for i,o in enumerate(obs) if o['camera']==c and o['video_frame']==f]);oo=[obs[i] for i in idx]
        raw=cv2.imread(str(args.observations.parent/oo[0]['source']))
        points=np.concatenate([np.array([o['pixel'] for o in oo]),before[idx],after[idx]])
        lo=np.maximum(np.floor(points.min(0)-55).astype(int),[0,72]);hi=np.minimum(np.ceil(points.max(0)+55).astype(int),[1920,1200])
        x,y=lo;u,v=hi;panels=[]
        for label,pred,colour in [('RAW / annotated spheres',None,(255,255,255)),('BEFORE / measured NOKOV',before[idx],(80,80,255)),('AFTER / fixed rig correction',after[idx],(80,255,80))]:
            image=raw[y:v,x:u].copy();image=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
            for j,o in enumerate(oo):
                q=tuple(np.rint((np.array(o['pixel'])-lo)*2).astype(int));cv2.drawMarker(image,q,(255,255,255),cv2.MARKER_CROSS,10,1)
                if pred is not None:
                    p=tuple(np.rint((pred[j]-lo)*2).astype(int));cv2.circle(image,p,7,colour,1);cv2.line(image,q,p,colour,1)
                cv2.putText(image,marker_label(o['marker_name']),(q[0]+7,q[1]+14),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,220,255),1,cv2.LINE_AA)
            header=np.zeros((54,image.shape[1],3),np.uint8)
            cv2.putText(header,label,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.46,colour,1,cv2.LINE_AA)
            detail=f'cam{c} frame{f} | {oo[0]["split"]} | n={len(oo)}'
            if pred is not None:detail+=f' | median={np.median(eb[idx] if label.startswith("BEFORE") else ea[idx]):.2f}px'
            cv2.putText(header,detail,(8,43),cv2.FONT_HERSHEY_SIMPLEX,.40,(220,220,220),1,cv2.LINE_AA)
            panels.append(np.vstack([header,image]))
        cv2.imwrite(str(args.output/f'marker_comparison_c{c}_f{f:04d}.png'),np.hstack(panels))
    # Only image-validated sparse frames may support image-domain MANO assessment.
    scope={(v['camera'],v['video_frame']) for v in byframe.values() if v['sparse_gate_pass']}
    rows=[]
    for i,o in enumerate(obs):
        local_measured=recording.local_points([o],delta)[0]
        local_fitted=recording.local_points([o],delta,'predicted_markers')[0]
        rows.append(dict(camera=o['camera'],video_frame=o['video_frame'],marker_name=o['marker_name'],
                         measured_image_error_px=float(ea[i]),attachment_image_error_px=float(ef[i]),
                         attachment_measured_projection_delta_px=float(np.linalg.norm(fitted[i]-after[i])),
                         attachment_measured_3d_mm=float(np.linalg.norm(local_fitted-local_measured)*1000),
                         role=o['split'],image_alignment_gate_pass=(o['camera'],o['video_frame']) in scope))
    with (args.output/'mano_local_observations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    def group_summary(group):
        return {key:statistics([r[key] for r in group]) for key in ['measured_image_error_px','attachment_image_error_px','attachment_measured_projection_delta_px','attachment_measured_3d_mm']}
    admitted=[r for r in rows if r['image_alignment_gate_pass']]
    summary=dict(status='blocked' if not scope else 'limited_sparse_scope',
                 gate='Each camera/frame separately: validation or final_test role, >=6 labeled observations, measured median <=3px and P95 <=5px; never pool cameras to pass',
                 by_camera_frame=byframe,admitted_camera_frames=[list(v) for v in sorted(scope)],admitted=group_summary(admitted),
                 role_limitation=annotation.get('role_limitation',''),
                 by_marker={name:group_summary([r for r in admitted if r['marker_name']==name]) for name in sorted({r['marker_name'] for r in admitted})},
                 by_finger={finger:group_summary([r for r in admitted if finger in r['marker_name']]) for finger in ['Thumb','Index','Middle','Ring','Pinky']},
                 limitations=['Global image alignment did not pass all held-out frames; only admitted sparse frames have image-domain MANO assessment.',
                              '3D attachment fit is measured independently of camera calibration, but measures marker fitting rather than anatomical shape.',
                              'No silhouette or anatomy ground truth; glove thickness, marker offset and MANO local surface error cannot be separately identified.'])
    (args.output/'mano_local_assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Camera calibration does not affect Euclidean marker-attachment fitting.
    d=recording.data
    errors=np.linalg.norm(d['predicted_markers']-d['markers_world_m'],axis=-1)*1000
    valid=d['marker_valid'] & np.isfinite(errors)
    marker_fit=dict(definition='Euclidean norm(predicted_markers - markers_world_m) in millimetres, finite and marker_valid only; unique NOKOV observations',
                    image_alignment_required=False,anatomical_accuracy_measured=False,
                    overall=statistics(errors[valid]),
                    by_marker={str(n):statistics(errors[:,k][valid[:,k]]) for k,n in enumerate(d['marker_names'])},
                    by_finger={name:statistics(errors[:,[name in str(n) for n in d['marker_names']]][valid[:,[name in str(n) for n in d['marker_names']]]]) for name in ['Thumb','Index','Middle','Ring','Pinky']})
    (args.output/'marker_fit_3d.json').write_text(json.dumps(marker_fit,indent=2)+'\n')
    if args.validation_output:
        by_camera={}
        for c in range(4):
            ids=[i for i,o in enumerate(obs) if o['camera']==c and o['split']!='fit']
            checks={str(v['video_frame']):v for v in byframe.values() if v['camera']==c and v['role']!='fit'}
            passed=[int(f) for f,v in checks.items() if v['sparse_gate_pass']]
            failed=[int(f) for f,v in checks.items() if v['status']=='failed']
            by_camera[str(c)]=dict(status='validated_sparse_scope' if passed else 'failed' if failed else 'not_validated',
                                  validation_frames=sorted(map(int,checks)),validated_frames=passed,failed_frames=failed,
                                  frame_roles={f:v['role'] for f,v in checks.items()},frame_checks=checks,
                                  before_px=statistics(eb[ids]),after_px=statistics(ea[ids]),
                                  limitations=['Errors aggregate explicitly named evaluation roles; development frames are not blind tests.',
                                               'No successful camera/frame has enough independent observations for the current image-domain MANO gate.' if not passed else 'Only listed camera/frame samples passed; surrounding video frames are untested.'])
        metadata=dict(status='partial',approved_for_default=False,
                      projection_config_sha256=sha256(args.config),
                      observations_sha256=sha256(args.observations),
                      summary='固定共享安装校正已改善实测点投影，但独立最终测试仍失败；未获得可用于图像域 MANO 形状验收的通过范围。',
                      by_camera=by_camera,split_policy=annotation.get('role_limitation',''),
                      time_policy='原相机-IMU偏移0；NOKOV→EGO +12.651796571ms；geometry linear+SLERP；timestamp match delta不是同步精度。',
                      limitations=summary['limitations'],
                      observations=os.path.relpath(args.observations.resolve(),args.validation_output.parent.resolve()),
                      fit_result=os.path.relpath(args.config.with_name('alignment_result.json').resolve(),args.validation_output.parent.resolve()),
                      mano_local_assessment=os.path.relpath((args.output/'mano_local_assessment.json').resolve(),args.validation_output.parent.resolve()),
                      source_reconstruction_unchanged=True,geometry_sampling='linear positions and SLERP at camera capture timestamp')
        args.validation_output.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'admitted_frames':sorted(scope),'admitted':summary['admitted']},indent=2))


if __name__=='__main__':main()
