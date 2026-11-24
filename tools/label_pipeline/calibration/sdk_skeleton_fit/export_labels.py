#!/usr/bin/env python3
"""Export a single assumed-internal SDK skeleton fit with explicit experimental provenance.

Uses the existing strict native label exporter/checker. Label array schema stays
unchanged; the assumption is carried by index.jsonl, summary.json and source metadata.
"""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parent/'ego_hand_system'
sys.path.insert(0,str(ROOT/'tools/personal_hand_calibration'))
from context import load_context
from run import import_mano,run_model
sys.path.insert(0,str(PROJECT/'scripts'))
from ego_data.calibration import CameraCalibration
from camera_models import project_points
from fit import mapping


def main():
    ap=argparse.ArgumentParser()
    for name in ['context','source','skeleton']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();c=load_context(a.context);fit_summary=json.loads((a.source/'summary.json').read_text())
    if fit_summary['strategy']!='sdk_skeleton_assumed_internal_joint_fit_v1' or fit_summary['approved_for_training'] is not False:raise ValueError('Expected explicitly experimental SDK fit')
    assumptions=fit_summary['assumptions'];assert assumptions['sdk_marker_to_internal_joint_offsets_configured'] is True and assumptions['assumption_verified'] is False
    out=a.source/'export';out.mkdir(exist_ok=False);(out/'logs').mkdir();mano_fit=a.source/'mano_fit';mano_fit.mkdir(exist_ok=False);fusion=a.source/'sdk_observations';fusion.mkdir(exist_ok=False)
    torch.set_num_threads(4);model=import_mano(PROJECT/'third_party/MANO').load(model_path=str(PROJECT/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1)
    base_summary=json.loads((c['baseline_fit']/'summary.json').read_text());summary=dict(base_summary)
    summary.update(fit_target='mano_joints',nokov_skeleton_fitting_enabled=True,approved_for_training=False,assumptions=assumptions,
                   root_policy=fit_summary['root_policy'],sdk_fit_source=str(a.source.resolve()),sdk_skeleton_file=str(a.skeleton.resolve()),
                   fit_error_point_semantics='mano_joints_vs_sdk_skeleton_assumed_internal',
                   parameters={'steps':350,'source_fit_summary':str((a.source/'summary.json').resolve())},tracks=[])
    tracks=[];replay_errors=[]
    for side,hand in enumerate(['left','right']):
        z=np.load(a.source/f'{hand}_pilot.npz');base=np.load(c['baseline_fit']/f'track_{side}.npz');n=len(z['pair_indices'])
        with torch.no_grad():v,j,res=run_model(model,torch.tensor(z['betas'])[None].expand(n,-1),torch.tensor(z['global_orient']),torch.tensor(z['hand_pose_pca']),torch.tensor(z['translation']))
        error=max(float(abs(v-torch.tensor(z['vertices'])).max()),float(abs(j-torch.tensor(z['joints'])).max()))
        if error>2e-6:raise ValueError(f'MANO parameter reconstruction mismatch: {error}')
        replay_errors.append(error);full=res.full_pose.numpy().reshape(n,16,3)
        values={k:z[k] for k in ['pair_indices','vertices','joints','betas','global_orient','hand_pose_pca','translation','faces']}
        values.update(mano_convention=base['mano_convention'],full_pose_axis_angle=full,hand_pose_axis_angle=full[:,1:],
                      joint_error_m=z['sdk_joint_error_mm']/1000,observation_error_m=z['sdk_joint_error_mm']/1000,
                      observation_points=z['joints'],render_valid=np.isfinite(z['vertices']).all((1,2)),
                      sdk_target21=z['sdk_target21'],sdk_segment_names=z['sdk_segment_names'],
                      fit_target=np.asarray('mano_joints'),approved_for_training=np.asarray(False))
        np.savez_compressed(mano_fit/f'track_{side}.npz',**values);tracks.append(z)
        summary['tracks'].append({'track_id':side,'handedness':hand.title(),'frames':n,'betas':z['betas'].tolist(),'sdk_mapping':z['sdk_segment_names'].tolist(),'fit_target':'mano_joints'})
    summary['maximum_mano_replay_error_m']=max(replay_errors);(mano_fit/'summary.json').write_text(json.dumps(summary,indent=2))
    sk=np.load(a.skeleton);root_policy=fit_summary['root_policy'];frames=sorted(set(tracks[0]['pair_indices'].tolist())|set(tracks[1]['pair_indices'].tolist()))
    cameras={f'camera{i}':CameraCalibration.load(c['dataset']/'calibration'/f'camera{i}.json') for i in range(4)};C0=cameras['camera0'];size=C0.image_size
    idx=[mapping(sk['names'][side],side,root_policy)[0] for side in [0,1]];rows=[]
    for frame in frames:
        row={'sync_index':int(frame),'hands':[]}
        for side in [0,1]:
            points=sk['positions_cam0_m'][side,frame,idx[side]]
            if not np.isfinite(points).all():continue
            base=points@C0.T_base_camera[:3,:3].T+C0.T_base_camera[:3,3]
            hand={'side':side,'joints_base_m':base.tolist(),'observed_joint_count':21,'views':{}}
            for camera,C in cameras.items():
                xyz=(base-C.T_base_camera[:3,3])@C.T_base_camera[:3,:3];uv,_=project_points(C,xyz)
                valid=np.isfinite(uv).all(1)&(xyz[:,2]>.03)&(uv[:,0]>=0)&(uv[:,0]<size[0])&(uv[:,1]>=0)&(uv[:,1]<size[1])
                if valid.all():hand['views'][camera]={'inlier_joint_count':21,'joints_2d':uv.tolist(),'detector_is_right':side,'observation_source':'SDK skeleton projection; internal offsets ASSUMED'}
            row['hands'].append(hand)
        rows.append(row)
    (fusion/'accepted.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (fusion/'summary.json').write_text(json.dumps({'stage':'nokov21_aligned_reference','source_group':'pair_20261111_192916__sdk_'+root_policy,
        'image_space':'native_raw','reference_camera':'camera0','accepted_frames':len(rows),'camera_ids':list(cameras),
        'pixel_target_provenance':'KB projection of SDK skeleton joint origins, internal offsets ASSUMED; not independent RGB truth',
        'assumptions':assumptions,'approved_for_training':False},indent=2))
    commands=[]
    def run(name,script,*args):
        cmd=[sys.executable,'-B',str(PROJECT/'scripts'/script),*map(str,args)];entry={'stage':name,'command':cmd,'returncode':None};commands.append(entry)
        (out/'commands.json').write_text(json.dumps(commands,indent=2));print(name,flush=True)
        env=dict(os.environ,PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
        with (out/'logs'/f'{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        entry['returncode']=r.returncode;(out/'commands.json').write_text(json.dumps(commands,indent=2))
        if r.returncode:raise RuntimeError(f'{name} failed; see {out/"logs"/(name+".log")}')
    dataset=out/'merged'
    run('export','export_native_wilor_training_dataset.py','--fusion',fusion,'--mano-fit',mano_fit,'--dataset',c['dataset'],'--reference-camera','camera0','--output',dataset,'--cameras',*list(cameras),'--view-filter','complete21','--image-space','native_raw','--sample-stride',1,'--require-both-hands')
    ds=json.loads((dataset/'summary.json').read_text());ds.update(approved_for_training=False,nokov_skeleton_fitting_enabled=True,assumptions=assumptions,
        root_policy=root_policy,source='SDK skeleton assumed internal joints + fixed personal MANO shape + native KB projection',
        fit_error_point_semantics='mano_joints_vs_sdk_skeleton_assumed_internal',source_fit_summary=str((a.source/'summary.json').resolve()))
    (dataset/'summary.json').write_text(json.dumps(ds,indent=2))
    rows=[json.loads(r) for r in (dataset/'index.jsonl').read_text().splitlines() if r.strip()]
    for row in rows:row.update(approved_for_training=False,nokov_skeleton_fitting_enabled=True,
        sdk_internal_offset_assumed=True,sdk_internal_offset_verified=False,sdk_root_policy=root_policy,
        source_experiment='SDK骨架假定内部关节 · '+root_policy,
        supervision_source='sdk_skeleton_assumed_internal_joint_fit',fit_error_point_semantics='mano_joints_vs_sdk_skeleton_assumed_internal')
    (dataset/'index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (dataset/'assumptions.json').write_text(json.dumps(assumptions,indent=2))
    run('check','check_wilor_training_dataset.py',dataset,'--mano-source',PROJECT/'third_party/MANO','--mano-model-dir',PROJECT/'models/mano','--device',c['device'])
    print(json.dumps({'labels':str(dataset),'images':ds['image_count'],'labels_count':ds['label_count'],'maximum_mano_replay_error_m':max(replay_errors)}),flush=True)


if __name__=='__main__':main()
