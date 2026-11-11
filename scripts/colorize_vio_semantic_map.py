#!/usr/bin/env python3
"""Add RGB and predicted semantic labels to existing frozen VIO/NFR maps.

Read original colour videos using the exact frame_map indices and observed
fisheye pixels. Fuse SegFormer probabilities across views, retain unknowns,
and export every original landmark in NOKOV and fixed initial-IMU frames.
This stage never reruns VIO/mapping or fits sensor/world transformations.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
from ego_data.fixed_relations import save_json,sha256
from ego_data.map_view import change_frame,initial_imu_frame,read_poses,refresh_map_view

DEFAULT_MODEL=ROOT/'models/semantic/segformer_b0_ade20k'
MODEL_ID='nvidia/segformer-b0-finetuned-ade-512-512'
UNKNOWN=255
CHINESE={'wall':'墙面','building':'建筑','sky':'天空','floor':'地面','tree':'树木','ceiling':'天花板',
 'road':'道路','bed':'床','windowpane':'窗户','grass':'草地','cabinet':'柜子','sidewalk':'人行道',
 'person':'人','earth':'土地','door':'门','table':'桌子','mountain':'山','plant':'植物',
 'curtain':'窗帘','chair':'椅子','car':'汽车','water':'水','painting':'画','sofa':'沙发',
 'shelf':'架子','house':'房屋','mirror':'镜子','rug':'地毯','armchair':'扶手椅','seat':'座位',
 'fence':'围栏','desk':'书桌','rock':'岩石','wardrobe':'衣柜','lamp':'灯','bathtub':'浴缸',
 'railing':'栏杆','cushion':'靠垫','box':'箱子','column':'柱子','signboard':'标牌','chest of drawers':'抽屉柜',
 'counter':'台面','sink':'水槽','fireplace':'壁炉','refrigerator':'冰箱','stairs':'楼梯',
 'case':'展柜','pool table':'台球桌','pillow':'枕头','screen door':'纱门','bookcase':'书架',
 'coffee table':'茶几','toilet':'马桶','flower':'花','book':'书','bench':'长凳','countertop':'台面',
 'stove':'炉灶','computer':'电脑','swivel chair':'转椅','bottle':'瓶子','television receiver':'电视',
 'fan':'风扇','apparel':'衣物','monitor':'显示器','bulletin board':'公告板','poster':'海报',
 'screen':'屏幕','whiteboard':'白板','basket':'篮子','bag':'包','blanket':'毯子','radiator':'暖气片',
 'microwave':'微波炉','glass':'玻璃','plate':'盘子','crt screen':'显示屏','trash can':'垃圾桶',
 'tray':'托盘','cup':'杯子','clock':'钟','light':'灯光','stool':'凳子','towel':'毛巾'}


def semantic_palette(count=256):
    palette=np.zeros((count,3),dtype=np.uint8)
    for k in range(count):
        palette[k]=np.round(np.array(colorsys.hls_to_rgb((k*.61803398875+.04)%1,.56,.65))*255)
    palette[UNKNOWN]=[148,163,184]
    return palette


def fuse_probabilities(total,weights,votes,views,min_confidence,min_agreement,min_views):
    posterior=total/np.maximum(weights[:,None],1e-12)
    candidate=posterior.argmax(axis=1)
    confidence=posterior[np.arange(len(total)),candidate]
    agreement=votes[np.arange(len(total)),candidate]/np.maximum(weights,1e-12)
    labels=candidate.astype(np.uint16)
    labels[(confidence<min_confidence)|(agreement<min_agreement)|(views<min_views)]=UNKNOWN
    return labels,confidence,agreement


def write_coloured_ply(path,points,colours,labels,confidence):
    with Path(path).open('w') as f:
        f.write(f'ply\nformat ascii 1.0\ncomment sparse NFR landmarks; predicted semantics; units metres\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort class_id\nproperty float confidence\nend_header\n')
        for p,c,label,score in zip(points,colours,labels,confidence):
            f.write(' '.join([*[f'{v:.16g}' for v in p],*[str(int(v)) for v in c],str(int(label)),f'{score:.8g}'])+'\n')


def download_model(directory):
    import urllib.request
    directory.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://huggingface.co/api/models/'+MODEL_ID,timeout=30) as f:
        revision=json.load(f)['sha']
    for name in ['config.json','preprocessor_config.json','model.safetensors']:
        temporary=directory/(name+'.tmp')
        with urllib.request.urlopen(f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}',timeout=120) as src,temporary.open('wb') as dst:
            while block:=src.read(1024*1024):dst.write(block)
        temporary.replace(directory/name)
    save_json(directory/'model_manifest.json',{'model_id':MODEL_ID,'revision':revision,'source_url':'https://huggingface.co/'+MODEL_ID,
        'files':{name:sha256(directory/name) for name in ['config.json','preprocessor_config.json','model.safetensors']}})


def plot_coloured_maps(output):
    """Static figures reuse existing geometry/annotations, never model inference."""
    import os
    os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    summary=json.loads((output/'summary.json').read_text());clouds=[];maps={}
    semantic=json.loads((output/'semantic_map_summary.json').read_text())
    for case in summary['cases']:
        directory=output/'maps'/case
        if not (directory/'semantic_map.npz').exists():return
        with (directory/'landmarks_nokov_world.csv').open() as f:rows=list(csv.DictReader(f))
        world=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in rows])
        _,poses=read_poses(output/'evaluation'/f'{case}_imu_nokov_world.csv');transform=initial_imu_frame(poses[0])
        _,reference=read_poses(output/'evaluation/nokov_imu_reference.csv')
        points=change_frame(world,transform);mask=np.linalg.norm(points,axis=1)<=5
        with np.load(directory/'semantic_map.npz',allow_pickle=False) as data:
            maps[case]={'points':points,'mask':mask,'online':change_frame(poses[:,:3,3],transform),
                'reference':change_frame(reference[:,:3,3],transform),'semantic':data['semantic_rgb']/255.,'rgb':data['rgb']/255.}
        clouds.append(points[mask])
    cloud=np.vstack(clouds);lo,hi=cloud.min(axis=0),cloud.max(axis=0);center=(lo+hi)/2;radius=max(hi-lo)*.55
    for mode in ['semantic','rgb']:
        fig=plt.figure(figsize=(16,6),constrained_layout=True)
        for i,(case,m) in enumerate(maps.items()):
            ax=fig.add_subplot(1,3,i+1,projection='3d');mask=m['mask']
            ax.scatter(*m['points'][mask].T,c=m[mode][mask],s=5,alpha=.85)
            ax.plot(*m['online'].T,color=summary['cases'][case]['color'],lw=1)
            ax.plot(*m['reference'].T,color='#172b3a',lw=1)
            ax.set(title=f'{summary["cases"][case]["label"]}\n{mask.sum()} / {len(mask)} landmarks within 5 m',xlabel='IMU0 X [m]',ylabel='Y [m]',zlabel='Z [m]',
                xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-55)
        fig.suptitle('Initial IMU frame / '+('predicted semantic classes (grey = unknown)' if mode=='semantic' else 'recorded image RGB')+' / sparse NFR map')
        if mode=='semantic':
            observed={int(k) for case in semantic['cases'].values() for k in case['class_counts']}
            legend=[Line2D([0],[0],marker='o',linestyle='',color=np.array(c['colour'])/255.,label=c['name'],markersize=5) for c in semantic['classes'] if c['id'] in observed]
            fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,-.05),ncol=5,fontsize=9,frameon=False)
        fig.savefig(output/f'{mode}_maps_imu0.png',dpi=160,bbox_inches='tight');plt.close(fig)


def build(args):
    sys.path.insert(0,str(ROOT/'python_deps/semantic_runtime'))
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
    start=time.perf_counter();output=args.comparison.resolve();model_dir=args.model_dir.resolve()
    if args.download_model and not all((model_dir/name).exists() for name in ['model_manifest.json','config.json','preprocessor_config.json','model.safetensors']):download_model(model_dir)
    manifest=json.loads((model_dir/'model_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha256(model_dir/name)!=digest:raise ValueError(f'model hash mismatch: {name}')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(2);cv2.setNumThreads(1)
    processor=SegformerImageProcessor.from_pretrained(model_dir,local_files_only=True)
    model=SegformerForSemanticSegmentation.from_pretrained(model_dir,local_files_only=True,use_safetensors=True).eval().to(device)
    classes={int(k):v for k,v in model.config.id2label.items()};nclass=len(classes);palette=semantic_palette()
    print(json.dumps({'device':device,'model':manifest,'classes':nclass}),flush=True)
    info=json.loads((output/'input_validation.json').read_text());summary=json.loads((output/'summary.json').read_text())
    protected=[output/'relations_used.json',output/'summary.json',output/'metrics.csv']
    protected.append(Path(summary['relations_file']))
    for case in summary['cases']:
        protected.extend([output/'runs'/case/'repeat_1/trajectory.csv',output/'runs'/case/'repeat_1/map_lm/sparse_landmarks.csv',
                          output/'maps'/case/'landmarks_nokov_world.csv',output/'evaluation'/f'{case}_imu_nokov_world.csv'])
    protected_hashes={str(p):sha256(p) for p in protected}
    with (output/'dataset_4cam/frame_map.csv').open() as f:
        frame_map={int(r['timestamp_ns']):r for r in csv.DictReader(f)}
    groups={};case_data={};rejected=0;source_hashes={}
    for case in summary['cases']:
        source=output/'runs'/case/'repeat_1/map_lm';landmark_file=source/'sparse_landmarks.csv';observation_file=source/'map_observations.csv'
        with landmark_file.open() as f:landmarks=list(csv.DictReader(f))
        lookup={int(r['landmark_id']):i for i,r in enumerate(landmarks)};n=len(landmarks)
        case_data[case]={'rows':landmarks,'probability':np.zeros((n,nclass)), 'weight':np.zeros(n),
            'votes':np.zeros((n,nclass)),'views':np.zeros(n,dtype=np.int32),'colours':[[] for _ in range(n)]}
        source_hashes[case]={'landmarks':sha256(landmark_file),'observations':sha256(observation_file)}
        with observation_file.open() as f:
            for row in csv.DictReader(f):
                error=float(row['error_px']);u,v=float(row['u_px']),float(row['v_px']);cam=int(row['cam_id'])
                width,height=info['resolution']
                if error>args.max_reprojection_error or not (0<=u<width and 0<=v<height):rejected+=1;continue
                mapping=frame_map[int(row['timestamp_ns'])];frame=int(mapping[f'cam{cam}_frame_index'])
                groups.setdefault((cam,frame),[]).append((case,lookup[int(row['landmark_id'])],u,v,error))
    preview_dir=output/'semantic_previews';preview_dir.mkdir(exist_ok=True);previews=set();processed=0;model_seconds=0.
    for cam,stream in enumerate(info['source_camera_order']):
        needed={frame:entries for (c,frame),entries in groups.items() if c==cam}
        if not needed:continue
        video=Path(info['session'])/info['source_meta']['files'][f'video_{stream}']
        capture=cv2.VideoCapture(str(video));next_frame=0
        if not capture.isOpened():raise FileNotFoundError(video)
        try:
            for frame_number,entries in sorted(needed.items()):
                while next_frame<frame_number:
                    if not capture.grab():raise ValueError(f'video ended before frame {frame_number}')
                    next_frame+=1
                ok,bgr=capture.read();next_frame+=1
                if not ok:raise ValueError(f'cannot decode {video} frame {frame_number}')
                bgr=cv2.resize(bgr,tuple(info['resolution']),interpolation=cv2.INTER_AREA)
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);height,width=rgb.shape[:2]
                uv=np.array([[e[2],e[3]] for e in entries],dtype=np.float32)
                inputs=processor(images=rgb,return_tensors='pt').to(device)
                before=time.perf_counter()
                with torch.inference_mode():
                    logits=model(**inputs).logits
                    grid=torch.as_tensor((uv+.5)/[width,height]*2-1,dtype=torch.float32,device=device).reshape(1,1,-1,2)
                    probabilities=F.grid_sample(logits,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0,:].softmax(dim=0).T.cpu().numpy()
                model_seconds+=time.perf_counter()-before
                for case in case_data:
                    select=np.array([i for i,e in enumerate(entries) if e[0]==case],dtype=int)
                    if not len(select):continue
                    selected=[entries[i] for i in select];indices=np.array([e[1] for e in selected]);weight=1/(1+np.array([e[4] for e in selected])**2)
                    d=case_data[case];p=probabilities[select]
                    np.add.at(d['probability'],indices,p*weight[:,None]);np.add.at(d['weight'],indices,weight)
                    np.add.at(d['votes'],(indices,p.argmax(axis=1)),weight);np.add.at(d['views'],indices,1)
                    for entry in selected:
                        colour=np.median(cv2.getRectSubPix(rgb,(3,3),(entry[2],entry[3])).reshape(-1,3),axis=0)
                        d['colours'][entry[1]].append(colour)
                if cam not in previews and frame_number>=300:
                    with torch.inference_mode():labels=F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False).argmax(dim=1)[0].cpu().numpy()
                    overlay=cv2.addWeighted(rgb,.48,palette[labels],.52,0)
                    sheet=np.concatenate([rgb,overlay],axis=1);cv2.imwrite(str(preview_dir/f'cam{cam}_rgb_semantic.png'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR));previews.add(cam)
                processed+=1
                if processed%50==0:print(f'Colour/semantic keyframe images {processed}/{len(groups)}',flush=True)
        finally:capture.release()
    annotation={'model':manifest,'device':device,'class_count':nclass,'unknown_class_id':UNKNOWN,
        'classes':[{'id':i,'name':name,'name_zh':CHINESE.get(name,name),'colour':palette[i].tolist()} for i,name in classes.items()]+[{'id':UNKNOWN,'name':'unknown','name_zh':'未知','colour':palette[UNKNOWN].tolist()}],
        'method':'Multi-view soft probability fusion weighted by 1/(1+reprojection_error_px^2); RGB is the median of 3x3 image patches across valid observations',
        'semantic_scope':'Model predictions on recorded distorted fisheye images; not manually verified semantic ground truth; original sparse geometry retained',
        'thresholds':{'max_reprojection_error_px':args.max_reprojection_error,'min_confidence':args.min_confidence,'min_agreement':args.min_agreement,'min_views':args.min_views},
        'keyframe_images_processed':processed,'rejected_observations':rejected,'model_inference_s':model_seconds,
        'source_hashes':source_hashes,'frame_map_sha256':sha256(output/'dataset_4cam/frame_map.csv'),
        'protected_output_sha256':protected_hashes,'cases':{}}
    for case,d in case_data.items():
        labels,confidence,agreement=fuse_probabilities(d['probability'],d['weight'],d['votes'],d['views'],args.min_confidence,args.min_agreement,args.min_views)
        rgb=np.array([np.median(c,axis=0) if len(c) else [148,163,184] for c in d['colours']],dtype=np.uint8)
        semantic_rgb=palette[labels];directory=output/'maps'/case
        with (directory/'landmarks_nokov_world.csv').open() as f:world_rows=list(csv.DictReader(f))
        ids=np.array([int(r['landmark_id']) for r in d['rows']],dtype=np.int64)
        if [int(r['landmark_id']) for r in world_rows]!=ids.tolist():raise ValueError('map point order changed')
        world=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in world_rows])
        _,poses=read_poses(output/'evaluation'/f'{case}_imu_nokov_world.csv');transform=initial_imu_frame(poses[0]);imu0=change_frame(world,transform)
        np.savez_compressed(directory/'semantic_map.npz',landmark_ids=ids,rgb=rgb,class_ids=labels,semantic_rgb=semantic_rgb,confidence=confidence,agreement=agreement,views=d['views'],source_map_sha256=np.array(source_hashes[case]['landmarks']))
        for frame,points in [('nokov_world',world),('imu0',imu0)]:
            write_coloured_ply(directory/f'map_rgb_{frame}.ply',points,rgb,labels,confidence)
            write_coloured_ply(directory/f'map_semantic_{frame}.ply',points,semantic_rgb,labels,confidence)
        with (directory/'semantic_landmarks.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['landmark_id','nokov_x_m','nokov_y_m','nokov_z_m','imu0_x_m','imu0_y_m','imu0_z_m','red','green','blue','class_id','class_name','confidence','agreement','colour_semantic_views'])
            for i,lid in enumerate(ids):w.writerow([int(lid),*world[i],*imu0[i],*rgb[i],int(labels[i]),classes.get(int(labels[i]),'unknown'),confidence[i],agreement[i],int(d['views'][i])])
        counts={str(int(label)):int((labels==label).sum()) for label in np.unique(labels)}
        annotation['cases'][case]={'landmarks':len(ids),'labelled':int((labels!=UNKNOWN).sum()),'unknown':int((labels==UNKNOWN).sum()),'without_colour_observations':int((d['views']==0).sum()),'class_counts':counts,'observations_fused':int(d['views'].sum())}
    for path,digest in protected_hashes.items():
        if sha256(path)!=digest:raise ValueError(f'protected VIO result changed: {path}')
    annotation['wall_s']=time.perf_counter()-start;annotation['protected_outputs_unchanged']=True
    save_json(output/'semantic_map_summary.json',annotation)
    plot_coloured_maps(output)
    refresh_map_view(output)
    print(json.dumps({'viewer':str(output/'viewer.html'),'images':processed,'wall_s':annotation['wall_s'],'cases':annotation['cases']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,default=ROOT/'output/ego_nokov_fixed_full_20261105')
    p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--download-model',action='store_true',help='Download public model files if not already present')
    p.add_argument('--max-reprojection-error',type=float,default=3.)
    p.add_argument('--min-confidence',type=float,default=.45)
    p.add_argument('--min-agreement',type=float,default=.6)
    p.add_argument('--min-views',type=int,default=2)
    p.add_argument('--refresh-viewer',action='store_true',help='Refresh display only, reuse existing RGB/semantic annotations')
    a=p.parse_args()
    if not 0<=a.min_confidence<=1 or not 0<=a.min_agreement<=1 or a.min_views<1 or a.max_reprojection_error<=0:
        p.error('invalid semantic fusion threshold')
    if a.refresh_viewer:
        plot_coloured_maps(a.comparison.resolve())
        refresh_map_view(a.comparison.resolve())
    else:build(a)
