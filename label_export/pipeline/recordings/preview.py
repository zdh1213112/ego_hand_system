"""Native RGB review renderer. Drawing consumes only persisted per-exposure training samples."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from stages.geometry import CameraCalibration, camera_label, project, reconstructed_poses, reference_to_camera
from stages.fingerprint import digest
from stages.signed import write_json
from unified.optics import identify, peaks

HANDS=('left','right')
COLORS=((206,112,247),(252,167,66),(113,222,102),(52,190,252),(93,88,238))
POSE_FIELDS=('global_orient','translation','hand_pose_pca','vertices','joints')


def label_pixels(label):
    points=(np.asarray(label['joints_3d'],np.float32)+np.asarray(label['trans'],np.float32)).astype(np.float64)
    if points.shape!=(21,3) or not np.isfinite(points).all() or (points[:,2]<=0).any():
        raise ValueError('训练读取器无法使用该相机下的 21 个关节')
    return cv2.fisheye.projectPoints(points[:,None],np.zeros(3),np.zeros(3),
        np.array(label['K'],dtype=np.float32).astype(float),np.asarray(label['distortion'],float))[0][:,0].astype(np.float32)


def source_geometry(native):
    cams=[CameraCalibration.load(native/f'calibration/camera{c}.json') for c in range(4)]
    with (native/'multiview_frames.csv').open() as f: rows=list(csv.DictReader(f))
    return cams,[reference_to_camera(cams[0],c) for c in cams],rows


def decode(cap,index):
    pos=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if not pos<=index<=pos+8:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    else:
        for _ in range(index-pos):
            if not cap.grab():raise IOError(index)
    ok,im=cap.read()
    if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=index+1:raise IOError(index)
    return im


def video(native,c):
    return cv2.VideoCapture(str(native/f'cameras/camera{c}/video.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])


def track(path,times=None):
    with np.load(path) as z:
        t=z['pair_indices'];i=np.arange(len(t)) if times is None else np.flatnonzero(np.isin(t,times))
        out={k:z[k][i] for k in (*POSE_FIELDS,'target24','predicted24','marker_valid')}
        out.update(pair_indices=t[i],betas=z['betas'],faces=z['faces'])
    return out,reconstructed_poses(out),{int(t):i for i,t in enumerate(out['pair_indices'])}


def image_peaks(im,xy):
    size=np.array(im.shape[1::-1]);valid=np.isfinite(xy).all(1)&(xy>=2).all(1)&(xy<size-2).all(1)
    if valid.sum()<5:return np.full_like(xy,np.nan)
    return identify(xy,peaks(im,xy[valid]),valid,np.flatnonzero(valid))


def save_sample(path,tr,poses,i,rt,cam,c,s,t,im,measured,observed=None,pure=None,glove=None,outward=None,source='fit_candidate',frame_index=None,legacy_wrist=False):
    try:
        sample=dict(camera_label(tr,i,poses,rt,cam,s) if pure is None else pure)
        joints=label_pixels(sample)
    except ValueError as e:
        if str(e) not in ('No visible mesh in a source-qualified view','invalid or behind-camera geometry','训练读取器无法使用该相机下的 21 个关节'):raise
        return None
    mesh=np.asarray(sample['joints_2d'])
    if mesh.shape!=(778,2):raise ValueError('训练网格投影字段错误')
    if glove is None:
        actual=tr['predicted24'][i,4:]
        rg,off=rt
        wrist=tr['joints'][i,0]
        if legacy_wrist:wrist=outward if outward is not None else np.full(3,np.nan)
        surface=np.vstack([wrist,actual])
        glove=(surface*np.array([1 if s else -1,1,1])@rg.T+off-sample['trans']).astype(np.float32)
    raw=project(measured,rt,cam,s)
    spots=image_peaks(im,raw) if observed is None else observed
    rg,off=rt
    nominal=np.asarray(glove)+sample['trans']
    guv=project(nominal,(np.eye(3),np.zeros(3)),cam,1)
    sample.update(training_joints_2d_native=joints,joints_3d_glove=glove,review_surface_uv=guv.astype('f'),
        review_marker_uv=np.asarray(spots,'f'),review_marker_projection=raw.astype('f'),review_faces=tr['faces'].astype(np.int32),
        review_sync=int(t),review_camera=c,review_side=s,review_source=source,source_frame_index=frame_index,
        review_outward_uv=(guv[0].astype('f') if legacy_wrist and np.isfinite(guv[0]).all() else None),
        approved_for_training=False)
    path.parent.mkdir(exist_ok=True,parents=True);np.save(path,sample)
    return path


def read_sample(path):
    if path is None:return None
    sample=np.load(path,allow_pickle=True).item()
    np.testing.assert_array_equal(sample['training_joints_2d_native'],label_pixels(sample))
    return sample


def draw(im,sample):
    if sample is None:return im.copy()
    out=im.copy()
    uv=sample['training_joints_2d_native'];mesh=np.asarray(sample['joints_2d'])
    faces=sample['review_faces'];good=np.isfinite(mesh).all(1)&(abs(mesh)<1e5).all(1)
    faces=faces[good[faces].all(1)]
    if len(faces):
        wire=out.copy();tris=np.rint(mesh[faces]).astype('i')
        cv2.polylines(wire,tris,True,(173,197,157),1,cv2.LINE_AA)
        cv2.addWeighted(wire,.28,out,.72,0,out)
    def coord(p):return tuple(np.rint(p).astype(int)) if np.isfinite(p).all() and (abs(p)<1e5).all() else None
    for finger,color in enumerate(COLORS):
        pts=np.rint(uv[[0,*range(1+4*finger,5+4*finger)]]).astype('i')
        cv2.polylines(out,[pts],False,color,2,cv2.LINE_AA)
        for p in pts[1:]:cv2.circle(out,tuple(p),3,color,-1,cv2.LINE_AA)
    for p in sample['review_surface_uv'][1:]:
        if (pos:=coord(p)) is not None: cv2.drawMarker(out,pos,(0,155,255),cv2.MARKER_CROSS,8,1)
    for p in sample['review_marker_uv']:
        if (pos:=coord(p)) is not None:cv2.circle(out,pos,4,(245,243,0),1,cv2.LINE_AA)
    if sample['review_outward_uv'] is not None and (p:=coord(sample['review_outward_uv'])) is not None:
        cv2.drawMarker(out,p,(0,215,255),cv2.MARKER_DIAMOND,15,1,cv2.LINE_AA)
    p=coord(uv[0]);cv2.circle(out,p,6,(30,24,22),2,cv2.LINE_AA);cv2.circle(out,p,4,(255,255,255),2,cv2.LINE_AA)
    return out


def panel(im,paths,titles,title):
    samples=[read_sample(p) for p in paths]
    clouds=[s['training_joints_2d_native'] for s in samples if s is not None]
    H,W=im.shape[:2];xy=np.concatenate(clouds) if clouds else np.empty((0,2))
    inside=xy[(xy>=0).all(1)&(xy<[W,H]).all(1)]
    if len(inside):
        lo=inside.min(0);hi=inside.max(0);size=int(np.clip(max(hi-lo)*1.55,200,min(H,W)))
        x,y=np.clip(np.rint((lo+hi-size)/2).astype(int),0,[W-size,H-size])
    else:size=min(H,W);x,y=(W-size)//2,(H-size)//2
    tiles=[]
    for label,arr,valid in [('RGB',im,True),*[(l,draw(im,s),s is not None) for l,s in zip(titles,samples)]]:
        crop=cv2.resize(arr[y:y+size,x:x+size],(300,300),interpolation=cv2.INTER_AREA)
        top=cv2.copyMakeBorder(crop,27,0,0,0,cv2.BORDER_CONSTANT,value=(29,34,37))
        text=title+' '+label if valid else title+' '+label+' / no sample'
        cv2.putText(top,text,(5,19),0,.45,(248,248,248),1,cv2.LINE_AA)
        if not len(inside):cv2.putText(top,'Outside or no valid pose',(9,287),0,.55,(0,180,255),1)
        tiles.append(top)
    return np.hstack(tiles)


def audit(samples,out):
    import re
    from collections import defaultdict
    sequence=sorted(set(samples))
    if not sequence:raise ValueError('没有可抽检的训练样本')
    buckets=defaultdict(list)
    for path in sequence:
        camera=int(re.search(r'_c([0-3])_',path.name).group(1))
        side=path.parent.name if path.parent.name in HANDS else path.stem.rsplit('_',1)[-1]
        buckets[camera,side].append(path)
    selected=[];keys=sorted(buckets)
    # Allocate across all represented physical hands and cameras before drawing additional exposures.
    quotas={k:0 for k in keys}
    while sum(quotas.values())<min(20,len(sequence)):
        for k in keys:
            if quotas[k]<len(buckets[k]) and sum(quotas.values())<min(20,len(sequence)):quotas[k]+=1
    for k in keys:
        pool=buckets[k];n=quotas[k]
        if n:selected.extend(pool[i] for i in np.linspace(0,len(pool)-1,n+2)[1:-1].round().astype(int) if n<len(pool))
        if n==len(pool):selected.extend(pool)
    if len(set(selected))!=min(20,len(sequence)):
        selected=list(dict.fromkeys(selected))
        selected.extend(p for p in sequence if p not in selected and len(selected)<min(20,len(sequence)))
    rows=[]
    for path in selected:
        label=read_sample(path);uv=label['training_joints_2d_native']
        rows.append(dict(sample=str(path.relative_to(out)),sync=label['review_sync'],camera=label['review_camera'],
            hand=HANDS[label['review_side']],joints=len(uv),max_abs_px=float(abs(uv-label_pixels(label)).max()),sha256=digest(path)))
    write_json(out/'pixel_audit.json',dict(checked=len(rows),renderer_source='training_joints_2d_native',
        loader_formula='GloveDataset native KB projection before augmentation and cropping',samples=rows))
    return len(rows)


def legend():
    return ('白圈为训练腕点 MANO J0，彩线为训练用的 21 个关节，浅绿线为同一份训练标签中的 778 顶点网格。'
       '青圈为原图独立识别的亮点，橙叉为与手指同名的模型表面挂点。没有可靠的亮点就留空。'
       '若有黄色菱形，它是 legacy 外旋腕部参照，不参与训练。')


def write_video(native,review,rows,paths,draw_label=None):
    cap=video(native,0);fps=cap.get(cv2.CAP_PROP_FPS);dst=review/'camera0_full.mp4'
    # Preserve source exposure cadence, including periods where neither hand has a label.
    frame_ids={int(r['camera0_frame_index']):int(r['sync_index']) for r in rows}
    expected_source=round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));original_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=min(960,original_width)//2*2;height=round(original_height*width/original_width/2)*2
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dst.with_suffix('.pending.mp4'))]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);count=0;labelled=0
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            t=frame_ids.get(count);shown=[]
            for side in (0,1):
                p=paths.get((t,0,side))
                if p:
                    im=draw_label(im,p) if draw_label else draw(im,read_sample(p))
                    shown.append(HANDS[side])
            labelled+=bool(shown)
            im=cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA)
            caption=f'cam0 frame {count}  sync {t}  '+(', '.join(shown) or 'no exported hand')
            cv2.putText(im,caption,(10,27),0,.60,(18,18,18),3,cv2.LINE_AA)
            cv2.putText(im,caption,(10,27),0,.60,(245,245,245),1,cv2.LINE_AA)
            proc.stdin.write(im.tobytes());count+=1
    finally:cap.release();proc.stdin.close();rc=proc.wait()
    if rc or not count or count!=expected_source or max(frame_ids,default=-1)>=count:
        raise RuntimeError(f'camera0 整段校验失败，解码 {count}/{expected_source} 帧，编码器状态 {rc}')
    pending=dst.with_suffix('.pending.mp4')
    check=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(pending)]))
    if int(check['streams'][0]['nb_frames'])!=count:raise ValueError('视频帧总数不一致')
    pending.replace(dst);return dict(frames=count,frames_with_labels=labelled,fps=fps)


def wrist_review(root,cfg):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    method=cfg['fit_method'];g=cfg['id'];review=root/'review';out=review/'wrist_preview';out.mkdir(exist_ok=True,parents=True)
    native=root/'native/processed/ego/normalized_multiview';cams,rt,rows=source_geometry(native)
    (review/'WRIST_REVIEW.md').write_text(f'# {g} · 腕点预览\n\n正在从逐曝光训练格式标签生成四机图像。\n')
    targets=json.loads((review/'wrist_targets.json').read_text());choices={};tracks=[];outward=[];labels={}
    motion=root/'legacy_stages/joint_motion_20261119'/g/'final'
    for s,h in enumerate(HANDS):
        if method=='legacy':
            tr,poses,lookup=track(motion/(h+'_pilot.npz'))
            with np.load(targets[h]['target_file']) as z:
                pt=z['times'];j=z['joints'];ref=int(targets[h]['sync'])
                aligned=root/'legacy_stages/nokov_label_training_20261118'/g/'aligned'/(h+'_pilot.npz')
                with np.load(aligned) as measurements:
                    ix=np.searchsorted(measurements['pair_indices'],pt)
                    np.testing.assert_array_equal(measurements['pair_indices'][ix],pt)
                    visible=np.isfinite(measurements['target24'][ix,:3]).all((1,2))
            outward.append(dict(zip(pt,j[:,0])))
        else:
            tr,poses,lookup=track(Path(targets[h]['target_file']))
            pt=tr['pair_indices'];j=tr['joints'];visible=tr['marker_valid'][:,:3].all(1);ref=targets[h]['sync']
            outward.append({})
        cand=np.flatnonzero(visible&((pt//120)%4==0))
        if not len(cand):cand=np.flatnonzero(visible)
        if not len(cand):raise ValueError(f'{h} 无有效腕部候选')
        reach=np.linalg.norm(j[:,0,None]-j[:,[8,12,16,20]],axis=2).mean(1)
        choices[h]=dict(reference=ref,extended=int(pt[cand[reach[cand].argmax()]]),flexed=int(pt[cand[reach[cand].argmin()]]))
        early=review/'step04_preview/source.json'
        if early.is_file():choices[h]['step04_comparison']=int(json.loads(early.read_text())['sync'])
        tracks.append((tr,poses,lookup))
    with np.load(root/'native/optimized_stride4/skeleton/aligned_skeleton.npz' if method=='legacy' else root/'raw_observations/aligned_skeleton.npz') as z:
        measured=z['marker24_cam0_m']
    review_times=set(t for poses in choices.values() for t in poses.values())
    src=motion/'export/merged'
    exported={}
    if method=='legacy':
        with (src/'index.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line);t=int(r['sync_index']);c=int(r['camera'][-1]);s=int(r['side'])
                if c==0 or t in review_times:exported[t,c,s]=r['label']
    panels={}
    def collect(c,selected):
        cap=video(native,c)
        try:
            for t in selected:
                frame=int(rows[t][f'camera{c}_frame_index']);im=decode(cap,frame)
                for s,h in enumerate(HANDS):
                    if c and t not in choices[h].values():continue
                    tr,poses,lut=tracks[s];i=lut.get(t);p=None
                    if i is not None:
                        strict=exported.get((t,c,s));pure=np.load(src/strict,allow_pickle=True).item() if strict else None
                        legacy_target=outward[s].get(t)
                        p=save_sample(out/'labels'/f'{t:05}_c{c}_{h}.npy',tr,poses,i,rt[c],cams[c],c,s,t,im,
                            measured[s,t]*[1 if s else -1,1,1],pure=pure,outward=legacy_target,frame_index=frame,
                            source=str(src/strict) if strict else 'step06 training-format candidate',legacy_wrist=method=='legacy')
                        if p is not None:labels[t,c,s]=p
                    if t in choices[h].values():panels[t,c,s]=panel(im,[p],['MANO'],f'{h} cam{c} {t}')
        finally:cap.release()
    for c in range(4):collect(c,sorted(review_times))
    media={}
    for s,h in enumerate(HANDS):
        for pose,t in choices[h].items():
            dest=out/f'{h}_{pose}.jpg'
            if not cv2.imwrite(str(dest),np.vstack([panels[t,c,s] for c in range(4)])):raise OSError(dest)
            media[str(dest.relative_to(review))]=digest(dest)
    initial=[f'# {g} · 腕点候选图','',legend(),'','各行是同一只手的四机图，左列原图，右列为本阶段训练格式标签。整段视频和抽检在继续生成。','']
    for h,vals in choices.items():
        initial+=['**'+h+'**　'+' · '.join(f'[{name} {t}](wrist_preview/{h}_{name}.jpg)' for name,t in vals.items()),'']
    (review/'WRIST_REVIEW.md').write_text('\n'.join(initial)+'\n')
    write_json(out/'images_ready.json',dict(selected=choices,images=media,complete_review=False))
    collect(0,(t for t in range(len(rows)) if t not in review_times))
    n=audit(labels.values(),out)
    from recordings.loader_audit import check
    check(out,lambda _:native)
    counts=write_video(native,out,rows,labels)
    with (out/'labels_sha256.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['sync','camera','side','relative_path','sha256'])
        for (t,c,s),p in sorted(labels.items()):w.writerow([t,c,s,p.relative_to(out),digest(p)])
    for f in ['camera0_full.mp4','labels_sha256.csv','pixel_audit.json']:media['wrist_preview/'+f]=digest(out/f)
    proof=review/'wrist_preview_targets.json';write_json(proof,dict(method=method,sample_count=len(labels),
        selected=choices,camera0=counts,targets_sha256=digest(review/'wrist_targets.json'),media=media))
    sign=review/'wrist_signoff.json';template=json.loads(sign.read_text())
    if template.get('approved'):raise ValueError('此腕点签收已经生效，不能重建')
    write_json(sign,dict(template,preview_sha256=digest(proof)))
    lines=[f'# {g} · {method} 腕点签收','',legend(),'',
           '当前为第 06 步候选。页面和视频从已保存的单相机训练格式 NPY 读取关节、网格和手套点。'
           '第 12 步会从最终导出文件再画一次，并列出同帧 J0 的变化。','']
    geometric=json.loads((root/'native/fourview_alignment/fourview_alignment.json').read_text())
    lines+=['四机空间配准的留出亮点用于检查拍摄几何。下列数值对应各机原始匹配时刻。','',
            '| 相机 | 匹配数 | 中位 px | P95 px |','|---|---:|---:|---:|']
    for camera,v in geometric['heldout_by_camera'].items():
        if v.get('count'):lines.append(f"| {camera} | {v['count']} | {v['median_px']:.2f} | {v['p95_px']:.2f} |")
    lines+=['']
    if method=='legacy':lines+=['本次需要确认黄色菱形所示的外旋参照。白圈是当前训练格式标签的 J0，两者分别存储。','']
    else:lines+=['本次需要确认白圈所示的唯一 J0。','']
    for h,vals in choices.items():
        lines+=['**'+h+'**　'+' · '.join(f'[{name} {t}](wrist_preview/{h}_{name}.jpg)' for name,t in vals.items()),'']
    if (review/'step04_preview/README.md').is_file():
        lines+=['[第 04 步同帧原图和双手骨架](step04_preview/README.md)。上面的 step04_comparison 图使用第 06 步已保存的标签，便于直接核对同一曝光。','']
    lines += [f"[camera0 整段视频](wrist_preview/camera0_full.mp4) · {counts['frames']} 帧，其中 {counts['frames_with_labels']} 帧至少有一手候选。",
        f'[抽检 {n} 份像素坐标](wrist_preview/pixel_audit.json) · [对应训练格式 NPY](wrist_preview/labels) · [签收版本](wrist_preview_targets.json)','',
        '请按手确认参照帧，或指出希望重看的同步帧号。当前等待用户签收。']
    (review/'WRIST_REVIEW.md').write_text('\n'.join(lines)+'\n')
    return [proof,review/'WRIST_REVIEW.md']


def verify_review(root):
    r=root/'review';sign=json.loads((r/'wrist_signoff.json').read_text());proof=r/'wrist_preview_targets.json'
    if not sign.get('approved') or sign.get('preview_sha256')!=digest(proof):
        raise ValueError('等待用户签收 '+str(r/'WRIST_REVIEW.md'))
    if sign.get('source')=='automatic_measured_reference':
        decision=json.loads((r/'adaptive_wrist/decision.json').read_text())
        selection=digest(r/'adaptive_wrist/selection.json')
        if selection!=sign.get('selection_sha256') or selection!=decision.get('selection_sha256'):
            raise ValueError('自动判定依据已更改')
        if 'observation_sha256' in decision or 'observation_sha256' in sign:
            measurements=digest(r/'adaptive_wrist/cuff_rgb.json')
            if decision.get('observation_sha256')!=measurements or sign.get('observation_sha256')!=measurements:
                raise ValueError('腕部多机观测与自动判定版本不符')
        if decision.get('preview_sha256')!=sign['preview_sha256'] or any(decision['targets'][h]!=sign[h] for h in HANDS):
            raise ValueError('自动参照与当前腕部目标版本不符')
    d=json.loads(proof.read_text())
    if d['targets_sha256']!=digest(r/'wrist_targets.json'):raise ValueError('腕点目标已更改')
    for k,v in d['media'].items():
        if digest(r/k)!=v:raise ValueError('腕部签收内容变化 '+k)
    with (r/'wrist_preview/labels_sha256.csv').open() as f:
        for row in csv.DictReader(f):
            if digest(r/'wrist_preview'/row['relative_path'])!=row['sha256']:raise ValueError('腕部标签变化')


def dataset_review(root,cfg):
    """Enrich exact accepted training NPYs (no geometry reconstruction), then render the full source video."""
    native=root/'native/processed/ego/normalized_multiview';cams,rt,frames=source_geometry(native)
    dest=root/'review';out=dest/'dataset_preview';out.mkdir(parents=True,exist_ok=True)
    with (root/'index.jsonl').open() as f:index={(int(r['sync_index']),int(r['camera'][-1]),int(r['side'])):r for r in map(json.loads,f)}
    if not index or any(r.get('approved_for_training') for r in index.values()):raise ValueError('需要未批准训练的候选数据')
    with np.load(root/'native/optimized_stride4/skeleton/aligned_skeleton.npz' if cfg['fit_method']=='legacy' else root/'raw_observations/aligned_skeleton.npz') as z:
        raw=z['marker24_cam0_m']
    times={};faces=[]
    wrist=json.loads((dest/'wrist_targets.json').read_text())
    for s,h in enumerate(HANDS):
        avail=np.unique([t for t,c,s0 in index if s0==s])
        pick=avail[np.linspace(0,len(avail)-1,min(8,len(avail))).round().astype(int)] if len(avail) else []
        early=dest/'step04_preview/source.json'
        fixed={int(json.loads(early.read_text())['sync'])} if early.is_file() else set()
        times[s]=sorted(set(map(int,pick))|{wrist[h]['sync']}|fixed)
        with np.load(root/'geometry_input'/cfg['id']/'tracks'/(h+'.npz') if cfg['fit_method']=='legacy' else root/'tracks'/(h+'.npz')) as z:faces.append(z['faces'])
    panels={};labels={};deltas={}
    for c in range(4):
        cap=video(native,c)
        try:
            selected=set(times[0]+times[1])
            if c==0:selected.update(t for t,v,s in index if v==0)
            for t in sorted(selected):
                frame=int(frames[t][f'camera{c}_frame_index']);im=decode(cap,frame)
                for s,h in enumerate(HANDS):
                    r=index.get((t,c,s));sample=None
                    if r and (c==0 or t in times[s]):
                        pure=np.load(root/r['label'],allow_pickle=True).item()
                        rich=np.load(root/r['glove_label'],allow_pickle=True).item()
                        for k in ('joints_3d','vertices','trans','K','joints_2d'):np.testing.assert_array_equal(np.asarray(pure[k]),np.asarray(rich[k]))
                        p=out/'labels'/f'{t:05}_c{c}_{h}.npy'
                        sample=save_sample(p,dict(faces=faces[s]),None,None,rt[c],cams[c],c,s,t,im,
                            raw[s,t]*[1 if s else -1,1,1],pure=pure,glove=rich['joints_3d_glove'],source=r['label'],frame_index=frame,legacy_wrist=cfg['fit_method']=='legacy')
                        if sample is None:raise ValueError('已导出样本无法从训练字段生成预览 '+r['label'])
                        labels[t,c,s]=sample
                        if t==wrist[h]['sync']:
                            draft=dest/'wrist_preview/labels'/p.name
                            if draft.exists():
                                before=read_sample(draft);after=read_sample(sample)
                                delta=((after['joints_3d'][0]+after['trans'])-(before['joints_3d'][0]+before['trans']))
                                deltas[f'{h}_cam{c}']=dict(mm=float(np.linalg.norm(delta)*1000),px=float(np.linalg.norm(after['training_joints_2d_native'][0]-before['training_joints_2d_native'][0])))
                    if t in times[s]:panels[t,c,s]=panel(im,[sample],['final'],f'{h} cam{c} {t}')
        finally:cap.release()
    n=audit(labels.values(),out)
    from recordings.loader_audit import check
    check(out,lambda _:native)
    v=write_video(native,out,frames,labels);files={}
    for s,h in enumerate(HANDS):
        for t in times[s]:
            p=out/f'{h}_sync{t:04}.jpg'
            if not cv2.imwrite(str(p),np.vstack([panels[t,c,s] for c in range(4)])):raise OSError(p)
            files[str(p.relative_to(dest))]=digest(p)
    for name in ('camera0_full.mp4','pixel_audit.json'):files['dataset_preview/'+name]=digest(out/name)
    target=dest/'dataset_targets.json';meta=json.loads(target.read_text())
    meta.update(review_image_sha256=files,stage06_to_final_J0=deltas,video=v,approved_for_training=False)
    write_json(target,meta);write_json(dest/'dataset_signoff.json',dict(approved=False,sha256=digest(target)))
    lines=[f"# {cfg['id']} · 最终候选标签",'',legend(),'',
        f"本段有 {len(index)} 份训练格式样本。相机视角没有通过导出时，图中保留原片。整段视频覆盖原始 camera0 的 {v['frames']} 帧，其中 {v['frames_with_labels']} 帧有导出标签。",'',
        '[打开整段视频](dataset_preview/camera0_full.mp4) · [逐样本训练 NPY](../labels) · [数据集索引](../index.jsonl)','']
    for s,h in enumerate(HANDS):lines+=['**'+h+'**　'+' · '.join(f'[{t}](dataset_preview/{h}_sync{t:04}.jpg)' for t in times[s]),'']
    for k,d in deltas.items():lines.append(f"- {k} 腕部参照帧，第 06 步至最终 J0 相距 {d['mm']:.2f} mm，在原图相距 {d['px']:.2f} px。")
    lines+=['',f'[像素抽检 {n} 份](dataset_preview/pixel_audit.json) · [解剖及签收摘要](dataset_targets.json) · [训练读取校验](../loader_check.json)','',
        '等待用户看图签收，当前训练批准标志为 false。']
    (dest/'DATASET_REVIEW.md').write_text('\n'.join(lines)+'\n')
    return [target,dest/'DATASET_REVIEW.md']
