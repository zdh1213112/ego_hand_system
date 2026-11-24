"""Generate physical camera views and bind accepted wrist targets to their bytes."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

def sha(path):
    with path.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)

def board(native,cams,rt,frames,t,side,joints,markers,title,glove=None):
    from observations import project,read_rgb
    tiles=[]
    for c,cam in enumerate(cams):
        cap=cv2.VideoCapture(str(native/f'cameras/camera{c}/video.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        try:im=read_rgb(cap,int(frames[t][f'camera{c}_frame_index']))
        finally:cap.release()
        uv=project(joints,rt[c],cam,side);points=project(markers,rt[c],cam,side)
        H,W=im.shape[:2]
        inside=lambda a:np.isfinite(a).all() and (a>=0).all() and (a<[W,H]).all()
        surface=None if glove is None else project(glove,rt[c],cam,side)
        visible=[p for cloud in (uv,points,surface) if cloud is not None for p in cloud if inside(p)]
        if visible:
            xy=np.array(visible);lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2
            size=min(max(150,int(max(hi-lo)*1.35)),H,W)
            x,y=np.clip(np.rint(center-size/2).astype(int),0,[W-size,H-size])
            crop=im[y:y+size,x:x+size].copy()
            for chain,color in zip([range(i,i+4) for i in (1,5,9,13,17)],[(215,120,240),(255,160,80),(130,220,100),(70,180,230),(70,70,240)]):
                ids=[0,*chain]
                for a,b in zip(ids,ids[1:]):
                    if inside(uv[a]) and inside(uv[b]):cv2.line(crop,tuple(np.rint(uv[a]-[x,y]).astype(int)),tuple(np.rint(uv[b]-[x,y]).astype(int)),color,2,cv2.LINE_AA)
            if surface is not None:
                for p in surface[1:]:
                    if inside(p):cv2.drawMarker(crop,tuple(np.rint(p-[x,y]).astype(int)),(0,175,255),cv2.MARKER_CROSS,8,1)
            for p in points:
                if inside(p):cv2.circle(crop,tuple(np.rint(p-[x,y]).astype(int)),4,(245,235,0),1,cv2.LINE_AA)
            display_wrist=uv[0] if surface is None else surface[0]
            if surface is not None and inside(uv[0]):
                cv2.drawMarker(crop,tuple(np.rint(uv[0]-[x,y]).astype(int)),(233,83,245),cv2.MARKER_CROSS,13,2,cv2.LINE_AA)
            if inside(display_wrist):
                cv2.drawMarker(crop,tuple(np.rint(display_wrist-[x,y]).astype(int)),(0,201,255),cv2.MARKER_DIAMOND,13,2,cv2.LINE_AA)
            pair=np.hstack([cv2.resize(im[y:y+size,x:x+size],(260,260)),cv2.resize(crop,(260,260))])
        else:
            thumb=cv2.resize(im,(260,260));pair=np.hstack([thumb,thumb]);cv2.putText(pair,'Outside field of view',(15,135),0,.65,(0,180,255),2)
        pad=cv2.copyMakeBorder(pair,32,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(pad,f'cam{c} / {title} / sync {t}',(8,22),0,.55,(240,240,240),1,cv2.LINE_AA);tiles.append(pad)
    return np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])

def wrists(case,required=None):
    from observations import geometry
    from fit_pivot import TIPS
    g=case.recording;review=case.root/'review';review.mkdir(exist_ok=True)
    native,cams,rt,frames=geometry(g)
    results={}
    for s,h in enumerate(('left','right')):
        path=case.stage('tip_pivot_20261119')/g/h/'geometry6_locked.npz'
        with np.load(path) as z:
            tt=z['times'];j=z['joints'];target=z['target24'];valid=z['marker_valid']
            # Geometry-only pivot ignores the cuff. Recover the three measured balls
            # from the same aligned observations used by the legacy wrist picker.
            aligned=case.stage('nokov_label_training_20261118')/g/'aligned'/(h+'_pilot.npz')
            with np.load(aligned) as measurements:
                slots=np.searchsorted(measurements['pair_indices'],tt)
                np.testing.assert_array_equal(measurements['pair_indices'][slots],tt)
                target[:,:3]=measurements['target24'][slots,:3]
            valid[:,:3]=np.isfinite(target[:,:3]).all(-1)
            eligible=valid[:,:3].all(1)
            cand=np.flatnonzero(eligible&((tt//120)%4==0))
            if not len(cand) and required and required.get(h) is not None:cand=np.flatnonzero(eligible)
            if not len(cand):raise ValueError(f'{h} 无完整腕带候选')
            reach=np.linalg.norm(j[:,0,None]-j[:,TIPS[1:]],axis=2).mean(1)
            pick=cand[np.argmin(abs(reach[cand]-np.percentile(reach[cand],75)))]
            if required and required.get(h) is not None:
                found=np.flatnonzero((tt==int(required[h]))&eligible)
                if len(found)!=1:raise ValueError(f'{h} 指定腕部时刻缺测，须重选')
                pick=found[0]
            choices={'reference':int(pick),'extended':int(cand[reach[cand].argmax()]),'flexed':int(cand[reach[cand].argmin()])}
            for label,i in choices.items():
                markers=np.where(valid[i,:,None],target[i],np.nan)
                image=board(native,cams,rt,frames,int(tt[i]),s,j[i],markers,h+' '+label)
                if not cv2.imwrite(str(review/f'{h}_{label}.jpg'),image):raise OSError(review)
            results[h]=dict(sync=int(tt[pick]),target_file=str(path),sha256=sha(path),wrist_right_canonical_m=j[pick,0].tolist(),candidate_count=len(cand))
    save(review/'wrist_targets.json',results)
    sign=review/'wrist_signoff.json'
    if not sign.exists() or json.loads(sign.read_text()).get('approved') is False:
        save(sign,dict(approved=False,**{h:{k:v[k] for k in ('sync','sha256')} for h,v in results.items()}))
    alias=case.root.parents[1]/'review'
    if not alias.exists():alias.symlink_to(review.relative_to(case.root.parents[1]),target_is_directory=True)
    (review/'WRIST_REVIEW.md').write_text('# 腕点签收\n\n每个视角左列为原图，右列为同一曝光的标签。橙菱形为外旋腕点，青圈为三颗袖口球及手指与手背的同名表面球。\n\n'+''.join(f"**{h} · {results[h]['sync']}**　[参照]({h}_reference.jpg) · [伸展]({h}_extended.jpg) · [卷曲]({h}_flexed.jpg)\n\n" for h in results)+'核对后在 wrist_signoff.json 中填写 approved、sync 和每手目标的 SHA-256。改选帧需按新帧重新生成参照和截图。\n')
    return results

def verify_wrist(case):
    review=case.root/'review'
    sign=json.loads((review/'wrist_signoff.json').read_text())
    targets=json.loads((review/'wrist_targets.json').read_text())
    if sign.get('approved') is not True:raise ValueError('等待腕点签收 '+str(review))
    for h,r in targets.items():
        if sign.get(h)!={'sync':r['sync'],'sha256':r['sha256']} or sha(Path(r['target_file']))!=r['sha256']:raise ValueError('腕点签收与目标不符 '+h)
    return {h:r['sync'] for h,r in targets.items()}

if __name__=='__main__':
    import argparse
    from raw_export import setup
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--reuse-signed-005',action='store_true');a=p.parse_args()
    case=setup(a.case);targets=wrists(case,case.config.get('wrist_sync'))
    if a.reuse_signed_005:
        if case.recording!='group_005' or {h:r['sync'] for h,r in targets.items()}!={'left':1928,'right':1032}:raise ValueError('历史许可仅覆盖 005 的 1928/1032')
        sign=dict(approved=True,source='ACCEPTANCE_17 historical reference frames; no new visual approval')
        sign.update({h:{k:r[k] for k in ('sync','sha256')} for h,r in targets.items()})
        save(case.root/'review/wrist_signoff.json',sign);verify_wrist(case)
