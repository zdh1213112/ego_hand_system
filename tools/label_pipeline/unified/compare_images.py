"""Three matching native image columns using the per-camera exported training label."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recordings.preview import HANDS,source_geometry,track,save_sample,decode,video,panel,audit,legend
from stages.signed import rows


def main():
    p=argparse.ArgumentParser()
    for k in ('source','exports','fixed','frames','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();out=a.output/'overlays';out.mkdir(parents=True,exist_ok=True)
    chosen=json.loads(a.frames.read_text());paths=[];lines=['# 逐帧四机对比','',legend(),'',
        '每张图从上到下是 cam0–3，三列为同一曝光的原图、legacy 和 unified。L 为已导出签收样本。'
        'L* 表示该视角未被纳入签收数据集，显示在此视角另行导出的诊断样本。U 为本次试算，所有试算样本仅供签收。','']
    for group in ('group_003','group_005'):
        native=a.source/group/'native';cams,rt,frames=source_geometry(native);exp=a.exports/group
        times={s:[r['sync'] for r in chosen[group][h]] for s,h in enumerate(HANDS)}
        lookup={(r['sync_index'],int(r['camera'][-1]),r['side']):r for r in rows(exp/'index.jsonl') if r['sync_index'] in times[r['side']]}
        for s,h in enumerate(HANDS):
            base=track(a.source/group/'tracks'/(h+'.npz'),times[s]);new=track(a.output/group/'tracks'/(h+'.npz'),times[s])
            with np.load(a.source/group/'wrist'/(h+'_outward.npz')) as z:
                wt=z['pair_indices'];wj=z['joints'][:,0]
            with np.load(native/'aligned_skeleton.npz') as z:raw=z['marker24_cam0_m'][s]
            bright=[]
            for c in range(4):
                with np.load(a.fixed/group/f'camera{c}.npz') as z:bright.append(z['uv'][s])
            caps=[video(native,c) for c in range(4)];lines += [f'## {group} {h}',''];links=[]
            try:
                for t in sorted(times[s]):
                    boards=[]
                    for c in range(4):
                        frame=int(frames[t][f'camera{c}_frame_index']);im=decode(caps[c],frame);pp=[];titles=[]
                        for k,(tr,poses,idx) in enumerate((base,new)):
                            method=('legacy','unified')[k];pure=None;gl=None;outward=None;name='L*' if k==0 else 'U'
                            origin='diagnostic '+method
                            if k==0 and (r:=lookup.get((t,c,s))) is not None:
                                pure=np.load(exp/r['label'],allow_pickle=True).item()
                                rich=np.load(exp/r['glove_label'],allow_pickle=True).item()
                                gl=rich['joints_3d_glove']
                                np.testing.assert_array_equal(pure['joints_3d'],rich['joints_3d'])
                                origin=str((exp/r['label']).resolve());name='L'
                            if k==0 and (j:=int(np.searchsorted(wt,t)))<len(wt) and wt[j]==t:outward=wj[j]
                            sample=None
                            if t in idx:
                                path=out/'labels'/group/h/f'{t:05}_c{c}_{method}.npy'
                                sample=save_sample(path,tr,poses,idx[t],rt[c],cams[c],c,s,t,im,
                                    raw[t]*np.array([1 if s else -1,1,1]),bright[c][t],pure,gl,outward,origin,frame,legacy_wrist=k==0)
                            pp.append(sample);titles.append(name)
                            if sample is not None:paths.append(sample)
                        boards.append(panel(im,pp,titles,f'cam{c} {t}'))
                    f=out/group/h/f'sync{t:04}.jpg';f.parent.mkdir(exist_ok=True,parents=True)
                    if not cv2.imwrite(str(f),np.vstack(boards),[cv2.IMWRITE_JPEG_QUALITY,93]):raise IOError(f)
                    links.append(f'[{t}]({f.relative_to(out)})')
            finally:
                for cap in caps:cap.release()
            lines+=[' · '.join(links),'']
    n=audit(paths,out)
    from recordings.loader_audit import check
    check(out,lambda sample:a.source/sample.parent.parent.name/'native')
    lines.append(f'[像素抽检 {n} 份](pixel_audit.json)')
    (out/'INDEX.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1);main()
