"""Execute unchanged numerical stages against one private, explicitly identified recording."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('case',type=Path);parser.add_argument('phase',choices=['imports','dense','motion','pivot','cuff','anatomy','mcp'])
    args=parser.parse_args();os.environ['LABEL_PIPELINE_CASE']=str(args.case.resolve())
    sys.path.insert(0,str(Path(__file__).with_name('solvers')))
    import case_settings as case
    import torch,cv2
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;cv2.setNumThreads(1)
    from cohort import HERE,CODE,HANDS
    import fit_cohort,cohort,choose_clips,fast_pinky,fit_pivot,observations,lock_tips
    import fit_recording,fit_wide,observed,refine_distal,fit_outward,preserve_pose,batch_constraints,tip_guard,guard_mcp_angle
    g=case.recording;opt=case.optimized
    co=HERE/g;motion=case.stage('joint_motion_20261119')
    def call(name,*argv):
        log=case.root/'records'/'legacy_solver'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-Bu',*map(str,argv)],cwd=case.ego,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} 失败，详见 {log}')
    if args.phase=='imports':
        print('loaded numerical stages for',g);return
    if args.phase=='dense':
        cohort.prepare(g);fit_cohort.fit_group(g)
        call('contour',CODE/'preserve_visible_contours.py','--selection',opt/'selection/selection.json','--source',co/'bounded/final','--calibrated',co/'calibrated','--skeleton',opt/'skeleton/aligned_skeleton.npz','--output',co/'contour')
        call('contour_export',CODE/'export.py','--context',opt/'context.json','--source',co/'contour/final')
    elif args.phase=='motion':
        out=motion/g/'final';out.mkdir(parents=True,exist_ok=True)
        keep=case.config.get('motion_keep',{})
        keep_args=[v for h,ts in keep.items() for t in ts for v in ('--keep',h,t)]
        call('motion',CODE/'temporal_articulation.py','--source',co/'contour/final','--report',co/'contour/summary.json','--reference',co/'reference/final','--skeleton',opt/'skeleton/aligned_skeleton.npz','--calibrated',co/'calibrated','--output',out,*keep_args)
        from temporal_articulation import write_report
        write_report(out,co/'contour/final',46,224,[[h,t] for h,ts in keep.items() for t in ts])
        call('motion_export',CODE/'export.py','--context',opt/'context.json','--source',out,'--reuse-native-images-from',co/'contour/final/export/merged')
        for filename,value in case.config['wilor_observations'].items():
            if filename not in ('native_wilor.npz','fast_wilor.npz','extra_wilor.npz'):raise ValueError(filename)
            link=out.parent/filename;source=(case.project/value).resolve(strict=True)
            if not link.exists():link.symlink_to(source)
        with (out/'export/merged/index.jsonl').open() as stream:sync=sorted({r['sync_index'] for r in map(json.loads,stream)})
        cutoff={g:sync[min(len(sync)-1,max(1,round(len(sync)*case.params['train_ratio'])))]}
        clips=choose_clips.choose(motion,HERE,cutoff,keep)[g]
        # Historical calibration records may specify reviewed windows. New recordings use the data-derived windows.
        if case.config.get('reviewed_fast_clips') is not None:clips=case.config['reviewed_fast_clips']
        save(motion/'fast_clips.json',{g:clips})
        for clip in clips:
            with np.load(out.parent/'fast_wilor.npz') as z:covered=set(z['times'])
            if not set(clip['frames'])<=covered:raise ValueError('同曝光 WiLoR 观测缺少快速片段，需单独确认推理。')
            fast_pinky.fit(clip)
    elif args.phase=='pivot':
        extra=int(case.config.get('wrist_candidate_samples',28))
        sys.argv=['fit_pivot','--groups',g,'--modes','geometry6','--extra',str(extra),'--steps','125']
        fit_pivot.main()
        observations.optical(g)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=fit_pivot.import_mano(case.ego/'third_party/MANO').load(model_path=str(case.ego/'models/mano'),is_rhand=True,use_pca=True,num_pca_comps=45,flat_hand_mean=False,batch_size=1).to(device).eval().requires_grad_(False)
        for h in HANDS:
            a,s,c=fit_pivot.array_pack(g,h,extra,model=model)
            lock_tips.lock(g,h,fit_pivot.Trial(model,a,s,c,device),fit_pivot.OUT/g/h,'geometry6')
    elif args.phase=='cuff':
        for h in HANDS:fit_recording.fit_hand(g,h)
    elif args.phase=='anatomy':
        observed.ROOT.mkdir(parents=True,exist_ok=True)
        for h in HANDS:fit_wide.build(g,h)
        for h in HANDS:observed.prepare(g,h)
        for h in HANDS:refine_distal.solve(g,h,preserve_anatomy=True)
    elif args.phase=='mcp':
        study=fit_outward.HERE
        for h in HANDS:
            amount=case.config.get('mcp_out_mm',case.params['mcp_out_mm']).get(h,case.params['mcp_out_mm'][h])
            fit_outward.run(g,h,'mild',target_mm=amount)
            preserve_pose.run(g,h);batch_constraints.run(g,h);tip_guard.run(g,h)
            if json.loads((study/g/'final'/f'{h}_tip_guard.json').read_text())['remaining']:
                tip_guard.run(g,h,palm=True);batch_constraints.run(g,h,tips_only=True)
        for h in HANDS:guard_mcp_angle.run(g,h)
        final=study/g/'final';archive=study/g/'before_mcp_angle_guard'
        if archive.exists():raise FileExistsError(archive)
        final.rename(archive);(study/'angle_guard'/g/'final').rename(final)
if __name__=='__main__':
    from phase_record import record
    with record(sys.argv[1],'solver_'+sys.argv[2]):main()
