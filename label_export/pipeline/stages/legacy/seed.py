"""Run the unmodified native MANO baseline and stride-four calibration in a private recording."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
def execute(out,placement,stride=4):
    ego=Path(__file__).resolve().parents[4]
    n=out/'native';processed=n/'processed';aligned=n/'fourview_alignment';baseline=n/'baseline';opt=n/'optimized_stride4'
    nokov,=(out/'converted').glob('session*/nokov')
    tasks=[('baseline',[ego/'label_export/run_nokov_native_labels.py','--processed-session',processed,'--aligned-nokov',aligned/'aligned_nokov21.npz','--hand-side','auto','--cameras','camera0','camera1','camera2','camera3','--sample-stride','8','--max-samples','0','--device','cuda','--output',baseline]),('stride4',[ego/'label_export/run_joint_refit_recording.py','--dataset',processed/'ego/normalized_multiview','--alignment',aligned/'fourview_alignment.json','--aligned21',aligned/'aligned_nokov21.npz','--baseline',baseline,'--nokov',nokov,'--placement',placement,'--output',opt,'--sample-stride',str(stride),'--device','cuda','--workers','2'])]
    for name,args in tasks:
        args=[sys.executable,'-Bu',*map(str,args)]
        log=out/'records'/'legacy_seed'/name;log.mkdir(parents=True,exist_ok=True)
        record=log/'run_record.json'
        if record.is_file():
            r=json.loads(record.read_text())
            if r['command']!=args:raise ValueError('入口参数变化 '+name)
            if r.get('return_code')==0:continue
        start=time.time();r=dict(command=args,start=start,fit_method='legacy')
        record.write_text(json.dumps(r,indent=2))
        with (log/'console.log').open('w') as stream:
            result=subprocess.run(args,cwd=ego,stdout=stream,stderr=subprocess.STDOUT)
        r.update(return_code=result.returncode,elapsed_s=time.time()-start)
        record.write_text(json.dumps(r,indent=2))
        result.check_returncode()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--placement',type=Path,required=True)
    a=p.parse_args();execute(a.output.resolve(),a.placement.resolve())
