"""Run the unmodified native MANO baseline and stride-four calibration in a private recording."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from recordings.marker_names import compatible_alignment

def execute(out,placement,stride=4):
    ego=Path(__file__).resolve().parents[3]
    n=out/'native';processed=n/'processed';aligned=n/'fourview_alignment';baseline=n/'baseline';opt=n/'optimized_stride4'
    alignment=compatible_alignment(out)
    nokov=Path(json.loads(alignment.read_text())['source_nokov'])
    tasks=[('baseline',[ego/'scripts/run_nokov_native_labels.py','--processed-session',processed,'--aligned-nokov',aligned/'aligned_nokov21.npz','--hand-side','auto','--cameras','camera0','camera1','camera2','camera3','--sample-stride','8','--max-samples','0','--device','cuda','--output',baseline]),('stride4',[Path(__file__).with_name('refit_entry.py'),'--dataset',processed/'ego/normalized_multiview','--alignment',alignment,'--aligned21',aligned/'aligned_nokov21.npz','--baseline',baseline,'--nokov',nokov,'--placement',placement,'--output',opt,'--sample-stride',str(stride),'--device','cuda','--workers','2'])]
    for name,args in tasks:
        args=[sys.executable,'-Bu',*map(str,args)]
        log=out/'records'/'legacy_seed'/name;log.mkdir(parents=True,exist_ok=True)
        record=log/'run_record.json'
        if record.is_file():
            r=json.loads(record.read_text())
            if r.get('return_code')==0:
                if r['command']!=args and r['command']!=[*args,'--resume']:raise ValueError('已完成的阶段入口变化 '+name)
                continue
            if name=='stride4' and opt.exists():
                args.append('--resume')
                with (log/'attempts.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')

        start=time.time();r=dict(command=args,start=start,fit_method='legacy')
        record.write_text(json.dumps(r,indent=2))
        with (log/'console.log').open('w') as stream:
            result=subprocess.run(args,cwd=ego,stdout=stream,stderr=subprocess.STDOUT)
        r.update(return_code=result.returncode,elapsed_s=time.time()-start)
        record.write_text(json.dumps(r,indent=2))
        result.check_returncode()
