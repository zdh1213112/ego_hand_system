"""Record executable inputs for each separately launched numerical phase."""
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,time

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

@contextmanager
def record(case_path,name):
    case_path=Path(case_path).resolve(strict=True)
    cfg=json.loads(case_path.read_text());root=Path(cfg['recording_output'])
    dest=root/'records'/'phases'/name;dest.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parents[2];seq=time.time_ns();record_file=dest/(str(seq)+'.json')
    row=dict(phase=name,fit_method=cfg.get('fit_method','legacy'),started=datetime.now(timezone.utc).isoformat(),config_sha256=checksum(case_path),default_sha256=checksum(code/'defaults.yaml'),state='running',code_sha256={str(p.relative_to(code)):checksum(p) for p in sorted(code.rglob('*')) if p.suffix in ('.cpp','.py')})
    def write():
        tmp=record_file.with_suffix('.part');tmp.write_text(json.dumps(row,indent=2)+'\n');os.replace(tmp,record_file)
    write()
    try:
        yield
        row['state']='complete'
    except BaseException as exc:
        row.update(state='failed',exception=repr(exc));raise
    finally:
        row['ended']=datetime.now(timezone.utc).isoformat();write()
