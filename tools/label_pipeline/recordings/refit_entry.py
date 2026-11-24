"""Delegate to the existing refit driver, selecting the missing-aware SDK observation reader."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve()
EGO=HERE.parents[3]
SOURCE=EGO/'scripts/run_joint_refit_recording.py'
EXPECTED=EGO/'tools/label_pipeline/calibration/nokov_skeleton_audit/analyze.py'


def main():
    spec=importlib.util.spec_from_file_location('recording_refit_driver',SOURCE)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    def execute(argv,**kwargs):
        if len(argv)>2 and Path(argv[2]).resolve()==(EGO/'scripts/joint_refit/dense_drive.py'):
            return subprocess.run([*argv[:2],str(HERE.with_name('dense_observations.py')),*argv[3:]],**kwargs)
        if len(argv)>2 and Path(argv[2]).resolve()==EXPECTED.resolve():
            effective=[*argv[:2],str(HERE.with_name('sdk_observations.py')),*argv[3:]]
            out=Path(sys.argv[sys.argv.index('--output')+1])
            (out/'logs/sdk_reader_command.json').write_text(json.dumps(dict(original=argv,actual=effective),indent=2)+'\n')
            return subprocess.run(effective,**kwargs)
        return subprocess.run(argv,**kwargs)
    driver.subprocess=SimpleNamespace(run=execute,STDOUT=subprocess.STDOUT)
    driver.main()


if __name__=='__main__':main()
