# Source work/nokov_label_training_20261118/fit_cohort.py, SHA-256 0c30dff09663dc1c0c6096544d8825c56498bac0e656ec8e013e5a9c64947087

import case_settings as case

import time

from cohort import HERE, CODE, OPT, run

def fit_group(group):
    dest=HERE/group
    reference=dest/'reference'
    if (dest/'bounded/summary.json').is_file():
        return
    if not all((reference/'final'/f'{h}_pilot.npz').is_file() for h in ('left','right')):
        raise FileNotFoundError(reference)
    scratch=dest/'bounded.partial'
    if scratch.exists():
        scratch.rename(dest/f'bounded.incomplete.{time.time_ns()}')
    opt=OPT[group]
    run(dest/'bounded.log',[CODE/'bounded_articulation.py','--selection',opt/'selection/selection.json',
        '--skeleton',opt/'skeleton/aligned_skeleton.npz','--calibrated',dest/'calibrated',
        '--source',reference/'final','--recover-empty-recording','--output',scratch])
    scratch.rename(dest/'bounded')
