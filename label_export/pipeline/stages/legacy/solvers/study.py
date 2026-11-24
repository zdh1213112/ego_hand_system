# Source work/group_003_right_mount_study_20261119/study.py, SHA-256 918cb4b651c732399bbcb342e8d930108b0375b21614fce5c29a62cdb4526b3c

import case_settings as case

import json

import numpy as np

ROOT = case.stage('group_003_right_mount_study_20261119')

WORK = ROOT.parent

BASE = WORK / "group_003_cuff_fit_20261119"

from fit_cuff import EGO, GROUP, HERE, PARENT, TIPS, geometry, save

from fit_pivot import OPT

from observations import project

def load(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}

def write_json(path, obj):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def summary(x):
    a = np.asarray(x)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(n=0, median=None, p95=None, max=None)
    return dict(n=int(a.size), median=float(np.median(a)), p95=float(np.percentile(a,95)), max=float(a.max()))

def units(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1.e-12)
