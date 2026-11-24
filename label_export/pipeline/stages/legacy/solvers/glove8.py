# Source work/glove_joints_20261120/glove8.py, SHA-256 008d38b7170ccd5cce69ed5375921e8fb9c8266f655f17dbf2cb838e851c07cf

import case_settings as case

import json

from pathlib import Path

import numpy as np

def dump(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def take(path, *keys):
    with np.load(path) as z:
        return {k: z[k] for k in keys}
