# Source work/group_010_anatomy_20261119/joint_anatomy_20261119/diagnose.py, SHA-256 cb8bd3ed7867b32350f8310272ce9de2a48196b73b39fb0c3c6f5f1fda7a7c02

import case_settings as case

import csv

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
