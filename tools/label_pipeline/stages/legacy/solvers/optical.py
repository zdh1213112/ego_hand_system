# Source work/soft35_export_20261119/optical.py, SHA-256 c5606b52705d9d3f96cd1487164251b459429dbad2d6616aaf65255d876dde4f

import case_settings as case

import json

import cv2

from common import HANDS, HERE, geometry, output

from bright_spots import detect_all, identify

SETS = {
    "wrist_nail": (0, 1, 2, 7, 11, 15, 19, 23),
    "index_middle": tuple(range(8, 16)),
    "ring_pinky": tuple(range(16, 24)),
}

def optical_times(group, hand):
    return json.loads((output(group) / "calibration" / (hand + "_split.json")).read_text())[
        "optical_sync"
    ]

def run(group):
    dest = output(group)
    cv2.setNumThreads(1)
    source, cams, rt, rows = geometry(group)
    times = {h: optical_times(group, h) for h in HANDS}
    jobs = [
        (h, key)
        for h in HANDS
        for key in SETS
        if not (dest / "optical" / (h + "_" + key) / "coverage.json").is_file()
    ]
    if not jobs:
        return
    raw, clouds = detect_all(
        source, cams, rt, rows, sorted({t for h, k in jobs for t in times[h]}), group
    )
    for hand, key in jobs:
        identify(
            hand + "_" + key,
            times[hand],
            raw,
            clouds,
            cams,
            rt,
            target=(hand, SETS[key]),
            source_track=HERE / group / "aligned" / (hand + "_pilot.npz"),
            output_root=dest,
        )
