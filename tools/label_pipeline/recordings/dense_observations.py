"""Use real Finger0 observations in the dense solver and verify all exported geometry."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

EGO = Path(__file__).resolve().parents[3]
SOLVER = EGO / 'scripts/joint_refit/dense_drive.py'
SHARED = {'faces', 'marker_names', 'geometry_space', 'approved_for_training', 'betas',
          'attachment_faces', 'barycentric', 'attachment_local_offsets_m', 'normal_offsets_m',
          'wrist_local', 'root_anchor_local_m'}


def source_for_solver(source, output):
    dest = output.parent / (output.name + '_observed_root')
    dest.mkdir(exist_ok=True)
    report = {}
    for path in sorted(source.glob('*_pilot.npz')):
        with np.load(path) as data:
            times = data['pair_indices']
            required = ('finger0_target', 'joints', 'vertices', 'hand_pose_pca', 'global_orient', 'translation')
            valid = np.ones(len(times), dtype=bool)
            reasons = {}
            for key in required:
                good = np.isfinite(data[key]).reshape(len(times), -1).all(1)
                reasons[key] = times[~good].tolist()
                valid &= good
            if not valid.any():
                raise ValueError(f'{path} has no valid initial poses with a measured root')
            target = dest / path.name
            if valid.all():
                if not target.exists():
                    target.symlink_to(os.path.relpath(path, dest))
                if not target.samefile(path):
                    raise ValueError(f'stale input {target}')
            else:
                arrays = {}
                for key in data:
                    value = data[key]
                    if key not in SHARED and value.ndim and value.shape[0] == len(times):
                        value = value[valid]
                    arrays[key] = value
                temp = target.with_suffix('.pending.npz')
                np.savez_compressed(temp, **arrays)
                temp.replace(target)
            report[path.name] = dict(total=len(times), used=int(valid.sum()), excluded_sync=times[~valid].tolist(), by_field=reasons)
    if not report:
        raise FileNotFoundError(source)
    (dest / 'measured_root.json').write_text(json.dumps(report, indent=2) + '\n')
    return dest


def main():
    args = sys.argv[1:]
    source_pos = args.index('--source') + 1
    output = Path(args[args.index('--output') + 1]).resolve()
    args[source_pos] = str(source_for_solver(Path(args[source_pos]).resolve(), output))
    # Both entry points export to <output_root>/legacy/<recording>, including single-recording runs.
    batch = next((p.parent.parent for p in output.parents if p.parent.name == 'legacy'), output.parent)
    logs = batch / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / 'dense_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([sys.executable, '-Bu', str(SOLVER), *args], check=True)
    for name in sorted((output / 'final').glob('*_pilot.npz')):
        with np.load(name) as result:
            for key in ('joints', 'vertices', 'translation', 'hand_pose_pca', 'global_orient', 'predicted24', 'root_anchor_error_mm'):
                if not np.isfinite(result[key]).all():
                    raise ValueError(f'nonfinite {key} in {name}')


if __name__ == '__main__':
    main()
