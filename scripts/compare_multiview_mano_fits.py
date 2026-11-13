#!/usr/bin/env python3
"""Compare matched-frame RGB marker errors with identical surface attachments."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from multiview_mano_constraints import MultiviewMarkerConstraints, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'baseline-fit', 'refined-fit', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.bundle, allow_pickle=False) as data:
        bundle = {k: data[k] for k in data.files}
    expected = json.loads(str(bundle['baseline_fit_sha256_json']))
    lookup = {int(pair): i for i, pair in enumerate(bundle['pair_indices'])}
    result = {'metric': 'RGB marker centre reprojection; same frames and attachments',
              'limitation': 'Same-recording heldout RGB; NOKOV supervision remains active. Not anatomical joint accuracy or RGB-only ablation.',
              'bundle_sha256': sha256(args.bundle), 'hands': {}}
    for side in (0, 1):
        base_path = args.baseline_fit / f'track_{side}.npz'
        if sha256(base_path) != expected[str(side)]:
            raise ValueError('Baseline differs from attachment calibration')
        with np.load(base_path, allow_pickle=False) as data:
            baseline = {k: data[k] for k in data.files}
        with np.load(args.refined_fit / f'track_{side}.npz', allow_pickle=False) as data:
            refined = {k: data[k] for k in data.files}
        base_ids = {int(pair): i for i, pair in enumerate(baseline['pair_indices'])}
        pairs = refined['pair_indices']
        good = np.asarray([int(p) in base_ids and int(p) in lookup and bool(refined['render_valid'][i]) and bool(baseline['render_valid'][base_ids[int(p)]]) for i, p in enumerate(pairs)])
        pairs = pairs[good]
        constraints = MultiviewMarkerConstraints(bundle, side, np.array([lookup[int(p)] for p in pairs]), 'cpu')
        metrics = {'matched_frames': len(pairs)}
        for name, vertices in [('baseline', baseline['vertices'][[base_ids[int(p)] for p in pairs]]), ('refined', refined['vertices'][good])]:
            with torch.no_grad():
                centres = constraints.centres(torch.as_tensor(vertices, dtype=torch.float32))
                metrics[name] = constraints.diagnostics(centres)
        result['hands']['left' if side == 0 else 'right'] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
