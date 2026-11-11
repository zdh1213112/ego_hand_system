#!/usr/bin/env python3
"""Fit, inspect and render one fixed NOKOV-to-EGO camera mount correction.

Example (from the repository root, with NumPy/SciPy/OpenCV installed)::

    python scripts/run_nokov_image_alignment.py \
        --mano output/nokov_right_hand_mano_20261111 \
        --observations output/nokov_right_hand_mano_20261111/video_overlay_aligned/diagnostics/marker_observations.json

The baseline is MANO_DIRECTORY/video_overlay unless --baseline is supplied.
Annotations contain native image coordinates, reviewed marker identities,
image hashes, localization uncertainty and explicit fit/development/test roles.
The fit uses measured NOKOV markers only, preserving the intrinsics, camera
relative poses, clock offset and MANO reconstruction. A successful optimizer
does not imply successful held-out image validation. See alignment_validation.json.

--preview-only writes review images instead of full videos. --skip-render only
fits and evaluates. Neither option certifies video completeness. To refresh the
3D viewer afterwards, run visualize_nokov_mano.py --input MANO_DIRECTORY --no-video
and build its mano_viewer.src.html with the visual-explainer renderer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mano', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview-only', action='store_true')
    mode.add_argument('--skip-render', action='store_true')
    args = parser.parse_args()
    mano = args.mano.resolve()
    baseline = (args.baseline or mano / 'video_overlay').resolve()
    output = (args.output or mano / 'video_overlay_aligned').resolve()
    if output == baseline:
        parser.error('Correction output must differ from the baseline directory')
    diagnostics = output / 'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    commands = []

    def run(script, *arguments):
        command = [sys.executable, str(scripts / script), *map(str, arguments)]
        commands.append(command)
        (output / 'run_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        subprocess.run(command, check=True)

    run('calibrate_nokov_image_markers.py', '--baseline', baseline, '--mano', mano,
        '--observations', args.observations.resolve(), '--output', diagnostics)
    run('inspect_nokov_image_alignment.py', '--baseline', baseline, '--mano', mano,
        '--observations', args.observations.resolve(), '--config', diagnostics / 'projection_config.json',
        '--output', diagnostics, '--validation-output', output / 'alignment_validation.json')
    if not args.skip_render:
        summary = json.loads((baseline / 'overlay_summary.json').read_text())
        calibration = baseline / 'calibration/nokov_calibration.json'
        nokov = json.loads(calibration.read_text())['reference_directory']
        run('render_nokov_mano_overlay.py', '--ego-session', summary['ego_session'], '--nokov', nokov,
            '--mano', mano, '--output', output, '--calibration-source', calibration,
            '--projection-config', diagnostics / 'projection_config.json',
            *(['--preview-only'] if args.preview_only else []))
    print(f'Alignment status: {output / "alignment_validation.json"}')


if __name__ == '__main__':
    main()
