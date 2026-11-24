#!/usr/bin/env python3
"""Materialize actual experimental labels through the existing native exporter."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT
sys.path.insert(0, str(ROOT / 'label_export/pipeline/calibration/personal_hand_calibration'))
from context import load_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    a = ap.parse_args(); c = load_context(a.context)
    s = json.loads((a.source / 'summary.json').read_text())
    if s['strategy'] != 'marker24_with_optional_sdk_direction_v1' or s['approved_for_training'] is not False:
        raise ValueError('Expected an experimental marker/direction fit')
    output = a.source / 'export'; output.mkdir(exist_ok=False); (output / 'logs').mkdir()
    commands = []
    def run(name, script, *args):
        command = [sys.executable, '-B', str(script), *map(str, args)]
        row = dict(stage=name, command=command, returncode=None); commands.append(row)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2)); print(name, flush=True)
        env = dict(os.environ, PYTHONNOUSERSITE='1'); env.pop('PYTHONPATH', None)
        with (output / 'logs' / f'{name}.log').open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        row['returncode'] = result.returncode
        (output / 'commands.json').write_text(json.dumps(commands, indent=2))
        if result.returncode: raise RuntimeError(f'{name} failed: {output / "logs" / (name + ".log")}')
    fit = a.source / 'mano_fit'
    run('materialize', ROOT / 'label_export/pipeline/calibration/personal_hand_calibration/materialize_fit.py', '--context', a.context, '--source', a.source, '--output', fit)
    metadata = dict(sdk_direction_enabled=s['variant'] == 'direction', sdk_direction_mapping_verified=False,
                    sdk_root_policy='finger0', sdk_root_position_used_as_mano_target=False,
                    sdk_joint_positions_used_as_targets=False, sdk_bone_lengths_used_as_targets=False,
                    shared_shape_frozen=True, marker_attachments_frozen=True, root_translation_frozen=True,
                    sdk_mapping_profile=str(Path(s['profile']) / 'mapping.json'),
                    source_optimized_fit=str(a.source.resolve()), approved_for_training=False)
    f = json.loads((fit / 'summary.json').read_text()); f.update(metadata)
    (fit / 'summary.json').write_text(json.dumps(f, indent=2))
    fusion = c['baseline_fit'].parent / 'nokov_observations_cam01'
    if not (fusion / 'accepted.jsonl').is_file(): raise FileNotFoundError(fusion)
    dataset = output / 'merged'
    run('export', PROJECT / 'scripts/export_native_wilor_training_dataset.py', '--fusion', fusion, '--mano-fit', fit,
        '--dataset', c['dataset'], '--reference-camera', 'camera0', '--cameras', 'camera0', 'camera1', 'camera2', 'camera3',
        '--output', dataset, '--image-space', 'native_raw', '--view-filter', 'complete21', '--sample-stride', 1, '--require-both-hands')
    ds = json.loads((dataset / 'summary.json').read_text()); ds.update(metadata)
    ds.update(supervision_source='nokov24_surface_fit_with_optional_calibrated_sdk_directions',
              fit_error_point_semantics='surface_marker_centres', motion_fitting_variant=s['variant'])
    (dataset / 'summary.json').write_text(json.dumps(ds, indent=2))
    rows = [json.loads(line) for line in (dataset / 'index.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        row.update(metadata, supervision_source='nokov24_surface_fit_with_optional_calibrated_sdk_directions', motion_fitting_variant=s['variant'])
    (dataset / 'index.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    run('check', PROJECT / 'scripts/check_wilor_training_dataset.py', dataset,
        '--mano-source', PROJECT / 'third_party/MANO', '--mano-model-dir', PROJECT / 'models/mano', '--device', c['device'])
    run('viewer', ROOT / 'label_export/pipeline/calibration/marker_placement/build_label_viewer.py', '--dataset', dataset, '--output', output / 'viewer', '--standalone', '--initial-sync', 632)
    print(json.dumps(dict(dataset=str(dataset), images=ds['image_count'], labels=ds['label_count'])), flush=True)


if __name__ == '__main__': main()
