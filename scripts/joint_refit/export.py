#!/usr/bin/env python3
"""Export a joint_refit result through the existing native WiLoR exporter and dataset checker."""
import argparse, json, os, subprocess, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]; ROOT = PROJECT.parent / 'wilor_training'  # shared fitting tools live in wilor_training/tools
sys.path.insert(0, str(ROOT / 'tools/personal_hand_calibration'))
from context import load_context

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--context', type=Path, required=True); ap.add_argument('--source', type=Path, required=True)
    a = ap.parse_args(); c = load_context(a.context)
    s = json.loads((a.source.parent / 'summary.json').read_text())
    if not s['strategy'].startswith('joint_refit_v'): raise ValueError('Expected a joint_refit result')
    (a.source / 'summary.json').write_text(json.dumps(dict(strategy=s['strategy'], stage=a.source.name, approved_for_training=False, joint_pose_attachment_optimization_enabled=True,
                                                             view_policy='accepted_quality_silhouettes_all_cameras'), indent=1))
    output = a.source / 'export'; output.mkdir(exist_ok=False); (output / 'logs').mkdir(); commands = []
    def run(name, script, *args):
        command = [sys.executable, '-B', str(script), *map(str, args)]; row = dict(stage=name, command=command, returncode=None); commands.append(row); print(name, flush=True)
        env = dict(os.environ, PYTHONNOUSERSITE='1'); env.pop('PYTHONPATH', None)
        with (output / 'logs' / f'{name}.log').open('w') as f: row['returncode'] = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        (output / 'commands.json').write_text(json.dumps(commands, indent=2))
        if row['returncode']: raise RuntimeError(f'{name} failed: {output / "logs" / (name + ".log")}')
    fit = a.source / 'mano_fit'
    run('materialize', ROOT / 'tools/personal_hand_calibration/materialize_fit.py', '--context', a.context, '--source', a.source, '--output', fit)
    meta = dict(joint_refit_stage=a.source.name, shared_shape_and_attachments_recalibrated=True, per_frame_pose_free=True,
                silhouette_supervision='accepted_quality_views_only', pose_prior='mano_scan_mahalanobis', source_optimized_fit=str(a.source.resolve()), approved_for_training=False)
    f = json.loads((fit / 'summary.json').read_text()); f.update(meta); (fit / 'summary.json').write_text(json.dumps(f, indent=2))
    fusion = c['baseline_fit'].parent / 'nokov_observations_cam01'; dataset = output / 'merged'
    run('export', PROJECT / 'scripts/export_native_wilor_training_dataset.py', '--fusion', fusion, '--mano-fit', fit, '--dataset', c['dataset'], '--reference-camera', 'camera0',
        '--cameras', 'camera0', 'camera1', 'camera2', 'camera3', '--output', dataset, '--image-space', 'native_raw', '--view-filter', 'complete21', '--sample-stride', 1,
        *(['--require-both-hands'] if all((a.source / f'{h}_pilot.npz').is_file() for h in ['left', 'right']) else []))
    ds = json.loads((dataset / 'summary.json').read_text()); ds.update(meta, supervision_source='nokov24_surface_fit_with_multiview_silhouettes'); (dataset / 'summary.json').write_text(json.dumps(ds, indent=2))
    rows = [json.loads(l) for l in (dataset / 'index.jsonl').read_text().splitlines() if l.strip()]
    for r in rows: r.update(meta, supervision_source='nokov24_surface_fit_with_multiview_silhouettes')
    (dataset / 'index.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    run('check', PROJECT / 'scripts/check_wilor_training_dataset.py', dataset, '--mano-source', PROJECT / 'third_party/MANO', '--mano-model-dir', PROJECT / 'models/mano', '--device', c['device'])
    run('viewer', ROOT / 'tools/marker_placement/build_label_viewer.py', '--dataset', dataset, '--output', output / 'viewer', '--standalone', '--initial-sync', 632)
    print(json.dumps(dict(dataset=str(dataset), images=ds['image_count'], labels=ds['label_count'])), flush=True)
if __name__ == '__main__': main()
