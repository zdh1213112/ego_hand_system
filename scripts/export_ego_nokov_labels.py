#!/usr/bin/env python3
"""One command: raw EGO/NOKOV recording to native dual-hand labels and videos."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from personal_marker24_config import load_personal_config, load_joint_refit_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'
DEFAULT_SEED = ROOT.parent/'work/20261109/alignment_172920/cam0_marker_calibrated_right/alignment.json'


def resolve_sources(recording=None, ego=None, nokov=None):
    if recording:
        if ego or nokov:
            raise ValueError('--recording cannot be combined with --ego/--nokov')
        recording = Path(recording).expanduser().resolve()
        egos = sorted(p for p in recording.glob('EGO*') if p.is_dir())
        nokovs = sorted(p for p in recording.glob('session*/nokov') if p.is_dir())
        if len(egos) != 1 or len(nokovs) != 1:
            raise ValueError('Expected one EGO* directory and one session*/nokov; use --ego and --nokov for other layouts')
        return egos[0], nokovs[0], recording.name
    if not ego or not nokov:
        raise ValueError('Provide --recording, or both --ego and --nokov')
    return Path(ego).expanduser().resolve(), Path(nokov).expanduser().resolve(), Path(ego).name


def raw_links(ego):
    if not (ego/'part0001').is_dir() or (ego/'part0002').exists():
        raise ValueError('This entry supports a single part0001 recording')
    links = {}
    for stream in ['left', 'right', 'right_side', 'left_side']:
        for suffix in ['.mp4', '_pts.csv']:
            matches = list((ego/'part0001').glob(f'*_camera_{stream}_part0001{suffix}'))
            if len(matches) != 1:
                raise ValueError(f'Expected one {stream}{suffix}: {matches}')
            links[f'{ego.name}_camera_{stream}{suffix}'] = matches[0]
    for pattern in ['*_calibration_camera.yaml', 'meta_*.json']:
        matches = list(ego.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f'Expected one {pattern}: {matches}')
        links[matches[0].name] = matches[0]
    imu = list((ego/'part0001').glob('*_imu_part0001.csv'))
    if len(imu) != 1:
        raise ValueError('Expected one IMU CSV')
    return links, imu[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, help='Pair directory containing EGO* and session*/nokov')
    p.add_argument('--ego', type=Path, help='EGO directory if stored separately')
    p.add_argument('--nokov', type=Path, help='NOKOV directory if stored separately')
    p.add_argument('--output', type=Path, help='New output directory; defaults to work/<recording>_<timestamp>')
    p.add_argument('--seed-alignment', type=Path, default=DEFAULT_SEED, help='Initial mount calibration; update after device remount')
    p.add_argument('--observations-only', action='store_true', help='Run dense RGB/KB consensus audit and videos, then stop before MANO')
    p.add_argument('--optimization-config',type=Path,help='Opt-in personal shape/24-marker/native-contour JSON; disabled by default')
    p.add_argument('--joint', action='store_true', help='Experimental MANO RGB refinement; default is baseline')
    p.add_argument('--marker-offset-mm', type=float, choices=[0, 2], default=0)
    p.add_argument('--sample-stride', type=int, default=8)
    p.add_argument('--source-fps', type=float, default=30)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print steps without writing or training')
    p.add_argument('--resume', action='store_true', help='Continue an interrupted run in --output; finished stages are skipped')
    a = p.parse_args()
    try:
        personal = load_personal_config(a.optimization_config)
        refit = load_joint_refit_config(a.optimization_config)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    if personal['enabled'] and (a.joint or a.observations_only):
        p.error('Personal marker24 optimization cannot be combined with --joint or --observations-only')
    if a.observations_only and a.joint:
        p.error("--observations-only cannot be combined with --joint")
    if refit['enabled'] and (a.joint or a.observations_only or personal['enabled']):
        p.error('joint_refit cannot be combined with --joint, --observations-only or personal_marker24')
    if a.resume and not a.output:
        p.error('--resume requires the --output directory of the interrupted run')
    try:
        ego, nokov, name = resolve_sources(a.recording, a.ego, a.nokov)
        links, imu = raw_links(ego)
        seed = a.seed_alignment.expanduser().resolve()
        if not (nokov/'nokov_rigid_bodies.csv').is_file():
            raise FileNotFoundError(nokov/'nokov_rigid_bodies.csv')
        if not seed.is_file():
            raise FileNotFoundError(seed)
        if a.sample_stride < 1 or not 0 < a.source_fps < 1000:
            raise ValueError('sample-stride and source-fps must be positive')
        out = a.output.expanduser().resolve() if a.output else ROOT.parent/'work'/f'{name}_native_{datetime.now():%Y%m%d_%H%M%S}'
        if out.exists() and not a.resume:
            raise FileExistsError(f'Use a new output directory, or pass --resume to continue it: {out}')
        if a.resume and not (out/'settings.json').is_file():
            raise FileNotFoundError(f'--resume: no previous run in {out}')
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        p.error(str(e))
    processed = out/'processed';dataset = processed/'ego/normalized_multiview'
    alignment = out/'fourview_alignment';base = out/'baseline';joint = out/'joint'
    env = os.environ.copy();env.pop('PYTHONPATH', None);env['PYTHONNOUSERSITE'] = '1'
    commands = []
    done_path = out/'completed_stages.json'
    done = set(json.loads(done_path.read_text())) if a.resume and done_path.is_file() else set()
    def run(stage, script, *args, clean=None):
        cmd = [sys.executable, '-B', str(script), *map(str, args)]
        commands.append(dict(stage=stage, command=cmd))
        if stage in done:
            print(f'[{stage}] finished earlier, skipped', flush=True)
            return
        print(f'[{stage}] '+ (' '.join(cmd) if a.dry_run else str(out/'logs'/f'{stage}.log')), flush=True)
        if a.dry_run:
            return
        if a.resume and clean is not None and Path(clean).exists():
            shutil.rmtree(clean)  # partial output of an interrupted stage inside this run
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with (out/'logs'/f'{stage}.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} failed; inspect {out / "logs" / (stage+".log")}; fix and rerun with --resume --output {out}')
        done.add(stage); done_path.write_text(json.dumps(sorted(done), indent=2)+'\n')
    config = a.optimization_config.expanduser().resolve() if a.optimization_config else None
    settings = dict(ego=str(ego), nokov=str(nokov), seed_alignment=str(seed), joint=a.joint, marker_offset_mm=a.marker_offset_mm if a.joint else None,
                    sample_stride=a.sample_stride, source_fps=a.source_fps, optimization_config=str(config) if config else None,
                    optimization_config_content=json.loads(config.read_text()) if config else None,
                    mode='joint_refit' if refit['enabled'] else 'personal_marker24' if personal['enabled'] else 'joint' if a.joint else 'baseline')
    if a.resume:
        # Runs written before these fields existed were baseline runs without a config.
        previous = dict(optimization_config=None, optimization_config_content=None, mode='baseline') | json.loads((out/'settings.json').read_text())
        changed = [k for k in settings if k in previous and previous[k] != settings[k]]
        if changed:
            p.error(f'--resume: settings differ from the interrupted run ({", ".join(changed)}); use a new --output')
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=a.resume);(out/'logs').mkdir(exist_ok=a.resume);target = processed/'ego_input';target.mkdir(parents=True, exist_ok=a.resume)
        for name, source in links.items():
            if not (target/name).is_symlink(): (target/name).symlink_to(source.resolve())
        (out/'settings.json').write_text(json.dumps(settings, indent=2, ensure_ascii=False)+'\n')
    run('normalize', SCRIPTS/'normalize_orbbec_ego4_session.py', '--session', processed/'ego_input', '--output', dataset, clean=dataset)
    run('synchronize', ROOT.parent/'nokov_ego_hand_validation/tools/synchronize_ego_imu_nokov.py', '--ego-imu-csv', imu, '--nokov-csv', nokov/'nokov_rigid_bodies.csv', '--rigid-body', 'orbbec_ego4', '--nokov-time-field', 'device_timestamp_raw', '--nokov-time-scale', .001, '--output-dir', processed/'alignment/synchronization', '--no-plot')
    run('align', SCRIPTS/'align_native_dual_nokov.py', '--dataset', dataset, '--nokov', nokov, '--sync', processed/'alignment/synchronization/imu_nokov_sync.json', '--seed-alignment', seed, '--output', processed/'alignment')
    run('fourview_alignment', SCRIPTS/'native_fourview_alignment.py', '--dataset', dataset, '--aligned-nokov', processed/'alignment/aligned_nokov21.npz', '--seed-alignment', processed/'alignment/dual_alignment.json', '--max-calibration-frames', 180, '--output', alignment, clean=alignment)
    if a.observations_only:
        run('rgb_observations', SCRIPTS/'audit_native_multiview_observations.py', '--dataset', dataset, '--aligned-nokov', alignment/'aligned_nokov21.npz', '--output', out/'rgb_observations', '--preview-stride', a.sample_stride)
        print(f'Observation report and videos: {out / "rgb_observations"}', flush=True)
        return
    common = ['--processed-session', processed, '--aligned-nokov', alignment/'aligned_nokov21.npz', '--hand-side', 'auto', '--cameras', 'camera0', 'camera1', 'camera2', 'camera3', '--sample-stride', a.sample_stride, '--max-samples', 0, '--device', a.device]
    run('baseline', SCRIPTS/'run_nokov_native_labels.py', *common, '--output', base, clean=base)
    def videos(label_root, stage):
        run(stage, SCRIPTS/'render_wilor_training_videos.py', '--dataset', label_root/'export/merged', '--output', label_root/'export/videos', '--fps', a.source_fps/a.sample_stride, '--workers', 2, '--video-codec', 'auto', clean=label_root/'export/videos')
    videos(base, 'baseline_videos')
    if a.joint:
        run('attachments', SCRIPTS/'prepare_multiview_mano_bundle.py', '--input', base/'mano_input_cam01.npz', '--baseline-fit', base/'mano_fit_full_cam01', '--rgb-correspondences', alignment/'rgb_correspondences.json', '--rgb-alignment', alignment/'fourview_alignment.json', '--marker-diameter-mm', 4, '--output', out/'constraints_source_2mm.npz')
        if not a.dry_run:
            import numpy as np
            with np.load(out/'constraints_source_2mm.npz', allow_pickle=False) as f:
                bundle = {k:f[k].copy() for k in f.files}
            bundle['offset_m'] = np.full_like(bundle['offset_m'], a.marker_offset_mm/1000)
            np.savez_compressed(out/'constraints.npz', **bundle)
            (out/'joint_config.json').write_text(json.dumps(dict(schema_version=1,multiview_mano_optimization=dict(enabled=True,bundle=str(out/'constraints.npz'),initial_fit=str(base/'mano_fit_full_cam01'),rgb_weight=.25,sigma_px=4.,pose_iterations=200)),indent=2)+'\n')
        run('joint', SCRIPTS/'run_nokov_native_labels.py', *common, '--label-config', out/'joint_config.json', '--output', joint)
        videos(joint, 'joint_videos')
    if refit['enabled']:
        optimized = out/'optimized'
        run('joint_refit', SCRIPTS/'run_joint_refit_recording.py', '--dataset', dataset, '--alignment', alignment/'fourview_alignment.json', '--aligned21', alignment/'aligned_nokov21.npz', '--baseline', base, '--nokov', nokov, '--placement', refit['placement'], '--output', optimized, '--sample-stride', refit['sample_stride'], '--device', a.device, *(['--resume'] if a.resume or optimized.exists() else []))
        final = optimized/'dense/fit/final'
        print(f'Baseline labels: {base / "export/merged"}\n'
              f'Optimized labels (for training): {final / "export/merged"}\n'
              f'Optimized viewer: {final / "export/viewer/index.html"}\n'
              f'Frame acceptance: {optimized / "dense/fit/summary.json"}\n'
              f'Image check: {final / "export/image_evidence_check.json"}\n'
              f'Add to the training set: python wilor_training/assemble_joint_refit_trainset.py --output <trainset dir>', flush=True)
        return
    if personal['enabled']:
        run('personal_marker24', SCRIPTS/'run_personal_marker24_pipeline.py', '--dataset', dataset, '--alignment', alignment/'fourview_alignment.json', '--aligned21', alignment/'aligned_nokov21.npz', '--baseline', base, '--config', a.optimization_config.resolve(), '--output', out/'optimized', '--sample-stride', a.sample_stride, '--source-fps', a.source_fps, '--device', a.device)
        if a.dry_run:
            subprocess.run([sys.executable, '-B', str(SCRIPTS/'run_personal_marker24_pipeline.py'), '--dataset', str(dataset), '--alignment', str(alignment/'fourview_alignment.json'), '--aligned21', str(alignment/'aligned_nokov21.npz'), '--baseline', str(base), '--config', str(a.optimization_config.resolve()), '--output', str(out/'optimized'), '--sample-stride', str(a.sample_stride), '--source-fps', str(a.source_fps), '--device', a.device, '--dry-run'], check=True)
        print(f'Baseline labels: {base / "export/merged"}\nExperimental outputs: {out / "optimized"}', flush=True)
        return
    print(f'Labels: {(joint if a.joint else base)/"export/merged"}\nVideos: {(joint if a.joint else base)/"export/videos"}', flush=True)


if __name__ == '__main__':
    main()
