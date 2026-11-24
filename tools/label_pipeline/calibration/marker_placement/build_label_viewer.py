#!/usr/bin/env python3
"""Build a local, offline original-image / exported-MANO inspection viewer."""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--initial-sync', type=int, help='Open the nearest available synchronized frame')
    ap.add_argument('--standalone', action='store_true', help='Dataset inspector without the separate marker placement tool')
    ap.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3] / 'ego_hand_system')
    args = ap.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output == dataset or dataset in output.parents and output.name in {'images','labels'}:
        raise ValueError('Viewer output must be separate from dataset root, images and labels')
    report_path = output / 'build_report.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('dataset') != str(dataset):
        raise ValueError('Viewer output belongs to a different dataset; choose a new --output')
    summary_path = dataset / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    sys.path.insert(0, str(args.project / 'scripts'))
    from wilor_label_projection import project_label_points
    from PIL import Image
    rows = [json.loads(x) for x in (dataset / 'index.jsonl').read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError('Dataset index.jsonl is empty')
    with (args.project / 'models/mano/MANO_RIGHT.pkl').open('rb') as f:
        faces = np.asarray(pickle.load(f, encoding='latin1')['f'], dtype=int).tolist()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    marker_link = output / 'marker_placement'
    if not args.standalone and not marker_link.exists():
        marker_link.symlink_to(Path(__file__).resolve().parent, target_is_directory=True)
    frames = {}
    max_error = 0.0
    for row in rows:
        key = (str(row.get('source_dataset_id', '')), str(row['camera']), int(row['sync_index']))
        sample = np.load(dataset / row['label'], allow_pickle=True).item()
        if sample.get('image_space') != 'native_raw' or sample.get('camera_model') != 'KB':
            raise ValueError('This viewer build requires original-image KB labels')
        xyz = np.asarray(sample['vertices'], dtype=float) + np.asarray(sample['trans'], dtype=float)
        uv = project_label_points(xyz, sample)
        stored = sample['joints_2d']
        if hasattr(stored, 'detach'):
            stored = stored.detach().cpu().numpy()
        error = float(np.max(np.abs(uv - np.asarray(stored))))
        if not np.isfinite(error) or error > 0.1:
            raise ValueError(f"Exported mesh projection mismatch: {row['label']}: {error} px")
        max_error = max(max_error, error)
        joints = project_label_points(np.asarray(sample['joints_3d']) + sample['trans'], sample)
        if key not in frames:
            source = (dataset / row['image']).resolve()
            dest = output / 'images' / f'{len(frames):06d}{source.suffix}'
            if dest.is_symlink() and dest.resolve() != source:
                raise ValueError(f'Existing image link points elsewhere: {dest}')
            if not dest.exists():
                dest.symlink_to(source)
            with Image.open(source) as image:
                size = list(image.size)
            frames[key] = dict(camera=row['camera'], sync=row['sync_index'], source=key[0],
                               source_name=row.get('source_experiment', ''), source_frame=row.get('source_frame_index'),
                               timestamp=row.get('reference_timestamp_ns'), image=str(dest.relative_to(output)),
                               size=size, hands=[])
        frames[key]['hands'].append(dict(side=int(row.get('physical_side', row['side'])),
            vertices=np.round(np.c_[uv, xyz[:, 2]], 5).tolist(), joints=np.round(joints, 4).tolist(),
            bbox=np.asarray(sample['bbox']).tolist(), label=str(dataset / row['label']),
            K=np.asarray(sample['K']).tolist(), distortion=np.asarray(sample['distortion']).tolist(),
            fit_median_px=row.get('fit_reprojection_median_px')))
    data = dict(dataset=str(dataset), projection='native_raw / KB', faces=faces,
                initial_sync=args.initial_sync, approved_for_training=summary.get('approved_for_training'),
                personal_marker24_optimization_enabled=summary.get('personal_marker24_optimization_enabled', False),
                frames=sorted(frames.values(), key=lambda f:(f['source'],f['sync'],f['camera'])),
                max_projection_difference_px=max_error, label_count=len(rows))
    (output / 'data.js').write_text('window.LABEL_DATA=' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
    html = Path(__file__).with_name('label_viewer.html').read_text()
    html = html.replace("resize();selectFrames();", "resize();selectFrames(D.initial_sync);")
    extra = "<script>for(const option of $('source').options){const f=D.frames.find(f=>f.source===option.value);option.textContent=f?.source_name||option.value||'当前录制'};const banner=document.createElement('div');banner.className='bar';banner.id='datasetBanner';banner.textContent='当前数据集：'+D.dataset+' · '+D.frames.length+' 张原图 / '+D.label_count+' 份手标签'+(D.approved_for_training===false?' · 实验标签，贴合质量尚未验收':'');document.querySelector('.views').before(banner);document.title='标签数据集查看器';</script>"
    if args.standalone:
        extra+="<script>$('markerLink')?.remove();</script>"
    (output / 'index.html').write_text(html.replace('</html>', extra+'</html>'))
    report = dict(dataset=str(dataset), frames=len(frames), labels=len(rows),
                  max_projection_difference_px=max_error, viewer=str(output / 'index.html'))
    (output / 'build_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
