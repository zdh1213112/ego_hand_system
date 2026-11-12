#!/usr/bin/env python3
"""Recover original detector boxes from completed dual-hypothesis predictions.

Only detection fields are exported: no pose, keypoints, mesh or camera estimate
is copied. Counts must match the original detector's completed summary, so a
pose adapter that dropped an entire detection cannot silently seed a new run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def detection_row(record, sync, camera, image_size):
    index = int(sync['sync_index'])
    if int(record['sync_index']) != index:
        raise ValueError(f'{camera}: unexpected sync index, expected {index}')
    if int(record['source_frame_index']) != int(sync[f'{camera}_frame_index']):
        raise ValueError(f'{camera}/{index}: source frame mismatch')
    if int(record['timestamp_ns']) != int(sync[f'{camera}_timestamp_ns']):
        raise ValueError(f'{camera}/{index}: timestamp mismatch')
    width, height = image_size
    boxes, hypotheses = {}, set()
    for hand in record['hands']:
        detection, hypothesis = int(hand['detection_index']), int(hand['hypothesis_index'])
        if hypothesis in hypotheses or detection < 0:
            raise ValueError(f'{camera}/{index}: duplicate hypothesis or negative detection ID')
        hypotheses.add(hypothesis)
        if hypothesis // 2 != detection or hypothesis % 2 != int(hand['is_right']):
            raise ValueError(f'{camera}/{index}: invalid dual-hypothesis mapping')
        box = np.asarray(hand['bbox_xyxy'], dtype=float)
        score, side = float(hand['confidence']), hand.get('detector_is_right')
        if (box.shape != (4,) or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError(f'{camera}/{index}: invalid detector box bounds')
        if not np.isfinite(score) or not 0 <= score <= 1 or side not in (None, 0, 1):
            raise ValueError(f'{camera}/{index}: invalid detector confidence/handedness')
        if hand.get('proposal_source') != 'detector':
            raise ValueError(f'{camera}/{index}: source is not an original detector box')
        value = {'detection_index': detection, 'bbox_xyxy': box.tolist(), 'confidence': score,
                 'detector_is_right': side, 'source': 'cached_original_detector'}
        if detection in boxes and boxes[detection] != value:
            raise ValueError(f'{camera}/{index}: duplicated detection hypotheses disagree')
        boxes[detection] = value
    if sorted(boxes) != list(range(len(boxes))):
        raise ValueError(f'{camera}/{index}: source adapter dropped a detector index')
    return {'sync_index': index, 'source_frame_index': int(record['source_frame_index']),
            'timestamp_ns': int(record['timestamp_ns']), 'boxes': [boxes[n] for n in sorted(boxes)]}


def export_cache(dataset, predictions, output):
    dataset, predictions, output = (Path(p).resolve() for p in (dataset, predictions, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    source_summary = json.loads((predictions / 'summary.json').read_text())
    if source_summary.get('stage') != 'gem_multiview_dual_hypothesis':
        raise ValueError('expected a completed GEM dual-hypothesis detector source')
    cameras = manifest['camera_ids']
    if source_summary['camera_ids'] != cameras or Path(source_summary['dataset']).resolve() != dataset:
        raise ValueError('source predictions do not identify the requested dataset/cameras')
    with (dataset / 'multiview_frames.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != source_summary['frame_count']:
        raise ValueError('source predictions do not cover the whole synchronized dataset')
    if len({int(r['sync_index']) for r in rows}) != len(rows):
        raise ValueError('duplicate synchronized frame index')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    camera_stats, inputs, outputs = {}, {}, {}
    try:
        for camera in cameras:
            path = predictions / camera / 'predictions.jsonl'
            target = temporary / camera / 'predictions.jsonl'
            target.parent.mkdir()
            count, detected_frames = 0, 0
            with path.open() as source, target.open('w') as destination:
                for sync in rows:
                    line = source.readline()
                    if not line:
                        raise ValueError(f'{camera}: incomplete source predictions')
                    record = detection_row(json.loads(line), sync, camera, manifest['image_size'])
                    count += len(record['boxes'])
                    detected_frames += bool(record['boxes'])
                    destination.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
                if any(line.strip() for line in source):
                    raise ValueError(f'{camera}: extra source frames outside the synchronization table')
            expected = source_summary['cameras'][camera]
            if (count != expected['physical_box_count'] or detected_frames != expected['detected_frame_count']
                    or len(rows) != expected['frame_count']):
                raise ValueError(f'{camera}: source pose adapter lost original detector boxes')
            inputs[camera] = fingerprint(path)
            outputs[camera] = {**fingerprint(target), 'path': str(output / camera / 'predictions.jsonl')}
            camera_stats[camera] = {'frame_count': len(rows), 'physical_box_count': count,
                                    'detected_frame_count': detected_frames,
                                    'detector_confidence': expected['detector_confidence']}
        report = {'stage': 'cached_original_multiview_detector', 'schema_version': 1,
                  'dataset': str(dataset), 'camera_ids': cameras, 'frame_count': len(rows),
                  'detector': source_summary['detector'], 'source_predictions': str(predictions),
                  'source_summary': fingerprint(predictions / 'summary.json'), 'source_prediction_files': inputs,
                  'dataset_manifest': fingerprint(dataset / 'manifest.json'),
                  'synchronization': fingerprint(dataset / 'multiview_frames.csv'),
                  'output_prediction_files': outputs, 'cameras': camera_stats,
                  'contains_pose_or_keypoints': False, 'detector_rerun': False,
                  'provenance': 'Exact original detector boxes, scores and side labels recovered from completed GEM run; no GEM pose output is used by WiLoR.'}
        (temporary / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'predictions', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_cache(args.dataset, args.predictions, args.output), indent=2))


if __name__ == '__main__':
    main()
