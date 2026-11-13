#!/usr/bin/env python3
"""Normalize a calibrated Orbbec EGO four-camera MCAP recording."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cameras', nargs='+', default=[f'camera{i}' for i in range(4)])
    parser.add_argument('--reference-camera', default='camera0')
    parser.add_argument('--max-delta-us', type=int, default=1500)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--allow-truncated', action='store_true')
    parser.add_argument('--calibration', '--calibration-dir', type=Path,
                        help='Orbbec YAML or directory with camera0..3.json; discovered from the recording by default')
    return parser.parse_args()


def synchronize_rows(
    frame_rows: dict[str, list[dict]], camera_ids: tuple[str, ...],
    reference_camera: str, max_delta_ns: int,
) -> list[dict]:
    """Nearest-neighbour synchronization without reusing a camera frame."""
    timestamps = {
        camera: [int(row["timestamp_ns"]) for row in frame_rows[camera]]
        for camera in camera_ids
    }
    used = {camera: set() for camera in camera_ids if camera != reference_camera}
    synchronized: list[dict] = []
    for reference_index, reference_ns in enumerate(timestamps[reference_camera]):
        selected = {reference_camera: reference_index}
        for camera in camera_ids:
            if camera == reference_camera:
                continue
            values = timestamps[camera]
            position = bisect_left(values, reference_ns)
            candidates = [index for index in (position - 1, position) if 0 <= index < len(values)]
            candidates = [index for index in candidates if index not in used[camera]]
            if not candidates:
                selected = {}
                break
            best = min(candidates, key=lambda index: abs(values[index] - reference_ns))
            if abs(values[best] - reference_ns) > max_delta_ns:
                selected = {}
                break
            selected[camera] = best
        if not selected:
            continue
        row: dict[str, int | float] = {
            "sync_index": len(synchronized),
            "reference_timestamp_ns": reference_ns,
        }
        for camera in camera_ids:
            index = selected[camera]
            timestamp_ns = timestamps[camera][index]
            row[f"{camera}_frame_index"] = index
            row[f"{camera}_timestamp_ns"] = timestamp_ns
            row[f"{camera}_delta_us"] = (timestamp_ns - reference_ns) / 1000.0
            if camera != reference_camera:
                used[camera].add(index)
        synchronized.append(row)
    return synchronized



def normalize(args):
    from normalize_orbbec_multiview_mcap import normalize as normalize_orbbec
    from ego_hand_inference.ego_data.calibration import load_orbbec_four_calibrations
    from ego_hand_inference.orbbec_io import discover_calibration, resolve_recording_input

    cameras = list(args.cameras)
    if len(cameras) < 2 or len(set(cameras)) != len(cameras) or not set(cameras) <= {f'camera{i}' for i in range(4)}:
        raise ValueError('Orbbec EGO4 requires two to four distinct camera0..camera3 IDs')
    if args.reference_camera not in cameras:
        raise ValueError('reference camera must be selected')
    source, _ = resolve_recording_input(args.input, 'mcap')
    calibration_source = getattr(args, 'calibration', None) or discover_calibration(source)
    calibration_source = Path(calibration_source).expanduser().resolve()
    calibration = load_orbbec_four_calibrations(calibration_source)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.orbbec_calibration_', dir=output.parent) as directory:
        directory = Path(directory)
        for camera, value in calibration.items():
            value.save(directory / f'{camera}.json')
        result = normalize_orbbec(SimpleNamespace(
            input=source, output=output, cameras=cameras, camera_topic=[],
            reference_camera=args.reference_camera, max_delta_us=args.max_delta_us,
            max_frames=args.max_frames, width=calibration[cameras[0]].image_size[0],
            height=calibration[cameras[0]].image_size[1], calibration_dir=directory,
            allow_truncated=getattr(args, 'allow_truncated', False),
        ))
    manifest_path = result / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['calibration']['source'] = str(calibration_source)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    output = normalize(parse_args())
    print(f'Normalized Orbbec multiview dataset: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
