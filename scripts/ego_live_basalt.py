"""Historical rolling-window prototype; not the continuous live VIO entry point.

Use ego_live_basalt_stream.py (or run_ego_live_multimodal.sh) for the stateful
native estimator. This unvalidated prototype is retained for source history.

Basalt 0.1.7's local executable loads a complete EuRoC dataset at startup.
This adapter builds bounded, overlapping windows while live capture continues.
It aligns consecutive local VIO coordinate systems using shared camera poses.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from prepare_basalt_dataset import build_basalt_calibration, calibrate_imu_sample


ROOT = Path(__file__).resolve().parents[1]
LENGTH = struct.Struct('!I')


def read_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        block = connection.recv(size - len(data))
        if not block:
            raise EOFError('live camera stream disconnected')
        data.extend(block)
    return bytes(data)


def read_frame(connection):
    (length,) = LENGTH.unpack(read_exact(connection, LENGTH.size))
    if not 0 < length <= 4_000_000:
        raise ValueError('invalid live frame metadata length')
    message = json.loads(read_exact(connection, length))
    if message.get('version') != 2 or not isinstance(message.get('row'), dict):
        raise ValueError('online Basalt requires live protocol v2')
    images = {}
    for entry in message.get('images', []):
        name, size = entry['camera'], entry['size']
        if name not in ('cam_0', 'cam_1') or name in images or not 0 < size <= 32_000_000:
            raise ValueError('online Basalt requires one cam_0 and one cam_1 JPEG')
        images[name] = read_exact(connection, size)
    if set(images) != {'cam_0', 'cam_1'}:
        raise ValueError('online Basalt requires both main cameras')
    return message['row'], images


def pose_from_basalt(value):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([value[key] for key in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
    T[:3, 3] = [value[key] for key in ('px', 'py', 'pz')]
    return T


def load_yaml(path):
    return yaml.safe_load('\n'.join(line for line in Path(path).read_text().splitlines()
                                    if not line.startswith('%YAML:')))


class RollingBasalt:
    def __init__(self, camera_calibration, imu_calibration, output, *, scale=.5,
                 window_s=8., stride_s=4., min_frames=40, binary=None, max_windows=0):
        if not .1 <= scale <= 1 or window_s < 3 or not 0 < stride_s < window_s:
            raise ValueError('scale must be [0.1,1], window >=3s, and 0<stride<window')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.vio_path = self.output / 'vio.jsonl'
        if self.vio_path.exists() and self.vio_path.stat().st_size:
            raise FileExistsError(f'VIO output already contains data: {self.vio_path}')
        camera_yaml = load_yaml(camera_calibration)
        by_id = {item['id']: item for item in camera_yaml['cameras']}
        camera_yaml['cameras'] = [by_id['cam_0'], by_id['cam_1']]
        imu_yaml = load_yaml(imu_calibration)
        self.calibration, metadata = build_basalt_calibration(camera_yaml, imu_yaml, scale, True)
        self.imu0 = imu_yaml['imu0']
        self.shift_ns = int(round(imu_yaml['cam0']['timeshift_cam_imu'] * 1e9))
        self.size = tuple(metadata['original_resolution'])
        self.resolution = tuple(self.calibration['value0']['resolution'][0])
        self.config = json.loads((ROOT / 'third_party/basalt/data/euroc_config.json').read_text())
        self.binary = Path(binary or ROOT / 'third_party/basalt_runtime/bin/basalt_vio').resolve()
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        self.runtime_lib = self.binary.parent.parent / 'lib'
        self.T_imu_cam = pose_from_basalt(self.calibration['value0']['T_imu_cam'][0])
        self.window_us = int(window_s * 1e6)
        self.stride_us = int(stride_s * 1e6)
        self.min_frames = min_frames
        self.max_windows = max_windows
        self.frames = deque()
        self.imu = {}
        self.lock = threading.Lock()
        self.stop = False
        self.error = None
        self.last_window_end = None
        self.previous_poses = {}
        self.last_published_us = None
        self.windows = 0
        self.thread = threading.Thread(target=self._worker, daemon=True, name='ego-live-basalt')
        self.thread.start()

    def add(self, row, images):
        stamps = row['timestamp_us']
        stamp = int(stamps['cam_0'])
        if abs(int(stamps['cam_1']) - stamp) > 1500:
            return
        grayscale = []
        for name in ('cam_0', 'cam_1'):
            frame = cv2.imdecode(np.frombuffer(images[name], np.uint8), cv2.IMREAD_GRAYSCALE)
            if frame is None or frame.shape[::-1] != self.size:
                raise ValueError(f'invalid live Basalt {name} frame size')
            if frame.shape[::-1] != self.resolution:
                frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError('failed to encode Basalt grayscale JPEG')
            grayscale.append(encoded.tobytes())
        with self.lock:
            if self.frames and stamp <= self.frames[-1][0]:
                raise ValueError('live Basalt camera timestamps must increase')
            self.frames.append((stamp, *grayscale))
            for kind in ('accel', 'gyro'):
                for sample in row.get('imu', {}).get(kind, []):
                    imu_stamp = int(sample['timestamp_us'])
                    value = np.asarray(sample['xyz'], dtype=np.float64)
                    if value.shape == (3,) and np.isfinite(value).all():
                        self.imu.setdefault(imu_stamp, {})[kind] = value
            cutoff = stamp - max(30_000_000, 2 * self.window_us)
            while self.frames and self.frames[0][0] < cutoff:
                self.frames.popleft()
            for old in [key for key in self.imu if key < cutoff]:
                del self.imu[old]

    def _snapshot(self):
        with self.lock:
            if not self.frames:
                return None
            end = self.frames[-1][0]
            if self.last_window_end is not None and end - self.last_window_end < self.stride_us:
                return None
            first = end - self.window_us
            frames = [frame for frame in self.frames if frame[0] >= first]
            if len(frames) < self.min_frames or frames[-1][0] - frames[0][0] < self.window_us * .8:
                return None
            start_imu = frames[0][0] - 100_000
            end_imu = frames[-1][0] + 100_000
            imu = [(stamp, data['accel'].copy(), data['gyro'].copy())
                   for stamp, data in sorted(self.imu.items())
                   if start_imu <= stamp <= end_imu and 'accel' in data and 'gyro' in data]
            if len(imu) < 50 or imu[0][0] > frames[0][0] or imu[-1][0] < frames[-1][0]:
                return None
            self.last_window_end = end
            return frames, imu

    def _write_dataset(self, root, frames, imu):
        mav0 = root / 'mav0'
        cameras = [mav0 / 'cam0' / 'data', mav0 / 'cam1' / 'data']
        for path in cameras:
            path.mkdir(parents=True)
        for stamp, left, right in frames:
            name = f'{stamp * 1000 + self.shift_ns}.jpg'
            (cameras[0] / name).write_bytes(left)
            (cameras[1] / name).write_bytes(right)
        for i in range(2):
            with (mav0 / f'cam{i}' / 'data.csv').open('w') as stream:
                stream.write('#timestamp [ns],filename\n')
                for stamp, *_ in frames:
                    key = stamp * 1000 + self.shift_ns
                    stream.write(f'{key},{key}.jpg\n')
        imu_path = mav0 / 'imu0' / 'data.csv'
        imu_path.parent.mkdir(parents=True)
        with imu_path.open('w') as stream:
            stream.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                         'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
            for stamp, accel, gyro in imu:
                accel, gyro = calibrate_imu_sample(accel, gyro, self.imu0)
                stream.write(f'{stamp * 1000},' + ','.join(f'{x:.12g}' for x in (*gyro, *accel)) + '\n')
        (root / 'calibration.json').write_text(json.dumps(self.calibration))
        (root / 'vio_config.json').write_text(json.dumps(self.config))

    def _run_window(self, frames, imu):
        with tempfile.TemporaryDirectory(prefix='.basalt-window-', dir=self.output) as temporary:
            root = Path(temporary)
            self._write_dataset(root, frames, imu)
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = str(self.runtime_lib) + (':' + env['LD_LIBRARY_PATH']
                                                              if env.get('LD_LIBRARY_PATH') else '')
            command = [str(self.binary), '--dataset-path', str(root), '--dataset-type', 'euroc',
                       '--cam-calib', str(root / 'calibration.json'), '--config-path', str(root / 'vio_config.json'),
                       '--show-gui', '0', '--save-trajectory', 'euroc', '--use-imu', '1', '--num-threads', '6']
            started = time.monotonic()
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f'Basalt failed: {result.stderr[-1500:]} {result.stdout[-1500:]}')
            trajectory = root / 'trajectory.csv'
            if not trajectory.is_file():
                raise RuntimeError('Basalt did not produce trajectory.csv')
            poses = {}
            with trajectory.open() as stream:
                for line in stream:
                    if not line.strip() or line.startswith('#'):
                        continue
                    values = line.strip().split(',')
                    if len(values) != 8:
                        continue
                    key = int(values[0])
                    x, y, z, qw, qx, qy, qz = map(float, values[1:])
                    T = np.eye(4)
                    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    T[:3, 3] = [x, y, z]
                    poses[(key - self.shift_ns) // 1000] = T @ self.T_imu_cam
            print(f'Basalt window frames={len(frames)} poses={len(poses)} '
                  f'compute={time.monotonic()-started:.1f}s', flush=True)
            return poses

    def _publish(self, poses):
        common = sorted(set(poses) & set(self.previous_poses))
        if self.previous_poses and not common:
            raise RuntimeError('Basalt windows have no shared pose; cannot maintain one world frame')
        alignment = (self.previous_poses[common[len(common)//2]] @ np.linalg.inv(poses[common[len(common)//2]])
                     if common else np.eye(4))
        world = {stamp: alignment @ T for stamp, T in poses.items()}
        new = [(stamp, T) for stamp, T in sorted(world.items())
               if self.last_published_us is None or stamp > self.last_published_us]
        if not new:
            raise RuntimeError('Basalt window produced no new camera poses')
        with self.vio_path.open('a', encoding='utf-8') as stream:
            for stamp, T in new:
                stream.write(json.dumps({'timestamp_us': {'cam_0': stamp},
                                         'vio': {'T_world_cam0': T.tolist()}}, separators=(',', ':')) + '\n')
            stream.flush()
        self.previous_poses = world
        self.last_published_us = new[-1][0]
        self.windows += 1
        print(f'Published {len(new)} VIO poses through cam0_us={self.last_published_us}', flush=True)

    def _worker(self):
        while not self.stop:
            snapshot = self._snapshot()
            if snapshot is None:
                time.sleep(.1)
                continue
            try:
                poses = self._run_window(*snapshot)
                self._publish(poses)
                if self.max_windows and self.windows >= self.max_windows:
                    self.stop = True
            except Exception as error:
                self.error = error
                self.stop = True

    def close(self):
        self.stop = True
        self.thread.join(timeout=2)
        if self.error:
            raise self.error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, default=Path('/tmp/ego-basalt-live.sock'))
    parser.add_argument('--camera-calibration', required=True, type=Path)
    parser.add_argument('--imu-calibration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scale', type=float, default=.5)
    parser.add_argument('--window-s', type=float, default=8.)
    parser.add_argument('--stride-s', type=float, default=4.)
    parser.add_argument('--max-windows', type=int, default=0)
    args = parser.parse_args(argv)
    path = args.socket.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    worker = RollingBasalt(args.camera_calibration, args.imu_calibration, args.output,
                           scale=args.scale, window_s=args.window_s, stride_s=args.stride_s,
                           max_windows=args.max_windows)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen(1)
        listener.settimeout(.5)
        print(f'Waiting for WiLoR capture packets at {path}', flush=True)
        try:
            while not worker.stop:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    print('Live Basalt capture connected', flush=True)
                    while not worker.stop:
                        try:
                            row, images = read_frame(connection)
                        except EOFError:
                            break
                        worker.add(row, images)
                print('Live Basalt capture disconnected', flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            worker.close()
            path.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
