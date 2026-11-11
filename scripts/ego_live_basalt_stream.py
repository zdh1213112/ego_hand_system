"""Continuous Basalt stereo/IMU processing from the WiLoR capture side channel.

The camera is opened only by the capture process. A single native estimator
keeps its state for this complete session; no offline windows are restarted.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

import cv2
import numpy as np
import yaml

from prepare_basalt_dataset import build_basalt_calibration

ROOT = Path(__file__).resolve().parents[1]
LENGTH = struct.Struct('!I')
HEADER = struct.Struct('<4sHHqIIII')
IMU_DTYPE = np.dtype([('time', '<i8'), ('gyro', '<f8', (3,)), ('accel', '<f8', (3,))])
assert IMU_DTYPE.itemsize == 56


def read_exact(connection, size, stop=None):
    data = bytearray()
    while len(data) < size:
        if stop is not None and stop.is_set():
            raise EOFError('live VIO stopped')
        try:
            block = connection.recv(min(size-len(data), 1_048_576))
        except socket.timeout:
            continue
        if not block:
            raise EOFError('live capture disconnected')
        data.extend(block)
    return bytes(data)


def read_frame(connection, stop=None):
    (length,) = LENGTH.unpack(read_exact(connection, LENGTH.size, stop))
    if not 0 < length <= 4_000_000:
        raise ValueError('invalid live VIO metadata length')
    event = json.loads(read_exact(connection, length, stop))
    if event.get('version') != 2 or not isinstance(event.get('row'), dict) or event.get('depth'):
        raise ValueError('VIO requires camera/IMU protocol v2 without depth')
    images = {}
    for entry in event.get('images', []):
        name, size = entry['camera'], entry['size']
        if name not in ('cam_0','cam_1') or name in images or not isinstance(size,int) or not 0 < size <= 32_000_000:
            raise ValueError('VIO requires one cam_0 and one cam_1 JPEG')
        images[name] = read_exact(connection, size, stop)
    if set(images) != {'cam_0','cam_1'}:
        raise ValueError('VIO requires both main camera images')
    return event['row'], images


def load_yaml(path):
    return yaml.safe_load('\n'.join(line for line in Path(path).read_text().splitlines()
                                    if not line.startswith('%YAML:')))


def merge_imu_rows(older, newer):
    combined = dict(newer)
    combined['imu'] = {}
    for kind in ('accel','gyro'):
        values = older.get('imu',{}).get(kind,[]) + newer.get('imu',{}).get(kind,[])
        if len(values) > 8192:
            raise RuntimeError('IMU backlog exceeded eight seconds; restart this session')
        combined['imu'][kind] = values
    return combined


class StreamingBasalt:
    def __init__(self, camera_calibration, imu_calibration, output, *, scale=.5,
                 binary=None, threads=6, max_frames=0, initialization_s=2.):
        if not .1 <= scale <= 1 or threads < 1 or max_frames < 0 or not 0 <= initialization_s <= 5:
            raise ValueError('invalid live Basalt scale/threads/frame limit')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ('vio.jsonl','world.jsonl','backend_events.jsonl'):
            if (self.output/name).exists():
                raise FileExistsError(self.output/name)
        camera = load_yaml(camera_calibration)
        by_id = {entry['id']:entry for entry in camera['cameras']}
        camera['cameras'] = [by_id[name] for name in ('cam_0','cam_1')]
        imu = load_yaml(imu_calibration)
        calibration, metadata = build_basalt_calibration(camera, imu, scale, True)
        self.original_size = tuple(metadata['original_resolution'])
        self.resolution = tuple(metadata['output_resolution'])
        self.imu_calibration = imu['imu0']
        self.shift_ns = int(round(imu['cam0']['timeshift_cam_imu']*1e9))
        (self.output/'calibration.json').write_text(json.dumps(calibration))
        config = json.loads((ROOT/'third_party/basalt/data/euroc_config.json').read_text())
        config['config.vio_enforce_realtime'] = True
        (self.output/'vio_config.json').write_text(json.dumps(config))
        metadata.update(method='basalt_native_continuous', image_timestamp_shift_ns=self.shift_ns,
                        output_world='right-handed metric, Z up from inertial gravity',
                        input_clock='Orbbec cam_0/IMU device microseconds',
                        camera_calibration=str(camera_calibration), imu_calibration=str(imu_calibration))
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        executable = Path(binary or ROOT/'build/ego_basalt_live').resolve()
        if not executable.is_file():
            raise FileNotFoundError(f'Build ego_basalt_live first: {executable}')
        self.condition = threading.Condition()
        self.pending = deque(maxlen=1)
        self.stop = threading.Event()
        self.closing = False
        self.error = None
        self.imu = {}
        self.last_received_us = None
        self.first_received_us = None
        self.last_sent_imu_ns = None
        self.initialized = False
        self.received = self.sent = self.dropped = self.startup_skipped = self.stale_imu = 0
        self.poses = self.maps = 0
        self.max_frames = max_frames
        self.initialization_s = initialization_s
        self.last_pose_us = None
        self.arrivals = {}
        self.latencies = []
        self.backend_log = (self.output/'backend.log').open('wb')
        env = os.environ.copy()
        runtime = str(ROOT/'third_party/basalt_runtime/lib')
        env['LD_LIBRARY_PATH'] = runtime + (os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        self.process = subprocess.Popen([str(executable),str(self.output/'calibration.json'),
            str(self.output/'vio_config.json'),str(self.output/'backend_events.jsonl'),
            str(self.shift_ns),str(threads)], stdin=subprocess.PIPE,
            stdout=self.backend_log, stderr=subprocess.STDOUT, bufsize=0, env=env)
        self.sender = threading.Thread(target=self._send,daemon=True,name='basalt-input')
        self.reader = threading.Thread(target=self._receive,daemon=True,name='basalt-output')
        self.sender.start()
        self.reader.start()

    def add(self, row, images):
        stamp = int(row['timestamp_us']['cam_0'])
        with self.condition:
            if self.error:
                raise self.error
            if self.last_received_us is not None and stamp <= self.last_received_us:
                raise ValueError('camera device timestamps must increase')
            self.last_received_us = stamp
            if self.first_received_us is None:
                self.first_received_us = stamp
            self.received += 1
            self.arrivals[stamp] = time.monotonic()
            cutoff = stamp-60_000_000
            self.arrivals = {t:v for t,v in self.arrivals.items() if t>=cutoff}
            if self.pending:
                older, _ = self.pending.pop()
                row = merge_imu_rows(older,row)
                self.dropped += 1
            self.pending.append((row,images))
            self.condition.notify_all()

    def _encode(self, row, images):
        stamp = int(row['timestamp_us']['cam_0'])
        image_ns = stamp*1000+self.shift_ns
        for kind in ('accel','gyro'):
            for sample in row.get('imu',{}).get(kind,[]):
                t = int(sample['timestamp_us'])
                vector = np.asarray(sample['xyz'],dtype=np.float64)
                if vector.shape!=(3,) or not np.isfinite(vector).all():
                    raise ValueError('invalid live IMU vector')
                # Firmware may drain old IMU samples before video starts.
                # Never invent a clock offset to make stale samples fit.
                if abs(t-stamp)>500_000:
                    self.stale_imu += 1
                    continue
                if self.last_sent_imu_ns is None or t*1000>self.last_sent_imu_ns:
                    self.imu.setdefault(t,{})[kind] = vector
        complete = sorted(t for t,values in self.imu.items() if 'accel' in values and 'gyro' in values)
        retention_us = max(1_000_000, int((self.initialization_s+.6)*1e6))
        self.imu = {t:v for t,v in self.imu.items() if t>=stamp-retention_us}
        complete = [t for t in complete if t in self.imu]
        if not self.initialized and stamp-self.first_received_us < self.initialization_s*1e6:
            self.startup_skipped += 1
            return None
        if not complete or complete[-1]*1000<image_ns:
            if not self.initialized and stamp-self.first_received_us>10_000_000:
                raise RuntimeError('no current IMU coverage after ten camera seconds')
            self.startup_skipped += 1
            return None
        if not self.initialized and (len(complete)<100 or complete[0]*1000>image_ns):
            self.startup_skipped += 1
            return None
        if abs(int(row['timestamp_us']['cam_1'])-stamp)>1500:
            self.dropped += 1
            return None
        if self.last_sent_imu_ns is not None and complete[0]*1000-self.last_sent_imu_ns>100_000_000:
            raise RuntimeError('IMU stream has a gap over 100 ms; start a new VIO session')
        if len(complete)>8192:
            raise RuntimeError('too many IMU samples for one live packet')
        samples = np.empty(len(complete),dtype=IMU_DTYPE)
        samples['time'] = np.asarray(complete,dtype=np.int64)*1000
        for kind, matrix, bias in (('accel','M_acc','AccBias'),('gyro','M_gyr','GyrBias')):
            raw = np.stack([self.imu[t][kind] for t in complete])
            samples[kind] = raw @ np.asarray(self.imu_calibration[matrix]).T - self.imu_calibration[bias]
        planes = []
        for camera in ('cam_0','cam_1'):
            gray = cv2.imdecode(np.frombuffer(images[camera],np.uint8),cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape[::-1]!=self.original_size:
                raise ValueError(f'invalid live {camera} resolution')
            if gray.shape[::-1]!=self.resolution:
                gray = cv2.resize(gray,self.resolution,interpolation=cv2.INTER_AREA)
            planes.append(np.ascontiguousarray(gray,dtype=np.uint8).tobytes())
        self.last_sent_imu_ns = int(samples['time'][-1])
        self.imu = {t:v for t,v in self.imu.items() if t*1000>self.last_sent_imu_ns}
        self.initialized = True
        return HEADER.pack(b'BVIO',2,0,image_ns,len(samples),len(planes[0]),len(planes[1]),0)+samples.tobytes()+planes[0]+planes[1]

    def _send(self):
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.closing:
                        self.condition.wait(.2)
                        if self.process.poll() is not None:
                            raise RuntimeError('Basalt backend exited; inspect backend.log')
                    if self.closing and not self.pending:
                        break
                    row,images = self.pending.pop()
                data = self._encode(row,images)
                if data is None:
                    continue
                if self.sent == 0:
                    (self.output/'first_packet.bin').write_bytes(data)
                view = memoryview(data)
                while view:
                    written = self.process.stdin.write(view)
                    if not written:
                        raise BrokenPipeError('Basalt backend input closed')
                    view = view[written:]
                self.sent += 1
                if self.max_frames and self.sent>=self.max_frames:
                    self.stop.set()
                    break
        except Exception as error:
            self.error = error
            self.stop.set()
        finally:
            try:self.process.stdin.close()
            except OSError:pass

    def _receive(self):
        try:
            path = self.output/'backend_events.jsonl'
            while not path.exists():
                if self.process.poll() is not None:
                    raise RuntimeError('Basalt failed to start; inspect backend.log')
                time.sleep(.02)
            with path.open() as source, (self.output/'vio.jsonl').open('w') as poses, (self.output/'world.jsonl').open('w') as world:
                while True:
                    position = source.tell()
                    line = source.readline()
                    if not line:
                        if self.process.poll() is not None:
                            break
                        time.sleep(.01)
                        continue
                    if not line.endswith('\n'):
                        source.seek(position)
                        time.sleep(.01)
                        continue
                    row = json.loads(line)
                    stamp = int(row['timestamp_us']['cam_0'])
                    if 'vio' in row:
                        T = np.asarray(row['vio']['T_world_cam0'])
                        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
                            raise ValueError('Basalt emitted an invalid camera pose')
                        if self.last_pose_us is not None and stamp<=self.last_pose_us:
                            raise ValueError('Basalt camera pose timestamps must increase')
                        self.last_pose_us = stamp
                        with self.condition:
                            arrival = self.arrivals.get(stamp)
                        if arrival is not None:
                            latency = 1000*(time.monotonic()-arrival)
                            self.latencies.append(latency)
                            row['vio']['adapter_latency_ms'] = latency
                        poses.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                        poses.flush()
                        self.poses += 1
                        if self.poses%100==0:
                            print(f'Basalt poses={self.poses} input={self.sent} source_cam0_us={stamp}',flush=True)
                    elif 'map_points_world_m' in row:
                        points = np.asarray(row['map_points_world_m'])
                        if points.size and (points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()):
                            raise ValueError('Basalt emitted an invalid sparse map')
                        self.maps += 1
                    else:
                        raise ValueError('unknown Basalt output event')
                    world.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    world.flush()
        except Exception as error:
            self.error = error
            self.stop.set()

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.sender.join(timeout=10)
        if self.sender.is_alive():
            self.process.terminate()
            self.sender.join(timeout=3)
            self.error = self.error or RuntimeError('Basalt input shutdown timed out')
        try:
            status = self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            status = self.process.wait(timeout=3)
            self.error = self.error or RuntimeError('Basalt estimator shutdown timed out')
        self.reader.join(timeout=5)
        self.backend_log.close()
        summary = dict(method='basalt_native_continuous',received_camera_packets=self.received,
            submitted_camera_frames=self.sent,dropped_camera_frames=self.dropped,
            initialization_window_s=self.initialization_s,
            startup_or_uncovered_frames=self.startup_skipped,discarded_stale_imu_samples=self.stale_imu,
            pose_updates=self.poses,sparse_map_updates=self.maps,last_pose_cam0_us=self.last_pose_us,
            backend_returncode=status,adapter_latency_median_ms=float(np.median(self.latencies)) if self.latencies else None,
            adapter_latency_p95_ms=float(np.percentile(self.latencies,95)) if self.latencies else None,
            error=str(self.error) if self.error else None)
        initialization = self.output/'backend_events.jsonl.initialization.json'
        if initialization.exists():
            summary['initialization'] = json.loads(initialization.read_text())
        (self.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
        if self.error:
            raise self.error
        if status:
            raise RuntimeError(f'Basalt exited with {status}; inspect {self.output/"backend.log"}')
        if self.sent and not self.poses:
            raise RuntimeError('Basalt processed images but produced no valid VIO poses')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket',type=Path,default=Path('/tmp/ego-basalt-live.sock'))
    parser.add_argument('--camera-calibration',type=Path,required=True)
    parser.add_argument('--imu-calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scale',type=float,default=.5)
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--initialization-s',type=float,default=2.,help='IMU warm-up before the first image; hold camera still')
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    path = args.socket.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    worker = StreamingBasalt(args.camera_calibration,args.imu_calibration,args.output,
                             scale=args.scale,threads=args.threads,max_frames=args.max_frames,binary=args.binary,
                             initialization_s=args.initialization_s)
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(1)
            listener.settimeout(.2)
            print(f'Continuous Basalt waiting for capture packets: {path}',flush=True)
            connection = None
            while connection is None and not worker.stop.is_set():
                try:connection,_ = listener.accept()
                except socket.timeout:continue
            if connection is not None:
                with connection:
                    connection.settimeout(.2)
                    while not worker.stop.is_set():
                        try:row,images = read_frame(connection,worker.stop)
                        except EOFError:break
                        worker.add(row,images)
    except KeyboardInterrupt:
        pass
    finally:
        try:worker.close()
        finally:path.unlink(missing_ok=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
