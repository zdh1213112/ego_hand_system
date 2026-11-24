# Source work/nov16_pipeline_20261116_230405/prepare_mcap_group.py, SHA-256 ec9f1a096fea9d85a579126a5b721dd38aeba0774766ce8c7f5cc4d62c445cb8
"""Lossless MCAP-to-existing-label-entry bridge; source recordings are read only."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'ego_hand_system/scripts'))
from ego_hand_inference.ego_data.h264 import H264MatroskaWriter
from ego_hand_inference.ego_data.calibration import cameras_from_ego_yaml
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    group, output = args.group.resolve(), args.output.resolve()
    ego = next(group.glob('EGO*'))
    session = next(group.glob('session*'))
    mcap = next(ego.glob('part0001/*.mcap'))
    meta_file = next(ego.glob('meta_*.json'))
    camera_calibration = next(ego.glob('*_calibration_camera.yaml'))
    stat = mcap.stat()
    identity = {'path':str(mcap), 'size_bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}
    if (output / 'conversion.json').is_file():
        existing = json.loads((output / 'conversion.json').read_text())
        if existing['source_mcap'] != identity:
            raise RuntimeError('Source MCAP changed since conversion')
        print('[bridge] reuse completed conversion ' + str(output), flush=True)
        return
    if output.exists():
        raise RuntimeError('Incomplete conversion retained; use a new output')
    output.mkdir(parents=True)
    (output / session.name).symlink_to(session, target_is_directory=True)
    destination = output / ego.name
    part = destination / 'part0001'
    part.mkdir(parents=True)
    for source in [meta_file, camera_calibration, *ego.glob('*_calibration_imu.yaml')]:
        (destination / source.name).symlink_to(source)
    calibration = cameras_from_ego_yaml(camera_calibration)
    streams = {'left':('camera0','cam_0'), 'right':('camera1','cam_1'),
               'right_side':('camera2','cam_2'), 'left_side':('camera3','cam_3')}
    topics = {'/camera/' + name:name for name in streams}
    writers = {}
    imu_count, angular_count = 0, 0
    imu_path = part / f'{ego.name}_imu_part0001.csv'
    started = time.time()
    try:
        for stream, (camera, cal_id) in streams.items():
            path = part / f'{ego.name}_camera_{stream}_part0001.mkv'
            writers[stream] = H264MatroskaWriter(camera, path, tuple(calibration[cal_id].image_size))
        factory = DecoderFactory()
        with imu_path.open('w', newline='') as imu_file, mcap.open('rb') as source:
            imu = csv.DictWriter(imu_file, fieldnames=['timestamp_us','x','y','z','type'])
            imu.writeheader()
            reader = make_reader(source, validate_crcs=True)
            for schema, channel, message in reader.iter_messages(topics=[*topics, '/imu']):
                decoded = factory.decoder_for(channel.message_encoding, schema)(message.data)
                timestamp = int(message.log_time)
                if channel.topic in topics:
                    if decoded.format.lower() != 'h264':
                        raise ValueError('Expected original H264 video, got ' + decoded.format)
                    stream = topics[channel.topic]
                    writers[stream].write(bytes(decoded.data), timestamp)
                    if stream == 'left' and writers[stream].packet_count % 2000 == 0:
                        print(f'[bridge] {group.name}: {writers[stream].packet_count} camera0 frames', flush=True)
                else:
                    stamp = decoded.timestamp.seconds*1_000_000_000 + decoded.timestamp.nanos
                    if abs(stamp-timestamp) > 1000:
                        raise ValueError('IMU timestamp and MCAP video clock disagree')
                    for field, kind in [('angular_velocity','gyro'),('linear_acceleration','accel')]:
                        value = getattr(decoded,field)
                        xyz = [float(value.x),float(value.y),float(value.z)]
                        if not all(math.isfinite(x) for x in xyz):
                            continue
                        imu.writerow(dict(timestamp_us=timestamp//1000, x=xyz[0], y=xyz[1], z=xyz[2], type=kind))
                        imu_count += 1
                        angular_count += kind == 'gyro'
    finally:
        for writer in writers.values():
            writer.close()
    if angular_count < 100:
        raise ValueError('Insufficient gyro data for EGO/NOKOV synchronization')
    stream_stats = {}
    for stream, writer in writers.items():
        camera, cal_id = streams[stream]
        decoded = writer.validate(tuple(calibration[cal_id].image_size))
        mp4 = part / f'{ego.name}_camera_{stream}_part0001.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(writer.path),'-map','0:v:0',
                        '-c:v','copy','-an','-movflags','+faststart',str(mp4)], check=True)
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
             '-show_entries','stream=nb_frames,width,height','-of','json',str(mp4)],text=True))['streams'][0]
        if int(info['nb_frames']) != decoded or [info['width'],info['height']] != list(calibration[cal_id].image_size):
            raise ValueError('MP4 remux count or resolution changed')
        pts = part / f'{ego.name}_camera_{stream}_part0001_pts.csv'
        with pts.open('w',newline='') as dest:
            csv_writer = csv.DictWriter(dest,fieldnames=list(writer.rows[0]))
            csv_writer.writeheader();csv_writer.writerows(writer.rows)
        stream_stats[camera] = writer.statistics(decoded)
        writer.path.unlink()  # own intermediate MKV; validated MP4 retains the same H264 bitstream
        print(f'[bridge] {camera}: validated {decoded} frames, lossless MP4 complete',flush=True)
    final_stat = mcap.stat()
    if (final_stat.st_size,final_stat.st_mtime_ns) != (stat.st_size,stat.st_mtime_ns):
        raise RuntimeError('Source changed during conversion')
    meta = json.loads(meta_file.read_text())
    report = dict(status='complete',source_group=str(group),source_mcap=identity,
        source_recording_status=meta['recording']['status'],source_failure_message=meta['recording'].get('failure_message'),
        output=str(output),video_transcoding=False,original_timestamps_preserved=True,
        imu_rows=imu_count,gyro_rows=angular_count,cameras=stream_stats,
        unequal_camera_counts_policy='existing normalizer uses timestamp intersection; unmatched frames are excluded',
        source_files_modified=False,seconds=time.time()-started)
    write_json(output/'conversion.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    # The CPU conversion queue can run ahead while a separate process fits
    # labels. A per-group lock prevents a duplicate conversion or partial reuse.
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ('.' + output.name + '.prepare.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
