"""Wire format, clock handling and bounded input for continuous Basalt."""
import json
from pathlib import Path
import socket
import struct
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ego_live_basalt_stream import HEADER, IMU_DTYPE, StreamingBasalt, merge_imu_rows, read_frame


def processor():
    worker=StreamingBasalt.__new__(StreamingBasalt)
    worker.original_size=(16,12)
    worker.resolution=(8,6)
    worker.shift_ns=1_801_936
    worker.imu_calibration=dict(M_acc=np.eye(3)*1.1,M_gyr=np.eye(3)*2,
                                AccBias=[.1,.2,.3],GyrBias=[.1,.1,.1])
    worker.imu={}
    worker.last_sent_imu_ns=None
    worker.first_received_us=1_000_000
    worker.initialized=False
    worker.initialization_s=0.
    worker.stale_imu=worker.startup_skipped=worker.dropped=0
    return worker


def packet(stamp=1_000_000, first=900_000, last=1_100_000):
    times=list(range(first,last+1,1000))
    row={'timestamp_us':{'cam_0':stamp,'cam_1':stamp+10},'imu':{
        'accel':[{'timestamp_us':t,'xyz':[1.,2.,9.]} for t in times],
        'gyro':[{'timestamp_us':t,'xyz':[.01,.02,.03]} for t in times]}}
    image=cv2.imencode('.jpg',np.full((12,16,3),128,np.uint8))[1].tobytes()
    return row,{'cam_0':image,'cam_1':image}


def test_native_gray_planes_and_one_time_calibration_and_clock_shift():
    worker=processor()
    row,images=packet()
    wire=worker._encode(row,images)
    header=HEADER.unpack_from(wire)
    assert header[:3]==(b'BVIO',2,0)
    assert header[3]==1_000_000*1000+worker.shift_ns
    assert header[5:7]==(48,48)
    samples=np.frombuffer(wire,dtype=IMU_DTYPE,count=header[4],offset=HEADER.size)
    assert samples['time'][0]==900_000_000
    np.testing.assert_allclose(samples['gyro'][0],[-.08,-.06,-.04])
    np.testing.assert_allclose(samples['accel'][0],[1.,2.,9.]*np.diag(worker.imu_calibration['M_acc'])-[.1,.2,.3])
    pixels=np.frombuffer(wire,dtype=np.uint8,offset=HEADER.size+header[4]*56)
    assert len(pixels)==96 and (pixels==128).all()


def test_stale_firmware_imu_is_discarded_without_inventing_offset():
    worker=processor()
    row,images=packet()
    for kind in ('accel','gyro'):
        row['imu'][kind].insert(0,{'timestamp_us':100,'xyz':[0,0,0]})
    wire=worker._encode(row,images)
    assert worker.stale_imu==2
    assert HEADER.unpack_from(wire)[3]==1_001_801_936


def test_no_current_imu_means_no_image_sent_to_estimator():
    worker=processor()
    row,images=packet(first=100,last=200)
    assert worker._encode(row,images) is None
    assert not worker.initialized


def test_imu_is_not_resent_in_subsequent_camera_packets():
    worker=processor()
    worker._encode(*packet())
    prior=worker.last_sent_imu_ns
    wire=worker._encode(*packet(stamp=1_150_000,first=1_101_000,last=1_200_000))
    header=HEADER.unpack_from(wire)
    samples=np.frombuffer(wire,dtype=IMU_DTYPE,count=header[4],offset=HEADER.size)
    assert (samples['time']>prior).all()


def test_capture_protocol_keeps_both_image_boundaries():
    row,_=packet()
    event=json.dumps({'version':2,'row':row,'images':[{'camera':'cam_0','size':3},{'camera':'cam_1','size':4}]}).encode()
    left,right=socket.socketpair()
    try:
        left.sendall(struct.pack('!I',len(event))+event+b'abc'+b'defg')
        received,images=read_frame(right)
    finally:left.close();right.close()
    assert received==row and images=={'cam_0':b'abc','cam_1':b'defg'}


def test_imu_survives_camera_queue_coalescing_and_overflow_is_explicit():
    row,_=packet()
    newer,_=packet(stamp=1_150_000,first=1_101_000,last=1_200_000)
    merged=merge_imu_rows(row,newer)
    assert len(merged['imu']['accel'])==len(row['imu']['accel'])+len(newer['imu']['accel'])
    row['imu']['accel']=[{}]*8192
    with pytest.raises(RuntimeError,match='backlog'):
        merge_imu_rows(row,newer)


def test_two_second_warmup_preserves_real_imu_for_stationary_initialization():
    worker=processor()
    worker.initialization_s=2.
    for stamp in range(1_000_000,3_000_000,33_333):
        assert worker._encode(*packet(stamp,stamp-33_333,stamp+3000)) is None
    wire=worker._encode(*packet(3_033_333,3_000_000,3_036_333))
    header=HEADER.unpack_from(wire)
    imu=np.frombuffer(wire,dtype=IMU_DTYPE,count=header[4],offset=HEADER.size)
    assert imu['time'][-1]-imu['time'][0] > 2_000_000_000
    assert worker.initialized
