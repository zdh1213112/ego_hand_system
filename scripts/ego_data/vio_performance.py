"""Read measured VIO performance from existing native outputs; no reruns.

Basalt stores Timer.elapsed() in seconds, even when its console format is
"ms".  stats_sums.measure is one visual-backend update per input image group;
its scope excludes optical flow and the IMU preintegration done before
measure().  Whole-process wall time is a separate throughput measurement.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import struct

import numpy as np
from ego_data.time_sync import require_zero_camera_offset


class _UBJSONReader:
    """Small dependency-free reader for nlohmann::json native stats files."""

    _numbers={b'i':('>b',1),b'U':('>B',1),b'I':('>h',2),b'l':('>i',4),
              b'L':('>q',8),b'd':('>f',4),b'D':('>d',8)}

    def __init__(self,data):
        self.data=data;self.index=0

    def take(self,size=1):
        result=self.data[self.index:self.index+size]
        if len(result)!=size:raise ValueError('truncated UBJSON stats')
        self.index+=size
        return result

    def peek(self):
        return self.data[self.index:self.index+1]

    def length(self):
        number=self.value()
        if not isinstance(number,int) or isinstance(number,bool) or number<0:
            raise ValueError('invalid UBJSON length')
        return number

    def value(self,marker=None):
        marker=self.take() if marker is None else marker
        while marker==b'N':marker=self.take()
        if marker in self._numbers:
            fmt,size=self._numbers[marker]
            return struct.unpack(fmt,self.take(size))[0]
        if marker==b'Z':return None
        if marker==b'T':return True
        if marker==b'F':return False
        if marker==b'C':return self.take().decode('utf-8')
        if marker==b'S':return self.take(self.length()).decode('utf-8')
        if marker not in [b'[',b'{']:
            raise ValueError(f'unsupported UBJSON stats marker: {marker!r}')
        closing=b']' if marker==b'[' else b'}';element_type=None;count=None
        if self.peek()==b'$':self.take();element_type=self.take()
        if self.peek()==b'#':self.take();count=self.length()
        result=[] if marker==b'[' else {}
        n=0
        while (n<count) if count is not None else (self.peek()!=closing):
            if marker==b'[':result.append(self.value(element_type))
            else:
                key=self.take(self.length()).decode('utf-8')
                result[key]=self.value(element_type)
            n+=1
        if count is None:self.take()
        return result


def read_stats(path):
    """Read JSON or UBJSON without changing native results."""
    path=Path(path)
    if path.suffix=='.json':return json.loads(path.read_text())
    reader=_UBJSONReader(path.read_bytes());result=reader.value()
    if reader.index!=len(reader.data):raise ValueError('trailing bytes in UBJSON stats')
    if not isinstance(result,dict):raise ValueError('VIO stats must be an object')
    return result


def _load_optional_stats(directory,stem):
    for suffix in ['.json','.ubjson']:
        path=directory/(stem+suffix)
        if path.exists():return read_stats(path),path
    return {},None


def _scalar(stats,key):
    values=stats.get(key,[])
    if isinstance(values,list):values=values[0] if values else None
    if values is None or not np.isfinite(values):return None
    return float(values)


def _distribution(seconds):
    samples=np.asarray(seconds,dtype=float)
    valid=np.isfinite(samples)&(samples>=0)
    values=samples[valid]*1000
    result={'count':int(valid.sum()),'invalid_count':int((~valid).sum()),
            **{k:None for k in ['mean','median','p90','p95','p99','min','max','std','total_s']}}
    if len(values):
        result.update(mean=float(values.mean()),median=float(np.median(values)),
                      p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),
                      p99=float(np.percentile(values,99)),min=float(values.min()),
                      max=float(values.max()),std=float(values.std()),total_s=float(values.sum()/1000))
    return result


def _resources(path):
    if not path.exists():return {}
    fields={}
    for line in path.read_text().splitlines():
        stripped=line.strip()
        for prefix,key in [('User time (seconds):','user_s'),('System time (seconds):','system_s'),
            ('Percent of CPU this job got:','cpu_percent'),('Maximum resident set size (kbytes):','rss_kib'),
            ('Major (requiring I/O) page faults:','major_page_faults'),('Minor (reclaiming a frame) page faults:','minor_page_faults'),
            ('Voluntary context switches:','voluntary_context_switches'),('Involuntary context switches:','involuntary_context_switches')]:
            if stripped.startswith(prefix):
                value=stripped[len(prefix):].strip().removesuffix('%')
                try:fields[key]=float(value)
                except ValueError:pass
    return fields


def _trajectory_stamps(path):
    if not path.exists():return []
    times=[]
    with path.open(newline='') as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith('#'):continue
            try:stamp=int(row[0])
            except ValueError:
                if not times:continue
                raise ValueError('invalid native trajectory timestamp')
            times.append(stamp)
    return times


def collect_vio_performance(output:Path,case:str,repeat:int=1)->dict:
    """Collect measured per-update distributions and whole-process throughput.

    The returned dictionary contains JSON-compatible numbers/null, with explicit
    scope and file sources.  Missing measurements stay null.  Frame timestamps
    are used for recording duration only, never as runtime measurements.
    """
    if repeat<1:raise ValueError('repeat must be positive')
    output=Path(output);directory=output/'runs'/case/f'repeat_{repeat}'
    sums,sums_path=_load_optional_stats(directory,'stats_sums')
    native,native_path=_load_optional_stats(directory,'stats_vio')
    stamps=_trajectory_stamps(directory/'trajectory.csv')
    measured=sums.get('measure',[]);backend=_distribution(measured)
    processed=len(stamps) if stamps else (len(measured) if len(measured) else None)
    timing_path=directory/'vio_timing.json'
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else {}
    wall=timing.get('wall_s')
    wall=float(wall) if wall is not None and np.isfinite(wall) and wall>0 else None
    duration=(stamps[-1]-stamps[0])*1e-9 if len(stamps)>1 else None
    resources_path=directory/'vio_resource.txt';resource=_resources(resources_path)
    cpu_time=(resource['user_s']+resource['system_s']) if all(k in resource for k in ['user_s','system_s']) else None
    peak_bytes=resource.get('rss_kib')
    peak_bytes=peak_bytes*1024 if peak_bytes is not None else _scalar(native,'resident_memory_peak')
    command_path=directory/'vio_command.json'
    command=json.loads(command_path.read_text()) if command_path.exists() else []
    threads=int(command[command.index('--num-threads')+1]) if '--num-threads' in command else None
    prepared_path=output/'input_validation.json'
    prepared=json.loads(prepared_path.read_text()).get('prepared_groups') if prepared_path.exists() else None
    if prepared_path.exists():require_zero_camera_offset(json.loads(prepared_path.read_text()),str(output))
    def relative(path):return str(path.relative_to(output)) if path else None
    sample_values=[float(v)*1000 if np.isfinite(v) and v>=0 else None for v in measured]
    result={'processed_frames':processed,'expected_frames':prepared,
        'output_coverage_percent':100*len(stamps)/prepared if prepared and stamps else None,
        'recording_duration_s':duration,'process_wall_s':wall,
        'wall_ms_per_frame':wall*1000/processed if wall and processed else None,
        'throughput_fps':processed/wall if wall and processed else None,
        'realtime_speedup':duration/wall if wall and duration else None,
        'rtf':wall/duration if wall and duration else None,
        'backend_frame_ms':backend,'optimized_frame_ms':_distribution(sums.get('optimize',[])),
        'marginalization_frame_ms':_distribution(sums.get('marginalize',[])),
        'backend_frame_samples_ms':sample_values,
        'backend_samples_match_poses':bool(stamps and len(measured)==len(stamps)),
        'native_pipeline_wall_s':_scalar(native,'exec_time_s'),
        'peak_rss_mib':peak_bytes/(1024**2) if peak_bytes is not None else None,
        'cpu_user_s':resource.get('user_s'),'cpu_system_s':resource.get('system_s'),
        'cpu_time_s':cpu_time,
        'cpu_percent':resource.get('cpu_percent',cpu_time/wall*100 if cpu_time is not None and wall else None),
        'configured_tbb_threads':threads,
        'marginalization_cache_written':'--marg-data' in command,
        'optical_flow_frame_ms':None,'end_to_end_frame_latency_ms':None,
        'sources':{'backend_frame':relative(sums_path),'native_pipeline':relative(native_path),
                   'process_wall':relative(timing_path) if timing_path.exists() else None,
                   'resources':relative(resources_path) if resources_path.exists() else None,
                   'processed_frames':relative(directory/'trajectory.csv') if stamps else relative(sums_path)},
        'source_scope':{
            'backend_frame_ms':'Measured wall Timer around each visual measure(): state prediction, visual update, optimization, marginalization and output queue pushes; excludes optical flow, preceding IMU preintegration and input vision queue waiting.',
            'wall_ms_per_frame':'Whole VIO process wall time / native output pose count; throughput average, not measured end-to-end latency of each frame. Includes image IO, optical flow, backend and output; excludes dataset preparation and later NFR mapping.',
            'rtf':'process_wall_s / recording_duration_s; lower is faster; realtime_speedup is its reciprocal.',
            'peak_rss_mib':'GNU time maximum resident set size in KiB / 1024; native resident_memory_peak bytes is used only when GNU time is unavailable. Per-iteration resident_memory sums are not memory peaks.',
            'cpu_percent':'GNU time aggregate multicore CPU percentage; 100 percent corresponds to one CPU core.',
            'unavailable':'No optical-flow durations or input-to-pose per-frame latency were recorded in this run.'}}
    if not measured:
        result['source_scope']['backend_frame_ms']='No saved per-update timing samples.'
    return result
