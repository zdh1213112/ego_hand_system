"""Start the single-camera-owner WiLoR / continuous VIO / body / Rerun chain."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def external_roots(root=ROOT):
    """Support both the standalone and EGO_glove checkout layouts."""
    return (root.parent, root.parent / 'EGO_glove', root.parent.parent)


def external_path(relative, root=ROOT):
    candidates = [base / relative for base in external_roots(root)]
    return next((path for path in candidates if path.exists()), candidates[0])


def wait_socket(path, child, timeout=20):
    deadline = time.monotonic()+timeout
    while not path.exists():
        if child.poll() is not None:
            raise RuntimeError(f'{path.name} producer failed; inspect logs')
        if time.monotonic()>=deadline:
            raise TimeoutError(f'timed out waiting for {path}')
        time.sleep(.05)


def stop_process(child, timeout=10, stop_signal=signal.SIGINT):
    try:os.killpg(child.pid,stop_signal)
    except ProcessLookupError:return
    try:child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=3)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/ego_live_multimodal'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--ego-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--visualizer',type=Path,default=external_path('ego-visualizer/.venv/bin/ego-visualizer'))
    parser.add_argument('--omni-root',type=Path,default=external_path('omniegocap_orbbec/OmniEgoCap'))
    parser.add_argument('--sdk-config',type=Path)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--no-body',action='store_true')
    parser.add_argument('--body-no-optimization',action='store_true')
    parser.add_argument('--body-optimization',action='store_true',
                        help='enable slower OmniEgoCap optimization for higher quality')
    parser.add_argument('--body-window-s',type=float,default=1.0)
    parser.add_argument('--body-stride-s',type=float,default=1.0)
    parser.add_argument('--body-ddim-steps',type=int,default=6,
                        help='fast live sampling; 30 restores the original OmniEgoCap preset')
    parser.add_argument('--body-max-windows',type=int,default=0)
    parser.add_argument('--camera-height',type=float,default=1.25,
                        help='initial cam_0 height above floor in metres; seated preset 1.25, measure for best pose')
    parser.add_argument('--max-frames',type=int,default=0,help='finite capture for validation; 0 runs until Ctrl+C')
    parser.add_argument('--vio-scale',type=float,default=.5)
    parser.add_argument('--display-width',type=int,default=1920)
    parser.add_argument('--display-cameras',nargs='+',choices=('cam_0','cam_1','cam_2','cam_3'),
                        default=['cam_0'])
    parser.add_argument('--display-preview-fps',type=float,default=10.0)
    parser.add_argument('--display-depth-hz',type=float,default=0.0,
                        help='opt in to background stereo depth computation; 0 disables it')
    parser.add_argument('--display-pose-fps',type=float,default=15.0)
    parser.add_argument('--display-map-fps',type=float,default=2.0)
    args=parser.parse_args(argv)
    if args.max_frames<0 or args.body_max_windows<0:
        parser.error('frame/window limits cannot be negative')
    if args.body_optimization and args.body_no_optimization:
        parser.error('--body-optimization and --body-no-optimization conflict')
    if len(set(args.display_cameras))!=len(args.display_cameras):
        parser.error('display cameras must be unique')
    if (args.display_width<160 or args.display_depth_hz<0 or args.display_preview_fps<=0 or args.display_pose_fps<0 or
            args.display_map_fps<0 or args.body_window_s<1 or args.body_stride_s<=0 or
            not 2<=args.body_ddim_steps<=30):
        parser.error('invalid live display rate, width or body window')
    sdk=args.sdk_config
    if sdk is None:
        candidates=[ROOT/'third_party/orbbec_sdk/shared/OrbbecSDKConfig.xml']
        for base in external_roots():
            candidates+=sorted(base.glob('OrbbecSDK*/shared/OrbbecSDKConfig.xml'))
        sdk=next((p for p in candidates if p.is_file()),None)
    if sdk is None or not sdk.is_file():
        raise FileNotFoundError('Orbbec SDK config not found; pass --sdk-config')
    omni_python=args.omni_root.parent/'venv/bin/python'
    required=[args.ego_python,args.visualizer,ROOT/'build/ego_live_bridge',ROOT/'build/ego_basalt_live']
    if not args.no_body:required += [omni_python,args.omni_root/'scripts/live_predict.py']
    for path in required:
        if not path.is_file():raise FileNotFoundError(path)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir()
    camera=output/'ego4_calibration.yaml'
    imu=output/'imu_calibration.yaml'
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env['PYTHONNOUSERSITE']='1'
    subprocess.run([str(ROOT/'build/ego_live_bridge'),'--sdk-config',str(sdk),
                    '--calibration-only','--calibration-out',str(camera),
                    '--imu-calibration-out',str(imu)],check=True,env=env)
    time.sleep(.5)
    viewer_socket=Path(f'/tmp/ego-mm-{os.getpid()}-view.sock')
    vio_socket=Path(f'/tmp/ego-mm-{os.getpid()}-vio.sock')
    children={}
    logs=[]
    commands={}
    def start(name,command,cwd=ROOT):
        command=[str(part) for part in command]
        commands[name]=command
        log=(output/'logs'/f'{name}.log').open('wb')
        logs.append(log)
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=child
        print(f'Started {name}: pid={child.pid}',flush=True)
        return child
    interrupted=False
    failed=None
    try:
        viewer_args=[args.visualizer,'live','--socket',viewer_socket,'--fast']
        if args.headless:viewer_args += ['--headless','--save',output/'live.rrd']
        viewer=start('viewer',viewer_args)
        wait_socket(viewer_socket,viewer)
        vio=start('vio',[args.ego_python,'-u',ROOT/'scripts/ego_live_basalt_stream.py',
                        '--socket',vio_socket,'--camera-calibration',camera,'--imu-calibration',imu,
                        '--output',output/'vio','--scale',args.vio_scale])
        wait_socket(vio_socket,vio)
        start('world_forwarder',[args.visualizer,'live-world','--socket',viewer_socket,
                                 '--input',output/'vio/world.jsonl','--follow',
                                 '--pose-fps',args.display_pose_fps,'--map-fps',args.display_map_fps])
        if not args.no_body:
            body_args=[omni_python,'-u',args.omni_root/'scripts/live_predict.py',
                       '--hands-jsonl',output/'wilor/predictions.jsonl','--vio-jsonl',output/'vio/vio.jsonl',
                       '--socket',viewer_socket,'--output-jsonl',output/'body.jsonl',
                       '--camera-height',args.camera_height,'--window-s',args.body_window_s,
                       '--stride-s',args.body_stride_s,'--ddim-steps',args.body_ddim_steps,
                       '--max-windows',args.body_max_windows]
            if not args.body_optimization:body_args.append('--no-optimization')
            start('body',body_args,args.omni_root)
        capture=start('capture',[args.ego_python,'-u',ROOT/'scripts/ego_live_wilor.py',
                         '--sdk-config',sdk,'--calibration',camera,'--output',output/'wilor',
                         '--no-preview','--live-socket',viewer_socket,'--vio-socket',vio_socket,
                         '--live-cameras',*args.display_cameras,'--live-width',args.display_width,
                         '--live-native-panel','--live-preview-fps',args.display_preview_fps,
                         '--live-depth-hz',args.display_depth_hz,'--max-frames',args.max_frames]
                         + (['--no-live-depth'] if args.display_depth_hz == 0 else []))
        (output/'launch.json').write_text(json.dumps(dict(started_at=datetime.now().isoformat(),commands=commands,
            headless=args.headless,body_optimization=args.body_optimization,
            body_ddim_steps=args.body_ddim_steps,
            camera_height_m=args.camera_height,
            display=dict(cameras=args.display_cameras,width=args.display_width,
                         preview_fps=args.display_preview_fps,depth_hz=args.display_depth_hz,pose_fps=args.display_pose_fps,
                         map_fps=args.display_map_fps),
            coordinate_note='VIO world is metric Z-up; default camera height/head extrinsic are assumptions'),indent=2)+'\n')
        print(f'LIVE RUN READY: {output}',flush=True)
        while capture.poll() is None:
            for name,child in children.items():
                if name=='capture':continue
                status=child.poll()
                if name=='vio' and status==0 and args.max_frames:
                    # The camera closes its VIO side channel before finishing
                    # GPU/process cleanup. This EOF is normal at a finite cap.
                    try:capture.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('VIO input ended while capture is still running')
                    break
                if status is not None and not (name=='body' and status==0 and args.body_max_windows):
                    raise RuntimeError(f'{name} exited unexpectedly with {status}; inspect {output/"logs"}')
            time.sleep(.2)
        if capture.returncode:
            raise RuntimeError(f'capture failed with {capture.returncode}; inspect capture.log')
        vio.wait(timeout=30)
        if vio.returncode:raise RuntimeError('VIO failed; inspect vio.log and vio/backend.log')
        if 'body' in children:
            deadline=time.monotonic()+45
            while time.monotonic()<deadline and children['body'].poll() is None:
                body_file=output/'body.jsonl'
                if body_file.exists() and body_file.stat().st_size and not args.body_max_windows:break
                time.sleep(.2)
            body_file=output/'body.jsonl'
            if not body_file.exists() or not body_file.stat().st_size:
                raise RuntimeError('no body prediction was produced; inspect body.log')
    except KeyboardInterrupt:
        interrupted=True
    except Exception as error:
        failed=error
    finally:
        # Stop only this launcher's process groups, keeping the display alive
        # until camera, VIO and body writers have flushed their final events.
        for name in ('capture','vio','body','world_forwarder'):
            if name in children:
                stop_process(children[name], stop_signal=signal.SIGTERM if name=='body' else signal.SIGINT)
        time.sleep(.3)
        if 'viewer' in children:stop_process(children['viewer'])
        for log in logs:log.close()
    states={name:child.returncode for name,child in children.items()}
    result=dict(output=str(output),interrupted=interrupted,error=str(failed) if failed else None,process_returncodes=states)
    summary=output/'vio/summary.json'
    if summary.exists():result['vio']=json.loads(summary.read_text())
    body=output/'body.jsonl'
    result['body_predictions']=sum(1 for line in body.read_text().splitlines() if line.strip()) if body.exists() else 0
    viewer_summary=output/'live.summary.json'
    if viewer_summary.exists():result['viewer']=json.loads(viewer_summary.read_text())
    (output/'run_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if failed:raise failed
    return 130 if interrupted else 0


if __name__=='__main__':
    raise SystemExit(main())
