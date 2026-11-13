#!/usr/bin/env python3
"""Build and serve the existing native-label mesh viewer for an explicit dataset."""
import argparse,json,os,socket,subprocess,sys
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUILDER=ROOT.parent/'wilor_training/tools/marker_placement/build_label_viewer.py'
def main():
 ap=argparse.ArgumentParser(__doc__)
 ap.add_argument('--dataset',type=Path,required=True,help='Dataset root containing index.jsonl, images/, labels/')
 ap.add_argument('--output',type=Path,help='Viewer cache; default <dataset-parent>/<dataset-name>_viewer')
 ap.add_argument('--port',type=int,default=8770)
 ap.add_argument('--sync',type=int,help='Initially open the nearest available sync frame')
 ap.add_argument('--build-only',action='store_true',help='Build HTML and projection cache without starting a server')
 a=ap.parse_args();dataset=a.dataset.expanduser().resolve();output=a.output.expanduser().resolve() if a.output else dataset.parent/(dataset.name+'_viewer')
 if not (dataset/'index.jsonl').is_file():ap.error(f'Missing {dataset/"index.jsonl"}; provide dataset root, not labels/')
 if not 1<=a.port<=65535:ap.error('port must be 1..65535')
 if not a.build_only:
  with socket.socket() as probe:
   try:probe.bind(('127.0.0.1',a.port))
   except OSError:ap.error(f'Port {a.port} is occupied; use --port with another port or reuse the running viewer')
 cmd=[sys.executable,'-B',str(BUILDER),'--dataset',str(dataset),'--output',str(output),'--project',str(ROOT),'--standalone']
 if a.sync is not None:cmd+=['--initial-sync',str(a.sync)]
 env=dict(os.environ,PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None);subprocess.run(cmd,check=True,env=env)
 report=json.loads((output/'build_report.json').read_text());(output/'launch.json').write_text(json.dumps(dict(dataset=str(dataset),output=str(output),port=a.port,initial_sync=a.sync),indent=2))
 print(f'已加载 {report["frames"]} 张原图、{report["labels"]} 份标签。',flush=True)
 if a.build_only:print(f'查看器：{output/"index.html"}',flush=True);return
 class Handler(SimpleHTTPRequestHandler):
  def end_headers(self):self.send_header('Cache-Control','no-cache');super().end_headers()
  def log_message(self,*args):pass
 print(f'标签查看器：http://127.0.0.1:{a.port}/\n保持终端运行，Ctrl+C 退出。',flush=True)
 server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(output)))
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()
