"""Re-render independent external boundaries for current, joint-all and leave-camera-out fits."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'personal_hand_calibration'))
from context import load_context
from refine_contours import project
from ablate_geometry import split_views

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--all-fit',type=Path,required=True);ap.add_argument('--held-fit',type=Path,required=True);ap.add_argument('--observations',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=load_context(a.context);width,height=json.loads((c['dataset']/'manifest.json').read_text())['image_size'];obs=json.loads(a.observations.read_text());report=dict(approved_for_training=False,hands={})
 for side,hand in enumerate(['left','right']):
  source=np.load(a.source/f'{hand}_pilot.npz');index={int(f):i for i,f in enumerate(source['pair_indices'])};records=[r for r in obs['records'] if r['side']==side];fit,held,_=split_views(records);allkeys={(r['camera'],r['sync']) for r in fit+held};heldkeys={(r['camera'],r['sync']) for r in held};scores={}
  for mode,path in [('before',a.source),('all',a.all_fit),('leave_one_out',a.held_fit)]:
   z=np.load(path/f'{hand}_pilot.npz');rows=[]
   for r in records:
    if (r['camera'],r['sync']) not in allkeys:continue
    i=index[r['sync']];uv=project(torch.tensor(z['vertices'][i],dtype=torch.float64),r,side).numpy();jp=project(torch.tensor(source['joints'][i],dtype=torch.float64),r,side).numpy();axis=jp[[5,9,13,17]].mean(0)-jp[0];axis/=max(np.linalg.norm(axis),1e-9);mask=np.zeros((height,width),np.uint8)
    for tri in np.rint(uv[z['faces']]).astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);pts=np.concatenate([q.reshape(-1,2) for q in contours]);pts=pts[(pts-jp[0])@axis>8]
    if not len(pts):raise ValueError('No distal boundary')
    pts=pts[np.linspace(0,len(pts)-1,min(200,len(pts))).astype(int)];gt=np.asarray(r['points']);error=np.r_[cKDTree(gt).query(pts)[0],cKDTree(pts).query(gt)[0]];rows.append(dict(camera=r['camera'],sync=r['sync'],held=(r['camera'],r['sync']) in heldkeys,distance_px=float(error.mean())))
   scores[mode]=rows;print(hand,mode,len(rows),float(np.mean([r['distance_px'] for r in rows])),flush=True)
  before=scores['before'];after=scores['all'];heldbefore=[r['distance_px'] for r in before if r['held']];heldafter=[r['distance_px'] for r in scores['leave_one_out'] if r['held']];d=np.array([n['distance_px']-o['distance_px'] for o,n in zip(before,after)]);delta=[dict(camera=b['camera'],sync=b['sync'],before_px=b['distance_px'],after_px=n['distance_px'],delta_px=float(x)) for b,n,x in zip(before,after,d)]
  report['hands'][hand]=dict(all_views_before_px=float(np.mean([r['distance_px'] for r in before])),all_views_after_px=float(np.mean([r['distance_px'] for r in after])),held_camera_before_px=float(np.mean(heldbefore)),held_camera_after_px=float(np.mean(heldafter)),held_camera_count=len(heldbefore),all_views_count=len(before),worsened_over_1px=int((d>1).sum()),improved_over_1px=int((d< -1).sum()),best=sorted(delta,key=lambda r:r['delta_px'])[:5],worst=sorted(delta,key=lambda r:-r['delta_px'])[:5],per_view=scores)
 a.output.write_text(json.dumps(report,indent=2));print(a.output)
if __name__=='__main__':main()
