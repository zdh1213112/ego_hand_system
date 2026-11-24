"""Align mesh boundary definition with observed external contours; no target-coordinate edits."""
import argparse,sys,json
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import EXP
from refine_contours import project
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path);ap.add_argument('--source',type=Path,default=EXP/'personal_calibration_v3');ap.add_argument('--observations',type=Path,default=EXP/'personal_contours_v1/observations.json');ap.add_argument('--output',type=Path,default=EXP/'personal_contours_external_v2');a=ap.parse_args()
 from context import load_context
 context=load_context(a.context);width,height=json.loads((context['dataset']/'manifest.json').read_text())['image_size']
 obs=json.loads(a.observations.read_text());zs=[np.load(a.source/f'{h}_pilot.npz') for h in ['left','right']];maps=[{int(f):i for i,f in enumerate(z['pair_indices'])} for z in zs];changes=[];kept=[]
 for r in obs['records']:
  side=r['side'];z=zs[side];i=maps[side][r['sync']];uv=project(torch.tensor(z['vertices'][i],dtype=torch.float64),r,side).numpy();jp=project(torch.tensor(z['joints'][i],dtype=torch.float64),r,side).numpy();mask=np.zeros((height,width),np.uint8)
  for tri in np.rint(uv[z['faces']]).astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);edge=np.zeros_like(mask);cv2.drawContours(edge,contours,-1,1,1);d=cv2.distanceTransform(1-edge,cv2.DIST_L2,3);ix=np.rint(uv).astype(int);inside=(ix[:,0]>=0)&(ix[:,0]<width)&(ix[:,1]>=0)&(ix[:,1]<height);ids=np.flatnonzero(inside);axis=jp[[5,9,13,17]].mean(0)-jp[0];axis/=max(np.linalg.norm(axis),1e-9);ids=ids[(d[ix[ids,1],ix[ids,0]]<2.5)&((uv[ids]-jp[0])@axis>8)]
  old=np.array(r['boundary_vertices']);removed=len(set(old)-set(ids));changes.append(dict(camera=r['camera'],sync=r['sync'],side=side,old_count=len(old),new_count=len(ids),old_indices_removed=removed))
  if len(ids)<25:obs['rejected'].append(dict(camera=r['camera'],sync=r['sync'],side=side,reason='external_mesh_boundary_insufficient'));continue
  r['boundary_vertices']=ids.tolist();kept.append(r)
 obs['records']=kept;obs['mesh_contour_definition']='external-only; v3 initialization; matches image RETR_EXTERNAL; internal holes excluded on both sides';obs['boundary_changes']=changes
 out=a.output;out.mkdir(exist_ok=False);(out/'observations.json').write_text(json.dumps(obs));print('Accepted',len(kept),'changed records',sum(q['old_indices_removed']>0 for q in changes))
if __name__=='__main__':main()
