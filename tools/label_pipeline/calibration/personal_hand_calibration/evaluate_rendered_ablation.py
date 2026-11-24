"""Re-render external silhouettes after fitting; independent of fixed loss vertex indices."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import EXP
from refine_contours import project
from ablate_geometry import split_views
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path);ap.add_argument('--source',type=Path,default=EXP/'personal_calibration_v3');ap.add_argument('--observations',type=Path,default=EXP/'personal_contours_external_v2/observations.json');ap.add_argument('--runs-root',type=Path,default=EXP);ap.add_argument('--output',type=Path,default=EXP/'personal_ablation_report_v2');a=ap.parse_args()
 from context import load_context
 context=load_context(a.context);width,height=json.loads((context['dataset']/'manifest.json').read_text())['image_size']
 out=a.output;report=json.loads((out/'report.json').read_text());obs=json.loads(a.observations.read_text());results={}
 for side,hand in enumerate(['left','right']):
  source=np.load(a.source/f'{hand}_pilot.npz');index={int(f):i for i,f in enumerate(source['pair_indices'])};_,held,_=split_views([r for r in obs['records'] if r['side']==side]);scores={}
  for mode in ['source','rigid','fingers','attachments','recalibrated_fingers']:
   z=source if mode=='source' else np.load(a.runs_root/f'personal_ablation_{mode}_v2'/f'{hand}_pilot.npz');rows=[]
   for r in held:
    i=index[r['sync']];uv=project(torch.tensor(z['vertices'][i],dtype=torch.float64),r,side).numpy();j0=project(torch.tensor(source['joints'][i],dtype=torch.float64),r,side).numpy();axis=j0[[5,9,13,17]].mean(0)-j0[0];axis/=np.linalg.norm(axis);mask=np.zeros((height,width),np.uint8)
    for tri in np.rint(uv[z['faces']]).astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);p=np.concatenate([c.reshape(-1,2) for c in contours]);p=p[(p-j0[0])@axis>8];p=p[np.linspace(0,len(p)-1,min(200,len(p))).astype(int)];gt=np.array(r['points']);e=np.r_[cKDTree(gt).query(p)[0],cKDTree(p).query(gt)[0]];rows.append(dict(camera=r['camera'],sync=r['sync'],distance_px=float(e.mean())))
   scores[mode]=rows
   print(hand,mode,'rendered',float(np.mean([q['distance_px'] for q in rows])),flush=True)
  for mode in report['modes']:
   before=scores['source'];after=scores[mode];d=np.array([a['distance_px']-b['distance_px'] for a,b in zip(after,before)]);r=report['modes'][mode][hand];r.update(rendered_before_px=float(np.mean([q['distance_px'] for q in before])),rendered_after_px=float(np.mean([q['distance_px'] for q in after])),rendered_worsened_over_1px=int((d>1).sum()),rendered_improved_over_1px=int((d< -1).sum()))
  results[hand]=scores
 (out/'rendered_external_contours.json').write_text(json.dumps(results,indent=2));(out/'report.json').write_text(json.dumps(report,indent=2))
 p=out/'index.html';h=p.read_text();start=h.index('data=')+len('data=');end=h.index(",$=x=>",start);h=h[:start]+json.dumps(report)+h[end:];h=h.replace("rows[side].before_px.toFixed(2)","rows[side].rendered_before_px.toFixed(2)").replace("rows[side].after_px.toFixed(2)","rows[side].rendered_after_px.toFixed(2)").replace('指标为自动外轮廓距离','表格使用重新渲染后的外轮廓距离，不依赖优化时固定的边界顶点。指标仍基于自动提取的原图边界');p.write_text(h)
if __name__=='__main__':main()
