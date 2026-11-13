#!/usr/bin/env python3
"""Explicit post-baseline personal shape, 24-marker, native contour and experimental export chain."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from personal_marker24_config import load_personal_config
ROOT=Path(__file__).resolve().parents[1];TOOLS=ROOT.parent/'wilor_training/tools';P=TOOLS/'personal_hand_calibration'
def main():
 ap=argparse.ArgumentParser(__doc__)
 for name in ['dataset','alignment','aligned21','baseline','config','output']:ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--sample-stride',type=int,default=8);ap.add_argument('--source-fps',type=float,default=30);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();cfg=load_personal_config(a.config)
 if not cfg['enabled']:print('personal_marker24_optimization disabled; baseline only');return
 if a.sample_stride<1 or not 0<a.source_fps<1000:ap.error('Invalid stride or fps')
 out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 context=dict(dataset=str(a.dataset.resolve()),alignment=str(a.alignment.resolve()),aligned21=str(a.aligned21.resolve()),baseline_fit=str(a.baseline.resolve()/'mano_fit_full_cam01'),placement=cfg['placement'],holdout=cfg['holdout'],dorsal_reference=cfg['dorsal_reference'],device=a.device)
 if not a.dry_run:
  for key in ['dataset','alignment','aligned21','baseline_fit','placement']:
   if not Path(context[key]).exists():raise FileNotFoundError(context[key])
  out.mkdir(parents=True);(out/'logs').mkdir();(out/'context.json').write_text(json.dumps(context,indent=2));(out/'optimization_config.json').write_text(json.dumps(cfg,indent=2))
 context_args=['--context',out/'context.json'];commands=[];env=dict(os.environ,PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
 def run(name,script,*arguments):
  cmd=[sys.executable,'-B',str(script),*map(str,arguments)];entry=dict(stage=name,command=cmd,returncode=None);commands.append(entry)
  if a.dry_run:print(json.dumps(entry));return
  (out/'commands.json').write_text(json.dumps(commands,indent=2));print(name,flush=True)
  with (out/'logs'/f'{name}.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  entry['returncode']=result.returncode;(out/'commands.json').write_text(json.dumps(commands,indent=2))
  if result.returncode:raise RuntimeError(f'{name} failed; see {out/"logs"/(name+".log")}')
 run('personal_shape_marker24',P/'run.py',*context_args,'--enable-personal-calibration','--output',out/'marker24','--steps',cfg['calibration_steps'],'--frames',cfg['calibration_frames'],'--sample-stride',a.sample_stride,'--pose-scope',cfg['pose_scope'])
 run('native_contours',P/'image_constraints.py',*context_args,'--source',out/'marker24','--output',out/'contours')
 run('contour_shape_refinement',P/'refine_contours.py',*context_args,'--source',out/'marker24','--observations',out/'contours/observations.json','--output',out/'contour_fit','--steps',cfg['contour_steps'])
 run('review',TOOLS/'marker24_pilot/build_review.py','--dataset',a.dataset,'--pilot',out/'contour_fit')
 final_fit=out/'contour_fit'
 joint_cfg=cfg['joint_pose_attachments']
 if joint_cfg['enabled']:
  run('joint_external_boundaries',P/'fix_boundary_correspondence.py',*context_args,'--source',final_fit,'--observations',out/'contours/observations.json','--output',out/'joint_contours')
  run('joint_pose_attachments',TOOLS/'joint_pose_attachment/run.py',*context_args,'--source',final_fit,'--observations',out/'joint_contours/observations.json','--output',out/'joint_fit','--steps',joint_cfg['steps'],'--view-policy',joint_cfg['view_policy'])
  run('joint_review',TOOLS/'marker24_pilot/build_review.py','--dataset',a.dataset,'--pilot',out/'joint_fit','--baseline-pilot',final_fit)
  final_fit=out/'joint_fit'
 if cfg['diagnostics']:
  obs=out/'diagnostic_contours/observations.json';runs=out/'diagnostics'
  run('external_boundaries',P/'fix_boundary_correspondence.py',*context_args,'--source',out/'contour_fit','--observations',out/'contours/observations.json','--output',obs.parent)
  for mode in ['rigid','fingers','attachments','recalibrated_fingers']:
   source=runs/'personal_ablation_attachments_v2' if mode=='recalibrated_fingers' else out/'contour_fit'
   run('diagnostic_'+mode,P/'ablate_geometry.py',*context_args,'--mode','fingers' if mode=='recalibrated_fingers' else mode,'--source',source,'--observations',obs,'--output',runs/f'personal_ablation_{mode}_v2','--steps',cfg['diagnostic_steps'])
  common=['--source',out/'contour_fit','--observations',obs,'--runs-root',runs,'--output',out/'diagnostic_report']
  run('diagnostic_report',P/'summarize_ablation.py',*common);run('rendered_diagnostic_scores',P/'evaluate_rendered_ablation.py',*context_args,*common)
 if cfg['export_optimized']:
  run('materialize_optimized_fit',P/'materialize_fit.py',*context_args,'--source',final_fit,'--output',out/'mano_fit')
  export=dict(mano_source=str(ROOT/'third_party/MANO'),mano_model_dir=str(ROOT/'models/mano'),recordings=[dict(name='personal_marker24',fusion=str(a.baseline.resolve()/'nokov_observations_cam01'),mano_fit=str(out/'mano_fit'),dataset=str(a.dataset.resolve()),reference_camera='camera0',cameras=['camera0','camera1','camera2','camera3'])])
  if not a.dry_run:(out/'native_export_config.json').write_text(json.dumps(export,indent=2))
  # Input fit is already sampled at the requested stride: do not sample it a second time.
  run('export_check_preview',ROOT/'scripts/run_native_wilor_export.py','--config',out/'native_export_config.json','--output',out/'export','--sample-stride',1,'--device',a.device,'--require-both-hands')
  run('videos',ROOT/'scripts/render_wilor_training_videos.py','--dataset',out/'export/merged','--output',out/'export/videos','--fps',a.source_fps/a.sample_stride,'--workers',2,'--video-codec','auto')
 if not a.dry_run:
  report=dict(status='experimental_pipeline_complete_not_accuracy_approved',approved_for_training=False,baseline=str(a.baseline.resolve()),shape_sharing=cfg['shape_sharing'],pose_scope=cfg['pose_scope'],sample_stride=a.sample_stride,fit=str(final_fit),joint_pose_attachments_enabled=joint_cfg['enabled'],labels=str(out/'export/merged') if cfg['export_optimized'] else None,videos=str(out/'export/videos') if cfg['export_optimized'] else None)
  (out/'pipeline_summary.json').write_text(json.dumps(report,indent=2))
 print('Optimized outputs:',out,flush=True)
if __name__=='__main__':main()
