#!/usr/bin/env python3
"""Full-mask driven shared MANO shape fitting on quality keyframes only.

The pure-shape control keeps all motion and marker attachments exactly fixed.
The second controlled variant permits small motion changes on calibration frames.
Neither variant fits an image or motion parameter of a held-out frame.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/label_pipeline/calibration/personal_hand_calibration'))
from context import load_context
from run import PROJECT, import_mano, run_model, stats
from render_masks import FullMaskBatch


def main():
    ap = argparse.ArgumentParser()
    for name in ['context', 'selection', 'source', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--refine-calibration-pose', action='store_true')
    a = ap.parse_args()
    if a.steps < 2 or a.resolution < 64:
        ap.error('Need >=2 steps and >=64 resolution')
    if a.output.exists():
        raise FileExistsError(a.output)
    c = load_context(a.context)
    if c['device'] != 'cuda':
        raise ValueError('This renderer requires CUDA')
    selection = json.loads(a.selection.read_text())
    torch.set_num_threads(4)
    torch.manual_seed(0)
    import cv2
    cv2.setNumThreads(2)
    model = import_mano(PROJECT/'third_party/MANO').load(model_path=str(PROJECT/'models/mano'),
        is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1).to('cuda')
    from mano.lbs import batch_rodrigues
    for p in model.parameters():
        p.requires_grad_(False)
    a.output.mkdir(parents=True)
    faces = model.faces_tensor.long()
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device='cuda')
    summary = {'strategy':'full_mask_shared_shape_v1', 'approved_for_training':False,
               'source':str(a.source.resolve()), 'selection':str(a.selection.resolve()),
               'shape_sharing':'one_beta_vector_per_physical_hand', 'full_mask_fitting':True,
               'refine_calibration_pose':a.refine_calibration_pose, 'marker_attachments_frozen':True,
               'noncalibration_pose_frozen':True, 'noncalibration_images_used':False,
               'hand_pose_representation':'45D axis angle before MANO pose mean (legacy field hand_pose_pca)', 'renderer':'nvdiffrast CUDA + analytic true KB projection', 'resolution':a.resolution,
               'mask_preprocessing':'Only enclosed <=64px and <=16px-wide speckle holes filled; finger/pinch gaps retained',
               'image_domain':'All distal hand pixels, fixed wrist plane >8px; cuff and forearm excluded',
               'objective':{'one_minus_iou':4.,'pixel_distance_term':1.,'robust_nokov24':.15,'beta_delta_prior':.008},
               'hands':{}}
    shared = json.loads((a.source/'shared_calibration.json').read_text())
    for side, hand in enumerate(['left','right']):
        if not (a.source/f'{hand}_pilot.npz').is_file(): continue
        start = time.monotonic()
        z = np.load(a.source/f'{hand}_pilot.npz')
        pairs = z['pair_indices']
        lookup = {int(f):i for i,f in enumerate(pairs)}
        chosen = selection['report']['hands'][hand]['calibration_sync']
        cal = np.asarray([lookup[f] for f in chosen])
        other = np.setdiff1d(np.arange(len(pairs)),cal)
        keys = {(r['camera'],r['sync']) for r in selection['report']['hands'][hand]['calibration_views']}
        records = [r for r in selection['records'] if r['side']==side and r['accepted_quality'] and (r['camera'],r['sync']) in keys]
        assert all(r['sync'] in chosen for r in records)
        batch = FullMaskBatch(records,lookup,z['joints'],z['vertices'],z['faces'],side,a.resolution)
        beta0, o0, p0, t0 = [to(z[k]) for k in ['betas','global_orient','hand_pose_pca','translation']]
        db = torch.nn.Parameter(torch.zeros_like(beta0))
        dp, do, dt = [torch.nn.Parameter(torch.zeros_like(x),requires_grad=False) for x in [p0,o0,t0]]
        bary, offset, wrist = [to(z[k]) for k in ['barycentric','normal_offsets_m','wrist_local']]
        face_ids = torch.as_tensor(z['attachment_faces'],dtype=torch.long,device='cuda')
        attachments = [x.clone() for x in [bary,offset,wrist]]
        target = to(z['target24'])
        weights = to([.2]*3+[1]*21)
        def geometry():
            return run_model(model,(beta0+db)[None].expand(len(pairs),-1),o0+do,p0+dp,t0+dt)[:2]
        def predict(v,j):
            tri=v[:,faces[face_ids]]
            normal=torch.nn.functional.normalize(torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
            surface=(tri*bary[None,:,:,None]).sum(2)+normal*offset[None,:,None]
            wr=torch.einsum('fij,kj->fki',batch_rodrigues(o0+do),wrist)+j[:,:1]
            return torch.cat([wr,surface],1)
        def objective():
            v,j=geometry();li,ld,iou=batch.terms(v)
            e=(predict(v,j)[cal]-target[cal]).norm(dim=-1)/.006
            lm=((torch.sqrt(1+e.square())-1)*weights).mean()
            poseprior=.03*(dp[cal]/.05).square().mean()+.03*(do[cal]/.025).square().mean()+.03*(dt[cal]/.003).square().mean()
            loss=4*li+ld+.15*lm+.008*db.square().mean()+poseprior
            return loss,(li,ld,iou,lm),v,j
        history=[]
        with torch.no_grad():
            initial_loss, terms, _, _ = objective()
            before={'loss':float(initial_loss),'soft_iou':float(terms[2]),'marker_term':float(terms[3])}
        optimizer=torch.optim.Adam([db],lr=.008)
        best_loss=float('inf');best=None
        for step in range(a.steps):
            if a.refine_calibration_pose and step==a.steps//2:
                for p in [dp,do,dt]:p.requires_grad_(True)
                optimizer=torch.optim.Adam([{'params':[db],'lr':.004},{'params':[dp,do],'lr':.0008},{'params':[dt],'lr':.00005}])
            optimizer.zero_grad()
            loss,terms,_,_=objective()
            if not torch.isfinite(loss):raise ValueError('Nonfinite full-mask objective')
            if float(loss.detach())<best_loss:
                best_loss=float(loss.detach());best=[p.detach().clone() for p in [db,dp,do,dt]]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                db.clamp_(-1.5,1.5);dp.clamp_(-.05,.05)
                do.mul_(torch.clamp(.025/do.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1))
                dt.mul_(torch.clamp(.003/dt.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1))
            if step%50==0 or step==a.steps-1:
                row={'step':step,'loss':float(loss.detach()),'soft_iou':float(terms[2].detach()),'mask_distance':float(terms[1].detach()),'marker_term':float(terms[3].detach())}
                history.append(row);print(hand,json.dumps(row),flush=True)
        with torch.no_grad():
            loss,terms,_,_=objective()
            if float(loss)<best_loss:best=[p.detach().clone() for p in [db,dp,do,dt]]
            for p,q in zip([db,dp,do,dt],best):p.copy_(q)
            loss,terms,v,j=objective();pred=predict(v,j);residual=(pred-target).norm(dim=-1)*1000
        assert all(torch.equal(x,y) for x,y in zip([bary,offset,wrist],attachments))
        for p in [dp,do,dt]:assert not torch.count_nonzero(p[other]),'Held-out motion changed'
        cpu=lambda x:x.detach().cpu().numpy()
        values={k:z[k] for k in z.files}
        values.update(betas=cpu(beta0+db),vertices=cpu(v),joints=cpu(j),baseline_vertices=z['vertices'],baseline_joints=z['joints'],
                      global_orient=cpu(o0+do),hand_pose_pca=cpu(p0+dp),translation=cpu(t0+dt),
                      predicted24=cpu(pred),marker_error_mm=cpu(residual),full_mask_shape_fitting=np.array(True),approved_for_training=np.array(False))
        np.savez_compressed(a.output/f'{hand}_pilot.npz',**values)
        # A true flat pose in PCA coordinates, shared template only; no frame motion.
        neutral=(torch.linalg.solve(model.hand_components.T,-model.hand_mean) if model.use_pca else -model.hand_mean).reshape(1,-1)
        with torch.no_grad():nv,nj,_=run_model(model,(beta0+db)[None],torch.zeros((1,3),device='cuda'),neutral,torch.zeros((1,3),device='cuda'))
        np.savez_compressed(a.output/f'{hand}_reference_hand.npz',vertices=cpu(nv[0]-nj[0,:1]),joints=cpu(nj[0]-nj[0,:1]),faces=z['faces'],betas=cpu(beta0+db),geometry_space='right_canonical',approved_for_training=False)
        shared['hands'][hand]['betas']=cpu(beta0+db).tolist()
        summary['hands'][hand]={'calibration_frames':len(cal),'calibration_views':len(records),'calibration_sync':chosen,
                               'noncalibration_frames':len(other),'before':before,
                               'after':{'loss':float(loss),'soft_iou':float(terms[2]),'marker_term':float(terms[3])},
                               'beta_delta':cpu(db).tolist(),'beta_limit_count':int((db.abs()>=1.499).sum()),
                               'max_calibration_hand_pose_delta_rad':float(dp.abs().max()),
                               'noncalibration_motion_exactly_unchanged':True,'attachments_exactly_unchanged':True,
                               'noncalibration_marker_residual_mm':stats(cpu(residual[other])),
                               'history':history,'elapsed_seconds':time.monotonic()-start}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        shared['provenance']={'source_dataset':str(c['dataset']),'source_fit':str(a.source.resolve()),'full_mask_shape_fit':str(a.output.resolve()),'mesh_accuracy_approved':False}
        (a.output/'shared_calibration.json').write_text(json.dumps(shared,indent=2))
        print(hand,'DONE',summary['hands'][hand]['elapsed_seconds'],flush=True)
    print(a.output,flush=True)


if __name__=='__main__':
    main()
