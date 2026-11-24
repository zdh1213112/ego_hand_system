#!/usr/bin/env python3
"""Label experimental artifacts accurately and save reproducibility metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--context', type=Path, required=True)
    a = ap.parse_args()
    selection = json.loads((a.output / 'selection.json').read_text())
    fit = a.output / 'fit'
    review = fit / 'review'
    html_path = review / 'index.html'
    html = html_path.read_text().replace('<h1>原图与 MANO Mesh 标签检查</h1>', '<h1>优质关键帧标定后固定手型 · KEYFRAME-CAL-3</h1>')
    html = html.replace('当前基线', '原始基线：NOKOV21 拟合').replace('24 点附着模型 · 实验结果', '优质关键帧标定＋固定手型／附着')
    html = html.replace('原图 / 基线 / 新方案三栏同步缩放。此页为实验结果，未通过验收；包含未满足完整标签导出条件的局部视图。',
                        '左右手各 16 个关键时刻标定共享手型和附着；本页只看未参加标定的时刻。其他帧用 NOKOV24 拟合动作，不用图像调整手型。原图保留畸变。结果尚未批准为训练真值；包含局部出画视图。')
    html = html.replace('src="data.js"', 'src="data.js?v=keyframe-cal-3"')
    html = html.replace('<div id="help">', '<div class="bar"><a href="../../selection_review/">查看入选样本和剔除原因</a> · <a href="../../validation.json">留出验证与稳定性报告</a> · <a href="../shared_calibration.json">固定标定参数</a></div><div id="help">')
    # Relative artifact links also work when review is served through the short alias.
    html = html.replace('</html>', "<script>if(location.pathname.startsWith('/keyframe_calibration/')){for(const a of document.querySelectorAll('a')){const h=a.getAttribute('href');if(h==='../../selection_review/')a.href='/quality_calibration/selection_review/';if(h==='../../validation.json')a.href='/quality_calibration/validation.json';if(h==='../shared_calibration.json')a.href='/quality_calibration/fit/shared_calibration.json';}}</script></html>")
    html = html.replace("if(!viewInitialized||!$('keepZoom')?.checked)fit();else draw();", "if(!viewInitialized){fit(f.hands.some(h=>h.side===1)?1:undefined);setTimeout(()=>{if(seq===token)fit(f.hands.some(h=>h.side===1)?1:undefined)},250)}else if(!$('keepZoom')?.checked)fit();else draw();")
    html_path.write_text(html)
    js = (review / 'data.js').read_text().replace('24-marker pilot: validation frames, NOT approved labels', 'Quality-keyframe calibration: NON-calibration frames; NOT approved ground truth')
    (review / 'data.js').write_text(js)
    # Also document the three-panel video legend with clear user-facing names.
    (review / 'VIDEO_LEGEND_zh.md').write_text('视频三栏从左到右：原图、原始 NOKOV21 基线、优质关键帧标定后固定手型和附着的结果。MARKER24 PILOT 为第三栏的旧通用渲染器标题。本页不含参与共享标定的时刻。\n')
    provenance = {'context': str(a.context.resolve()), 'context_sha256': hashlib.sha256(a.context.read_bytes()).hexdigest(),
                  'selection_sha256': hashlib.sha256((a.output / 'selection.json').read_bytes()).hexdigest(),
                  'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                  'native_wilor_predictions_used': False, 'fixed_2mm_subtraction_applied': False,
                  'shared_parameters_frozen_during_other_frame_fitting': True,
                  'approved_for_training': False, 'checks': {}}
    holdout = selection['report']['holdout']
    for side, hand in enumerate(['left', 'right']):
        z = np.load(fit / f'{hand}_pilot.npz')
        q = selection['report']['hands'][hand]
        chosen = set(q['calibration_sync'])
        assert not any(holdout[0] <= sync < holdout[1] for sync in chosen)
        assert all(sum(r['sync'] == sync for r in q['calibration_views']) >= 2 for sync in chosen)
        assert np.isfinite(z['vertices']).all() and z['betas'].shape == (10,)
        assert z['barycentric'].shape == (21, 3) and np.allclose(z['barycentric'].sum(-1), 1)
        assert np.array_equal(z['calibration_mask'], np.isin(z['pair_indices'], list(chosen)))
        noncal = ~z['calibration_mask']
        residual = z['marker_error_mm'][noncal]
        provenance['checks'][hand] = {'keyframe_count': len(chosen), 'shared_beta_vectors': 1,
                                     'noncalibration_frames': int(noncal.sum()), 'finite_mesh': True,
                                     'finger_surface_marker_median_mm': float(np.median(residual[:, 3:])),
                                     'finger_surface_marker_p95_mm': float(np.percentile(residual[:, 3:], 95)),
                                     'wrist_marker_median_mm': float(np.median(residual[:, :3]))}
    # A known contaminated contour must never reenter the shared calibration.
    known = [r for r in selection['records'] if r['camera'] == 'camera0' and r['sync'] == 632 and r['side'] == 1]
    assert len(known) == 1 and not known[0]['accepted_quality']
    provenance['checks']['known_background_contour_632_right_rejected'] = True
    (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance['checks'], indent=2))


if __name__ == '__main__':
    main()
