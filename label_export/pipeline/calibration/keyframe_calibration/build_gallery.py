#!/usr/bin/env python3
"""Publish the selected calibration views with masks, rejection reasons and contact sheets."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REASONS = {
    'background_or_excessive_extent': '疑似连接背景／超出 marker 包络',
    'incomplete_marker_or_fingertip_coverage': '手指或 marker 覆盖不完整',
    'fingertip_overlap_or_occlusion': '指尖重叠／遮挡，不适合标定手型',
    'blur_or_low_detail': '模糊或细节不足',
    'gray_glove_may_continue_outside_mask': '疑似漏掉灰色手套边缘',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.selection.read_text())
    audit = Path(d['report']['source_audit'])
    original = json.loads((audit / 'data.js').read_text().removeprefix('window.CONTOUR_AUDIT=').strip().rstrip(';'))
    lookup = {(r['camera'], r['sync'], r['side']): r for r in original['records']}
    a.output.mkdir(parents=True, exist_ok=True)
    asset = a.output / 'assets'
    if not asset.exists():
        asset.symlink_to(audit.resolve(), target_is_directory=True)
    records = []
    counts = {}
    for r in d['records']:
        f = dict(lookup[(r['camera'], r['sync'], r['side'])])
        hand = ['left', 'right'][r['side']]
        selected = {(v['camera'], v['sync']) for v in d['report']['hands'][hand]['calibration_views']}
        f.update(accepted=r['accepted_quality'], used_in_joint_fit=(r['camera'], r['sync']) in selected,
                 selected_for_calibration=(r['camera'], r['sync']) in selected,
                 metrics=r['metrics'], reason='；'.join(REASONS[x] for x in r['reasons']))
        f['image'] = 'assets/' + f['image']
        if f['mask']:
            f['mask'] = 'assets/' + f['mask']
        records.append(f)
        for reason in r['reasons']:
            counts[REASONS[reason]] = counts.get(REASONS[reason], 0) + 1
    report = dict(observations=str(a.selection.resolve()), hands=d['report']['hands'],
                  original_views=len(original['records']), scored=len(records),
                  good=sum(r['accepted'] for r in records),
                  selected=sum(r['used_in_joint_fit'] for r in records), reasons=counts)
    data = {'report': report, 'records': records}
    (a.output / 'data.js').write_text('window.CONTOUR_AUDIT=' + json.dumps(data, separators=(',', ':')) + ';\n')
    html = (ROOT / 'label_export/pipeline/calibration/joint_pose_attachment/contour_audit.html').read_text()
    html = html.replace('实际优化轮廓检查 · CONTOUR-AUDIT-1', '优质关键帧筛选 · KEYFRAME-CAL-3')
    html = html.replace('实际优化轮廓检查', '优质关键帧筛选')
    html = html.replace('参与本轮联合优化', '用于固定手型标定').replace('全部接受的轮廓', '全部通过质量门槛的轮廓')
    html = html.replace('绿色区域可能包含袖口；只有紫色点参与轮廓损失。本页不显示 MANO 网格。',
                        '仅少量互补关键帧参与固定手型标定；其他帧只用 NOKOV 驱动动作。本页可直接复核原图和分割，不需要逐点点击。自动通过不等于人工真值。')
    html = html.replace("${f.used_in_joint_fit?'参与联合优化':'接受但未参与本轮联合优化'}", "${f.used_in_joint_fit?'用于固定手型标定':'未用于固定手型标定'}")
    start = html.index("$('report').textContent=`总计：")
    end = html.index(';resize();select();', start)
    html = html[:start] + "$('report').textContent=`原始手/视图样本 ${D.report.original_views} 份，原分割通过并可重放 ${D.report.scored} 份；严格筛选保留 ${D.report.good} 份，其中 ${D.report.selected} 份用于标定。左右手各 16 个时间点，每个时间点至少两路相机通过。其余图像不调整固定手型。剔除原因可重叠：${JSON.stringify(D.report.reasons)}`" + html[end:]
    html = html.replace('参与联合优化', '用于固定手型标定')
    html = html.replace('resize();select();', "$('hand').value='0';resize();select(56);")
    (a.output / 'index.html').write_text(html)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    cv2.setNumThreads(2)
    for side, hand in enumerate(['left', 'right']):
        frames = d['report']['hands'][hand]['calibration_sync']
        for rank in [0, 1]:
            tiles = []
            for sync in frames:
                views = sorted([r for r in records if r['side'] == side and r['sync'] == sync and r['used_in_joint_fit']], key=lambda r: r['camera'])
                r = views[rank]
                image = cv2.imread(str(a.output / r['image']))
                alpha = cv2.imread(str(a.output / r['mask']), cv2.IMREAD_UNCHANGED)[:, :, 3] > 0
                overlay = image.copy()
                overlay[alpha] = (.75 * image[alpha] + .25 * np.array([50, 240, 90])).astype(np.uint8)
                x1, y1, x2, y2 = np.rint(r['bbox']).astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
                tile = np.zeros((270, 400, 3), np.uint8)
                for k, im in enumerate([image, overlay]):
                    crop = im[y1:y2, x1:x2]
                    scale = min(200 / crop.shape[1], 240 / crop.shape[0])
                    crop = cv2.resize(crop, None, fx=scale, fy=scale)
                    h, w = crop.shape[:2]
                    tile[30:30+h, k*200:k*200+w] = crop
                cv2.putText(tile, f'{hand} {sync} {r["camera"]}', (5, 20), 0, .5, (255, 255, 255), 1)
                tiles.append(tile)
            sheet = np.concatenate([np.concatenate(tiles[k:k+4], axis=1) for k in range(0, len(tiles), 4)], axis=0)
            cv2.imwrite(str(a.output / f'{hand}_selected_view{rank+1}.jpg'), sheet)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
