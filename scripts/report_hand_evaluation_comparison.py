#!/usr/bin/env python3
"""Verify and present completed single-view and shared multiview evaluations."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

BACKENDS = ('wilor', 'gem', 'fah')
NAMES = {'wilor': 'WiLoR', 'gem': 'GEM-X', 'fah': 'FAH-S'}
MODES = {'single': '单视角独立推理', 'shared': '多视角融合'}
METRICS = {'root_relative': 'root_relative_mpjpe_mm', 'pa': 'pa_mpjpe_mm',
           'absolute': 'absolute_mpjpe_mm', 'pixel': 'mean_2d_error_px'}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_summary(backend, mode, summary):
    return dict(backend=backend, mode=mode, samples=summary['samples'],
        accepted=summary['pose_success']['count'], coverage=summary['pose_success']['rate'],
        **{name: value['mean'] for name, value in summary['metrics'].items()},
        **{f'{kind}_pck_{threshold}': value for kind in METRICS
           for threshold, value in summary.get(kind, {}).get('pck', {}).items()},
        **{f'{kind}_auc_0_50mm': summary[kind]['auc_0_50mm'] for kind in METRICS if kind != 'pixel'})


def load_and_verify(directory):
    summary = read_json(directory / 'summary.json')
    with np.load(directory / 'joint_errors.npz', allow_pickle=False) as value:
        errors = {key: value[key].copy() for key in value.files}
    with (directory / 'per_sample.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_array_equal(errors['ids'], [int(r['index']) for r in rows])
    assert len(rows) == summary['overall']['samples']
    assert len(set(errors['ids'])) == len(rows)
    assert int(errors['valid'].sum()) == summary['overall']['labeled_joints']
    for flag in ('pose_success', 'placement_success', 'pa_success'):
        count = sum(r[flag] == 'True' for r in rows)
        assert count == summary['overall'][flag]['count']
        np.testing.assert_allclose(count / len(rows), summary['overall'][flag]['rate'])
    for column, reference in summary['overall']['metrics'].items():
        values = np.asarray([float(r[column]) for r in rows if r[column]], dtype=float)
        assert np.isfinite(values).all()
        assert len(values) == reference['count']
        if len(values):
            np.testing.assert_allclose(values.mean(), reference['mean'], atol=1e-9, rtol=1e-10)
    for kind, column in METRICS.items():
        values = errors[kind]
        valid = errors['valid']
        per_hand = np.asarray([v[m & np.isfinite(v)].mean() if np.any(m & np.isfinite(v)) else np.nan
                               for v, m in zip(values, valid)])
        expected = np.asarray([float(r[column]) if r[column] else np.nan for r in rows])
        np.testing.assert_allclose(per_hand, expected, atol=1e-8, rtol=1e-10, equal_nan=True)
        flat = values[valid]
        for threshold, pck in summary['overall'][kind]['pck'].items():
            np.testing.assert_allclose(np.count_nonzero(flat <= float(threshold)) / len(flat), pck)
        if kind != 'pixel':
            auc = np.where(np.isfinite(flat), np.clip(1 - flat / 50, 0, 1), 0).mean()
            np.testing.assert_allclose(auc, summary['overall'][kind]['auc_0_50mm'])
    if 'shared_evaluation' in summary:
        status = summary['shared_evaluation']['status_counts']
        assert sum(status.values()) == len(rows)
        assert status.get('accepted', 0) == summary['overall']['pose_success']['count']
        if status.get('not_in_run', 0) or status.get('not_processed', 0):
            raise ValueError(f'incomplete recording coverage: {directory}')
    provenance = {name: digest(directory / name) for name in
                  ('summary.json', 'joint_errors.npz', 'per_sample.csv', 'run_config.json')}
    return summary, errors, provenance


def table(columns, rows):
    def formatted(key, value):
        if value is None:
            return '不适用'
        if isinstance(value, float):
            return f'{value:.4f}' if 'auc_' in key else f'{value:.2f}'
        return str(value)
    header = ''.join(f'<th>{html.escape(label)}</th>' for _, label in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(formatted(key, row.get(key)))}</td>'
                                  for key, _ in columns) + '</tr>' for row in rows)
    return f'<div class="table"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def display_rows(rows):
    return [{**r, 'backend': NAMES[r['backend']], 'mode': MODES.get(r.get('mode'), r.get('mode')),
             'coverage_percent': r.get('coverage', 0) * 100} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-root', required=True, type=Path)
    parser.add_argument('--shared-root', required=True, type=Path,
                        help='comparison root containing backend/summary.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, arrays, sources = {}, {}, {}
    overall, by_camera, unique, mesh = [], [], [], []
    fingerprint = None
    reference = None
    for backend in BACKENDS:
        for mode, root in (('single', args.single_root), ('shared', args.shared_root)):
            directory = (root / backend).resolve()
            summary, errors, hashes = load_and_verify(directory)
            if summary['backend'] != backend:
                raise ValueError(f'backend mismatch: {directory}')
            if reference is None:
                reference = errors
                fingerprint = summary['dataset']['fingerprint']
            assert fingerprint == summary['dataset']['fingerprint']
            np.testing.assert_array_equal(reference['ids'], errors['ids'])
            np.testing.assert_array_equal(reference['valid'], errors['valid'])
            summaries[backend, mode], arrays[backend, mode] = summary, errors
            sources[f'{backend}/{mode}'] = dict(directory=str(directory), sha256=hashes)
            overall.append(row_summary(backend, mode, summary['overall']))
            for camera, values in summary['by']['camera'].items():
                by_camera.append(dict(camera=camera, **row_summary(backend, mode, values)))
            mesh.append(dict(backend=backend, mode=mode, **{
                k: v['mean'] for k, v in summary['overall']['metrics'].items() if 'mpvpe' in k}))
            if mode == 'shared':
                unique.append(row_summary(backend, mode, summary['unique_hand_frames']))

    valid = reference['valid']
    common_all = np.ones(len(valid), bool)
    for data in arrays.values():
        common_all &= np.all(np.isfinite(data['root_relative']) | ~valid, axis=1)
    paired, common_rows, paired_ids = [], [], {}
    for backend in BACKENDS:
        first, second = arrays[backend, 'single'], arrays[backend, 'shared']
        common = np.all((np.isfinite(first['root_relative']) & np.isfinite(second['root_relative'])) | ~valid, axis=1)
        if not common.any():
            raise ValueError(f'no paired samples for {backend}')
        paired_ids[backend] = reference['ids'][common].tolist()
        row = dict(backend=backend, common_samples=int(common.sum()))
        for kind, metric in METRICS.items():
            for mode, data in (('single', first), ('shared', second)):
                values = data[kind][common][valid[common]]
                values = values[np.isfinite(values)]
                row[f'{mode}_{metric}'] = float(values.mean()) if len(values) else None
            if row[f'single_{metric}'] is not None and row[f'shared_{metric}'] is not None:
                row[f'delta_{metric}'] = row[f'shared_{metric}'] - row[f'single_{metric}']
        paired.append(row)
        for mode in MODES:
            data = arrays[backend, mode]
            common_row = dict(backend=backend, mode=mode, common_samples=int(common_all.sum()))
            for kind, metric in METRICS.items():
                values = data[kind][common_all][valid[common_all]]
                values = values[np.isfinite(values)]
                common_row[metric] = float(values.mean()) if len(values) else None
            common_rows.append(common_row)

    result = dict(generated_at=datetime.now().astimezone().isoformat(), dataset=summaries['wilor', 'single']['dataset'],
                  overall=overall, paired=paired, all_six_common_samples=int(common_all.sum()),
                  all_six_common=common_rows, by_camera=by_camera, unique_hand_frames=unique, mesh=mesh,
                  sources=sources,
                  protocols={mode: summaries['wilor', mode]['protocol'] for mode in MODES})
    write_json(args.output / 'summary.json', result)
    write_json(args.output / 'paired_sample_ids.json', dict(per_backend=paired_ids, all_six=reference['ids'][common_all].tolist()))
    for name, rows in [('overall', overall), ('paired', paired), ('all_six_common', common_rows),
                       ('by_camera', by_camera), ('unique_hand_frames', unique), ('mesh', mesh)]:
        write_csv(args.output / f'{name}.csv', rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    thresholds = np.linspace(0, 50, 501)
    for ax, kind, title in zip(axes, ('root_relative', 'absolute'), ('Wrist-aligned 3D PCK', 'Absolute 3D PCK')):
        for backend, color in zip(BACKENDS, ('#2166ac', '#d8781c', '#31945f')):
            for mode, linestyle in (('single', '--'), ('shared', '-')):
                flat = arrays[backend, mode][kind][valid]
                finite = np.sort(flat[np.isfinite(flat)])
                curve = np.searchsorted(finite, thresholds, side='right') / len(flat)
                ax.plot(thresholds, curve * 100, color=color, linestyle=linestyle,
                        label=f'{NAMES[backend]} / {mode}', linewidth=1.8)
        ax.set(title=title, xlabel='Threshold (mm)', ylabel='PCK (%)', xlim=(0, 50), ylim=(0, 100))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('All labeled joints; missing shared predictions count as failures', fontsize=11)
    fig.savefig(args.output / 'pck_curves.png', dpi=180)
    fig.savefig(args.output / 'pck_curves.svg')
    plt.close(fig)

    columns = [('backend', '算法'), ('mode', '方式'), ('accepted', '有结果样本数'),
               ('coverage_percent', '覆盖率 %'), ('root_relative_mpjpe_mm', '腕部对齐 MPJPE / mm'),
               ('pa_mpjpe_mm', 'PA-MPJPE / mm'), ('absolute_mpjpe_mm', '绝对 MPJPE / mm'),
               ('mean_2d_error_px', '二维误差 / px')]
    paired_columns = [('backend', '算法'), ('metric', '指标'), ('common_samples', '共同成功样本数'),
                      ('single_mm', '单视角 / mm'), ('shared_mm', '融合 / mm'), ('delta_mm', '融合 − 单视角 / mm')]
    paired_display = []
    for row in paired:
        for kind, label in [('root_relative', '腕部对齐 MPJPE'), ('pa', 'PA-MPJPE'), ('absolute', '绝对 MPJPE')]:
            metric = METRICS[kind]
            paired_display.append(dict(backend=row['backend'], metric=label, common_samples=row['common_samples'],
                single_mm=row[f'single_{metric}'], shared_mm=row[f'shared_{metric}'], delta_mm=row[f'delta_{metric}']))
    body = '<h1>单视角与多视角融合精度评估</h1>'
    body += (f'<p class="intro">全量评估完成 · {len(valid):,} 份相机／手标签 · '
             f'{result["dataset"]["selected_images"]:,} 张标签图像 · WiLoR / GEM-X / FAH-S</p>')
    body += '<p class="note">单视角在四路图像上分别独立推理，使用 GT 框和左右手；融合使用完整原始录像、检测器和几何关联。'
    body += '主表均值只统计有预测的样本；覆盖率、PCK 和 AUC 保留失败样本。GT 是拟合参考标签，不是直接测量的 21 关节点。</p>'
    body += f'<h2>全量结果</h2><p>每行的评估范围均为 {len(valid):,} 份标签。误差越小越好，覆盖率越高越好。数值按相机／手标签等权。</p>'
    body += table(columns, display_rows(overall))
    body += '<h2>相同样本上的单视角／融合配对比较</h2><p>每个算法只选择两种方式均成功的标签；不同算法的配对集合不同。'
    body += '差值小于零表示该项误差降低。由于输入、检测框、左右手关联和位置求解协议不同，不能将差值全部归因于融合。</p>'
    body += table(paired_columns, display_rows(paired_display))
    body += f'<h2>六组结果共同成功的 {int(common_all.sum()):,} 份标签</h2><p>以下六行使用完全相同的样本 ID；仍保留两种流程的协议差异。</p>'
    body += table([columns[0], columns[1], ('common_samples', '共同样本数'), *columns[4:]], display_rows(common_rows))
    body += '<h2>全量 PCK 曲线</h2><p>分母包含全部有效标签关节，缺失预测计为失败。虚线为单视角，实线为多视角融合。</p><img src="pck_curves.png" alt="腕部对齐及绝对三维 PCK 曲线，缺失预测计为失败">'
    pck_rows = [{**r, **{f'{kind}_pck20_percent': r[f'{kind}_pck_20'] * 100 for kind in ('root_relative', 'absolute', 'pa')}} for r in overall]
    pck_columns = [columns[0], columns[1]]
    for kind, label in [('root_relative', '腕部对齐'), ('absolute', '绝对'), ('pa', 'PA')]:
        pck_columns.extend([(f'{kind}_pck20_percent', f'{label} PCK@20mm %'), (f'{kind}_auc_0_50mm', f'{label} AUC 0–50')])
    body += '<p>PCK／AUC 越大越好；AUC 已归一化到 [0,1]。</p>' + table(pck_columns, display_rows(pck_rows))
    body += '<h2>各相机结果</h2>' + table([('camera', '相机'), *columns], display_rows(by_camera))
    body += f'<h2>物理手／时刻去重后的融合结果</h2><p>按时间戳与物理左右手去重，每个算法的分母均为 {unique[0]["samples"]:,} 个手／时刻。</p>'
    body += table([*columns[:4], *columns[4:7]], display_rows(unique))
    body += '<h2>网格误差</h2><p>WiLoR／FAH 共享 MANO，GEM 共享 SOMA。SOMA 与 MANO 顶点没有逐点对应，因此 GEM 网格误差不适用。'
    body += '单视角 FAH 的拟合网格与关节来自不同输出字段，网格离群误差已保留。</p>'
    body += table([*columns[:2], ('absolute_mpvpe_mm', '绝对 MPVPE / mm'), ('root_relative_mpvpe_mm', '腕部对齐 MPVPE / mm'),
                   ('pa_mpvpe_joint_aligned_mm', '关节 PA 变换后 MPVPE / mm')], display_rows(mesh))
    body += '<h2>指标与来源</h2><ul><li>单视角绝对位置：根据网络自己的二维预测与标定求解平移；融合绝对位置：直接使用保存的共享三维结果，不重新拟合。</li>'
    body += '<li>单视角二维指标来自网络二维预测；融合二维指标来自共享三维模型通过真实镜头的投影。它们衡量不同输出。</li>'
    body += '<li>PA 允许旋转、平移和尺度对齐，不允许反射；腕部对齐指标包含腕部的零误差。</li>'
    body += '<li>所有标签均已匹配到录像时刻，无未处理或未匹配标签。融合拒绝或没有共享手模型的样本仍保留为失败。</li></ul>'
    body += '<p>下载：<a href="overall.csv">全量结果 CSV</a> · <a href="paired.csv">配对结果 CSV</a> · <a href="by_camera.csv">各相机 CSV</a> · '
    body += '<a href="summary.json">完整 JSON</a> · <a href="verification.json">一致性验证</a> · <a href="pck_curves.svg">PCK 矢量图</a></p>'
    body += '<details><summary>原始报告路径与 SHA256</summary><pre>' + html.escape(json.dumps(sources, indent=2)) + '</pre></details>'
    css = 'body{max-width:1320px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui,sans-serif;color:#192f43;background:#f8fafc}h1{font-size:30px}h2{margin-top:36px}.intro{color:#486581}.note{background:#eaf2f8;border-left:4px solid #2166ac;padding:16px}.table{overflow:auto;background:white;border:1px solid #d9e2ec;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px 12px;border-bottom:1px solid #e4eaf0;text-align:right;white-space:nowrap}th{background:#edf3f8}td:first-child,th:first-child{text-align:left}img{max-width:100%;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#185f9b}@media print{body{margin:0;padding:0;background:white}.table{overflow:visible}table{font-size:10px}td,th{padding:5px}h2{break-after:avoid}}'
    (args.output / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单视角与多视角融合精度评估</title><style>' + css + '</style><body>' + body + '</body></html>')
    write_json(args.output / 'verification.json', dict(status='passed', sources=sources,
        evaluations_verified=6, samples_each=len(valid), identical_sample_ids_and_masks=True,
        all_label_timestamps_matched=True, summary_matches_per_sample_and_joint_errors=True,
        coverage_pck_auc_recomputed=True, all_six_common_samples=int(common_all.sum()),
        builder_sha256=digest(Path(__file__))))
    print(json.dumps(dict(output=str(args.output.resolve()), evaluations=6, samples_each=len(valid),
                          all_six_common_samples=int(common_all.sum()), verified=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
