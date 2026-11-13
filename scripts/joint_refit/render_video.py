#!/usr/bin/env python3
"""Per-camera review videos: raw | previous label | new label for both hands on every fitted frame."""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np, torch
PROJECT = Path(__file__).resolve().parents[2]; ROOT = PROJECT.parent / 'wilor_training'  # shared fitting tools live in wilor_training/tools
sys.path[:0] = [str(ROOT / 'tools/full_silhouette'), str(ROOT / 'tools/personal_hand_calibration')]
from render_masks import project_native
from context import load_context

def main():
    ap = argparse.ArgumentParser()
    for n in ['context', 'selection', 'previous', 'result', 'output']: ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--size', type=int, default=400); ap.add_argument('--fps', type=float, default=3.75)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); c = load_context(a.context); S = a.size
    recs = json.loads(a.selection.read_text())['records']; cams = {}
    for r in recs: cams.setdefault(r['camera'], r)
    rows = list(csv.DictReader((c['dataset'] / 'multiview_frames.csv').open()))
    present = [h for h in ['left', 'right'] if (a.result / f'{h}_pilot.npz').is_file()]
    if not present: raise ValueError('result has no hand fit')
    fits = [[np.load(d / f'{h}_pilot.npz') if (d / f'{h}_pilot.npz').is_file() else None for h in ['left', 'right']] for d in [a.previous, a.result]]
    syncs = sorted(set().union(*[set(f['pair_indices'].tolist()) for f in fits[1] if f is not None]))
    for cam in sorted(cams):
        r = cams[cam]; cap = cv2.VideoCapture(str(c['dataset'] / 'cameras' / cam / 'video.mp4')); tmp = a.output / f'{cam}_tmp.mp4'
        W = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (3 * S, 2 * S)); pos = -1; written = 0
        for s in syncs:
            fi = int(rows[s][f'{cam}_frame_index'])
            if fi != pos + 1: cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read(); pos = fi
            if not ok: continue
            canvas = np.zeros((2 * S, 3 * S, 3), np.uint8); any_hand = False
            for side in [0, 1]:
                z1 = fits[1][side]
                if z1 is None: continue
                i = np.where(z1['pair_indices'] == s)[0]
                if not len(i): continue
                uv, depth = project_native(torch.as_tensor(z1['vertices'][i[0]][None], dtype=torch.float64), [r], side); uv = uv[0].numpy()
                if depth.min() < .05 or uv[:, 0].max() < 0 or uv[:, 1].max() < 0 or uv[:, 0].min() > img.shape[1] or uv[:, 1].min() > img.shape[0]: continue
                lo, hi = uv.min(0) - 50, uv.max(0) + 50; size = max(hi - lo); lo = (lo + hi) / 2 - size / 2; k = S / size
                M = np.array([[k, 0, -lo[0] * k], [0, k, -lo[1] * k]]); raw = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_CUBIC); panels = [raw]
                for z, color in [(fits[0][side], (255, 200, 0)), (z1, (0, 220, 255))]:
                    j = np.where(z['pair_indices'] == s)[0] if z is not None else []; p = raw.copy()
                    if len(j):
                        q, _ = project_native(torch.as_tensor(z['vertices'][j[0]][None], dtype=torch.float64), [r], side); q = (q[0].numpy() - lo) * k
                        sil = np.zeros((S, S), np.uint8)
                        for f in z['faces']: cv2.fillConvexPoly(sil, np.round(q[f]).astype(np.int32), 255)
                        tint = p.copy(); tint[sil > 0] = color; p = cv2.addWeighted(p, .78, tint, .22, 0)
                        cv2.drawContours(p, cv2.findContours(sil, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], -1, color, 1, cv2.LINE_AA)
                    panels.append(p)
                row = np.hstack(panels)
                for x, t in enumerate(['RAW', 'PREVIOUS', 'JOINT REFIT']): cv2.putText(row, f"{['LEFT', 'RIGHT'][side]} {cam} sync{s} {t}", (x * S + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[side * S:(side + 1) * S] = row; any_hand = True
            if any_hand:
                W.write(canvas); written += 1
                if s in (576, 632, 664, 688, 1304): cv2.imwrite(str(a.output / f'{cam}_sync{s}.jpg'), canvas)
        W.release(); cap.release(); out = a.output / f'{cam}_comparison.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(out)], check=True); tmp.unlink(); print(cam, written, out, flush=True)
    (a.output / 'index.html').write_text('<meta charset="utf-8"><title>joint refit</title><body style="background:#111;color:#eee;font-family:sans-serif"><h3>原图 | 之前标签(青) | 联合重拟合(黄)</h3>' + ''.join(f'<p>{cam}</p><video src="{cam}_comparison.mp4" controls loop style="max-width:100%"></video>' for cam in sorted(cams)) + '</body>')
if __name__ == '__main__': main()
