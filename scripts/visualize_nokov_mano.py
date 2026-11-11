#!/usr/bin/env python3
"""Export a standalone marker/MANO viewer, scientific plots, and a video."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

EDGES = [(0, b) for b in (1,5,9,13,17)] + [(b+i,b+i+1) for b in (1,5,9,13,17) for i in range(3)]
OVERLAY_VIEWS = {'4views': '四路拼接', 'cam0_left': '前左相机 cam0',
                 'cam1_right': '前右相机 cam1', 'cam2_right_side': '右侧相机 cam2',
                 'cam3_left_side': '左侧相机 cam3'}


def overlay_media_files(directory, summary, validation):
    """Expose only completed video files with matching encoding metadata."""
    videos = {}
    if summary.get('preview_only', True):
        return videos
    for item in validation.get('encoded_videos', []):
        name = item.get('file', '')
        view = next((key for key in OVERLAY_VIEWS if name == f'mano_overlay_{key}.mp4'), None)
        if view is None or item.get('tail_decodable') is not True:
            continue
        path = directory / name
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            frame_count = int(item.get('nb_frames', 0))
            duration = float(item['duration'])
            width, height = int(item['width']), int(item['height'])
        except (KeyError, TypeError, ValueError):
            continue
        if item.get('bytes') != path.stat().st_size or frame_count != summary.get('frames'):
            continue
        if not np.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
            continue
        videos[view] = dict(label=OVERLAY_VIEWS[view], file=f'{directory.name}/{name}',
                            duration_s=duration, width=width, height=height)
    return videos


def overlay_catalog(directory):
    def read_json(path):
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    variants = {}
    for key, folder, label in [('original', 'video_overlay', '原始估计'),
                               ('aligned', 'video_overlay_aligned', '图像校正')]:
        root = directory / folder
        summary = read_json(root / 'overlay_summary.json')
        validation = read_json(root / 'overlay_validation.json')
        alignment = read_json(root / 'alignment_validation.json') if key == 'aligned' else {}
        if key == 'aligned' and alignment.get('approved_for_default') is not True:
            label = '图像校正（候选）'
        for camera in alignment.get('by_camera', {}).values():
            metrics = camera.get('after_px', {})
            if camera.get('status') in ('validated', 'validated_sparse_scope') and not (
                camera.get('validation_frames') and metrics.get('count', 0) > 0
                and all(isinstance(metrics.get(name), (int, float))
                        and np.isfinite(metrics[name]) and metrics[name] >= 0 for name in ('median', 'p95'))
            ):
                camera['status'] = 'not_validated'
        videos = overlay_media_files(root, summary, validation)
        expected_config = alignment.get('projection_config_sha256')
        if expected_config and (summary.get('projection_config_sha256') != expected_config
                                or validation.get('projection_config_sha256') != expected_config):
            # Recalibration must not relabel old encoded media as the new result.
            videos = {}
        links = {name: f'{folder}/{name}' for name in ['overlay_summary.json', 'overlay_validation.json',
                 'alignment_validation.json', 'frame_associations.csv'] if (root / name).is_file()}
        poster_path = root / 'overlay_4views_frame0120.jpg'
        poster = ('data:image/jpeg;base64,' + base64.b64encode(poster_path.read_bytes()).decode()
                  if poster_path.is_file() else '')
        variants[key] = dict(label=label, videos=videos, summary=summary, alignment=alignment,
                             links=links, poster=poster)
    aligned = variants['aligned']
    verified_views = {f'cam{camera}_{suffix}' for camera, suffix in enumerate(
        ['left', 'right', 'right_side', 'left_side'])
        if aligned['alignment'].get('by_camera', {}).get(str(camera), {}).get('status') in ('validated', 'validated_sparse_scope')}
    accepted = (aligned['alignment'].get('approved_for_default') is True
                and aligned['alignment'].get('status') in ('validated_sparse_scope', 'partial')
                and bool(verified_views.intersection(aligned['videos'])))
    return dict(variants=variants, default='aligned' if accepted else 'original')


def quantized(points, origin):
    finite = np.isfinite(points).all(-1)
    values = np.rint((np.nan_to_num(points)-origin[:,None])/.0001)
    if np.any(np.abs(values[finite]) >= 32767):
        raise ValueError('viewer geometry exceeds the int16 relative coordinate range')
    values[~finite] = -32768
    return base64.b64encode(values.astype('<i2').tobytes()).decode()


def frame_residual_statistics(marker_error_m):
    """Return metre-valued statistics over finite original residuals, or None."""
    statistics = {key: [] for key in ('mean', 'max', 'median', 'p95')}
    for row in np.asarray(marker_error_m, dtype=np.float64):
        valid = row[np.isfinite(row)]
        if valid.size:
            median, p95 = np.percentile(valid, [50, 95])
            values = (float(valid.mean()), float(valid.max()), float(median), float(p95))
        else:
            values = (None, None, None, None)
        for key, value in zip(statistics, values):
            statistics[key].append(value)
    return statistics


def export_html(directory, data, summary):
    ids = np.unique(np.r_[np.arange(0, len(data['frame_no']), 3), len(data['frame_no'])-1])
    origin = data['joints'][ids,0]
    payload = dict(frames=data['frame_no'][ids].tolist(),time=data['time_s'][ids].round(6).tolist(),
                   origin=origin.tolist(),faces=data['faces'].tolist(),edges=EDGES,
                   joint_names=data['joint_names'].tolist(),marker_names=data['marker_names'].tolist(),
                   valid=data['marker_valid'][ids].sum(-1).tolist(),summary=summary)
    payload.update(frame_residual_statistics(data['marker_error_m'][ids]))
    payload['marker_error_m']=base64.b64encode(data['marker_error_m'][ids].astype('<f4').tobytes()).decode()
    payload['attachment_face_ids']=data['attachment_face_ids'].tolist()
    payload['overlays'] = overlay_catalog(directory)
    for key in ('vertices','joints','markers_world_m','predicted_markers'):
        payload[key] = quantized(data[key][ids], origin)
    template = (Path(__file__).parent/'templates/nokov_mano.html').read_text()
    interaction=(Path(__file__).parent/'templates/nokov_marker_interaction.js').read_text()
    page = template.replace('__DATA__',json.dumps(payload,separators=(',',':'),ensure_ascii=False))
    page=page.replace('__MARKER_INTERACTION__',interaction)
    plot_uri='data:image/png;base64,'+base64.b64encode((directory/'fit_residuals.png').read_bytes()).decode()
    page=page.replace('src="fit_residuals.png"',f'src="{plot_uri}"')
    (directory/'mano_viewer.src.html').write_text(page)


def export_plots(directory, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    error = data['marker_error_m']*1000
    fig, ax = plt.subplots(2,1,figsize=(13,7),gridspec_kw={'height_ratios':[1.2,1]})
    ax[0].plot(data['time_s'],np.nanpercentile(error,95,axis=1),c='#9aaabb',lw=.6,label='Frame P95')
    ax[0].plot(data['time_s'],np.nanmedian(error,axis=1),c='#126b80',lw=.7,label='Frame median')
    ax[0].set(xlabel='Time from first NOKOV frame (s)',ylabel='Marker fit residual (mm)',
              title='Right-hand MANO: residual against valid measured markers')
    ax[0].legend(frameon=False); ax[0].set_xlim(data['time_s'][[0,-1]])
    x=np.arange(24)
    ax[1].bar(x,np.nanmedian(error,axis=0),color='#126b80',width=.65,label='Median')
    ax[1].scatter(x,np.nanpercentile(error,95,axis=0),s=14,c='#c36a29',label='P95')
    ax[1].set_xticks(x,data['marker_names'],rotation=45,ha='right',fontsize=8)
    ax[1].set_ylabel('Marker fit residual (mm)'); ax[1].legend(frameon=False)
    for a in ax:
        a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15)
        a.set_ylim(bottom=0)
    fig.tight_layout();fig.savefig(directory/'fit_residuals.png',dpi=160);plt.close(fig)


def camera_matrix(yaw, pitch):
    cy,sy,cp,sp=np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch)
    return np.array([[cy,-sy,0],[sy*sp,cy*sp,cp],[-sy*cp,-cy*cp,sp]])


def render_panel(canvas, vertices, joints, markers, predicted, faces, panel, rotation, title):
    x,y,w,h=panel
    center=joints[0]
    # A fixed metric scale makes fingers comparable throughout the sequence.
    scale=min(w,h)/.32
    def project(points):
        p=(points-center)@rotation.T
        p[:,0]=p[:,0]*scale+x+w/2
        p[:,1]=-p[:,1]*scale+y+h*.72
        return p
    pv,pj,pm,pp=[project(v) for v in (vertices,joints,markers,predicted)]
    tri=vertices[faces]@rotation.T
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    light=np.abs(normal@np.array([.3,.5,.81]))
    color=(np.array([156,177,195])[None]*(.48+.48*light[:,None])).astype(np.uint8)
    pixels=np.rint(pv[:,:2]).astype(np.int32)
    for k in np.argsort(tri[:,:,2].mean(-1)):
        cv2.fillConvexPoly(canvas,pixels[faces[k]],tuple(map(int,color[k])),lineType=cv2.LINE_AA)
    jp=np.rint(pj[:,:2]).astype(np.int32)
    for a,b in EDGES:
        cv2.line(canvas,tuple(jp[a]),tuple(jp[b]),(45,175,235),2,cv2.LINE_AA)
    for j,p in enumerate(jp):
        cv2.circle(canvas,tuple(p),3,(45,175,235),-1,cv2.LINE_AA)
        if w>650:
            cv2.putText(canvas,str(j),tuple(p+[5,-4]),cv2.FONT_HERSHEY_SIMPLEX,.33,(230,232,235),1,cv2.LINE_AA)
    valid=np.isfinite(pm).all(-1)
    for i in np.flatnonzero(valid):
        p=np.rint(pm[i,:2]).astype(int); q=np.rint(pp[i,:2]).astype(int)
        cv2.line(canvas,tuple(p),tuple(q),(150,160,165),1,cv2.LINE_AA)
        cv2.rectangle(canvas,tuple(p-3),tuple(p+3),(145,217,99),-1,cv2.LINE_AA)
    cv2.putText(canvas,title,(x+18,y+26),cv2.FONT_HERSHEY_SIMPLEX,.62,(211,218,225),1,cv2.LINE_AA)
    length=int(.05*scale)
    cv2.line(canvas,(x+25,y+h-25),(x+25+length,y+h-25),(210,218,225),2)
    cv2.putText(canvas,'50 mm',(x+25,y+h-35),cv2.FONT_HERSHEY_SIMPLEX,.42,(210,218,225),1,cv2.LINE_AA)


def export_video(directory,data):
    width,height=1440,900
    path=directory/'mano_reconstruction.mp4'
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
             '-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    rotations=[camera_matrix(-.12,1.12),camera_matrix(-.65,.15)]
    panels=[(15,100,900,700),(930,100,490,700)]
    chosen=[0,int(len(data['frame_no'])*.25),int(len(data['frame_no'])*.5),int(len(data['frame_no'])*.75)]
    chosen=[3*(i//3) for i in chosen]
    for i in range(0,len(data['frame_no']),3):
        canvas=np.full((height,width,3),(24,28,33),np.uint8)
        cv2.putText(canvas,'NOKOV / MANO   |   RIGHT HAND',(35,44),cv2.FONT_HERSHEY_SIMPLEX,.9,(229,234,238),2,cv2.LINE_AA)
        label=f"{data['time_s'][i]:06.2f} s  |  NOKOV frame {data['frame_no'][i]}  |  {data['marker_valid'][i].sum()}/24 observed markers"
        cv2.putText(canvas,label,(35,78),cv2.FONT_HERSHEY_SIMPLEX,.6,(170,183,195),1,cv2.LINE_AA)
        for panel,rot,title in zip(panels,rotations,['21 MANO joints + surface mesh','Side view / metric scale bar below']):
            render_panel(canvas,data['vertices'][i],data['joints'][i],data['markers_world_m'][i],data['predicted_markers'][i],data['faces'],panel,rot,title)
        error=data['marker_error_m'][i]*1000
        cv2.putText(canvas,f'Marker residual: median {np.nanmedian(error):.2f} mm / P95 {np.nanpercentile(error,95):.2f} mm',
                    (35,830),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,225,230),1,cv2.LINE_AA)
        cv2.putText(canvas,'Orange circles: MANO joints   Green squares: measured markers   Grey lines: surface residuals',
                    (35,864),cv2.FONT_HERSHEY_SIMPLEX,.52,(170,183,195),1,cv2.LINE_AA)
        if i in chosen:
            cv2.imwrite(str(directory/f'preview_{i:05d}.png'),canvas)
        proc.stdin.write(canvas.tobytes())
        if i%900==0: print(f'rendered {i}/{len(data["frame_no"])} source frames',flush=True)
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg encoding failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--no-video',action='store_true');a=p.parse_args()
    data=dict(np.load(a.input/'reconstruction.npz'));summary=json.loads((a.input/'summary.json').read_text())
    export_plots(a.input,data);export_html(a.input,data,summary)
    for i in (0,len(data['frame_no'])//2):
        with (a.input/f'mesh_frame_{int(data["frame_no"][i])}.obj').open('w') as f:
            f.write('# MANO right hand, NOKOV world coordinates, metres\n')
            for v in data['vertices'][i]:f.write('v '+' '.join(f'{x:.8f}' for x in v)+'\n')
            for face in data['faces']:f.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
    if not a.no_video:export_video(a.input,data)


if __name__=='__main__':main()
