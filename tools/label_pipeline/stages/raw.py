"""Lossless transfer and the existing native temporal and spatial alignments."""
from pathlib import Path
import subprocess
import sys
from .geometry import EGO
from export_ego_nokov_labels import raw_links

def recording_files(group):
    ego, = (p for p in group.glob('EGO*') if p.is_dir())
    archives = list((ego/'part0001').glob('*.mcap'))
    if len(archives) > 1 or (ego/'part0002').exists():
        raise ValueError(f'需要单分段 EGO 录制 {ego}')
    if archives:
        return archives
    files, imu = raw_links(ego)
    return [*files.values(), imu]

def run(no,cfg,out):
    def call(program,*args):
        subprocess.run([sys.executable,'-B',str(program),*map(str,args)],check=True)
    src=Path(cfg['raw_group']);converted=out/'converted';data=out/'native'
    if no==1:
        if not list(src.glob('EGO*/part0001/*.mcap')):
            files=recording_files(src)
            converted.mkdir(exist_ok=True,parents=True)
            for pattern in ('EGO*','session*'):
                source,=(p for p in src.glob(pattern) if p.is_dir())
                target=converted/source.name
                if not target.exists():target.symlink_to(source.resolve(),target_is_directory=True)
                if not target.samefile(source):raise ValueError(target)
            from .signed import write_json
            note=converted/'conversion.json'
            write_json(note,dict(format='native_mp4',files=[str(p.resolve()) for p in files]))
            return [note]
        call(Path(__file__).with_name('convert.py'),'--group',src,'--output',converted)
        return [converted/'conversion.json']
    ego,=converted.glob('EGO*');session,=converted.glob('session*')
    nokov=session/'nokov';processed=data/'processed';normalized=processed/'ego/normalized_multiview'
    sync=processed/'alignment/synchronization';align=processed/'alignment'
    if no==2:
        dest=processed/'ego_input';dest.mkdir(parents=True,exist_ok=True)
        for name,source in raw_links(ego)[0].items():
            link=dest/name
            if not link.exists():link.symlink_to(source.resolve(strict=True))
        call(EGO/'scripts/normalize_orbbec_ego4_session.py','--session',dest,'--output',normalized)
        imu,=ego.glob('part0001/*_imu_part0001.csv')
        call(Path(__file__).with_name('synchronize.py'),'--ego-imu-csv',imu,'--nokov-csv',nokov/'nokov_rigid_bodies.csv','--rigid-body','orbbec_ego4','--nokov-time-field','device_timestamp_raw','--nokov-time-scale','0.001','--output-dir',sync,'--no-plot')
        return [normalized/'multiview_frames.csv',sync/'imu_nokov_sync.json']
    if no==3:
        call(EGO/'scripts/align_native_dual_nokov.py','--dataset',normalized,'--nokov',nokov,'--sync',sync/'imu_nokov_sync.json','--seed-alignment',cfg['initial_alignment'],'--output',align)
        call(EGO/'scripts/native_fourview_alignment.py','--dataset',normalized,'--aligned-nokov',align/'aligned_nokov21.npz','--seed-alignment',align/'dual_alignment.json','--max-calibration-frames','180','--output',data/'fourview_alignment')
        return [data/'fourview_alignment/fourview_alignment.json',data/'fourview_alignment/aligned_nokov21.npz']
    raise ValueError('原始重拟合阶段须先通过 005 回归验收。')
