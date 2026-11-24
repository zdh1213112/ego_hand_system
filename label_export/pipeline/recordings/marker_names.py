"""Expose descriptor-verified marker identities to readers expecting the original glove set names."""
import json
from pathlib import Path

from stages.fingerprint import digest
from stages.signed import write_json

NAMES = ['WristM','WristIn','WristOut','HandOffset'] + [
    f'Finger{finger}{joint}' for finger in ('Thumb','Index','Middle','Ring','Pinky') for joint in range(1,5)]


def compatible_alignment(root):
    original = root/'native/fourview_alignment/fourview_alignment.json'
    alignment = json.loads(original.read_text())
    if alignment.get('status') != 'accepted':
        raise ValueError('四机空间对齐未通过')
    source = Path(alignment['source_nokov'])
    descriptor = json.loads((source/'asset_descriptions.json').read_text())
    names = descriptor['marker_sets']
    rename = {}
    for hand in ('left','right'):
        found = [k for k in names if k.startswith(hand+'_hand')]
        if len(found)!=1 or names[found[0]]!=NAMES:
            raise ValueError(f'无法逐指验证 {hand} 的 24 个 marker')
        rename[found[0]]=hand+'_hand_4mm'
    if all(k==v for k,v in rename.items()): return original
    dest=root/'native/marker_name_compatibility';dest.mkdir(exist_ok=True)
    proof=dest/'source.json';new_alignment=dest/'fourview_alignment.json'
    if proof.exists():
        stored=json.loads(proof.read_text())
        if stored['original_alignment_sha256']!=digest(original) or stored['renamed_csv_sha256']!=digest(dest/'nokov/nokov_markers.csv'):
            raise ValueError('marker 名称兼容记录和原始输入不符')
        return new_alignment
    out=dest/'nokov';out.mkdir(exist_ok=True)
    for old in source.iterdir():
        dst=out/old.name
        if old.name=='asset_descriptions.json':
            write_json(dst,dict(descriptor,marker_sets={rename.get(k,k):v for k,v in names.items()}))
        elif old.name=='nokov_markers.csv':
            # Only the exact CSV field token changes. Coordinates, index, clock and validity retain their original bytes.
            repl={b','+k.encode()+b',':b','+v.encode()+b',' for k,v in rename.items()}
            temp=dst.with_suffix('.tmp')
            counts={k:0 for k in repl}
            with old.open('rb') as ins,temp.open('wb') as outs:
                for line in ins:
                    for k,v in repl.items():
                        if k in line: counts[k]+=1;line=line.replace(k,v)
                    outs.write(line)
            if min(counts.values())<1:raise ValueError('未找到录制中的手套名称')
            temp.replace(dst)
        elif not dst.exists():dst.symlink_to(old)
    write_json(new_alignment,dict(alignment,source_nokov=str(out.resolve())))
    write_json(proof,dict(original_alignment=str(original),original_alignment_sha256=digest(original),
        original_nokov=str(source),marker_set_aliases=rename,marker_names=NAMES,
        original_csv_sha256=digest(source/'nokov_markers.csv'),renamed_csv_sha256=digest(out/'nokov_markers.csv'),
        renamed_rows={k.decode().strip(','):v for k,v in counts.items()}))
    return new_alignment
