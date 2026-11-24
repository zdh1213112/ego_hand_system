"""Validate all declared frozen inputs, including symlink targets, before exporting."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def digest(path):
    with path.open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
        return h.hexdigest()

def verify(root):
    manifest=root/'MANIFEST.json'
    data=json.loads(manifest.read_text())
    records=data['files']
    if len(records)!=data['file_count'] or len({r['path'] for r in records})!=len(records):
        raise ValueError('冻结包清单计数或路径重复')
    def one(r):
        rel=Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'冻结清单路径无效 {rel}')
        p=root/rel
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise ValueError(f'冻结输入发生变化 {rel}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(one,records):pass
    return {'manifest_sha256':digest(manifest),'verified':len(records)}
