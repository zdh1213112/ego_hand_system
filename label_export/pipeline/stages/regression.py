"""Compare regenerated arrays and independent frozen digests by physical exposure."""
import csv
import json
from collections import Counter
import numpy as np
from .fingerprint import digest
from .storage_digest import canonical_digest
import hashlib

def key(row):
    return int(row['sync_index']),row['camera'],int(row['side'])

def read_index(path):
    with path.open() as f:return {key(r):r for r in map(json.loads,f)}

def compare(candidate,original,prefix=''):
    if type(candidate)!=type(original):
        yield prefix+'.type',None
        return
    if isinstance(original,dict):
        if sorted(candidate)!=sorted(original):yield prefix+'.keys',None
        for k in sorted(candidate.keys() & original.keys()):
            yield from compare(candidate[k],original[k],prefix+'.'+k)
    elif hasattr(original,'shape'):
        a,b=np.asarray(candidate),np.asarray(original)
        if a.dtype!=b.dtype or a.shape!=b.shape:yield prefix+'.dtype_or_shape',None
        elif not np.array_equal(a,b,equal_nan=True):
            v=np.abs(a.astype(float)-b.astype(float))
            yield prefix,float(np.nanmax(v)) if np.isfinite(v).any() else None
    elif candidate!=original:yield prefix,None

def audit(group,out,original,limit=None):
    pred=read_index(out/'index.jsonl');truth=read_index(group/'reference/index.jsonl')
    with (group/'reference/exposure_hashes.csv').open() as f:
        hashes={(int(r['sync']),r['camera'],int(r['side'])):r for r in csv.DictReader(f)}
    if truth.keys()!=hashes.keys():raise ValueError('冻结曝光和文件摘要不匹配')
    counts=Counter();detail={};bad=[];im_cache={}
    shared=sorted(pred.keys()&truth.keys())
    if limit is not None:shared=shared[:limit]
    for k in shared:
        a,b=pred[k],truth[k];ref=hashes[k];errors=[]
        for field in ('single_hand','reference_timestamp_ns','source_frame_index'):
            if a[field]!=b[field]:errors.append((field,None))
        for pathfield,hashfield in [('label','strict_sha256'),('glove_label','surface_sha256'),('image','image_sha256')]:
            p=out/a[pathfield]
            if pathfield=='image':
                if p not in im_cache:im_cache[p]=digest(p)
                h=im_cache[p]
            else:h=digest(p)
            counts[hashfield+'_match']+=int(h==ref[hashfield])
            if pathfield=='image':continue
            actual_bytes=p.read_bytes();signed_bytes=(original/b[pathfield]).read_bytes()
            if hashlib.sha256(signed_bytes).hexdigest()!=ref[hashfield]:raise ValueError(f'签收原件摘要变化 {k} {pathfield}')
            counts[pathfield+'_storage_normalized_sha256_match']+=int(canonical_digest(actual_bytes)==canonical_digest(signed_bytes))
            x=np.load(p,allow_pickle=True).item()
            y=np.load(original/b[pathfield],allow_pickle=True).item()
            errors.extend(compare(x,y,pathfield))
        if errors:
            bad.append(k)
            for field,diff in errors:
                row=detail.setdefault(field,dict(count=0,first=k,maximum=None))
                row['count']+=1
                if diff is not None:row['maximum']=max(diff,row['maximum'] or 0.)
        counts['all_fields_equal']+=int(not errors)
    return dict(group=group.name,expected=len(truth),generated=len(pred),compared=len(shared),missing=sorted(truth.keys()-pred.keys()),extra=sorted(pred.keys()-truth.keys()),counts=dict(counts),field_differences=detail,first_affected_exposures=bad[:10])
