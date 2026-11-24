"""Normalize only the opaque PyTorch storage identifier and its encoded lengths."""
import hashlib
import pickletools
import re
storage=re.compile(rb'(ctorch\nFloatStorage\nq.)(X)(.{4})([0-9]{8,19})(q.)(X\x03\x00\x00\x00cpu)',re.S)
def mask(blob):
    match=storage.search(blob)
    if not match:raise ValueError('expected torch FloatStorage')
    key=match[4]
    token=b'X'+len(key).to_bytes(4,'little')+key
    if blob.count(token)!=2:raise ValueError('unexpected number of storage keys')
    # Normalize key length and the two enclosing structural lengths as well.
    off=blob.index(b'\x80\x04',100,256)
    if blob[off+2]!=0x95:raise ValueError('unexpected pickle frame')
    ops=list(pickletools.genops(blob[off:]))
    [(payload,pos)]=[(arg,pos+off) for op,arg,pos in ops if op.name=='BINBYTES' and token in arg and b'FloatStorage' in arg]
    standard=b'X\x01\x00\x00\x000'
    delta=2*(len(standard)-len(token))
    data=bytearray(blob)
    frame=off+3
    data[frame:frame+8]=(int.from_bytes(data[frame:frame+8],'little')+delta).to_bytes(8,'little')
    data[pos+1:pos+5]=(len(payload)+delta).to_bytes(4,'little')
    return bytes(data).replace(token,standard)

def canonical_digest(blob):
    return hashlib.sha256(mask(blob)).hexdigest()
