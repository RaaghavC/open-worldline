"""Bounded NumPy-only artifact reads. Does not import producer/model code."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import numpy as np


def need(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parse(data):
    def pairs(rows):
        out = {}
        for k, v in rows:
            need(k not in out, 'Duplicate JSON key')
            out[k] = v
        return out
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def path(root, name):
    rel = PurePosixPath(name)
    need(not rel.is_absolute() and rel.parts and rel.as_posix() == name and
         '..' not in rel.parts and '\\' not in name and '\0' not in name, 'Restricted artifact path')
    p = Path(root)
    for part in rel.parts:
        p = p / part
        need(not p.is_symlink(), 'No artifact symlinks')
    return p


def read(p):
    p = Path(p)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 8 * 2**20, 'Bounded regular JSON')
    return parse(p.read_bytes())


def record(p):
    p = Path(p)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size < 100_000_000, 'Bounded regular artifact')
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}


def inventory(root):
    out = {}
    for p in sorted(Path(root).rglob('*')):
        rel = p.relative_to(root).as_posix()
        path(root, rel)
        if not p.is_dir():
            out[rel] = record(p)
    return out


def tensor_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def tensors(p, expected=None):
    p = Path(p)
    need(8 < record(p)['bytes'] <= 64 * 2**20, 'Bounded tensor file')
    with p.open('rb') as f:
        hsize = struct.unpack('<Q', f.read(8))[0]
        need(2 <= hsize <= 65536, 'Bounded tensor header')
        header = parse(f.read(hsize))
    header.pop('__metadata__', None)
    if expected is not None:
        need(set(header) == set(expected), 'Exact tensor names')
    spans = []
    for name, r in header.items():
        need(r['dtype'] in ('F32', 'I64', 'U8') and isinstance(r['shape'], list) and
             all(type(v) is int and v >= 0 for v in r['shape']), 'Tensor dtype/shape')
        if expected is not None:
            need({k: r[k] for k in ('shape', 'dtype')} == expected[name], 'Exact tensor dimensions/dtype')
        dtype = {'F32': '<f4', 'I64': '<i8', 'U8': 'u1'}[r['dtype']]
        a, b = r['data_offsets']
        need(type(a) is int and type(b) is int and 0 <= a <= b and
             b - a == math.prod(r['shape']) * np.dtype(dtype).itemsize, 'Tensor span')
        spans.append((a, b))
    end = 0
    for a, b in sorted(spans):
        need(a == end, 'Contiguous tensor payload')
        end = b
    need(8 + hsize + end == p.stat().st_size, 'Exact tensor file bytes')
    out = {}
    for name, r in header.items():
        dtype = {'F32': '<f4', 'I64': '<i8', 'U8': 'u1'}[r['dtype']]
        a, b = r['data_offsets']
        with p.open('rb') as f:
            f.seek(8 + hsize + a)
            x = np.frombuffer(f.read(b - a), dtype=dtype).reshape(r['shape'])
        need(np.isfinite(x).all(), 'Finite saved tensor: ' + name)
        out[name] = x
    return out
