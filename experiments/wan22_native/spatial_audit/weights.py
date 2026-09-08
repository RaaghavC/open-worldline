# SPDX-License-Identifier: Apache-2.0
"""Compare VAE load records with original ZIP storage bytes, without Torch."""
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import zipfile

VAE_SHA = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
VAE_BYTES = 2818839170
PARAMETERS = 704688668
CONFIG = {'dim': 160, 'dec_dim': 256, 'z_dim': 48, 'dim_mult': [1, 2, 4, 4],
          'num_res_blocks': 2, 'attn_scales': [], 'temperal_downsample': [False, True, True], 'dropout': 0.0}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def regular(path):
    path = Path(path).absolute()
    if not path.is_file() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('A regular file without symlink ancestors is required')
    return path


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON keys are not accepted')
        result[key] = value
    return result


def _invalid(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def record(path):
    path = regular(path)
    if path.stat().st_size > 2**20:
        raise ValueError('Bounded decoder load record required')
    value = json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=_invalid)
    if (not isinstance(value, dict) or value.get('weight_sha256') != VAE_SHA
            or value.get('compute_dtype') != 'float32' or value.get('parameters') != PARAMETERS
            or value.get('config') != CONFIG or not isinstance(value.get('tensors'), dict)
            or len(value['tensors']) != 196):
        raise ValueError('Original FP32 native decoder load record required')
    for name, row in value['tensors'].items():
        if (not isinstance(name, str) or not name or not isinstance(row, dict)
                or row.get('cuda_copy_exact') is not True):
            raise ValueError('Every original decoder copy record is required')
        shape, digest = row.get('shape'), row.get('sha256')
        if (not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape)
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Exact positive shapes and SHA256 records required')
    if sum(math.prod(row['shape']) for row in value['tensors'].values()) != PARAMETERS:
        raise ValueError('Decoder shapes do not sum to the original parameter count')
    return value


def _rebuilt(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    if (not isinstance(storage, dict) or set(storage) != {'storage_key', 'elements'}
            or type(offset) is not int or offset != 0
            or not isinstance(size, tuple) or not isinstance(stride, tuple) or len(size) != len(stride)
            or not size or any(type(x) is not int or x <= 0 for x in size)
            or any(type(x) is not int or x <= 0 for x in stride)
            or type(requires_grad) is not bool):
        raise ValueError('Only contiguous original tensor metadata is accepted')
    expected = []; total = 1
    for dimension in reversed(size):
        expected.append(total); total *= dimension
    if stride != tuple(reversed(expected)) or total != storage['elements']:
        raise ValueError('Storage length or contiguous strides differ')
    return dict(storage, shape=list(size))


class MetadataOnly(pickle.Unpickler):
    """Resolve three inert metadata constructs, never model or tensor classes."""
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return _rebuilt
        if (module, name) == ('torch', 'FloatStorage'):
            return 'FLOAT32_STORAGE_TAG'
        raise ValueError('Unapproved pickle global in checkpoint metadata')

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or value[0] != 'storage'
                or value[1] != 'FLOAT32_STORAGE_TAG' or not isinstance(value[2], str)
                or not value[2].isdigit() or not isinstance(value[3], str)
                or type(value[4]) is not int or value[4] <= 0 or value[4] > PARAMETERS):
            raise ValueError('Only bounded original FP32 storage metadata is accepted')
        return {'storage_key': value[2], 'elements': value[4]}


def read_metadata(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate checkpoint archive names are not accepted')
    candidates = [name for name in names if name.endswith('/data.pkl')]
    if len(candidates) != 1:
        raise ValueError('Exactly one original metadata entry required')
    name = candidates[0]
    if archive.getinfo(name).file_size > 2**20:
        raise ValueError('Checkpoint metadata exceeds its size bound')
    stream = io.BytesIO(archive.read(name)); value = MetadataOnly(stream).load()
    if stream.read(1) or not isinstance(value, (dict, OrderedDict)):
        raise ValueError('One complete original parameter mapping required')
    return name[:-len('data.pkl')], value


def compare_storages(archive, prefix, metadata, rows):
    """Check contiguous raw storage bytes against rows; no numeric conversion."""
    if set(metadata) != set(rows) or len({v['storage_key'] for v in metadata.values()}) != len(metadata):
        raise ValueError('Exact independent original parameter/storage identities required')
    checked = {}; total = 0
    for name, specification in metadata.items():
        row = rows[name]
        if specification['shape'] != row['shape'] or row.get('cuda_copy_exact') is not True:
            raise ValueError('Original storage shape or CUDA copy declaration differs: ' + name)
        target = prefix + 'data/' + specification['storage_key']
        size = specification['elements'] * 4
        if archive.getinfo(target).file_size != size:
            raise ValueError('Original storage byte length differs: ' + name)
        digest = hashlib.sha256(); observed = 0
        with archive.open(target) as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                observed += len(block); digest.update(block)
        if observed != size or digest.hexdigest() != row['sha256']:
            raise ValueError('Original storage byte hash differs: ' + name)
        checked[name] = digest.hexdigest(); total += specification['elements']
    return checked, total


def audit_vae_weights(checkpoint, load_records):
    """Audit one or more matching load reports against the pinned 2.8 GB file."""
    checkpoint = regular(checkpoint)
    if not isinstance(load_records, (list, tuple)) or not 1 <= len(load_records) <= 2:
        raise ValueError('One or two explicit decoder load records required')
    paths = [regular(path) for path in load_records]
    before = [sha(path) for path in paths]
    records = [record(path) for path in paths]
    if [sha(path) for path in paths] != before:
        raise ValueError('Decoder load records changed while parsing')
    if any(value != records[0] for value in records[1:]):
        raise ValueError('Codec and decoder load records differ')
    if checkpoint.stat().st_size != VAE_BYTES or sha(checkpoint) != VAE_SHA:
        raise ValueError('Pinned original decoder checkpoint bytes required')
    with zipfile.ZipFile(checkpoint) as archive:
        prefix, metadata = read_metadata(archive)
        if len(metadata) != 196:
            raise ValueError('Exactly 196 original decoder storages required')
        checked, total = compare_storages(archive, prefix, metadata, records[0]['tensors'])
    if (total != PARAMETERS or [sha(path) for path in paths] != before
            or checkpoint.stat().st_size != VAE_BYTES or sha(checkpoint) != VAE_SHA):
        raise ValueError('Original parameter count and unchanged checkpoint/load records required')
    return {'status': 'passed', 'checkpoint_sha256': VAE_SHA, 'checkpoint_bytes': VAE_BYTES,
            'load_record_sha256': before, 'matched_storages': len(checked), 'matched_parameters': total,
            'tensor_sha256': checked, 'torch_imported_by_this_module': False,
            'checkpoint_hash_verified_before_and_after': True, 'load_records_unchanged': True,
            'method': 'Restricted inert metadata parser and streamed original ZIP storage bytes; no model/tensor construction',
            'limitations': ['Checks the retained copy records against the source checkpoint, not live GPU memory.']}
