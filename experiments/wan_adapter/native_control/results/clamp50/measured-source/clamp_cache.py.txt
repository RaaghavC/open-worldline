# SPDX-License-Identifier: Apache-2.0
"""Materialize only the independently encoded observation from the fixed cache."""
import hashlib
import json
from pathlib import Path
from safetensors import safe_open
import torch
from fetch_weights import sha, FILES

MANIFEST_SHA256 = '8d7675d96f9f71d27cd4107c9fd2540b05ebc02eb6834c24a98dcf6a76802a5f'


def read_observation(directory):
    root = Path(directory).resolve()
    manifest_file = root/'manifest.json'
    if sha(manifest_file) != MANIFEST_SHA256:
        raise ValueError('Require the fixed published Atrium observation cache')
    manifest = json.loads(manifest_file.read_text())
    if manifest.get('status') != 'passed' or manifest.get('schema') != 'worldline-wan-atrium-cache-v1':
        raise ValueError('Completed original observation cache required')
    if manifest['protocol_sha256'] != sha(Path(__file__).parent.parent/'PROTOCOL.md') or manifest['vae_weights_sha256'] != FILES['Wan2.1_VAE.pth'][1]:
        raise ValueError('Observation protocol or codec identity differs')
    checks = manifest.get('causal_checks', [])
    if len(checks) != 11 or not all(check.get('passed') is True for check in checks):
        raise ValueError('All 11 independent-observation cache checks must pass')
    rows = [row for row in manifest['windows'] if row['id'] == 'open-0000']
    if len(rows) != 1:
        raise ValueError('Expected one fixed open-0000 entry')
    row = rows[0]
    relative = Path(row['file']); path = (root/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root) or sha(path) != row['sha256']:
        raise ValueError('Observation cache path or file hash mismatch')
    # Whole-file hashing checks opaque integrity. Only this tensor is read.
    with safe_open(path, framework='pt', device='cpu') as tensors:
        observation = tensors.get_tensor('observation')
    if observation.shape != (1,16,1,36,64) or observation.dtype != torch.float32 or not torch.isfinite(observation).all():
        raise ValueError('Invalid independently encoded observation')
    digest = hashlib.sha256(observation.contiguous().numpy().tobytes()).hexdigest()
    metadata = row['tensors']['observation']
    if metadata['shape'] != list(observation.shape) or metadata['dtype'] != str(observation.dtype) or metadata['sha256'] != digest:
        raise ValueError('Observation tensor integrity mismatch')
    provenance = {'cache_manifest_sha256': MANIFEST_SHA256, 'window': 'open-0000',
        'cache_file_sha256': row['sha256'], 'observation_tensor_sha256': digest,
        'materialized_tensor_keys': ['observation'], 'target_materialized': False, 'actions_materialized': False,
        'observed_rgb_source': row['source']['sources'][0], 'data_license': row['source']['data_license'],
        'encoding': 'First RGB image encoded alone with fresh official VAE temporal cache',
        'encoding_compute_dtype': manifest['compute_dtype'], 'stored_dtype': str(observation.dtype),
        'causal_cache_checks_passed': 11}
    return observation[0].contiguous(), provenance
