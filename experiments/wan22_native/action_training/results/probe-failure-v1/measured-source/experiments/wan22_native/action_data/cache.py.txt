# SPDX-License-Identifier: Apache-2.0
"""Strict training and inference readers for the separate 48-channel cache."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .data import sha, SELECTION, CHANNELS
from .operations import tensor_sha

SCHEMA = 'worldline-wan22-atrium-action-cache-v1'
SHAPES = {'target':(1,48,5,18,32),'observation':(1,48,1,18,32),'commands':(1,16,6)}


def checked_file(directory, entry):
    directory = Path(directory).resolve()
    path = directory/entry['file']
    if (Path(entry['file']).is_absolute() or '..' in Path(entry['file']).parts
            or not path.resolve().is_relative_to(directory) or sha(path) != entry['sha256']):
        raise ValueError('Cache file path or hash differs')
    return path


def read_window(directory, identity, *, conditioning_only=False):
    directory = Path(directory)
    if (directory/'watchdog-stop.json').exists():
        raise ValueError('Stopped cache cannot be used')
    metrics = json.loads((directory/'metrics.json').read_text())
    if (metrics.get('status') != 'passed' or metrics.get('finite_output') is not True
            or metrics.get('cache_clear_after_operations') is not True):
        raise ValueError('Completed worker metrics required; partial cache rejected')
    manifest_path = directory/'manifest.json'
    if sha(manifest_path) != metrics.get('cache_manifest_sha256'):
        raise ValueError('Manifest differs from completed worker evidence')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'passed'
            or manifest.get('command_channels') != list(CHANNELS)):
        raise ValueError('Completed matching 48-channel command cache required')
    expected = [f'{arm}-{start:04d}' for arm,start in SELECTION]
    if [e.get('id') for e in manifest.get('windows',[])] != expected:
        raise ValueError('Exactly the eight prescribed windows required')
    expected_checks = []
    for arm,start in SELECTION:
        key = f'{arm}-{start:04d}'
        expected_checks.append(key+'_target_prefix_vs_independent_initial')
        if (arm,start) in (('open',0),('open',8)):
            expected_checks.append(key+'_future_rgb_cannot_change_initial_latent')
            if start==0: expected_checks.append('shared_initial_A_after_full_B_is_identical')
    checks = manifest.get('causal_checks',[])
    if ([c.get('name') for c in checks] != expected_checks or any(c.get('passed') is not True
            or c.get('max_abs') != 0.0 or c.get('target_prefix_replaced') is not False
            or c.get('target_prefix_sha256') != c.get('observation_sha256') for c in checks)):
        raise ValueError('Exactly the 11 completed native causal checks required')
    expected_observations = ['start-0000-shared']+[f'{a}-{s:04d}' for a,s in SELECTION if s!=0]
    observation_rows = manifest.get('observations',[])
    if [o.get('id') for o in observation_rows] != expected_observations:
        raise ValueError('Exactly seven independently encoded observations required')
    observations = {o['id']:o for o in observation_rows}
    for observation in observation_rows:
        checked_file(directory,observation)
        if (set(observation.get('tensors',{})) != {'observation'} or observation.get('encoded_rgb_frames') != 1):
            raise ValueError('Independent one-image observation record required')
    for entry,(arm,start) in zip(manifest['windows'],SELECTION):
        observation_id = 'start-0000-shared' if start==0 else entry['id']
        if (entry.get('observation_id') != observation_id
                or entry.get('tensors',{}).get('observation') != observations[observation_id]['tensors']['observation']):
            raise ValueError('Window and independently encoded observation differ')
        check = next(c for c in checks if c['name']==entry['id']+'_target_prefix_vs_independent_initial')
        if check['observation_sha256'] != entry['tensors']['observation']['sha256']:
            raise ValueError('Causal check binds a different observation')
    entries = {e['id']:e for e in manifest['windows']}
    if identity not in entries:
        raise ValueError('Unknown window')
    entry = entries[identity]; path = checked_file(directory,entry)
    names = ('observation','commands') if conditioning_only else tuple(SHAPES)
    values = {}
    with safe_open(path,framework='pt',device='cpu') as handle:
        if set(handle.keys()) != set(SHAPES):
            raise ValueError('Unexpected window tensor keys')
        for name in names:
            value = handle.get_tensor(name)
            spec = entry['tensors'][name]
            if (tuple(value.shape) != SHAPES[name] or value.dtype != torch.float32 or not torch.isfinite(value).all()
                    or spec != {'shape':list(SHAPES[name]),'dtype':'float32','sha256':tensor_sha(value)}):
                raise ValueError('Cache tensor contract or hash differs: '+name)
            values[name] = value
    return values, {'manifest_sha256':sha(directory/'manifest.json'),'window_id':identity,
                    'file_sha256':entry['sha256'],'materialized_tensor_keys':list(names),'source':entry['source']}


def load_training_window(directory, identity):
    return read_window(directory,identity,conditioning_only=False)


def load_condition(directory, identity):
    return read_window(directory,identity,conditioning_only=True)
