# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU tensors only; no learned weights or model execution."""
import copy
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from . import predictions as p
from ..cuda_reference.vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(value) + '\n')


def save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    save_file({key: value.contiguous() for key, value in values.items()}, str(path))


def make_fixture(root):
    packet, codec = root / 'packet', root / 'codec'
    packet.mkdir(); codec.mkdir()
    tensors = {'reference_observation': torch.zeros(1, 48, 1, 18, 32)}
    observations = {}
    for name, shape in p.SHAPES.items():
        tensors[name + '_noise'] = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 100000
        tensors[name + '_rgb'] = torch.zeros(1, 3, 1, shape[2] * 16, shape[3] * 16)
        observations[name] = torch.full((1, 48, 1, shape[2], shape[3]), 0.125)
    save(packet / 'prepared.safetensors', tensors)
    manifest = {'schema': 'wan22-two-size-inputs-v1', 'status': 'prepared',
                'files': {'prepared.safetensors': {'sha256': p._sha(packet / 'prepared.safetensors'),
                                                  'bytes': (packet / 'prepared.safetensors').stat().st_size}},
                'tensor_sha256': {name: p._tensor_sha(value) for name, value in tensors.items()}}
    write(packet / 'manifest.json', manifest)
    packet_sha = p._sha(packet / 'manifest.json')
    mapping = {name: p._sha(p.REPO / name) for name in (p.CATALOG_NAME, p.SOLVER_NAME)}

    def bind(run, mode, profile, stage, extra):
        result = run / stage / 'result'
        result.mkdir(parents=True, exist_ok=True)
        for name in mapping:
            target = run / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.REPO / name, target)
        identity = {'mode': mode, 'profile': profile, 'source_sha256': mapping, 'input_manifest_sha256': packet_sha}
        worker = {**identity, 'status': 'passed', 'stage': stage, 'finite_outputs': True, 'inputs_unchanged': True,
                  'hardware': {'name': 'synthetic CPU fixture'},
                  'predictions': 2 if mode == 'pair' else 100 if mode == 'clip' else 0,
                  'solver_updates': 50 if mode == 'clip' else 0, **extra,
                  'output_sha256': {f.relative_to(result).as_posix(): p._sha(f) for f in result.rglob('*') if f.is_file()}}
        write(result / 'metrics.json', worker)
        write(run / stage / 'terminal.json', {'status': 'complete', 'exit_code': 0, 'error': None})
        report = {**identity, 'status': 'passed', 'model_execution': True,
                  'settings': {'steps': 50, 'shift': 5.0, 'guidance': 5.0},
                  'child_reports': {stage + '/result/metrics.json': p._sha(result / 'metrics.json')},
                  'child_terminals': {stage + '/terminal.json': p._sha(run / stage / 'terminal.json')}}
        if stage == 'core': report['codec_result_report_sha256'] = p._sha(codec / 'metrics.json')
        write(run / 'metrics.json', report)

    result = codec / 'codec/result'
    save(result / 'observations.safetensors', observations)
    for name, value in observations.items(): save(result / name / 'observation.safetensors', {'observation': value})
    bind(codec, 'codec', 'both', 'codec', {'profiles': {name: {'observation_tensor_sha256': p._tensor_sha(value)}
                                                         for name, value in observations.items()}})
    catalog = json.loads((p.REPO / p.CATALOG_NAME).read_text())['tensors']
    weights = {'tensor_count': 825, 'parameter_count': 4999787712, 'parameter_bytes': 19999150848,
               'convert_model_dtype': False, 'cuda_copy_exact': True, 'all_shards_verified': True,
               'tensors': {name: {'shape': row['shape'], 'shard': row['shard'], 'original_dtype': 'float32',
                   'loaded_dtype': 'float32', 'source_sha256': row['original_sha256'],
                   'loaded_sha256': row['original_sha256'], 'source_owner_released': True,
                   'cuda_copy_exact': True} for name, row in catalog.items()}}
    solver = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    solver.set_timesteps(50, device='cpu', shift=5.0)
    for profile, shape in p.SHAPES.items():
        prefix = shape[2] // 2 * (shape[3] // 2)
        times = torch.full((1, prefix * 5), 999, dtype=torch.int64); times[:, :prefix] = 0
        latent = tensors[profile + '_noise'].clone(); latent[:, :1] = observations[profile][0]
        values = {'initial_noise': tensors[profile + '_noise'], 'initial_latent': latent,
                  'observation': observations[profile], 'token_times': times}
        for mode in ('pair', 'clip'):
            run = root / (mode + '-' + profile); result = run / 'core/result'
            save(result / 'sampling-inputs.safetensors', values)
            write(result / 'weight-load.json', weights)
            if mode == 'pair':
                positive, negative = latent * 0.1 + 0.02, latent * -0.03
                guided = negative + 5.0 * (positive - negative)
                save(result / 'completed-positive.safetensors', {'positive_velocity': positive})
                save(result / 'completed-negative.safetensors', {'positive_velocity': positive, 'negative_velocity': negative})
                save(result / 'outputs.safetensors', {'positive_velocity': positive, 'negative_velocity': negative, 'guided_velocity': guided})
            else:
                save(result / 'step-01.safetensors', {'latent': latent})
                # Identical saved states are valid synthetic evidence. Hardlinks
                # keep the 50-file fixture small; mutation helpers detach first.
                for number in range(2, 51): os.link(result / 'step-01.safetensors', result / f'step-{number:02d}.safetensors')
                os.link(result / 'step-01.safetensors', result / 'latents.safetensors')
                rows = [{'step': number, 'timestep': int(t), 'prefix_exact': True,
                         'latent_sha256': p._tensor_sha(latent), 'seconds': float(number)} for number, t in enumerate(solver.timesteps, 1)]
                (result / 'steps.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
            bind(run, mode, profile, 'core', {'sampling_input_tensor_sha256': {key: p._tensor_sha(value) for key, value in values.items()}})


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.base_temp = tempfile.TemporaryDirectory(); cls.base = Path(cls.base_temp.name).resolve()
        make_fixture(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.base_temp.cleanup(); torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'fixture'
        shutil.copytree(self.base, self.root, copy_function=os.link)

    def audit(self, mode='pair', profile='baseline'):
        return getattr(p, 'audit_' + mode)(self.root / (mode + '-' + profile), self.root / 'codec', self.root / 'packet', profile)

    def refresh(self, mode='pair', profile='baseline', edit_worker=None):
        run = self.root / (mode + '-' + profile); result = run / 'core/result'
        worker = json.loads((result / 'metrics.json').read_text())
        worker['output_sha256'] = {f.relative_to(result).as_posix(): p._sha(f) for f in result.rglob('*') if f.is_file() and f.name != 'metrics.json'}
        if edit_worker: edit_worker(worker)
        write(result / 'metrics.json', worker)
        parent = json.loads((run / 'metrics.json').read_text()); parent['child_reports']['core/result/metrics.json'] = p._sha(result / 'metrics.json')
        write(run / 'metrics.json', parent)

    def alter_tensor(self, name, change, mode='pair', profile='baseline', refresh=True):
        path = self.root / (mode + '-' + profile) / 'core/result' / name
        values = {k: v.clone() for k, v in load_file(str(path)).items()}; change(values); save(path, values)
        if refresh: self.refresh(mode, profile)
        return values

    def test_pair_both_exact_shapes_and_numpy_guidance(self):
        with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            for profile in p.SHAPES:
                result = self.audit('pair', profile)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['latent_shape'], list(p.SHAPES[profile]))
                self.assertEqual(result['guidance_residual'], {'max_absolute': 0., 'rmse': 0., 'different_bytes': 0})
                self.assertEqual(result['weights']['tensor_count'], 825)

    def test_clip_both_shapes_retains_explicit_no_replay_limit(self):
        for profile in p.SHAPES:
            result = self.audit('clip', profile)
            self.assertEqual(len(result['steps']), 50)
            self.assertFalse(result['solver_replayed'])
            self.assertTrue(result['checks']['final_equals_step_50'])

    def test_guidance_corruption_rejected_even_after_file_hash_refresh(self):
        self.alter_tensor('outputs.safetensors', lambda v: v['guided_velocity'].__setitem__((0, 2, 1, 1), 100))
        with self.assertRaisesRegex(ValueError, r'negative \+'):
            self.audit()

    def test_partials_must_match_each_other_and_final(self):
        self.alter_tensor('completed-positive.safetensors', lambda v: v['positive_velocity'].add_(0.25))
        with self.assertRaisesRegex(ValueError, 'partials'): self.audit()

    def test_prefix_time_and_noise_semantics_independent_of_recorded_hash(self):
        for key, change in (
            ('initial_latent', lambda t: t.__setitem__((0, 0, 0, 0), -1)),
            ('initial_noise', lambda t: t.__setitem__((0, 2, 0, 0), -1)),
            ('token_times', lambda t: t.__setitem__((0, 0), 999)),
            ('observation', lambda t: t.__setitem__((0, 0, 0, 0, 0), -1))):
            with self.subTest(key=key):
                original = self.base / 'pair-baseline/core/result/sampling-inputs.safetensors'
                target = self.root / 'pair-baseline/core/result/sampling-inputs.safetensors'
                target.unlink(); shutil.copyfile(original, target)
                values = self.alter_tensor('sampling-inputs.safetensors', lambda v: change(v[key]))
                self.refresh(edit_worker=lambda worker: worker.update(sampling_input_tensor_sha256={k: p._tensor_sha(v) for k, v in values.items()}))
                with self.assertRaises(ValueError): self.audit()

    def test_clip_prefix_corruption_rejected_with_consistent_new_hash(self):
        values = self.alter_tensor('step-24.safetensors', lambda v: v['latent'].__setitem__((0, 0, 0, 0), -1), 'clip')
        path = self.root / 'clip-baseline/core/result/steps.jsonl'; rows = path.read_text().splitlines()
        row = json.loads(rows[23]); row['latent_sha256'] = p._tensor_sha(values['latent']); rows[23] = json.dumps(row)
        path.unlink(); path.write_text('\n'.join(rows) + '\n'); self.refresh('clip')
        with self.assertRaisesRegex(ValueError, 'prefix at step 24'): self.audit('clip')

    def test_clip_order_schedule_and_final_equality(self):
        path = self.root / 'clip-baseline/core/result/steps.jsonl'; original = path.read_text()
        rows = original.splitlines(); rows[0], rows[1] = rows[1], rows[0]
        path.unlink(); path.write_text('\n'.join(rows) + '\n'); self.refresh('clip')
        with self.assertRaisesRegex(ValueError, 'schedule'): self.audit('clip')
        path.unlink(); path.write_text(original); self.refresh('clip')
        self.alter_tensor('latents.safetensors', lambda v: v['latent'].__setitem__((0, 3, 0, 0), -1), 'clip')
        with self.assertRaisesRegex(ValueError, 'step 50'): self.audit('clip')

    def test_nonfinite_dtype_and_unbound_file_corruption(self):
        self.alter_tensor('outputs.safetensors', lambda v: v['guided_velocity'].fill_(float('nan')))
        with self.assertRaisesRegex(ValueError, 'Finite'): self.audit()
        self.alter_tensor('outputs.safetensors', lambda v: v.update(guided_velocity=v['guided_velocity'].double()))
        with self.assertRaisesRegex(ValueError, 'dtype'): self.audit()
        path = self.root / 'pair-baseline/core/result/outputs.safetensors'
        raw = path.read_bytes(); path.unlink(); path.write_bytes(raw + b'x')
        with self.assertRaisesRegex(ValueError, 'hash'): self.audit()

    def test_original_weight_identity_rejected_after_metadata_rebinding(self):
        path = self.root / 'pair-baseline/core/result/weight-load.json'; values = json.loads(path.read_text())
        values['tensors']['patch_embedding.weight']['loaded_sha256'] = '0' * 64
        write(path, values); self.refresh()
        with self.assertRaisesRegex(ValueError, 'parameter identity'): self.audit()

    def test_header_wrong_shape_metadata_and_bound_before_materialization(self):
        path = self.root / 'bad.safetensors'
        shape = (2,)
        cases = [({'x': {'shape': [3], 'dtype': 'F32', 'data_offsets': [0, 8]}}, b'\0' * 8),
                 ({'__metadata__': {'bad': 3}, 'x': {'shape': [2], 'dtype': 'F32', 'data_offsets': [0, 8]}}, b'\0' * 8),
                 ({'x': {'shape': [2], 'dtype': 'F32', 'data_offsets': [1, 9]}}, b'\0' * 9)]
        for header, payload in cases:
            raw = json.dumps(header).encode(); path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
            with mock.patch.object(p, 'safe_open', side_effect=AssertionError('Materialization forbidden')):
                with self.assertRaises(ValueError): p._read(path, {'x': (shape, 'F32')})
        path.write_bytes(struct.pack('<Q', p.HEADER_LIMIT + 1) + b'{}')
        with self.assertRaisesRegex(ValueError, 'header'): p._read(path, {'x': (shape, 'F32')})

    def test_reject_symlink_and_undeclared_profile(self):
        with self.assertRaises(ValueError): self.audit(profile='other')
        path = self.root / 'pair-baseline/core/result/outputs.safetensors'; path.unlink()
        path.symlink_to(self.base / 'pair-baseline/core/result/outputs.safetensors')
        with self.assertRaisesRegex(ValueError, 'Symlink'): self.audit()


if __name__ == '__main__':
    unittest.main()
