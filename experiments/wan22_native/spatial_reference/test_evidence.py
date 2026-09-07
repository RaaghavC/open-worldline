# SPDX-License-Identifier: Apache-2.0
"""Synthetic files and timing arithmetic only. No weights, models or CUDA."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from . import evidence as e
from .config import SETTINGS
from .guards import limits
from .config import spec, SPECS
from .sampling import scheduler


GPU = 'NVIDIA A100-SXM4-80GB'
HARDWARE = {'name': GPU, 'total_memory_bytes': 80 * 2**30, 'bf16_supported': True,
            'capability': [8, 0], 'torch': '2.5.1+cu124', 'cuda': '12.4',
            'flash_attn': '2.7.4.post1', 'flash_attention_2_available': True,
            'flash_attention_3_available': False}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This instantiates shapes only on meta. No parameter values, weights,
        # forward call or CUDA context is created for the VAE fixture.
        import torch
        from ..cuda_reference.vendor.vae2_2 import WanVAE_
        config = dict(dim=160, dec_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                      attn_scales=[], temperal_downsample=[False, True, True], dropout=0.)
        with torch.device('meta'):
            model = WanVAE_(**config)
        cls.vae_weights = {'weight_sha256': '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36',
                           'compute_dtype': 'float32', 'parameters': 704688668, 'config': config,
                           'tensors': {name: {'shape': list(value.shape), 'sha256': 'a' * 64,
                                            'cuda_copy_exact': True} for name, value in model.state_dict().items()}}
        del model

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'source/module.py'
        self.source.parent.mkdir()
        self.source.write_bytes(b'original source\n')
        self.mapping = {'module.py': e.sha(self.source)}
        write(self.root / 'inputs/manifest.json', {'synthetic': True})
        self.packet_sha = e.sha(self.root / 'inputs/manifest.json')
        self.loader = mock.patch.object(e, 'load_prepared', return_value=({'x': object()}, {'text': object()}, {'synthetic': True})).start()
        self.addCleanup(mock.patch.stopall)

    def build(self, mode='pair', profile='spatial'):
        stages = ('codec',) if mode == 'codec' else ('core',) if mode == 'pair' else ('core', 'decode')
        root = {'status': 'passed', 'model_execution': True, 'mode': mode, 'profile': profile,
                'source_sha256': self.mapping, 'input_manifest_sha256': self.packet_sha,
                'settings': SETTINGS, 'limits': limits(mode), 'child_reports': {}, 'child_terminals': {}}
        for stage in stages:
            result = self.root / stage / 'result'
            result.mkdir(parents=True)
            (result / 'output.bin').write_bytes(b'finite fixture output')
            monitor = {'status': 'complete', 'mode': mode, 'limits': limits(mode), 'error': None}
            write(result / 'monitor-terminal.json', monitor)
            child = {**root, 'stage': stage, 'hardware': copy.deepcopy(HARDWARE),
                     'finite_outputs': True, 'inputs_unchanged': True,
                     'predictions': 2 if mode == 'pair' else 100 if stage == 'core' else 0,
                     'solver_updates': 50 if stage == 'core' and mode == 'clip' else 0,
                     'output_sha256': {name: e.sha(result / name) for name in ('output.bin', 'monitor-terminal.json')}}
            self.scientific(result, child, mode, profile)
            child['output_sha256'] = {p.relative_to(result).as_posix(): e.sha(p)
                                      for p in result.rglob('*') if p.is_file()}
            write(result / 'metrics.json', child)
            terminal = {'status': 'complete', 'exit_code': 0, 'mode': mode,
                        'limits': limits(mode), 'error': None, 'cleanup_error': None}
            write(self.root / stage / 'terminal.json', terminal)
            root['child_reports'][stage + '/result/metrics.json'] = e.sha(result / 'metrics.json')
            root['child_terminals'][stage + '/terminal.json'] = e.sha(self.root / stage / 'terminal.json')
        write(self.root / 'metrics.json', root)
        return root

    def scientific(self, result, child, mode, profile):
        def touch(name):
            path = result / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture payload; not a model prediction')
        if child['stage'] == 'core':
            expected = json.loads((e.REPO / e.PREFIX / 'cuda_reference/expected-weights.json').read_text())['tensors']
            weights = {'tensor_count': 825, 'parameter_count': 4999787712, 'parameter_bytes': 19999150848,
                       'convert_model_dtype': False, 'all_shards_verified': True, 'cuda_copy_exact': True,
                       'tensors': {name: {'shape': row['shape'], 'shard': row['shard'],
                           'original_dtype': 'float32', 'loaded_dtype': 'float32',
                           'source_sha256': row['original_sha256'], 'loaded_sha256': row['original_sha256'],
                           'cuda_copy_exact': True, 'source_owner_released': True} for name, row in expected.items()}}
            write(result / 'weight-load.json', weights)
            child['precision'] = e.PRECISION
            touch('sampling-inputs.safetensors')
            if mode == 'pair':
                for name in ('outputs.safetensors', 'completed-positive.safetensors', 'completed-negative.safetensors'):
                    touch(name)
            else:
                touch('latents.safetensors'); rows = []
                for number, timestep in enumerate(scheduler().timesteps, 1):
                    touch(f'step-{number:02d}.safetensors')
                    rows.append({'step': number, 'timestep': int(timestep), 'prefix_exact': True,
                                 'latent_sha256': 'a' * 64, 'seconds': float(number)})
                (result / 'steps.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
        else:
            child.update(decoder_cache_clear=True, load_seconds=5.)
            write(result / 'weight-load.json', self.vae_weights)
            if child['stage'] == 'codec':
                touch('observations.safetensors'); child['profiles'] = {}
                for name, selected in SPECS.items():
                    for item in ('observation.safetensors', 'timing-proxy.safetensors', 'reconstructed-initial.png'):
                        touch(name + '/' + item)
                    relative = name + '/proxy-rgb/index.json'
                    self.rgb(result, relative, selected, 'repeated_observation_codec_timing_proxy')
                    child['profiles'][name] = {'encode_seconds': 1., 'decode_seconds': 2.,
                        'observation_tensor_sha256': 'a' * 64, 'raw_rgb_index': relative}
            else:
                selected = spec(profile)
                self.rgb(result, 'rgb/index.json', selected, 'generated_clip')
                for name in ('preview.gif', 'comparison.png', *[f'frames/{n:04d}.png' for n in range(17)]):
                    touch(name)
                child.update(decode_seconds=2., images={'frames': 17, 'height': selected.height,
                    'width': selected.width, 'conditioned_initial_frames': 1,
                    'new_future_frames': 16, 'contact_images_resized': False})

    def rgb(self, result, relative, selected, purpose):
        folder = (result / relative).parent; folder.mkdir(parents=True)
        records = []
        for number in range(17):
            name = f'{number:04d}.safetensors'; path = folder / name
            path.write_bytes(b'fixture raw RGB payload')
            records.append({'index': number, 'file': name, 'bytes': path.stat().st_size,
                            'sha256': e.sha(path), 'tensor_sha256': 'a' * 64})
        write(result / relative, {'schema': 'wan22-rgb-frames-v1', 'purpose': purpose,
              'shape': [1, 3, 17, selected.height, selected.width], 'dtype': 'float32',
              'range': [-1, 1], 'frames': records})

    def check(self, mode='pair', profile='spatial'):
        return e.validate_completed(self.root, mode, profile, self.mapping, self.packet_sha, GPU)

    def mutate_child(self, change, stage='core'):
        path = self.root / stage / 'result/metrics.json'
        child = json.loads(path.read_text()); change(child); write(path, child)
        root = json.loads((self.root / 'metrics.json').read_text())
        root['child_reports'][stage + '/result/metrics.json'] = e.sha(path)
        write(self.root / 'metrics.json', root)

    def refresh_outputs(self, stage='core'):
        result = self.root / stage / 'result'
        self.mutate_child(lambda child: child.update(output_sha256={p.relative_to(result).as_posix(): e.sha(p)
            for p in result.rglob('*') if p.is_file() and p.name not in e.IGNORED_OUTPUTS}), stage)

    def test_completed_pair_returns_unchanged_root(self):
        expected = self.build()
        self.assertEqual(self.check(), expected)
        self.loader.assert_called_once_with(self.root / 'inputs')

    def test_codec_and_two_child_clip(self):
        self.build('codec', 'both')
        self.assertEqual(self.check('codec', 'both')['mode'], 'codec')
        # Reuse the scratch directory for the second isolated run, retaining
        # only that run's declared children.
        shutil.rmtree(self.root / 'codec')
        self.build('clip', 'baseline')
        self.assertEqual(self.check('clip', 'baseline')['mode'], 'clip')

    def test_manifest_hash_is_checked_before_loader(self):
        with self.assertRaises(ValueError):
            e.validate_input_packet(self.root / 'inputs', '0' * 64)
        self.loader.assert_not_called()

    def test_manifest_change_during_load_rejected(self):
        def change(_):
            write(self.root / 'inputs/manifest.json', {'changed': True})
            return {}, {}, {}
        self.loader.side_effect = change
        with self.assertRaises(ValueError):
            e.validate_input_packet(self.root / 'inputs', self.packet_sha)

    def test_input_symlink_rejected_before_materialization(self):
        (self.root / 'inputs/link').symlink_to(self.source)
        with self.assertRaises(ValueError):
            e.validate_input_packet(self.root / 'inputs', self.packet_sha)
        self.loader.assert_not_called()

    def test_plan_incorrect_identity_and_incomplete_child_maps_rejected(self):
        original = self.build()
        changes = [('model_execution', False), ('status', 'planned'), ('profile', 'baseline'),
                   ('mode', 'clip'), ('source_sha256', {}), ('input_manifest_sha256', 'f' * 64),
                   ('child_reports', {}), ('child_terminals', {}), ('settings', {'steps': 49}),
                   ('limits', limits('clip'))]
        for key, value in changes:
            with self.subTest(key=key):
                altered = copy.deepcopy(original); altered[key] = value
                write(self.root / 'metrics.json', altered)
                with self.assertRaises(ValueError): self.check()

    def test_child_and_terminal_hash_tampering_rejected(self):
        self.build()
        for relative in ('core/result/metrics.json', 'core/terminal.json'):
            path = self.root / relative; original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.assertRaises(ValueError): self.check()
            path.write_bytes(original)

    def test_failed_and_boolean_exit_not_accepted(self):
        original = self.build()
        path = self.root / 'core/terminal.json'
        saved = json.loads(path.read_text())
        for key, value in [('status', 'failed'), ('exit_code', 1), ('exit_code', False),
                           ('error', 'interrupted'), ('cleanup_error', 'kill failed')]:
            with self.subTest(key=key, value=value):
                terminal = {**saved, key: value}; write(path, terminal)
                root = copy.deepcopy(original)
                root['child_terminals']['core/terminal.json'] = e.sha(path)
                write(self.root / 'metrics.json', root)
                with self.assertRaises(ValueError): self.check()

    def test_prescribed_counts_and_finite_integrity_flags(self):
        self.build()
        for key, value in [('predictions', 1), ('predictions', 2.0), ('solver_updates', False),
                           ('finite_outputs', False), ('inputs_unchanged', False), ('stage', 'decode')]:
            path = self.root / 'core/result/metrics.json'; original = path.read_bytes()
            self.mutate_child(lambda child: child.update({key: value}))
            with self.assertRaises(ValueError): self.check()
            path.write_bytes(original)
        # Restore the parent's hash after the final mutation.

    def test_hardware_mismatch_across_clip_children(self):
        self.build('clip', 'spatial')
        self.mutate_child(lambda child: child['hardware'].update(cudnn=999), 'decode')
        with self.assertRaises(ValueError): self.check('clip', 'spatial')

    def test_wrong_gpu_or_missing_hardware_rejected(self):
        self.build()
        with self.assertRaises(ValueError):
            e.validate_completed(self.root, 'pair', 'spatial', self.mapping, self.packet_sha, 'Other GPU')
        self.mutate_child(lambda child: child.pop('hardware'))
        with self.assertRaises(ValueError): self.check()

    def test_output_modified_missing_or_unlisted(self):
        self.build(); path = self.root / 'core/result/output.bin'; original = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.check()
        path.unlink()
        with self.assertRaises(ValueError): self.check()
        path.write_bytes(original)
        (path.parent / 'unbound.bin').write_bytes(b'x')
        with self.assertRaises(ValueError): self.check()

    def test_traversal_absolute_and_symlink_outputs_rejected(self):
        self.build()
        path = self.root / 'core/result/metrics.json'; original = path.read_bytes()
        for name in ('../outside', '/outside', 'dir//file', './file', 'dir\\file'):
            path.write_bytes(original)
            self.mutate_child(lambda child: child['output_sha256'].update({name: '0' * 64}))
            with self.assertRaises(ValueError): self.check()
        path.write_bytes(original)
        target = self.root / 'core/result/output.bin'; target.unlink(); target.symlink_to(self.source)
        with self.assertRaises(ValueError): self.check()

    def test_source_snapshot_changed_or_extra_rejected(self):
        self.build(); original = self.source.read_bytes()
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.check()
        self.source.write_bytes(original)
        (self.source.parent / 'extra.py').write_bytes(b'extra')
        with self.assertRaises(ValueError): self.check()

    def test_watchdog_and_missing_monitor_rejected(self):
        self.build()
        stop = self.root / 'core/watchdog-stop.json'; write(stop, {'status': 'stopped'})
        with self.assertRaises(ValueError): self.check()
        stop.unlink(); (self.root / 'core/result/monitor-terminal.json').unlink()
        self.mutate_child(lambda child: child['output_sha256'].pop('monitor-terminal.json'))
        with self.assertRaises(ValueError): self.check()

    def test_missing_scientific_artifact_cannot_be_removed_from_manifest(self):
        self.build()
        (self.root / 'core/result/completed-negative.safetensors').unlink()
        self.refresh_outputs()
        with self.assertRaises(ValueError): self.check()

    def test_all_825_core_parameter_identities_and_precision_checked(self):
        self.build(); path = self.root / 'core/result/weight-load.json'
        original = json.loads(path.read_text())
        for alter in (lambda data: data['tensors'].pop('patch_embedding.weight'),
                      lambda data: data['tensors']['patch_embedding.weight'].update(loaded_sha256='0' * 64),
                      lambda data: data['tensors']['patch_embedding.weight'].update(source_owner_released=False),
                      lambda data: data.update(convert_model_dtype=True)):
            data = copy.deepcopy(original); alter(data); write(path, data); self.refresh_outputs()
            with self.assertRaises(ValueError): self.check()
        write(path, original); self.refresh_outputs()
        self.mutate_child(lambda child: child.update(precision={}))
        with self.assertRaises(ValueError): self.check()

    def test_codec_missing_frame_record_cache_and_load_evidence(self):
        self.build('codec', 'both')
        index = self.root / 'codec/result/spatial/proxy-rgb/index.json'
        original = json.loads(index.read_text()); changed = copy.deepcopy(original)
        changed['frames'].pop(); write(index, changed); self.refresh_outputs('codec')
        with self.assertRaises(ValueError): self.check('codec', 'both')
        write(index, original); self.refresh_outputs('codec')
        self.mutate_child(lambda child: child.update(decoder_cache_clear=False), 'codec')
        with self.assertRaises(ValueError): self.check('codec', 'both')
        self.mutate_child(lambda child: child.update(decoder_cache_clear=True), 'codec')
        path = self.root / 'codec/result/weight-load.json'
        data = json.loads(path.read_text()); data['tensors'].pop(next(iter(data['tensors'])))
        write(path, data); self.refresh_outputs('codec')
        with self.assertRaises(ValueError): self.check('codec', 'both')

    def test_clip_requires_all_steps_in_order_and_all_pngs(self):
        self.build('clip', 'spatial')
        path = self.root / 'core/result/steps.jsonl'; original = path.read_text()
        rows = original.splitlines(); rows[0], rows[1] = rows[1], rows[0]
        path.write_text('\n'.join(rows) + '\n'); self.refresh_outputs()
        with self.assertRaises(ValueError): self.check('clip', 'spatial')
        path.write_text(original); self.refresh_outputs()
        (self.root / 'decode/result/frames/0016.png').unlink(); self.refresh_outputs('decode')
        with self.assertRaises(ValueError): self.check('clip', 'spatial')

    def test_duplicate_and_nonfinite_json_rejected_before_load(self):
        path = self.root / 'inputs/manifest.json'
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e400}', '[]'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                e.validate_input_packet(path.parent, e.sha(path))
        self.loader.assert_not_called()

    def test_sources_are_explicit_stable_and_snapshot_never_overwrites(self):
        repo = self.root / 'repo'; repo.mkdir()
        (repo / 'a.py').write_bytes(b'a\r\n'); (repo / 'b.json').write_bytes(b'{}')
        (repo / 'test_extra.py').write_bytes(b'not a runtime dependency')
        output = self.root / 'snapshot'; output.mkdir()
        with mock.patch.multiple(e, REPO=repo, NAMES=('a.py', 'b.json')):
            mapping = e.sources(); self.assertEqual(list(mapping), ['a.py', 'b.json'])
            self.assertEqual(e.snapshot(output), mapping)
            for name in mapping:
                self.assertEqual((repo / name).read_bytes(), (output / 'source' / name).read_bytes())
            with self.assertRaises(FileExistsError): e.snapshot(output)


class AdmissionTests(unittest.TestCase):
    def reports(self):
        identity = {'status': 'passed', 'hardware': HARDWARE, 'source_sha256': {'x': '0' * 64},
                    'input_manifest_sha256': '1' * 64}
        pair = {**identity, 'mode': 'pair', 'profile': 'spatial', 'stage': 'core',
                'predictions': 2, 'solver_updates': 0, 'load_seconds': 100., 'pair_seconds': 2.}
        codec = {**identity, 'mode': 'codec', 'profile': 'both', 'stage': 'codec', 'load_seconds': 20.,
                 'profiles': {'baseline': {'decode_seconds': 10.}, 'spatial': {'decode_seconds': 30.}}}
        return pair, codec

    def test_exact_equation_uses_selected_codec_profile(self):
        pair, codec = self.reports()
        result = e.admit_clip(pair, codec, 'spatial')
        self.assertEqual(result['estimated_seconds'], 390.)
        self.assertEqual(result['limit_seconds'], 1800)
        self.assertTrue(result['estimate_is_not_a_completion_guarantee'])
        pair['profile'] = 'baseline'
        self.assertEqual(e.admit_clip(pair, codec, 'baseline')['estimated_seconds'], 366.)

    def test_strict_limit_and_bad_duration_values(self):
        pair, codec = self.reports()
        # 1510 + 200 + 60 + 30 is exactly 1800, and must fail.
        pair['load_seconds'] = 1510
        with self.assertRaises(ValueError): e.admit_clip(pair, codec, 'spatial')
        for value in (0, -1, True, float('nan'), float('inf'), '2', 10**400):
            pair, codec = self.reports(); pair['pair_seconds'] = value
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                e.admit_clip(pair, codec, 'spatial')

    def test_cross_identity_or_missing_measurement_rejected(self):
        for mutate in (lambda p, c: c.update(hardware={}),
                       lambda p, c: p.update(profile='baseline'),
                       lambda p, c: c['profiles']['spatial'].pop('decode_seconds'),
                       lambda p, c: p.update(status='failed')):
            pair, codec = self.reports(); mutate(pair, codec)
            with self.assertRaises(ValueError): e.admit_clip(pair, codec, 'spatial')


if __name__ == '__main__':
    unittest.main()
