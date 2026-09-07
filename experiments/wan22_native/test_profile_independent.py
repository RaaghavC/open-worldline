# SPDX-License-Identifier: Apache-2.0
"""Bounded profiler tests using synthetic observations and a stand-in denoiser."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from . import profile_core as p


def configuration(output):
    return {'output': str(output), 'device': 'cpu', 'weights': 'never-opened',
        'observation_cache': 'synthetic', 'text_cache': 'synthetic',
        'source_sha256': {name: p.sha256(p.HERE / name) for name in p.SOURCE_NAMES},
        'reused_source_sha256': {name: p.sha256(p.REPO / name) for name in p.REUSED_NAMES}}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def worker_fixture(self, directory, failure=None):
        observation = torch.linspace(-.8, .7, 48 * 18 * 32).reshape(1, 48, 1, 18, 32)
        positive, negative = torch.ones(2, 4096), torch.full((3, 4096), -1.)
        calls = []

        def denoiser(latents, times, contexts, seq_len):
            calls.append({'latent': latents[0].clone(), 'times': times.clone(),
                          'context': contexts[0].clone(), 'seq_len': seq_len,
                          'inference_mode': torch.is_inference_mode_enabled()})
            if failure is not None and len(calls) == 2:
                raise failure
            result = .2 if contexts[0][0, 0] > 0 else -.1
            return [torch.full_like(latents[0], result)]

        with mock.patch.object(p, 'load_observation', return_value=(observation, {'synthetic': True})), \
             mock.patch.object(p, 'load_text', return_value=(positive, negative, {'synthetic': True})), \
             mock.patch.object(p, 'load_core', return_value=(denoiser, {'synthetic': True})), \
             mock.patch.object(p.torch.backends.mps, 'is_available', side_effect=AssertionError('No GPU capability check needed')):
            if failure is None:
                p.worker(configuration(directory))
            else:
                with self.assertRaises(type(failure)):
                    p.worker(configuration(directory))
        return observation, positive, negative, calls

    def test_worker_runs_exactly_one_pair_and_one_native_cpu_solver_step(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            observation, positive, negative, calls = self.worker_fixture(output)
            metrics = json.loads((output / 'metrics.json').read_text())
            inputs = load_file(output / 'inputs.safetensors')
            outputs = load_file(output / 'outputs.safetensors')
            self.assertEqual(len(calls), 2)
            self.assertTrue(torch.equal(calls[0]['context'], positive))
            self.assertTrue(torch.equal(calls[1]['context'], negative))
            self.assertTrue(all(call['inference_mode'] and call['seq_len'] == 720 for call in calls))
            self.assertTrue(torch.equal(calls[0]['latent'], calls[1]['latent']))
            self.assertTrue(torch.equal(calls[0]['times'], calls[1]['times']))
            self.assertTrue(torch.equal(calls[0]['latent'], inputs['initial_latent']))
            self.assertTrue(torch.equal(inputs['initial_latent'][:, :1], observation[0]))
            self.assertTrue(torch.equal(inputs['initial_latent'][:, 1:], inputs['initial_noise'][:, 1:]))
            scheduler = p.make_scheduler()
            self.assertTrue((inputs['token_times'][:, :144] == 0).all())
            self.assertTrue((inputs['token_times'][:, 144:] == scheduler.timesteps[0]).all())
            expected_guidance = outputs['negative_velocity'] + 5 * (outputs['positive_velocity'] - outputs['negative_velocity'])
            self.assertTrue(torch.equal(outputs['guided_velocity'], expected_guidance))
            # The first UniPC step is first-order: x_next = x + delta_sigma*v.
            expected_next = inputs['initial_latent'] + (scheduler.sigmas[1] - scheduler.sigmas[0]) * expected_guidance
            expected_next[:, :1] = observation[0]
            torch.testing.assert_close(outputs['one_step_latent'], expected_next, atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.equal(outputs['one_step_latent'][:, :1], observation[0]))
            stages = [row['stage'] for row in metrics['timings']]
            self.assertEqual(stages.count('positive native forward'), 1)
            self.assertEqual(stages.count('negative native forward'), 1)
            self.assertEqual(stages.count('one official CPU UniPC step'), 1)
            measured_pair = sum(row['seconds'] for row in metrics['timings'] if row['stage'] in ('positive native forward', 'negative native forward'))
            self.assertEqual(metrics['pair_seconds'], measured_pair)
            self.assertEqual(metrics['fifty_pairs_linear_estimate_seconds'], 50 * measured_pair)
            self.assertFalse(metrics['full_video_generated'])
            self.assertFalse(metrics['quality_measured'])
            self.assertEqual(metrics['status'], 'passed')
            self.assertEqual((metrics['max_seconds'], metrics['max_memory_gib'], metrics['minimum_available_gib']), (900, 18, 2))
            self.assertTrue((output / 'memory.jsonl').is_file())

    def test_worker_retains_partial_inputs_and_failure_status(self):
        for error in (RuntimeError('synthetic second forward failure'), KeyboardInterrupt('synthetic interruption')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                _, _, _, calls = self.worker_fixture(output, failure=error)
                record = json.loads((output / 'metrics.json').read_text())
                self.assertEqual(len(calls), 2)
                self.assertEqual(record['status'], 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
                self.assertEqual(record['error_type'], type(error).__name__)
                self.assertTrue((output / 'inputs.safetensors').is_file())
                self.assertFalse((output / 'outputs.safetensors').exists())
                self.assertNotIn('pair_seconds', record)
                self.assertFalse(record['full_video_generated'])

    def test_changed_source_rejected_before_any_weight_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(directory)
            config['source_sha256']['portable.py'] = '0' * 64
            with mock.patch.object(p, 'load_core', side_effect=AssertionError('Forbidden weight load')):
                with self.assertRaisesRegex(ValueError, 'Core source changed'):
                    p.worker(config)

    def test_observation_reader_rejects_mixed_keys_before_materializing_any_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'original.png').write_bytes(b'synthetic original identity fixture')
            image_hash = p.sha256(root / 'original.png')
            observation = torch.zeros(1, 48, 1, 18, 32)
            cache = root / 'observation.safetensors'
            save_file({'observation': observation, 'target': torch.zeros(1), 'actions': torch.zeros(1)}, cache)
            names = ('codec.py', 'codec_profile.py', 'codec-source.json', 'test_codec.py', 'vendor/vae2_2.py')
            provenance = json.loads((p.HERE / 'provenance.json').read_text())
            record = {'status': 'passed', 'finite_output': True, 'input_rgb_frames': 1,
                'input_image_sha256': image_hash, 'future_rgb_read': False, 'old_latent_read': False,
                'actions_read': False, 'target_read': False, 'expected_latent_shape': [1,48,1,18,32],
                'cache_clear_after_encode': True, 'cache_clear_after_decode': True,
                'source_sha256': {name: p.sha256(p.HERE / name) for name in names},
                'codec': {'weight_sha256': provenance['native_vae_sha256']},
                'output_sha256': {cache.name: p.sha256(cache)}, 'latent_tensor_sha256': p.tensor_sha256(observation)}
            (root / 'metrics.json').write_text(json.dumps(record))
            real_open = p.safe_open
            reads = []

            class Reader:
                def __init__(self, *args, **kwargs):
                    self.inner = real_open(*args, **kwargs)
                def __enter__(self):
                    self.handle = self.inner.__enter__()
                    return self
                def __exit__(self, *args):
                    return self.inner.__exit__(*args)
                def keys(self):
                    return self.handle.keys()
                def get_tensor(self, key):
                    reads.append(key)
                    raise AssertionError('No mixed-file tensor may be read')

            with mock.patch.object(p, 'IMAGE_SHA256', image_hash), mock.patch.object(p, 'safe_open', Reader):
                with self.assertRaises(ValueError):
                    p.load_observation(root)
            self.assertEqual(reads, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
