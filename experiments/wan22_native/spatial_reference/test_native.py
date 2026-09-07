# SPDX-License-Identifier: Apache-2.0
"""CPU mocks and a meta-only constructor check. No real weights or GPU work."""
from contextlib import ExitStack, contextmanager, nullcontext
import unittest
from unittest import mock

import torch

from ..cuda_reference import native as frozen
from ..cuda_reference.vendor import attention
from . import native


def inputs(shape=(48, 5, 18, 32), time=999, context_length=25):
    tokens = shape[1] * (shape[2] // 2) * (shape[3] // 2)
    prefix = (shape[2] // 2) * (shape[3] // 2)
    latent = torch.full(shape, .125, dtype=torch.float32)
    times = torch.full((1, tokens), time, dtype=torch.int64)
    times[:, :prefix] = 0
    context = torch.full((context_length, 4096), .25, dtype=torch.float32)
    return latent, times, context


@contextmanager
def cpu_cuda_dispatch():
    """Keep all actual storage on CPU while recording the requested transfers."""
    original_to = torch.Tensor.to
    transfers = []

    def transfer(value, *args, **kwargs):
        device = args[0] if args else kwargs.get('device')
        if device == 'cuda:0':
            transfers.append((value.clone(), args, kwargs.copy()))
            return value.clone()
        return original_to(value, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('Actual CUDA initialization forbidden')))
        stack.enter_context(mock.patch.object(torch.Tensor, 'to', transfer))
        autocast = stack.enter_context(mock.patch.object(torch, 'autocast', return_value=nullcontext()))
        synchronize = stack.enter_context(mock.patch.object(torch.cuda, 'synchronize'))
        stack.enter_context(mock.patch.object(attention, 'FLASH_ATTN_2_AVAILABLE', True))
        stack.enter_context(mock.patch.object(attention, 'FLASH_ATTN_3_AVAILABLE', False))
        yield transfers, autocast, synchronize


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_loader_functions_are_exact_frozen_aliases(self):
        self.assertIs(native.meta_model, frozen.meta_model)
        self.assertIs(native.load_model, frozen.load_model)
        frozen.verify_sources()

    def test_actual_constructor_is_meta_only_with_original_complex_rope(self):
        with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden')):
            model = native.meta_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4999787712)
        self.assertEqual(len(list(model.parameters())), 825)
        self.assertTrue(all(p.device.type == 'meta' and p.dtype == torch.float32 and not p.requires_grad
                            for p in model.parameters()))
        self.assertFalse(model.training)
        self.assertEqual(model.freqs.device.type, 'cpu')
        self.assertEqual(model.freqs.dtype, torch.complex128)
        self.assertEqual(model.__class__.__module__, 'experiments.wan22_native.cuda_reference.vendor.model')

    def check_dispatch(self, shape, expected_tokens, expected_prefix):
        latent, times, context = inputs(shape)
        snapshots = [v.clone() for v in (latent, times, context)]
        calls = []

        def model(xs, ts, cs, seq_len):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(len(xs), 1)
            self.assertEqual(len(cs), 1)
            self.assertEqual(seq_len, expected_tokens)
            self.assertEqual(tuple(ts.shape), (1, expected_tokens))
            self.assertEqual(ts.dtype, torch.int64)
            self.assertTrue(bool((ts[:, :expected_prefix] == 0).all()))
            self.assertTrue(bool((ts[:, expected_prefix:] == 999).all()))
            self.assertTrue(torch.equal(xs[0], snapshots[0]))
            self.assertTrue(torch.equal(cs[0], snapshots[2]))
            calls.append(seq_len)
            return [xs[0] * 2 + cs[0][0, 0]]

        with cpu_cuda_dispatch() as (transfers, autocast, synchronize):
            output = native.predict(model, latent, times, context)
        self.assertEqual(calls, [expected_tokens])
        self.assertEqual(len(transfers), 3)
        for row, expected in zip(transfers, snapshots):
            self.assertTrue(torch.equal(row[0], expected))
            self.assertEqual(row[1], ('cuda:0',))
            self.assertEqual(row[2], {})
        autocast.assert_called_once_with('cuda', dtype=torch.bfloat16)
        synchronize.assert_called_once_with()
        self.assertEqual(output.device.type, 'cpu')
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), shape)
        self.assertTrue(torch.equal(output, latent * 2 + .25))
        for value, expected in zip((latent, times, context), snapshots):
            self.assertTrue(torch.equal(value, expected))

    def test_low_resolution_dispatch_and_input_ownership(self):
        self.check_dispatch((48, 5, 18, 32), 720, 144)

    def test_native_area_dispatch_and_input_ownership(self):
        self.check_dispatch((48, 5, 44, 78), 4290, 858)

    def assert_invalid_before_cuda(self, values):
        with mock.patch.object(torch.Tensor, 'to', side_effect=AssertionError('Transfer before validation')), \
                mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA before validation')), \
                mock.patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CUDA before validation')), \
                mock.patch.object(torch, 'autocast', side_effect=AssertionError('Autocast before validation')), \
                mock.patch.object(native._frozen, 'verify_sources', side_effect=AssertionError('Expected earlier rejection')):
            with self.assertRaises(ValueError):
                native.predict(mock.Mock(side_effect=AssertionError('Model before validation')), *values)

    def test_malformed_latent_rejected_before_cuda(self):
        good = inputs()
        bad_values = [None, [], torch.zeros(48, 18, 32), torch.zeros(48, 1, 18, 32),
                      torch.zeros(16, 5, 18, 32), torch.zeros(48, 5, 20, 32),
                      good[0].double(), good[0].half(), good[0].to_sparse(),
                      torch.empty_like(good[0], device='meta')]
        for number in (float('nan'), float('inf'), -float('inf')):
            value = good[0].clone()
            value[0, 0, 0, 0] = number
            bad_values.append(value)
        for value in bad_values:
            with self.subTest(type=type(value), shape=getattr(value, 'shape', None)):
                self.assert_invalid_before_cuda((value, good[1], good[2]))

    def test_malformed_token_times_rejected_before_cuda(self):
        good = inputs((48, 5, 44, 78))
        bad_values = [None, good[1][0], torch.zeros(1, 720, dtype=torch.int64),
                      good[1].int(), good[1].float(), good[1].to_sparse(),
                      torch.empty_like(good[1], device='meta')]
        for coordinate, number in [(0, 1), (857, 999), (858, 998), (4289, 998)]:
            value = good[1].clone()
            value[0, coordinate] = number
            bad_values.append(value)
        for number in (-1, 1000):
            value = good[1].clone()
            value[:, 858:] = number
            bad_values.append(value)
        for value in bad_values:
            with self.subTest(type=type(value), shape=getattr(value, 'shape', None)):
                self.assert_invalid_before_cuda((good[0], value, good[2]))

    def test_malformed_context_rejected_before_cuda(self):
        good = inputs()
        bad_values = [None, [], torch.zeros(4096), torch.zeros(1, 25, 4096),
                      torch.zeros(0, 4096), torch.zeros(513, 4096), torch.zeros(25, 4095),
                      good[2].double(), good[2].bfloat16(), good[2].to_sparse(),
                      torch.empty_like(good[2], device='meta')]
        for number in (float('nan'), float('inf'), -float('inf')):
            value = good[2].clone()
            value[0, 0] = number
            bad_values.append(value)
        for value in bad_values:
            with self.subTest(type=type(value), shape=getattr(value, 'shape', None)):
                self.assert_invalid_before_cuda((good[0], good[1], value))

    def test_time_and_context_boundary_values_are_accepted(self):
        for time, length in [(0, 1), (999, 512)]:
            with self.subTest(time=time, context_length=length), cpu_cuda_dispatch():
                values = inputs(time=time, context_length=length)
                output = native.predict(lambda xs, *_: [xs[0]], *values)
                self.assertTrue(torch.equal(output, values[0]))

    def test_fa2_required_and_fa3_rejected_before_transfer(self):
        values = inputs()
        for fa2, fa3 in [(False, False), (False, True), (True, True)]:
            with self.subTest(fa2=fa2, fa3=fa3), \
                    mock.patch.object(attention, 'FLASH_ATTN_2_AVAILABLE', fa2), \
                    mock.patch.object(attention, 'FLASH_ATTN_3_AVAILABLE', fa3), \
                    mock.patch.object(torch.Tensor, 'to', side_effect=AssertionError('Transfer before backend gate')):
                with self.assertRaisesRegex(RuntimeError, 'FlashAttention 2 only'):
                    native.predict(mock.Mock(), *values)

    def test_source_failure_precedes_transfer_and_model(self):
        values = inputs()
        with mock.patch.object(native._frozen, 'verify_sources', side_effect=ValueError('Pinned source changed')), \
                mock.patch.object(torch.Tensor, 'to', side_effect=AssertionError('Transfer before source gate')):
            with self.assertRaisesRegex(ValueError, 'Pinned source changed'):
                native.predict(mock.Mock(), *values)

    def test_malformed_model_result_is_rejected(self):
        values = inputs()
        results = [None, [], [values[0], values[0]], [None], [values[0][0]],
                   [torch.zeros_like(values[0], dtype=torch.int64)], [values[0].to_sparse()]]
        for result in results:
            with self.subTest(type=type(result)), cpu_cuda_dispatch():
                with self.assertRaises(RuntimeError):
                    native.predict(lambda *_: result, *values)

    def test_nonfinite_and_float32_overflow_outputs_are_rejected(self):
        values = inputs()
        for number, dtype in [(float('nan'), torch.float32), (float('inf'), torch.float32),
                              (-float('inf'), torch.float32), (1e300, torch.float64)]:
            with self.subTest(number=number), cpu_cuda_dispatch():
                result = torch.full_like(values[0], number, dtype=dtype)
                with self.assertRaises(FloatingPointError):
                    native.predict(lambda *_: [result], *values)

    def test_lower_precision_result_is_converted_to_float32(self):
        values = inputs()
        with cpu_cuda_dispatch():
            output = native.predict(lambda xs, *_: [xs[0].bfloat16()], *values)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, values[0]))

    def test_model_interruption_propagates_without_input_mutation(self):
        values = inputs()
        before = [v.clone() for v in values]
        with cpu_cuda_dispatch() as (_, _, synchronize):
            with self.assertRaises(KeyboardInterrupt):
                native.predict(mock.Mock(side_effect=KeyboardInterrupt('Fixture')), *values)
        synchronize.assert_not_called()
        for value, previous in zip(values, before):
            self.assertTrue(torch.equal(value, previous))

    def test_prediction_disables_input_gradient_recording(self):
        latent, times, context = inputs()
        latent.requires_grad_(True)
        context.requires_grad_(True)
        with cpu_cuda_dispatch():
            output = native.predict(lambda xs, ts, cs, n: [xs[0] + cs[0].mean()], latent, times, context)
        self.assertFalse(output.requires_grad)
        self.assertIsNone(latent.grad)
        self.assertIsNone(context.grad)


if __name__ == '__main__':
    unittest.main()
