# SPDX-License-Identifier: Apache-2.0
"""CPU mocks and pixel checks only; no native model, weights or CUDA execution."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from . import codec
from ..cuda_reference.decode import load_codec as frozen_load_codec


class FakeCodec:
    def __init__(self, height, width, failure=None, malformed=None):
        self.height, self.width = height, width
        self.failure, self.malformed = failure, malformed
        self.clears = 0
        self.calls = []

    def clear_cache(self):
        self.clears += 1

    def _result(self, operation, value, scale, shape):
        self.calls.append((operation, tuple(value.shape), value.dtype, scale,
                           torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        if self.failure:
            raise self.failure
        result = torch.zeros((), dtype=torch.float32).expand(shape)
        if self.malformed == 'shape':
            return result[..., :-1]
        if self.malformed == 'dtype':
            return result.to(torch.float16)
        if self.malformed == 'finite':
            return torch.full((), float('nan')).expand(shape)
        return result

    def encode(self, value, scale):
        return self._result('encode', value, scale,
                            (1, 48, 1, self.height // 16, self.width // 16))

    def decode(self, value, scale):
        return self._result('decode', value, scale,
                            (1, 3, 17, self.height, self.width))


@contextmanager
def cpu_boundaries():
    with ExitStack() as stack:
        transfer = stack.enter_context(patch.object(codec, '_to_cuda', side_effect=lambda value: value))
        sync = stack.enter_context(patch.object(codec.torch.cuda, 'synchronize'))
        # No CUDA autocast is entered during these CPU fixtures. Its requested
        # arguments are asserted separately, without substituting codec math.
        autocast = stack.enter_context(patch.object(codec.torch, 'autocast'))
        yield transfer, sync, autocast


class CodecTests(unittest.TestCase):
    def test_loader_and_source_pins_are_reused(self):
        self.assertIs(codec.load_codec, frozen_load_codec)
        codec.verify_sources()

    def test_both_native_shapes_independent_encode_and_full_decode(self):
        scale = object()
        with cpu_boundaries() as (transfer, sync, autocast):
            for height, width in sorted(codec.SIZES):
                model = FakeCodec(height, width)
                image = torch.zeros(1, 3, 1, height, width)
                observation = codec.encode(model, scale, image, height, width)
                self.assertEqual(tuple(observation.shape), (1, 48, 1, height // 16, width // 16))
                latent = observation[:, :, 0].unsqueeze(2).expand(1, 48, 5, height // 16, width // 16)[0]
                video = codec.decode(model, scale, latent, height, width)
                self.assertEqual(tuple(video.shape), (1, 3, 17, height, width))
                self.assertEqual(video.dtype, torch.float32)
                self.assertEqual(video.device.type, 'cpu')
                self.assertEqual(model.clears, 4)
                self.assertEqual(model.calls[0][:3], ('encode', (1, 3, 1, height, width), torch.float32))
                self.assertEqual(model.calls[1][:3], ('decode', (1, 48, 5, height // 16, width // 16), torch.float32))
                self.assertTrue(all(call[3] is scale and call[4:] == (False, True) for call in model.calls))
            self.assertEqual(transfer.call_count, 4)
            self.assertEqual(sync.call_count, 4)
            self.assertEqual(autocast.call_count, 4)
            for call in autocast.call_args_list:
                self.assertEqual(call.args, ('cuda',))
                self.assertEqual(call.kwargs, {'enabled': False})

    def test_invalid_inputs_are_rejected_before_native_access(self):
        model = FakeCodec(288, 512)
        image = torch.zeros(1, 3, 1, 288, 512)
        with cpu_boundaries() as (transfer, _, _):
            for size in ((288, 513), (704.0, 1248), (512, 288), (True, 512)):
                with self.assertRaises(ValueError):
                    codec.encode(model, None, image, *size)
            bad_images = (image[:, :, :0], image.to(torch.float16),
                          torch.full_like(image, float('inf')), image + 2)
            for bad in bad_images:
                with self.assertRaises(ValueError):
                    codec.encode(model, None, bad, 288, 512)
            for bad in (torch.zeros(48, 1, 18, 32), torch.zeros(48, 5, 18, 32, dtype=torch.float64),
                        torch.full((48, 5, 18, 32), float('nan'))):
                with self.assertRaises(ValueError):
                    codec.decode(model, None, bad, 288, 512)
            self.assertEqual(transfer.call_count, 0)
            self.assertEqual(model.calls, [])

    def test_cleanup_after_native_interrupt_and_output_validation_failure(self):
        image = torch.zeros(1, 3, 1, 288, 512)
        latent = torch.zeros(48, 5, 18, 32)
        with cpu_boundaries():
            for function, value in ((codec.encode, image), (codec.decode, latent)):
                for failure in (RuntimeError('native failed'), KeyboardInterrupt()):
                    model = FakeCodec(288, 512, failure=failure)
                    with self.assertRaises(type(failure)):
                        function(model, None, value, 288, 512)
                    self.assertEqual(model.clears, 2)
                for malformed in ('shape', 'dtype', 'finite'):
                    model = FakeCodec(288, 512, malformed=malformed)
                    with self.assertRaises(ValueError):
                        function(model, None, value, 288, 512)
                    self.assertEqual(model.clears, 2)

    def test_decode_clamps_only_finite_native_output(self):
        model = FakeCodec(288, 512)
        native = torch.tensor([-2., .125, 3.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 17, 288, 512)
        with cpu_boundaries(), patch.object(model, 'decode', return_value=native):
            video = codec.decode(model, None, torch.zeros(48, 5, 18, 32), 288, 512)
        self.assertEqual(video[0, :, 0, 0, 0].tolist(), [-1., .125, 1.])
        self.assertEqual(native[0, :, 0, 0, 0].tolist(), [-2., .125, 3.])
        self.assertEqual(model.clears, 2)

    def test_full_pixel_mapping_contact_and_gif_timing(self):
        height, width = 288, 512
        # Different RGB values distinguish every frame, with endpoints and
        # rounding ties represented directly in the retained FP32 array.
        video = torch.empty(1, 3, 17, height, width)
        for index in range(17):
            video[:, :, index] = -1 + index / 8
        video[0, :, 0, 0, :4] = torch.tensor([-1., 0., 1., .5])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = codec.images(video, root, height, width)
            self.assertEqual(report['frames'], 17)
            self.assertEqual(report['preview_encoded_frame_durations_ms'], [120] * 17)
            self.assertEqual(report['preview_encoded_duration_ms'], 2040)
            self.assertEqual(report['preview_requested_playback_fps'], 1000 / 120)
            with Image.open(root / 'comparison.png') as contact:
                self.assertEqual(contact.size, (2 * width, 5 * (height + 28)))
                for index in range(17):
                    expected = np.rint((video[0, :, index].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    with Image.open(root / 'frames' / f'{index:04d}.png') as frame:
                        self.assertEqual(frame.mode, 'RGB')
                        self.assertEqual(frame.size, (width, height))
                        np.testing.assert_array_equal(np.asarray(frame), expected)
                    if index in codec.CONTACT_FRAMES:
                        slot = codec.CONTACT_FRAMES.index(index)
                        x, y = slot % 2 * width, slot // 2 * (height + 28) + 28
                        np.testing.assert_array_equal(np.asarray(contact.crop((x, y, x + width, y + height))), expected)
            with self.assertRaises(FileExistsError):
                codec.images(video, root, height, width)
        self.assertEqual(video[0, 0, 0, 0, :4].tolist(), [-1., 0., 1., .5])

    def test_image_export_rejects_invalid_values_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'unused'
            for value in (float('nan'), 2.):
                bad = torch.full((), value).expand(1, 3, 17, 288, 512)
                with self.assertRaises(ValueError):
                    codec.images(bad, root, 288, 512)
                self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
