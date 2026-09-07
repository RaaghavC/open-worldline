# SPDX-License-Identifier: Apache-2.0
"""Independent CPU mathematics and input-contract checks, with random weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch import nn

from native_control.portable import create_model as native_model, cpu_sinusoidal_embedding
from native_control.sampling import make_scheduler
from tokenwise_time.portable import token_times, create_model, extend_model, block_forward, head_forward


CONFIG = dict(model_type='t2v', dim=256, ffn_dim=384, freq_dim=64, num_heads=2,
              num_layers=2, text_dim=32, text_len=8, in_dim=16, out_dim=16)


class IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_shape_scheduler_times_and_temporal_spatial_order(self):
        grid = torch.tensor([[5, 18, 32], [3, 2, 3]], dtype=torch.int64)
        scheduler = make_scheduler()
        retained = grid.clone()
        for step in scheduler.timesteps:
            supplied = torch.tensor([step.item(), step.item()], dtype=torch.int64)
            result = token_times(grid, supplied, 2884)
            for batch, (frames, height, width) in enumerate(grid.tolist()):
                for token in range(2884):
                    expected = 0 if token < height * width else step.item()
                    self.assertEqual(result[batch, token].item(), expected)
            self.assertTrue(torch.equal(supplied, torch.tensor([step.item()] * 2)))
        self.assertTrue(torch.equal(grid, retained))
        empty_prefix = token_times(grid, torch.tensor([500, 50]), 2884, observed_latent_frames=0)
        self.assertTrue(torch.equal(empty_prefix, torch.tensor([500, 50])[:, None].expand(2, 2884)))

    def test_clean_frame_time_zero_is_embedded_in_every_block_and_head(self):
        torch.manual_seed(245)
        model = create_model(**CONFIG).eval()
        latent = torch.randn(16, 3, 4, 6)
        context = torch.randn(3, 32)
        times = token_times(torch.tensor([[3, 2, 3]]), torch.tensor([777]), 20)
        observed = []

        def block_pre(_, args, kwargs):
            observed.append(kwargs['e'].detach().clone())

        hooks = [block.register_forward_pre_hook(block_pre, with_kwargs=True) for block in model.blocks]
        head_inputs = []
        hooks.append(model.head.register_forward_pre_hook(lambda _, args: head_inputs.append(args[1].detach().clone())))
        with torch.no_grad():
            output = model([latent], times, [context], 20)
            # Compute two scalar embeddings independently, then inspect every token.
            e0 = model.time_embedding(cpu_sinusoidal_embedding(model.freq_dim, torch.tensor([0])))
            e777 = model.time_embedding(cpu_sinusoidal_embedding(model.freq_dim, torch.tensor([777])))
            modulation0 = model.time_projection(e0).reshape(6, model.dim)
            modulation777 = model.time_projection(e777).reshape(6, model.dim)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(observed), 2)
        self.assertEqual(len(head_inputs), 1)
        for index in range(20):
            expected_modulation = modulation0 if index < 6 else modulation777
            expected_head = e0[0] if index < 6 else e777[0]
            for block_value in observed:
                torch.testing.assert_close(block_value[0, index], expected_modulation, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(head_inputs[0][0, index], expected_head, atol=2e-6, rtol=2e-6)
        self.assertGreater(e0.abs().sum().item(), 0)
        self.assertEqual(output[0].shape, latent.shape)

    def test_patch_embedding_token_order_matches_prefix_assignment(self):
        model = create_model(**CONFIG).eval()
        with torch.no_grad():
            model.patch_embedding.weight.zero_()
            model.patch_embedding.bias.zero_()
            model.patch_embedding.weight[0, 0, 0, 0, 0] = 1
        latent = torch.zeros(1, 16, 3, 4, 6)
        expected = []
        for frame in range(3):
            for y in range(2):
                for x in range(3):
                    value = frame * 100 + y * 10 + x
                    latent[0, 0, frame, 2*y, 2*x] = value
                    expected.append(value)
        tokens = model.patch_embedding(latent).flatten(2).transpose(1, 2)
        self.assertEqual(tokens[0, :, 0].tolist(), expected)
        times = token_times(torch.tensor([[3, 2, 3]]), torch.tensor([999]), 18)
        self.assertEqual([int(v // 100) for v in tokens[0, times[0] == 0, 0]], [0] * 6)

    def test_block_and_head_use_per_token_shift_scale_and_gates(self):
        class Attention(nn.Module):
            def forward(self, x, *args):
                return x * .3

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 4
                self.modulation = nn.Parameter(torch.arange(24).reshape(1, 6, 4).float() * .01)
                self.norm1 = self.norm2 = self.norm3 = nn.Identity()
                self.self_attn = self.cross_attn = Attention()
                self.ffn = nn.Linear(4, 4)

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.modulation = nn.Parameter(torch.arange(8).reshape(1, 2, 4).float() * .01)
                self.norm = nn.Identity()
                self.head = nn.Linear(4, 5)

        torch.manual_seed(593)
        block, head = Block(), Head()
        x, e = torch.randn(2, 7, 4), torch.randn(2, 7, 6, 4) * .1
        actual = block_forward(block, x, e, None, None, None, None, None)
        expected = torch.empty_like(actual)
        for batch in range(2):
            for token in range(7):
                p = block.modulation[0] + e[batch, token]
                value = x[batch, token]
                value = value + .3 * (value * (1 + p[1]) + p[0]) * p[2]
                value = value + .3 * value
                value = value + block.ffn(value * (1 + p[4]) + p[3]) * p[5]
                expected[batch, token] = value
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        head_e = torch.randn(2, 7, 4)
        actual_head = head_forward(head, actual, head_e)
        expected_head = torch.stack([torch.stack([
            head.head(actual[batch, token] * (1 + head.modulation[0, 1] + head_e[batch, token])
                      + head.modulation[0, 0] + head_e[batch, token])
            for token in range(7)]) for batch in range(2)])
        torch.testing.assert_close(actual_head, expected_head, atol=2e-7, rtol=2e-6)

    def test_instance_extension_preserves_tensors_parameters_and_native_behavior(self):
        original = native_model(portable=True, **CONFIG)
        other = native_model(portable=True, **CONFIG)
        before = {name: value.clone() for name, value in original.state_dict().items()}
        identities = {name: id(value) for name, value in original.named_parameters()}
        old_function = other.forward.__func__
        self.assertIs(extend_model(original), original)
        self.assertIs(other.forward.__func__, old_function)
        self.assertEqual(identities, {name: id(value) for name, value in original.named_parameters()})
        self.assertEqual(list(before), list(original.state_dict()))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in original.state_dict().items()))
        with self.assertRaises(ValueError):
            extend_model(original)

    def test_invalid_grids_and_scalar_times_are_rejected(self):
        for grid, time_value, length, prefix in [
            (torch.tensor([[0, 2, 2]]), torch.tensor([999]), 20, 1),
            (torch.tensor([[2, 2, 2]]), torch.tensor([999]), 7, 1),
            (torch.tensor([[2, 2, 2]]), torch.tensor([999]), 8, 2),
            (torch.tensor([[2, 2, 2]]), torch.tensor([1001]), 8, 1),
            (torch.tensor([[2., 2., 2.]]), torch.tensor([500]), 8, 1),
        ]:
            with self.assertRaises(ValueError):
                token_times(grid, time_value, length, observed_latent_frames=prefix)
        model = create_model(**CONFIG)
        with self.assertRaises(ValueError):
            model([torch.zeros(16, 3, 4, 6)], torch.tensor([500]), [torch.zeros(1, 32)], 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Evidence output must be new')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
        warnings.filterwarnings('ignore', message="User provided device_type of 'cuda'.*")
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests))
    here = Path(__file__).parent
    files = [Path(__file__), here / 'portable.py', here.parent / 'native_control/portable.py',
             here.parent / 'native_control/sampling.py', here.parent / 'native_control/vendor/model.py',
             here.parent / 'native_control/vendor/fm_solvers_unipc.py']
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'seconds': time.perf_counter() - started,
              'device': 'cpu', 'torch': torch.__version__, 'pretrained_weights_loaded': False,
              'source_sha256': {str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'scope': 'Token order, exact integer noise labels, every block/head modulation, parameter identity and input guards. Untrained mathematical extension only; no GPU, image sampling or world-quality result.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
