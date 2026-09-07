"""Independent CPU checks for the optional Wan adapter, with no external weights.

Run with the isolated Wan environment: python test_independent.py.
The pooled-token test records a baseline limitation rather than a quality gain.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import ActionObservationAdapter, ActionObservationResidual
from compat import install, real_rope_apply, real_rope_params
from vendor.wan21.layers.rope import rope_apply as upstream_apply
from vendor.wan21.layers.rope import rope_params as upstream_params
from vendor.wan21.model import Wan21Model


def digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class IndependentCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9401)

    def test_actual_head_width_uses_unequal_rope_partitions(self):
        # Wan 1.3B head width is128, so its temporal/spatial dimensions are
        # 44/42/42, unlike the equal partitions exercised by the first test.
        raw = torch.randn(1, 62, 2, 128, requires_grad=True)
        grid = torch.tensor([[3, 4, 5]])
        old = torch.cat([upstream_params(64, d) for d in (44, 42, 42)], dim=1)
        new = torch.cat([real_rope_params(64, d) for d in (44, 42, 42)], dim=1)
        reference = upstream_apply(raw, grid, old)
        actual = real_rope_apply(raw, grid, new)
        probe = torch.randn_like(reference)
        grad0 = torch.autograd.grad((reference * probe).sum(), raw, retain_graph=True)[0]
        grad1 = torch.autograd.grad((actual * probe).sum(), raw)[0]
        self.assertTrue(torch.allclose(reference, actual, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad0, grad1, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(actual[:, 60:], raw[:, 60:]))

    def test_action_groups_keep_all_six_channels_and_order(self):
        residual = ActionObservationResidual(32, width=16)
        commands = torch.arange(16 * 6, dtype=torch.float32).reshape(1, 16, 6)
        seen = []
        handle = residual.action[0].register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        try:
            residual(torch.randn(1, 20, 32), commands, torch.randn(1, 16, 4, 4), (5, 2, 2))
        finally:
            handle.remove()
        self.assertEqual(tuple(seen[0].shape), (1, 5, 24))
        self.assertEqual(seen[0][:, 0].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(seen[0][:, 1:].reshape(1, 16, 6), commands))

    def test_observation_token_order_is_a_documented_limitation(self):
        residual = ActionObservationResidual(96, width=32)
        with torch.no_grad():
            torch.nn.init.normal_(residual.output.weight, std=.03)
        tokens = torch.randn(1, 12, 96)
        commands = torch.randn(1, 4, 6)
        observed = torch.randn(1, 16, 4, 8)
        rearranged = observed.flatten(2)[:, :, torch.randperm(32)].reshape_as(observed)
        original = residual(tokens, commands, observed, (2, 2, 3))
        permuted = residual(tokens, commands, rearranged, (2, 2, 3))
        changed_content = residual(tokens, commands, torch.zeros_like(observed), (2, 2, 3))
        self.assertFalse(torch.equal(observed, rearranged))
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6, rtol=1e-6))
        self.assertGreater((original - changed_content).abs().max().item(), 1e-5)

    def test_checkpoint_hooks_match_two_sequential_updates(self):
        install()
        config = dict(dim=96, ffn_dim=192, num_heads=4, num_layers=3,
                      freq_dim=16, text_dim=32, text_len=8)
        normal = Wan21Model(**config).float().eval()
        with torch.no_grad():
            torch.nn.init.normal_(normal.head.head.weight, std=.05)
        normal.requires_grad_(False)
        checkpointed = copy.deepcopy(normal)
        checkpointed.gradient_checkpointing = True
        adapter0 = ActionObservationAdapter(96, (0, 1, 2), width=32)
        adapter1 = copy.deepcopy(adapter0)
        optimizers = [torch.optim.AdamW(a.parameters(), lr=.003) for a in (adapter0, adapter1)]
        before = digest(normal)
        for _ in range(2):
            commands = torch.randn(1, 4, 6)
            observed = torch.randn(1, 16, 4, 6)
            latent = torch.randn(16, 2, 4, 6)
            context = [torch.randn(8, 32)]
            timestep = torch.tensor([370.])
            target = torch.randn(1, 16, 2, 4, 6)
            adapter0.attach(normal, commands, observed, (2, 2, 3))
            adapter1.attach(checkpointed, commands, observed, (2, 2, 3))
            try:
                y0 = normal([latent], timestep, context, 12)
                y1 = checkpointed([latent], timestep, context, 12)
                self.assertTrue(torch.allclose(y0, y1, atol=1e-6, rtol=1e-6))
                F.mse_loss(y0, target).backward()
                F.mse_loss(y1, target).backward()
                for p0, p1 in zip(adapter0.parameters(), adapter1.parameters()):
                    self.assertIsNotNone(p0.grad)
                    self.assertIsNotNone(p1.grad)
                    self.assertTrue(torch.isfinite(p0.grad).all())
                    self.assertTrue(torch.allclose(p0.grad, p1.grad, atol=1e-6, rtol=1e-6))
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            finally:
                # Hooks remain present for recomputation until both backwards finish.
                adapter0.detach()
                adapter1.detach()
        self.assertEqual(before, digest(normal))
        self.assertEqual(before, digest(checkpointed))
        self.assertTrue(all(p.grad is None for p in normal.parameters()))
        self.assertTrue(all(p.grad is None for p in checkpointed.parameters()))

    def test_observation_batch_mismatch_is_rejected(self):
        residual = ActionObservationResidual(32, width=16)
        with self.assertRaises(ValueError):
            residual(torch.randn(2, 12, 32), torch.randn(2, 4, 6),
                     torch.randn(1, 16, 4, 6), (2, 2, 3))

    def test_runtime_refuses_to_overwrite_prior_evidence(self):
        import runtime_probe
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            evidence = output / "metrics.json"
            original = b'{"status":"passed","marker":"preserve"}\n'
            evidence.write_bytes(original)
            args = ["runtime_probe.py", "--weights", str(Path(temporary) / "missing-weights"),
                    "--output", str(output), "--device", "cpu", "--synthetic-runtime-only"]
            with patch.object(sys, "argv", args):
                with self.assertRaises((ValueError, FileExistsError, SystemExit)):
                    runtime_probe.main()
            self.assertEqual(evidence.read_bytes(), original)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["metrics.json"])

    def test_selected_vendor_files_match_source_manifest(self):
        root = Path(__file__).resolve().parent
        manifest = json.loads((root / "vendor/source-manifest.json").read_text())
        for item in manifest["files"]:
            path = root / item["local_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["local_sha256"],
                             str(path.relative_to(root)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
