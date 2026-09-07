"""CPU numerical and gradient checks, without pretrained weights or a GPU."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from vendor.wan21.layers.rope import rope_params as upstream_params, rope_apply as upstream_apply
from vendor.wan21.model import Wan21Model
from vendor.wan21 import attention
from compat import install, real_rope_params, real_rope_apply, portable_attention
from adapter import ActionObservationAdapter


def digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run():
    torch.set_num_threads(2)
    torch.manual_seed(731)
    heads, dim = 2, 24
    grids = torch.tensor([[2, 3, 4]])
    raw = torch.randn(1, 26, heads, dim, requires_grad=True)
    f0 = torch.cat([upstream_params(1024, 8)] * 3, dim=1)
    f1 = torch.cat([real_rope_params(1024, 8)] * 3, dim=1)
    y0 = upstream_apply(raw, grids, f0)
    y1 = real_rope_apply(raw, grids, f1)
    upstream_grad = torch.autograd.grad(y0.square().sum(), raw, retain_graph=True)[0]
    portable_grad = torch.autograd.grad(y1.square().sum(), raw)[0]
    rope_error = (y0 - y1).abs().max().item()
    rope_grad_error = (upstream_grad - portable_grad).abs().max().item()
    assert rope_error < 1e-6 and rope_grad_error < 3e-6
    # Padding must not allow padded keys to influence valid query outputs.
    q = torch.randn(1, 5, 2, 12)
    k = torch.randn(1, 6, 2, 12)
    v = torch.randn_like(k)
    masked = portable_attention(q, k, v, q_lens=torch.tensor([3]), k_lens=torch.tensor([4]))
    reference = portable_attention(q[:, :3], k[:, :4], v[:, :4])
    assert torch.allclose(masked[:, :3], reference, atol=1e-6)
    assert masked[:, 3:].count_nonzero() == 0
    attention.attention = portable_attention
    config = dict(dim=96, ffn_dim=192, num_heads=4, num_layers=3, freq_dim=16, text_dim=32, text_len=8)
    original = Wan21Model(**config).float().eval()
    torch.nn.init.normal_(original.head.head.weight, std=.05)
    x = torch.randn(16, 2, 4, 6, requires_grad=True)
    ctx = [torch.randn(8, 32)]
    t = torch.tensor([500.])
    output0 = original([x], t, ctx, 12)
    gx0 = torch.autograd.grad(output0.square().mean(), x)[0]
    install()
    core = Wan21Model(**config).float().eval()
    core.load_state_dict(original.state_dict(), strict=True)
    output1 = core([x], t, ctx, 12)
    gx1 = torch.autograd.grad(output1.square().mean(), x)[0]
    core_error = (output0 - output1).abs().max().item()
    core_grad_error = (gx0 - gx1).abs().max().item()
    assert core_error < 1e-5 and core_grad_error < 1e-6
    core.requires_grad_(False)
    x = x.detach()
    before = digest(core)
    baseline = core([x], t, ctx, 12).detach()
    adapter = ActionObservationAdapter(96, (0, 1, 2), width=32)
    actions = torch.randn(1, 4, 6)
    observed = torch.randn(1, 16, 4, 6)
    adapter.attach(core, actions, observed, (2, 2, 3))
    zero_output = core([x], t, ctx, 12)
    assert torch.equal(baseline, zero_output), 'Zero-init must leave the core output exactly unchanged'
    target = torch.randn_like(zero_output)
    loss = F.mse_loss(zero_output, target)
    loss.backward()
    grads = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None for p in core.parameters())
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.003)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    normal = core([x], t, ctx, 12)
    normal_loss = F.mse_loss(normal, target)
    normal_loss.backward()
    gradients0 = [p.grad.clone() for p in adapter.parameters()]
    optimizer.zero_grad(set_to_none=True)
    core.gradient_checkpointing = True
    checkpointed = core([x], t, ctx, 12)
    F.mse_loss(checkpointed, target).backward()
    checkpoint_grad_error = max((p.grad - g).abs().max().item() for p, g in zip(adapter.parameters(), gradients0))
    assert torch.allclose(normal, checkpointed, atol=1e-6)
    assert checkpoint_grad_error < 1e-6
    original_actions = checkpointed.detach()
    adapter.detach()
    adapter.attach(core, actions.flip(1), observed, (2, 2, 3))
    reordered = core([x], t, ctx, 12).detach()
    action_difference = (original_actions - reordered).abs().mean().item()
    assert action_difference > 1e-8, 'Ordered actions should affect the trained nonzero adapter'
    adapter.detach()
    assert before == digest(core)
    assert all(p.grad is None for p in core.parameters())
    return {'status': 'passed', 'device': 'cpu', 'pretrained_weights': False,
            'quality_evidence': False, 'torch': torch.__version__,
            'rope_max_abs_error': rope_error, 'rope_gradient_max_abs_error': rope_grad_error,
            'full_tiny_core_max_abs_error': core_error, 'full_tiny_core_gradient_max_abs_error': core_grad_error,
            'zero_init_bit_exact': True, 'checkpoint_gradient_max_abs_error': checkpoint_grad_error,
            'ordered_action_mean_abs_difference_after_one_update': action_difference,
            'all_adapter_gradients_finite': True, 'base_parameters_unchanged': True,
            'base_sha256': before, 'note': 'Random tiny core, synthetic tensors, numerical and gradient checks only.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
