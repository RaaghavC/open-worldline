# SPDX-License-Identifier: Apache-2.0
"""Small full-forward parity against the literal pinned official FP32 model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from native_control.portable import create_model


def measured_forward(model, inputs, context, timestep):
    trace = {}
    hooks = []
    modules = {'time_embedding': model.time_embedding, 'time_projection': model.time_projection,
               'text_embedding': model.text_embedding, 'head': model.head,
               **{f'block_{i}': block for i, block in enumerate(model.blocks)}}
    for name, module in modules.items():
        def capture(module, args, output, name=name):
            trace[name] = output.detach().clone()
        hooks.append(module.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            output = model(inputs, torch.full((len(inputs),), timestep, dtype=torch.int64), context, 14)
    finally:
        for hook in hooks:
            hook.remove()
    for index, value in enumerate(output):
        trace[f'velocity_{index}'] = value.detach()
    return trace


def run():
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    configuration = dict(model_type='t2v', dim=256, ffn_dim=512, freq_dim=256, num_heads=2,
                         num_layers=2, text_dim=4096, text_len=512, in_dim=16, out_dim=16)
    reference = create_model(portable=False, **configuration).eval()
    candidate = create_model(portable=True, **configuration).eval()
    with torch.no_grad():
        # Official initialization zeros the output projection. Make final outputs
        # nontrivial before copying exactly the same state into both models.
        reference.head.head.weight.normal_(mean=0, std=.025)
        reference.head.head.bias.normal_(mean=0, std=.01)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    inputs = [torch.randn(16, 2, 4, 6), torch.randn(16, 1, 4, 4)]
    contexts = [torch.randn(25, 4096)*.2, torch.randn(1, 4096)*.2]
    rows = []
    for timestep in (999, 500, 50):
        official = measured_forward(reference, inputs, contexts, timestep)
        portable = measured_forward(candidate, inputs, contexts, timestep)
        for key in official:
            a, b = official[key], portable[key]
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                raise AssertionError(f'Non-FP32 output at {timestep}/{key}')
            difference = (a-b).abs()
            row = {'timestep': timestep, 'stage': key, 'shape': list(a.shape),
                   'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()),
                   'reference_max_abs': float(a.abs().max())}
            if not torch.isfinite(b).all() or not torch.allclose(a, b, atol=2e-5, rtol=2e-5):
                raise AssertionError(f'Official FP32 parity failed: {row}')
            rows.append(row)
    return {'status': 'passed', 'configuration': configuration, 'timesteps': [999, 500, 50],
            'atol': 2e-5, 'rtol': 2e-5, 'head_width': 128, 'rows': rows,
            'reference': 'Literal unmodified official model source and FP32 parameters/activations. Both paths use declared FP32 SDPA instead of CUDA FlashAttention.',
            'candidate_changes': 'CPU-double timestep calculation transferred as FP32; real-valued FP32 RoPE tables and rotations. No neural layer or activation changes.',
            'coverage_limits': 'Small random-weight CPU model; not pretrained full-model parity against CUDA BF16/FlashAttention or MPS arithmetic.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output file must be new')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'running'}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
            warnings.filterwarnings('ignore', message="User provided device_type of 'cuda'.*")
            report = run()
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        files = [Path(__file__), Path(__file__).with_name('portable.py'), Path(__file__).parent/'vendor/model.py']
        report.update(elapsed_seconds=time.perf_counter()-started, torch=torch.__version__,
            source_sha256={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
