"""Read-only comparison of one CPU and one MPS native5B forward pair."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'outputs/open-worldline'))
from experiments.wan22_native.load_weights import sha256, tensor_sha256

CPU = ROOT / 'work/wan22-native-cpu-pair-v1'
MPS = ROOT / 'work/wan22-native-pair-v1'
OUT = ROOT / 'work/wan22-device-comparison-v1'
OUT.mkdir(exist_ok=False)

reports = {}
for label, path in (('cpu', CPU), ('mps', MPS)):
    r = json.loads((path / 'metrics.json').read_text())
    terminal = json.loads((path / 'terminal.json').read_text())
    assert r['status'] == 'passed' and r['device'] == label
    assert terminal['status'] == 'complete' and terminal['exit_code'] == 0
    assert not (path / 'watchdog-stop.json').exists()
    for name, digest in r['output_sha256'].items(): assert sha256(path / name) == digest
    reports[label] = r
for key in ('source_sha256', 'reused_source_sha256', 'sampling_configuration', 'seed', 'latent_shape',
            'tokens', 'observed_tokens', 'text', 'observation'):
    assert reports['cpu'][key] == reports['mps'][key], key
cpu_load = json.loads((CPU / 'weight-load.json').read_text())
mps_load = json.loads((MPS / 'weight-load.json').read_text())
assert cpu_load == mps_load

def tensors(path):
    with safe_open(path, framework='pt', device='cpu') as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}

inputs = {label: tensors(path / 'inputs.safetensors') for label, path in (('cpu', CPU), ('mps', MPS))}
assert inputs['cpu'].keys() == inputs['mps'].keys()
assert all(torch.equal(value, inputs['mps'][name]) for name, value in inputs['cpu'].items())
outputs = {label: tensors(path / 'outputs.safetensors') for label, path in (('cpu', CPU), ('mps', MPS))}
assert outputs['cpu'].keys() == outputs['mps'].keys() == {
    'positive_velocity', 'negative_velocity', 'guided_velocity', 'one_step_latent'}

def score(cpu, mps):
    a, b = cpu.double().flatten(), mps.double().flatten()
    assert a.numel() > 0 and torch.isfinite(a).all() and torch.isfinite(b).all()
    difference = b - a
    mse = float(difference.square().mean())
    rms = float(a.square().mean().sqrt())
    product = float(a.norm() * b.norm())
    return {'elements': a.numel(), 'mean_absolute_difference': float(difference.abs().mean()),
        'root_mean_square_difference': mse**.5, 'maximum_absolute_difference': float(difference.abs().max()),
        'cpu_root_mean_square': rms, 'relative_rms_difference': mse**.5 / rms if rms else None,
        'cosine_similarity': float(torch.dot(a, b)) / product if product else None,
        'exactly_equal_elements': int((a == b).sum())}

comparison = {}
for name, cpu in outputs['cpu'].items():
    mps = outputs['mps'][name]
    assert cpu.shape == mps.shape == (48, 5, 18, 32)
    assert cpu.dtype == mps.dtype == torch.float32
    comparison[name] = {'all_latents': score(cpu, mps), 'observed_latent': score(cpu[:, :1], mps[:, :1]),
        'future_latents': score(cpu[:, 1:], mps[:, 1:]),
        'per_latent_frame': [score(cpu[:, i], mps[:, i]) for i in range(5)]}
for label in ('cpu', 'mps'):
    assert torch.equal(outputs[label]['one_step_latent'][:, :1], inputs[label]['observation'][0])

record = {
    'status': 'comparison_complete', 'source_sha256': sha256(Path(__file__)),
    'cpu_metrics_sha256': sha256(CPU / 'metrics.json'), 'mps_metrics_sha256': sha256(MPS / 'metrics.json'),
    'input_tensors_bit_exact': True, 'text_source_observation_sampling_identities_equal': True,
    'all_loaded_weight_records_equal': True, 'weight_tensors': len(cpu_load['tensors']),
    'known_prefix_preserved_on_both_devices': True,
    'output_tensor_hashes': {label: {key: tensor_sha256(value) for key, value in values.items()}
                             for label, values in outputs.items()},
    'comparison': comparison,
    'model_inference_performed_by_this_script': False,
    'scope': 'One initial positive/negative pair at the exact same inputs and declared mixed precision. CPU is a second execution backend, not a full-FP32 or CUDA ground truth. This comparison does not establish full-rollout agreement or explain visual artifacts by itself.',
}
(OUT / 'report.json').write_text(json.dumps(record, indent=2) + '\n')
shutil.copyfile(Path(__file__), OUT / 'compare-wan22-devices.py')
print(json.dumps({name: values['future_latents'] for name, values in comparison.items()}, indent=2))
