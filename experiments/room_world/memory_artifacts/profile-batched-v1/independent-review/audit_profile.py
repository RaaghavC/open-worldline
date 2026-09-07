"""Read-only CPU artifact audit of the completed batched profile and its stepwise reference."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile', type=Path, required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--stepwise', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT, REPO, OLD = args.profile.resolve(), args.repo.resolve(), args.stepwise.resolve()
if args.output.exists():
    raise FileExistsError('Audit output must be new')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(values):
    return {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
        'sha256': hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for key, value in sorted(values.items())}


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


study = json.loads((ROOT / 'study.json').read_text())
old_study = json.loads((OLD / 'study.json').read_text())
assert study['sequence_path'] == 'batched'
assert old_study.get('sequence_path', 'stepwise') == 'stepwise'
assert old_study['status'] == 'complete' and old_study['matched_budgets'] is True
for key in ('requested_updates_per_arm', 'seeds', 'arms', 'max_seconds_per_arm', 'base_checkpoint_sha256', 'training', 'validation', 'schedules'):
    assert study[key] == old_study[key]
assert set(study['source_sha256']) == {'memory_train.py', 'memory_evaluate.py', 'memory_model.py', 'memory_data.py', 'model.py', 'memory_sequence.py'}
assert study['status'] == 'complete' and study['phase'] == 'profile' and study['matched_budgets'] is True
assert study['requested_updates_per_arm'] == 50 and study['seeds'] == [20260907]
assert not study['reserved_test_opened'] and not study['evaluation_executed'] and not study['controls_evaluated']
for name, expected in study['source_sha256'].items():
    assert sha(ROOT / 'measured-source' / (name + '.txt')) == expected
    if name in ('memory_model.py', 'memory_data.py', 'model.py'):
        assert old_study['source_sha256'][name] == expected
for name, expected in old_study['source_sha256'].items():
    assert sha(OLD / 'measured-source' / (name + '.txt')) == expected
base = REPO / 'experiments/room_world/artifacts/predictor/model.pt'
assert sha(base) == study['base_checkpoint_sha256'] == '9376e684e3abd103313d13d04bd4a1afc1b202fccb7e7b6fe133e7232a486d69'
base_values = torch.load(base, weights_only=True, map_location='cpu')['state_dict']
initial = torch.load(ROOT / '20260907/initial-memory.pt', weights_only=True, map_location='cpu')
assert sum(t.numel() for t in initial.values()) == 24960
old_initial = torch.load(OLD / '20260907/initial-memory.pt', weights_only=True, map_location='cpu')
assert hashes(initial) == hashes(old_initial)
assert sha(ROOT / '20260907/initial-memory.pt') == sha(OLD / '20260907/initial-memory.pt')
finite(initial)
results, raw = [], []
for mode in ('carry', 'reset'):
    entries = [r for r in study['runs'] if r['mode'] == mode and r['seed'] == 20260907]
    assert len(entries) == 1
    entry = entries[0]
    path = ROOT / entry['metrics']
    assert sha(path) == entry['metrics_sha256']
    measured = json.loads(path.read_text())
    raw.append(measured)
    assert entry['sequence_path'] == measured['sequence_path'] == 'batched'
    assert measured['loss_implementation'] == 'experiments.room_world.memory_sequence.sequence_loss_batched'
    old_entry = next(row for row in old_study['runs'] if row['mode'] == mode and row['seed'] == 20260907)
    old_path = OLD / old_entry['metrics']
    assert sha(old_path) == old_entry['metrics_sha256']
    previous = json.loads(old_path.read_text())
    for key in ('initial_memory', 'base_before', 'base_after', 'data', 'scene_schedule', 'completed_scene_schedule', 'optimizer', 'parameter_counts'):
        assert measured[key] == previous[key]
    terminal = json.loads((path.parent.parent / 'terminal.json').read_text())
    assert measured['status'] == terminal['status'] == 'complete' and terminal['exit_code'] == 0 and terminal['reason'] is None
    assert measured['mode'] == mode and measured['seed'] == 20260907 and measured['phase'] == 'profile'
    assert measured['completed_updates'] == measured['requested_updates'] == entry['completed_updates'] == 50
    assert measured['scene_schedule'] == measured['completed_scene_schedule'] == study['schedules']['20260907']
    updates = measured['updates']
    assert len(updates) == 50 and [u['update'] for u in updates] == list(range(1, 51))
    assert [u['scene'] for u in updates] == measured['scene_schedule']
    assert set(measured['scene_schedule']) == set(range(5000, 5032))
    assert measured['initial_memory'] == hashes(initial)
    assert measured['base_before'] == measured['base_after'] == hashes(base_values)
    assert torch.device(measured['device']).type == 'mps' and measured['runtime']['mps_fallback'] == '0'
    assert measured['parameter_counts']['trainable'] == 24960
    checkpoints = {}
    for name, field in [('memory-final.pt', 'memory_final_sha256'), ('memory-last.pt', 'memory_last_sha256'), ('recovery-last.pt', 'recovery_last_sha256')]:
        file = path.parent / name
        assert sha(file) == measured[field]
        loaded = torch.load(file, weights_only=True, map_location='cpu')
        finite(loaded)
        assert loaded['mode'] == mode and loaded['seed'] == 20260907 and loaded['completed_updates'] == 50
        assert loaded['sequence_path'] == 'batched'
        assert loaded['base_checkpoint_sha256'] == sha(base)
        assert hashes(loaded['memory_state_dict']) == measured['final_memory']
        checkpoints[name] = loaded
    recovery = checkpoints['recovery-last.pt']
    assert recovery['completed_scene_schedule'] == measured['scene_schedule']
    assert set(recovery['rng']) == {'torch_cpu', 'torch_mps', 'python', 'numpy'}
    assert len(recovery['optimizer_state_dict']['state']) == len(initial)
    optimizer = recovery['optimizer_state_dict']
    assert len(optimizer['param_groups']) == 1
    group = optimizer['param_groups'][0]
    assert group['params'] == list(range(len(initial)))
    for key, expected in [('lr', .0003), ('weight_decay', .0001), ('betas', (.9, .999)), ('eps', 1e-8)]:
        assert group[key] == expected
    for index, (name, value) in enumerate(initial.items()):
        state = optimizer['state'][index]
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and state['step'].item() == 50
        assert state['exp_avg'].shape == state['exp_avg_sq'].shape == value.shape
        assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == value.dtype
        assert (state['exp_avg_sq'] >= 0).all()
    for name, value in checkpoints['memory-final.pt']['memory_state_dict'].items():
        assert torch.equal(value, recovery['memory_state_dict'][name])
    assert all(not torch.equal(value, initial[name]) for name, value in recovery['memory_state_dict'].items())
    for update in updates:
        assert math.isfinite(update['loss']) and update['loss'] >= 0
        assert math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip'] > 0
        assert math.isfinite(update['seconds_with_recovery_write']) and update['seconds_with_recovery_write'] > 0
    assert all(row['rss_bytes'] <= 18 * 2**30 and row['available_bytes'] >= 2 * 2**30 for row in terminal['samples'])
    assert max(u['mps_driver_bytes'] for u in updates) <= 18 * 2**30
    results.append({'mode': mode, 'completed_updates': 50, 'training_seconds': measured['elapsed_seconds'],
        'supervised_seconds': terminal['elapsed_seconds'],
        'stepwise_training_seconds': previous['elapsed_seconds'],
        'observed_stepwise_to_batched_training_time_ratio': previous['elapsed_seconds'] / measured['elapsed_seconds'],
        'linear_1024_update_training_seconds': measured['elapsed_seconds'] * 1024 / 50,
        'update_seconds_mean': statistics.mean(u['seconds_with_recovery_write'] for u in updates),
        'update_seconds_median': statistics.median(u['seconds_with_recovery_write'] for u in updates),
        'update_seconds_range': [min(u['seconds_with_recovery_write'] for u in updates), max(u['seconds_with_recovery_write'] for u in updates)],
        'loss_first': updates[0]['loss'], 'loss_last': updates[-1]['loss'],
        'loss_range': [min(u['loss'] for u in updates), max(u['loss'] for u in updates)],
        'gradient_norm_range': [min(u['gradient_norm_before_clip'] for u in updates), max(u['gradient_norm_before_clip'] for u in updates)],
        'all_memory_tensors_changed': True, 'final_sha256': measured['memory_final_sha256'], 'recovery_sha256': measured['recovery_last_sha256'],
        'peak_mps_driver_sampled_gib': max(u['mps_driver_bytes'] for u in updates) / 2**30,
        'peak_mps_active_sampled_gib': max(u['mps_active_bytes'] for u in updates) / 2**30,
        'peak_rss_sampled_gib': max(row['rss_bytes'] for row in terminal['samples']) / 2**30,
        'minimum_available_sampled_gib': min(row['available_bytes'] for row in terminal['samples']) / 2**30})
for key in ('optimizer', 'initial_memory', 'base_before', 'base_after', 'data', 'scene_schedule', 'completed_scene_schedule'):
    assert raw[0][key] == raw[1][key]
report = {'schema': 'worldline-room-memory-batched-profile-independent-audit-v1', 'status': 'passed', 'device_used_for_audit': 'cpu',
    'audit_source_sha256': sha(__file__), 'study_sha256': sha(ROOT / 'study.json'), 'source_sha256': study['source_sha256'],
    'base_checkpoint_sha256': sha(base), 'initial_memory_sha256': sha(ROOT / '20260907/initial-memory.pt'),
    'sequence_path': 'batched', 'old_stepwise_study_sha256': sha(OLD / 'study.json'),
    'same_initialization_schedule_data_optimizer_as_stepwise': True, 'checkpoint_path_labels_verified': True,
    'optimizer_tensor_shapes_and_nonnegative_second_moments_verified': True,
    'matched_realized_updates': True, 'completed_updates_each': 50, 'seed': 20260907, 'full_temporal_graph_steps': 65,
    'base_tensors_equal_to_published_checkpoint': True, 'recovery_optimizer_steps_all50': True, 'reserved_test_opened': False,
    'quality_evaluation_executed': False, 'arms': results,
    'limits': 'One 50-update profile per arm, one initialization seed. Elapsed-time ratios are observed across two separate runs, not a multi-run speed benchmark. The 1024-update estimates are linear, exclude initialization and evaluation, and must retain the 600-second cap. No validation/control result or learned-memory claim. Metal/RSS samples overlap and are not proven instantaneous peaks. Original L1-gradient failure remains unchanged.'}
args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2))
