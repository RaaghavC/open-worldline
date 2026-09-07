"""CPU arithmetic/file audit only; imports no model and executes no inference."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, help='Portable training study root; verifies identical checkpoint hashes without using retained absolute paths')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('New audit output required')
    torch.set_num_threads(1)
    started = time.perf_counter()
    run = args.run.resolve()
    report = json.loads((run/'diagnostic.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    terminal = json.loads((run/'terminal.json').read_text())
    manifest = json.loads((args.validation/'manifest.json').read_text())
    summary = json.loads(args.summary.read_text())
    assert report['status'] == terminal['status'] == 'complete' and terminal['exit_code'] == 0
    assert report['device'] == plan['device'] == 'cpu'
    assert report['reserved_test_opened'] is plan['reserved_test_opened'] is False
    assert report['plan'] == plan
    assert not list(run.glob('*watchdog-stop*'))
    assert terminal['max_seconds'] == 600 and terminal['max_rss_gib'] == 18 and terminal['minimum_available_gib'] == 2
    assert plan['scene_ids'] == list(range(300000,300008)) == manifest['scene_seeds']
    assert manifest['reserved_test_generated'] is False and manifest['split'] == 'validation'
    assert sha(args.validation/'manifest.json') == plan['data_manifest_sha256']
    assert summary['source_sha256'] == sha(run/'diagnostic.json') and summary['post_hoc'] is True
    for name,digest in plan['sources'].items():
        assert sha(run/'source'/(name+'.txt')) == digest
    for job in plan['jobs']:
        checkpoint = args.checkpoint_root/str(job['seed'])/job['mode']/'training/memory-final.pt' if args.checkpoint_root else Path(job['checkpoint'])
        assert sha(checkpoint) == job['checkpoint_sha256']
    expected = set(itertools.product((20260907,20260908,20260909),('carry','reset'),range(300000,300008)))
    assert {(row['seed'],row['mode'],row['scene_seed']) for row in report['cases']} == expected
    assert len(report['cases']) == 48
    arrays, input_files = {}, {}
    for row in manifest['records']:
        path = args.validation/row['file']
        assert sha(path) == row['file_sha256']
        with np.load(path,allow_pickle=False) as archive:
            assert set(archive.files) == {'observations','actions'}
            rgb,actions = archive['observations'],archive['actions']
        assert rgb.shape == (2,66,3,64,64) and rgb.dtype == np.uint8
        assert actions.shape == (2,65) and actions.dtype == np.int64
        assert raw_sha(rgb) == row['observations_sha256'] and raw_sha(actions) == row['actions_sha256']
        assert np.array_equal(rgb[0,38:42],rgb[1,38:42])
        assert np.array_equal(actions[0,37:],actions[1,37:])
        normalized = (torch.from_numpy(rgb.copy()).float()/127.5-1).numpy()
        arrays[row['scene_seed']] = normalized,actions
        input_files[row['file']] = sha(path)
    comparisons, max_difference = 0, 0.
    def equal(actual, expected):
        nonlocal comparisons,max_difference
        if isinstance(expected,dict):
            for key,value in expected.items():
                equal(actual[key],value)
        elif isinstance(expected,list):
            assert len(actual) == len(expected)
            for a,b in zip(actual,expected):
                equal(a,b)
        elif isinstance(expected,(float,np.floating)):
            difference = abs(float(actual)-float(expected))
            max_difference = max(max_difference,difference)
            assert np.isfinite(actual) and difference <= 5e-15, (actual,expected,difference)
            comparisons += 1
        else:
            assert actual == expected, (actual,expected)
            comparisons += 1
    cases = []
    tensors_checked = 0
    for row in report['cases']:
        path = (run/row['file']).resolve()
        assert path.is_relative_to(run) and sha(path) == row['file_sha256']
        loaded = torch.load(path,weights_only=True,map_location='cpu')
        assert set(loaded) == set(row['tensors'])
        values = {}
        for key,tensor in loaded.items():
            assert tensor.device.type == 'cpu' and tensor.is_contiguous() and torch.isfinite(tensor).all()
            actual = {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':raw_sha(tensor.numpy())}
            assert actual == row['tensors'][key]
            values[key] = tensor.numpy()
            tensors_checked += 1
        rgb,actions = arrays[row['scene_seed']]
        assert np.array_equal(values['boundary_history'],rgb[:,38:42])
        assert np.array_equal(values['return_actions'],actions[:,41:])
        state = values['state_normal'].astype(np.float64)
        assert state.shape == (2,32) and np.max(np.abs(state)) <= 1
        assert np.array_equal(values['state_swapped'],values['state_normal'][::-1])
        assert not np.count_nonzero(values['state_zero'])
        normal = values['prediction_normal']
        for name in ('normal','swapped','zero'):
            prediction = values['prediction_'+name]
            assert prediction.shape == (2,24,3,64,64) and prediction.dtype == np.float32
            assert np.max(np.abs(prediction)) <= 1
        # Identical histories/actions make branch-swapping the complete prediction a required symmetry.
        assert np.array_equal(values['prediction_swapped'],normal[::-1])
        assert np.array_equal(values['prediction_zero'][0],values['prediction_zero'][1])
        truth = rgb[:,42:].astype(np.float64)
        mask = np.any(truth[0] != truth[1],axis=1)
        region = np.broadcast_to(mask[:,None],truth[0].shape)
        pair_region = np.broadcast_to(region,truth.shape)
        scores = {}
        for name in ('normal','swapped','zero'):
            pred = values['prediction_'+name].astype(np.float64)
            branches = []
            for branch in range(2):
                own = float(np.abs(pred[branch]-truth[branch])[region].mean()/2)
                other = float(np.abs(pred[branch]-truth[1-branch])[region].mean()/2)
                branches.append({'branch':branch,'own_truth_mae':own,'opposite_truth_mae':other,'own_strictly_closer':own<other})
            scores[name] = {'return_frames':24,'different_pixel_count':int(mask.sum()),
                'return_full_frame_mae':float(np.abs(pred-truth).mean()/2),'eligible_pair':True,
                'pair_correct':all(branch['own_strictly_closer'] for branch in branches),
                'return_region_mae':sum(branch['own_truth_mae'] for branch in branches)/2,'branches':branches}
        changes = {}
        for name in ('swapped','zero'):
            delta = np.abs(values['prediction_'+name].astype(np.float64)-normal.astype(np.float64))/2
            changes[name] = {'full_return_prediction_change_mae':float(delta.mean()),
                'paired_region_prediction_change_mae':float(delta[pair_region].mean()),
                'maximum_rgb_change':float(delta.max()),'changed_channel_values':int(np.count_nonzero(delta)),
                'bit_exact_to_normal':bool(np.array_equal(values['prediction_'+name],normal))}
        difference = state[0]-state[1]
        computed = {'seed':row['seed'],'mode':row['mode'],'scene_seed':row['scene_seed'],
            'state_branch_l2_difference':float(np.linalg.norm(difference)),
            'state_branch_max_absolute_difference':float(np.abs(difference).max()),
            'state_branch_l2_norms':np.linalg.norm(state,axis=1).tolist(),
            'zero_intervention_changes_input_state':bool(np.count_nonzero(state)),
            'intervention_scores':scores,'prediction_changes':changes,
            'all_interventions_bit_exact':all(item['bit_exact_to_normal'] for item in changes.values()),
            'observed_history_equal':True,'recent_actions_equal':True,'return_actions_equal':True}
        equal(row,computed)
        if row['mode'] == 'reset':
            assert computed['all_interventions_bit_exact'] and row['reset_invariance_passed'] is True
        cases.append(computed)
    groups = {}
    for mode in ('carry','reset'):
        rows = [row for row in cases if row['mode']==mode]
        state_diffs = [row['state_branch_l2_difference'] for row in rows]
        group = {'cases':len(rows),'state_branch_l2_difference':{'mean':float(np.mean(state_diffs)),
            'minimum':min(state_diffs),'maximum':max(state_diffs)},'interventions':{}}
        for name in ('normal','swapped','zero'):
            item = {'pair_correct_count':sum(row['intervention_scores'][name]['pair_correct'] for row in rows),
                'mean_return_region_mae':float(np.mean([row['intervention_scores'][name]['return_region_mae'] for row in rows]))}
            if name != 'normal':
                changes = [row['prediction_changes'][name] for row in rows]
                item.update(mean_region_prediction_change=float(np.mean([v['paired_region_prediction_change_mae'] for v in changes])),
                    max_region_prediction_change=max(v['paired_region_prediction_change_mae'] for v in changes),
                    bit_exact_cases=sum(v['bit_exact_to_normal'] for v in changes))
            group['interventions'][name] = item
        groups[mode] = group
    equal(summary['groups'],groups)
    artifact_hashes = {str(path.relative_to(run)):sha(path) for path in sorted(run.rglob('*')) if path.is_file()}
    result = {'schema':'worldline-room-state-sensitivity-independent-audit-v1','status':'passed',
        'audit_script_sha256':sha(__file__),'seconds':time.perf_counter()-started,
        'input_report_sha256':sha(run/'diagnostic.json'),'input_plan_sha256':sha(run/'plan.json'),
        'input_summary_sha256':sha(args.summary),'validation_manifest_sha256':sha(args.validation/'manifest.json'),
        'validation_files_sha256':input_files,'measured_artifact_sha256':artifact_hashes,
        'measured_source_sha256':plan['sources'],'cases':len(cases),'saved_tensors_checked':tensors_checked,
        'prediction_tensors_checked':144,'independent_scalar_checks':comparisons,'maximum_scalar_difference':max_difference,
        'groups':groups,'per_case_recomputed':cases,'worker_seconds':report['elapsed_seconds'],
        'parent_seconds':terminal['elapsed_seconds'],'parent_sampled_peak_rss_bytes':max(v['rss_bytes'] for v in terminal['samples']),
        'parent_sampled_minimum_available_bytes':min(v['available_bytes'] for v in terminal['samples']),
        'gpu_access':False,'model_imported':False,'model_inference_rerun':False,'checkpoint_tensor_values_loaded':False,
        'reserved_test_opened':False,'reset_invariance_cases':24,'swapped_prediction_branch_symmetry_cases':48,
        'limits':'Post hoc sensitivity on existing development data. Exact source/checkpoint/file bindings and retained outputs are audited; warm-state values are not independently regenerated. No semantic probe, scientific novelty, useful memory, new acceptance gate, or held-out test claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','cases','saved_tensors_checked','independent_scalar_checks','maximum_scalar_difference','seconds')}))


if __name__ == '__main__':
    main()
