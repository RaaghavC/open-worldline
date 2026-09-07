# SPDX-License-Identifier: Apache-2.0
"""Post hoc state sensitivity on the completed fixed Room memory study.

Default execution prints a plan without loading model tensors. --execute runs
the CPU-only diagnostic under a supervised time/memory bound. No training,
threshold selection, fitted probe or reserved scene is involved.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .memory_evaluate import generated_rollout, history_at, score_return
from .memory_train import (BASE, BASE_SHA256, DevelopmentDataset, atomic_write,
    handoff_and_guard, make_model, new_directory, restore_memory, sha256, tensor_hashes)

HERE = Path(__file__).parent
SEEDS = (20260907, 20260908, 20260909)
SCENES = tuple(range(300000,300008))
STUDY_SHA256 = '71a6ed7982293da9255e00b8d3e1e6adabb0d8156665cc6dfc6ec559d8594c4f'
SOURCES = ('memory_state_diagnostic.py','memory_train.py','memory_evaluate.py',
           'memory_model.py','memory_data.py','model.py','memory_sequence.py')
INTERVENTIONS = ('normal','swapped','zero')


@torch.inference_mode()
def boundary_state(model, observed_prefix, prefix_actions, *, mode='carry'):
    """Warm only the supplied observed prefix, ending before the return action."""
    if mode not in ('carry','reset'):
        raise ValueError('Expected carry or reset mode')
    if (observed_prefix.ndim != 5 or observed_prefix.shape[0] != 2 or observed_prefix.shape[2] != 3
            or observed_prefix.dtype != torch.float32 or observed_prefix.device.type != 'cpu'
            or prefix_actions.dtype != torch.int64 or prefix_actions.device.type != 'cpu'
            or prefix_actions.shape != (2,observed_prefix.shape[1]-1) or prefix_actions.shape[1] < 4
            or not torch.isfinite(observed_prefix).all() or observed_prefix.abs().max() > 1
            or (prefix_actions < 0).any() or (prefix_actions > 5).any()):
        raise ValueError('Expected a finite paired CPU RGB/action prefix')
    model.eval()
    state = model.initial_state(2)
    if state.device.type != 'cpu':
        raise ValueError('This diagnostic is CPU only')
    for transition in range(prefix_actions.shape[1]):
        _, state = model(history_at(observed_prefix,transition),prefix_actions[:,transition],state,mode=mode)
    history = history_at(observed_prefix,prefix_actions.shape[1]).clone()
    if not torch.equal(history[0],history[1]) or not torch.equal(prefix_actions[0,-4:],prefix_actions[1,-4:]):
        raise ValueError('Recent observed RGB and commands must be identical between branches')
    if state.shape != (2,32) or state.dtype != torch.float32 or not torch.isfinite(state).all():
        raise ValueError('Expected finite paired GRU32 state')
    return history,state.clone()


@torch.inference_mode()
def intervene(model, observed_prefix, prefix_actions, return_actions, *, mode='carry'):
    """Generate all returns before a caller supplies truth for scoring.

    The only intervention is the starting state at the aliased return boundary.
    Subsequent RGB and state evolve through the unchanged model at every step.
    """
    if (return_actions.ndim != 2 or return_actions.shape[0] != 2 or return_actions.shape[1] < 1
            or return_actions.dtype != torch.int64 or return_actions.device.type != 'cpu'
            or not torch.equal(return_actions[0],return_actions[1])):
        raise ValueError('Both branches need the same CPU return-command sequence')
    history,state = boundary_state(model,observed_prefix,prefix_actions,mode=mode)
    states = {'normal':state.clone(),'swapped':state.flip(0).clone(),'zero':torch.zeros_like(state)}
    predictions = {name:generated_rollout(model,history.clone(),return_actions.clone(),value.clone(),mode=mode).cpu()
                   for name,value in states.items()}
    return {'boundary_history':history,'return_actions':return_actions.clone(),
            **{f'state_{name}':value for name,value in states.items()},
            **{f'prediction_{name}':value for name,value in predictions.items()}}


def summarize(values, truth):
    """Use existing paired scoring only after every intervention has generated."""
    normal = values['prediction_normal']
    if truth.shape != normal.shape:
        raise ValueError('Truth must align with the complete generated returns')
    state = values['state_normal'].double()
    difference = state[0]-state[1]
    mask = (truth[0] != truth[1]).any(dim=1).unsqueeze(0).unsqueeze(2).expand_as(normal)
    if not mask.any():
        raise ValueError('Paired return must contain differing truth pixels')
    changes = {}
    for name in ('swapped','zero'):
        delta = (values[f'prediction_{name}'].double()-normal.double()).abs()/2
        changes[name] = {'full_return_prediction_change_mae':float(delta.mean()),
            'paired_region_prediction_change_mae':float(delta[mask].mean()),
            'maximum_rgb_change':float(delta.max()),'changed_channel_values':int(torch.count_nonzero(delta)),
            'bit_exact_to_normal':torch.equal(values[f'prediction_{name}'],normal)}
    return {'state_branch_l2_difference':float(torch.linalg.vector_norm(difference)),
        'state_branch_max_absolute_difference':float(difference.abs().max()),
        'state_branch_l2_norms':torch.linalg.vector_norm(state,dim=1).tolist(),
        'zero_intervention_changes_input_state':bool(torch.count_nonzero(state)),
        'intervention_scores':{name:score_return(values[f'prediction_{name}'],truth) for name in INTERVENTIONS},
        'prediction_changes':changes,
        'all_interventions_bit_exact':all(row['bit_exact_to_normal'] for row in changes.values())}


def build_plan(study,validation,evaluation):
    root = Path(study).resolve()
    if sha256(root/'study.json') != STUDY_SHA256:
        raise ValueError('Only the completed fixed 512-update study is prescribed')
    trained = json.loads((root/'study.json').read_text())
    evaluated = json.loads((Path(evaluation)/'validation.json').read_text())
    if (trained['status'] != 'complete' or trained['phase'] != 'train' or trained['matched_budgets'] is not True
            or trained['requested_updates_per_arm'] != 512 or trained['seeds'] != list(SEEDS)
            or trained['sequence_path'] != 'batched' or trained['reserved_test_opened'] is not False
            or evaluated['status'] != 'complete' or evaluated['training_study_sha256'] != STUDY_SHA256
            or evaluated['reserved_test_opened'] is not False):
        raise ValueError('Complete fixed development evidence is required')
    data = DevelopmentDataset(validation,'validation')
    if tuple(data.scene_ids) != SCENES or data.manifest_sha256 != trained['validation']['manifest_sha256']:
        raise ValueError('The exact eight development scenes are required')
    for name,digest in trained['source_sha256'].items():
        if sha256(HERE/name) != digest or sha256(root/'measured-source'/(name+'.txt')) != digest:
            raise ValueError('A measured training source changed')
    if sha256(BASE) != BASE_SHA256:
        raise ValueError('Frozen base checkpoint changed')
    jobs = []
    for seed in SEEDS:
        for mode in ('carry','reset'):
            rows = [row for row in trained['runs'] if row['seed']==seed and row['mode']==mode]
            if len(rows) != 1:
                raise ValueError('Missing or duplicated model')
            metrics = (root/rows[0]['metrics']).resolve()
            if not metrics.is_relative_to(root) or sha256(metrics) != rows[0]['metrics_sha256']:
                raise ValueError('Training metrics identity differs')
            measured = json.loads(metrics.read_text())
            checkpoint = metrics.parent/'memory-final.pt'
            if measured['status'] != 'complete' or measured['completed_updates'] != 512 or sha256(checkpoint) != measured['memory_final_sha256']:
                raise ValueError('Only the fixed final checkpoint is eligible')
            jobs.append({'seed':seed,'mode':mode,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint)})
    return {'schema':'worldline-room-state-sensitivity-plan-v1','purpose':'Post hoc causal sensitivity, not a new acceptance test or scientific novelty claim',
        'training_study_sha256':STUDY_SHA256,'evaluation_sha256':sha256(Path(evaluation)/'validation.json'),
        'data_manifest_sha256':data.manifest_sha256,'base_checkpoint_sha256':BASE_SHA256,
        'sources':{name:sha256(HERE/name) for name in SOURCES},'jobs':jobs,'scene_ids':list(SCENES),
        'first_return_action':41,'observed_prefix_frames':42,'generated_return_frames':24,
        'interventions':list(INTERVENTIONS),'device':'cpu','reserved_test_opened':False,
        'training':False,'fitted_probes':False,'all_cases':48,'prediction_tensors':144,
        'raw_prediction_bytes':48*3*2*24*3*64*64*4}


def worker(config):
    out = Path(config['output']);plan = config['plan'];torch.set_num_threads(1)
    for name,digest in plan['sources'].items():
        if sha256(HERE/name) != digest:raise ValueError('Source changed after snapshot')
    dataset = DevelopmentDataset(config['validation'],'validation')
    if dataset.manifest_sha256 != plan['data_manifest_sha256']:raise ValueError('Development data changed')
    report = {'status':'running','plan':plan,'cases':[],'device':'cpu','reserved_test_opened':False,
        'interpretation':'State perturbation measures sensitivity of these trained networks. It does not establish semantic storage, useful memory or a new acceptance result.'}
    started = time.monotonic()
    try:
        atomic_write(out/'diagnostic.json',report)
        for job in plan['jobs']:
            if sha256(job['checkpoint']) != job['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
            payload = torch.load(job['checkpoint'],weights_only=True,map_location='cpu')
            if (payload['completed_updates'] != 512 or payload['seed'] != job['seed'] or payload['mode'] != job['mode']
                    or payload['sequence_path'] != 'batched' or payload['base_checkpoint_sha256'] != BASE_SHA256):
                raise ValueError('Final checkpoint metadata differs')
            model = make_model(BASE,job['seed'],'cpu');restore_memory(model,payload['memory_state_dict']);model.eval()
            weights_before = tensor_hashes(model.state_dict())
            for scene in SCENES:
                observations,actions = dataset.load_pair(scene)
                # Production diagnostic uses the complete fixed standard 65-step episode.
                if observations.shape != (2,66,3,64,64) or actions.shape != (2,65):raise ValueError('Fixed episode shape differs')
                values = intervene(model,observations[:,:42],actions[:,:41],actions[:,41:],mode=job['mode'])
                scores = summarize(values,observations[:,42:])
                path = out/f'{job["seed"]}-{job["mode"]}-scene-{scene}.pt'
                atomic_write(path,values,tensor=True)
                report['cases'].append(dict(scores,seed=job['seed'],mode=job['mode'],scene_seed=scene,
                    reset_invariance_passed=scores['all_interventions_bit_exact'] if job['mode']=='reset' else None,
                    observed_history_equal=True,recent_actions_equal=True,return_actions_equal=True,
                    file=path.name,file_sha256=sha256(path),tensors=tensor_hashes(values)))
                atomic_write(out/'diagnostic.json',report)
                print(json.dumps({'completed_cases':len(report['cases']),'seed':job['seed'],'mode':job['mode'],'scene':scene}),flush=True)
            if tensor_hashes(model.state_dict()) != weights_before:raise RuntimeError('Diagnostic changed model parameters')
            del model,payload
        report['reset_control_passed'] = all(row['reset_invariance_passed'] for row in report['cases'] if row['mode']=='reset')
        report['status'] = 'complete' if report['reset_control_passed'] else 'failed'
        if len(report['cases']) != 48:raise RuntimeError('Incomplete case coverage')
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;atomic_write(out/'diagnostic.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    for name in ('study','validation','evaluation','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=600)
    args=parser.parse_args()
    if args.worker:
        worker(json.loads(sys.stdin.read(65536)));return
    if any(getattr(args,name) is None for name in ('study','validation','evaluation')):parser.error('Study, validation capture and completed evaluation are required')
    if not 0 < args.max_seconds <= 600:parser.error('Time cap must lie in (0,600] seconds')
    plan=build_plan(args.study,args.validation,args.evaluation)
    if not args.execute:
        print(json.dumps(plan,indent=2));return
    if args.output is None:parser.error('Execution needs a new output directory')
    if args.output.resolve().is_relative_to(HERE.resolve()):raise ValueError('Output must be outside measured source')
    out=new_directory(args.output);atomic_write(out/'plan.json',plan)
    for name,digest in plan['sources'].items():
        target=out/'source'/(name+'.txt');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,target)
        if sha256(target)!=digest:raise RuntimeError('Source changed while snapshotting')
    config={'output':str(out.resolve()),'validation':str(args.validation.resolve()),'plan':plan}
    with (out/'worker.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-m','experiments.room_world.memory_state_diagnostic','--worker'],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'))
        terminal=handoff_and_guard(process,config,out,max_seconds=args.max_seconds,max_rss_gib=18,minimum_available_gib=2)
    if terminal['status']!='complete':raise SystemExit(1)
    report=json.loads((out/'diagnostic.json').read_text())
    if report['status']!='complete':raise SystemExit(1)
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),'reset_control_passed':report['reset_control_passed']}))


if __name__=='__main__':main()
