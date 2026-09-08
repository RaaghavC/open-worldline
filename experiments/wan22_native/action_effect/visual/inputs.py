"""Bounded saved evidence and conditions for the final128 matched visual pair."""
import json
from pathlib import Path
import shutil
import sys
import math

sys.path.insert(0, str(Path.cwd()))
import torch
from safetensors.torch import save_file
import sampler
from experiments.wan22_native.action_cuda import data, cache_run, probe, probe_evidence as pe
from experiments.wan22_native.spatial_reference import evidence as se
from experiments.wan22_native.spatial_reference.config import SPECS, SETTINGS
from experiments.wan22_native.spatial_reference.guards import atomic, limits
from experiments.wan22_native.action_training import objective

HERE = Path(__file__).resolve().parent
REPO = pe.REPO
sha = pe.sha
read = pe.read_json
file = pe.relative_file
SCHEMA = 'worldline-action-effect128-visual-pair-v1'
SCOPE = 'effect128-matched-wait-interact-visual-pair'
SAMPLER_SHA = '374e2d1b68015c7ce0b436dcddb68c6f5d8b247584feaa9140b94ec1aae3c314'
TRAINING_SOURCES = read(HERE/'training-source.json')


def sources():
    repo = {**pe.source_hashes(independent=True), **se.sources()}
    local = {p:sha(HERE/p) for p in ('inputs.py','run.py','sampler.py','test_run.py','training-source.json','training-contract.json')}
    if local['sampler.py'] != SAMPLER_SHA: raise ValueError('Use the exact corrected reviewed sampler')
    return {'repository':repo, 'local':local}


def root(path, fresh=False):
    p = Path(path).absolute()
    if any(q.is_symlink() for q in (p,*p.parents)) or p.resolve().is_relative_to(REPO):
        raise ValueError('Use a regular artifact path outside the repository')
    if (p.exists() if fresh else not p.is_dir()): raise ValueError('Output must be fresh, inputs must exist')
    return p


def checked_outputs(base, rows):
    if not isinstance(rows,dict) or not rows: raise ValueError('Nonempty output inventory required')
    for name,digest in rows.items():
        if sha(file(base,name)) != digest: raise ValueError('Saved output changed: '+name)


def training(directory):
    base=root(directory); plan=read(file(base,'plan.json')); parent=read(file(base,'metrics.json'))
    worker=read(file(base,'worker/metrics.json')); result=read(file(base,'training/metrics.json'))
    terminal=read(file(base,'terminal.json')); pointer=read(file(base,'training/last-valid.json'))
    if (parent.get('schema')!='worldline-action-effect128-run-v1' or parent.get('status')!='passed'
        or parent.get('completed_updates')!=128 or parent.get('model_execution') is not True
        or worker.get('status')!='passed' or worker.get('completed_updates')!=128 or worker.get('base_unchanged') is not True
        or result.get('status')!='passed' or result.get('completed_updates')!=128 or result.get('base_unchanged') is not True
        or result.get('training_forwards')!=256 or result.get('bridge_predictions')!=258
        or result.get('auxiliary_feature_extracts')!=64 or result.get('auxiliary_head_predictions')!=128
        or result.get('total_training_feature_forwards')!=320 or result.get('total_training_head_predictions')!=384
        or result.get('native_reference_predictions')!=2 or result.get('zero_adapter_gate_passed') is not True
        or terminal.get('status')!='complete' or type(terminal.get('exit_code')) is not int or terminal['exit_code']!=0
        or terminal.get('cleanup_error') is not None or list(base.rglob('watchdog-stop.json'))
        or parent.get('worker_metrics_sha256')!=sha(base/'worker/metrics.json')
        or parent.get('terminal_sha256')!=sha(base/'terminal.json')
        or parent.get('plan_sha256')!=sha(base/'plan.json') or worker.get('plan_sha256')!=sha(base/'plan.json')
        or worker.get('training_metrics_sha256')!=sha(base/'training/metrics.json')
        or plan.get('profile')!='spatial' or plan.get('warm_start') is not False):
        raise ValueError('Completed fresh effect128 spatial training required')
    validate_effect_contract(plan)
    expected=TRAINING_SOURCES
    if plan.get('protocol',{}).get('paired_updates')!=128 or plan['protocol'].get('starts')!=[0,8,32,49]*32 or plan['protocol'].get('completed_checkpoints')!=list(range(0,129,16)) or plan.get('resume_supported') is not False:raise ValueError('Only the fixed128 duration and final-checkpoint policy are accepted')
    if plan.get('source_sha256')!=expected or worker.get('source_sha256')!=expected:
        raise ValueError('Training source contract differs')
    for group,rows in expected.items():
        for name,digest in rows.items():
            if sha(file(base/'source'/group,name))!=digest: raise ValueError('Training source snapshot changed')
    checked_outputs(base,worker['output_sha256'])
    names=set(worker['output_sha256'])
    wanted={f'training/prediction-{i:04d}-{arm}.safetensors' for i in range(1,129) for arm in ('closed','open')}|{f'training/gradients-after-clip-{i:04d}.safetensors' for i in range(1,129)}
    if not wanted<=names or {p.name for p in (base/'training').glob('checkpoint-*')}!={f'checkpoint-{i:04d}' for i in range(0,129,16)}:raise ValueError('Full128 prediction/gradient and declared checkpoint coverage required')
    auxiliary={f'training/auxiliary-{i:04d}-{arm}-{context}.safetensors' for i in range(1,129,4) for arm in ('closed','open') for context in ('positive','negative')}
    if not auxiliary<=names:raise ValueError('All128 auxiliary prediction files required')
    before=read(file(base,'training/core-before.json')); after=read(file(base,'training/core-after.json'))
    if before!=after or before!=probe._original_weights(read(file(base,'worker/weight-load.json'))):
        raise ValueError('Original 825 foundation values differ')
    if pointer.get('directory')!='checkpoint-0128' or pointer.get('completed_updates')!=128:
        raise ValueError('The last valid checkpoint must be exactly 128')
    folder=base/'training/checkpoint-0128'; manifest=read(file(folder,'manifest.json'))
    if (pointer.get('manifest_sha256')!=sha(folder/'manifest.json') or manifest.get('completed_updates')!=128
        or manifest.get('identity',{}).get('plan_sha256')!=sha(base/'plan.json')):
        raise ValueError('Final checkpoint identity differs')
    checked_outputs(folder,manifest['files'])
    adapter, record=sampler.load_adapter_checkpoint(folder/'adapter.safetensors',manifest['files']['adapter.safetensors'])
    if record['tensor_sha256']!={k:v['sha256'] for k,v in manifest['tensors'].items()}:
        raise ValueError('Final adapter tensor records differ')
    del adapter
    return plan, folder/'adapter.safetensors', {'parent_sha256':sha(base/'metrics.json'),
        'terminal_sha256':sha(base/'terminal.json'),'worker_sha256':sha(base/'worker/metrics.json'),'training_sha256':sha(base/'training/metrics.json'),
        'plan_sha256':sha(base/'plan.json'),'checkpoint_manifest_sha256':sha(folder/'manifest.json'),
        'checkpoint_sha256':record['checkpoint_sha256'],'adapter':record,'hardware':worker['hardware'],
        'core_records':before,'training_input_identity':plan['input_identity']}


def validate_effect_contract(plan):
    """This family changes the objective; it must keep every declared original draw."""
    contract=read(HERE/'training-contract.json')
    for key in ('schema','scope','protocol','schedule','limits','original128_input_plan_sha256','heldout_assessment','checkpoint_selection','warm_start','resume_supported','negative_context_adapter_training'):
        if plan.get(key)!=contract[key]:raise ValueError('Exact effect128 training contract differs: '+key)
    if plan.get('source_sha256')!=TRAINING_SOURCES:raise ValueError('Exact effect128 source differs')


def conditions(directory, trained):
    """Hash original evidence, materializing observation/commands and small prefix proofs only."""
    base=root(directory); wanted=trained['cache']; parent=read(file(base,'metrics.json'))
    for relative,key in [('metrics.json','cache_parent_sha256'),('worker/metrics.json','cache_worker_sha256'),
                         ('terminal.json','cache_terminal_sha256'),('result/completion.json','cache_completion_sha256'),
                         ('result/manifest.json','cache_manifest_sha256')]:
        if sha(file(base,relative))!=wanted[key]: raise ValueError('Completed training cache identity changed')
    if (parent.get('status')!='passed' or parent.get('model_execution') is not True
        or parent.get('source_sha256')!=cache_run.sources() or parent.get('hardware')!=wanted['hardware']
        or list(base.rglob('watchdog-stop.json'))): raise ValueError('Current completed native CUDA cache required')
    records={}; rows={}
    for arm in ('closed','open'):
        key=arm+'-0000'; values,provenance=data.read_window(base/'result',key,conditioning_only=True)
        if set(values)!={'observation','commands'}: raise ValueError('Conditioning-only reader returned a target')
        if any(sampler.tensor_sha(value)!=wanted['windows'][key]['tensor_sha256'][name] for name,value in values.items()):
            raise ValueError('Actual trained conditions changed')
        records[arm]=values; rows[arm]=provenance
    validate_commands(records)
    return records,rows


def validate_commands(records):
    if set(records)!={'closed','open'} or not torch.equal(records['closed']['observation'],records['open']['observation']):
        raise ValueError('Both branches must have the exact same independent initial observation')
    for arm in records:
        expected=torch.zeros(1,16,6); expected[:,1:,3]=math.pi/24; expected[:,0,5]=int(arm=='open')
        if not torch.equal(records[arm]['commands'],expected): raise ValueError('Use unchanged original wait/interact then 15 left-turn commands')


def baseline(directory, observation):
    base=root(directory); report=read(file(base,'metrics.json')); packet=report['input_manifest_sha256']
    se.validate_completed(base,'clip','spatial',se.sources(),packet,'NVIDIA A100-SXM4-80GB')
    core=read(file(base,'core/result/metrics.json')); decode=read(file(base,'decode/result/metrics.json'))
    path=base/'core/result/sampling-inputs.safetensors'; shape=SPECS['spatial'].latent_shape
    values=pe.tensors(path,{'initial_noise':(shape,'F32'),'initial_latent':(shape,'F32'),
        'observation':(SPECS['spatial'].observation_shape,'F32'),'token_times':((1,4290),'I64')},core['output_sha256']['sampling-inputs.safetensors'])
    _,contexts,_=se.validate_input_packet(base/'inputs',packet)
    if not torch.equal(values['observation'],observation):
        raise ValueError('Training observation differs from the retained native baseline; stop rather than substitute it')
    timing={'core_load_seconds':core['load_seconds'],'one_native_sampling_seconds':core['sampling_seconds'],
            'complete_decode_stage_seconds':decode['elapsed_seconds'],'full_baseline_seconds':report['elapsed_seconds']}
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in timing.values()): raise ValueError('Finite measured baseline timing required')
    return values,contexts,{'parent_sha256':sha(base/'metrics.json'),'core_sha256':sha(base/'core/result/metrics.json'),
        'decode_sha256':sha(base/'decode/result/metrics.json'),'hardware':core['hardware'],'timing':timing,
        'input_tensor_sha256':{k:sampler.tensor_sha(v) for k,v in values.items()},
        'context_tensor_sha256':{k:sampler.tensor_sha(v) for k,v in contexts.items()}}


def collect(training_run,cache_run,baseline_run):
    plan,checkpoint,trained=training(training_run)
    records,condition_records=conditions(cache_run,plan['input_identity'])
    values,contexts,native=baseline(baseline_run,records['closed']['observation'])
    import run
    run.require_hardware(trained['hardware'],native['hardware'])
    for arm in records: sampler.validate_inputs(values,contexts,records[arm]['commands'],'spatial')
    return checkpoint,values,contexts,{arm:r['commands'] for arm,r in records.items()}, {
        'training':trained,'conditions':condition_records,'baseline':native}


def preflight(path):
    report=read(path)
    if (report.get('schema')!='worldline-action-visual-cpu-v1' or report.get('status')!='passed'
        or report.get('source_sha256')!=sources() or report.get('failures')!=0 or report.get('errors')!=0
        or report.get('skipped')!=0 or type(report.get('tests')) is not int or report['tests']<6):
        raise ValueError('Current source-bound CPU checks required')
    return sha(path)


def prepare(training_run,cache_run,baseline_run,cpu_report,training_audit,output):
    out=root(output,fresh=True)
    for p in (training_run,cache_run,baseline_run):
        if out.is_relative_to(Path(p).resolve()): raise ValueError('Output may not be inside an input')
    out.mkdir(parents=True); report={'schema':SCHEMA,'status':'preparing','model_execution':False}
    try:
        review=preflight(cpu_report); mapping=sources()
        checkpoint,values,contexts,commands,identity=collect(training_run,cache_run,baseline_run)
        audit_sha=validate_training_audit(training_audit,identity['training'])
        for name,tensors in [('sampling-inputs.safetensors',values),('contexts.safetensors',contexts),('commands.safetensors',commands)]:
            save_file(tensors,str(out/name))
        shutil.copyfile(checkpoint,out/'adapter.safetensors'); shutil.copyfile(cpu_report,out/'cpu-report.json');shutil.copyfile(training_audit,out/'training-audit.json')
        artifacts={name:sha(out/name) for name in ('sampling-inputs.safetensors','contexts.safetensors','commands.safetensors','adapter.safetensors','cpu-report.json')}
        for group,rows in mapping.items():
            for name,digest in rows.items():
                src=(REPO if group=='repository' else HERE)/name; dst=out/'source'/group/name
                dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
                if sha(dst)!=digest: raise ValueError('Source changed during copy')
        evidence_files={}
        selections={'training':(training_run, ['metrics.json','plan.json','terminal.json','worker/metrics.json','worker/weight-load.json',
            'training/metrics.json','training/core-before.json','training/core-after.json','training/last-valid.json','training/checkpoint-0128/manifest.json']),
            'cache':(cache_run,['metrics.json','terminal.json','worker/metrics.json','result/completion.json','result/manifest.json']),
            'baseline':(baseline_run,['metrics.json','core/result/metrics.json','core/result/weight-load.json','core/terminal.json','decode/result/metrics.json','decode/terminal.json'])}
        for group,(base,names) in selections.items():
            for name in names:
                src=file(base,name); dst=out/'evidence'/group/name; dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dst); evidence_files[str(dst.relative_to(out))]=sha(src)
                if sha(dst)!=sha(src): raise ValueError('Evidence changed during copy')
        plan={'schema':SCHEMA,'status':'prepared','scope':SCOPE,'profile':'spatial','arms':['closed','open'],
              'settings':SETTINGS,'limits':limits('clip'),'expected_gpu':'NVIDIA A100-SXM4-80GB',
              'source_sha256':mapping,'cpu_report_sha256':review,'input_identity':identity,'artifacts':artifacts,
              'evidence_sha256':evidence_files, 'complete_prior_runs_verified_during_preparation':True,
              'training':False,'quality_assessed':False,'fresh_noise_drawn':False,
              'training_audit_sha256':audit_sha,'final_checkpoint_updates':128,'checkpoint_selection':False,
              'same_initial_observation_noise_text':True,'negative_context_adapter_training':True,
              'future_targets_materialized':False,'frame_count_per_arm':17,'new_frames_per_arm':16}
        atomic(out/'plan.json',plan); read_prepared(out)
        report.update(status='prepared',plan_sha256=sha(out/'plan.json')); return plan
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error)); raise
    finally: atomic(out/'metrics.json',report)


def read_prepared(directory):
    out=root(directory); plan=read(file(out,'plan.json'))
    if (plan.get('schema')!=SCHEMA or plan.get('status')!='prepared' or plan.get('scope')!=SCOPE
        or plan.get('profile')!='spatial' or plan.get('arms')!=['closed','open'] or plan.get('settings')!=SETTINGS
        or plan.get('limits')!=limits('clip') or plan.get('source_sha256')!=sources()
        or plan.get('negative_context_adapter_training') is not True or plan.get('training') is not False or plan.get('final_checkpoint_updates')!=128 or plan.get('checkpoint_selection') is not False): raise ValueError('Exact current matched visual plan required')
    for group,rows in plan['source_sha256'].items():
        for name,digest in rows.items():
            if sha(file(out/'source'/group,name))!=digest: raise ValueError('Prepared source snapshot changed')
    if preflight(file(out,'cpu-report.json'))!=plan['cpu_report_sha256']: raise ValueError('CPU report changed')
    checked_outputs(out,plan['evidence_sha256'])
    links={'evidence/training/metrics.json':plan['input_identity']['training']['parent_sha256'],
           'evidence/training/worker/metrics.json':plan['input_identity']['training']['worker_sha256'],
           'evidence/training/terminal.json':plan['input_identity']['training']['terminal_sha256'],
           'evidence/training/training/metrics.json':plan['input_identity']['training']['training_sha256'],
           'evidence/training/plan.json':plan['input_identity']['training']['plan_sha256'],
           'evidence/training/training/checkpoint-0128/manifest.json':plan['input_identity']['training']['checkpoint_manifest_sha256'],
           'evidence/baseline/metrics.json':plan['input_identity']['baseline']['parent_sha256'],
           'evidence/baseline/core/result/metrics.json':plan['input_identity']['baseline']['core_sha256'],
           'evidence/baseline/decode/result/metrics.json':plan['input_identity']['baseline']['decode_sha256']}
    if any(plan['evidence_sha256'].get(k)!=v for k,v in links.items()): raise ValueError('Prior evidence identity differs')
    expected={'sampling-inputs.safetensors','contexts.safetensors','commands.safetensors','adapter.safetensors','cpu-report.json'}
    if set(plan['artifacts'])!=expected: raise ValueError('All prepared artifacts required')
    checked_outputs(out,plan['artifacts'])
    trained=plan['input_identity']['training']
    if validate_training_audit(out/'training-audit.json',trained)!=plan['training_audit_sha256']:raise ValueError('Final128 audit identity differs')
    if plan['artifacts']['adapter.safetensors']!=trained['checkpoint_sha256']: raise ValueError('Prepared checkpoint changed')
    adapter,record=sampler.load_adapter_checkpoint(out/'adapter.safetensors',trained['checkpoint_sha256']); del adapter
    if record!=trained['adapter']: raise ValueError('Final checkpoint parameter identity differs')
    shape=SPECS['spatial'].latent_shape
    values=pe.tensors(out/'sampling-inputs.safetensors',{'initial_noise':(shape,'F32'),'initial_latent':(shape,'F32'),
        'observation':(SPECS['spatial'].observation_shape,'F32'),'token_times':((1,4290),'I64')},plan['artifacts']['sampling-inputs.safetensors'])
    contexts=pe.tensors(out/'contexts.safetensors',{'atrium':((25,4096),'F32'),'native_negative':((126,4096),'F32')},plan['artifacts']['contexts.safetensors'])
    commands=pe.tensors(out/'commands.safetensors',{arm:((1,16,6),'F32') for arm in ('closed','open')},plan['artifacts']['commands.safetensors'])
    baseline=plan['input_identity']['baseline']
    if ({k:sampler.tensor_sha(v) for k,v in values.items()}!=baseline['input_tensor_sha256']
        or {k:sampler.tensor_sha(v) for k,v in contexts.items()}!=baseline['context_tensor_sha256']):
        raise ValueError('Exact native baseline inputs required')
    validate_commands({arm:{'commands':command,'observation':values['observation']} for arm,command in commands.items()})
    for arm,command in commands.items():
        window=trained['training_input_identity']['cache']['windows'][arm+'-0000']['tensor_sha256']
        if sampler.tensor_sha(command)!=window['commands'] or sampler.tensor_sha(values['observation'])!=window['observation']:
            raise ValueError('Prepared commands or observation differ from training')
        sampler.validate_inputs(values,contexts,command,'spatial')
    return plan,values,contexts,commands


def validate_training_audit(path,trained):
    """Future complete independent audit, bound to the actual final128 result."""
    value=read(path)
    identity={key:trained[key] for key in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
    identity.update(final_checkpoint_manifest_sha256=trained['checkpoint_manifest_sha256'],
                    final_checkpoint_sha256=trained['checkpoint_sha256'],source_sha256=TRAINING_SOURCES)
    if value.get('schema')!='worldline-action-effect128-actual-independent-v1' or value.get('status')!='passed' or value.get('completed_updates')!=128 or value.get('foundation_values_unchanged') is not True or value.get('final_checkpoint_only') is not True or value.get('auxiliary_updates')!=32 or value.get('auxiliary_head_predictions')!=128 or any(value.get('identity',{}).get(k)!=v for k,v in identity.items()):
        raise ValueError('Passed audit of this exact final128 training result required')
    return sha(path)
