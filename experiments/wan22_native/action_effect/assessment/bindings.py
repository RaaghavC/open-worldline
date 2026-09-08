"""Exact completed-effect checkpoint eligibility, without foundation loading."""
import json
from pathlib import Path

AUDIT_SCHEMA='worldline-action-effect128-actual-independent-v1'
RUN_SCHEMA='worldline-action-effect128-run-v1'
RESULT_SCHEMA='worldline-action-effect128-prototype-result-v1'
SELECTED=('metrics.json','plan.json','terminal.json','worker/metrics.json','worker/weight-load.json',
    'training/metrics.json','training/core-before.json','training/core-after.json','training/last-valid.json',
    'training/checkpoint-0128/manifest.json')

def require(ok,label):
    if not ok:raise ValueError(label)

def audit_identity(identity,source_map):
    return {k:identity[k] for k in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}|{
        'final_checkpoint_manifest_sha256':identity['checkpoint_manifest_sha256'],
        'final_checkpoint_sha256':identity['checkpoint_sha256'],'source_sha256':source_map}

def validate_audit(report,identity,source_map):
    require(report.get('schema')==AUDIT_SCHEMA and report.get('status')=='passed'
        and report.get('completed_updates')==128 and report.get('foundation_values_unchanged')is True
        and report.get('final_checkpoint_only')is True and report.get('main_predictions')==256
        and report.get('auxiliary_updates')==32 and report.get('auxiliary_feature_extracts')==64
        and report.get('auxiliary_head_predictions')==128 and report.get('identity')==audit_identity(identity,source_map),
        'Completed exact effect128 independent audit required')

def validate_reports(plan,parent,worker,training,terminal,pointer,manifest,expected_plan,source_map):
    require(plan==expected_plan and plan.get('source_sha256')==source_map,'Exact predeclared effect training plan required')
    require(parent.get('schema')==worker.get('schema')==RUN_SCHEMA and training.get('schema')==RESULT_SCHEMA,
        'Effect training runtime schemas differ')
    require(all(r.get('status')=='passed' and r.get('completed_updates')==128 for r in (parent,worker,training))
        and parent.get('model_execution')is True and worker.get('base_unchanged')is True and training.get('base_unchanged')is True
        and terminal.get('status')=='complete' and type(terminal.get('exit_code'))is int and terminal['exit_code']==0
        and terminal.get('cleanup_error')is None,'Complete guarded effect128 run required')
    require(training.get('training_forwards')==256 and training.get('auxiliary_feature_extracts')==64
        and training.get('auxiliary_head_predictions')==128 and training.get('zero_adapter_gate_passed')is True
        and training.get('negative_context_adapter_training')is True and worker.get('negative_context_adapter_training')is True,
        'Exact main/auxiliary counts and negative-context training declaration required')
    require(worker.get('source_sha256')==source_map and training.get('schedule')==expected_plan['schedule'],'Training sources or schedule differ')
    rows=training.get('updates',[]);require(len(rows)==128,'All128 update records required')
    require(all(r.get('update')==i+1 and r.get('auxiliary',{}).get('enabled')is (i%4==0) for i,r in enumerate(rows)),'Exactly32 start0 auxiliary updates required')
    require(pointer.get('directory')=='checkpoint-0128' and pointer.get('completed_updates')==128
        and manifest.get('completed_updates')==128,'Only final128 checkpoint may be evaluated')

def completed(directory,audit_path,ref,here):
    vi=ref.vi;root=vi.root(directory);source_map=vi.read(here/'effect-training-source.json');expected=vi.read(here/'effect-training-plan.json')
    values={n:vi.read(vi.file(root,n)) for n in SELECTED};plan=values['plan.json'];parent=values['metrics.json'];worker=values['worker/metrics.json'];training=values['training/metrics.json']
    manifest=values['training/checkpoint-0128/manifest.json'];pointer=values['training/last-valid.json']
    validate_reports(plan,parent,worker,training,values['terminal.json'],pointer,manifest,expected,source_map)
    require(not list(root.rglob('watchdog-stop.json')) and not list(root.rglob('*cleanup-error.json')),'Stopped or failed cleanup evidence cannot qualify')
    for group,rows in source_map.items():
        for name,digest in rows.items():require(vi.sha(vi.file(root/'source'/group,name))==digest,'Training measured-source snapshot differs')
    vi.checked_outputs(root,worker['output_sha256'])
    before=values['training/core-before.json'];after=values['training/core-after.json']
    require(before==after==ref.probe._original_weights(values['worker/weight-load.json']) and len(before)==825,'All825 original frozen values required')
    ph=vi.sha(root/'plan.json');folder=root/'training/checkpoint-0128'
    require(parent.get('plan_sha256')==worker.get('plan_sha256')==ph
        and parent.get('worker_metrics_sha256')==vi.sha(root/'worker/metrics.json')
        and parent.get('terminal_sha256')==vi.sha(root/'terminal.json')
        and worker.get('training_metrics_sha256')==vi.sha(root/'training/metrics.json')
        and pointer.get('manifest_sha256')==vi.sha(folder/'manifest.json')
        and manifest.get('identity',{}).get('plan_sha256')==ph,'Completed evidence hash links differ')
    vi.checked_outputs(folder,manifest['files'])
    checkpoint=folder/'adapter.safetensors';adapter,record=ref.sampler.load_adapter_checkpoint(checkpoint,manifest['files']['adapter.safetensors']);del adapter
    require(record['tensor_sha256']=={k:v['sha256'] for k,v in manifest['tensors'].items()},'Exact final parameter records required')
    identity={'parent_sha256':vi.sha(root/'metrics.json'),'worker_sha256':vi.sha(root/'worker/metrics.json'),
        'training_sha256':vi.sha(root/'training/metrics.json'),'terminal_sha256':vi.sha(root/'terminal.json'),
        'plan_sha256':ph,'checkpoint_manifest_sha256':vi.sha(folder/'manifest.json'),'checkpoint_sha256':record['checkpoint_sha256'],
        'adapter':record,'hardware':worker['hardware'],'core_records':before,'training_input_identity':plan['input_identity']}
    validate_audit(vi.read(audit_path),identity,source_map)
    return checkpoint,identity
