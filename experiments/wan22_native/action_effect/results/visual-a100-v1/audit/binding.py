"""Bind later completed evidence without changing the frozen numerical auditor."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
SCHEMA='worldline-action-effect-visual-audit-binding-v1'
REVIEW_SCHEMA='worldline-action-effect-visual-actual-plan-review-v1'

def need(value,label):
    if not value:raise ValueError(label)

def digest(value):
    need(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'Exact lowercase SHA256 required')
    return value

def regular(path):
    p=Path(path).absolute()
    need(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'Regular file without symlink ancestors required')
    return p

def sha(path):
    with regular(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    p=regular(path);need(p.stat().st_size<=16*2**20,'Bounded JSON required')
    def pairs(rows):
        value={}
        for key,item in rows:
            need(key not in value,'Duplicate JSON key');value[key]=item
        return value
    def nonfinite(_):raise ValueError('Finite JSON required')
    value=json.loads(p.read_text(),object_pairs_hook=pairs,parse_constant=nonfinite)
    need(isinstance(value,dict),'JSON object required');return value

def member(root,name):
    p=Path(name)
    need(isinstance(name,str) and name and not p.is_absolute() and '..' not in p.parts,'Relative artifact path required')
    return regular(Path(root)/p)

def inspect(prepared,source_review,plan_review):
    """Read actual files only. No provisional checkpoint or unknown hash is accepted."""
    prepared=Path(prepared).absolute();pins=read(HERE/'pins.json')
    plan=read(member(prepared,'plan.json'));mapping=pins['visual_source_sha256']
    need(plan.get('schema')=='worldline-action-effect128-visual-pair-v1'
         and plan.get('scope')=='effect128-matched-wait-interact-visual-pair'
         and plan.get('status')=='prepared' and plan.get('source_sha256')==mapping
         and plan.get('negative_context_adapter_training') is True
         and plan.get('final_checkpoint_updates')==128 and plan.get('checkpoint_selection') is False,
         'Actual prepared effect128 final-only visual plan required')
    need(plan.get('training') is False and plan.get('profile')=='spatial'
         and plan.get('arms')==['closed','open'] and plan.get('settings')=={'steps':50,'shift':5.,'guidance':5.},'Unchanged matched visual settings')
    for group,rows in mapping.items():
        for name,value in rows.items():need(sha(member(prepared,'source/'+group+'/'+name))==digest(value),'Exact prepared source '+name)
    for key in ('artifacts','evidence_sha256'):
        need(isinstance(plan.get(key),dict) and plan[key],'Complete prepared evidence required')
        for name,value in plan[key].items():need(sha(member(prepared,name))==digest(value),'Exact prepared evidence '+name)
    need(sha(source_review)==pins['source_review_sha256'],'Frozen independent program review required')
    review=read(source_review)
    need(review.get('status')=='passed' and review.get('source_sha256')==mapping,'Program review matches sources')
    cpu=read(member(prepared,'cpu-report.json'))
    need(sha(prepared/'cpu-report.json')==pins['visual_cpu_sha256']==plan['cpu_report_sha256']
         and cpu.get('status')=='passed' and cpu.get('tests')==11 and cpu.get('source_unchanged') is True
         and cpu.get('source_sha256')==mapping and all(cpu.get(k)==0 for k in ('failures','errors','skipped')),
         'Exact passed visual CPU report required')
    trained=plan['input_identity']['training'];completed=read(member(prepared,'training-audit.json'))
    cp=digest(trained['checkpoint_sha256'])
    need(sha(member(prepared,'adapter.safetensors'))==cp==plan['artifacts']['adapter.safetensors'],'Actual final adapter bytes required')
    identity={k:trained[k] for k in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
    identity.update(final_checkpoint_manifest_sha256=trained['checkpoint_manifest_sha256'],final_checkpoint_sha256=cp,source_sha256=pins['training_source_sha256'])
    need(completed.get('schema')=='worldline-action-effect128-actual-independent-v1'
         and completed.get('status')=='passed' and completed.get('completed_updates')==128
         and completed.get('foundation_values_unchanged') is True and completed.get('final_checkpoint_only') is True
         and completed.get('auxiliary_updates')==32 and completed.get('auxiliary_head_predictions')==128
         and completed.get('identity')==identity,'Exact completed effect128 training audit required')
    training_plan=read(member(prepared,'evidence/training/plan.json'))
    need(sha(prepared/'evidence/training/plan.json')==pins['training_plan_sha256']==trained['plan_sha256'],'Frozen actual training plan required')
    need(training_plan.get('source_sha256')==pins['training_source_sha256']
         and all(training_plan.get(k)==v for k,v in pins['training_contract'].items()),'Unchanged new-family training contract')
    audit_sha=sha(prepared/'training-audit.json');need(audit_sha==plan['training_audit_sha256'],'Prepared actual audit hash')
    plan_sha=sha(prepared/'plan.json');actual_review=read(plan_review)
    expected={'schema':REVIEW_SCHEMA,'status':'passed','visual_plan_sha256':plan_sha,
              'actual_audit_sha256':audit_sha,'final_checkpoint_sha256':cp,'source_sha256':mapping}
    need(all(actual_review.get(k)==v for k,v in expected.items()),'Passed independent review of the exact actual visual plan required')
    return {'schema':SCHEMA,'status':'bound','prepared':str(prepared),'source_review':str(Path(source_review).absolute()),
            'plan_review':str(Path(plan_review).absolute()),'plan_sha256':plan_sha,'source_review_sha256':sha(source_review),
            'plan_review_sha256':sha(plan_review),'training_audit_sha256':audit_sha,'checkpoint_sha256':cp,
            'cpu_sha256':sha(prepared/'cpu-report.json'),'pins_sha256':sha(HERE/'pins.json'),
            'binding_source_sha256':sha(__file__),'model_execution':False,'training_admission':False}

def create(prepared,source_review,plan_review,output):
    out=Path(output).absolute()
    need(not out.exists() and not any(q.is_symlink() for q in (out,*out.parents)),'Fresh binding file required')
    need(not out.resolve().is_relative_to(Path(prepared).resolve()) and not out.resolve().is_relative_to(HERE),'Binding must be separate from prepared and frozen auditor files')
    value=inspect(prepared,source_review,plan_review)
    with out.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return value

def load(path,expected_sha256):
    need(sha(path)==digest(expected_sha256),'Parent-reviewed binding checksum required')
    value=read(path)
    need(value.get('schema')==SCHEMA and value.get('status')=='bound','Completed actual-evidence binding required')
    fresh=inspect(value['prepared'],value['source_review'],value['plan_review'])
    need(value==fresh,'Bound evidence changed after review');return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source-review','plan-review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=create(args.prepared,args.source_review,args.plan_review,args.output)
    print(json.dumps({'status':result['status'],'binding_sha256':sha(args.output)}))
