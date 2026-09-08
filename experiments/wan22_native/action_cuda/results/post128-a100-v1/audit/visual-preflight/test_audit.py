"""Focused CPU checks for the final128-only audit adaptation."""
import ast
import copy
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('final128_visual_audit',HERE/'audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def bound_inputs():
    p=a.read(a.PREP/'plan.json')
    return p,a.read(a.PREP/'training-audit.json'),a.read(a.PREP/'source/local/training-source.json')


def test_actual_audited_final128_preparation():
    p,record,sources=bound_inputs()
    assert a.sha(a.PREP/'plan.json')==a.PLAN_SHA
    assert a.sha(a.PREP/'training-audit.json')==a.TRAINING_AUDIT_SHA
    assert a.sha(a.PREP/'adapter.safetensors')==a.CHECKPOINT_SHA
    assert a.sha(a.PREP/'cpu-report.json')==a.CPU_SHA
    assert a.sha(a.REVIEW)==a.REVIEW_SHA and a.sha(a.PLAN_REVIEW)==a.PLAN_REVIEW_SHA
    a.check_training_binding(p,record,sources)
    for group,rows in p['source_sha256'].items():
        for name,digest in rows.items():
            assert a.sha(a.PREP/'source'/group/name)==digest


def test_final_checkpoint_and_complete_audit_cannot_be_substituted():
    p,record,sources=bound_inputs()
    for key,value in [('completed_updates',16),('status','running'),('foundation_values_unchanged',False)]:
        bad=copy.deepcopy(record);bad[key]=value
        with pytest.raises(AssertionError):a.check_training_binding(p,bad,sources)
    changed=copy.deepcopy(p);changed['artifacts']['adapter.safetensors']='0'*64
    with pytest.raises(AssertionError):a.check_training_binding(changed,record,sources)
    changed=copy.deepcopy(record);changed['identity']['source_sha256']['local']['runner.py']='0'*64
    with pytest.raises(AssertionError):a.check_training_binding(p,changed,sources)
    changed=copy.deepcopy(p);changed['checkpoint_selection']=True
    with pytest.raises(AssertionError):a.check_training_binding(changed,record,sources)


def test_hardware_allows_only_integer_equal_or_greater_capacity():
    p,_,_=bound_inputs();trained=p['input_identity']['training']['hardware']
    def recorded(actual):
        return {'policy':'All fields exact except reported total GPU bytes may be greater',
                'training_total_memory_bytes':trained['total_memory_bytes'],
                'actual_total_memory_bytes':actual['total_memory_bytes'],
                'additional_reported_bytes':actual['total_memory_bytes']-trained['total_memory_bytes']}
    for added in (0,2**21):
        actual=copy.deepcopy(trained);actual['total_memory_bytes']+=added
        a.check_hardware(actual,trained,recorded(actual))
    for capacity in (trained['total_memory_bytes']-1,True,float(trained['total_memory_bytes'])):
        actual=copy.deepcopy(trained);actual['total_memory_bytes']=capacity
        with pytest.raises(AssertionError):a.check_hardware(actual,trained,recorded(actual))
    for key,value in [('name','another GPU'),('bf16_supported',1),('unexpected',False)]:
        actual=copy.deepcopy(trained);actual[key]=value
        with pytest.raises(AssertionError):a.check_hardware(actual,trained,recorded(actual))
    actual=copy.deepcopy(trained);wrong=recorded(actual);wrong['additional_reported_bytes']=1
    with pytest.raises(AssertionError):a.check_hardware(actual,trained,wrong)
    assert a.LIMITS['seconds']==1800 and a.LIMITS['cuda_reserved_bytes']==60*2**30


def test_retained_array_rgb_and_trajectory_checks_unchanged():
    old=(a.BASE/'work/wan22-action-cuda-visual-actual-audit-v1/audit.py').read_text()
    new=(HERE/'audit.py').read_text()
    def functions(text):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    x,y=functions(old),functions(new)
    for name in ('array_file','tensor_sha','inventory','jsonlines','check_rgb'):
        assert x[name]==y[name]
    start="        schedule=[r['timestep']"
    end="        need(records['core']['model_frozen']"
    assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
    previous=a.BASE/'work/wan22-action-cuda-visual-prep-v1/sampler.py'
    current=a.BASE/'work/wan22-action-cuda-final128-eval-prep-v1/sampler.py'
    assert previous.read_bytes()==current.read_bytes()
