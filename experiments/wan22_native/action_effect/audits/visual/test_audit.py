"""Synthetic future-result binding failures; no actual checkpoint or model call."""
import ast
import copy
import json
from pathlib import Path
import pytest
import binding as b
import audit as a

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

@pytest.fixture
def packet(tmp_path,monkeypatch):
    here=tmp_path/'auditor';here.mkdir();(here/'binding.py').write_bytes(Path(b.__file__).read_bytes())
    monkeypatch.setattr(b,'HERE',here)
    prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'source/local').mkdir(parents=True)
    (prepared/'source/local/fixture.py').write_text('# synthetic source fixture\n')
    mapping={'local':{'fixture.py':b.sha(prepared/'source/local/fixture.py')}}
    sources={'local':{'training.py':'1'*64}}
    contract={'schema':'worldline-action-effect128-run-v1','schedule':[{'k':506,'sigma':.506}],'negative_context_adapter_training':True}
    training={**contract,'source_sha256':sources};put(prepared/'evidence/training/plan.json',training)
    training_sha=b.sha(prepared/'evidence/training/plan.json')
    (prepared/'adapter.safetensors').write_bytes(b'Synthetic checksum fixture. This is not a model checkpoint.')
    cp=b.sha(prepared/'adapter.safetensors')
    trained={k:str(i)*64 for i,k in enumerate(('parent_sha256','worker_sha256','training_sha256','terminal_sha256'),1)}
    trained.update(plan_sha256=training_sha,checkpoint_sha256=cp,checkpoint_manifest_sha256='5'*64)
    identity={k:trained[k] for k in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
    identity.update(final_checkpoint_manifest_sha256=trained['checkpoint_manifest_sha256'],final_checkpoint_sha256=cp,source_sha256=sources)
    completed={'schema':'worldline-action-effect128-actual-independent-v1','status':'passed','completed_updates':128,'foundation_values_unchanged':True,'final_checkpoint_only':True,'auxiliary_updates':32,'auxiliary_head_predictions':128,'identity':identity}
    put(prepared/'training-audit.json',completed)
    cpu={'status':'passed','tests':11,'source_unchanged':True,'source_sha256':mapping,'failures':0,'errors':0,'skipped':0}
    put(prepared/'cpu-report.json',cpu)
    source_review=tmp_path/'source-review.json';put(source_review,{'status':'passed','source_sha256':mapping})
    plan={'schema':'worldline-action-effect128-visual-pair-v1','scope':'effect128-matched-wait-interact-visual-pair','status':'prepared','source_sha256':mapping,'negative_context_adapter_training':True,'final_checkpoint_updates':128,'checkpoint_selection':False,'training':False,'profile':'spatial','arms':['closed','open'],'settings':{'steps':50,'shift':5.,'guidance':5.},'input_identity':{'training':trained},'artifacts':{name:b.sha(prepared/name) for name in ('adapter.safetensors','cpu-report.json')},'evidence_sha256':{'evidence/training/plan.json':training_sha},'cpu_report_sha256':b.sha(prepared/'cpu-report.json'),'training_audit_sha256':b.sha(prepared/'training-audit.json')}
    put(prepared/'plan.json',plan)
    review=tmp_path/'plan-review.json';put(review,{'schema':b.REVIEW_SCHEMA,'status':'passed','visual_plan_sha256':b.sha(prepared/'plan.json'),'actual_audit_sha256':plan['training_audit_sha256'],'final_checkpoint_sha256':cp,'source_sha256':mapping})
    pins={'visual_source_sha256':mapping,'visual_cpu_sha256':b.sha(prepared/'cpu-report.json'),'source_review_sha256':b.sha(source_review),'training_source_sha256':sources,'training_contract':contract,'training_plan_sha256':training_sha}
    put(here/'pins.json',pins)
    return prepared,source_review,review,tmp_path/'binding.json',plan,completed,sources

def test_actual_file_binding_and_hash_required(packet):
    p,s,r,out,*_=packet;value=b.create(p,s,r,out)
    assert b.load(out,b.sha(out))==value
    with pytest.raises(ValueError,match='checksum'):b.load(out,'0'*64)
    with pytest.raises(ValueError,match='Fresh'):b.create(p,s,r,out)

def test_missing_training_audit_cannot_create_binding(packet):
    p,s,r,out,*_=packet;(p/'training-audit.json').unlink()
    with pytest.raises(ValueError):b.create(p,s,r,out)
    assert not out.exists()

def test_incomplete_or_wrong_checkpoint_and_auxiliary_identity(packet):
    p,s,r,out,plan,completed,sources=packet
    a.check_training_binding(plan,completed,sources,plan['artifacts']['adapter.safetensors'])
    for key,value in [('status','running'),('completed_updates',16),('auxiliary_updates',31),('auxiliary_head_predictions',127),('foundation_values_unchanged',False)]:
        bad=copy.deepcopy(completed);bad[key]=value
        with pytest.raises(AssertionError):a.check_training_binding(plan,bad,sources,plan['artifacts']['adapter.safetensors'])
    with pytest.raises(AssertionError):a.check_training_binding(plan,completed,sources,'0'*64)
    bad=copy.deepcopy(plan);bad['negative_context_adapter_training']=False
    with pytest.raises(AssertionError):a.check_training_binding(bad,completed,sources,plan['artifacts']['adapter.safetensors'])

def test_changed_actual_evidence_and_unreviewed_plan_rejected(packet):
    p,s,r,out,*_=packet;b.create(p,s,r,out);expected=b.sha(out)
    (p/'adapter.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError):b.load(out,expected)
    value=b.read(r);value['visual_plan_sha256']='0'*64;put(r,value)
    with pytest.raises(ValueError):b.inspect(p,s,r)

def test_source_or_training_contract_change_rejected(packet):
    p,s,r,out,*_=packet
    (p/'source/local/fixture.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='source'):b.create(p,s,r,out)
    assert not out.exists()
    for name in ('../escape','/absolute'):
        with pytest.raises(ValueError):b.member(p,name)

def test_numerical_auditor_and_sampler_preserved():
    here=Path(a.__file__).resolve().parent
    old=(here/'reference/final128-audit.py').read_text();new=(here/'audit.py').read_text()
    def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    x,y=functions(old),functions(new)
    for name in ('array_file','tensor_sha','inventory','jsonlines','check_rgb','check_hardware'):assert x[name]==y[name]
    begin="            steps=jsonlines(directory/'steps.jsonl')";end="        need(records['core']['model_frozen']"
    assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)]
    begin="        values=array_file(root/'sampling-inputs.safetensors'";end="        schedule=[r['timestep']"
    assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)]
    pins=json.loads((here/'pins.json').read_text())
    assert pins['visual_source_sha256']['local']['sampler.py']=='374e2d1b68015c7ce0b436dcddb68c6f5d8b247584feaa9140b94ec1aae3c314'
