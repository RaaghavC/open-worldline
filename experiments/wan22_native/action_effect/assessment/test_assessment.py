"""Bounded CPU integration, without external weights or CUDA."""
import copy,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import assessment as a
b=a.bindings;torch=a.torch;HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def reports():
    plan=a.vi.read(HERE/'effect-training-plan.json');sources=plan['source_sha256']
    parent={'schema':b.RUN_SCHEMA,'status':'passed','completed_updates':128,'model_execution':True}
    worker={'schema':b.RUN_SCHEMA,'status':'passed','completed_updates':128,'base_unchanged':True,'source_sha256':sources,'negative_context_adapter_training':True}
    training={'schema':b.RESULT_SCHEMA,'status':'passed','completed_updates':128,'base_unchanged':True,'training_forwards':256,'auxiliary_feature_extracts':64,'auxiliary_head_predictions':128,'zero_adapter_gate_passed':True,'negative_context_adapter_training':True,'schedule':copy.deepcopy(plan['schedule']),'updates':[{'update':i+1,'auxiliary':{'enabled':i%4==0}} for i in range(128)]}
    return [plan,parent,worker,training,{'status':'complete','exit_code':0},{'directory':'checkpoint-0128','completed_updates':128},{'completed_updates':128},copy.deepcopy(plan),sources]

def test_completed_training_schema_schedule_counts_and_negative_context():
    base=reports();b.validate_reports(*base)
    for i,key,value in [(1,'completed_updates',16),(2,'base_unchanged',False),(3,'auxiliary_head_predictions',127),(3,'negative_context_adapter_training',False),(4,'exit_code',False),(5,'directory','checkpoint-0016')]:
        bad=copy.deepcopy(base);bad[i][key]=value
        with pytest.raises(ValueError):b.validate_reports(*bad)
    bad=copy.deepcopy(base);bad[0]['schedule'][16]['k']+=1
    with pytest.raises(ValueError):b.validate_reports(*bad)
    bad=copy.deepcopy(base);bad[3]['updates'][17]['auxiliary']['enabled']=True
    with pytest.raises(ValueError):b.validate_reports(*bad)

def test_audit_exact_final_checkpoint_source_terminal_and_counts():
    identity={k:str(i)*64 for i,k in enumerate(('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256','checkpoint_manifest_sha256','checkpoint_sha256'))};sources={'repository':{'file.py':'a'*64}}
    report={'schema':b.AUDIT_SCHEMA,'status':'passed','completed_updates':128,'foundation_values_unchanged':True,'final_checkpoint_only':True,'main_predictions':256,'auxiliary_updates':32,'auxiliary_feature_extracts':64,'auxiliary_head_predictions':128,'identity':b.audit_identity(identity,sources)}
    b.validate_audit(report,identity,sources)
    for key in ('terminal_sha256','final_checkpoint_sha256','source_sha256'):
        bad=copy.deepcopy(report);bad['identity'].pop(key)
        with pytest.raises(ValueError):b.validate_audit(bad,identity,sources)
    for key in ('main_predictions','auxiliary_updates','completed_updates'):
        bad=copy.deepcopy(report);bad[key]-=1
        with pytest.raises(ValueError):b.validate_audit(bad,identity,sources)

def test_actual_original_input_reader_and_small_checkpoint_loads():
    root=ROOT/'work/wan22-action-cuda-comparison128-prepared-v1'
    assert a.vi.sha(root/'plan.json')==a.OLD_PLAN_SHA
    plan,old,values,contexts,commands,cases=a.ref.read_prepared(root)
    assert plan['checkpoint_128']['checkpoint_sha256']==a.OLD128_SHA
    assert list(contexts['native_negative'].shape)==[126,4096]
    assert list(values['observation'].shape)==[1,48,1,44,78]
    assert torch.equal(cases['closed_times'],cases['open_times']) and set(torch.unique(cases['closed_times']).tolist())=={0,506}
    assert not torch.equal(cases['closed_noisy'][:,:,1:],cases['open_noisy'][:,:,1:])
    assert torch.equal(cases['closed_noisy'][:,:,:1],values['observation']) and not torch.cuda.is_initialized()

def test_same_bridge_feature_reuse_survives_adapter_state_changes():
    from experiments.wan22_native.action_cuda.test_cpu import fixture,assert_no_hooks
    bridge,x,t,c,commands,obs=fixture();adapter=bridge.adapter
    zero={k:v.detach().clone() for k,v in adapter.state_dict().items()};states={cp:copy.deepcopy(zero) for cp in a.reader.CHECKPOINTS}
    name=next(k for k,v in zero.items() if k.startswith('output.') and v.ndim==1)
    states['old128'][name].fill_(.01);states['new128'][name].fill_(.03)
    ids={cp:{'tensor_sha256':{k:a.ref.sampler.tensor_sha(v) for k,v in rows.items()}} for cp,rows in states.items()}
    features=bridge.extract_features(x,t,c);before=features.hidden.clone();outputs={}
    for cp in ('zero','old128','new128','zero'):
        a.activate_checkpoint(adapter,states,ids,cp)
        value=bridge.predict_from_features(features,commands,obs,track_grad=False)
        direct=bridge(x,t,c,commands=commands,observation=obs,track_grad=False)
        assert torch.equal(value,direct)
        if cp in outputs:assert torch.equal(value,outputs[cp])
        outputs[cp]=value
    assert not torch.equal(outputs['zero'],outputs['new128']) and torch.equal(features.hidden,before)
    assert all(p.grad is None for p in adapter.parameters());assert_no_hooks(bridge.core)
    bad=copy.deepcopy(ids);bad['zero']['tensor_sha256'][name]='0'*64
    with pytest.raises(ValueError):a.activate_checkpoint(adapter,states,bad,'zero')

def test_distinct_additional_inputs_and_exact_eighteen_saved_heads():
    shape=a.reader.SHAPE;obs=np.ones((1,48,1,44,78),np.float32);x=np.zeros(shape,np.float32);x[:,:,:1]=obs
    t=np.full((1,4290),999,np.int64);t[:,:858]=0;values={'initial_latent':torch.from_numpy(x[0]),'token_times':torch.from_numpy(t)}
    contexts={'positive':np.zeros((25,4096),np.float32),'negative':np.ones((126,4096),np.float32)}
    cmds={arm:np.zeros((1,16,6),np.float32) for arm in a.reader.ARMS};cmds['open'][0,0,5]=1
    cases={};extracts=[];saved=[];calls=[]
    for i,arm in enumerate(a.reader.ARMS):
        noisy=x.copy();noisy[:,:,1:]=2+i;times=t.copy();times[:,858:]=506
        cases.update({arm+'_noisy':torch.from_numpy(noisy),arm+'_times':torch.from_numpy(times),arm+'_target':torch.full(shape,4.)})
    def extract(name,text,latent,times,context):
        handle=object();extracts.append((handle,name,text,latent.copy(),times.copy()));latent[:]=123;return handle
    def predict(cp,feature,command,observed):
        calls.append((cp,feature,float(command[0,0,5])));assert np.array_equal(observed,obs);return np.zeros(shape,np.float32)
    result=a.additional_checks(values,cases,obs,cmds,contexts,np.ones(shape,np.float32),extract,predict,lambda name,v:saved.append(name),lambda:None)
    assert len(saved)==18 and len(extracts)==4 and len(calls)==18
    assert [sum(c[1] is f[0] for c in calls) for f in extracts]==[3,3,6,6]
    assert all(result['seen506'][cp]['closed']['future_flow_mse_fp64']==16 for cp in a.reader.CHECKPOINTS)
    assert all(result['original999'][cp]['normalized_contrast_mse']==1 for cp in a.reader.CHECKPOINTS)
    assert np.all(values['initial_latent'].numpy()[:,1:]==0)

def test_expired_deadline_blocks_launch_and_failed_child_is_reaped(tmp_path):
    out=tmp_path/'expired';out.mkdir();(out/'plan.json').write_text('{}');decision=tmp_path/'decision.json';decision.write_text('{}');digest=a.vi.sha(decision)
    with patch.object(a,'read_prepared',return_value=({},)),patch.object(a,'admission',return_value=digest),patch.object(a.time,'monotonic',side_effect=[1.,902.,903.]),patch.object(a.subprocess,'Popen') as popen:
        with pytest.raises(RuntimeError,match='Deadline before launch'):a.execute(out,tmp_path,decision)
        popen.assert_not_called()
    assert a.vi.read(out/'metrics.json')['model_execution']is False
    out=tmp_path/'failed';out.mkdir();(out/'plan.json').write_text('{}')
    class Child:returncode=124
    child=Child()
    with patch.object(a,'read_prepared',return_value=({},)),patch.object(a,'admission',return_value=digest),patch.object(a.subprocess,'Popen',return_value=child),patch.object(a.ref,'supervise',side_effect=RuntimeError('stop')),patch.object(a.ref,'stop_child') as stop:
        with pytest.raises(RuntimeError,match='stop'):a.execute(out,tmp_path,decision)
        stop.assert_called_once_with(child)
    assert a.vi.read(out/'metrics.json')['model_execution']is None
    with pytest.raises(FileExistsError):a.execute(out,tmp_path,decision)

def test_cpu_review_and_admission_require_exact_identity_and_counts(tmp_path):
    source=a.sources();r={'status':'passed','source_sha256':source,'tests':14,'failures':0,'errors':0,'skipped':0,'exit_code':0,'source_unchanged':True,'cuda_initialized':False}
    p=tmp_path/'cpu.json';p.write_text(json.dumps(r));a.cpu_review(p)
    r['cuda_initialized']=True;p.write_text(json.dumps(r))
    with pytest.raises(ValueError):a.cpu_review(p)
    plan={'source_sha256':source,'checkpoint_sha256':{'zero':a.ZERO_SHA,'old128':a.OLD128_SHA,'new128':'f'*64},'new_training_audit_sha256':'e'*64,'cpu_report_sha256':'d'*64};(tmp_path/'plan.json').write_text(json.dumps(plan))
    d={'schema':'worldline-action-effect-assessment-admission-v1','decision':'admit','issued_by':'parent-agent','plan_sha256':a.vi.sha(tmp_path/'plan.json'),'source_sha256':source,'limits':a.ref.limits('pair'),'counts':a.counts(),'checkpoint_sha256':plan['checkpoint_sha256'],'heldout_manifest_sha256':a.reader.MANIFEST_SHA,'new_training_audit_sha256':'e'*64,'cpu_report_sha256':'d'*64,'training_admitted':False,'sampling_admitted':False}
    p=tmp_path/'decision.json';p.write_text(json.dumps(d));a.admission(p,tmp_path,plan)
    d['counts']['head_predictions']=65;p.write_text(json.dumps(d))
    with pytest.raises(ValueError):a.admission(p,tmp_path,plan)
