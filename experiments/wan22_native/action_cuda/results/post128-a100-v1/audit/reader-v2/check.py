"""Bounded prepared-input, arithmetic and exact call-coverage fixtures."""
import ast,copy,json,struct,tempfile,time,sys
from pathlib import Path
import numpy as np
import audit as a

started=time.monotonic();before=a.sources();checks=[]
plan,old,cases,values,contexts,commands=a.prepared_inputs(a.PREPARED)
checks.append('Actual final128 plan/audit/checkpoint and unchanged original14 k506/k999 input bytes')
record=a.read_json(a.PREPARED/'training-audit.json');mapping=a.read_json(a.PREPARED/'source/training_reader/training-source.json')
for key,value in [('completed_updates',16),('foundation_values_unchanged',False),('status','running')]:
 bad=copy.deepcopy(record);bad[key]=value
 try:a.training_binding(plan,bad,mapping)
 except ValueError:pass
 else:raise AssertionError('Accepted bad final128 audit '+key)
bad=copy.deepcopy(plan);bad['checkpoint_128']['checkpoint_sha256']='0'*64
try:a.training_binding(bad,record,mapping)
except ValueError:pass
else:raise AssertionError('Accepted wrong final128 checkpoint')
checks.append('Rejects incomplete/nonfrozen/wrong-checkpoint actual128 audit evidence')
names=a.prediction_names()
assert len(names)==len(set(names))==20 and names[:2]==['native-atrium','native-native_negative']
assert names[-6:]==['objective-128-closed','objective-128-open','causal-128-closed-atrium','causal-128-closed-native_negative','causal-128-open-atrium','causal-128-open-native_negative']
assert len([n for n in names if n.startswith('native-')])==2
assert len([n for n in names if n.startswith('objective-')])==6
assert len([n for n in names if n.startswith('causal-0-')])==4
checks.append('Two native calls once plus exactly six calls for each0/16/128 checkpoint')
shape=(1,2,5,2,2);target=np.zeros(shape,np.float32);prediction=np.ones(shape,np.float32);prediction[:,:,:1]=1000
assert a.losses(prediction,target)['float64_mse']==1 and a.losses(prediction,target)['numpy_float32_mse']==1
pc=np.full(shape,2,np.float32);po=np.full(shape,2.5,np.float32);nc=np.full(shape,1,np.float32);no=np.full(shape,1.625,np.float32)
assert np.all(a.cfg(pc,nc)==6) and np.all(a.cfg(po,no)==6)
assert a.contrast(a.cfg(po,no),a.cfg(pc,nc))['difference_l2']==0 and np.all(po-pc==.5) and np.all(no-nc==.625)
checks.append('Known prefix excluded from objective and nonzero text-specific effects can cancel exactly under CFG5')
old_source=(a.BASE/'work/wan22-action-cuda-diagnostic-actual-audit-v1/source/audit.py').read_text()
new_source=Path(a.__file__).read_text()
def funcs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
x,y=funcs(old_source),funcs(new_source)
for name in ('tensors','contrast','cfg','losses','exact'):assert x[name]==y[name]
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'fixture.safetensors';data=np.arange(12,dtype='<f4').reshape(3,4)
 head={'v':{'shape':[3,4],'dtype':'F32','data_offsets':[0,48]}}
 raw=json.dumps(head).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+data.tobytes())
 a.exact(a.tensors(path,{'v':((3,4),'F32')})['v'],data,'independent byte fixture')
 path.write_bytes(path.read_bytes()+b'bad')
 try:a.tensors(path,{'v':((3,4),'F32')})
 except ValueError:pass
 else:raise AssertionError('Accepted trailing tensor bytes')
checks.append('Original tensor/contrast/loss/CFG function ASTs unchanged; independent bounded byte fixture rejects trailing data')
sample={'host_rss_bytes':836259840,'host_available_bytes':2071829778432,
        'cuda_reserved_bytes':1721761792,'cuda_allocated_bytes':1742798848,'cuda_available_bytes':82926567424}
a.check_worker_sample(sample)
for key,value in [('cuda_reserved_bytes',60*2**30+1),('host_rss_bytes',48*2**30+1),('host_available_bytes',8*2**30-1),('cuda_available_bytes',8*2**30-1),('cuda_allocated_bytes',True)]:
 bad=dict(sample);bad[key]=value
 try:a.check_worker_sample(bad)
 except ValueError:pass
 else:raise AssertionError('Accepted declared cap/type violation '+key)
checks.append('Sequential allocated/reserved mismatch accepted descriptively; all declared resource caps and integer types still reject violations')
assert before==a.sources() and 'torch' not in sys.modules
r={'status':'passed','checks':checks,'tests':len(checks),'elapsed_seconds':time.monotonic()-started,'source_sha256':before,'fixture_source_sha256':a.sha(Path(__file__)),'prepared_plan_sha256':a.PLAN_SHA,'numpy':np.__version__,'model_execution':False,'cloud_calls':False,'torch_imported':False,'actual20_run_audited':False}
(a.HERE/'cpu-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'passed','checks':len(checks),'seconds':r['elapsed_seconds'],'audit_sha256':a.sha(Path(a.__file__)),'report_sha256':a.sha(a.HERE/'cpu-check.json')}))
