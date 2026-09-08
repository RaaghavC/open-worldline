# SPDX-License-Identifier: Apache-2.0
import json,struct,tempfile,time
from pathlib import Path
import numpy as np
import audit as a

started=time.monotonic();before=a.sources();checks=[]
plan,*_=a.prepared_inputs(a.PREPARED)
checks.append('Exact prepared-v3 source/input/checkpoint hashes, k506 corruption, 858-token observed prefixes, k999 initial noise and original commands')
shape=(1,2,5,2,2);target=np.zeros(shape,np.float32);prediction=np.ones(shape,np.float32);prediction[:,:,:1]=1000
assert a.losses(prediction,target)['float64_mse']==1 and a.losses(prediction,target)['numpy_float32_mse']==1
checks.append('Future-only MSE excludes a large deliberate initial-latent error')
positive_closed=np.full(shape,2,np.float32);positive_open=np.full(shape,2.5,np.float32)
negative_closed=np.full(shape,1,np.float32);negative_open=np.full(shape,1.625,np.float32)
gc=a.cfg(positive_closed,negative_closed);go=a.cfg(positive_open,negative_open)
assert np.all(gc==6) and np.all(go==6) and a.contrast(go,gc)['difference_l2']==0
assert np.all(positive_open-positive_closed==.5) and np.all(negative_open-negative_closed==.625)
checks.append('Analytic nonzero positive/negative command effects cancel exactly under FP32 CFG5')
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory).resolve();path=root/'fixture.safetensors';values={'f':np.arange(12,dtype=np.float32).reshape(3,4),'t':np.array([[0,999]],dtype=np.int64)}
 head={};offset=0;payload=[]
 for name,value in values.items():
  raw=value.tobytes();head[name]={'shape':list(value.shape),'dtype':'F32' if value.dtype==np.float32 else 'I64','data_offsets':[offset,offset+len(raw)]};offset+=len(raw);payload.append(raw)
 head['__metadata__']={'purpose':'synthetic reader check'};header=json.dumps(head).encode();path.write_bytes(struct.pack('<Q',len(header))+header+b''.join(payload))
 read=a.tensors(path,{'f':((3,4),'F32'),'t':((1,2),'I64')})
 for name,value in values.items():a.exact(read[name],value,'synthetic')
 checks.append('Bounded raw F32/I64 reader agrees with independently assembled byte fixture')
 for mutation in ['metadata','offsets','trailing']:
  h=json.loads(header);body=b''.join(payload)
  if mutation=='metadata':h['__metadata__']['purpose']=1
  elif mutation=='offsets':h['t']['data_offsets'][0]-=4
  else:body+=b'BAD!'
  raw=json.dumps(h).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+body)
  try:a.tensors(path,{'f':((3,4),'F32'),'t':((1,2),'I64')})
  except ValueError:pass
  else:raise AssertionError('Accepted malformed '+mutation)
 checks.append('Non-string metadata, overlapping ranges and trailing payload are rejected')
assert before==a.sources()
r={'status':'passed','checks':checks,'elapsed_seconds':time.monotonic()-started,'source_sha256':before,'fixture_source_sha256':a.sha(Path(__file__)),'prepared_plan_sha256':a.PLAN_SHA,'numpy':np.__version__,'model_execution':False,'cloud_calls':False,'recovered_diagnostic_audited':False}
(a.HERE/'cpu-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'passed','checks':len(checks),'seconds':r['elapsed_seconds'],'report_sha256':a.sha(a.HERE/'cpu-check.json')}))
