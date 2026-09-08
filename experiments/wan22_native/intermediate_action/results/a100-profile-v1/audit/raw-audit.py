"""Read-only audit of verified recovered CPU arrays. No model construction or replay."""
from pathlib import Path
import hashlib,json,math,sys,time,subprocess
import numpy as np
import torch
from safetensors.numpy import load_file
W=Path.cwd(); R=W/'work/intermediate-profile-recovered-stages-v1'; ROOT=R/'recovered'
OUT=W/'work/intermediate-action-profile-audit-v1'; started=time.monotonic()
sys.dont_write_bytecode=True;torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def th(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def arr(p):
 d=load_file(p)
 assert all(np.isfinite(a).all() for a in d.values())
 return d
def close(a,b):
 assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-8),(a,b)
def resources(root,worker,mode):
 p=js(root/'metrics.json'); t=js(root/'terminal.json'); w=js(root/worker/'metrics.json'); m=js(root/worker/'monitor-terminal.json')
 limit=dict(seconds=float(900 if mode=='pair' else 600),host_rss_bytes=48*2**30,cuda_reserved_bytes=60*2**30,minimum_host_available_bytes=8*2**30,minimum_cuda_available_bytes=8*2**30,minimum_gpu_total_bytes=70*2**30)
 for v in (p,t,w,m):assert v['limits']==limit and 0<=v['elapsed_seconds']<limit['seconds']
 assert p['status']==w['status']=='passed' and t['status']==m['status']=='complete' and t['exit_code']==0 and t['cleanup_error'] is None
 assert t['peak_combined_rss_bytes']<=limit['host_rss_bytes'] and t['minimum_host_available_bytes']>=limit['minimum_host_available_bytes']
 assert p['terminal_sha256']==sha(root/'terminal.json') and p['source_sha256']==w['source_sha256']
 assert p['plan_sha256']==w['plan_sha256']==sha(root/'plan.json') and p['admission']==w['admission']
 assert p['admission']['sha256']==sha(root/'executed-admission.json')
 assert p.get('result_metrics_sha256',p.get('worker_metrics_sha256'))==sha(root/worker/'metrics.json')
 assert not list(root.rglob('watchdog-stop.json'))
 plan=js(root/'plan.json');assert plan['source_sha256']==w['source_sha256']
 for name,digest in plan['source_sha256'].items():assert sha(root/'source'/name)==digest
 samples=[json.loads(line) for line in (root/worker/'memory.jsonl').read_text().splitlines()]
 assert len(samples)==m['sample_count']>0
 last=-1;non_atomic=0
 for s in samples:
  assert type(s['seconds']) in (float,int) and math.isfinite(s['seconds']) and last<=s['seconds']<limit['seconds'];last=s['seconds']
  for k in ('host_rss_bytes','host_available_bytes','cuda_reserved_bytes','cuda_allocated_bytes','cuda_available_bytes'):assert type(s[k]) is int and s[k]>=0
  assert s['host_rss_bytes']<=48*2**30 and s['cuda_reserved_bytes']<=60*2**30 and s['cuda_allocated_bytes']<=60*2**30 and s['host_available_bytes']>=8*2**30 and s['cuda_available_bytes']>=8*2**30
  non_atomic+=s['cuda_allocated_bytes']>s['cuda_reserved_bytes']
 assert w['hardware']['total_memory_bytes']>=70*2**30
 return dict(samples=len(samples),non_atomic_allocated_above_reserved=non_atomic,elapsed_parent=p['elapsed_seconds'],peak_sampled_reserved=max(s['cuda_reserved_bytes'] for s in samples)),p,w,plan
report=dict(schema='worldline-intermediate-profile-and-factorial-cache-raw-audit-v1',status='running',model_execution=False)
try:
 assert sha(R/'index.json')=='9a25833ebe9839b77021c3a4efce1a04f638836cd35188607fc053534886d95f'
 idx=js(R/'index.json'); assert len(idx['files'])==511
 paths={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()};assert paths==set(idx['files'])
 for name,row in idx['files'].items():
  p=ROOT/name;assert not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
 assert sum(r['bytes'] for r in idx['files'].values())==412466579
 report['recovery']=dict(index_sha256=sha(R/'index.json'),verification_sha256=sha(R/'recovery-verified.json'),files=511,bytes=412466579)
 # Cache: retain a full independent raw tensor scan; invoke the exact read-only producer gate separately.
 cache_root=ROOT/'action-results/factorial-native-cache-v1'
 res,parent,worker,plan=resources(cache_root,'worker','codec'); c=js(cache_root/'result/completion.json')
 assert worker['result_completion_sha256']==sha(cache_root/'result/completion.json')
 for n,h in worker['output_sha256'].items():assert sha(cache_root/n)==h
 loaded=js(cache_root/'worker/weight-load.json');before=js(cache_root/'worker/codec-before.json');after=js(cache_root/'worker/codec-after.json')
 expected={n:dict(shape=r['shape'],dtype='float32',sha256=r['sha256']) for n,r in loaded['tensors'].items()}
 assert len(before)==196 and before==after==expected and c['identity']['codec_provenance']==loaded
 assert loaded['weight_sha256']=='20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
 assert c['encoder_attempts']==c['encoder_completed']==8 and c['one_frame_calls']==2 and c['seventeen_frame_calls']==6
 tensors={}
 for name,row in c['files'].items():
  p=cache_root/'result'/name;assert sha(p)==row['sha256'];d=arr(p);tensors[name]=d
  assert set(d)==set(row['tensors'])
  for k,a in d.items():assert str(a.dtype)=='float32' and list(a.shape)==row['tensors'][k]['shape'] and th(a)==row['tensors'][k]['sha256']
 shared=tensors['observation.safetensors']['observation'];assert shared.shape==(1,48,1,44,78)
 first=tensors[c['arms'][0]+'.safetensors']['target'][:,:,:1]
 checked=[]
 for row in c['checks']:
  name=row['name']
  if name.endswith('_cross_length'):a=tensors[name.removesuffix('_cross_length')+'.safetensors']['target'][:,:,:1];b=shared
  elif name.endswith('_same_length'):a=tensors[name.removesuffix('_same_length')+'.safetensors']['target'][:,:,:1];b=first
  else:assert name=='shared_image_after_six_videos';a=tensors['observation-repeat.safetensors']['observation'];b=shared
  assert np.array_equal(a,b) and th(a)==row['candidate_sha256'] and th(b)==row['reference_sha256']
  assert row['passed'] and row['bit_exact_equal'] and row['max_abs']==row['relative_l2']==0
  checked.append(name)
 assert len(checked)==12
 for arm in c['arms']:
  d=tensors[arm+'.safetensors'];assert d['target'].shape==(1,48,5,44,78) and np.array_equal(d['observation'],shared)
  assert d['commands'].shape==(1,16,6) and th(d['commands'])==plan['input_plan']['arms'][arm]['arrays']['commands']['sha256']==c['identity']['commands_sha256'][arm]
 code="import sys,json,torch;from pathlib import Path;sys.dont_write_bytecode=True;torch.set_num_threads(1);sys.path.insert(0,str(Path('outputs/open-worldline').resolve()));sys.path.insert(0,str(Path('work/atrium-factorial-native-cache-v1').resolve()));import run;r=Path(sys.argv[1]);v,p=run.read_completed(r,'stationary_closed',conditioning_only=False);assert not torch.cuda.is_initialized();print(json.dumps(dict(status='passed',source_sha256=p['identity']['source_sha256'],completion_sha256=p['completion_sha256'])))"
 result=subprocess.run([sys.executable,'-c',code,str(cache_root)],capture_output=True,text=True,check=True)
 gate=json.loads(result.stdout);assert gate['status']=='passed'
 report['cache']=dict(status='passed',parent_sha256=sha(cache_root/'metrics.json'),worker_sha256=sha(cache_root/'worker/metrics.json'),completion_sha256=sha(cache_root/'result/completion.json'),plan_sha256=sha(cache_root/'plan.json'),source_sha256=plan['source_sha256'],terminal_sha256=sha(cache_root/'terminal.json'),encoded_files=8,bit_exact_prefix_checks=12,original_values_unchanged=196,source_reader_passed=True,resources=res)
 (OUT/'cache-raw-audit.json').write_text(json.dumps(report['cache'],indent=2)+'\n')
 print('CACHE RAW PASS',flush=True)
 # Original profile reader is CPU-only, exact input/source checking with no core construction.
 sys.path.insert(0,str(W/'outputs/open-worldline'));sys.path.insert(0,str(W/'work/intermediate-action-cuda-profile-v1'))
 import packet
 from experiments.wan22_native.action_cuda.probe import _original_weights
 from experiments.wan22_native.action_cuda import probe_math
 from experiments.wan22_native.action_effect.training import effect
 root=ROOT/'action-results/intermediate-profile-spatial-v1';res,parent,w,plan=resources(root,'result','pair'); checked_plan,data=packet.read_prepared(root)
 assert checked_plan==plan
 for n,h in w['output_sha256'].items():assert sha(root/'result'/n)==h
 expected=_original_weights(js(root/'result/weight-load.json'));assert len(expected)==825
 for n in ('before','after-parity','after-block28','after-block29'):assert js(root/f'result/core-{n}.json')==expected
 assert w['completed_updates']==4 and w['zero_gate_passed'] and w['all825_current_value_hashes_verified']
 assert w['rotary_before_sha256']==w['rotary_after_sha256'] and w['rotary_copy_exact']
 for row in w['parity']:
  ref=arr(root/f"result/parity/native-{row['context']}.safetensors")['velocity']
  a=arr(root/f"result/parity/block{row['block_index']}-{row['context']}-{row['arm']}-{row['path']}.safetensors")['velocity']
  assert a.shape==ref.shape==(1,48,5,44,78) and np.array_equal(a,ref) and th(a)==row['bridged_sha256']==row['native_sha256']
 assert len(w['parity'])==16
 raw=list((root/'result').glob('parity/*.safetensors'))+list((root/'result').glob('block*/main-*.safetensors'))+list((root/'result').glob('block*/auxiliary-*.safetensors'))+list((root/'result').glob('block*/gradients-after-clip-*.safetensors'))
 assert len(raw)==46
 for p in raw:arr(p)
 results=[];checkpoints=[]
 for index in (28,29):
  for step in range(3):
   cp=root/f'result/block{index}/checkpoint-{step:04d}'; m=js(cp/'manifest.json');a=arr(cp/'adapter.safetensors')
   assert m['completed_updates']==step and m['identity']['placement']==f'block{index}' and m['identity']['plan_sha256']==sha(root/'plan.json') and m['identity']['source_sha256']==plan['source_sha256']
   for n,h in m['files'].items():assert sha(cp/n)==h
   assert set(a)==set(m['tensors'])==set(data['initial']) and sum(v.size for v in a.values())==947712
   for n,v in a.items():assert th(v)==m['tensors'][n]['sha256'] and list(v.shape)==m['tensors'][n]['shape']
   state=torch.load(cp/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
   assert state['completed_updates']==step and state['identity']==m['identity'] and torch.equal(state['draw_rng_state'],data['rng'])
   group=state['optimizer']['param_groups'][0]
   for k,v in probe_math.OPTIMIZER.items():assert group[k]==v
   assert len(state['optimizer']['state'])==(18 if step else 0)
   for v in state['optimizer']['state'].values():
    assert v['step'].item()==step and all(torch.isfinite(t).all() for t in v.values() if isinstance(t,torch.Tensor))
   if not step:
    assert all(np.array_equal(a[n],data['initial'][n].numpy()) for n in a)
   checkpoints.append(dict(placement=index,step=step,manifest_sha256=sha(cp/'manifest.json')))
  assert js(root/f'result/block{index}/last-valid.json')['completed_updates']==2
  for step,row in enumerate(w['placements'][str(index)]['updates'],1):
   draw=data['schedule'][step-1];noise=data['draws'][draw['noise_key']];losses=[]
   for arm,window,record in zip(('closed','open'),data['windows'],row['branches']):
    x,t,v=probe_math.flow_inputs(window['target'],window['observation'],noise,draw['k'])
    hashes=dict(noisy=th(x.numpy()),token_times=th(t.numpy()),flow_target=th(v.numpy()),observation=th(window['observation'].numpy()),commands=th(window['commands'].numpy()))
    assert hashes==record['input_sha256'] and np.array_equal(x[:,:,:1].numpy(),window['observation'].numpy())
    pred=arr(root/f'result/block{index}/main-{step:04d}-{arm}.safetensors')['velocity']
    loss=float(np.square(pred[:,:,1:].astype(np.float64)-v[:,:,1:].numpy().astype(np.float64)).mean());close(loss,record['future_flow_mse']);losses.append(loss)
   x,t,target_tensor=effect.auxiliary_inputs(data['windows'],noise)
   ident=dict(pure_noise_sha256=th(noise.numpy()),input_sha256=th(x.numpy()),times_sha256=th(t.numpy()),observation_sha256=th(data['windows'][0]['observation'].numpy()),target_difference_sha256=th(target_tensor.numpy()),commands_sha256={a:th(v['commands'].numpy()) for a,v in zip(('closed','open'),data['windows'])},context_sha256={k:th(data[k].numpy()) for k in ('positive','negative')})
   assert ident==row['auxiliary']['input_identity']
   gs=[]
   for arm in ('closed','open'):
    pp=arr(root/f'result/block{index}/auxiliary-{step:04d}-{arm}-positive.safetensors')['velocity'];nn=arr(root/f'result/block{index}/auxiliary-{step:04d}-{arm}-negative.safetensors')['velocity']
    gs.append(nn+np.float32(5)*(pp-nn))
   target=(data['windows'][1]['target']-data['windows'][0]['target']).numpy()
   aux=float(np.square((-(gs[1]-gs[0]))[:,:,1:].astype(np.float64)-target[:,:,1:].astype(np.float64)).mean());close(aux,row['auxiliary']['future_clean_difference_mse'])
   close(sum(losses)/2,row['paired_mean_future_flow_mse']);close(sum(losses)/2+aux,row['total_objective'])
   grad=arr(root/f'result/block{index}/gradients-after-clip-{step:04d}.safetensors');assert set(grad)==set(data['initial'])
   norm=math.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in grad.values()));close(norm,row['gradient_l2_after_clip']);assert norm<=1
   gru=math.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for n,v in grad.items() if n.startswith('command_gru.')))
   close(gru,row['command_gru_gradient_l2']);assert (gru==0 if step==1 else gru>0)
   results.append(dict(placement=index,update=step,main_future_mse_fp64=sum(losses)/2,auxiliary_mse_fp64=aux,gradient_l2_fp64=norm,gru_gradient_l2_fp64=gru,reported_gru_gradient=row['command_gru_gradient_l2']))
 report['profile']=dict(status='passed',parent_sha256=sha(root/'metrics.json'),result_sha256=sha(root/'result/metrics.json'),terminal_sha256=sha(root/'terminal.json'),plan_sha256=sha(root/'plan.json'),source_sha256=plan['source_sha256'],raw_prediction_gradient_bundles=46,bit_exact_native_comparisons=16,checkpoints=checkpoints,all825_maps_equal_at_four_points=True,updates=results,resources=res)
 assert not torch.cuda.is_initialized()
 report.update(status='passed',elapsed_seconds=time.monotonic()-started,source_sha256=sha(__file__),limitations=['No native replay or backward replay. Original weight identity is checked through retained loader/value records, not the absent 21 GB checkpoints.','FP64 local descriptive losses are compared with GPU FP32 reductions using relative 2e-6, absolute 2e-8 reporting tolerance; exact tensor/input/parity/prefix gates remain exact.','No generated images or control quality are assessed by this numerical profile.'])
except BaseException as error:
 report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
 raise
finally:
 (OUT/'raw-audit-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'],report_sha256=sha(OUT/'raw-audit-report.json'))))
