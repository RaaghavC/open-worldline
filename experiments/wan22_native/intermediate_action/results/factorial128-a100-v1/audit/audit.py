"""Independent saved factorial128 audit. No model, backward, CUDA or network execution."""
from pathlib import Path
import argparse,json,math,os,time,shutil
import numpy as np
import checks as c
HERE=Path(__file__).resolve().parent
PLAN='030f04106f588dc903d5f9e3de3078badd7bf870f0923dadf0f5b796eaae314e'
ARMS=tuple(m+'_'+a for m in ('stationary','left','right') for a in ('closed','interact'))

def read(path):
    path=Path(path);c.need(path.is_file() and not path.is_symlink() and path.stat().st_size<=16*2**20,'Bounded regular JSON required')
    return c.parse(path.read_bytes())

def file(root,name):
    q=Path(name);c.need(not q.is_absolute() and q.parts and all(p not in ('.','..') for p in q.parts),'Relative evidence path required')
    p=root/q;c.need(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'Regular evidence file required');return p

def inventory(root):
    c.need(not any(p.is_symlink() for p in root.rglob('*')),'No evidence symlinks')
    return {str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':c.sha(p)} for p in sorted(root.rglob('*')) if p.is_file()}

def flow(window,noise,k):
    target=window['target'];obs=window['observation'];c.fp32(target);c.fp32(noise)
    c.need(np.array_equal(target[:,:,:1],obs),'Raw target prefix must already equal observation')
    x=np.add(np.multiply(np.float32(1-k/1000),target,dtype=np.float32),np.multiply(np.float32(k/1000),noise,dtype=np.float32),dtype=np.float32)
    x[:,:,:1]=obs;v=np.subtract(noise,target,dtype=np.float32);t=np.full((1,4290),k,np.int64);t[:,:858]=0
    identity={n:c.tensor_sha(a) for n,a in dict(noisy=x,token_times=t,flow_target=v,observation=obs,commands=window['commands']).items()}
    return x,t,v,identity

def schedule(plan,original):
    c.need(len(plan)==len(original)==128,'All128 original draws required')
    for i,(a,b) in enumerate(zip(plan,original)):
        motion=('stationary','left','right')[i%3]
        expected=dict(update=i+1,motion=motion,branches=[motion+'_closed',motion+'_interact'],auxiliary=i%4==0,
                      **{k:b[k] for k in ('noise_key','noise_sha256','k','sigma','rng_after_sha256')})
        c.need(a==expected and a['noise_key']==f'noise_{i:04d}' and type(a['k']) is int and 50<=a['k']<=950 and a['sigma']==a['k']/1000,'Exact prospective motion and original draw schedule required')

def resource_sample(sample):
    c.need(c.finite_number(sample.get('seconds')) and sample['seconds']<900,'Finite nonnegative sample time below deadline')
    required={'host_available_bytes','combined_rss_bytes'} if 'combined_rss_bytes' in sample else {'host_available_bytes','host_rss_bytes','cuda_reserved_bytes','cuda_allocated_bytes','cuda_available_bytes'}
    c.need(required<=set(sample) and all(type(sample[k]) is int and sample[k]>=0 for k in required),'Nonnegative integer memory counters required')
    c.need(sample['host_available_bytes']>=8*2**30 and sample.get('combined_rss_bytes',sample.get('host_rss_bytes'))<=48*2**30,'Original host resource caps')
    if 'cuda_reserved_bytes' in sample:c.need(sample['cuda_reserved_bytes']<=60*2**30 and sample['cuda_available_bytes']>=8*2**30,'Original CUDA resource caps')

def finite_tree(value):
    import torch
    if isinstance(value,torch.Tensor):c.need(value.device.type=='cpu' and torch.isfinite(value).all().item(),'Finite CPU recovery tensor required')
    elif isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)
    elif isinstance(value,float):c.need(math.isfinite(value),'Finite recovery scalar required')

def execute(root,repo,out,recovery=None):
    root=Path(root).resolve();repo=Path(repo).resolve();out=Path(out).absolute()
    c.need(not out.exists() and not out.resolve().is_relative_to(root) and not out.resolve().is_relative_to(repo),'Fresh separate audit output required')
    out.mkdir(parents=True);began=time.monotonic();report=dict(schema='worldline-factorial-intermediate128-actual-audit-v1',status='running',run=str(root),model_execution=False,gpu_replay=False,backward_replay=False,checks={})
    for name in ('audit.py','checks.py','expected-plan.json','expected-weights.json'):
        shutil.copyfile(HERE/name,out/name)
    report['auditor_source_sha256']={n:c.sha(out/n) for n in ('audit.py','checks.py','expected-plan.json','expected-weights.json')}
    try:
        before=inventory(root);r=root/'result';p=read(root/'plan.json');w=read(r/'metrics.json');parent=read(root/'metrics.json');terminal=read(root/'terminal.json')
        c.need(c.sha(HERE/'expected-plan.json')==PLAN==c.sha(root/'plan.json') and p==read(HERE/'expected-plan.json'),'Exact reviewed prepared plan')
        if recovery is not None:
            rec=Path(recovery).resolve();verified=read(rec/'recovery-verified.json');index=read(rec/'index.json')
            c.need(verified['status']=='verified' and c.sha(rec/'index.json')==verified['index_sha256'] and root.is_relative_to(rec/'recovered'),'Exact completed original recovery required')
            prefix=root.relative_to(rec/'recovered').as_posix()
            for n,v in before.items():c.need(index['files'][prefix+'/'+n]==v,'Verified recovered member '+n)
            report['recovery_verified_sha256']=c.sha(rec/'recovery-verified.json')
        for x in (parent,w):
            c.need(x.get('status')=='passed' and x.get('model_execution') is True and x.get('plan_sha256')==PLAN and x.get('source_sha256')==p['source_sha256'],'Completed exact source-bound parent/worker')
            c.need(x.get('limits')==p['protocol']['limits'] and 0<x['elapsed_seconds']<900,'Original limits and elapsed time')
        c.need(terminal.get('status')=='complete' and terminal.get('exit_code')==0 and terminal.get('cleanup_error') is None,'Completed zero-exit terminal')
        hardware=w['hardware'];c.need(type(hardware.get('total_memory_bytes')) is int and hardware['total_memory_bytes']>=70*2**30 and hardware['name']=='NVIDIA A100-SXM4-80GB' and hardware['torch']=='2.5.1+cu124' and hardware['cuda']=='12.4' and hardware['flash_attn']=='2.7.4.post1' and hardware['flash_attention_2_available'] is True,'Pinned A100 CUDA/FA2 runtime and minimum capacity')
        expected_precision=dict(parameter_storage='Original FP32',convert_model_dtype=False,outer_autocast='Native CUDA BF16 with default cache setting',inner_contexts='Unmodified upstream contexts',attention='Unmodified upstream FlashAttention 2',rope='Unmodified upstream complex RoPE')
        c.need(w['precision']==expected_precision and w['rotary_before_sha256']==w['rotary_after_sha256'],'Original precision and rotary identity')
        c.need(parent['result_metrics_sha256']==c.sha(r/'metrics.json') and parent['terminal_sha256']==c.sha(root/'terminal.json'),'Parent terminal/output binding')
        c.need(not list(root.rglob('watchdog-stop.json')) and not list(root.rglob('*cleanup-error.json')),'No stopped or failed cleanup')
        admitted=read(root/'executed-admission.json');admission_sha=c.sha(root/'executed-admission.json')
        c.need(parent['admission']==w['admission'] and w['admission']['sha256']==admission_sha,'Exact executed admission binding')
        required_admission=dict(schema=p['schema'],scope=p['protocol']['scope'],decision='admit',plan_sha256=PLAN,source_sha256=p['source_sha256'],prior_receipt_sha256=p['prior_receipt_sha256'],cpu_report_sha256=p['cpu_report_sha256'],expected_gpu=w['hardware']['name'],limits=p['protocol']['limits'],image_generation=False)
        c.need(all(admitted.get(k)==v for k,v in required_admission.items()) and w['admission']==dict(sha256=admission_sha,**required_admission),'Exact admission source/plan/limits/policy')
        for n,d in p['source_sha256'].items():
            c.need(c.sha(file(root/'source',n))==d,'Measured source '+n)
            if n.startswith('repo/'):c.need(c.sha(file(repo,n[5:]))==d,'Current repository source '+n)
        c.need(len(p['source_sha256'])==88,'All88 measured sources')
        cpu=read(root/'cpu-report.json');c.need(c.sha(root/'cpu-report.json')==p['cpu_report_sha256'] and cpu['status']=='passed' and cpu['source_sha256']==p['source_sha256'],'Passed current CPU review')
        c.need(cpu['tests']==15 and all(cpu[k]==0 for k in ('pytest_exit_code','failures','errors','skipped')),'Complete15 CPU checks')
        for n,v in p['files'].items():
            q=file(root,n);c.need(q.stat().st_size==v['bytes'] and c.sha(q)==v['sha256'],'Exact prepared file '+n)
        for label,d in p['prior_receipt_sha256'].items():c.need(c.sha(root/(label+'-validation.json'))==d,'Exact prior actual receipt')
        c.need(parent['prior_receipt_sha256']==w['prior_receipt_sha256']==p['prior_receipt_sha256'],'Prior proof bindings')
        c.need(w['protocol']==p['protocol'] and [s['schedule'] for s in w['updates']]==p['schedule'],'Exact executed protocol and128 schedule')
        schedule(p['schedule'],read(root/'original/plan.json')['schedule'])
        for key,value in dict(completed_updates=128,main_predictions=256,auxiliary_predictions=128,auxiliary_feature_extracts=64,auxiliary_updates=32,zero_gate_passed=True,all825_current_value_hashes_verified=True,base_unchanged=True,rotary_copy_exact=True,sources_unchanged=True,inputs_unchanged=True).items():c.need(w.get(key)==value,'Completed counter/identity '+key)
        outputs=w['output_sha256'];actual={n for n in before if n.startswith('result/') and Path(n).name not in ('metrics.json','memory.jsonl')}
        c.need(actual=={'result/'+n for n in outputs},'Complete worker output inventory')
        for n,d in outputs.items():c.need(c.sha(file(r,n))==d,'Retained output '+n)
        for pattern,expected_names in [('main-*.safetensors',{f'main-{i:04d}-{a}.safetensors' for i in range(1,129) for a in ('closed','interact')}),('auxiliary-*.safetensors',{f'auxiliary-{i:04d}-{a}-{ctx}.safetensors' for i in range(1,129,4) for a in ('closed','open') for ctx in ('positive','negative')}),('gradients-after-clip-*.safetensors',{f'gradients-after-clip-{i:04d}.safetensors' for i in range(1,129)}),('cuda-rng-*.safetensors',{f'cuda-rng-{i:04d}.safetensors' for i in range(0,129,16)})]:
            c.need({q.name for q in r.glob(pattern)}==expected_names,'Exact expected retained set '+pattern)
        catalog=read(HERE/'expected-weights.json');key='repo/experiments/wan22_native/cuda_reference/expected-weights.json'
        c.need(c.sha(HERE/'expected-weights.json')==p['source_sha256'][key],'Original825 catalog pin')
        expected={n:dict(shape=v['shape'],dtype='float32',sha256=v['original_sha256']) for n,v in catalog['tensors'].items()}
        c.need(len(expected)==825 and read(r/'core-before.json')==read(r/'core-after.json')==expected,'All825 original current-value records unchanged')
        loaded=read(r/'weight-load.json');c.need(loaded['tensor_count']==825 and loaded['parameter_count']==4999787712 and loaded['parameter_bytes']==19999150848 and loaded['convert_model_dtype'] is False and loaded['all_shards_verified'] is True and loaded['cuda_copy_exact'] is True and set(loaded['tensors'])==set(expected),'Exact native load metadata')
        for n,e in expected.items():
            q=loaded['tensors'][n];c.need(q['shape']==e['shape'] and q['source_sha256']==q['loaded_sha256']==e['sha256'] and q['original_dtype']==q['loaded_dtype']=='float32' and q['cuda_copy_exact'] is True,'Exact FP32 CUDA copy record '+n)
        windows={a:c.arrays(root/('cache/result/'+a+'.safetensors'),('target','observation','commands')) for a in ARMS}
        positive=c.arrays(root/'original/positive.safetensors')['context'];negative=c.arrays(root/'original/negative.safetensors')['context']
        c.fp32(positive,(25,4096));c.fp32(negative,(126,4096));initial=c.arrays(root/'original/initial-adapter.safetensors')
        initial_rng=c.arrays(root/'original/initial-cpu-rng.safetensors')['rng']
        c.need(w['input_identity']==dict(windows={a:{k:c.tensor_sha(v) for k,v in vals.items()} for a,vals in windows.items()},positive=c.tensor_sha(positive),negative=c.tensor_sha(negative),initial={n:c.tensor_sha(v) for n,v in initial.items()},initial_cpu_rng=c.tensor_sha(initial_rng)),'Exact original input tensor identity')
        observation=None
        for arm,window in windows.items():
            c.fp32(window['target']);c.fp32(window['observation'],(1,48,1,44,78));c.fp32(window['commands'],(1,16,6))
            c.need(np.array_equal(window['target'][:,:,:1],window['observation']),'Raw independent target prefix exact')
            if observation is not None:c.need(np.array_equal(observation,window['observation']),'All six shared observations')
            observation=window['observation'];motion,branch=arm.split('_');expected_commands=np.zeros((1,16,6),np.float32);expected_commands[0,:,3]={'stationary':0.,'left':math.pi/120,'right':-math.pi/120}[motion];expected_commands[0,0,5]=float(branch=='interact')
            c.need(np.array_equal(expected_commands,window['commands']),'Exact six-arm destination commands')
        parity=[]
        for index,arm in enumerate(ARMS[:2]):
            a=c.arrays(r/f'parity-{arm}-native.safetensors',['velocity'])['velocity'];b=c.arrays(r/f'parity-{arm}-bridge.safetensors',['velocity'])['velocity'];c.fp32(a);c.fp32(b)
            row=w['parity'][index];c.need(a.tobytes()==b.tobytes() and row['arm']==arm and row['exact_equal'] is True and row['passed'] is True and row['max_absolute']==row['relative_l2']==0 and row['native_sha256']==c.tensor_sha(a)==row['bridged_sha256'],'Both native zero outputs bit-exact')
            parity.append(dict(arm=arm,sha256=c.tensor_sha(a),exact_equal=True))
        c.need(len(w['parity'])==2,'Exactly two parity rows')
        updates=[];rngs={};draws=None
        for i,(row,step) in enumerate(zip(p['schedule'],w['updates'])):
            if i%16==0:draws=c.arrays(root/f'original/draws-{i:04d}-{i+15:04d}.safetensors')
            noise=draws[row['noise_key']];c.fp32(noise);rngs[i+1]=draws[f'rng_after_{i:04d}'].copy()
            if i==0:rngs[0]=draws['rng_initial'].copy()
            c.need(c.tensor_sha(noise)==row['noise_sha256'] and c.tensor_sha(rngs[i+1])==row['rng_after_sha256'],'Recorded noise/RNG bytes, never regenerated')
            c.need(step['live_sequential_forwards']==2 and step['optimizer_updates']==1 and [b['branch'] for b in step['branches']]==['closed','open'],'Two half-loss branches then one update')
            branches=[];pair=[windows[a] for a in row['branches']]
            for j,(window,record) in enumerate(zip(pair,step['branches'])):
                x,t,v,identity=flow(window,noise,row['k']);c.need(record['input_sha256']==identity and record['observed_input_prefix_exact'] is True,'Exact shared draw / branch flow inputs')
                pred=c.arrays(r/f'main-{i+1:04d}-{("closed","interact")[j]}.safetensors',['velocity'])['velocity'];branches.append(c.loss_record(pred,v,record['future_flow_mse']))
                if i==0:
                    par=w['parity'][j];expected_id={k:c.tensor_sha(a) for k,a in window.items()};expected_id.update(noisy=c.tensor_sha(x),times=c.tensor_sha(t),context=c.tensor_sha(positive));c.need(par['input_identity']==expected_id and par['inputs_unchanged'] is True and par['observed_prefix_exact'] is True,'Parity exact input binding')
            c.need(step['paired_mean_future_flow_mse']==sum(b['future_flow_mse'] for b in step['branches'])/2,'Paired half-loss scalar')
            if row['auxiliary']:
                predictions={a+'-'+ctx:c.arrays(r/f'auxiliary-{i+1:04d}-{a}-{ctx}.safetensors',['velocity'])['velocity'] for a in ('closed','open') for ctx in ('positive','negative')}
                aux=c.auxiliary_record(step['auxiliary'],predictions,pair,noise,positive,negative)
            else:
                c.need(step['auxiliary']==dict(enabled=False,**{'lambda':1.},feature_extracts=0,head_predictions=0,weighted_loss=0.),'No auxiliary work outside32 declared updates');aux=None
            c.need(step['total_objective']==step['paired_mean_future_flow_mse']+step['auxiliary']['weighted_loss'],'Main plus original lambda1 auxiliary')
            gradients=c.arrays(r/f'gradients-after-clip-{i+1:04d}.safetensors',initial)
            for n,g in gradients.items():c.fp32(g,initial[n].shape)
            g=c.clipped_gradient_record(gradients,step)
            if i==0:c.need(step['command_gru_gradient_l2']==0,'Zero first upstream gradient')
            if i==1:c.need(step['command_gru_gradient_l2']>0,'Positive second upstream gradient')
            updates.append(dict(update=i+1,motion=row['motion'],branches=branches,auxiliary=aux,gradient=g))
        del draws
        os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
        import torch
        torch.set_num_threads(1);c.need(not torch.cuda.is_initialized(),'No CUDA initialized by audit')
        checkpoint_identity=dict(schema=p['schema'],source_sha256=p['source_sha256'],plan_sha256=PLAN,admission_sha256=admission_sha,prior_receipt_sha256=p['prior_receipt_sha256'],protocol=p['protocol'])
        cps=[];expected_cps={f'checkpoint-{i:04d}' for i in range(0,129,16)}
        c.need({q.name for q in r.glob('checkpoint-*')}==expected_cps,'Exactly nine checkpoint directories')
        for i in range(0,129,16):
            folder=r/f'checkpoint-{i:04d}';m=read(folder/'manifest.json');values=c.arrays(folder/'adapter.safetensors',initial)
            c.need(m['completed_updates']==i and m['identity']==checkpoint_identity and set(m['files'])=={'adapter.safetensors','optimizer-and-rng.pt'} and m['external_core_weights_included'] is False,'Checkpoint identity and exact files')
            c.need(sum(v.size for v in values.values())==947712 and set(m['tensors'])==set(initial),'Original18 adapter tensors')
            for n,v in values.items():c.fp32(v,initial[n].shape);c.need(m['tensors'][n]==dict(shape=list(v.shape),dtype='float32',sha256=c.tensor_sha(v)),'Checkpoint tensor manifest')
            for n,d in m['files'].items():c.need(c.sha(folder/n)==d,'Checkpoint recovery/file hash')
            c.need((folder/'optimizer-and-rng.pt').stat().st_size<32*2**20,'Bounded recovery file')
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):recovery_state=torch.load(folder/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
            finite_tree(recovery_state);c.need(recovery_state['identity']==checkpoint_identity and recovery_state['completed_updates']==i,'Safe recovery source/update identity')
            c.need(np.array_equal(recovery_state['torch_cpu_rng'].numpy(),initial_rng) and np.array_equal(recovery_state['draw_rng_state'].numpy(),rngs[i]),'Original saved CPU/draw RNG recovery')
            opt=recovery_state['optimizer'];c.need(len(opt['param_groups'])==1,'Single optimizer group');group=opt['param_groups'][0]
            c.need(group['lr']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01 and group['params']==list(range(18)),'Original AdamW and18 parameters')
            if i==0:c.need(not opt['state'] and all(np.array_equal(values[n],initial[n]) for n in initial),'Fresh saved zero initialization')
            else:
                c.need(set(opt['state'])==set(range(18)),'All18 AdamW states')
                for j,n in enumerate(m['tensors']):
                    state=opt['state'][j];c.need(float(state['step'])==i and tuple(state['exp_avg'].shape)==values[n].shape and tuple(state['exp_avg_sq'].shape)==values[n].shape,'Optimizer step/moment shapes')
            c.arrays(r/f'cuda-rng-{i:04d}.safetensors',['rng'])
            cps.append(dict(completed_updates=i,manifest_sha256=c.sha(folder/'manifest.json'),adapter_sha256=c.sha(folder/'adapter.safetensors')))
        c.need(not np.count_nonzero(initial['output.weight']) and not np.count_nonzero(initial['output.bias']),'Original zero output adapter')
        pointer=read(r/'last-valid.json');c.need(pointer['directory']=='checkpoint-0128' and pointer['completed_updates']==128 and pointer['manifest_sha256']==cps[-1]['manifest_sha256'],'Last-valid points only to completed128')
        c.need(w['last_checkpoint']==dict(directory='checkpoint-0128',completed_updates=128,manifest_sha256=cps[-1]['manifest_sha256']),'Worker final checkpoint binding')
        resource=[]
        for name in ('parent-memory.jsonl','result/memory.jsonl'):
            rows=[]
            with file(root,name).open() as stream:
                for line in stream:
                    c.need(len(line)<65536,'Bounded memory sample');sample=c.parse(line);resource_sample(sample)
                    rows.append(sample)
            c.need(rows,'Retained resource samples required');resource.append(dict(file=name,samples=len(rows)))
        monitor=read(r/'monitor-terminal.json');c.need(monitor['status']=='complete' and monitor['sample_count']==resource[1]['samples'],'Complete monitored sample count')
        c.need(not torch.cuda.is_initialized(),'Audit remains CPU-only')
        after=inventory(root);c.need(after==before,'All raw bytes unchanged during audit')
        (out/'inventory.json').write_text(json.dumps(before,indent=2)+'\n')
        report.update(status='passed',completed_updates=128,auxiliary_updates=32,main_predictions=256,auxiliary_predictions=128,auxiliary_feature_extracts=64,all825_unchanged=True,final_checkpoint_only=True,
          identity=dict(parent_sha256=c.sha(root/'metrics.json'),result_sha256=c.sha(r/'metrics.json'),terminal_sha256=c.sha(root/'terminal.json'),plan_sha256=PLAN,final_checkpoint_manifest_sha256=cps[-1]['manifest_sha256'],final_checkpoint_sha256=cps[-1]['adapter_sha256'],source_sha256=p['source_sha256']),
          checks=dict(sources=88,prepared_files=19,parity=parity,updates=updates,checkpoints=cps,resources=resource,raw_files_unchanged=len(before),raw_bytes=sum(v['bytes'] for v in before.values())),
          limitations=['Saved artifacts verify recorded forward losses and combined clipped gradient norms, not a model or backward replay.','Separate main and auxiliary gradient paths cannot be reconstructed from combined gradients.','Optimizer recovery states are checked for identities, finite values, steps and dimensions; this audit does not reconstruct every AdamW operation.','No images or visible action control are assessed. New data and placement changed together, so no placement-only advantage is established.'])
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began;(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('run-root','repo','output'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--recovery',type=Path);args=parser.parse_args();r=execute(args.run_root,args.repo,args.output,args.recovery)
    print(json.dumps(dict(status=r['status'],report_sha256=c.sha(args.output/'report.json'),final_checkpoint_sha256=r['identity']['final_checkpoint_sha256'])))
if __name__=='__main__':main()
