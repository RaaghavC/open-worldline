"""Bounded read-only probe audit. No model forward, GPU or provider calls."""
from pathlib import Path
import hashlib,json,math,os,sys,time
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
BASE=Path(__file__).resolve().parent.parent;REPO=BASE/'outputs/open-worldline'
ROOT=BASE/'work/wan22-action-cuda-recovered-probe-v1/recovered/action-results/probe-spatial-v1'
CACHE=ROOT.parent/'cache-spatial-run-v1';OUT=BASE/'work/wan22-action-cuda-probe-actual-audit-v1'
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from safetensors.numpy import load_file
from experiments.wan22_native.action_cuda import probe
from experiments.wan22_native.cuda_reference.guards import LIMITS
torch.set_num_threads(1)
OUT.mkdir(exist_ok=False);start=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def need(ok,label):
    if not ok:raise AssertionError(label)
def inventory():return {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(ROOT.rglob('*'))if p.is_file()}
def norm(a):return float(np.sqrt(np.sum(np.asarray(a,dtype=np.float64)**2)))
def finite_tree(v):
    if isinstance(v,torch.Tensor):need(torch.isfinite(v).all().item(),'finite recovery tensor')
    elif isinstance(v,float):need(math.isfinite(v),'finite recovery scalar')
    elif isinstance(v,dict):
        for x in v.values():finite_tree(x)
    elif isinstance(v,(list,tuple)):
        for x in v:finite_tree(x)
report={'schema':'worldline-action-cuda-probe-actual-independent-v1','status':'running',
        'run':str(ROOT.relative_to(BASE)),'model_execution':False,'gpu_replay':False,
        'audit_source_sha256':sha(__file__),'checks':{}}
before_inventory=inventory();(OUT/'audit.py').write_bytes(Path(__file__).read_bytes())
try:
    parent=read(ROOT/'metrics.json');result=read(ROOT/'result/metrics.json')
    terminal=read(ROOT/'terminal.json');monitor=read(ROOT/'result/monitor-terminal.json')
    plan,windows,initial,draws,context,cpu_rng=probe.read_prepared(ROOT,CACHE)
    need(parent['status']==result['status']=='passed' and terminal['status']==monitor['status']=='complete' and terminal['exit_code']==0,'all terminal states')
    need(not list(ROOT.rglob('watchdog-stop.json')) and not list(ROOT.rglob('*cleanup-error.json')),'no guard/cleanup failure')
    need(all(terminal[k]is None for k in ['error','error_type','cleanup_error','cleanup_error_type']),'terminal errors')
    need(parent['result_metrics_sha256']==sha(ROOT/'result/metrics.json') and parent['terminal_sha256']==sha(ROOT/'terminal.json'),'parent output hashes')
    for record in [parent,result]:
        need(record['source_sha256']==plan['source_sha256'] and record['input_identity']==plan['input_identity'],'exact plan identities')
        need(record['plan_sha256']==sha(ROOT/'plan.json') and record['profile']=='spatial','plan/profile')
        need(record['scope']=='two-update-numerical-feasibility-only' and record['image_generation']is False and record['fixed16_admitted']is False and record['quality_assessed']is False,'numerical-only scope')
        need(record['model_execution']is True,'actual execution')
    for record in [parent,result,terminal,monitor]:
        need(record['limits']==LIMITS and 0<record['elapsed_seconds']<900,'same fixed limit/deadline')
    need(result['completed_updates']==2 and result['native_reference_predictions']==2 and result['bridge_predictions']==6,'prescribed counts')
    need(result['zero_adapter_gate_passed']is True and result['base_unchanged']is True and result['all825_current_value_hashes_verified']is True,'successful numerical gates')
    need(result['inputs_unchanged']is True and result['sources_unchanged']is True and result['automatic_promotion']is False,'no automatic promotion')
    need(result['protocol']==plan['protocol'] and result['schedule']==plan['schedule'],'fixed training settings')
    admitted=read(ROOT/'executed-admission.json')
    need(admitted['scope']=='two-update-numerical-feasibility-only' and admitted['decision']=='admit','explicit admission')
    need(admitted['plan_sha256']==sha(ROOT/'plan.json') and admitted['source_sha256']==plan['source_sha256'] and admitted['input_identity']==plan['input_identity'],'admission exact bytes')
    need(parent['admission']['admission_sha256']==sha(ROOT/'executed-admission.json')==result['admission']['admission_sha256'],'admission file identity')
    for name,digest in result['output_sha256'].items():need(sha(ROOT/'result'/name)==digest,'retained output hash '+name)
    core_before=read(ROOT/'result/core-before.json');core_after=read(ROOT/'result/core-after.json')
    loaded=read(ROOT/'result/weight-load.json');catalog=read(REPO/'experiments/wan22_native/cuda_reference/expected-weights.json')['tensors']
    need(core_before==core_after and len(core_before)==825 and set(core_before)==set(catalog)==set(loaded['tensors']),'complete unchanged original core')
    need(loaded['convert_model_dtype']is False and loaded['all_shards_verified']is True and loaded['cuda_copy_exact']is True,'FP32 shard/copy checks')
    for name,row in catalog.items():
        wanted={'shape':row['shape'],'dtype':'float32','sha256':row['original_sha256']}
        need(core_before[name]==wanted,'original before/after value '+name)
        item=loaded['tensors'][name]
        need(item['shape']==row['shape'] and item['loaded_sha256']==item['source_sha256']==row['original_sha256'],'original loaded value '+name)
        need(item['cuda_copy_exact']is True and item['source_owner_released']is True and item['loaded_dtype']==item['original_dtype']=='float32','loaded storage/owner '+name)
    report['checks']['core']={'passed':True,'tensor_count':825,'parameters':sum(math.prod(v['shape'])for v in core_before.values()),'before_after_exact':True,'before_sha256':sha(ROOT/'result/core-before.json'),'after_sha256':sha(ROOT/'result/core-after.json'),'full_weight_reread':False}
    parity=[]
    for row in result['native_comparisons']:
        key=row['window_id'];a=load_file(str(ROOT/'result'/('parity-'+key+'-native.safetensors')))
        b=load_file(str(ROOT/'result'/('parity-'+key+'-bridge.safetensors')))
        need(set(a)=={'native_velocity'}and set(b)=={'bridged_velocity'},'parity raw keys')
        a=a['native_velocity'];b=b['bridged_velocity']
        need(a.shape==b.shape==(1,48,5,44,78)and a.dtype==b.dtype==np.float32 and np.isfinite(a).all()and np.isfinite(b).all(),'parity shape/finite')
        difference=b.astype(np.float64)-a.astype(np.float64);maximum=float(abs(difference).max());denominator=norm(a);numerator=norm(difference)
        relative=numerator/denominator if denominator else (0. if numerator==0 else None)
        need(maximum<=1e-6 and relative is not None and relative<=1e-6,'fixed dual parity bound')
        need(row['passed']is True and row['max_absolute']==maximum and math.isclose(row['relative_l2'],relative,rel_tol=1e-12,abs_tol=1e-15),'parity recorded values')
        exact=a.tobytes()==b.tobytes();need(row['exact_equal']==exact and row['tolerances']=={'max_absolute':1e-6,'relative_l2':1e-6},'parity equality/tolerances')
        need(thash(a)==row['native_sha256']and thash(b)==row['bridged_sha256'],'parity tensor identities')
        parity.append({'window_id':key,'max_absolute':maximum,'relative_l2':relative,'bit_exact_equal':exact,'native_sha256':thash(a),'bridged_sha256':thash(b)})
    need([p['window_id']for p in parity]==['closed-0000','open-0000'],'two declared comparisons')
    report['checks']['native_zero_parity']={'passed':True,'comparisons':parity}
    inputs=[]
    for step,row in enumerate(plan['schedule']):
        update=result['updates'][step];noise=draws[row['noise_key']].numpy();k=row['k']
        need(update['update']==step+1 and update['start']==row['start']and update['optimizer_updates']==1 and update['live_sequential_forwards']==2,'ordered update')
        for index,key in enumerate(row['branches']):
            window={name:value.numpy()for name,value in windows[key].items()}
            target=window['target'];observation=window['observation'];commands=window['commands']
            noisy=np.float32(1-k/1000.)*target+np.float32(k/1000.)*noise;noisy[:,:,:1]=observation
            velocity=noise-target;times=np.full((1,4290),k,dtype=np.int64);times[:,:858]=0
            identities={n:thash(v)for n,v in {'noisy':noisy,'token_times':times,'flow_target':velocity,'observation':observation,'commands':commands}.items()}
            branch=update['branches'][index]
            need(branch['input_sha256']==identities and branch['observed_input_prefix_exact']is True,'independent FM input identity')
            need(branch['branch']==['closed','open'][index]and math.isfinite(branch['future_flow_mse'])and branch['future_flow_mse']>=0,'branch loss/order')
            if step==0:need(result['native_comparisons'][index]['input_sha256']==identities,'parity uses training draw')
            inputs.append({'update':step+1,'window_id':key,'input_sha256':identities,'future_flow_mse':branch['future_flow_mse']})
        need(update['paired_mean_future_flow_mse']==sum(x['future_flow_mse']for x in update['branches'])/2,'half-loss accumulation record')
        for field in ['gradient_l2_before_clip','gradient_l2_after_clip','output_gradient_l2']:
            need(math.isfinite(update[field])and update[field]>0,'finite positive '+field)
    need(result['updates'][0]['command_gru_gradient_l2']==0 and result['updates'][1]['command_gru_gradient_l2']>0,'first-zero/second-positive GRU')
    report['checks']['paired_inputs_and_recorded_gradients']={'passed':True,'inputs':inputs,'updates':result['updates'],'independent_backward_replay':False}
    checkpoints=[];params=[];recoveries=[]
    for step in range(3):
        directory=ROOT/'result'/f'checkpoint-{step:04d}';manifest=read(directory/'manifest.json')
        need(manifest['completed_updates']==step and manifest['external_core_weights_included']is False,'checkpoint stage/scope')
        need(set(manifest['files'])=={'adapter.safetensors','optimizer-and-rng.pt'},'checkpoint files')
        for name,digest in manifest['files'].items():need(sha(directory/name)==digest,'checkpoint file identity')
        values=load_file(str(directory/'adapter.safetensors'))
        need(set(values)==set(manifest['tensors'])==set(initial),'adapter parameter keys')
        for name,value in values.items():
            need(value.dtype==np.float32 and np.isfinite(value).all(),'finite FP32 adapter')
            need(manifest['tensors'][name]=={'shape':list(value.shape),'dtype':'float32','sha256':thash(value)},'adapter tensor hash')
        need(sum(v.size for v in values.values())==947712,'original adapter count')
        # TorchVersion is the installed Torch string subclass used in runtime metadata.
        # Permit only that one type while retaining the safe weights-only loader.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            recovery=torch.load(directory/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
        finite_tree(recovery)
        need(recovery['identity']==manifest['identity']and recovery['completed_updates']==step,'recovery identity/update')
        identity=manifest['identity']
        need(identity['source_sha256']==plan['source_sha256']and identity['plan_sha256']==sha(ROOT/'plan.json')and identity['input_identity']==plan['input_identity'],'checkpoint input/source binding')
        need(identity['admission_sha256']==sha(ROOT/'executed-admission.json')and identity['protocol']==plan['protocol'],'checkpoint protocol/admission')
        expected_rng=draws['rng_initial'if step==0 else f'rng_after_{step-1:04d}']
        need(torch.equal(recovery['draw_rng_state'],expected_rng),'private draw RNG retained')
        need(torch.equal(recovery['torch_cpu_rng'],cpu_rng),'CPU RNG unchanged after preparation')
        opt=recovery['optimizer'];need(len(opt['param_groups'])==1,'one adapter optimizer group')
        group=opt['param_groups'][0]
        need(group['lr']==1e-4 and tuple(group['betas'])==(.9,.999)and group['eps']==1e-8 and group['weight_decay']==.01,'fixed AdamW values')
        names=list(manifest['tensors']);need(group['params']==list(range(len(names)))and len(names)==18,'ordered optimizer parameter mapping')
        if step==0:
            need(not opt['state'],'fresh optimizer state')
            need(all(np.array_equal(values[n],initial[n].numpy())for n in values),'checkpoint0 equals prepared bytes')
            need(not np.count_nonzero(values['output.weight'])and not np.count_nonzero(values['output.bias']),'zero output initialization')
        else:
            need(set(opt['state'])==set(group['params']),'all adapter moments present')
            for number,name in enumerate(names):
                state=opt['state'][number]
                need(set(state)=={'step','exp_avg','exp_avg_sq'}and float(state['step'])==step,'all AdamW counters')
                need(tuple(state['exp_avg'].shape)==values[name].shape==tuple(state['exp_avg_sq'].shape),'optimizer moment shape')
                need(torch.all(state['exp_avg_sq']>=0).item(),'nonnegative second moment')
        cuda_rng=load_file(str(ROOT/'result'/f'cuda-rng-{step:04d}.safetensors'))
        need(set(cuda_rng)=={'cuda_rng_0'}and cuda_rng['cuda_rng_0'].dtype==np.uint8,'retained CUDA RNG bytes')
        checkpoints.append({'step':step,'manifest_sha256':sha(directory/'manifest.json'),'files':manifest['files'],'parameters':947712,'optimizer_state_tensors':len(opt['state'])*3})
        params.append(values);recoveries.append(recovery)
    descriptive=[]
    for step in [1,2]:
        sums=0.;count=0;maximum=0.;gradient_square=0.;gru_square=0.;moments_max=0.
        names=list(read(ROOT/'result'/f'checkpoint-{step:04d}'/'manifest.json')['tensors'])
        for number,name in enumerate(names):
            state=recoveries[step]['optimizer']['state'][number]
            m=state['exp_avg'].numpy().astype(np.float64);v=state['exp_avg_sq'].numpy().astype(np.float64)
            expected=params[step-1][name].astype(np.float64)*(1-1e-4*.01)-(1e-4*m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            difference=params[step][name].astype(np.float64)-expected
            maximum=max(maximum,float(np.abs(difference).max()));sums+=float(np.sum(difference**2));count+=difference.size
            oldm=0. if step==1 else recoveries[step-1]['optimizer']['state'][number]['exp_avg'].numpy().astype(np.float64)
            oldv=0. if step==1 else recoveries[step-1]['optimizer']['state'][number]['exp_avg_sq'].numpy().astype(np.float64)
            inferred=(m-.9*oldm)/.1;gradient_square+=float(np.sum(inferred**2))
            if name.startswith('command_gru.'):gru_square+=float(np.sum(inferred**2))
            moments_max=max(moments_max,float(np.abs(v-(.999*oldv+.001*inferred**2)).max()))
        descriptive.append({'update':step,'parameter_equation_max_abs':maximum,'parameter_equation_rmse':math.sqrt(sums/count),
            'gradient_l2_inferred_from_first_moments':math.sqrt(gradient_square),'gru_gradient_l2_inferred_from_first_moments':math.sqrt(gru_square),
            'second_moment_recurrence_max_abs':moments_max,'scope':'Descriptive FP64 reconstruction from saved FP32 moments; no GPU replay or new pass threshold.'})
    last=read(ROOT/'result/last-valid.json')
    need(last=={'directory':'checkpoint-0002','manifest_sha256':checkpoints[2]['manifest_sha256'],'completed_updates':2},'last validated checkpoint')
    report['checks']['checkpoints']={'passed':True,'checkpoints':checkpoints,'adamw_equation_diagnostics':descriptive,
        'safe_load_note':'Initial strict weights_only inspection rejected TorchVersion metadata. Audit permits only the installed torch.torch_version.TorchVersion string subclass in weights_only=True. No unsafe pickle load.'}
    samples=[json.loads(x)for x in (ROOT/'result/memory.jsonl').read_text().splitlines()];parent_samples=[json.loads(x)for x in (ROOT/'parent-memory.jsonl').read_text().splitlines()]
    need(len(samples)==monitor['sample_count']and samples and parent_samples,'memory count')
    for row in samples:
        need(row['host_rss_bytes']<=LIMITS['host_rss_bytes']and row['cuda_reserved_bytes']<=LIMITS['cuda_reserved_bytes'],'memory caps')
        need(row['host_available_bytes']>=LIMITS['minimum_host_available_bytes']and row['cuda_available_bytes']>=LIMITS['minimum_cuda_available_bytes'],'memory floors')
    for row in parent_samples:need(row['combined_rss_bytes']<=LIMITS['host_rss_bytes']and row['host_available_bytes']>=LIMITS['minimum_host_available_bytes'],'parent caps')
    need(max(x['combined_rss_bytes']for x in parent_samples)==terminal['peak_combined_rss_bytes'],'sampled peak binding')
    report['checks']['resources']={'passed':True,'worker_samples':len(samples),'parent_samples':len(parent_samples),
        'peak_combined_rss_bytes':terminal['peak_combined_rss_bytes'],'peak_cuda_reserved_bytes':max(x['cuda_reserved_bytes']for x in samples),
        'minimum_cuda_available_bytes':min(x['cuda_available_bytes']for x in samples),'parent_seconds':parent['elapsed_seconds'],
        'worker_seconds':result['elapsed_seconds'],'load_seconds':result['load_seconds'],'before_hash_seconds':result['before_hash_seconds'],'after_hash_seconds':result['after_hash_seconds']}
    need(inventory()==before_inventory,'all recovered evidence unchanged')
    report.update(status='passed',recovered_bytes_unchanged=True,recovered_files=len(before_inventory),
        recovered_bytes=sum(v['bytes']for v in before_inventory.values()),plan_sha256=sha(ROOT/'plan.json'),
        limitations=['Numerical and optimizer feasibility only; no generated-image or action-control evaluation.',
          'No gradients or full forward graph were replayed. Step2 predictions were not separately retained for independent loss recomputation.',
          'Frozen base identity is checked from recorded hashes against the original catalog; no full weight reread.',
          'Resource limits hold at recorded sampled instants.'])
except BaseException as error:
    report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    report['elapsed_seconds']=time.monotonic()-start;report['cuda_initialized']=torch.cuda.is_initialized()
    (OUT/'inventory.json').write_text(json.dumps(before_inventory,indent=2)+'\n');report['inventory_sha256']=sha(OUT/'inventory.json')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'report':str((OUT/'report.json').relative_to(BASE)),
                     'sha256':sha(OUT/'report.json'),'elapsed_seconds':report['elapsed_seconds']}))
