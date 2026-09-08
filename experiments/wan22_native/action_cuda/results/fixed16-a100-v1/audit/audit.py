"""Independent retained-tensor audit; no model forward, CUDA or cloud calls."""
from pathlib import Path
import hashlib,json,math,os,sys,time
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
BASE=Path(__file__).resolve().parent.parent;REPO=BASE/'outputs/open-worldline'
RECOVERY=BASE/'work/wan22-action-cuda-recovered-final-v1'
PARENT=RECOVERY/'recovered/action-results';ROOT=PARENT/'fixed16-spatial-v1';TRAIN=ROOT/'training'
PREP=BASE/'work/wan22-action-cuda-fixed16-prep-v1';OUT=BASE/'work/wan22-action-cuda-fixed16-actual-audit-v1'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(PREP));os.chdir(REPO)
import numpy as np
import torch
from safetensors.numpy import load_file
import runner
torch.set_num_threads(1)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def need(ok,label):
    if not ok:raise AssertionError(label)
def norm(a):return float(np.sqrt(np.sum(np.asarray(a,dtype=np.float64)**2)))
def inventory():return {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(ROOT.rglob('*'))if p.is_file()}
def arrays(path,keys=None):
    values=load_file(str(path))
    if keys is not None:need(set(values)==set(keys),'exact tensor keys '+str(path.name))
    for value in values.values():need(np.isfinite(value).all(),'finite retained tensor')
    return values
OUT.mkdir(exist_ok=False);began=time.monotonic();before_inventory=inventory()
(OUT/'audit.py').write_bytes(Path(__file__).read_bytes())
report={'schema':'worldline-action-cuda-fixed16-actual-independent-v1','status':'running','run':str(ROOT.relative_to(BASE)),
        'model_execution':False,'gpu_replay':False,'audit_source_sha256':sha(__file__),'checks':{}}
try:
    recovered=read(RECOVERY/'recovery-verified.json');need(recovered['status']=='verified','complete recovery gate')
    parent=read(ROOT/'metrics.json');worker=read(ROOT/'worker/metrics.json');training=read(TRAIN/'metrics.json')
    terminal=read(ROOT/'terminal.json');monitor=read(ROOT/'worker/monitor-terminal.json')
    plan,windows,initial,draws,context,rng=runner.read_prepared(ROOT,PARENT/'probe-spatial-v1',PARENT/'cache-spatial-run-v1',REPO/'experiments/wan_adapter/text_cache/native-results')
    admission=read(ROOT/'executed-admission.json');decision=runner.admission(ROOT/'executed-admission.json',ROOT,plan)
    need(all(x['status']=='passed'and x['completed_updates']==16 for x in [parent,worker,training]),'complete sixteen-update stages')
    need(terminal['status']==monitor['status']=='complete'and terminal['exit_code']==0 and terminal['cleanup_error']is None,'complete supervisor and monitor')
    need(not list(ROOT.rglob('watchdog-stop.json'))and not list(ROOT.rglob('*cleanup-error.json')),'no stopped/failed cleanup')
    need(parent['worker_metrics_sha256']==sha(ROOT/'worker/metrics.json')and parent['terminal_sha256']==sha(ROOT/'terminal.json'),'parent output hashes')
    need(worker['training_metrics_sha256']==sha(TRAIN/'metrics.json'),'training metrics hash')
    need(parent['plan_sha256']==worker['plan_sha256']==sha(ROOT/'plan.json'),'actual plan identity')
    need(parent['admission']==worker['admission']==decision,'exact admitted run')
    need(worker['source_sha256']==plan['source_sha256']and worker['hardware']==plan['input_identity']['cache']['hardware'],'source/runtime identity')
    need(training['identity']['source_sha256']==plan['source_sha256']and training['identity']['input_identity']==plan['input_identity'],'training source/input identity')
    for x in [parent,worker,terminal,monitor]:need(x['limits']==runner.LIMITS and 0<x['elapsed_seconds']<900,'unchanged internal resource/time limit')
    need(training['quality_assessed']is False and training['image_generation']is False and plan['warm_start']is False,'scope and fresh reset')
    need(training['training_forwards']==32 and training['bridge_predictions']==34 and training['native_reference_predictions']==2,'native/bridge/training counts')
    need(training['base_unchanged']is True and worker['base_unchanged']is True and training['inputs_unchanged']is True,'unchanged base/inputs')
    need(training['schedule']==plan['schedule']and [r['start']for r in plan['schedule']]==[0,8,32,49]*4,'fixed sixteen draw order')
    for group,mapping in plan['source_sha256'].items():
        for name,digest in mapping.items():
            need(sha(ROOT/'source'/group/name)==digest,'retained source '+name)
            if group=='local':need(sha(PARENT/'fixed16-executed-source'/name)==digest,'executed local source '+name)
    cpu=read(PARENT/'fixed16-executed-source/runner-cpu-report-v2.json')
    need(cpu['status']=='passed'and cpu['tests']==9 and cpu['sources_unchanged']is True and cpu['source_sha256']==plan['source_sha256'],'source-bound reviewed CPU result')
    need(admission['reviewed_cpu_report_sha256']==sha(PARENT/'fixed16-executed-source/runner-cpu-report-v2.json'),'CPU review admission')
    for name,digest in worker['output_sha256'].items():need(sha(ROOT/name)==digest,'worker output '+name)
    for name,digest in training['raw_files'].items():need(sha(TRAIN/name)==digest,'raw prediction/gradient '+name)
    catalog=read(REPO/'experiments/wan22_native/cuda_reference/expected-weights.json')['tensors']
    original=read(TRAIN/'core-before.json');after=read(TRAIN/'core-after.json');loaded=read(ROOT/'worker/weight-load.json')
    need(original==after==runner.probe._original_weights(loaded)and len(original)==825,'all frozen source values')
    for name,row in catalog.items():need(original[name]=={'shape':row['shape'],'dtype':'float32','sha256':row['original_sha256']},'catalog identity '+name)
    report['checks']['source_and_core']={'passed':True,'repository_sources':len(plan['source_sha256']['repository']),'local_sources':4,'original_core_tensors':825,
        'original_parameters':sum(math.prod(x['shape'])for x in original.values()),'core_before_sha256':sha(TRAIN/'core-before.json'),'core_after_sha256':sha(TRAIN/'core-after.json'),
        'reviewed_cpu_report_sha256':sha(PARENT/'fixed16-executed-source/runner-cpu-report-v2.json'),'plan_sha256':sha(ROOT/'plan.json')}
    parity=[]
    for row in training['native_comparisons']:
        key=row['window_id'];a=arrays(TRAIN/('parity-'+key+'-native.safetensors'),['native_velocity'])['native_velocity']
        b=arrays(TRAIN/('parity-'+key+'-bridge.safetensors'),['bridged_velocity'])['bridged_velocity']
        need(a.shape==b.shape==(1,48,5,44,78)and a.dtype==b.dtype==np.float32,'raw parity shape/dtype')
        difference=b.astype(np.float64)-a.astype(np.float64);maximum=float(abs(difference).max());n=norm(a);relative=norm(difference)/n if n else (0. if not np.any(difference)else None)
        need(maximum<=1e-6 and relative is not None and relative<=1e-6 and row['passed']is True,'unchanged native-zero dual gate')
        need(row['max_absolute']==maximum and row['relative_l2']==relative and row['tolerances']=={'max_absolute':1e-6,'relative_l2':1e-6},'saved parity arithmetic')
        need(row['native_sha256']==ahash(a)and row['bridged_sha256']==ahash(b),'parity tensor hashes')
        parity.append({'window_id':key,'bit_exact_equal':a.tobytes()==b.tobytes(),'max_abs':maximum,'relative_l2':relative})
    need([x['window_id']for x in parity]==['closed-0000','open-0000'],'two ordered parity branches')
    need(training['zero_adapter_gate_passed']is True,'parity completed before optimizer')
    report['checks']['native_zero_parity']={'passed':True,'comparisons':parity}
    updates=[]
    for index,row in enumerate(plan['schedule'],1):
        update=training['updates'][index-1];noise=draws[row['noise_key']].numpy();k=row['k'];branches=[]
        need(update['update']==index and update['start']==row['start']and update['live_sequential_forwards']==2 and update['optimizer_updates']==1,'one chronological paired optimizer update')
        for branch,key in zip(['closed','open'],row['branches']):
            values={name:v.numpy()for name,v in windows[key].items()};target=values['target'];observation=values['observation']
            noisy=np.float32(1-k/1000.)*target+np.float32(k/1000.)*noise;noisy[:,:,:1]=observation
            velocity=noise-target;times=np.full((1,4290),k,dtype=np.int64);times[:,:858]=0
            hashes={name:ahash(v)for name,v in {'noisy':noisy,'token_times':times,'flow_target':velocity,'observation':observation,'commands':values['commands']}.items()}
            saved=next(x for x in update['branches']if x['branch']==branch)
            need(saved['input_sha256']==hashes and saved['observed_input_prefix_exact']is True,'exact saved training input')
            prediction=arrays(TRAIN/f'prediction-{index:04d}-{branch}.safetensors',['velocity'])['velocity']
            need(prediction.shape==(1,48,5,44,78)and prediction.dtype==np.float32,'native prediction shape/precision')
            loss=float(np.mean((prediction[:,:,1:].astype(np.float64)-velocity[:,:,1:].astype(np.float64))**2))
            need(math.isfinite(saved['future_flow_mse'])and saved['future_flow_mse']>=0,'finite recorded future loss')
            branches.append({'window_id':key,'future_flow_mse_fp64':loss,'recorded_future_flow_mse':saved['future_flow_mse'],
                'absolute_loss_reduction_difference':abs(loss-saved['future_flow_mse']),'prediction_sha256':ahash(prediction)})
        need(update['paired_mean_future_flow_mse']==sum(x['future_flow_mse']for x in update['branches'])/2,'half-loss accumulation')
        for field in ['gradient_l2_before_clip','gradient_l2_after_clip','output_gradient_l2']:
            need(math.isfinite(update[field])and update[field]>0,'positive finite '+field)
        updates.append({'update':index,'start':row['start'],'seconds':update['seconds'],'branches':branches})
    need(training['updates'][0]['command_gru_gradient_l2']==0 and training['updates'][1]['command_gru_gradient_l2']>0,'first-zero then second-positive recurrent gradient')
    report['checks']['velocities_and_objective']={'passed':True,'predictions':32,'future_only_loss_recomputations':32,'updates':updates,
        'scope':'Independent FP64 scoring of retained predictions; GPU reduction roundoff reported without a new threshold or backward replay.'}
    checkpoints=[];equations=[];previous_values=None;previous_state=None
    for step in range(17):
        directory=TRAIN/f'checkpoint-{step:04d}';manifest=read(directory/'manifest.json');values=arrays(directory/'adapter.safetensors')
        need(manifest['completed_updates']==step and set(manifest['files'])=={'adapter.safetensors','optimizer-and-rng.pt'},'checkpoint stage/files')
        for name,digest in manifest['files'].items():need(sha(directory/name)==digest,'checkpoint file hash')
        need(set(values)==set(initial)==set(manifest['tensors'])and sum(v.size for v in values.values())==947712,'all adapter parameters')
        for name,value in values.items():need(value.dtype==np.float32 and manifest['tensors'][name]=={'shape':list(value.shape),'dtype':'float32','sha256':ahash(value)},'parameter tensor identity')
        need(manifest['identity']==training['identity']and manifest['external_core_weights_included']is False,'checkpoint source/input identity')
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            recovery=torch.load(directory/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
        need(recovery['completed_updates']==step and recovery['identity']==manifest['identity'],'safe recovery identity')
        need(torch.equal(recovery['torch_cpu_rng'],rng)and torch.equal(recovery['draw_rng_state'],draws['rng_initial'if step==0 else f'rng_after_{step-1:04d}']),'canonical RNG records')
        optimizer=recovery['optimizer'];need(len(optimizer['param_groups'])==1,'single adapter group');group=optimizer['param_groups'][0]
        need(group['lr']==1e-4 and group['betas']==(.9,.999)and group['eps']==1e-8 and group['weight_decay']==.01,'fixed AdamW hyperparameters')
        names=list(manifest['tensors']);need(len(names)==18 and group['params']==list(range(18)),'ordered optimizer ownership')
        if step==0:
            need(not optimizer['state']and all(np.array_equal(values[n],initial[n].numpy())for n in values),'fresh checkpoint0')
            need(not np.count_nonzero(values['output.weight'])and not np.count_nonzero(values['output.bias']),'zero initial adapter')
        else:
            gradients=arrays(TRAIN/f'gradients-after-clip-{step:04d}.safetensors',names)
            need(set(optimizer['state'])==set(range(18)),'all gradient/moment states')
            parameter_max=moment_max=variance_max=0.;parameter_sum=gradient_sum=gru_sum=0.;element_count=0
            for number,name in enumerate(names):
                state=optimizer['state'][number];need(set(state)=={'step','exp_avg','exp_avg_sq'}and float(state['step'])==step,'all optimizer step counters')
                gradient=gradients[name];need(gradient.dtype==np.float32 and gradient.shape==values[name].shape,'saved post-clip gradient shape')
                m=state['exp_avg'].numpy();v=state['exp_avg_sq'].numpy()
                need(m.shape==v.shape==gradient.shape and m.dtype==v.dtype==np.float32 and np.isfinite(m).all()and np.isfinite(v).all()and np.all(v>=0),'finite moment tensors')
                oldm=0. if step==1 else previous_state[number]['exp_avg'].numpy().astype(np.float64)
                oldv=0. if step==1 else previous_state[number]['exp_avg_sq'].numpy().astype(np.float64)
                g=gradient.astype(np.float64);md=m.astype(np.float64);vd=v.astype(np.float64)
                moment_max=max(moment_max,float(abs(md-(.9*oldm+.1*g)).max()))
                variance_max=max(variance_max,float(abs(vd-(.999*oldv+.001*g*g)).max()))
                expected=previous_values[name].astype(np.float64)*(1-1e-4*.01)-(1e-4*md/(1-.9**step))/(np.sqrt(vd/(1-.999**step))+1e-8)
                difference=values[name].astype(np.float64)-expected
                parameter_max=max(parameter_max,float(abs(difference).max()));parameter_sum+=float(np.sum(difference**2));element_count+=difference.size
                gradient_sum+=float(np.sum(g*g))
                if name.startswith('command_gru.'):gru_sum+=float(np.sum(g*g))
            actual_grad=math.sqrt(gradient_sum);actual_gru=math.sqrt(gru_sum)
            need(actual_grad>0 and math.isfinite(actual_grad),'all-step positive finite saved gradient')
            if step==1:need(actual_gru==0.,'first saved GRU gradient zero')
            if step==2:need(actual_gru>0.,'second saved GRU gradient nonzero')
            equations.append({'update':step,'saved_post_clip_gradient_l2_fp64':actual_grad,'saved_gru_gradient_l2_fp64':actual_gru,
                'reported_post_clip_gradient_l2':training['updates'][step-1]['gradient_l2_after_clip'],
                'first_moment_equation_max_abs':moment_max,'second_moment_equation_max_abs':variance_max,
                'parameter_equation_max_abs':parameter_max,'parameter_equation_rmse':math.sqrt(parameter_sum/element_count)})
        checkpoints.append({'update':step,'manifest_sha256':sha(directory/'manifest.json'),'files':manifest['files'],'parameter_count':947712})
        previous_values=values;previous_state=optimizer['state']
    need(read(TRAIN/'last-valid.json')=={'directory':'checkpoint-0016','manifest_sha256':checkpoints[-1]['manifest_sha256'],'completed_updates':16},'last valid checkpoint')
    report['checks']['checkpoints_and_gradients']={'passed':True,'checkpoints':checkpoints,'gradient_bundles':16,'gradient_arrays':288,
        'equation_diagnostics':equations,'scope':'FP64 equations use retained FP32 gradients/moments; residuals are descriptive, not a GPU replay or new acceptance threshold.',
        'safe_loader':'weights_only=True with only the installed TorchVersion string subclass allowed for runtime metadata.'}
    samples=[json.loads(x)for x in (ROOT/'worker/memory.jsonl').read_text().splitlines()];ps=[json.loads(x)for x in (ROOT/'parent-memory.jsonl').read_text().splitlines()]
    need(samples and ps and len(samples)==monitor['sample_count'],'sample counts')
    cap=runner.LIMITS
    for row in samples:need(row['host_rss_bytes']<=cap['host_rss_bytes']and row['cuda_reserved_bytes']<=cap['cuda_reserved_bytes']and row['host_available_bytes']>=cap['minimum_host_available_bytes']and row['cuda_available_bytes']>=cap['minimum_cuda_available_bytes'],'worker sampled caps/floors')
    for row in ps:need(row['combined_rss_bytes']<=cap['host_rss_bytes']and row['host_available_bytes']>=cap['minimum_host_available_bytes'],'parent sampled caps/floors')
    need(max(x['combined_rss_bytes']for x in ps)==terminal['peak_combined_rss_bytes'],'terminal peak')
    report['checks']['resources']={'passed':True,'parent_seconds':parent['elapsed_seconds'],'worker_seconds':worker['elapsed_seconds'],'training_function_seconds_including_hashes_and_artifacts':training['elapsed_seconds'],
        'worker_samples':len(samples),'parent_samples':len(ps),'peak_cuda_reserved_bytes':max(x['cuda_reserved_bytes']for x in samples),
        'peak_combined_rss_bytes':terminal['peak_combined_rss_bytes'],'minimum_cuda_available_bytes':min(x['cuda_available_bytes']for x in samples),
        'recorded_external_timeout':admission['external_process_group_timeout'],'internal_limits':cap}
    need(inventory()==before_inventory,'recovered run bytes unchanged')
    report.update(status='passed',recovered_bytes_unchanged=True,recovered_files=len(before_inventory),recovered_bytes=sum(x['bytes']for x in before_inventory.values()),
        recovery_verified_sha256=sha(RECOVERY/'recovery-verified.json'),limitations=['No model or backward pass replay; checks use retained tensors and runtime records.',
        'Sixteen paired updates on one development layout do not establish image quality, command following or generalization.',
        'Different draws/windows prevent interpreting the chronological loss series as measured learning improvement.',
        'Resource bounds are verified at saved sampling instants; external timeout configuration is retained admission metadata.'])
except BaseException as error:
    report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    report['elapsed_seconds']=time.monotonic()-began;report['cuda_initialized']=torch.cuda.is_initialized()
    (OUT/'inventory.json').write_text(json.dumps(before_inventory,indent=2)+'\n');report['inventory_sha256']=sha(OUT/'inventory.json')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'report':str((OUT/'report.json').relative_to(BASE)),'sha256':sha(OUT/'report.json'),'elapsed_seconds':report['elapsed_seconds']}))
