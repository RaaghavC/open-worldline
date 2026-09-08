"""Independent effect128 retained-artifact audit; no model/backward/CUDA/cloud execution."""
from pathlib import Path
import argparse,json,math,os,sys,time,hashlib
import checks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','runner-source','recovery','run-root','probe-root','cache-root','diagnostic-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    REPO=args.repo.resolve();PREP=args.runner_source.resolve();RECOVERY=args.recovery.resolve()
    ROOT=args.run_root.resolve();TRAIN=ROOT/'training';OUT=args.output.absolute();HERE=Path(__file__).resolve().parent
    if OUT.exists() or OUT.resolve().is_relative_to(ROOT) or OUT.resolve().is_relative_to(REPO) or OUT.resolve().is_relative_to(RECOVERY):
        parser.error('Output must be fresh and outside the run, recovery and repository')
    OUT.mkdir(parents=True,exist_ok=False);began=time.monotonic();before_inventory={}
    sha=checks.sha;ahash=checks.tensor_sha;need=checks.need;norm=checks.norm;arrays=checks.arrays
    def read(p):
        p=Path(p);need(p.is_file()and not p.is_symlink()and p.stat().st_size<=16*2**20,'Bounded regular JSON');return checks.parse(p.read_bytes())
    def inventory():
        need(not any(p.is_symlink()for p in ROOT.rglob('*')),'No recovered symlinks')
        return {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(ROOT.rglob('*'))if p.is_file()}
    report={'schema':'worldline-action-effect128-actual-independent-v1','status':'running','run':str(ROOT),
            'model_execution':False,'gpu_replay':False,'backward_replay':False,'audit_source_sha256':sha(__file__),'checks_source_sha256':sha(HERE/'checks.py'),'checks':{}}
    for name in ('audit.py','checks.py','pins.json'):
        if(HERE/name).is_file():(OUT/name).write_bytes((HERE/name).read_bytes())
    torch=None
    try:
        pins=read(HERE/'pins.json');need(pins['status']=='frozen','Final source freeze required')
        for group,rows in pins['source_sha256'].items():
            base=REPO if group=='repository' else PREP
            for name,digest in rows.items():need(sha(base/name)==digest,'Frozen source dependency '+name)
        need(sha(HERE/'reference/fixed128-audit.py')==pins['original_auditor_sha256'],'Preserved original auditor')
        before_inventory=inventory()
        os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
        sys.path.insert(0,str(REPO));sys.path.insert(0,str(PREP));os.chdir(REPO)
        import numpy as np
        import torch
        import runner
        import extension
        torch.set_num_threads(1)
        need(not torch.cuda.is_initialized(),'Audit must not initialize CUDA')
        recovered=read(RECOVERY/'recovery-verified.json');need(recovered['status']=='verified','complete recovery gate')
        need(ROOT.is_relative_to(RECOVERY/'recovered'),'run is inside the verified recovery')
        need(sha(RECOVERY/'index.json')==recovered['index_sha256'],'verified original recovery index')
        recovery_index=read(RECOVERY/'index.json');relative_run=ROOT.relative_to(RECOVERY/'recovered').as_posix()
        for name,row in before_inventory.items():need(recovery_index['files'][relative_run+'/'+name]==row,'exact recovered member: '+name)
        need(not any(p.is_symlink() for p in ROOT.rglob('*')),'no recovered symlink indirection')
        parent=read(ROOT/'metrics.json');worker=read(ROOT/'worker/metrics.json');training=read(TRAIN/'metrics.json')
        terminal=read(ROOT/'terminal.json');monitor=read(ROOT/'worker/monitor-terminal.json')
        plan,windows,initial,draws,context,rng=runner.read_prepared(ROOT,args.probe_root,args.cache_root,REPO/'experiments/wan_adapter/text_cache/native-results')
        diagnostic=extension.validate_diagnostic(args.diagnostic_root,plan['diagnostic_plan_sha256'])
        admission=read(ROOT/'executed-admission.json');decision=runner.admission(ROOT/'executed-admission.json',ROOT,plan,diagnostic)
        need(all(x['status']=='passed'and x['completed_updates']==128 for x in [parent,worker,training]),'complete 128-update stages')
        need(terminal['status']==monitor['status']=='complete'and terminal['exit_code']==0 and terminal['cleanup_error']is None,'complete supervisor and monitor')
        need(not list(ROOT.rglob('watchdog-stop.json'))and not list(ROOT.rglob('*cleanup-error.json')),'no stopped/failed cleanup')
        need(parent['worker_metrics_sha256']==sha(ROOT/'worker/metrics.json')and parent['terminal_sha256']==sha(ROOT/'terminal.json'),'parent output hashes')
        need(worker['training_metrics_sha256']==sha(TRAIN/'metrics.json'),'training metrics hash')
        need(parent['plan_sha256']==worker['plan_sha256']==sha(ROOT/'plan.json'),'actual plan identity')
        need(parent['admission']==worker['admission']==decision,'exact admitted run')
        need(worker['source_sha256']==plan['source_sha256'],'source/runtime identity')
        hardware_comparison=extension.diagnostic().visual.require_hardware(worker['hardware'],plan['input_identity']['cache']['hardware'])
        need(hardware_comparison==worker['hardware_comparison'],'declared hardware compatibility only')
        expected_identity={'schema':runner.SCHEMA,'plan_sha256':sha(ROOT/'plan.json'),'admission':decision,
            'prepared_draw_files':{name:row['sha256'] for name,row in plan['input_identity']['artifacts'].items() if name.startswith('draws-')},
            'diagnostic_result_sha256':diagnostic['result_sha256']}
        need(training['identity']==expected_identity,'training source/input/diagnostic identity')
        need(extension.diagnostic_identity(worker['diagnostic_evidence'])==extension.diagnostic_identity(diagnostic),'actual diagnostic evidence identity')
        for x in [parent,worker,terminal,monitor]:need(x['limits']==runner.LIMITS and 0<x['elapsed_seconds']<900,'unchanged internal resource/time limit')
        need(training['quality_assessed']is False and training['image_generation']is False and plan['warm_start']is False,'scope and fresh reset')
        need(training['training_forwards']==256 and training['bridge_predictions']==258 and training['native_reference_predictions']==2,'native/bridge/training counts')
        active=checks.schedule_counts(training,plan['schedule'])
        need(plan['source_sha256']==pins['source_sha256'] and sha(ROOT/'plan.json')==pins['prepared_plan_sha256'],'frozen effect source and prepared plan')
        need(sha(ROOT/'cpu-report.json')==pins['cpu_report_sha256'],'final frozen CPU gate')
        need(training['base_unchanged']is True and worker['base_unchanged']is True and training['inputs_unchanged']is True,'unchanged base/inputs')
        need(training['schedule']==plan['schedule']and [r['start']for r in plan['schedule']]==[0,8,32,49]*32,'fixed 128 draw order')
        for group,mapping in plan['source_sha256'].items():
            for name,digest in mapping.items():
                need(sha(ROOT/'source'/group/name)==digest,'retained source '+name)
        cpu=read(ROOT/'cpu-report.json')
        need(cpu['status']=='passed'and cpu['tests']>=5 and cpu['source_sha256']==plan['source_sha256'],'source-bound reviewed CPU result')
        need(all(cpu[key]==0 for key in ('failures','errors','skipped')),'no CPU review failure/skip')
        need(admission['cpu_report_sha256']==plan['cpu_report_sha256']==sha(ROOT/'cpu-report.json'),'CPU review admission')
        ref=extension.references()
        need(plan['schedule'][:16]==ref['schedule'],'original first16 schedule exact')
        need(sha(ROOT/'draws-0000-0015.safetensors')==ref['draw_file_sha256'],'original first16 saved file exact')
        need(sha(ROOT/'initial-cpu-rng.safetensors')==ref['initial_cpu_rng_file_sha256'],'original global CPU RNG exact')
        need(ahash(draws['rng_initial'].numpy())==ref['rng_initial_sha256'],'original private draw RNG exact')
        for i,row in enumerate(plan['schedule']):
            need(ahash(draws[row['noise_key']].numpy())==row['noise_sha256'] and ahash(draws[f'rng_after_{i:04d}'].numpy())==row['rng_after_sha256'],'saved chronological draw identities')
        original_plan=read(PREP/'original128-plan.json')
        need(sha(PREP/'original128-plan.json')==checks.ORIGINAL_PLAN_SHA,'pinned original128 full plan')
        need(plan['schedule']==original_plan['schedule'],'all128 original main schedule rows exact')
        for name,row in original_plan['input_identity']['artifacts'].items():
            need(plan['input_identity']['artifacts'][name]==row and sha(ROOT/name)==row['sha256'],'all original128 input file bytes exact: '+name)
        report['checks']['saved_draws']={'passed':True,'saved_noises':128,'regenerated_noises':0,'all128_original_draw_files_and_schedule_exact':True,'original128_plan_sha256':checks.ORIGINAL_PLAN_SHA,'first16_file_sha256':ref['draw_file_sha256'],'first16_schedule_exact':True,'initial_cp0_exact':True}
        expected_worker_outputs={p.relative_to(ROOT).as_posix() for folder in (ROOT/'worker',TRAIN) for p in folder.rglob('*') if p.is_file() and p!=ROOT/'worker/metrics.json' and p.name!='memory.jsonl'}
        need(set(worker['output_sha256'])==expected_worker_outputs,'complete worker output hash coverage')
        for name,digest in worker['output_sha256'].items():need(sha(runner.evidence.relative_file(ROOT,name))==digest,'worker output '+name)
        for name,digest in training['raw_files'].items():need(sha(runner.evidence.relative_file(TRAIN,name))==digest,'raw prediction/gradient '+name)
        catalog=read(REPO/'experiments/wan22_native/cuda_reference/expected-weights.json')['tensors']
        original=read(TRAIN/'core-before.json');after=read(TRAIN/'core-after.json');loaded=read(ROOT/'worker/weight-load.json')
        need(original==after==runner.probe._original_weights(loaded)and len(original)==825,'all frozen source values')
        for name,row in catalog.items():need(original[name]=={'shape':row['shape'],'dtype':'float32','sha256':row['original_sha256']},'catalog identity '+name)
        report['checks']['source_and_core']={'passed':True,'repository_sources':len(plan['source_sha256']['repository']),'local_sources':len(plan['source_sha256']['local']),'reference_sources':len(plan['source_sha256']['reference']),'original_core_tensors':825,
            'original_parameters':sum(math.prod(x['shape'])for x in original.values()),'core_before_sha256':sha(TRAIN/'core-before.json'),'core_after_sha256':sha(TRAIN/'core-after.json'),
            'reviewed_cpu_report_sha256':sha(ROOT/'cpu-report.json'),'plan_sha256':sha(ROOT/'plan.json')}
        parity=[]
        for row in training['native_comparisons']:
            key=row['window_id'];a=arrays(TRAIN/('parity-'+key+'-native.safetensors'),['native_velocity'])['native_velocity']
            b=arrays(TRAIN/('parity-'+key+'-bridge.safetensors'),['bridged_velocity'])['bridged_velocity']
            need(a.shape==b.shape==(1,48,5,44,78)and a.dtype==b.dtype==np.float32,'raw parity shape/dtype')
            difference=b.astype(np.float64)-a.astype(np.float64);maximum=float(abs(difference).max());n=norm(a);relative=norm(difference)/n if n else (0. if not np.any(difference)else None)
            need(maximum<=1e-6 and relative is not None and relative<=1e-6 and row['passed']is True,'unchanged native-zero dual gate')
            need(row['max_absolute']==maximum and math.isclose(row['relative_l2'],relative,rel_tol=1e-12,abs_tol=1e-15) and row['tolerances']=={'max_absolute':1e-6,'relative_l2':1e-6},'saved parity arithmetic')
            need(row['native_sha256']==ahash(a)and row['bridged_sha256']==ahash(b),'parity tensor hashes')
            parity.append({'window_id':key,'bit_exact_equal':a.tobytes()==b.tobytes(),'max_abs':maximum,'relative_l2':relative})
        need([x['window_id']for x in parity]==['closed-0000','open-0000'],'two ordered parity branches')
        need(training['zero_adapter_gate_passed']is True,'parity completed before optimizer')
        report['checks']['native_zero_parity']={'passed':True,'comparisons':parity,'descriptive_relative_l2_comparison':{'rel_tol':1e-12,'abs_tol':1e-15},'actual_max_and_relative_gate_each':1e-6}
        need(len(training['updates'])==128,'all update records present')
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
                loss_check=checks.loss_record(prediction,velocity,saved['future_flow_mse']);loss=loss_check['future_mse_fp64']
                need(math.isfinite(saved['future_flow_mse'])and saved['future_flow_mse']>=0,'finite recorded future loss')
                branches.append({'window_id':key,'future_flow_mse_fp64':loss,'recorded_future_flow_mse':saved['future_flow_mse'],
                    'absolute_loss_reduction_difference':abs(loss-saved['future_flow_mse']),'prediction_sha256':ahash(prediction)})
            need([x['branch'] for x in update['branches']]==['closed','open'],'exact two ordered branch records')
            need(update['paired_mean_future_flow_mse']==sum(x['future_flow_mse']for x in update['branches'])/2,'half-loss accumulation')
            for field in ['gradient_l2_before_clip','gradient_l2_after_clip','output_gradient_l2']:
                need(math.isfinite(update[field])and update[field]>0,'positive finite '+field)
            updates.append({'update':index,'start':row['start'],'seconds':update['seconds'],'branches':branches})
        need(training['updates'][0]['command_gru_gradient_l2']==0 and training['updates'][1]['command_gru_gradient_l2']>0,'first-zero then second-positive recurrent gradient')
        report['checks']['velocities_and_objective']={'passed':True,'predictions':256,'future_only_loss_recomputations':256,'updates':updates,
            'record_consistency_tolerance':checks.LOSS_TOLERANCE,'scope':'Independent FP64 scoring; narrow recorded-loss consistency only, not a quality threshold or backward replay.'}
        positive=arrays(TRAIN/'positive.safetensors',['context'])['context'];negative=arrays(TRAIN/'negative.safetensors',['context'])['context']
        need(positive.shape==(25,4096) and negative.shape==(126,4096),'fixed genuine text shapes')
        need(np.array_equal(positive,context.numpy()),'same saved positive context')
        need(ahash(negative)==plan['input_identity']['negative_text']['negative_tensor_sha256'],'same genuine negative context')
        need(np.array_equal(negative,arrays(ROOT/'negative.safetensors',['context'])['context']),'prepared negative context exact')
        start0=[{k:v.numpy()for k,v in windows[name].items()}for name in ('closed-0000','open-0000')]
        saved_difference=arrays(TRAIN/'auxiliary-target-difference.safetensors',['difference'])['difference']
        need(np.array_equal(saved_difference,start0[1]['target']-start0[0]['target']),'unchanged raw target difference')
        expected_aux={f'auxiliary-{i:04d}-{arm}-{text}.safetensors'for i in active for arm in ('closed','open')for text in ('positive','negative')}
        need({p.name for p in TRAIN.glob('auxiliary-[0-9]*.safetensors')}==expected_aux,'exact128 auxiliary prediction files')
        auxiliary=[]
        for index in active:
            row=plan['schedule'][index-1];record=training['updates'][index-1]['auxiliary'];noise=draws[row['noise_key']].numpy()
            predictions={arm+'-'+text:arrays(TRAIN/f'auxiliary-{index:04d}-{arm}-{text}.safetensors',['velocity'])['velocity']for arm in ('closed','open')for text in ('positive','negative')}
            for value in predictions.values():checks.fp32(value)
            result=checks.auxiliary_record(record,predictions,start0,noise,positive,negative)
            auxiliary.append({'update':index,**result})
        report['checks']['auxiliary']={'passed':True,'updates':32,'feature_extracts_recorded':64,'raw_head_predictions_verified':128,'records':auxiliary,
            'arithmetic':'Separate FP32 G=N+5*(P-N), then -(Gopen-Gclosed); raw open-minus-closed target, future four latents only.',
            'endpoint_scale':1.0,'literal_solver_sigma_claim':False,'record_consistency_tolerance':checks.LOSS_TOLERANCE,
            'parameter_gradient_scope':'Only combined post-clip parameter gradients are retained; separate main/auxiliary parameter gradients and frozen feature tensors are not reconstructed.'}
        checkpoints=[];equations=[];gradient_records=[];previous_values=None;previous_state=None
        need({p.name for p in TRAIN.glob('checkpoint-*')}=={f'checkpoint-{i:04d}' for i in range(0,129,16)},'exact nine completed checkpoint directories')
        need({p.name for p in TRAIN.glob('gradients-after-clip-*.safetensors')}=={f'gradients-after-clip-{i:04d}.safetensors' for i in range(1,129)},'exact 128 saved gradient bundles')
        need({p.name for p in TRAIN.glob('prediction-*.safetensors')}=={f'prediction-{i:04d}-{arm}.safetensors' for i in range(1,129) for arm in ('closed','open')},'exact 256 raw training predictions')
        for step in range(0,129,16):
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
                need(sha(directory/'adapter.safetensors')==ref['checkpoint_zero_sha256'],'original checkpoint0 exact file')
                need(not np.count_nonzero(values['output.weight'])and not np.count_nonzero(values['output.bias']),'zero initial adapter')
            else:
                need(set(optimizer['state'])==set(range(18)),'all gradient/moment states')
                # Start from each exact saved checkpoint. Compute 16 FP64 AdamW
                # equations using saved post-clip gradients, without GPU replay.
                predicted={name:previous_values[name].astype(np.float64) for name in names}
                moments={name:(np.zeros_like(predicted[name]) if step==16 else previous_state[i]['exp_avg'].numpy().astype(np.float64)) for i,name in enumerate(names)}
                variances={name:(np.zeros_like(predicted[name]) if step==16 else previous_state[i]['exp_avg_sq'].numpy().astype(np.float64)) for i,name in enumerate(names)}
                for update in range(step-15,step+1):
                    gradients=arrays(TRAIN/f'gradients-after-clip-{update:04d}.safetensors',names)
                    combined_check=checks.clipped_gradient_record(gradients,training['updates'][update-1])
                    gradient_sum=gru_sum=output_sum=0.
                    for name in names:
                        g=gradients[name]
                        need(g.dtype==np.float32 and g.shape==values[name].shape,'saved post-clip gradient shape')
                        g=g.astype(np.float64);size=float(np.sum(g*g));gradient_sum+=size
                        if name.startswith('command_gru.'):gru_sum+=size
                        if name.startswith('output.'):output_sum+=size
                        moments[name]=.9*moments[name]+.1*g
                        variances[name]=.999*variances[name]+.001*g*g
                        predicted[name]*=(1-1e-4*.01)
                        predicted[name]-=(1e-4*moments[name]/(1-.9**update))/(np.sqrt(variances[name]/(1-.999**update))+1e-8)
                        need(np.isfinite(predicted[name]).all(),'finite descriptive AdamW calculation')
                    actual_grad=math.sqrt(gradient_sum);actual_gru=math.sqrt(gru_sum)
                    need(actual_grad>0 and math.isfinite(actual_grad)and output_sum>0,'positive finite saved gradient/output projection')
                    if update==1:need(actual_gru==0.,'first saved GRU gradient zero')
                    if update==2:need(actual_gru>0.,'second saved GRU gradient nonzero')
                    need(math.isfinite(training['updates'][update-1]['command_gru_gradient_l2'])and training['updates'][update-1]['command_gru_gradient_l2']>=0,'finite recorded recurrent gradient')
                    gradient_records.append({'update':update,'saved_post_clip_gradient_l2_fp64':actual_grad,'saved_gru_gradient_l2_fp64':actual_gru,
                        'reported_post_clip_gradient_l2':training['updates'][update-1]['gradient_l2_after_clip'],'combined_gradient_check':combined_check})
                parameter_max=moment_max=variance_max=parameter_sum=0.;element_count=0
                for number,name in enumerate(names):
                    state=optimizer['state'][number];need(set(state)=={'step','exp_avg','exp_avg_sq'}and float(state['step'])==step,'all optimizer step counters')
                    m=state['exp_avg'].numpy();v=state['exp_avg_sq'].numpy()
                    need(m.shape==v.shape==values[name].shape and m.dtype==v.dtype==np.float32 and np.isfinite(m).all()and np.isfinite(v).all()and np.all(v>=0),'finite moment tensors')
                    moment_max=max(moment_max,float(abs(m.astype(np.float64)-moments[name]).max()))
                    variance_max=max(variance_max,float(abs(v.astype(np.float64)-variances[name]).max()))
                    difference=values[name].astype(np.float64)-predicted[name]
                    parameter_max=max(parameter_max,float(abs(difference).max()));parameter_sum+=float(np.sum(difference**2));element_count+=difference.size
                equations.append({'from_checkpoint':step-16,'to_checkpoint':step,'saved_gradients_used':16,
                    'first_moment_equation_max_abs':moment_max,'second_moment_equation_max_abs':variance_max,
                    'parameter_equation_max_abs':parameter_max,'parameter_equation_rmse':math.sqrt(parameter_sum/element_count)})
            checkpoints.append({'update':step,'manifest_sha256':sha(directory/'manifest.json'),'files':manifest['files'],'parameter_count':947712})
            previous_values=values;previous_state=optimizer['state']
        need(read(TRAIN/'last-valid.json')=={'directory':'checkpoint-0128','manifest_sha256':checkpoints[-1]['manifest_sha256'],'completed_updates':128},'last valid checkpoint')
        report['checks']['checkpoints_and_gradients']={'passed':True,'checkpoints':checkpoints,'gradient_bundles':128,'gradient_arrays':2304,
            'gradients':gradient_records,'equation_diagnostics':equations,
            'scope':'FP64 AdamW equations start from each exact 16-update checkpoint and consume all saved post-clip gradients. Residuals at the next checkpoint are descriptive, not a GPU replay or a new acceptance threshold.',
            'safe_loader':'weights_only=True with only the installed TorchVersion string subclass allowed for runtime metadata.'}
        samples=[json.loads(x)for x in (ROOT/'worker/memory.jsonl').read_text().splitlines()];ps=[json.loads(x)for x in (ROOT/'parent-memory.jsonl').read_text().splitlines()]
        need(samples and ps and len(samples)==monitor['sample_count'],'sample counts')
        cap=runner.LIMITS
        for row in samples:need(all(type(row[key])is int and row[key]>=0 for key in ('host_rss_bytes','cuda_reserved_bytes','host_available_bytes','cuda_available_bytes')),'strict finite worker memory fields')
        for row in samples:need(row['host_rss_bytes']<=cap['host_rss_bytes']and row['cuda_reserved_bytes']<=cap['cuda_reserved_bytes']and row['host_available_bytes']>=cap['minimum_host_available_bytes']and row['cuda_available_bytes']>=cap['minimum_cuda_available_bytes'],'worker sampled caps/floors')
        for row in ps:need(all(type(row[key])is int and row[key]>=0 for key in ('combined_rss_bytes','host_available_bytes')),'strict finite parent memory fields')
        for row in ps:need(row['combined_rss_bytes']<=cap['host_rss_bytes']and row['host_available_bytes']>=cap['minimum_host_available_bytes'],'parent sampled caps/floors')
        need(max(x['combined_rss_bytes']for x in ps)==terminal['peak_combined_rss_bytes'],'terminal peak')
        report['checks']['resources']={'passed':True,'parent_seconds':parent['elapsed_seconds'],'worker_seconds':worker['elapsed_seconds'],'training_function_seconds_including_hashes_and_artifacts':training['elapsed_seconds'],
            'worker_samples':len(samples),'parent_samples':len(ps),'peak_cuda_reserved_bytes':max(x['cuda_reserved_bytes']for x in samples),
            'peak_combined_rss_bytes':terminal['peak_combined_rss_bytes'],'minimum_cuda_available_bytes':min(x['cuda_available_bytes']for x in samples),
            'recorded_external_timeout':admission.get('external_process_group_timeout'),'internal_limits':cap}
        report['identity']={'parent_sha256':sha(ROOT/'metrics.json'),'worker_sha256':sha(ROOT/'worker/metrics.json'),
            'training_sha256':sha(TRAIN/'metrics.json'),'terminal_sha256':sha(ROOT/'terminal.json'),'plan_sha256':sha(ROOT/'plan.json'),
            'final_checkpoint_manifest_sha256':sha(TRAIN/'checkpoint-0128/manifest.json'),
            'final_checkpoint_sha256':sha(TRAIN/'checkpoint-0128/adapter.safetensors'),'source_sha256':plan['source_sha256']}
        need(inventory()==before_inventory,'recovered run bytes unchanged')
        report.update(status='passed',completed_updates=128,foundation_values_unchanged=True,final_checkpoint_only=True,auxiliary_updates=32,auxiliary_feature_extracts=64,auxiliary_head_predictions=128,main_predictions=256,recovered_bytes_unchanged=True,recovered_files=len(before_inventory),recovered_bytes=sum(x['bytes']for x in before_inventory.values()),
            recovery_verified_sha256=sha(RECOVERY/'recovery-verified.json'),limitations=['No model or backward replay. Only combined post-clip gradients are retained; source-bound accumulation order and optimizer evidence do not independently reconstruct separate main/auxiliary parameter gradients.',
            '128 paired updates on one development layout do not establish image quality, command following or generalization.',
            'Different draws/windows prevent interpreting the chronological loss series as measured learning improvement.',
            'Resource bounds are verified at saved sampling instants; external timeout configuration is retained admission metadata.'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began
        report['cuda_initialized']=False if torch is None else torch.cuda.is_initialized()
        (OUT/'inventory.json').write_text(json.dumps(before_inventory,indent=2)+'\n');report['inventory_sha256']=sha(OUT/'inventory.json')
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'],'report':str(OUT/'report.json'),'sha256':sha(OUT/'report.json'),'elapsed_seconds':report['elapsed_seconds']}))

if __name__=='__main__':main()
