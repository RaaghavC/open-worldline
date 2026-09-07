#!/usr/bin/env python3
"""Read saved probe values and recompute input/optimizer arithmetic, no model."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from safetensors.torch import load_file


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def need(v,label):
    if not v:raise AssertionError(label)
def finite_tree(value):
    if isinstance(value,torch.Tensor):need(value.device.type=='cpu' and bool(torch.isfinite(value).all()),'Finite CPU recovery tensor')
    elif isinstance(value,float):need(math.isfinite(value),'Finite recovery scalar')
    elif isinstance(value,dict):
        for item in value.values():finite_tree(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_tree(item)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--failed-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);root=a.run;result=root/'result'
    report={'schema':'worldline-original-action-probe-independent-v2','status':'running','audit_source_sha256':sha(__file__),
        'model_inference':False,'gpu_used':False,'optimizer_execution':False,'source_and_input_mutation':False,
        'initial_audit_attempt':'Retained under wan22-action-probe-v2-independent-attempt1. The audit initially expected text key positive; measured train.py stores atrium. Corrected audit schema only; no measured file changed.',
        'scope':'Read-only saved tensors and metadata; CPU noise regeneration, FM input arithmetic and Adam-moment gradient reconstruction. No native foundation values loaded.'}
    try:
        plan=read(root/'plan.json');launch=read(root/'launch.json');terminal=read(root/'terminal.json');m=read(result/'metrics.json')
        need(plan['status']=='complete' and terminal['status']=='complete' and terminal['exit_code']==0 and m['status']=='passed','Completed parent and worker')
        need(plan['terminal_sha256']==sha(root/'terminal.json'),'Terminal identity')
        need(plan['protocol']==m['protocol'] and m['protocol']['mode']=='probe' and m['protocol']['paired_updates']==m['completed_updates']==2,'Prescribed two updates')
        need(not m['generated_images'] and m['quality_evaluation']is False and m['automatic_promotion_to_fixed16']is False,'No quality/promotion claim')
        need(m['admission_evidence']['visual_admission']['scope']=='optimizer-feasibility-only' and m['admission_evidence']['visual_admission']['foundation_visual_status']=='failed','Failed foundation status retained')
        need(m['source_sha256']==plan['source_sha256']==launch['source_sha256'] and len(m['source_sha256'])==42,'42-source identity')
        for name,digest in m['source_sha256'].items():
            need(sha(root/'measured-source'/(name+'.txt'))==sha(result/'measured-source'/(name+'.txt'))==digest,'Parent/worker source '+name)
        need('experiments/wan22_native/action_adapter/pooling.py' in m['source_sha256'],'Pooling helper source bound')
        need(m['input_evidence']==plan['input_evidence']==launch['input_evidence'],'Data/text identity')
        for name,digest in m['output_sha256'].items():need(sha(result/name)==digest,'Output identity '+name)
        before=read(result/'core-before.json');after=read(result/'core-after.json');loader=read(result/'weight-load.json')
        expected={n:{'shape':v['shape'],'dtype':v['loaded_dtype'],'sha256':v['loaded_sha256']}for n,v in loader['tensors'].items()}
        need(before==after==expected and len(before)==825,'All 825 current value records equal loader before/after')
        need(sum(math.prod(row['shape'])for row in before.values())==4_999_787_712,'Foundation parameter count')
        need(m['base_unchanged']is True and m['all825_current_value_hashes_verified']is True,'Frozen value assertions')
        report['frozen_core']={'records':825,'parameters':4999787712,'all_before_after_loader_records_equal':True,
            'before_file_sha256':sha(result/'core-before.json'),'after_file_sha256':sha(result/'core-after.json'),
            'no_full_foundation_tensors_reloaded':True,
            'limit':'These retained actual-value hashes and source-bound frozen/gradient assertions are checked; this audit does not reopen all foundation tensor values.'}
        draws=load_file(str(root/'draws.safetensors'));g=torch.Generator(device='cpu').manual_seed(20260907)
        initial_rng=g.get_state().clone();schedule=m['schedule'];need(schedule==plan['schedule']==launch['schedule'] and len(schedule)==2,'Schedule identity')
        need(sha(root/'draws.safetensors')==plan['draw_file_sha256']==m['draw_file_sha256'],'Noise file hash')
        need(set(draws)=={'noise_0000','noise_0001','rng_after_0000','rng_after_0001'},'Exact draw keys')
        for i,row in enumerate(schedule):
            k=int(torch.randint(50,951,(1,),generator=g));noise=torch.randn((1,48,5,18,32),generator=g,dtype=torch.float32)
            need(k==row['k'] and row['sigma']==k/1000 and row['start']==[0,8][i],'Fresh scheduled k/start')
            need(torch.equal(noise,draws[row['noise_key']]) and tsha(noise)==row['noise_sha256'],'Fresh CPU noise bytes')
            need(torch.equal(g.get_state(),draws[f'rng_after_{i:04d}']) and tsha(g.get_state())==row['rng_after_sha256'],'Private RNG state')
        need(not torch.equal(draws['noise_0000'],draws['noise_0001']),'Two different draws')
        positive=load_file(str(root/'positive.safetensors'))
        need(sha(root/'positive.safetensors')==plan['positive_file_sha256'],'Cached real text file')
        need(set(positive)=={'atrium'},'Positive text only')
        cache_manifest=read(a.cache/'manifest.json');need(sha(a.cache/'manifest.json')==m['input_evidence']['cache_manifest_sha256'],'Bound cache manifest')
        windows={row['id']:row for row in cache_manifest['windows']};branch_checks=[]
        need(len(windows)==8 and cache_manifest['split']=='development' and cache_manifest['independent_layouts']==1,'One-layout development cache')
        for index,(row,update) in enumerate(zip(schedule,m['updates'])):
            need(update['update']==index+1 and update['start']==row['start'] and update['k']==row['k'] and update['noise_sha256']==row['noise_sha256'],'Update schedule')
            need(update['optimizer_updates']==1 and update['live_sequential_forwards']==2,'One update per paired two forwards')
            need(abs(sum(b['future_flow_mse']for b in update['branches'])/2-update['paired_mean_future_flow_mse'])<1e-15,'Paired loss arithmetic')
            for branch,identifier in zip(update['branches'],row['branches']):
                entry=windows[identifier];file=a.cache/entry['file'];need(sha(file)==entry['sha256']==m['input_evidence']['windows'][identifier]['provenance']['file_sha256'],'Window hash')
                tensors=load_file(str(file));need(set(tensors)=={'target','observation','commands'},'No hidden conditioning keys')
                for name,value in tensors.items():
                    need(value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Finite FP32 cache')
                    need(tsha(value)==entry['tensors'][name]['sha256']==m['input_evidence']['windows'][identifier]['tensors'][name],'Cache tensor hash')
                target,obs=tensors['target'],tensors['observation'];need(torch.equal(target[:,:,:1],obs),'Independent observation equals target prefix')
                noise=draws[row['noise_key']];sigma=row['k']/1000.
                noisy=(1-sigma)*target+sigma*noise;noisy[:,:,:1]=obs;velocity=noise-target
                times=torch.full((1,720),row['k'],dtype=torch.int64);times[:,:144]=0
                recomputed={'noisy':noisy,'flow_target':velocity,'token_times':times,'observation':obs,'commands':tensors['commands']}
                need({n:tsha(v)for n,v in recomputed.items()}==branch['input_sha256'],'Recorded branch FM inputs recompute exactly')
                branch_checks.append({'id':identifier,'noise_sha256':tsha(noise),'k':row['k'],'all_five_input_hashes_exact':True})
        report['paired_inputs']={'source_files':42,'snapshot_copies':84,'draws_regenerated_exactly':True,'branch_checks':branch_checks,
            'first_pair_shared_observation':m['updates'][0]['branches'][0]['input_sha256']['observation']==m['updates'][0]['branches'][1]['input_sha256']['observation'],
            'second_pair_commands_identical':m['updates'][1]['branches'][0]['input_sha256']['commands']==m['updates'][1]['branches'][1]['input_sha256']['commands'],
            'different_windows_and_noise_per_update':True,'source_scene_family':'single_atrium_layout_v1','split':'development'}
        bundles=[];identity=None;all_state={};parameter_names=None
        for step in range(3):
            directory=result/f'checkpoint-{step:04d}';manifest=read(directory/'manifest.json');weights=load_file(str(directory/'adapter.safetensors'))
            recovery=torch.load(directory/'optimizer-and-rng.pt',map_location='cpu',weights_only=True);finite_tree(recovery)
            need(manifest['completed_updates']==recovery['completed_updates']==step,'Recovery step')
            need(manifest['identity']==recovery['identity'],'Recovery identity')
            if identity is None:identity=manifest['identity']
            need(manifest['identity']==identity and identity['source_sha256']==m['source_sha256'],'Shared checkpoint provenance')
            need(identity['core_before_sha256']==sha(result/'core-before.json'),'Checkpoint base identity')
            need(identity['positive_tensor_sha256']==tsha(positive['atrium']),'Checkpoint genuine text identity')
            for file,digest in manifest['files'].items():need(sha(directory/file)==digest,'Checkpoint artifact hash')
            need(set(weights)==set(manifest['tensors']) and sum(v.numel()for v in weights.values())==947712,'Exactly adapter parameters')
            for name,value in weights.items():
                rec=manifest['tensors'][name];need(list(value.shape)==rec['shape'] and value.dtype==torch.float32 and bool(torch.isfinite(value).all()) and tsha(value)==rec['sha256'],'Finite checkpoint tensor '+name)
            parameter_names=list(manifest['tensors']);optimizer=recovery['optimizer'];groups=optimizer['param_groups'];need(len(groups)==1,'One adapter optimizer group')
            group=groups[0];ids=group['params'];need(len(ids)==len(parameter_names)==18 and len(set(ids))==18,'Exact optimizer ownership')
            need(group['lr']==1e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01,'Optimizer hyperparameters')
            need(len(optimizer['state'])==(18 if step else 0),'Optimizer state coverage')
            state={}
            for pid,name in zip(ids,parameter_names):
                if step:
                    saved=optimizer['state'][pid];need(int(saved['step'])==step,'Every optimizer step')
                    need(saved['exp_avg'].shape==weights[name].shape==saved['exp_avg_sq'].shape,'Moment shape')
                    need(saved['exp_avg'].dtype==saved['exp_avg_sq'].dtype==torch.float32 and bool((saved['exp_avg_sq']>=0).all()),'Moment dtype/sign')
                    state[name]=saved
            all_state[step]=state
            expected_rng=initial_rng if step==0 else draws[f'rng_after_{step-1:04d}'];need(torch.equal(recovery['draw_rng_state'],expected_rng),'Recovery private draw state')
            bundles.append({'completed_updates':step,'parameter_tensors':18,'optimizer_states':len(state),'all_finite':True,'manifest_sha256':sha(directory/'manifest.json')})
        last=read(result/'last-valid.json');need(last=={'directory':'checkpoint-0002','manifest_sha256':sha(result/'checkpoint-0002/manifest.json'),'completed_updates':2},'Last-valid pointer')
        need(m['final_checkpoint']==m['last_checkpoint'] and m['final_checkpoint']['manifest_sha256']==last['manifest_sha256'],'Final/last-valid report identity')
        initial_weights=load_file(str(result/'checkpoint-0000/adapter.safetensors'))
        need(torch.count_nonzero(initial_weights['output.weight'])==torch.count_nonzero(initial_weights['output.bias'])==0,'Zero output initialization')
        need(sha(result/'checkpoint-0000/adapter.safetensors')==m['initial_adapter_sha256']==sha(a.failed_run/'result/checkpoint-0000/adapter.safetensors'),'Fresh initialization equals failed v1, not warm start')
        need(sha(root/'draws.safetensors')==sha(a.failed_run/'draws.safetensors'),'Draws unchanged from failed v1')
        derived=[]
        for step,update in enumerate(m['updates'],1):
            gradients={}
            for name,state in all_state[step].items():
                prev=torch.zeros_like(state['exp_avg'],dtype=torch.float64) if step==1 else all_state[step-1][name]['exp_avg'].double()
                gradients[name]=(state['exp_avg'].double()-.9*prev)/.1
            def norm(prefix=None):return math.sqrt(sum(float(v.square().sum())for n,v in gradients.items()if prefix is None or n.startswith(prefix)))
            values={'gradient_l2_after_clip':norm(),'command_gru_gradient_l2':norm('command_gru.'),'output_gradient_l2':norm('output.')}
            need(update['gradient_l2_before_clip']==update['gradient_l2_after_clip']<1,'Clipping inactive for both updates')
            for name,value in values.items():need(math.isclose(value,update[name],rel_tol=2e-5,abs_tol=1e-9),'Moment-derived norm '+name)
            derived.append({'update':step,'derived_from_adam_first_moments':values,'relative_tolerance':2e-5,'absolute_tolerance':1e-9,
                'meaning':'Approximate reconstruction through stored FP32 moment arithmetic, not retained raw gradient tensors.'})
        need(m['updates'][0]['command_gru_gradient_l2']==0 and m['updates'][1]['command_gru_gradient_l2']>0,'Second update reaches GRU')
        report['checkpoints']={'bundles':bundles,'last_valid_completed_updates':2,'fresh_seed_matches_failed_v1':True,'recovery_weights_only_load':True,'moment_derived_gradient_checks':derived}
        need(m['elapsed_seconds']<=terminal['elapsed_seconds']<900,'Elapsed guard')
        need(sum(x['seconds']for x in m['timings'])<=m['elapsed_seconds'],'Stage sum')
        need(m['peaks_sampled']['mps_driver_bytes']<18*2**30 and min(x['available_bytes']for x in terminal['samples'])>=2*2**30,'Memory guards')
        report['timings']={'worker_seconds':m['elapsed_seconds'],'parent_seconds':terminal['elapsed_seconds'],'stage_seconds':m['timings'],'peak_driver_bytes':m['peaks_sampled']['mps_driver_bytes']}
        report['interpretation']={'execution':'Two prescribed numerical optimizer updates completed; frozen-base and recovery checks pass.',
            'losses':[x['paired_mean_future_flow_mse']for x in m['updates']],
            'not_a_learning_curve':'The two losses use different start windows (0 and 8), different noise draws and k=506 versus628. Their decrease cannot measure learning improvement.',
            'images_generated':0,'quality_demonstrated':False,'novelty_demonstrated':False,'fixed16_admitted':False}
        report.update(status='passed',source_metrics_sha256=sha(result/'metrics.json'),plan_sha256=sha(root/'plan.json'),terminal_sha256=sha(root/'terminal.json'),findings=[])
    except BaseException as error:report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('status','frozen_core','checkpoints','interpretation')},indent=2))
if __name__=='__main__':main()
