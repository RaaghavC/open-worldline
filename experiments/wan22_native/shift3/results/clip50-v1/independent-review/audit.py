#!/usr/bin/env python3
"""Read-only saved-artifact audit. No model imports, inference or GPU work."""
from pathlib import Path
import argparse,ast,hashlib,json,math,platform
import numpy as np
from PIL import Image
from safetensors.numpy import load_file


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def tensor_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise AssertionError(message)

def schedule(shift):
    # Independently transcribed arithmetic from the retained pinned UniPC source.
    # Its construction has shift=1; set_timesteps applies the requested shift once.
    sigma_max=float(np.float32(1-.001))
    unshifted=np.linspace(sigma_max,0.,51)[:-1]
    shifted=shift*unshifted/(1+(shift-1)*unshifted)
    return (shifted*1000).astype(np.int64),np.r_[shifted,0].astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--published',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out={'schema':'worldline-shift3-independent-artifact-audit-v1','status':'running','audit_source_sha256':sha(__file__),
        'model_inference':False,'gpu_used':False,'read_only_inputs':True,'python':platform.python_version(),'numpy':np.__version__,
        'scope':'Saved artifact identities and arithmetic, plus separately stated inspection of existing PNGs. Does not rerun denoising or prove per-step tensors that were not retained.'}
    try:
        root,old,pub=a.run,a.control,a.published
        metrics=read(root/'metrics.json');control=read(old/'metrics.json');core=read(root/'core/result/metrics.json');decode=read(root/'decode/result/metrics.json')
        oldcore=read(old/'core/result/metrics.json');olddecode=read(old/'decode/result/metrics.json');publication=read(pub/'publication.json')
        require(sha(old/'metrics.json')=='17d52167ef71b804f4de36b9855607410ae3134bb0d4d36cdc948dd6f9bea1bc','Pinned prior control changed')
        require(metrics['status']==core['status']==decode['status']=='passed','Execution incomplete')
        require(not list(root.rglob('watchdog-stop.json')),'Stopped run')
        require(metrics['sampling_configuration']=={'steps':50,'shift':3.,'guidance':5.},'Wrong settings')
        require(control['sampling_configuration']=={'steps':50,'shift':5.,'guidance':5.},'Wrong control settings')
        require(metrics['shift5_control']['metrics_sha256']==sha(old/'metrics.json'),'Control identity')
        stage_terminals={}
        for name,m in [('core',core),('decode',decode)]:
            require(metrics[name+'_metrics_sha256']==sha(root/name/'result/metrics.json'),'Stage metrics hash')
            terminal=read(root/name/'terminal.json');stage_terminals[name]=terminal
            require(terminal['status']=='complete' and terminal['exit_code']==0,'Worker incomplete')
            for file,digest in m['output_sha256'].items():require(sha(root/name/'result'/file)==digest,'Output hash: '+file)
        raw_files={str(f.relative_to(root)):f for f in root.rglob('*') if f.is_file()}
        listed={entry['file']:entry for entry in publication['files']}
        require(len(listed)==len(publication['files'])==publication['file_count']==len(raw_files)==83,'Publication coverage/count')
        require(set(listed)==set(raw_files),'Publication missing or extra measured file')
        byte_total=0
        for name,entry in listed.items():
            raw=raw_files[name];public=pub/name
            require(raw.stat().st_size==public.stat().st_size==entry['bytes'],'Published byte size: '+name)
            require(sha(raw)==sha(public)==entry['sha256'],'Byte-exact published copy: '+name)
            byte_total+=entry['bytes']
        require(byte_total==publication['total_bytes'],'Publication byte sum')
        out['publication']={'files':len(listed),'bytes':byte_total,'all_raw_files_byte_exact':True,'publication_sha256':sha(pub/'publication.json')}
        source_counts={}
        for key,folder in [('source_sha256','measured-source'),('reused_source_sha256','reused-source'),('ablation_source_sha256','ablation-source')]:
            for name,digest in metrics[key].items():require(sha(root/folder/(name+'.txt'))==digest,'Measured source '+name)
            source_counts[key]=len(metrics[key])
            if key!='ablation_source_sha256':require(metrics[key]==control[key],'Shared source changed: '+key)
        source_nodes={}
        for label,file in [('shift3',root/'ablation-source/sample.py.txt'),('native',root/'measured-source/sample_clip.py.txt')]:
            tree=ast.parse(file.read_text());source_nodes[label]={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        require(source_nodes['shift3']['integrate']==source_nodes['native']['integrate'],'Sampling loop differs beyond injected factory')
        out['source_identity']={'snapshot_counts':source_counts,'all_shared_sources_match_control':True,'integrate_ast_exact':True,
            'solver_source_sha256':metrics['reused_source_sha256']['experiments/wan_adapter/native_control/vendor/fm_solvers_unipc.py']}
        values=load_file(str(root/'inputs.safetensors'));oldvalues=load_file(str(old/'inputs.safetensors'))
        require(set(values)==set(oldvalues)=={'initial_noise','initial_latent','observation','token_times','positive','negative'},'Input keys')
        require(sha(root/'inputs.safetensors')==sha(old/'inputs.safetensors')==metrics['input_sha256'],'All-input file identity')
        for key,x in values.items():
            require(np.isfinite(x).all() and np.array_equal(x,oldvalues[key]),'Input differs: '+key)
            require(tensor_sha(x)==metrics['input_evidence']['tensor_sha256'][key],'Input tensor digest: '+key)
        require(values['initial_noise'].shape==(48,5,18,32),'Noise shape')
        require(values['observation'].shape==(1,48,1,18,32),'Observation shape')
        expected_initial=values['initial_noise'].copy();expected_initial[:,:1]=values['observation'][0]
        require(np.array_equal(values['initial_latent'],expected_initial),'Initial prefix-only replacement')
        require(values['token_times'].shape==(1,720) and (values['token_times'][0,:144]==0).all() and (values['token_times'][0,144:]==999).all(),'Initial native token times')
        require(read(root/'core/result/weight-load.json')==read(old/'core/result/weight-load.json'),'Full 825-tensor load evidence differs')
        require(core['runtime_environment']==oldcore['runtime_environment'] and decode['runtime_environment']==olddecode['runtime_environment'],'Runtime environment changed')
        out['input_identity']={'file_sha256':sha(root/'inputs.safetensors'),'all_six_inputs_exact':True,
            'tensor_sha256':{k:tensor_sha(v) for k,v in values.items()},'all_825_weight_load_records_equal':True,'stage_environments_equal':True}
        planned=read(root/'schedule.json');times,sigmas=schedule(3);oldtimes,oldsigmas=schedule(5)
        require(np.array_equal(np.array(planned['timesteps']),times),'Independent 50 timestep arithmetic')
        require(np.array_equal(np.array(planned['sigmas']),sigmas.astype(float)),'Independent 51 sigma arithmetic')
        steps=core['steps'];require(len(steps)==core['completed_steps']==50 and core['completed_core_calls']==100,'Completed count')
        for i,row in enumerate(steps):
            require(row['step']==i+1 and row['model_timestep']==int(times[i]) and row['sigma']==float(sigmas[i]),'Step schedule '+str(i))
            require(row['observed_tokens']==144 and row['observed_token_time']==0 and row['future_token_time']==int(times[i]),'Step token schedule')
            require(row['cfg_calls_prefix_exact']==[True,True] and row['post_step_prefix_exact'] is True,'Prefix declaration')
            parts=[row[k] for k in ('positive_seconds','negative_seconds','input_transfer_seconds','guidance_and_output_transfer_seconds','cpu_solver_seconds')]
            require(all(math.isfinite(v) and v>=0 for v in parts+[row['seconds']]),'Step timing finite/nonnegative')
            require(sum(parts)<=row['seconds']+1e-6,'Step timing sum exceeds interval')
        require(core['all_prefix_checks_passed'] is True,'Prefix aggregate')
        final=load_file(str(root/'core/result/latents.safetensors'))['latent']
        last=load_file(str(root/'core/result/last-latent.safetensors'))['latent']
        require(final.shape==(1,48,5,18,32) and final.dtype==np.float32 and np.isfinite(final).all(),'Final latent')
        require(np.array_equal(final[0,:,:1],values['observation'][0]),'Retained final prefix')
        require(np.array_equal(final,last if last.shape==final.shape else last[None]),'Last/final latent equality')
        out['schedule']={'steps':50,'sigmas':51,'recorded_cfg_calls':100,'all_100_recorded_prefix_boundaries_passed':True,
            'all_50_recorded_post_step_prefix_checks_passed':True,'independent_all_times_and_sigmas_exact':True,
            'first_model_time_both':int(times[0]),'first_sigma_shift3':float(sigmas[0]),'first_sigma_shift5':float(oldsigmas[0]),
            'last_model_time':int(times[-1]),'terminal_sigma':float(sigmas[-1]),'retained_final_prefix_exact':True,
            'limitation':'Intermediate model inputs/velocities were not all saved; their prefix/finiteness checks are source-bound execution records, not independently replayed tensors.'}
        video=load_file(str(root/'decode/result/decoded.safetensors'))['video']
        require(video.shape==(1,3,17,288,512) and video.dtype==np.float32 and np.isfinite(video).all(),'Raw decoded video')
        require(video.min()>=-1 and video.max()<=1,'Raw RGB convention')
        pixels=np.rint(((video[0].transpose(1,2,3,0)+1)/2).clip(0,1)*255).astype(np.uint8)
        frames=root/'decode/result/frames';require(len(list(frames.glob('*.png')))==17,'All PNG frames')
        for i in range(17):require(np.array_equal(np.asarray(Image.open(frames/f'{i:04d}.png')),pixels[i]),'PNG differs from exact raw conversion')
        sheet=np.asarray(Image.open(root/'decode/result/comparison.png'))
        for pos,i in enumerate((0,5,10,16)):
            x,y=(pos%2)*512,(pos//2)*320
            require(np.array_equal(sheet[y+32:y+320,x:x+512],pixels[i]),'Contact sheet resampled or changed frame')
        oldvideo=load_file(str(old/'decode/result/decoded.safetensors'))['video']
        require(np.array_equal(oldvideo[:,:,0],video[:,:,0]),'First reconstructed RGB differs')
        difference=(video[:,:,1:].astype(np.float64)-oldvideo[:,:,1:].astype(np.float64))/2
        out['outputs']={'raw_shape':list(video.shape),'raw_dtype':str(video.dtype),'raw_range':[float(video.min()),float(video.max())],
            'all_17_frames_finite':True,'all_17_pngs_exact_raw_quantization':True,'four_contact_sheet_tiles_exact':True,
            'first_frame_exact_to_shift5':True,'future_rgb_mae_between_shifts':float(np.mean(abs(difference))),
            'raw_file_sha256':sha(root/'decode/result/decoded.safetensors'),'final_latent_file_sha256':sha(root/'core/result/latents.safetensors'),
            'known_initial_reconstruction_frames':1,'generated_future_frames':16,'quality_metric':False}
        gate=metrics['timing_gate'];terms=['measured_core_load_seconds','fifty_measured_pairs_seconds','fifty_conservative_auxiliary_seconds','measured_full_cpu_solver_seconds','measured_complete_decoder_seconds','startup_and_artifact_allowance_seconds','already_elapsed_preflight_seconds']
        require(abs(sum(gate[k]for k in terms)-gate['total_estimate_seconds'])<1e-8,'Timing estimate arithmetic')
        require(abs(900-gate['total_estimate_seconds']-gate['remaining_estimated_headroom_seconds'])<1e-8,'Headroom arithmetic')
        timings={row['stage']:row['seconds'] for m in (core,decode) for row in m['timings']}
        step_seconds=sum(row['seconds']for row in steps)
        require(step_seconds<=timings['native_50_step_sampling']<=core['elapsed_seconds'],'Sampling interval containment')
        require(sum(stage_terminals[k]['elapsed_seconds']for k in stage_terminals)<=metrics['elapsed_seconds']<900,'Sequential parent interval')
        peak_review={}
        for name,m in [('core',core),('decode',decode)]:
            rows=[json.loads(x)for x in (root/name/'result/memory.jsonl').read_text().splitlines()]
            for key,value in m['peaks_sampled'].items():require(max(x[key]for x in rows)==value,'Saved peak arithmetic '+name+' '+key)
            require(m['peaks_sampled']['mps_driver_bytes']<=18*2**30,'Driver cap')
            peak_review[name]=m['peaks_sampled']
        parent_rows=[json.loads(x)for x in (root/'parent-memory.jsonl').read_text().splitlines()]
        require(max(x['combined_rss_bytes']for x in parent_rows)==metrics['sampled_parent_limits']['combined_rss_bytes'],'Parent RSS max')
        require(min(x['available_bytes']for x in parent_rows)==metrics['sampled_parent_limits']['minimum_available_bytes']>=2*2**30,'Available memory floor')
        out['timing_and_memory']={'stages_seconds':timings,'summed_step_intervals_seconds':step_seconds,'parent_seconds':metrics['elapsed_seconds'],
            'summed_child_terminal_seconds':sum(t['elapsed_seconds']for t in stage_terminals.values()),'estimate_seconds':gate['total_estimate_seconds'],
            'sampled_peaks':peak_review,'minimum_parent_available_bytes':metrics['sampled_parent_limits']['minimum_available_bytes'],
            'not_an_interactive_fps_claim':True}
        image_paths=[root/'decode/result/comparison.png',root/'decode/result/frames/0005.png',root/'decode/result/frames/0016.png',old/'decode/result/comparison.png']
        out['visual_inspection']={'status':'failed','images':[{'run':'shift3'if path.is_relative_to(root)else'shift5','file':str(path.relative_to(root if path.is_relative_to(root)else old)),'sha256':sha(path)}for path in image_paths],
            'finding':'Both contact sheets retain a recognizable brown door, but generated frames contain severe colored prismatic patterns, warped wall and floor surfaces, and smeared edges. Original shift3 PNGs 5 and 16 confirm the contact-sheet failure. Shift3 does not resolve this distortion.',
            'limitation':'Inspection of retained frames 0,5,10,16 and two original-size future PNGs, not a blinded quality study or a claim that both entire videos have identical pixels.'}
        out.update(status='passed_artifact_audit_visual_failure_retained',source_metrics_sha256=sha(root/'metrics.json'),control_metrics_sha256=sha(old/'metrics.json'),
            evidence_sha256={str(path.relative_to(root)):sha(path)for path in [root/'metrics.json',root/'schedule.json',root/'core/result/metrics.json',root/'decode/result/metrics.json',root/'core/result/weight-load.json']},findings=[])
    except BaseException as error:
        out.update(status='failed_audit',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k]for k in ('status','publication','schedule','timing_and_memory')},indent=2))
if __name__=='__main__':main()
