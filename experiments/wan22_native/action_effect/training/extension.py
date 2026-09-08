"""Exact saved-draw continuation and read-only diagnostic admission checks."""
import importlib.util,json,math,sys
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from experiments.wan22_native.action_training import objective
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
HERE=Path(__file__).resolve().parent


def diagnostic():
    name='_fixed128_diagnostic_reference'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,HERE/'reference_diagnostic/diagnostic.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def references():
    return json.loads((HERE/'reference-identities.json').read_text())


def extend_draws(prefix_schedule,prefix_draws,shape):
    """Copy the measured first16 bytes; generate112 once from their last RNG state."""
    if len(prefix_schedule)!=16 or set(prefix_draws)!={'rng_initial'}|{f'{kind}_{i:04d}' for i in range(16) for kind in ('noise','rng_after')}:
        raise ValueError('Complete16-draw saved prefix required')
    rows=json.loads(json.dumps(prefix_schedule));draws={k:v.clone() for k,v in prefix_draws.items()}
    for i,row in enumerate(rows):
        if row['update']!=i+1 or row['start']!=(0,8,32,49)[i%4] or row['noise_key']!=f'noise_{i:04d}' or row['noise_sha256']!=tensor_sha(draws[row['noise_key']]) or row['rng_after_sha256']!=tensor_sha(draws[f'rng_after_{i:04d}']) or tuple(draws[row['noise_key']].shape)!=shape:
            raise ValueError('Original prefix row or bytes differ')
    generator=torch.Generator(device='cpu');generator.set_state(draws['rng_after_0015'])
    for i in range(16,128):
        k=int(torch.randint(50,951,(1,),generator=generator).item())
        noise=torch.randn(shape,generator=generator,dtype=torch.float32,device='cpu');key=f'noise_{i:04d}'
        draws[key]=noise;draws[f'rng_after_{i:04d}']=generator.get_state().clone();start=(0,8,32,49)[i%4]
        rows.append(dict(update=i+1,start=start,branches=[f'closed-{start:04d}',f'open-{start:04d}'],k=k,sigma=k/1000.,noise_key=key,noise_sha256=tensor_sha(noise),rng_after_sha256=tensor_sha(draws[f'rng_after_{i:04d}'])))
    return rows,draws


def load_prefix(directory):
    root=Path(directory);ref=references()
    paths={'metrics.json':'parent_sha256','plan.json':'plan_sha256','training/metrics.json':'training_sha256','training/checkpoint-0000/manifest.json':'checkpoint_zero_manifest_sha256','training/checkpoint-0000/adapter.safetensors':'checkpoint_zero_sha256','training/fresh-draws.safetensors':'draw_file_sha256','initial-cpu-rng.safetensors':'initial_cpu_rng_file_sha256'}
    for name,key in paths.items():
        p=root/name
        if p.is_symlink() or not p.is_file() or sha(p)!=ref[key]:raise ValueError('Original fixed16 input file differs: '+name)
    initial=load_file(str(root/'training/checkpoint-0000/adapter.safetensors'),device='cpu')
    draws=load_file(str(root/'training/fresh-draws.safetensors'),device='cpu')
    rng=load_file(str(root/'initial-cpu-rng.safetensors'),device='cpu')['rng']
    if {k:tensor_sha(v) for k,v in initial.items()}!={k:v['sha256'] for k,v in ref['checkpoint_zero_tensors'].items()} or torch.count_nonzero(initial['output.weight']) or torch.count_nonzero(initial['output.bias']):raise ValueError('Exact original cp0 required')
    return initial,ref['schedule'],draws,rng


def chunks(draws):
    for first in range(0,128,16):
        yield f'draws-{first:04d}-{first+15:04d}.safetensors',{k:v for k,v in draws.items() if k=='rng_initial' and first==0 or k!='rng_initial' and first<=int(k.rsplit('_',1)[1])<first+16}


def prefix_identity(initial,rows,draws):
    ref=references()
    if rows[:16]!=ref['schedule'] or {k:tensor_sha(v) for k,v in initial.items()}!={k:v['sha256'] for k,v in ref['checkpoint_zero_tensors'].items()}:raise ValueError('Exact original cp0 and first16 schedule required')
    if tensor_sha(draws['rng_initial'])!=ref['rng_initial_sha256']:raise ValueError('Original initial draw RNG differs')
    for i,row in enumerate(ref['schedule']):
        if tensor_sha(draws[f'noise_{i:04d}'])!=row['noise_sha256'] or tensor_sha(draws[f'rng_after_{i:04d}'])!=row['rng_after_sha256']:raise ValueError('First16 draw bytes differ')


def scalar_gate(objective_rows,positive_effect,guided_effect):
    """Necessary evidence only. Parent still decides whether CFG needs a change."""
    numbers=[objective_rows[cp][arm] for cp in ('0','16') for arm in ('closed','open')]+[positive_effect,guided_effect]
    if any(type(n) not in (int,float) or not math.isfinite(n) or n<0 for n in numbers):raise ValueError('Finite nonnegative measurements required')
    if not all(objective_rows['16'][arm]<objective_rows['0'][arm] for arm in ('closed','open')):raise ValueError('Both fixed own-corruption objectives must improve')
    if positive_effect<=0 or guided_effect<=0:raise ValueError('Finite nonzero positive and guided command responses required')
    return True


def validate_diagnostic(directory,expected_plan_sha):
    """Remeasure retained14 velocities. Never accept a summary without raw evidence."""
    d=diagnostic();root=d.vi.root(directory);prepared,old,values,contexts,commands,cases=d.read_prepared(root)
    if sha(root/'plan.json')!=expected_plan_sha:raise ValueError('Different diagnostic input plan')
    parent=d.vi.read(root/'metrics.json');terminal=d.vi.read(root/'terminal.json');result=d.vi.read(root/'result/metrics.json')
    if parent.get('status')!='passed' or parent.get('model_execution') is not True or parent.get('result_sha256')!=sha(root/'result/metrics.json') or parent.get('terminal_sha256')!=sha(root/'terminal.json') or parent.get('plan_sha256')!=expected_plan_sha or terminal.get('status')!='complete' or type(terminal.get('exit_code')) is not int or terminal['exit_code']!=0 or terminal.get('cleanup_error') is not None or list(root.rglob('watchdog-stop.json')):
        raise ValueError('Actual diagnostic parent must complete without a watchdog stop')
    if result.get('status')!='passed' or result.get('predictions')!=14 or result.get('model_frozen') is not True or result.get('plan_sha256')!=expected_plan_sha or result.get('source_sha256')!=prepared['source_sha256'] or result.get('limits')!=d.limits('pair'):raise ValueError('Completed current14-call worker required')
    d.vi.checked_outputs(root/'result',result['output_sha256'])
    before=d.vi.read(root/'result/core-before.json');after=d.vi.read(root/'result/core-after.json')
    loaded=d.vi.read(root/'result/weight-load.json')
    if before!=after or before!=d.probe._original_weights(loaded) or before!=old['input_identity']['training']['core_records']:raise ValueError('Frozen825 original foundation records must agree')
    shape=(1,48,5,44,78);spec={'velocity':(shape,'F32')};raw={}
    for call in result['calls']:
        name=call['name']
        if name in raw:raise ValueError('Repeated diagnostic prediction')
        path=root/'result'/(name+'.safetensors');raw[name]=d.vi.pe.tensors(path,spec,result['output_sha256'][name+'.safetensors'])['velocity']
        if tensor_sha(raw[name])!=call['prediction_sha256']:raise ValueError('Prediction identity differs')
    expected={'native-'+text for text in ('atrium','native_negative')}|{'objective-'+cp+'-'+arm for cp in ('0','16') for arm in ('closed','open')}|{'causal-'+cp+'-'+arm+'-'+text for cp in ('0','16') for arm in ('closed','open') for text in ('atrium','native_negative')}
    if set(raw)!=expected or len(result['calls'])!=14:raise ValueError('All14 raw velocities required')
    def native(x,t,c):return raw['native-'+('atrium' if torch.equal(c,contexts['atrium']) else 'native_negative')].clone()
    def adapt(cp,x,t,c,a):
        arm='open' if a[0,0,5] else 'closed';text='atrium' if torch.equal(c,contexts['atrium']) else 'native_negative'
        return raw['objective-'+cp+'-'+arm if int(t[0,-1])==506 else 'causal-'+cp+'-'+arm+'-'+text].clone()
    recomputed=d.evaluate(native,adapt,cases,values,contexts,commands,lambda *_:None,lambda:None)
    if recomputed['calls']!=result['calls']:raise ValueError('Per-call input/command/context binding differs')
    effect=recomputed['causal']['16'];scalar_gate(recomputed['objective'],effect['positive_command_effect']['rms'],effect['guided_command_effect']['rms'])
    cp0=prepared['checkpoint_zero']['checkpoint_sha256'];cp16=old['artifacts']['adapter.safetensors']
    if cp0!=references()['checkpoint_zero_sha256'] or cp16!=references()['checkpoint16_sha256']:raise ValueError('Diagnostic must use original cp0/cp16')
    positive=raw['causal-16-open-atrium']-raw['causal-16-closed-atrium'];negative=raw['causal-16-open-native_negative']-raw['causal-16-closed-native_negative']
    a=5*positive[:,:,1:].double();b=-4*negative[:,:,1:].double();den=float(a.norm()*b.norm())
    return {'parent_sha256':sha(root/'metrics.json'),'result_sha256':sha(root/'result/metrics.json'),'terminal_sha256':sha(root/'terminal.json'),'plan_sha256':expected_plan_sha,'source_sha256':prepared['source_sha256'],'objective':recomputed['objective'],'relative_objective_improvement':recomputed['relative_objective_improvement'],'causal':effect,'weighted_cfg_contribution_cosine':float((a*b).sum())/den if den else None,'guided_to_positive_rms_ratio':effect['guided_command_effect']['rms']/effect['positive_command_effect']['rms'],'necessary_numeric_gate_passed':True,'automatic_training_admission':False}


def diagnostic_identity(record):
    # Exact admission binds retained bytes, not host-specific floating reductions.
    return {key:record[key] for key in ('parent_sha256','result_sha256','terminal_sha256','plan_sha256','source_sha256','necessary_numeric_gate_passed','automatic_training_admission')}
