# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy audit of fourteen retained predictions; no model imports."""
import argparse,hashlib,json,math,struct
from pathlib import Path,PurePosixPath
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
REPO=BASE/'outputs/open-worldline'
PREPARED=BASE/'work/wan22-action-cuda-diagnostic-prepared-v3'
PROGRAM=BASE/'work/wan22-action-cuda-diagnostic-prep-v1'
PLAN_SHA='c17ad42baf1dcd9898c7fcecb1aa3458113576ad92dd977c9ee7376e7f50c1e6'
SHAPE=(1,48,5,44,78);TIMES=(1,4290);OBS=(1,48,1,44,78)
ARMS=('closed','open');TEXTS=('atrium','native_negative')
LOSS_TOL={'relative':2e-6,'absolute':1e-8}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def tsha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def parse(raw):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'Duplicate JSON key');d[k]=v
        return d
    def bad(value):raise ValueError('Nonfinite JSON constant')
    d=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    require(isinstance(d,dict),'JSON object required');return d


def file(root,name):
    require(isinstance(name,str) and '\\' not in name and not PurePosixPath(name).is_absolute()
            and str(PurePosixPath(name))==name and all(p not in ('','.','..') for p in name.split('/')),'Canonical relative file required')
    p=Path(root)
    require(p.is_dir() and not p.is_symlink(),'Regular input directory required')
    for part in name.split('/'):
        p=p/part;require(not p.is_symlink(),'No symlink inputs')
    require(p.is_file(),'Missing file: '+name);return p


def read_json(path):
    require(path.stat().st_size<=4*2**20,'Bounded JSON required');return parse(path.read_bytes())


def tensors(path,expected):
    """Read exact declared F32/I64 payloads after header and range validation."""
    require(not path.is_symlink() and 10<=path.stat().st_size<=32*2**20,'Bounded regular tensor file required')
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];require(2<=n<=65536 and 8+n<=path.stat().st_size,'Bounded header required')
        h=parse(f.read(n));metadata=h.pop('__metadata__',{})
        require(isinstance(metadata,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in metadata.items()),'String-only metadata required')
        require(set(h)==set(expected),'Exact tensor keys required');intervals=[]
        for name,(shape,dtype) in expected.items():
            row=h[name];require(set(row)=={'shape','dtype','data_offsets'} and row['shape']==list(shape)
                and all(type(x)is int and x>0 for x in row['shape']) and row['dtype']==dtype and dtype in ('F32','I64'),'Tensor shape/type differs')
            a,b=row['data_offsets'];require(type(a)is int and type(b)is int and 0<=a<b and b-a==math.prod(shape)*(4 if dtype=='F32' else 8),'Bad offsets')
            intervals.append((a,b))
        end=0
        for a,b in sorted(intervals):require(a==end,'Gap/overlap in tensor storage');end=b
        require(end==path.stat().st_size-8-n,'Trailing/truncated storage')
        values={}
        for name,(shape,dtype) in expected.items():
            a,b=h[name]['data_offsets'];f.seek(8+n+a);raw=f.read(b-a);require(len(raw)==b-a,'Truncated tensor')
            value=np.frombuffer(raw,dtype='<f4' if dtype=='F32' else '<i8').reshape(shape)
            require(np.isfinite(value).all(),'Nonfinite tensor');values[name]=value
    return values


def exact(a,b,label):require(a.shape==b.shape and a.dtype==b.dtype and tsha(a)==tsha(b),'Exact bytes differ: '+label)


def contrast(a,b,future=True):
    if future:a=a[:,:,1:];b=b[:,:,1:]
    d=a.astype(np.float64)-b.astype(np.float64);norm=float(np.linalg.norm(d.ravel()));den=float(np.linalg.norm(b.astype(np.float64).ravel()))
    return {'exact_equal':tsha(a)==tsha(b),'max_absolute':float(np.abs(d).max()),'rms':float(np.sqrt(np.mean(d*d,dtype=np.float64))),
            'relative_l2':norm/den if den else None,'difference_l2':norm,'reference_l2':den}


def cfg(positive,negative):
    return np.add(negative,np.multiply(np.float32(5),np.subtract(positive,negative,dtype=np.float32),dtype=np.float32),dtype=np.float32)


def losses(prediction,target):
    delta=prediction[:,:,1:].astype(np.float64)-target[:,:,1:].astype(np.float64)
    fp32_delta=np.subtract(prediction[:,:,1:],target[:,:,1:],dtype=np.float32)
    squared=np.multiply(fp32_delta,fp32_delta,dtype=np.float32)
    return {'float64_mse':float(np.mean(delta*delta,dtype=np.float64)),
            'float32_elementwise_float64_mean':float(np.mean(squared,dtype=np.float64)),
            'numpy_float32_mse':float(np.mean(squared,dtype=np.float32))}


def sources():
    return {'reader':sha(Path(__file__)),'diagnostic':sha(PROGRAM/'diagnostic.py'),'diagnostic_test':sha(PROGRAM/'test_diagnostic.py'),
            'expected_weights':sha(REPO/'experiments/wan22_native/cuda_reference/expected-weights.json')}


def prepared_inputs(root):
    root=Path(root);plan_path=file(root,'plan.json');require(sha(plan_path)==PLAN_SHA,'Exact prepared-v3 plan required');plan=read_json(plan_path)
    require(sha(PROGRAM/'diagnostic.py')==plan['source_sha256']['diagnostic']['diagnostic.py'],'Frozen diagnostic source differs')
    catalog_name='experiments/wan22_native/cuda_reference/expected-weights.json'
    require(sha(file(REPO,catalog_name))==plan['source_sha256']['visual']['repository'][catalog_name],
            'Original weight catalog differs from pinned preparation')
    for name,digest in plan['source_sha256']['diagnostic'].items():require(sha(file(root,name+'.txt'))==digest,'Diagnostic snapshot differs')
    visual=root/'visual-inputs';old=read_json(file(visual,'plan.json'));require(sha(visual/'plan.json')==plan['visual_plan_sha256'],'Visual plan differs')
    for group,rows in plan['source_sha256']['visual'].items():
        group='repository' if group=='repository' else 'local'
        for name,digest in rows.items():require(sha(file(visual/'source'/group,name))==digest,'Visual source snapshot differs')
    require(sha(file(root,'checkpoint-0000.safetensors'))==plan['checkpoint_zero']['checkpoint_sha256'],'Zero checkpoint file differs')
    for name,digest in old['artifacts'].items():require(sha(file(visual,name))==digest,'Prepared visual artifact differs')
    require(sha(file(root,'objective-inputs.safetensors'))==plan['objective_input_sha256'],'Objective input file differs')
    require(sha(file(root,'original-training-inputs.safetensors'))==plan['original_training_input_sha256'],'Raw training input file differs')
    cases=tensors(root/'objective-inputs.safetensors',{arm+'_'+kind:(TIMES if kind=='times' else SHAPE,'I64' if kind=='times' else 'F32') for arm in ARMS for kind in ('noisy','times','target')})
    raw=tensors(root/'original-training-inputs.safetensors',{'noise':(SHAPE,'F32'),'closed_target':(SHAPE,'F32'),'open_target':(SHAPE,'F32'),'observation':(OBS,'F32')})
    values=tensors(visual/'sampling-inputs.safetensors',{'initial_noise':(SHAPE[1:],'F32'),'initial_latent':(SHAPE[1:],'F32'),'observation':(OBS,'F32'),'token_times':(TIMES,'I64')})
    contexts=tensors(visual/'contexts.safetensors',{'atrium':((25,4096),'F32'),'native_negative':((126,4096),'F32')})
    commands=tensors(visual/'commands.safetensors',{arm:((1,16,6),'F32') for arm in ARMS})
    require(tsha(raw['noise'])==plan['first_draw']['noise_sha256'] and plan['first_draw']['k']==506,'First draw identity differs')
    exact(raw['observation'],values['observation'],'independent observation')
    expected=values['initial_noise'].copy();expected[:,:1]=values['observation'][0];exact(values['initial_latent'],expected,'pure-noise causal initial input')
    require(np.all(values['token_times'][:,:858]==0) and np.all(values['token_times'][:,858:]==999),'Causal token times differ')
    for arm in ARMS:
        target=raw[arm+'_target'];expected=np.add(np.multiply(np.float32(.494),target,dtype=np.float32),np.multiply(np.float32(.506),raw['noise'],dtype=np.float32),dtype=np.float32);expected[:,:,:1]=raw['observation']
        exact(expected,cases[arm+'_noisy'],'fixed k506 corruption');exact(np.subtract(raw['noise'],target,dtype=np.float32),cases[arm+'_target'],'future flow target')
        require(np.all(cases[arm+'_times'][:,:858]==0) and np.all(cases[arm+'_times'][:,858:]==506),'Objective token times differ')
        expected=np.zeros((1,16,6),np.float32);expected[:,1:,3]=np.float32(math.pi/24);expected[:,0,5]=arm=='open';exact(commands[arm],expected,'original destination commands')
        pinned=old['input_identity']['training']['training_input_identity']['cache']['windows'][arm+'-0000']['tensor_sha256']
        require(tsha(target)==pinned['target'],'Original target tensor differs')
    return plan,old,cases,values,contexts,commands


def audit(root):
    root=Path(root);plan,old,cases,values,contexts,commands=prepared_inputs(root)
    # Canonical local preparation is an additional independent byte anchor.
    prepared_inputs(PREPARED)
    parent=read_json(file(root,'metrics.json'));terminal=read_json(file(root,'terminal.json'));result=root/'result';worker=read_json(file(result,'metrics.json'))
    require(parent.get('status')=='passed' and parent.get('model_execution') is True and parent.get('result_sha256')==sha(result/'metrics.json')
            and parent.get('terminal_sha256')==sha(root/'terminal.json') and parent.get('plan_sha256')==PLAN_SHA,'Completed bound parent required')
    require(terminal.get('status')=='complete' and type(terminal.get('exit_code'))is int and terminal['exit_code']==0 and not terminal.get('cleanup_error')
            and not list(root.rglob('watchdog-stop.json')),'Complete unstopped worker required')
    require(worker.get('status')=='passed' and worker.get('model_execution') is True and worker.get('predictions')==14
            and worker.get('completed_prediction_files')==14 and worker.get('model_frozen') is True and worker.get('plan_sha256')==PLAN_SHA
            and worker.get('source_sha256')==plan['source_sha256'],'Fourteen completed current predictions required')
    inventory={}
    for name,digest in worker['output_sha256'].items():
        p=file(result,name);require(sha(p)==digest,'Retained output hash differs');inventory[name]={'sha256':digest,'bytes':p.stat().st_size}
    catalog=read_json(REPO/'experiments/wan22_native/cuda_reference/expected-weights.json')
    expected={k:{'shape':v['shape'],'dtype':'float32','sha256':v['original_sha256']} for k,v in catalog['tensors'].items()}
    require(len(expected)==825 and read_json(file(result,'core-before.json'))==expected==read_json(file(result,'core-after.json')),'Original 825 before/after records differ')
    load=read_json(file(result,'weight-load.json'));require(load['tensor_count']==825 and set(load['tensors'])==set(expected) and load['cuda_copy_exact'] is True,'Original foundation load record differs')
    for name,row in load['tensors'].items():
        require(row['shape']==expected[name]['shape'] and row['source_sha256']==row['loaded_sha256']==expected[name]['sha256']
                and row['original_dtype']==row['loaded_dtype']=='float32' and row['cuda_copy_exact'] is True,'Original tensor identity differs')
    names=['native-'+text for text in TEXTS]
    for cp in ('0','16'):
        names += ['objective-'+cp+'-'+arm for arm in ARMS]
        names += ['causal-'+cp+'-'+arm+'-'+text for arm in ARMS for text in TEXTS]
    require([r['name'] for r in worker['calls']]==names,'Fourteen call names/order differ')
    expected_files={name+'.safetensors' for name in names}|{'command-delta-'+cp+'-'+label+'.safetensors' for cp in ('0','16') for label in ('positive','negative')}
    require({p.name for p in result.glob('*.safetensors')}==expected_files and expected_files<=set(inventory),'Exact 18 saved tensor files required')
    predictions={};identities=[]
    for name,row in zip(names,worker['calls']):
        v=tensors(file(result,name+'.safetensors'),{'velocity':(SHAPE,'F32')})['velocity'];predictions[name]=v
        if name.startswith('native-'):cp='native';text=name.removeprefix('native-');x=values['initial_latent'][None];t=values['token_times'];a=None
        elif name.startswith('objective-'):_,cp,arm=name.split('-');text='atrium';x=cases[arm+'_noisy'];t=cases[arm+'_times'];a=commands[arm]
        else:_,cp,arm,text=name.split('-',3);x=values['initial_latent'][None];t=values['token_times'];a=commands[arm]
        identity={'name':name,'checkpoint':cp,'input_sha256':tsha(x),'times_sha256':tsha(t),'context_sha256':tsha(contexts[text]),'commands_sha256':None if a is None else tsha(a),'prediction_sha256':tsha(v)}
        require(identity==row,'Recorded call tensor identity differs: '+name);identities.append(identity)
    objective={};parity={};effects={}
    for cp in ('0','16'):
        objective[cp]={}
        for arm in ARMS:
            m=losses(predictions['objective-'+cp+'-'+arm],cases[arm+'_target']);reported=worker['objective'][cp][arm]
            require(isinstance(reported,(int,float)) and math.isfinite(reported) and math.isclose(reported,m['numpy_float32_mse'],rel_tol=LOSS_TOL['relative'],abs_tol=LOSS_TOL['absolute']),'Reported loss inconsistent with retained values')
            objective[cp][arm]={**m,'reported_float32_mse':reported,'reported_minus_float64':reported-m['float64_mse']}
        for arm in ARMS:
            if cp=='0':
                for text in TEXTS:
                    m=contrast(predictions['causal-0-'+arm+'-'+text],predictions['native-'+text],future=False)
                    relative=m['relative_l2'] if m['reference_l2'] else 0. if m['difference_l2']==0 else None
                    require(m['max_absolute']<=1e-6 and relative is not None and relative<=1e-6,'Original zero-adapter parity bound failed')
                    parity[arm+'-'+text]=m
        pos={a:predictions['causal-'+cp+'-'+a+'-atrium'] for a in ARMS};neg={a:predictions['causal-'+cp+'-'+a+'-native_negative'] for a in ARMS}
        dp=np.subtract(pos['open'],pos['closed'],dtype=np.float32);dn=np.subtract(neg['open'],neg['closed'],dtype=np.float32)
        for label,value in [('positive',dp),('negative',dn)]:exact(value,tensors(file(result,'command-delta-'+cp+'-'+label+'.safetensors'),{'velocity':(SHAPE,'F32')})['velocity'],'saved command delta')
        guided={a:cfg(pos[a],neg[a]) for a in ARMS};gd=np.subtract(guided['open'],guided['closed'],dtype=np.float32)
        decomposition=np.subtract(np.multiply(np.float32(5),dp,dtype=np.float32),np.multiply(np.float32(4),dn,dtype=np.float32),dtype=np.float32)
        roundoff=float(np.max(np.abs(np.subtract(gd,decomposition,dtype=np.float32))))
        measured={'positive_command_effect':contrast(pos['open'],pos['closed']), 'negative_command_effect':contrast(neg['open'],neg['closed']),
           'guided_command_effect':contrast(guided['open'],guided['closed']), 'positive_residual_vs_native':contrast(pos['closed'],predictions['native-atrium']),
           'negative_residual_vs_native':contrast(neg['closed'],predictions['native-native_negative'])}
        for label,row in measured.items():
            for key in ('max_absolute','rms','relative_l2'):
                actual=worker['causal'][cp][label][key];expected_value=row[key]
                require(actual is None if expected_value is None else type(actual) in (int,float) and math.isclose(actual,expected_value,rel_tol=1e-10,abs_tol=1e-14),'Reported contrast differs')
        require(worker['causal'][cp]['guided_delta_decomposition_max_roundoff']==roundoff,'Reported FP32 decomposition residual differs')
        p=dp[:,:,1:].astype(np.float64).ravel();n=dn[:,:,1:].astype(np.float64).ravel();g=gd[:,:,1:].astype(np.float64).ravel()
        pn=float(np.linalg.norm(p));nn=float(np.linalg.norm(n));gn=float(np.linalg.norm(g));den=5*pn+4*nn
        effects[cp]={**measured,'positive_negative_command_delta_cosine':float(np.dot(p,n)/(pn*nn)) if pn and nn else None,
            'guided_l2_over_sum_weighted_component_l2':gn/den if den else None,
            'guided_l2_over_5x_positive_l2':gn/(5*pn) if pn else None,
            'guided_delta_decomposition_max_fp32_roundoff_all_latents':roundoff,
            'computed_guided_tensor_sha256':{a:tsha(v) for a,v in guided.items()},
            'guided_values_retained_by_runtime':False,'meaning':'Ratio and cosine describe combination/cancellation, with no causal diagnosis or quality threshold.'}
    return {'schema':'worldline-fourteen-prediction-independent-audit-v1','status':'passed','source_sha256':sources(),
        'prepared_plan_sha256':PLAN_SHA,'parent_sha256':sha(root/'metrics.json'),'worker_sha256':sha(result/'metrics.json'),'terminal_sha256':sha(root/'terminal.json'),
        'inventory':inventory,'prediction_calls':identities,'predictions_verified':14,'command_delta_files_verified':4,'original_foundation_records_verified':825,
        'objective_k506':objective,'relative_objective_improvement_float64':{a:1-objective['16'][a]['float64_mse']/objective['0'][a]['float64_mse'] if objective['0'][a]['float64_mse'] else None for a in ARMS},
        'loss_reduction_comparison_tolerance':LOSS_TOL,'zero_checkpoint_native_parity':parity,'causal_k999':effects,
        'guidance_equation':'Separate FP32 subtract, multiply by 5, then add negative; both commands on both text branches. Guided outputs reconstructed independently, not compared with saved guided tensors because none were retained.',
        'model_execution':False,'cloud_calls':False,'limitations':['Four losses use the same retained k506 draw and branch-specific target-corrupted inputs; these are seen training inputs, not held-out or action-causal evidence.',
        'The k999 panel keeps initial noise/observation/text fixed when changing commands and excludes future targets as conditions.',
        'The audit verifies saved tensors and reported arithmetic, not model execution or backward replay; foundation verification uses retained value-hash records, not original weight values.',
        'No image generation, perceptual-quality measurement, or preset explanation of failed control. Cancellation ratios alone do not establish the failure cause.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Fresh report directory required');a.output.mkdir(parents=True)
    try:report=audit(a.run)
    except BaseException as e:
        (a.output/'failed.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'error':str(e),'reader_sha256':sha(Path(__file__))},indent=2)+'\n');raise
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'predictions':14,'model_execution':False,'report_sha256':sha(a.output/'report.json')}))

if __name__=='__main__':main()
