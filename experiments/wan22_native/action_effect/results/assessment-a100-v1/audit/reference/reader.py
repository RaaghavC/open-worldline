"""Bounded NumPy input/score reader and injected evaluation wiring. No model imports."""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

SHAPE=(1,48,5,44,78)
MANIFEST_SHA='058e8c34570032f9a76a34ed040d89490fc25aede5bcaa9ffe40559c43111c19'
CHECKPOINTS=('zero','old128','new128')
ARMS=('closed','open')
TEXTS=('positive','negative')

def require(ok,label):
    if not ok:raise ValueError(label)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def pairs(rows):
    result={}
    for k,v in rows:
        require(k not in result,'Duplicate JSON key');result[k]=v
    return result

def parse(data):return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def file(root,name,limit):
    require(isinstance(name,str) and name and Path(name).name==name,'A plain relative filename is required')
    root=Path(root).absolute();p=root/name
    require(not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file() and p.stat().st_size<=limit,'Bounded regular file required')
    return p

def read_json(path,limit=262144):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'Bounded JSON file required')
    return parse(p.read_bytes())

def read_tensor_file(path,expected,expected_file_sha):
    """Validate bounds and exact keys before materializing any array."""
    p=Path(path);require(not p.is_symlink() and p.is_file() and p.stat().st_size<=4*1024**2,'Bounded tensor file required')
    require(sha(p)==expected_file_sha,'Tensor file hash differs')
    data=p.read_bytes();require(len(data)>=8,'Missing safetensor header')
    size=struct.unpack('<Q',data[:8])[0];require(2<=size<=65536 and 8+size<=len(data),'Bounded header required')
    header=parse(data[8:8+size]);require(set(header)==set(expected),'Exact tensor keys required')
    ranges=[]
    for name,(shape,dtype) in expected.items():
        row=header[name];require(set(row)=={'dtype','shape','data_offsets'},'Exact tensor header fields required')
        require(row['shape']==list(shape) and row['dtype']==dtype,'Tensor shape or dtype differs')
        off=row['data_offsets'];require(isinstance(off,list) and len(off)==2 and all(type(v)is int for v in off),'Integer tensor offsets required')
        n=int(np.prod(shape))*(4 if dtype=='F32' else 1)
        require(dtype in ('F32','U8') and 0<=off[0]<=off[1] and off[1]-off[0]==n,'Tensor offsets differ from expected bytes')
        ranges.append(tuple(off))
    ordered=sorted(ranges);require(ordered[0][0]==0 and all(a[1]==b[0] for a,b in zip(ordered,ordered[1:])) and 8+size+ordered[-1][1]==len(data),'Payload must be contiguous with no trailing bytes')
    result={}
    for name,(shape,dtype) in expected.items():
        first,last=header[name]['data_offsets'];a=np.frombuffer(data[8+size+first:8+size+last],dtype='<f4' if dtype=='F32' else 'u1').reshape(shape).copy()
        require(np.isfinite(a).all(),'Nonfinite saved tensor');result[name]=a
    return result

def read_heldout(directory,expected_manifest_sha=MANIFEST_SHA):
    root=Path(directory);path=file(root,'manifest.json',262144)
    require(sha(path)==expected_manifest_sha==MANIFEST_SHA,'Canonical held-out manifest differs')
    m=read_json(path)
    require(m['schema']=='worldline-action-effect-heldout-noise-v1' and m['status']=='prepared' and m['evaluation_only']is True and m['training_use_prohibited']is True and m['seed']==2026090801 and m['count']==4 and m['shape']==list(SHAPE) and m['dtype']=='float32','Canonical evaluation-only protocol differs')
    names=[f'noise-{i:04d}.safetensors' for i in range(4)]
    require(m['noise_files']==names and set(m['files'])==set(names)|{'generator-states.safetensors','prepare.py','environment.json','training-comparison.json','generation-status.json'},'Exact packet file inventory required')
    require({p.name for p in root.iterdir()}==set(m['files'])|{'manifest.json'},'Unrecorded packet member')
    noises={}
    for name,row in m['files'].items():
        p=file(root,name,4*1024**2);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Packet file identity differs')
        if name in names:expected={'noise':(SHAPE,'F32')}
        elif name=='generator-states.safetensors':expected={k:((5056,),'U8') for k in ('initial','after_0000','after_0001','after_0002','after_0003')}
        else:continue
        values=read_tensor_file(p,expected,row['sha256'])
        require(set(values)==set(row['tensors']),'Tensor inventory differs')
        for key,a in values.items():require(row['tensors'][key]=={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':tensor_sha(a)},'Saved array identity differs')
        if name in names:noises[name]=values['noise']
    env=read_json(root/'environment.json');comparison=read_json(root/'training-comparison.json')
    require(env['seed']==m['seed'] and env['device']=='cpu' and env['global_cpu_rng_unchanged']is True and env['cuda_initialized']is False,'Generation environment differs')
    require(comparison['status']=='passed' and comparison['full_tensor_comparisons']==512 and comparison['future_only_comparisons']==512 and comparison['equal_training_pairs']==[] and comparison['equal_future_pairs']==[] and len(comparison['training_noise_records'])==128,'Training separation record differs')
    for i,a in enumerate(noises.values()):
        for b in list(noises.values())[:i]:require(not np.array_equal(a[:,:,1:],b[:,:,1:]),'Repeated evaluation future noise')
    return m,noises

def verify_training_disjoint(directory,training_directory):
    """Independent reread of saved NumPy bytes, never regenerate a Gaussian."""
    m,noises=read_heldout(directory);comparison=read_json(Path(directory)/'training-comparison.json')
    from safetensors import safe_open
    count=0
    for name,digest in m['training_shard_sha256'].items():
        p=file(training_directory,name,64*1024**2);require(sha(p)==digest,'Original training shard differs')
        records=[r for r in comparison['training_noise_records'] if r['shard']==name]
        require(len(records)==16,'Sixteen noise records per original shard required')
        with safe_open(p,framework='np') as f:
            for row in records:
                a=f.get_tensor(row['key']);require(a.shape==SHAPE and a.dtype==np.float32 and tensor_sha(a)==row['sha256'],'Original training array differs')
                for b in noises.values():require(not np.array_equal(a,b) and not np.array_equal(a[:,:,1:],b[:,:,1:]),'Held-out noise overlaps training')
                count+=1
    require(count==128,'All128 original noises required')
    return {'status':'passed','training_arrays':128,'full_comparisons':512,'future_comparisons':512,'noises_regenerated':0}

def finite(a,shape=SHAPE):
    require(isinstance(a,np.ndarray) and a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite FP32 array of declared shape required')

def magnitude(a):
    x=a[:,:,1:].astype(np.float64)
    return {'rms':float(np.sqrt(np.mean(x*x))),'mean_absolute':float(np.mean(np.abs(x))),'maximum_absolute':float(np.max(np.abs(x)))}

def score(predictions,target_difference):
    require(set(predictions)=={a+'-'+t for a in ARMS for t in TEXTS},'Four named predictions required')
    for value in predictions.values():finite(value)
    finite(target_difference)
    guided={}
    for arm in ARMS:
        p,n=predictions[arm+'-positive'],predictions[arm+'-negative']
        guided[arm]=np.add(n,np.multiply(np.float32(5),np.subtract(p,n,dtype=np.float32),dtype=np.float32),dtype=np.float32)
        finite(guided[arm])
    d=np.negative(np.subtract(guided['open'],guided['closed'],dtype=np.float32));finite(d)
    x=d[:,:,1:].astype(np.float64);y=target_difference[:,:,1:].astype(np.float64)
    denominator=float(np.mean(y*y));require(denominator>0,'A nonzero recorded target contrast is required')
    mse=float(np.mean((x-y)**2));norm_product=float(np.linalg.norm(x)*np.linalg.norm(y))
    return {'future_contrast_mse':mse,'target_difference_mean_square':denominator,'normalized_contrast_mse':mse/denominator,
        'cosine':float(np.sum(x*y))/norm_product if norm_product else None,
        'predicted_clean_difference_sha256':tensor_sha(d),
        'magnitudes':{k:magnitude(v) for k,v in {**predictions,**{a+'-guided':v for a,v in guided.items()},'predicted-clean-difference':d,'target-difference':target_difference}.items()},
        'reduction':'FP64 scoring after native-order FP32 CFG5 and s1 endpoint difference; observed latent excluded',
        'quality_claim':False}

def evaluate_matrix(noises,observation,commands,contexts,extract,predict,retain,check=lambda:None):
    """Wiring-only callback driver. Parent must admit exact three checkpoints.

    extract(name,text,x,times,context) runs the original frozen core. It must not
    depend on an adapter. predict(checkpoint,features,commands,observation) applies
    one exact adapter through the same bridge owner and native head. No target is
    passed here. All three checkpoints share each of eight feature extractions.
    """
    require(list(noises)==[f'noise-{i:04d}.safetensors' for i in range(4)],'Fixed four-noise order required')
    finite(observation,(1,48,1,44,78));require(set(commands)==set(ARMS) and set(contexts)==set(TEXTS),'Exact condition names required')
    for arm in ARMS:finite(commands[arm],(1,16,6))
    expected=np.zeros((1,16,6),np.float32);expected[0,0,5]=1
    require(np.array_equal(commands['open']-commands['closed'],expected),'Only the first interaction pulse may differ')
    for text,length in (('positive',25),('negative',126)):finite(contexts[text],(length,4096))
    records=[];extracts=0
    for i,(name,noise) in enumerate(noises.items()):
        finite(noise);x=noise.copy();x[:,:,:1]=observation
        times=np.full((1,4290),999,np.int64);times[:,:858]=0
        features={}
        try:
            for text in TEXTS:
                check();features[text]=extract(name,text,x.copy(),times.copy(),contexts[text].copy());extracts+=1
            for cp in CHECKPOINTS:
                for arm in ARMS:
                    for text in TEXTS:
                        check();v=predict(cp,features[text],commands[arm].copy(),observation.copy());finite(v)
                        label=f'heldout-{i:04d}-{cp}-{arm}-{text}'
                        retain(label,{'velocity':v.copy()})
                        records.append({'name':label,'checkpoint':cp,'noise':name,'arm':arm,'text':text,
                            'input_sha256':tensor_sha(x),'time_sha256':tensor_sha(times),'context_sha256':tensor_sha(contexts[text]),
                            'commands_sha256':tensor_sha(commands[arm]),'prediction_sha256':tensor_sha(v)})
        finally:features.clear()
    require(extracts==8 and len(records)==48,'Exactly8 feature extracts and48 retained head outputs required')
    return {'feature_extracts':extracts,'head_predictions':48,'calls':records,'target_conditioning':False,
        'checkpoint_admission_checked_here':False,'meaning':'Injected callback protocol only; no model/loader/guard implementation or completed-training admission is provided.'}


def read_saved_scores(directory,file_sha256,calls,target_difference):
    """Read the exact48 retained velocity files, in fixed12 groups of four."""
    names=[f'heldout-{i:04d}-{cp}-{arm}-{text}' for i in range(4) for cp in CHECKPOINTS for arm in ARMS for text in TEXTS]
    require([r['name'] for r in calls]==names and set(file_sha256)=={n+'.safetensors' for n in names},'Exact held-out raw prediction inventory required')
    result=[]
    for start in range(0,48,4):
        rows=calls[start:start+4];values={}
        for row in rows:
            name=row['name']+'.safetensors';path=file(directory,name,4*1024**2)
            a=read_tensor_file(path,{'velocity':(SHAPE,'F32')},file_sha256[name])['velocity']
            require(tensor_sha(a)==row['prediction_sha256'],'Retained prediction array differs')
            values[row['arm']+'-'+row['text']]=a
        result.append({'noise_index':start//12,'checkpoint':CHECKPOINTS[(start%12)//4],**score(values,target_difference)})
    return result
