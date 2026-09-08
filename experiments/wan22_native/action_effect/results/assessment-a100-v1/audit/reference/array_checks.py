"""Independent CPU artifact arithmetic; no Torch, model, CUDA or network imports."""
from pathlib import Path
import hashlib,json,math,struct
import numpy as np
SHAPE=(1,48,5,44,78)
AUXILIARY_UPDATES=tuple(range(1,129,4))
LOSS_TOLERANCE={'relative':2e-6,'absolute':1e-8}
GRADIENT_TOLERANCE={'relative':2e-6,'absolute':1e-8}
ORIGINAL_PLAN_SHA='94c2df86e6d62de09fc9fa469b2fca7e04fd7f5fe894b4182349fd36e6af833a'

def need(value,message):
    if not value:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def tensor_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def parse(data):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'Duplicate JSON key');result[key]=value
        return result
    def bad(_):raise ValueError('Nonfinite JSON constant')
    value=json.loads(data,object_pairs_hook=pairs,parse_constant=bad)
    need(isinstance(value,dict),'JSON object required');return value

def arrays(path,keys=None,maximum=64*2**20):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and 10<=path.stat().st_size<=maximum,'Bounded regular tensor file required')
    with path.open('rb') as f:
        count=struct.unpack('<Q',f.read(8))[0]
        need(2<=count<=65536 and 8+count<=path.stat().st_size,'Bounded tensor header required')
        header=parse(f.read(count));metadata=header.pop('__metadata__',{})
        need(isinstance(metadata,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in metadata.items()),'String-only metadata required')
        need(bool(header) and (keys is None or set(header)==set(keys)),'Exact tensor key set required')
        intervals=[];dtypes={'F32':np.dtype('<f4'),'I64':np.dtype('<i8'),'U8':np.dtype('u1')}
        for name,row in header.items():
            need(isinstance(row,dict) and set(row)=={'shape','dtype','data_offsets'},'Exact tensor descriptor required')
            shape=row['shape'];need(isinstance(shape,list) and all(type(n)is int and n>0 for n in shape),'Positive integer shape required')
            need(row['dtype'] in dtypes,'Unsupported tensor dtype');offsets=row['data_offsets']
            need(isinstance(offsets,list) and len(offsets)==2 and all(type(n)is int for n in offsets),'Integer offsets required')
            a,b=offsets;need(0<=a<b and b-a==math.prod(shape)*dtypes[row['dtype']].itemsize,'Tensor payload size mismatch')
            intervals.append((a,b,name))
        end=0
        for a,b,name in sorted(intervals):need(a==end,'Tensor gap or overlap');end=b
        need(8+count+end==path.stat().st_size,'Trailing or missing tensor bytes')
        result={}
        for a,b,name in sorted(intervals):
            row=header[name];f.seek(8+count+a);raw=f.read(b-a);need(len(raw)==b-a,'Truncated tensor')
            value=np.frombuffer(raw,dtype=dtypes[row['dtype']]).copy().reshape(row['shape'])
            need(np.isfinite(value).all(),'Nonfinite retained tensor');result[name]=value
        return result

def finite_number(value,*,nonnegative=True):
    return type(value) in (int,float) and math.isfinite(value) and (not nonnegative or value>=0)

def fp32(value,shape=SHAPE):
    need(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all(),'Exact finite FP32 tensor required')
    return value

def norm(value):return float(np.sqrt(np.sum(np.asarray(value,dtype=np.float64)**2)))

def loss_record(prediction,target,recorded):
    fp32(prediction,target.shape);fp32(target,target.shape)
    need(prediction.ndim==5 and prediction.shape[0]==1 and prediction.shape[2]==5,'B1 five-latent loss required')
    residual=prediction[:,:,1:].astype(np.float64)-target[:,:,1:].astype(np.float64)
    value=float(np.mean(residual*residual))
    need(finite_number(recorded),'Finite recorded loss required')
    need(math.isclose(value,recorded,rel_tol=LOSS_TOLERANCE['relative'],abs_tol=LOSS_TOLERANCE['absolute']),'Recorded loss disagrees with retained predictions')
    return {'future_mse_fp64':value,'recorded_future_mse':recorded,'absolute_reduction_difference':abs(value-recorded)}

def guided(positive,negative):
    fp32(positive,negative.shape);fp32(negative,negative.shape)
    # Separate FP32 operations match the frozen eager equation, with no fused expression.
    difference=np.subtract(positive,negative,dtype=np.float32)
    scaled=np.multiply(np.float32(5),difference,dtype=np.float32)
    result=np.add(negative,scaled,dtype=np.float32)
    need(np.isfinite(result).all(),'Nonfinite reconstructed guidance');return result

def auxiliary_conditions(windows,noise,positive,negative):
    need(len(windows)==2,'Exactly two ordered windows required')
    closed,opened=windows;shape=noise.shape;fp32(noise,shape)
    need(len(shape)==5 and shape[0]==1 and shape[2]==5 and shape[-2]%2==shape[-1]%2==0,'Native B1 five-latent grid required')
    for w in windows:
        need(set(w)=={'target','observation','commands'},'Exact RGB/action cache tensor interface required')
        fp32(w['target'],shape);fp32(w['observation'],(1,shape[1],1,*shape[-2:]));fp32(w['commands'],(1,16,6))
    need(np.array_equal(closed['observation'],opened['observation']),'Common independent observation required')
    delta=opened['commands']-closed['commands'];expected=np.zeros((1,16,6),np.float32);expected[0,0,5]=1
    need(np.array_equal(delta,expected),'Only initial wait/interact command may change')
    x=noise.copy();x[:,:,:1]=closed['observation'];prefix=(shape[-2]//2)*(shape[-1]//2)
    times=np.full((1,5*prefix),999,np.int64);times[:,:prefix]=0
    target=np.subtract(opened['target'],closed['target'],dtype=np.float32)
    identities={'pure_noise_sha256':tensor_sha(noise),'input_sha256':tensor_sha(x),'times_sha256':tensor_sha(times),'observation_sha256':tensor_sha(closed['observation']),'target_difference_sha256':tensor_sha(target),'commands_sha256':{a:tensor_sha(w['commands'])for a,w in zip(('closed','open'),windows)},'context_sha256':{'positive':tensor_sha(positive),'negative':tensor_sha(negative)}}
    return x,times,target,identities

def auxiliary_record(record,predictions,windows,noise,positive,negative):
    x,times,target,identity=auxiliary_conditions(windows,noise,positive,negative)
    need(record.get('enabled') is True and record.get('lambda')==1. and record.get('endpoint_scale')==1.,'Fixed enabled auxiliary loss required')
    need(record.get('feature_extracts')==2 and record.get('head_predictions')==4,'Two frozen features and four heads required')
    need(record.get('prefix_excluded') is True and record.get('target_conditioning') is False,'Future-only loss and target isolation required')
    need(record.get('input_identity')==identity,'Exact auxiliary input identities required')
    expected={a+'-'+t for a in ('closed','open') for t in ('positive','negative')}
    need(set(predictions)==expected,'Exactly four auxiliary paths required')
    for value in predictions.values():fp32(value,noise.shape)
    closed=guided(predictions['closed-positive'],predictions['closed-negative']);opened=guided(predictions['open-positive'],predictions['open-negative'])
    predicted=np.negative(np.subtract(opened,closed,dtype=np.float32),dtype=np.float32)
    loss=loss_record(predicted,target,record.get('future_clean_difference_mse'))
    need(record.get('weighted_loss')==record['future_clean_difference_mse'],'Fixed lambda1 weighted auxiliary loss')
    return {**loss,'input_identity':identity,'closed_guided_sha256':tensor_sha(closed),'open_guided_sha256':tensor_sha(opened),'predicted_endpoint_difference_sha256':tensor_sha(predicted),'target_difference_sha256':tensor_sha(target),'ideal_endpoint_scale':1.0,'literal_solver_sigma_claim':False}

def schedule_counts(training,schedule):
    need(len(schedule)==len(training['updates'])==128,'All128 ordered updates required')
    active=[]
    for index,(row,step) in enumerate(zip(schedule,training['updates']),1):
        need(row['start']==(0,8,32,49)[(index-1)%4] and step['start']==row['start'] and step['update']==index,'Original four-start ordering required')
        need(step['optimizer_updates']==1 and step['live_sequential_forwards']==2,'One optimizer update after two main branches required')
        aux=step['auxiliary']
        if index in AUXILIARY_UPDATES:
            need(aux.get('enabled') is True and aux.get('lambda')==1 and aux.get('endpoint_scale')==1 and aux.get('feature_extracts')==2 and aux.get('head_predictions')==4,'Prescribed auxiliary update missing or malformed');active.append(index)
        else:need(aux=={'enabled':False,'lambda':1.0,'feature_extracts':0,'head_predictions':0,'weighted_loss':0.},'Unexpected auxiliary work on another window')
        need(step['total_objective']==step['paired_mean_future_flow_mse']+aux['weighted_loss'],'Main plus lambda1 auxiliary objective required')
    expected={'main_feature_forwards':256,'main_head_predictions':256,'total_training_feature_forwards':320,'total_training_head_predictions':384,'auxiliary_feature_extracts':64,'auxiliary_head_predictions':128,'parity_native_forwards':2,'parity_bridge_forwards':2}
    need(all(training.get(k)==v for k,v in expected.items()),'Explicit main/auxiliary/parity counters differ')
    need(training.get('negative_context_adapter_training') is True,'Negative-context auxiliary training must be recorded')
    return active

def clipped_gradient_record(values,row):
    need(values and all(v.dtype==np.float32 and np.isfinite(v).all() for v in values.values()),'Finite FP32 combined gradients required')
    total=math.sqrt(sum(norm(v)**2 for v in values.values()));gru=math.sqrt(sum(norm(v)**2 for n,v in values.items()if n.startswith('command_gru.')));output=math.sqrt(sum(norm(v)**2 for n,v in values.items()if n.startswith('output.')))
    before=row['gradient_l2_before_clip'];need(finite_number(before) and before>0,'Finite positive pre-clip norm required')
    scale=min(1.,1./(before+1e-6))
    for actual,recorded in [(total,row['gradient_l2_after_clip']),(gru,row['command_gru_gradient_l2']*scale),(output,row['output_gradient_l2']*scale)]:
        need(finite_number(recorded) and math.isclose(actual,recorded,rel_tol=GRADIENT_TOLERANCE['relative'],abs_tol=GRADIENT_TOLERANCE['absolute']),'Combined gradient norm or clip scaling mismatch')
    need(0<total<=1+2e-6,'One original L2 clip required')
    return {'saved_post_clip_l2_fp64':total,'saved_post_clip_gru_l2_fp64':gru,'reported_clip_coefficient':scale,'separate_component_gradients_retained':False}
