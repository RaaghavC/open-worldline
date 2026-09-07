# SPDX-License-Identifier: Apache-2.0
"""Native settings and explicit CPU solver; injected predictions in CPU tests."""
import torch
from .vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler

SHAPE=(48,5,18,32)
SETTINGS={'steps':50,'shift':5.,'guidance':5.}


def scheduler():
    value=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
    value.set_timesteps(50,device='cpu',shift=5.)
    return value


def times_at(t):
    t=torch.as_tensor(t,device='cpu')
    if t.numel()!=1 or t.dtype!=torch.int64 or not 0<=int(t)<=999:
        raise ValueError('Exact integer scheduler time required')
    out=t.reshape(1,1).expand(1,720).clone();out[:,:144]=0
    return out


def pair(predict,x,times,positive,negative,observation,event=None):
    """Positive then negative, identical projected state and token times."""
    prefix=observation[0]
    outputs={}
    for label,context in [('positive',positive),('negative',negative)]:
        value=x.clone();value[:,:1]=prefix
        prediction=predict(value,times.clone(),context)
        if prediction.dtype!=torch.float32 or prediction.device.type!='cpu' or tuple(prediction.shape)!=SHAPE or not torch.isfinite(prediction).all():
            raise FloatingPointError('Finite FP32 CPU prediction with exact latent shape required')
        outputs[label+'_velocity']=prediction.clone()
        if event:event(label,outputs)
    guided=outputs['negative_velocity']+5.*(outputs['positive_velocity']-outputs['negative_velocity'])
    if not torch.isfinite(guided).all():raise FloatingPointError('Nonfinite guided velocity')
    outputs['guided_velocity']=guided
    return outputs


def sample(predict,values,contexts,*,event=None):
    solver=scheduler();x=values['initial_noise'].clone();observation=values['observation']
    x[:,:1]=observation[0]
    if not torch.equal(x,values['initial_latent']) or not torch.equal(times_at(solver.timesteps[0]),values['token_times']):
        raise ValueError('Saved initial state/time differs from native settings')
    for step,t in enumerate(solver.timesteps):
        x[:,:1]=observation[0]
        result=pair(predict,x,times_at(t),contexts['atrium'],contexts['native_negative'],observation)
        x=solver.step(result['guided_velocity'].unsqueeze(0),t,x.unsqueeze(0),return_dict=False)[0][0]
        x[:,:1]=observation[0]
        if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite solver state')
        if event:event(step,t,x,result)
    return x
