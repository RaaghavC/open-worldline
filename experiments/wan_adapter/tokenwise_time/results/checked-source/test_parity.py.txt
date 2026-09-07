# SPDX-License-Identifier: Apache-2.0
"""CPU parity for an untrained tokenwise-time conditioning extension."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from native_control.portable import create_model as create_original_model
from tokenwise_time.portable import create_model, token_times


def trace_forward(model,inputs,contexts,times,seq_len):
    trace={}; hooks=[]
    for name,module in {'time_embedding':model.time_embedding,'time_projection':model.time_projection,
                        'text_embedding':model.text_embedding,'head_output':model.head,
                        **{f'block{i}_output':b for i,b in enumerate(model.blocks)}}.items():
        def capture(module,args,out,name=name):trace[name]=out.detach().clone()
        hooks.append(module.register_forward_hook(capture))
    for i,block in enumerate(model.blocks):
        def capture_block(module,args,kwargs,i=i):
            e=kwargs['e']; trace[f'block{i}_time_input']=e.detach().clone()
            trace[f'block{i}_modulation']=(module.modulation.unsqueeze(0)+e if e.ndim==4 else module.modulation+e).detach().clone()
        hooks.append(block.register_forward_pre_hook(capture_block,with_kwargs=True))
    def capture_head(module,args):
        e=args[1]; trace['head_time_input']=e.detach().clone()
        trace['head_modulation']=(module.modulation.unsqueeze(0)+e.unsqueeze(2) if e.ndim==3 else module.modulation+e.unsqueeze(1)).detach().clone()
    hooks.append(model.head.register_forward_pre_hook(capture_head))
    try:
        with torch.inference_mode():out=model(inputs,times,contexts,seq_len)
    finally:
        for hook in hooks:hook.remove()
    trace.update({f'velocity{i}':v.detach() for i,v in enumerate(out)})
    return trace


def run():
    torch.set_num_threads(4);torch.manual_seed(20260910)
    config=dict(model_type='t2v',dim=256,ffn_dim=512,freq_dim=256,num_heads=2,num_layers=2,
                text_dim=4096,text_len=512,in_dim=16,out_dim=16)
    original=create_original_model(portable=True,**config).eval()
    with torch.no_grad():
        original.head.head.weight.normal_(std=.025);original.head.head.bias.normal_(std=.01)
    extended=create_model(**config).eval();extended.load_state_dict(original.state_dict(),strict=True)
    assert list(original.state_dict())==list(extended.state_dict())
    assert all(torch.equal(v,extended.state_dict()[k]) for k,v in original.state_dict().items())
    inputs=[torch.randn(16,3,4,6),torch.randn(16,2,2,4)]
    contexts=[torch.randn(25,4096)*.2,torch.randn(1,4096)*.2]
    rows=[];seq_len=20
    for t in (999,500,50,0):
        scalar=torch.tensor([t,t//2],dtype=torch.int64)
        before=trace_forward(original,inputs,contexts,scalar,seq_len)
        after=trace_forward(extended,inputs,contexts,scalar[:,None].expand(-1,seq_len).clone(),seq_len)
        for key,a in before.items():
            b=after[key]
            if a.shape!=b.shape:
                if key=='time_projection':a=a[:,None].expand_as(b)
                else:a=a[:,None].expand_as(b)
            delta=(a-b).abs()
            row={'timestep_batch':scalar.tolist(),'stage':key,'shape':list(b.shape),'max_abs':delta.max().item()}
            assert torch.isfinite(b).all() and torch.allclose(a,b,atol=2e-5,rtol=2e-5),row
            rows.append(row)
    mixed=token_times(torch.tensor([[3,2,3],[2,1,2]]),torch.tensor([999,500]),seq_len)
    mixed_trace=trace_forward(extended,inputs,contexts,mixed,seq_len)
    assert all(torch.isfinite(x).all() for x in mixed_trace.values())
    # Instance method binding must leave a subsequently created original untouched.
    fresh=create_original_model(portable=True,**config)
    assert fresh.forward.__func__ is original.forward.__func__
    return {'status':'passed','configuration':config,'head_width':128,'atol':2e-5,'rtol':2e-5,
            'rows':rows,'state_dict_keys_identical':True,'parameter_tensors_identical':True,
            'mixed_token_times_finite':True,'no_class_or_original_instance_mutation':True,
            'limits':'Tiny random-weight CPU mathematical test only. No pretrained weights loaded; no image generation or training. Mixed token times remain untrained for Wan2.1 T2V.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Evidence output must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter();report={'status':'running'}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='.*torch.cuda.amp.autocast.*')
            warnings.filterwarnings('ignore',message="User provided device_type of 'cuda'.*")
            report=run()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        here=Path(__file__).parent
        files=[Path(__file__),here/'portable.py',here.parent/'native_control/portable.py',here.parent/'native_control/vendor/model.py']
        report.update(elapsed_seconds=time.perf_counter()-start,torch=torch.__version__,source_sha256={str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
