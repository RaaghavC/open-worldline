# SPDX-License-Identifier: Apache-2.0
"""Native profile worker. Imported only by explicit dispatch, never plan mode."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from run_profile import SCHEMA, paths, protocol, sha


def worker(config):
    if config.get('schema') != SCHEMA or config.get('protocol') != protocol():
        raise ValueError('Exact profile protocol required')
    for key in ('repository','controller_source','training_source'):
        sys.path.insert(0,config[key])
    import torch
    from safetensors.torch import save_file
    spec=importlib.util.spec_from_file_location('command_profile_inputs',Path(config['training_source'])/'packet.py')
    packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
    import math_steps
    from controller import CommandAttentionController, DEFAULT_PARAMETER_COUNT
    from bridge import NativeCommandAttentionBridge
    from engine import execute
    from experiments.wan22_native.action_cuda.probe import _original_weights, _runtime_flags, PRECISION
    from experiments.wan22_native.action_training.objective import parameter_records
    from experiments.wan22_native.cuda_reference import native
    from experiments.wan22_native.official_cpu.streaming import tensor_sha
    from experiments.wan22_native.spatial_reference.guards import atomic, hardware, Monitor, _deadline

    out = Path(config['output'])/'result'; out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(schema=SCHEMA,status='running',model_execution=False,quality_assessed=False,
                  image_generation=False,completed_updates=0,automatic_training_promotion=False)
    atomic(out/'metrics.json',report)
    try:
        _deadline(config['deadline'],'pair')
        bindings = paths(argparse.Namespace(**config))
        if {n:sha(p) for n,p in bindings.items()} != config['source_sha256']:
            raise ValueError('Supplied profile, controller, training or native source changed')
        prepared = Path(config['prepared'])
        reader=packet.Inputs(prepared,config['inputs_sha256'])
        plan=reader.plan
        data=dict(windows=reader.windows,positive=reader.positive,negative=reader.negative)
        rows=plan['schedule'][:2]
        initial=reader.initial
        if sum(v.numel() for v in initial.values()) != DEFAULT_PARAMETER_COUNT:
            raise ValueError('Exact default controller parameter count required')
        identity = dict(windows={a:{k:tensor_sha(v) for k,v in w.items()} for a,w in data['windows'].items()},
                        positive=tensor_sha(data['positive']),negative=tensor_sha(data['negative']))
        report.update(source_sha256=config['source_sha256'],inputs_sha256=config['inputs_sha256'],
                      initial_controller_sha256=plan['files']['initial-controller.safetensors']['sha256'],input_identity=identity,
                      rows=rows,precision=PRECISION,hardware=hardware(config['expected_gpu']),runtime_flags=_runtime_flags())
        torch.set_num_threads(1)
        with Monitor(out,config['deadline'],'pair') as monitor:
            def check():
                monitor.check()
                if _runtime_flags()!=report['runtime_flags']:
                    raise RuntimeError('Runtime precision flags changed')
            check(); report['model_execution']=True; atomic(out/'metrics.json',report)
            start=time.monotonic();core,weights=native.load_model(config['weights'],check);torch.cuda.synchronize()
            report['load_seconds']=time.monotonic()-start
            report['load_memory']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            atomic(out/'weight-load.json',weights)
            expected=_original_weights(weights)
            def verify_core(label):
                start=time.monotonic();values=parameter_records(core,check=check,expected=expected)
                atomic(out/('core-'+label+'.json'),values)
                report.setdefault('foundation_hash_seconds',{})[label]=time.monotonic()-start
                if any(m._forward_hooks or m._forward_pre_hooks for m in core.modules()):
                    raise RuntimeError('Controller leaked native hooks')
            verify_core('before')
            rotary=core.freqs.detach().cpu().clone()
            with torch.inference_mode(False):
                controller=CommandAttentionController();controller.load_state_dict(initial,strict=True);controller.to('cuda:0')
            if any(torch.count_nonzero(p.b.weight) for p in controller.projections.values()):
                raise ValueError('All low-rank output matrices must initialize at exactly zero')
            if any(tensor_sha(v.detach().cpu())!=tensor_sha(initial[n]) for n,v in controller.state_dict().items()):
                raise RuntimeError('Saved controller CUDA initialization differs')
            bridge=NativeCommandAttentionBridge(core,controller,profile='spatial')
            def memory():
                return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                            allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))
            def retain(name,values):
                path=out/(name+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise ValueError('Refuse overwriting retained profile evidence')
                save_file({n:v.detach().cpu().contiguous() for n,v in values.items()},str(path))
            def native_predict(noisy,times,context):
                if tuple(noisy.shape)!=(1,48,5,44,78) or tuple(times.shape)!=(1,4290):
                    raise ValueError('Only literal native 1248 by 704 inputs admitted')
                with torch.inference_mode(False),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    value=core([noisy[0].to('cuda:0')],times.to('cuda:0'),[context.to('cuda:0')],4290)
                torch.cuda.synchronize()
                return value[0].detach().float().cpu().unsqueeze(0)
            def progress(current):
                report.update(current);report.update(status='running',memory=memory())
                atomic(out/'metrics.json',report)
            def run_step(row,noise,optimizer,store):
                return math_steps.update(bridge,data['windows'],row,noise,data['positive'],data['negative'],optimizer,
                                         check=check,retain=store,synchronize=torch.cuda.synchronize)
            result=execute(bridge,data,rows,noise=lambda i:reader.noise(rows[i]),
                flow_inputs=math_steps.flow_inputs,native_predict=native_predict,tensor_sha=tensor_sha,
                make_optimizer=lambda:torch.optim.AdamW(controller.parameters(),**math_steps.OPTIMIZER),run_step=run_step,
                retain=retain,progress=progress,verify_core=verify_core,check=check,synchronize=torch.cuda.synchronize,
                reset_peak=lambda:torch.cuda.reset_peak_memory_stats(0),memory=memory)
            report.update(result,status='running')
            if core.freqs.is_inference() or not torch.equal(core.freqs.detach().cpu(),rotary):
                raise RuntimeError('Original rotary values or normal-tensor status changed')
            report.update(rotary_unchanged=True,all825_current_value_hashes_verified=True)
            if {a:{k:tensor_sha(v) for k,v in w.items()} for a,w in data['windows'].items()} != identity['windows']:
                raise RuntimeError('Window values changed during profile')
            if any(tensor_sha(data[k])!=identity[k] for k in ('positive','negative')):
                raise RuntimeError('Context values changed during profile')
            check()
        if {n:sha(p) for n,p in bindings.items()} != config['source_sha256']:
            raise RuntimeError('Executed sources changed during profile')
        if sha(prepared/'inputs.json')!=config['inputs_sha256']:
            raise RuntimeError('Input identity changed during profile')
        _deadline(config['deadline'],'pair')
        report['output_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
                                 if p.is_file() and p.name!='metrics.json'}
        report.update(status='passed',sources_unchanged=True,inputs_unchanged=True)
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;atomic(out/'metrics.json',report)
