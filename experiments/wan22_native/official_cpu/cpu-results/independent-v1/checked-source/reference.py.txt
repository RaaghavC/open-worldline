# SPDX-License-Identifier: Apache-2.0
"""Translate only official CUDA FP32/disabled autocast calls, retaining all math."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import torch

HERE = Path(__file__).resolve().parent
MODEL_SHA256 = '8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432'


def translated_source():
    path = HERE/'vendor/model.py';raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MODEL_SHA256:
        raise ValueError('Pinned official model source changed')
    tree=ast.parse(raw);original=copy.deepcopy(tree);changes=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or ast.unparse(node.func)!='torch.amp.autocast':continue
        kwargs={x.arg:ast.unparse(x.value)for x in node.keywords}
        if len(node.args)!=1 or ast.literal_eval(node.args[0])!='cuda' or kwargs not in ({'enabled':'False'},{'dtype':'torch.float32'}):
            raise ValueError('Unreviewed autocast call')
        before=copy.deepcopy(node)
        node.args=[ast.Constant('cpu')];node.keywords=[ast.keyword(arg='enabled',value=ast.Constant(False))]
        changes.append({'line':node.lineno,'before':ast.unparse(before),'after':ast.unparse(node),
                        'original_ast':ast.dump(before,include_attributes=False)})
    if len(changes)!=7:
        raise ValueError('Expected exactly seven reviewed native precision contexts')
    ast.fix_missing_locations(tree)
    # Reverse precisely those nodes and require the complete original AST.
    reverse=copy.deepcopy(tree);original_calls={n.lineno:n for n in ast.walk(original)
        if isinstance(n,ast.Call)and ast.unparse(n.func)=='torch.amp.autocast'}
    for node in ast.walk(reverse):
        if isinstance(node,ast.Call)and node.lineno in original_calls and ast.unparse(node.func)=='torch.amp.autocast':
            prior=original_calls[node.lineno];node.args=copy.deepcopy(prior.args);node.keywords=copy.deepcopy(prior.keywords)
    if ast.dump(reverse,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise RuntimeError('Translation changed an unapproved AST node')
    code=ast.unparse(tree)+'\n'
    report={'original_source_sha256':MODEL_SHA256,'translated_source_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'complete_ast_reversal_exact':True,'changed_calls':changes,
        'other_ast_nodes_changed':False,'attention_substitution':'Package-local independent CPU BF16 SDPA, not CUDA FlashAttention',
        'model_equations_replaced':False,'literal_complex_fp64_rope':True}
    return code,report


def native_module():
    code,report=translated_source();name=__package__+'.vendor._translated_native'
    module=types.ModuleType(name);module.__package__=__package__+'.vendor';module.__file__=str(HERE/'vendor/model.py')
    sys.modules[name]=module
    exec(compile(code,str(HERE/'vendor/model.py')+' [CPU contexts]','exec'),module.__dict__)
    return module,report


def create_meta(configuration=None):
    if configuration is None and hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()!='d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc':
        raise ValueError('Original full-model configuration differs')
    config=json.loads((HERE/'config.json').read_text()) if configuration is None else dict(configuration)
    config={k:v for k,v in config.items()if not k.startswith('_')}
    module,report=native_module()
    with torch.device('meta'):model=module.WanModel(**config)
    # Native initializer creates this non-parameter table on meta in this context.
    # Recompute with the exact native function on CPU, without real-valued substitution.
    d=model.dim//model.num_heads
    model.freqs=torch.cat([module.rope_params(1024,d-4*(d//6)),module.rope_params(1024,2*(d//6)),
                          module.rope_params(1024,2*(d//6))],dim=1)
    if model.freqs.dtype!=torch.complex128 or model.freqs.device.type!='cpu':
        raise RuntimeError('Literal complex128 rotary table required')
    return model.eval().requires_grad_(False),report


def forward(model,latent,times,context,seq_len):
    if any(x.device.type!='cpu' for x in (latent,times,context)):
        raise ValueError('This reference has no GPU execution path')
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        return model([latent],times,[context],seq_len)[0]
