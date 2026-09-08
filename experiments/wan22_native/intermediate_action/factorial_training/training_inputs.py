# SPDX-License-Identifier: Apache-2.0
"""Exact six-arm packet. Prior bare-module validators run only in isolation."""
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from experiments.wan22_native.action_cuda import probe_evidence as evidence
from experiments.wan22_native.intermediate_action.cuda_profile import packet as profile_sources
from experiments.atrium_factorial.native_cache import packet as cache_sources
from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
from experiments.wan22_native.spatial_reference.guards import atomic,limits
import training_math

HERE=Path(__file__).resolve().parent
REPO=evidence.REPO
SCHEMA='worldline-factorial-intermediate128-v1'
SCOPE='fresh-block28-factorial128-training-only'
CATALOG_SHA='dbbd9d7e4f3150be15cd94359e4ec1cb61e40846fd548af9d0142eb5ac9da54a'
RECEIPTS={'cache':'5b0ce7941d64b6a6a767bec38bdeebf576f492386ec996d8a962569faf913c15',
          'profile':'3ed83d887b33dabfa64993a37cbb28c13c69d4947a5e4ef7a1630811fd6adf89'}
PROOF_FILES={'metrics.json':'parent_sha256','terminal.json':'terminal_sha256','plan.json':'plan_sha256'}


def source_paths():
    result={}
    for collector in (profile_sources.source_paths,cache_sources.source_paths):
        for path in collector().values():result['repo/'+str(path.relative_to(REPO))]=path
    for name in ('training_inputs.py','training_math.py','training_run.py','validate_prior.py','test_cpu.py',
                 'original-inputs.json','cache-validation-v1.json','profile-validation-v1.json'):
        result['local/'+name]=HERE/name
    return dict(sorted(result.items()))


def sources():return {k:sha(v) for k,v in source_paths().items()}


def protocol():
    return dict(scope=SCOPE,block_index=28,profile='spatial',updates=128,motions=list(training_math.MOTIONS),
        motion_pair_counts={'stationary':43,'left':43,'right':42},
        auxiliary_updates=32,main_predictions=256,auxiliary_predictions=128,auxiliary_feature_extracts=64,
        initial='Exact original saved zero adapter; fresh AdamW; no warm start',
        loss='Unchanged effect.paired_update; auxiliary enabled on updates 1, 5, ... 125',
        optimizer=dict(lr=.0001,betas=[.9,.999],eps=1e-8,weight_decay=.01),clip_l2=1.,
        target_prefix='Must already be bit-exact to independent observation; no target replacement',
        checkpoints=list(training_math.CHECKPOINTS),zero_native_checks=2,zero_gate='bit-exact',
        limits=limits('pair'),quality_assessed=False,image_generation=False,automatic_promotion=False,
        negative_context_adapter_training=True,
        placement_comparison='New data; no matched-data block 28 versus 29 superiority comparison')


def checked_tensors(path,record,keys=None):
    path=Path(path)
    if (not 0<record['bytes']<=64*2**20 or path.is_symlink() or path.stat().st_size!=record['bytes']
            or sha(path)!=record['sha256']):raise ValueError('Saved tensor file differs or exceeds bound')
    header=record['header'];keys=list(header) if keys is None else keys
    if not set(keys)<=set(header):raise ValueError('Selected tensor is absent from exact header')
    with safe_open(path,framework='pt',device='cpu') as handle:
        if set(handle.keys())!=set(header):raise ValueError('Exact saved tensor names required')
        for name,spec in header.items():
            view=handle.get_slice(name)
            if view.get_shape()!=spec['shape'] or view.get_dtype()!=spec['dtype']:raise ValueError('Saved tensor header differs')
        result={}
        for name in keys:
            value=handle.get_tensor(name)
            if not torch.isfinite(value).all():raise ValueError('Finite saved tensor required')
            if 'sha256' in header[name] and tensor_sha(value)!=header[name]['sha256']:raise ValueError('Saved tensor hash differs')
            result[name]=value
        return result


def review(path):
    value=evidence.read_json(path)
    if (value.get('status')!='passed' or value.get('tests',0)<8 or value.get('source_sha256')!=sources()
            or value.get('sources_unchanged') is not True or any(value.get(k)!=0 for k in ('pytest_exit_code','failures','errors','skipped'))):
        raise ValueError('Current completed CPU evidence required')
    return sha(path)


def catalog():
    path=HERE/'original-inputs.json'
    if sha(path)!=CATALOG_SHA:raise ValueError('Frozen original public input catalog changed')
    value=evidence.read_json(path)
    if value['original_plan']['sha256']!='bd6dc4082ddf1f19033f8b967eaf5cb3356d6e0484530e039e250d68a24c68e4':
        raise ValueError('Exact original 128 plan required')
    # Operational original paths are provenance; the new packet uses basenames.
    result=dict(value,files={Path(name).name:record for name,record in value['files'].items()})
    if len(result['files'])!=len(value['files']):raise ValueError('Input basename collision')
    return result


def receipts_at(root,prepared=False):
    result={}
    for kind,digest in RECEIPTS.items():
        path=Path(root)/(kind+('-validation.json' if prepared else '-validation-v1.json'))
        if sha(path)!=digest:raise ValueError('Pinned actual stage receipt differs')
        value=evidence.read_json(path)
        expected=(cache_sources if kind=='cache' else profile_sources).sources()
        if value.get('status')!='passed' or value.get('source_sha256')!=expected:
            raise ValueError('Actual prior source graph differs from current imported dependencies')
        result[kind]=value
    if (result['cache'].get('raw_target_prefix_bit_exact') is not True or result['cache'].get('all196_unchanged') is not True
            or result['profile'].get('exact_zero_comparisons')!=16 or result['profile'].get('all825_unchanged') is not True):
        raise ValueError('Actual cache and CUDA profile gates required')
    return result


def proof_map(kind,receipt):
    paths=dict(PROOF_FILES)
    paths[('worker' if kind=='cache' else 'result')+'/metrics.json']='worker_sha256'
    if kind=='cache':paths['result/completion.json']='completion_sha256'
    return {name:receipt[key] for name,key in paths.items()}


def expected_files(original,receipts):
    result={'original/'+name:record for name,record in original['files'].items()}
    for name,record in receipts['cache']['selected_files'].items():
        result['cache/'+name]=dict(record,header=receipts['cache']['windows'][Path(name).stem])
    return result


def prepare(original_run,cache_run,profile_run,cpu_report,output):
    output=Path(output)
    if output.exists():raise ValueError('Fresh training packet required')
    cpu_hash=review(cpu_report);original=catalog();receipts=receipts_at(HERE)
    output.mkdir(parents=True)
    files=expected_files(original,receipts)
    for name,record in files.items():
        directory,relative=name.split('/',1)
        source=evidence.relative_file(original_run if directory=='original' else cache_run,relative)
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        if sha(source)!=record['sha256'] or source.stat().st_size!=record['bytes']:raise ValueError('Original input changed')
        shutil.copyfile(source,target)
    for kind in receipts:
        shutil.copyfile(HERE/(kind+'-validation-v1.json'),output/(kind+'-validation.json'))
        for name,digest in proof_map(kind,receipts[kind]).items():
            source=evidence.relative_file(cache_run if kind=='cache' else profile_run,name)
            if sha(source)!=digest:raise ValueError('Actual prior parent/worker evidence changed')
            target=output/'prior'/kind/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(cpu_report,output/'cpu-report.json')
    mapping=sources()
    for name,path in source_paths().items():
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    plan=dict(schema=SCHEMA,status='prepared',protocol=protocol(),schedule=training_math.schedule(original['schedule']),
        source_sha256=mapping,files=files,cpu_report_sha256=cpu_hash,prior_receipt_sha256=RECEIPTS,
        original_catalog_sha256=CATALOG_SHA,model_execution=False)
    atomic(output/'plan.json',plan);read_prepared(output);return plan


class Draws:
    """One original shard at a time, with no RNG reconstruction or new draws."""
    def __init__(self,root,records,rows):
        self.root=Path(root);self.records=records;self.rows=rows;self.current=None;self.values={}
    def chunk(self,index):
        first=index//16*16;name=f'draws-{first:04d}-{first+15:04d}.safetensors'
        if name!=self.current:
            self.values={};self.current=None
            values=checked_tensors(self.root/name,self.records[name])
            for row in self.rows[first:first+16]:
                if tensor_sha(values[row['noise_key']])!=row['noise_sha256']:raise ValueError('Original noise differs')
                if tensor_sha(values[f'rng_after_{row["update"]-1:04d}'])!=row['rng_after_sha256']:raise ValueError('Original saved draw RNG differs')
            self.values=values;self.current=name
        return self.values
    def noise(self,row):
        if row!=self.rows[row['update']-1]:raise ValueError('Requested draw row differs')
        return self.chunk(row['update']-1)[row['noise_key']]
    def rng(self,completed):
        if type(completed)is not int or not 0<=completed<=128:raise ValueError('Completed draw count invalid')
        return self.chunk(max(completed-1,0))['rng_initial' if not completed else f'rng_after_{completed-1:04d}']
    def validate_all(self):
        for index in range(0,128,16):self.chunk(index)
        self.values={};self.current=None


def read_prepared(root):
    root=Path(root);plan=evidence.read_json(root/'plan.json');original=catalog()
    if (plan.get('schema')!=SCHEMA or plan.get('status')!='prepared' or plan.get('protocol')!=protocol()
            or plan.get('source_sha256')!=sources() or plan.get('schedule')!=training_math.schedule(original['schedule'])
            or plan.get('prior_receipt_sha256')!=RECEIPTS or plan.get('original_catalog_sha256')!=CATALOG_SHA):
        raise ValueError('Prospective protocol/source/schedule differs')
    for name,digest in plan['source_sha256'].items():
        if sha(evidence.relative_file(root/'source',name))!=digest:raise ValueError('Source snapshot differs')
    if review(root/'cpu-report.json')!=plan['cpu_report_sha256']:raise ValueError('CPU report changed')
    receipts=receipts_at(root,prepared=True)
    for kind,receipt in receipts.items():
        for name,digest in proof_map(kind,receipt).items():
            if sha(evidence.relative_file(root/'prior'/kind,name))!=digest:raise ValueError('Actual prior evidence changed')
    if plan['files']!=expected_files(original,receipts):raise ValueError('Prepared file catalog differs')
    for name,record in plan['files'].items():
        path=evidence.relative_file(root,name)
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('Prepared artifact differs')
    windows={a:checked_tensors(root/'cache/result'/f'{a}.safetensors',plan['files'][f'cache/result/{a}.safetensors']) for a in receipts['cache']['windows']}
    training_math.validate_windows(windows)
    def one(name,key=None):
        values=checked_tensors(root/'original'/name,original['files'][name]);return values if key is None else values[key]
    initial=one('initial-adapter.safetensors')
    if sum(v.numel() for v in initial.values())!=947712 or any(torch.count_nonzero(initial[k]) for k in ('output.weight','output.bias')):raise ValueError('Original zero adapter required')
    positive=one('positive.safetensors','context');negative=one('negative.safetensors','context');rng=one('initial-cpu-rng.safetensors','rng')
    draws=Draws(root/'original',original['files'],plan['schedule']);draws.validate_all()
    return plan,dict(windows=windows,initial=initial,positive=positive,negative=negative,rng=rng,noise=draws.noise,draw_rng=draws.rng)


def admission(path,root,plan,gpu):
    value=evidence.read_json(path);required=dict(schema=SCHEMA,scope=SCOPE,decision='admit',
        plan_sha256=sha(Path(root)/'plan.json'),source_sha256=plan['source_sha256'],prior_receipt_sha256=plan['prior_receipt_sha256'],
        cpu_report_sha256=plan['cpu_report_sha256'],expected_gpu=gpu,limits=limits('pair'),image_generation=False)
    if not gpu or any(value.get(k)!=v for k,v in required.items()):raise ValueError('New exact source/input-bound training admission required')
    return dict(sha256=sha(path),**required)
