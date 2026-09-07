"""Read-only completed CUDA clip audit; NumPy/Pillow, no Torch or cloud calls."""
import os
import sys
_HERE=os.path.dirname(os.path.realpath(__file__))
sys.path[:]=[p for p in sys.path if os.path.realpath(p or os.getcwd())!=_HERE]
import ast
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import struct
import zipfile
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
TASK=HERE.parents[1]
ROOT=HERE/'recovered-clip-v1/results/clip-run-v1'
PAIR=HERE/'recovered-pair-v1/results/pair-run-v1'
REPO=TASK/'outputs/open-worldline'
OUT=HERE/'clip-independent-audit-v1.json'
VAE=TASK/'work/wan22-ti2v5b-weights/Wan2.2_VAE.pth'
VAE_SHA='20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def obj(path):return json.loads(Path(path).read_text())
def tensor_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def arrays(path):
    raw=Path(path).read_bytes();assert 8<=len(raw)<=64*2**20
    count=struct.unpack('<Q',raw[:8])[0];assert 2<=count<=2**20 and count+8<=len(raw)
    header=json.loads(raw[8:count+8]);body=memoryview(raw)[count+8:];values={};intervals=[]
    for name,row in header.items():
        if name=='__metadata__':continue
        assert row['dtype']in ('F32','I64')
        dtype=np.dtype('<f4'if row['dtype']=='F32'else'<i8');lo,hi=row['data_offsets']
        assert 0<=lo<hi<=len(body)and hi-lo==math.prod(row['shape'])*dtype.itemsize
        a=np.frombuffer(body[lo:hi],dtype=dtype).reshape(row['shape']);assert np.isfinite(a).all()
        values[name]=a;intervals.append((lo,hi))
    intervals.sort();assert intervals[0][0]==0 and intervals[-1][1]==len(body)
    assert all(a[1]==b[0]for a,b in zip(intervals,intervals[1:]))
    return values


def rebuilt(storage,offset,size,stride,requires_grad,hooks,metadata=None):
    assert isinstance(storage,dict)and type(offset)is int and offset==0
    assert isinstance(size,tuple)and isinstance(stride,tuple)and len(size)==len(stride)
    assert all(type(n)is int and n>0 for n in size)
    expected=[];product=1
    for n in reversed(size):expected.append(product);product*=n
    assert stride==tuple(reversed(expected))and product==storage['elements']
    return dict(storage,shape=list(size),offset=offset,stride=list(stride))


class MetadataOnly(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return OrderedDict
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuilt
        if (module,name)==('torch','FloatStorage'):return 'FLOAT32_STORAGE_TAG'
        raise ValueError('Unapproved pickle global; no dynamic model classes allowed')
    def persistent_load(self,pid):
        assert isinstance(pid,tuple)and len(pid)==5 and pid[0]=='storage'and pid[1]=='FLOAT32_STORAGE_TAG'
        assert isinstance(pid[2],str)and pid[2].isdigit()and type(pid[4])is int and pid[4]>0
        return {'storage_key':pid[2],'elements':pid[4]}


def vae_storage_audit(report):
    assert VAE.stat().st_size==2818839170 and sha(VAE)==VAE_SHA==report['weight_sha256']
    assert report['compute_dtype']=='float32'and report['parameters']==704688668
    checked={};total=0
    with zipfile.ZipFile(VAE)as z:
        meta_names=[n for n in z.namelist()if n.endswith('/data.pkl')];assert len(meta_names)==1
        prefix=meta_names[0][:-len('data.pkl')];raw=z.read(meta_names[0]);assert len(raw)<2**20
        metadata=MetadataOnly(io.BytesIO(raw)).load()
        assert set(metadata)==set(report['tensors'])and len(metadata)==196
        assert len({v['storage_key']for v in metadata.values()})==196
        for name,spec in metadata.items():
            row=report['tensors'][name];assert spec['shape']==row['shape']and row['cuda_copy_exact']is True
            storage=prefix+'data/'+spec['storage_key'];assert z.getinfo(storage).file_size==spec['elements']*4
            h=hashlib.sha256()
            with z.open(storage)as f:
                for block in iter(lambda:f.read(2**20),b''):h.update(block)
            assert h.hexdigest()==row['sha256'],name
            checked[name]=h.hexdigest();total+=spec['elements']
    assert total==704688668
    return {'original_checkpoint_sha256':VAE_SHA,'matched_tensor_storages':196,'matched_parameters':total,
        'method':'Restricted metadata parser accepts only OrderedDict, a non-executing FloatStorage tag and a metadata-only tensor reconstruction function. Contiguous original storage bytes are streamed and hashed; no Torch import or tensor/model construction.',
        'tensor_sha256':checked}


def main():
    if OUT.exists():raise ValueError('Fresh audit output required')
    before={str(p.relative_to(ROOT)):sha(p)for p in ROOT.rglob('*')if p.is_file()}
    parent=obj(ROOT/'metrics.json');core=obj(ROOT/'core/result/metrics.json');decode=obj(ROOT/'decode/result/metrics.json')
    pair=obj(PAIR/'metrics.json');paircore=obj(PAIR/'core/result/metrics.json')
    package=REPO/'experiments/wan22_native/cuda_reference'
    assert parent['status']==core['status']==decode['status']=='passed'and parent['mode']==core['mode']==decode['mode']=='clip'
    assert parent['execute_requested']is True and parent['model_execution']is True
    assert core['predictions']==100 and core['solver_updates']==50 and decode['predictions']==decode['solver_updates']==0
    assert core['settings']==decode['settings']=={'steps':50,'shift':5.,'guidance':5.}
    assert core['input_tensors_unchanged']and core['finite_outputs']and decode['finite_outputs']and decode['decoder_cache_clear']
    assert not any(r['actions_read']or r['future_target_read']for r in (core,decode))
    assert not list(ROOT.rglob('*watchdog-stop*'))
    sources={}
    for node in ast.parse((package/'evidence.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name)and target.id in ('NAMES','SHARED'):
                    sources.update({('shared/'+n if target.id=='SHARED'else n):sha((REPO if target.id=='SHARED'else package)/n)for n in ast.literal_eval(node.value)})
    assert parent['source_sha256']==core['source_sha256']==decode['source_sha256']==pair['source_sha256']==sources
    for name,digest in sources.items():assert sha(ROOT/'measured-source'/(name+'.txt'))==digest
    for name,key in [('cpu-report.json','cpu_report_sha256'),('independent-report.json','independent_report_sha256')]:
        assert sha(ROOT/name)==parent['gates'][key]==sha(PAIR/name)
    assert sha(ROOT/'independent-test.py.txt')==sha(package/'test_independent.py')
    assert parent['input_identity']==core['input_identity']==decode['input_identity']==pair['input_identity']
    for name,digest in parent['copied_input_sha256'].items():assert sha(ROOT/name)==digest==sha(PAIR/name)
    assert parent['copied_input_sha256']==pair['copied_input_sha256']
    admission=parent['pair_admission'];assert admission['pair_report_sha256']==sha(PAIR/'metrics.json')
    assert admission['pair_worker_sha256']==sha(PAIR/'core/result/metrics.json')
    assert admission['hardware']==core['hardware']==decode['hardware']==paircore['hardware']
    expected_estimate=1.2*(paircore['load_seconds']+50*paircore['pair_seconds'])+120+30
    assert admission['estimated_seconds']==expected_estimate and expected_estimate<=900
    assert admission['decoder_cost_measured']is False
    inputs=arrays(ROOT/'inputs.safetensors');observation=inputs['observation'][0]
    assert observation.shape==(48,1,18,32)and np.array_equal(inputs['initial_latent'][:,:1],observation)
    assert np.array_equal(inputs['initial_latent'][:,1:],inputs['initial_noise'][:,1:])
    assert (inputs['token_times'][:,:144]==0).all()and (inputs['token_times'][:,144:]==999).all()
    for name,array in inputs.items():assert tensor_sha(array)==parent['input_identity']['input_tensor_sha256'][name]
    original=obj(REPO/'experiments/wan22_native/core-results/cpu-pair-v1/weight-load.json')['tensors']
    weights=obj(ROOT/'core/result/weight-load.json');assert weights==obj(PAIR/'core/result/weight-load.json')
    assert set(weights['tensors'])==set(original)and weights['tensor_count']==len(original)==825
    for name,row in weights['tensors'].items():
        assert row['source_sha256']==row['loaded_sha256']==original[name]['original_sha256']
        assert row['shape']==original[name]['shape']and row['shard']==original[name]['shard']
        assert row['original_dtype']==row['loaded_dtype']=='float32'and row['cuda_copy_exact']and row['source_owner_released']
    assert sum(math.prod(v['shape'])for v in weights['tensors'].values())==weights['parameter_count']==4999787712
    assert core['precision']==paircore['precision']
    rows=[json.loads(line)for line in (ROOT/'core/result/steps.jsonl').read_text().splitlines()]
    assert len(rows)==50 and [r['step']for r in rows]==list(range(1,51))
    # Exact recorded official schedule, retained in the published source-gated CPU tests.
    # Its sigmas are linearly sampled between the native initial maximum/minimum before one shift.
    initial=np.linspace(1.,0.,1001)[::-1][1:]
    # The scheduler's constructor uses training timesteps 1..1000, reversed, then 1 - t/1000.
    native_sigma_min=0.;native_sigma_max=float(np.float32(0.999))
    sigmas=np.linspace(native_sigma_max,native_sigma_min,51)[:-1]
    scheduled=(5*sigmas/(1+4*sigmas)*1000).astype(np.int64)
    assert [r['timestep']for r in rows]==scheduled.tolist()
    for r in rows:
        path=ROOT/f'core/result/step-{r["step"]:02d}.safetensors';value=arrays(path)
        assert set(value)=={'latent'}and value['latent'].shape==(48,5,18,32)and value['latent'].dtype==np.dtype('<f4')
        assert r['prefix_exact']is True and np.array_equal(value['latent'][:,:1],observation)
        assert tensor_sha(value['latent'])==r['latent_sha256']
    final=arrays(ROOT/'core/result/latents.safetensors');assert np.array_equal(final['latent'],value['latent'])
    decoder_launch=obj(ROOT/'decode/launch.json')
    assert decoder_launch['latent_sha256']==sha(ROOT/'core/result/latents.safetensors')
    assert decoder_launch['source_sha256']==sources and decoder_launch['pair_admission']==admission
    assert obj(ROOT/'core/launch.json')['deadline']==decoder_launch['deadline']
    vae=vae_storage_audit(obj(ROOT/'decode/result/weight-load.json'))
    codec_info=obj(package/'codec-source.json');assert codec_info['weight_sha256']==VAE_SHA
    codec_source=ast.parse((package/'vendor/vae2_2.py').read_text());normalization={}
    for node in ast.walk(codec_source):
        if isinstance(node,ast.Assign)and isinstance(node.value,ast.Call)and ast.unparse(node.value.func)=='torch.tensor':
            for t in node.targets:
                if isinstance(t,ast.Name)and t.id in ('mean','std')and node.value.args:
                    try:vals=ast.literal_eval(node.value.args[0])
                    except (ValueError,TypeError):continue
                    if isinstance(vals,list)and len(vals)==48:normalization[t.id]=vals
    assert normalization==codec_info['normalization']
    decoded=arrays(ROOT/'decode/result/decoded.safetensors');assert set(decoded)=={'decoded_rgb'}
    rgb=decoded['decoded_rgb'];assert rgb.shape==(1,3,17,288,512)and rgb.dtype==np.dtype('<f4')and rgb.min()>=-1 and rgb.max()<=1
    image_rows=[]
    for i in range(17):
        path=ROOT/f'decode/result/frames/{i:04d}.png'
        with Image.open(path)as im:
            assert im.mode=='RGB'and im.size==(512,288)
            actual=np.array(im)
        expected=np.rint((rgb[0,:,i].transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)
        assert np.array_equal(actual,expected)
        image_rows.append({'frame':i,'png_sha256':sha(path),'rgb_pixel_sha256':tensor_sha(actual),'exact_saved_fp32_round_mapping':True})
    with Image.open(ROOT/'decode/result/comparison.png')as contact:
        assert contact.size==(1024,1580)
        for j,i in enumerate([0,1,2,4,6,8,10,12,14,16]):
            x=j%2*512;y=j//2*316+28
            with Image.open(ROOT/f'decode/result/frames/{i:04d}.png')as im:
                assert np.array_equal(np.array(contact.crop((x,y,x+512,y+288))),np.array(im))
    with Image.open(ROOT/'decode/result/preview.gif')as gif:
        assert gif.n_frames==17;durations=[]
        for i in range(gif.n_frames):gif.seek(i);durations.append(gif.info['duration'])
    resources={}
    for stage,report in [('core',core),('decode',decode)]:
        assert sha(ROOT/stage/'result/metrics.json')==parent[stage+'_report_sha256']
        terminal=obj(ROOT/stage/'terminal.json');assert terminal['status']=='complete'and terminal['exit_code']==0
        if stage=='core':assert sha(ROOT/stage/'terminal.json')==parent['core_terminal_sha256']
        for name,digest in report['output_sha256'].items():assert sha(ROOT/stage/'result'/name)==digest
        memory=[json.loads(line)for line in (ROOT/stage/'result/memory.jsonl').read_text().splitlines()]
        host=[json.loads(line)for line in (ROOT/stage/'parent-memory.jsonl').read_text().splitlines()]
        limits=report['limits'];assert limits==parent['limits']and limits['seconds']==900
        assert terminal['peak_combined_rss_bytes']==max(r['combined_rss_bytes']for r in host)<=limits['host_rss_bytes']
        assert terminal['minimum_host_available_bytes']==min(r['host_available_bytes']for r in host)>=limits['minimum_host_available_bytes']
        assert max(r['cuda_reserved_bytes']for r in memory)<=limits['cuda_reserved_bytes']
        assert min(r['cuda_available_bytes']for r in memory)>=limits['minimum_cuda_available_bytes']
        resources[stage]={'worker_seconds':report['elapsed_seconds'],'load_seconds':report['load_seconds'],
            'operation_seconds':report['sampling_seconds']if stage=='core'else report['decode_seconds'],
            'peak_combined_rss_bytes':terminal['peak_combined_rss_bytes'],
            'peak_sampled_cuda_reserved_bytes':max(r['cuda_reserved_bytes']for r in memory),
            'minimum_sampled_cuda_free_bytes':min(r['cuda_available_bytes']for r in memory),'sampled_guards_passed':True}
    assert parent['elapsed_seconds']<900
    assert before=={str(p.relative_to(ROOT)):sha(p)for p in ROOT.rglob('*')if p.is_file()}
    result={'status':'passed','scope':'Independent retained-file, original VAE byte and image-pixel audit. No Torch import, model, CUDA, cloud or key call.',
        'source_sha256':sha(__file__),'raw_files_unchanged':True,'raw_file_sha256':before,'raw_run_files':len(before),
        'source_input_and_pair_admission_identities_match':True,'matched_original_core_tensors':825,'vae_original_storage_audit':vae,
        'completed_predictions_recorded':100,'completed_solver_updates':50,'saved_step_prefixes_exact':50,'saved_step_latent_hashes_exact':50,
        'call_boundary_evidence':'The source-bound runner makes two calls per recorded solver step and resets the prefix before both. It does not retain separate tensors for all 100 call inputs.',
        'original_48_channel_normalization_source_match':True,'decoder_cache_clear_recorded':True,'finite_rgb_shape':list(rgb.shape),
        'png_frames_verified':image_rows,'contact_sheet_frame_crops_exact':10,'preview_frames':17,'preview_frame_durations_ms':durations,
        'preview_actual_mean_playback_fps':17000/sum(durations),'preview_requested_playback_fps':decode['images']['preview_playback_fps'],
        'preview_timing_note':'GIF stores delays in 10 ms units; the requested 125 ms delay can be rounded by the encoder. This is playback timing, not model throughput.',
        'parent_seconds':parent['elapsed_seconds'],'pair_estimate_seconds':expected_estimate,'resources':resources,
        'conditioned_initial_frames':1,'generated_future_frames':16,
        'future_frames_per_second_core_sampling_interval':16/core['sampling_seconds'],
        'future_frames_per_second_including_all_parent_work':16/parent['elapsed_seconds'],
        'visual_inspection':'The retained contact sheet has a clear conditioned doorway, followed by colored bands, distorted objects and warped wall/floor textures. Completion passed; future-frame visual quality failed. The audit does not identify the cause or claim intended-resolution performance.',
        'visual_inspection_artifact_sha256':sha(ROOT/'decode/result/comparison.png')}
    with OUT.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'passed','audit_sha256':sha(OUT),'source_sha256':result['source_sha256'],
        'core_tensors':825,'VAE_storages':196,'prefix_checks':50,'PNG_frames':17,'preview_ms':durations,'resources':resources},indent=2))


if __name__=='__main__':main()
