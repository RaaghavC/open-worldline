# SPDX-License-Identifier: Apache-2.0
"""Plan or explicitly execute one bounded approximate decoder comparison."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from safetensors.torch import save_file
from .. import sample_clip as limits
from ..codec_decode_profile import RUNTIME_VARIABLES
from .codec import HERE,TinyDecoder,WEIGHT_BYTES,WEIGHT_SHA256,sha,tensor_sha
from .inputs import describe_run,load_latent,load_references_after_decode,read_tensor

SOURCE_NAMES=('__init__.py','codec.py','inputs.py','run.py','test_cpu.py','vendor/__init__.py','vendor/taehv.py','vendor/abot_taehv.py','LICENSE-MIT.txt')
REUSED_NAMES=('experiments/wan22_native/codec.py','experiments/wan22_native/sample_clip.py','experiments/wan22_native/codec_profile.py',
    'experiments/wan22_native/codec_decode_profile.py','experiments/room_world/memory_train.py')
ENVIRONMENT=dict(zip(RUNTIME_VARIABLES,('0.6',None,'0',None,None)))


def source_hashes():return {name:sha(HERE/name) for name in SOURCE_NAMES}
def reused_hashes():return {name:sha(limits.REPO/name) for name in REUSED_NAMES}


def check_cpu(path):
    report=json.loads(Path(path).read_text())
    if report.get('status')!='passed' or report.get('tests',0)<8 or report.get('source_sha256')!=source_hashes() or report.get('reused_source_sha256')!=reused_hashes():
        raise ValueError('Passed source-bound tiny-decoder CPU preflight required')
    return sha(path)


def snapshot(output):
    local,reused=source_hashes(),reused_hashes()
    for base,group,values in ((HERE,'checked-source',local),(limits.REPO,'reused-source',reused)):
        for name,digest in values.items():
            destination=output/group/(name+'.txt');destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(base/name,destination)
            if sha(destination)!=digest:raise RuntimeError('Source changed during snapshot')
    return local,reused


def score(reference,prediction):
    # Caller passes RGB in [0,1], after decoder output has been produced.
    if reference.shape!=prediction.shape or not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError('Same finite RGB shapes required')
    def row(a,b):
        d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);mse=float(np.mean(d*d))
        return {'mse_rgb_0_1':mse,'mae_rgb_0_1':float(np.mean(abs(d))),'psnr_db':None if mse==0 else float(-10*np.log10(mse)),
                'exact':mse==0}
    return {'all_frames':row(reference,prediction),'first_frame':row(reference[:1],prediction[:1]),
        'later_frames':row(reference[1:],prediction[1:]),'per_frame':[dict(frame=i,**row(a,b)) for i,(a,b) in enumerate(zip(reference,prediction))]}


def save_outputs(out,rgb01,baseline,truth,kind):
    raw=rgb01.detach().cpu().contiguous()
    save_file({'rgb01':raw},str(out/'decoded.safetensors'))
    tiny=raw[0].numpy();full=((baseline[0].permute(1,0,2,3).numpy()+1)/2).clip(0,1)
    result={'full_vae_agreement':score(full,tiny),'comparison_meaning':'Agreement with full decoder, not ground-truth accuracy for generated clips'}
    if truth is not None:
        reference=truth.permute(0,3,1,2).numpy().astype(np.float32)/255
        result.update(original_rgb_reconstruction=score(reference,tiny),full_vae_reconstruction=score(reference,full))
    frames=out/'frames';frames.mkdir();images=[]
    for index,array in enumerate(tiny):
        pixels=np.rint(array.transpose(1,2,0)*255).astype(np.uint8)
        image=Image.fromarray(pixels);image.save(frames/f'{index:04d}.png');images.append(image)
    images[0].save(out/'preview.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    sheet=Image.new('RGB',(1024,4*320),(242,241,235));draw=ImageDraw.Draw(sheet)
    for row,index in enumerate((0,5,10,16)):
        draw.text((8,row*320+7),f'Full native VAE | {kind} | frame {index}',fill=(20,20,20))
        draw.text((520,row*320+7),'TAEHV approximate FP32 decoder | same latent',fill=(20,20,20))
        full_image=Image.fromarray(np.rint(full[index].transpose(1,2,0)*255).astype(np.uint8))
        sheet.paste(full_image,(0,row*320+32));sheet.paste(images[index],(512,row*320+32))
    sheet.save(out/'comparison.png')
    result['output_sha256']={str(p.relative_to(out)):sha(p) for p in [*sorted(frames.glob('*.png')),out/'decoded.safetensors',out/'preview.gif',out/'comparison.png']}
    result.update(frame_count=17,preview_playback_fps=10,extra_output_upscaling=False,frames_dropped_or_padded=0,
        gif_is_display_compressed=True,raw_rgb01_retained=True)
    return result


def worker(config):
    if config['source_sha256']!=source_hashes() or config['reused_source_sha256']!=reused_hashes():raise ValueError('Worker source changed')
    if {n:os.environ.get(n) for n in ENVIRONMENT}!=ENVIRONMENT:raise ValueError('Worker environment must be set before launch')
    device=config['device']
    if device not in ('cpu','mps'):raise ValueError('Only CPU or MPS')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(2);out=Path(config['output']);guard=limits.Guard(out,device);error=None;complete=False
    guard.report.update(kind=config['kind'],external_approximate_decoder=True,original_model=False,dtype='float32',
        source_sha256=config['source_sha256'],reused_source_sha256=config['reused_source_sha256'],
        source_evidence=config['source_evidence'],runtime_environment=ENVIRONMENT,model_training=False,
        transformer_executed=False,actions_read=False,text_read=False,extra_mean_std_transform=False,
        reference_pixels_loaded_before_decode=False)
    try:
        def deadline():
            if time.monotonic()>=config['deadline']:raise TimeoutError('900-second combined deadline')
        deadline()
        if sha(Path(config['run'])/'input.safetensors')!=config['input_file_sha256']:raise ValueError('Saved input changed')
        latent=read_tensor(Path(config['run'])/'input.safetensors','latent',{'latent'},(1,48,5,18,32),torch.float32,2**21)
        if tensor_sha(latent)!=config['latent_tensor_sha256']:raise ValueError('Normalized latent changed')
        codec=guard.measure('verify_and_load_external_fp32_tiny_decoder',lambda:TinyDecoder(config['weights'],device,check=deadline))
        guard.report['decoder']=codec.provenance;guard.report['chunks']=[]
        def chunk(row,pixels):
            deadline();guard.report['chunks'].append(row);guard.save()
        decoded,rows=guard.measure('decode_all5_latents_into17_frames',lambda:codec.decode(latent,chunks=(3,2),on_chunk=chunk))
        decoded=decoded.cpu().contiguous()
        if decoded.shape!=(1,17,3,288,512) or [r['output_frames'] for r in rows]!=[9,8] or not codec.cache_is_clear():
            raise RuntimeError('Expected9+8 native output and cleared streaming state')
        guard.report.update(cache_clear_after_decode=True,finite_output=True,decoded_shape=list(decoded.shape),
            latent_tensor_sha256=tensor_sha(latent),startup_frames_trimmed_once=3)
        # References are first materialized here, after every tiny-decoder frame exists.
        baseline,truth=load_references_after_decode(config['source_run'],config['source_evidence'])
        guard.report['references_loaded_after_decode']=True
        guard.report.update(guard.measure('compare_and_save_all17_frames',lambda:save_outputs(out,decoded,baseline,truth,config['kind'])))
        deadline();complete=True
    except BaseException as problem:error=problem;raise
    finally:
        if not complete and error is None:error=RuntimeError('Tiny decoder did not complete')
        guard.close(error)


class OneChildGuard(limits.CombinedGuard):
    def launch(self,config,wrapper):
        if time.monotonic()>=self.deadline:raise TimeoutError('900-second deadline before worker')
        wrapper.mkdir();limits.atomic_write(wrapper/'launch.json',config);error=None;began=time.monotonic()
        with (wrapper/'worker.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.wan22_native.tiny_decoder.run','--worker'],
                stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                env=limits.child_environment({'environment':ENVIRONMENT}))
            self.process=proc
            try:proc.communicate(json.dumps(config).encode())
            except BaseException as problem:error=problem;self.terminate();raise
            finally:limits.atomic_write(wrapper/'terminal.json',{'status':'failed' if error or proc.returncode!=0 else 'complete',
                'exit_code':proc.returncode,'error_type':type(error).__name__ if error else None,'elapsed_seconds':time.monotonic()-began})
        self.process=None
        if proc.returncode!=0:raise RuntimeError('Tiny decoder worker failed; evidence retained')
        report=json.loads((wrapper/'result/metrics.json').read_text())
        if report.get('status')!='passed' or report.get('frame_count')!=17 or (wrapper/'result/watchdog-stop.json').exists():
            raise RuntimeError('Tiny decoder incomplete or stopped')
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--kind',choices=('reconstruction','shift5','shift3'))
    for name in ('source-run','weights','cpu-report','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--device',choices=('cpu','mps'),default='mps');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(json.loads(sys.stdin.read(131072)));return
    if any(getattr(args,n)is None for n in ('kind','source_run','cpu_report','output')):parser.error('Kind, completed source run, CPU report and fresh output required')
    if args.execute and args.weights is None:parser.error('Execution requires the pinned local weights; downloads are not performed')
    if args.output.resolve().is_relative_to(HERE.parent):raise ValueError('Fresh output outside source package required')
    out=limits.new_directory(args.output);guard=OneChildGuard(out)
    report={'status':'running','experiment':'External approximate TAEHV FP32 decoder comparison','model_training':False,
        'kind':args.kind,'device':args.device,'fixed_chunks':[3,2],'expected_output_frames':[9,8],
        'weight_sha256_required':WEIGHT_SHA256,'weight_bytes_required':WEIGHT_BYTES,
        'max_combined_seconds':900,'max_memory_gib':18,'minimum_available_gib':2,
        'single_separate_decoder_process':True,'quality_or_speed_claim':False,
        'dependencies':{n:importlib.metadata.version(n) for n in ('torch','numpy','pillow','safetensors','psutil','tqdm')}}
    limits.atomic_write(out/'metrics.json',report)
    try:
        torch.set_num_threads(1);report['cpu_report_sha256']=check_cpu(args.cpu_report)
        evidence=describe_run(args.source_run,args.kind);latent,identity=load_latent(args.source_run,evidence)
        source,reused=snapshot(out);save_file({'latent':latent.contiguous()},str(out/'input.safetensors'))
        report.update(source_evidence=evidence,latent_identity=identity,source_sha256=source,reused_source_sha256=reused,
            input_file_sha256=sha(out/'input.safetensors'),reference_pixels_loaded_before_decode=False)
        shutil.copyfile(args.cpu_report,out/'cpu-tests.json')
        if not args.execute:report.update(status='planned',weights_loaded=False,model_execution=False);return
        if args.weights.stat().st_size!=WEIGHT_BYTES or sha(args.weights)!=WEIGHT_SHA256:raise ValueError('Exact pinned tiny weights required')
        config={'run':str(out.resolve()),'output':str((out/'decoder/result').resolve()),'weights':str(args.weights.resolve()),
            'source_run':str(args.source_run.resolve()),'kind':args.kind,'device':args.device,'deadline':guard.deadline,
            'source_sha256':source,'reused_source_sha256':reused,'source_evidence':evidence,
            'input_file_sha256':report['input_file_sha256'],'latent_tensor_sha256':identity['tensor_sha256']}
        result=guard.launch(config,out/'decoder')
        report.update(status='passed',frame_count=17,model_execution=True,decoder_metrics_sha256=sha(out/'decoder/result/metrics.json'))
    except BaseException as problem:
        report.update(status='interrupted' if isinstance(problem,KeyboardInterrupt) else 'failed',error_type=type(problem).__name__,error=str(problem));raise
    finally:
        guard.close();report.update(elapsed_seconds=time.monotonic()-guard.started,sampled_parent_limits=guard.peaks)
        limits.atomic_write(out/'metrics.json',report)


if __name__=='__main__':main()
