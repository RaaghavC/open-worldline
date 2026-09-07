# SPDX-License-Identifier: Apache-2.0
"""Guarded FP32 reconstruction of one independently observed original RGB frame.

Separate process from the transformer and text encoder. This does not read an
old latent, actions, future frame or training target. No downloads occur.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
import psutil
import torch
from safetensors.torch import save_file
from wan22_native.codec import Wan22Codec,cleanup_temporal_chunks,HERE,sha

SOURCE_NAMES=('codec.py','codec-source.json','codec_profile.py','test_codec.py','vendor/vae2_2.py')
IMAGE_SHA256='7bdfa121eb2917b837af3ee1faae9e697751cdd3cdee0b21422c1b1f2a53e780'


def atomic_json(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def validate_cpu(path):
    report=json.loads(Path(path).read_text())
    if report.get('status')!='passed' or report.get('tests',0)<8:raise ValueError('Passed new-codec CPU report required')
    for name in SOURCE_NAMES:
        if report.get('source_sha256',{}).get(name)!=sha(HERE/name):raise ValueError('Codec source changed after CPU tests: '+name)
    return sha(path)


def read_image(path):
    if sha(path)!=IMAGE_SHA256:raise ValueError('Only the original open-0000 first RGB image is prescribed')
    with Image.open(path) as im:
        if im.mode not in ('RGB','RGBA') or im.size!=(512,288):raise ValueError('Expected original RGB or opaque RGBA 512x288 without resizing')
        pixels=np.array(im,dtype=np.uint8,copy=True)
        if im.mode=='RGBA':
            if not np.all(pixels[:,:,3]==255):raise ValueError('RGBA alpha must be fully opaque; no compositing is allowed')
            pixels=pixels[:,:,:3].copy()
    video=torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0).unsqueeze(2).float()/127.5-1.
    return pixels,video


class Guard:
    """CPU process monitor, with optional MPS allocator readings and one hard cap."""
    def __init__(self,output,device):
        self.output=Path(output)
        if self.output.exists() or self.output.is_symlink() or self.output.resolve().is_relative_to(HERE):raise ValueError('New output directory outside source required')
        self.output.mkdir(parents=True);self.device=device;self.started=time.perf_counter();self.stage='validate'
        self.stop=threading.Event();self.peaks={'rss_bytes':0,'mps_active_bytes':0,'mps_driver_bytes':0}
        self.report={'status':'running','max_seconds':900,'max_memory_gib':18,'minimum_available_gib':2,'timings':[]}
        if device=='mps':torch.mps.set_per_process_memory_fraction(min(1.,18*2**30/torch.mps.recommended_max_memory()))
        self.save()
        def watch():
            process=psutil.Process()
            with (self.output/'memory.jsonl').open('x') as log:
                while not self.stop.is_set():
                    row={'seconds':time.perf_counter()-self.started,'stage':self.stage,'rss_bytes':process.memory_info().rss,
                         'available_bytes':psutil.virtual_memory().available,'mps_active_bytes':0,'mps_driver_bytes':0}
                    if device=='mps':row.update(mps_active_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
                    for key in self.peaks:self.peaks[key]=max(self.peaks[key],row[key])
                    log.write(json.dumps(row)+'\n');log.flush()
                    reason=None
                    if row['seconds']>900:reason='900-second limit'
                    elif max(row['rss_bytes'],row['mps_driver_bytes'])>18*2**30:reason='18-GiB memory limit'
                    elif row['available_bytes']<2*2**30:reason='Less than2 GiB available system RAM'
                    if reason:
                        atomic_json(self.output/'watchdog-stop.json',{'status':'stopped','reason':reason,'last_sample':row})
                        os._exit(124)
                    self.stop.wait(.5)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
    def save(self):
        self.report['peaks_sampled']=dict(self.peaks);atomic_json(self.output/'metrics.json',self.report)
    def measure(self,name,function):
        self.stage=name
        if self.device=='mps':torch.mps.synchronize()
        started=time.perf_counter();result=function()
        if self.device=='mps':torch.mps.synchronize()
        self.report['timings'].append({'stage':name,'seconds':time.perf_counter()-started});self.save();return result
    def close(self,error):
        self.stop.set();self.thread.join(timeout=2)
        self.report.update(status='failed' if error else 'passed',elapsed_seconds=time.perf_counter()-self.started)
        if error:self.report.update(error_type=type(error).__name__,error=str(error))
        self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','image','cpu-report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','mps'),default='cpu')
    p.add_argument('--allocator-cleanup',action='store_true')
    args=p.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':raise RuntimeError('Automatic CPU fallback must be disabled')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4);guard=Guard(args.output,args.device);r=guard.report;error=None;completed=False
    r.update(experiment='Official Wan2.2 FP32 first-image codec reconstruction',original_model=False,external_pretrained_codec=True,
        device=args.device,dtype='float32',input_image_sha256=IMAGE_SHA256,input_rgb_frames=1,future_rgb_read=False,
        old_latent_read=False,actions_read=False,target_read=False,source_image='Original Atrium open/0000.png',data_license='CC0-1.0',
        input_transform='Original 512x288 RGB bytes from RGB or fully opaque RGBA PNG; discard alpha only when all values are 255; pixel/127.5-1; no crop, resize or compositing',
        expected_latent_shape=[1,48,1,18,32],expected_reconstruction_shape=[1,3,1,288,512],
        optional_allocator_cleanup=args.allocator_cleanup,automatic_mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0'),
        platform=platform.platform(),dependencies={n:importlib.metadata.version(n) for n in ('torch','einops','numpy','psutil','safetensors','pillow')})
    try:
        r['cpu_report_sha256']=validate_cpu(args.cpu_report)
        shutil.copyfile(args.cpu_report,args.output/'cpu-tests.json')
        r['source_sha256']={}
        for name in SOURCE_NAMES:
            dest=args.output/'measured-source'/(name+'.txt');dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(HERE/name,dest);r['source_sha256'][name]=sha(dest)
        pixels,video=guard.measure('verify_and_load_one_rgb',lambda:read_image(args.image))
        with Image.open(args.image) as source_image:
            r['input_image_metadata']={'mode':source_image.mode,'size':list(source_image.size),
                'alpha_discarded':source_image.mode=='RGBA','alpha_value':255 if source_image.mode=='RGBA' else None,
                'rgb_bytes_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
        shutil.copyfile(args.image,args.output/'original.png')
        codec=guard.measure('verify_and_load_fp32_codec',lambda:Wan22Codec(args.weights,args.device))
        r['codec']=codec.provenance
        if args.device=='mps':torch.mps.empty_cache()
        with cleanup_temporal_chunks(codec) if args.allocator_cleanup else nullcontext() as counts:
            latent=guard.measure('encode_initial_rgb_alone',lambda:codec.encode(video))
            r['cache_clear_after_encode']=codec.cache_is_clear()
            latent_cpu=latent.cpu().contiguous()
            save_file({'observation':latent_cpu},str(args.output/'observation.safetensors'))
            reconstructed=guard.measure('decode_initial_observation',lambda:codec.decode(latent))
            r['cache_clear_after_decode']=codec.cache_is_clear()
        r['allocator_chunk_counts']=counts
        if list(latent.shape)!=r['expected_latent_shape'] or list(reconstructed.shape)!=r['expected_reconstruction_shape']:raise RuntimeError('Unexpected actual codec shapes')
        if not r['cache_clear_after_encode'] or not r['cache_clear_after_decode']:raise RuntimeError('Native codec cache was retained')
        prediction=((reconstructed[0,:,0].permute(1,2,0).cpu().numpy()+1)/2).clip(0,1)
        target=pixels.astype(np.float32)/255.
        mse=float(np.mean((prediction-target)**2));mae=float(np.mean(np.abs(prediction-target)))
        r['reconstruction']={'mse_rgb_0_1':mse,'mae_rgb_0_1':mae,'psnr_db':None if mse==0 else -10*math.log10(mse),
            'scope':'Known original first-frame codec reconstruction, not generated future quality'}
        Image.fromarray(np.rint(prediction*255).astype(np.uint8)).save(args.output/'reconstruction.png')
        sheet=Image.new('RGB',(1024,288));sheet.paste(Image.fromarray(pixels),(0,0));sheet.paste(Image.open(args.output/'reconstruction.png'),(512,0));sheet.save(args.output/'comparison.png')
        r['latent_tensor_sha256']=hashlib.sha256(latent_cpu.numpy().tobytes()).hexdigest()
        r['output_sha256']={name:sha(args.output/name) for name in ('observation.safetensors','original.png','reconstruction.png','comparison.png')}
        r['finite_output']=bool(torch.isfinite(latent).all() and torch.isfinite(reconstructed).all())
        completed=True
    except BaseException as problem:error=problem;raise
    finally:
        if not completed and error is None:error=RuntimeError('Codec profile did not complete')
        guard.close(error)
    print(json.dumps({'status':r['status'],'elapsed_seconds':r['elapsed_seconds'],'reconstruction':r['reconstruction'],'peaks':r['peaks_sampled']},indent=2))

if __name__=='__main__':main()
