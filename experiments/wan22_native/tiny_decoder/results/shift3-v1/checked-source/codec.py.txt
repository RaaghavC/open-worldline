# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 wrapper around unmodified upstream TAEHV streaming equations."""
import hashlib
import time
from pathlib import Path

import torch
from .vendor.taehv import TAEHV, StreamingTAEHV

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = '99545f093b36c16297ad182ce18a5dce2ebc98bb611f5e8dfa2dce431ba39ce2'
WEIGHT_SHA256 = 'd053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a'
WEIGHT_BYTES = 22_884_021
MODEL_PARAMETERS = 11_418_108
DECODER_PARAMETERS = 9_923_532


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20), b''):digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_model():
    """Exact upstream taew2_2 architecture without a weight read."""
    if sha(HERE/'vendor/taehv.py') != SOURCE_SHA256:
        raise ValueError('Pinned literal upstream source differs')
    return TAEHV(checkpoint_path=None, arch_name='taew2_2').float().eval().requires_grad_(False)


class TinyDecoder:
    def __init__(self, weights, device='cpu', check=None):
        path = Path(weights)
        if path.stat().st_size != WEIGHT_BYTES or sha(path) != WEIGHT_SHA256:
            raise ValueError('Exact pinned 22884021-byte taew2_2 weight file required')
        if check:check()
        model = make_model()
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError('Exact upstream encoder and decoder state keys required')
        for key,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(value).all()):
                raise ValueError('Invalid pinned state tensor: '+key)
        model.load_state_dict(state, strict=True)
        for key,value in model.state_dict().items():
            if value.dtype != torch.float32 or not torch.equal(value,state[key].float()):
                raise ValueError('FP32 checkpoint conversion differs: '+key)
        del state,expected
        self._initialize(model,device)
        self.provenance.update(weight_sha256=WEIGHT_SHA256,weight_bytes=WEIGHT_BYTES,
            full_weights_verified=True,parameter_count=MODEL_PARAMETERS,decoder_parameters=DECODER_PARAMETERS)
        if check:check()

    @classmethod
    def from_model(cls,model):
        """Random CPU equation fixture only, never a production weight fallback."""
        if next(model.parameters()).device.type!='cpu':raise ValueError('Fixture must stay on CPU')
        result=cls.__new__(cls);result._initialize(model,'cpu');result.provenance['fixture_random_weights']=True
        return result

    def _initialize(self,model,device):
        if device not in ('cpu','mps'):raise ValueError('Only CPU or MPS')
        if (not isinstance(model,TAEHV) or model.latent_channels!=48 or model.patch_size!=2
                or model.t_upscale!=4 or model.frames_to_trim!=3
                or sum(p.numel() for p in model.parameters())!=MODEL_PARAMETERS
                or sum(p.numel() for p in model.decoder.parameters())!=DECODER_PARAMETERS):
            raise ValueError('Exact original taew2_2 architecture required')
        self.device=torch.device(device)
        self.model=model.float().eval().requires_grad_(False).to(self.device)
        self.stream=StreamingTAEHV(self.model)
        self.provenance={'external_model':'TAEHV taew2_2','original_worldline_model':False,
            'source_sha256':SOURCE_SHA256,'source_commit':'011dfc2112197741c540e0bdd5b7b67bcc930771',
            'dtype':'float32','autocast_enabled':False,'normalized_native48_latents_used_directly':True,
            'full_vae_inverse_mean_std_applied':False,'warmup_frames':0,
            'extra_spatial_upscaling':False,'extra_frame_padding':False,'extra_output_crop':False,
            'upstream_architecture_spatial_scale':16,'output_convention':'FP32 [B,T,3,H*16,W*16] in [0,1]'}

    def reset(self):self.stream.reset()

    def cache_is_clear(self):
        return (not self.stream.decoder_work_queue and not self.stream.encoder_work_queue
            and all(x is None for x in self.stream.decoder_memory+self.stream.encoder_memory)
            and self.stream.n_frames_decoded==self.stream.n_frames_encoded==0)

    @torch.inference_mode()
    def decode(self,normalized,*,chunks=(3,2),on_chunk=None):
        """Start a new independent clip, trim only the upstream three startup frames."""
        if (not isinstance(normalized,torch.Tensor) or normalized.dtype!=torch.float32 or normalized.ndim!=5
                or normalized.shape[0]!=1 or normalized.shape[1]!=48 or min(normalized.shape[2:])<1
                or not torch.isfinite(normalized).all()):
            raise ValueError('Finite normalized FP32[1,48,F,H,W] required')
        if not isinstance(chunks,(tuple,list)) or any(type(n)is not int or n<1 for n in chunks) or sum(chunks)!=normalized.shape[2]:
            raise ValueError('Positive chunk lengths must consume every supplied latent exactly once')
        before=tensor_sha(normalized);self.reset();outputs=[];rows=[];offset=0
        try:
            with torch.autocast(device_type=self.device.type,enabled=False):
                for index,count in enumerate(chunks):
                    if self.device.type=='mps':torch.mps.synchronize()
                    started=time.monotonic()
                    latent=normalized[:,:,offset:offset+count].permute(0,2,1,3,4).to(self.device,dtype=torch.float32,copy=True)
                    frames=[];frame=self.stream.decode(latent)
                    while frame is not None:
                        frames.append(frame);frame=self.stream.decode()
                    expected=4*count-(3 if index==0 else 0)
                    if len(frames)!=expected:raise RuntimeError('Upstream streaming output frame count differs')
                    chunk=torch.cat(frames,1)
                    if (chunk.dtype!=torch.float32 or chunk.shape!=(1,expected,3,normalized.shape[3]*16,normalized.shape[4]*16)
                            or not torch.isfinite(chunk).all() or chunk.min()<0 or chunk.max()>1):
                        raise RuntimeError('Invalid exact-size upstream FP32 output')
                    if self.device.type=='mps':torch.mps.synchronize()
                    row={'decode_and_transfer_seconds':time.monotonic()-started,'chunk':index,'latent_start':offset,'latent_count':count,'output_frames':expected,
                         'startup_frames_trimmed':3 if index==0 else 0,'output_shape':list(chunk.shape)}
                    rows.append(row);outputs.append(chunk);offset+=count
                    if on_chunk:on_chunk(row,chunk)
            output=torch.cat(outputs,1)
            if tensor_sha(normalized)!=before:raise RuntimeError('Normalized input was mutated')
            return output,rows
        finally:self.reset()
