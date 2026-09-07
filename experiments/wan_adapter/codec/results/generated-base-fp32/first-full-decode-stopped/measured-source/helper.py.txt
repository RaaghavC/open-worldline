"""Original portable loading and encode/decode helper for the official Wan VAE."""
import hashlib
from pathlib import Path
import torch
from .vendor.wan_vae import WanVAE_

VAE_SHA256 = '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'
MEAN = [-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]
STD = [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()


class OfficialWanCodec:
    """Pretrained external VAE, with explicit dtype and no CUDA-only autocast.

    Normalization constants and network configuration match the pinned official
    WanVAE wrapper. Loading uses weights_only=True and mmap=True. Source neural
    layers and their original temporal cache behavior are unchanged.
    """
    def __init__(self, weights, device='mps', dtype=torch.float32):
        if Path(weights).stat().st_size != 507609880 or sha(weights) != VAE_SHA256:
            raise ValueError('Official pinned VAE SHA256 does not match')
        self.device=torch.device(device)
        self.dtype=dtype
        with torch.device('meta'):
            self.model=WanVAE_(dim=96,z_dim=16,dim_mult=[1,2,4,4],num_res_blocks=2,
                attn_scales=[],temperal_downsample=[False,True,True],dropout=0.)
        state=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
        loaded=self.model.load_state_dict(state,strict=True,assign=True)
        self.loaded_keys=len(state)
        self.missing_keys=loaded.missing_keys
        self.unexpected_keys=loaded.unexpected_keys
        self.model=self.model.eval().requires_grad_(False).to(device=self.device,dtype=dtype)
        self.scale=[torch.tensor(MEAN,device=self.device,dtype=dtype),
                    torch.tensor(STD,device=self.device,dtype=dtype).reciprocal()]

    @torch.inference_mode()
    def encode(self, video):
        """Input [B,3,T,H,W] in [-1,1]; T must equal 1 modulo four."""
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] % 4 != 1:
            raise ValueError('Expected [B,3,1+4n,H,W] RGB video')
        return self.model.encode(video.to(self.device,self.dtype),self.scale)

    @torch.inference_mode()
    def decode(self, latent):
        """Return [B,3,T,H,W] reconstruction clamped to official [-1,1]."""
        if latent.ndim != 5 or latent.shape[1] != 16:
            raise ValueError('Expected [B,16,F,H/8,W/8] Wan latents')
        return self.model.decode(latent.to(self.device,self.dtype),self.scale).float().clamp(-1,1)
