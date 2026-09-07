# SPDX-License-Identifier: Apache-2.0
"""Tokenwise time conditioning for the existing Wan2.1 T2V parameter layout.

Forward equations are adapted from the Apache-2.0 official Wan2.1 model and
Wan2.2 tokenwise time broadcasting. No pretrained parameters are added, changed
or loaded here. The extension is untrained for mixed token times. It does not
reproduce the pretrained Wan2.2 TI2V model or provide an image sampler.
"""
from types import MethodType
import torch
from native_control.portable import create_model as create_original_model
from native_control.portable import cpu_sinusoidal_embedding


def token_times(grid_sizes, timestep, seq_len, *, observed_latent_frames=1):
    """Return [B,L] exact integer times in patch embedding's F,H,W order.

    grid_sizes contains the post-patch grid dimensions, not pixel dimensions.
    Temporal patch size must be 1. Padding uses the corresponding future time.
    This constructs a time field only; it never edits latent values.
    """
    if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3 or grid_sizes.dtype != torch.int64:
        raise ValueError('grid_sizes must be int64 [B,3] post-patch F,H,W')
    if timestep.ndim != 1 or timestep.shape[0] != grid_sizes.shape[0] or timestep.dtype != torch.int64:
        raise ValueError('timestep must be int64 [B] with matching batch')
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError('seq_len must be positive')
    if not isinstance(observed_latent_frames, int) or observed_latent_frames < 0:
        raise ValueError('observed_latent_frames must be a nonnegative integer')
    if (timestep < 0).any() or (timestep > 1000).any():
        raise ValueError('timestep must be within [0,1000]')
    rows = []
    for i, (frames, height, width) in enumerate(grid_sizes.tolist()):
        if min(frames, height, width) <= 0 or frames*height*width > seq_len:
            raise ValueError('Invalid grid or insufficient padded sequence length')
        if observed_latent_frames >= frames:
            raise ValueError('At least one future latent frame is required')
        row = timestep[i].expand(seq_len).clone()
        row[:observed_latent_frames*height*width] = 0
        rows.append(row)
    return torch.stack(rows)


def block_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    """Same official block operations, with e[B,L,6,D]."""
    if e.dtype != torch.float32 or e.shape != (*x.shape[:2], 6, self.dim):
        raise ValueError('Block time modulation must be float32 [B,L,6,D]')
    parts = (self.modulation.unsqueeze(0) + e).unbind(dim=2)
    y = self.self_attn(self.norm1(x).float()*(1+parts[1])+parts[0],
                       seq_lens, grid_sizes, freqs)
    x = x + y*parts[2]
    x = x + self.cross_attn(self.norm3(x), context, context_lens)
    y = self.ffn(self.norm2(x).float()*(1+parts[4])+parts[3])
    return x + y*parts[5]


def head_forward(self, x, e):
    """Same official head operations, with e[B,L,D]."""
    if e.dtype != torch.float32 or e.shape != x.shape:
        raise ValueError('Head time embedding must be float32 [B,L,D]')
    shift, scale = (self.modulation.unsqueeze(0)+e.unsqueeze(2)).unbind(dim=2)
    return self.head(self.norm(x)*(1+scale)+shift)


def model_forward(self, x, t, context, seq_len):
    """Bidirectional T2V-core forward with explicit [B,L] token times.

    This intentionally has no image, adapter, action or future-target argument.
    Caller latents may contain observations, but this function does not clamp or
    sample them. FP32 is the only validated arithmetic; no autocast is applied.
    """
    device = self.patch_embedding.weight.device
    if any(p.dtype != torch.float32 for p in self.parameters()):
        raise TypeError('Only full float32 model parameters are validated')
    if not isinstance(x, list) or not x or len(x) != len(context):
        raise ValueError('x/context must be nonempty lists of the same length')
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError('seq_len must be positive')
    if t.dtype != torch.int64 or t.shape != (len(x), seq_len) or t.device != device:
        raise ValueError('t must be int64 [B,L] on the model device')
    if (t < 0).any() or (t > 1000).any():
        raise ValueError('Token timesteps must be within [0,1000]')
    for u, c in zip(x, context):
        if u.ndim != 4 or u.shape[0] != self.in_dim or u.dtype != torch.float32 or u.device != device:
            raise ValueError('Each latent must be float32 [C,F,H,W] on the model device')
        if any(n % p for n,p in zip(u.shape[1:],self.patch_size)):
            raise ValueError('Latent dimensions must divide exactly into patches')
        if c.ndim != 2 or c.shape[1] != self.text_dim or c.shape[0] > self.text_len or c.dtype != torch.float32 or c.device != device:
            raise ValueError('Each text context must be float32 [L,text_dim] within text_len')
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    patches = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.tensor([u.shape[2:] for u in patches], dtype=torch.int64)
    tokens = [u.flatten(2).transpose(1,2) for u in patches]
    seq_lens = torch.tensor([u.shape[1] for u in tokens], dtype=torch.int64)
    if seq_lens.max().item() > seq_len:
        raise ValueError('seq_len is smaller than the actual patch sequence')
    tokens = torch.cat([torch.cat([u,u.new_zeros(1,seq_len-u.shape[1],u.shape[2])],dim=1)
                        for u in tokens])
    embeddings = cpu_sinusoidal_embedding(self.freq_dim,t.flatten()).unflatten(0,(len(x),seq_len))
    e = self.time_embedding(embeddings)
    e0 = self.time_projection(e).unflatten(2,(6,self.dim))
    # This preserves the native padding-before-projection and no text mask.
    context = self.text_embedding(torch.stack([
        torch.cat([c,c.new_zeros(self.text_len-c.shape[0],c.shape[1])]) for c in context]))
    kwargs = dict(e=e0,seq_lens=seq_lens,grid_sizes=grid_sizes,freqs=self.freqs,
                  context=context,context_lens=None)
    for block in self.blocks:
        tokens = block(tokens,**kwargs)
    return [u.float() for u in self.unpatchify(self.head(tokens,e),grid_sizes)]


def extend_model(model):
    """Bind methods to this instance only; no class/global source mutation."""
    if model.model_type != 't2v' or tuple(model.patch_size) != (1,2,2):
        raise ValueError('Only T2V with patch size (1,2,2) is supported')
    if getattr(model, '_tokenwise_time_untrained', False):
        raise ValueError('Model is already extended')
    model.forward = MethodType(model_forward,model)
    for block in model.blocks:
        block.forward = MethodType(block_forward,block)
    model.head.forward = MethodType(head_forward,model.head)
    model._tokenwise_time_untrained = True
    return model


def create_model(**configuration):
    return extend_model(create_original_model(portable=True,**configuration))
