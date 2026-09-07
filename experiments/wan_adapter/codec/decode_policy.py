# SPDX-License-Identifier: Apache-2.0
"""Allocator-only hook around the official decoder's existing temporal chunks."""
from contextlib import contextmanager
import torch


@contextmanager
def cleanup_after_temporal_chunk(codec):
    """Leave feature caches and decoder math intact; release unused MPS buffers.

    WanVAE_.decode already calls Decoder3d once for each latent time index. This
    forward hook runs after that call and never mutates an input or an output.
    On CPU it counts the identical calls without allocator operations.
    """
    evidence = {'completed_temporal_chunks': 0}
    def after_chunk(module, inputs, output):
        if codec.device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
        evidence['completed_temporal_chunks'] += 1
    hook = codec.model.decoder.register_forward_hook(after_chunk)
    try:
        yield evidence
    finally:
        hook.remove()
