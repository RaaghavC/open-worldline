# SPDX-License-Identifier: Apache-2.0
"""Optional allocator cleanup after native causal convolutions; no math edits."""
from contextlib import contextmanager
import time

import torch

from .vendor.vae2_2 import CausalConv3d


@contextmanager
def cleanup_causal_convolutions(codec, callback=None, event_callback=None):
    """Retain native outputs/caches; release only unused allocator buffers.

    A provided CPU callback permits independent hook/equation tests. Production
    MPS cleanup explicitly synchronizes outstanding work before empty_cache.
    Every handle is removed even if registration, convolution or cleanup fails.
    """
    injected = callback is not None
    if callback is None:
        if codec.device.type == "mps":
            def callback():
                torch.mps.synchronize()
                torch.mps.empty_cache()
        else:
            callback = lambda: None
    layers = [(name, module) for name, module in codec.model.named_modules() if isinstance(module, CausalConv3d)]
    if not layers:
        raise ValueError("Native causal convolution modules are required")
    records = {name: {"calls": 0, "completed_cleanups": 0, "cleanup_seconds": 0.} for name, _ in layers}
    report = {"registered_layers": len(layers), "cleanup_calls": 0, "completed_cleanups": 0,
              "cleanup_seconds": 0., "layers": records, "events": [], "hooks_removed": False,
              "policy": "synchronize then empty_cache after native CausalConv3d forward",
              "backend": "injected callback" if injected else "mps" if codec.device.type == "mps" else "cpu no-op"}
    handles = []
    def hook(name):
        def after(module, args, output):
            started = time.monotonic()
            row = {"call": report["cleanup_calls"] + 1, "layer": name, "status": "running"}
            report["cleanup_calls"] += 1
            records[name]["calls"] += 1
            try:
                callback()
                row["status"] = "passed"
                report["completed_cleanups"] += 1
                records[name]["completed_cleanups"] += 1
            except BaseException as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                elapsed = time.monotonic() - started
                row["cleanup_seconds"] = elapsed
                report["cleanup_seconds"] += elapsed
                records[name]["cleanup_seconds"] += elapsed
                report["events"].append(row)
                if event_callback is not None:
                    event_callback(dict(row))
            # Return None: the native output tensor is left untouched.
        return after
    try:
        for name, module in layers:
            handle = module.register_forward_hook(hook(name))
            handles.append((module, handle))
        yield report
    finally:
        for module, handle in handles:
            handle.remove()
        report["hooks_removed"] = all(handle.id not in module._forward_hooks for module, handle in handles)
        codec.model.clear_cache()
