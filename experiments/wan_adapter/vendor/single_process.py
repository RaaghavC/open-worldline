"""Original single-process shim. Distributed operations are deliberately rejected."""
from types import SimpleNamespace
_STATE = SimpleNamespace(sp_enabled=False, sp=1, sp_rank=0, dp_rank=0, dp_enabled=False)
def get_parallel_state():
    return _STATE
def sp_all_gather(*args, **kwargs):
    raise RuntimeError('This isolated experiment does not implement distributed collectives')
def sp_all_to_all_4D(*args, **kwargs):
    raise RuntimeError('This isolated experiment does not implement distributed collectives')
