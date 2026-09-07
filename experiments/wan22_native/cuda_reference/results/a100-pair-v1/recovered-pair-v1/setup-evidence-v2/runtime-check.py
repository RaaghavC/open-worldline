"""Prepared container check. No model weights or attention kernel are executed."""
import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 11), sys.version
    assert sys.platform == 'linux' and platform.machine() == 'x86_64'
    assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__
    assert torch.version.cuda == '12.4', torch.version.cuda
    assert torch._C._GLIBCXX_USE_CXX11_ABI is False
    compiler = subprocess.check_output(['nvcc', '--version'], text=True)
    assert 'release 12.4,' in compiler, compiler
    driver_rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True
    ).strip().splitlines()
    assert driver_rows
    for driver in driver_rows:
        assert re.fullmatch(r'\d+\.\d+(?:\.\d+)?', driver), driver
        parts = tuple(int(v) for v in driver.split('.'))
        assert parts + (0,) * (3 - len(parts)) >= (550, 54, 15), driver
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    assert torch.cuda.is_bf16_supported()
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory >= 70 * 1024**3, props.total_memory
    result = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cxx11abi': torch._C._GLIBCXX_USE_CXX11_ABI,
        'nvcc': compiler,
        'driver_versions': driver_rows,
        'gpu': props.name,
        'gpu_memory_bytes': props.total_memory,
        'device_count': torch.cuda.device_count(),
        'native_bf16': torch.cuda.is_bf16_supported(),
        'cudnn': torch.backends.cudnn.version(),
        'attention_kernel_executed': False,
        'foundation_model_executed': False,
    }
    if args.flash:
        import flash_attn
        import flash_attn_2_cuda
        version = importlib.metadata.version('flash-attn')
        assert version.split('+')[0] == '2.7.4.post1', version
        result.update(flash_attention=version, flash_module=flash_attn.__file__,
                      flash_cuda_extension=flash_attn_2_cuda.__file__)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
