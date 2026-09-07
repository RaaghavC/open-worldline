# SPDX-License-Identifier: Apache-2.0
"""Source gates and one bounded separate codec process, with no downloads."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..codec import HERE as CODEC_ROOT, WEIGHT_SHA256, SOURCE_SHA256, CONFIG
from ..codec_profile import atomic_json, validate_cpu as validate_codec_cpu
from ..codec_decode_profile import validate_decode_cpu, validate_memory_cpu, RUNTIME_VARIABLES
from .data import HERE, REPO, sha

LOCAL_SOURCES = ('__init__.py','data.py','operations.py','cache.py','runtime.py','worker.py','build.py','roundtrip.py','test_data.py','test_codec_path.py','source-plan.md.txt')
REUSED_SOURCES = ('experiments/wan22_native/codec.py','experiments/wan22_native/codec-source.json',
    'experiments/wan22_native/vendor/vae2_2.py','experiments/wan22_native/codec_profile.py',
    'experiments/wan22_native/codec_decode_profile.py','experiments/wan22_native/codec_memory.py',
    'experiments/wan_adapter/capture_data.py')
ENVIRONMENT = dict(zip(RUNTIME_VARIABLES,('0.6',None,'0',None,None)))


def source_hashes():
    return {**{'action_data/'+n:sha(HERE/n) for n in LOCAL_SOURCES},
            **{n:sha(REPO/n) for n in REUSED_SOURCES}}


def validate_preflight(path):
    r = json.loads(Path(path).read_text())
    if r.get('status') != 'passed' or r.get('tests',0) < 10 or r.get('source_sha256') != source_hashes():
        raise ValueError('Passed source-matching action-data CPU report required')
    if r.get('measurements',{}).get('full_17_encode_cleanup_max_abs') != 0.0 or r.get('measurements',{}).get('full_17_decode_cleanup_max_abs') != 0.0:
        raise ValueError('Exact 17-frame encode/decode cleanup equality required')
    return sha(path)


def validate_decoder_run(directory):
    directory = Path(directory)
    r = json.loads((directory/'metrics.json').read_text())
    if (r.get('status') != 'passed' or r.get('finite_output') is not True or (directory/'watchdog-stop.json').exists()
            or r.get('dtype') != 'float32' or r.get('decoded_shape') != [1,3,17,288,512]
            or r.get('per_convolution_cleanup') is not True or r.get('optional_allocator_cleanup') is not True
            or not r.get('cache_clear_after_decode')):
        raise ValueError('Completed full-shape cleanup decoder evidence required')
    codec = r.get('codec',{})
    if codec.get('weight_sha256') != WEIGHT_SHA256 or codec.get('source_sha256') != SOURCE_SHA256 or codec.get('config') != CONFIG:
        raise ValueError('Decoder weights or native configuration differ')
    for name in ('codec.py','codec-source.json','vendor/vae2_2.py','codec_profile.py','codec_decode_profile.py','codec_memory.py'):
        if r.get('source_sha256',{}).get(name) != sha(CODEC_ROOT/name):
            raise ValueError('Decoder measured source differs: '+name)
    cleanup = r.get('per_convolution_cleanup_report',{})
    if (cleanup.get('hooks_removed') is not True or cleanup.get('cleanup_calls',0) < 1
            or cleanup.get('cleanup_calls') != cleanup.get('completed_cleanups')):
        raise ValueError('Incomplete decoder cleanup')
    if r.get('runtime_environment',{}).get('environment') != ENVIRONMENT:
        raise ValueError('Different measured decoder process environment')
    seconds = r.get('elapsed_seconds',0)
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0 < seconds < 900:
        raise ValueError('Invalid completed decoder duration')
    return {'metrics_sha256':sha(directory/'metrics.json'),'elapsed_seconds':seconds,
            'runtime_environment':r['runtime_environment'],'policy':'Per-native-convolution sync/empty_cache plus temporal chunk cleanup'}


def gates(cpu_report, decoder_run):
    return {'action_data_cpu_sha256':validate_preflight(cpu_report),
        'initial_codec_cpu_sha256':validate_codec_cpu(CODEC_ROOT/'codec-results/cpu-v3/tests.json'),
        'decode_cpu_sha256':validate_decode_cpu(CODEC_ROOT/'codec-results/decode-cpu-v4/tests.json'),
        'memory_cpu_sha256':validate_memory_cpu(CODEC_ROOT/'codec-results/memory-cpu-v1/tests.json'),
        'completed_decoder':validate_decoder_run(decoder_run)}


def snapshot(output):
    hashes = source_hashes()
    for name in LOCAL_SOURCES:
        destination = output/'measured-source'/'action_data'/(name+'.txt')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,destination)
    for name in REUSED_SOURCES:
        destination = output/'measured-source'/(name+'.txt')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,destination)
    return hashes


def new_output(path):
    path = Path(path).resolve()
    if path.exists() or path.is_symlink() or path.is_relative_to(CODEC_ROOT):
        raise ValueError('Fresh output outside the source package required')
    path.mkdir(parents=True)
    return path


def run_child(output, request):
    """Parent wall limit covers both launch and the only codec process."""
    output = Path(output); started = time.monotonic()
    env = os.environ.copy()
    for name,value in ENVIRONMENT.items():
        if value is None: env.pop(name,None)
        else: env[name] = value
    env['PYTHONUNBUFFERED'] = '1'
    atomic_json(output/'request.json',request)
    atomic_json(output/'launch.json',{'worker_environment':{n:env.get(n) for n in RUNTIME_VARIABLES},
        'max_seconds':900,'max_memory_gib':18,'minimum_available_gib':2,
        'source_sha256':source_hashes(),'single_separate_codec_process':True})
    process = None; error = None; terminal = {'status':'running'}
    try:
        with (output/'stdout.log').open('x') as stdout,(output/'stderr.log').open('x') as stderr:
            process = subprocess.Popen([sys.executable,'-m','experiments.wan22_native.action_data.worker',
                '--request',str(output/'request.json'),'--output',str(output/'result')],cwd=REPO,env=env,stdout=stdout,stderr=stderr)
            while process.poll() is None:
                if time.monotonic()-started >= 900:
                    raise TimeoutError('900-second total process limit')
                time.sleep(.2)
            terminal = {'status':'complete' if process.returncode==0 else 'failed','exit_code':process.returncode}
            if process.returncode != 0:
                raise RuntimeError('Codec child failed; inspect retained stderr, metrics and watchdog records')
    except BaseException as problem:
        error = problem
        terminal.update(status='failed',error_type=type(problem).__name__,error=str(problem))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=3)
        terminal.update(elapsed_seconds=time.monotonic()-started,exit_code=process.returncode if process else None)
        atomic_json(output/'terminal.json',terminal)
