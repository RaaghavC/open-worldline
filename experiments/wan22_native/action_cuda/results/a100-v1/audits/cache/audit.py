"""Read-only actual-cache audit. CPU tensors only; no model or provider calls."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
BASE = Path(__file__).resolve().parent.parent
REPO = BASE / 'outputs/open-worldline'
ROOT = BASE / 'work/wan22-action-cuda-recovered-cache-v1/recovered/action-results/cache-spatial-run-v1'
CAPTURE = BASE / 'work/atrium-pilot/dense-pair'
OUT = BASE / 'work/wan22-action-cuda-cache-actual-audit-v1'
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from PIL import Image
from safetensors.numpy import load_file
from experiments.wan22_native.action_cuda import data, cache_run
from experiments.wan22_native.spatial_reference.guards import limits

torch.set_num_threads(1)
OUT.mkdir(exist_ok=False)
started = time.monotonic()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def thash(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def inventory(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}
def check(condition, message):
    if not condition: raise AssertionError(message)
def tensor_file(directory, row):
    p = directory / row['file']
    check(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(directory.resolve()), 'tensor path')
    check(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'tensor file hash/size')
    tensors = load_file(str(p))
    check(set(tensors) == set(row['tensors']), 'tensor keys')
    for name, value in tensors.items():
        check(value.dtype == np.float32 and np.isfinite(value).all(), 'finite FP32 tensor')
        check(row['tensors'][name] == {'shape': list(value.shape), 'dtype': 'float32', 'sha256': thash(value)}, 'tensor metadata')
    return tensors

report = {'schema': 'worldline-action-cuda-cache-actual-independent-v1', 'status': 'running',
          'run': str(ROOT.relative_to(BASE)), 'model_execution': False, 'cuda_initialized': False,
          'audit_source_sha256': sha(__file__), 'checks': {}}
before = inventory(ROOT)
(OUT/'audit.py').write_bytes(Path(__file__).read_bytes())
try:
    parent = read(ROOT/'metrics.json'); worker = read(ROOT/'worker/metrics.json')
    terminal = read(ROOT/'terminal.json'); monitor = read(ROOT/'worker/monitor-terminal.json')
    plan = read(ROOT/'plan.json'); completion = read(ROOT/'result/completion.json')
    manifest = read(ROOT/'result/manifest.json'); prepared = read(ROOT/'result/plan.json')
    cap = limits('codec')
    check(parent['status'] == worker['status'] == completion['status'] == manifest['status'] == 'passed', 'all stages passed')
    check(terminal['status'] == monitor['status'] == 'complete' and terminal['exit_code'] == 0, 'terminal completion')
    check(all(terminal[k] is None for k in ['error','error_type','cleanup_error','cleanup_error_type']), 'terminal errors')
    check(not list(ROOT.rglob('watchdog-stop.json')) and not list(ROOT.rglob('*cleanup-error.json')), 'no watchdog/cleanup stop')
    check(worker['codec_caches_clear'] is True, 'codec cleanup')
    for record in [parent,worker,terminal,monitor]:
        check(record['limits'] == cap and 0 < record['elapsed_seconds'] < cap['seconds'], 'time/limits')
    for record in [parent,worker]:
        check(record['profile'] == 'spatial' and record['sources_unchanged'] is True and record['inputs_unchanged'] is True, 'parent worker identity')
        check(record['quality_assessed'] is False and record['model_execution'] is True, 'scope')
        for key in ['source_sha256','input_plan_sha256','cpu_report_sha256']:
            check(record[key] == plan[key], 'plan record mismatch '+key)
        check(record['result_completion_sha256'] == sha(ROOT/'result/completion.json'), 'completion hash')
    check(parent['worker_metrics_sha256'] == sha(ROOT/'worker/metrics.json'), 'worker hash')
    check(plan['source_sha256'] == cache_run.sources(), 'current source map')
    check(cache_run.preflight(ROOT/'cpu-report.json') == plan['cpu_report_sha256'], 'CPU source gate')
    for name,digest in plan['source_sha256'].items():
        check(sha(ROOT/'source'/name) == digest == sha(REPO/name), 'retained/current source '+name)
    expected = data.plan(CAPTURE, 'spatial')
    check(expected == prepared == plan['input_plan'] == parent['input_plan'], 'original RGB/preprocessing/command plan')
    check(cache_run.canonical_sha(expected) == plan['input_plan_sha256'], 'input plan hash')
    report['checks']['source_and_original_plan'] = {'passed':True,'source_files':len(plan['source_sha256']),
        'original_rgb_files':len(expected['original_file_inventory']), 'input_plan_sha256':plan['input_plan_sha256'],
        'cpu_report_sha256':plan['cpu_report_sha256'],'current_source_sha256':plan['source_sha256']}
    original = read(CAPTURE/'manifest.json')
    check(sha(CAPTURE/'manifest.json') == '942eaf38badb1c2de5489fac59b7727e5c2a3d5699ec44aa415b7862c8440ef0', 'original capture manifest')
    rgb = {}
    for arm in ['closed','open']:
        with Image.open(CAPTURE/arm/'0000.png') as image:
            check(image.mode in ['RGB','RGBA'] and image.size == (512,288), 'original image format')
            if image.mode == 'RGBA': check(image.getchannel('A').getextrema() == (255,255), 'opaque original')
            rgb[arm] = np.array(image.convert('RGB'))
    delta = rgb['open'].astype(np.int16)-rgb['closed'].astype(np.int16)
    check(np.count_nonzero(delta)==18 and np.abs(delta).max()==1, 'canonical raw difference')
    windows = []; saved_observations = {}
    for row in manifest['observations']:
        v=tensor_file(ROOT/'result',row)['observation']; saved_observations[row['id']]=v
        check(v.shape == (1,48,1,44,78) and row['encoded_rgb_frames']==1, 'one-frame observation')
    check(len(saved_observations)==7, 'seven observations')
    for row in manifest['windows']:
        identity=row['id']; arm,start=identity.split('-');start=int(start)
        values,_ = data.read_window(ROOT/'result',identity)
        retained=tensor_file(ROOT/'result',row)
        for key,value in values.items(): check(np.array_equal(value.numpy(),retained[key]), 'reader independent tensor match')
        check(retained['target'].shape==(1,48,5,44,78), 'target native shape')
        check(np.array_equal(retained['observation'],saved_observations[row['observation_id']]), 'standalone observation matches window')
        commands=np.zeros((1,16,6),dtype=np.float32); labels=[]
        for step,raw in enumerate(original['arms'][arm][start+1:start+17]):
            command=raw['action_from_previous'];labels.append(command)
            check(command in ['left','right','wait','interact'], 'known command')
            if command in ['left','right']: commands[0,step,3]=math.pi/24*(1 if command=='left' else -1)
            if command=='interact': commands[0,step,5]=1
        check(np.array_equal(commands,retained['commands']), 'destination-aligned independent commands')
        windows.append({'id':identity,'file_sha256':row['sha256'],'tensors':row['tensors'],
                        'command_labels':labels,'observation_id':row['observation_id']})
    check(len(windows)==8 and windows[0]['observation_id']==windows[1]['observation_id']=='start-0000-shared', 'eight/shared start0')
    report['checks']['windows']={'passed':True,'count':8,'unique_observations':7,'raw_canonical_changed_channel_values':18,
                                  'raw_canonical_max_change':1,'raw_target_prefix_replaced':False,'windows':windows}
    prefixes=[]
    check([x['name'] for x in manifest['causal_checks']]==data.check_names(), 'eleven ordered checks')
    for row in manifest['causal_checks']:
        v=tensor_file(ROOT/'result',row['evidence']); a=v['candidate'];b=v['reference']
        difference=a.astype(np.float64)-b.astype(np.float64)
        maximum=float(np.max(np.abs(difference))); norm=float(np.sqrt(np.sum(b.astype(np.float64)**2)))
        numerator=float(np.sqrt(np.sum(difference**2))); relative=numerator/norm if norm else (0. if numerator==0 else None)
        exact=a.tobytes()==b.tobytes(); require_exact=not row['name'].endswith('_target_vs_independent')
        check(maximum<=1e-5 and relative is not None and relative<=1e-5 and (exact or not require_exact), 'fixed prefix gate '+row['name'])
        check(row['passed'] is True and row['max_abs']==maximum and row['exact_equal']==np.array_equal(a,b), 'recorded prefix stats')
        check(row['bit_exact_equal']==exact and row['requires_bit_exact']==require_exact, 'prefix exactness contract')
        check(row['limits']=={'max_abs':1e-5,'relative_l2':1e-5}, 'unchanged prefix bounds')
        check(math.isclose(relative,row['relative_l2'],rel_tol=1e-12,abs_tol=1e-15), 'descriptive relative norm')
        check(thash(a)==row['candidate_sha256'] and thash(b)==row['reference_sha256'], 'prefix tensor hashes')
        prefixes.append({'name':row['name'],'max_abs':maximum,'relative_l2':relative,'bit_exact_equal':exact,
                         'requires_bit_exact':require_exact,'candidate_sha256':thash(a),'reference_sha256':thash(b)})
    check(completion['encoder_calls_completed']==completion['encoder_call_attempts']=={'one_frame':8,'seventeen_frames':10,'total':18}, 'eighteen native calls')
    report['checks']['prefixes']={'passed':True,'count':11,'all_bit_exact':all(p['bit_exact_equal'] for p in prefixes),'comparisons':prefixes}
    weight=read(ROOT/'worker/weight-load.json')
    oldweight=read(BASE/'work/wan22-spatial-recovered-final-v1/recovered/spatial-results/codec-both-run-v1/codec/result/weight-load.json')
    check(weight==oldweight==completion['codec_provenance']==manifest['codec_provenance'], 'previous original VAE load records')
    check(weight['weight_sha256']=='20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36' and weight['compute_dtype']=='float32', 'original VAE identity')
    check(len(weight['tensors'])==196 and sum(math.prod(v['shape'])for v in weight['tensors'].values())==704688668 and all(v['cuda_copy_exact']is True for v in weight['tensors'].values()), 'all original VAE tensors')
    report['checks']['vae']={'passed':True,'weight_sha256':weight['weight_sha256'],'tensor_records':196,'parameters':704688668,
                           'same_as_previous_independently_audited_cuda_codec':True,'new_full_weight_read':False}
    parent_samples=[json.loads(x)for x in (ROOT/'parent-memory.jsonl').read_text().splitlines()]
    samples=[json.loads(x)for x in (ROOT/'worker/memory.jsonl').read_text().splitlines()]
    check(len(samples)==monitor['sample_count'] and parent_samples and samples, 'retained memory sample counts')
    for row in samples:
        check(row['host_rss_bytes']<=cap['host_rss_bytes'] and row['cuda_reserved_bytes']<=cap['cuda_reserved_bytes'], 'sampled memory upper caps')
        check(row['host_available_bytes']>=cap['minimum_host_available_bytes'] and row['cuda_available_bytes']>=cap['minimum_cuda_available_bytes'], 'sampled free floors')
    for row in parent_samples:
        check(row['combined_rss_bytes']<=cap['host_rss_bytes'] and row['host_available_bytes']>=cap['minimum_host_available_bytes'], 'parent memory caps')
    check(max(x['combined_rss_bytes']for x in parent_samples)==terminal['peak_combined_rss_bytes'], 'terminal sampled peak')
    check(min(x['host_available_bytes']for x in parent_samples)==terminal['minimum_host_available_bytes'], 'terminal sampled available')
    report['checks']['resources']={'passed':True,'worker_samples':len(samples),'parent_samples':len(parent_samples),
        'peak_combined_rss_bytes':terminal['peak_combined_rss_bytes'],
        'peak_cuda_reserved_bytes':max(x['cuda_reserved_bytes']for x in samples),
        'peak_cuda_allocated_bytes':max(x['cuda_allocated_bytes']for x in samples),
        'minimum_cuda_available_bytes':min(x['cuda_available_bytes']for x in samples),
        'minimum_host_available_bytes':min(x['host_available_bytes']for x in samples+parent_samples),
        'parent_elapsed_seconds':parent['elapsed_seconds'],'worker_elapsed_seconds':worker['elapsed_seconds'],
        'vae_load_seconds':worker['load_seconds'],'monitor_elapsed_seconds':monitor['elapsed_seconds'],'limits':cap,
        'hardware':worker['hardware'],'codec_caches_clear':True}
    after=inventory(ROOT);check(after==before,'recovered bytes unchanged')
    report.update(status='passed',recovered_bytes_unchanged=True,
                  recovered_files=len(before),recovered_bytes=sum(v['bytes']for v in before.values()),
                  manifest_sha256=sha(ROOT/'result/manifest.json'),completion_sha256=sha(ROOT/'result/completion.json'),
                  limitations=['Read-only retained evidence audit; no model re-execution or quality evaluation.',
                    'Memory limits are checked at recorded sampling instants.',
                    'All eight windows belong to one development layout; spatial inputs enlarge original 512x288 captures.',
                    'Exact causal-prefix results do not establish action control or future-frame reconstruction quality.',
                    'VAE value records match the prior audited original load; original checkpoint bytes were not reread.'])
except BaseException as error:
    report.update(status='failed',error_type=type(error).__name__,error=str(error))
    raise
finally:
    report['elapsed_seconds']=time.monotonic()-started
    report['cuda_initialized']=torch.cuda.is_initialized()
    (OUT/'inventory.json').write_text(json.dumps(before,indent=2)+'\n')
    report['inventory_sha256']=sha(OUT/'inventory.json')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'report':str((OUT/'report.json').relative_to(BASE)),
                     'sha256':sha(OUT/'report.json'),'elapsed_seconds':report['elapsed_seconds']}))
