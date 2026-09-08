"""Audit recovered trained-adapter visual artifacts without Torch or model replay."""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image


BASE = Path(__file__).resolve().parents[2]
REPO = BASE / 'outputs/open-worldline'
BASELINE = BASE / 'work/wan22-spatial-recovered-final-v1/recovered/spatial-results/clip-spatial-run-v1'
import binding as bindings
RGB_PATH = REPO / 'experiments/wan22_native/spatial_audit/rgb.py'
spec = importlib.util.spec_from_file_location('independent_rgb', RGB_PATH)
rgb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rgb)
SHAPE = (48, 5, 44, 78)
LIMITS = dict(seconds=1800.0, host_rss_bytes=48*2**30, cuda_reserved_bytes=60*2**30,
              minimum_host_available_bytes=8*2**30, minimum_cuda_available_bytes=8*2**30,
              minimum_gpu_total_bytes=70*2**30)


def need(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    return rgb._json(Path(path))


def sha(path):
    return rgb._sha(path)


def tensor_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array_file(path, shapes, maximum=16*2**20):
    """Check the header before reading the small FP32/int64 input payloads."""
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and 8 < path.stat().st_size <= maximum,
         'bounded regular tensor file: ' + path.name)
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        need(2 <= size <= 65536, 'bounded tensor header')
        header = rgb._parse(stream.read(size))
        header.pop('__metadata__', None)
        need(set(header) == set(shapes), 'exact tensor names: ' + path.name)
        spans = []
        for name, (shape, dtype) in shapes.items():
            row = header[name]
            need(row['shape'] == list(shape) and row['dtype'] == dtype, 'tensor shape/dtype: ' + name)
            a, b = row['data_offsets']
            need(type(a) is int and type(b) is int and 0 <= a < b
                 and b-a == math.prod(shape)*(4 if dtype == 'F32' else 8), 'exact tensor byte span')
            spans.append((a, b))
        end = 0
        for a, b in sorted(spans):
            need(a == end, 'contiguous nonoverlapping payload'); end = b
        need(8+size+end == path.stat().st_size, 'exact tensor payload length')
        values = {}
        for name, (shape, dtype) in shapes.items():
            a, b = header[name]['data_offsets']; stream.seek(8+size+a)
            value = np.frombuffer(stream.read(b-a), dtype='<f4' if dtype == 'F32' else '<i8').reshape(shape)
            need(np.isfinite(value).all(), 'finite tensor: ' + name)
            values[name] = value
    return values


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'no symlink artifacts')
        if path.is_file():
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result


def jsonlines(path):
    return [rgb._parse(line) for line in Path(path).read_bytes().splitlines() if line]


def check_rgb(result, arm, report):
    """Reuse the independent bounded shard reader, adapting only path prefixes."""
    height, width = 704, 1248
    row = report['arms'][arm]; images = row['images']
    need(images['frames'] == 17 and images['height'] == height and images['width'] == width
         and images['conditioned_initial_frames'] == 1 and images['new_future_frames'] == 16
         and images['contact_frame_indices'] == list(rgb.CONTACT)
         and images['contact_images_resized'] is False, 'native presentation contract')
    need({p.name for p in (result/arm/'frames').iterdir()} == {f'{i:04d}.png' for i in range(17)},
         'exact authoritative PNG frame set')
    contact_path = rgb._artifact(result, arm+'/comparison.png', report, rgb.IMAGE_LIMIT)
    with Image.open(contact_path) as opened:
        need(opened.format == 'PNG' and opened.mode == 'RGB'
             and opened.size == (2*width, 5*(height+28)), 'contact sheet dimensions')
        contact = opened.copy()
    def inspect(number, value):
        expected = rgb._pixels(value)
        png = rgb._artifact(result, f'{arm}/frames/{number:04d}.png', report, rgb.IMAGE_LIMIT)
        need(np.array_equal(rgb._png(png, height, width), expected), 'exact raw-to-PNG rounded pixels')
        if number in rgb.CONTACT:
            slot = rgb.CONTACT.index(number); x, y = slot%2*width, slot//2*(height+28)
            need(np.array_equal(np.asarray(contact.crop((x,y+28,x+width,y+28+height))), expected),
                 'unresized contact frame pixels')
            label = np.asarray(contact.crop((x+8,y+6,x+width,y+28)))
            need(np.any(np.max(label,axis=-1)<128), 'dark label ink in contact header')
    try:
        raw = rgb._frames(result, arm+'/rgb', 'spatial', 'generated_clip', report, inspect)
    finally:
        contact.close()
    with Image.open(rgb._artifact(result,arm+'/preview.gif',report,rgb.IMAGE_LIMIT)) as preview:
        need(preview.format == 'GIF' and preview.size == (width,height)
             and 1 <= preview.n_frames <= 17 and preview.info.get('loop') == 0, 'GIF contract')
        durations = []
        for frame in range(preview.n_frames):
            preview.seek(frame); duration = preview.info.get('duration')
            need(type(duration) is int and duration > 0 and duration%120 == 0, 'GIF timing unit')
            durations.append(duration)
    need(sum(durations)==2040 and images['preview_requested_frame_duration_ms']==120
         and images['preview_encoded_frames']==len(durations)
         and images['preview_encoded_frame_durations_ms']==durations
         and images['preview_encoded_duration_ms']==sum(durations), 'actual GIF timing metadata')
    rgb._close(images['preview_requested_playback_fps'],1000/120,'GIF requested playback FPS')
    return dict(raw_rgb=raw, png_frames_exact=17, contact_frame_regions_exact=10,
                contact_label_check='Dark header ink only; no font or glyph byte claim.',
                gif_frame_durations_ms=durations, gif_color_equality_checked=False,
                initial_conditioned_frames=1, new_future_frames=16, cache_clear=row['cache_clear'],
                decode_seconds=row['decode_seconds'])


def check_training_binding(plan, completed, training_sources, checkpoint_sha):
    """Bind the final checkpoint to the independently audited 128-update run."""
    need(plan['schema']=='worldline-action-effect128-visual-pair-v1'
         and plan['scope']=='effect128-matched-wait-interact-visual-pair'
         and plan['negative_context_adapter_training'] is True
         and plan['final_checkpoint_updates']==128 and plan['checkpoint_selection'] is False,
         'fixed final128 checkpoint eligibility')
    trained=plan['input_identity']['training']
    need(plan['artifacts']['adapter.safetensors']==trained['checkpoint_sha256']==checkpoint_sha,
         'actual final128 checkpoint bytes')
    identity={k:trained[k] for k in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
    identity.update(final_checkpoint_manifest_sha256=trained['checkpoint_manifest_sha256'],
                    final_checkpoint_sha256=checkpoint_sha,source_sha256=training_sources)
    need(completed['schema']=='worldline-action-effect128-actual-independent-v1'
         and completed['status']=='passed' and completed['completed_updates']==128
         and completed['foundation_values_unchanged'] is True and completed['final_checkpoint_only'] is True
         and completed['auxiliary_updates']==32 and completed['auxiliary_head_predictions']==128
         and completed['identity']==identity, 'exact actual128 audit and frozen training-source identity')


def check_hardware(actual, trained, recorded):
    """Apply the reviewed capacity rule; all remaining JSON values and types stay exact."""
    capacity='total_memory_bytes'
    need(isinstance(actual,dict) and isinstance(trained,dict) and set(actual)==set(trained)
         and type(actual.get(capacity)) is int and type(trained.get(capacity)) is int
         and actual[capacity]>=trained[capacity]>=LIMITS['minimum_gpu_total_bytes'],
         'actual GPU capacity at least trained capacity and fixed minimum')
    other=lambda value:json.dumps({k:v for k,v in value.items() if k!=capacity},sort_keys=True,allow_nan=False)
    need(other(actual)==other(trained), 'all other native runtime fields and JSON types equal')
    need(recorded=={'policy':'All fields exact except reported total GPU bytes may be greater',
         'training_total_memory_bytes':trained[capacity],'actual_total_memory_bytes':actual[capacity],
         'additional_reported_bytes':actual[capacity]-trained[capacity]}, 'recorded hardware comparison')


def audit(root, recovery, baseline, out, binding_path, binding_sha256):
    bound=bindings.load(binding_path,binding_sha256)
    PREP=Path(bound['prepared']); REVIEW=Path(bound['source_review']); PLAN_REVIEW=Path(bound['plan_review'])
    PLAN_SHA=bound['plan_sha256']; REVIEW_SHA=bound['source_review_sha256']; PLAN_REVIEW_SHA=bound['plan_review_sha256']
    TRAINING_AUDIT_SHA=bound['training_audit_sha256']; CHECKPOINT_SHA=bound['checkpoint_sha256']; CPU_SHA=bound['cpu_sha256']
    need(not (out/'report.json').exists(), 'fresh report required')
    began=time.monotonic(); before=inventory(root)
    report={'schema':'worldline-action-effect128-visual-actual-independent-v1','status':'running',
            'model_execution':False,'model_replay':False,'cloud_operations':False,
            'audit_source_sha256':sha(__file__),'binding_sha256':binding_sha256,'binding_source_sha256':sha(bindings.__file__),'pins_sha256':sha(bindings.HERE/'pins.json'),'reused_rgb_auditor_sha256':sha(RGB_PATH),'checks':{}}
    try:
        recovered=read(recovery); need(recovered['status']=='verified','verified complete recovery')
        need(sha(REVIEW)==REVIEW_SHA and sha(root/'plan.json')==PLAN_SHA==sha(PREP/'plan.json'),
             'exact independently reviewed final128 plan')
        plan=read(root/'plan.json'); prepared=read(PREP/'plan.json')
        need(plan==prepared and plan['limits']==LIMITS and plan['settings']=={'steps':50,'shift':5.0,'guidance':5.0},
             'unchanged native settings and fixed limits')
        for scope,mapping in plan['source_sha256'].items():
            for name,digest in mapping.items():
                need(sha(root/'source'/scope/name)==digest,'retained source '+name)
        need(read(REVIEW)['source_sha256']==plan['source_sha256'], 'reviewed actual source graph')
        need(sha(PLAN_REVIEW)==PLAN_REVIEW_SHA, 'exact actual-plan review')
        planned_review=read(PLAN_REVIEW)
        need(planned_review['status']=='passed' and planned_review['visual_plan_sha256']==PLAN_SHA
             and planned_review['actual_audit_sha256']==TRAINING_AUDIT_SHA
             and planned_review['final_checkpoint_sha256']==CHECKPOINT_SHA, 'actual final128 plan review binding')
        need(sha(root/'training-audit.json')==TRAINING_AUDIT_SHA==plan['training_audit_sha256'],
             'actual completed128 audit bytes')
        training_sources=read(root/'source/local/training-source.json')
        check_training_binding(plan,read(root/'training-audit.json'),training_sources,CHECKPOINT_SHA)
        need(sha(root/'cpu-report.json')==CPU_SHA==plan['cpu_report_sha256'], 'exact reviewed CPU report')
        cpu=read(root/'cpu-report.json')
        need(cpu['status']=='passed' and cpu['tests']==11 and cpu['source_unchanged'] is True
             and cpu['source_sha256']==plan['source_sha256'] and cpu['native_model_execution'] is False
             and cpu['cloud_actions'] is False and all(cpu[k]==0 for k in ('failures','errors','skipped')),
             'source-bound eleven-case CPU report')
        for name,digest in plan['artifacts'].items():
            need(sha(root/name)==digest==sha(PREP/name),'canonical prepared artifact '+name)
        for name,digest in plan['evidence_sha256'].items():
            need(sha(root/name)==digest,'retained prior evidence '+name)
        parent=read(root/'metrics.json'); admission=read(root/'executed-admission.json')
        need(parent['status']=='passed' and parent['model_execution'] is True and parent['model_frozen'] is True
             and parent['predictions']==200 and parent['solver_updates']==100 and parent['frames_per_arm']==17,
             'complete visual parent')
        need(parent['source_sha256']==plan['source_sha256'] and parent['plan_sha256']==PLAN_SHA
             and parent['limits']==LIMITS and parent['training'] is False and parent['quality_assessed'] is False,
             'parent scope/source/limits')
        required={'schema':'worldline-action-cuda-visual-admission-v1','decision':'admit','issued_by':'parent-agent',
                  'scope':plan['scope'],'plan_sha256':PLAN_SHA,'source_sha256':plan['source_sha256'],
                  'input_identity':plan['input_identity'],'artifacts':plan['artifacts'],'limits':LIMITS,
                  'settings':plan['settings'],'profile':'spatial','arms':['closed','open'],'training_admitted':False,
                  'final128_audit_sha256':TRAINING_AUDIT_SHA,
                  'cpu_report_sha256':plan['cpu_report_sha256']}
        need(all(admission.get(k)==v for k,v in required.items()) and bool(admission['reason'].strip()), 'exact explicit parent admission')
        need(parent['admission']=={'sha256':sha(root/'executed-admission.json'),'scope':plan['scope']}, 'executed decision hash')
        need(not list(root.rglob('watchdog-stop.json')) and not list(root.rglob('*cleanup-error.json')), 'no watchdog or cleanup failure')
        records={}; resources={}; launches={}
        for stage in ('core','decode'):
            result=root/stage/'result'; row=read(result/'metrics.json'); terminal=read(root/stage/'terminal.json')
            monitor=read(result/'monitor-terminal.json'); launch=read(root/stage/'launch.json'); launches[stage]=launch
            need(row['status']=='passed' and row['stage']==stage and row['model_execution'] is True
                 and row['training'] is False and row['future_targets_materialized'] is False, 'complete child scope')
            need(row['plan_sha256']==PLAN_SHA and row['source_sha256']==plan['source_sha256'], 'actual child source and inputs')
            need(terminal['status']==monitor['status']=='complete' and type(terminal['exit_code']) is int
                 and terminal['exit_code']==0 and terminal['cleanup_error'] is None, 'complete supervisor/monitor')
            need(parent['child_reports'][stage]==sha(result/'metrics.json')
                 and parent['child_terminals'][stage]==sha(root/stage/'terminal.json'), 'parent child hashes')
            for r in (row,terminal,monitor):
                need(r['limits']==LIMITS and 0<r['elapsed_seconds']<1800, 'child measured deadline and caps')
            need(launch['stage']==stage and launch['plan_sha256']==PLAN_SHA and launch['admission']==parent['admission'], 'child launch identity')
            outputs=row['output_sha256']; need(outputs and 'weight-load.json' in outputs,'nonempty output inventory')
            for name,digest in outputs.items(): need(before[f'{stage}/result/{name}']['sha256']==digest,'saved output '+name)
            actual=row['hardware']; trained=plan['input_identity']['training']['hardware']
            check_hardware(actual,trained,row['hardware_comparison'])
            samples=jsonlines(result/'memory.jsonl'); parents=jsonlines(root/stage/'parent-memory.jsonl')
            need(samples and parents and len(samples)==monitor['sample_count'],'resource sample counts')
            for sample in samples:
                need(sample['host_rss_bytes']<=LIMITS['host_rss_bytes'] and sample['cuda_reserved_bytes']<=LIMITS['cuda_reserved_bytes']
                     and sample['host_available_bytes']>=LIMITS['minimum_host_available_bytes']
                     and sample['cuda_available_bytes']>=LIMITS['minimum_cuda_available_bytes'], 'sampled child memory limits')
            for sample in parents:
                need(sample['combined_rss_bytes']<=LIMITS['host_rss_bytes']
                     and sample['host_available_bytes']>=LIMITS['minimum_host_available_bytes'], 'sampled combined host limits')
            need(max(x['combined_rss_bytes'] for x in parents)==terminal['peak_combined_rss_bytes'], 'recorded combined peak')
            need(min(x['host_available_bytes'] for x in parents)==terminal['minimum_host_available_bytes'], 'recorded host floor')
            need(before[f'{stage}/worker.log']['bytes']>=0,'retained child log')
            resources[stage]={'worker_seconds':row['elapsed_seconds'],'load_seconds':row['load_seconds'],
                 'parent_samples':len(parents),'worker_samples':len(samples),
                 'peak_combined_rss_bytes':terminal['peak_combined_rss_bytes'],
                 'peak_cuda_reserved_bytes':max(x['cuda_reserved_bytes'] for x in samples),
                 'minimum_cuda_available_bytes':min(x['cuda_available_bytes'] for x in samples),
                 'worker_log_sha256':sha(root/stage/'worker.log')}
            records[stage]=row
        need(launches['core']['deadline']==launches['decode']['deadline'] and 0<parent['elapsed_seconds']<1800,
             'one shared complete-run deadline')
        need(launches['decode']['core_metrics_sha256']==sha(root/'core/result/metrics.json'),'serialized core-to-decoder handoff')
        report['checks']['completion_resources']={'passed':True,'parent_seconds':parent['elapsed_seconds'],'stages':resources,'limits':LIMITS}

        values=array_file(root/'sampling-inputs.safetensors',{'initial_noise':(SHAPE,'F32'),'initial_latent':(SHAPE,'F32'),
                     'observation':((1,48,1,44,78),'F32'),'token_times':((1,4290),'I64')})
        contexts=array_file(root/'contexts.safetensors',{'atrium':((25,4096),'F32'),'native_negative':((126,4096),'F32')})
        commands=array_file(root/'commands.safetensors',{'closed':((1,16,6),'F32'),'open':((1,16,6),'F32')})
        diff=np.argwhere(commands['closed']!=commands['open']); need(diff.tolist()==[[0,0,5]],'only first interaction command differs')
        expected=np.zeros((1,16,6),dtype=np.float32); expected[0,1:,3]=np.float32(math.pi/24)
        need(np.array_equal(commands['closed'],expected),'closed branch exact wait/turn commands')
        expected[0,0,5]=1; need(np.array_equal(commands['open'],expected),'open branch exact interaction/turn commands')
        restored=values['initial_noise'].copy();restored[:,:1]=values['observation'][0]
        need(restored.tobytes()==values['initial_latent'].tobytes(),'canonical noise with independent clean observation')
        times=np.full((1,4290),999,dtype=np.int64);times[:,:858]=0
        need(np.array_equal(times,values['token_times']),'858 observed tokens time zero, remaining tokens 999')
        baseline_identity=plan['input_identity']['baseline']
        need(sha(baseline/'metrics.json')==baseline_identity['parent_sha256']
             and sha(baseline/'core/result/metrics.json')==baseline_identity['core_sha256']
             and sha(baseline/'decode/result/metrics.json')==baseline_identity['decode_sha256'], 'retained actual baseline identity')
        baseline_values=array_file(baseline/'core/result/sampling-inputs.safetensors',
                                  {k:(v.shape,'I64' if v.dtype==np.int64 else 'F32') for k,v in values.items()})
        need(all(values[k].tobytes()==baseline_values[k].tobytes() for k in values),'all baseline sampling values exact')
        need({k:tensor_sha(v) for k,v in values.items()}==baseline_identity['input_tensor_sha256'],'baseline value hashes')
        need({k:tensor_sha(v) for k,v in contexts.items()}==baseline_identity['context_tensor_sha256'],'genuine baseline context hashes')
        report['checks']['conditions']={'passed':True,'changed_command_indices':diff.tolist(),
             'input_tensor_sha256':{k:tensor_sha(v) for k,v in values.items()},
             'context_tensor_sha256':{k:tensor_sha(v) for k,v in contexts.items()},
             'commands_sha256':{k:tensor_sha(v) for k,v in commands.items()},'native_baseline_values_bit_exact':True}

        catalog=read(REPO/'experiments/wan22_native/cuda_reference/expected-weights.json')['tensors']
        original=read(root/'core/result/core-before.json'); after=read(root/'core/result/core-after.json')
        loaded=read(root/'core/result/weight-load.json'); need(original==after and len(original)==825,'825 unchanged frozen values')
        need(original==plan['input_identity']['training']['core_records'],'same trained foundation values')
        need(loaded['tensor_count']==825 and loaded['parameter_count']==4999787712 and loaded['convert_model_dtype'] is False
             and loaded['all_shards_verified'] is True and loaded['cuda_copy_exact'] is True,'original FP32 foundation load')
        need(set(loaded['tensors'])==set(original)==set(catalog),'exact 825-parameter name coverage')
        for name,row in catalog.items():
            expected={'shape':row['shape'],'dtype':'float32','sha256':row['original_sha256']}
            need(original[name]==expected,'published original parameter '+name)
            item=loaded['tensors'][name]
            need(item['shape']==row['shape'] and item['original_dtype']==item['loaded_dtype']=='float32'
                 and item['source_sha256']==item['loaded_sha256']==row['original_sha256']
                 and item['source_owner_released'] is True and item['cuda_copy_exact'] is True,'exact CUDA load '+name)
        vae=read(root/'decode/result/weight-load.json'); reference_vae=read(baseline/'decode/result/weight-load.json')
        need(read(baseline/'decode/result/metrics.json')['output_sha256']['weight-load.json']==sha(baseline/'decode/result/weight-load.json'),
             'baseline decoder weight-report source binding')
        need(vae==reference_vae and len(vae['tensors'])==196 and vae['parameters']==704688668
             and vae['compute_dtype']=='float32','same original native FP32 decoder')
        report['checks']['frozen_weights']={'passed':True,'core_tensors':825,'core_parameters':4999787712,
            'core_before_sha256':sha(root/'core/result/core-before.json'),'core_after_sha256':sha(root/'core/result/core-after.json'),
            'vae_tensors':196,'vae_parameters':704688668,'checkpoint128_sha256':plan['artifacts']['adapter.safetensors']}

        schedule=[r['timestep'] for r in jsonlines(baseline/'core/result/steps.jsonl')]
        need(len(schedule)==50 and schedule[0]==999,'retained source-verified native schedule')
        arms={}
        for arm in ('closed','open'):
            directory=root/'core/result'/arm; row=read(directory/'metrics.json')
            need(row==records['core']['arms'][arm] and row['predictions']==row['clean_prefix_calls']==100
                 and row['solver_updates']==50 and row['settings']==plan['settings'],'complete arm counters/settings')
            need(row['input_tensor_sha256']==report['checks']['conditions']['input_tensor_sha256']
                 and row['text_tensor_sha256']==report['checks']['conditions']['context_tensor_sha256']
                 and row['commands_sha256']==tensor_sha(commands[arm]),'per-arm unchanged input identity')
            need(row['adapter']==plan['input_identity']['training']['adapter']
                 and row['commands_on_cfg_branches']==['positive','native_negative']
                 and row['negative_context_adapter_training'] is True
                 and row['negative_context_training_scope']=='start0 auxiliary CFG contrast; main FM uses positive text only'
                 and row['training_scope_metadata_corrected_by']=='effect128 visual runner'
                 and row['quality_assessed'] is False,'actual adapter/CFG scope')
            steps=jsonlines(directory/'steps.jsonl');need(len(steps)==50,'50 retained chronological states')
            need({p.name for p in directory.glob('step-*.safetensors')}=={f'step-{i:02d}.safetensors' for i in range(1,51)},'exact trajectory file set')
            hashes=[]
            for number,step in enumerate(steps,1):
                latent=array_file(directory/f'step-{number:02d}.safetensors',{'latent':(SHAPE,'F32')})['latent']
                need(step['step']==number and step['timestep']==schedule[number-1] and step['prefix_exact'] is True
                     and step['latent_sha256']==tensor_sha(latent),'step order/time/retained tensor hash')
                need(np.ascontiguousarray(latent[:,:1]).tobytes()==values['observation'][0].tobytes(),'exact saved clean prefix')
                hashes.append(step['latent_sha256'])
            final=array_file(directory/'latents.safetensors',{'latent':(SHAPE,'F32')})['latent']
            need(final.tobytes()==latent.tobytes() and tensor_sha(final)==row['final_latent_sha256'],'final is retained step 50')
            velocities=array_file(directory/'initial-velocities.safetensors',{k:(SHAPE,'F32') for k in ('positive_velocity','negative_velocity','guided_velocity')})
            guided=velocities['negative_velocity']+np.float32(5)*(velocities['positive_velocity']-velocities['negative_velocity'])
            need(guided.tobytes()==velocities['guided_velocity'].tobytes(),'retained first-step exact FP32 CFG arithmetic')
            decoded=records['decode']['arms'][arm]
            need(decoded['latent_sha256']==tensor_sha(final) and decoded['cache_clear'] is True,'generated-latent decoder handoff/cache reset')
            arms[arm]={'predictions_recorded':100,'saved_prefix_checks_recomputed':50,'state_tensor_sha256':hashes,
                 'final_latent_sha256':tensor_sha(final),'sampling_seconds':row['seconds'],
                 'initial_velocity_tensor_sha256':{k:tensor_sha(v) for k,v in velocities.items()},
                 'rgb':check_rgb(root/'decode/result',arm,records['decode'])}
        need(records['core']['model_frozen'] is True and records['core']['predictions']==200 and records['core']['solver_updates']==100,'complete two-arm core')
        report['checks']['arms']=arms
        # Both decodes share the exact same causal first observation. Record the
        # observed equality rather than requiring cross-run decoder equivalence.
        firsts={a:arms[a]['rgb']['raw_rgb']['frame_records'][0] for a in arms}
        report['checks']['initial_reconstruction']={'raw_tensor_equal_between_arms':firsts['closed']['tensor_sha256']==firsts['open']['tensor_sha256'],
             'png_pixels_equal_between_arms':firsts['closed']['pixel_sha256']==firsts['open']['pixel_sha256']}
        need(inventory(root)==before,'recovered artifacts unchanged by audit')
        report.update(status='passed',identity={'plan_sha256':PLAN_SHA,'parent_sha256':sha(root/'metrics.json'),
             'core_sha256':sha(root/'core/result/metrics.json'),'decode_sha256':sha(root/'decode/result/metrics.json'),
             'admission_sha256':sha(root/'executed-admission.json'),'training_audit_sha256':TRAINING_AUDIT_SHA,
             'checkpoint128_sha256':CHECKPOINT_SHA,'source_sha256':plan['source_sha256']},
            recovery_verified_sha256=sha(recovery),recovered_files=len(before),
            recovered_bytes=sum(v['bytes'] for v in before.values()),recovered_bytes_unchanged=True,
            limitations=['Saved states establish 50 prefix checks per arm; 100 prediction boundaries per arm rely on source-bound execution counters. Only first-step raw velocities are retained.',
             'No denoiser, decoder, solver trajectory or gradient replay is performed. All array checks use recovered bytes.',
             'Raw RGB and PNG identity is separate from visual quality or correct command response; independent visual analysis is separate.',
             'Resource limits are checked at retained sampling instants. Worker deadlines and logs do not prove provider billing termination.',
             'Each arm contains one conditioned reconstruction and 16 generated future frames. GIF timing is playback, not model speed.'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began
        (out/'inventory.json').write_text(json.dumps(before,indent=2)+'\n')
        report['inventory_sha256']=sha(out/'inventory.json')
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'],'report':str(out/'report.json'),'sha256':sha(out/'report.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--binding-sha256',required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--recovery-verified',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,default=BASELINE)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source_root=args.run_root.resolve(); output=args.output.resolve()
    need(not output.exists() and not output.is_relative_to(source_root)
         and not output.is_relative_to(REPO.resolve()), 'fresh output outside measured run and repository')
    output.mkdir(parents=True)
    shutil.copyfile(__file__,output/'audit.py')
    for name in ('binding.py','pins.json'):shutil.copyfile(bindings.HERE/name,output/name)
    shutil.copyfile(args.binding,output/'binding.json')
    audit(source_root,args.recovery_verified.resolve(),args.baseline_root.resolve(),output,args.binding,args.binding_sha256)
