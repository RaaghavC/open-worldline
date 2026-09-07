# SPDX-License-Identifier: Apache-2.0
"""Separate, plan-by-default codec, pair and clip diagnostics; no cloud control."""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from safetensors.torch import save_file
from ..official_cpu.streaming import sha, tensor_sha
from ..cuda_reference.evidence import PRECISION
from .config import SPECS, SETTINGS, spec
from .evidence import sources, snapshot, validate_input_packet, validate_completed, admit_clip
from .guards import limits, atomic, hardware, Monitor, stop_child, supervise
from .output import read_tensors, save_rgb, first_frame, difference
from .sampling import pair, sample, scheduler, times_at
from .inputs import PREPARED_NAMES

HERE = Path(__file__).resolve().parent


def preflight(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 2**20:
        raise ValueError('Bounded regular CPU report required')
    report = json.loads(path.read_text())
    tests = {p.name: sha(p) for p in sorted(HERE.glob('test_*.py'))}
    if (report.get('schema') != 'wan22-spatial-cpu-review-v1' or report.get('status') != 'passed'
            or report.get('source_sha256') != sources() or report.get('test_sha256') != tests
            or report.get('failures') != 0 or report.get('errors') != 0 or report.get('skipped') != 0
            or type(report.get('tests_run')) is not int or report['tests_run'] < 50):
        raise ValueError('A passing CPU review bound to every current source and test is required')
    return sha(path)


def _identity(config):
    return {key: config[key] for key in ('mode', 'profile', 'source_sha256', 'input_manifest_sha256')}


def worker(config):
    out = Path(config['output']); out.mkdir()
    started = time.monotonic(); complete = False; error = None
    report = {**_identity(config), 'stage': config['stage'], 'status': 'running',
              'limits': limits(config['mode']), 'predictions': 0, 'solver_updates': 0,
              'actions_read': False, 'future_targets_read': False, 'quality_assessed': False,
              'original_worldline_model': False, 'model': 'External pretrained Wan2.2 TI2V-5B'}
    atomic(out / 'metrics.json', report)
    try:
        if sources() != config['source_sha256']:
            raise ValueError('Source changed after plan')
        report['hardware'] = hardware(config['expected_gpu'])
        if config.get('admitted_hardware') is not None and report['hardware'] != config['admitted_hardware']:
            raise ValueError('Hardware or environment differs from the measured admission')
        atomic(out / 'metrics.json', report)
        with Monitor(out, config['deadline'], config['mode']) as guard:
            tensors, contexts, manifest = validate_input_packet(config['inputs'], config['input_manifest_sha256'])
            input_hashes = {k: tensor_sha(v) for k, v in tensors.items()}
            context_hashes = {k: tensor_sha(v) for k, v in contexts.items()}
            if config['stage'] == 'codec':
                from .codec import load_codec, encode, decode
                begin = time.monotonic()
                model, scale, weights = load_codec(Path(config['weights']) / 'Wan2.2_VAE.pth', guard.check)
                report['load_seconds'] = time.monotonic() - begin
                atomic(out / 'weight-load.json', weights)
                report['profiles'] = {}; observations = {}
                for name, selected in SPECS.items():
                    folder = out / name; folder.mkdir()
                    guard.check(); begin = time.monotonic()
                    observation = encode(model, scale, tensors[name + '_rgb'], selected.height, selected.width)
                    guard.check(); encode_seconds = time.monotonic() - begin
                    observations[name] = observation
                    save_file({'observation': observation}, str(folder / 'observation.safetensors'))
                    # Five repeated image latents exercise 17-frame decoding. They
                    # are a codec timing input, never a generated prediction.
                    proxy = observation[0].repeat(1, 5, 1, 1)
                    save_file({'latent': proxy}, str(folder / 'timing-proxy.safetensors'))
                    begin = time.monotonic()
                    video = decode(model, scale, proxy, selected.height, selected.width)
                    guard.check(); decode_seconds = time.monotonic() - begin
                    save_rgb(video, folder / 'proxy-rgb', selected, purpose='repeated_observation_codec_timing_proxy')
                    first_frame(video, folder / 'reconstructed-initial.png')
                    mse = ((video[:, :, :1].double() - tensors[name + '_rgb'].double()) / 2).square().mean().item()
                    row = {'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                           'observation_tensor_sha256': tensor_sha(observation),
                           'timing_proxy': 'Five repeated observation latents; no denoiser or future target',
                           'initial_frame_mse_0_1': mse,
                           'initial_frame_psnr_db': -10 * math.log10(mse) if mse > 0 else None,
                           'exact_initial_rgb_tensor': mse == 0, 'raw_rgb_index': name + '/proxy-rgb/index.json'}
                    if name == 'baseline':
                        row['retained_mps_observation_difference'] = difference(observation, tensors['reference_observation'])
                    report['profiles'][name] = row
                    atomic(out / 'metrics.json', report)
                    del video, proxy
                save_file(observations, str(out / 'observations.safetensors'))
                report['decoder_cache_clear'] = all(v is None for v in model._feat_map + model._enc_feat_map)
                if not report['decoder_cache_clear']:
                    raise RuntimeError('Codec caches retained')
                del model; gc.collect(); torch.cuda.empty_cache()
            elif config['stage'] == 'core':
                from .native import load_model, predict
                selected = spec(config['profile']); name = selected.name
                observation_path = Path(config['observation_file'])
                if sha(observation_path) != config['observation_sha256']:
                    raise ValueError('Measured encoded observations changed')
                observation = read_tensors(observation_path, {k: s.observation_shape for k, s in SPECS.items()})[name]
                if tensor_sha(observation) != config['observation_tensor_sha256']:
                    raise ValueError('Encoded observation identity differs')
                noise = tensors[name + '_noise'].clone(); initial = noise.clone(); initial[:, :1] = observation[0]
                values = {'initial_noise': noise, 'initial_latent': initial, 'observation': observation,
                          'token_times': times_at(scheduler().timesteps[0], selected.latent_shape)}
                save_file(values, str(out / 'sampling-inputs.safetensors'))
                report['sampling_input_tensor_sha256'] = {k: tensor_sha(v) for k, v in values.items()}
                begin = time.monotonic(); model, weights = load_model(config['weights'], guard.check)
                torch.cuda.synchronize(); report['load_seconds'] = time.monotonic() - begin
                atomic(out / 'weight-load.json', weights); report['precision'] = PRECISION
                def call(x, times, context):
                    guard.check(); result = predict(model, x, times, context); guard.check()
                    report['predictions'] += 1
                    return result
                if config['mode'] == 'pair':
                    def partial(label, values):
                        save_file(values, str(out / f'completed-{label}.safetensors'))
                        atomic(out / 'metrics.json', report)
                    begin = time.monotonic()
                    result = pair(call, initial, values['token_times'], contexts['atrium'], contexts['native_negative'],
                                  observation, latent_shape=selected.latent_shape, event=partial)
                    report['pair_seconds'] = time.monotonic() - begin
                    save_file(result, str(out / 'outputs.safetensors'))
                else:
                    begin = time.monotonic()
                    with (out / 'steps.jsonl').open('x') as stream:
                        def event(index, t, latent, result):
                            exact = torch.equal(latent[:, :1], observation[0])
                            if not exact:
                                raise RuntimeError('Observed prefix changed')
                            save_file({'latent': latent}, str(out / f'step-{index + 1:02d}.safetensors'))
                            stream.write(json.dumps({'step': index + 1, 'timestep': int(t), 'prefix_exact': exact,
                                'latent_sha256': tensor_sha(latent), 'seconds': time.monotonic() - begin}) + '\n')
                            stream.flush(); report['solver_updates'] = index + 1; atomic(out / 'metrics.json', report)
                        latent = sample(call, values, contexts, latent_shape=selected.latent_shape, event=event)
                    report['sampling_seconds'] = time.monotonic() - begin
                    save_file({'latent': latent}, str(out / 'latents.safetensors'))
                if {k: tensor_sha(v) for k, v in values.items()} != report['sampling_input_tensor_sha256']:
                    raise RuntimeError('Caller sampling inputs changed')
                del model; gc.collect(); torch.cuda.empty_cache()
            elif config['stage'] == 'decode':
                from .codec import load_codec, decode, images
                selected = spec(config['profile']); latent_path = Path(config['latent_file'])
                if sha(latent_path) != config['latent_sha256']:
                    raise ValueError('Generated latent changed before decoding')
                latent = read_tensors(latent_path, {'latent': selected.latent_shape})['latent']
                begin = time.monotonic()
                model, scale, weights = load_codec(Path(config['weights']) / 'Wan2.2_VAE.pth', guard.check)
                report['load_seconds'] = time.monotonic() - begin; atomic(out / 'weight-load.json', weights)
                begin = time.monotonic(); video = decode(model, scale, latent, selected.height, selected.width)
                guard.check(); report['decode_seconds'] = time.monotonic() - begin
                save_rgb(video, out / 'rgb', selected, purpose='generated_clip')
                report['images'] = images(video, out, selected.height, selected.width)
                report['decoder_cache_clear'] = all(v is None for v in model._feat_map + model._enc_feat_map)
                if not report['decoder_cache_clear']:
                    raise RuntimeError('Decoder caches retained')
                del model; gc.collect(); torch.cuda.empty_cache()
            else:
                raise ValueError('Unknown worker stage')
            if ({k: tensor_sha(v) for k, v in tensors.items()} != input_hashes
                    or {k: tensor_sha(v) for k, v in contexts.items()} != context_hashes):
                raise RuntimeError('Prepared caller tensors changed')
            report['inputs_unchanged'] = True; report['finite_outputs'] = True
            guard.check()
        report['output_sha256'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in ('metrics.json', 'memory.jsonl', 'watchdog-stop.json')}
        guard.check(); complete = True
    except BaseException as caught:
        error = caught; raise
    finally:
        report.update(status='passed' if complete else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      elapsed_seconds=time.monotonic() - started,
                      error_type=type(error).__name__ if error else None, error=str(error) if error else None)
        atomic(out / 'metrics.json', report)


def launch(config, out):
    out = Path(out); out.mkdir(); atomic(out / 'launch.json', config)
    with (out / 'worker.log').open('x') as log:
        proc = None
        try:
            proc = subprocess.Popen([sys.executable, '-m', __package__ + '.run', '--worker'],
                                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            proc.stdin.write(json.dumps(config).encode()); proc.stdin.close()
            supervise(proc, out, config['deadline'], config['mode'])
        except BaseException as error:
            cleanup_error = None
            try:
                if proc is not None:
                    stop_child(proc)
            except BaseException as cleanup:
                cleanup_error = str(cleanup)
            finally:
                if not (out / 'terminal.json').exists():
                    atomic(out / 'terminal.json', {'status': 'failed', 'exit_code': proc.returncode if proc else None,
                        'error': str(error), 'cleanup_error': cleanup_error, 'stage': 'handoff'})
                elif cleanup_error:
                    atomic(out / 'handoff-cleanup-error.json', {'error': str(error), 'cleanup_error': cleanup_error})
            raise
    terminal = json.loads((out / 'terminal.json').read_text())
    report = json.loads((out / 'result/metrics.json').read_text())
    if (terminal['status'] != 'complete' or terminal['exit_code'] != 0 or report['status'] != 'passed'
            or list(out.rglob('watchdog-stop.json'))):
        raise RuntimeError('Child failed; completed and partial artifacts are retained')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('codec', 'pair', 'clip'))
    parser.add_argument('--profile', choices=('both', 'baseline', 'spatial'))
    parser.add_argument('--expected-gpu'); parser.add_argument('--input-manifest-sha256')
    for name in ('weights', 'inputs', 'cpu-report', 'output', 'codec-result', 'pair-result'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        worker(json.loads(sys.stdin.read(131072))); return
    required = ('mode', 'profile', 'expected_gpu', 'input_manifest_sha256', 'weights', 'inputs', 'cpu_report', 'output')
    if any(getattr(args, key) is None for key in required):
        parser.error('Mode, profile, expected GPU, exact packet hash and all input/review/output paths are required')
    if (args.mode == 'codec') != (args.profile == 'both'):
        parser.error('Codec requires both profiles; pair and clip require one explicit profile')
    if (args.mode != 'codec') != (args.codec_result is not None):
        parser.error('Pair and clip require a completed codec result')
    if (args.mode == 'clip') != (args.pair_result is not None):
        parser.error('Only clip requires a completed pair result')
    if args.output.exists() or args.output.resolve().is_relative_to(HERE):
        raise ValueError('Fresh output outside the source package required')
    args.output.mkdir(parents=True); started = time.monotonic()
    report = {'status': 'running', 'mode': args.mode, 'profile': args.profile,
              'execute_requested': args.execute, 'model_execution': False, 'limits': limits(args.mode),
              'settings': SETTINGS, 'child_reports': {}, 'child_terminals': {}}
    atomic(args.output / 'metrics.json', report)
    try:
        report['cpu_report_sha256'] = preflight(args.cpu_report)
        validate_input_packet(args.inputs, args.input_manifest_sha256)
        report['source_sha256'] = mapping = snapshot(args.output)
        report['input_manifest_sha256'] = args.input_manifest_sha256
        (args.output / 'inputs').mkdir()
        for name in ('manifest.json', *PREPARED_NAMES):
            shutil.copyfile(args.inputs / name, args.output / 'inputs' / name)
        validate_input_packet(args.output / 'inputs', args.input_manifest_sha256)
        shutil.copyfile(args.cpu_report, args.output / 'cpu-report.json')
        admission = None; observation = {}; admitted_hardware = None
        if args.mode != 'codec':
            validate_completed(args.codec_result, 'codec', 'both', mapping, args.input_manifest_sha256, args.expected_gpu)
            codec_report = json.loads((args.codec_result / 'codec/result/metrics.json').read_text())
            admitted_hardware = codec_report['hardware']
            observed = args.codec_result / 'codec/result/observations.safetensors'
            observed_sha = codec_report['output_sha256']['observations.safetensors']
            if sha(observed) != observed_sha:
                raise ValueError('Validated codec observations changed before copying')
            shutil.copyfile(observed, args.output / 'observations.safetensors')
            if sha(args.output / 'observations.safetensors') != observed_sha:
                raise ValueError('Copied codec observation bytes differ')
            observation = {'observation_file': str((args.output / 'observations.safetensors').resolve()),
                           'observation_sha256': observed_sha,
                           'observation_tensor_sha256': codec_report['profiles'][args.profile]['observation_tensor_sha256']}
            report['codec_result_report_sha256'] = sha(args.codec_result / 'metrics.json')
            if args.mode == 'clip':
                validate_completed(args.pair_result, 'pair', args.profile, mapping, args.input_manifest_sha256, args.expected_gpu)
                measured = json.loads((args.pair_result / 'core/result/metrics.json').read_text())
                pair_root = json.loads((args.pair_result / 'metrics.json').read_text())
                if measured['hardware'] != admitted_hardware or pair_root['codec_result_report_sha256'] != report['codec_result_report_sha256']:
                    raise ValueError('Pair used different codec evidence or hardware')
                admission = admit_clip(measured, codec_report, args.profile)
                report['pair_result_report_sha256'] = sha(args.pair_result / 'metrics.json')
        report.update(status='planned', admission=admission,
                      plan_checks='Input, source, CPU review and measured prior-stage evidence; no model or account access')
        atomic(args.output / 'metrics.json', report)
        if not args.execute:
            return report
        report.update(model_execution=None, model_execution_attempted=True)
        atomic(args.output / 'metrics.json', report)
        config = {'mode': args.mode, 'profile': args.profile, 'expected_gpu': args.expected_gpu,
                  'deadline': started + limits(args.mode)['seconds'], 'source_sha256': mapping,
                  'input_manifest_sha256': args.input_manifest_sha256, 'admitted_hardware': admitted_hardware,
                  'inputs': str((args.output / 'inputs').resolve()), 'weights': str(args.weights.resolve()), **observation}
        stages = ['codec'] if args.mode == 'codec' else ['core']
        if args.mode == 'clip':
            stages.append('decode')
        for stage in stages:
            extra = {}
            if stage == 'decode':
                latent = args.output / 'core/result/latents.safetensors'
                extra = {'latent_file': str(latent.resolve()), 'latent_sha256': sha(latent)}
            result = launch(dict(config, stage=stage, output=str((args.output / stage / 'result').resolve()), **extra), args.output / stage)
            if stage == 'core' and (result['predictions'] != (2 if args.mode == 'pair' else 100)
                                   or result['solver_updates'] != (0 if args.mode == 'pair' else 50)):
                raise RuntimeError('Incomplete prescribed denoiser or solver count')
            report['child_reports'][stage + '/result/metrics.json'] = sha(args.output / stage / 'result/metrics.json')
            report['child_terminals'][stage + '/terminal.json'] = sha(args.output / stage / 'terminal.json')
            atomic(args.output / 'metrics.json', report)
        report.update(status='passed', model_execution=True)
        atomic(args.output / 'metrics.json', report)
        validate_completed(args.output, args.mode, args.profile, mapping, args.input_manifest_sha256, args.expected_gpu)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error)); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started; atomic(args.output / 'metrics.json', report)


if __name__ == '__main__':
    main()
