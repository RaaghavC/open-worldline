# SPDX-License-Identifier: Apache-2.0
"""Audit recovered results on CPU; never launch a model or contact a provider."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

from ..spatial_reference.evidence import sources as runtime_sources, validate_completed, validate_input_packet
from .weights import audit_vae_weights, sha

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAX_RUN_BYTES = 2 * 2**30


def sources():
    result = runtime_sources()
    root = HERE.parents[2]
    for name in ('__init__.py', 'weights.py', 'predictions.py', 'rgb.py', 'run.py'):
        path = HERE / name
        result[path.relative_to(root).as_posix()] = sha(path)
    return result


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inventory(path):
    root = Path(path).absolute()
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('A regular result directory without symlink ancestors is required')
    files = {}; total = 0
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            item = Path(folder) / name
            if item.is_symlink():
                raise ValueError('Result symlinks are not accepted')
        for name in names:
            item = Path(folder) / name
            if not item.is_file():
                raise ValueError('Every retained artifact must be a regular file')
            size = item.stat().st_size; total += size
            if size > 100_000_000 or total > MAX_RUN_BYTES:
                raise ValueError('Declared split-result file or directory size bound exceeded')
            files[item.relative_to(root).as_posix()] = {'bytes': size, 'sha256': sha(item)}
    if not files:
        raise ValueError('An actual nonempty retained run is required')
    return files


def obj(path):
    if path.stat().st_size > 4 * 2**20:
        raise ValueError('Bounded JSON report required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON report key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('A retained report object is required')
    return value


def check_prior(clip_root, pair_root, codec_root, profile):
    """Recompute the admission arithmetic and bind the exact prior reports."""
    clip = obj(clip_root / 'metrics.json'); pair = obj(pair_root / 'metrics.json')
    pair_core = obj(pair_root / 'core/result/metrics.json')
    codec = obj(codec_root / 'codec/result/metrics.json')
    clip_core = obj(clip_root / 'core/result/metrics.json')
    clip_decode = obj(clip_root / 'decode/result/metrics.json')
    if (clip.get('pair_result_report_sha256') != sha(pair_root / 'metrics.json')
            or clip.get('codec_result_report_sha256') != sha(codec_root / 'metrics.json')
            or pair.get('codec_result_report_sha256') != sha(codec_root / 'metrics.json')
            or any(row['hardware'] != codec['hardware'] for row in (pair_core, clip_core, clip_decode))):
        raise ValueError('Clip must use the exact completed codec and pair evidence')
    values = {'core_load_seconds': pair_core['load_seconds'], 'pair_seconds': pair_core['pair_seconds'],
              'codec_load_seconds': codec['load_seconds'], 'decode_seconds': codec['profiles'][profile]['decode_seconds']}
    for value in values.values():
        if type(value) not in (int, float) or value <= 0:
            raise ValueError('Positive measured admission times required')
        try:
            if not math.isfinite(value):
                raise ValueError('Finite measured admission times required')
        except OverflowError as error:
            raise ValueError('Bounded finite admission times required') from error
    expected = values['core_load_seconds'] + 100.0 * values['pair_seconds'] + 1.2 * (
        values['codec_load_seconds'] + values['decode_seconds']) + 30
    reported = clip.get('admission', {})
    if (not math.isfinite(expected) or expected >= 1800 or reported.get('estimated_seconds') != expected
            or reported.get('limit_seconds') != 1800 or reported.get('profile') != profile
            or reported.get('measured_seconds') != values
            or reported.get('factors') != {'pairs': 50, 'pair_safety': 2.0, 'codec_safety': 1.2, 'artifact_seconds': 30}):
        raise ValueError('Recorded clip admission differs from recomputed measured estimate')
    return {'status': 'passed', 'estimated_seconds_recomputed': expected,
            'limit_seconds': 1800, 'prior_report_hashes_match': True,
            'does_not_verify_provider_deadline_or_billing': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('codec', 'pair', 'clip'), required=True)
    parser.add_argument('--profile', choices=('both', 'baseline', 'spatial'), required=True)
    parser.add_argument('--input-manifest-sha256', required=True)
    parser.add_argument('--expected-gpu', required=True)
    for name in ('run-root', 'packet-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('codec-root', 'pair-root', 'vae-weights'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if (args.mode == 'codec') != (args.profile == 'both'):
        parser.error('Codec audits require both profiles; pair and clip require one profile')
    if (args.mode != 'codec') != (args.codec_root is not None):
        parser.error('Pair and clip audits require their completed codec result')
    if (args.mode == 'clip') != (args.pair_root is not None):
        parser.error('Only clip audits require their completed pair result')
    if (args.mode in ('codec', 'clip')) != (args.vae_weights is not None):
        parser.error('Codec and clip audits require the original local VAE checkpoint')
    output = args.output.absolute()
    roots = {'run': args.run_root.absolute(), 'packet': args.packet_root.absolute()}
    if args.codec_root is not None:
        roots['codec'] = args.codec_root.absolute()
    if args.pair_root is not None:
        roots['pair'] = args.pair_root.absolute()
    if (output.exists() or output.resolve().is_relative_to(REPO)
            or any(p.is_symlink() for p in (output, *output.parents))
            or any(output.resolve().is_relative_to(root.resolve()) for root in roots.values())):
        raise ValueError('Fresh audit output outside source and measured directories required')
    output.mkdir(parents=True); started = time.monotonic()
    report = {'schema': 'wan22-spatial-independent-audit-v1', 'status': 'running', 'mode': args.mode,
              'profile': args.profile, 'new_model_execution': False, 'provider_access': False,
              'quality_assessed': False, 'input_manifest_sha256': args.input_manifest_sha256,
              'checks': {}, 'limitations': ['Passing an artifact audit does not establish visual quality, action control or a research advance.']}
    atomic(output / 'report.json', report)
    try:
        before = {name: inventory(root) for name, root in roots.items()}
        atomic(output / 'input-inventory.json', before)
        report['input_inventory_sha256'] = sha(output / 'input-inventory.json')
        report['source_sha256'] = source_map = sources()
        source_root = output / 'source'
        for name, digest in source_map.items():
            original = HERE.parents[2] / name; target = source_root / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, target)
            if sha(target) != digest:
                raise ValueError('Audit source changed during snapshot')
        report['retained_files'] = {name: {'count': len(rows), 'bytes': sum(row['bytes'] for row in rows.values())}
                                    for name, rows in before.items()}
        validate_input_packet(roots['packet'], args.input_manifest_sha256)
        mapping = runtime_sources()
        validate_completed(roots['run'], args.mode, args.profile, mapping,
                           args.input_manifest_sha256, args.expected_gpu)
        if args.mode != 'codec':
            validate_completed(roots['codec'], 'codec', 'both', mapping,
                               args.input_manifest_sha256, args.expected_gpu)
        if args.mode == 'codec':
            from .rgb import audit_codec
            report['checks']['codec'] = audit_codec(roots['run'], roots['packet'])
            atomic(output / 'report.json', report)
            report['checks']['original_vae_storages'] = audit_vae_weights(args.vae_weights,
                [roots['run'] / 'codec/result/weight-load.json'])
        elif args.mode == 'pair':
            from .predictions import audit_pair
            report['checks']['predictions'] = audit_pair(roots['run'], roots['codec'], roots['packet'], args.profile)
        else:
            from .predictions import audit_pair, audit_clip
            from .rgb import audit_decoded_clip
            validate_completed(roots['pair'], 'pair', args.profile, mapping,
                               args.input_manifest_sha256, args.expected_gpu)
            report['checks']['admitted_pair'] = audit_pair(roots['pair'], roots['codec'], roots['packet'], args.profile)
            report['checks']['admission'] = check_prior(roots['run'], roots['pair'], roots['codec'], args.profile)
            report['checks']['predictions'] = audit_clip(roots['run'], roots['codec'], roots['packet'], args.profile)
            atomic(output / 'report.json', report)
            report['checks']['decoded_rgb'] = audit_decoded_clip(roots['run'], args.profile)
            atomic(output / 'report.json', report)
            report['checks']['original_vae_storages'] = audit_vae_weights(args.vae_weights,
                [roots['codec'] / 'codec/result/weight-load.json', roots['run'] / 'decode/result/weight-load.json'])
        if source_map != sources() or any(inventory(roots[name]) != rows for name, rows in before.items()):
            raise ValueError('Audited sources or artifacts changed during the audit')
        report.update(status='passed', sources_unchanged=True, inputs_unchanged=True)
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        atomic(output / 'report.json', report)


if __name__ == '__main__':
    main()
