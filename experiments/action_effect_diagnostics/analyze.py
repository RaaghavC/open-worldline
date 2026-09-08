# SPDX-License-Identifier: Apache-2.0
"""Post hoc diagnostic of retained action-effect predictions, not a new model run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

INDEX_SHA256 = '4b75683b81ea7538311801393f82506ff0e3baa48e0f84d6f20d9c1761da7626'
SHAPE = (1, 48, 5, 44, 78)
RUN = 'action-results/effect-assessment-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar_fit(prediction, target):
    """Oracle scalar uses the target; it is neither deployable nor held-out accuracy."""
    x, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Finite equal-shaped inputs required')
    xx, yy, xy = float(np.sum(x*x)), float(np.sum(y*y)), float(np.sum(x*y))
    require(yy > 0, 'Nonzero target required')
    alpha = xy/xx if xx else 0.0
    positive_alpha = max(0.0, alpha)
    return {
        'actual_normalized_mse': float(np.sum((x-y)**2)/yy),
        'unconstrained_oracle_gain': alpha,
        'unconstrained_oracle_normalized_mse': float(np.sum((alpha*x-y)**2)/yy),
        'nonnegative_oracle_gain': positive_alpha,
        'nonnegative_oracle_normalized_mse': float(np.sum((positive_alpha*x-y)**2)/yy),
        'cosine': xy/(xx*yy)**0.5 if xx else None,
        'response_rms': float(np.sqrt(xx/x.size)),
        'target_rms': float(np.sqrt(yy/y.size)),
    }


def patches(value):
    """Future latent values to [frame, patch_y, patch_x, channel*dy*dx]."""
    a = np.asarray(value)
    require(a.shape == SHAPE, 'Declared latent shape required')
    a = a[0, :, 1:]
    return a.reshape(48, 4, 22, 2, 39, 2).transpose(1, 2, 4, 0, 3, 5).reshape(4, 22, 39, 192).astype(np.float64)


def spatial_measurements(prediction, target):
    """Separate spatially repeated patch vectors from position-varying components."""
    x, y = patches(prediction), patches(target)
    xm, ym = x.mean(axis=(1, 2), keepdims=True), y.mean(axis=(1, 2), keepdims=True)
    xx, yy = float(np.sum(x*x)), float(np.sum(y*y))
    xc, yc = x-xm, y-ym
    target_energy = (y*y).sum(axis=-1)
    response_energy = (x*x).sum(axis=-1)
    mask = np.zeros(target_energy.size, dtype=bool)
    # A descriptive target-derived support, not an independently labelled door mask.
    count = int(np.ceil(mask.size * 0.1))
    mask[np.argsort(target_energy.reshape(-1), kind='stable')[-count:]] = True
    mask = mask.reshape(target_energy.shape)
    return {
        'response_energy_in_spatial_mean_fraction': float(np.sum(np.broadcast_to(xm, x.shape)**2)/xx) if xx else None,
        'target_energy_in_spatial_mean_fraction': float(np.sum(np.broadcast_to(ym, y.shape)**2)/yy),
        'centered_response_vs_centered_target': scalar_fit(xc, yc),
        'spatial_mean_prediction_normalized_mse': float(np.sum((np.broadcast_to(xm, x.shape)-y)**2)/yy),
        'oracle_spatial_mean_normalized_mse': float(np.sum(yc*yc)/yy),
        'top_target_energy_patch_fraction': count/mask.size,
        'target_energy_in_top_patches_fraction': float(target_energy[mask].sum()/yy),
        'response_energy_in_top_target_patches_fraction': float(response_energy[mask].sum()/xx) if xx else None,
    }


def guided_difference(values):
    # Preserve the experiment's explicit FP32 operation order.
    guided = {}
    for arm in ('closed', 'open'):
        p, n = values[arm+'-positive'], values[arm+'-negative']
        guided[arm] = np.add(n, np.multiply(np.float32(5), np.subtract(p, n, dtype=np.float32), dtype=np.float32), dtype=np.float32)
    return np.negative(np.subtract(guided['open'], guided['closed'], dtype=np.float32))


def branch_projection(values, target):
    """Optimistic target-fitted span of the positive/negative command contrasts.

    FP32 branch differences are promoted before fitting. The experiment's
    rounded CFG subtraction is not replaced by this diagnostic arithmetic.
    """
    columns = []
    for text in ('positive', 'negative'):
        difference = -np.subtract(values['open-'+text], values['closed-'+text], dtype=np.float32)
        columns.append(difference[:, :, 1:].astype(np.float64).reshape(-1))
    matrix = np.stack(columns, axis=1)
    y = target[:, :, 1:].astype(np.float64).reshape(-1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, y, rcond=None)
    prediction = matrix @ coefficients
    yy = float(y @ y)
    return {
        'coefficient_order': ['positive_clean_command_contrast', 'negative_clean_command_contrast'],
        'oracle_coefficients': coefficients.tolist(),
        'oracle_normalized_mse': float(np.sum((prediction-y)**2)/yy),
        'rank': int(rank),
        'singular_values': singular_values.tolist(),
        'fit_uses_target': True,
        'is_cfg_sampler': False,
    }


def analyze(download, output):
    download, output = Path(download), Path(output)
    require(not output.exists(), 'Output must be fresh')
    require(sha(download/'index.json') == INDEX_SHA256, 'Published assessment index differs')
    index = json.loads((download/'index.json').read_text())
    raw = download/'recovered'
    consumed = {}

    def checked_file(name):
        require(name in index['files'], 'Input absent from published index')
        p, expected = raw/name, index['files'][name]
        require(not p.is_symlink() and p.is_file(), 'Regular input required')
        require(p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], 'Published input bytes differ: '+name)
        consumed[name] = expected
        return p

    metrics = json.loads(checked_file(RUN+'/result/metrics.json').read_text())
    targets = load_file(checked_file(RUN+'/reference-inputs/original14/original-training-inputs.safetensors'))
    for arm in ('closed', 'open'):
        value = targets[arm+'_target']
        require(value.shape == SHAPE and value.dtype == np.float32 and np.isfinite(value).all(), 'Exact finite FP32 target required')
    target = np.subtract(targets['open_target'], targets['closed_target'], dtype=np.float32)
    require(not np.count_nonzero(target[:, :, :1]), 'Observed prefix target difference must be zero')
    cases, maps = [], {'target_patch_rms': np.sqrt((patches(target)**2).mean(axis=-1))}
    for noise in range(4):
        for checkpoint in ('zero', 'old128', 'new128'):
            values = {}
            for arm in ('closed', 'open'):
                for text in ('positive', 'negative'):
                    name = f'heldout-{noise:04d}-{checkpoint}-{arm}-{text}.safetensors'
                    loaded = load_file(checked_file(RUN+'/result/'+name))
                    require(set(loaded) == {'velocity'}, 'Exact velocity file required')
                    value = loaded['velocity']
                    require(value.shape == SHAPE and value.dtype == np.float32 and np.isfinite(value).all(), 'Exact finite FP32 velocity required')
                    values[arm+'-'+text] = value
            prediction = guided_difference(values)
            summary = scalar_fit(prediction[:, :, 1:], target[:, :, 1:])
            saved = metrics['heldout_scores'][noise*3+('zero', 'old128', 'new128').index(checkpoint)]
            require(saved['noise_index'] == noise and saved['checkpoint'] == checkpoint, 'Saved score order differs')
            require(np.isclose(summary['actual_normalized_mse'], saved['normalized_contrast_mse'], rtol=1e-12, atol=1e-15), 'Existing published score not reproduced')
            cases.append({'noise_index': noise, 'checkpoint': checkpoint, 'global_scalar': summary,
                          'per_future_latent_frame_scalar': [scalar_fit(prediction[:, :, f], target[:, :, f]) for f in range(1, 5)],
                          'spatial': spatial_measurements(prediction, target),
                          'positive_negative_span_oracle': branch_projection(values, target)})
            if checkpoint == 'new128':
                maps[f'new128_noise_{noise}_patch_rms'] = np.sqrt((patches(prediction)**2).mean(axis=-1))
    report = {
        'schema': 'worldline-action-effect-direction-diagnostic-v1',
        'status': 'computed',
        'source_sha256': sha(__file__),
        'public_assessment_index_sha256': INDEX_SHA256,
        'consumed_files': consumed,
        'existing_12_normalized_scores_reproduced': True,
        'cases': cases,
        'model_execution': False,
        'training': False,
        'target_used_for_oracle_fits': True,
        'limitations': [
            'Post hoc diagnostic on one already evaluated room; not a new held-out evaluation or acceptance gate.',
            'Oracle multipliers use the answer and cannot be deployed as measured performance.',
            'Only the first pure-noise endpoint contrast is analyzed; changing solver guidance would also change future states and may rotate later predictions.',
            'Spatial measurements concern codec latent patches, not a labelled door region or pixel segmentation.',
            'Concentration or direction errors do not by themselves prove a representational bottleneck or its cause.',
            'No evidence of Genie 3 parity, reliable controls, or scientific novelty is produced.',
        ],
    }
    output.mkdir(parents=True)
    np.savez(output/'patch-rms.npz', **maps)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'input_files': len(consumed), 'cases': len(cases), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.download, args.output)
