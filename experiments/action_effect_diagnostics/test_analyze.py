# SPDX-License-Identifier: Apache-2.0
import numpy as np
from experiments.action_effect_diagnostics.analyze import scalar_fit, patches, spatial_measurements, branch_projection, SHAPE


def test_scale_oracle_distinguishes_small_signal_from_wrong_direction():
    y = np.array([1., 0.])
    aligned = scalar_fit(np.array([0.01, 0.]), y)
    orthogonal = scalar_fit(np.array([0., 0.01]), y)
    opposite = scalar_fit(-y, y)
    assert aligned['unconstrained_oracle_gain'] == 100
    assert aligned['unconstrained_oracle_normalized_mse'] == 0
    assert orthogonal['unconstrained_oracle_normalized_mse'] == 1
    assert opposite['unconstrained_oracle_normalized_mse'] == 0
    assert opposite['nonnegative_oracle_normalized_mse'] == 1
    assert scalar_fit(np.zeros(2), y)['cosine'] is None


def test_patch_layout_preserves_every_future_value_and_explicit_coordinate():
    a = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    p = patches(a)
    assert p.shape == (4, 22, 39, 192)
    assert p[2, 7, 11, 13*4+1*2+0] == a[0, 13, 3, 15, 22]
    restored = p.reshape(4, 22, 39, 48, 2, 2).transpose(3, 0, 1, 4, 2, 5).reshape(48, 4, 44, 78)
    assert np.array_equal(restored, a[0, :, 1:])


def test_uniform_patch_response_is_separated_from_local_target():
    prediction = np.zeros(SHAPE, dtype=np.float32)
    prediction[:, :, 1:] = 1
    target = np.zeros(SHAPE, dtype=np.float32)
    target[:, :, 1:, :2, :2] = 1
    result = spatial_measurements(prediction, target)
    assert result['response_energy_in_spatial_mean_fraction'] == 1
    assert np.isclose(result['target_energy_in_spatial_mean_fraction'], 1/(22*39))
    assert result['centered_response_vs_centered_target']['cosine'] is None


def test_branch_span_fits_independent_directions_and_handles_zero_rank():
    values = {a+'-'+t: np.zeros(SHAPE, dtype=np.float32)
              for a in ('closed', 'open') for t in ('positive', 'negative')}
    values['open-positive'][0, 0, 1, 0, 0] = -1
    values['open-negative'][0, 1, 1, 0, 0] = -1
    target = np.zeros(SHAPE, dtype=np.float32)
    target[0, 0, 1, 0, 0] = 2
    target[0, 1, 1, 0, 0] = 3
    result = branch_projection(values, target)
    assert result['rank'] == 2
    assert np.allclose(result['oracle_coefficients'], [2, 3])
    assert result['oracle_normalized_mse'] < 1e-28
    for v in values.values():
        v.fill(0)
    result = branch_projection(values, target)
    assert result['rank'] == 0
    assert result['oracle_normalized_mse'] == 1
