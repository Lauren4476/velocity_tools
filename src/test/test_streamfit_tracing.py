import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from velocity_tools import extract_streamline
from velocity_tools import gradient_descent

jax.config.update("jax_enable_x64", True)


def _fake_forward_model(opt_params, fixed_params, distance_pc):
    """Small synthetic model with duplicate points to exercise tie tracing."""
    shift = 1e-3 * opt_params['r0']
    ra_model = jnp.array([2.0, 1.5, 1.5, 1.0, 0.5], dtype=jnp.float64) + shift
    dec_model = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    v_model = jnp.array([7.20, 7.00, 7.00, 6.80, 6.60], dtype=jnp.float64)
    return ra_model, dec_model, v_model


def test_get_distance_metric_trace_metadata() -> None:
    ra = jnp.array([3.0, 2.0, 1.0, 0.5, 0.25], dtype=jnp.float64)
    dec = jnp.zeros_like(ra)

    distance_metric, trace = extract_streamline.get_distance_metric(ra, dec, return_trace=True)

    assert distance_metric.shape == ra.shape
    assert distance_metric.dtype == jnp.float64
    assert trace['n_points'] == 5
    assert np.all(np.isfinite(np.asarray(distance_metric)))


def test_get_distance_metric_branch_cut_inputs_remain_float64() -> None:
    r = jnp.array([0.45, 0.55, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64)
    theta = jnp.array([
        jnp.pi - 0.03,
        -jnp.pi + 0.02,
        -1.0,
        -0.6,
        0.4,
        1.0,
    ], dtype=jnp.float64)
    ra = r * jnp.cos(theta)
    dec = r * jnp.sin(theta)

    distance_metric, trace = extract_streamline.get_distance_metric(ra, dec, return_trace=True)

    assert distance_metric.dtype == jnp.float64
    assert trace['n_points'] == 6
    assert np.all(np.isfinite(np.asarray(distance_metric)))


def test_match_model_to_data_curve_trace_detects_duplicate_metric() -> None:
    ra_model = jnp.array([3.0, 2.0, 2.0, 2.0, 1.0, 0.5], dtype=jnp.float64)
    dec_model = jnp.zeros_like(ra_model)
    v_model = jnp.array([7.0, 6.8, 6.8, 6.8, 6.4, 6.2], dtype=jnp.float64)

    ra_data = jnp.array([2.6, 1.6, 0.8], dtype=jnp.float64)
    dec_data = jnp.zeros_like(ra_data)

    ra_interp, dec_interp, v_interp, valid, trace = gradient_descent.match_model_to_data_curve(
        ra_model, dec_model, v_model, ra_data, dec_data, return_trace=True
    )

    assert ra_interp.dtype == jnp.float64
    assert dec_interp.dtype == jnp.float64
    assert v_interp.dtype == jnp.float64
    assert valid.shape[0] == ra_data.shape[0]
    assert trace['model_nan_count'] == 0
    assert trace['model_metric_duplicate_count'] >= 1
    assert trace['model_metric_near_tie_count'] >= trace['model_metric_duplicate_count']
    assert 'distance_metric_model' in trace
    assert 'distance_metric_data' in trace


def test_chi2_loss_returns_trace(monkeypatch) -> None:
    monkeypatch.setattr(gradient_descent, 'forward_model', _fake_forward_model)

    opt_params = {
        'r0': 540.0,
        'theta0': 0.7,
        'phi0': 1.2,
        'log_omega': np.log(4e-12),
        'v_r0': -0.2,
    }
    fixed_params = {
        'mass': 3.2,
        'inc': -0.8,
        'pa': 2.4,
        'rmin': 50.0,
        'deltar': 40.0,
        'v_lsr': 7.0,
    }
    data = (
        jnp.array([2.2, 1.7, 0.9], dtype=jnp.float64),
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.array([7.15, 6.95, 6.70], dtype=jnp.float64),
    )
    uncertainties = (
        jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64),
        jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64),
        jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64),
    )

    loss, trace = gradient_descent.chi2_loss(
        opt_params, fixed_params, data, uncertainties, 147.0, return_trace=True
    )

    assert np.isfinite(float(loss))
    assert jnp.asarray(loss).dtype == jnp.float64
    assert 'chi2_components' in trace
    assert 'matching' in trace
    assert trace['matching']['model_nan_count'] == 0
    assert trace['matching']['model_metric_duplicate_count'] >= 1
    assert trace['chi2_components']['chi2_total'] == pytest.approx(float(loss))


def test_fit_streamline_writes_trace_csv(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gradient_descent, 'forward_model', _fake_forward_model)

    initial_opt_params = {
        'r0': 540.0,
        'theta0': 0.7,
        'phi0': 1.2,
        'log_omega': np.log(4e-12),
        'v_r0': -0.2,
    }
    fixed_params = {
        'mass': 3.2,
        'inc': -0.8,
        'pa': 2.4,
        'rmin': 50.0,
        'deltar': 40.0,
        'v_lsr': 7.0,
    }
    data = (
        jnp.array([2.2, 1.7, 0.9], dtype=jnp.float64),
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.array([7.15, 6.95, 6.70], dtype=jnp.float64),
    )
    uncertainties = (
        jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64),
        jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64),
        jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64),
    )

    trace_file = tmp_path / 'streamfit_trace.csv'
    log_file = tmp_path / 'streamfit_log.csv'

    best_opt_params, loss_history, _ = gradient_descent.fit_streamline(
        initial_opt_params,
        fixed_params,
        data,
        uncertainties,
        147.0,
        n_epochs=3,
        info_every=100,
        early_stopping_patience=10,
        log_file=str(log_file),
        trace_file=str(trace_file),
        trace_every=1,
    )

    assert len(loss_history) == 3
    assert 'log_omega' in best_opt_params
    assert 'omega' in best_opt_params
    for key in ('r0', 'theta0', 'phi0', 'log_omega', 'omega', 'v_r0'):
        assert jnp.asarray(best_opt_params[key]).dtype == jnp.float64
    assert float(best_opt_params['omega']) == pytest.approx(
        float(np.exp(float(best_opt_params['log_omega']))), rel=1e-6
    )

    with open(trace_file, newline='') as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 4  # epoch 0 + 3 optimization epochs
    expected_columns = {
        'epoch',
        'loss',
        'theta_ref_model',
        'theta_ref_data',
        'model_nan_count',
        'model_metric_duplicate_count',
        'chi2_total',
    }
    assert expected_columns.issubset(set(rows[0].keys()))

    with open(log_file, newline='') as fh:
        log_rows = list(csv.DictReader(fh))

    assert len(log_rows) == 4  # epoch 0 + 3 optimization epochs
    expected_log_columns = {
        'epoch',
        'loss',
        'r0',
        'theta0',
        'phi0',
        'log_omega',
        'omega',
        'v_r0',
    }
    assert expected_log_columns.issubset(set(log_rows[0].keys()))

    for row in log_rows:
        log_omega = float(row['log_omega'])
        omega = float(row['omega'])
        assert omega == pytest.approx(float(np.exp(log_omega)), rel=1e-6)
