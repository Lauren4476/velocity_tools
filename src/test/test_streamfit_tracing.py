import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from velocity_tools import extract_streamline
from velocity_tools import gradient_descent

jax.config.update("jax_enable_x64", True)


def _get_param(name, opt_params, fixed_params):
    """Resolve a model parameter from optimizable or fixed dictionaries."""
    if name in opt_params:
        return opt_params[name]
    return fixed_params[name]


def _fake_forward_model(opt_params, fixed_params, distance_pc):
    """Small synthetic model with duplicate points to exercise tie tracing."""
    r0 = _get_param('r0', opt_params, fixed_params)
    mass = _get_param('mass', opt_params, fixed_params)
    v_lsr = _get_param('v_lsr', opt_params, fixed_params)

    shift = (1e-3 * r0) + (5e-4 * mass)
    ra_model = jnp.array([2.0, 1.5, 1.5, 1.0, 0.5], dtype=jnp.float64) + shift
    dec_model = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    v_model = jnp.array([7.20, 7.00, 7.00, 6.80, 6.60], dtype=jnp.float64) + (v_lsr - 7.0)
    return ra_model, dec_model, v_model


def _bounds_for_keys(keys):
    """Return deterministic bounds for optimized-key subsets used in tests."""
    template = {
        'r0': (200.0, 2000.0),
        'theta0': (0.0, float(jnp.pi)),
        'phi0': (0.0, float(2.0 * jnp.pi)),
        'log_omega': (float(np.log(1e-14)), float(np.log(1e-10))),
        'v_r0': (-5.0, 5.0),
        'mass': (0.1, 10.0),
        'inc': (-float(jnp.pi), float(jnp.pi)),
        'pa': (0.0, float(2.0 * jnp.pi)),
        'rmin': (1.0, 500.0),
        'deltar': (1.0, 200.0),
        'v_lsr': (0.0, 20.0),
    }
    return {key: template[key] for key in keys}


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
        param_bounds=_bounds_for_keys(initial_opt_params.keys()),
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


def test_fit_streamline_allows_custom_opt_partition(monkeypatch) -> None:
    monkeypatch.setattr(gradient_descent, 'forward_model', _fake_forward_model)

    initial_opt_params = {
        'mass': 3.2,
        'v_lsr': 7.0,
    }
    fixed_params = {
        'r0': 540.0,
        'theta0': 0.7,
        'phi0': 1.2,
        'log_omega': np.log(4e-12),
        'v_r0': -0.2,
        'inc': -0.8,
        'pa': 2.4,
        'rmin': 50.0,
        'deltar': 40.0,
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

    best_opt_params, loss_history, _ = gradient_descent.fit_streamline(
        initial_opt_params,
        fixed_params,
        data,
        uncertainties,
        147.0,
        param_bounds=_bounds_for_keys(initial_opt_params.keys()),
        n_epochs=3,
        info_every=100,
        early_stopping_patience=10,
    )

    assert len(loss_history) == 3
    assert set(best_opt_params.keys()) == {'mass', 'v_lsr'}
    assert jnp.asarray(best_opt_params['mass']).dtype == jnp.float64
    assert jnp.asarray(best_opt_params['v_lsr']).dtype == jnp.float64


def test_fit_streamline_stops_after_threshold_streak(monkeypatch) -> None:
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

    # The synthetic setup has loss ~1.1, so threshold 2.0 is met immediately.
    # With a 2-epoch streak requirement, training should stop after epoch 2.
    best_opt_params, loss_history, _ = gradient_descent.fit_streamline(
        initial_opt_params,
        fixed_params,
        data,
        uncertainties,
        147.0,
        param_bounds=_bounds_for_keys(initial_opt_params.keys()),
        n_epochs=10,
        info_every=100,
        early_stopping_patience=10,
        loss_threshold=2.0,
        loss_threshold_epochs=2,
    )

    assert len(loss_history) == 2
    assert all(loss <= 2.0 for loss in loss_history)
    assert 'omega' in best_opt_params


def test_fit_streamline_requires_bounds_for_all_optimized_params(monkeypatch) -> None:
    monkeypatch.setattr(gradient_descent, 'forward_model', _fake_forward_model)

    initial_opt_params = {
        'r0': 540.0,
        'theta0': 0.7,
    }
    fixed_params = {
        'phi0': 1.2,
        'log_omega': np.log(4e-12),
        'v_r0': -0.2,
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

    with pytest.raises(ValueError, match='Missing bounds for optimized parameters'):
        gradient_descent.fit_streamline(
            initial_opt_params,
            fixed_params,
            data,
            uncertainties,
            147.0,
            param_bounds={'r0': (200.0, 2000.0)},
            n_epochs=1,
            info_every=100,
            early_stopping_patience=10,
        )


def test_fit_streamline_accepts_omega_alias_bounds(monkeypatch) -> None:
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

    param_bounds = {
        'r0': (200.0, 2000.0),
        'theta0': (0.0, float(jnp.pi)),
        'phi0': (0.0, float(2.0 * jnp.pi)),
        'omega': (1e-14, 1e-10),
        'v_r0': (-5.0, 5.0),
    }

    best_opt_params, _, _ = gradient_descent.fit_streamline(
        initial_opt_params,
        fixed_params,
        data,
        uncertainties,
        147.0,
        param_bounds=param_bounds,
        n_epochs=1,
        info_every=100,
        early_stopping_patience=10,
    )

    assert 'log_omega' in best_opt_params
    assert 'omega' in best_opt_params
    assert jnp.asarray(best_opt_params['log_omega']).dtype == jnp.float64
    assert jnp.asarray(best_opt_params['omega']).dtype == jnp.float64


def test_fit_streamline_rejects_nonpositive_learning_rate(monkeypatch) -> None:
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

    with pytest.raises(ValueError, match='learning_rate must be > 0'):
        gradient_descent.fit_streamline(
            initial_opt_params,
            fixed_params,
            data,
            uncertainties,
            147.0,
            learning_rate=0.0,
            param_bounds=_bounds_for_keys(initial_opt_params.keys()),
            n_epochs=1,
            info_every=100,
            early_stopping_patience=10,
        )
