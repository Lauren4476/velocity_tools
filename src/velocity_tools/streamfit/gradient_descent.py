'''
This file contains the loss function and optimization routines for streamfit.

The optimization uses Adam (adaptive moment estimation) optimizer to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 02-02-26
'''

import jax.numpy as jnp
from jax import jit, grad, value_and_grad, lax
import jax
import optax
from skimage import data
from . import stream_lines_grad
from . import extract_streamline
import csv
import math

jax.config.update("jax_enable_x64", True)

FLOAT_DTYPE = jnp.float64


TRACE_FIELDNAMES = [
    'epoch',
    'loss',
    'chi2_ra',
    'chi2_dec',
    'chi2_v',
    'chi2_total',
    'theta_ref_model',
    'theta_ref_data',
    'theta_ref_delta',
    'model_points_total',
    'model_nan_count',
    'model_valid_points',
    'model_metric_span',
    'model_metric_min_gap',
    'model_metric_near_tie_count',
    'model_metric_duplicate_count',
    'model_metric_non_monotonic_count',
    'model_r_thresh',
    'model_inner_count',
    'data_r_thresh',
    'data_inner_count',
]


STREAMLINE_MODEL_PARAM_KEYS = (
    'r0',
    'theta0',
    'phi0',
    'log_omega',
    'v_r0',
    'mass',
    'inc',
    'pa',
    'rmin',
    'deltar',
    'v_lsr',
)


DEFAULT_OPTIMIZABLE_PARAM_KEYS = (
    'r0',
    'theta0',
    'phi0',
    'log_omega',
    'v_r0',
)


def _is_numeric_value(value):
    """Return True for scalar/array-like numeric values, excluding booleans."""
    try:
        arr = jnp.asarray(value)
    except Exception:
        return False
    if arr.dtype == jnp.bool_:
        return False
    return bool(jnp.issubdtype(arr.dtype, jnp.number))


def _to_float64(value):
    """Convert a numeric value or array-like input to float64."""
    return jnp.asarray(value, dtype=FLOAT_DTYPE)


def _coerce_opt_params_float64(opt_params):
    """Return optimization parameters coerced to float64."""
    coerced = {}
    for key, value in opt_params.items():
        if _is_numeric_value(value):
            coerced[key] = _to_float64(value)
        else:
            coerced[key] = value
    return coerced


def _coerce_fixed_params_float64(fixed_params):
    """Return fixed-parameter dictionary with numeric values coerced to float64."""
    coerced = {}
    for key, value in fixed_params.items():
        if value is None or isinstance(value, bool):
            coerced[key] = value
        elif _is_numeric_value(value):
            coerced[key] = _to_float64(value)
        else:
            coerced[key] = value
    return coerced


def _coerce_data_tuple_float64(values):
    """Coerce tuple/list of arrays to float64 arrays."""
    return tuple(_to_float64(value) for value in values)


def _sanitize_model_param_dict(params, dict_name):
    """Coerce a parameter dictionary to float64 and normalize aliases."""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"{dict_name} must be a dictionary, got {type(params).__name__}.")

    if dict_name == 'initial_opt_params':
        sanitized = _coerce_opt_params_float64(params.copy())
    else:
        sanitized = _coerce_fixed_params_float64(params.copy())

    if 'omega' in sanitized and 'log_omega' not in sanitized:
        sanitized['log_omega'] = jnp.log(_to_float64(sanitized['omega']))
    if 'omega' in sanitized:
        del sanitized['omega']

    unknown = sorted(key for key in sanitized if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown parameter keys in {dict_name}: {unknown}. "
            f"Supported keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return sanitized


def _validate_param_value_types(opt_params, fixed_params):
    """Validate numeric/None value types for model parameters."""
    for key, value in opt_params.items():
        if key == 'rmin' and value is None:
            raise ValueError("Optimizable parameter 'rmin' cannot be None.")
        if isinstance(value, bool) or not _is_numeric_value(value):
            raise TypeError(
                f"Optimizable parameter '{key}' must be numeric. "
                f"Got value of type {type(value).__name__}."
            )

    for key, value in fixed_params.items():
        if key == 'rmin' and value is None:
            continue
        if isinstance(value, bool) or not _is_numeric_value(value):
            raise TypeError(
                f"Fixed parameter '{key}' must be numeric"
                " (or None only for 'rmin'). "
                f"Got value of type {type(value).__name__}."
            )


def _sanitize_param_partition(initial_opt_params, fixed_params, require_nonempty_opt=False):
    """Sanitize and validate opt/fixed parameter partition for streamline modeling."""
    opt_params = _sanitize_model_param_dict(initial_opt_params, 'initial_opt_params')
    fixed_params = _sanitize_model_param_dict(fixed_params, 'fixed_params')

    overlap = sorted(set(opt_params) & set(fixed_params))
    if overlap:
        raise KeyError(
            f"Parameters cannot be present in both initial_opt_params and fixed_params: {overlap}"
        )

    missing = [
        key for key in STREAMLINE_MODEL_PARAM_KEYS
        if key not in opt_params and key not in fixed_params
    ]
    if missing:
        raise KeyError(
            "Missing required streamline parameters across initial_opt_params and fixed_params: "
            f"{missing}. Supported model keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    if require_nonempty_opt and len(opt_params) == 0:
        raise ValueError(
            "initial_opt_params must contain at least one optimizable parameter. "
            f"You can choose any subset of: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    _validate_param_value_types(opt_params, fixed_params)

    return opt_params, fixed_params


def _resolve_model_params(opt_params, fixed_params):
    """Return merged model parameters and sanitized opt/fixed dictionaries."""
    opt_params, fixed_params = _sanitize_param_partition(opt_params, fixed_params)
    model_params = fixed_params.copy()
    model_params.update(opt_params)
    return model_params, opt_params, fixed_params


def _normalize_learning_rate_dict(learning_rate_dict):
    """Normalize per-parameter learning-rate keys to canonical model keys."""
    if learning_rate_dict is None:
        return None

    normalized = dict(learning_rate_dict)
    if 'omega' in normalized:
        if 'log_omega' in normalized:
            raise KeyError(
                "learning_rate_dict contains both 'omega' and 'log_omega'. "
                "Please provide only one key."
            )
        normalized['log_omega'] = normalized.pop('omega')

    unknown = sorted(key for key in normalized if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown keys in learning_rate_dict: {unknown}. "
            f"Supported keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return normalized


def _normalize_param_bounds(param_bounds):
    """Normalize parameter-bound keys and convert omega bounds to log-space."""
    if param_bounds is None:
        return None

    normalized = dict(param_bounds)
    if 'omega' in normalized:
        if 'log_omega' in normalized:
            raise KeyError(
                "param_bounds contains both 'omega' and 'log_omega'. "
                "Please provide only one key."
            )
        omega_min, omega_max = normalized.pop('omega')
        omega_min = float(omega_min)
        omega_max = float(omega_max)
        if omega_min <= 0 or omega_max <= 0:
            raise ValueError("Omega bounds must be strictly positive when using 'omega' bounds.")
        if omega_min >= omega_max:
            raise ValueError("Omega bounds must satisfy omega_min < omega_max.")
        normalized['log_omega'] = (math.log(omega_min), math.log(omega_max))

    unknown = sorted(key for key in normalized if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown keys in param_bounds: {unknown}. "
            f"Supported keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return normalized

def _params_dict_to_vector(opt_params):
    """Convert parameter dict to ordered vector."""
    keys = list(opt_params.keys())
    vec = jnp.array([opt_params[k] for k in keys], dtype=FLOAT_DTYPE)
    return vec, keys


def _vector_to_params_dict(vec, keys):
    """Convert parameter vector back to dict."""
    return {k: vec[i] for i, k in enumerate(keys)}

def _omega_from_log_omega(log_omega):
    """Convert optimization-space log_omega to physical omega (1/s)."""
    return jnp.exp(log_omega)


def _with_derived_omega(opt_params):
    """Return a shallow copy including derived physical omega when available."""
    params_with_omega = opt_params.copy()
    if 'log_omega' in params_with_omega and 'omega' not in params_with_omega:
        params_with_omega['omega'] = _omega_from_log_omega(params_with_omega['log_omega'])
    return params_with_omega


def _as_float_or_value(value):
    """Convert scalar-like values to Python floats for readable diagnostics."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _dict_nonfinite_keys(values_dict):
    """Return dict keys whose values contain NaN/Inf."""
    bad_keys = []
    for key, value in values_dict.items():
        if not bool(jnp.all(jnp.isfinite(value))):
            bad_keys.append(key)
    return bad_keys


def _tree_has_nonfinite_values(tree):
    """Check whether any numeric leaf in a pytree contains NaN/Inf."""
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if leaf is None:
            continue
        try:
            if not bool(jnp.all(jnp.isfinite(leaf))):
                return True
        except TypeError:
            # Non-numeric leaf (e.g., metadata), ignore.
            continue
    return False


#not used anymore
def _debug_epoch_snapshot(epoch, stage, opt_params, fixed_params, loss_probe=None, grads=None, updates=None):
    """Print a detailed optimization snapshot to trace NaN/Inf origins."""
    print(f"\n[debug-trace] epoch={epoch}, stage={stage}")

    model_params, _, _ = _resolve_model_params(opt_params, fixed_params)

    params_printable = {key: _as_float_or_value(value) for key, value in model_params.items()}
    omega = _omega_from_log_omega(model_params['log_omega'])
    params_printable['omega'] = _as_float_or_value(omega)
    print(f"  opt_params={params_printable}")

    mass = model_params.get('mass', jnp.nan)
    rmin = model_params.get('rmin', jnp.nan)
    deltar = model_params.get('deltar', jnp.nan)
    r0 = model_params.get('r0', jnp.nan)
    rc = stream_lines_grad.r_cent(mass=mass, omega=omega, r0=r0)
    r_low = jnp.maximum(rmin, rc * 0.5) if rmin is not None else rc * 0.5
    r_start = r0 - deltar
    arange_ok = bool(jnp.isfinite(r_start) & jnp.isfinite(r_low) & (r_start > r_low))

    print(
        "  derived="
        f"r0_minus_deltar={_as_float_or_value(r_start)}, "
        f"rc={_as_float_or_value(rc)}, "
        f"r_low={_as_float_or_value(r_low)}, "
        f"arange_ok={arange_ok}"
    )

    if loss_probe is not None:
        print(f"  loss_probe={_as_float_or_value(loss_probe)}")

    if grads is not None:
        grads_printable = {key: _as_float_or_value(value) for key, value in grads.items()}
        print(f"  grads={grads_printable}")

    if updates is not None:
        updates_printable = {key: _as_float_or_value(value) for key, value in updates.items()}
        print(f"  updates={updates_printable}")


def _build_trace_row(epoch, loss_value, loss_trace):
    """Flatten nested trace dictionary into a CSV row."""
    chi2_components = loss_trace.get('chi2_components', {})
    matching = loss_trace.get('matching', {})
    model_metric_trace = matching.get('distance_metric_model', {})
    data_metric_trace = matching.get('distance_metric_data', {})

    theta_ref_model = matching.get('theta_ref_model', float('nan'))
    theta_ref_data = matching.get('theta_ref_data', float('nan'))

    return {
        'epoch': epoch,
        'loss': loss_value,
        'chi2_ra': chi2_components.get('chi2_ra', float('nan')),
        'chi2_dec': chi2_components.get('chi2_dec', float('nan')),
        'chi2_v': chi2_components.get('chi2_v', float('nan')),
        'chi2_total': chi2_components.get('chi2_total', float('nan')),
        'theta_ref_model': theta_ref_model,
        'theta_ref_data': theta_ref_data,
        'theta_ref_delta': theta_ref_model - theta_ref_data,
        'model_points_total': matching.get('model_points_total', 0),
        'model_nan_count': matching.get('model_nan_count', 0),
        'model_valid_points': matching.get('model_valid_points', 0),
        'model_metric_span': matching.get('model_metric_span', float('nan')),
        'model_metric_min_gap': matching.get('model_metric_min_gap', float('nan')),
        'model_metric_near_tie_count': matching.get('model_metric_near_tie_count', 0),
        'model_metric_duplicate_count': matching.get('model_metric_duplicate_count', 0),
        'model_metric_non_monotonic_count': matching.get('model_metric_non_monotonic_count', 0),
        'model_r_thresh': model_metric_trace.get('r_thresh', float('nan')),
        'model_inner_count': model_metric_trace.get('inner_count', 0),
        'data_r_thresh': data_metric_trace.get('r_thresh', float('nan')),
        'data_inner_count': data_metric_trace.get('inner_count', 0),
    }


def forward_model(opt_params, fixed_params, distance_pc):
    """
    Run the forward model using stream_lines_grad.xyz_stream
    
    Parameters:
    -----------
    opt_params : dict
        Dictionary containing optimizable parameters (any subset of
        STREAMLINE_MODEL_PARAM_KEYS).
    fixed_params : dict
        Dictionary containing fixed parameters (the complementary subset).
        Together with opt_params, this must define all keys in
        STREAMLINE_MODEL_PARAM_KEYS exactly once.
    distance_pc : float
        Distance to source in parsecs
        
    Returns:
    --------
    tuple: (ra_offsets, dec_offsets, velocities) each in appropriate units
        - RA offsets in arcsec (negative for standard convention)
        - Dec offsets in arcsec
        - Line-of-sight velocities in km/s
    """
    model_params, _, _ = _resolve_model_params(opt_params, fixed_params)
    distance_pc = _to_float64(distance_pc)

    omega = _omega_from_log_omega(model_params['log_omega'])

    # Run the forward model - returns positions in au, velocities in km/s
    (x, y, z), (vx, vy, vz) = stream_lines_grad.xyz_stream(
        mass=model_params['mass'],
        r0=model_params['r0'],
        theta0=model_params['theta0'],
        phi0=model_params['phi0'],
        omega=omega,
        v_r0=model_params['v_r0'],
        inc=model_params['inc'],
        pa=model_params['pa'],
        rmin=model_params['rmin'],
        deltar=model_params['deltar']
    )
    
    # Filter out sentinel values (used for points below rmin)
    # Sentinel value is -1e10, which is unphysical for positions
    # TODO: is this used?
    sentinel = -1e10
    valid_mask = (x > sentinel + 1e8)  # Points where x is NOT the sentinel
    
    # Convert positions from au to arcsec offsets
    # x = RA offset (with negative for standard RA convention)
    # z = Dec offset
    # y = line-of-sight velocity
    ra_model = jnp.where(valid_mask, -x / distance_pc, jnp.nan)  # arcsec
    dec_model = jnp.where(valid_mask, z / distance_pc, jnp.nan)  # arcsec
    v_model = jnp.where(valid_mask, vy + model_params['v_lsr'], jnp.nan)  # km/s (add systemic velocity)

    return ra_model, dec_model, v_model


def forward_fill_nans(arr):
    """
    Forward-fill NaN values in a JAX-compatible way.
    Each NaN is replaced with the last non-NaN value before it.
    Uses lax.scan for JIT compatibility.
    
    Parameters:
    -----------
    arr : array
        1D array potentially containing NaN values
        
    Returns:
    --------
    filled : array
        Array with NaN values forward-filled
    """
    arr = _to_float64(arr)
    is_nan = jnp.isnan(arr)
    num_nans = int(jnp.sum(is_nan))
    if num_nans > 0:
        print(f"[forward_fill_nans] Found {num_nans} NaN values in array of size {arr.size}")
    arr_clean = jnp.nan_to_num(arr, nan=0.0)
    
    # Forward-fill using scan
    def body_fn(last_valid, x_and_is_nan):
        x, x_is_nan = x_and_is_nan
        new_val = jnp.where(x_is_nan, last_valid, x)
        return new_val, new_val
    
    init_carry = arr_clean[0]
    _, filled = jax.lax.scan(body_fn, init_carry, (arr_clean, is_nan))
    return filled




def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data, return_trace=False):
    """
    Extract model values corresponding to data positions, using the same distance metric
    as used for binning the point cloud.
    Uses get_distance_metric from extract_streamline

    Parameters
    ----------
    return_trace : bool
        If True, also return a trace dictionary containing diagnostics on
        distance metric stability and model-point ordering.
    """

    ra_model = _to_float64(ra_model)
    dec_model = _to_float64(dec_model)
    v_model = _to_float64(v_model)
    ra_data = _to_float64(ra_data)
    dec_data = _to_float64(dec_data)

    # Forward-fill NaNs in model arrays - TODO does this make sense? should we be masking these instead?
    ra_model_filled = forward_fill_nans(ra_model)
    dec_model_filled = forward_fill_nans(dec_model)
    v_model_filled = forward_fill_nans(v_model)

    # compute distance metrics for full model and data
    # (no clipping - interpolation will handle matching)
    if return_trace:
        dmetric_model, dmetric_model_trace = extract_streamline.get_distance_metric(
            ra_model_filled, dec_model_filled, return_trace=True)
        dmetric_data, dmetric_data_trace = extract_streamline.get_distance_metric(
            ra_data, dec_data, return_trace=True)
    else:
        dmetric_model = extract_streamline.get_distance_metric(
            ra_model_filled, dec_model_filled)
        dmetric_data = extract_streamline.get_distance_metric(
            ra_data, dec_data) 

    # sort model once
    sort_idx = jnp.argsort(dmetric_model)
    d_model_sorted = dmetric_model[sort_idx]
    ra_sorted = ra_model_filled[sort_idx]
    dec_sorted = dec_model_filled[sort_idx]
    v_sorted = v_model_filled[sort_idx]

    # interpolate model to the *actual* data distance metric
    ra_model_interp = jnp.interp(dmetric_data, d_model_sorted, ra_sorted)
    dec_model_interp = jnp.interp(dmetric_data, d_model_sorted, dec_sorted)
    v_model_interp  = jnp.interp(dmetric_data, d_model_sorted, v_sorted)
        
    n_points = len(ra_data)

    valid = jnp.ones(n_points, dtype=bool)

    if not return_trace:
        return ra_model_interp, dec_model_interp, v_model_interp, valid

    model_nan_mask = jnp.isnan(ra_model) | jnp.isnan(dec_model) | jnp.isnan(v_model)
    model_nan_count = int(jnp.sum(model_nan_mask))
    model_points_total = int(ra_model.size)
    model_valid_points = model_points_total - model_nan_count

    d_diff = jnp.diff(d_model_sorted)
    if d_diff.size > 0:
        model_metric_min_gap = float(jnp.min(d_diff))
        model_metric_near_tie_count = int(jnp.sum(jnp.abs(d_diff) <= 1e-8))
        model_metric_duplicate_count = int(jnp.sum(d_diff == 0.0))
        model_metric_non_monotonic_count = int(jnp.sum(d_diff < 0.0))
    else:
        model_metric_min_gap = float('nan')
        model_metric_near_tie_count = 0
        model_metric_duplicate_count = 0
        model_metric_non_monotonic_count = 0

    model_metric_span = float(d_model_sorted[-1] - d_model_sorted[0]) if d_model_sorted.size > 1 else 0.0

    matching_trace = {
        'model_points_total': model_points_total,
        'model_nan_count': model_nan_count,
        'model_valid_points': model_valid_points,
        'model_metric_span': model_metric_span,
        'model_metric_min_gap': model_metric_min_gap,
        'model_metric_near_tie_count': model_metric_near_tie_count,
        'model_metric_duplicate_count': model_metric_duplicate_count,
        'model_metric_non_monotonic_count': model_metric_non_monotonic_count,
        'distance_metric_model': dmetric_model_trace,
        'distance_metric_data': dmetric_data_trace,
    }

    return ra_model_interp, dec_model_interp, v_model_interp, valid, matching_trace


def chi2_loss(opt_params, fixed_params, data, uncertainties, distance_pc, return_trace=False):
    """
    Compute chi-squared loss between model and data (RA, Dec, LOS velocity)
    
    Parameters:
    -----------
    opt_params : dict
        Optimizable streamline model parameters (any subset of
        STREAMLINE_MODEL_PARAM_KEYS).
    fixed_params : dict
        Fixed streamline model parameters (complementary subset).
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data
    distance_pc : float
        Distance to source in parsecs

        
    Returns:
    --------
    float: Chi-squared loss value
    """

    opt_params, fixed_params = _sanitize_param_partition(opt_params, fixed_params)
    distance_pc = _to_float64(distance_pc)

    ra_data, dec_data, v_data = _coerce_data_tuple_float64(data)
    ra_sigma, dec_sigma, v_sigma = _coerce_data_tuple_float64(uncertainties)
    # small values to avoid division by zero
    eps = _to_float64(1e-8)
    ra_sigma = jnp.maximum(ra_sigma, eps)
    dec_sigma = jnp.maximum(dec_sigma, eps)
    v_sigma = jnp.maximum(v_sigma, eps)
    
    # Run forward model
    ra_model, dec_model, v_model = forward_model(opt_params, fixed_params, distance_pc)
    

    # Match model to data using arc-length parameterisation
    if return_trace:
        ra_model_interp, dec_model_interp, v_model_interp, _, matching_trace = match_model_to_data_curve(
            ra_model, dec_model, v_model, ra_data, dec_data, return_trace=True)
    else:
        ra_model_interp, dec_model_interp, v_model_interp, _ = match_model_to_data_curve(
            ra_model, dec_model, v_model, ra_data, dec_data)


    # Compute chi-squared components
    chi2_ra = jnp.sum(((ra_data - ra_model_interp) / ra_sigma)**2)
    chi2_dec = jnp.sum(((dec_data - dec_model_interp) / dec_sigma)**2)
    chi2_v = jnp.sum(((v_data - v_model_interp) / v_sigma)**2)
    # Total chi-squared
    chi2_total = chi2_ra + chi2_dec + chi2_v

    if return_trace:
        loss_trace = {
            'chi2_components': {
                'chi2_ra': float(chi2_ra),
                'chi2_dec': float(chi2_dec),
                'chi2_v': float(chi2_v),
                'chi2_total': float(chi2_total),
            },
            'matching': matching_trace,
        }
        return chi2_total, loss_trace

    return chi2_total

##### DEPRECATED MANUAL ADAM IMPLEMENTATION - WE USE OPTAX INSTEAD #####
'''
def adam_step(opt_params, grads, m, v, t, learning_rate=0.001, learning_rate_dict=None, 
              beta1=0.9, beta2=0.999, eps=1e-8, param_bounds=None):
    """
    Perform one Adam optimization step (only for optimizable parameters).
    
    Parameters:
    -----------
    opt_params : dict
        Current optimizable parameter values
    grads : dict
        Gradients of loss w.r.t. optimizable parameters
    m : dict
        First moment estimates (momentum)
    v : dict
        Second moment estimates (adaptive learning rate)
    t : int
        Time step (iteration number)
    learning_rate : float
        Default learning rate (alpha) - used if learning_rate_dict doesn't have a learning rate for a specific param.
    learning_rate_dict : dict or None
        Optional per-parameter learning rates. If provided, overrides learning_rate for each param in the dict.
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    eps : float
        Small constant for numerical stability
    param_bounds : dict or None
        Optional bounds for each parameter: {param_name: (min, max)}
        
    Returns:
    --------
    tuple: (new_opt_params, new_m, new_v)
    """
    new_opt_params = {}
    new_m = {}
    new_v = {}
    
    for key in opt_params.keys():
        # Update biased first moment estimate
        new_m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
        
        # Update biased second raw moment estimate
        new_v[key] = beta2 * v[key] + (1 - beta2) * grads[key]**2
        
        # Compute bias-corrected first moment estimate
        m_hat = new_m[key] / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = new_v[key] / (1 - beta2**t)

        # Get learning rate for this parameter
        if learning_rate_dict is not None and key in learning_rate_dict:
            lr = learning_rate_dict[key]
        else:
            lr = learning_rate
        
        # Update parameters
        new_opt_params[key] = opt_params[key] - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        # Apply bounds if provided
        if param_bounds is not None and key in param_bounds:
            min_val, max_val = param_bounds[key]
            new_opt_params[key] = jnp.clip(new_opt_params[key], min_val, max_val)
           

    return new_opt_params, new_m, new_v
'''

def estimate_parameter_errors(best_opt_params, fixed_params, data, uncertainties, distance_pc):
    """
    Estimate parameter uncertainties using Hessian of chi2 loss.

    Returns
    -------
    dict
        1-sigma uncertainties for each optimizable parameter
    array
        covariance matrix
    """

    # convert dict -> vector
    theta0, keys = _params_dict_to_vector(best_opt_params)

    def loss_vec(theta_vec):
        params = _vector_to_params_dict(theta_vec, keys)
        return chi2_loss(params, fixed_params, data, uncertainties, distance_pc)

    # compute Hessian
    H = jax.hessian(loss_vec)(theta0)

    # invert to get covariance
    cov = jnp.linalg.inv(H)

    print(f"Hessian matrix:\n{H}"
          f"\nCovariance matrix:\n{cov}")

    # parameter errors
    errors = jnp.sqrt(jnp.diag(cov))

    error_dict = {k: float(errors[i]) for i, k in enumerate(keys)}

    return error_dict, cov

def fit_streamline(initial_opt_params, fixed_params, data, uncertainties, distance_pc,
                   learning_rate=0.001, learning_rate_dict=None, param_bounds=None, n_epochs=1000, 
                   beta1=0.9, beta2=0.999, 
                   info_every=100, early_stopping_patience=50, log_file=None,
                   trace_file=None, trace_every=1, output_uncertainties=False):
    """
    Fit streamline model parameters to data using Adam optimizer.
    Any supported streamline parameter can be optimized or fixed.
    Parameters are split by dictionary membership:
    - keys in initial_opt_params are optimized
    - keys in fixed_params are held fixed
    The union must contain each key in STREAMLINE_MODEL_PARAM_KEYS exactly once.
    
    Parameters:
    -----------
    initial_opt_params : dict
        Initial guesses for the parameters to optimize.
        Allowed keys are STREAMLINE_MODEL_PARAM_KEYS.
        Historically, the default optimized subset is:
        DEFAULT_OPTIMIZABLE_PARAM_KEYS.
    fixed_params : dict
        Fixed (non-optimized) parameters using the same key space.
        Together with initial_opt_params, this must provide a full,
        non-overlapping partition of STREAMLINE_MODEL_PARAM_KEYS.
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data
    distance_pc : float
            Distance to source in parsecs
    learning_rate : float
        Default learning rate for Adam optimizer. Used if no specific rate provided for a parameter.
    learning_rate_dict : dict or None
        Per-parameter learning rates keyed by model parameter names.
        If provided, overrides learning_rate for specified optimized keys.
    param_bounds : dict or None
        Parameter bounds in optimization space.
        Bounds are applied only to optimized keys that have entries here.
        You may provide 'omega' bounds as linear bounds; these are converted
        to 'log_omega' bounds internally.
    n_epochs : int
        Maximum number of optimization iterations
    beta1 : float
        Adam exponential decay rate for first moment
    beta2 : float
        Adam exponential decay rate for second moment
    info_every : int
        Print loss every N epochs
    early_stopping_patience : int
        Stop if loss doesn't improve for N epochs
    log_file : str or None
        If provided, log epoch, loss, and parameter values to this CSV file.
        File will be created/overwritten at start and updated after each epoch.
    trace_file : str or None
        If provided, log per-epoch matching diagnostics (theta_ref, NaN counts,
        metric gaps/ties) to this CSV file.
    trace_every : int
        Frequency (in epochs) for writing rows to trace_file.
        Must be >= 1.
        
    Returns:
    --------   
    dict: Optimized parameters (same keys as initial_opt_params), including
        derived 'omega' when 'log_omega' is optimized.
    list: Loss history
    """
    # Initialize parameters
    opt_params, fixed_params = _sanitize_param_partition(
        initial_opt_params,
        fixed_params,
        require_nonempty_opt=True,
    )
    data = _coerce_data_tuple_float64(data)
    uncertainties = _coerce_data_tuple_float64(uncertainties)
    distance_pc = _to_float64(distance_pc)
    learning_rate_dict = _normalize_learning_rate_dict(learning_rate_dict)
    param_bounds = _normalize_param_bounds(param_bounds)

    # Build optimizer (supports optional per-parameter learning rates)
    if learning_rate_dict is not None:
        param_labels = {
            key: key if key in learning_rate_dict else 'default'
            for key in opt_params.keys()
        }
        transforms = {
            'default': optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2)
        }
        for key, lr in learning_rate_dict.items():
            if key in opt_params:
                transforms[key] = optax.adam(learning_rate=lr, b1=beta1, b2=beta2)
        solver = optax.multi_transform(transforms, param_labels)
    else:
        solver = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2)

    opt_state = solver.init(opt_params)
    
    # Create gradient function (only w.r.t. opt_params)
    loss_and_grad_fn = value_and_grad(chi2_loss, argnums=0)
    
    # Track loss history
    loss_history = []
    initial_loss = float(chi2_loss(opt_params, fixed_params, data, uncertainties, distance_pc))
    best_loss = initial_loss
    best_opt_params = opt_params.copy()
    best_epoch = 0
    patience_counter = 0

    if trace_every < 1:
        raise ValueError('trace_every must be >= 1')
    
    # Initialize CSV log file if requested
    csv_file = None
    csv_writer = None
    if log_file is not None:
        csv_file = open(log_file, 'w', newline='')
        # Create header: epoch, loss, then all optimizable params
        fieldnames = ['epoch', 'loss'] + list(opt_params.keys())
        if 'log_omega' in opt_params and 'omega' not in fieldnames:
            fieldnames.append('omega')
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_file.flush()

    trace_csv_file = None
    trace_csv_writer = None
    if trace_file is not None:
        trace_csv_file = open(trace_file, 'w', newline='')
        trace_csv_writer = csv.DictWriter(trace_csv_file, fieldnames=TRACE_FIELDNAMES)
        trace_csv_writer.writeheader()
        trace_csv_file.flush()
    
    print(f"Starting optimization with {n_epochs} epochs...")
    print(f"Optimizing parameters: {list(opt_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Initial optimizable values:")
    for key in opt_params.keys():
        print(f"  {key}: {opt_params[key]:.3e}")
    
    # Log initial parameters and initial loss (epoch 0) if CSV logging is enabled
    if csv_writer is not None:
        row = {'epoch': 0, 'loss': initial_loss}
        for key in opt_params.keys():
            row[key] = float(opt_params[key])
        if 'log_omega' in opt_params:
            row['omega'] = float(_omega_from_log_omega(opt_params['log_omega']))
        csv_writer.writerow(row)
        csv_file.flush()

    if trace_csv_writer is not None:
        initial_loss_for_trace, initial_trace = chi2_loss(
            opt_params, fixed_params, data, uncertainties, distance_pc, return_trace=True)
        initial_trace_row = _build_trace_row(0, float(initial_loss_for_trace), initial_trace)
        trace_csv_writer.writerow(initial_trace_row)
        trace_csv_file.flush()
    
    try:
        for epoch in range(1, n_epochs + 1):
            if epoch % info_every == 0:
                print(f"\n Starting Epoch {epoch} -------------------------")
            # Compute gradients at current parameters (pre-update)
            _, grads = loss_and_grad_fn(opt_params, fixed_params, data, uncertainties, distance_pc)

            # Perform Optax Adam step
            updates, opt_state = solver.update(grads, opt_state, params=opt_params)
            opt_params = optax.apply_updates(opt_params, updates)

            # Apply bounds if provided
            if param_bounds is not None:
                for key in opt_params.keys():
                    if key in param_bounds:
                        min_val, max_val = param_bounds[key]
                        opt_params[key] = jnp.clip(opt_params[key], min_val, max_val)

            # Compute loss at updated parameters (post-update)
            if trace_csv_writer is not None and epoch % trace_every == 0:
                loss_eval, loss_trace = chi2_loss(
                    opt_params, fixed_params, data, uncertainties, distance_pc, return_trace=True)
                loss_value = float(loss_eval)
            else:
                loss_value = float(chi2_loss(opt_params, fixed_params, data, uncertainties, distance_pc))
                loss_trace = None
        
            # Track loss
            loss_history.append(loss_value)
        
            # Log to CSV if requested
            if csv_writer is not None:
                row = {'epoch': epoch, 'loss': loss_value}
                # Add all optimizable parameter values
                for key in opt_params.keys():
                    row[key] = float(opt_params[key])
                if 'log_omega' in opt_params:
                    row['omega'] = float(_omega_from_log_omega(opt_params['log_omega']))
                csv_writer.writerow(row)
                csv_file.flush()  # Ensure data is written after each epoch

            if trace_csv_writer is not None and loss_trace is not None:
                trace_row = _build_trace_row(epoch, loss_value, loss_trace)
                trace_csv_writer.writerow(trace_row)
                trace_csv_file.flush()
        
            # Early stopping check
            if loss_value < best_loss:
                best_loss = loss_value
                best_opt_params = opt_params.copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
        
            # Print progress
            if epoch % info_every == 0:
                print(f'Epoch {epoch}/{n_epochs}, Loss: {loss_value:.6f}, Best Loss: {best_loss:.6f}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} - no improvement for {early_stopping_patience} epochs")
                break
    
    finally:
        # Always close the CSV file if it was opened
        if csv_file is not None:
            csv_file.close()
            print(f"Optimization log saved to: {log_file}")
        if trace_csv_file is not None:
            trace_csv_file.close()
            print(f"Matching trace log saved to: {trace_file}")

    print(f"\nOptimization complete!")
    print(f"Final loss: {best_loss:.6f}")
    print(f"Best-fit parameters found at epoch: {best_epoch}")
    for key in best_opt_params.keys():
        print(f"  {key}: {best_opt_params[key]:.3e}")

    # compute errors on best-fit parameters
    if output_uncertainties:
        print("\nEstimating parameter uncertainties from Hessian...")
        param_errors, cov_matrix = estimate_parameter_errors(
            best_opt_params,
            fixed_params,
            data,
            uncertainties,
            distance_pc
        )
        print("\nParameter uncertainties (1-sigma):")
        for k, v in param_errors.items():
            print(f"  {k}: {v}")
    else:
        param_errors = None


    return _with_derived_omega(best_opt_params), loss_history, param_errors