'''
This file contains the loss function and optimization routines for streamfit.

The optimization uses Adam (adaptive moment estimation) optimizer to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 30-03-26
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
    'grad_norm',
    'theta_ref_model',
    'theta_ref_data',
    'model_points_total',
    'model_nan_count',
    'model_valid_points',
    'model_metric_span',
    'model_metric_min_gap',
    'model_metric_near_tie_count',
    'model_metric_duplicate_count',
    'model_metric_non_monotonic_count',
    'model_inner_count',
    'data_inner_count',
    'data_points_total',
    'data_valid_points',
    'data_retained_count',
    'model_retained_count',
    'overlap_r_min',
    'overlap_r_max',
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


def _build_normalization_spec(opt_params, param_bounds):
    """Build bounds-derived shift/scale metadata for optimized parameters."""
    if param_bounds is None:
        raise ValueError(
            "param_bounds is required because optimization is performed in normalized space. "
            "Provide bounds for every optimized parameter."
        )

    missing = sorted(key for key in opt_params if key not in param_bounds)
    if missing:
        raise ValueError(
            "Missing bounds for optimized parameters: "
            f"{missing}. Please add (min, max) entries for all optimized keys."
        )

    normalization_spec = {}
    for key, value in opt_params.items():
        bounds = param_bounds[key]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for '{key}' must be a 2-element (min, max) tuple/list. "
                f"Got: {bounds!r}"
            )

        lower = _to_float64(bounds[0])
        upper = _to_float64(bounds[1])
        if not bool(jnp.isfinite(lower)) or not bool(jnp.isfinite(upper)):
            raise ValueError(f"Bounds for '{key}' must be finite. Got ({bounds[0]}, {bounds[1]}).")
        if not bool(upper > lower):
            raise ValueError(
                f"Bounds for '{key}' must satisfy min < max. Got ({float(lower)}, {float(upper)})."
            )

        value = _to_float64(value)
        if not bool(jnp.isfinite(value)):
            raise ValueError(f"Initial value for '{key}' must be finite. Got {value}.")
        if not bool((value >= lower) & (value <= upper)):
            raise ValueError(
                f"Initial value for '{key}' ({float(value)}) is outside bounds "
                f"({float(lower)}, {float(upper)})."
            )

        scale = upper - lower
        normalization_spec[key] = {
            'offset': lower,
            'scale': scale,
        }

    return normalization_spec


def _normalize_opt_params(opt_params, normalization_spec):
    """Normalize optimized parameters to [0, 1] using x_norm=(x-min)/(max-min)."""
    normalized = {}
    for key, value in opt_params.items():
        offset = normalization_spec[key]['offset']
        scale = normalization_spec[key]['scale']
        normalized[key] = (_to_float64(value) - offset) / scale
    return normalized


def _denormalize_opt_params(norm_opt_params, normalization_spec):
    """Convert normalized optimized parameters back to physical/log parameter values."""
    denormalized = {}
    for key, value in norm_opt_params.items():
        offset = normalization_spec[key]['offset']
        scale = normalization_spec[key]['scale']
        denormalized[key] = _to_float64(value) * scale + offset
    return denormalized


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


def _gradient_l2_norm(grad_tree):
    """Compute L2 norm of gradients across all leaves in a pytree."""
    grad_leaves = jax.tree_util.tree_leaves(grad_tree)
    grad_sum_sq = jnp.asarray(0.0, dtype=FLOAT_DTYPE)
    for grad_leaf in grad_leaves:
        grad_sum_sq = grad_sum_sq + jnp.sum(jnp.square(grad_leaf))
    return jnp.sqrt(grad_sum_sq)


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


def _build_trace_row(epoch, loss_value, loss_trace, grad_norm):
    """Flatten nested trace dictionary into a CSV row."""
    chi2_components = loss_trace.get('chi2_components', {})
    matching = loss_trace.get('matching', {})
    model_metric_trace = matching.get('distance_metric_model', {})
    data_metric_trace = matching.get('distance_metric_data', {})


    return {
        'epoch': epoch,
        'loss': loss_value,
        'chi2_ra': chi2_components.get('chi2_ra', float('nan')),
        'chi2_dec': chi2_components.get('chi2_dec', float('nan')),
        'chi2_v': chi2_components.get('chi2_v', float('nan')),
        'chi2_total': chi2_components.get('chi2_total', float('nan')),
        'grad_norm': grad_norm,
        'theta_ref_model': model_metric_trace.get('theta_ref', float('nan')),
        'theta_ref_data': data_metric_trace.get('theta_ref', float('nan')),
        'model_points_total': matching.get('model_points_total', 0),
        'model_nan_count': matching.get('model_nan_count', 0),
        'model_valid_points': matching.get('model_valid_points', 0),
        'model_metric_span': matching.get('model_metric_span', float('nan')),
        'model_metric_min_gap': matching.get('model_metric_min_gap', float('nan')),
        'model_metric_near_tie_count': matching.get('model_metric_near_tie_count', 0),
        'model_metric_duplicate_count': matching.get('model_metric_duplicate_count', 0),
        'model_metric_non_monotonic_count': matching.get('model_metric_non_monotonic_count', 0),
        'model_inner_count': model_metric_trace.get('inner_count', 0),
        'data_inner_count': data_metric_trace.get('inner_count', 0),
        'data_points_total': matching.get('data_points_total', 0),
        'data_valid_points': matching.get('data_valid_points', 0),
        'data_retained_count': matching.get('data_retained_count', 0),
        'model_retained_count': matching.get('model_retained_count', 0),
        'overlap_r_min': matching.get('overlap_r_min', float('nan')),
        'overlap_r_max': matching.get('overlap_r_max', float('nan')),
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
        - Line-of-sight velocities in km/s, relative to v_lsr
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
    # make velocity absolute by adding back v_lsr 
    v_model = jnp.where(valid_mask, vy + model_params['v_lsr'], jnp.nan)  # km/s  (absolute)

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
    # if num_nans > 0:
    #     jax.debug.print(f"[forward_fill_nans] Found {num_nans} NaN values in array of size {arr.size}")
    # else:
    #     jax.debug.print("[forward_fill_nans] No forward filling needed")
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
    Extract model values corresponding to data positions using the projected
    radial distance metric from extract_streamline.get_distance_metric.

    Matching is restricted to the physically overlapping radial domain:
    1. Data is restricted to the model-supported radial range.
    2. Model is restricted to the data-supported radial range.
    3. Interpolation is performed only on the overlap support.

    Parameters
    ----------
    return_trace : bool
        If True, also return a trace dictionary containing diagnostics on
        distance metric stability and model-point ordering.

    Returns
    -------
    tuple
        (ra_model_interp, dec_model_interp, v_model_interp, valid)
        where valid is a boolean mask with shape len(original data), marking
        retained data points inside the overlap domain.

    Raises
    ------
    ValueError
        If no valid model/data points exist, there is no radial overlap, or
        fewer than two model points remain in the overlap domain.
    """

    ra_model = _to_float64(ra_model)
    dec_model = _to_float64(dec_model)
    v_model = _to_float64(v_model)
    ra_data = _to_float64(ra_data)
    dec_data = _to_float64(dec_data)

    # Compute projected radial metric for model/data in float64.
    if return_trace:
        dmetric_model, dmetric_model_trace = extract_streamline.get_distance_metric(
            ra_model, dec_model, return_trace=True)
        dmetric_data, dmetric_data_trace = extract_streamline.get_distance_metric(
            ra_data, dec_data, return_trace=True)
    else:
        dmetric_model = extract_streamline.get_distance_metric(
            ra_model, dec_model)
        dmetric_data = extract_streamline.get_distance_metric(
            ra_data, dec_data) 

    model_finite_mask = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )
    data_finite_mask = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )

    if not bool(jnp.any(model_finite_mask)):
        raise ValueError('No finite model points are available for model-data matching.')
    if not bool(jnp.any(data_finite_mask)):
        raise ValueError('No finite data points are available for model-data matching.')

    model_metric_for_min = jnp.where(model_finite_mask, dmetric_model, jnp.inf)
    model_metric_for_max = jnp.where(model_finite_mask, dmetric_model, -jnp.inf)
    data_metric_for_min = jnp.where(data_finite_mask, dmetric_data, jnp.inf)
    data_metric_for_max = jnp.where(data_finite_mask, dmetric_data, -jnp.inf)

    model_min = jnp.min(model_metric_for_min)
    model_max = jnp.max(model_metric_for_max)
    data_min = jnp.min(data_metric_for_min)
    data_max = jnp.max(data_metric_for_max)

    overlap_min = jnp.maximum(model_min, data_min)
    overlap_max = jnp.minimum(model_max, data_max)

    if not bool(overlap_max >= overlap_min):
        raise ValueError(
            'No physically valid radial overlap between model and data. '
            f'Model range [{float(model_min):.6g}, {float(model_max):.6g}], '
            f'data range [{float(data_min):.6g}, {float(data_max):.6g}]'
        )

    data_keep = data_finite_mask & (dmetric_data >= overlap_min) & (dmetric_data <= overlap_max)
    model_keep = model_finite_mask & (dmetric_model >= overlap_min) & (dmetric_model <= overlap_max)

    if not bool(jnp.any(data_keep)):
        raise ValueError(
            'No retained data points after overlap filtering. '
            f'Overlap range [{float(overlap_min):.6g}, {float(overlap_max):.6g}]'
        )

    model_retained_count = int(jnp.sum(model_keep))
    if model_retained_count < 2:
        raise ValueError(
            'Insufficient retained model support for interpolation after overlap filtering: '
            f'{model_retained_count} point(s) available; need at least 2.'
        )

    # Sort model once by metric.
    sort_idx = jnp.argsort(dmetric_model)
    d_model_sorted = dmetric_model[sort_idx]
    ra_sorted = ra_model[sort_idx]
    dec_sorted = dec_model[sort_idx]
    v_sorted = v_model[sort_idx]
    model_keep_sorted = model_keep[sort_idx]

    # Build edge anchors on retained support so clipped interpolation uses only
    # overlap-domain endpoints.
    first_keep_idx = jnp.argmax(model_keep_sorted)
    last_keep_idx = model_keep_sorted.size - 1 - jnp.argmax(jnp.flip(model_keep_sorted))

    ra_first = ra_sorted[first_keep_idx]
    dec_first = dec_sorted[first_keep_idx]
    v_first = v_sorted[first_keep_idx]
    ra_last = ra_sorted[last_keep_idx]
    dec_last = dec_sorted[last_keep_idx]
    v_last = v_sorted[last_keep_idx]

    is_below_overlap = d_model_sorted < overlap_min
    ra_support = jnp.where(model_keep_sorted, ra_sorted, jnp.where(is_below_overlap, ra_first, ra_last))
    dec_support = jnp.where(model_keep_sorted, dec_sorted, jnp.where(is_below_overlap, dec_first, dec_last))
    v_support = jnp.where(model_keep_sorted, v_sorted, jnp.where(is_below_overlap, v_first, v_last))
    d_support = jnp.clip(d_model_sorted, overlap_min, overlap_max)

    # Interpolate only on overlap support. For dropped data points, query at
    # overlap_min and then overwrite with finite placeholders.
    d_query = jnp.where(data_keep, dmetric_data, overlap_min)
    ra_interp_all = jnp.interp(d_query, d_support, ra_support)
    dec_interp_all = jnp.interp(d_query, d_support, dec_support)
    v_interp_all = jnp.interp(d_query, d_support, v_support)

    ra_model_interp = jnp.where(data_keep, ra_interp_all, ra_data)
    dec_model_interp = jnp.where(data_keep, dec_interp_all, dec_data)
    v_model_interp = jnp.where(data_keep, v_interp_all, _to_float64(0.0))

    valid = data_keep

    if not return_trace:
        return ra_model_interp, dec_model_interp, v_model_interp, valid

    model_nan_mask = ~model_finite_mask
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
        'model_retained_count': model_retained_count,
        'data_points_total': int(ra_data.size),
        'data_valid_points': int(jnp.sum(data_finite_mask)),
        'data_retained_count': int(jnp.sum(data_keep)),
        'overlap_r_min': float(overlap_min),
        'overlap_r_max': float(overlap_max),
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
        ra_model_interp, dec_model_interp, v_model_interp, valid, matching_trace = match_model_to_data_curve(
            ra_model, dec_model, v_model, ra_data, dec_data, return_trace=True)
    else:
        ra_model_interp, dec_model_interp, v_model_interp, valid = match_model_to_data_curve(
            ra_model, dec_model, v_model, ra_data, dec_data)
        

    ### smooth overlap and weighting - penalty for being outside overlap
    dmetric_data = extract_streamline.get_distance_metric(ra_data, dec_data)
    dmetric_model = extract_streamline.get_distance_metric(ra_model, dec_model)

    model_finite = jnp.isfinite(dmetric_model)
    data_finite = jnp.isfinite(dmetric_data)

    model_min = jnp.min(jnp.where(model_finite, dmetric_model, jnp.inf))
    model_max = jnp.max(jnp.where(model_finite, dmetric_model, -jnp.inf))
    data_min = jnp.min(jnp.where(data_finite, dmetric_data, jnp.inf))
    data_max = jnp.max(jnp.where(data_finite, dmetric_data, -jnp.inf))

    overlap_min = jnp.maximum(model_min, data_min)
    overlap_max = jnp.minimum(model_max, data_max)

    margin = _to_float64(0.05)  # tune this

    dist_to_overlap = jnp.minimum(
        jnp.abs(dmetric_data - overlap_min),
        jnp.abs(dmetric_data - overlap_max)
    )

    weights = jnp.exp(- (dist_to_overlap / margin) ** 2)

    penalty = jnp.maximum(0.0, overlap_min - dmetric_data) + \
              jnp.maximum(0.0, dmetric_data - overlap_max)

    chi2_penalty = jnp.sum((penalty / margin) ** 2)

    ### main polar plane of sky / velocity loss

    r_data, theta_data = extract_streamline.cartesian_to_polar(ra_data, dec_data)
    _, theta_model = extract_streamline.cartesian_to_polar(ra_model_interp, dec_model_interp)

    # angular difference -> arc length distance
    dtheta = extract_streamline._wrap_to_pi(theta_data - theta_model)
    dsky = r_data * dtheta # gives distance in au, with dtheta in rad and r_data in au
    sigma_dsky = jnp.sqrt(ra_sigma**2 + dec_sigma**2) # approximate uncertainty on dsky

    chi2_dsky = jnp.sum(weights * ((dsky / sigma_dsky)**2))
    chi2_v = jnp.sum(weights * (((v_data - v_model_interp) / v_sigma)**2))
    chi2_total = chi2_dsky + chi2_v + chi2_penalty


    if return_trace:
        loss_trace = {
            'chi2_components': {
                'chi2_dsky': float(chi2_dsky),
                'chi2_v': float(chi2_v),
                'chi2_penalty': float(chi2_penalty),
                'overlap_width': float(overlap_max - overlap_min),
                'chi2_total': float(chi2_total),
            },
            'matching': matching_trace,
        }
        return chi2_total, loss_trace

    return chi2_total

    '''

    ### Original RA/Dec/velocity loss

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
    '''

def estimate_parameter_errors(
    best_opt_params,
    fixed_params,
    data,
    uncertainties,
    distance_pc,
    gradient_tol=1e-1,
    normalization_spec=None,
):
    """
    Estimate parameter uncertainties using Hessian of chi2 loss.

    Parameters
    ----------
    gradient_tol : float or None
        Tolerance on gradient norm in normalized space. If provided and
        normalized-space gradient norm > gradient_tol at best params, a
        warning is issued because the quadratic approximation may not be valid.
    normalization_spec : dict or None
        Bounds-derived normalization metadata for optimized parameters.
        Required to evaluate gradient_tol in normalized space.

    Returns
    -------
    dict
        1-sigma uncertainties for each optimizable parameter
    array
        covariance matrix
    """

    if gradient_tol is not None:
        gradient_tol = float(gradient_tol)
        if not math.isfinite(gradient_tol):
            raise ValueError('gradient_tol must be finite when provided.')
        if gradient_tol <= 0:
            raise ValueError('gradient_tol must be positive when provided.')

    # convert dict -> vector
    params_vec, keys = _params_dict_to_vector(best_opt_params)

    def loss_vec(theta_vec):
        params = _vector_to_params_dict(theta_vec, keys)
        return chi2_loss(params, fixed_params, data, uncertainties, distance_pc)

    # Check gradient magnitude at best-fit parameters in normalized space.
    if gradient_tol is not None:
        if normalization_spec is None:
            print(
                "WARNING: gradient_tol is interpreted in normalized space, but "
                "normalization_spec was not provided. Skipping gradient_tol check "
                "for uncertainty estimation."
            )
        else:
            missing_norm_keys = [key for key in keys if key not in normalization_spec]
            if missing_norm_keys:
                raise ValueError(
                    "normalization_spec is missing optimized parameter keys required "
                    f"for gradient_tol check: {missing_norm_keys}"
                )

            norm_opt_params = _normalize_opt_params(best_opt_params, normalization_spec)
            norm_params_vec, _ = _params_dict_to_vector(norm_opt_params)

            def norm_loss_vec(theta_norm_vec):
                norm_params = _vector_to_params_dict(theta_norm_vec, keys)
                physical_params = _denormalize_opt_params(norm_params, normalization_spec)
                return chi2_loss(physical_params, fixed_params, data, uncertainties, distance_pc)

            norm_grad_vec = jax.grad(norm_loss_vec)(norm_params_vec)
            norm_grad_norm = float(_gradient_l2_norm(norm_grad_vec))

            if norm_grad_norm > gradient_tol:
                print(
                    "WARNING: Normalized-space gradient norm at best fit = "
                    f"{norm_grad_norm:.3e} exceeds tolerance {gradient_tol:.3e}"
                )
                print("Optimization may not have reached a minimum yet.")
                print("Parameter uncertainties may be unreliable or incalculable. Consider:")
                print("    - Increasing n_epochs")
                print("    - Reducing learning rate for finer convergence")
                print("    - Reducing loss_threshold if used")

    # compute Hessian
    H = jax.hessian(loss_vec)(params_vec)

    # invert to get covariance
    cov = jnp.linalg.inv(H)

    # parameter errors
    errors = jnp.sqrt(jnp.diag(cov))

    error_dict = {k: float(errors[i]) for i, k in enumerate(keys)}

    return error_dict, cov

def fit_streamline(initial_opt_params, fixed_params, data, uncertainties, distance_pc,
                   learning_rate=0.001, param_bounds=None, n_epochs=1000,
                   beta1=0.9, beta2=0.999,
                   info_every=100, loss_threshold=None, loss_threshold_epochs=1,
                   gradient_tol=None, gradient_tol_epochs=1,
                   early_stopping_patience=50,
                   log_file=None, trace_file=None, trace_every=1,
                   output_uncertainties=False,
                   ):
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
        Adam learning rate applied uniformly to all normalized parameters.
    param_bounds : dict or None
        Parameter bounds in physical/log parameter units.
        Optimization is performed in normalized space using
        x_norm = (x - min) / (max - min), so bounds are required for all
        optimized keys and are used as normalization anchors.
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
    loss_threshold : float or None
        Optional absolute loss threshold for threshold-based stopping.
        If provided, optimization stops after loss is <= loss_threshold for
        loss_threshold_epochs consecutive epochs.
    loss_threshold_epochs : int
        Number of consecutive epochs with loss <= loss_threshold required to
        trigger threshold-based early stopping. Must be >= 1.
    gradient_tol : float or None
        Optional gradient norm tolerance for stopping in normalized space.
        If provided, optimization stops when the L2 norm of gradients with
        respect to normalized parameters
        is less than this threshold for gradient_tol_epochs consecutive epochs,
        indicating convergence.
    gradient_tol_epochs : int
        Number of consecutive epochs with ||grad|| < gradient_tol required to
        trigger normalized-space gradient norm-based early stopping. Must be >= 1.
        
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
    opt_param_keys = list(opt_params.keys())
    data = _coerce_data_tuple_float64(data)
    uncertainties = _coerce_data_tuple_float64(uncertainties)
    distance_pc = _to_float64(distance_pc)
    learning_rate = _to_float64(learning_rate)
    if not bool(jnp.isfinite(learning_rate)):
        raise ValueError(f'learning_rate must be finite. Got {learning_rate}.')
    if not bool(learning_rate > 0):
        raise ValueError(f'learning_rate must be > 0. Got {float(learning_rate)}.')
    param_bounds = _normalize_param_bounds(param_bounds)
    normalization_spec = _build_normalization_spec(opt_params, param_bounds)

    # Keep optimization variables in normalized coordinates; convert back to
    # physical/log units only when evaluating the forward model and diagnostics.
    opt_params_norm = _normalize_opt_params(opt_params, normalization_spec)

    # Use one global learning rate on normalized parameters.
    solver = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2)

    opt_state = solver.init(opt_params_norm)

    def loss_from_normalized(norm_opt_params):
        physical_opt_params = _denormalize_opt_params(norm_opt_params, normalization_spec)
        return chi2_loss(physical_opt_params, fixed_params, data, uncertainties, distance_pc)

    # Create gradient function in normalized space.
    loss_and_grad_fn = value_and_grad(loss_from_normalized)
    
    # Track loss history
    loss_history = []
    initial_loss = float(loss_from_normalized(opt_params_norm))
    best_loss = initial_loss
    best_opt_params = opt_params.copy()
    best_epoch = 0
    patience_counter = 0
    loss_threshold_counter = 0
    gradient_tol_counter = 0
    ordered_best_opt_params = {k: best_opt_params[k] for k in opt_param_keys}

    if trace_every < 1:
        raise ValueError('trace_every must be >= 1')
    if loss_threshold is not None:
        loss_threshold = float(loss_threshold)
        if not math.isfinite(loss_threshold):
            raise ValueError('loss_threshold must be finite when provided.')
        if loss_threshold_epochs < 1:
            raise ValueError('loss_threshold_epochs must be >= 1 when loss_threshold is provided.')
    if gradient_tol is not None:
        gradient_tol = float(gradient_tol)
        if not math.isfinite(gradient_tol):
            raise ValueError('gradient_tol must be finite when provided.')
        if gradient_tol <= 0:
            raise ValueError('gradient_tol must be positive when provided.')
        if gradient_tol_epochs < 1:
            raise ValueError('gradient_tol_epochs must be >= 1 when gradient_tol is provided.')
    
    # Initialize CSV log file if requested
    csv_file = None
    csv_writer = None
    if log_file is not None:
        csv_file = open(log_file, 'w', newline='')
        # Create header: epoch, loss, then all optimizable params
        fieldnames = ['epoch', 'loss'] + opt_param_keys
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
    print(f"Optimizing parameters: {opt_param_keys}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    if loss_threshold is not None:
        print(
            f"Threshold-based stopping enabled: loss <= {loss_threshold:.6g} "
            f"for {loss_threshold_epochs} consecutive epochs."
        )
    if gradient_tol is not None:
        print(
            f"Gradient norm stopping enabled (normalized space): ||grad|| < {gradient_tol:.6g} "
            f"for {gradient_tol_epochs} consecutive epochs."
        )
    print(f"Initial optimizable values:")
    for key in opt_param_keys:
        print(f"  {key}: {opt_params[key]:.3e}")
    
    # Log initial parameters and initial loss (epoch 0) if CSV logging is enabled
    if csv_writer is not None:
        row = {'epoch': 0, 'loss': initial_loss}
        for key in opt_param_keys:
            row[key] = float(opt_params[key])
        if 'log_omega' in opt_params:
            row['omega'] = float(_omega_from_log_omega(opt_params['log_omega']))
        csv_writer.writerow(row)
        csv_file.flush()

    if trace_csv_writer is not None:
        initial_loss_for_trace, initial_trace = chi2_loss(
            opt_params, fixed_params, data, uncertainties, distance_pc, return_trace=True)
        initial_norm_grads = loss_and_grad_fn(opt_params_norm)[1]
        initial_grad_norm = float(_gradient_l2_norm(initial_norm_grads))
        initial_trace_row = _build_trace_row(0, float(initial_loss_for_trace), initial_trace, initial_grad_norm)
        trace_csv_writer.writerow(initial_trace_row)
        trace_csv_file.flush()
    
    try:
        for epoch in range(1, n_epochs + 1):
            if epoch % info_every == 0:
                print(f"\n Starting Epoch {epoch} -------------------------")
            # Compute gradients at current normalized parameters (pre-update)
            _, norm_grads = loss_and_grad_fn(opt_params_norm)

            # Compute gradient norm in normalized space and reuse for
            # tracing/progress/stopping.
            grad_norm = float(_gradient_l2_norm(norm_grads))

            # Perform Optax Adam step in normalized space.
            updates, opt_state = solver.update(norm_grads, opt_state, params=opt_params_norm)
            opt_params_norm = optax.apply_updates(opt_params_norm, updates)

            # Enforce normalized bounds and map back to physical/log values.
            for key in opt_param_keys:
                opt_params_norm[key] = jnp.clip(opt_params_norm[key], 0.0, 1.0)
            opt_params = _denormalize_opt_params(opt_params_norm, normalization_spec)

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
                for key in opt_param_keys:
                    row[key] = float(opt_params[key])
                if 'log_omega' in opt_params:
                    row['omega'] = float(_omega_from_log_omega(opt_params['log_omega']))
                csv_writer.writerow(row)
                csv_file.flush()  # Ensure data is written after each epoch

            if trace_csv_writer is not None and loss_trace is not None:
                trace_row = _build_trace_row(epoch, loss_value, loss_trace, grad_norm)
                trace_csv_writer.writerow(trace_row)
                trace_csv_file.flush()
        
            # Early stopping checks
            if loss_value < best_loss:
                best_loss = loss_value
                best_opt_params = opt_params.copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if loss_threshold is not None:
                if loss_value <= loss_threshold:
                    loss_threshold_counter += 1
                else:
                    loss_threshold_counter = 0
            
            if gradient_tol is not None:
                if grad_norm < gradient_tol:
                    gradient_tol_counter += 1
                else:
                    gradient_tol_counter = 0
        
            # Print progress
            if epoch % info_every == 0:
                if gradient_tol is not None:
                    print(f'Epoch {epoch}/{n_epochs}, Loss: {loss_value:.6f}, Best Loss: {best_loss:.6f}, ||grad||: {grad_norm:.6e}')
                else:
                    print(f'Epoch {epoch}/{n_epochs}, Loss: {loss_value:.6f}, Best Loss: {best_loss:.6f}')

            # Early stopping conditions (any one is sufficient to stop)
            if loss_threshold is not None and loss_threshold_counter >= loss_threshold_epochs:
                print(
                    f"\nEarly stopping at epoch {epoch}: loss <= {loss_threshold:.6g} "
                    f"for {loss_threshold_epochs} consecutive epochs"
                )
                break

            if gradient_tol is not None and gradient_tol_counter >= gradient_tol_epochs:
                print(
                    f"\nEarly stopping at epoch {epoch}: normalized gradient norm {grad_norm:.6e} < {gradient_tol:.6e} "
                    f"for {gradient_tol_epochs} consecutive epochs"
                )
                break
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}: no improvement for {early_stopping_patience} epochs")
                break
    
        # restore canonical parameter order before returning
        ordered_best_opt_params = {k: best_opt_params[k] for k in opt_param_keys}

    finally:
        # Always close the CSV file if it was opened
        if csv_file is not None:
            csv_file.close()
            print(f"Optimization log saved to: {log_file}")
        if trace_csv_file is not None:
            trace_csv_file.close()
            print(f"Matching trace log saved to: {trace_file}")

    print(f"Optimization complete!")
    print(f"\nFinal loss: {best_loss:.6f}")
    print(f"Best-fit parameters found at epoch: {best_epoch}")
    for key in ordered_best_opt_params.keys():
        print(f"  {key}: {ordered_best_opt_params[key]:.3e}")

    # compute errors on best-fit parameters
    if output_uncertainties:
        print("\nEstimating parameter uncertainties from Hessian...")
        param_errors, cov_matrix = estimate_parameter_errors(
            ordered_best_opt_params,
            fixed_params,
            data,
            uncertainties,
            distance_pc,
            gradient_tol=gradient_tol,
            normalization_spec=normalization_spec,
        )
        print("\nParameter uncertainties (1-sigma):")
        for k, v in param_errors.items():
            print(f"  {k}: {v}")
    else:
        param_errors = None


    return _with_derived_omega(ordered_best_opt_params), loss_history, param_errors