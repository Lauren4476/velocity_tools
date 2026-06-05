'''
This file contains the loss function and optimisation routines for streamfit.

The optimisation uses adam (adaptive moment estimation) optimiser to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 02-06-26
'''

import jax.numpy as jnp
from jax import value_and_grad, lax
import jax
import optax
from . import stream_lines_grad
from . import extract_streamline
from . import outputs
import csv
import math

jax.config.update("jax_enable_x64", True)

# settings and constants
VR0_MIN = 1e-6

LOSS_METHOD_CHOICES = ('radecvel', 'rthetavel')

LOSS_METHOD_COMPONENT_KEYS = {
    'radecvel': ('chi2_ra', 'chi2_dec', 'chi2_v'),
    'rthetavel': ('chi2_r', 'chi2_theta', 'chi2_v'),
}

TRACE_COMMON_FIELDNAMES = [
    'epoch',
    'loss',
    # 'chi2_penalty',
    # 'low_shortfall_penalty',
    # 'low_excess_penalty',
    # 'high_shortfall_penalty',
    # 'high_excess_penalty',
    'chi2_total',
    'grad_norm',
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
    'overlap_metric_min',
    'overlap_metric_max',
]


def check_loss_method(loss_method):
    """Check that the selected loss method is valid and return it"""
    if loss_method not in LOSS_METHOD_CHOICES:
        raise ValueError(
            f"Unknown loss_method '{loss_method}'. "
            f"Choose from: {list(LOSS_METHOD_CHOICES)}"
        )
    return loss_method


def trace_fieldnames_for_loss_method(loss_method):
    """Return the trace csv headers for the chosen loss method"""
    loss_method = check_loss_method(loss_method)
    return ['epoch', 'loss', *LOSS_METHOD_COMPONENT_KEYS[loss_method], *TRACE_COMMON_FIELDNAMES[2:]]


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


def is_numeric_value(value):
    """Return True for scalar/array-like numeric values"""
    try:
        arr = jnp.asarray(value)
    except Exception:
        return False
    if arr.dtype == jnp.bool_:
        return False
    return bool(jnp.issubdtype(arr.dtype, jnp.number))


def to_float64(value):
    """Convert a numeric value or array-like input to float64"""
    return jnp.asarray(value, dtype=jnp.float64)


def make_opt_params_float64(opt_params):
    """Return optimisation parameters as float64"""
    coerced = {}
    for key, value in opt_params.items():
        if is_numeric_value(value):
            coerced[key] = to_float64(value)
        else:
            coerced[key] = value
    return coerced


def make_fixed_params_float64(fixed_params):
    """Return fixed-parameter dictionary with numeric values as float64"""
    coerced = {}
    for key, value in fixed_params.items():
        if value is None or isinstance(value, bool):
            coerced[key] = value
        elif is_numeric_value(value):
            coerced[key] = to_float64(value)
        else:
            coerced[key] = value
    return coerced


def make_data_tuple_float64(values):
    """Convert tuple/list of arrays to float64 arrays"""
    return tuple(to_float64(value) for value in values)


def clean_model_param_dict(params, dict_name):
    """Convert parameter dictionary to float64 and standardise it"""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"{dict_name} must be a dictionary, got {type(params).__name__}.")

    if dict_name == 'initial_opt_params':
        sanitized = make_opt_params_float64(params.copy())
    else:
        sanitized = make_fixed_params_float64(params.copy())

    if 'omega' in sanitized and 'log_omega' not in sanitized:
        sanitized['log_omega'] = jnp.log(to_float64(sanitized['omega']))
    if 'omega' in sanitized:
        del sanitized['omega']

    tiny = to_float64(1e-8)
    # Protect against exact polar-angle edge values which can cause
    # downstream numerical issues (theta=0 or theta=pi). 
    # If the uservsupplied exactly 0 or pi, 
    # nudge by a tiny amount into the open interval (0, pi).
    if 'theta0' in sanitized:
        try:
            theta_val = to_float64(sanitized['theta0'])
            if bool(jnp.all(jnp.isclose(theta_val, to_float64(0.0)))):
                sanitized['theta0'] = theta_val + tiny
            elif bool(jnp.all(jnp.isclose(theta_val, to_float64(jnp.pi)))):
                sanitized['theta0'] = theta_val - tiny
        except Exception:
            pass

    unknown = sorted(key for key in sanitized if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown parameter keys in {dict_name}: {unknown} "
            f"Supported keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return sanitized


def check_param_types(opt_params, fixed_params):
    """Check that model parameters are of the correct type (numeric or None for rmin)"""
    for key, value in opt_params.items():
        if key == 'rmin' and value is None:
            raise ValueError("'rmin' cannot be None")
        if isinstance(value, bool) or not is_numeric_value(value):
            raise TypeError(
                f"Optimisable parameter '{key}' must be numeric, "
                f"got value of type {type(value).__name__}."
            )

    for key, value in fixed_params.items():
        if key == 'rmin' and value is None:
            continue
        if isinstance(value, bool) or not is_numeric_value(value):
            raise TypeError(
                f"Fixed parameter '{key}' must be numeric"
                " (or None only for 'rmin'), "
                f"got value of type {type(value).__name__}."
            )


def sanitize_param_partition(initial_opt_params, fixed_params, require_nonempty_opt=False):
    """Sanitize and validate opt/fixed parameter partition for streamline modeling"""
    opt_params = clean_model_param_dict(initial_opt_params, 'initial_opt_params')
    fixed_params = clean_model_param_dict(fixed_params, 'fixed_params')

    overlap = sorted(set(opt_params) & set(fixed_params))
    if overlap:
        raise KeyError(
            f"Parameters cannot be present in both initial_opt_params and fixed_params! Overlap: {overlap}"
        )

    missing = []
    for key in STREAMLINE_MODEL_PARAM_KEYS:
        if key not in opt_params and key not in fixed_params:
            missing.append(key)
    if missing:
        raise KeyError(
            "Missing required streamline parameters across initial_opt_params and fixed_params: "
            f"{missing}. The list of parameters is: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    if require_nonempty_opt and len(opt_params) == 0:
        raise ValueError(
            "initial_opt_params must contain at least one optimisable parameter. "
        )

    check_param_types(opt_params, fixed_params)

    return opt_params, fixed_params


def prepare_model_params(opt_params, fixed_params):
    """Construct merged model parameters and clean opt/fixed dictionaries"""
    opt_params, fixed_params = sanitize_param_partition(opt_params, fixed_params)
    model_params = fixed_params.copy()
    model_params.update(opt_params)
    return model_params, opt_params, fixed_params


def standardise_param_bounds(param_bounds):
    """Check/standardise parameter-bound keys and convert omega bounds to log-space."""
    if param_bounds is None:
        return None

    standardised = dict(param_bounds)
    if 'omega' in standardised:
        if 'log_omega' in standardised:
            raise KeyError(
                "param_bounds contains both 'omega' and 'log_omega'. "
                "Please provide only one of these."
            )
        omega_min, omega_max = standardised.pop('omega')
        omega_min = float(omega_min)
        omega_max = float(omega_max)
        if omega_min <= 0 or omega_max <= 0:
            raise ValueError("'omega' bounds must be positive")
        if omega_min >= omega_max:
            raise ValueError("'omega' bounds must satisfy omega_min < omega_max")
        standardised['log_omega'] = (math.log(omega_min), math.log(omega_max))

    unknown = sorted(key for key in standardised if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown params in param_bounds: {unknown}. "
            f"Supported params are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return standardised


def build_normalisation_spec(opt_params, param_bounds):
    """Build shift and scale for normalisation ofoptimised parameters, from bounds."""
    if param_bounds is None:
        raise ValueError(
            "param_bounds is required because optimisation is performed in normalised space. "
            "Provide bounds for every parameter you want to optimise."
        )
 
    missing = []
    for key in opt_params:        
        if key not in param_bounds:
            missing.append(key)
    if missing:
        raise ValueError(
            "Missing bounds for optimised parameters: "
            f"{missing}. Please add (min, max) entries for all parameters you want to optimise."
        )

    normalisation_spec = {}
    for key, value in opt_params.items():
        bounds = param_bounds[key]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for '{key}' must be a 2-element (min, max) tuple."
                f"Got: {bounds!r}"
            )

        lower_bound = to_float64(bounds[0])
        upper_bound = to_float64(bounds[1])
        if not bool(jnp.isfinite(lower_bound)) or not bool(jnp.isfinite(upper_bound)):
            raise ValueError(f"Bounds for '{key}' must be finite. Got ({lower_bound}, {upper_bound})")
        if not bool(upper_bound > lower_bound):
            raise ValueError(
                f"Bounds for '{key}' must satisfy min < max. Got ({float(lower_bound)}, {float(upper_bound)})"
            )

        value = to_float64(value)
        if not bool((value >= lower_bound) & (value <= upper_bound)):
            raise ValueError(
                f"Initial value for '{key}' ({float(value)}) is outside bounds"
                f"({float(lower_bound)}, {float(upper_bound)})."
            )

        scale = upper_bound - lower_bound
        normalisation_spec[key] = {
            'offset': lower_bound,
            'scale': scale,
        }

    return normalisation_spec


def normalise_opt_params(opt_params, normalisation_spec):
    """normalise optimised parameters to [0, 1]"""
    normalised = {}
    for key, value in opt_params.items():
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        normalised[key] = (to_float64(value) - offset) / scale
    return normalised


def denormalise_opt_params(norm_opt_params, normalisation_spec):
    """Convert normalised optimised parameters back to physical/log parameter values"""
    denormalised = {}
    for key, value in norm_opt_params.items():
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        denormalised[key] = to_float64(value) * scale + offset
    return denormalised


def params_dict_to_vector(opt_params):
    """Convert parameter dict to ordered vector"""
    keys = list(opt_params.keys())
    vec = jnp.array([opt_params[k] for k in keys], dtype=jnp.float64)
    return vec, keys


def vector_to_params_dict(vec, keys):
    """Convert parameter vector back to dict"""
    return {k: vec[i] for i, k in enumerate(keys)}


def with_derived_omega(opt_params):
    """Return a copy of the opt params including omega, when it is available"""
    params_with_omega = opt_params.copy()
    if 'log_omega' in params_with_omega and 'omega' not in params_with_omega:
        params_with_omega['omega'] = jnp.exp(params_with_omega['log_omega'])
    return params_with_omega

def build_trace_row(epoch, loss_value, loss_trace, grad_norm, loss_method):
    """Flatten trace dictionary into a CSV row for output"""
    loss_method = check_loss_method(loss_method)
    chi2_components = loss_trace.get('chi2_components', {})
    matching = loss_trace.get('matching', {})
    model_metric_trace = matching.get('distance_metric_model', {})
    data_metric_trace = matching.get('distance_metric_data', {})

    row = {
        'epoch': epoch,
        'loss': loss_value,
    }
    for component_key in LOSS_METHOD_COMPONENT_KEYS[loss_method]:
        row[component_key] = chi2_components.get(component_key, float('nan'))

    row.update({
        # 'low_shortfall_penalty': chi2_components.get('low_shortfall_penalty', float('nan')),
        # 'low_excess_penalty': chi2_components.get('low_excess_penalty', float('nan')),
        # 'high_shortfall_penalty': chi2_components.get('high_shortfall_penalty', float('nan')),
        # 'high_excess_penalty': chi2_components.get('high_excess_penalty', float('nan')),
        # 'chi2_penalty': chi2_components.get('chi2_penalty', float('nan')),
        'chi2_total': chi2_components.get('chi2_total', float('nan')),
        'grad_norm': grad_norm,
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
        'overlap_metric_min': matching.get('overlap_metric_min', float('nan')),
        'overlap_metric_max': matching.get('overlap_metric_max', float('nan')),
    })

    return row

def trace_tree_to_python(value):
    """Go through the trace tree and convert JAX arrays to Python scalars where possible"""
    # go through containers, converting JAX arrays to Python scalars where possible, and leaving non-numeric values as-is
    if isinstance(value, dict):
        return {key: trace_tree_to_python(v) for key, v in value.items()}
    if isinstance(value, list):
        return [trace_tree_to_python(v) for v in value]
    if isinstance(value, tuple):
        return tuple(trace_tree_to_python(v) for v in value)
    # preserve Nones as-is
    if value is None:
        return None
    # convert jax arrays to python scalars where posible
    try:
        array_value = jnp.asarray(value)
    except Exception:
        return value
    # if it's a scalar array, convert to scalar
    if array_value.ndim == 0:
        return array_value.item()
    return value


@jax.jit
def gradient_l2_norm(grad_tree):
    """Compute L2 norm of gradients across all leaves in a pytree
    (a pytree is a nested structure of lists/dicts/tuples containing arrays, 
    used by jax for gradients)."""
    grad_leaves = jax.tree_util.tree_leaves(grad_tree)
    grad_sum_sq = jnp.asarray(0.0, dtype=jnp.float64)
    for grad_leaf in grad_leaves:
        grad_sum_sq = grad_sum_sq + jnp.sum(jnp.square(grad_leaf))
    return jnp.sqrt(grad_sum_sq)

@jax.jit
# obsolete - penalties not used anymore
def softplus_barrier(value, tau):
    """Smooth approximation to max(0, value) with transition scale tau."""
    tau = to_float64(tau)
    return tau * jnp.logaddexp(to_float64(0.0), to_float64(value) / tau)

@jax.jit
# obsolete - penalties not used anymore
def coverage_penalties(dmetric_model, model_finite_mask, data_min, data_max):
    """Penalise differences between model and data coverage in distance metric,
    using smooth barrier functions to keep differentiability"""
    margin = to_float64(0.2 * (data_max - data_min))
    tau = 0.8 * margin

    model_metric_for_min = jnp.where(model_finite_mask, dmetric_model, jnp.inf)
    model_metric_for_max = jnp.where(model_finite_mask, dmetric_model, -jnp.inf)
    model_min = jnp.min(model_metric_for_min)
    model_max = jnp.max(model_metric_for_max)

    #model starts too far out
    low_shortfall = softplus_barrier(model_min - data_min + margin, tau)
    #model starts too far in
    low_excess = softplus_barrier(data_min - model_min - margin, tau)
    #model ends too far in
    high_shortfall = softplus_barrier(data_max - model_max + margin, tau)
    #model ends too far out
    high_excess = softplus_barrier(model_max - data_max - margin, tau)


    low_penalty = ((low_shortfall / margin) ** 2 + (low_excess / margin) **2)
    high_penalty = ((high_shortfall / margin) ** 2 + (high_excess / margin) ** 2)
    total_penalty = (low_penalty + high_penalty)

    return total_penalty, low_shortfall, low_excess, high_shortfall, high_excess



def forward_model(opt_params, fixed_params, distance_pc):
    """
    Run the forward model using stream_lines_grad.xyz_stream
    
    Parameters:
    -----------
    opt_params : dict
        Dictionary containing optimisable parameters (any subset of
        STREAMLINE_MODEL_PARAM_KEYS)
    fixed_params : dict
        Dictionary containing fixed parameters (the complementary subset)
        Together with opt_params, this must define all keys in
        STREAMLINE_MODEL_PARAM_KEYS exactly once
    distance_pc : float
        Distance to source in parsecs
        
    Returns:
    --------
    tuple: (ra_offsets, dec_offsets, velocities)
        - RA offsets in arcsec (negative for standard convention)
        - Dec offsets in arcsec
        - Line-of-sight velocities in km/s, relative to v_lsr
    """
    model_params, _, _ = prepare_model_params(opt_params, fixed_params)
    distance_pc = to_float64(distance_pc)

    omega = jnp.exp(model_params['log_omega'])

    # Protect near-zero v_r0 from creating singularities in physics calculations
    # Allow negative v_r0, but replace exact-zero or tiny values with signed epsilon
    v_r0_protected = model_params['v_r0']
    threshold = to_float64(1e-6)
    v_r0_protected = jnp.where(
        jnp.isclose(v_r0_protected, to_float64(0.0)),
        - jnp.sign(v_r0_protected) * threshold,
        v_r0_protected
        )

    # Run the forward model - returns positions in au, velocities in km/s
    (x, y, z), (vx, vy, vz) = stream_lines_grad.xyz_stream(
        mass=model_params['mass'],
        r0=model_params['r0'],
        theta0=model_params['theta0'],
        phi0=model_params['phi0'],
        omega=omega,
        v_r0=v_r0_protected,
        inc=model_params['inc'],
        pa=model_params['pa'],
        rmin=model_params['rmin'],
        deltar=model_params['deltar']
    )
     
    # Convert positions from au to arcsec offsets
    # x = RA offset (with negative for standard RA convention)
    # z = Dec offset
    # y = line-of-sight velocity
    ra_model = -x / distance_pc  # arcsec
    dec_model = z / distance_pc  # arcsec
    # make velocity absolute by adding back v_lsr
    v_model = vy + model_params['v_lsr']  # km/s 

    return ra_model, dec_model, v_model


def distance_metric_overlap(dmetric_model, model_finite_mask, dmetric_data, data_finite_mask):
    """Compute the overlapping range in the streamline distance metric between data and model"""
    model_metric = dmetric_model[model_finite_mask]
    data_metric = dmetric_data[data_finite_mask]

    model_min = jnp.min(model_metric)
    model_max = jnp.max(model_metric)
    data_min = jnp.min(data_metric)
    data_max = jnp.max(data_metric)

    overlap_min = jnp.maximum(model_min, data_min)
    overlap_max = jnp.minimum(model_max, data_max)
    return model_min, model_max, data_min, data_max, overlap_min, overlap_max

@jax.jit
def order_model_by_metric(dmetric_model, ra_model, dec_model, v_model, sort_tol=1e-12):
    """Order finite model support by distance metric, skipping argsort when already monotonic."""
    dmetric_model = to_float64(dmetric_model)
    ra_model = to_float64(ra_model)
    dec_model = to_float64(dec_model)
    v_model = to_float64(v_model)

    if dmetric_model.size <= 1:
        return dmetric_model, ra_model, dec_model, v_model

    sort_tol = to_float64(sort_tol)
    d_diff = jnp.diff(dmetric_model)
    ascending = jnp.all(d_diff >= -sort_tol)
    descending = jnp.all(d_diff <= sort_tol)

    def keep_order(_):
        return dmetric_model, ra_model, dec_model, v_model

    def reverse_order(_):
        return dmetric_model[::-1], ra_model[::-1], dec_model[::-1], v_model[::-1]

    def sorted_order(_):
        sort_idx = jnp.argsort(dmetric_model)
        return (dmetric_model[sort_idx], ra_model[sort_idx], dec_model[sort_idx], v_model[sort_idx])

    return lax.cond(
        ascending,
        keep_order,
        lambda _: lax.cond(descending, reverse_order, sorted_order, operand=None),
        operand=None,
    )


def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data, return_trace=False):
    """
    Extract model values corresponding to data positions using the distance metric from
    extract_streamline.get_distance_metric

    Method:
    1. Compute the distance metric for model and data points
    2. Apply finite masks
    3. Normalise both metrics to [0, 1] based on their finite ranges
    4. Map data normalised positions to model normalised positions
    5. Interpolate model RA, Dec, and velocity at the mapped positions

    Parameters
    ----------
    return_trace : bool
        If True, also return a trace dictionary containing diagnostics

    Returns
    -------
    ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model, matching_trace
        where valid is a boolean mask with shape len(original data), marking
        retained data points
    """

    ra_model = to_float64(ra_model)
    dec_model = to_float64(dec_model)
    v_model = to_float64(v_model)
    ra_data = to_float64(ra_data)
    dec_data = to_float64(dec_data)

    # Compute the streamline distance metric for model/data in float64.
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

    # only finite values are valid
    model_valid_mask = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )
    data_valid_mask = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )

    d_model_f = dmetric_model[model_valid_mask]
    ra_model_f = ra_model[model_valid_mask]
    dec_model_f = dec_model[model_valid_mask]
    v_model_f = v_model[model_valid_mask]
    d_data_f = dmetric_data[data_valid_mask]

    # filter model to keep only model points with dmetric >= minimum of data dmetric
    # this is becuase the model shouldn't go further in than the innermost data point
    # as this is where we no longer observe the streamer
    dmetric_min_data = jnp.min(d_data_f)
    model_keep_mask = d_model_f >= dmetric_min_data

    d_model_f = d_model_f[model_keep_mask]
    ra_model_f = ra_model_f[model_keep_mask]
    dec_model_f = dec_model_f[model_keep_mask]
    v_model_f = v_model_f[model_keep_mask]

    model_points_total = ra_model.size
    data_points_total = ra_data.size
    model_nan_count = model_points_total - jnp.sum(model_valid_mask)
    data_nan_count = data_points_total - jnp.sum(data_valid_mask)



    # check there are enough points for matching
    model_valid_points = d_model_f.size
    data_valid_points = d_data_f.size
    model_has_enough = model_valid_points >= 2
    data_has_any = data_valid_points >= 1


    d_model_sorted, ra_sorted, dec_sorted, v_sorted = order_model_by_metric(
        d_model_f,
        ra_model_f,
        dec_model_f,
        v_model_f,
    )

    # model diffs and stats for trace
    d_diff_mod = jnp.diff(d_model_sorted)

    model_metric_min_gap = jnp.where(d_diff_mod.size > 0, jnp.min(d_diff_mod), jnp.nan)
    model_metric_near_tie_count = jnp.where(d_diff_mod.size > 0, jnp.sum(jnp.abs(d_diff_mod) <= 1e-8), 0)
    model_metric_duplicate_count = jnp.where(d_diff_mod.size > 0, jnp.sum(d_diff_mod == 0.0), 0)
    model_metric_non_monotonic_count = jnp.where(d_diff_mod.size > 0, jnp.sum(d_diff_mod < 0.0), 0)

    # metric ranges
    model_min = jnp.min(d_model_sorted)
    model_max = jnp.max(d_model_sorted)
    data_min = jnp.min(d_data_f)
    data_max = jnp.max(d_data_f)
    model_span = model_max - model_min
    data_span = data_max - data_min
    model_span_safe = jnp.where(model_span == 0.0, 1.0, model_span)
    data_span_safe = jnp.where(data_span == 0.0, 1.0, data_span)

    # normalise to [0, 1] and map data metric to model metric space
    d_data_norm = (d_data_f - data_min) / data_span_safe
    d_model_goal = model_min + d_data_norm * model_span_safe

    # interpolation to get model values at the exact goal positions
    ra_model_interp = jnp.interp(d_model_goal, d_model_sorted, ra_sorted)
    dec_model_interp = jnp.interp(d_model_goal, d_model_sorted, dec_sorted)
    v_model_interp = jnp.interp(d_model_goal, d_model_sorted, v_sorted)

    valid = data_valid_mask


    if not return_trace:
        return ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model
    
    matching_trace = {
        "model_points_total": model_points_total,
        "model_nan_count": model_nan_count,
        "model_valid_points": model_valid_points,
        "data_points_total": data_points_total,
        "data_nan_count": data_nan_count,
        "data_valid_points": data_valid_points,
        "model_metric_min_gap": model_metric_min_gap,
        "model_metric_near_tie_count": model_metric_near_tie_count,
        "model_metric_duplicate_count": model_metric_duplicate_count,
        "model_metric_non_monotonic_count": model_metric_non_monotonic_count,
        "model_metric_min": model_min,
        "model_metric_max": model_max,
        "data_metric_min": data_min,
        "data_metric_max": data_max,
        "model_metric_span": model_span,
        "data_metric_span": data_span,
    }

    return ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model, matching_trace


def chi2_loss_raw(
    opt_params,
    fixed_params,
    distance_pc,
    prepared_data,
    return_trace=False,
    loss_method='radecvel',
):
    """Compute chi-squared loss and optionally return a diagnostic trace tree"""

    loss_method = check_loss_method(loss_method)

    opt_params, fixed_params = sanitize_param_partition(opt_params, fixed_params)
    distance_pc = to_float64(distance_pc)

    ra_data = prepared_data.ra_data
    dec_data = prepared_data.dec_data
    v_data = prepared_data.v_data
    ra_sigma = prepared_data.ra_sigma_safe
    dec_sigma = prepared_data.dec_sigma_safe
    v_sigma = prepared_data.v_sigma_safe

    ra_model, dec_model, v_model = forward_model(opt_params, fixed_params, distance_pc)

    ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model = (
        match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data)
    )

    dmetric_data = prepared_data.dmetric_data

    model_finite_mask = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )

    # penalties not used anymore
    # chi2_penalty, low_shortfall, low_excess, high_shortfall, high_excess  = coverage_penalties(
    #     dmetric_model,
    #     model_finite_mask,
    #     prepared_data.data_min,
    #     prepared_data.data_max,
    # )

    # Only compute chi2 on valid/retained data points to avoid penalizing points outside overlap domain
    chi2_v = jnp.sum((((v_data[valid] - v_model_interp[valid]) / v_sigma[valid]) ** 2))

    if loss_method == 'radecvel':
        chi2_ra = jnp.sum((((ra_data[valid] - ra_model_interp[valid]) / ra_sigma[valid]) ** 2))
        chi2_dec = jnp.sum((((dec_data[valid] - dec_model_interp[valid]) / dec_sigma[valid]) ** 2))
        chi2_total = chi2_ra + chi2_dec + chi2_v # + chi2_penalty
    else:
        r_proj_data = prepared_data.r_proj_data
        theta_proj_data = prepared_data.theta_proj_data
        r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(
            ra_model_interp,
            dec_model_interp,
        )

        dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)

        sigma_r = jnp.sqrt(ra_sigma**2 + dec_sigma**2)
        r_eps = to_float64(1e-8)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt(((dec_data * ra_sigma)**2 + (ra_data * dec_sigma)**2)) / (r_safe**2)
        sigma_theta = jnp.maximum(sigma_theta, r_eps)

        # Only compute chi2 on valid/retained data points
        chi2_r = jnp.sum((((r_proj_data[valid] - r_proj_model[valid]) / sigma_r[valid]) ** 2))
        chi2_theta = jnp.sum(((dtheta[valid] / sigma_theta[valid]) ** 2))
        chi2_total = chi2_r + chi2_theta + chi2_v # + chi2_penalty

    if not return_trace:
        return chi2_total

    data_finite_mask = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )

    model_min, model_max, data_min, data_max, overlap_min, overlap_max = distance_metric_overlap(
        dmetric_model,
        model_finite_mask,
        dmetric_data,
        data_finite_mask,
    )

    model_nan_count = jnp.sum(~model_finite_mask)
    model_points_total = ra_model.size
    model_valid_points = model_points_total - model_nan_count

    data_keep = data_finite_mask & (dmetric_data >= overlap_min) & (dmetric_data <= overlap_max)
    model_keep = model_finite_mask & (dmetric_model >= overlap_min) & (dmetric_model <= overlap_max)

    sort_idx = jnp.argsort(dmetric_model)
    d_model_sorted = dmetric_model[sort_idx]
    d_diff = jnp.diff(d_model_sorted)
    if d_diff.size > 0:
        model_metric_min_gap = jnp.min(d_diff)
        model_metric_near_tie_count = jnp.sum(jnp.abs(d_diff) <= 1e-8)
        model_metric_duplicate_count = jnp.sum(d_diff == 0.0)
        model_metric_non_monotonic_count = jnp.sum(d_diff < 0.0)
    else:
        model_metric_min_gap = to_float64(float('nan'))
        model_metric_near_tie_count = to_float64(0.0)
        model_metric_duplicate_count = to_float64(0.0)
        model_metric_non_monotonic_count = to_float64(0.0)

    model_metric_span = d_model_sorted[-1] - d_model_sorted[0] if d_model_sorted.size > 1 else to_float64(0.0)

    if loss_method == 'radecvel':
        chi2_components = {
            'chi2_ra': chi2_ra,
            'chi2_dec': chi2_dec,
            'chi2_v': chi2_v,
            # 'chi2_penalty': chi2_penalty,
            # 'low_shortfall_penalty': low_shortfall,
            # 'low_excess_penalty': low_excess,
            # 'high_shortfall_penalty': high_shortfall,
            # 'high_excess_penalty': high_excess,
            'overlap_width': overlap_max - overlap_min,
            'chi2_total': chi2_total,
        }
    else:
        chi2_components = {
            'chi2_r': chi2_r,
            'chi2_theta': chi2_theta,
            'chi2_v': chi2_v,
            # 'chi2_penalty': chi2_penalty,
            # 'low_shortfall_penalty': low_shortfall,
            # 'low_excess_penalty': low_excess,
            # 'high_shortfall_penalty': high_shortfall,
            # 'high_excess_penalty': high_excess,
            'overlap_width': overlap_max - overlap_min,
            'chi2_total': chi2_total,
        }

    matching_trace = {
        'model_points_total': model_points_total,
        'model_nan_count': model_nan_count,
        'model_valid_points': model_valid_points,
        'model_retained_count': jnp.sum(model_keep),
        'data_points_total': ra_data.size,
        'data_valid_points': jnp.sum(data_finite_mask),
        'data_retained_count': jnp.sum(data_keep),
        'overlap_metric_min': overlap_min,
        'overlap_metric_max': overlap_max,
        'model_metric_span': model_metric_span,
        'model_metric_min_gap': model_metric_min_gap,
        'model_metric_near_tie_count': model_metric_near_tie_count,
        'model_metric_duplicate_count': model_metric_duplicate_count,
        'model_metric_non_monotonic_count': model_metric_non_monotonic_count,
    }

    loss_trace = {
        'chi2_components': chi2_components,
        'matching': matching_trace,
        'loss_method': loss_method,
    }
    return chi2_total, loss_trace


def chi2_loss(
    opt_params,
    fixed_params,
    distance_pc,
    prepared_data,
    return_trace=False,
    loss_method='radecvel',
):
    """
    Compute chi-squared loss between model and data using one of two modes:
    - 'radecvel': RA, Dec, and LOS velocity residuals
    - 'rthetavel': projected radial distance, polar angle, and LOS velocity residuals
    
    Parameters:
    -----------
    opt_params : dict
        optimisable streamline model parameters (any subset of
        STREAMLINE_MODEL_PARAM_KEYS).
    fixed_params : dict
        Fixed streamline model parameters (complementary subset).
    distance_pc : float
        Distance to source in parsecs
    prepared_data : PreparedData
        Precomputed data-only quantities (distance metrics, bounds, polar coords).
        Created via extract_streamline.prepare_data(data, uncertainties).

        
            Created via extract_streamline.prepare_data(data, uncertainties).
    --------
    float: Chi-squared loss value
    """

    loss_method = check_loss_method(loss_method)

    opt_params, fixed_params = sanitize_param_partition(opt_params, fixed_params)
    distance_pc = to_float64(distance_pc)

    ra_data = prepared_data.ra_data
    dec_data = prepared_data.dec_data
    v_data = prepared_data.v_data
    ra_sigma = prepared_data.ra_sigma_safe
    dec_sigma = prepared_data.dec_sigma_safe
    v_sigma = prepared_data.v_sigma_safe

    # Run forward model
    ra_model, dec_model, v_model = forward_model(opt_params, fixed_params, distance_pc)
    

    # Match model to data using arc-length parameterisation
    if return_trace:
        ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model, matching_trace = (
            match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data, return_trace=True)
        )
    else:
        ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model = (
            match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data)
        )

    model_finite_mask = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )

    # penalties not used anymore
    # chi2_penalty, low_shortfall, low_excess, high_shortfall, high_excess = coverage_penalties(
    #     dmetric_model,
    #     model_finite_mask,
    #     prepared_data.data_min,
    #     prepared_data.data_max,
    # )

    # Only compute chi2 on valid/retained data points to avoid penalizing points outside overlap domain
    chi2_v = jnp.sum((((v_data[valid] - v_model_interp[valid]) / v_sigma[valid]) ** 2))

    if loss_method == 'radecvel':
        chi2_ra = jnp.sum((((ra_data[valid] - ra_model_interp[valid]) / ra_sigma[valid]) ** 2))
        chi2_dec = jnp.sum((((dec_data[valid] - dec_model_interp[valid]) / dec_sigma[valid]) ** 2))
        chi2_total = chi2_ra + chi2_dec + chi2_v # + chi2_penalty
    else:
        # r/theta are defined on the projected plane of the sky from (RA, Dec).
        # Use precomputed data coordinates
        r_proj_data = prepared_data.r_proj_data
        theta_proj_data = prepared_data.theta_proj_data
        r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(
            ra_model_interp,
            dec_model_interp,
        )

        dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)

        sigma_r = jnp.sqrt(ra_sigma**2 + dec_sigma**2)
        r_eps = to_float64(1e-8)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt(((dec_data * ra_sigma)**2 + (ra_data * dec_sigma)**2)) / (r_safe**2)
        sigma_theta = jnp.maximum(sigma_theta, r_eps)

        # Only compute chi2 on valid/retained data points
        chi2_r = jnp.sum((((r_proj_data[valid] - r_proj_model[valid]) / sigma_r[valid]) ** 2))
        chi2_theta = jnp.sum(((dtheta[valid] / sigma_theta[valid]) ** 2))
        chi2_total = chi2_r + chi2_theta + chi2_v # + chi2_penalty

    if return_trace:
        if loss_method == 'radecvel':
            chi2_components = {
                'chi2_ra': float(chi2_ra),
                'chi2_dec': float(chi2_dec),
                'chi2_v': float(chi2_v),
                # 'chi2_penalty': float(chi2_penalty),
                # 'low_shortfall_penalty': float(low_shortfall),
                # 'low_excess_penalty': float(low_excess),
                # 'high_shortfall_penalty': float(high_shortfall),
                # 'high_excess_penalty': float(high_excess),
                'chi2_total': float(chi2_total),
            }
        else:
            chi2_components = {
                'chi2_r': float(chi2_r),
                'chi2_theta': float(chi2_theta),
                'chi2_v': float(chi2_v),
                # 'chi2_penalty': float(chi2_penalty),
                # 'low_shortfall_penalty': float(low_shortfall),
                # 'low_excess_penalty': float(low_excess),
                # 'high_shortfall_penalty': float(high_shortfall),
                # 'high_excess_penalty': float(high_excess),
                'chi2_total': float(chi2_total),
            }

        loss_trace = {
            'chi2_components': chi2_components,
            'matching': matching_trace,
            'loss_method': loss_method,
        }
        return chi2_total, loss_trace

    return chi2_total


def estimate_parameter_errors(
    best_opt_params,
    fixed_params,
    data,
    uncertainties,
    distance_pc,
    prepared_data,
    loss_method='radecvel',
    gradient_tol=1e-1,
    normalisation_spec=None,
):
    """
    Estimate parameter uncertainties using Hessian of chi2 loss.

    Parameters
    ----------
    prepared_data : PreparedData
        Precomputed data-only quantities (created via extract_streamline.prepare_data).
    gradient_tol : float or None
        Tolerance on gradient norm in normalised space. If provided and
        normalised-space gradient norm > gradient_tol at best params, a
        warning is issued because the quadratic approximation may not be valid.
    normalisation_spec : dict or None
        Bounds-derived normalisation metadata for optimised parameters.
        Required to evaluate gradient_tol in normalised space.

    Returns
    -------
    dict
        1-sigma uncertainties for each optimisable parameter
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
    params_vec, keys = params_dict_to_vector(best_opt_params)
    loss_method = check_loss_method(loss_method)

    def loss_vec(theta_vec):
        params = vector_to_params_dict(theta_vec, keys)
        return chi2_loss(
            params,
            fixed_params,
            distance_pc,
            prepared_data,
            loss_method=loss_method,
        )

    # Check gradient magnitude at best-fit parameters in normalised space.
    if gradient_tol is not None:
        if normalisation_spec is None:
            print(
                "WARNING: gradient_tol is interpreted in normalised space, but "
                "normalisation_spec was not provided. Skipping gradient_tol check "
                "for uncertainty estimation."
            )
        else:
            missing_norm_keys = [key for key in keys if key not in normalisation_spec]
            if missing_norm_keys:
                raise ValueError(
                    "normalisation_spec is missing optimised parameter keys required "
                    f"for gradient_tol check: {missing_norm_keys}"
                )

            norm_opt_params = normalise_opt_params(best_opt_params, normalisation_spec)
            norm_params_vec, _ = params_dict_to_vector(norm_opt_params)

            def norm_loss_vec(theta_norm_vec):
                norm_params = vector_to_params_dict(theta_norm_vec, keys)
                physical_params = denormalise_opt_params(norm_params, normalisation_spec)
                return chi2_loss(
                    physical_params,
                    fixed_params,
                    distance_pc,
                    prepared_data,
                    loss_method=loss_method,
                )

            norm_grad_vec = jax.grad(norm_loss_vec)(norm_params_vec)
            norm_grad_norm = float(gradient_l2_norm(norm_grad_vec))

            if norm_grad_norm > gradient_tol:
                print(
                    "WARNING: normalised-space gradient norm at best fit = "
                    f"{norm_grad_norm:.3e} exceeds tolerance {gradient_tol:.3e}"
                )
                print("optimisation may not have reached a minimum yet.")
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
                   loss_method='radecvel',
                   output_uncertainties=False,
                   ):
    """
    Fit streamline model parameters to data using Adam optimiser.
    Any supported streamline parameter can be optimised or fixed.
    Parameters are split by dictionary membership:
    - keys in initial_opt_params are optimised
    - keys in fixed_params are held fixed
    The union must contain each key in STREAMLINE_MODEL_PARAM_KEYS exactly once.
    
    Parameters:
    -----------
    initial_opt_params : dict
        Initial guesses for the parameters to optimise.
        Allowed keys are STREAMLINE_MODEL_PARAM_KEYS.
    fixed_params : dict
        Fixed (non-optimised) parameters using the same key space.
        Together with initial_opt_params, this must provide a full,
        non-overlapping partition of STREAMLINE_MODEL_PARAM_KEYS.
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data
    distance_pc : float
            Distance to source in parsecs
    learning_rate : float
        Adam learning rate applied uniformly to all normalised parameters.
    param_bounds : dict or None
        Parameter bounds in physical/log parameter units.
        optimisation is performed in normalised space using
        x_norm = (x - min) / (max - min), so bounds are required for all
        optimised keys and are used as normalisation anchors.
        You may provide 'omega' bounds as linear bounds; these are converted
        to 'log_omega' bounds internally.
    n_epochs : int
        Maximum number of optimisation iterations
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
    loss_method : str
        Loss definition to use. Options:
        - 'radecvel': optimise RA, Dec, and velocity residuals.
        - 'rthetavel': optimise radial distance, polar angle, and velocity residuals.
        Both options use the same model-data matching and overlap penalty.
    loss_threshold : float or None
        Optional absolute loss threshold for threshold-based stopping.
        If provided, optimisation stops after loss is <= loss_threshold for
        loss_threshold_epochs consecutive epochs.
    loss_threshold_epochs : int
        Number of consecutive epochs with loss <= loss_threshold required to
        trigger threshold-based early stopping. Must be >= 1.
    gradient_tol : float or None
        Optional gradient norm tolerance for stopping in normalised space.
        If provided, optimisation stops when the L2 norm of gradients with
        respect to normalised parameters
        is less than this threshold for gradient_tol_epochs consecutive epochs,
        indicating convergence.
    gradient_tol_epochs : int
        Number of consecutive epochs with ||grad|| < gradient_tol required to
        trigger normalised-space gradient norm-based early stopping. Must be >= 1.
        
    **IMPORTANT: Epoch and Loss Semantics**
    
    CSV and trace logging includes epochs 0 through N:
    - Epoch 0: loss and parameters at INITIAL state (before any updates)
    - Epoch i (1 <= i <= N): loss and parameters AFTER applying update i
    
    This means:
    - Loss at epoch i = loss evaluated at the state after i updates have been applied
    - Parameters at epoch i = parameters after i updates have been applied
    - epoch 0 loss = initial loss (loss before epoch 1 update)
    - epoch N loss = final loss (after epoch N update)
    
    Loss history is 0-indexed and aligned with epoch numbers:
    loss_history[i] = loss logged for CSV epoch i
        
    Returns:
    --------   
    dict: optimised parameters (same keys as initial_opt_params), including
        derived 'omega' when 'log_omega' is optimised.
    list: Loss history (indexed by epoch: loss_history[i] = loss at epoch i)
    """
    # Initialize parameters
    loss_method = check_loss_method(loss_method)
    opt_params, fixed_params = sanitize_param_partition(
        initial_opt_params,
        fixed_params,
        require_nonempty_opt=True,
    )
    opt_param_keys = list(opt_params.keys())
    data = make_data_tuple_float64(data)
    uncertainties = make_data_tuple_float64(uncertainties)
    distance_pc = to_float64(distance_pc)
    learning_rate = to_float64(learning_rate)
    if not bool(jnp.isfinite(learning_rate)):
        raise ValueError(f'learning_rate must be finite. Got {learning_rate}.')
    if not bool(learning_rate > 0):
        raise ValueError(f'learning_rate must be > 0. Got {float(learning_rate)}.')
    param_bounds = standardise_param_bounds(param_bounds)
    normalisation_spec = build_normalisation_spec(opt_params, param_bounds)

    # Keep optimisation variables in normalised coordinates; convert back to
    # physical/log units only when evaluating the forward model and diagnostics.
    opt_params_norm = normalise_opt_params(opt_params, normalisation_spec)

    # Use one global learning rate on normalised parameters.
    solver = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2)

    opt_state = solver.init(opt_params_norm)

    # Precompute data-only quantities once before optimisation loop
    prepared_data = extract_streamline.prepare_data(data, uncertainties)

    def loss_from_normalised(norm_opt_params):
        physical_opt_params = denormalise_opt_params(norm_opt_params, normalisation_spec)
        return chi2_loss(
            physical_opt_params,
            fixed_params,
            distance_pc,
            loss_method=loss_method,
            prepared_data=prepared_data,
        )

    def loss_from_normalised_with_trace(norm_opt_params):
        physical_opt_params = denormalise_opt_params(norm_opt_params, normalisation_spec)
        return chi2_loss_raw(
            physical_opt_params,
            fixed_params,
            distance_pc,
            prepared_data,
            return_trace=True,
            loss_method=loss_method,
        )

    # Create gradient functions in normalised space.
    loss_and_grad_fn = value_and_grad(loss_from_normalised)
    loss_and_trace_fn = value_and_grad(loss_from_normalised_with_trace, has_aux=True)
    
    # Track loss history
    loss_history = []
    initial_loss = float(loss_from_normalised(opt_params_norm))
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
        # Create header: epoch, loss, then all optimisable params
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
        trace_csv_writer = csv.DictWriter(
            trace_csv_file,
            fieldnames=trace_fieldnames_for_loss_method(loss_method),
        )
        trace_csv_writer.writeheader()
        trace_csv_file.flush()
    
    print(f"Starting optimisation with {n_epochs} epochs...")
    print(f"Loss method: {loss_method}")
    print(f"optimising parameters: {opt_param_keys}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    if loss_threshold is not None:
        print(
            f"Threshold-based stopping enabled: loss <= {loss_threshold:.6g} "
            f"for {loss_threshold_epochs} consecutive epochs."
        )
    if gradient_tol is not None:
        print(
            f"Gradient norm stopping enabled (normalised space): ||grad|| < {gradient_tol:.6g} "
            f"for {gradient_tol_epochs} consecutive epochs."
        )
    print(f"Initial optimisable values:")
    for key in opt_param_keys:
        print(f"  {key}: {opt_params[key]:.3e}")
    
    # Log epoch 0: initial state (before any updates)
    initial_loss = float(initial_loss)
    if csv_writer is not None:
        row = {'epoch': 0, 'loss': initial_loss}
        for key in opt_param_keys:
            row[key] = float(opt_params[key])
        if 'log_omega' in opt_params:
            row['omega'] = float(jnp.exp(opt_params['log_omega']))
        csv_writer.writerow(row)
        csv_file.flush()
    
    # Log epoch 0 trace if trace file is requested
    if trace_csv_writer is not None:
        # Compute initial loss and trace
        (loss_value_trace, loss_trace_raw), norm_grads_trace = loss_and_trace_fn(opt_params_norm)
        loss_trace = trace_tree_to_python(loss_trace_raw)
        grad_norm = float(gradient_l2_norm(norm_grads_trace))
        
        # Build and write trace row for epoch 0
        trace_row = build_trace_row(0, float(loss_value_trace), loss_trace, grad_norm, loss_method)
        trace_csv_writer.writerow(trace_row)
        trace_csv_file.flush()
    
    try:
        for epoch in range(1, n_epochs + 1):
            if epoch % info_every == 0:
                print(f"\n Starting Epoch {epoch} -------------------------")
            # Compute loss and gradients at current (pre-update) normalised parameters.
            # At the START of iteration i, we're at state S(i-1).
            # The loss computed here is loss(S(i-1)), which is what we want to log for CSV epoch (i-1).
            trace_requested = trace_csv_writer is not None and epoch % trace_every == 0
            loss_trace = None
            if trace_requested:
                (loss_value, loss_trace_raw), norm_grads = loss_and_trace_fn(opt_params_norm)
                loss_trace = trace_tree_to_python(loss_trace_raw)
            else:
                loss_value, norm_grads = loss_and_grad_fn(opt_params_norm)
            loss_value = float(loss_value)

            # Compute gradient norm in normalised space for stopping criteria and logging.
            grad_norm = float(gradient_l2_norm(norm_grads))

            # LOG DEFERRED EPOCH: Log epoch (epoch - 1) using loss computed at current state (S(epoch-1))
            # (This is the loss AFTER applying update epoch-1, which is what we want for CSV epoch epoch-1)
            if epoch > 1:
                if csv_writer is not None:
                    row = {'epoch': (epoch - 1), 'loss': loss_value}
                    # opt_params is still S(epoch-1) before this iteration's update
                    for key in opt_param_keys:
                        row[key] = float(opt_params[key])
                    if 'log_omega' in opt_params:
                        row['omega'] = float(jnp.exp(opt_params['log_omega']))
                    csv_writer.writerow(row)
                    csv_file.flush()

            # Perform Optax Adam step in normalised space (apply update).
            updates, opt_state = solver.update(norm_grads, opt_state, params=opt_params_norm)
            opt_params_norm = optax.apply_updates(opt_params_norm, updates)

            # Enforce normalised bounds and map back to physical/log values.
            for key in opt_param_keys:
                opt_params_norm[key] = jnp.clip(opt_params_norm[key], 0.0, 1.0)

            # Gradient-aware epsilon protection for v_r0 near zero:
            # When v_r0 is very close to zero, use the sign of the gradient to determine
            # which direction to protect towards, allowing the optimiser to continue smoothly.
            if 'v_r0' in opt_param_keys:
                threshold_norm = to_float64(1e-12)  # normalised space threshold
                v_r0_norm_val = opt_params_norm['v_r0']
                if bool(jnp.all(jnp.abs(v_r0_norm_val) < threshold_norm)):
                    # v_r0 is very close to zero; check gradient direction
                    grad_v_r0 = norm_grads['v_r0']
                    # In gradient descent, we move opposite to gradient:
                    # If grad > 0, param should decrease (negative direction)
                    # If grad < 0, param should increase (positive direction)
                    protect_sign = -jnp.sign(grad_v_r0)
                    # Default to positive if gradient is exactly zero
                    protect_sign = jnp.where(protect_sign == 0, 1.0, protect_sign)
                    # Set v_r0 to small epsilon in the gradient-indicated direction
                    epsilon_norm = threshold_norm
                    opt_params_norm['v_r0'] = protect_sign * epsilon_norm

            # Now materialize physical parameters from the (possibly clamped)
            # normalised parameters.
            opt_params = denormalise_opt_params(opt_params_norm, normalisation_spec)
        
            # Track loss (store the loss before the update for the loss_history)
            loss_history.append(loss_value)

            if trace_csv_writer is not None and loss_trace is not None:
                # Use the loss value from before the update for trace logging
                trace_row = build_trace_row(epoch, loss_value, loss_trace, grad_norm, loss_method)
                trace_csv_writer.writerow(trace_row)
                trace_csv_file.flush()
        
            # Early stopping checks (use loss before update)
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
                    f"\nEarly stopping at epoch {epoch}: normalised gradient norm {grad_norm:.6e} < {gradient_tol:.6e} "
                    f"for {gradient_tol_epochs} consecutive epochs"
                )
                break
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}: no improvement for {early_stopping_patience} epochs")
                break
    
        # After the loop, log the final epoch (epoch N or the epoch where we stopped)
        # At this point, opt_params contains the parameters from the end of the final iteration.
        # We need to compute the loss at these parameters to complete the CSV epoch logging.
        if csv_writer is not None:
            loss_final = float(loss_from_normalised(opt_params_norm))
            # epoch is the last epoch number from the loop (either n_epochs or early stopping)
            row = {'epoch': epoch, 'loss': loss_final}
            for key in opt_param_keys:
                row[key] = float(opt_params[key])
            if 'log_omega' in opt_params:
                row['omega'] = float(jnp.exp(opt_params['log_omega']))
            csv_writer.writerow(row)
            csv_file.flush()
        
        # Optionally compute and log trace diagnostics at the best parameters found.
        # This is independent of the CSV epoch logging and serves as diagnostics for the best fit.
        if trace_csv_writer is not None and best_epoch % trace_every != 0:
            # best_epoch was not logged via regular trace_every sampling; compute and log now
            best_loss_for_trace, best_trace = chi2_loss(
                best_opt_params,
                fixed_params,
                distance_pc,
                prepared_data,
                return_trace=True,
                loss_method=loss_method,
            )
            best_loss_for_trace = float(best_loss_for_trace)
            
            # Compute best gradient norm for trace
            best_norm_grads = loss_and_grad_fn(normalise_opt_params(best_opt_params, normalisation_spec))[1]
            best_grad_norm = float(gradient_l2_norm(best_norm_grads))
            
            # Log trace row for best epoch
            best_trace_row = build_trace_row(best_epoch, best_loss_for_trace, best_trace, best_grad_norm, loss_method)
            trace_csv_writer.writerow(best_trace_row)
            trace_csv_file.flush()
    
        # restore canonical parameter order before returning
        ordered_best_opt_params = {k: best_opt_params[k] for k in opt_param_keys}

    finally:
        # Always close the CSV file if it was opened
        if csv_file is not None:
            csv_file.close()
            print(f"Optimisation log saved to: {log_file}")
        if trace_csv_file is not None:
            trace_csv_file.close()
            print(f"Matching trace log saved to: {trace_file}")

    print(f"Optimisation complete!")
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
            prepared_data,
            loss_method=loss_method,
            gradient_tol=gradient_tol,
            normalisation_spec=normalisation_spec,
        )
        print("\nParameter uncertainties (1-sigma):")
        for k, v in param_errors.items():
            print(f"  {k}: {v}")
    else:
        param_errors = None


    return with_derived_omega(ordered_best_opt_params), loss_history, param_errors