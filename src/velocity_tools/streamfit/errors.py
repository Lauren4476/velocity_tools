# For computing the uncertainties, we must use the old versions of the forward model, loss, etc.
# This is because the new versions were modified to be jit compatible by having constant length arrays.
# But this has a side effect of requiring computations such as argsort,
# which are not compatible with taking second derivatives for the Hessian-based uncertainty estimation.

# So we put the old (non-jit) versions of the relevant functions here, and use them in the uncertainty estimation.

def estimate_parameter_errors(
    best_opt_params,
    fixed_params,
    data,
    uncertainties,
    distance_pc,
    prepared_data,
    loss_method=0,
    gradient_tol=1e-1,
    normalisation_spec=None,
):
    """
    Estimate parameter uncertainties using Hessian of chi2 loss.

    Parameters
    ----------
    prepared_data : PreparedData
        Precomputed data-only quantities (created via prepare_data).
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
        chi2_total, _ = chi2_loss(
            params,
            fixed_params,
            distance_pc,
            prepared_data,
            loss_method=loss_method,
        )
        return chi2_total

    
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
                chi2_total, _ = chi2_loss(
                     physical_params,
                    fixed_params,
                    distance_pc,
                    prepared_data,
                    loss_method=loss_method,
                )
                return chi2_total

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


#-------------------- old gradient_descent.py ---------------------

import jax.numpy as jnp
import jax
from jax.experimental import checkify
import astropy.units as u
import math

# settings and constants

BIG = 1e30


LOSS_METHOD_CHOICES = [0, 1]


CANONICAL_UNITS = {
    "r0": u.au,
    "theta0": u.rad,
    "phi0": u.rad,
    "inc": u.rad,
    "pa": u.rad,
    "v_r0": u.km / u.s,
    "omega": 1 / u.s,
    "mass": u.Msun,
    "rmin": u.au,
    "deltar": u.au,
    "v_lsr": u.km / u.s,
}

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

def check_loss_method(loss_method):
    """Check that the selected loss method is valid and return it"""
    if loss_method not in LOSS_METHOD_CHOICES:
        raise ValueError(
            f"Unknown loss_method '{loss_method}'. "
            f"Choose from: 0: radecvel 1: rthetavel"
        )
    return loss_method




def is_numeric_value(value):
    """Return True for scalar/array-like numeric values"""
    try:
        arr = jnp.asarray(value)
    except Exception:
        return False
    if arr.dtype == jnp.bool_:
        return False
    return bool(jnp.issubdtype(arr.dtype, jnp.number))


@jax.jit
def to_float64(value):
    """Convert a numeric value or array-like input to float64"""
    return jnp.asarray(value, dtype=jnp.float64)








def clean_model_param_dict(params, dict_name):
    """Convert parameter dictionary to float64 and standardise it"""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"{dict_name} must be a dictionary, got {type(params).__name__}.")

    sanitized = {}

    for key, val in params.items():
        if isinstance(val, u.Quantity):
            if key not in CANONICAL_UNITS:
                raise ValueError(f"The parameter {key} doesn't have defined canonical units...")
            val = val.to(CANONICAL_UNITS[key]).value
        # if it's already a raw number, assume it's already correct
        sanitized[key] = jnp.asarray(val, dtype=jnp.float64)

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



@jax.jit
def normalise_opt_params(opt_params, normalisation_spec):
    """normalise optimised parameters to [0, 1]"""
    normalised = {}
    for key, value in opt_params.items():
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        normalised[key] = (to_float64(value) - offset) / scale
    return normalised


@jax.jit
def denormalise_opt_params(norm_opt_params, normalisation_spec):
    """Convert normalised optimised parameters back to physical/log parameter values"""
    denormalised = {}
    for key, value in norm_opt_params.items():
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        if key == 'phi0':
            # special handling for phi0 because circular
            denormalised[key] = jnp.mod(to_float64(value) * scale + offset, 2*jnp.pi)
        else:
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


def forward_model(opt_params, fixed_params, distance_pc):
    """
    Run the forward model using xyz_stream
    
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
    model_params, opt_params, fixed_params = prepare_model_params(opt_params, fixed_params)
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
    (x, y, z), (vx, vy, vz) = xyz_stream(
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



@jax.jit
def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data):
    """
    Extract model values corresponding to data positions using the distance metric from
    get_distance_metric

    Method:
    1. Compute the distance metric for model and data points
    2. Apply finite masks
    3. Normalise both metrics to [0, 1] based on their finite ranges
    4. Map data normalised positions to model normalised positions
    5. Interpolate model RA, Dec, and velocity at the mapped positions

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

    # get distance metrics
    dmetric_model, _ = get_distance_metric(ra_model, dec_model)
    dmetric_data, _ = get_distance_metric(ra_data, dec_data)

    # only finite values are valid
    model_valid = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )

    data_valid = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )

    # we also filter model to keep only model points with dmetric >= minimum of data dmetric
    # this is becuase the model shouldn't go further in than the innermost data point
    # as this is where we no longer observe the streamer
    d_data_valid = jnp.where(data_valid, dmetric_data, jnp.inf)
    data_min = jnp.min(d_data_valid)

    # enforce both constraints on model
    model_keep = model_valid & (dmetric_model >= data_min)

    # weights: 0 = ignore, 1 = use. This is for jax/jit compatibility
    w_model = model_keep.astype(jnp.float64)
    w_data = data_valid.astype(jnp.float64)

    d_model = jnp.where(model_keep, dmetric_model, 0.0)
    d_data  = jnp.where(data_valid, dmetric_data, 0.0)

    ra = ra_model
    dec = dec_model
    v = v_model

    # ---- sort ONLY MODEL using metric + weight penalty ----
    model_sort_key = d_model + (1.0 - w_model) * BIG
    model_idx = jnp.argsort(model_sort_key)

    d_model_s = d_model[model_idx]
    ra_s = ra[model_idx]
    dec_s = dec[model_idx]
    v_s = v[model_idx]
    w_model_s = w_model[model_idx]

    # stats for trace and interpolation domain
    data_min_eff = jnp.min(jnp.where(data_valid, d_data, jnp.inf))
    data_max_eff = jnp.max(jnp.where(data_valid, d_data, -jnp.inf))

    model_min = jnp.min(jnp.where(model_keep, d_model, jnp.inf))
    model_max = jnp.max(jnp.where(model_keep, d_model, -jnp.inf))

    model_span = model_max - model_min
    data_span = data_max_eff - data_min_eff
    model_span_safe = jnp.where(model_span == 0.0, 1.0, model_span)
    data_span_safe = jnp.where(data_span == 0.0, 1.0, data_span)

    # normalise data metric
    d_data_norm = (d_data - data_min_eff) / data_span
    d_goal = model_min + d_data_norm * model_span

    # interpolate model at data points, using weights to ignore invalid model points 
    # by giving them huge distance values so they don't affect the interpolation
    xp = jnp.where(w_model_s > 0, d_model_s, BIG)

    ra_interp = jnp.interp(d_goal, xp, ra_s)
    dec_interp = jnp.interp(d_goal, xp, dec_s)
    v_interp = jnp.interp(d_goal, xp, v_s)

    # things for trace
    valid = data_valid

    matching_trace = {
    "model_points_total": model_idx.size,
    "model_nan_count": jnp.sum(jnp.isnan(d_model)),
    "model_valid_points": model_valid.sum(),
    "data_points_total": ra_data.size,
    "data_nan_count": jnp.sum(jnp.isnan(d_data)),
    "data_valid_points": data_valid.sum(),
    "model_metric_min": model_min,
    "model_metric_max": model_max,
    "data_metric_min": data_min_eff,
    "data_metric_max": data_max_eff,
    "model_metric_span": model_span_safe,
    "data_metric_span": data_span_safe}

    return ra_interp, dec_interp, v_interp, valid, dmetric_model, matching_trace



checked_matching = checkify.checkify(match_model_to_data_curve)

def checked_match_model_to_data_curve(*args, **kwargs):
    """Wrapper around match_model_to_data_curve with checkify checks for errors (to remain jax compatible)"""
    errors, result = checked_matching(*args, **kwargs)
    errors.throw()
    return result


#@jax.jit(static_argnames=("loss_method"))
def chi2_loss(
    opt_params,
    fixed_params,
    distance_pc,
    prepared_data,
    loss_method=0,
):
    """
    Compute chi-squared loss between model and data using one of two modes:
    - 0: RA, Dec, and LOS velocity residuals
    - 1: projected radial distance, polar angle, and LOS velocity residuals
    
    Parameters:
    -----------
    opt_params : dict
        optimisable streamline model parameters (any subset of
        STREAMLINE_MODEL_PARAM_KEYS). already unitless
    fixed_params : dict
        Fixed streamline model parameters (complementary subset). already unitless
    distance_pc : float
        Distance to source in parsecs
    prepared_data : PreparedData
        Precomputed data-only quantities (distance metrics, bounds, polar coords).
        Created via prepare_data(data, uncertainties).

        
            Created via prepare_data(data, uncertainties).
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
    ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model, matching_trace = (
        checked_match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data)
    )

    valid = jnp.asarray(valid, dtype=bool)
    # Only compute chi2 on valid/retained data points to avoid penalizing points outside overlap domain
    chi2_v = jnp.sum((((v_data[valid] - v_model_interp[valid]) / v_sigma[valid]) ** 2))

    if loss_method == 0:
        chi2_ra = jnp.sum((((ra_data[valid] - ra_model_interp[valid]) / ra_sigma[valid]) ** 2))
        chi2_dec = jnp.sum((((dec_data[valid] - dec_model_interp[valid]) / dec_sigma[valid]) ** 2))
        chi2_total = chi2_ra + chi2_dec + chi2_v # + chi2_penalty
    else:
        # r/theta are defined on the projected plane of the sky from (RA, Dec).
        # Use precomputed data coordinates
        r_proj_data = prepared_data.r_proj_data
        theta_proj_data = prepared_data.theta_proj_data
        r_proj_model, theta_proj_model = cartesian_to_polar(
            ra_model_interp,
            dec_model_interp,
        )

        dtheta = wrap_to_pi(theta_proj_data - theta_proj_model)

        sigma_r = jnp.sqrt(ra_sigma**2 + dec_sigma**2)
        r_eps = to_float64(1e-8)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt(((dec_data * ra_sigma)**2 + (ra_data * dec_sigma)**2)) / (r_safe**2)
        sigma_theta = jnp.maximum(sigma_theta, r_eps)

        # Only compute chi2 on valid/retained data points
        chi2_r = jnp.sum((((r_proj_data[valid] - r_proj_model[valid]) / sigma_r[valid]) ** 2))
        chi2_theta = jnp.sum(((dtheta[valid] / sigma_theta[valid]) ** 2))
        chi2_total = chi2_r + chi2_theta + chi2_v # + chi2_penalty

    if loss_method == 0:
        chi2_components = {
            'chi2_ra': chi2_ra.astype(float),
            'chi2_dec': chi2_dec.astype(float),
            'chi2_v': chi2_v.astype(float),
            # 'chi2_penalty': float(chi2_penalty),
            # 'low_shortfall_penalty': float(low_shortfall),
            # 'low_excess_penalty': float(low_excess),
            # 'high_shortfall_penalty': float(high_shortfall),
            # 'high_excess_penalty': float(high_excess),
            'chi2_total': chi2_total.astype(float),
        }
    else:
        chi2_components = {
            'chi2_r': chi2_r.astype(float),
            'chi2_theta': chi2_theta.astype(float),
            'chi2_v': chi2_v.astype(float),
            # 'chi2_penalty': float(chi2_penalty),
            # 'low_shortfall_penalty': float(low_shortfall),
            # 'low_excess_penalty': float(low_excess),
            # 'high_shortfall_penalty': float(high_shortfall),
            # 'high_excess_penalty': float(high_excess),
            'chi2_total': chi2_total.astype(float),
        }

    loss_trace = {
        'chi2_components': chi2_components,
        'matching': matching_trace,
        'loss_method': loss_method,
    }
    return chi2_total, loss_trace
















# ------------------ old stream_lines_grad.py --------------------


import astropy.units as u
from ..helper_functions import *
import jax
import jax.numpy as jnp
# from jax import lax
# from jax import debug
jax.config.update("jax_enable_x64", True)
from typing import NamedTuple


## constants 
eps = 1e-8 # small value to avoid division by zero
FLOAT_DTYPE = jnp.float64

## important streamline quantities (for easy reuse)
class StreamState(NamedTuple):
    rc: jnp.ndarray
    mu: jnp.ndarray
    nu: jnp.ndarray
    epsilon: jnp.ndarray
    ecc: jnp.ndarray
    vk0: jnp.ndarray


def to_float64(value):
    '''input must be a number or array-like'''
    return jnp.asarray(value, dtype=FLOAT_DTYPE)

@jax.jit
def v_k(radius, mass=0.5):
    '''
    Velocity term that is repeated in all velocity components.
    It corresponds to v_k in Mendoza+(2009)
    :param radius: au
    :param mass: Msun
    :return: v_k, km/s
    '''
    arg = G * mass / radius
    return jnp.sqrt(arg)

@jax.jit
def r_cent(mass, omega=1e-14, r0=1e4):
    '''
    Centrifugal radius or disk radius in the Ulrich (1976) model.
    r_u in Mendoza's nomenclature.

    :param mass: Central mass for the protostar, Msun
    :param omega: Angular speed at the r0 radius, 1/s
    :param r0: Initial radius of the streamline, au
    :return: r_cent, au
    '''
    r_cent = (jnp.power(r0, 4) * jnp.power(omega, 2) / (G * mass)) # in au^3 km^-2
    r_cent_au = r_cent * (jnp.power(au_in_km, 2)) # in au
    return r_cent_au


@jax.jit
def build_stream_quantities(mass, r0, theta0, omega, v_r0):
    '''
    precompute streamer quantities reused throughout file, and 
    store in class StreamState (near top)
    '''

    mass = jnp.asarray(mass, dtype=FLOAT_DTYPE)
    r0 = jnp.asarray(r0, dtype=FLOAT_DTYPE)
    theta0 = jnp.asarray(theta0, dtype=FLOAT_DTYPE)
    omega = jnp.asarray(omega, dtype=FLOAT_DTYPE)
    v_r0 = jnp.asarray(v_r0, dtype=FLOAT_DTYPE)

    # Protect near-zero v_r0 from creating singularities in nu calculation
    # Allow negative v_r0, but replace exact-zero or tiny values with signed epsilon
    threshold = to_float64(1e-6)
    v_r0 = jnp.where(
        jnp.isclose(v_r0, to_float64(0.0)),
        - jnp.sign(v_r0) * threshold, #let it continue in the direction it was going
        v_r0  # normal values -> unchanged
        )

    rc = r_cent(mass=mass, omega=omega, r0=r0)
    mu = rc / r0
    nu = v_r0 * jnp.sqrt(rc / (G * mass))
    sin_theta0 = jnp.sin(theta0)
    sin_theta0_sq = jnp.power(sin_theta0, 2)
    epsilon = jnp.power(nu, 2) + jnp.power(mu, 2) * sin_theta0_sq - 2 * mu
    ecc = jnp.sqrt(1.0 + epsilon * sin_theta0_sq)
    vk0 = v_k(rc, mass=mass)

    return StreamState(rc=rc, mu=mu, nu=nu, epsilon=epsilon, ecc=ecc, vk0=vk0)

@jax.jit
def safe_arccos(x, eps=1e-8):
    '''
    Safe arccos function with clipping to valid range [-1, 1],
    with a small margin to avoid numerical issues in gradients near the boundaries

    :param x: input value
    :param eps: small offset
    :return: arccos of clipped input
    '''
    x = jnp.asarray(x)
    x = x.astype(FLOAT_DTYPE)

    # Keep away from +/-1 by at least a few ULPs of the active dtype
    eps_user = jnp.asarray(eps, dtype=x.dtype)
    eps_floor = jnp.asarray(32.0 * jnp.finfo(x.dtype).eps, dtype=x.dtype)
    eps_eff = jnp.maximum(eps_user, eps_floor)

    x_safe = jnp.clip(x, -1.0 + eps_eff, 1.0 - eps_eff)
    return jnp.arccos(x_safe)

@jax.jit
def get_theta(theta0, orb_ang, orb_ang0):
    '''
    Gets theta from theta0, orb_ang, and orb_ang0, in radians.
    Eqn (8) in Mendoza+2009
    
    :param theta0: radians
    :param orb_ang: radians
    :param orb_ang0: radians
    :return theta: radians
    '''
    cos_theta = jnp.cos(theta0) * jnp.cos(orb_ang - orb_ang0)
    theta = safe_arccos(cos_theta)
    return theta


@jax.jit
def get_orb_ang(r_to_rc, theta0, ecc):
    '''
    Gets orb_ang (varphi in Mendoza+2009), in radians.
    To get initial orb_ang, set r_to_rc = r0/rc = 1/mu
    
    :param r_to_rc: radius divided by centrifugal radius
    :param theta0: radius
    :param ecc: eccentricity
    :return orb_ang: radians
    '''
    cos_orb_ang = (1/ecc) * (1 - (jnp.power(jnp.sin(theta0), 2) / r_to_rc))
    orb_ang = safe_arccos(cos_orb_ang)
    return orb_ang

@jax.jit
def get_dphi(theta, theta0=jnp.radians(30)):
    '''
    Gets the difference in Phi between initial and current, in radians.

    :param theta: radians
    :param theta0: radians
    :return: difference in Phi angle, radians
    '''
    arg = jnp.tan(theta0) / jnp.tan(theta)
    return safe_arccos(arg)



@jax.jit
def stream_line(r, stream_state, theta0=jnp.radians(30), phi0=jnp.radians(15)):
    '''
    It calculates the stream line following Mendoza et al. (2009),
    only for r < r0. Point r = r0 is handled outside the function.
    It takes the radial velocity and rotation at the streamline
    initial radius and it describes the entire trajectory.

    :param r: au
    :param mass: Msun
    :param r0: au
    :param theta0: radians
    :param phi0: radians
    :param omega: 1/s
    :param v_r0: Initial radial velocity, km/s
    :return: theta, radians
    '''
    r = jnp.asarray(r, dtype=FLOAT_DTYPE)
    rc = stream_state.rc
    mu = stream_state.mu
    ecc = stream_state.ecc

    # orb_ang is varphi in Mendoza+2009
    # at initial position r_to_rc = r0/rc = 1/mu
    orb_ang0 = get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)

    # vectorised computation over array of r values
    r_to_rc = r / rc
    orb_ang = get_orb_ang(r_to_rc=r_to_rc, theta0=theta0, ecc=ecc)
    theta = get_theta(theta0, orb_ang, orb_ang0)
    phi = phi0 + get_dphi(theta, theta0=theta0)

    # remove values where r_to_rc < 0.5 (inside centrifugal radius)
    mask = r_to_rc >= 0.5
    orb_ang = jnp.where(mask, orb_ang, jnp.nan)
    theta = jnp.where(mask, theta, jnp.nan)
    phi = jnp.where(mask, phi, jnp.nan)

    return orb_ang, theta, phi #in radians


@jax.jit
def stream_line_vel(
    r,
    theta,
    orb_ang,
    stream_state,
    theta0=jnp.radians(30),
):
    '''
    It calculates the velocity along the stream line following Mendoza+(2009)
    It takes the radial velocity and rotation at the streamline
    initial radius and it describes the entire trajectory.

    :param theta: radians
    :param r: au
    :param mass: Msun
    :param r0: au
    :param theta0: radians
    :param phi0: radians
    :param omega: 1/s
    :param v_r0: Initial radial velocity, km/s
    :return: v_r, v_theta, v_phi in units of km/s
    '''
    rc = stream_state.rc
    ecc = stream_state.ecc
    vk0 = stream_state.vk0
    r_to_rc = (r / rc)
    #
    v_r_all = -ecc * jnp.sin(theta0) * jnp.sin(orb_ang) / r_to_rc /(1 - ecc*jnp.cos(orb_ang))
    v_theta_all = jnp.sin(theta0) / jnp.sin(theta) / r_to_rc \
                  * jnp.sqrt(jnp.power(jnp.cos(theta0),2) - jnp.power(jnp.cos(theta),2))
    v_phi_all = jnp.power(jnp.sin(theta0), 2) / (jnp.sin(theta) * r_to_rc)

    return v_r_all * vk0, v_theta_all * vk0, v_phi_all * vk0

@jax.jit
def build_rotation_matrix(inc, pa):
    '''constructs combined inclination/position-angle rotation matrix'''

    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)

    ci = jnp.cos(inc)
    si = jnp.sin(inc)
    cp = jnp.cos(pa)
    sp = jnp.sin(pa)

    return jnp.array([
        [cp, sp * si, -sp * ci],
        [0.0, ci, si],
        [sp, -cp * si, cp * ci],
    ], dtype=FLOAT_DTYPE)

@jax.jit
def rotate_xyz(x, y, z, rotation_matrix):
    '''
    Rotate on inclination and PA
    x-axis and y-axis are on the plane on the sky,
    z-axis is the

    Rotation around x is inclination angle
    Rotation around y is PA angle

    Using example matrices as described in:
    https://en.wikipedia.org/wiki/3D_projection

    :param x: cartesian x-coordinate, in the direction of decreasing RA
    :param y: cartesian y-coordinate, in the direction away of the observer
    :param z: cartesian z-coordinate, in the direction of increasing Dec.
    :param rotation_matrix: 3x3 rotation matrix combining inclination and PA rotations.
    :return: new x, y, and z-coordinates as observed on the sky, with the
    same units as the input ones.

    '''
    x = jnp.asarray(x, dtype=FLOAT_DTYPE)
    y = jnp.asarray(y, dtype=FLOAT_DTYPE)
    z = jnp.asarray(z, dtype=FLOAT_DTYPE)

    xyz = jnp.stack((x, y, z), axis=0)

    xyz_rot = rotation_matrix @ xyz

    return xyz_rot[0], xyz_rot[1], xyz_rot[2]


def xyz_stream(mass=0.5, r0=1e4, theta0=30,
               phi0=15, omega=1e-14, v_r0=0,
               inc=0, pa=0, rmin=None, deltar=1):
    '''
    it gets xyz coordinates and velocities for a stream line.
    They are also rotated in PA and inclination along the line of sight.
    This is a wrapper around stream_line() and rotate_xyz()

    Spherical into cartesian transformation is done for position and velocity
    using:
    https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    :param mass: Central mass (Msun)
    :param r0: Initial radius of streamline (au)
    :param theta0: Initial polar angle of streamline (degrees)
    :param phi0: Initial azimuthal angle of streamline (degrees)
    :param omega: Angular rotation. (defined positive), (1/s)
    :param v_r0: Initial radial velocity of the streamline, (km/s)
    :param inc: inclination with respect of line-of-sight, inc=0 is an edge-on-disk (degrees)
    :param pa: Position angle of the rotation axis, measured due East from North. This is usually estimated from the outflow PA, or the disk PA-90deg., (degrees)
    :param rmin: smallest radius for calculation, (au)
    :param deltar: spacing between two consecutive radii in the sampling of the streamer, in (au)
    :return: x, y, z in (au), v_x, v_y, v_z in (km/s)
    '''

    mass = jnp.asarray(mass, dtype=FLOAT_DTYPE)
    r0 = jnp.asarray(r0, dtype=FLOAT_DTYPE)
    theta0 = jnp.asarray(theta0, dtype=FLOAT_DTYPE)
    phi0 = jnp.asarray(phi0, dtype=FLOAT_DTYPE)
    omega = jnp.asarray(omega, dtype=FLOAT_DTYPE)
    v_r0 = jnp.asarray(v_r0, dtype=FLOAT_DTYPE)
    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)
    deltar = jnp.asarray(deltar, dtype=FLOAT_DTYPE)
    if rmin is not None:
        rmin = jnp.asarray(rmin, dtype=FLOAT_DTYPE)

    stream_state = build_stream_quantities(mass=mass, r0=r0, theta0=theta0, omega=omega, v_r0=v_r0)
    rc = stream_state.rc
    mu = stream_state.mu
    ecc = stream_state.ecc

    rotation_matrix = build_rotation_matrix(inc, pa)


    # checkify.check(rc <= r0, "Centrifugal radius is larger that start of streamline.")

    r_low = jnp.maximum(rmin, rc*0.5) if rmin is not None else rc*0.5
    # r is values internal to the initial radius r0 for computation
    r = jnp.arange(r0 - deltar, r_low, step=-1*deltar, dtype=FLOAT_DTYPE)

    # calculate positions and velocities inside r0
    orb_ang, theta, phi = stream_line(r, stream_state=stream_state, theta0=theta0, phi0=phi0)
    v_r, v_theta, v_phi = stream_line_vel(r, theta, orb_ang, stream_state=stream_state, theta0=theta0)

    # prepend initial positions and velocities at r0
    r_full = jnp.concatenate((jnp.asarray([r0], dtype=FLOAT_DTYPE), r))
    theta_full = jnp.concatenate((jnp.asarray([theta0], dtype=FLOAT_DTYPE), theta))
    phi_full = jnp.concatenate((jnp.asarray([phi0], dtype=FLOAT_DTYPE), phi))
    orb_ang0 = get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)
    orb_ang_full = jnp.concatenate((jnp.asarray([orb_ang0], dtype=FLOAT_DTYPE), orb_ang))
    v_r_full = jnp.concatenate((jnp.asarray([v_r0], dtype=FLOAT_DTYPE), v_r))
    v_theta_full = jnp.concatenate((jnp.asarray([0.0], dtype=FLOAT_DTYPE), v_theta))
    # we need to calculate v_phi0
    v_phi0 = stream_state.vk0 * jnp.sin(theta0) * stream_state.mu
    v_phi_full = jnp.concatenate((jnp.asarray([v_phi0], dtype=FLOAT_DTYPE), v_phi))

    # convert from spherical into cartesian coordinates
    v_x = v_r_full * jnp.sin(theta_full) * jnp.cos(phi_full) \
          + v_theta_full * jnp.cos(theta_full) * jnp.cos(phi_full) \
          - v_phi_full * jnp.sin(phi_full)
    v_y = v_r_full * jnp.sin(theta_full) * jnp.sin(phi_full) \
          + v_theta_full * jnp.cos(theta_full) * jnp.sin(phi_full) \
          + v_phi_full * jnp.cos(phi_full)
    v_z = v_r_full * jnp.cos(theta_full) \
          - v_theta_full * jnp.sin(theta_full)
    x = r_full * jnp.sin(theta_full) * jnp.cos(phi_full)
    y = r_full * jnp.sin(theta_full) * jnp.sin(phi_full)
    z = r_full * jnp.cos(theta_full)
    # get mask from smallest radius for calculation
    if rmin is None:
        gd_rmin = jnp.ones_like(r, dtype=bool)
    else:
        gd_rmin = (r_full > rmin)
    gd_rmin = gd_rmin.astype(x.dtype)
    # apply mask before rotation
    x = jnp.where(gd_rmin, x, jnp.nan)  
    y = jnp.where(gd_rmin, y, jnp.nan)
    z = jnp.where(gd_rmin, z, jnp.nan)
    v_x = jnp.where(gd_rmin, v_x, jnp.nan)
    v_y = jnp.where(gd_rmin, v_y, jnp.nan)
    v_z = jnp.where(gd_rmin, v_z, jnp.nan)
    # rotate
    return rotate_xyz(x, y, z, rotation_matrix=rotation_matrix), \
           rotate_xyz(v_x, v_y, v_z, rotation_matrix=rotation_matrix)

# ----------------- old extract_streamline.py ------------------


import numpy as np
from astropy import units as u
from collections import namedtuple
import jax.numpy as jnp
import jax


PreparedData = namedtuple('PreparedData', [
    'ra_data', 'dec_data', 'v_data',
    'ra_sigma_safe', 'dec_sigma_safe', 'v_sigma_safe',
    'dmetric_data', 'data_finite_mask',
    'data_min', 'data_max',
    'r_proj_data', 'theta_proj_data',
])



@jax.jit
def circular_median(theta_vals, weights):
    '''Branch-cut-safe median angle. (unrwap, linear median, rewrap)
    theta values with weight = 0 are ignored in the median calculation'''
    weights = weights / (jnp.sum(weights) + 1e-12) # normalize weights to sum to 1, add small value to avoid division by zero
    theta_anchor = jnp.arctan2(
        jnp.sum(weights * jnp.sin(theta_vals)),
        jnp.sum(weights * jnp.cos(theta_vals))
    )
    theta_delta = wrap_to_pi(theta_vals - theta_anchor)
    theta_unwrapped = theta_anchor + theta_delta
    sort_idx = jnp.argsort(theta_unwrapped)
    sorted_vals = theta_unwrapped[sort_idx]
    sorted_weights = weights[sort_idx]
    cumulative_weights = jnp.cumsum(sorted_weights)
    cutoff = 0.5 * jnp.sum(sorted_weights)
    median_idx = jnp.argmax(cumulative_weights >= cutoff)
    theta_ref = sorted_vals[median_idx]
    return wrap_to_pi(theta_ref)



@jax.jit
def wrap_to_pi(angle):
    '''Wrap angles to [-pi, pi)'''
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi



        
@jax.jit
def get_distance_metric(ra_coords, dec_coords):
    '''
    Compute radial + angular distance metric for point cloud binning
    Uses a circular angular deviation to avoid branch-cut artifacts.
    '''
    pc_r, pc_theta = cartesian_to_polar(ra_coords, dec_coords)

    finite_mask = jnp.isfinite(pc_r) & jnp.isfinite(pc_theta)
    # finite_r = pc_r[finite_mask]
    # finite_theta = pc_theta[finite_mask]

    # deal with if there are no valid points
    def empty_case(_):

        distance_metric = jnp.full_like(pc_r, jnp.nan)
        trace = {
            "n_points": pc_r.size,
            "n_finite_points": 0,
            "n_reference_points": 0,
            "r_percentile_thresh": jnp.nan,
            "r_thresh": jnp.nan,
            "theta_ref": 0.0,
            "theta_weight": 1.0,
            "close_point_count": 0,
        }
        return distance_metric, trace

    def notempty_case(_):

        theta_weight = 1.0 # maybe make this a tunable parameter
        finite_count = jnp.sum(finite_mask)
        # reference theta is obtained from points within a radius threshold
        n_ref = jnp.clip(finite_count, 1, 10)
        percentile = 100.0 / n_ref
        # jnp.percentile can deal with nans
        r_thresh = jnp.percentile(
            jnp.where(finite_mask, pc_r, jnp.nan),
            percentile)
        
        close_mask = (finite_mask & (pc_r <= r_thresh)).astype(jnp.float64)
        theta_ref = circular_median(pc_theta, weights=close_mask)

        # cyclic angular deviation
        theta_dev = jnp.pi - jnp.abs(
            jnp.pi - jnp.abs(wrap_to_pi(pc_theta - theta_ref))
        )

        distance_metric = pc_r * jnp.sqrt(1.0 + (theta_weight * theta_dev) ** 2)
        distance_metric = jnp.where(finite_mask, distance_metric, jnp.inf)

        trace = {
            "n_points": pc_r.size,
            "n_finite_points": finite_count,
            "n_reference_points": n_ref,
            "r_percentile_thresh": percentile,
            "r_thresh": r_thresh,
            "theta_ref": theta_ref,
            "theta_weight": theta_weight,
            "close_point_count": close_mask.size,
        }

        return distance_metric, trace
    
    return jax.lax.cond(finite_mask.any(), notempty_case, empty_case, operand=None)

@jax.jit
def cartesian_to_polar(x, y):
    '''
    Convert cartesian coordinates (x,y) to polar coordinates
    e.g. inputs could be RA and Dec offsets
    Note theta is returned in radians
    '''
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x) # angle wrt x-axis, in radians

    return (r, theta)





