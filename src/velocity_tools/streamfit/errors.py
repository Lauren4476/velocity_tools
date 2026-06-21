# For computing the uncertainties, we must use the old versions of the forward model, loss, etc.
# This is because the new versions were modified to be jit compatible by having constant length arrays.
# But this has a side effect of requiring computations such as argsort,
# which are not compatible with taking second derivatives for the Hessian-based uncertainty estimation.

# So we put the old (non-jit) versions of the relevant functions here, and use them in the uncertainty estimation.

import jax.numpy as jnp
import jax
from jax.experimental import checkify
import astropy.units as u
import math
jax.config.update("jax_enable_x64", True)
from typing import NamedTuple
from collections import namedtuple
from . import stream_lines_grad, extract_streamline, gradient_descent


## constants 
eps = 1e-8 # small value to avoid division by zero
G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11) # in au (km/s)^2 * Msol^-1
au_in_km = 1.4959787e8 #km
FLOAT_DTYPE = jnp.float64

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
    "mass": u.Msun,
    "rmin": u.au,
    "deltar": u.au,
    "v_lsr": u.km / u.s,
    "rc": u.au, 
    "omega": 1/u.s,
    # mu = rc/r0 is dimensionless, so no units
}

STREAMLINE_MODEL_PARAM_KEYS = (
    'r0',
    'theta0',
    'phi0',
    'rc',
    'omega',
    'mu',
    'v_r0',
    'mass',
    'inc',
    'pa',
    'rmin',
    'deltar',
    'v_lsr',
)


def estimate_covariance_at_best_fit(
    best_opt_params,
    initial_opt_params,
    fixed_params,
    data,
    uncertainties,
    distance,
    loss_method=0,
    gradient_tol=1e-1,
    rotation_key=None,
):
    """ wrapper around estimate_parameter_errors for convenient using after fit_streamline has finished"""
    opt_keys = list(initial_opt_params.keys())
    best_for_cov = {k: float(best_opt_params[k]) for k in opt_keys}
    # prepare data-only quantities once
    prepared_data = extract_streamline.prepare_data(data, uncertainties, n_elements=len(data[0]))
    param_errors, cov, cov_transformed_dict = estimate_parameter_errors(
        best_for_cov,
        fixed_params,
        distance,
        prepared_data,
        loss_method=loss_method,
        gradient_tol=gradient_tol,
        normalisation_spec=None,
        rotation_key=rotation_key,
    )
    return opt_keys, param_errors, cov, cov_transformed_dict


def estimate_parameter_errors(
    best_opt_params,
    fixed_params,
    distance_pc,
    prepared_data,
    loss_method=0,
    gradient_tol=1e-1,
    normalisation_spec=None,
    best_norm_opt_params=None,
    rotation_key=None,
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
    best_norm_opt_params : dict or None
        Normalised parameters at best-fit state, used for gradient_tol_check.
        If not provided, will be computed from best_opt_params and normalisation_spec, but providing it can save a redundant computation
    rotation_key : str or None
        If provided, must be 'rc' or 'omega'. Used to transform covariance matrix from optimised 'mu' to rotation_key

    Returns
    -------
    dict
        1-sigma uncertainties for each optimisable parameter
    array
        covariance matrix
    dict or None
        If rotation_key was given and 'mu' was optimised: {'keys': new_keys, 'cov': new_cov, 'errors': error_dict} 
        where new_keys is same as original keys but with 'mu' replaced by rotation_key, 
        new_cov is the covariance matrix transformed into the original parameter space, 
        and error_dict is the dict of 1-sigma errors for each parameter in new_keys.
    """
    if gradient_tol is not None:
        gradient_tol = float(gradient_tol)
        if not math.isfinite(gradient_tol):
            raise ValueError('gradient_tol must be finite when provided.')
        if gradient_tol <= 0:
            raise ValueError('gradient_tol must be positive when provided.')

    # convert dict -> vector
    params_vec, keys = params_dict_to_vector(best_opt_params)
    loss_method = gradient_descent.check_loss_method(loss_method)

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

            if best_norm_opt_params is not None:
                norm_opt_params = best_norm_opt_params
            else:
                norm_opt_params = gradient_descent.normalise_opt_params(best_opt_params, normalisation_spec)
            norm_params_vec, norm_keys = params_dict_to_vector(norm_opt_params)

            def norm_loss_vec(theta_norm_vec):
                norm_params = vector_to_params_dict(theta_norm_vec, norm_keys)
                physical_params = gradient_descent.denormalise_opt_params(norm_params, normalisation_spec)
                chi2_total, _ = chi2_loss(
                    physical_params,
                    fixed_params,
                    distance_pc,
                    prepared_data,
                    loss_method=loss_method,
                )
                return chi2_total

            norm_grad_vec = jax.grad(norm_loss_vec)(norm_params_vec)
            norm_grad_norm = float(gradient_descent.gradient_l2_norm(norm_grad_vec))
            print(f"Gradient magnitude at best-fit parameters: {norm_grad_norm:.3e}")

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

    cov_transformed_dict = None
    if rotation_key is not None and 'mu' in keys:
        cov_transformed_dict = transform_cov_matrix(cov, keys, best_opt_params, fixed_params, rotation_key)

    return error_dict, cov, cov_transformed_dict

def transform_cov_matrix(cov, keys, best_opt_params, fixed_params, rotation_key):
    """Transform a covariance matrix computed in optimisation space (rotation param is mu)
    into the equivalent covariance matrix for the original input rotation parameter (rc or omega).

    Maths:
    if A = original optimisation-space parameter vector (with mu)
    and B = transformed parameter vector with with rc or omega instead of mu,
    then covariance in B space is

    cov_B = J @ cov_A @ J^T

    where J is the Jacobian of the transformation from A to B, evaluated at the best-fit parameters
    J = d(B) / d(A)  (identity except for row corresponding to mu)

    Parameters:
    cov: covariance matrix from estimate_parameter_errors, in same parameter order as 'keys'
    keys: list of str. optimised parameter names, must include 'mu'
    best_opt_params: dict of best-fit optimised parameters (the point at which we evaluate J)
    fixed_params: dict of fixed parameters (the point at which we evaluate J)
    rotation_key: str, 'rc' or 'omega', the original rotation parameter which we want to transform into

    Returns:
    new_cov: covariance matrix transformed into original parameter space, with same order as keys but with 'mu' replaced by rotation_key
    new_keys: list of str, same as keys but with 'mu' replaced by rotation_key
    errors: dict of 1-sigma errors for each parameter in new_keys
    """

    if rotation_key not in ['rc', 'omega']:
        raise ValueError(f"rotation_key must be 'rc' or 'omega', got {rotation_key}")
    if 'mu' not in keys:
        raise ValueError(f"keys must include 'mu' for covariance transformation, got {keys}")
    # check that we have mass and r0 available to convert mu to rc or omega
    for required_key in ('mass', 'r0'):
        if required_key not in best_opt_params and required_key not in fixed_params:
            raise ValueError(f"'{required_key}' must be present in either best_opt_params or fixed_params")
        
    params_vec = jnp.array([best_opt_params[k] for k in keys], dtype=jnp.float64)

    def transform(vec_A):
        opt_params = vector_to_params_dict(vec_A, keys)
        combined_params = {**fixed_params, **opt_params}
        mu = combined_params['mu']
        mass = combined_params['mass']
        r0 = combined_params['r0']
        if rotation_key == 'rc':
            rotation_val = mu * r0
        else: # 'omega'
            rotation_val = stream_lines_grad.omega_from_mu(mu=mu, mass=mass, r0=r0)
        output = []
        for k in keys:
            if k == 'mu':
                output.append(rotation_val)
            else:
                output.append(opt_params[k])
        return jnp.stack(output)
    
    J = jax.jacobian(transform)(params_vec)
    new_cov = J @ cov @ J.T

    new_keys = [rotation_key if k == 'mu' else k for k in keys]
    new_sigmas = jnp.sqrt(jnp.diag(new_cov))
    error_dict = {k: float(new_sigmas[i]) for i, k in enumerate(new_keys)}

    return {'keys': new_keys, 'cov': new_cov, 'errors': error_dict}

#-------------------- old gradient_descent.py ---------------------



def params_dict_to_vector(opt_params):
    """Convert parameter dict to ordered vector"""
    keys = list(opt_params.keys())
    vec = jnp.array([opt_params[k] for k in keys], dtype=jnp.float64)
    return vec, keys

def vector_to_params_dict(vec, keys):
    """Convert parameter vector back to dict"""
    return {k: vec[i] for i, k in enumerate(keys)}


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
    model_params, opt_params, fixed_params = gradient_descent.prepare_model_params(opt_params, fixed_params)
    distance_pc = extract_streamline.to_float64(distance_pc)


    # Protect near-zero v_r0 from creating singularities in physics calculations
    # Allow negative v_r0, but replace exact-zero or tiny values with signed epsilon
    v_r0_protected = model_params['v_r0']
    threshold = extract_streamline.to_float64(1e-6)
    v_r0_protected = jnp.where(
        jnp.isclose(v_r0_protected, extract_streamline.to_float64(0.0)),
        - jnp.sign(v_r0_protected) * threshold,
        v_r0_protected
        )

    # derive mu from rc or omega (whicever is provided)
    if 'mu' in model_params:
        mu = model_params['mu']
    elif 'rc' in model_params:
        mu = model_params['rc'] / model_params['r0']
    elif 'omega' in model_params:
        mu = stream_lines_grad.mu_from_omega(omega=model_params['omega'], mass=model_params['mass'], r0=model_params['r0'])
    else:
        raise ValueError("model_params must contain either 'rc', 'omega', or 'mu'")
    model_params['mu'] = mu

    # Run the forward model - returns positions in au, velocities in km/s
    (x, y, z), (vx, vy, vz) = xyz_stream(
        mass=model_params['mass'],
        r0=model_params['r0'],
        theta0=model_params['theta0'],
        phi0=model_params['phi0'],
        mu=model_params['mu'],
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
    extract_streamline.get_distance_metric

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
    ra_model = extract_streamline.to_float64(ra_model)
    dec_model = extract_streamline.to_float64(dec_model)
    v_model = extract_streamline.to_float64(v_model)
    ra_data = extract_streamline.to_float64(ra_data)
    dec_data = extract_streamline.to_float64(dec_data)

    # get distance metrics
    dmetric_model, _ = extract_streamline.get_distance_metric(ra_model, dec_model)
    dmetric_data, _ = extract_streamline.get_distance_metric(ra_data, dec_data)

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
    d_data_norm = (d_data - data_min_eff) / data_span_safe
    d_goal = model_min + d_data_norm * model_span_safe

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
        Created via extract_streamline.prepare_data(data, uncertainties).

        
            Created via extract_streamline.prepare_data(data, uncertainties).
    --------
    float: Chi-squared loss value
    """

    loss_method = gradient_descent.check_loss_method(loss_method)

    opt_params, fixed_params = gradient_descent.sanitize_param_partition(opt_params, fixed_params)
    distance_pc = extract_streamline.to_float64(distance_pc)

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
    chi2_v_by_point = (((v_data[valid] - v_model_interp[valid]) / v_sigma[valid]) ** 2)
    chi2_v = jnp.sum((((v_data[valid] - v_model_interp[valid]) / v_sigma[valid]) ** 2))

    if loss_method == 0:
        chi2_ra = jnp.sum((((ra_data[valid] - ra_model_interp[valid]) / ra_sigma[valid]) ** 2))
        chi2_dec = jnp.sum((((dec_data[valid] - dec_model_interp[valid]) / dec_sigma[valid]) ** 2))
        chi2_total = chi2_ra + chi2_dec + chi2_v
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
        r_eps = extract_streamline.to_float64(1e-8)
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
            'chi2_total': chi2_total.astype(float),
        }
    else:
        chi2_components = {
            'chi2_r': chi2_r.astype(float),
            'chi2_theta': chi2_theta.astype(float),
            'chi2_v': chi2_v.astype(float),
            'chi2_total': chi2_total.astype(float),
        }

    loss_trace = {
        'chi2_components': chi2_components,
        'matching': matching_trace,
        'loss_method': loss_method,
    }
    return chi2_total, loss_trace




# ------------------ old stream_lines_grad.py --------------------


## important streamline quantities (for easy reuse)
class StreamState(NamedTuple):
    rc: jnp.ndarray
    mu: jnp.ndarray
    nu: jnp.ndarray
    epsilon: jnp.ndarray
    ecc: jnp.ndarray
    vk0: jnp.ndarray



@jax.jit
def stream_line(r, stream_state, theta0=jnp.radians(30), phi0=jnp.radians(15)):
    '''
    It calculates the stream line following Mendoza et al. (2009),
    only for r < r0. Point r = r0 is handled outside the function.
    It takes the radial velocity and rotation at the streamline
    initial radius and it describes the entire trajectory.

    :param r: au
    :param stream_state: StreamState named tuple containing precomputed quantities for the streamline
    :param theta0: radians
    :param phi0: radians
    :return: theta, radians
    '''
    r = jnp.asarray(r, dtype=FLOAT_DTYPE)
    rc = stream_state.rc
    mu = stream_state.mu
    ecc = stream_state.ecc

    # orb_ang is varphi in Mendoza+2009
    # at initial position r_to_rc = r0/rc = 1/mu
    orb_ang0 = stream_lines_grad.get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)

    # vectorised computation over array of r values
    r_to_rc = r / rc
    orb_ang = stream_lines_grad.get_orb_ang(r_to_rc=r_to_rc, theta0=theta0, ecc=ecc)
    theta = stream_lines_grad.get_theta(theta0, orb_ang, orb_ang0)
    phi = phi0 + stream_lines_grad.get_dphi(theta, theta0=theta0)

    # remove values where r_to_rc < 0.5 (inside centrifugal radius)
    mask = r_to_rc >= 0.5
    orb_ang = jnp.where(mask, orb_ang, jnp.nan)
    theta = jnp.where(mask, theta, jnp.nan)
    phi = jnp.where(mask, phi, jnp.nan)

    return orb_ang, theta, phi #in radians


def xyz_stream(mass=0.5, r0=1e4, theta0=jnp.radians(30),
               phi0=jnp.radians(15), mu=0.1, v_r0=0,
               inc=jnp.radians(0), pa=jnp.radians(0), rmin=None, deltar=1):
    '''
    it gets xyz coordinates and velocities for a stream line.
    They are also rotated in PA and inclination along the line of sight.
    This is a wrapper around stream_line() and stream_lines_grad.rotate_xyz()

    Spherical into cartesian transformation is done for position and velocity
    using:
    https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    :param mass: Central mass (Msun)
    :param r0: Initial radius of streamline (au)
    :param theta0: Initial polar angle of streamline (radians)
    :param phi0: Initial azimuthal angle of streamline (radians)
    :param mu: dimensionless parameter related to rotation, mu = r_cent/r0
    :param v_r0: Initial radial velocity of the streamline, (km/s)
    :param inc: inclination with respect of line-of-sight, inc=0 is an edge-on-disk (radians)
    :param pa: Position angle of the rotation axis, measured due East from North. This is usually estimated from the outflow PA, or the disk PA-90deg., (radians)
    :param rmin: smallest radius for calculation, (au)
    :param deltar: spacing between two consecutive radii in the sampling of the streamer, in (au)
    :return: x, y, z in (au), v_x, v_y, v_z in (km/s)
    '''

    mass = jnp.asarray(mass, dtype=FLOAT_DTYPE)
    r0 = jnp.asarray(r0, dtype=FLOAT_DTYPE)
    theta0 = jnp.asarray(theta0, dtype=FLOAT_DTYPE)
    phi0 = jnp.asarray(phi0, dtype=FLOAT_DTYPE)
    mu = jnp.asarray(mu, dtype=FLOAT_DTYPE)
    v_r0 = jnp.asarray(v_r0, dtype=FLOAT_DTYPE)
    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)
    deltar = jnp.asarray(deltar, dtype=FLOAT_DTYPE)
    if rmin is not None:
        rmin = jnp.asarray(rmin, dtype=FLOAT_DTYPE)

    stream_state = stream_lines_grad.build_stream_quantities(mass=mass, r0=r0, theta0=theta0, mu=mu, v_r0=v_r0)
    rc = stream_state.rc
    mu = stream_state.mu
    ecc = stream_state.ecc

    rotation_matrix = stream_lines_grad.build_rotation_matrix(inc, pa)


    # checkify.check(rc <= r0, "Centrifugal radius is larger that start of streamline.")

    r_low = jnp.maximum(rmin, rc*0.5) if rmin is not None else rc*0.5
    # r is values internal to the initial radius r0 for computation
    r = jnp.arange(r0 - deltar, r_low, step=-1*deltar, dtype=FLOAT_DTYPE)

    # calculate positions and velocities inside r0
    orb_ang, theta, phi = stream_line(r, stream_state=stream_state, theta0=theta0, phi0=phi0)
    v_r, v_theta, v_phi = stream_lines_grad.stream_line_vel(r, theta, orb_ang, stream_state=stream_state, theta0=theta0)

    # prepend initial positions and velocities at r0
    r_full = jnp.concatenate((jnp.asarray([r0], dtype=FLOAT_DTYPE), r))
    theta_full = jnp.concatenate((jnp.asarray([theta0], dtype=FLOAT_DTYPE), theta))
    phi_full = jnp.concatenate((jnp.asarray([phi0], dtype=FLOAT_DTYPE), phi))
    orb_ang0 = stream_lines_grad.get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)
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
    return stream_lines_grad.rotate_xyz(x, y, z, rotation_matrix=rotation_matrix), \
           stream_lines_grad.rotate_xyz(v_x, v_y, v_z, rotation_matrix=rotation_matrix)

# ----------------- old extract_streamline.py ------------------

PreparedData = namedtuple('PreparedData', [
    'ra_data', 'dec_data', 'v_data',
    'ra_sigma_safe', 'dec_sigma_safe', 'v_sigma_safe',
    'dmetric_data', 'data_finite_mask',
    'data_min', 'data_max',
    'r_proj_data', 'theta_proj_data',
])



    





