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


REQUIRED_OPT_PARAM_KEYS = (
    'r0',
    'theta0',
    'phi0',
    'log_omega',
    'v_r0',
)

def _params_dict_to_vector(opt_params):
    """Convert parameter dict to ordered vector."""
    keys = list(opt_params.keys())
    vec = jnp.array([opt_params[k] for k in keys])
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


def _sanitize_opt_params(initial_opt_params):
    """Normalize optimization params to the log_omega API and validate required keys."""
    opt_params = initial_opt_params.copy()

    if 'omega' in opt_params and 'log_omega' in opt_params:
        # If caller passes both (e.g. reusing fit output), prefer optimization-space key.
        del opt_params['omega']

    if 'omega' in opt_params and 'log_omega' not in opt_params:
        raise KeyError(
            "Optimization parameters now require 'log_omega' (natural log of omega). "
            "Convert input using log_omega = log(omega)."
        )

    missing = [key for key in REQUIRED_OPT_PARAM_KEYS if key not in opt_params]
    if missing:
        raise KeyError(
            f"Missing required optimizable parameters: {missing}. "
            f"Required keys are: {list(REQUIRED_OPT_PARAM_KEYS)}"
        )

    return opt_params


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
        Dictionary containing optimizable streamline parameters:
        - 'r0': initial radius (au)
        - 'theta0': initial polar angle (radians)
        - 'phi0': initial azimuthal angle (radians)
        - 'log_omega': natural log of angular rotation (log(1/s))
        - 'v_r0': initial radial velocity (km/s)
    fixed_params : dict
        Dictionary containing fixed streamline parameters:
        - 'mass': stellar mass (Msun)
        - 'inc': inclination (radians)
        - 'pa': position angle (radians)
        - 'rmin': minimum radius (au)
        - 'deltar': radial spacing (au)
        - 'v_lsr': systemic velocity (km/s)
    distance_pc : float
        Distance to source in parsecs
        
    Returns:
    --------
    tuple: (ra_offsets, dec_offsets, velocities) each in appropriate units
        - RA offsets in arcsec (negative for standard convention)
        - Dec offsets in arcsec
        - Line-of-sight velocities in km/s
    """
    if 'log_omega' not in opt_params:
        raise KeyError(
            "forward_model expects 'log_omega' in opt_params. "
            "Use log_omega = log(omega)."
        )

    omega = _omega_from_log_omega(opt_params['log_omega'])

    # Run the forward model - returns positions in au, velocities in km/s
    (x, y, z), (vx, vy, vz) = stream_lines_grad.xyz_stream(
        mass=fixed_params['mass'],
        r0=opt_params['r0'],
        theta0=opt_params['theta0'],
        phi0=opt_params['phi0'],
        omega=omega,
        v_r0=opt_params['v_r0'],
        inc=fixed_params['inc'],
        pa=fixed_params['pa'],
        rmin=fixed_params['rmin'],
        deltar=fixed_params['deltar']
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
    v_model = jnp.where(valid_mask, vy + fixed_params['v_lsr'], jnp.nan)  # km/s (add systemic velocity)

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


# def arc_length_2d(x, z):
#     """
#     Compute cumulative 2D arc length along curve in the x-z plane (POS).
#     NaN values are forward-filled before computation.
    
#     Parameters:
#     -----------
#     x : array
#         x positions (i.e. RA offset)
#     z : array
#         z positions (i.e. Dec offset)
        
#     Returns:
#     --------
#     s : Array, cumulative arc length along the curve
#     """
#     # Forward-fill NaN values to handle invalid region points
#     # TODO: consider interpolation instead of forward-fill?
#     x_filled = forward_fill_nans(x)
#     z_filled = forward_fill_nans(z)
    
#     dx = jnp.diff(x_filled)
#     dz = jnp.diff(z_filled)
#     ds = jnp.sqrt(dx**2 + dz**2)
#     s = jnp.concatenate((jnp.array([0.0]), jnp.cumsum(ds)))
#     return s


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

    # Forward-fill NaNs in model arrays
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
    # print(f"dmetric_model: {dmetric_model}")
    # print(f"dmetric_data: {dmetric_data}")

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
        Optimizable streamline model parameters
    fixed_params : dict
        Fixed streamline model parameters
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

    ra_data, dec_data, v_data = data
    ra_sigma, dec_sigma, v_sigma = uncertainties
    # small values to avoid division by zero
    eps = 1e-8
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
    Only optimizes: r0, theta0, phi0, log_omega, v_r0
    Keeps fixed: mass, inc, pa, rmin, deltar, v_lsr
    
    Parameters:
    -----------
    initial_opt_params : dict
        Initial guess for optimizable parameters:
        - 'r0': initial radius (au)
        - 'theta0': initial polar angle (radians)
        - 'phi0': initial azimuthal angle (radians)
        - 'log_omega': natural log of angular rotation (log(1/s))
        - 'v_r0': initial radial velocity (km/s)
    fixed_params : dict
        Fixed parameters (not optimized):
        - 'mass': stellar mass (Msun)
        - 'inc': inclination (radians)
        - 'pa': position angle (radians)
        - 'rmin': minimum radius (au)
        - 'deltar': radial spacing (au)
        - 'v_lsr': systemic velocity (km/s)
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data
    distance_pc : float
            Distance to source in parsecs
    learning_rate : float
        Default learning rate for Adam optimizer. Used if no specific rate provided for a parameter.
    learning_rate_dict : dict or None
        Per-parameter learning rates: {'r0': 1e-2, 'log_omega': 1e-2, ...}
        If provided, overrides learning_rate for specified parameters.
    param_bounds : dict or None
        Parameter bounds in optimization space: {'log_omega': (log(1e-15), log(1e-10)), ...}
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
    dict: Optimized parameters including both 'log_omega' and derived 'omega'
    list: Loss history
    """
    # Initialize parameters
    opt_params = _sanitize_opt_params(initial_opt_params)

    if learning_rate_dict is not None and 'omega' in learning_rate_dict and 'log_omega' not in learning_rate_dict:
        raise KeyError(
            "learning_rate_dict now expects 'log_omega' instead of 'omega'. "
            "Use log-space learning rates keyed by 'log_omega'."
        )

    if param_bounds is not None and 'omega' in param_bounds and 'log_omega' not in param_bounds:
        raise KeyError(
            "param_bounds now expects 'log_omega' bounds instead of 'omega'. "
            "Use natural-log bounds, e.g. (log(min_omega), log(max_omega))."
        )

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
        print(f"Logging optimization progress to: {log_file}")

    trace_csv_file = None
    trace_csv_writer = None
    if trace_file is not None:
        trace_csv_file = open(trace_file, 'w', newline='')
        trace_csv_writer = csv.DictWriter(trace_csv_file, fieldnames=TRACE_FIELDNAMES)
        trace_csv_writer.writeheader()
        trace_csv_file.flush()
        print(f"Logging matching traces to: {trace_file}")
    
    print(f"Starting optimization with {n_epochs} epochs...")
    print(f"Optimizing parameters: {list(opt_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print("Omega is optimized in log space: omega = exp(log_omega)")
    print(f"Initial optimizable values: {_with_derived_omega(opt_params)}")
    
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
                print(f'  Current optimizable params: {_with_derived_omega(opt_params)}')
            
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

    # compute errors on best-fit parameters
    # Estimate parameter uncertainties
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
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {best_loss:.6f}")
    print(f"Best-fit parameters found at epoch: {best_epoch}")

    return _with_derived_omega(best_opt_params), loss_history, param_errors
