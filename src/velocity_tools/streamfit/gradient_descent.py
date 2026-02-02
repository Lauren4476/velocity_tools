'''
This file contains the loss function and optimization routines for streamfit.

The optimization uses Adam (adaptive moment estimation) optimizer to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 02-02-26
'''

import jax.numpy as jnp
from jax import jit, grad, value_and_grad, lax
import jax
from . import stream_lines_grad
from . import extract_streamline
import csv


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
        - 'omega': angular rotation (1/s)
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
    # Run the forward model - returns positions in au, velocities in km/s
    (x, y, z), (vx, vy, vz) = stream_lines_grad.xyz_stream(
        mass=fixed_params['mass'],
        r0=opt_params['r0'],
        theta0=opt_params['theta0'],
        phi0=opt_params['phi0'],
        omega=opt_params['omega'],
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


def arc_length_2d(x, z):
    """
    Compute cumulative 2D arc length along curve in the x-z plane (POS).
    NaN values are forward-filled before computation.
    
    Parameters:
    -----------
    x : array
        x positions (i.e. RA offset)
    z : array
        z positions (i.e. Dec offset)
        
    Returns:
    --------
    s : Array, cumulative arc length along the curve
    """
    # Forward-fill NaN values to handle invalid region points
    # TODO: consider interpolation instead of forward-fill?
    x_filled = forward_fill_nans(x)
    z_filled = forward_fill_nans(z)
    
    dx = jnp.diff(x_filled)
    dz = jnp.diff(z_filled)
    ds = jnp.sqrt(dx**2 + dz**2)
    s = jnp.concatenate((jnp.array([0.0]), jnp.cumsum(ds)))
    return s


def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data):
    """
    Extract model values corresponding to data positions, using the same distance metric
    as used for binning the point cloud.
    Uses get_distance_metric from extract_streamline
    """

    # Forward-fill NaNs in model arrays
    ra_model_filled = forward_fill_nans(ra_model)
    dec_model_filled = forward_fill_nans(dec_model)
    v_model_filled = forward_fill_nans(v_model)

    # compute distance metrics for full model and data
    # (no clipping - interpolation will handle matching)
    dmetric_model, _ = extract_streamline.get_distance_metric(
        ra_model_filled, dec_model_filled)
    dmetric_data, _ = extract_streamline.get_distance_metric(
        ra_data, dec_data)
    
    # Replace any NaNs in distance metrics with forward fill
    # TODO: may need to get rid of forward fill here if it causes issues
    # dmetric_model = forward_fill_nans(dmetric_model)
    # dmetric_data = forward_fill_nans(dmetric_data)
    
    # Sample distance metrics to n_points using percentiles
    n_points = len(ra_data)

    percentiles = jnp.linspace(0, 100, n_points)
    dmetric_model_sampled = jnp.percentile(dmetric_model, percentiles)
    dmetric_data_sampled = jnp.percentile(dmetric_data, percentiles)
    
    # Sort the full model arrays by distance metric
    sort_idx_full = jnp.argsort(dmetric_model)
    dmetric_model_full_sorted = dmetric_model[sort_idx_full]
    ra_model_full_sorted = ra_model_filled[sort_idx_full]
    dec_model_full_sorted = dec_model_filled[sort_idx_full]
    v_model_full_sorted = v_model_filled[sort_idx_full]
    
    # Interpolate model to sampled distance metric positions
    ra_model_sampled = jnp.interp(dmetric_model_sampled, dmetric_model_full_sorted, ra_model_full_sorted)
    dec_model_sampled = jnp.interp(dmetric_model_sampled, dmetric_model_full_sorted, dec_model_full_sorted)
    v_model_sampled = jnp.interp(dmetric_model_sampled, dmetric_model_full_sorted, v_model_full_sorted)
    
    # Sort sampled model by its distance metric
    sort_idx = jnp.argsort(dmetric_model_sampled)
    dmetric_model_sampled_sorted = dmetric_model_sampled[sort_idx]
    ra_model_sampled_sorted = ra_model_sampled[sort_idx]
    dec_model_sampled_sorted = dec_model_sampled[sort_idx]
    v_model_sampled_sorted = v_model_sampled[sort_idx]
    
    # Now interpolate the sampled model to data distance metric positions
    ra_model_interp = jnp.interp(dmetric_data_sampled, dmetric_model_sampled_sorted, ra_model_sampled_sorted)
    dec_model_interp = jnp.interp(dmetric_data_sampled, dmetric_model_sampled_sorted, dec_model_sampled_sorted)
    v_model_interp = jnp.interp(dmetric_data_sampled, dmetric_model_sampled_sorted, v_model_sampled_sorted)

    return ra_model_interp, dec_model_interp, v_model_interp, jnp.ones(n_points, dtype=bool)


def chi2_loss(opt_params, fixed_params, data, uncertainties, distance_pc):
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
    ra_model_interp, dec_model_interp, v_model_interp, _ = match_model_to_data_curve(
        ra_model, dec_model, v_model, ra_data, dec_data)


    # Compute chi-squared components
    chi2_ra = jnp.sum(((ra_data - ra_model_interp) / ra_sigma)**2)
    chi2_dec = jnp.sum(((dec_data - dec_model_interp) / dec_sigma)**2)
    chi2_v = jnp.sum(((v_data - v_model_interp) / v_sigma)**2)
    # Total chi-squared
    chi2_total = chi2_ra + chi2_dec + chi2_v
    
    return chi2_total


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


def fit_streamline(initial_opt_params, fixed_params, data, uncertainties, distance_pc,
                   learning_rate=0.001, learning_rate_dict=None, param_bounds=None, n_epochs=1000, 
                   beta1=0.9, beta2=0.999, 
                   info_every=100, early_stopping_patience=50, log_file=None):
    """
    Fit streamline model parameters to data using Adam optimizer.
    Only optimizes: r0, theta0, phi0, omega, v_r0
    Keeps fixed: mass, inc, pa, rmin, deltar, v_lsr
    
    Parameters:
    -----------
    initial_opt_params : dict
        Initial guess for optimizable parameters:
        - 'r0': initial radius (au)
        - 'theta0': initial polar angle (radians)
        - 'phi0': initial azimuthal angle (radians)
        - 'omega': angular rotation (1/s)
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
        Per-parameter learning rates: {'r0': 1e-2, 'omega': 1e-3, ...}
        If provided, overrides learning_rate for specified parameters.
    param_bounds : dict or None
        Parameter bounds: {'omega': (1e-15, 1e-10), ...}
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
        
    Returns:
    --------   
    dict: Optimized parameters (opt_params only)
    list: Loss history
    """
    # Initialize parameters
    opt_params = initial_opt_params.copy()
    
    # Initialize Adam moments (only for optimizable parameters)
    m = {key: 0.0 for key in opt_params.keys()}
    v = {key: 0.0 for key in opt_params.keys()}
    
    # Create gradient function (only w.r.t. opt_params)
    loss_and_grad_fn = value_and_grad(chi2_loss, argnums=0)
    
    # Track loss history
    loss_history = []
    best_loss = float('inf')
    best_opt_params = opt_params.copy()
    best_epoch = 0
    patience_counter = 0
    
    # Initialize CSV log file if requested
    csv_file = None
    csv_writer = None
    if log_file is not None:
        csv_file = open(log_file, 'w', newline='')
        # Create header: epoch, loss, then all optimizable params
        fieldnames = ['epoch', 'loss'] + list(opt_params.keys())
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"Logging optimization progress to: {log_file}")
    
    print(f"Starting optimization with {n_epochs} epochs...")
    print(f"Optimizing parameters: {list(opt_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Initial optimizable values: {opt_params}")
    
    # Log initial parameters (epoch 0) if CSV logging is enabled
    if csv_writer is not None:
        initial_loss = chi2_loss(opt_params, fixed_params, data, uncertainties, distance_pc)
        row = {'epoch': 0, 'loss': float(initial_loss)}
        for key in opt_params.keys():
            row[key] = float(opt_params[key])
        csv_writer.writerow(row)
        csv_file.flush()
    
    try:
        for epoch in range(1, n_epochs + 1):
            if epoch % info_every == 0:
                print(f"\n Starting Epoch {epoch} -------------------------")
            # Compute loss and gradients (only w.r.t. opt_params)
            loss_value, grads = loss_and_grad_fn(opt_params, fixed_params, data, uncertainties, distance_pc)
        
            # Perform Adam step
            opt_params, m, v = adam_step(opt_params, grads, m, v, epoch, 
                                     learning_rate=learning_rate,
                                     learning_rate_dict=learning_rate_dict,
                                     param_bounds=param_bounds,
                                     beta1=beta1, beta2=beta2)
        
            # Track loss
            loss_history.append(float(loss_value))
        
            # Log to CSV if requested
            if csv_writer is not None:
                row = {'epoch': epoch, 'loss': float(loss_value)}
                # Add all optimizable parameter values
                for key in opt_params.keys():
                    row[key] = float(opt_params[key])
                csv_writer.writerow(row)
                csv_file.flush()  # Ensure data is written after each epoch
        
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
                print(f'  Current optimizable params: {opt_params}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} - no improvement for {early_stopping_patience} epochs")
                break
    
    finally:
        # Always close the CSV file if it was opened
        if csv_file is not None:
            csv_file.close()
            print(f"Optimization log saved to: {log_file}")
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {best_loss:.6f}")
    print(f"Best-fit parameters found at epoch: {best_epoch}")
    
    return best_opt_params, loss_history
