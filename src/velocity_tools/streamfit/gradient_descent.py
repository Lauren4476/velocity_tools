'''
This file contains the loss function and optimization routines for streamfit.

The optimization uses Adam (adaptive moment estimation) optimizer to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 22-01-26
'''

import jax.numpy as jnp
from jax import jit, grad, value_and_grad, lax
import jax
from . import stream_lines_grad


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
    x_filled = forward_fill_nans(x)
    z_filled = forward_fill_nans(z)
    
    dx = jnp.diff(x_filled)
    dz = jnp.diff(z_filled)
    ds = jnp.sqrt(dx**2 + dz**2)
    s = jnp.concatenate((jnp.array([0.0]), jnp.cumsum(ds)))
    return s


def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data):
    """
    Match model output to data, using plane-of-sky curve length parameterisation.
    NaN values in model arrays are handled by forward-filling.
    """
    # arc-lengths (forward-fill handles NaN via arc_length_2d)
    s_model = arc_length_2d(ra_model, dec_model)
    s_data = arc_length_2d(ra_data, dec_data)

    # cut off where model ends
    s_max = jnp.max(s_model)

    # clamp s_data to valid range [0, s_max]
    s_data_clamped = jnp.clip(s_data, 0.0, s_max)

    # Forward-fill NaN values in model arrays before interpolation
    ra_model_filled = forward_fill_nans(ra_model)
    dec_model_filled = forward_fill_nans(dec_model)
    v_model_filled = forward_fill_nans(v_model)
    
    # interpolate model quantities to data arc-lengths
    ra_model_interp = jnp.interp(s_data_clamped, s_model, ra_model_filled)
    dec_model_interp = jnp.interp(s_data_clamped, s_model, dec_model_filled)
    v_model_interp = jnp.interp(s_data_clamped, s_model, v_model_filled)

    return ra_model_interp, dec_model_interp, v_model_interp, jnp.ones_like(s_data, dtype=bool)




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
                   info_every=100, early_stopping_patience=50):
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
    patience_counter = 0
    
    print(f"Starting optimization with {n_epochs} epochs...")
    print(f"Optimizing parameters: {list(opt_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Initial optimizable values: {opt_params}")
    
    for epoch in range(1, n_epochs + 1):
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
        
        # Early stopping check
        if loss_value < best_loss:
            best_loss = loss_value
            best_opt_params = opt_params.copy()
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
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {best_loss:.6f}")
    print(f"Best optimized parameters: {best_opt_params}")
    
    return best_opt_params, loss_history
