'''
This file contains the loss function and optimization routines for streamfit.

The optimization uses Adam (adaptive moment estimation) optimizer to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 22-01-26
'''

import jax.numpy as jnp
from jax import jit, grad, value_and_grad
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
    
    # Convert positions from au to arcsec offsets
    # x = RA offset (with negative for standard RA convention)
    # z = Dec offset
    # y = line-of-sight velocity
    ra_model = -x / distance_pc  # arcsec
    dec_model = z / distance_pc  # arcsec
    v_model = vy + fixed_params['v_lsr']  # km/s (add systemic velocity)
    
    return ra_model, dec_model, v_model


def arc_length_2d(x, z):
    """
    Compute cumulative 2D arc length along curve in the x-z plane (POS).
    
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
    dx = jnp.diff(x)
    dz = jnp.diff(z)
    ds = jnp.sqrt(dx**2 + dz**2)
    s = jnp.concatenate((jnp.array([0.0]), jnp.cumsum(ds)))
    return s


def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data):
    """
    Match model output to data, using plane-of-sky curve length parameterisation.
    """
    # arc-lengths
    s_model = arc_length_2d(ra_model, dec_model)
    s_data = arc_length_2d(ra_data, dec_data)
    # cut off where model ends
    s_max = jnp.max(s_model)
    valid = s_data <= s_max
    s_data = s_data[valid]
    ra_data = ra_data[valid]
    dec_data = dec_data[valid]
    # interpolate model quantities to data arc-lengths
    ra_model_interp = jnp.interp(s_data, s_model, ra_model)
    dec_model_interp = jnp.interp(s_data, s_model, dec_model)
    v_model_interp = jnp.interp(s_data, s_model, v_model)

    return ra_model_interp, dec_model_interp, v_model_interp, valid




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
    
    # Run forward model
    ra_model, dec_model, v_model = forward_model(opt_params, fixed_params, distance_pc)
    # Remove NaNs (due to rmin) from model
    not_nan = ~jnp.isnan(ra_model) & ~jnp.isnan(dec_model) & ~jnp.isnan(v_model)
    ra_model = ra_model[not_nan]
    dec_model = dec_model[not_nan]
    v_model = v_model[not_nan]
    
    # Match model to data using arc-length parameterisation
    ra_model_interp, dec_model_interp, v_model_interp, valid = match_model_to_data_curve(
        ra_model, dec_model, v_model, ra_data, dec_data)
    
    # mask for valid (only for arrays that haven't already been masked in matching)
    v_data = v_data[valid]
    ra_sigma = ra_sigma[valid]
    dec_sigma = dec_sigma[valid]
    v_sigma = v_sigma[valid]

    # Compute chi-squared components
    chi2_ra = jnp.sum(((ra_data - ra_model_interp) / ra_sigma)**2)
    chi2_dec = jnp.sum(((dec_data - dec_model_interp) / dec_sigma)**2)
    chi2_v = jnp.sum(((v_data - v_model_interp) / v_sigma)**2)
    # Total chi-squared
    chi2_total = chi2_ra + chi2_dec + chi2_v
    
    return chi2_total


def adam_step(opt_params, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
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
        Learning rate (alpha)
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    eps : float
        Small constant for numerical stability
        
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
        
        # Update parameters
        new_opt_params[key] = opt_params[key] - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    
    return new_opt_params, new_m, new_v


def fit_streamline(initial_opt_params, fixed_params, data, uncertainties, distance_pc,
                   learning_rate=0.001, num_epochs=1000, 
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
        Learning rate for Adam optimizer
    num_epochs : int
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
    
    print(f"Starting optimization with {num_epochs} epochs...")
    print(f"Optimizing parameters: {list(opt_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Initial optimizable values: {opt_params}")
    
    for epoch in range(1, num_epochs + 1):
        # Compute loss and gradients (only w.r.t. opt_params)
        loss_value, grads = loss_and_grad_fn(opt_params, fixed_params, data, uncertainties, distance_pc)
        
        # Perform Adam step
        opt_params, m, v = adam_step(opt_params, grads, m, v, epoch, 
                                     learning_rate=learning_rate,
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
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss_value:.6f}, Best Loss: {best_loss:.6f}')
            print(f'  Current optimizable params: {opt_params}')
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} - no improvement for {early_stopping_patience} epochs")
            break
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {best_loss:.6f}")
    print(f"Best optimized parameters: {best_opt_params}")
    
    return best_opt_params, loss_history
