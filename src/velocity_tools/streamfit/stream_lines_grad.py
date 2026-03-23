'''
This file will provide streamline functions that are fully differentiable using JAX.

JAX numpy does not accept units. Therefore, all inputs for jax must be unitless.
The assumed input units are:
- Distance: au
- Velocity: km/s
- Mass: solar masses
- Angles (PA, i, theta, phi...): radians
'''

import astropy.units as u
from scipy import optimize
from ..helper_functions import *
#from astropy.constants import G
import jax
import jax.numpy as jnp
from jax import lax
from jax import debug
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
from jax.scipy.optimize import minimize # may not be needed if using jaxopt
from jaxopt import LBFGSB # may not be needed if using custom Newton method

#
# Implementation of stream lines using the prescription from
# Mendoza et al. (2009)  doi:10.1111/j.1365-2966.2008.14210.x
#

# Constants 
eps = 1e-8 # small value to avoid division by zero
FLOAT_DTYPE = jnp.float64


# JAX functions (no Astropy units allowed here)
# Assumed units: mass (Msun), distances (au), velocities (km/s), angles (radians)

def v_k(radius, mass=0.5):
    """
    Velocity term that is repeated in all velocity component.
    It corresponds to v_k in Mendoza+(2009)
    :param radius: au
    :param mass: Msun
    :return: v_k, km/s
    """
    arg = G * mass / radius
    return jnp.power(arg, 0.5)

def r_cent(mass, omega=1e-14, r0=1e4):
    """
    Centrifugal radius or disk radius in the Ulrich (1976)'s model.
    r_u in Mendoza's nomenclature.

    :param mass: Central mass for the protostar, Msun
    :param omega: Angular speed at the r0 radius, 1/s
    :param r0: Initial radius of the streamline, au
    :return: r_cent, au
    """
    r_cent = (jnp.power(r0, 4) * jnp.power(omega, 2) / (G * mass)) # in au^3 km^-2
    r_cent_au = r_cent * (jnp.power(au_in_km, 2)) # in au
    # jax.debug.print("rc={0} au", r_cent_au)
    return r_cent_au

def safe_arccos(x, eps=1e-8):
    """
    Safe arccos function with clipping to valid range [-1, 1].
    Fully differentiable with JAX.

    :param x: input value
    :param eps: small offset from boundaries to avoid numerical issues
    :return: arccos of clipped input
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(FLOAT_DTYPE)
    x = x.astype(FLOAT_DTYPE)

    # Keep away from +/-1 by at least a few ULPs of the active dtype.
    # This stabilizes gradients without changing equations away from boundary.
    eps_user = jnp.asarray(eps, dtype=x.dtype)
    eps_floor = jnp.asarray(32.0 * jnp.finfo(x.dtype).eps, dtype=x.dtype)
    eps_eff = jnp.maximum(eps_user, eps_floor)
    #jax.debug.print("eps used = {eps_eff}", eps_eff=eps_eff)

    x_safe = jnp.clip(x, -1.0 + eps_eff, 1.0 - eps_eff)
    # print if clipping is happening
    # if jnp.any(x < -1.0 + eps_eff) or jnp.any(x > 1.0 - eps_eff):
    #     jax.debug.print("Warning: input to arccos was clipped. Original x={x}, clipped x={x_safe}", x=x, x_safe=x_safe)
    return jnp.arccos(x_safe)


def get_theta(theta0, orb_ang, orb_ang0):
    """
    Gets theta from theta0, orb_ang, and orb_ang0, in radians.
    Eqn (8) in Mendoza+2009
    
    :param theta0: radians
    :param orb_ang: radians
    :param orb_ang0: radians
    """
    cos_theta = jnp.cos(theta0) * jnp.cos(orb_ang - orb_ang0)
    theta = safe_arccos(cos_theta)
    return theta


def get_orb_ang(r_to_rc, theta0, ecc):
    """
    Gets orb_ang (varphi in Mendoza+2009), in radians.
    To get initial orb_ang, set r_to_rc = r0/rc = 1/mu
    
    :param r_to_rc: Description
    :param theta0: Description
    :param ecc: Description
    """
    cos_orb_ang = (1/ecc) * (1 - (jnp.power(jnp.sin(theta0), 2) / r_to_rc))
    orb_ang = safe_arccos(cos_orb_ang)
    return orb_ang

def get_dphi(theta, theta0=jnp.radians(30)):
    """
    Gets the difference in Phi between initial and current, in radians.

    :param theta: radians
    :param theta0: radians
    :return: difference in Phi angle, radians
    """
    arg = jnp.tan(theta0) / jnp.tan(theta)
    return safe_arccos(arg)



#TODO: come back and check this function at end
def stream_line(r, mass=0.5, r0=1e4, theta0=jnp.radians(30), phi0=jnp.radians(15),
                omega=1e-14, v_r0=0):
    """
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
    """
    rc = r_cent(mass=mass, omega=omega, r0=r0)

    # mu and nu are dimensionless
    mu = (rc / r0)
    nu = v_r0 * jnp.power((rc / (G * mass)), 0.5)
    # epsilon is the dimensionless energy
    epsilon = jnp.power(nu, 2) + jnp.power(mu, 2) * jnp.power(jnp.sin(theta0), 2) - 2 * mu
    ecc = jnp.power((1 + epsilon * jnp.power(jnp.sin(theta0), 2)), 0.5)
    # jax.debug.print("ecc={0}", ecc)
    # jax.debug.print("mu={0}", mu)
    # jax.debug.print("theta0={0}", theta0)

    # orb_ang is varphi in Mendoza+2009
    # at initial position r_to_rc = r0/rc = 1/mu
    orb_ang0 = get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)

    # Initialise arrays
    orb_ang = jnp.zeros_like(r)
    theta = jnp.zeros_like(r)
    phi = jnp.zeros_like(r)

    # get initial orb_ang at r0
    # orb_ang is varphi in Mendoza+2009
    #at initial position r_to_rc = r0/rc = 1/mu
    orb_ang0 = get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)

    for ind in range(len(r)):
        r_i = (r[ind] / rc)
        orb_ang_i = get_orb_ang(r_to_rc=r_i, theta0=theta0, ecc=ecc)
        orb_ang = orb_ang.at[ind].set(orb_ang_i)
        theta_i = get_theta(theta0, orb_ang_i, orb_ang0)
        theta = theta.at[ind].set(theta_i)
        dphi = get_dphi(theta_i, theta0=theta0)
        phi = phi.at[ind].set(phi0 + dphi)

    # remove values where r_to_rc < 0.5 (inside centrifugal radius)
    mask = (r / rc) >= 0.5
    orb_ang = jnp.where(mask, orb_ang, jnp.nan)
    theta = jnp.where(mask, theta, jnp.nan)
    phi = jnp.where(mask, phi, jnp.nan)

    # jax.debug.print("orb_ang = {orb_ang}", orb_ang=orb_ang)
    # jax.debug.print("theta = {theta}", theta=theta)
    # jax.debug.print("phi = {phi}", phi=phi)
    
    return orb_ang, theta, phi #in radians


def stream_line_vel(r, theta, orb_ang, mass=0.5, r0=1e4, theta0=jnp.radians(30),
                omega=1e-14, v_r0=0):
    """
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
    """
    rc = r_cent(mass=mass, omega=omega, r0=r0)
    r_to_rc = (r / rc)
    v_k0 = v_k(rc, mass=mass)
    # mu and nu are dimensionless
    mu = (rc / r0)
    nu = v_r0 * jnp.power((rc / (G * mass)), 0.5)
    epsilon = jnp.power(nu, 2) + jnp.power(mu, 2) * jnp.power(jnp.sin(theta0), 2) - 2 * mu
    ecc = jnp.power((1 + epsilon * jnp.power(jnp.sin(theta0), 2)), 0.5)
    #
    v_r_all = -ecc * jnp.sin(theta0) * jnp.sin(orb_ang) / r_to_rc /(1 - ecc*jnp.cos(orb_ang))
    v_theta_all = jnp.sin(theta0) / jnp.sin(theta) / r_to_rc \
                  * jnp.power((jnp.power(jnp.cos(theta0),2) - jnp.power(jnp.cos(theta),2)), 0.5)
    v_phi_all = jnp.power(jnp.sin(theta0), 2) / (jnp.sin(theta) * r_to_rc)

    return v_r_all * v_k0, v_theta_all * v_k0, v_phi_all * v_k0


def rotate_xyz(x, y, z, inc=jnp.radians(30), pa=jnp.radians(30)):
    """
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
    :param inc: Inclination angle. 0=no change. radians
    :param pa: Change the PA angle. Measured from North due East. radians
    :return: new x, y, and z-coordinates as observed on the sky, with the
    same units as the input ones.

    """
    x = jnp.asarray(x, dtype=FLOAT_DTYPE)
    y = jnp.asarray(y, dtype=FLOAT_DTYPE)
    z = jnp.asarray(z, dtype=FLOAT_DTYPE)
    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)

    xyz = jnp.stack([x, y, z], axis=0)

    rot_inc = jnp.array([[1, 0, 0],
                        [0, jnp.cos(inc), jnp.sin(inc)],
                        [0, -jnp.sin(inc), jnp.cos(inc)]], dtype=FLOAT_DTYPE)
    rot_pa = jnp.array([[jnp.cos(pa), 0, -jnp.sin(pa)],
                       [0, 1, 0],
                       [jnp.sin(pa), 0, jnp.cos(pa)]], dtype=FLOAT_DTYPE)
    
    xyz_new = rot_pa @ rot_inc @ xyz
    x_new, y_new, z_new = jnp.unstack(xyz_new, axis=0)
    return x_new, y_new, z_new


# Astropy wrapper - handles astropy units and calls jax-compatible maths
def xyz_stream(mass=0.5*u.Msun, r0=1e4*u.au, theta0=30*u.deg,
               phi0=15*u.deg, omega=1e-14/u.s, v_r0=0*u.km/u.s,
               inc=0*u.deg, pa=0*u.deg, rmin=None, deltar=1*u.au):
    """
    it gets xyz coordinates and velocities for a stream line.
    They are also rotated in PA and inclination along the line of sight.
    This is a wrapper around stream_line() and rotate_xyz()

    Spherical into cartesian transformation is done for position and velocity
    using:
    https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    :param mass: Central mass, Msun
    :param r0: Initial radius of streamline, au
    :param theta0: Initial polar angle of streamline, degrees
    :param phi0: Initial azimuthal angle of streamline, degrees
    :param omega: Angular rotation. (defined positive), 1/s
    :param v_r0: Initial radial velocity of the streamline, km/s
    :param inc: inclination with respect of line-of-sight, inc=0 is an edge-on-disk, degrees
    :param pa: Position angle of the rotation axis, measured due East from North. This is usually estimated from the outflow PA, or the disk PA-90deg., degrees
    :param rmin: smallest radius for calculation, au
    :param deltar: spacing between two consecutive radii in the sampling of the streamer, in au
    :return: x, y, z in au, v_x, v_y, v_z in km/s
    """

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

    # quantities we will need later
    rc = r_cent(mass=mass, omega=omega, r0=r0)
    #jax.debug.print("rc={0} au", rc)
    mu = (rc / r0)
    nu = v_r0 * jnp.power((rc / (G * mass)), 0.5)
    epsilon = jnp.power(nu, 2) + jnp.power(mu, 2) * jnp.power(jnp.sin(theta0), 2) - 2 * mu
    ecc = jnp.power((1 + epsilon * jnp.power(jnp.sin(theta0), 2)), 0.5)
    if rc > r0:
        # early stop if centrifugal radius is larger than r0
        # TODO: ideally centrifugal radius should be fed in as the minimum of r0
        jax.debug.print("Centrifugal radius (rc={0} au) > initial radius (r0={1} au).", rc, r0)
        raise ValueError('Centrifugal radius is larger than start of streamline')
    r_low = jnp.maximum(rmin, rc*0.5) if rmin is not None else rc*0.5
    # r is values internal to the initial radius r0 for computation
    r = jnp.arange(r0 - deltar, r_low, step=-1*deltar, dtype=FLOAT_DTYPE)
    # print("r = {0}".format(r))

    # calculate positions and velocities inside r0
    orb_ang, theta, phi = stream_line(r, mass=mass, r0=r0, theta0=theta0, phi0=phi0,
                        omega=omega, v_r0=v_r0)
    #
    v_r, v_theta, v_phi = stream_line_vel(r, theta, orb_ang, mass=mass, r0=r0,
                                          theta0=theta0, omega=omega, v_r0=v_r0)
    # prepend initial positions and velocities at r0
    r_full = jnp.concatenate((jnp.asarray([r0], dtype=FLOAT_DTYPE), r))
    theta_full = jnp.concatenate((jnp.asarray([theta0], dtype=FLOAT_DTYPE), theta))
    phi_full = jnp.concatenate((jnp.asarray([phi0], dtype=FLOAT_DTYPE), phi))
    orb_ang0 = get_orb_ang(r_to_rc=1/mu, theta0=theta0, ecc=ecc)
    orb_ang_full = jnp.concatenate((jnp.asarray([orb_ang0], dtype=FLOAT_DTYPE), orb_ang))
    v_r_full = jnp.concatenate((jnp.asarray([v_r0], dtype=FLOAT_DTYPE), v_r))
    v_theta_full = jnp.concatenate((jnp.asarray([0.0], dtype=FLOAT_DTYPE), v_theta))
    # we need to calculate v_phi0 (multiply by v_k0)
    v_k0 = v_k(rc, mass=mass)
    v_phi0 = v_k0 * jnp.power(jnp.sin(theta0), 2) / (jnp.sin(theta0) * (r0/rc))
    v_phi_full = jnp.concatenate((jnp.asarray([v_phi0], dtype=FLOAT_DTYPE), v_phi))


    v_x = v_r_full * jnp.sin(theta_full) * jnp.cos(phi_full) \
          + v_theta_full * jnp.cos(theta_full) * jnp.cos(phi_full) \
          - v_phi_full * jnp.sin(phi_full)
    v_y = v_r_full * jnp.sin(theta_full) * jnp.sin(phi_full) \
          + v_theta_full * jnp.cos(theta_full) * jnp.sin(phi_full) \
          + v_phi_full * jnp.cos(phi_full)
    v_z = v_r_full * jnp.cos(theta_full) \
          - v_theta_full * jnp.sin(theta_full)
    # Convert from spherical into cartesian coordinates
    x = r_full * jnp.sin(theta_full) * jnp.cos(phi_full)
    y = r_full * jnp.sin(theta_full) * jnp.sin(phi_full)
    z = r_full * jnp.cos(theta_full)
    # Get mask from smallest radius for calculation
    if rmin is None:
        gd_rmin = jnp.ones_like(r, dtype=bool)
    else:
        gd_rmin = (r_full > rmin)
    gd_rmin = gd_rmin.astype(x.dtype)
    # Apply mask before rotation
    x = jnp.where(gd_rmin, x, jnp.nan)  
    y = jnp.where(gd_rmin, y, jnp.nan)
    z = jnp.where(gd_rmin, z, jnp.nan)
    v_x = jnp.where(gd_rmin, v_x, jnp.nan)
    v_y = jnp.where(gd_rmin, v_y, jnp.nan)
    v_z = jnp.where(gd_rmin, v_z, jnp.nan)
    # Rotate
    return rotate_xyz(x, y, z, inc=inc, pa=pa), \
           rotate_xyz(v_x, v_y, v_z, inc=inc, pa=pa)

    '''
    OLD mask and rotation logic:

        if rmin is not None:
        gd_rmin = (r > rmin)
        if gd_rmin.sum() > 0:
            return rotate_xyz(x[gd_rmin], y[gd_rmin], z[gd_rmin], inc=inc, pa=pa),\
                rotate_xyz(v_x[gd_rmin], v_y[gd_rmin], v_z[gd_rmin], inc=inc, pa=pa)
        else:
            return [jnp.nan], [jnp.nan], [jnp.nan], [jnp.nan], [jnp.nan], [jnp.nan]
    else:
        return rotate_xyz(x, y, z, inc=inc, pa=pa), \
               rotate_xyz(v_x, v_y, v_z, inc=inc, pa=pa)
    '''
