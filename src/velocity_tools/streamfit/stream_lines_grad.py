'''
Contains all functions needed for a forward model of a streamline.
All are fully differentiable using JAX, except for the astropy wrapper function (bottom), 
which handles units and calls the jax-compatible functions

Streamline implementation is based on Mendoza et al. (2009) doi:10.1111/j.1365-2966.2008.14210.x

The assumed input units are:
- distance on sky: au
- velocity: km/s
- mass: solar masses
- angles (PA, i, theta, phi...): radians
- Omega (angular velocity): 1/s
- distance to source: pc
'''

import astropy.units as u
from ..helper_functions import *
import jax
import jax.numpy as jnp
# from jax import lax
# from jax import debug
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
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


# Astropy wrapper - handles astropy units and calls jax-compatible maths
# TODO: lauren you need to make this jax compatible
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

    if rc > r0:
        # early stop if centrifugal radius is larger than r0
        # TODO: ideally centrifugal radius should be fed in as the minimum of r0
        raise ValueError('Centrifugal radius is larger than start of streamline')
    r_low = jnp.maximum(rmin, rc*0.5) if rmin is not None else rc*0.5
    # r is values internal to the initial radius r0 for computation
    r = jnp.arange(r0 - deltar, r_low, step=-1*deltar, dtype=FLOAT_DTYPE)
    # print("r = {0}".format(r))

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
