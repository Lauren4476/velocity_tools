'''
This file will provide streamline functions that are fully differentiable using JAX.

JAX numpy does not accept units. Therefore, all inputs for jax must be unitless.
The wrapper function is xyz_stream. Currently, user inputs quantities with astropy units into this 
The assumed input units are:
- Distance: au
- Velocity: km/s
- Mass: solar masses
- Angles (PA, i, theta, phi...): degrees
    - these are then immediately converted into radians for calculations in all other functions in this file.
'''

'''
TODO:
- Add bounds to minimisation code for getting theta?
- Replace python for loops with vectorisation or lax.scan
- Remove unused imports
- Test
'''

import astropy.units as u
from scipy import optimize
from ..helper_functions import *
#from astropy.constants import G
import jax
import jax.numpy as jnp
from jax import lax
from jax import debug
from jax.scipy.optimize import minimize


#
# Implementation of stream lines using the prescription from
# Mendoza et al. (2009)  doi:10.1111/j.1365-2966.2008.14210.x
#

# Constants 
eps = 1e-8


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
    return jnp.sqrt(G * mass / radius)


def r_cent(mass, omega=1e-14, r0=1e4):
    """
    Centrifugal radius or disk radius in the Ulrich (1976)'s model.
    r_u in Mendoza's nomenclature.

    :param mass: Central mass for the protostar, Msun
    :param omega: Angular speed at the r0 radius, 1/s
    :param r0: Initial radius of the streamline, au
    :return: r_cent, au
    """
    r_cent = (r0 ** 4 * omega ** 2 / (G * mass)) # in au^3 km^-2
    return r_cent * (au_in_km**2) # in au


def theta_abs(theta, r_to_rc=0.1, theta0=jnp.radians(30), ecc=1.,
              orb_ang=jnp.pi / 2):
    """
    function to determine theta numerically by finding the root of a function
    This is equation (9) in Mendoza+(2009)

    :param theta: angle of streamline, radians
    :param r_to_rc: radius in units of the centrifugal radius
    :param theta0: Initial angle of the streamline, radians
    :param ecc: eccentricity of the orbit (equation 6)
    :param orb_ang: angle in the orbital motion (equation 7), radians
    :return: returns the difference between the radius and the predicted one,
           a value of 0 corresponds to a proper streamline
    """
    cos_ratio = jnp.cos(theta) / jnp.cos(theta0)
    safe_cos_ratio = jnp.clip(cos_ratio, -1.0 + eps, 1.0 - eps)
    xi = jnp.arccos(safe_cos_ratio) + orb_ang # <- this is in radians
    geom = jnp.sin(theta0)**2 / (1 - ecc * jnp.cos(xi))
    return jnp.sum((r_to_rc - geom)**2) #new, gradient is smooth
    #return jnp.sum(jnp.abs(r_to_rc - geom)) # old, gradient has kink at 0


def get_dphi(theta, theta0=jnp.radians(30)):
    """
    Gets the difference in Phi, in radians.

    :param theta: radians
    :param theta0: radians
    :return: difference in Phi angle, radians
    """
    arg = jnp.tan(theta0) / jnp.tan(theta)
    arg = jnp.clip(arg, -1 + eps, 1 - eps)
    return jnp.arccos(arg)



#TODO: come back and check this function at end
def stream_line(r, mass=0.5, r0=1e4, theta0=jnp.radians(30),
                omega=1e-14, v_r0=0):
    """
    It calculates the stream line following Mendoza et al. (2009)
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
    #print("rc={0}".format(rc))
    theta = jnp.zeros_like(r) + jnp.nan
    # mu and nu are dimensionless
    mu = (rc / r0)
    nu = (v_r0 * jnp.sqrt(rc / (G * mass)))
    # epsilon is the dimensionless energy
    epsilon = nu**2 + mu**2 * jnp.sin(theta0)**2 - 2 * mu
    ecc = jnp.sqrt(1 + epsilon * jnp.sin(theta0)**2)
    # definition of orb_ang here
    orb_ang = jnp.arccos((1 - mu * jnp.sin(theta0)**2) / ecc) #<- this is in radians

    # the first element in the streamline is the starting point
    theta = theta.at[0].set(theta0)
    # Initial guess at largest radius is theta0 +- initguess towards the midplane
    deltar = jnp.amin(jnp.abs(jnp.roll(r,1) - r))
    #print(deltar)
    # we use a constant of 6e-5 for an epsilon of 0.01 km/s
    # this result will be in radians
    tol = (6.e-5 * (au_to_m(deltar) / 1000) * omega / (v_r0+ 0.1))
    # the initial guess will be 10 times the tolerance for now, in testing
    initguess = 10 * tol
    #print('tolerance ', tol)

    ### New minimisation code (jax compatible):
    # theta_bracket unused so far as BFGS does not take bounds...
    if theta0 < jnp.radians(90):
        theta_i = theta0 + initguess
        theta_bracket = [(theta0, jnp.pi/2.)]
    else:
        theta_i = theta0 - initguess
        theta_bracket = [(jnp.pi/2., theta0)]
    theta_i_vec = jnp.atleast_1d(theta_i) # jax minimize expects an array

    for ind in jnp.arange(1, len(r)):
        r_i = (r[ind] / rc)
        if r_i > 0.5:
            # Using jax.scipy.optimize.minimize
            # It uses the BFGS method (currently this is the only method supported)
            # It DOES NOT TAKE BOUNDS
            # It does not parse any optimiser-specific options (e.g. from an options_dict)
            result = minimize(theta_abs, theta_i_vec,
                              args=(r_i, theta0, ecc, orb_ang),
                              method='BFGS',
                              tol=tol)
            theta_i = result.x
            theta_i = theta_i[0] # get the scalar out of the array
            # These prints are to diagnose if the minimization is converging
            #print(ind, result.success)
            #print(result.message, result.status, result.nit)
            theta = theta.at[ind].set(theta_i)
            
    ''' OLD minimisation code (not jax compatible):
    
    if theta0 < jnp.radians(90):
        theta_i = theta0 + initguess
        theta_bracket = [(theta0, jnp.pi/2.)]
    else:
        theta_i = theta0 - initguess
        theta_bracket = [(jnp.pi/2., theta0)]
    for ind in jnp.arange(1, len(r)):
        r_i = (r[ind] / rc)
        if r_i > 0.5:
            # print('initial guess of theta_i = {0}'.format(theta_i))
            # result = optimize.minimize(theta_abs, theta_i,
            #                            bounds=theta_bracket,
            #                            args=(r_i, theta0, ecc, orb_ang))
            # By default, when minimize receives bounds and no constrains,
            # it uses the L-BFGS-B method:
            # ftol is the tolerance in the function evaluation
            # "The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol"
            # gtol corresponds to the parameter pgtol in fmin_l_bfgs_b
            # "The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol"
            # eps corresponds to the absolute step size used for numerical approximation of the jacobian via forward differences.
            options_dict = {'gtol': tol/10., 'eps': tol, 'ftol': tol}
            result = optimize.minimize(theta_abs, theta_i,
                                       bounds=theta_bracket,
                                       args=(r_i, theta0, ecc, orb_ang),
                                       options=options_dict)
            theta_i = result.x
            # These prints are to diagnose if the minimization is converging
            # print(ind, result.success)
            # print(result.message, result.status, result.nit)
            theta_i = theta_i[0] # get the scalar out of the array
            theta = theta.at[ind].set(theta_i)
    '''
    return theta #in radians


def stream_line_vel(r, theta, mass=0.5, r0=1e4, theta0=jnp.radians(30),
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
    nu = (v_r0 * jnp.sqrt(rc / (G * mass)))
    epsilon = nu**2 + mu**2 * jnp.sin(theta0)**2 - 2 * mu
    ecc = jnp.sqrt(1 + epsilon*jnp.sin(theta0)**2)
    orb_ang = jnp.arccos((1 - mu * jnp.sin(theta0)**2) / ecc) # <- this is in radians
    cos_ratio = jnp.cos(theta) / jnp.cos(theta0)
    xi = jnp.arccos(cos_ratio) + orb_ang # <- this is in radians
    #
    v_r_all = -ecc * jnp.sin(theta0) * jnp.sin(xi) / r_to_rc /(1 - ecc*jnp.cos(xi))
    v_theta_all = jnp.sin(theta0) / jnp.sin(theta) / r_to_rc \
                  * jnp.sqrt(jnp.cos(theta0)**2 - jnp.cos(theta)**2)
    v_phi_all = jnp.sin(theta0)**2 / jnp.sin(theta) / r_to_rc

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
    xyz = jnp.stack([x, y, z], axis=0)

    rot_inc = jnp.array([[1, 0, 0],
                        [0, jnp.cos(inc), jnp.sin(inc)],
                        [0, -jnp.sin(inc), jnp.cos(inc)]])
    rot_pa = jnp.array([[jnp.cos(pa), 0, -jnp.sin(pa)],
                       [0, 1, 0],
                       [jnp.sin(pa), 0, jnp.cos(pa)]])
    
    xyz_new = rot_pa @ rot_inc @ xyz
    x_new, y_new, z_new = jnp.unstack(xyz_new, axis=0)
    return x_new, y_new, z_new


# Astropy wrapper - handles astropy units and calls jax-compatible maths
@u.quantity_input
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
    # First we convert angles to radians, and strip units from all quantities
    mass = mass.to(u.Msun).value
    r0 = r0.to(u.au).value
    theta0 = theta0.to(u.rad).value
    phi0 = phi0.to(u.rad).value
    omega = omega.to(1/u.s).value
    v_r0 = v_r0.to(u.km/u.s).value
    inc = inc.to(u.rad).value
    pa = pa.to(u.rad).value
    deltar = deltar.to(u.au).value

    #rest of function
    rc = r_cent(mass=mass, omega=omega, r0=r0)
    #if rc > r0:
        #print('Centrifugal radius is larger than start of streamline')
    r = jnp.arange(r0, rc*0.5, step=-1*deltar)
    theta = stream_line(r, mass=mass, r0=r0, theta0=theta0,
                        omega=omega, v_r0=v_r0)
    d_phi = get_dphi(theta, theta0=theta0)
    phi = phi0 + d_phi
    #
    v_r, v_theta, v_phi = stream_line_vel(r, theta, mass=mass, r0=r0,
                                          theta0=theta0, omega=omega, v_r0=v_r0)
    v_x = v_r * jnp.sin(theta) * jnp.cos(phi) \
          + v_theta * jnp.cos(theta) * jnp.cos(phi) \
          - v_phi * jnp.sin(phi)
    v_y = v_r * jnp.sin(theta) * jnp.sin(phi) \
          + v_theta * jnp.cos(theta) * jnp.sin(phi) \
          + v_phi * jnp.cos(phi)
    v_z = v_r * jnp.cos(theta) \
          - v_theta * jnp.sin(theta)
    # Convert from spherical into cartesian coordinates
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    # Get mask from smallest radius for calculation
    if rmin is None:
        gd_rmin = jnp.ones_like(r, dtype=bool)
    else:
        gd_rmin = (r > rmin)
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
