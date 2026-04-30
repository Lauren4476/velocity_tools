'''
This file contains functions to extract streamline emission from a data cube,
and extract a 1D streamline from that.

It will perform multigaussian fits, decide the best number of fits,
use umap and dbscan for clustering,
and produce a final data cube containing just the streamer emission.

Then it will extract a streamline from this cube.
'''

import numpy as np
from astropy import units as u
from collections import namedtuple
import jax.numpy as jnp


PreparedData = namedtuple('PreparedData', [
    'ra_data', 'dec_data', 'v_data',
    'ra_sigma_safe', 'dec_sigma_safe', 'v_sigma_safe',
    'dmetric_data', 'data_finite_mask',
    'data_min', 'data_max',
    'r_proj_data', 'theta_proj_data',
])


def _wrap_to_pi(angle):
    """Wrap angles to [-pi, pi)."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _circular_median(theta_vals):
    """
    Compute a branch-cut-safe median angle.

    Angles are first unwrapped around a circular-mean anchor, then a linear
    median is taken in that unwrapped frame, and wrapped back to [-pi, pi).
    """
    theta_anchor = jnp.arctan2(jnp.mean(jnp.sin(theta_vals)), jnp.mean(jnp.cos(theta_vals)))
    theta_delta = _wrap_to_pi(theta_vals - theta_anchor)
    theta_unwrapped = theta_anchor + theta_delta
    theta_ref = jnp.median(theta_unwrapped)
    return _wrap_to_pi(theta_ref)

def reduce_to_1D(streamer_cube, n_elements=10):
    '''
    This function will reduce a cube of emission to a 1D 'streamline', 
    by weighted means.

    In this function, 'pc' is short for point cloud.

    Inspired by method in TIPSY 

    Parameters
    ----------
    streamer_cube : SpectralCube object, should contain only streamer emission
    n_elements : int, number of elements to reduce the cube to


    Returns
    -------
    pc_means : array of shape (3, n_elements), the weighted mean coordinates of each bin
    index 0 = RA offsets (arcsec)
    index 1 = Dec offsets (arcsec)
    index 2 = velocity (km/s)
    '''
    print('Starting reduction')
    nz, ny, nx = streamer_cube.shape

    # create coordinate arrays for RA and Dec in arcsec
    y_indices, x_indices = np.mgrid[0:ny, 0:nx]
    world_coords = streamer_cube.wcs.celestial.pixel_to_world_values(x_indices.ravel(), y_indices.ravel())
    ra_coords = (world_coords[0].reshape(ny, nx) - streamer_cube.header['CRVAL1']) * 60 * 60
    ra_coords = ra_coords * np.cos(streamer_cube.header['CRVAL2'] * np.pi / 180) # cos(dec) correct for declination. in arcsec
    dec_coords = (world_coords[1].reshape(ny, nx) - streamer_cube.header['CRVAL2']) * 60 * 60 # in arcsec

    # create velocity array in km/s
    #TODO: fix this to use spectral_axis and WCS instead of header keywords, to be more robust
    v_coords = streamer_cube.spectral_axis.to(u.km/u.s).value - (streamer_cube.header['CRVAL3']*1e-3)

    print('Created coordinate arrays')

    # get data and mask
    pcloud = np.array(streamer_cube)
    rms_mask = ~np.isnan(pcloud)
    flux = pcloud[rms_mask]

    # get indices of valid points in pc
    pc_indices = np.indices(pcloud.shape) # indices of all points in pc
    pc_z = pc_indices[0][rms_mask] # z indices of points in pc
    pc_y = pc_indices[1][rms_mask] # y indices of points in pc
    pc_x = pc_indices[2][rms_mask] # x indices of points in pc

    print('Got point cloud with', len(flux), 'points')

    # extract coordinates of valid points using the arrays above
    pc_ra = ra_coords[pc_y, pc_x]
    pc_dec = dec_coords[pc_y, pc_x]
    pc_v = v_coords[pc_z]
    pc_coords = np.array([pc_ra, pc_dec, pc_v]) # shape (3, n_points)   

    # compute distance metric to bin the point cloud
    distance_metric = get_distance_metric(pc_coords[0], pc_coords[1])
    b_per = np.linspace(0, 100, n_elements+1) # percentiles to bin the pc into
    partitions = np.array([np.percentile(distance_metric, per) for per in b_per])

    print("Partition boundaries for projected distance metric:", np.round(partitions, 3))

    # take flux-weighted means and stds in each bin
    pc_means = np.zeros((3, n_elements))
    pc_stds = np.zeros((3, n_elements))
    for i in range(n_elements):
        # identify points in this bin, add weighted means and weighted stds
        distance_indices = (distance_metric > partitions[i]) & (distance_metric <= partitions[i+1])
        pc_means[:, i] = np.average(pc_coords.T[distance_indices],
                                 axis=0,
                                 weights=flux[distance_indices])
        pc_stds[:, i] = np.sqrt(np.average((pc_coords.T[distance_indices] - pc_means[:, i])**2,
                                         axis=0,
                                         weights=flux[distance_indices]))
        
    # flip arrays so that they go from large to small distance (towards star)
    pc_means = pc_means[:, ::-1]
    pc_stds = pc_stds[:, ::-1]
    
    return pc_coords, pc_means, pc_stds

        

def get_distance_metric(ra_coords, dec_coords, return_trace=False):
    '''
    Compute distance metric - used to bin the point cloud into n_elements
    Basically just distance on the plane of the sky

    (used to include polar angle as well)

    Parameters
    ----------
    ra_coords : array
        RA offsets.
    dec_coords : array
        Dec offsets.
    n_elements : int
        Number of elements used in distance partitioning.
    return_trace : bool
        If True, return extra diagnostics useful for debugging metric instability.

    Returns
    -------
    distance_metric, theta_ref
        Default return values.
    distance_metric, theta_ref, trace_dict
        Returned when return_trace=True.
    '''
    pc_r, pc_theta = cartesian_to_polar(ra_coords, dec_coords)
    distance_metric = pc_r

    if return_trace:
        trace = {
            'n_points': int(pc_r.size),
        }
        return distance_metric, trace

    return distance_metric


def cartesian_to_polar(x, y):
    '''
    Convert cartesian coordinates (x,y) to polar coordinates
    e.g. inputs could be RA and Dec offsets

    Parameters
    ----------
    x : array of x coordinates
    y : array of y coordinates

    Returns
    -------
    r : array of radial distances
    theta : array of angles in radians
    '''
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x) # angle wrt x-axis, in radians

    return (r, theta)


def prepare_data(data, uncertainties):
    '''
    Precompute all data-only quantities used by streamfit loss evaluation.

    Parameters
    ----------
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data

    Returns
    -------
    PreparedData
        Container with precomputed data-only quantities.
    '''
    ra_data = jnp.asarray(data[0], dtype=jnp.float64)
    dec_data = jnp.asarray(data[1], dtype=jnp.float64)
    v_data = jnp.asarray(data[2], dtype=jnp.float64)

    ra_sigma = jnp.asarray(uncertainties[0], dtype=jnp.float64)
    dec_sigma = jnp.asarray(uncertainties[1], dtype=jnp.float64)
    v_sigma = jnp.asarray(uncertainties[2], dtype=jnp.float64)

    eps = jnp.asarray(1e-8, dtype=jnp.float64)
    ra_sigma_safe = jnp.maximum(ra_sigma, eps)
    dec_sigma_safe = jnp.maximum(dec_sigma, eps)
    v_sigma_safe = jnp.maximum(v_sigma, eps)

    dmetric_data = get_distance_metric(ra_data, dec_data)
    data_finite_mask = jnp.isfinite(ra_data) & jnp.isfinite(dec_data) & jnp.isfinite(dmetric_data)

    data_metric_for_min = jnp.where(data_finite_mask, dmetric_data, jnp.inf)
    data_metric_for_max = jnp.where(data_finite_mask, dmetric_data, -jnp.inf)
    data_min = jnp.min(data_metric_for_min)
    data_max = jnp.max(data_metric_for_max)

    r_proj_data, theta_proj_data = cartesian_to_polar(ra_data, dec_data)

    return PreparedData(
        ra_data=ra_data,
        dec_data=dec_data,
        v_data=v_data,
        ra_sigma_safe=ra_sigma_safe,
        dec_sigma_safe=dec_sigma_safe,
        v_sigma_safe=v_sigma_safe,
        dmetric_data=dmetric_data,
        data_finite_mask=data_finite_mask,
        data_min=data_min,
        data_max=data_max,
        r_proj_data=r_proj_data,
        theta_proj_data=theta_proj_data,
    )


