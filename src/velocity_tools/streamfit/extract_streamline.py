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
from astropy.coordinates import SkyCoord
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
def _wrap_to_pi(angle):
    """Wrap angles to [-pi, pi)."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

@jax.jit
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


def _wrap_to_pi_numpy(angle):
    '''Wrap angles to [-pi, pi).'''
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def reduce_to_1D(streamer_cube, yso_centre, n_elements=10):
    '''
    This function will reduce a cube of emission to a 1D 'streamline', 
    by weighted means.

    In this function, 'pc' is short for point cloud.

    Inspired by method in TIPSY 

    Parameters
    ----------
    streamer_cube : SpectralCube object, should contain only streamer emission
    yso_centre : SkyCoord, the coordinates of the star, used to compute RA and Dec offsets in arcsec
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
    yso_centre_icrs = yso_centre.icrs

    # create coordinate arrays for RA/Dec offsets in arcsec without assuming input celestial units
    # Use spherical offsets to robustly handle RA wrapping and high-latitude geometry.
    y_indices, x_indices = np.mgrid[0:ny, 0:nx]
    world_coords = streamer_cube.wcs.celestial.pixel_to_world_values(x_indices.ravel(), y_indices.ravel())
    ra_unit = u.Unit(streamer_cube.header.get('CUNIT1', streamer_cube.wcs.celestial.world_axis_units[0]))
    dec_unit = u.Unit(streamer_cube.header.get('CUNIT2', streamer_cube.wcs.celestial.world_axis_units[1]))
    world_sky = SkyCoord(
        ra=world_coords[0] * ra_unit,
        dec=world_coords[1] * dec_unit,
        frame='icrs'
    )
    dra, ddec = yso_centre_icrs.spherical_offsets_to(world_sky)
    ra_coords = dra.to(u.arcsec).value.reshape(ny, nx)
    dec_coords = ddec.to(u.arcsec).value.reshape(ny, nx)

    # create velocity array relative to the reference channel, then express it in km/s
    spectral_unit = u.Unit(streamer_cube.header.get('CUNIT3', streamer_cube.spectral_axis.unit))
    spectral_axis = streamer_cube.spectral_axis.to(spectral_unit)
    v_coords = spectral_axis.to(u.km / u.s).value

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
    """
    Compute radial + angular distance metric for point cloud binning.
    Uses a circular angular deviation to avoid branch-cut artifacts.
    """
    pc_r, pc_theta = cartesian_to_polar(ra_coords, dec_coords)

    finite_mask = jnp.isfinite(pc_r) & jnp.isfinite(pc_theta)
    finite_r = pc_r[finite_mask]
    finite_theta = pc_theta[finite_mask]

    # No valid points
    if finite_r.size == 0:
        if return_trace:
            return pc_r, {
                "n_points": int(pc_r.size),
                "n_finite_points": 0,
                "n_reference_points": 0,
                "r_percentile_thresh": float("nan"),
                "r_thresh": float("nan"),
                "theta_ref": 0.0,
                "theta_weight": 1.0,
                "close_point_count": 0,
            }
        return pc_r

    theta_weight = 1.0

    # Get reference angle from points within a radius percentile threshold
    n_ref = min(10, max(1, int(finite_r.size)))
    percentile = 100.0 / n_ref
    r_thresh = jnp.percentile(finite_r, percentile)

    close_theta = finite_theta[finite_r <= r_thresh]
    ref_theta_source = close_theta if close_theta.size else finite_theta
    theta_ref = _circular_median(ref_theta_source)

    # Cyclic angular deviation
    theta_dev = jnp.pi - jnp.abs(
        jnp.pi - jnp.abs(_wrap_to_pi(pc_theta - theta_ref))
    )

    distance_metric = pc_r * jnp.sqrt(1.0 + (theta_weight * theta_dev) ** 2)
    distance_metric = jnp.where(finite_mask, distance_metric, jnp.inf)

    if return_trace:
        return distance_metric, {
            "n_points": int(pc_r.size),
            "n_finite_points": int(finite_r.size),
            "n_reference_points": n_ref,
            "r_percentile_thresh": percentile,
            "r_thresh": float(r_thresh),
            "theta_ref": float(theta_ref),
            "theta_weight": theta_weight,
            "close_point_count": int(close_theta.size),
        }

    return distance_metric

@jax.jit
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


def get_metric_partitions(pc_coords, n_elements):
    '''
    Compute percentile partitions for the streamline distance metric.

    Parameters
    ----------
    pc_coords : array
        Point cloud coordinates. Index 0 = RA, Index 1 = Dec, Index 2 = velocity.
    n_elements : int
        Number of partitions to compute.

    Returns
    -------
    partitions : ndarray
        Percentile boundaries of the distance metric.
    '''
    if n_elements < 1:
        raise ValueError('n_elements must be >= 1')

    ra_coords = pc_coords[0]
    dec_coords = pc_coords[1]
    distance_metric = np.asarray(get_distance_metric(ra_coords, dec_coords))
    finite_mask = np.isfinite(distance_metric)
    finite_metric = distance_metric[finite_mask]

    if finite_metric.size == 0:
        return np.full(n_elements + 1, np.nan, dtype=np.float64)

    b_per = np.linspace(0.0, 100.0, n_elements + 1)
    return np.asarray([np.percentile(finite_metric, per) for per in b_per], dtype=np.float64)


def _get_metric_reference_trace(pc_coords):
    '''Return the metric reference angle and weight used for boundary sampling.'''
    ra_coords = pc_coords[0]
    dec_coords = pc_coords[1]
    _, trace = get_distance_metric(ra_coords, dec_coords, return_trace=True)
    theta_ref = float(trace.get('theta_ref', 0.0))
    theta_weight = float(trace.get('theta_weight', 1.0))
    return theta_ref, theta_weight


def sample_metric_boundary(partition_radius, theta_ref, theta_weight=1.0, n_samples=720):
    '''Sample a constant-metric boundary as a closed RA/Dec curve.'''
    if n_samples < 4:
        raise ValueError('n_samples must be >= 4')

    theta = jnp.linspace(-jnp.pi, jnp.pi, n_samples, endpoint=False)
    theta_dev = jnp.pi - jnp.abs(jnp.pi - jnp.abs(_wrap_to_pi(theta - theta_ref)))
    radius = partition_radius / jnp.sqrt(1.0 + (theta_weight * theta_dev) ** 2)
    ra = radius * jnp.cos(theta)
    dec = radius * jnp.sin(theta)
    return ra, dec


def sample_metric_boundaries(pc_coords, partitions, n_samples=720):
    '''Sample all metric boundary curves for a point cloud and partition set.'''
    theta_ref, theta_weight = _get_metric_reference_trace(pc_coords)
    curves = [
        sample_metric_boundary(partition_radius, theta_ref, theta_weight=theta_weight, n_samples=n_samples)
        for partition_radius in np.asarray(partitions)
    ]
    trace = {
        'theta_ref': theta_ref,
        'theta_weight': theta_weight,
    }
    return curves, trace


def plot_metric_boundaries(
    ax,
    pc_coords,
    partitions,
    color='lightgrey',
    linewidth=1,
    alpha=0.5,
    n_samples=720,
    zorder=1,
):
    '''Plot metric boundary curves on an RA/Dec axis.'''
    curves, trace = sample_metric_boundaries(pc_coords, partitions, n_samples=n_samples)
    for ra, dec in curves:
        ax.plot(ra, dec, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    return curves, trace


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


