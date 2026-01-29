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
import jax.numpy as jnp

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
    v_coords = streamer_cube.spectral_axis.to(u.km/u.s).value - streamer_cube.header['CRVAL3'] # in km/s

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
    distance_metric, theta_ref = get_distance_metric(pc_coords[0], pc_coords[1], n_elements=n_elements)
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

        

def get_distance_metric(ra_coords, dec_coords, n_elements=10):
    '''
    Compute distance metric - used to bin the point cloud into n_elements
    This combines projected distance from star, and angular deviation from initial angle
    This is needed to get good sampling along a streamer which curves

    The theta used here is the angle in polar coordinates in the plane of the sky, wrt RA axis

    This distance metric is the same as in TIPSY
    '''
    pc_r, pc_theta = cartesian_to_polar(ra_coords, dec_coords)
    # get reference theta from close points
    r_percentile_threshold = 100 / n_elements
    r_thresh = jnp.percentile(pc_r, r_percentile_threshold)
    theta_ref = jnp.median(pc_theta[pc_r < r_thresh])
    # for the distance metric, use deviation from this theta_ref
    pc_theta2 = jnp.pi - jnp.abs(jnp.pi - jnp.abs(pc_theta - theta_ref))
    distance_metric = pc_r * jnp.sqrt(1+pc_theta2**2)

    return distance_metric, theta_ref


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


