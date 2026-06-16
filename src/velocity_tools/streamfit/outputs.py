'''
This file contains functions related to outputs from streamfit optimisation,
such as saving logs and plotting results.

Last updated: 03-06-26
'''
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os


from . import gradient_descent
from . import extract_streamline


def _ensure_clean_dir(path):
    """Create directory if missing and remove any files inside it.

    Keeps behaviour consistent across plotting functions that write epoch frames.
    """
    os.makedirs(path, exist_ok=True)
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass


def _create_video_from_images(output_dir, input_pattern, output_name, fps=5):
    """Call ffmpeg to make a video from numbered image frames.

    This function is tolerant of missing ffmpeg and surfaces a readable message.
    """
    import subprocess

    output_video = os.path.join(output_dir, output_name)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf",
        "setpts='PTS/(1+0.01*N)',pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        output_video,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create the video.")


def plot_loss(loss_history, save_folder=None):
    '''Plot loss as a function of epochs'''
    # plot loss vs epoch nicely
    # matplotlib serif font
    plt.rcParams['font.family'] = 'serif'
    # Plot loss history
    # Epoch indexing: epoch 0 = initial state, epoch i (i >= 1) = after update i
    # loss_history is 0-indexed: loss_history[i] = loss at epoch i
    plt.figure(figsize=(12, 3))
    epochs = range(len(loss_history))
    plt.plot(epochs, loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimisation Progress')
    plt.yscale('log')
    plt.grid(True, alpha=0.5)
    if save_folder is not None:
        plt.savefig(f'{save_folder}/loss_history.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def find_spikes(loss, threshold=0.1):
    '''Identify epochs where the loss increases by more than a certain percentage compared to adjacent epochs.'''
    spikes = []
    for i in range(1, len(loss) - 1):
        if loss[i] > loss[i - 1] * (1 + threshold) and loss[i] > loss[i + 1] * (1 + threshold):
            spikes.append(i)
    return spikes

def plot_morphology_by_epoch(
    optimisation_log,
    param_names,
    gradient_descent,
    fixed_params,
    distance,
    ra_data=None,
    dec_data=None,
    ra_sigma=None,
    dec_sigma=None,
    pc_coords=None,
    n_points=None,
    output_dir="streamfit_test_output/morphology_epochs",
    make_video=False
):
    """
    Create and save one streamline morphology plot per optimisation epoch, in output_dir
    """

    epochs = optimisation_log['epoch'].values

    # create the models
    epoch_models = []
    for idx, epoch in enumerate(epochs):

        row = optimisation_log.iloc[idx]
        opt_params_epoch = {param: float(row[param]) for param in param_names}
        opt_params_epoch_full, opt_params_epoch, fixed_params = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(opt_params_epoch_full, distance)
        valid_mask_model = valid_mask_model.astype(bool)

        (ra_model_interp, dec_model_interp, _, valid, model_keep, dmetric_model, matching_trace) = gradient_descent.checked_match_model_to_data_curve(
            ra_model,
            dec_model,
            v_model,
            valid_mask_model,
            ra_data,
            dec_data,
        )

        epoch_models.append(
            dict(
                epoch=epoch,
                opt_params_epoch=opt_params_epoch,
                ra_model=ra_model,
                dec_model=dec_model,
                ra_model_interp=ra_model_interp,
                dec_model_interp=dec_model_interp,
                valid=valid,
                model_keep=model_keep,
            )
        )

    # get constant axis limits from the whole collection of epochs
    all_ra = []
    all_dec = []

    for epoch_data in epoch_models:
        all_ra.extend(epoch_data['ra_model'])
        all_dec.extend(epoch_data['dec_model'])

    # include observational data
    all_ra.extend(ra_data)
    all_dec.extend(dec_data)
    # include point cloud
    all_ra.extend(pc_coords[0])
    all_dec.extend(pc_coords[1])

    all_ra = jnp.array(all_ra)
    all_dec = jnp.array(all_dec)
    mask = ~jnp.isnan(all_ra) & ~jnp.isnan(all_dec)
    all_ra = all_ra[mask]
    all_dec = all_dec[mask]

    pad_ra = 0.05 * (all_ra.max() - all_ra.min())
    pad_dec = 0.05 * (all_dec.max() - all_dec.min())

    ra_lim = (all_ra.min() - pad_ra, all_ra.max() + pad_ra)
    dec_lim = (all_dec.min() - pad_dec, all_dec.max() + pad_dec)

    # prepare clean output folder for epoch frames
    _ensure_clean_dir(output_dir)

    partitions = extract_streamline.get_metric_partitions(pc_coords, n_points)
    metric_boundaries, trace = extract_streamline.sample_metric_boundaries(pc_coords, partitions)

    # plot and save for each epoch
    for model in epoch_models:
        plot_morphology(
            ra_model=model["ra_model"],
            dec_model=model["dec_model"],
            ra_data=ra_data,
            dec_data=dec_data,
            ra_sigma=ra_sigma,
            dec_sigma=dec_sigma,
            ra_model_interp=model["ra_model_interp"],
            dec_model_interp=model["dec_model_interp"],
            valid=model["valid"],
            pc_coords=pc_coords,
            metric_boundaries=metric_boundaries,
            title=f"Epoch: {int(model['epoch'])}",
            xlim=ra_lim,
            ylim=dec_lim,
            save_folder=output_dir,
            save_name=f"morphology_epoch_{int(model['epoch']):03d}",
            show=False
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "morphology_epoch_%03d.png")
        _create_video_from_images(output_dir, input_pattern, "streamline_morphology_evolution.mp4", fps=5)

def plot_morphology(
    ra_model=None,
    dec_model=None,
    ra_data=None,
    dec_data=None,
    ra_sigma=None,
    dec_sigma=None,
    ra_model_interp=None,
    dec_model_interp=None,
    valid=None,
    by_eye=None,
    pc_coords=None,
    metric_boundaries=None,
    title=None,
    xlim=None,
    ylim=None,
    legend_loc='lower right',
    save_folder=None,
    save_name='streamline_morphology',
    show=True,
):
    '''Plot offsets in RA/Dec. Optionally include: model, model points, data points, best fit, background overlay, metric partitions.'''


    fig, ax = plt.subplots(figsize=(6.5, 7))
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)

    # model curve if given
    if ra_model is not None and dec_model is not None:
        ax.plot(ra_model, dec_model, color='blue', linewidth=2, label='Best-fit', zorder=7)

    # model points if given
    if ra_model_interp is not None and dec_model_interp is not None and valid is not None:
        if valid is not None:
            ax.scatter(
                np.asarray(ra_model_interp, dtype=float)[valid],
                np.asarray(dec_model_interp, dtype=float)[valid],
                s=25,
                color='blue',
                zorder=7,
            )

    # by eye model if given
    if by_eye is not None:
        ra_by_eye, dec_by_eye, _ = by_eye
        ax.plot(
            np.asarray(ra_by_eye, dtype=float),
            np.asarray(dec_by_eye, dtype=float),
            color='tab:green',
            linewidth=2,
            label='By-eye',
            zorder=8,
        )

    # data points streamline if given
    if ra_data is not None and dec_data is not None:
        ra_data = np.asarray(ra_data, dtype=float)
        dec_data = np.asarray(dec_data, dtype=float)
        if ra_sigma is not None and dec_sigma is not None:
            ax.errorbar(
                ra_data,
                dec_data,
                xerr=np.asarray(ra_sigma, dtype=float),
                yerr=np.asarray(dec_sigma, dtype=float),
                fmt='o-',
                label='Extracted 1D Streamline',
                color='red',
                zorder=5,
            )
        else:
            ax.plot(
                ra_data,
                dec_data,
                'o-',
                label='Extracted 1D Streamline',
                color='red',
                zorder=5,
            )
        if valid is not None:
            ax.scatter(
                ra_data[valid], dec_data[valid],
                s=45, facecolor='none', edgecolor='cyan', linewidth=1.2, zorder=6,
                label='Retained data',
            )

    ax.scatter(0, 0, marker='*', s=100, color='yellow', edgecolor='black', zorder=10)
    ax.set_xlabel('RA Offset (arcsec)')
    ax.set_ylabel('Dec Offset (arcsec)')

    if pc_coords is not None:
        pc_coords = np.asarray(pc_coords, dtype=float)
        ax.scatter(
            pc_coords[0],
            pc_coords[1],
            s=1,
            color='gray',
            alpha=0.3,
            label='Point cloud',
            zorder=4,
        )

        if metric_boundaries is not None:
            # get current axes limits
            ax_limits = ax.get_xlim(), ax.get_ylim()
            # ax.set_facecolor('lightgrey')

            # for partition_radius in partitions:
            #     circle = plt.Circle((0, 0), partition_radius, facecolor='white', edgecolor='none', zorder=1)
            #     ax.add_patch(circle)

            extract_streamline.plot_metric_boundaries(
                ax,
                pc_coords,
                metric_boundaries,
                color='gray',
                linewidth=1,
                alpha=0.3,
            )
            # restore axes limits
            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.invert_xaxis()
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    if save_folder is not None:
        plt.savefig(f'{save_folder}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    elif show is not False:
        plt.show()

def plot_ra_vel_by_epoch(
    optimisation_log,
    param_names,
    gradient_descent,
    fixed_params,
    distance,
    ra_data=None,
    dec_data=None,
    v_data=None,
    ra_sigma=None,
    v_sigma=None,
    pc_coords=None,
    output_dir="streamfit_test_output/ra_vel_epochs",
    make_video=False,
):
    """
    Create RA–velocity plots for every epoch
    """

    epochs = optimisation_log['epoch'].values

    epoch_models = []

    # make models
    for idx, epoch in enumerate(epochs):

        row = optimisation_log.iloc[idx]
        opt_params_epoch = {p: float(row[p]) for p in param_names}
        opt_params_epoch_full, _, _ = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(opt_params_epoch_full, distance)
        valid_mask_model = valid_mask_model.astype(bool)
        ra_model_interp, _, v_model_interp, valid, model_keep, dmetric_model, matching_trace = (
            gradient_descent.checked_match_model_to_data_curve(ra_model, dec_model, v_model, valid_mask_model, ra_data, dec_data)
        )

        epoch_models.append({
            "epoch": epoch,
            "ra_model": ra_model,
            "v_model": v_model,
            "ra_model_interp": ra_model_interp,
            "v_model_interp": v_model_interp,
            "valid": valid,
            "model_keep": model_keep,
        })

    # global velocity limits
    v_list = [m["v_model"] for m in epoch_models]
    if v_data is not None:
        v_list.append(v_data)
    all_v = np.concatenate(v_list)
    vlim = (np.nanmin(all_v), np.nanmax(all_v))

    # global RA limits
    ra_list = [m["ra_model"] for m in epoch_models]
    if ra_data is not None:
        ra_list.append(ra_data)
    all_ra = np.concatenate(ra_list)
    ralim = (np.nanmin(all_ra), np.nanmax(all_ra))

    # make clean output folder
    _ensure_clean_dir(output_dir)


    # make the plots
    for model in epoch_models:
        plot_ra_vel(
            ra_model=model["ra_model"],
            v_model=model["v_model"],
            ra_data=ra_data,
            v_data=v_data,
            ra_sigma=ra_sigma,
            v_sigma=v_sigma,
            ra_model_interp=model["ra_model_interp"],
            v_model_interp=model["v_model_interp"],
            valid=model["valid"],
            model_keep=model["model_keep"],
            pc_coords=pc_coords,
            title=f"Epoch: {int(model['epoch'])}",
            vlim=vlim,
            ralim=ralim,
            save_path=os.path.join(
                output_dir,
                f"ra_vel_epoch_{int(model['epoch']):03d}.png"
            ),
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "ra_vel_epoch_%03d.png")
        _create_video_from_images(output_dir, input_pattern, "streamline_ra_vel_evolution.mp4", fps=5)
        

def plot_ra_vel(
    ra_model,
    v_model,
    *,
    ra_data=None,
    v_data=None,
    ra_sigma=None,
    v_sigma=None,
    ra_model_interp=None,
    v_model_interp=None,
    valid=None,
    model_keep=None,
    pc_coords=None,
    title=None,
    vlim=None,
    ralim=None,
    legend_loc='lower right',
    save_path=None,
    show=False,
):
    ra_model = np.asarray(ra_model, dtype=float)
    v_model = np.asarray(v_model, dtype=float)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
    if model_keep is not None:
        model_keep = np.asarray(model_keep, dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 5))
    

    # point cloud (RA vs velocity)
    if pc_coords is not None:
        ax.scatter(pc_coords[0], pc_coords[2], s=1, alpha=0.3, color='grey', label='Point cloud')

    # data
    if ra_data is not None and v_data is not None:
        if ra_sigma is not None and v_sigma is not None:
            ax.errorbar(ra_data, v_data, xerr=ra_sigma, yerr=v_sigma, fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data')
        else:
            ax.plot(ra_data, v_data, 'o', color='red', label='Data')

    # model curve
    if ra_model is not None and v_model is not None:
        ax.plot(ra_model, v_model, color='blue', linewidth=2, label='Model Streamline', zorder=7)

    # interpolated points
    if ra_model_interp is not None and v_model_interp is not None and valid is not None:
        ax.scatter(np.asarray(ra_model_interp)[valid], np.asarray(v_model_interp)[valid], s=25, color='blue', zorder=5, label='Model at data positions')

    # selected data points
    if ra_data is not None and v_data is not None and valid is not None:
        ax.scatter(np.asarray(ra_data)[valid], np.asarray(v_data)[valid], s=45, facecolor='none', edgecolor='cyan', linewidth=1.2, zorder=6, label='Retained data')

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title(title or "RA vs Velocity")

    if vlim is not None:
        ax.set_ylim(vlim)
    if ralim is not None:
        ax.set_xlim(ralim)

    # flip RA axis to match astronomical convention
    ax.invert_xaxis()

    ax.legend(loc=legend_loc)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


#########
def plot_dec_vel_by_epoch(
    optimisation_log,
    param_names,
    gradient_descent,
    fixed_params,
    distance,
    ra_data=None,
    dec_data=None,
    v_data=None,
    dec_sigma=None,
    v_sigma=None,
    pc_coords=None,
    output_dir="streamfit_test_output/dec_vel_epochs",
    make_video=False,
):
    """
    Create DEC–velocity plots for every epoch
    """

    epochs = optimisation_log['epoch'].values

    epoch_models = []

    # make models
    for idx, epoch in enumerate(epochs):

        row = optimisation_log.iloc[idx]
        opt_params_epoch = {p: float(row[p]) for p in param_names}
        opt_params_epoch_full, _, _ = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(opt_params_epoch_full, distance)
        valid_mask_model = valid_mask_model.astype(bool)
        ra_model_interp, dec_model_interp, v_model_interp, valid, model_keep, dmetric_model, matching_trace = (
            gradient_descent.checked_match_model_to_data_curve(ra_model, dec_model, v_model, valid_mask_model, ra_data, dec_data)
        )

        epoch_models.append({
            "epoch": epoch,
            "ra_model": ra_model,
            "dec_model": dec_model,
            "v_model": v_model,
            "ra_model_interp": ra_model_interp,
            "dec_model_interp": dec_model_interp,
            "v_model_interp": v_model_interp,
            "valid": valid,
            "model_keep": model_keep,
        })

    # global velocity limits
    v_list = [m["v_model"] for m in epoch_models]
    if v_data is not None:
        v_list.append(v_data)
    all_v = np.concatenate(v_list)
    vlim = (np.nanmin(all_v), np.nanmax(all_v))

    # global dec limits
    dec_list = [m["dec_model"] for m in epoch_models]
    if dec_data is not None:
        dec_list.append(dec_data)
    all_dec = np.concatenate(dec_list)
    declim = (np.nanmin(all_dec), np.nanmax(all_dec))

    # make clean output folder
    _ensure_clean_dir(output_dir)


    # make the plots
    for model in epoch_models:
        plot_dec_vel(
            dec_model=model["dec_model"],
            v_model=model["v_model"],
            dec_data=dec_data,
            v_data=v_data,
            dec_sigma=dec_sigma,
            v_sigma=v_sigma,
            dec_model_interp=model["dec_model_interp"],
            v_model_interp=model["v_model_interp"],
            valid=model["valid"],
            model_keep=model["model_keep"],
            pc_coords=pc_coords,
            title=f"Epoch: {int(model['epoch'])}",
            vlim=vlim,
            declim=declim,
            save_path=os.path.join(
                output_dir,
                f"dec_vel_epoch_{int(model['epoch']):03d}.png"
            ),
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "dec_vel_epoch_%03d.png")
        _create_video_from_images(output_dir, input_pattern, "streamline_dec_vel_evolution.mp4", fps=5)
        

def plot_dec_vel(
    dec_model,
    v_model,
    *,
    dec_data=None,
    v_data=None,
    dec_sigma=None,
    v_sigma=None,
    dec_model_interp=None,
    v_model_interp=None,
    valid=None,
    model_keep=None,
    pc_coords=None,
    title=None,
    vlim=None,
    declim=None,
    legend_loc='lower right',
    save_path=None,
    show=False,
):
    dec_model = np.asarray(dec_model, dtype=float)
    v_model = np.asarray(v_model, dtype=float)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
    if model_keep is not None:
        model_keep = np.asarray(model_keep, dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 5))

    # point cloud (DEC vs velocity)
    if pc_coords is not None:
        # pc_coords layout: [ra, dec, velocity]
        ax.scatter(pc_coords[1], pc_coords[2], s=1, alpha=0.3, color='grey', label='Point cloud')

    # data
    if dec_data is not None and v_data is not None:
        if dec_sigma is not None and v_sigma is not None:
            ax.errorbar(dec_data, v_data, xerr=dec_sigma, yerr=v_sigma, fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data')
        else:
            ax.plot(dec_data, v_data, 'o', color='red', label='Data')

    # model curve
    if dec_model is not None and v_model is not None:
        ax.plot(dec_model, v_model, color='blue', linewidth=2, label='Model Streamline')

    # interpolated points
    if dec_model_interp is not None and v_model_interp is not None and valid is not None:
        ax.scatter(np.asarray(dec_model_interp)[valid], np.asarray(v_model_interp)[valid], s=25, color='blue', zorder=5, label='Model at data positions')

    # selected data points
    if dec_data is not None and v_data is not None and valid is not None:
        ax.scatter(np.asarray(dec_data)[valid], np.asarray(v_data)[valid], s=45, facecolor='none', edgecolor='cyan', linewidth=1.2, zorder=6, label='Retained data')

    ax.set_xlabel("DEC Offset (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title(title or "DEC vs Velocity")

    if vlim is not None:
        ax.set_ylim(vlim)
    if declim is not None:
        ax.set_xlim(declim)

    ax.legend(loc=legend_loc)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def build_velocity_radius_kde(
    ra_data,
    dec_data,
    vlos_data,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    grid_size=100,
    sigma_levels=None,
):
    """Build a KDE background grid for projected radius vs velocity plots.

    Parameters
    ----------
    rproj_data : array-like
        Projected radial distance samples (arcsec).
    vlos_data : array-like
        Line-of-sight velocity samples (km/s).
    xmin, xmax, ymin, ymax : float, optional
        Plot limits for KDE grid. If omitted, finite data limits are used.
    grid_size : int, optional
        Number of grid points per axis.
    sigma_levels : array-like, optional
        Sigma values used to build cumulative Gaussian-like contour levels.

    Returns
    -------
    dict
        Dictionary with keys: "xx", "yy", "zz", "levels", "xlim", "ylim".
    """
    from scipy import stats
    ra = np.asarray(ra_data)
    dec = np.asarray(dec_data)
    rproj = np.sqrt(ra**2 + dec**2)
    vlos = np.asarray(vlos_data, dtype=float)
    finite = np.isfinite(rproj) & np.isfinite(vlos)
    if np.sum(finite) < 3:
        raise ValueError("Need at least 3 finite samples to build KDE background.")

    rproj = rproj[finite]
    vlos = vlos[finite]

    if xmin is None:
        xmin = float(np.nanmin(rproj) - 1)
    if xmax is None:
        xmax = float(np.nanmax(rproj) + 1)
    if ymin is None:
        ymin = float(np.nanmin(vlos) - 1)
    if ymax is None:
        ymax = float(np.nanmax(vlos) + 1)

    xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([rproj, vlos])

    kernel = stats.gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)
    zmax = np.nanmax(zz)
    if np.isfinite(zmax) and zmax > 0:
        zz = zz / zmax

    if sigma_levels is None:
        sigma_levels = np.arange(1.0, 2.1, 0.5)
    sigma_levels = np.asarray(sigma_levels, dtype=float)
    levels = np.append(np.exp(-0.5 * sigma_levels**2)[::-1], [1.0])

    return {
        "xx": xx,
        "yy": yy,
        "zz": zz,
        "levels": levels,
        "xlim": (xmin, xmax),
        "ylim": (ymin, ymax),
    }


def plot_vel_radius(
    ra_model,
    dec_model,
    v_model,
    *,
    ra_data=None,
    dec_data=None,
    v_data=None,
    ra_sigma=None,
    dec_sigma=None,
    v_sigma=None,
    ra_model_interp=None,
    dec_model_interp=None,
    v_model_interp=None,
    valid=None,
    model_keep=None,
    kde_background=None,
    velocity_reference=None,
    title=None,
    xlim=None,
    ylim=None,
    legend_loc='lower right',
    save_path=None,
    show=False,
):
    """Plot velocity vs projected radius for one model (optionally with KDE background)."""
    ra_model = np.asarray(ra_model, dtype=float)
    dec_model = np.asarray(dec_model, dtype=float)
    v_model = np.asarray(v_model, dtype=float)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
    if model_keep is not None:
        model_keep = np.asarray(model_keep, dtype=bool)
        ra_model = ra_model[model_keep]
        dec_model = dec_model[model_keep]
        v_model = v_model[model_keep]

    rproj_model = np.sqrt(ra_model**2 + dec_model**2)
    order_model = np.argsort(rproj_model)

    fig, ax = plt.subplots(figsize=(6.5 * 1.3, 4 * 1.3))

    if kde_background is not None:
        ax.contourf(
            kde_background["xx"],
            kde_background["yy"],
            kde_background["zz"],
            levels=kde_background["levels"],
            cmap='Greys',
            vmin=0,
            vmax=1.2,
            zorder=1,
        )

    # Central source marker in this projection (r=0, v=v_lsr)
    if velocity_reference is not None:
        ax.scatter(
            0,
            float(velocity_reference),
            marker='*',
            s=100,
            color='yellow',
            edgecolor='black',
            zorder=10,
            label='Central Source',
        )

    data_handle = None
    if ra_data is not None and dec_data is not None and v_data is not None:
        ra_data = np.asarray(ra_data, dtype=float)
        dec_data = np.asarray(dec_data, dtype=float)
        v_data = np.asarray(v_data, dtype=float)
        rproj_data = np.sqrt(ra_data**2 + dec_data**2)

        if ra_sigma is not None and dec_sigma is not None and v_sigma is not None:
            ra_sigma = np.asarray(ra_sigma, dtype=float)
            dec_sigma = np.asarray(dec_sigma, dtype=float)
            v_sigma = np.asarray(v_sigma, dtype=float)
            denom = np.maximum(rproj_data, 1e-8)
            rproj_sigma = np.sqrt((ra_data * ra_sigma) ** 2 + (dec_data * dec_sigma) ** 2) / denom
            data_handle = ax.errorbar(
                rproj_data,
                v_data,
                xerr=rproj_sigma,
                yerr=v_sigma,
                fmt='o',
                color='red',
                ecolor='red',
                ms=4,
                alpha=0.9,
                label='Extracted 1D Streamline',
                zorder=6,
            )
        else:
            data_handle = ax.plot(
                rproj_data,
                v_data,
                'o',
                color='red',
                label='Extracted 1D Streamline',
                zorder=6,
            )[0]

    model_handle, = ax.plot(
        rproj_model[order_model],
        v_model[order_model],
        color='blue',
        linewidth=2,
        label='Model Streamline',
        zorder=7,
    )

    if (
        ra_model_interp is not None
        and dec_model_interp is not None
        and v_model_interp is not None
        and valid is not None
    ):
        ra_model_interp = np.asarray(ra_model_interp, dtype=float)
        dec_model_interp = np.asarray(dec_model_interp, dtype=float)
        v_model_interp = np.asarray(v_model_interp, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        rproj_interp = np.sqrt(ra_model_interp**2 + dec_model_interp**2)
        ax.scatter(
            rproj_interp[valid],
            v_model_interp[valid],
            s=25,
            color='blue',
            label='Model at retained data arc lengths',
            zorder=8,
        )

    if velocity_reference is not None:
        ax.axhline(
            float(velocity_reference),
            color='black',
            linestyle='--',
            label='Systemic Velocity',
            zorder=4,
        )

    ax.set_xlabel('Projected Distance from Source (arcsec)')
    ax.set_ylabel('Velocity (km/s)')
    ax.set_title(title or 'Velocity vs Projected Radius')

    if xlim is not None:
        ax.set_xlim(xlim)
    elif kde_background is not None:
        ax.set_xlim(kde_background["xlim"])

    if ylim is not None:
        ax.set_ylim(ylim)
    elif kde_background is not None:
        ax.set_ylim(kde_background["ylim"])

    if data_handle is not None:
        ax.legend(handles=[data_handle, model_handle], loc=legend_loc)
    else:
        ax.legend(loc=legend_loc)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def plot_vel_radius_by_epoch(
    optimisation_log,
    param_names,
    gradient_descent,
    fixed_params,
    distance,
    *,
    ra_data=None,
    dec_data=None,
    v_data=None,
    ra_sigma=None,
    dec_sigma=None,
    v_sigma=None,
    kde_background=None,
    velocity_reference=None,
    output_dir="streamfit_test_output/vel_radius_epochs",
    make_video=False,
):
    """Create velocity vs projected radius plots for every epoch."""
    epochs = optimisation_log['epoch'].values
    epoch_models = []

    for idx, epoch in enumerate(epochs):
        row = optimisation_log.iloc[idx]
        opt_params_epoch = {p: float(row[p]) for p in param_names}
        opt_params_epoch_full, _, _ = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(
            opt_params_epoch_full,
            distance,
        )

        valid_mask_model = valid_mask_model.astype(bool)

        ra_model_interp, dec_model_interp, v_model_interp, valid, model_keep, dmetric_model, matching_trace = (
            gradient_descent.checked_match_model_to_data_curve(
                ra_model,
                dec_model,
                v_model,
                valid_mask_model,
                ra_data,
                dec_data,
            )
        )

        if model_keep is not None:
            model_keep = model_keep.astype(bool)

        epoch_models.append({
            "epoch": epoch,
            "ra_model": ra_model,
            "dec_model": dec_model,
            "v_model": v_model,
            "ra_model_interp": ra_model_interp,
            "dec_model_interp": dec_model_interp,
            "v_model_interp": v_model_interp,
            "valid": valid,
            "model_keep": model_keep,
        })

    # Set consistent axis limits across epochs
    rproj_list = []
    v_list = [np.asarray(m["v_model"], dtype=float) for m in epoch_models]
    for model in epoch_models:
        ra_m = np.asarray(model["ra_model"], dtype=float)
        dec_m = np.asarray(model["dec_model"], dtype=float)
        rproj_list.append(np.sqrt(ra_m**2 + dec_m**2))

    if ra_data is not None and dec_data is not None:
        rproj_list.append(np.sqrt(np.asarray(ra_data, dtype=float) ** 2 + np.asarray(dec_data, dtype=float) ** 2))
    if v_data is not None:
        v_list.append(np.asarray(v_data, dtype=float))

    all_rproj = np.concatenate(rproj_list)
    all_v = np.concatenate(v_list)
    xlim = (np.nanmin(all_rproj), np.nanmax(all_rproj))
    ylim = (np.nanmin(all_v), np.nanmax(all_v))

    _ensure_clean_dir(output_dir)

    for model in epoch_models:
        plot_vel_radius(
            ra_model=model["ra_model"],
            dec_model=model["dec_model"],
            v_model=model["v_model"],
            ra_data=ra_data,
            dec_data=dec_data,
            v_data=v_data,
            ra_sigma=ra_sigma,
            dec_sigma=dec_sigma,
            v_sigma=v_sigma,
            ra_model_interp=model["ra_model_interp"],
            dec_model_interp=model["dec_model_interp"],
            v_model_interp=model["v_model_interp"],
            valid=model["valid"],
            model_keep=model["model_keep"],
            kde_background=kde_background,
            velocity_reference=velocity_reference,
            title=f"Epoch: {int(model['epoch'])}",
            xlim=xlim,
            ylim=ylim,
            save_path=os.path.join(
                output_dir,
                f"vel_radius_epoch_{int(model['epoch']):03d}.png",
            ),
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "vel_radius_epoch_%03d.png")
        _create_video_from_images(
            output_dir,
            input_pattern,
            "streamline_vel_radius_evolution.mp4",
            fps=5,
        )

    return epoch_models

def plot_param_uncertainties(opt_keys, opt_params, opt_sigmas, save_folder=None):
    eps = 1e-12
    norm_errs = np.abs(opt_sigmas / (opt_params + eps))
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ypos = np.arange(len(opt_keys))
    ax.barh(
        ypos,
        norm_errs,
        color='tab:blue',
        alpha=0.8
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(opt_keys)
    ax.set_xlabel('Relative uncertainty ($\\sigma / |x|$)')
    ax.set_title('Normalized Parameter Uncertainties')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(f'{save_folder}/parameter_uncertainties.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_param_correlations(param_names, covariance, annotate=True, save_folder=None):
    '''
    Plot a parameter correlation matrix derived from the covariance matrix, as a heatmpa.
    
    Parameters:
    ----------
    param_names: list of str
        Names of the parameters, in the same order as the covariance matrix.
    covariance: 2D array
        Covariance matrix of the parameters.
    annotate: bool, optional
        Whether to annotate the heatmap with correlation values.
    '''
    cov_np = np.array(covariance, dtype=float)

    diag = np.sqrt(np.clip(np.diag(cov_np), 1e-30, None))
    corr = cov_np / np.outer(diag, diag)
    corr = np.clip(corr, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm_r')

    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(param_names, fontsize=11)
    ax.set_title('Parameter Correlation Matrix')

    # Create colorbar axis with matched height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Correlation coefficient')

    if annotate:
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                ax.text(
                    j, i,
                    f'{corr[i, j]:.2f}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(f'{save_folder}/parameter_correlation_matrix.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_streamline_covariance_samples(streamline_samples,
                                      best_opt_params,
                                      fixed_params,
                                      distance,
                                      data,
                                      uncertainties,
                                      velocity_reference=None,
                                      save_folder=None):
    """
    Plot streamline uncertainty from precomputed streamline samples.

    Parameters
    ----------
    streamlines : list of dict
        Output from evaluate_streamline_samples().
    best_params : dict
        Best-fit optimised parameters.
    fixed_params : dict
        Fixed model parameters.
    distance : float
        Source distance in pc.
    data : tuple
        (ra_data, dec_data, v_data)
    uncertainties : tuple
        (ra_sigma, dec_sigma, v_sigma)
    velocity_reference : float, optional
        Draw a horizontal reference line on the velocity panel.
    """
    fig, (ax_sky, ax_v) = plt.subplots(1, 2, figsize=(10, 5))
    ra_data, dec_data, v_data = data
    ra_sigma, dec_sigma, v_sigma = uncertainties

    # plot samples streamlines
    for streamline in streamline_samples:
        ra = streamline['ra']
        dec = streamline['dec']
        vel = streamline['v']

        rproj = np.sqrt(ra**2 + dec**2)
        order = np.argsort(rproj)
        ax_sky.plot(ra, dec, color='tab:blue', alpha=0.1, lw=1)
        ax_v.plot(rproj[order], vel[order], color='tab:blue', alpha=0.1, lw=1)

    # plot best fit streamline
    best_opt_full_params, best_opt_params, fixed_params = gradient_descent.prepare_model_params(best_opt_params, fixed_params)
    ra_best, dec_best, v_best, valid_mask_best, err = gradient_descent.forward_model(best_opt_full_params, distance)
    ra_best = np.asarray(ra_best, dtype=float)
    dec_best = np.asarray(dec_best, dtype=float)
    v_best = np.asarray(v_best, dtype=float)
    valid_mask_best = valid_mask_best.astype(bool)
    ra_best = ra_best[valid_mask_best]
    dec_best = dec_best[valid_mask_best]
    v_best = v_best[valid_mask_best]
    rproj_best = np.sqrt(ra_best**2 + dec_best**2)
    order_best = np.argsort(rproj_best)
    ax_sky.plot(ra_best, dec_best, color='blue', lw=2, label='Best-fit')
    ax_v.plot(rproj_best[order_best], v_best[order_best], color='blue', lw=2, label='Best-fit')

    # plot data
    ax_sky.errorbar(
        ra_data, dec_data, xerr=ra_sigma, yerr=dec_sigma,
        fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data'
        )
    rproj_data = np.sqrt(ra_data**2 + dec_data**2)
    # get errors in rproj_data
    rproj_sigma = np.sqrt((ra_data * ra_sigma)**2 + (dec_data * dec_sigma)**2) / rproj_data
    order_data = np.argsort(rproj_data)
    ax_v.errorbar(
        rproj_data[order_data], np.asarray(v_data)[order_data], yerr=np.asarray(v_sigma)[order_data], xerr=np.asarray(rproj_sigma)[order_data],
        fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data'
        )
    if velocity_reference is not None:
        xmin, xmax = ax_v.get_xlim()
        ax_v.hlines(velocity_reference, xmin=xmin, xmax=xmax, colors='k', linestyles='--', alpha=0.6,)
        ax_v.set_xlim(xmin, xmax)

    # finalise plots
    ax_sky.invert_xaxis()
    ax_sky.set_xlabel('RA Offset (arcsec)')
    ax_sky.set_ylabel('Dec Offset (arcsec)')
    ax_sky.set_title('Covariance Sampling')
    ax_sky.legend()

    ax_v.set_xlabel('Projected distance(arcsec)')
    ax_v.set_ylabel('Velocity (km/s)')
    ax_v.set_title('Covariance Sampling')
    ax_v.legend()

    plt.tight_layout()

    if save_folder is not None:
        plt.savefig(f'{save_folder}/streamline_covariance_samples.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def evaluate_streamlines_samples(param_samples, opt_keys, fixed_params, distance):
    """
    Evaluate streamline models for sampled parameter vectors
    Returns
    -------
    streamlines : list of dict
        Each entry contains:
        {
            "ra": ...,
            "dec": ...,
            "v": ...,
            "dmetric": ...
        }
    """
    streamlines = []
    for sample in param_samples:
        sample_params = {
            key: float(value)
            for key, value in zip(opt_keys, sample)
        }
        sample_params_full, sample_params, fixed_params = gradient_descent.prepare_model_params(sample_params, fixed_params)
        ra, dec, vel, valid_mask, err = gradient_descent.forward_model(sample_params_full, distance)
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        vel = np.asarray(vel, dtype=float)
        valid_mask = valid_mask.astype(bool)
        ra = ra[valid_mask]
        dec = dec[valid_mask]
        vel = vel[valid_mask]

        dmetric, trace = extract_streamline.get_distance_metric(ra, dec)
        dmetric = np.asarray(dmetric, dtype=float)

        streamlines.append(
            {
                "ra": ra,
                "dec": dec,
                "v": vel,
                "dmetric": dmetric,
            }
        )

    return streamlines

def sample_parameter_sets_from_covariance(best_params, covariance, opt_keys, param_bounds=None, n_samples=100, seed=42):
    """
    Draw parameter samples from a covariance matrix.
    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
        Sampled parameter vectors
    """
    rng = np.random.default_rng(seed)
    mu = np.array(
        [best_params[key] for key in opt_keys],
        dtype=float,
    )
    cov = np.asarray(covariance, dtype=float)
    samples = rng.multivariate_normal(
        mu,
        cov,
        size=n_samples,
    )
    if param_bounds is not None:
        param_bounds = gradient_descent.convert_and_strip_bound_units(param_bounds)
        for j, key in enumerate(opt_keys):
            if key in param_bounds:
                low, high = param_bounds[key]
                samples[:, j] = np.clip(samples[:, j], low, high)

    return samples

def plot_param_optimisation_history(
    optimisation_log,
    trace_log,
    trace_component_cols,
    spikes,
    save_folder=None
):
    epochs = optimisation_log["epoch"].values
    loss = optimisation_log["loss"].values

    param_names = []
    for c in optimisation_log.columns:
        if c not in ("epoch", "loss"):
            param_names.append(c)

    fig, axes = plt.subplots(len(param_names) + 2, 1, figsize=(8, 3 * (len(param_names) + 1)), sharex=True)

    plot_loss_panel(axes[0], epochs, loss, spikes)

    for col in trace_component_cols:
        axes[1].plot(epochs, trace_log[col].values, label=col)
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(True)

    for ax, param in zip(axes[2:], param_names):
        values = optimisation_log[param].values
        ax.plot(epochs, values)
        ax.scatter(epochs[spikes], values[spikes], color="orange")
        ax.set_ylabel(param)
        ax.grid(True)

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(f'{save_folder}/parameter_optimisation_history.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_trace_diagnostics(optimisation_log, trace_log, spikes, save_folder=None):
    epochs = optimisation_log["epoch"].values
    loss = optimisation_log["loss"].values

    trace_cols = []
    for c in trace_log.columns:
        if c not in ("epoch", "loss"):
            trace_cols.append(c)

    fig, axes = plt.subplots(len(trace_cols) + 1, 1, figsize=(10, 2.5 * (len(trace_cols) + 1)), sharex=True)

    plot_loss_panel(axes[0], epochs, loss, spikes)

    spike_epochs = epochs[spikes]

    for ax, col in zip(axes[1:], trace_cols):
        ax.plot(trace_log["epoch"], trace_log[col])
        mask = (trace_log["epoch"].astype(int).isin(spike_epochs))
        ax.scatter(trace_log.loc[mask, "epoch"], trace_log.loc[mask, col], color="orange")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(f'{save_folder}/trace_diagnostics.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_loss_panel(ax, epochs, loss, spikes):
    lowest_loss = np.min(loss)
    best_idx = np.argmin(loss)
    best_epoch = epochs[best_idx]
    ax.plot(epochs, loss, color="black")
    ax.scatter(best_epoch, lowest_loss, color="green", label=f"Best Epoch: {best_epoch}")
    ax.scatter(epochs[spikes], loss[spikes], color="orange", label="Spikes")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

def detect_trace_loss_method(trace_log):
    if trace_log is None:
        return None, []
    if {"chi2_ra", "chi2_dec", "chi2_v"}.issubset(trace_log.columns):
        return "radecvel", ["chi2_ra", "chi2_dec", "chi2_v"]
    if {"chi2_r", "chi2_theta", "chi2_v"}.issubset(trace_log.columns):
        return "rthetavel", ["chi2_r", "chi2_theta", "chi2_v"]
    return "unknown", []

def load_optimisation_logs(logs_dir):
    log_path = os.path.join(logs_dir, "optimisation_log.csv")
    optimisation_log = pd.read_csv(log_path)
    # trace only exists if trace_every != None
    trace_path = os.path.join(logs_dir,"optimisation_trace.csv")
    if os.path.exists(trace_path):
        trace_log = pd.read_csv(trace_path)
        print(f"Loaded tracer log: {trace_path} "
              f"({len(trace_log)} rows)"
        )
    else:
        trace_log = None
        print(
            f"Tracer log not found at {trace_path}. "
            f"Re-run with trace_every != None to save tracer log. "
        )
    return optimisation_log, trace_log


