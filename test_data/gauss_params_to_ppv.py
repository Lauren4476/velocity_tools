#!/usr/bin/env python3

# infile is a fits file of gauss params: amplitude, central velocity, sigma and their uncertainties, for each pixel in a data cube.
# outfile is a position-position-velocity cube, where each pixel has the gaussian spectrum.

import argparse
from datetime import datetime, timezone

import numpy as np
import astropy.units as u
from astropy.io import fits


SPEED_OF_LIGHT_KM_S = 299792.458
SPEED_OF_LIGHT_M_S = SPEED_OF_LIGHT_KM_S * 1000.0


def make_gaussian(amplitude, center_vel, sigma, velocity_axis):
    return amplitude * np.exp(-0.5 * ((velocity_axis - center_vel) / sigma) ** 2)

def get_velocity_unit(header):
    unit_string = header.get("CUNIT3", "km s-1")
    try:
        return u.Unit(unit_string)
    except Exception as exc:
        raise ValueError(
            f"Could not parse input spectral unit {unit_string!r}; pass a FITS cube with a valid CUNIT3 card."
        ) from exc


def infer_velocity_axis(center_vel, sigma, vref=None, dv=None, nsigma=10.0, samples_per_sigma=5.0):
    finite_center = np.asarray(center_vel, dtype=float)
    finite_sigma = np.asarray(sigma, dtype=float)

    if vref is None:
        vref = np.nanmedian(finite_center)
    if dv is None:
        dv = np.nanmedian(np.abs(finite_sigma)) / samples_per_sigma

    if not np.isfinite(vref):
        raise ValueError("Could not infer a reference velocity from the fit parameters; pass --vref.")
    if not np.isfinite(dv) or dv <= 0:
        raise ValueError("Could not infer a spectral channel spacing from the fit parameters; pass --dv.")

    span = np.nanmax(np.abs(finite_center - vref) + nsigma * np.abs(finite_sigma))
    if not np.isfinite(span) or span <= 0:
        span = dv

    nchan = int(np.ceil((2.0 * span) / dv)) + 1
    if nchan < 2:
        nchan = 2

    velocity_axis = vref + (np.arange(nchan) - (nchan - 1) / 2.0) * dv
    return velocity_axis, vref, dv


def build_output_header(header, shape, vref, dv, restfreq_hz=None, line=None, specsys=None, velref=None):
    nchan, ny, nx = shape
    out_header = header.copy()

    out_header["NAXIS"] = 3
    out_header["NAXIS1"] = nx
    out_header["NAXIS2"] = ny
    out_header["NAXIS3"] = nchan

    out_header["CTYPE3"] = "VELO-LSR"
    # get vel from input header if possible
    vel_unit = get_velocity_unit(header)
    if vel_unit.is_equivalent(u.km / u.s):
        out_header["CUNIT3"] = "km/s"
    elif vel_unit.is_equivalent(u.m / u.s):
        out_header["CUNIT3"] = "m/s"
    else:
        raise ValueError(f"Input spectral unit {vel_unit} is not compatible with velocity; pass a FITS cube with a valid CUNIT3 card.")
    out_header["CRPIX3"] = (nchan + 1) / 2.0
    out_header["CRVAL3"] = vref
    out_header["CDELT3"] = dv
    out_header["CROTA3"] = 0.0

    if restfreq_hz is not None:
        out_header["RESTFRQ"] = restfreq_hz
        out_header["RESTFREQ"] = restfreq_hz
        out_header["ALTRPIX"] = out_header["CRPIX3"]
        if out_header["CDELT3"] != 0:
            if out_header["CUNIT3"] == "km/s":
                out_header["ALTRDEL"] = restfreq_hz * (dv / SPEED_OF_LIGHT_KM_S)
                out_header["ALTRVAL"] = restfreq_hz * (1.0 - (vref / SPEED_OF_LIGHT_KM_S))
            elif out_header["CUNIT3"] == "m/s":
                out_header["ALTRDEL"] = restfreq_hz * (dv / SPEED_OF_LIGHT_M_S)
                out_header["ALTRVAL"] = restfreq_hz * (1.0 - (vref / SPEED_OF_LIGHT_M_S))
            else:
                raise ValueError(f"Output spectral unit {out_header['CUNIT3']} is not compatible with velocity; this should not happen.")

    if line is not None:
        out_header["LINE"] = line
    if specsys is not None:
        out_header["SPECSYS"] = specsys
    if velref is not None:
        out_header["VELREF"] = velref

    out_header["WCSAXES"] = 3
    out_header["DATE"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_header.setdefault("ORIGIN", "velocity_tools")
    return out_header


def gauss_params_to_ppv(infile, outfile, vref=None, dv=None, nsigma=10.0, samples_per_sigma=5.0,
                        restfreq=None, line=None, specsys=None, velref=257):
    with fits.open(infile) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    if data.ndim != 3:
        raise ValueError("Input data must have shape (n_params, ny, nx)")
    n_params, ny, nx = data.shape
    if n_params < 3:
        raise ValueError("Input data must have at least 3 parameters (amplitude, velocity, sigma)")
    amplitude = data[0]
    centre_vel = data[1]
    sigma = data[2]
    velocity_axis, vref, dv = infer_velocity_axis(centre_vel, sigma, vref=vref, dv=dv, nsigma=nsigma, samples_per_sigma=samples_per_sigma)

    gaussians_cube = np.full((velocity_axis.size, ny, nx), np.nan, dtype=float)
    for y in range(ny):
        for x in range(nx):
            amp = amplitude[y, x]
            cen_vel = centre_vel[y, x]
            sig = sigma[y, x]
            if not np.isfinite(amp) or not np.isfinite(cen_vel) or not np.isfinite(sig) or sig <= 0:
                continue
            gaussians_cube[:, y, x] = make_gaussian(amp, cen_vel, sig, velocity_axis)


    if restfreq is None:
        restfreq = header.get("RESTFRQ", header.get("RESTFREQ"))

    out_header = build_output_header(header, gaussians_cube.shape, vref, dv, restfreq_hz=restfreq, line=line or header.get("LINE"), specsys=specsys or header.get("SPECSYS", "LSRK"), velref=velref)

    hdu = fits.PrimaryHDU(gaussians_cube, header=out_header)
    hdu.writeto(outfile, overwrite=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a FITS cube of Gaussian fit parameters into a PPV cube."
    )
    parser.add_argument("infile", help="Input FITS file containing Gaussian fit parameters")
    parser.add_argument("outfile", help="Output FITS file for the PPV cube")
    parser.add_argument("--vref", type=float, default=None, help="Reference velocity in km/s")
    parser.add_argument("--dv", type=float, default=None, help="Channel spacing in km/s")
    parser.add_argument("--nsigma", type=float, default=10.0, help="Velocity span in units of sigma")
    parser.add_argument(
        "--samples-per-sigma",
        type=float,
        default=5.0,
        help="Number of channels per fitted sigma when dv is inferred",
    )
    parser.add_argument("--restfreq", type=float, default=None, help="Line rest frequency in Hz")
    parser.add_argument("--line", type=str, default=None, help="Line name to store in the output header")
    parser.add_argument("--specsys", type=str, default=None, help="Spectral reference frame for the output header")
    parser.add_argument("--velref", type=int, default=257, help="Legacy VELREF code for the spectral axis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gauss_params_to_ppv(
        args.infile,
        args.outfile,
        vref=args.vref,
        dv=args.dv,
        nsigma=args.nsigma,
        samples_per_sigma=args.samples_per_sigma,
        restfreq=args.restfreq,
        line=args.line,
        specsys=args.specsys,
        velref=args.velref,
    )