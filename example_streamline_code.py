import os
import matplotlib.pyplot as plt
from velocity_tools import stream_lines
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5
from astropy.visualization.wcsaxes import WCSAxes
from astropy.io import fits
#
Per2_c = SkyCoord("3h32m17.92s", "+30d49m48.03s", frame='fk5')
Per2_ref = Per2_c.skyoffset_frame()

distance = 300 #parsecs
file_TdV = 'Per-emb-2-HC3N_10-9_TdV.fits'
# The file can be downloaded from:
if not os.path.isfile(file_TdV):
    import urllib.request
    link_file = 'https://github.com/jpinedaf/NOEMA_streamer_analysis/raw/eb3908e651e00dc9f110a8e82304222b83bb51fe/data/Per-emb-2-HC3N_10-9_TdV.fits'
    urllib.request.urlretrieve(link_file, filename=file_TdV)

# Create the streamline model
theta0 = 130.*u.deg
r0 = 0.9e4*u.au
phi0 = 365.*u.deg
v_r0 = 0*u.km/u.s
omega0 = 4e-13/u.s
v_lsr = 7.05*u.km/u.s
inc = -43*u.deg
PA_ang = 130*u.deg
Mstar = 3.2*u.Msun
# these are the results in astronomical units
(x1, y1, z1), (vx1, vy1, vz1) = stream_lines.xyz_stream(
                mass=Mstar, r0=r0, theta0=theta0, phi0=phi0,
                omega=omega0, v_r0=v_r0, inc=inc, pa=PA_ang,
                rmin=5.5e3*u.au, deltar=50*u.au) #<- decrease deltar for more accurate streamer calculation
dra_stream = -x1.value / distance * u.arcsec
ddec_stream = z1.value / distance * u.arcsec
fil = SkyCoord(dra_stream, ddec_stream, frame=Per2_ref).transform_to(FK5)
# Load the data
hdu = fits.open(file_TdV)[0]
wcs_TdV = WCS(hdu.header)
# create figure
plt.close('all')
fig = plt.figure(1, figsize=(6, 6))
ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs_TdV)
fig.add_axes(ax)  # note that the axes have to be explicitly added to the figure
im = ax.imshow(hdu.data, cmap='inferno',
               vmin=0, vmax=160.e-3)#, transform=ax.get_transform(wcs_TdV))
ax.scatter(Per2_c.ra, Per2_c.dec, marker='*', transform=ax.get_transform('world'),
    facecolor='white', edgecolor='black')
ax.plot(fil.ra, fil.dec,  color='black', transform=ax.get_transform('world'), linewidth=5)
ax.plot(fil.ra, fil.dec,  color='red', transform=ax.get_transform('world'), linewidth=2)
plt.show()
