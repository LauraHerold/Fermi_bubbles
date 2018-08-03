""" Plots fits file integrated over energy. To get the right units, GAPLPROP fits file has to be multiplied by E^2 """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
import auxil

##################################################################################################### parameters

source_class = True
gnomview = False

binmin = 6
binmax = 11

binmin = 12
binmax = 17

#binmin = 18
#binmax = 23

save_fn = '../../plots/Plots_9-year/Mollwiede_GALPROP_model_source_range2_log.pdf'

##################################################################################################### constants

smooth_map = True
mask_point_sources = True

scale_min = 0
scale_max = 1e-5

scale_min = -6.5
scale_max = -4

normalized = False
#unit = r'$I\ [\mathrm{GeV\ s^{-1}\ sr^{-1}\ cm^{-2}}]$'
unit = r'$\log_{10}(I\ [\mathrm{GeV\ s^{-1}\ sr^{-1}\ cm^{-2}}])$'

cmap = pyplot.cm.hot_r # jet, hot

GeV2MeV = 1000.
delta = 0.3837641821164575
smooth_sigma = np.deg2rad(.6)
npix = 196608
nside = healpy.npix2nside(npix)

##################################################################################################### load data from a fits file

if source_class:
    map_fn = '../../data/P8_P302_Source_z100_w009_w478/residuals/9years_Source_z100_refit_PS_resid_signal_bubbles_counts.fits'# Map is in unit counts
    data_fn = '../../data/P8_P302_Source_z100_w009_w478/maps/counts_P8_P302_Source_z100_w009_w478_healpix_o7_24bins.fits'
    expo_fn = '../../data/P8_P302_Source_z100_w009_w478/irfs/expcube_P8_P302_Source_z100_w009_w478_P8R2_SOURCE_V6_healpix_o7_24bins.fits'
    
else:
    map_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/residuals/9years_UltracleanVeto_z90_refit_PS_resid_signal_bubbles_counts.fits'               # Map is in unit counts
    expo_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/irfs/expcube_P8_P302_UltracleanVeto_z90_w009_w478_P8R2_ULTRACLEANVETO_V6_healpix_o7_24bins.fits'

mask_fn = '../../data/ps_masks/ps_mask_3FGL_OG_nside128.npy'     
    
hdu = pyfits.open(map_fn)
res = hdu[1].data.field('Spectra')
Es = hdu[2].data.field('MeV')/GeV2MeV
hdu_expo = pyfits.open(expo_fn)
exposure = hdu_expo[1].data.field('Spectra')
hdu_dat = pyfits.open(data_fn)
dat = hdu_dat[1].data.field('Spectra')

deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))
dOmega = 4. * np.pi / npix

if normalized:
    res /= (binmax - binmin + 1)


##################################################################################################### transpose the data matrix and find the mask

res = res.T
dat = dat.T
exposure = exposure.T
mod = dat - res

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)

##################################################################################################### sum over energy bins (from binmin to binmax)

print 'sum over energy bins...'

plot_map = np.zeros(npix)
for i in range(binmin, binmax+1):
    for j in range(len(plot_map)):
        plot_map[j] += Es[i] * mod[i][j] / exposure[i][j] / dOmega
       


##################################################################################################### smooth with smooth_sigma Gaussian

if smooth_map:
    plot_map = np.array(hlib.heal(plot_map, mask), dtype = np.float64)
    plot_map = healpy.smoothing(plot_map, sigma=smooth_sigma)

for pixel in xrange(npix):
    if mask[pixel] == 0:
        plot_map[pixel] = None

masked_array = np.ma.array(plot_map, mask=np.isnan(plot_map))
cmap.set_bad('grey', 1.)
  

##################################################################################################### show Mollweide view skymap

emin = Es[binmin] * np.exp(-delta/2)
emax = Es[binmax] * np.exp(delta/2)

title = r'$E = %.0f$' %emin + r'$ - %.0f\ \mathrm{GeV}$' %emax

auxil.setup_figure_pars(plot_type = 'map')

if gnomview:
    healpy.gnomview(plot_map, rot = ([0,0]), xsize = 3000, ysize = 1000, min =scale_min, max=scale_max, unit = unit, title = title, notext = True)
    
else:
    healpy.mollview(np.log10(plot_map), unit=unit, title = title,  min=scale_min, max=scale_max, cmap=cmap)

healpy.graticule(dpar=10., dmer=10.)

pyplot.savefig(save_fn)
