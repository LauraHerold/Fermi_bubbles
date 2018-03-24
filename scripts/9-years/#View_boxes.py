""" Plots fits file integrated over energy. Since fits file saves data in differential flux one has to multiply by (deltaE/E) in order to get integral flux. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
import auxil

##################################################################################################### parameters

source_class = True
low_energy_range = 0                              # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)

gnomview = False

binmin = 0
binmax = 1

#binmin = 6
#binmax = 11

#binmin = 12
#binmax = 17


save_fn = '../../plots/Plots_9-year/Low_energy_range' + str(low_energy_range) +'/Mollweide_Boxes_source_range1.pdf'

##################################################################################################### constants

lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]
smooth_map = True
mask_point_sources = True
symmask = True

scale_min = -2.e-6
scale_max = 4.e-6
cmap = pyplot.cm.hot_r # jet, hot

normalized = False
unit = r'$\mathrm{GeV /(s\ sr\ cm}^2)}$'


GeV2MeV = 1000.
delta = 0.3837641821164575
smooth_sigma = np.deg2rad(.5)
npix = 196608
nside = healpy.npix2nside(int(npix))
dOmega = 4. * np.pi / npix


##################################################################################################### load data from a fits file

if source_class:
    boxes_fn = 'fits/test/Boxes_residual+boxes_0.3-1.0GeV_flux_ultraclean_test.fits'
    #boxes_fn = 'fits/Boxes_' + lowE_ranges[low_energy_range] + 'GeV_source.fits'
    data_fn = '../../data/P8_P302_Source_z100_w009_w478/maps/counts_P8_P302_Source_z100_w009_w478_healpix_o7_24bins.fits'
    expo_fn = '../../data/P8_P302_Source_z100_w009_w478/irfs/expcube_P8_P302_Source_z100_w009_w478_P8R2_SOURCE_V6_healpix_o7_24bins.fits' # Exposure
else:
    boxes_fn = 'fits/Boxes_' + lowE_ranges[low_energy_range] + 'GeV_ultraclean.fits'
    data_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/maps/counts_P8_P302_UltracleanVeto_z90_w009_w478_healpix_o7_24bins.fits'
    expo_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/irfs/expcube_P8_P302_UltracleanVeto_z90_w009_w478_P8R2_ULTRACLEANVETO_V6_healpix_o7_24bins.fits' # Exposure


mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'


hdu_boxes = pyfits.open(boxes_fn)
boxes = hdu_boxes[1].data.field('Spectra').T
Es = hdu_boxes[2].data.field('GeV')
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))

#hdu_expo = pyfits.open(expo_fn)
#expo = hdu_expo[1].data.field('Spectra').T[6:24]

#hdu = pyfits.open(data_fn)
#data = hdu[1].data.field('Spectra').T[6:24]


#deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))

#flux_data = [data[E] * Es[E]**2 / deltaE[E] /expo[E] /dOmega for E in range(18)]
#tot_map = boxes + flux_data

print 'boxes.shape: ', boxes.shape
print 'boxes: ', boxes

    
##################################################################################################### find the mask

mask = np.ones(int(npix))
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmask:
        mask *= mask[::-1]

##################################################################################################### sum over energy bins (from binmin to binmax)

print 'sum over energy bins...'

plot_map = np.zeros(npix)
for i in range(npix):
    for j in range(binmin, binmax + 1):
        plot_map[i] += boxes[j][i] * deltaE[j] / Es[j]                       # Integrated flux = differential flux * deltaE / E


##################################################################################################### smooth with smooth_sigma Gaussian

if smooth_map:
    plot_map = np.array(hlib.heal(plot_map, mask), dtype = np.float64)
    plot_map = healpy.smoothing(plot_map, sigma=smooth_sigma)

for pixel in xrange(int(npix)):
    if mask[pixel] == 0:
        plot_map[pixel] = float('nan')

masked_array = np.ma.array(plot_map, mask=np.isnan(plot_map))
cmap.set_bad('grey', 1.)

##################################################################################################### show Mollweide view skymap

emin = Es[binmin] * np.exp(-delta/2)
emax = Es[binmax] * np.exp(delta/2)

auxil.setup_figure_pars(plot_type = 'map')

#f = pyplot.gcf().get_children()
#CbAx = f[1]

#unit_text_obj = pyplot.gcf().get_children()[0]
#unit_text_obj.set_fontsize(20)


title = r'$E = %.0f$' %emin + r'$ - %.0f\ \mathrm{GeV}$' %emax

if gnomview:
    healpy.gnomview(plot_map, rot = ([0,0]), xsize = 3000, ysize = 1000, min =scale_min, max=scale_max, unit = unit, title = title, notext = True)
    
else:
    healpy.mollview(plot_map, unit=unit, title = title, min=scale_min, max=scale_max, cmap=cmap) #  

healpy.graticule(dpar=10., dmer=10.)

pyplot.savefig(save_fn)
