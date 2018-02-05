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
binmax = 5

#binmin = 6
#binmax = 11

#binmin = 12
#binmax = 17


save_fn = '../../plots/Plots_9-year/Low_energy_range' + str(low_energy_range) +'/Mollweide_LowE_0.3-1.0GeV_source_range1.pdf'

##################################################################################################### constants

lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]
smooth_map = True
mask_point_sources = True
symmask = True

scale_min = -2.e-6   #-10e-6
scale_max = 4.e-6   #10e-6
cmap = pyplot.cm.hot_r # jet, hot

normalized = False
unit = r'$\mathrm{GeV /(s\ sr\ cm}^2)}$'


GeV2MeV = 1000.
delta = 0.3837641821164575
smooth_sigma = np.deg2rad(.5)
npix = 196608
nside = healpy.npix2nside(npix)

##################################################################################################### load data from a fits file

if source_class:
    map_fn = 'fits/LowE_' + lowE_ranges[low_energy_range] + 'GeV_source.fits'
else:
    map_fn = 'fits/LowE_' + lowE_ranges[low_energy_range] + 'GeV_ultraclean.fits'

 
mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'


hdu = pyfits.open(map_fn)
data = hdu[1].data.field('Spectra')
Es = hdu[2].data.field('GeV')
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))


if normalized:
    data /= (binmax - binmin + 1)


##################################################################################################### transpose the data matrix and find the mask

data = data.T

    
##################################################################################################### find the mask

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmask:
        mask *= mask[::-1]

##################################################################################################### sum over energy bins (from binmin to binmax)

print 'sum over energy bins...'

plot_map = np.zeros(npix)
for i in range(binmin, binmax + 1):
    for j in range(len(plot_map)):
        plot_map[j] += data[i][j] * deltaE[i] / Es[i]                           # Integrated flux = differential flux * deltaE / E


##################################################################################################### smooth with smooth_sigma Gaussian

if smooth_map:
    plot_map = np.array(hlib.heal(plot_map, mask), dtype = np.float64)
    plot_map = healpy.smoothing(plot_map, sigma=smooth_sigma)

for pixel in xrange(npix):
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
    healpy.mollview((plot_map), unit=unit, title = title,  min=scale_min, max=scale_max, cmap=cmap)

healpy.graticule(dpar=10., dmer=10.)

pyplot.savefig(save_fn)
