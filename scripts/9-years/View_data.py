""" 
Plots fits file integrated over energy. Since fits file saves data in differential flux one has to multiply by (deltaE/E) in order to get integral flux. 
python View_data.py
"""

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
import auxil

##################################################################################################### Parameters



log_plot = True
if log_plot:
    scale_max = [-3.5, -4, -4.5]
    scale_min = [-6, -6.5, -7]
else:
    scale_max = [5.e-5, 3.e-5, 1.e-5]
    scale_min = np.zeros(3)

save_plots = 0
show_plots = 1

##################################################################################################### Constants

highE_ranges = ((6,11),(12,17),(18,23))        # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)


smooth_map = True
mask_point_sources = False
symmask = False
cmap = pyplot.cm.hot_r # jet, hot
cmap = None

normalized = False
if log_plot:
    unit = r'${\rm log_{10}} (I [\mathrm{GeV\ s^{-1}\ sr^{-1}\ cm^{-2}}])$'
else:
    unit = r'$I [\mathrm{GeV\ s^{-1}\ sr^{-1}\ cm^{-2}]}$'


GeV2MeV = 1000.
delta = 0.3837641821164575
smooth_sigmas = np.deg2rad([0., 0.5, 0.7]) #0.5
nside = 128
npix = healpy.nside2npix(nside)

map_fn = '../../data/P8_P302_Source_z100_w009_w478/maps/counts_P8_P302_Source_z100_w009_w478_healpix_o7_24bins.fits'
expo_fn = '../../data/P8_P302_Source_z100_w009_w478/irfs/expcube_P8_P302_Source_z100_w009_w478_P8R2_SOURCE_V6_healpix_o7_24bins.fits'
mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'


##################################################################################################### Read map and mask

hdu = pyfits.open(map_fn)
counts = hdu[1].data.field('Spectra') # Shape (npix, nE)
Es = hdu[2].data.field('MeV') / GeV2MeV
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))
hdu_expo = pyfits.open(expo_fn)
exposure = hdu_expo[1].data.field('Spectra')

deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))
dOmega = 4. * np.pi / npix

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmask:
        mask *= mask[::-1]

flux = Es**2 * counts / (deltaE * exposure * dOmega)
flux = flux.T

##################################################################################################### Loop over 3 high-energy ranges

for highE in (0,1,2):
    print "Range ", highE
    binmin = highE_ranges[highE][0]
    binmax = highE_ranges[highE][1]

    plot_map = np.zeros(npix)
    for i in range(binmin, binmax + 1):
        plot_map += flux[i] * deltaE[i] / Es[i] # integrate SED

    if smooth_map and smooth_sigmas[highE] > 0.:
        if mask_point_sources:
            plot_map = np.array(hlib.heal(plot_map, mask), dtype = np.float64)
        plot_map = healpy.smoothing(plot_map, sigma=smooth_sigmas[highE])

    if mask_point_sources:
        for pixel in xrange(npix):
            if mask[pixel] == 0:
                plot_map[pixel] = float('nan')

        masked_array = np.ma.array(plot_map, mask=np.isnan(plot_map))
        cmap.set_bad('grey', 1.)


    emin = Es[binmin] * np.exp(-delta/2)
    emax = Es[binmax] * np.exp(delta/2)

    auxil.setup_figure_pars(plot_type = 'map')
    pyplot.rcParams['figure.figsize'] = [6,4]
    pyplot.rcParams['axes.labelsize'] = 30 # 30
    pyplot.rcParams['axes.titlesize'] = 20 # 20
    pyplot.rcParams['xtick.labelsize'] = 15 # 15
    print pyplot.rcParams['figure.figsize']
    
    title = r'$E = %.0f$' %emin + r'$ - %.0f\ \mathrm{GeV}$' %emax
    
    if 0:
        pyplot.figure()
        healpy.gnomview(plot_map, rot = ([0,0]), xsize = 3000, ysize = 1000, min =scale_min, max=scale_max[highE], unit = unit, title = title, notext = True)

        healpy.graticule(dpar=10., dmer=10.)
        save_fn = '../../plots/Plots_9-year/Gnomview_data_source_range_'+ str(highE) + '.pdf'
        pyplot.savefig(save_fn)

    print pyplot.rcParams['figure.figsize']



    #pyplot.figure()
    if log_plot:
        plot_map = np.maximum(plot_map, 10**scale_min[highE])
        plot_map = np.log10(plot_map)
    healpy.mollview(plot_map, unit=unit, title=title,  min=scale_min[highE], max=scale_max[highE],
                         cmap=cmap, xsize=1500)
    #print ax
    pyplot.rcParams['figure.figsize'] = [12,8]
    pyplot.rcParams['axes.labelsize'] = 30
    pyplot.rcParams['axes.titlesize'] = 20
    pyplot.rcParams['xtick.labelsize'] = 15
    print pyplot.rcParams['figure.figsize']
    #pyplot.colorbar().set_label(label=unit, size=30, weight='bold')
    healpy.graticule(dpar=10., dmer=10.)
    save_fn = '../../plots/Plots_9-year/Mollweide_data_source_range_'+ str(highE) + '.pdf'


    if save_plots:
        print 'save figure to file:'
        print save_fn
        pyplot.savefig(save_fn)
if show_plots:
    pyplot.show()

