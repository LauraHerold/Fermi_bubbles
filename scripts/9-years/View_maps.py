""" Plots fits file integrated over energy. Since fits file saves data in differential flux one has to multiply by (deltaE/E) in order to get integral flux. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
import auxil
from optparse import OptionParser

##################################################################################################### Parameters


parser = OptionParser()
parser.add_option("-f", "--file_name", dest = "map_fn", default = '', help ='file name') 
parser.add_option("-n", "--scale_min", dest="scale_min", default=" -2.e-6", help="Minimal value in Mollweide map")
parser.add_option("-x", "--scale_max", dest="scale_max", default=" 4.e-6", help="Maximal value in Mollweide map")
(options, args) = parser.parse_args()

map_fn = str(options.map_fn)
scale_min = float(options.scale_min)
scale_max = float(options.scale_max)


##################################################################################################### Constants

highE_ranges = ((0,5),(6,11),(12,17))        # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)

log_scale = False

smooth_map = True
mask_point_sources = True
symmask = True
cmap = pyplot.cm.hot_r # jet, hot

normalized = False
unit = r'$I\ [\mathrm{GeV\ s^{-1}\ sr^{-1}\ cm^{-2}}]}$'
if log_scale:
    unit = r'$\log_{10}(I\ [\mathrm{GeV\ s^{-1}\ sr^{-1}\ cm^{-2}}])$'



GeV2MeV = 1000.
delta = 0.3837641821164575
smooth_sigma = np.deg2rad(.5)
npix = 196608
nside = healpy.npix2nside(npix)

 
mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'


##################################################################################################### Read map and mask

hdu = pyfits.open("fits/" + map_fn)
data = hdu[1].data.field('Spectra').T # Shape (nE, npix)
Es = hdu[2].data.field('GeV')
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))

if (highE_ranges[2][1] - highE_ranges[0][0] + 1) > len(data):
    highE_ranges = ((0,3),(4,9),(10,15))
    print "Changed high-energy ranges."

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmask:
        mask *= mask[::-1]

##################################################################################################### Loop over 3 high-energy ranges

for highE in [0,1,2]:
    print "Range ", highE
    binmin = highE_ranges[highE][0]
    binmax = highE_ranges[highE][1]

    plot_map = np.zeros(npix)
    for i in range(binmin, binmax + 1):
        plot_map += data[i] * deltaE[i] / Es[i]
        #for j in range(len(plot_map)):
        #    plot_map[j] += data[i][j] * deltaE[i] / Es[i]                           # Integrated flux = differential flux * deltaE / E

    if smooth_map:
        plot_map = np.array(hlib.heal(plot_map, mask), dtype = np.float64)
        plot_map = healpy.smoothing(plot_map, sigma=smooth_sigma)

    for pixel in xrange(npix):
        if mask[pixel] == 0:
            plot_map[pixel] = float('nan')

    masked_array = np.ma.array(plot_map, mask=np.isnan(plot_map))
    cmap.set_bad('grey', 1.)


    emin = Es[binmin] * np.exp(-delta/2)
    emax = Es[binmax] * np.exp(delta/2)

    #auxil.setup_figure_pars(plot_type = 'map')
    pyplot.rcParams['figure.figsize'] = [1.5,1]
    pyplot.rcParams['axes.labelsize'] = 30
    pyplot.rcParams['axes.titlesize'] = 20
    pyplot.rcParams['xtick.labelsize'] = 15
    print pyplot.rcParams['figure.figsize']
    
    title = r'$E = %.0f$' %emin + r'$ - %.0f\ \mathrm{GeV}$' %emax

    #pyplot.figure()
    #healpy.gnomview(plot_map, rot = ([0,0]), xsize = 3000, ysize = 1000, min =scale_min, max=scale_max, unit = unit, title = title, notext = True)

    #healpy.graticule(dpar=10., dmer=10.)
    #save_fn = '../../plots/Plots_9-year/Gnomview_' + map_fn[:-5] + '_range_'+ str(highE) + '.pdf'
    #pyplot.savefig(save_fn)

    #print pyplot.rcParams['figure.figsize']

    pyplot.figure()
    if log_scale:
        ax = healpy.mollview(np.log10(plot_map), unit=unit, title = title,  min=scale_min, max=scale_max, cmap=cmap, xsize = 1500)
    else:
        ax = healpy.mollview((plot_map), unit=unit, title = title,  min=scale_min, max=scale_max, cmap=cmap, xsize = 1500)
   
    print ax
    #pyplot.colorbar().set_label(label=unit, size=30, weight='bold')
    healpy.graticule(dpar=10., dmer=10.)
    save_fn = '../../plots/Plots_9-year/Mollweide_' + map_fn[:-5] + '_range_'+ str(highE) + '.pdf'
    if log_scale:
        save_fn = '../../plots/Plots_9-year/Mollweide_' + map_fn[:-5] + '_range_'+ str(highE) + '_log.pdf'
    pyplot.savefig(save_fn)
