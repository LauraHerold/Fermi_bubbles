""" Plots the SED of all latitude stripes necessary to observe the Fermi bubbles. """

import numpy as np
import pyfits
from matplotlib import pyplot
import healpylib as hlib
import dio
from yaml import load
import auxil

########################################################################################################################## Parameters


latitude = 8# 7 is GP

fn_ending = ".pdf"


########################################################################################################################## Constants

colours = ["blue", "red"]
lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]


dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575                                                              # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'



########################################################################################################################## Load dictionaries


dct_boxes = dio.loaddict('dct/Low_energy_range0/dct_boxes_source.yaml')
dct_lowE = dio.loaddict('dct/Low_energy_range0/dct_lowE_source.yaml')
dct_GALPROP = dio.loaddict('dct/Low_energy_range0/dct_GALPROP_source.yaml')
dct_lowE1 = dio.loaddict('dct/Low_energy_range1/dct_lowE_source.yaml')
dct_lowE2 = dio.loaddict('dct/Low_energy_range2/dct_lowE_source.yaml')
dct_lowE3 = dio.loaddict('dct/Low_energy_range3/dct_lowE_source.yaml')
dct_boxes1 = dio.loaddict('dct/Low_energy_range1/dct_boxes_source.yaml')
dct_boxes2 = dio.loaddict('dct/Low_energy_range2/dct_boxes_source.yaml')
dct_boxes3 = dio.loaddict('dct/Low_energy_range3/dct_boxes_source.yaml')

SED_boxes = dct_boxes['6) Differential_flux_profiles']
SED_lowE = dct_lowE['6) Differential_flux_profiles']
SED_GALPROP = dct_GALPROP['6) Differential_flux_profiles']
SED_lowE1 = dct_lowE1['6) Differential_flux_profiles']
SED_lowE2 = dct_lowE2['6) Differential_flux_profiles']
SED_lowE3 = [0,0] + dct_lowE3['6) Differential_flux_profiles']
SED_boxes1 = dct_boxes1['6) Differential_flux_profiles']
SED_boxes2 = dct_boxes2['6) Differential_flux_profiles']
SED_boxes3 = dct_boxes3['6) Differential_flux_profiles']

std_boxes = dct_boxes['7) Standard_deviation_profiles']


Lc = dct_boxes['3) Center_of_lon_bins']
Bc = dct_boxes['4) Center_of_lat_bins']
Es = np.asarray(dct_boxes['5) Energy_bins'])

nB = len(SED_boxes)
nL = len(SED_boxes[0])
nE = len(SED_boxes[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)

########################################################################################################################## Plot


b = latitude
print "Bc[b]: ",  Bc[b]
auxil.setup_figure_pars(plot_type = 'spectrum')
pyplot.figure()
index = 0

    
for l in xrange(nL):
     
    baseline  = np.asarray(SED_boxes[b][l])
    std = np.asarray(std_boxes[b][l])
    
    #syst_max = np.maximum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], SED_boxes1[b][l], SED_boxes2[b][l]])
    #syst_min = np.minimum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], SED_boxes1[b][l], SED_boxes2[b][l]])

    syst_max = np.maximum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], [0,0] + SED_lowE3[b][l], SED_boxes1[b][l], SED_boxes2[b][l], [0,0] + SED_boxes3[b][l]])
    syst_min = np.minimum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], [0,0] + SED_lowE3[b][l], SED_boxes1[b][l], SED_boxes2[b][l], [0,0] + SED_boxes3[b][l]])
    print Es.shape
    print syst_max.shape
               
    label = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2)

    
    pyplot.errorbar(Es, baseline, std, color=colours[index], marker='s', linestyle = '', label=label)
    pyplot.fill_between(Es, syst_min, syst_max, color = colours[index], alpha = 0.5)

    index += 1


########################################################################################################################## Cosmetics, safe plot

    
lg = pyplot.legend(loc='upper left', ncol=2)
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.xlabel('$E\ \mathrm{[GeV]}$')
#pyplot.ylabel('Counts')
pyplot.ylabel(r'$ E^2\frac{\mathrm{d}N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
pyplot.title(r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2))

name = 'Summary_SED_' + str(int(Bc[b]))
fn = plot_dir + name + fn_ending
pyplot.xscale('log')
pyplot.yscale('log')
pyplot.ylim((1.e-8,4.e-4))
pyplot.savefig(fn, format = 'pdf')


