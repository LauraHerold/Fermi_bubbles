""" Plots the SED of all latitude stripes necessary to observe the Fermi bubbles. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

import dio
from yaml import load

########################################################################################################################## parameters

input_data = 'boxes'  # data, lowE, boxes, GALPROP

fit_plaw = False
fit_plaw_cut = True

bin_start_fit = 7 # Energy bin where fit starts (is halved if combine_two_energy_bins)
binmin = 0
binmax = 19

fn_ending = '.pdf'
colours = ['blue', 'red']


########################################################################################################################## Constants

GeV2MeV = 1000.
delta = 0.346573590092441 # logarithmic distance between two energy bins
plot_dir = '../plots/'

########################################################################################################################## Load dictionaries

dct  = dio.loaddict('../dct/dct_' + input_data + '.yaml')

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']

nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = len(diff_profiles[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)

########################################################################################################################## Integrate over energies

profiles = np.zeros((nL, nB))
std = np.zeros((nL, nB))

for b in xrange(nB):
    for l in xrange(nL):
        for E in xrange(nE):
            profiles[l,b] += diff_profiles[b][l][E]
            std[l,b] += std_profiles[b][l][E]**2

std = np.sqrt(std)
    
            
########################################################################################################################## plot energy flux

label = "Eastern bubble (left)"
pyplot.errorbar(Bc, profiles[1], std[1], color = "red", marker='s', markersize=4, label=label)
label = "Western bubble (right)"
pyplot.errorbar(Bc, profiles[0], std[0], color = "blue", marker='s', markersize=4, label=label)
       
########################################################################################################################## cosmetics, safe plot

lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.xlabel('$b$ [deg]')
pyplot.ylabel(r'$ E^2\frac{dN}{dE}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
#pyplot.title(r'SED in latitude stripes, $b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2))

name = 'Profiles_'+ input_data
fn = plot_dir + name + fn_ending
pyplot.yscale('log')
pyplot.savefig(fn, format = 'pdf')

