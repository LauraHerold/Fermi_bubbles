""" Plots the SED of all models in one latitude stripes. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

import dio
from yaml import load

########################################################################################################################## parameters

input_data = ['data', 'lowE', 'GALPROP', 'boxes']


fn_ending = '.pdf'
colours = ['black', 'blue', 'red', 'green']
markers = ['s', 'o', 'D', '<']


########################################################################################################################## Constants

GeV2MeV = 1000.
delta = 0.346573590092441 # logarithmic distance between two energy bins
plot_dir = '../plots/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('../dct/dct_' + input_data[0] + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))


nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = len(diff_profiles[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)



########################################################################################################################## Plot profiles



for l in xrange(nL):
    pyplot.figure()
    colour_index = 0
    marker_index = 0

    
    for input in input_data:
        dct  = dio.loaddict('../dct/dct_' + input + '.yaml')
        diff_profiles = dct['6) Differential_flux_profiles']
        std_profiles = dct['7) Standard_deviation_profiles']

        profiles = np.zeros(nB)
        std = np.zeros(nB)
        
        for b in xrange(nB):
            for E in xrange(nE):
                profiles[b] += diff_profiles[b][l][E] *  deltaE[E] / Es[E]
                std[b] += (std_profiles[b][l][E] * deltaE[E] / Es[E])**2
        std = np.sqrt(std)

        pyplot.errorbar(Bc, profiles, std, color = colours[colour_index], marker=markers[marker_index], markersize=4, label=input)
        colour_index += 1
        marker_index += 1
    
    lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$b$ [deg]')
    pyplot.ylabel(r'$ F \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    pyplot.title(r'$\ell \in (%i^\circ$' % (Lc[l] - 5) + ', $%i^\circ)$' % (Lc[l] + 5))

    name = 'Profiles_'+ str(l)
    fn = plot_dir + name + fn_ending
    pyplot.yscale('log')
    pyplot.savefig(fn, format = 'pdf')
            
