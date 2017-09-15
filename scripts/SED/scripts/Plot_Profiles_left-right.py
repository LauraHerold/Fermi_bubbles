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

binmin = 12
binmax = 19 # Noch zu implementieren.

input_data = ['data', 'lowE', 'GALPROP', 'boxes']


fn_ending = '_high.pdf'
colours = ['black', 'blue', 'red', 'green']
markers = ['s', 'o', 'D', '<']

plot_diff_rightleft = True


########################################################################################################################## Constants

GeV2MeV = 1000.
delta = 0.346573590092441 # logarithmic distance between two energy bins
plot_dir = '../plots/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('../dct/dct_' + input_data[0] + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])[binmin:binmax+1]
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))


nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = binmax - binmin +1
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)



########################################################################################################################## Plot difference right - left

if plot_diff_rightleft:
    input = 'data'
    dct  = dio.loaddict('../dct/dct_' + input + '.yaml')
    diff_profiles = dct['6) Differential_flux_profiles']
    std_profiles = dct['7) Standard_deviation_profiles']

    profiles_data = np.zeros((nL, nB))
    std_data = np.zeros((nL, nB))
        
    for l in xrange(nL):
        for b in xrange(nB):
            for E in xrange(nE):
                profiles_data[l,b] += diff_profiles[b][l][binmin+E] *  deltaE[E] / Es[E]
                std_data[l,b] += (std_profiles[b][l][binmin+E] * deltaE[E] / Es[E])**2

    difference_profiles = profiles_data[0] - profiles_data[1]            
    difference_std = np.sqrt(std_data[0] + std_data[1])


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
                profiles[b] += diff_profiles[b][l][binmin+E] *  deltaE[E] / Es[E]
                std[b] += (std_profiles[b][l][binmin+E] * deltaE[E] / Es[E])**2
        std = np.sqrt(std) 

        pyplot.errorbar(Bc, profiles, std, color = colours[colour_index], marker=markers[marker_index], markersize=4, label=input)
        colour_index += 1
        marker_index += 1

    if plot_diff_rightleft:
        labl0 = labl1 = 'difference data'
        
        for b in xrange(nB):
            if l==0 and difference_profiles[b] > 0:
                pyplot.errorbar(Bc[b], difference_profiles[b], difference_std[b], color = 'grey', marker='>', markersize=6, label=labl0)
                labl0 = None
            if l==1 and difference_profiles[b] < 0:
                pyplot.errorbar(Bc[b], -difference_profiles[b], difference_std[b], color = 'grey', marker='>', markersize=6, label=labl1)
                labl1 = None
        
    lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$b$ [deg]')
    pyplot.ylabel(r'$ F \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')

    emin = Es[0] * np.exp(-delta/2)
    emax = Es[nE-1] * np.exp(delta/2)
    pyplot.title(r'$\ell \in (%i^\circ$' % (Lc[l] - 5) + '$,\ %i^\circ)$, ' % (Lc[l] + 5) + r'$E\in (%.1f\ \mathrm{GeV}$' %emin + r'$,\ %.1f$' %emax + r'$\ \mathrm{GeV})$')

    name = 'Profiles_'+ str(l)
    fn = plot_dir + name + fn_ending
    pyplot.yscale('log')
    pyplot.ylim(5.e-8, 1.e-4)
    pyplot.savefig(fn, format = 'pdf')
            
