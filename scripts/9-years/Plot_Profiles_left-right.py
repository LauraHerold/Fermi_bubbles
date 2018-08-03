""" Plots the latitude profiles of all models. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit
from optparse import OptionParser
import dio
from yaml import load
import auxil

########################################################################################################################## parameters

plot_diff_rightleft = False
input_data = ['data', 'lowE', 'boxes', 'GALPROP']
colours = ['black', 'blue', 'red', 'green']
markers = ['s', 'o', 'D', '<']
titles = ['West', 'East']


parser = OptionParser()
parser.add_option("-c", "--data_class", dest = "data_class", default = "source", help="data class (source or ultraclean)")
parser.add_option("-E", "--lowE_range", dest="lowE_range", default='0', help="There are 3 low-energy ranges: (3,5), (3,3), (4,5), (6,7)")
(options, args) = parser.parse_args()

data_class = str(options.data_class)
low_energy_range = int(options.lowE_range) # 0: baseline, 4: test

########################################################################################################################## Constants

labels_dct = {"data":"Data", "lowE": "LowE", "boxes": "Rectangles", "GALPROP":"GALPROP"}

highE_ranges = ((0,5),(6,11),(12,17))        # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) + '/dct_' + input_data[0] + '_' + data_class + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))


nB = len(diff_profiles)
nL = len(diff_profiles[0])


########################################################################################################################## Plot difference right - left


for highE in range(3):
    binmin = highE_ranges[highE][0]
    binmax = highE_ranges[highE][1]
    nE = binmax - binmin + 1
    
    if plot_diff_rightleft:
        input = 'data'
        dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) + '/dct_' + input + '_' + data_class + '.yaml')
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
        auxil.setup_figure_pars(plot_type = 'spectrum')
        f = pyplot.figure()
        colour_index = 0
        marker_index = 0
    
        for input in input_data:
            print input
            dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) + '/dct_' + input + '_' + data_class + '.yaml')
            diff_profiles = dct['6) Differential_flux_profiles']
            std_profiles = dct['7) Standard_deviation_profiles']

            profiles = np.zeros(nB)
            std = np.zeros(nB)
        
            for b in xrange(nB):
                for E in xrange(nE):
                    profiles[b] += diff_profiles[b][l][binmin+E] *  deltaE[E] / Es[E]
                    std[b] += (std_profiles[b][l][binmin+E] * deltaE[E] / Es[E])**2
            std = np.sqrt(std) 
            
            
            pyplot.errorbar(Bc, profiles, std, color = colours[colour_index], marker=markers[marker_index], linewidth = 1.3,  label=labels_dct[input])
            colour_index += 1
            marker_index += 1

        if plot_diff_rightleft:
            labl0 = labl1 = 'difference data'
        
            for b in xrange(nB):
                if l==0 and difference_profiles[b] > 0:
                    pyplot.errorbar(Bc[b], difference_profiles[b], difference_std[b], color = 'grey', marker='>', linewidth = 1.3, label=labl0)
                    labl0 = None
                if l==1 and difference_profiles[b] < 0:
                    pyplot.errorbar(Bc[b], -difference_profiles[b], difference_std[b], color = 'grey', marker='>', linewidth = 1.3, label=labl1)
                    labl1 = None


        emin = Es[binmin] * np.exp(-delta/2)
        emax = Es[binmax] * np.exp(delta/2)
        ax = f.add_subplot(111)
        #pyplot.text(0.1, 0.9,'matplotlib', ha='right', va='top')#, transform=ax.transAxes)
        textstr = r'$\ell \in (%.0f^\circ$' % (Lc[l] - 5) + '$,\ %.0f^\circ)$\n' % (Lc[l] + 5) + r'$E\in (%.0f$' %emin + r'$,\ %.0f$' %emax + r'$)\ \mathrm{GeV}$'
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize = 20, verticalalignment='top')
        
        lg = pyplot.legend(loc='upper right', ncol=1, fontsize = 'medium')
        lg.get_frame().set_linewidth(0)
        pyplot.grid(True)
        pyplot.xlabel('$b\ [\mathrm{deg}]$')
        pyplot.ylabel(r'$ I\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')

        
        pyplot.title(titles[l])
        


        name = 'Profiles_l='+ str(l)
        fn = plot_dir + '/Low_energy_range'+ str(low_energy_range)+ '/' + name + '_' + data_class + '_range_' + str(highE) +'.pdf'
        pyplot.yscale('log')
        pyplot.ylim(3.e-7, 7.e-5)
        pyplot.savefig(fn, format = 'pdf')
            
