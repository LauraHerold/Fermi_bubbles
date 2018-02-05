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
plot_diff_leftright = True

data_class = 'ultraclean'
binmin = 0
binmax = 31

fn_ending = '.pdf'
colours = ['black', 'blue', 'red', 'green']
markers = ['s', 'o', 'D', '<']


########################################################################################################################## Constants

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('dct/dct_' + input_data[0] + '_' + data_class + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])


nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = binmax - binmin +1
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)


########################################################################################################################## Plot spectra

for b in xrange(nB):
         
    for l in xrange(nL):

        fig = pyplot.figure()
        colour_index = 0
        marker_index = 0
        
        for input in input_data:    
            print input
            
            dct  = dio.loaddict('dct/dct_' + input + '_' + data_class + '.yaml')
            diff_profiles = dct['6) Differential_flux_profiles']
            std_profiles = dct['7) Standard_deviation_profiles']
        
            map  = np.asarray(diff_profiles[b][l])
            std_map = np.asarray(std_profiles[b][l])    
            

########################################################################################################################## plot energy flux with error bars

            label = input
            pyplot.errorbar(Es, map, std_map, color=colours[colour_index], marker=markers[marker_index], markersize=6, markeredgewidth=0.4, linestyle='-', linewidth=0.1, label=label)
            colour_index += 1
            marker_index += 1
            
########################################################################################################################## Plot difference between right and left

        
        if plot_diff_leftright:
            print 'difference data'
            dct  = dio.loaddict('dct/dct_data_' + data_class + '.yaml')
            diff_profiles = dct['6) Differential_flux_profiles']
            std_profiles = dct['7) Standard_deviation_profiles']

            map = [0,0]
            std_map = [0,0]
        
            for ell in xrange(nL):
                map[ell]  = np.asarray(diff_profiles[b][ell])
                std_map[ell] = np.asarray(std_profiles[b][ell])

            difference = map[0] - map[1]

            total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)

            lab0 = lab1 = 'difference data'
            
            for reading_point in range(len(difference)):

                if difference[reading_point] > 0 and l==0:
                    pyplot.errorbar(Es[reading_point], difference[reading_point], total_std[reading_point], color='grey', marker='>', markersize=6., markeredgewidth=0.4, linestyle=':', linewidth=0.1, label = lab0)
                    lab0 = None
                if difference[reading_point] < 0 and l==1:
                    pyplot.errorbar(Es[reading_point], difference[reading_point], total_std[reading_point], color='grey', marker='>', markersize=6., markeredgewidth=0.4, linestyle=':', linewidth=0.1, label = lab1)
                    lab1 = None
                
                
    
       
########################################################################################################################## cosmetics, safe plot
        

        lg = pyplot.legend(loc='upper right', ncol=2, fontsize = 'small')
        lg.get_frame().set_linewidth(0)
        #pyplot.grid(True)
        pyplot.xlabel('$E$ [GeV]')
        pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
        pyplot.title(r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2) + r', $b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2))
    
        pyplot.axis('tight')
        name = 'SED_all_models_' + data_class + '_l=' + str(int(Lc[l])) + '_b=' + str(int(Bc[b]))
        fn = plot_dir + name + fn_ending
        pyplot.xscale('log')
        pyplot.yscale('log')

        if dB[b] == 10:
            pyplot.ylim((1.e-8,1.e-4))
        else:
            pyplot.ylim((1.e-8,1.e-3))

        pyplot.xlim((1., 2.e3))

        pyplot.savefig(fn, format = 'pdf')

        pyplot.close(fig)
        print 'plotted '+ str(l)

