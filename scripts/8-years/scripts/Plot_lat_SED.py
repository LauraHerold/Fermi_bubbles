""" Plots the SED of all models in one latitude stripes. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

import dio
from yaml import load

########################################################################################################################## Parameters

average_over_larger_stripes = True

binmin = 0
binmax = 31

fn_ending = 'rough_av.pdf'




########################################################################################################################## Constants

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.346573590092441 # logarithmic distance between two energy bins
plot_dir = '../plots/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('../dct/dct_boxes.yaml')
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])


nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = len(diff_profiles[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)

nB_av = (nB + 1) / 2

        
########################################################################################################################## Plot energy flux with errorbars

for l in xrange(nL):
    
    fig = pyplot.figure()


    if average_over_larger_stripes:
        labels = [r'$b \in (-2^\circ, 2^\circ)$', r'$b \in (\pm 2^\circ, \pm6^\circ)$', r'$b \in (\pm 6^\circ, \pm10^\circ)$', r'$b \in (\pm 10^\circ, \pm30^\circ)$', r'$b \in (\pm 30^\circ, \pm60^\circ)$']
        colours = ['black', 'red', 'blue', 'green', 'orange']
        markers = ['s', 'o', 'D', '>', '<']
        ls = ['-', '--', '-.', ':', ':']
        plot_profiles = np.zeros((5, nE))
        plot_profiles[0] = np.asarray(diff_profiles[7][l])
        plot_profiles[1] = (np.asarray(diff_profiles[6][l]) + np.asarray(diff_profiles[8][l])) / 2
        plot_profiles[2] = (np.asarray(diff_profiles[5][l]) + np.asarray(diff_profiles[9][l])) / 2
        plot_profiles[3] = (np.asarray(diff_profiles[3][l]) + np.asarray(diff_profiles[4][l]) + np.asarray(diff_profiles[10][l]) + np.asarray(diff_profiles[11][l])) / 4
        plot_profiles[4] = (np.asarray(diff_profiles[0][l]) + np.asarray(diff_profiles[1][l]) + np.asarray(diff_profiles[2][l]) + np.asarray(diff_profiles[12][l]) + np.asarray(diff_profiles[13][l]) + np.asarray(diff_profiles[14][l])) / 6

        plot_std_profiles = np.zeros((5, nE))
        plot_std_profiles[0] = np.asarray(std_profiles[7][l])
        plot_std_profiles[1] = np.sqrt( (np.asarray(std_profiles[6][l])**2 + np.asarray(std_profiles[8][l])**2) / 2)
        plot_std_profiles[2] = np.sqrt( (np.asarray(std_profiles[5][l])**2 + np.asarray(std_profiles[9][l])**2) / 2)
        plot_std_profiles[3] = np.sqrt( (np.asarray(std_profiles[3][l])**2 + np.asarray(std_profiles[4][l])**2 + np.asarray(std_profiles[10][l])**2 + np.asarray(std_profiles[11][l])**2) / 4)
        plot_std_profiles[4] = np.sqrt( (np.asarray(std_profiles[0][l])**2 + np.asarray(std_profiles[1][l])**2 + np.asarray(std_profiles[2][l])**2 + np.asarray(std_profiles[12][l])**2 + np.asarray(std_profiles[13][l])**2 + np.asarray(std_profiles[14][l])**2) / 6)
        
        for index in xrange(5):
            pyplot.errorbar(Es, plot_profiles[index], plot_std_profiles[index], color=colours[index], marker=markers[index], markeredgewidth=0.4, ls=ls[index], linewidth=0.1, label=labels[index])
    else:
        colours = ['grey', 'brown', 'orange', 'purple', 'green', 'blue', 'red', 'black']
        markers = ['1', '2', '8', 'H', '<', 's', 'o', 'D']
        colour_index = 0
        marker_index = 0           
        for b in xrange(nB_av):

            map_top  = np.asarray(diff_profiles[b][l])
            std_map_top = np.asarray(std_profiles[b][l])
            print Bc[b]

            map_bottom  = np.asarray(diff_profiles[nB-b-1][l])
            std_map_bottom = np.asarray(std_profiles[nB-b-1][l])
            print Bc[nB-b-1]
                
            map = (map_top + map_bottom) / 2
            std_map = np.sqrt((std_map_top**2 + std_map_bottom**2)/2)
                
            label = r'$b \in (\pm%i^\circ$' % np.absolute(Bc[b] - dB[b]/2) + ', $\pm%i^\circ)$' % np.absolute(Bc[b] + dB[b]/2)
            pyplot.errorbar(Es, map, std_map, color=colours[colour_index], marker=markers[marker_index], markeredgewidth=0.4, linestyle='-', linewidth=0.1, label=label)
            colour_index += 1
            marker_index += 1
        
       
########################################################################################################################## Cosmetics, safe plot
        

    lg = pyplot.legend(loc='upper right', ncol=2, fontsize = 'small')
    lg.get_frame().set_linewidth(0)
    #pyplot.grid(True)
    pyplot.xlabel('$E$ [GeV]')
    pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    pyplot.title(r'SED of boxes model, $\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2))
    pyplot.axis('tight')
    name = 'Lat_spectra' + '_l=' + str(int(Lc[l]))
    fn = plot_dir + name + fn_ending
    pyplot.xscale('log')
    pyplot.yscale('log')

    pyplot.ylim((5.e-8,1.e-5))
    pyplot.xlim((1., 2.e3))

    pyplot.savefig(fn, format = 'pdf')

    pyplot.close(fig)
    print 'plotted '+ str(l)

