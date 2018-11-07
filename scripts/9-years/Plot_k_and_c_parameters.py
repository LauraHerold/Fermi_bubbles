import dio
from matplotlib import pyplot
import numpy as np


low_energy_range = "0.3-1.0GeV"
input_data = 'lowE' # capital letters
data_class = "source"


colours = ['grey', 'grey', 'grey', 'grey', 'grey', 'darkorange', 'green', 'red', 'blue', 'magenta', 'grey', 'grey', 'grey', 'grey', 'grey']
markers = ['.', '.', '.', '.', '.', '.', 'o', 's', 'D', '.', '.', '.', '.', '.', '.']

dct  = dio.loaddict('fits/' + input_data + '_' + low_energy_range + '_' + data_class + '.yaml')

c_array = dct['c_array'] # shape: [nB, nE]
k_array = dct['k_array']

nE = len(c_array)
nB = len(c_array[0])
print nB

spectra_dct = dio.loaddict('dct/Low_energy_range0/dct_lowE_' + data_class + '.yaml')

Bc = spectra_dct['4) Center_of_lat_bins']
Es = np.asarray(spectra_dct['5) Energy_bins'])

plot_dir = '../../plots/Plots_9-year/'




####################################################################################################################### print spectra



y_labels = [r'$E \cdot c_b\ [\mathrm{counts\ s^{-1} sr^{-1}}]$', r'$E\cdot k_b$']
names = ['c-parameter_spectrum', 'k-parameter_spectrum']
data_index = 0

for data in [c_array, k_array]:
    pyplot.figure()
    index = 0
    for b in range(0,15):

        if colours[index] == 'grey':
            label = None
        else:
            label = r'$b \in (%.0f^\circ$' % (Bc[b] - 2) + '$,\ %.0f^\circ)$' % (Bc[b] + 2)
        pyplot.plot(Es, Es * data[b], color = colours[index], marker = markers[index], label = label)


        index += 1

        
    lg = pyplot.legend(loc='lower left', ncol=1, fontsize = 'medium')
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$E\ [\mathrm{GeV}]$')
    pyplot.ylabel(y_labels[data_index])
    pyplot.yscale('log')
    pyplot.xscale('log')

    if data == k_array:
        pyplot.ylim(1e-4, 1e0)
    fn = plot_dir + 'Low_energy_range0/' + names[data_index] + '_' + input_data + '.pdf'
    pyplot.savefig(fn, format = 'pdf')

    data_index += 1

    

        
