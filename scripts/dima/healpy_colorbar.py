# python healpy_colorbar.py

import numpy as np
import healpy as hp
import matplotlib.pyplot as pyplot
#from matplotlib import rc
from matplotlib import transforms

import auxil
auxil.setup_figure_pars()

#pyplot.rcParams['figure.subplot.bottom'] = 0.20
#pyplot.rcParams['figure.subplot.top'] = 0.95
#pyplot.rcParams['xtick.labelsize'] = 12
#spyplot.rcParams['ytick.labelsize'] = 12

m = 1. * np.arange(hp.nside2npix(32))
m = m / 1000.
unit = r'$I [\mathrm{GeV\ s^{-1} sr^{-1} cm^{-2}]}$'

hp.mollview(m, cbar=None)
auxil.add_mollview_colorbar(m, label=unit, nticks=5)

pyplot.title('Mollview colorbar test')

auxil.save_figure('mollview_colorbar_test', ext=['pdf'], save_plots=0)

pyplot.show()
