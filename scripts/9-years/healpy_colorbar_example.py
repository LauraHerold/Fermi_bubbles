# python healpy_colorbar_example.py

import numpy as np
import healpy as hp
import matplotlib.pyplot as pyplot
import os

import auxil

auxil.setup_figure_pars()


m = 1. * np.arange(hp.nside2npix(32))
m = m / 1000.
unit = r'$I [\mathrm{GeV\ s^{-1} sr^{-1} cm^{-2}]}$'

hp.mollview(m, cbar=None)
auxil.add_mollview_colorbar(m, label=unit, nticks=5)

pyplot.title('Mollview colorbar test')

#if not os.path.isdir('tmp'):
#    os.mkdir('tmp')

#auxil.save_figure('tmp/mollview_colorbar_test', ext=['pdf'], save_plots=1)

pyplot.show()
