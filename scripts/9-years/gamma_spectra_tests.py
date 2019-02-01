# tests of gamma-ray spectra library
# python gamma_spectra_tests.py

import numpy as np
from matplotlib import pyplot
import os

import gamma_spectra as gs

folder = 'gs_tests'
fn = '%s/pp_spectrum.pdf' % folder
if not os.path.isdir(folder):
    os.mkdir(folder)

index = -2.
cutoff_power = 5. + np.log10(5.)

npts = 200

E_g = 10.**np.linspace(-2., cutoff_power, npts)
p_p = 10**np.linspace(-0.5, cutoff_power + 1, npts)
# norm, index, cutoff
pars_p = [1., index, 10**cutoff_power]
dNdp_p = gs.plaw_cut(pars_p)(p_p)

EdNdE_gamma0 = gs.pi0_spectrum(dNdp_p, p_p)
EdNdE_gamma_vec0 = np.frompyfunc(EdNdE_gamma0, 1, 1)
gamma_spec0 = E_g * EdNdE_gamma_vec0(E_g)

EdNdE_gamma = gs.EdQdE_pp(dNdp_p, p_p)
EdNdE_gamma_vec = np.frompyfunc(EdNdE_gamma, 1, 1)
gamma_spec = E_g * EdNdE_gamma_vec(E_g)

E0 = 1.
norm = EdNdE_gamma_vec0(E0) / EdNdE_gamma_vec(E0)
gamma_spec *= norm

pyplot.figure()

pyplot.loglog(E_g, gamma_spec0, label='')
pyplot.loglog(E_g, gamma_spec, label='')

pyplot.xlabel(r'$\rm E_\gamma\; (GeV)$')
ylabel = r'${E_\gamma}\frac{d N}{d E_\gamma}$'


ymax = np.max(gamma_spec) * 10
ymin = np.max(gamma_spec) / 100
pyplot.ylim(ymin, ymax)
pyplot.title('pi0 gamma-ray spectrum')

pyplot.savefig(fn)
pyplot.show()