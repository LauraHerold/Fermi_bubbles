# determine the volume average ISRF for Popescu et al.
# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima/
# python average_ISRF_galp_v56.py -m R12 -r 10 -z 6


import numpy as np
import pyfits
import pandas as pd
from scipy import interpolate
import scipy
from matplotlib import pyplot
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--model", dest = "model", default = "R12",
                  help="ISRF model: R12 or F98")
parser.add_option("-r", "--rmax", dest = "rmax", default = "10",
                  help="max radius in deg")
parser.add_option("-z", "--zmax", dest = "zmax", default = "6",
                  help="max height in deg")
parser.add_option("-w", "--show", dest="show_plots", default=0,
                  help="show plots")

(options, args) = parser.parse_args()

rmax = float(options.rmax)
zmax = float(options.zmax)
model = options.model
show_plots = int(options.show_plots)


erg2eV = 1.e12/1.6
pc2sm = 3.086e18
mk2angstrom = 1.e4
nano2mikro = 1.e-3


R_sun = 8.5 # kpc
R_max = np.deg2rad(rmax) * R_sun
z_max = np.deg2rad(zmax) * R_sun


out_dir = '/Users/Dmitry/Work/student_works/github_bubbles/data/ISRF_average'
key = 'GC_average'

folder0 = '/Users/Dmitry/data/Galprop_v56/Porter_etal_ApJ_846_67_2017_SEDonly/'
folder = '%s/%s/' % (folder0, model)

files = os.listdir(folder)
files = [fn for fn in files if fn.endswith('Flux.fits.gz')]

#print files
print 'get coordinates'
coords = np.zeros((len(files), 3))
for i, fn in enumerate(files):
    coords[i] = np.array(fn.split('_')[3:6], dtype=float)

crds = []
dcrds = []
for i in range(3):
    crd = np.sort(scipy.unique(coords[:,i]))
    dcrd = np.zeros_like(crd)
    bndr = np.zeros((len(crd) + 1))
    bndr[1:-1] = (crd[1:] + crd[:-1]) / 2.
    bndr[0] = crd[0]
    bndr[-1] = crd[-1]
    dcrd = (bndr[1:] - bndr[:-1])
    crds.append(crd)
    dcrds.append(dcrd)


print 'get data'
tot_vol = 0.
ld_dUdld = 0.
lds = None
for i, fn in enumerate(files):
    crd = np.array(fn.split('_')[3:6], dtype=float)
    if np.sqrt(crd[0]**2 + crd[1]**2) < R_max and np.abs(crd[2]) < z_max:
        hdu = pyfits.open(folder + fn)
        if lds is None:
            lds = hdu[1].data.field('Wavelength')

        vol = 1.
        crd_check = []
        for k in range(3):
            ind = np.argmin(np.abs(crd[k] - crds[k]))
            crd_check.append(crds[k][ind])
            vol *= dcrds[k][ind]
        tot_vol += vol
        ld_dUdld += vol * hdu[1].data.field('Total')
        
        if np.max(hdu[1].data.field('Total')) < 1.:
            print hdu[1].header
            print fn
            print crd_check
            pyplot.figure()
            pyplot.loglog(lds, hdu[1].data.field('Total'))
            pyplot.show()
            exit()


ld_dUdld /= tot_vol

arr = np.array([lds, ld_dUdld]).T
df = pd.DataFrame(arr, columns=['lambda(mkm)', 'ldUld(eV/cm^3)'])


fn = '%s/ldUld_%s_%s.csv' % (out_dir, key, model)
print 'save the model to file'
print fn
df.to_csv(fn, index=False)

if show_plots:
    pyplot.figure()
    pyplot.loglog(lds, ld_dUdld)
    pyplot.ylim(1.e-6, 1.e2)
    pyplot.show()
