# determine the volume average ISRF for Popescu et al.
# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima/
# python average_ISRF_galp_v54.py -r 10 -z 6


import numpy as np
import pyfits
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-r", "--rmax", dest = "rmax", default = "10",
                  help="max radius in deg")
parser.add_option("-z", "--zmax", dest = "zmax", default = "6",
                  help="max height in deg")
parser.add_option("-w", "--show", dest="show_plots", default=0,
                  help="show plots")

(options, args) = parser.parse_args()

rmax = float(options.rmax)
zmax = float(options.zmax)
show_plots = int(options.show_plots)


R_sun = 8.5 # kpc
R_max = np.deg2rad(rmax) * R_sun
z_max = np.deg2rad(zmax) * R_sun

out_dir = '/Users/Dmitry/Work/student_works/github_bubbles/data/ISRF_average'
key = 'GC_average'
model = 'v54'

# volume grid
rgrid = np.linspace(0, R_max, 100)
rc = (rgrid[1:] + rgrid[:-1]) / 2.
dr = rgrid[1:] - rgrid[:-1]
zgrid = np.linspace(0, z_max, 80)
zc = (zgrid[1:] + zgrid[:-1]) / 2.
dz = zgrid[1:] - zgrid[:-1]
vol = np.sum(rc * dr).T * np.sum(dz)

# load the data
print 'Load lambdas'
folder = '../../data/ISRF_flux/'
fn0 = folder + 'Standard_0_0_0_Flux.fits.gz'
hdu = pyfits.open(fn0)
ld_dUdld = hdu[1].data.field('Total')
lds = hdu[1].data.field('Wavelength')
rs_str = ['0', '2', '4',]
zs_str = ['0', '0.1', '0.2', '0.5', '1', '2']

rs = np.array(rs_str, dtype=float)
zs = np.array(zs_str, dtype=float)

# interpolate
u_data = np.zeros((len(rs), len(zs), len(lds)))
for i, r in enumerate(rs_str):
    for j, z in enumerate(zs_str):
        fn = folder + 'Standard_%s_0_%s_Flux.fits.gz' % (r, z)
        hdu = pyfits.open(fn)
        u_data[i, j] = hdu[1].data.field('Total')


# find the average
fs_list = []
for i, ld in enumerate(lds):
    fs = 1. * u_data[:,:,i]
    fs_func = interpolate.interp2d(rs, zs, fs.T, kind='linear',
                                   bounds_error=False, fill_value=0.)
    fs_list.append(fs_func)


ldUld = [np.sum((fs_list[i](rc, zc) * rc * dr).T * dz) / vol for i in range(len(lds))]

arr = np.array([lds, ldUld]).T
df = pd.DataFrame(arr, columns=['lambda(mkm)', 'ldUld(eV/cm^3)'])


fn = '%s/ldUld_%s_%s.csv' % (out_dir, key, model)
print 'save the model to file'
print fn
df.to_csv(fn, index=False)

if show_plots:
    pyplot.figure()
    pyplot.loglog(lds, ldUld)
    pyplot.ylim(1.e-6, 1.e2)
    pyplot.show()
