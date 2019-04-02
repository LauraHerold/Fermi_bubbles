# determine the volume average ISRF for Popescu et al.
# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima/
# python average_ISRF_Popescu.py -f IR -r 10 -z 6


import numpy as np
import pyfits
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--field", dest = "field", default = "IR",
                  help="ISRF field: IR or SL")
parser.add_option("-r", "--rmax", dest = "rmax", default = "10",
                  help="max radius in deg")
parser.add_option("-z", "--zmax", dest = "zmax", default = "6",
                  help="max height in deg")
parser.add_option("-w", "--show", dest="show_plots", default=0,
                  help="show plots")


(options, args) = parser.parse_args()

spec = options.field
rmax = float(options.rmax)
zmax = float(options.zmax)
show_plots = int(options.show_plots)
#spec = 'IR'

erg2eV = 1.e12/1.6
pc2sm = 3.086e18
mk2angstrom = 1.e4
nano2mikro = 1.e-3
R_sun = 8500 # pc

R_max = np.deg2rad(rmax) * R_sun
z_max = np.deg2rad(zmax) * R_sun

folder = '/Users/Dmitry/Work/student_works/github_bubbles/data/ISRF_popescu/'

# coordinates
hdu = pyfits.open(folder + 'ISRF_IR/RR.fits')
rs = hdu[0].data

hdu = pyfits.open(folder + 'ISRF_IR/ZZ.fits')
zs = hdu[0].data

rgrid = np.linspace(0, R_max, 100)
rc = (rgrid[1:] + rgrid[:-1]) / 2.
dr = rgrid[1:] - rgrid[:-1]
zgrid = np.linspace(0, z_max, 80)
zc = (zgrid[1:] + zgrid[:-1]) / 2.
dz = zgrid[1:] - zgrid[:-1]
vol = np.sum(rc * dr).T * np.sum(dz)


if spec == 'IR':
    hdu = pyfits.open(folder + 'ISRF_IR/lambda_arr.fits')
    lds = hdu[0].data

    hdu = pyfits.open(folder + 'ISRF_IR/urad_out_arr.fits')
    u_data = hdu[0].data.T

elif spec == 'SL':
    bands_fn = folder + 'ISRF_SL/bands.csv'
    bands = pd.read_csv(bands_fn)
    lds = np.array(bands['lambda(nm)']) * nano2mikro
    u_data = np.zeros((len(rs), len(zs), len(lds)))
    for i, r in enumerate(rs):
        for j, z in enumerate(zs):
            fn = folder + 'ISRF/ldUld_r%ipc_z%ipc_SL.csv' % (r, z)
            u_data[i, j] = np.loadtxt(fn, delimiter=',', skiprows=1).T[1]


fs_list = []
for i, ld in enumerate(lds):
    fs = 1. * u_data[:,:,i]
    if spec == 'IR':
        fs *= lds[i] * mk2angstrom * erg2eV / pc2sm**3
    fs_func = interpolate.interp2d(rs, zs, fs.T, kind='linear',
                                   bounds_error=False, fill_value=0.)
    fs_list.append(fs_func)


ldUld = [np.sum((fs_list[i](rc, zc) * rc * dr).T * dz) / vol for i in range(len(lds))]
arr = np.array([lds, ldUld]).T
df = pd.DataFrame(arr, columns=['lambda(mkm)', 'ldUld(eV/cm^3)'])
key = 'GC_average'
fn = folder + 'ISRF/ldUld_%s_%s.csv' % (key, spec)
print 'save the model to file'
print fn
df.to_csv(fn, index=False)

if show_plots:
    pyplot.figure()
    pyplot.loglog(lds, ldUld)
    pyplot.show()
