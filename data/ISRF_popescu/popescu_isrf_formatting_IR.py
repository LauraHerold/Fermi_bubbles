# format the IR data from Popescu
# python popescu_isrf_formatting_IR.py


import numpy as np
import pandas as pd
import pyfits

erg2eV = 1.e12/1.6
pc2sm = 3.086e18
mk2angstrom = 1.e4

hdu = pyfits.open('IR_model/urad_out_arr.fits')
u_data = hdu[0].data.T
#print u_data.shape

if 0:
    hdu = pyfits.open('IR_model/RR.fits')
    rs = hdu[0].data

    hdu = pyfits.open('IR_model/ZZ.fits')
    zs = hdu[0].data

hdu = pyfits.open('IR_model/lambda_arr.fits')
lds = hdu[0].data
#print hdu[0].header
#print lds

#print u_data[0,0]


ldUld = u_data[0,0] * lds * mk2angstrom * erg2eV / pc2sm**3

arr = np.array([lds, ldUld]).T

r = 0
z = 0
key = 'r%ipc_z%ipc' % (r, z)
df = pd.DataFrame(arr, columns=['lambda(mkm)', 'ldUld(eV/cm^3)'])
fn = 'ISRF/ldUld_%s_IR.csv' % key
print 'save the model to file'
print fn
df.to_csv(fn, index=False)
