# format the SL data from Popescu
# python popescu_isrf_formatting.py

import numpy as np
import pandas as pd

erg2eV = 1.e12/1.6
pc2sm = 3.086e18
nm2m = 1.e-9
nano2mikro = 1.e-3

c_light = 2.9979e8 # m/s speed of light
h_Planck = 4.1357e-15 # eV * s Planck constant


bands_fn = 'models/bands.csv'
bands = pd.read_csv(bands_fn)

lds = np.array(bands['lambda(nm)'], dtype=float) * nm2m
Es = c_light * h_Planck / lds
#print (c_light * h_Planck / 0.1)
#print Es
#exit()

res_dict = {}

#df = pd.DataFrame(arr, columns=['lambda(nm)', 'ldUld(eV/cm^3)'])

for ind, band in enumerate(bands['Band']):
    print band
    ld = bands['lambda(nm)'].loc[ind]
    fn = 'models/u_%s.dat' % band
    #df = pd.read_table(fn, header=80, sep=' ')
    data = np.loadtxt(fn, skiprows=82).T
    for i in range(data.shape[1]):
        r = data[0][i]
        z = data[1][i]
        u = data[2][i]
        key = 'r%ipc_z%ipc' % (r, z)
        if ind == 0:
            print key
            arr = np.zeros_like(bands, dtype=np.float)
            res_dict[key] = pd.DataFrame(arr, columns=['lambda(mkm)', 'ldUld(eV/cm^3)'])
        ld_angstrom = 10. * ld
        
        ldUld = u * ld * 10 * erg2eV / pc2sm**3
        res_dict[key].loc[ind] = ld * nano2mikro, ldUld

for key in res_dict.keys():
    fn = 'ISRF/ldUld_%s_SL.csv' % key
    print 'save the model to file'
    print fn
    res_dict[key].to_csv(fn, index=False)

#print res_dict