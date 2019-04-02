# combine IR and SL ISRFs from Popescu
# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima/
# python combine_ISRF_Popescu.py


import numpy as np
import pyfits
import pandas as pd
from matplotlib import pyplot

import auxil


out_dir = '/Users/Dmitry/Work/student_works/github_bubbles/data/ISRF_average'
key = 'GC_average'
model = 'Popescu'

print 'load lambdas from Galprop model'
IRFmap_fn0 = '../../data/ISRF_average/ldUld_GC_average_F98.csv'
data = np.loadtxt(IRFmap_fn0, delimiter=',', skiprows=1).T
lds = data[0]

print 'Take Popescu ISRF model'
isrf = auxil.get_popescu_isrf(field='total')
isrf_SL = auxil.get_popescu_isrf(field='SL')
ld_dUdld = isrf(lds)


arr = np.array([lds, ld_dUdld]).T
df = pd.DataFrame(arr, columns=['lambda(mkm)', 'ldUld(eV/cm^3)'])


fn = '%s/ldUld_%s_%s.csv' % (out_dir, key, model)
print 'save the model to file'
print fn
df.to_csv(fn, index=False)

if 1:
    pyplot.figure()
    #pyplot.loglog(lds, ld_dUdld)
    pyplot.loglog(lds, isrf_SL(lds), ls='--', lw=2)
    pyplot.loglog(lds, isrf_SL(lds)+1.e-20)
    pyplot.xlim(0.01, 100)
    pyplot.ylim(0.001, 10)
    pyplot.show()
