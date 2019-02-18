# calculate the ratio of norms for different models of CRe at the GC
# python CRe_spectra.py

import numpy as np

import dio


dct = dio.loaddict('ISRF_CRe.yaml')

for key in dct.keys():
    norm_ratio = dct[key]['N_0']/dct['v54']['N_0']
    print '%s %.2g, %.2f' % (key, norm_ratio, dct[key]['gamma'])
