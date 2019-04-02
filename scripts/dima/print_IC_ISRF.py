# print the relative normalizations of CRe SEDs for different ISRF models
# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima
# python print_IC_ISRF.py

import numpy as np

import dio

fn = 'data/ISRF_CRe.yaml'

dct = dio.loaddict(fn)

m0 = 'v54'
N = dct[m0]['N_0']

for key in dct.keys():
    print '%s, %.3g & %.3g' % (key, dct[key]['N_0'] / N, dct[key]['gamma'])