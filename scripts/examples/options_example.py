# an example on how to use command line options
# python options_example.py -x 5. -p 2 -c "result: 5^2 ="

import numpy as np
import scipy
from optparse import OptionParser


parser = OptionParser()



parser.add_option("-x", "--base", dest="x",
                  default='0.',
                  help="base of the number")

parser.add_option("-p", "--power", dest="pwr",
                  default='1.',
                  help="power of the number")

parser.add_option("-c", "--comment", dest="comment",
                  default='',
                  help="comment")

(options, args) = parser.parse_args()


x = float(options.x)
p = float(options.pwr)

st = options.comment


print '%s %.3g' % (st, x**p)