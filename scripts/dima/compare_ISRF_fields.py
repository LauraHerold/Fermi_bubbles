# compare different ISRF models

# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima
# python compare_ISRF_fields.py -v0 -w0


import numpy as np
from matplotlib import pyplot
from optparse import OptionParser

import auxil
import dio

parser = OptionParser()
parser.add_option("-w", "--show", dest="show_plots", default=0,
                  help="show plots")
parser.add_option("-v", "--save", dest="save_plots", default=0,
                  help="save plots")

(options, args) = parser.parse_args()

show_plots = int(options.show_plots)
save_plots = int(options.save_plots)


c_light = 2.9979e8 # m/s speed of light
h_Planck = 4.1357e-15 # eV * s Planck constant
mk2m = 1.e-6

plot_dir = '../../paper/plots/'

def get_isrf(model):
    fn = '../../data/ISRF_average/ldUld_GC_average_%s.csv' % model
    data = np.loadtxt(fn, delimiter=',', skiprows=1).T
    lds = data[0]
    ld_dUdld = data[1]
    Es = lambda2eV(lds * mk2m)[::-1]
    return Es, ld_dUdld

def get_isrf_ld(model):
    fn = '../../data/ISRF_average/ldUld_GC_average_%s.csv' % model
    data = np.loadtxt(fn, delimiter=',', skiprows=1).T
    lds = data[0]
    ld_dUdld = data[1]
    return lds, ld_dUdld


def lambda2eV(ld):
    """
        transform lambda (m) to energy (eV)
        """
    return c_light * h_Planck / ld



if show_plots or save_plots:
    auxil.setup_figure_pars(plot_type = 'spectrum')
    pyplot.rcParams['figure.subplot.top'] = 0.93
    pyplot.rcParams['figure.subplot.bottom'] = 0.15
    pyplot.rcParams['xtick.labelsize'] = 18
    pyplot.rcParams['ytick.labelsize'] = 18

    pyplot.figure()
    Es, ld_dUdld = get_isrf('v54')
    pyplot.loglog(Es, ld_dUdld[::-1], label='Porter et al. (2008)')

    Es, ld_dUdld = get_isrf('Popescu')
    pyplot.loglog(Es, ld_dUdld[::-1], ls='--', label='Popescu et al. (2017) ')


    Es, ld_dUdld = get_isrf('R12')
    pyplot.loglog(Es, ld_dUdld[::-1], ls='-.', label='Porter et al. (2017) R12')


    Es, ld_dUdld = get_isrf('F98')
    pyplot.loglog(Es, ld_dUdld[::-1], ls=':', label='Porter et al. (2017) F98')

    lg = pyplot.legend(loc='best', ncol=1, numpoints=1, labelspacing=0.4)
    lg.get_frame().set_linewidth(0)  #To get rid of the box

    pyplot.xlabel(r'$E_{\rm ISRF}\ \mathrm{[eV]}$')
    ylabel = r'$\lambda\frac{\mathrm{d}U}{\mathrm{d}\lambda}\ \left[ \frac{\mathrm{eV}}{\mathrm{cm^3}} \right]$'
    pyplot.ylabel(ylabel)

    pyplot.xlim(3.e-4, 30)
    pyplot.ylim(1.e-3, 100)

    fn = plot_dir + 'ISRF_comparison'
    auxil.save_figure(fn, ext=['pdf','png'], save_plots=save_plots)


    pyplot.figure()
    lds, ld_dUdld = get_isrf_ld('v54')
    pyplot.loglog(lds, ld_dUdld, label='Porter et al. (2008)')
    
    lds, ld_dUdld = get_isrf_ld('Popescu')
    pyplot.loglog(lds, ld_dUdld, ls='--', lw=1.5,
                  label='Popescu et al. (2017) ')
    
    
    lds, ld_dUdld = get_isrf_ld('R12')
    pyplot.loglog(lds, ld_dUdld, ls='-.', lw=1.5,
                  label='Porter et al. (2017) R12')
    
    
    lds, ld_dUdld = get_isrf_ld('F98')
    pyplot.loglog(lds, ld_dUdld, ls=':', lw=2., c='black',
                  label='Porter et al. (2017) F98')
    
    lg = pyplot.legend(loc='best', ncol=1, numpoints=1, labelspacing=0.4)
    lg.get_frame().set_linewidth(0)  #To get rid of the box
    
    pyplot.xlabel(r'$\lambda\ \mathrm{[\mu m]}$')
    ylabel = r'$\lambda\frac{\mathrm{d}U}{\mathrm{d}\lambda}\ \left[ \frac{\mathrm{eV}}{\mathrm{cm^3}} \right]$'
    pyplot.ylabel(ylabel)
    
    pyplot.xlim(6.e-2, 1.e3)
    pyplot.ylim(1.e-3, 100)


    fn = plot_dir + 'ISRF_comparison_ld'
    auxil.save_figure(fn, ext=['pdf','png'], save_plots=save_plots)

save_data = 1
if save_data:
    out_dict = {}
    out_dict['comment'] = 'volume average of ISRF energy density over the cylinder around GC '
    out_dict['comment'] += 'visible at |GLon| < 10 deg and |GLat| < 2 deg from the Earth'
    
    lds, ld_dUdld = get_isrf_ld('v54')
    key = 'porter_2008'
    out_dict[key] = {}
    out_dict[key]['lambdas'] = lds
    out_dict[key]['lambdas_unit'] = 'micro m'
    out_dict[key]['ld_dU_dld'] = ld_dUdld
    out_dict[key]['ld_dU_dld_unit'] = 'ev / cm^3'
    out_dict['comment'] = 'ISRF from Porter et al (2008)'


    lds, ld_dUdld = get_isrf_ld('Popescu')
    key = 'popescu_2017'
    out_dict[key] = {}
    out_dict[key]['lambdas'] = lds
    out_dict[key]['lambdas_unit'] = 'micro m'
    out_dict[key]['ld_dU_dld'] = ld_dUdld
    out_dict[key]['ld_dU_dld_unit'] = 'ev / cm^3'
    out_dict['comment'] = 'ISRF from Popescu et al (2017)'


    lds, ld_dUdld = get_isrf_ld('R12')
    key = 'Porter_R12_2017'
    out_dict[key] = {}
    out_dict[key]['lambdas'] = lds
    out_dict[key]['lambdas_unit'] = 'micro m'
    out_dict[key]['ld_dU_dld'] = ld_dUdld
    out_dict[key]['ld_dU_dld_unit'] = 'ev / cm^3'
    out_dict['comment'] = 'ISRF from Porter et al (2017) R12 model'



    lds, ld_dUdld = get_isrf_ld('F98')
    key = 'Porter_F98_2017'
    out_dict[key] = {}
    out_dict[key]['lambdas'] = lds
    out_dict[key]['lambdas_unit'] = 'micro m'
    out_dict[key]['ld_dU_dld'] = ld_dUdld
    out_dict[key]['ld_dU_dld_unit'] = 'ev / cm^3'
    out_dict['comment'] = 'ISRF from Porter et al (2017) F98 model'

    out_fn = 'results/FigC1_ISRF_energy_density_comparison.yaml'
    dio.savedict(out_dict, out_fn)


if show_plots:
    pyplot.show()

