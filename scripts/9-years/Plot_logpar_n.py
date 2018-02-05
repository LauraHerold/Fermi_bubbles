""" Plots the cutoff energies of all models as a function of latitude, left and right separately. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

import dio
from yaml import load
import auxil

#data_class = source 


lats = np.array([-55.0, -45.0, -35.0, -25.0, -15.0, -8.0, -4.0, 0.0, 4.0, 8.0, 15.0, 25.0, 35.0, 45.0, 55.0])


GALPROP_source =   np.array( [[0.020193204398083198, -1.7050281033646413], [-0.7553814904859283, -0.9079979757209884], [-0.44445046996646187, -1.222915419852999], [-1.2118312366535977, -0.972401622430998], [-0.9615656106379082, -0.9866235068380204], [-0.7077874743127229, -1.1666090108072247], [0.2580448417647462, -5.034039201082924], [0.19458503815044836, -2.9205442806340285], [0.20468947400458337, -0.24540062925194006], [0.12256536791934469, -0.7827272797015828], [-1.666858456585664, -1.8795328019273152], [-0.6384118149204406, -0.7940574357382184], [-0.5759523284815387, -1.2769115732947107], [-0.6173667731509174, -0.5260881130858588], [-5.292248336907553, -0.04964672385880109]] ).T

lowE_source_range0 =np.array( [[0.052525154319261025, -1.6185738788112811], [-0.7416168502890491, -0.9012619616632569], [-0.8240501609457702, -1.3563026540356677], [-1.1677866430932118, -1.0978619177947126], [-1.0129576784932637, -1.1979496602805546], [-0.8925911802329224, -0.9718661566908202], [-0.13902361657491813, -0.8987739728786489], [-0.006838606557156068, -2.3055498796259517], [-0.3085572334492649, -1.1569499848421014], [-0.02673647143489366, -0.7491107445265095], [-1.0468467767041627, -1.3126226608387936], [-4.194358332886106, -25.4945288893271], [-9.209508766799585, -8.008088929139788], [-6.168081589152466, -3.824310525539481], [-4.957081664366074, -1.8010516285678997]] ).T

boxes_source_range0 = np.array( [[0.25748176414638596, -0.9191153392696467], [-0.7572927770744888, -0.9254943458498345], [-0.7066205769073981, -1.2200304865574159], [-1.042603647988206, -0.9437793691124918], [-0.9276764427924605, -1.0837481984412292], [-1.081621399393681, -1.1571168375763399], [-0.34172239597933035, -0.8355197037339088], [0.023771842845361824, -1.851048778014973], [-0.24433188750772544, -0.9969361810227247], [-0.08114232723577286, -0.7652980884538036], [-0.4800263902737494, -1.490368741343436], [0.16232224397850703, 0.4960243105465444], [0.5296672516134804, -0.03672316069722564], [0.12794807178595824, 0.6525605437050528], [0.7323504022193577, 0.7001724196772388]] ).T

data_source = np.array( [[-0.45045643245520206, -1.1007861873834948], [-0.779519538318318, -0.8618563054598094], [-0.7812238583565777, -1.0193676919178514], [-0.9368599071156744, -0.8209864074901879], [-0.736562208586016, -0.7741081879059506], [-0.6144759845634502, -0.6193936034143421], [-0.24130118499496622, -0.4284978962353906], [-0.17101027053414142, -0.5016513205609339], [-0.3337075369686768, -0.44077890073677595], [-0.3720342207286602, -0.6136834883264732], [-0.6165911234328962, -0.8175265380173252], [-0.6181189334751922, -0.5934806365908687], [-0.66882310637426, -0.8246080540831423], [-0.7681302181578038, -0.70811491555127], [-0.8080362705036228, -0.7613815284161392]] ).T

########################################################################################################################## Right

auxil.setup_figure_pars(plot_type = 'spectrum')
pyplot.figure()

pyplot.plot(lats, data_source[0], label = "data")
pyplot.plot(lats, lowE_source_range0[0],  label = "lowE")
pyplot.plot(lats, boxes_source_range0[0], label = "boxes")
pyplot.plot(lats, GALPROP_source[0],  label = "GALPROP")

lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.xlabel(r'$b\ \mathrm{[deg]}$')
pyplot.ylabel(r'$ n(500\ \mathrm{GeV})$')

pyplot.title(r'Inclination $n(E)= -\alpha-2\beta\ \log E$ for $\ell \in (-10^\circ,0^\circ)$', fontsize=20)#' for log parabola $\left(E\ \frac{\mathrm{d}N}{\mathrm{d}E}\right) = N_0 E^{-\alpha-\beta\ \log E}$ at $E = 500$ GeV', 

plot_dir = '../../plots/Plots_9-year/Low_energy_range0/'

name = 'LogParabola_n(500GeV)_l_in_(-10,0).pdf'
fn = plot_dir + name
pyplot.ylim(-2., 2)

pyplot.savefig(fn, format = 'pdf')


########################################################################################################################## Left

pyplot.figure()

pyplot.plot(lats, data_source[1], label = "data")
pyplot.plot(lats, lowE_source_range0[1],  label = "lowE")
pyplot.plot(lats, boxes_source_range0[1], label = "boxes")
pyplot.plot(lats, GALPROP_source[1],  label = "GALPROP")

lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.xlabel(r'$b\ \mathrm{[deg]}$')
pyplot.ylabel(r'$ n(500\ \mathrm{GeV})$')

pyplot.title(r'Inclination $n(E)= -\alpha-2\beta\ \log E$ for $\ell \in (0^\circ,10^\circ)$', fontsize=20)#' for log parabola $\left(E\ \frac{\mathrm{d}N}{\mathrm{d}E}\right) = N_0 E^{-\alpha-\beta\ \log E}$ at $E = 500$ GeV', 

plot_dir = '../../plots/Plots_9-year/Low_energy_range0/'

name = 'LogParabola_n(500GeV)_l_in_(0,10).pdf'
fn = plot_dir + name 
#pyplot.yscale('log')
pyplot.ylim(-2., 2)
auxil.setup_figure_pars(plot_type = 'spectrum')
pyplot.savefig(fn, format = 'pdf')
