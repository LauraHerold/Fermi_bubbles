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


GALPROP_source = np.array( [[0.020193204398083198, -1.7050281033646413], [-0.7553814904859283, -0.9079979757209884], [-0.44445046996646187, -1.222915419852999], [-1.2118312366535977, -0.972401622430998], [-0.9615656106379082, -0.9866235068380204], [-0.7077874743127229, -1.1666090108072247], [0.2580448417647462, -5.034039201082924], [0.19458503815044836, -2.9205442806340285], [0.20468947400458337, -0.24540062925194006], [0.12256536791934469, -0.7827272797015828], [-1.666858456585664, -1.8795328019273152], [-0.6384118149204406, -0.7940574357382184], [-0.5759523284815387, -1.2769115732947107], [-0.6173667731509174, -0.5260881130858588], [-5.292248336907553, -0.04964672385880109]] ).T

lowE_source_range0 =  np.array( [[0.254880207806775, -1.4122129326499895], [-0.7937615794731187, -0.8081089582565935], [-0.8867736990451527, -1.2237805662445573], [-1.1770879159552758, -1.249603860057942], [-0.9881370303620244, -1.137752288205181], [-0.7627437040142118, -0.828953019461467], [-0.22669135452892625, -0.8769281123220406], [0.03963299349748153, -21.230892965750993], [-0.21318855534809047, -1.1588460220955543], [-0.20460880842658455, -0.7965516056926153], [-0.7841599273444992, -1.515204495381882], [-0.5008919284229727, -0.4550494942704671], [-0.46156656558617926, -0.6443518428229151], [-0.7659351060803523, -0.4110284820087814], [-10.1645210397837, 0.7530038024013328]] ).T

boxes_source_range0 = np.array( [[0.04009137687886588, -1.0538746531407042], [-0.9252545136523364, -0.5885464457908967], [-0.6194889424955493, -1.1582618268793001], [-1.0769691560826686, -1.2129517719740783], [-0.970288636807253, -1.1986173586885838], [-0.39780919140846926, -0.7358854044181655], [-0.25229652249955353, -0.6661441841523166], [-0.09895516671665015, -0.40370595781434326], [-0.17982103378955355, -2.0004610436384835], [-0.5491175825299884, -0.8071367035577898], [-0.8308361947155896, -1.3315742459608042], [-0.4973859878716527, -0.3550677514510764], [-0.504859529800893, -0.295360624984501], [-0.8578509616990857, -0.4409916916847258], [-0.20438951421332402, 1.2398602809663162]] ).T

data_source = np.array( [[-0.45045643245520206, -1.1007861873834948], [-0.779519538318318, -0.8618563054598094], [-0.7812238583565777, -1.0193676919178514], [-0.9368599071156744, -0.8209864074901879], [-0.736562208586016, -0.7741081879059506], [-0.6144759845634502, -0.6193936034143421], [-0.24130118499496622, -0.4284978962353906], [-0.17101027053414142, -0.5016513205609339], [-0.3337075369686768, -0.44077890073677595], [-0.3720342207286602, -0.6136834883264732], [-0.6165911234328962, -0.8175265380173252], [-0.6181189334751922, -0.5934806365908687], [-0.66882310637426, -0.8246080540831423], [-0.7681302181578038, -0.70811491555127], [-0.8080362705036228, -0.7613815284161392]] ).T

########################################################################################################################## Right

auxil.setup_figure_pars(plot_type = 'spectrum')
pyplot.figure()

pyplot.plot(lats, 2 - data_source[0], label = "Data", color = "black", linewidth = 1.3)
pyplot.plot(lats, 2 - lowE_source_range0[0],  label = "LowE", color = "blue", linewidth = 1.3)
pyplot.plot(lats, 2 - boxes_source_range0[0], label = "Rectangles", color = "red", linewidth = 1.3)
pyplot.plot(lats, 2 - GALPROP_source[0],  label = "GALPROP", color = "green", linewidth = 1.3)

lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.xlabel(r'$b\ \mathrm{[deg]}$')
pyplot.ylabel(r'$2 - n(500\, \mathrm{GeV})$')

#pyplot.title(r'Inclination $n(E)= -\alpha-2\beta\ \log E$ for $\ell \in (-10^\circ,0^\circ)$', fontsize=20)#' for log parabola $\left(E\ \frac{\mathrm{d}N}{\mathrm{d}E}\right) = N_0 E^{-\alpha-\beta\ \log E}$ at $E = 500$ GeV',
pyplot.title(r"$\ell \in (-10^\circ,0^\circ)$")

plot_dir = '../../plots/Plots_9-year/Low_energy_range0/'

name = 'LogParabola_n(500GeV)_l_in_(-10,0).pdf'
fn = plot_dir + name
pyplot.ylim(1.5, 4.5)

pyplot.savefig(fn, format = 'pdf')


########################################################################################################################## Left


pyplot.figure()

pyplot.plot(lats, 2 - data_source[1], label = "Data", color = "black", linewidth = 1.3)
pyplot.plot(lats, 2 - lowE_source_range0[1],  label = "LowE", color = "blue", linewidth = 1.3)
pyplot.plot(lats, 2 - boxes_source_range0[1], label = "Rectangles", color = "red", linewidth = 1.3)
pyplot.plot(lats, 2 - GALPROP_source[1],  label = "GALPROP", color = "green", linewidth = 1.3)

lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.xlabel(r'$b\ \mathrm{[deg]}$')
pyplot.ylabel(r'$2 - n(500\, \mathrm{GeV})$')

pyplot.title(r'Inclination $n(E)= -\alpha-2\beta\ \log E$ for $\ell \in (0^\circ,10^\circ)$', fontsize=20)#' for log parabola $\left(E\ \frac{\mathrm{d}N}{\mathrm{d}E}\right) = N_0 E^{-\alpha-\beta\ \log E}$ at $E = 500$ GeV',
pyplot.title(r"$\ell \in (0^\circ,10^\circ)$")

plot_dir = '../../plots/Plots_9-year/Low_energy_range0/'

name = 'LogParabola_n(500GeV)_l_in_(0,10).pdf'
fn = plot_dir + name 
#pyplot.yscale('log')
pyplot.ylim(1.5, 4.5)
auxil.setup_figure_pars(plot_type = 'spectrum')
pyplot.savefig(fn, format = 'pdf')
