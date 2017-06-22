# transform a catalog to a dictionary and save
"""



python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/30MeV_7-5years/model+PS/No_PS_mask_diffuse_PS_MC_data_30MeV_100MeV_1bins_gridstep0.5l-0.5b_ds9data.list -o data/catalogs/30MeV_7-5years/dictcats/No_PS_mask_diffuse_PS_MC_data_30MeV_100MeV_1bins_gridstep0.5l-0.5b_ds9data.yaml -t PGwave2 -e 30-100
python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/1000-3000/counts_map_3D.list -o data/catalogs/dictcats/PGwave_PSlist_en_1000_3000_flux.yaml -t PGwave -e 1000-3000

type=3FGL_simul
python scripts/pylib/pylib_diffuse/cat2dict.py -i /nfs/farm/g/glast/u/mdimauro/LogNLogS1GeV/simulationdima/files/incatalog_bpl.fits -o data/catalogs/GC_simulation/3FGL-simulation_allsky.yaml -t 3FGL_simul


set type=PGwave
foreach er (100-300 300-1000 1000-3000 3000-10000 10000-200000)
set infile=data/catalogs/"$er"/counts_map_3D.list
set outfile=data/catalogs/dictcats/PGwave_PSlist_"$er"MeV.yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t PGwave -e $er
end

python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/3000-10000/counts_map_3D.list -o data/catalogs/dictcats/PGwave_PSlist_en_3000_10000.yaml -t PGwave -e 3000_10000

type=PGwavefin
infile=models/PS/PSGalCentList_copy.txt
outfile=models/PS/GC_PGwave_PSlist.yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t PGwave

type=3FGL
infile=data/catalogs/gll_psc_v14.fit
outfile=data/catalogs/dictcats/3FGL_dict.yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t $type

python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/gll_psc_v14.fit -o data/catalogs/dictcats/3FGL_dict_err.yaml -t 3FGL

type=3FGLnew
python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/30MeV_7-5years/gll_psc_v16.fit -o data/catalogs/30MeV_7-5years/dictcats/3FGL_dict_30_100MeV_flux_100-300_extension.yaml -t 3FGLnew


type=4FGL
python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/4FGL_Tobi-comp/_afs_slac_g_glast_groups__catalog_pointlike_skymodels_P302_7years_uw985_P302_7years_uw985.fits -o data/catalogs/4FGL_Tobi-comp/dictcats/4FGL_prelim_P302_7years_uw985.yaml -t 4FGL_prel
python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/4FGL_Tobi-comp/_afs_slac_g_glast_groups__catalog_pointlike_skymodels_P302_7years_uw985_P302_7years_uw985.fits -o data/catalogs/30MeV_7-5years/dictcats/4FGL_prelim_P302_7years_30-100MeV_uw985.yaml -t 4FGL_30-100MeV


foreach TS (10 25)
set type=gttscube
set infile=data/catalogs/listsources_complete_candidates_gtlike_all_TS"$TS".txt
set outfile=data/catalogs/dictcats/gttscubeTS"$TS"_PS_list.yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t $type
end

python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/listsources_complete_candidates_all_TS10-2.txt -o data/catalogs/dictcats/gttscubeTS10_PS_err_list.yaml -t gttscube

set type=MRfilter
foreach fn (MRFSeeds-CEL_noEXT_2sigmapeak2 MRFSeeds-CEL_noEXT_energy PGWSeeds-CEL_noEXT)
set infile=data/catalogs/MRfilter/"$fn".txt
set outfile=data/catalogs/dictcats/"$fn".yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t $type
end

type=MRfilter
for fn in MRFSeeds-CEL_noEXT_2sigmapeak MRFSeeds-CEL_noEXT_energy PGWSeeds-CEL_noEXT
do
infile=data/catalogs/MRfilter/"$fn".txt
outfile=data/catalogs/dictcats/"$fn".yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t $type
done


python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/listsources_complete_candidates_gtlike_all_TS25.txt -o data/catalogs/dictcats/DiMauroListTS25.yaml -t gttscube

type=DiMauroListTS10
infile=data/catalogs/listsources_complete_candidates_all_TS10.txt
outfile=data/catalogs/dictcats/DiMauroListTS10.yaml
python scripts/pylib/pylib_diffuse/cat2dict.py -i $infile -o $outfile -t $type

python scripts/pylib/pylib_diffuse/cat2dict.py -i data/catalogs/listsources_complete_candidates_gtlike_all_TS10.txt -o data/catalogs/dictcats/DiMauroListTS10.yaml -t DiMauroListTS10
"""

from optparse import OptionParser
import os
import numpy as np
import pyfits
#import healpy
#import matplotlib
#matplotlib.use("agg")
#from matplotlib import pyplot

import auxil_gcfit as auxil
import dio
import ps_util
import math



# input from command line
parser = OptionParser()

parser.add_option("-i", "--infile", dest="infile",
                  default='',
                  help="input file")

parser.add_option("-o", "--outfile", dest="outfile",
                  default='',
                  help="output file")

parser.add_option("-t", "--type", dest="type",
                  default='',
                  help="input type")


parser.add_option("-e", "--enbin", dest="enbin",
                  default='',
                  help="energy bin")


(options, args) = parser.parse_args()

# input and output options
infn = options.infile
outfn = options.outfile
intype = options.type
enbin = options.enbin

if intype == 'PGwavefin':
    dt = np.dtype([('ind', np.int), ('b', np.float), ('l', np.float), ('pos_err', np.float),
                   ('nass', np.int), ('cats', 'S60')])
    data = np.loadtxt(infn, dtype=dt)
    res_dict = {}
    for key in ['b', 'l']:
        res_dict[key] = data[key]
    cats = []
    for i in range(len(data)):
        cat_list = data['cats'][i]
        print cat_list
        #cat_list = cat_list.replace('3003', '300 3')
        #cat_list = cat_list.replace('3001', '300 1')
        #cat_list = cat_list.split(' ')
        cat_list = cat_list.split(',')
        print cat_list
        cats.append(cat_list)
    res_dict['cats'] = cats
    #print cats
    #print res_dict.keys()
    dio.savedict(res_dict, outfn)

if intype == 'MRfilter':
    dt = np.dtype([('name', 'S60'), ('ra', np.float), ('dec', np.float), ('sigma', np.float), \
                   ('pos_err', np.float), ('y', np.float), ('cats', 'S60')])
    
    res_dict = {}
    
    # format the input to be readable by numpy
    print 'format the input to be readable by numpy'
    inf = open(infn)
    instr = inf.read()
    inf.close()
    instr = instr.replace('Name', '#Name')
    instr = instr.replace(') (', ')(')
    instr = instr.replace(', ', ',')
    infn_tmp = infn.replace('.txt', '_temp.txt')
    inf_tmp = open(infn_tmp, 'w')
    inf_tmp.write(instr)
    inf_tmp.flush()
    inf_tmp.close()
    
    # import the data and remove the tmp file
    data = np.loadtxt(infn_tmp, dtype=dt)
    os.remove(infn_tmp)
    
    # transform from ra-dec to galactic coordinates
    print 'transform from ra-dec to galactic coordinates'
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    ctab = SkyCoord(ra=data['ra'] * u.degree, dec=data['dec'] * u.degree)

    res_dict['b'] = ctab.galactic.b.value
    res_dict['l'] = ctab.galactic.l.value
    for key in ['name', 'ra', 'dec', 'pos_err', 'sigma']:
        res_dict[key] = data[key]
    
    dio.savedict(res_dict, outfn)
    if 0:
        print data['name'][:10]
        dict_test = dio.loaddict(outfn)
        print dict_test['name'][:10]

if intype == "3FGL":
    hdu	= pyfits.open(infn)
    bs = hdu[1].data.field('GLAT')
    ls = hdu[1].data.field('GLON')
    errs = hdu[1].data.field('Conf_68_SemiMajor')  #or  'Conf_68_PosAng'
    flux = hdu[1].data.field('Flux1000_3000')
    errflux = hdu[1].data.field('Unc_Flux1000_3000')
    res_dict = {}
    res_dict['b'] = bs
    res_dict['l'] = ls
    res_dict['err'] = errs
    res_dict['cats'] = [intype] * len(bs)
    res_dict['ERR_flux_1-3GeV'] = errflux
    res_dict['flux_1-3GeV'] = flux
    dio.savedict(res_dict, outfn)

if intype == "3FGLnew":
    hdu	= pyfits.open(infn)
    bs = hdu[1].data.field('GLAT')
    ls = hdu[1].data.field('GLON')
    errs = hdu[1].data.field('Conf_68_SemiMajor')  #or  'Conf_68_PosAng'
    fluxd = hdu[1].data.field('Flux_Density')
    unc_fluxd = hdu[1].data.field('Unc_Flux_Density')
    flux_100_300 = hdu[1].data.field('Flux100_300')
    ts = hdu[1].data.field('Sqrt_TS100_300')
    extension = hdu[1].data.field('Extended_Source_Name')
    n = range(0, len(hdu[1].data.field('GLAT')),1)
    flux = []
    errflux = []
    for i in n:
        func_i = ps_util.get_pssp(i, hdu[1], cat_name='3FGL')
        flux_i = func_i(0.05477)
        flux.append(flux_i)
        errflux.append(flux_i * (unc_fluxd[i]/fluxd[i]))
    res_dict = {}
    res_dict['b'] = bs
    res_dict['l'] = ls
    res_dict['err'] = errs
    res_dict['cats'] = [intype] * len(bs)
    res_dict['unc_flux'] = errflux
    res_dict['flux_100-300'] = flux_100_300
    res_dict['flux'] = flux
    res_dict['TS'] = ts
    res_dict['Extension_Name'] = extension
    print "saved TS"
    dio.savedict(res_dict, outfn)

if intype == "3FGL_simul":
    hdu	= pyfits.open(infn)
    bs = hdu[1].data.field('GLAT')
    ls = hdu[1].data.field('GLON')
    flux = hdu[1].data.field('Flux')
    res_dict = {}
    res_dict['b'] = bs
    res_dict['l'] = ls
    res_dict['cats'] = [intype] * len(bs)
    res_dict['flux_1-10GeV'] = flux
    print "saved 3FGL-simul"
    dio.savedict(res_dict, outfn)

#============
if intype == "4FGL_30-100MeV":
    #######
    hdu = pyfits.open(infn)
    b = []
    l = []
    err = []
    ts = []
    bs = hdu[1].data.field('GLAT')
    ls = hdu[1].data.field('GLON')
    errs = hdu[1].data.field('Conf_95_SemiMajor')  #or  'Conf_68_PosAng'
    fluxd = hdu[1].data.field('Flux_Density')
    tss = hdu[1].data.field('Test_Statistic')
    unc_fluxd = hdu[1].data.field('Unc_Flux_Density')
    n = range(0, len(hdu[1].data.field('GLAT')),1)
    
    thdu = hdu[1]
    n = range(0, len(bs),1)
    flux = []
    errflux = []
    for i in n:
        func_i = ps_util.get_pssp(i, hdu[1], cat_name='P8uw')
        flux_i = func_i(0.05477)
        err_flux_i = flux_i * (unc_fluxd[i]/fluxd[i])
            
        if (not math.isnan(flux_i)) and (not math.isnan(err_flux_i)) :
            b.append(float(bs[i]))
            l.append(float(ls[i]))
            err.append(float(errs[i]))
            ts.append(float(tss[i]))
            flux.append(float(flux_i))
            errflux.append(float(err_flux_i))
        else:
            print "eliminated nan"
        
    
    res_dict = {}
    res_dict['b'] = b
    res_dict['l'] = l
    res_dict['err'] = err
    res_dict['cats'] = [intype] * len(b)
    res_dict['unc_flux'] = errflux
    res_dict['flux'] = flux
    res_dict['TS'] = ts
    dio.savedict(res_dict, outfn)
#=============


#============
if intype == "4FGL_prel":
    #######
    Eb = np.logspace(0., 1., 100)
    Es = (Eb[1:] + Eb[:-1]) / 2
    print "Es", Es
    dE = (Eb[1:] - Eb[:-1])
    print "Es", dE
    hdu = pyfits.open(infn)
    bs = hdu[1].data.field('GLAT')
    ls = hdu[1].data.field('GLON')
    semi_ma =  hdu[1].data.field('Conf_95_SemiMajor')    
    flux1 = hdu[1].data.field('Flux_Density')            #photon/cm**2/MeV/s
    errflux1 = hdu[1].data.field('Unc_Flux_Density')     #photon/cm**2/MeV/s
    ts = hdu[1].data.field('Test_Statistic')
    ind = hdu[1].data.field('Exp_Index')
    spectr = hdu[1].data.field('SpectrumType')
    
    thdu = hdu[1]
    n = range(0, len(bs),1)
    flux = []
    for i in n:
        ps_ind = i
        sp = ps_util.get_pssp(ps_ind, thdu, cat_name='P8uw')
        flux.append(np.sum(sp(Es) * dE))
    print "flux",flux
    
######
    
    res_dict = {}
    res_dict['b'] = bs
    res_dict['l'] = ls
    res_dict['cats'] = [intype] * len(bs)
    res_dict['flux'] = flux
    res_dict['flux_tot'] = flux1
    res_dict['unc_flux_tot'] = errflux1
    res_dict['TS'] = ts
    res_dict['exp_index'] = ind
    res_dict['spectral_type'] = spectr
    res_dict['err'] = semi_ma
    print "saved 4FGL-prel"
    dio.savedict(res_dict, outfn)
#=============

if intype == "fermipy":
    hdu	= pyfits.open(infn)
    bs = hdu[1].data.field('GLAT')
    ls = hdu[1].data.field('GLON')
    flux = hdu[1].data.field('flux')
    res_dict = {}
    res_dict['b'] = bs
    res_dict['l'] = ls
    res_dict['cats'] = [intype] * len(bs)
    res_dict['flux'] = flux
    print "saved fermipy dictionary:"
    dio.savedict(res_dict, outfn)



if intype == "gttscube":
    data = np.loadtxt(infn).T
    res_dict = {}
    res_dict['b'] = data[1]
    res_dict['l'] = data[0]
    res_dict['err'] = data[2]   #68%
    res_dict['TS'] = data[4]
    n = range(0, len(data[1]), 1)
    cat = []
    for i in n :
        cat.append([intype])
    res_dict['cats'] = cat
    dio.savedict(res_dict, outfn)

if intype == "gttscube_new":
    data = np.loadtxt(infn).T
    res_dict = {}
    res_dict['b'] = data[1]
    res_dict['l'] = data[0]
    res_dict['TS'] = data[2]
    res_dict['flux'] = data[6]
    res_dict['cats'] = [intype] * len(data[0])
    dio.savedict(res_dict, outfn)


if intype == "PGwave":
    data = np.loadtxt(infn).T
    print data.shape
    res_dict = {}
    res_dict['b'] = data[4]
    res_dict['l'] = data[3]
    res_dict['err'] = data[5]
    res_dict['SNR'] = data[6]
    res_dict['cats'] = [enbin] * data.shape[1]
    res_dict['flux'] = data[8]
    #print res_dict
    #exit()
    dio.savedict(res_dict, outfn)

if intype == "PGwave2":
    data = np.loadtxt(infn).T
    print data.shape
    
        
    res_dict = {}
    res_dict['x'] = data[1]
    res_dict['y'] = data[2]
    res_dict['b'] = data[4]
    res_dict['l'] = data[3]
    res_dict['err'] = data[5]
    res_dict['SNR'] = data[6]
    res_dict['K-signif'] = data[7]
    if isinstance(data[1],list):
        res_dict['cats'] = [enbin] * data.shape[1]
    else:
        res_dict['cats'] = ['30-100']
            
    res_dict['counts'] = data[8]
    res_dict['err_counts'] = data[9]
    dio.savedict(res_dict, outfn)
        