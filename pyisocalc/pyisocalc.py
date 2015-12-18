#!/usr/bin/env python
#########################################################################
# Author: Andy Ohlin (debian.user@gmx.com)
# Modified by: Andrew Palmer (palmer@embl.de)
#              Artem Tarasov (lomereiter@gmail.com)
#
# Example usage:
# pyisocalc('Fe(ClO3)5',plot=false,gauss=0.25,charge=-2,resolution=250)
# Do "pyisocalc('--help') to find out more
#
##########################################################################
ver='0.2 (5 Sep. 2015)'
# Version 0.2 -- Modified from v0.8 of Andy Ohlin's code
#
# Dependencies:
# python2.7, python-numpy, python-matplotlib
# pyms-mass_spectrum
#
# Isotopic abundances and masses were copied from Wsearch32.
# Elemental oxidation states were mainly
# taken from Matthew Monroe's molecular weight calculator, with some changes.
#########################################################################

import re #for regular expressions
import sys
import time #for code execution analysis
import numpy as np
from numpy import shape,asarray,prod,zeros,repeat #for cartesian product
from numpy import random,histogram # for binning
from numpy import pi,sqrt,exp,true_divide,multiply # misc math functions
from numpy import linspace #for gaussian
from numpy import copysign
from itertools import groupby, imap
import operator
from ..mass_spectrum import mass_spectrum as MassSpectrum
from ..centroid_detection import gradient

# values updated on 16/12/2015 from http://www.ciaaw.org/isotopic-abundances.htm / http://www.ciaaw.org/atomic-masses.htm
PeriodicTable ={
'H': [1, 1, [1.007825032, 2.014101778], [0.999855, 0.000145]],
'He': [2, 0, [3.01602932, 4.002603254], [0.000002, 0.999998]],
'Li': [3, 1, [6.015122887, 7.01600344], [0.0485, 0.9515]],
'Be': [4, 2, [9.0121831], [1.0]],
'B': [5, 3, [10.012937, 11.009305], [0.1965, 0.8035]],
'C': [6, -4, [12, 13.00335484], [0.9894, 0.0106]],
'N': [7, 5, [14.003074, 15.0001089], [0.996205, 0.003795]],
'O': [8, -2, [15.99491462, 16.99913176, 17.99915961], [0.99757, 0.0003835, 0.002045]],
'F': [9, -1, [18.99840316], [1.0]],
'Ne': [10, 0, [19.99244018, 20.9938467, 21.9913851], [0.9048, 0.0027, 0.0925]],
'Na': [11, 1, [22.98976928], [1.0]],
'Mg': [12, 2, [23.9850417, 24.985837, 25.982593], [0.78965, 0.10011, 0.11025]],
'Al': [13, 3, [26.9815385], [1.0]],
'Si': [14, 4, [27.97692654, 28.97649467, 29.97377001], [0.922545, 0.04672, 0.030735]],
'P': [15, 5, [30.973762], [1.0]],
'S': [16, -2, [31.97207117, 32.97145891, 33.967867, 35.967081], [0.9485, 0.00763, 0.04365, 0.000158]],
'Cl': [17, -1, [34.9688527, 36.9659026], [0.758, 0.242]],
'Ar': [18, 0, [35.9675451, 37.962732, 39.96238312], [0.003336, 0.000629, 0.996035]],
'K': [19, 1, [38.96370649, 39.9639982, 40.96182526], [0.932581, 0.000117, 0.067302]],
'Ca': [20, 2, [39.9625909, 41.958618, 42.958766, 43.955482, 45.95369, 47.9525228], [0.96941, 0.00647, 0.00135, 0.02086, 0.00004, 0.00187]],
'Sc': [21, 3, [44.955908], [1.0]],
'Ti': [22, 4, [45.952628, 46.951759, 47.947942, 48.947866, 49.944787], [0.0825, 0.0744, 0.7372, 0.0541, 0.0518]],
'V': [23, 5, [49.947156, 50.943957], [0.0025, 0.9975]],
'Cr': [24, 2, [49.946042, 51.940506, 52.940648, 53.938879], [0.04345, 0.83789, 0.09501, 0.02365]],
'Mn': [25, 2, [54.938044], [1.0]],
'Fe': [26, 3, [53.939609, 55.934936, 56.935393, 57.933274], [0.05845, 0.91754, 0.02119, 0.00282]],
'Co': [27, 2, [58.933194], [1.0]],
'Ni': [28, 2, [57.935342, 59.930786, 60.931056, 61.928345, 63.927967], [0.680769, 0.262231, 0.011399, 0.036345, 0.009256]],
'Cu': [29, 2, [62.929598, 64.92779], [0.6915, 0.3085]],
'Zn': [30, 2, [63.929142, 65.926034, 66.927128, 67.924845, 69.92532], [0.4917, 0.2773, 0.0404, 0.1845, 0.0061]],
'Ga': [31, 3, [68.925574, 70.924703], [0.60108, 0.39892]],
'Ge': [32, 2, [69.924249, 71.9220758, 72.923459, 73.92117776, 75.9214027], [0.2052, 0.2745, 0.0776, 0.3652, 0.0775]],
'As': [33, 3, [74.921595], [1.0]],
'Se': [34, 4, [73.9224759, 75.9192137, 76.9199142, 77.917309, 79.916522, 81.9167], [0.0086, 0.0923, 0.076, 0.2369, 0.498, 0.0882]],
'Br': [35, -1, [78.918338, 80.91629], [0.5065, 0.4935]],
'Kr': [36, 0, [77.920365, 79.916378, 81.913483, 82.914127, 83.91149773, 85.91061063], [0.00355, 0.02286, 0.11593, 0.115, 0.56987, 0.17279]],
'Rb': [37, 1, [84.91178974, 86.90918053], [0.7217, 0.2783]],
'Sr': [38, 2, [83.913419, 85.909261, 86.908878, 87.905613], [0.0056, 0.0986, 0.07, 0.8258]],
'Y': [39, 3, [88.90584], [1.0]],
'Zr': [40, 4, [89.9047, 90.90564, 91.90503, 93.90631, 95.90827], [0.5145, 0.1122, 0.1715, 0.1738, 0.028]],
'Nb': [41, 5, [92.90637], [1.0]],
'Mo': [42, 6, [91.906808, 93.905085, 94.905839, 95.904676, 96.906018, 97.905405, 99.907472], [0.14649, 0.09187, 0.15873, 0.16673, 0.09582, 0.24292, 0.09744]],
'Tc': [43, 2, [97.90721], [1.0]],
'Ru': [44, 3, [95.90759, 97.90529, 98.905934, 99.904214, 100.905577, 101.904344, 103.90543], [0.0554, 0.0187, 0.1276, 0.126, 0.1706, 0.3155, 0.1862]],
'Rh': [45, 2, [102.9055], [1.0]],
'Pd': [46, 2, [101.9056, 103.904031, 104.90508, 105.90348, 107.903892, 109.905172], [0.0102, 0.1114, 0.2233, 0.2733, 0.2646, 0.1172]],
'Ag': [47, 1, [106.90509, 108.904755], [0.51839, 0.48161]],
'Cd': [48, 2, [105.90646, 107.904183, 109.903007, 110.904183, 111.902763, 112.904408, 113.903365, 115.904763], [0.01245, 0.00888, 0.1247, 0.12795, 0.24109, 0.12227, 0.28754, 0.07512]],
'In': [49, 3, [112.904062, 114.9038788], [0.04281, 0.95719]],
'Sn': [50, 4, [111.904824, 113.902783, 114.9033447, 115.901743, 116.902954, 117.901607, 118.903311, 119.902202, 121.90344, 123.905277], [0.0097, 0.0066, 0.0034, 0.1454, 0.0768, 0.2422, 0.0859, 0.3258, 0.0463, 0.0579]],
'Sb': [51, 3, [120.90381, 122.90421], [0.5721, 0.4279]],
'Te': [52, 4, [119.90406, 121.90304, 122.90427, 123.90282, 124.90443, 125.90331, 127.904461, 129.9062228], [0.0009, 0.0255, 0.0089, 0.0474, 0.0707, 0.1884, 0.3174, 0.3408]],
'I': [53, -1, [126.90447], [1.0]],
'Xe': [54, 0, [123.90589, 125.9043, 127.903531, 128.9047809, 129.9035094, 130.905084, 131.9041551, 133.905395, 135.9072145], [0.00095, 0.00089, 0.0191, 0.26401, 0.04071, 0.21232, 0.26909, 0.10436, 0.08857]],
'Cs': [55, 1, [132.905452], [1.0]],
'Ba': [56, 2, [129.90632, 131.905061, 133.904508, 134.905688, 135.904576, 136.905827, 137.905247], [0.0011, 0.001, 0.0242, 0.0659, 0.0785, 0.1123, 0.717]],
'La': [57, 3, [137.90712, 138.90636], [0.0008881, 0.9991119]],
'Ce': [58, 3, [135.907129, 137.90599, 139.90544, 141.90925], [0.00186, 0.00251, 0.88449, 0.11114]],
'Pr': [59, 3, [140.90766], [1.0]],
'Nd': [60, 3, [141.90773, 142.90982, 143.91009, 144.91258, 145.91312, 147.9169, 149.9209], [0.27153, 0.12173, 0.23798, 0.08293, 0.17189, 0.05756, 0.05638]],
'Pm': [61, 3, [144.91276], [1.0]],
'Sm': [62, 3, [143.91201, 146.9149, 147.91483, 148.91719, 149.91728, 151.91974, 153.92222], [0.0308, 0.15, 0.1125, 0.1382, 0.0737, 0.2674, 0.2274]],
'Eu': [63, 3, [150.91986, 152.92124], [0.4781, 0.5219]],
'Gd': [64, 3, [151.9198, 153.92087, 154.92263, 155.92213, 156.92397, 157.92411, 159.92706], [0.002, 0.0218, 0.148, 0.2047, 0.1565, 0.2484, 0.2186]],
'Tb': [65, 4, [158.92535], [1.0]],
'Dy': [66, 3, [155.92428, 157.92442, 159.9252, 160.92694, 161.92681, 162.92874, 163.92918], [0.00056, 0.00095, 0.02329, 0.18889, 0.25475, 0.24896, 0.2826]],
'Ho': [67, 3, [164.93033], [1.0]],
'Er': [68, 3, [161.92879, 163.92921, 165.9303, 166.93205, 167.93238, 169.93547], [0.00139, 0.01601, 0.33503, 0.22869, 0.26978, 0.1491]],
'Tm': [69, 3, [168.93422], [1.0]],
'Yb': [70, 3, [167.93389, 169.93477, 170.93633, 171.93639, 172.93822, 173.93887, 175.94258], [0.00126, 0.03023, 0.14216, 0.21754, 0.16098, 0.31896, 0.12887]],
'Lu': [71, 3, [174.94078, 175.94269], [0.97401, 0.02599]],
'Hf': [72, 4, [173.94005, 175.94141, 176.94323, 177.94371, 178.94582, 179.94656], [0.0016, 0.0526, 0.186, 0.2728, 0.1362, 0.3508]],
'Ta': [73, 5, [179.94746, 180.948], [0.0001201, 0.9998799]],
'W': [74, 6, [179.94671, 181.948204, 182.950223, 183.950931, 185.95436], [0.0012, 0.265, 0.1431, 0.3064, 0.2843]],
'Re': [75, 2, [184.952955, 186.95575], [0.374, 0.626]],
'Os': [76, 4, [183.952489, 185.95384, 186.95575, 187.95584, 188.95814, 189.95844, 191.96148], [0.0002, 0.0159, 0.0196, 0.1324, 0.1615, 0.2626, 0.4078]],
'Ir': [77, 4, [190.96059, 192.96292], [0.373, 0.627]],
'Pt': [78, 4, [189.95993, 191.96104, 193.962681, 194.964792, 195.964952, 197.96789], [0.00012, 0.00782, 0.32864, 0.33775, 0.25211, 0.07356]],
'Au': [79, 3, [196.966569], [1.0]],
'Hg': [80, 2, [195.96583, 197.966769, 198.968281, 199.968327, 200.970303, 201.970643, 203.973494], [0.0015, 0.1004, 0.1694, 0.2314, 0.1317, 0.2974, 0.0682]],
'Tl': [81, 1, [202.972345, 204.974428], [0.29515, 0.70485]],
'Pb': [82, 2, [203.973044, 205.974466, 206.975897, 207.976653], [0.014, 0.241, 0.221, 0.524]],
'Bi': [83, 3, [208.9804], [1.0]],
'Po': [84, 4, [209], [1.0]],
'At': [85, 7, [210], [1.0]],
'Rn': [86, 0, [222], [1.0]],
'Fr': [87, 1, [223], [1.0]],
'Ra': [88, 2, [226], [1.0]],
'Ac': [89, 3, [227], [1.0]],
'Th': [90, 4, [230.03313, 232.03806], [0.0002, 0.9998]],
'Pa': [91, 4, [231.03588], [1.0]],
'U': [92, 6, [234.04095, 235.04393, 238.05079], [0.000054 , 0.007204, 0.992742]],
'Ee':[0,0,[0.000548597],[1.0]]}
from collections import namedtuple
FormulaSegment = namedtuple('FormulaSegment', ['atom', 'number'])

#######################################
# Collect properties
#######################################
def getAverageMass(segment):
    masses, ratios = PeriodicTable[segment.atom][2:4]
    atomic_mass = np.dot(masses, ratios)
    return atomic_mass * segment.number

def getCharge(segment):
    atomic_charge = PeriodicTable[segment.atom][1]
    return atomic_charge * segment.number

#####################################################
# Iterate over expanded formula to collect property
#####################################################
def getSegments(formula):
    segments = re.findall('([A-Z][a-z]*)([0-9]*)',formula)
    for atom, number in segments:
        number = int(number) if number else 1
        yield FormulaSegment(atom, number)

def molmass(formula):
    return sum(imap(getAverageMass, getSegments(formula)))

def molcharge(formula):
    return sum(imap(getCharge, getSegments(formula)))

################################################################################
#expands ((((M)N)O)P)Q to M*N*O*P*Q
################################################################################

def formulaExpander(formula):
    while len(re.findall('\(\w*\)',formula))>0:
        parenthetical=re.findall('\(\w*\)[0-9]+',formula)
        for i in parenthetical:
            p=re.findall('[0-9]+',str(re.findall('\)[0-9]+',i)))
            j=re.findall('[A-Z][a-z]*[0-9]*',i)
            oldj=j
            for n in range(0,len(j)):
                numero=re.findall('[0-9]+',j[n])
                if len(numero)!=0:
                    for k in numero:
                        nu=re.sub(k,str(int(int(k)*int(p[0]))),j[n])
                else:
                    nu=re.sub(j[n],j[n]+p[0],j[n])
                j[n]=nu
            newphrase=""
            for m in j:
                newphrase+=str(m)
            formula=formula.replace(i,newphrase)
        if (len((re.findall('\(\w*\)[0-9]+',formula)))==0) and (len(re.findall('\(\w*\)',formula))!=0):
            formula=formula.replace('(','')
            formula=formula.replace(')','')
    lopoff=re.findall('[A-Z][a-z]*0',formula)
    if lopoff!=[]:
        formula=formula.replace(lopoff[0],'')
    return formula

def singleElementPattern(segment, threshold=1e-9):
    # see 'Efficient Calculation of Exact Fine Structure Isotope Patterns via the
    #      Multidimensional Fourier Transform' (A. Ipsen, 2014)
    element, amount = segment.atom, segment.number
    iso_mass, iso_abundance = map(np.array, PeriodicTable[element][2:4])
    if len(iso_abundance) == 1:
        return np.array([1.0]), iso_mass * amount
    if amount == 1:
        return iso_abundance, iso_mass
    dim = len(iso_abundance) - 1
    abundance = np.zeros([amount + 1] * dim)
    abundance.flat[0] = iso_abundance[0]
    abundance.flat[(amount+1)**np.arange(dim)] = iso_abundance[-1:0:-1]
    abundance = np.real(np.fft.ifftn(np.fft.fftn(abundance) ** amount))
    significant = np.where(abundance > threshold)
    intensities = abundance[significant]
    masses = amount * iso_mass[0] + (iso_mass[1:] - iso_mass[0]).dot(significant)
    return intensities, masses

def trim(ry, my):
    my, inv = np.unique(my, return_inverse=True)
    ry = np.bincount(inv, weights=ry)
    return ry, my

def cartesian(rx, mx, cutoff):
    ry, my = asarray(rx[0]), asarray(mx[0])
    for i in xrange(1, len(rx)):
        newr = np.outer(rx[i], ry).ravel()
        newm = np.add.outer(mx[i], my).ravel()
        js = np.where(newr > cutoff)[0]
        ry, my = newr[js], newm[js]
    return trim(ry, my)

def isotopes(segments, cutoff):
    patterns = [singleElementPattern(x, cutoff) for x in segments]
    ratios = [x[0] for x in patterns]
    masses = [x[1] for x in patterns]
    return cartesian(ratios,masses,cutoff)

##################################################################################
# Does housekeeping to generate final intensity ratios and puts it into a dictionary
##################################################################################
def genDict(m,n,charges,cutoff):
    m, n = np.asarray(m).round(8), np.asarray(n).round(8)
    filter = n > cutoff
    m, n = m[filter], n[filter]
    n *= 100.0 / max(n)
    m -= charges * PeriodicTable['Ee'][2]
    m /= abs(charges)
    return dict(zip(m, n))

def genGaussian(final,sigma, pts):
    mzs = np.array(final.keys())
    intensities = np.array(final.values())
    xvector = np.linspace(min(mzs)-1,max(mzs)+1,pts)
    yvector = intensities.dot(exp(-0.5 * (np.add.outer(mzs, -xvector)/sigma)**2))
    yvector *= 100.0 / max(yvector)
    return (xvector,yvector)

def mz(a,b,c):
    if c==0:
        c=b
        if b==0:
            c=1
    mz=a/c
    return mz

def checkhelpcall(sf):
    print_help = False
    exit = False
    if sf=='--help':
        exit = True
    if sf == '':
        print_help = True
    if print_help:
        print " "
        print "\t\tThis is pyisocalc, an isotopic pattern calculator written in python (2.x)."
        print "\t\tGet the latest version from http://sourceforge.net/p/pyisocalc"
        print "\t\tThis is version",ver
        print "\tUsage:"
        print "\t-h\t--help   \tYou're looking at it."
        print "\t-f\t--formula\tFormula enclosed in apostrophes, e.g. 'Al2(NO3)4'."
        print "\t-c\t--charge\tCharge, e.g. -2 or 3. Must be an integer. If not provided the charge will be calculated"
        print "\t  \t        \tbased on default oxidation states as defined in this file."
        print "\t-o\t--output\tFilename to save data into. The data will be saved as a tab-separated file. No output by default."
        print "\t-p\t--plot   \tWhether to plot or not. Can be yes, YES, Yes, Y, y. Default is no."
        print "\t-g\t--gauss  \tGaussian broadening factor (affects resolution). Default is 0.35. Lower value gives higher resolution."
        print "\t  \t         \tAdjust this factor to make the spectrum look like the experimentally observed one."
        print "\t-r\t--resolution\tNumber of points to use for the m/z axis (affects resolution). Default is 500. Higher is slower."
        print " "
        print "\t Example:"
        print "\t./pyisocalc.py -f 'Fe(ClO3)5' -p y -g 0.25 -o ironperchlorate.dat -c -2 -r 250"
        print ""
        exit=True
    return exit
def resolution2pts(min_x,max_x,resolution):
    # turn resolving power into ft pts
    # resolution = fwhm/max height
    # turn resolution in points per mz then multipy by mz range
    pts = resolution/1000 * (max(max_x-min_x,1))
    return pts
def checkoutput(output):
    save = True
    if output == '':
        save = False
    return save
########
# main function#
########
def isodist(molecules,charges=0,output='',plot=False,sigma=0.05,resolution=50000,cutoff=0.0001,do_centroid=True,verbose=False):

    #exit = checkhelpcall(molecules)
    #save = checkoutput(output)
    #if exit==True:
    #    sys.exit(0)

    molecules=molecules.split(',')
    for element in molecules:
        element=formulaExpander(element)
        if verbose:
            print ('The mass of %(substance)s is %(Mass)f and the calculated charge is %(Charge)d with m/z of %(Mz)f.' % {'substance': \
            element, 'Mass': molmass(element), 'Charge': molcharge(element),'Mz':mz(molmass(element),molcharge(element),charges)})

    segments = list(getSegments(element))

    if charges==0:
        charges=sum(getCharge(x) for x in segments)
        if charges==0:
            charges=1
    else:
        if verbose:
            print "Using user-supplied charge of %d for mass spectrum" % charges

    ratios, masses = isotopes(segments, cutoff)
    final = genDict(masses, ratios, charges, cutoff)

    ms_output = MassSpectrum()
    pts = resolution2pts(min(final.keys()),max(final.keys()),resolution)
    xvector,yvector=genGaussian(final,sigma,pts)
    ms_output.add_spectrum(xvector,yvector)
    if do_centroid:
        mz_list,intensity_list,centroid_list = gradient(ms_output.get_spectrum()[0],ms_output.get_spectrum()[1],max_output=-1,weighted_bins=5)
        ms_output.add_centroids(mz_list,intensity_list)
    else:
        ms_output.add_centroids(np.array(final.keys()),np.array(final.values()))

    if plot==True:
        import matplotlib.pyplot as plt #for plotting        
        plt.plot(xvector,yvector)
        plt.plot(mz_list,intensity_list,'rx')
        plt.show()

    #if save==True:
    #    g=open(savefile,'w')
    #    xs=xvector.tolist()
    #    ys=yvector.tolist()
    #    for i in range(0,len(xs)):
    #        g.write(str(xs[i])+"\t"+str(ys[i])+"\n")
    #    g.close
    return ms_output


def str_to_el(str_in):
    import re
    atom_number = re.split('([A-Z][a-z]*)', str_in)
    el = {}
    for atom, number in zip(atom_number[1::2], atom_number[2::2]):
        if atom not in PeriodicTable:
            raise ValueError("Element not recognised: {} in {}".format(atom, str_in))
        if number == '':
            number = '1'
        number = int(number)
        if not atom in el:
            el[atom] = number
        else:
            el[atom] += number
    return el


def rm_1bracket(str_in):
    # find first and last brackets
    rb = str_in.index(')')
    lb = str_in[0:rb].rindex('(')

    # check if multiplier after last bracket
    if len(str_in) == rb + 1:  # end of string
        mult = "1"
        mult_idx = 0
    else:
        mult = str_in[rb + 1:]
        mult_idx = len(mult)
    if not mult.isdigit():  # not a number
        mult = '1'
        mult_idx = 0
    # exband brackets
    str_tmp = ""
    for m in range(0, int(mult)):
        str_tmp = str_tmp + str_in[lb + 1:rb]
    if lb == 0:
        str_strt = ""
    else:
        str_strt = str_in[0:lb]
    if rb == len(str_in) - 1:
        str_end = ""
    else:
        str_end = str_in[rb + 1 + mult_idx:]
    return str_strt + str_tmp + str_end


def strip_bracket(str_in):
    go = True
    try:
        while go == True:
            str_in = rm_1bracket(str_in)
    except ValueError as e:
        if str(e) != "substring not found":
            raise
    return str_in


def process_sf(str_in):
    import re
    # split_sign
    sub_strings = re.split('([\+-])', str_in)
    if not sub_strings[0] in (('+', '-')):
        sub_strings = ["+"] + sub_strings
    el = {}
    for sign, sf in zip(sub_strings[0::2], sub_strings[1::2]):
        # remove brackets
        str_in = strip_bracket(sf)
        # count elements
        el_ = str_to_el(str_in)
        for atom in el_:
            number = int('{}1'.format(sign)) * el_[atom]
            if not atom in el:
                el[atom] = number
            else:
                el[atom] += number
    return el


def process_complexes(str_in):
    """
    Function splits strings at '.' and moves any preceding number to the end
    so A.nB -> A+(B)n
    :param str_in: molecular formula that may or may not contain complexes
    :return: reformatted string
    """
    def _move_num_to_end(s):
        # move initial numbers to end
        alpha_idx = [ss.isalpha() for ss in s].index(True)
        str_re = "({}){}".format(s[alpha_idx:],s[0:alpha_idx])
        return str_re
    str_in = str_in.split(".")
    str_out = ["{}".format(s)  if s[0].isalpha()   else _move_num_to_end(s)  for s in str_in ]
    str_out =  "+".join(str_out)
    return str_out


def prep_str(str_in):
    str_in = process_complexes(str_in) #turn A.nB into A+(B)n
    str_in.split("+")
    return str_in

def complex_to_simple(str_in):
    str_in = prep_str(str_in)
    el_dict = process_sf(str_in)
    if any((all([e==0 for e in el_dict.values()]),any([e<0 for e in el_dict.values()]))):
        return None
    sf_str = "".join(["{}{}".format(a,el_dict[a]) for a in el_dict if el_dict[a]>0])
    return sf_str