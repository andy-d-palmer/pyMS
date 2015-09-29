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

# values are taken from http://www.ciaaw.org/pubs/TICE-2009.pdf
PeriodicTable ={
'H': [1, 1, [1.007825032, 2.014101778], [0.999885, 0.000115]],
'He': [2, 0, [3.016029319, 4.002603254], [1.34e-06, 0.99999866]],
'Li': [3, 1, [6.0151223, 7.0160041], [0.0759, 0.9241]],
'Be': [4, 2, [9.0121822], [1.0]],
'B': [5, 3, [10.0129371, 11.0093055], [0.199, 0.801]],
'C': [6, -4, [12.0, 13.00335484], [0.9893, 0.0107]],
'N': [7, 5, [14.00307401, 15.00010897], [0.99636, 0.00364]],
'O': [8, -2, [15.99491462, 16.9991315, 17.9991604], [0.99757, 0.00038, 0.00205]],
'F': [9, -1, [18.9984032], [1.0]],
'Ne': [10, 0, [19.99244018, 20.99384668, 21.99138511], [0.9048, 0.0027, 0.0925]],
'Na': [11, 1, [22.98976966], [1.0]],
'Mg': [12, 2, [23.98504187, 24.985837, 25.982593], [0.7899, 0.1, 0.1101]],
'Al': [13, 3, [26.98153863], [1.0]],
'Si': [14, 4, [27.97692649, 28.97649468, 29.97377018], [0.92223, 0.04685, 0.03092]],
'P': [15, 5, [30.97376149], [1.0]],
'S': [16, -2, [31.97207073, 32.97145854, 33.96786687, 35.96708088], [0.9499, 0.0075, 0.0425, 0.0001]],
'Cl': [17, -1, [34.96885271, 36.9659026], [0.7576, 0.2424]],
'Ar': [18, 0, [35.96754511, 37.9627324, 39.96238312], [0.003336, 0.000629, 0.996035]],
'K': [19, 1, [38.9637069, 39.96399867, 40.96182597], [0.932581, 0.000117, 0.067302]],
'Ca': [20, 2, [39.9625912, 41.9586183, 42.9587668, 43.9554811, 45.9536927, 47.952533], [0.96941, 0.00647, 0.00135, 0.02086, 4e-05, 0.00187]],
'Sc': [21, 3, [44.9559119], [1.0]],
'Ti': [22, 4, [45.9526316, 46.9517631, 47.9479463, 48.94787, 49.9447912], [0.0825, 0.0744, 0.7372, 0.0541, 0.0518]],
'V': [23, 5, [49.9471585, 50.9439595], [0.0025, 0.9975]],
'Cr': [24, 2, [49.9460442, 51.9405075, 52.9406494, 53.9388804], [0.04345, 0.83789, 0.09501, 0.02365]],
'Mn': [25, 2, [54.9380451], [1.0]],
'Fe': [26, 3, [53.9396147, 55.9349418, 56.9353983, 57.9332801], [0.05845, 0.91754, 0.02119, 0.00282]],
'Co': [27, 2, [58.933195], [1.0]],
'Ni': [28, 2, [57.9353429, 59.9307864, 60.931056, 61.9283451, 63.927966], [0.68077, 0.26223, 0.011399, 0.036346, 0.009255]],
'Cu': [29, 2, [62.9295975, 64.9277895], [0.6915, 0.3085]],
'Zn': [30, 2, [63.9291422, 65.9260334, 66.9271273, 67.9248442, 69.9253193], [0.4917, 0.2773, 0.0404, 0.1845, 0.0061]],
'Ga': [31, 3, [68.9255736, 70.9247013], [0.60108, 0.39892]],
'Ge': [32, 2, [69.9242474, 71.9220758, 72.9234589, 73.9211778, 75.9214026], [0.2057, 0.2745, 0.0775, 0.3650, 0.0773]],
'As': [33, 3, [74.9215965], [1.0]],
'Se': [34, 4, [73.9224764, 75.9192136, 76.919914, 77.9173091, 79.9165213, 81.9166994], [0.0089, 0.0937, 0.0763, 0.2377, 0.4961, 0.0873]],
'Br': [35, -1, [78.9183379, 80.916291], [0.5069, 0.4931]],
'Kr': [36, 0, [77.9203648, 79.916379, 81.9134836, 82.914136, 83.911507, 85.91061073], [0.00355, 0.02286, 0.11593, 0.115, 0.56987, 0.17279]],
'Rb': [37, 1, [84.91178974, 86.90918053], [0.7217, 0.2783]],
'Sr': [38, 2, [83.913425, 85.9092602, 86.9088771, 87.9056121], [0.0056, 0.0986, 0.07, 0.8258]],
'Y': [39, 3, [88.9058483], [1.0]],
'Zr': [40, 4, [89.9047044, 90.9056458, 91.9050408, 93.9063152, 95.9082734], [0.5145, 0.1122, 0.1715, 0.1738, 0.028]],
'Nb': [41, 5, [92.9063781], [1.0]],
'Mo': [42, 6, [91.906811, 93.9050883, 94.9058421, 95.9046795, 96.9060215, 97.9054082, 99.90747], [0.1453, 0.0915, 0.1584, 0.1667, 0.0960, 0.2439, 0.0982]],
'Tc': [43, 2, [96.9064], [1.0]],
'Ru': [44, 3, [95.907598, 97.905287, 98.9059393, 99.9042195, 100.9055821, 101.9043493, 103.905433], [0.0554, 0.0187, 0.1276, 0.126, 0.1706, 0.3155, 0.1862]],
'Rh': [45, 2, [102.905504], [1.0]],
'Pd': [46, 2, [101.905609, 103.904036, 104.905085, 105.903486, 107.903892, 109.905153], [0.0102, 0.1114, 0.2233, 0.2733, 0.2646, 0.1172]],
'Ag': [47, 1, [106.905097, 108.904752], [0.51839, 0.48161]],
'Cd': [48, 2, [105.906459, 107.904184, 109.9030021, 110.9041781, 111.9027578, 112.9044017, 113.9033585, 115.904756], [0.0125, 0.0089, 0.1249, 0.128, 0.2413, 0.1222, 0.2873, 0.0749]],
'In': [49, 3, [112.904058, 114.903878], [0.0429, 0.9571]],
'Sn': [50, 4, [111.904818, 113.902779, 114.903342, 115.901741, 116.902952, 117.901603, 118.903308, 119.9021947, 121.903439, 123.9052739], [0.0097, 0.0066, 0.0034, 0.1454, 0.0768, 0.2422, 0.0859, 0.3258, 0.0463, 0.0579]],
'Sb': [51, 3, [120.9038157, 122.904214], [0.5721, 0.4279]],
'Te': [52, 4, [119.90402, 121.9030439, 122.90427, 123.9028179, 124.9044307, 125.9033117, 127.9044631, 129.9062244], [0.0009, 0.0255, 0.0089, 0.0474, 0.0707, 0.1884, 0.3174, 0.3408]],
'I': [53, -1, [126.904473], [1.0]],
'Xe': [54, 0, [123.905893, 125.904274, 127.9035313, 128.9047794, 129.903508, 130.9050824, 131.9041535, 133.9053945, 135.907219], [0.000952, 0.00089, 0.019102, 0.264006, 0.04071, 0.212324, 0.269086, 0.104357, 0.088573]],
'Cs': [55, 1, [132.9054519], [1.0]],
'Ba': [56, 2, [129.9063208, 131.9050613, 133.9045084, 134.9056886, 135.9045759, 136.9058274, 137.9052472], [0.00106, 0.00101, 0.02417, 0.06592, 0.07854, 0.11232, 0.71698]],
'La': [57, 3, [137.907112, 138.9063533], [0.0008881, 0.9991119]],
'Ce': [58, 3, [135.907172, 137.905991, 139.9054387, 141.909244], [0.00185, 0.00251, 0.8845, 0.11114]],
'Pr': [59, 3, [140.9076528], [1.0]],
'Nd': [60, 3, [141.9077233, 142.9098143, 143.9100873, 144.9125736, 145.9131169, 147.916893, 149.920891], [0.27152, 0.12174, 0.23798, 0.08293, 0.17189, 0.05756, 0.05638]],
'Pm': [61, 3, [144.9127], [1.0]],
'Sm': [62, 3, [143.911999, 146.9148979, 147.9148227, 148.9171847, 149.9172755, 151.9197324, 153.9222093], [0.0307, 0.1499, 0.1124, 0.1382, 0.0738, 0.2675, 0.2275]],
'Eu': [63, 3, [150.9198502, 152.9212303], [0.4781, 0.5219]],
'Gd': [64, 3, [151.919791, 153.9208656, 154.922622, 155.9221227, 156.9239601, 157.9241039, 159.9270541], [0.002, 0.0218, 0.148, 0.2047, 0.1565, 0.2484, 0.2186]],
'Tb': [65, 4, [158.9253468], [1.0]],
'Dy': [66, 3, [155.924283, 157.924409, 159.9251975, 160.9269334, 161.9267984, 162.9287312, 163.9291748], [0.00056, 0.00095, 0.02329, 0.18889, 0.25475, 0.24896, 0.2826]],
'Ho': [67, 3, [164.9303221], [1.0]],
'Er': [68, 3, [161.928778, 163.9292, 165.9302931, 166.9320482, 167.9323702, 169.9354643], [0.00139, 0.01601, 0.33503, 0.22869, 0.26978, 0.1491]],
'Tm': [69, 3, [168.9342133], [1.0]],
'Yb': [70, 3, [167.933897, 169.9347618, 170.9363258, 171.9363815, 172.9382108, 173.9388621, 175.9425717], [0.00123, 0.02982, 0.1409, 0.2168, 0.16103, 0.32026, 0.12996]],
'Lu': [71, 3, [174.9407718, 175.9426863], [0.97401, 0.02599]],
'Hf': [72, 4, [173.940046, 175.9414086, 176.9432207, 177.9436988, 178.9458161, 179.94655], [0.0016, 0.0526, 0.186, 0.2728, 0.1362, 0.3508]],
'Ta': [73, 5, [179.9474648, 180.9479958], [0.0001201, 0.9998799]],
'W': [74, 6, [179.946704, 181.9482042, 182.950223, 183.9509312, 185.9543641], [0.0012, 0.265, 0.1431, 0.3064, 0.2843]],
'Re': [75, 2, [184.952955, 186.9557531], [0.374, 0.626]],
'Os': [76, 4, [183.9524891, 185.9538382, 186.9557505, 187.9558382, 188.9581475, 189.958447, 191.9614807], [0.0002, 0.0159, 0.0196, 0.1324, 0.1615, 0.2626, 0.4078]],
'Ir': [77, 4, [190.960594, 192.9629264], [0.373, 0.627]],
'Pt': [78, 4, [189.959932, 191.961038, 193.9626803, 194.9647911, 195.9649515, 197.967893], [0.00012, 0.00782, 0.3286, 0.3378, 0.2521, 0.07356]],
'Au': [79, 3, [196.9665687], [1.0]],
'Hg': [80, 2, [195.965833, 197.966769, 198.9682799, 199.968326, 200.9703023, 201.970643, 203.9734939], [0.0015, 0.0997, 0.1687, 0.231, 0.1318, 0.2986, 0.0687]],
'Tl': [81, 1, [202.9723442, 204.9744275], [0.2952, 0.7048]],
'Pb': [82, 2, [203.9730436, 205.9744653, 206.9758969, 207.9766521], [0.014, 0.241, 0.221, 0.524]],
'Bi': [83, 3, [208.9803987], [1.0]],
'Po': [84, 4, [209.0], [1.0]],
'At': [85, 7, [210.0], [1.0]],
'Rn': [86, 0, [220.0], [1.0]],
'Fr': [87, 1, [223.0], [1.0]],
'Ra': [88, 2, [226.0], [1.0]],
'Ac': [89, 3, [227.0], [1.0]],
'Th': [90, 4, [232.0380553], [1.0]],
'Pa': [91, 4, [231.035884], [1.0]],
'U': [92, 6, [234.0409521, 235.0439299, 238.0507882], [5.4e-05, 0.007204, 0.992742]],
'Np': [93, 5, [237.0], [1.0]],
'Pu': [94, 3, [244.0], [1.0]],
'Am': [95, 2, [243.0], [1.0]],
'Cm': [96, 3, [247.0], [1.0]],
'Bk': [97, 3, [247.0], [1.0]],
'Cf': [98, 0, [251.0], [1.0]],
'Es': [99, 0, [252.0], [1.0]],
'Fm': [100, 0, [257.0], [1.0]],
'Md': [101, 0, [258.0], [1.0]],
'No': [102, 0, [259.0], [1.0]],
'Lr': [103, 0, [262.0], [1.0]],
'Rf': [104, 0, [261.0], [1.0]],
'Db': [105, 0, [262.0], [1.0]],
'Sg': [106, 0, [266.0], [1.0]],
'Ee':[0,0,[0.000548597],[1.0]]}

mass_electron = 0.00054857990924
 
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
    m -= charges * mass_electron
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
def isodist(molecules,charges=0,output='',plot=False,sigma=0.35,resolution=250,cutoff=0.0001,do_centroid=True,verbose=False):

    exit = checkhelpcall(molecules)
    save = checkoutput(output)
    if exit==True:
        sys.exit(0) 

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
    if do_centroid:
        pts = resolution2pts(min(final.keys()),max(final.keys()),resolution)
        xvector,yvector=genGaussian(final,sigma,pts)
        ms_output.add_spectrum(xvector,yvector)
        mz_list,intensity_list,centroid_list = gradient(ms_output.get_spectrum()[0],ms_output.get_spectrum()[1],max_output=-1,weighted_bins=5)
        ms_output.add_centroids(mz_list,intensity_list)
    else: 
        mz_idx = sorted(final.keys())
        ms_output.add_centroids(mz_idx,[final[f] for f in mz_idx])
    if plot==True:
        import matplotlib.pyplot as plt #for plotting        
        plt.plot(xvector,yvector)
        plt.plot(mz_list,intensity_list,'rx')
        plt.show()

    if save==True:
        g=open(savefile,'w')
        xs=xvector.tolist()
        ys=yvector.tolist()
        for i in range(0,len(xs)):
            g.write(str(xs[i])+"\t"+str(ys[i])+"\n")
        g.close
    return ms_output


def str_to_el(str_in):
    import re
    atom_number = re.split('([A-Z][a-z]*)', str_in)
    el = {}
    for atom, number in zip(atom_number[1::2], atom_number[2::2]):
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
        mult = str_in[rb + 1]
        mult_idx = 1
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

def complex_to_simple(str_in):
    el_dict = process_sf(str_in)
    if any([e<0 for e in el_dict.values()]):
        return None
    sf_str = "".join(["{}{}".format(a,el_dict[a]) for a in el_dict if el_dict[a]>0])
    return sf_str