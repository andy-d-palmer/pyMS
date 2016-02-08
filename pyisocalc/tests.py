__author__ = 'palmer'
import unittest
import numpy as np
import sys
sys.path.append('/Users/palmer/Documents/python_codebase')
from pyMS.pyisocalc import pyisocalc

class PyisocalcTest(unittest.TestCase):
    def test_correct_element_parsing(self):
        str_el = {"m+H":{"H":1}, #m: bug/hack, lower case letters will be ignored
                  "m-H":{"H":-1},
                  "m+H2":{"H":2},
                  "m-H2":{"H":-2},
                  "m-H2O+K":{"H":-2,"O":-1,"K":1},
                  "(C1H2O3)1-H1":{"C":1,"H":1,"O":3},
                  "Fe(ClO3)5":{"Fe":1,"Cl":5,"O":15},
                  "C5H10Ru.3Cl":{"C":5,"H":10,"Ru":1,"Cl":3},}

        for s in str_el:
            str_in = pyisocalc.prep_str(s)
            print str_in
            e_ = pyisocalc.process_sf(str_in)
            self.assertItemsEqual(e_.keys(),str_el[s].keys())
            for el in str_el[s]:
                self.assertEqual(str_el[s][el], e_[el],msg="{} != {}  in {}".format(str_el[s][el], e_[el],el))

