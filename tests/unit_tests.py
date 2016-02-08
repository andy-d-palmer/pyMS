__author__ = 'palmer'
import unittest
import numpy as np
import sys
sys.path.append('/Users/palmer/Documents/python_codebase')
from pyMS.mass_spectrum import mass_spectrum


class mass_spectrum_TestIO(unittest.TestCase):

    def test_add_spectrum(self):
        mzs_counts_list = [ [[0,1,2],[2,2,2]],
                            [[0,0,0],[0,0,0]],
                            [[0,1,2], [2,]]
                            ]
        test_vals_should_add= [  True,
                                 True,
                                 False,
                               ]

        for mzs_counts,should_add in zip(mzs_counts_list,test_vals_should_add):
            this_spectrum = mass_spectrum()
            if should_add:
                this_spectrum.add_spectrum(mzs_counts[0],mzs_counts[1])
                self.assertItemsEqual(this_spectrum.mzs,mzs_counts[0])
                self.assertItemsEqual(this_spectrum.intensities,mzs_counts[1])
            else:
                with self.assertRaises(IOError):
                    this_spectrum.add_spectrum(mzs_counts[0],mzs_counts[1])

    def test_add_centroids(self):
        mzs_counts_list = [ [[0,1,2],[2,2,2]],
                            [[0,0,0],[0,0,0]],
                            [[0,1,2], [2,]]
                            ]
        test_vals_should_add= [  True,
                                 True,
                                 False,
                               ]
        for mzs_counts,should_add in zip(mzs_counts_list,test_vals_should_add):
            this_spectrum = mass_spectrum()
            if should_add:
                this_spectrum.add_centroids(mzs_counts[0],mzs_counts[1])
                self.assertItemsEqual(this_spectrum.centroids,mzs_counts[0])
                self.assertItemsEqual(this_spectrum.centroids_intensity,mzs_counts[1])
            else:
                with self.assertRaises(IOError):
                    this_spectrum.add_centroids(mzs_counts[0],mzs_counts[1])

    def test_get_profile_spectrum(self):
        mzs = [0,1,2,3]
        vals = [1,2,1,0]
        this_spectrum = mass_spectrum()
        this_spectrum.add_spectrum(mzs,vals)
        [mzs_,vals_] = this_spectrum.get_spectrum(source='profile')
        self.assertItemsEqual(mzs,mzs_)
        self.assertItemsEqual(vals,vals_)

    def test_get_centroid_spectrum(self):
        mzs = [0,1,2,3]
        vals = [1,2,1,0]
        this_spectrum = mass_spectrum()
        this_spectrum.add_centroids(mzs,vals)
        [mzs_,vals_] = this_spectrum.get_spectrum(source='centroids')
        self.assertItemsEqual(mzs,mzs_)
        self.assertItemsEqual(vals,vals_)

class mass_spectrum_TestNormalisation(unittest.TestCase):
    def make_spectrum(self,mzs,vals,centroid_mzs,centroid_vals):
        this_spectrum = mass_spectrum()
        this_spectrum.add_spectrum(mzs,vals)
        this_spectrum.add_centroids(centroid_mzs,centroid_vals)
        return this_spectrum

    def test_tic(self):
        mzs_vals_centroidMzs_centroidVals  = [[[0,1,2,3,4,5,6,7,8,9,10],[1,1,1,1,1,1,1,1,1,1,1],[2,4,6],[1,1,1]],
                                              ]
        for this_vars in mzs_vals_centroidMzs_centroidVals:
            mzs,vals,centroid_mzs,centroid_vals = this_vars
            this_spectrum = self.make_spectrum(mzs,vals,centroid_mzs,centroid_vals)
            this_spectrum.normalise_spectrum(method='tic')
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='profile')[1]),1)
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='centroids')[1]),1)

    def test_sqrt(self):
        mzs_vals_centroidMzs_centroidVals  = [
                    [[0,1,2,3,4,5,6,7,8,9,10],[1,1,1,1,1,1,1,1,1,1,1],[2,4,6],[1,1,1]],
                    [[1,1,1,1,1,1,1,1,1,1,1],[0,1,2,3,4,5,6,7,8,9,10],[1,1,1],[2,4,6]],
                    [[1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0 ,0],[1,1,1],[0,0,0]],
            ]
        expected_sum_vals = [[11.0, 3.0],
                             [22.46827, 5.8637],
                             [0.0,0.0]]
        for this_vars, this_result in zip(mzs_vals_centroidMzs_centroidVals, expected_sum_vals):
            mzs,vals,centroid_mzs,centroid_vals = this_vars
            this_spectrum = self.make_spectrum(mzs,vals,centroid_mzs,centroid_vals)
            this_spectrum.normalise_spectrum(method='sqrt')
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='profile')[1]),this_result[0],places=4)
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='centroids')[1]),this_result[1],places=4)

    def test_mad(self):
        mzs_vals_centroidMzs_centroidVals  = [
                    [[0,1,2,3,4,5,6,7,8,9,10],[1,1,1,1,1,1,1,1,1,1,1],[2,4,6],[1,1,1]],
                    [[1,1,1,1,1,1,1,1,1,1,1],[0,1,2,3,4,5,6,7,8,9,10],[1,1,1],[2,4,6]],
                                              ]
        expected_sum_vals = [[0.0, 0.0],
                             [18.33333, 6.0]]
        for this_vars, this_result in zip(mzs_vals_centroidMzs_centroidVals, expected_sum_vals):
            mzs,vals,centroid_mzs,centroid_vals = this_vars
            this_spectrum = self.make_spectrum(mzs,vals,centroid_mzs,centroid_vals)
            this_spectrum.normalise_spectrum(method='mad')
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='profile')[1]),this_result[0],places=4)
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='centroids')[1]),this_result[1],places=4)

    def test_rms(self):
        mzs_vals_centroidMzs_centroidVals  = [[[0,1,2,3,4,5,6,7,8,9,10],[1,1,1,1,1,1,1,1,1,1,1],[2,4,6],[1,1,1]],
                                              [[1,1,1,1,1,1,1,1,1,1,1],[0,1,2,3,4,5,6,7,8,9,10],[1,1,1],[2,4,6]]]
        expected_sum_vals = [[11.0, 3.0],
                             [9.29669, 2.77746]]
        for this_vars, this_result in zip(mzs_vals_centroidMzs_centroidVals, expected_sum_vals):
            mzs,vals,centroid_mzs,centroid_vals = this_vars
            this_spectrum = self.make_spectrum(mzs,vals,centroid_mzs,centroid_vals)
            this_spectrum.normalise_spectrum(method='rms')
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='profile')[1]),this_result[0],places=4)
            self.assertAlmostEqual(np.sum(this_spectrum.get_spectrum(source='centroids')[1]),this_result[1],places=4)

import pyMS.normalisation as normalisation
class normalisation_TestFunctions(unittest.TestCase):
    def test_none(self):
        counts_list = [[0,0,0,0,0],
                       [1,2,3,4,5]]
        vals_list = [[0,0,0,0,0],
                     [1,2,3,4,5]]
        for counts,vals in zip(counts_list,vals_list):
            counts_ = normalisation.none(counts)
            self.assertEqual(len(counts),len(counts_))
            np.testing.assert_array_almost_equal(counts_,vals)


    def test_tic(self):
        counts_list = [np.asarray([0,0,0,0,0]),
                       np.asarray([1,2,3,4,5])]
        vals_list = [[0,0,0,0,0],
                     [n/15. for n in [1,2,3,4,5]]
                    ]
        for counts,vals in zip(counts_list,vals_list):
            counts_ = normalisation.tic(counts)
            self.assertEqual(len(counts),len(counts_))
            np.testing.assert_array_almost_equal(counts_,vals)

    def test_rms(self):
        counts_list = [np.asarray([0,0,0,0,0]),
                       np.asarray([3,3,3]),
                       ]
        vals_list = [[0,0,0,0,0],
                     [1,1,1]
                     ]
        for counts,vals in zip(counts_list,vals_list):
            counts_ = normalisation.rms(counts)
            self.assertEqual(len(counts),len(counts_))
            np.testing.assert_array_almost_equal(counts_,vals)


    def test_mad(self):
        counts_list = [np.asarray([0,0,0,0,0]),
                       np.asarray([1,2,3,4,5])]
        vals_list = [[0,0,0,0,0],
                     [1,2,3,4,5]]
        for counts,vals in zip(counts_list,vals_list):
            counts_ = normalisation.mad(counts)
            self.assertEqual(len(counts),len(counts_))
            np.testing.assert_array_almost_equal(counts_,vals)


    def test_sqrt(self):
        counts_list = [[0.,0.,0.,0.,0.],
                       ]
        vals_list = [[0.,0.,0.,0.,0.],
                     ]
        for counts,vals in zip(counts_list,vals_list):
            counts_ = normalisation.sqrt(counts)
            self.assertEqual(len(counts),len(counts_))
            np.testing.assert_array_almost_equal(counts_,vals)


    #def test_apply_normalisation(counts,type_str=""):
    #    normToApply = {"none": none,
    #                 "tic":tic,
    #                 "rms":rms,
    #                 "mad":mad,
    #                 "sqrt":sqrt}
    #    if type_str not in normToApply.keys():
    #        raise ValueError("{} not in {}".format(type_str,normToApply.keys()))
    #    return normToApply[type_str](counts)