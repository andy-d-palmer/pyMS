__author__ = 'palmer'
import numpy as np

def apply_normalisation(counts,type_str=""):
    """
    helper function to apply a normalisation function (with some input testing etc)
    :param counts: numpy array of values to normalise
    :param type_str: normalisation type to apply (name)
    :return: numpy array of normalised counts
    """
    normToApply = {"none": none,
                 "tic":tic,
                 "rms":rms,
                 "mad":mad,
                 "sqrt":sqrt}
    if type_str not in normToApply.keys():
        raise ValueError("{} not in {}".format(type_str,normToApply.keys()))
    counts = np.asarray(counts,dtype=float)
    return normToApply[type_str](counts)

def shift_and_scale(counts,scale=1.0,shift=0.0):
    """
    applys the generic scaling a shifting operation
    :param counts: numpy array
    :param scale: float
    :param shift: float
    :return: numpy array of scaled and shifted values
    """
    scale = float(scale)
    if scale == 0:
        return np.zeros(np.shape(counts))
    if shift > 0:
        counts -= shift
    return (counts)/scale


def check_zeros(counts):
    """
    helper function to check if vector is all zero
    :param counts:
    :return: bool
    """
    if sum(counts) == 0:
        return True
    else:
        return False

def none(counts):
    """
    does nothing, just returns input. is a dummy for programmatic case where a function must be supplied
    :param counts:  numpy array
    :return: counts:
    """
    return np.asarray(counts,dtype=float)


def tic(counts):
    """
    normalisation function, divides each intensity by the sum of all intensities (each spectrum sums to 1)
    :param counts: numpy array
    :return:counts normalised: numpy array
    """
    if check_zeros(counts):
        return counts
    return shift_and_scale(counts,scale=np.sum(counts))


def rms(counts):
    """
    normalisation function, divides each intensity by the root-mean-square of all intensities
    :param counts: numpy array
    :return:counts normalised: numpy array
    """
    if check_zeros(counts):
        return counts
    return shift_and_scale(counts,scale=np.sqrt(np.mean(np.square(counts,dtype=float))))


def mad(counts):
    """
    normalisation function, divides each intensity by the median-absolute-deviation of all intensities
    :param counts: numpy array
    :return:counts normalised: numpy array
    """
    if check_zeros(counts):
        return counts
    return shift_and_scale(counts,scale=np.median(np.abs(counts - np.median(counts))))



def sqrt(counts):
    """
    normalisation function, returns the square root of intensities
    :param counts: numpy array
    :return:counts normalised: numpy array
    """
    if check_zeros(counts):
        return counts
    counts_norm = np.sqrt(counts)
    counts_norm[counts==0] = 0
    return counts_norm


