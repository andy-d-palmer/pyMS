__author__ = 'palmer'

def nosmooth(mzs,intensities):
    return intensities

def sg_smooth(mzs,intensities,n_smooth=1,w_size=5):
    import scipy.signal as signal
    for n in range(0,n_smooth):
        intensities = signal.savgol_filter(intensities,w_size,2)
    intensities[intensities<0]=0
    return intensities

def apodization(mzs,intensities,w_size=10):
    import scipy.signal as signal
    win = signal.hann(w_size)
    win = signal.slepian(w_size,0.3)
    intensities = signal.fftconvolve(intensities, win, mode='same') / sum(win)
    intensities[intensities<1e-6]=0
    return intensities


def rebin(mzs,intensities,delta_mz=0.1):
    import numpy as np
    n_bins = np.round((mzs[-1]-mzs[0])/delta_mz)
    new_mzs = np.linspace(mzs[0],mzs[-1]+delta_mz,n_bins)
    mz_idx = np.digitize(mzs,new_mzs[0:-1])
    new_intensities = np.bincount(mz_idx,weights=intensities,minlength=len(new_mzs))
    return new_mzs,new_intensities

def fast_change(mzs,intensities,diff_thresh=0.01):
    import numpy as np
    import scipy.signal as signal
    diff =  np.concatenate((np.abs(np.diff(intensities)),[1]))
    diff = signal.medfilt(diff)
    intensities[diff<diff_thresh] = 0
    return intensities

def median(mzs,intensities, w_size=3):
    import scipy.signal as signal
    return signal.medfilt(intensities,kernel_size=w_size)
