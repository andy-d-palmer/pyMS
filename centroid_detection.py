import numpy as np


def gradient(mzs, intensities, **opt_args):
    function_args = {'max_output': -1, 'weighted_bins': 1, 'min_intensity': 1e-5,'grad_type':'gradient'}
    for key, val in opt_args.iteritems():
        if key in function_args.keys():
            function_args[key] = val
        else:
            print 'possible arguments:'
            for i in function_args.keys():
                print i
            raise NameError('gradient does not take argument: %s' % (key))
    mzMaxNum = function_args['max_output']
    weighted_bins = function_args['weighted_bins']
    min_intensity = function_args['min_intensity']
    gradient_type=function_args['grad_type']
    assert mzMaxNum < len(mzs)
    assert len(mzs)==len(intensities)
    assert weighted_bins < len(mzs)/2.
    # calc first&sectond differential
    if gradient_type == 'gradient':
        MZgrad = np.gradient(intensities)
        MZgrad2 = np.gradient(MZgrad)[0:-1]
    elif gradient_type == 'diff':
        MZgrad = np.concatenate((np.diff(intensities),[1]))
        MZgrad2 = np.diff(MZgrad)
    else:
        raise ValueError('gradient type {} not known'.format(gradient_type))
    # detect crossing points
    cPoint = MZgrad[0:-1] * MZgrad[1:] <= 0
    mPoint = MZgrad2<0
    indices = cPoint & mPoint
    # bool->list of indices
    # Could check left/right of crossing point
    indices_list_l = np.transpose(np.where(indices==True))
    indices_list_r = indices_list_l+1
    intensites_list_l=intensities[indices_list_l]
    intensites_list_r=intensities[indices_list_r]
    indices_list = np.zeros((len(intensites_list_l),1),dtype=int)
    for ii in range(0,len(intensites_list_l)):
        if intensites_list_l[ii] > intensites_list_r[ii]:
            indices_list[ii] = indices_list_l[ii]
        else:
            indices_list[ii] = indices_list_r[ii]
    indices_list=np.unique(np.asarray(indices_list))
    # Remove any 'peaks' that aren't real
    indices_list = indices_list[intensities[indices_list] > min_intensity]
    # Select the peaks
    intensities_list = intensities[indices_list]
    mzs_list = mzs[indices_list]
    # Tidy up if required
    if mzMaxNum > 0:
        if len(mzs_list) > mzMaxNum:
            sort_idx = np.argsort(intensities_list)
            intensities_list = intensities_list[sort_idx[-mzMaxNum:]]
            mzs_list = mzs_list[sort_idx[-mzMaxNum:]]
            indices_list = indices_list[sort_idx[-mzMaxNum:]]
        elif len(mzs) < mzMaxNum:
            lengthDiff = mzMaxNum - len(indices_list)
            mzs_list = np.concatenate((mzs_list, np.zeros((lengthDiff, 1))))
            intensities_list = np.concatenate((intensities_list, np.zeros((lengthDiff, 1))))
            indices_list = np.concatenate((indices_list, np.zeros((lengthDiff, 1))))
    if weighted_bins > 0:
        # check no peaks within bin width of spectrum edge
        good_idx = (indices_list > weighted_bins) & (indices_list < (len(mzs) - weighted_bins))
        mzs_list = mzs_list[good_idx]
        intensities_list = intensities_list[good_idx]
        indices_list = indices_list[good_idx]
        bin_shift = range(-weighted_bins, weighted_bins + 1)
        for ii in range(0, len(mzs_list)):
            bin_idx = bin_shift + indices_list[ii]
            mzs_list[ii] = np.average(mzs[bin_idx], weights=intensities[bin_idx])
            indices_list[ii] = indices_list[ii] + np.argmax(intensities[bin_idx]) - (1+weighted_bins)/2
            intensities_list[ii] = intensities[indices_list[ii]]
            # mzs_list[ii] = np.mean(mzs[bin_idx])
    return (mzs_list, intensities_list, indices_list)
