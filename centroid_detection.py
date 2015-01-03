import numpy as np
def gradient(mzs,intensities,**opt_args):
    function_args = {'max_output': -1, 'weighted_bins':1,'min_intensity':0}
    for key, val in opt_args.iteritems(): 
        if key in function_args.keys():
            function_args[key] = val
        else:
            print 'possible arguments:'
            for i in function_args.keys():
                   print i
            raise NameError('gradient does not take argument: %s'%(key))
    mzMaxNum = function_args['max_output']
    weighted_bins = function_args['weighted_bins']
    min_intensity = function_args['min_intensity']
    #calc first differential
    MZgrad = np.gradient(intensities)
    #calc second differential
    MZgrad2 = np.gradient(MZgrad)
    """
    #It could either be the left or the right index so check which is the
    #actual maximum value
    indiciesRight = indicies + 1
    indiciesLeft = indicies - 1

    toKeep = indiciesRight<len(intensities)
    indicies= indicies[toKeep]
    indiciesRight=indiciesRight[toKeep]
    indiciesLeft=indiciesLeft[toKeep]

    toKeep = indiciesLeft>=0
    indicies= indicies[toKeep]
    indiciesRight=indiciesRight[toKeep]
    indiciesLeft=indiciesLeft[toKeep]
    collected = np.argmax(np.concatenate((intensities[indicies], intensities[indiciesRight], intensities[indiciesLeft]),axis=1), axis=1)
    indicies = np.unique(np.concatenate(indicies[collected == 0],indiciesRight[collected == 1], indiciesLeft[collected == 2]))

"""    
    indices = MZgrad[0:-2]*MZgrad[1:-1]<=0 # detect crossing points
    indices[MZgrad2[indices]<0] = False #pick maxima
    indices = np.concatenate(([0],indices,[0]))==1
    indices_list = np.transpose(np.where(indices==True))
    
    #Remove any 'peaks' that aren't real
    indices_list = indices_list[intensities[indices_list] > min_intensity]

    #Select the peaks    
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
            mzs_list = np.concatenate((mzs_list, np.zeros((lengthDiff,1))))
            intensities_list  = np.concatenate((intensities_list, np.zeros((lengthDiff,1))))
            indices_list = np.concatenate((indices_list, np.zeros((lengthDiff,1))))
    
    if weighted_bins > 0:
        bin_shift = range(-weighted_bins,weighted_bins+1)
        for ii in range(0,len(mzs_list)):
            bin_idx = bin_shift+indices_list[ii]
            mzs_list[ii] = np.average(mzs[bin_idx],weights=intensities[bin_idx])
            indices_list[ii] = indices_list[ii] + np.argmax(intensities[bin_idx]) - weighted_bins
            intensities_list[ii] = intensities[indices_list[ii]]
            #mzs_list[ii] = np.mean(mzs[bin_idx])
    return  (mzs_list,intensities_list,indices_list)
