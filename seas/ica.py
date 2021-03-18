import numpy as np
from datetime import datetime
from sklearn.decomposition import FastICA
from scipy import linalg
import os
import re
from timeit import default_timer as timer
from seas.waveletAnalysis import waveletAnalysis
from seas.signal import butterworth


from seas.hdf5manager import hdf5manager
# import matplotlib.pyplot as plt


def project(vector, shape, roimask=None, savepath=None, 
    n_components=None, svd_multiplier=None, calc_residuals=True):
    '''
    Find eigenvectors of a 2D vector, projecting along the first 
    dimension.  Generally, we want the first dimension to be the spatial 
    component of our matrix.  PCA_project returns a list with sorted 
    significance of the PC vectors, and a 3D reshaped eigenbrain matrix 
    for easy spatial visualization.  This matrix has dimensions 
    (PCcomponent, x, y).
    '''
    print('\nCalculating Eigenspace\n-----------------------')
    assert(vector.ndim == 2), ('vector was not a two-dimensional np array.'
        'If input is a movie, be sure to convert shape to (xy, t)')

    if roimask is not None:
        print('Using roimask to crop video')
        assert roimask.size == vector.shape[0], \
        'Vector was not the same size as the cropped mask'

        print('Original size:', vector.shape)
        maskind = np.where(roimask.flat == 1)
        vector = vector[maskind]
        print('Reduced size:', vector.shape)

    mean = np.mean(vector, 0).flatten()
    vector = vector - mean

    components = {}
    components['mean'] = mean
    components['roimask'] = roimask
    components['shape'] = shape

    if svd_multiplier is None:
        svd_multiplier = 5


    if vector.dtype == np.float16:
        vector = vector.astype('float32', copy=False)

    if n_components is None: 
        print('Calculating ICA (with n_component SVD estimator)...')

        t0 = timer()
        try:
            u, ev, _ = linalg.svd(vector, full_matrices=False)
        except ValueError:
            u, ev, _ = linalg.svd(vector, full_matrices=False, 
                lapack_driver='gesvd') # LAPACK error if matricies are too big

        components['svd_eigval'] = ev
        ev -= ev.min()
        ev = ev/ev.sum()
        integrate = np.cumsum(ev)
        x = np.arange(ev.size)

        p = np.polyfit(x, integrate, deg=2)
        y = np.polyval(p,x)

        cross_1 = np.where(integrate > y)[0][0]
        cross_2 = np.where(integrate[cross_1:] < y[cross_1:])[0][0] + cross_1

        # n_components = min(cross_1*svd_multiplier, cross_2 // 2)
        n_components = cross_1*svd_multiplier

        components['increased_cutoff'] = 0

        while True:
            print('\nCalculating ICA with', n_components, 'components...')

            w_init = u[:n_components,:n_components].astype('float64')
            ica = FastICA(n_components=n_components, random_state=1000, w_init=w_init)

            eig_vec = ica.fit_transform(vector)
            eig_mix = ica.mixing_

            noise, cutoff = sortNoise(eig_mix.T)

            p_signal = (1 - noise.sum() / noise.size)*100

            if noise.size == shape[0]: # all components are being used
                break 
            elif p_signal < 75:
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                break
            elif n_components >= shape[0]:
                print('ICA components were under 75% signal ({0}% signal).'\
                    .format(p_signal))
                print('However, number of components is maxed out.')
                print('Using this decomposition...')
                break
            else:
                print('ICA components were over 75% signal ({0}% signal).'\
                    .format(p_signal))
                print('Recalculating with more components...' )
                n_components += n_components // 2
                components['increased_cutoff'] += 1

                if n_components > shape[0]:
                    print('\nComponents maxed out!')
                    print('\tAttempted:', n_components)
                    n_components = shape[0]
                    print('\tReduced to:', shape[0])

        components['lag1_full'] = tca.lagNAutoCorr(eig_mix.T, 1)
        components['svd_multiplier'] = svd_multiplier

        try:
            print('Cropping excess noise components')
            components['svd_cutoff'] = n_components
            reduced_n_components = int((noise.size - noise.sum()) * 1.25)

            print('reduced_n_components:', reduced_n_components)

            if reduced_n_components < n_components:
                print('Cropping', n_components, 'to', reduced_n_components)

                ev_sort = np.argsort(eig_mix.std(axis=0))
                eig_vec = eig_vec[:,ev_sort][:,::-1]
                eig_mix = eig_mix[:, ev_sort][:,::-1]
                noise = noise[ev_sort][::-1]

                eig_vec = eig_vec[:,:reduced_n_components]
                eig_mix = eig_mix[:,:reduced_n_components]
                n_components = reduced_n_components
                noise = noise[:reduced_n_components]

                components['lag1_full'] = components['lag1_full'][ev_sort][::-1]
            else:
                print('Less than 75% signal.  Not cropping excess noise.')

        except Exception as e:
            print('Error cropping!!')
            print('\t', e)

        components['noise_components'] = noise
        components['cutoff'] = cutoff
        t = timer()-t0
        print('Independent Component Analysis took: {0} sec'.format(t))


    else: 
        print('Calculating ICA (' + str(n_components)+' components)...')

        t0 = timer()
        ica = FastICA(n_components=n_components, random_state=1000)

        try:
            eig_vec = ica.fit_transform(vector)  # Eigenbrains
        except ValueError: 
            print('Calculation exceeded float32 maximum.')
            print('Trying again with float64 vector...')
            #value error if any value exceeds float32 maximum.  
            #overcome this by converting to float64 
            eig_vec = ica.fit_transform(vector.astype('float64'))

        t = timer()-t0
        print('Independent Component Analysis took: {0} sec'.format(t))
        eig_mix = ica.mixing_  # Get estimated mixing matrix


    print('components shape:', eig_vec.shape)

    ev_sort = np.argsort(eig_mix.std(axis=0))
    eig_vec = eig_vec[:,ev_sort][:,::-1]
    eig_mix = eig_mix[:, ev_sort][:,::-1]
    eig_val = eig_mix.std(axis=0)

    components['eig_mix'] = eig_mix
    components['timecourses'] = eig_mix.T

    n_components = eig_vec.shape[1]
    components['eig_vec'] = eig_vec
    components['eig_val'] = eig_val
    components['n_components'] = n_components
    components['time'] = t

    if calc_residuals:
        try:
            vector = vector.astype('float64')
            rebuilt = PCA_rebuild(components, artifact_components='none', vector=True).T

            rebuilt -= rebuilt.mean(axis=0)
            vector -= vector.mean(axis=0)

            residuals = np.abs(vector - rebuilt)
            
            residuals_temporal = residuals.mean(axis=0)

            if roimask is not None:
                residuals_spatial = np.zeros(roimask.shape)
                residuals_spatial.flat[maskind] = residuals.mean(axis=1)
            else:
                residuals_spatial = np.reshape(residuals.mean(axis=1), (shape[1], shape[2]))
                
            components['residuals_spatial'] = residuals_spatial
            components['residuals_temporal'] = residuals_temporal
        
        except Exception as e:
            print('Residual Calculation Failed!!')
            print('\t', e)

    # Save information about how and when movie was filtered in dictionary
    filtermeta = {}
    filtermeta['date'] = \
        datetime.now().strftime('%Y%m%d')[2:]
    fmt = '%Y-%m-%dT%H:%M:%SZ'
    filtermeta['tstmp'] = \
        datetime.now().strftime(fmt)
    filtermeta['n_components'] = n_components
    components['filter'] = filtermeta

    if savepath is not None:
        f = hdf5manager(savepath)
        f.save(components)
    else:
        print('No path provided, not saving components')

    print('\n')
    return components


def filter_mean(mean, filtermethod='wavelet', low_cutoff=0.5):

    print('Highpass filter signal timecourse: ' + str(low_cutoff) + 'Hz')
    print('Filter method:', filtermethod)

    if filtermethod == 'butterworth':
        variance = mean.var()
        mean_filtered = butterworth(mean, low=low_cutoff)
        percent_variance = np.round(mean.var() / variance * 100)
        print(str(percent_variance) + '% variance retained')

    elif filtermethod == 'wavelet':
        wavelet = waveletAnalysis(mean.astype('float64'), fps=10)
        mean_filtered = wavelet.noiseFilter(upperPeriod=1/low_cutoff)

    else:
        raise Exception("Filter method '" + str(filtermethod)\
         + "' not supported!\n\t Supported methods: butterworth, wavelet")

    return mean_filtered
