from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import numpy as np
import math
from itertools import compress


def sort_noise(timecourses=None,
               lag1=None,
               return_logpdf=False,
               method='KDE',
               verbose=False):
    '''
    Sorts timecourses into two clusters (signal and noise) based on 
    lag-1 autocorrelation.  
    Timecourses should be a np array of shape (n, t).

    Returns noise_components, a np array with 1 value for all noise 
    timecourses detected, as well as the cutoff value detected
    '''
    if method == 'KDE':

        # calculate lag autocorrelations
        if lag1 is None:
            assert timecourses is not None, 'sortNoise requires either timecourses or lag1'
            lag1 = lag_n_autocorr(timecourses, 1)

        # calculate minimum between min and max peaks
        kde_skl = KernelDensity(kernel='gaussian',
                                bandwidth=0.05).fit(lag1[:, np.newaxis])
        x_grid = np.linspace(-0.2, 1.2, 1200)

        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

        maxima = argrelextrema(np.exp(log_pdf), np.greater)[0]
        if len(maxima) <= 1:
            if verbose:
                print('Only one cluster found')
            cutoff = 0
        else:
            cutoff_index = np.argmin(np.exp(log_pdf)[maxima[0]:maxima[-1]]) \
                + maxima[0]
            cutoff = x_grid[cutoff_index]
            if verbose:
                print('autocorr cutoff:', cutoff)

        noise_components = (lag1 < cutoff).astype('uint8')
    else:
        raise Exception('method: {0} is unknown!'.format(method))

    if return_logpdf:
        return noise_components, cutoff, log_pdf
    else:
        return noise_components, cutoff


def get_peak_separation(log_pdf, x_grid=None):

    if x_grid is None:
        x_grid = np.linspace(-0.2, 1.2, 1200)

    maxima = argrelextrema(np.exp(log_pdf), np.greater)[0]

    if len(maxima) > 2:
        maxima = np.delete(maxima, np.argmin(np.exp(log_pdf)[maxima]))
    peak_separation = x_grid[maxima[-1]] - x_grid[maxima[0]]

    return peak_separation


def lag_n_autocorr(x, n, verbose=True):

    if x.ndim == 1:
        return np.corrcoef(x[n:], x[:-n])[0, 1]
    elif x.ndim == 2:
        if verbose:
            print('calculating {0}-lag autocorrelation'.format(n),
                  'along first dimension:', x.shape)
        nt = x.shape[0]
        corrmatrix = np.corrcoef(x[:, n:], x[:, :-n])
        return np.diag(corrmatrix[:nt, nt:])
    else:
        print('Invalid input!!')
        raise AssertionError


def butterworth(data, high=None, low=None, fps=10, order=5):
    from scipy import signal

    def butter_highpass(cutoff, fps, order=order):
        nyq = 0.5 * fps
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_lowpass(cutoff, fps, order=order):
        nyq = 0.5 * fps
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    if low is not None:
        b, a = butter_highpass(low, fps, order=order)
        data = signal.filtfilt(b, a, data)

    if high is not None:
        b, a = butter_lowpass(high, fps, order=order)
        data = signal.filtfilt(b, a, data)

    return data


def local_max(xvalues, array1d, sig=None):
    # finds the local max array values and their respective position (xvalues)

    # by giving a significance cuttoff value array of the same size as the array1d, this will also return
    # the cutoff significance at each local maxima

    if sig is not None:
        i = np.r_[True, array1d[1:] > array1d[:-1]] & np.r_[
            array1d[:-1] > array1d[1:], True]
        return list(compress(xvalues,
                             i)), list(compress(array1d,
                                                i)), list(compress(sig, i))
    else:
        i = np.r_[True, array1d[1:] > array1d[:-1]] & np.r_[
            array1d[:-1] > array1d[1:], True]
        return list(compress(xvalues, i)), list(compress(array1d, i))


def local_min(xvalues, array1d, sig=None):
    # finds the local min array values and their respective position (xvalues)

    # by giving a significance cutoff value array of the same size as the array1d, this will also return
    # the cutoff significance at each local minima

    if sig is not None:
        i = np.r_[True, array1d[1:] < array1d[:-1]] & np.r_[
            array1d[:-1] < array1d[1:], True]
        return list(compress(xvalues,
                             i)), list(compress(array1d,
                                                i)), list(compress(sig, i))
    else:
        i = np.r_[True, array1d[1:] < array1d[:-1]] & np.r_[
            array1d[:-1] < array1d[1:], True]
        return list(compress(xvalues, i)), list(compress(array1d, i))


def abline(slope, intercept, nframe, label=None, color=None):
    """Plot a line from slope and intercept"""
    x_vals = np.array((0, nframe))
    y_vals = intercept + slope * x_vals
    if color != None:
        plt.plot(x_vals, y_vals, label=label, color=color)
    else:
        plt.plot(x_vals, y_vals, label=label)


def linear_regression(time, signal, verbose=True):

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(time.reshape(-1, 1), signal.reshape(-1, 1))
    wsumpred = regr.predict(signal.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_[0]
    if verbose:
        print('Coefficients: \n', regr.coef_)
        print('Mean squared error: %.2f' % mean_squared_error(signal, wsumpred))
        print('Variance score: %.2f' % r2_score(signal, wsumpred))

    return slope, intercept


def tdelay_correlation(vectors, n, max_offset=150, return_window=False):
    '''
    Calculates correlations of timecourses stored in an array 'vectors', of shape 
    (n, t) against the 'n'th element of the array, or an input vector of size 't'.  
    Returns the correlation of each vector with vector 'n', and time offset.
    '''

    if type(n) is int:
        tc = vectors[n].copy()

    elif type(n) is np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        assert n.size == vectors[0].size, \
            'vector `n` shape ({0}) was not same shape as vectors in array ({1})'\
                .format(n.size, vectors[0].size)
        tc = n

    tc = (tc - tc.mean()) / (tc.std() * len(tc))

    vectors = (vectors.copy() - vectors.mean(axis=1)[:,None]) / \
        vectors.std(axis=1)[:,None]
    # print('vectors', vectors)

    n_elements = vectors[:, 0].size
    x_corr = np.zeros(n_elements)
    t_delay = np.zeros(n_elements, dtype=np.int32)

    if max_offset > tc.size:
        max_offset = tc.size

    if return_window:
        corr_window = np.zeros((n_elements, 2 * max_offset + 1))

    for i, v in enumerate(vectors):
        corr = np.correlate(v, tc, mode='full')  # full correlation
        corr = corr[tc.size - max_offset - 1:tc.size +
                    max_offset]  # crop to window
        maxind = np.argsort(np.abs(corr))[-1]  #get largest value
        x_corr[i] = np.abs(corr)[maxind]
        t_delay[i] = maxind - max_offset

        if return_window:
            corr_window[i] = corr

    if return_window:
        return x_corr, t_delay, corr_window
    else:
        return x_corr, t_delay


def gaussian_smooth_2d(matrix, dj, dt):
    sigma = [dj, dt]
    smooth_matrix = gaussian_filter(matrix.real, sigma=sigma)
    smooth_matrix += gaussian_filter(matrix.imag, sigma=sigma).imag

    return smooth_matrix


def short_time_fourier_transform(data,
                                 fps=10,
                                 fftLen=100,
                                 overlap=99,
                                 verbose=False):

    padEndSz = fftLen
    # the last segment can overlap the end of the data array by no more
    # than one window size
    nyq = fps / 2  # Nyquist frequency

    if verbose:
        print("Calculating STFT of window size {0} and an overlap of {1}\
            \n--------------------------------------------------\
            ".format(fftLen, overlap))

    hopSz = np.int32(np.floor(fftLen - overlap))
    # calculates the how far the next STFT is from the last
    numSeg = np.int32(np.ceil(len(data) / np.float32(hopSz)))
    # Number of segments of through the all the data
    window = np.hanning(fftLen)
    # create a Hanning window of the appropriate length
    inPad = np.zeros(fftLen)  # zeros to pad each individual segment

    padData = np.concatenate((data, np.zeros(padEndSz)))
    # the padded data to process
    result = np.empty((fftLen, numSeg), dtype=np.float32)
    # space to hold the result

    for i in range(numSeg):
        hop = hopSz * i  # figure out the current segment offset
        seg = padData[hop:hop + fftLen]  # get the current segment
        windowed = (seg * window)  # apply a Hanning Window
        padded = np.append(windowed, inPad)
        # add zeros to double the length of the data
        spectrum = np.fft.fft(padded) / fftLen
        # take the Fourier Transform and scale by the number of data points
        autopower = np.abs(spectrum * np.conj(spectrum))
        # find the autopower spectrum
        result[:, i] = autopower[:fftLen]  # append to the results array

    result = np.flipud(result[0:math.floor(2 * nyq * fps), :]) / overlap
    # clip values greater than the nyquist sampling rate
    maxData = np.amax(result)
    minData = np.amin(result)

    return result, fps, nyq, maxData
