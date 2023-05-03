from sklearn.neighbors import KernelDensity
from scipy import signal
from scipy.signal import argrelextrema
import numpy as np
import math
from itertools import compress
from typing import Tuple, List


def sort_noise(timecourses: np.ndarray = None,
               lag1: np.ndarray = None,
               return_logpdf: bool = False,
               method: str = 'KDE',
               verbose: bool = False) -> Tuple[np.ndarray, int, np.ndarray]:
    '''
    Sorts timecourses into two clusters (signal and noise) based on 
    lag-1 autocorrelation.  

    Arguments:
        timecourses: Input to calculate noise threshold. 
            Should be a np array of shape (n, t).
        lag1: Required if the timecourses are not provided.
            alternate input to calculate noise threshold.
            Should be a np array of shape (n, t).
        return_logpdf: Whether to return the KDE log density function.
        method: The method to calculate the cutoff.  Currently only KDE is supported.
        verbose: Whether to record a verbose output.

    Returns:
        noise_components, a np array with a value of 1 where all noise 
            timecourses detected. as well as the cutoff value detected.
        cutoff: The cutoff index, anything above this value is considered to be noise. 
        log_pdf: Returned only if return_logpdf is True.  
            The pdf function evaluated between -0.2 and 1.2.
    '''
    if method == 'KDE':

        # Calculate lag autocorrelations.
        if lag1 is None:
            assert timecourses is not None, 'sortNoise requires either timecourses or lag1'
            lag1 = lag_n_autocorr(timecourses, 1)

        # Calculate minimum between min and max peaks.
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


def get_peak_separation(log_pdf: np.ndarray,
                        x_grid: np.ndarray = None) -> float:
    '''
    Get peak to peak separation value from the log pdf.
    This is used in 

    Arguments:
        log_pdf: The log pdf function specifying the density distribution.
        x_grid: Optional argument specifying the coordiates matching the log_pdf.
            If empty, -0.2 : 1.2 in 1200 points is the default.

    Returns: 
        peak_separation: The peak to peak distance of the log_pdf.
    '''
    if x_grid is None:
        x_grid = np.linspace(-0.2, 1.2, 1200)

    maxima = argrelextrema(np.exp(log_pdf), np.greater)[0]

    if len(maxima) > 2:
        maxima = np.delete(maxima, np.argmin(np.exp(log_pdf)[maxima]))
    peak_separation = x_grid[maxima[-1]] - x_grid[maxima[0]]

    return peak_separation


def lag_n_autocorr(x: np.ndarray, n: int, verbose: bool = True):
    '''
    Calculate the lag-n autocorrelation.  Lower values are more likely to be noise.

    Arguments:
        x: The 1-D input vector to calculate lag autcorrelation from.
        n: The integer lag to calculate.
        verbose: Whether to print a verbose output.
    '''
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


def butterworth(data: np.ndarray,
                high: float = None,
                low: float = None,
                fps: int = 10,
                order: int = 5) -> np.ndarray:
    '''
    Apply a butterworth filter on the data.

    Arguments:
        data: A 1-D time series array to filter.
        high: The high pass filter to apply.
        low: The low pass filter to apply.
        fps: The number of frames per second of the input data.
        order: The butterworth filter order to apply.

    Returns:
        data: The filtered dataset.
    '''

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


def local_max(x_values:np.ndarray, 
              array1d: np.ndarray, 
              sig: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Finds the local max array values and their respective position (xvalues)
    by giving a significance cuttoff value array of the same size as the array1d, 
    this will also return the cutoff significance at each local maxima.

    Args:
        x_values: positional information (i.e. time, sequence, etc.)
        array1d: data values (i.e. Dfof, etc.)
        sig: third list, significance of signal, that is returned 
            (in realitity this could be any array of equal len as array1d) 

    Returns:
        xvalues of where the local max 
        data values of the local max
        third list of significance of where the local max

    '''
    if sig is not None:
        i = np.r_[True, array1d[1:] > array1d[:-1]] & np.r_[array1d[:-1] >
                                                            array1d[1:], True]
        return list(compress(x_values,
                             i)), list(compress(array1d,
                                                i)), list(compress(sig, i))
    else:
        i = np.r_[True, array1d[1:] > array1d[:-1]] & np.r_[array1d[:-1] >
                                                            array1d[1:], True]
        return list(compress(x_values, i)), list(compress(array1d, i))


def local_min(x_values: np.ndarray,
              array1d: np.ndarray,
              sig=None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    '''
    Finds the local min array values and their respective position (xvalues)
    by giving a significance cuttoff value array of the same size as the array1d, 
    this will also return the cutoff significance at each local maxima.

    Args:
        x_values: positional information (i.e. time, sequence, etc.)
        array1d: data values (i.e. Dfof, etc.)
        sig: third list, significance of signal, that is returned 
            (in realitity this could be any array of equal len as array1d) 

    Returns:
        xvalues of where the local min 
        data values of the local min
        third list of significance of where the local min
    '''
    
    if sig is not None:
        i = np.r_[True, array1d[1:] < array1d[:-1]] & np.r_[array1d[:-1] <
                                                            array1d[1:], True]
        return list(compress(x_values,
                             i)), list(compress(array1d,
                                                i)), list(compress(sig, i))
    else:
        i = np.r_[True, array1d[1:] < array1d[:-1]] & np.r_[array1d[:-1] <
                                                            array1d[1:], True]
        return list(compress(x_values, i)), list(compress(array1d, i))


def abline(slope: float,
           intercept: float,
           x_max: float,
           label: str = None,
           color: str = None,
           x_min: float = 0) -> None:
    '''
    Plot a line to the currently active plt figure based on slope and intercept.

    Arguments:
        slope: The line slope.
        intercept: The line y intercept.
        x_max: The maximum x value to draw the line to.
        x_min: The minimum x value to draw the line to.
        label: The line label, if applicable.
        color: The matplotlib color string.  If not specified, uses the next default option.

    Returns:
        None
    '''
    x_vals = np.array((0, x_min))
    y_vals = intercept + slope * x_vals
    if color != None:
        plt.plot(x_vals, y_vals, label=label, color=color)
    else:
        plt.plot(x_vals, y_vals, label=label)


def linear_regression(time: np.ndarray,
                      signal: np.ndarray,
                      verbose=True) -> Tuple[float, float]:
    '''
    Applies a linear regression to the time-series signal.

    Arguments:
        time: The time, or x axis to fit the linear regression to.
        signal: The signal, or y axis to fit the linear regression to.
        verbose: Whether to print a verbose output specifying the fit results.

    Returns:
        slope: The linear regression slope.
        intercept: The linear regression intercept.
    '''
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


def tdelay_correlation(vectors: np.ndarray, 
                       n: int, 
                       max_offset: int = 150, 
                       return_window: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculates correlations of timecourses stored in an array 'vectors', of shape 
    (m, t) against the 'n'th element of the array, or an input vector of size 't'.  
    Returns 

    Arguments:
        vectors: a (m,t) numpy array containing all time series, including the one to compare to.
        n: The element to compare each other element to.
        max_offset: The integer maximum time offset to calculate correlation out to.
        return_window: Whether to return the maximum correlation value in the context of the correlation window.

    Returns:
        x_corr: the correlation of each vector with vector 'n', and time offset.
        t_delay: The time delay which maximizes the correlation value.
        corr_window: Returns the maximum correlation at the given time delay for each time series,
            all other values are zero.  Only returned if return_window is True.
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


def gaussian_smooth_2d(matrix: np.ndarray,
                       dj: float,
                       dt: float) -> np.ndarray:
    '''
    This takes a 2-d matrix and applies a smoothing gaussian
    filter defined by the two sigma variables (dj, dt)

    Arguments:
        matrix: 2d matrix of values to smooth
        dj = sigma to define the first dimension gaussian
        dt = sigma to define the second dimension gaussian

    Returns:
        Smoothed 2d matrix

    '''
    sigma = [dj, dt]
    smooth_matrix = gaussian_filter(matrix.real, sigma=sigma)
    smooth_matrix += gaussian_filter(matrix.imag, sigma=sigma).imag

    return smooth_matrix


def short_time_fourier_transform(data: np.ndarray,
                                 fps: float = 10,
                                 fft_len: int = 100,
                                 overlap: int = 50,
                                 verbose: bool = False) -> np.array:
    '''
    Creates a short time windowed fourier transform of a time series

    Arguments:
        data: Time series (1d vector)
        fps: frames per second
        fft_len: window length (number of data points to run fft)
        overlap: Overlab between adjacent windows (number of datapoints that are 
            similar between adajacent windows)
        verbose: Boolean to be verbose or not

    Returns:
        result: 2d matrix of short time windowed Fourier transform
        fps: frames per second used in fourier transform
        nyq: nyquist frequency
        maxData: location of the maxima of the 2d transform
    '''
    
    padEndSz = fft_len
    # The last segment can overlap the end of the data array by no more
    # than one window size.
    nyq = fps / 2  # Nyquist frequency.

    if verbose:
        print("Calculating STFT of window size {0} and an overlap of {1}\
            \n--------------------------------------------------\
            ".format(fft_len, overlap))

    hopSz = np.int32(np.floor(fft_len - overlap))
    # Calculates the how far the next STFT is from the last.
    numSeg = np.int32(np.ceil(len(data) / np.float32(hopSz)))
    # Number of segments of through the all the data.
    window = np.hanning(fft_len)
    # Create a Hanning window of the appropriate length.
    inPad = np.zeros(fft_len)  # zeros to pad each individual segment

    padData = np.concatenate((data, np.zeros(padEndSz)))
    # The padded data to process.
    result = np.empty((fft_len, numSeg), dtype=np.float32)
    # Space to hold the result.

    for i in range(numSeg):
        hop = hopSz * i  # Figure out the current segment offset.
        seg = padData[hop:hop + fft_len]  # Get the current segment.
        windowed = (seg * window)  # Apply a Hanning Window.
        padded = np.append(windowed, inPad)
        # Add zeros to double the length of the data.
        spectrum = np.fft.fft(padded) / fft_len
        # Take the Fourier Transform and scale by the number of data points.
        autopower = np.abs(spectrum * np.conj(spectrum))
        # Find the autopower spectrum.
        result[:, i] = autopower[:fft_len]  # Append to the results array.

    result = np.flipud(result[0:math.floor(2 * nyq * fps), :]) / overlap
    # Clip values greater than the nyquist sampling rate.
    maxData = np.amax(result)
    minData = np.amin(result)

    return result, fps, nyq, maxData
