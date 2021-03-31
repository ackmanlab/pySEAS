#!/usr/bin/env python3

import sys
#sys.path.insert(0, '/home/mike/Dropbox/0_firebird_research/fit_peaks/single_fits/')
import numpy as np
from seas.waveletFunctions import *
import matplotlib.pylab as plt
import matplotlib
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from datetime import datetime
from seas.signalanalysis import local_max, linear_regression, abline, gaussian_smooth_2d, lag_n_autocorr
#sys.path.insert(0, '/home/mike/Dropbox/0_firebird_research/microburst_characterization/')
#from src import remove_dropouts
import operator

#Adapted from __author__ = 'Mykhaylo Shumko' : __init__, waveletTransform, plotPower, lagNAutoCorr
#All other functions written by Brian Mullen
'''
To find the global wavelet spectrum run these two scripts:
    waveletAnalysis.globalWaveletSpectrum()

To find events:
    Need to work on this. Depending on which range of frequencies, you get varying results
    waveletAnalysis.averageWaveletPower(periodLim =[0.25, 8])

To find signal change:
    waveletAnalysis.tsSignal()
    There seems to be very little difference between binned data and full data

'''


def waveletCoherence(tc1, tc2, fps=10, siglvl=0.95, sigtest=0):

    tc1 = np.asarray(np.squeeze(tc1))
    tc2 = np.asarray(np.squeeze(tc2))

    assert tc1.ndim == 1 and tc2.ndim == 1, 'Timecouses should only be one dimension'
    assert tc1.shape[0] == tc2.shape[0], 'Timecourses are not the same shape'

    #run wavelet transform
    w1 = waveletAnalysis(tc1, fps=fps)
    w2 = waveletAnalysis(tc2, fps=fps)

    # cross wavelet transform
    w1w2 = w1.wave * np.conj(
        w2.wave)  #/ (np.ones([1, tc1.size]) * w1.scale[:, None])
    xwt = np.sqrt(w1w2.real**2 + w1w2.imag**2)

    # calculate phase
    phase = np.angle(w1w2)

    # #set up smoothing window
    # win_s = np.floor(w1.wave.shape[1]/w1.dj)+1
    # win_t = 2*np.floor(w1.wave.shape[0]/w1.cadence)+1

    # win_s = 10
    # win_t = 10

    # window2D = np.dot(windowFunc(win_t, win_type = 'ham')[:,np.newaxis] , windowFunc(win_s, win_type = 'ham')[:,np.newaxis].T)

    #smooth
    s1 = gaussian_smooth_2d(w1.power, w1.dj, w1.cadence)
    s2 = gaussian_smooth_2d(w2.power, w2.dj, w2.cadence)
    s1s2 = gaussian_smooth_2d(w1w2, w1.dj, w1.cadence)

    #calculate coherency
    coherency = s1s2 / np.sqrt(s1 * s2)

    #calculate coherence
    coherence = (s1s2.real**2 + s1s2.imag**2) / (s1 * s2)

    # significance?
    acor1 = 0.5 * (lag_n_autocorr(tc1, 1) + lag_n_autocorr(tc1, 2))
    acor2 = 0.5 * (lag_n_autocorr(tc2, 1) + lag_n_autocorr(tc2, 2))

    xwt_signif = wave_cohere_signif(([1.0]), ([1.0]),
                                    dt=1 / fps,
                                    sigtest=sigtest,
                                    scale=w1.scale,
                                    x_lag1=acor2,
                                    y_lag1=acor1,
                                    mother=w1.mother,
                                    siglvl=siglvl)
    xwt_signif = xwt_signif[:, np.newaxis].dot(np.ones(w1.n)[np.newaxis, :])

    return coherency, coherence, xwt, phase, xwt_signif


def wave_cohere_signif(X,
                       Y,
                       dt,
                       scale,
                       sigtest=-1,
                       y_lag1=-1,
                       x_lag1=-1,
                       siglvl=-1,
                       dof=-1,
                       mother=-1,
                       param=-1):
    '''
    Modified from waveletFunctions to take into account two time series in calculating the 
    significance of the cross wavelet transform.
    '''
    n1 = len(np.atleast_1d(Y))
    n2 = len(np.atleast_1d(X))

    J1 = len(scale) - 1
    s0 = np.min(scale)
    dj = np.log2(scale[1] / scale[0])

    if n1 == 1:
        y_variance = Y
    else:
        y_variance = np.std(Y)**2

    if n2 == 1:
        x_variance = X
    else:
        x_variance = np.std(X)**2

    if sigtest == -1:
        sigtest = 0
    if x_lag1 == -1:
        x_lag1 = 0.0
    if y_lag1 == -1:
        y_lag1 = 0.0
    if siglvl == -1:
        siglvl = 0.95
    if mother == -1:
        mother = 'MORLET'

    # get the appropriate parameters [see Table(2)]
    if mother == 'MORLET':  #----------------------------------  Morlet
        empir = ([2, -1, -1, -1])
        if param == -1 or param == 6:
            param = 6.
            empir[1:] = ([0.776, 2.39, 0.60])
        if param == 4:
            empir[1:] = ([1.151, 2.5, 0.60])
        k0 = param
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0**2)
                                       )  # Scale-->Fourier [Sec.3h]
    elif mother == 'PAUL':
        empir = ([2, -1, -1, -1])
        if param == -1:
            param = 4
            empir[1:] = ([1.132, 1.17, 1.5])
        m = param
        fourier_factor = (4 * np.pi) / (2 * m + 1)
    elif mother == 'DOG':  #-------------------------------------Paul
        empir = ([1., -1, -1, -1])
        if param == -1 or param == 2:
            param = 2.
            empir[1:] = ([3.541, 1.43, 1.4])
        elif param == 6:  #--------------------------------------DOG
            empir[1:] = ([1.966, 1.37, 0.97])
        m = param
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
    else:
        print('Mother must be one of MORLET, PAUL, DOG')

    period = scale * fourier_factor
    dofmin = empir[0]  # Degrees of freedom with no smoothing
    Cdelta = empir[1]  # reconstruction factor
    gamma_fac = empir[2]  # time-decorrelation factor
    dj0 = empir[3]  # scale-decorrelation factor
    freq = dt / period  # normalized frequency

    y_fft_theor = (1 - y_lag1**2) / (
        1 - 2 * y_lag1 * np.cos(freq * 2 * np.pi) + y_lag1**2)  # [Eqn(16)]
    y_fft_theor = y_variance * y_fft_theor  # include time-series variance

    x_fft_theor = (1 - x_lag1**2) / (
        1 - 2 * x_lag1 * np.cos(freq * 2 * np.pi) + x_lag1**2)  # [Eqn(16)]
    x_fft_theor = x_variance * x_fft_theor  # include time-series variance

    fft_theor = np.sqrt(y_fft_theor * x_fft_theor)
    signif = fft_theor

    if len(np.atleast_1d(dof)) == 1:
        if dof == -1:
            dof = dofmin
    if sigtest == 0:  # no smoothing, DOF=dofmin [Sec.4]
        dof = dofmin
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = fft_theor * chisquare  # [Eqn(18)]
    elif sigtest == 1:  # time-averaged significance
        if len(np.atleast_1d(dof)) == 1:
            dof = np.zeros(J1) + dof
        dof[dof < 1] = 1
        dof = dofmin * np.sqrt(1 +
                               (dof * dt / gamma_fac / scale)**2)  # [Eqn(23)]
        dof[dof < dofmin] = dofmin  # minimum DOF is dofmin
        for a1 in range(0, J1 + 1):
            chisquare = chisquare_inv(siglvl, dof[a1]) / dof[a1]
            signif[a1] = fft_theor[a1] * chisquare
        # print("Chi squared: %e " % chisquare)
    elif sigtest == 2:  # time-averaged significance
        if len(dof) != 2:
            print(
                'ERROR: DOF must be set to [S1,S2], the range of scale-averages'
            )
        if Cdelta == -1:
            print('ERROR: Cdelta & dj0 not defined for ' + mother +
                  ' with param = ' + str(param))

        s1 = dof[0]
        s2 = dof[1]
        avg = np.logical_and(scale >= s1, scale < s2)  # scales between S1 & S2
        navg = np.sum(
            np.array(np.logical_and(scale >= s1, scale < s2), dtype=int))
        if navg == 0:
            print('ERROR: No valid scales between ' + str(s1) + ' and ' +
                  str(s2))
        Savg = 1. / np.sum(1. / scale[avg])  # [Eqn(25)]
        Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)  # power-of-two midpoint
        dof = (dofmin * navg * Savg / Smid) * np.sqrt(
            1 + (navg * dj / dj0)**2)  # [Eqn(28)]
        fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])  # [Eqn(27)]
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare  # [Eqn(26)]
    else:
        print('ERROR: sigtest must be either 0, 1, or 2')

    return signif


def lagWaveletCoherency(tc1, tc2, fps=10, lag=5):
    for i in np.arange(0, lag + 1):
        if i == 0:
            #initialize for proper shape
            cor, coh, xwt, phase, xwt_sig = waveletCoherence(tc1, tc2)

            # create ouputs
            lag_cor = np.zeros(
                (2 * lag + 1, cor.shape[0], cor.shape[1])) * np.nan
            lag_coh = np.zeros(
                (2 * lag + 1, cor.shape[0], cor.shape[1])) * np.nan
            lag_xwt = np.zeros(
                (2 * lag + 1, cor.shape[0], cor.shape[1])) * np.nan
            lag_phase = np.zeros(
                (2 * lag + 1, cor.shape[0], cor.shape[1])) * np.nan
            lag_xwt_sig = np.zeros(
                (2 * lag + 1, cor.shape[0], cor.shape[1])) * np.nan

            #store in the proper index
            lag_cor[lag] = cor
            lag_coh[lag] = coh
            lag_xwt[lag] = xwt
            lag_phase[lag] = phase
            lag_xwt_sig[lag] = xwt_sig
        else:
            #finish up the remaining lags (right shifted)
            lag_cor[lag + i, : , i:], lag_coh[lag + i, : , i:], lag_xwt[lag + i, : , i:],\
            lag_phase[lag + i, : , i:], lag_xwt_sig[lag + i, : , i:] = waveletCoherence(tc1[i:],tc2[:-i], fps = fps)

            #(left shifted)
            lag_cor[lag - i, : , :-i],lag_coh[lag - i, : , :-i],lag_xwt[lag - i, : , :-i],\
            lag_phase[lag - i, : , :-i],lag_xwt_sig[lag - i, : , :-i] = waveletCoherence(tc1[:-i],tc2[i:], fps = fps)

    return lag_cor, lag_coh, lag_xwt, lag_phase, lag_xwt_sig


class waveletAnalysis:

    def __init__(self, data, fps, **kwargs):
        """
        Initialize the wavelet parameters
        """

        assert data.ndim == 1, 'Time series is the wrong shape. It should be a 1-dim vector'

        self.dataCopy = data
        self.data = (data - np.mean(data)) / np.std(data, ddof=1)
        self.n = len(self.data)
        self.cadence = 1 / fps
        self.time = np.arange(self.n) * self.cadence

        #default parameters
        #print/ plot statements
        self.verbose = kwargs.get('verbose', False)
        self.plot = kwargs.get('plot', False)

        #wavelet parameters
        self.mother = kwargs.get('mother', 'MORLET')
        self.param = kwargs.get('param', 4)
        self.j1 = kwargs.get('j1', 80)
        self.pad = kwargs.get('pad', 1)
        self.dj = kwargs.get('dj', 0.125)
        self.s0 = kwargs.get('s0', 2 * self.cadence)

        #noies modeling parameter
        self.siglvl = kwargs.get('siglvl', 0.95)
        self.lag1 = 0.5 * (lag_n_autocorr(data, 1) + lag_n_autocorr(data, 2))
        # self.lag1 = self.lagNAutoCorr(data, 1)

        self.waveletTransform()

    def waveletTransform(self):
        # Wavelet transform:
        self.wave, self.period, self.scale, self.coi = \
            wavelet(self.data, self.cadence, self.pad, self.dj, self.s0, self.j1, self.mother, self.param)

        if len(self.time) != len(self.coi):
            self.coi = self.coi[1:]

        self.power = (np.abs(self.wave))**2  # compute wavelet power spectrum

        # Significance levels: (variance=1 for the normalized data)
        self.signif = wave_signif(([1.0]), dt=self.cadence, sigtest=0, scale=self.scale, \
            lag1=self.lag1, mother=self.mother, siglvl = self.siglvl)
        self.sig95 = self.signif[:, np.newaxis].dot(
            np.ones(self.n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
        self.sig95 = self.power / self.sig95  # where ratio > 1, power is significant
        return

    def inverseWaveletTransform(self,
                                waveFlt=None,
                                C_d=1.151,
                                psi0=np.pi**(-0.25) * 0.85):
        """
        Supply own C_d and psi0 if not using a DOG m = 2 wavelet.
        """
        if waveFlt == None:
            waveFlt = self.waveFlt

        InvTranform = np.zeros(self.n)
        tansformConstant = ((self.dj * math.sqrt(self.cadence)) / (C_d * psi0)
                           )  # Reconstruction constant.

        # For more information, see article: "A Practical Guide to Wavelet Analysis", C. Torrence and G. P. Compo, 1998.
        for i in range(waveFlt.shape[0]):
            waveFlt[i, :] /= math.sqrt(self.period[i])
        for i in range(self.n):
            InvTranform[i] = np.sum(np.real(waveFlt[:, i]), axis=0)
        self.dataFlt = tansformConstant * InvTranform

    def plotPower(self, ax=None):
        """
        ax is the subplot argument.
        """
        self.levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        colors = [
            'navy', 'slateblue', 'c', 'g', 'gold', 'orange', 'tomato', 'crimson'
        ]

        if ax == None:
            f = plt.figure()
            f, ax = plt.subplots(1)
        else:
            ax = np.ravel(ax)[0]

        # Max period is fourier_factor*S0*2^(j1*dj), fourier_factor = 3.97383530632
        CS = ax.contourf(self.time,
                         self.period,
                         np.log2(self.power),
                         len(self.levels),
                         colors=colors)
        im = ax.contourf(CS, levels=np.log2(self.levels), colors=colors)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Period (s)')
        ax.set_title('Wavelet Power Spectrum')
        # 95 significance contour, levels at -99 (fake) and 1 (95# signif)
        ax.contour(self.time, self.period, self.sig95, [-99, 1], colors='k')
        # # cone-of-influence, anything "below" is dubious
        ax.fill_between(self.time,
                        np.max(self.period),
                        self.coi,
                        alpha=0.5,
                        facecolor='white',
                        zorder=3)
        ax.plot(self.time, self.coi, 'k')
        # # format y-scale

        # different matplotlib versions available for python < 3.8.
        try:
            ax.set_yscale('log', base=2, subs=None)
        except ValueError:
            ax.set_yscale('log', basey=2, subsy=None)

        ax.set_ylim([np.min(self.period), np.max(self.period)])
        axy = ax.yaxis
        axy.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(
            axis='y', style='plain')  ## causes issues with tkinter mpl canvas
        ax.invert_yaxis()
        # set up the size and location of the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        plt.colorbar(im, cax=cax, orientation='horizontal')

    def waveletFilter(self,
                      lowerPeriod=None,
                      upperPeriod=None,
                      movement_vector=None,
                      sigLevel=1):
        """
        NAME:    waveletFilter(wave, sig95, l_s = 0, u_s = 29)
        USE:     This does a band pass and power filtering on the wavelet period-frequency domain. Give it the                   wavelet amplitude array, the power significance array, and the upper and lower scales that will                         be used for filtering. 
        RETURNS: The filtered wavelet amplitude array.
        MOD:     2016-04-13
        
        # WAVELET SCALE CALCULATOR ################# 
        # jth scale indicie = ln(S/S0)/(dJ * ln(2))
        ##################################    
        """
        self.waveFlt = self.wave.copy()

        # Band pass filter
        # Proporionally decreases power in specific wavelengths with respect to a 'motion vector'
        # More motion, more severe the filtration of power in the given frequencies
        if movement_vector is not None:
            movement_vector -= movement_vector.min()
            movement_vector /= movement_vector.max()
            movement_vector = (1. - movement_vector)

            if lowerPeriod != None:
                if lowerPeriod > self.period[0]:
                    lower_ind = np.where(self.period < lowerPeriod)[0][-1]
                    #print('Lower Period: ', self.period[lower_ind], 'Upper Freq: ', 1/self.period[lower_ind])
                    self.waveFlt[:
                                 lower_ind, :] = self.waveFlt[:lower_ind, :] * movement_vector
            if upperPeriod != None:
                if upperPeriod < self.period[-1]:
                    upper_ind = np.where(self.period > upperPeriod)[0][0]
                    #print('Upper Period: ', self.period[upper_ind], 'Lower Freq: ', 1/self.period[upper_ind])
                    self.waveFlt[upper_ind:, :] = self.waveFlt[
                        upper_ind:, :] * movement_vector

        # Band pass filter
        # Zero out parts of the wavlet space that we don't want to reconstruct.
        else:
            if lowerPeriod != None:
                if lowerPeriod > self.period[0]:
                    lower_ind = np.where(self.period < lowerPeriod)[0][-1]
                    #print('Lower Period: ', self.period[lower_ind], 'Lower Freq: ', 1/self.period[lower_ind])
                    self.waveFlt[:lower_ind, :] = 0
            if upperPeriod != None:
                if upperPeriod < self.period[-1]:
                    upper_ind = np.where(self.period > upperPeriod)[0][0]
                    #print('Upper Period: ', self.period[upper_ind], 'Upper Freq: ', 1/self.period[upper_ind])
                    self.waveFlt[upper_ind:, :] = 0

        # Significance filter
        notSigInd = np.where(
            self.sig95 < sigLevel
        )  # Only pass data that has power of (100% - sigThreshold). Usually sigThreshold is 95%. Was 0.25.
        self.waveFlt[notSigInd] = 0

    ############################################

    def nanCOI(self):
        # get rid of all values outside the cone of influence
        #   wave = np.log2(wave)
        self.nanCOImat = self.power.copy()
        for i in range(self.power.shape[1]):
            cutoff = np.where(self.coi[i] < self.period)
            self.nanCOImat[cutoff, i] = np.nan

    def nanSig(self):
        # get rid of all values that are not significant
        self.nanSigmat = self.wave
        self.nanSigmat[np.where(wavelet.sig95 < 1)] = np.nan

    def sigLost(self, slope, intercept):
        loss = 1 - (intercept + slope * self.n * self.cadence) / intercept
        if self.verbose:
            print('We have lost approximately {0:.2f} % power over the movie'.
                  format(loss[0] * 100))

        return loss

    def binSignal(self, binsz=30):  # binsz in seconds
        binnum = (self.n * self.cadence) // binsz
        padsz = math.ceil(float(self.n) / binnum) * binnum - self.n
        binsignal = np.append(self.signal, np.zeros(int(padsz)) * np.nan)
        binsignal = np.nanmean(binsignal.reshape(int(binnum), -1), axis=1)
        bintime = np.arange(binsz, self.n * self.cadence + 1, binsz) - binsz / 2

        return bintime, binsignal

    def familySig(self, sigList=[0.9, 0.95, 0.99, 0.999], dof=-1, sigtest=0):
        # plot a family of significance curves for visualization and analysis

        if isinstance(sigList, float):
            if sigtest < 2:
                fam_signif = np.zeros((1, self.scale.shape[0])) * np.nan
            if sigtest == 2:
                fam_signif = np.nan
            fam_signif = wave_signif([1.0],
                                     dt=self.cadence,
                                     scale=self.scale,
                                     sigtest=sigtest,
                                     lag1=self.lag1,
                                     siglvl=sigList,
                                     dof=dof,
                                     mother=self.mother,
                                     param=self.param)

        if isinstance(sigList, list):
            if sigtest < 2:
                fam_signif = np.zeros(
                    (len(sigList), self.scale.shape[0])) * np.nan
            if sigtest == 2:
                fam_signif = np.zeros((len(sigList), 1)) * np.nan

            for i, sig in enumerate(sigList):
                fam_signif[i] = wave_signif([1.0],
                                            dt=self.cadence,
                                            scale=self.scale,
                                            sigtest=sigtest,
                                            lag1=self.lag1,
                                            siglvl=sig,
                                            dof=dof,
                                            mother=self.mother,
                                            param=self.param)

        return np.squeeze(fam_signif), np.squeeze(sigList)

    def sumAcrossPeriod(self, perLim=[0, 100]):
        #sum wavelet power across select periods

        if self.verbose:
            print('Summing across {0} to {1} periods on wavelet run with mother {2} at paramter {3}'.\
                  format(perLim[0], perLim[1], self.mother, self.param) )

        cdelta = None
        if self.mother == 'MORLET' and self.param == 6:
            cdelta = 0.776
        elif self.mother == 'MORLET' and self.param == 4:
            cdelta = 1.151
        elif self.mother == 'DOG' and self.param == 2:
            cdelta = 3.541
        elif self.mother == 'DOG' and self.param == 6:
            cdelta = 1.966
        else:
            assert cdelta != None, 'Unknown c value based on wavelet choice'

        l_per_lim = np.min(np.where(perLim[0] < self.period))
        u_per_lim = np.min(np.where(perLim[1] < self.period))

        if not hasattr(self, 'nanCOImat'):
            self.nanCOI()

        period_sum = (self.dj * self.cadence) / cdelta * np.nansum(
            (self.nanCOImat[l_per_lim:u_per_lim]**2 /
             self.period[l_per_lim:u_per_lim, None]),
            axis=0)

        return np.squeeze(period_sum)

    def globalWaveletSpectrum(self):
        if self.verbose:
            print('Assessing wavelet mother {0} at paramter {1}'.format(
                self.mother, self.param))

        # calulate the global self spectrum
        self.nanCOI()
        # if np.sum(~np.isnan(self.nanCOImat))!=0:
        self.period_size = np.sum(~np.isnan(self.nanCOImat), axis=1)
        nan_ind = np.where(self.period_size == 0)[0]
        self.gws = np.zeros_like(self.period) * np.nan
        if nan_ind.any():
            self.gws[:nan_ind[0]] = np.nanmean(self.nanCOImat[:nan_ind[0], :],
                                               axis=1)
            self.gws[nan_ind] = 0
        else:
            self.gws = np.nanmean(self.nanCOImat, axis=1)

        if self.period_size.shape[0] != self.period.shape[0]:
            dif = self.period_size.shape[0] - self.period.shape[0]
            if dif < 0:
                self.period_size = np.append(self.period_size,
                                             np.zeros(np.abs(dif)))
            else:
                self.period_size = self.period_size[:self.period.shape[0]]

        # calculate the average significance
        self.gws_sig, self.gws_sigList = self.familySig(sigList=[0.95],
                                                        dof=self.period_size,
                                                        sigtest=1)

        if self.verbose:
            print('Auto-correlation value: {0:.4g}'.format(self.lag1))
        # determine fourier wavelength
        if self.mother == 'DOG':
            self.flambda = (2 * np.pi * 1 / self.period) / np.sqrt(self.param +
                                                                   .5)
        if self.mother == 'MORLET':
            self.flambda = (4 * np.pi * 1 / self.period) / (
                self.param + np.sqrt(2 + np.square(self.param)))

        mx_wav, mx_gws, mx_sig = local_max(self.period, self.gws, self.gws_sig)
        fl_wav, mx_gws, mx_sig = local_max(self.flambda, self.gws, self.gws_sig)

        lwav = []
        lgws = []
        lfl = []
        for i in range(len(mx_wav)):
            if mx_gws[i] > mx_sig[i]:
                lwav.append(mx_wav[i])
                lgws.append(mx_gws[i])
                lfl.append(fl_wav[i])
        lwav_inv = [x**(-1) for x in lwav]

        self.gws_localmax_power = lgws
        self.gws_localmax_freq = lfl

        #find the lowest and highest frequencies that are still significant
        hiwav = np.nan
        hival = np.nan
        lowav = np.nan
        loval = np.nan

        if np.where(self.gws > self.gws_sig)[0].shape[0] > 0:
            hival = self.gws[np.where(self.gws > self.gws_sig)][0]
            hiwav = self.flambda[np.where(self.gws > self.gws_sig)][0]

            loval = self.gws[np.where(self.gws > self.gws_sig)][-1]
            lowav = self.flambda[np.where(self.gws > self.gws_sig)][-1]

        self.gws_lo_high_freq = [(lowav, loval), (hiwav, hival)]

        if nan_ind.any():
            self.gws[nan_ind] = np.nan

        if self.verbose:
            print('Low frequency: ', lowav)
            print('High freqency: ', hiwav)

        if self.plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
            nyq = 0.5 * 1 / self.cadence
            linetype = ['-', '-.', '--', ':']
            fam_signif, sigList = self.familySig(dof=self.period_size,
                                                 sigtest=1)

            #self period graph
            ax1.plot(self.period, self.gws)
            ax1.plot(lwav, lgws, 'ro')

            for i in range(len(sigList)):
                if i >= len(linetype) - 1:
                    j = len(linetype) - 1
                else:
                    j = i
                ax1.plot(self.period,
                         fam_signif[i],
                         label=sigList[i],
                         ls=linetype[j],
                         color='k')
            if not lgws:
                ax1.set_ylim([0, 10])
            else:
                ax1.set_ylim([0, np.ceil(max(lgws))])

            if not lwav or np.ceil(max(lwav)) < 10:
                ax1.set_xlim([0, 10])
            else:
                ax1.set_xlim([0, np.ceil(max(lwav))])
            # ax1.legend()
            ax1.set_xlabel(' Wavelet {0} {1} period(s)'.format(
                self.mother, self.param))
            ax1.set_ylabel('normalized power (to variance)')
            ax1.set_title('Power spectrum by period')
            # plt.show()

            #         #self frequency graph
            #         plt.plot(1/self.period, gws)
            #         plt.plot(lwav_inv,lgws, 'ro')

            #         for i in range(len(sigList)):
            #             if i >= len(linetype) - 1:
            #                 j = len(linetype) - 1
            #             else:
            #                 j = i
            #             plt.plot(1/self.period, fam_signif[i], label = sigList[i], ls = linetype[j], color='k')
            #         if not lgws:
            #             plt.ylim([0,10])
            #         else:
            #             plt.ylim([0,np.ceil(max(lgws))])

            #         if not lwav_inv or np.ceil(max(lwav_inv)) < nyq:
            #             plt.xlim([0,nyq])
            #         else:
            #             plt.xlim([0, np.ceil(max(lwav_inv))])
            #         plt.xlabel(self.mother + ' frequency')
            #         plt.ylabel('normalized power (to variance)')
            #         plt.legend()
            #         plt.show()

            #Fourier space lambda
            ax2.plot(self.flambda, self.gws)
            for i in range(len(sigList)):
                if i >= len(linetype) - 1:
                    j = len(linetype) - 1
                else:
                    j = i
                ax2.plot(self.flambda,
                         fam_signif[i],
                         label=sigList[i],
                         ls=linetype[j],
                         color='k')

            if not lgws:
                ax2.set_ylim([0, 10])
            else:
                ax2.set_ylim([0, np.ceil(max(lgws))])

            if not lfl or np.ceil(max(lfl)) < nyq:
                ax2.set_xlim([0, nyq])
            else:
                ax2.set_xlim([0, np.ceil(max(lfl))])
            ax2.plot(lowav, loval, 'go', label='lowSigFreq')
            ax2.plot(hiwav, hival, 'bo', label='highSigFreq')
            ax2.plot(lfl, lgws, 'ro', label='localMax')
            ax2.legend()
            ax2.set_xlabel('Fourier frquency')
            ax2.set_title('Power spectrum by frequency')

            # ax2.set_ylabel('normalized power (to variance)')
            plt.tight_layout()
            plt.show()

        # return lfl, lgws, [lowav,loval], [hiwav, hival]

    def averageWaveletPower(self, periodLim=[0.25, 8]):

        assert len(
            periodLim) == 2, 'Period limit list must only include 2 values'

        if self.verbose:
            print('Creating scaled average of the timeseries, \
             created with wavelet mother {0} at paramter {1}'.format(
                self.mother, self.param))

        self.period_sum = self.sumAcrossPeriod(perLim=periodLim)
        if self.period_sum.shape[0] != self.time.shape[0]:
            dif = self.period_sum.shape[0] - self.time.shape[0]
            if dif < 0:
                self.period_sum = np.append(self.period_sum,
                                            np.zeros(np.abs(dif)))
            else:
                self.period_sum = self.period_sum[:self.period.shape[0]]

        # calculate the average significance

    #     sig = wavelet.signif/np.sqrt(1+((self.period_size * wavelet.cadence)/(gamma * wavelet.period)))
        self.sig_period_sum, _ = self.familySig(sigList=[0.95],
                                                dof=periodLim,
                                                sigtest=2)

        #find coordinates of local max values
        mx_wav, mx_gws = local_max(self.time, self.period_sum)

        #Return only those above significance threshold
        ltime = []
        lgws = []
        for i in range(len(mx_wav)):
            if mx_gws[i] > self.sig_period_sum:
                ltime.append(mx_wav[i])
                lgws.append(mx_gws[i])
        if self.plot:
            sigline = np.zeros(self.period_sum.shape[0]) + self.sig_period_sum

            plt.plot(self.time, self.period_sum)
            plt.plot(self.time, np.squeeze(sigline), 'k--')
            plt.plot(ltime, lgws, 'ro')
            # plt.xlim([150, 180])
            # plt.ylim([0, 20])

            #     plt.xlim([0,8])
            #     plt.ylim([0,1])
            plt.show()

        self.events = ltime
        self.events_value = lgws

    def tsSignal(self, binSig=False, periodLim=[0.5, 4]):
        '''
        Using wavelet transforms to assess how the power of signal changes 
        over time
        '''
        self.binSig = binSig
        assert len(periodLim) == 2, 'Period limits are wrong size'

        #determine the time and power across neural signal periods
        self.signal = self.sumAcrossPeriod(perLim=periodLim)
        self.signal_sig, _ = self.familySig(sigList=[0.95],
                                            dof=periodLim,
                                            sigtest=2)

        if self.binSig:
            bintime, binsignal = self.binSignal()
            self.slope, self.intercept = linear_regression(bintime,
                                                           binsignal,
                                                           verbose=self.verbose)
            self.signalloss = self.sigLost(self.slope, self.intercept)
        else:
            self.slope, self.intercept = linear_regression(self.time,
                                                           self.signal,
                                                           verbose=self.verbose)
            self.signalloss = self.sigLost(self.slope, self.intercept)

        if self.plot:
            plt.plot(self.time, self.signal, label='signal power')
            plt.plot(self.time,
                     np.squeeze(np.ones(self.n) * self.signal_sig),
                     'k--',
                     label='{0} sign level'.format(self.siglvl))
            abline(self.slope,
                   self.intercept,
                   self.n * self.cadence,
                   label='trend',
                   color=None)
            if self.binSig:
                plt.scatter(bintime, binsignal, color='r', label='binvalues')
            plt.xlabel('time(s)')
            plt.ylabel('summed power (period: {0} - {1})'.format(
                periodLim[0], periodLim[1]))
            plt.legend()
            plt.show()

    def noiseFilter(self,
                    lowerPeriod=None,
                    upperPeriod=10,
                    movement_vector=None,
                    sigLevel=0.25):

        if lowerPeriod is None:
            lowerPeriod = 2 * self.cadence  #nyquist sampling rate

        if movement_vector is not None:
            movement_vector = np.array(np.squeeze(movement_vector),
                                       dtype='float64')
            assert movement_vector.shape[
                0] == self.n, 'Movement vector is not the same size as the data'

        self.waveletFilter(lowerPeriod=lowerPeriod,
                           upperPeriod=upperPeriod,
                           movement_vector=movement_vector,
                           sigLevel=sigLevel)
        self.inverseWaveletTransform()
        filtData = (self.dataFlt * np.std(self.dataCopy, ddof=1)) + np.mean(
            self.dataCopy)

        if self.plot:
            plt.figure(num=None,
                       figsize=(20, 3),
                       dpi=80,
                       facecolor='w',
                       edgecolor='k')
            ax1 = plt.subplot(211)
            ax1.plot(self.time, self.dataCopy, color='k', label='origional')
            ax1.set_ylabel('dfof')
            ax1.legend()

            ax2 = plt.subplot(212)
            ax2.plot(self.time, filtData, color='blue', label='filtered')
            ax2.plot(self.time,
                     self.dataCopy - filtData,
                     color='orange',
                     label='residual')
            ax2.set_ylabel('dfof')
            ax2.set_xlabel('time(s)')
            ax2.legend()
            plt.show()

        return filtData

    def waveletEventDetection(self):

        self.waveletFilter(lowerPeriod=2 * self.cadence,
                           upperPeriod=100,
                           sigLevel=1)
        self.inverseWaveletTransform()
        fstd = np.std(self.dataFlt)

        xmax, ymax = local_max(self.time, self.dataFlt)

        timeIndex = np.array(xmax)[np.array(np.squeeze(ymax > (fstd)))]

        frameIndex = (timeIndex / self.cadence).astype('uint16')
        eventVal = self.dataCopy[frameIndex]

        if self.plot:
            fig = plt.figure(num=None,
                             figsize=(20, 3 * 4),
                             dpi=80,
                             facecolor='w',
                             edgecolor='k')
            gs = GridSpec(4, 5)  # 2 rows, 3 columns
            # ax = fig.add_subplot(gs[0,0])
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :])
            ax3 = fig.add_subplot(gs[2, :])
            ax4 = fig.add_subplot(gs[3, :])

            # ax.text(0,0,'Domain number: {0}'.format(index))
            # ax.imshow(bl)
            # ax.axis('off')
            ax1.plot(self.time, self.data, label='dFoF', c='b')
            ax1.plot(self.time, self.dataFlt, label='Filtered', c='k')
            ax1.legend()
            ax2.plot(self.time,
                     self.data - self.dataFlt,
                     label='residuals',
                     c='orange')
            ax2.legend()
            ax3.plot(self.time, self.dataCopy, label='dFoF', c='b')
            ax3.scatter(timeIndex, eventVal, c='r', label='event')
            ax3.legend()
            ax4.plot(self.time, self.dataCopy, label='dFoF', c='b')
            ax4.eventplot(timeIndex, lineoffsets=-1, linelengths=0.5, color='k')
            a, b = ax1.get_ylim()
            ax2.set_ylim([a, b])

            ax4.set_xlabel('time(s)')
            ax1.set_ylabel('dfof')
            ax2.set_ylabel('dfof')
            ax3.set_ylabel('dfof')
            ax4.set_ylabel('dfof')
            ax1.legend()
            ax2.legend()
            ax3.legend()

            fig.tight_layout()
            # plt.savefig(savefnm)
            plt.show()

        return frameIndex, eventVal
