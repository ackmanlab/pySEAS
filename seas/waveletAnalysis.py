#!/usr/bin/env python3
'''
Adapted from __author__ = 'Mykhaylo Shumko' : __init__, waveletTransform, plotPower, lagNAutoCorr
All other functions written by Brian Mullen.

To find the global wavelet spectrum run these two scripts:
    wave = waveletAnalysis(data) #create object & runs wavelet transform
    wave.globalWaveletSpectrum() #runs function to create global wavelet spectrum
    
To find signal change:
    waveletAnalysis.tsSignal()
    There seems to be very little difference between binned data and full data
'''
import sys
import numpy as np
from seas.waveletFunctions import *
import matplotlib.pylab as plt
import matplotlib
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from datetime import datetime
from seas.signalanalysis import local_max, linear_regression, abline, lag_n_autocorr

from typing import List
import operator


class waveletAnalysis:

    def __init__(self, data: np.ndarray, fps: int, **kwargs):
        """
        A class for managing wavelet analysis. 
        Initialize the wavelet parameters and run the wavelet tranform.
        We have only fully implemented Morlet wavelet in this code. One should update the 
        paramters if other types of Wavelets are used.

        Functions:
            waveletTransform: Wavelet transform
            inverseWaveletTransform: inverse wavelet transform
            plotPower: visualization function to look at power
                function used in the GUI
            waveletFilter: 
            nanCOI: creates cone of influence (COI) 
                values outside the cone are assigned np.nan
            nanSig: creates np.nan values outside of significance 
                (used to visualize significant waveforms)
            familySig: creates several cutoff values based on the red noise autoregression
                based on percent cutoff value
            sumAcrossPeriod: Sum across defined frequencies
            globalWaveletSpectrum: Average across time to find significant frequencies
            averageWaveletPower: Average across frequency to find significant times when waveforms
                are present
            noiseFilter: Filters timeseries based on high and/or low pass filters and based on
                significance filters to get rid of low power noise. Used to filter global mean.

        Arguments:
            data: time course data
            fps: Frames per second, rate of data acquisition (default=10)
            verbose: Boolean on printed outputs (default=False)
            plot: Plots outcomes of several functions (default=False)
            mother: mother wavelet (default=Morelt)
            param: defining parameter for mother wavelet(default=4)
            j1: parameter for wavelet transform (see wavelet functions; 
                default=based on wavelt/parameters)
            pad: parameter for wavelet transform (see wavelet functions; 
                default=based on wavelt/parameters)
            dj: parameter for wavelet transform (see wavelet functions; 
                default=based on wavelt/parameters)
            cdelta: parameter for wavelet transform (see wavelet functions; 
                default=based on wavelt/parameters)
            psi0: constant factor to ensure a total energy of unity
                (see Torrence and Compo; page 65)
            siglvl: set significance level (default=0.95)
            lag1: defines autoregression red-noise model
        """

        assert data.ndim == 1, 'Time series is the wrong shape. It should be a 1-dim vector'

        self.dataCopy = data
        self.data = (data - np.mean(data)) / np.std(data, ddof=1)
        self.n = len(self.data)
        self.cadence = 1 / fps
        self.time = np.arange(self.n) * self.cadence

        # Default parameters.
        # Print/ plot statements.
        self.verbose = kwargs.get('verbose', False)
        self.plot = kwargs.get('plot', False)

        # Wavelet parameters.
        self.mother = kwargs.get('mother', 'MORLET')
        self.param = kwargs.get('param', 4)
        self.j1 = kwargs.get('j1', 80)
        self.pad = kwargs.get('pad', 1)
        self.dj = kwargs.get('dj', 0.125)
        self.s0 = kwargs.get('s0', 2 * self.cadence)
        self.cdelta = None
        if self.mother == 'MORLET' and self.param == 6:
            self.cdelta = 0.776
            self.psi0 = np.pi**(-0.25) * 0.85
        elif self.mother == 'MORLET' and self.param == 4:
            self.cdelta = 1.151
            self.psi0 = np.pi**(-0.25) * 0.85
        elif self.mother == 'DOG' and self.param == 2:
            self.cdelta = 3.541
            self.psi0 = None  #used in wavelet inverse, see (Torrence and Compo 1998)
        elif self.mother == 'DOG' and self.param == 6:
            self.cdelta = 1.966
            self.psi0 = None  #used in wavelet inverse, see (Torrence and Compo 1998)
        else:
            assert self.cdelta != None, 'Unknown c value based on wavelet choice'
        if self.psi0 == None:
            if verbose:
                'Unknown Psi 0, must be input before inverse can be calculated'
        # Noise modeling parameter.
        self.siglvl = kwargs.get('siglvl', 0.95)
        self.lag1 = 0.5 * (lag_n_autocorr(data, 1) + lag_n_autocorr(data, 2))

        self.waveletTransform()

    def waveletTransform(self):
        ''' 
        Runs wavelet transform
        Establishes the amount of power per wavelet at each timepoint
        Determines significance based on red noise spectra determined from the lag1 parameter

        Arguments:
            None

        Returns:
            None
        '''

        self.wave, self.period, self.scale, self.coi = \
            wavelet(self.data, self.cadence, self.pad, self.dj, self.s0, self.j1, self.mother, self.param)

        if len(self.time) != len(self.coi):
            self.coi = self.coi[1:]

        self.power = (np.abs(self.wave))**2  # Compute wavelet power spectrum.

        # Significance levels: (variance=1 for the normalized data).
        self.signif = wave_signif(([1.0]), dt=self.cadence, sigtest=0, scale=self.scale, \
            lag1=self.lag1, mother=self.mother, siglvl = self.siglvl)
        self.sig95 = self.signif[:, np.newaxis].dot(
            np.ones(
                self.n)[np.newaxis, :])  # Expand signif --> (J+1)x(N) array.
        self.sig95 = self.power / self.sig95  # Where ratio > 1, power is significant.

    def inverseWaveletTransform(self, waveFlt: np.ndarray = None):
        ''' 
        Inverse wavelet transform from 2d power spectra back to a timecourse

        Arguments:
            None

        Returns:
            None
        '''

        if waveFlt == None:
            waveFlt = self.waveFlt

        InvTranform = np.zeros(self.n)
        tansformConstant = (
            (self.dj * math.sqrt(self.cadence)) / (self.cdelta * self.psi0)
        )  # Reconstruction constant.

        for i in range(waveFlt.shape[0]):
            waveFlt[i, :] /= math.sqrt(self.period[i])
        for i in range(self.n):
            InvTranform[i] = np.sum(np.real(waveFlt[:, i]), axis=0)
        self.dataFlt = tansformConstant * InvTranform

    def plotPower(self, ax=None):
        '''
        Plot log power spectorgram, with cone of influence

        Arguments:
            ax: specify axis to plot on (see matplotlib.pyplot)

        Returns:
            None
        '''

        self.levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        colors = [
            'navy', 'slateblue', 'c', 'g', 'gold', 'orange', 'tomato', 'crimson'
        ]

        if ax == None:
            f = plt.figure()
            f, ax = plt.subplots(1)
        else:
            ax = np.ravel(ax)[0]

        # Max period is fourier_factor*S0*2^(j1*dj), fourier_factor = 3.97383530632.
        CS = ax.contourf(self.time,
                         self.period,
                         np.log2(self.power),
                         len(self.levels),
                         colors=colors)
        im = ax.contourf(CS, levels=np.log2(self.levels), colors=colors)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Period (s)')
        ax.set_title('Wavelet Power Spectrum')
        # 95 significance contour, levels at -99 (fake) and 1 (95# signif).
        ax.contour(self.time, self.period, self.sig95, [-99, 1], colors='k')
        # Cone-of-influence, anything "below" is dubious.
        ax.fill_between(self.time,
                        np.max(self.period),
                        self.coi,
                        alpha=0.5,
                        facecolor='white',
                        zorder=3)
        ax.plot(self.time, self.coi, 'k')
        # Format y-scale.

        # Different matplotlib versions available for python < 3.8.
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
        # Set up the size and location of the colorbar.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        plt.colorbar(im, cax=cax, orientation='horizontal')

    def waveletFilter(self,
                      lowerPeriod: float = None,
                      upperPeriod: float = None,
                      sigLevel: float = 1):
        '''
        High-pass and low-pass filter possibilities.

        In seas.ica filter_method == 'wavelet' will filter high frequencies past the nyquist
        smapling rate and very low power spectra (significance ratio less than 0.25)

        See seas.waveletAnalysis.noiseFilter

        Arguments:
            lowerPeriod: High pass filter, specify lowest period and 
                it will exclude all under that period
            upperPeriod: Low pass filter, specify highest period and 
                it will exclude all over that period
            sigLevel: Significance filter, eliminate low power waveforms
                (0.25 is used in noiseFilter)

        Returns:
            None
        '''

        self.waveFlt = self.wave.copy()

        # Band pass filter:
        # Zero out parts of the wavlet space that we don't want to reconstruct.
        if lowerPeriod != None:
            if lowerPeriod > self.period[0]:
                lower_ind = np.where(self.period < lowerPeriod)[0][-1]
                self.waveFlt[:lower_ind, :] = 0
        if upperPeriod != None:
            if upperPeriod < self.period[-1]:
                upper_ind = np.where(self.period > upperPeriod)[0][0]
                self.waveFlt[upper_ind:, :] = 0

        # Significance filter:
        notSigInd = np.where(
            self.sig95 < sigLevel
        )  # Only pass data that has power of (100% - sigThreshold). Usually sigThreshold is 95%. Was 0.25.
        self.waveFlt[notSigInd] = 0

    def nanCOI(self):
        '''
        Get rid of all values outside the cone of influence.
        Sets those values to np.nan

        Arguments:
            None

        Returns:
            None
        '''
        self.nanCOImat = self.power.copy()
        for i in range(self.power.shape[1]):
            cutoff = np.where(self.coi[i] < self.period)
            self.nanCOImat[cutoff, i] = np.nan

    def nanSig(self):
        '''
        Get rid of all values not significant.
        Sets those values to np.nan

        Arguments:
            None

        Returns:
            None
        '''
        self.nanSigmat = self.wave
        self.nanSigmat[np.where(wavelet.sig95 < 1)] = np.nan

    def familySig(self,
                  sigList: List[float, float, float,
                                float] = [0.9, 0.95, 0.99, 0.999],
                  dof: int = -1,
                  sigtest: float = 0):
        '''
        Plot a family of significance curves for visualization and analysis.

        Arguments:
            sigList: List or float of significant values
            dof: degrees of freedom (see waveletFunctions)
            sigtest: which significance test is used (see waveletFunctions for
                more information)
                    0: regular chi-square test (for full wavelet transform; non-smoothed)
                    1: time-average test (for globalWaveletSpectrum)
                    2: scale-average test (for averageWaveletPower)

        Returns:
            fam_significance: significance cutoffs
            sigList: List of requested significants 
        '''

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

    def sumAcrossPeriod(self, perLim: List[float, float] = [0, 100]):
        '''
        Sum wavelet power across select periods.
        
        Arguments:
            perLim: tuple for lower and upper periods. Power summed
                between these defined periods. 

        Returns:
            period_sum: sum across the power of defined frequencies
        '''

        if self.verbose:
            print('Summing across {0} to {1} periods on wavelet run with mother {2} at paramter {3}'.\
                  format(perLim[0], perLim[1], self.mother, self.param) )

        l_per_lim = np.min(np.where(perLim[0] < self.period))
        u_per_lim = np.min(np.where(perLim[1] < self.period))

        if not hasattr(self, 'nanCOImat'):
            self.nanCOI()

        period_sum = (self.dj * self.cadence) / self.cdelta * np.nansum(
            (self.nanCOImat[l_per_lim:u_per_lim]**2 /
             self.period[l_per_lim:u_per_lim, None]),
            axis=0)

        return np.squeeze(period_sum)

    def globalWaveletSpectrum(self):
        '''
        Global Wavelet Spectrum, Average across time to find significant 
        prominant frequencies
        
        Arguments:
            None

        Returns:
            None
        '''
        if self.verbose:
            print('Assessing wavelet mother {0} at paramter {1}'.format(
                self.mother, self.param))

        # Calulate the global self spectrum.
        self.nanCOI()
        # If np.sum(~np.isnan(self.nanCOImat))!=0:.
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

        # Calculate the average significance.
        self.gws_sig, self.gws_sigList = self.familySig(sigList=[0.95],
                                                        dof=self.period_size,
                                                        sigtest=1)

        if self.verbose:
            print('Auto-correlation value: {0:.4g}'.format(self.lag1))
        # Determine fourier wavelength.
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

        # Find the lowest and highest frequencies that are still significant.
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

            # Self period graph.
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

            # Fourier space lambda.
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

    def averageWaveletPower(self, perLim: List[float, float] = [0.25, 8]):
        '''
        Average wavelet power, Average across periods to find significant 
        prominant times of activity.
        
        Arguments:
            perLim: tuple for lower and upper periods. Power summed
                between these defined periods. 

        Returns:
            None
        '''
        assert len(perLim) == 2, 'Period limit list must only include 2 values'

        if self.verbose:
            print('Creating scaled average of the timeseries, \
             created with wavelet mother {0} at paramter {1}'.format(
                self.mother, self.param))

        self.period_sum = self.sumAcrossPeriod(perLim=perLim)
        if self.period_sum.shape[0] != self.time.shape[0]:
            dif = self.period_sum.shape[0] - self.time.shape[0]
            if dif < 0:
                self.period_sum = np.append(self.period_sum,
                                            np.zeros(np.abs(dif)))
            else:
                self.period_sum = self.period_sum[:self.period.shape[0]]

        # calculate the average significance
        self.sig_period_sum, _ = self.familySig(sigList=[0.95],
                                                dof=perLim,
                                                sigtest=2)

        # Find coordinates of local max values.
        mx_wav, mx_gws = local_max(self.time, self.period_sum)

        # Return only those above significance threshold.
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

    def noiseFilter(self,
                    lowerPeriod: float = None,
                    upperPeriod: float = 10,
                    sigLevel: float = 0.25):
        '''
        High-pass and low-pass filter possibilities.

        In seas.ica filter_method == 'wavelet' will filter high frequencies past the nyquist
        smapling rate and very low power spectra (significance ratio less than 0.25)

        See seas.waveletAnalysis.noiseFilter

        Arguments:
            lowerPeriod: High pass filter, specify lowest period and 
                it will exclude all under that period
            upperPeriod: Low pass filter, specify highest period and 
                it will exclude all over that period
            sigLevel: Significance filter, eliminate low power waveforms
                (0.25 is used in noiseFilter)

        Returns:
            filtData: inverse wavlet transform based on removing periods/power
                defined in the parameters
        '''

        if lowerPeriod is None:
            lowerPeriod = 2 * self.cadence  # Nyquist sampling rate.

        self.waveletFilter(lowerPeriod=lowerPeriod,
                           upperPeriod=upperPeriod,
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
