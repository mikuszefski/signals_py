# coding=utf-8
"""
NMI 2018-06-27
class to analyse wave data
Note: this is for Python3, i.e. type(a/b) = <float> even if a and b are
integers
"""

import numpy as np
from scipy.signal import flattop, hann, gaussian ## fft window functions
from random import random


class Wave( object ):
    """
    class that simplifies some fft procedures on periodic data
    """

    def __init__( self, waveData, sampleRate ):
        self._sampleRate = sampleRate
        self._sampleSpacing = 1 / self._sampleRate
        self._wave = np.array ( waveData, dtype=np.double ) ## real signal
        self._wavePoints = len( waveData )
        self._timelist = None
        self._flist = None
        self._fft = None

    def wave( self ):
        return( self._wave )

    def number_of_points( self ):
        return self._wavePoints

    def sample_rate( self ):
        return self._sampleRate

    def sample_spacing( self ):
        return self._sampleSpacing

    def make_time_x( self ):
        """
        generate an x-axis according to the given sample rate
        """
        if self._timelist:
            out = self._timelist 
        else:
            data = np.arange( self._wavePoints )
            out = data * self._sampleSpacing
            self._timelist = out
        return out

    def make_freq_x( self ):
        """
        provides the list of frequencies that correspnd to the fft
        values follow the order (0, ...(n - 1 )/2, -( u ), ..., -1 ),
        where u depends on whether  n is even or odd. 
        see numpys fftfreq()
        """
        if self._flist is None:
            out = np.fft.fftfreq(
                self._wavePoints,
                self._sampleSpacing
            )
            self._flist = out
        else:
            out = self._flist
        return out

    def make_window( self, windowType=['FlatTop'], **kwargs):
        """
        window functions for fft
        """
        if windowType[0] == 'FlatTop':
            window_func = flattop
        elif windowType[0] == 'Hann':
            window_func = hann
        elif windowType[0] == 'Gaussian':
            window_func = gaussian
        else:
            window_func = np.ones
        window = window_func( self._wavePoints, *(windowType[1:]) )
        normalisation = self._wavePoints / window.sum()
        out = np.array( window ) * normalisation
        return out

    def fft( self, **kwargs ):
        """
        fft of the wave using a window function given in kwargs.
        Additional arguments are passed to the window function itself.
        """
        out = np.fft.fft( self._wave * self.make_window( **kwargs ) )
        return out / self._wavePoints

    def power_spectrum_single_sided( self, **kwargs ):
        powerS = self.fft( **kwargs )
        powerS = ( powerS * np.conj( powerS ) ).real
        powerS =  4 * powerS[ : self._wavePoints // 2 ] ### single sided so: 2 * Amplitude * 2 conj( Amplitude )
        powerF = self.make_freq_x()
        powerF = powerF[ : self._wavePoints // 2 ]
        return powerF, powerS

    def spectrum_single_sided( self, **kwargs ):
        pf, ps = self.power_spectrum_single_sided( **kwargs )
        return pf, np.sqrt( ps )

    def harmonic_positions_list( self, frequency, hMax=5 ):
        """
        provide the positiond of the first n harmonics
        in the frequency data
        """
        out = [
            1 + int(
                i * self._wavePoints * frequency * self._sampleSpacing
            )
            for i in range( 1, hMax + 1, 1 )
        ]
        return out

    def get_base_frequency( self, relativeSigmaQuotient=8 ):
        """
        uses the fact that the fourier of a gaussian is a gaussian.
        folding the virtual deltapeak of a frequency results in a gaussian.
        taking the log and looking at two consecutive bins, preferably near the maximum, 
        allows to calculate the position of the maximum, i.e. the frequency
        (square terms of the gaussians cancel and a linear equation is solved)
        
        8 is my choice ... seems to work well... 
        just narrow enough to be zero on the edges and not mix peaks
        """
        sigma = self._wavePoints / relativeSigmaQuotient
        tau = self._wavePoints / ( 2.0 * np.pi * sigma )
        pf, ps = self.spectrum_single_sided(
            windowType=[ 'Gaussian', sigma ]
        )
        n = np.argmax( ps ) - 1
        slg = tau**2 * np.log( ps[ n + 1 ] / ps[ n ] )
        theoreticalbin = slg + n + 0.5
        return theoreticalbin * self._sampleRate / self._wavePoints

    def thd_f( self, frequency=None, harmonicslist=None ):
        """
        This is sqrt(v1^2 + v2^2 + v3^2 +...)/ v0
        so it is 0 for pure signal and diverges if all intensity goes 
        into higher orders 
        """
        if frequency:
            baseF = frequency
        else:
            baseF = self.get_base_frequency()
        if harmonicslist is None:
            harmonicslist = range( 1, 5 + 1 )
        harmonicslist = sorted( harmonicslist )
        if 1 not in harmonicslist:
            harmonicslist = [1] + [f for f in harmonicslist]
        hM = max( harmonicslist )
        hpl = self.harmonic_positions_list( baseF, hMax=hM )
        hpl = [ hpl[ i - 1 ] for i in harmonicslist ]
        hpl = [ p for p in hpl if p < self._wavePoints // 2 ]           ## position cannot be higher than data available
        spec = self.spectrum_single_sided( windowType=[ 'FlatTop' ] )
        vList = [ spec[1][ p ] for p in hpl ]
        out = sum( [ x**2 for x in vList[ 1: ] ] )
        out = np.sqrt( out ) / vList[0]
        return out, vList, harmonicslist
