import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
from collections.abc import Iterable


def createPlots(f_c, f, abs_h_f):

    def plotCutoffFreqs(f_c, delta_f):

            plt.axvline(f_c - delta_f, color='r', linestyle='--')
            plt.axvline(f_c, color='k')
            plt.axvline(f_c + delta_f, color='r', linestyle='--')

    plt.figure()
    plt.plot(f, abs_h_f)
    plt.xlim([f[0], f[len(f) - 1]])
    plt.grid()

    if(not isinstance(f_c, Iterable)):
        plotCutoffFreqs(f_c, delta_f)
    else:
        for f_c_i in f_c:
            plotCutoffFreqs(f_c_i, delta_f)
    
    plt.show()


if(__name__=='__main__'):

    filter_type = 'lowpass';  f_c = 20
    #filter_type = 'highpass'; f_c = 20
    #filter_type = 'bandpass'; f_c = [20, 80]
    #filter_type = 'bandstop'; f_c = [20, 80]

    f_s = 200
    delta_f = 7
    f_bin_width = 0.1

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    f = np.arange(f_bin_width, (f_s/2) + f_bin_width, f_bin_width)

    raiseCutoffFreqValueErrorFlag = False
    raiseFilterTypeValueErrorFlag = False

    if(type(f_c) is int):
        num_f_c = 1
    elif(type(f_c) is list):
        num_f_c = np.array(f_c).shape[0]
    else:
        raiseCutoffFreqValueErrorFlag = True
    
    if(not raiseCutoffFreqValueErrorFlag):
        if(num_f_c == 1):

            if(filter_type == 'lowpass'):
                abs_h_f = special.expit(-(f - f_c))
            elif(filter_type == 'highpass'):
                abs_h_f = special.expit(f - f_c)
            else:
                raiseFilterTypeValueErrorFlag = True

        elif(num_f_c == 2):

            if(filter_type == 'bandpass'):
                abs_h_f = special.expit(f - f_c[0])*special.expit(-(f - f_c[1]))
            elif(filter_type == 'bandstop'):
                abs_h_f = special.expit(-(f - f_c[0])) + special.expit(f - f_c[1])
            else:
                raiseFilterTypeValueErrorFlag = True

        else:
            raiseCutoffFreqValueErrorFlag = True

    if(raiseCutoffFreqValueErrorFlag):
        raise ValueError('Cutoff frequencies are not interpretable.')

    if(raiseFilterTypeValueErrorFlag):
        raise ValueError('Filter type is not interpretable.')

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    createPlots(f_c, f, abs_h_f)
