import numpy as np
import scipy.signal as dsp
import scipy.special as special
import matplotlib.pyplot as plt
import utility


def calculateFrequencyMagnitudeResponse(lowpass_or_highpass, f_s, f_c, delta_f, f_bin_width, AUC, min_dB):

    if(lowpass_or_highpass == 'lowpass'):
        sign = -1
    elif(lowpass_or_highpass == 'highpass'):
        sign = 1
    else:
        raise ValueError('The ''lowpass_or_highpass'' variable must either be ''lowpass'' or ''highpass''.')

    f = np.arange(f_bin_width, (f_s/2) + f_bin_width, f_bin_width)
    sigma = delta_f/np.log((1 + AUC)/(1 - AUC))
    epsilon = 10**(min_dB/10)
    abs_h_f = (1/(1 + epsilon))*(special.expit(sign*(f - f_c)/sigma) + epsilon)

    return [f, abs_h_f]


def showPlots(f, abs_h_f):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    [fig, axs] = plt.subplots(1, 2, figsize=(10, 4))
    mag_plot_ax    = axs[0]
    mag_plot_dB_ax = axs[1]

    mag_plot_ax.plot(f, abs_h_f)
    mag_plot_ax.set_xlim([f[0], f[len(f) - 1]])
    mag_plot_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    mag_plot_ax.axvline(f_c, color='k')
    mag_plot_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    mag_plot_ax.grid()
    mag_plot_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    mag_plot_ax.set_xlabel('Frequency (Hz)')
    mag_plot_ax.set_ylabel(r'$|h_(f)|$')
    mag_plot_ax.set_title('Frequency Magnitude Response')

    mag_plot_dB_ax.plot(f, 10*np.log10(abs_h_f))
    mag_plot_dB_ax.set_xlim([f[0], f[len(f) - 1]])
    mag_plot_dB_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    mag_plot_dB_ax.axvline(f_c, color='k')
    mag_plot_dB_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    mag_plot_dB_ax.grid()
    mag_plot_dB_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    mag_plot_dB_ax.set_xlabel('Frequency (Hz)')
    mag_plot_dB_ax.set_ylabel(r'$|h_(f)|_{dB}$')
    mag_plot_dB_ax.set_title('Frequency Magnitude Response (dB)')
    
    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):


    exponent = 4

    withinUnitCircle = False
    MA_or_AR = 'MA'
    lowpass_or_highpass = 'lowpass'
    f_s     = (90.0)*np.power(10, exponent)
    f_c     = (10.0)*np.power(10, exponent)
    delta_f = ( 2.5)*np.power(10, exponent)

    f_bin_width = 0.01
    AUC = 0.99
    min_dB = -120

    [f, abs_h_f] = calculateFrequencyMagnitudeResponse(lowpass_or_highpass, f_s, f_c, delta_f, f_bin_width, AUC, min_dB)

    [abs_h_f_dB, 
     root_repeating_factor, 
     squared_reduced_abs_h_f_cheb_poly_root_dicts_list] = \
        utility.spectralEstimation.estimateUniqueSquaredSpectralChebyshevPolynomialRoots(f_bin_width, 
                                                                                         f, 
                                                                                         abs_h_f, 
                                                                                         min_dB, 
                                                                                         withinUnitCircle, 
                                                                                         MA_or_AR)

    [MA_z_coefs, 
     AR_z_coefs] =\
        utility.spectralEstimation.estimateSpectralZTransCoefs(root_repeating_factor, 
                                                               squared_reduced_abs_h_f_cheb_poly_root_dicts_list)

    [abs_h_f_theo, 
     angle_deg_h_f_theo] = \
        utility.spectralEstimation.estimateSpectralMagnitudeAndPhase(f, 
                                                                     root_repeating_factor, 
                                                                     squared_reduced_abs_h_f_cheb_poly_root_dicts_list)

    showPlots(f, abs_h_f)
