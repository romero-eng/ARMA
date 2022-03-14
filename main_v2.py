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


def showPlots(exponent, f, abs_h_f, abs_h_f_theo, abs_h_f_emp):

    f_begin = f[0]
    f_end = f[len(f) - 1]

    mag_plot_ax_idxs         = np.array([0, 0])
    mag_plot_dB_ax_idxs      = np.array([1, 0])
    theo_mag_plot_ax_idxs    = np.array([0, 1])
    theo_mag_plot_dB_ax_idxs = np.array([1, 1])
    emp_mag_plot_ax_idxs     = np.array([0, 2])
    emp_mag_plot_dB_ax_idxs  = np.array([1, 2])
    
    idxs_matrix = \
        np.vstack((mag_plot_ax_idxs,
                   mag_plot_dB_ax_idxs,
                   theo_mag_plot_ax_idxs,
                   theo_mag_plot_dB_ax_idxs,
                   emp_mag_plot_ax_idxs,
                   emp_mag_plot_dB_ax_idxs))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    num_rows = np.amax(idxs_matrix[:, 0]) + 1
    num_cols = np.amax(idxs_matrix[:, 1]) + 1
    [fig, axs] = plt.subplots(num_rows, num_cols, figsize=(16, 7))

    if(len(axs.shape) == 2):
        mag_plot_ax         = axs[        mag_plot_ax_idxs[0],         mag_plot_ax_idxs[1]]
        mag_plot_dB_ax      = axs[     mag_plot_dB_ax_idxs[0],      mag_plot_dB_ax_idxs[1]]
        theo_mag_plot_ax    = axs[   theo_mag_plot_ax_idxs[0],    theo_mag_plot_ax_idxs[1]]
        theo_mag_plot_dB_ax = axs[theo_mag_plot_dB_ax_idxs[0], theo_mag_plot_dB_ax_idxs[1]]
        emp_mag_plot_ax     = axs[    emp_mag_plot_ax_idxs[0],     emp_mag_plot_ax_idxs[1]]
        emp_mag_plot_dB_ax  = axs[ emp_mag_plot_dB_ax_idxs[0],  emp_mag_plot_dB_ax_idxs[1]]
    else:
        mag_plot_ax         = axs[np.amax(        mag_plot_ax_idxs)]
        mag_plot_dB_ax      = axs[np.amax(     mag_plot_dB_ax_idxs)]
        theo_mag_plot_ax    = axs[np.amax(   theo_mag_plot_ax_idxs)]
        theo_mag_plot_dB_ax = axs[np.amax(theo_mag_plot_dB_ax_idxs)]
        emp_mag_plot_ax     = axs[np.amax(    emp_mag_plot_ax_idxs)]
        emp_mag_plot_dB_ax  = axs[np.amax( emp_mag_plot_dB_ax_idxs)]

    mag_plot_ax.plot(f, abs_h_f)
    mag_plot_ax.set_xlim([f_begin, f_end])
    mag_plot_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    mag_plot_ax.axvline(f_c, color='k')
    mag_plot_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    mag_plot_ax.grid()
    mag_plot_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    mag_plot_ax.set_xlabel('Frequency (Hz)')
    mag_plot_ax.set_ylabel(r'$|h_(f)|$')
    mag_plot_ax.set_title('Frequency Magnitude Response')

    mag_plot_dB_ax.plot(f, 10*np.log10(abs_h_f))
    mag_plot_dB_ax.set_xlim([f_begin, f_end])
    mag_plot_dB_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    mag_plot_dB_ax.axvline(f_c, color='k')
    mag_plot_dB_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    mag_plot_dB_ax.grid()
    mag_plot_dB_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    mag_plot_dB_ax.set_xlabel('Frequency (Hz)')
    mag_plot_dB_ax.set_ylabel(r'$|h_(f)|_{dB}$')
    mag_plot_dB_ax.set_title('Frequency Magnitude Response (dB)')

    theo_mag_plot_ax.plot(f, abs_h_f_theo)
    theo_mag_plot_ax.set_xlim([f_begin, f_end])
    theo_mag_plot_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    theo_mag_plot_ax.axvline(f_c, color='k')
    theo_mag_plot_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    theo_mag_plot_ax.grid()
    theo_mag_plot_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    theo_mag_plot_ax.set_xlabel('Frequency (Hz)')
    theo_mag_plot_ax.set_ylabel(r'$|h_(f)|$')
    theo_mag_plot_ax.set_title('Theoretical Frequency Magnitude Response')

    theo_mag_plot_dB_ax.plot(f, 10*np.log10(abs_h_f_theo))
    theo_mag_plot_dB_ax.set_xlim([f_begin, f_end])
    theo_mag_plot_dB_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    theo_mag_plot_dB_ax.axvline(f_c, color='k')
    theo_mag_plot_dB_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    theo_mag_plot_dB_ax.grid()
    theo_mag_plot_dB_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    theo_mag_plot_dB_ax.set_xlabel('Frequency (Hz)')
    theo_mag_plot_dB_ax.set_ylabel(r'$|h_(f)|_{dB}$')
    theo_mag_plot_dB_ax.set_title('Theoretical Frequency Magnitude Response (dB)')

    emp_mag_plot_ax.plot(f, abs_h_f_emp)
    emp_mag_plot_ax.set_xlim([f_begin, f_end])
    emp_mag_plot_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    emp_mag_plot_ax.axvline(f_c, color='k')
    emp_mag_plot_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    emp_mag_plot_ax.grid()
    emp_mag_plot_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    emp_mag_plot_ax.set_xlabel('Frequency (Hz)')
    emp_mag_plot_ax.set_ylabel(r'$|h_(f)|$')
    emp_mag_plot_ax.set_title('Empirical Frequency Magnitude Response')

    emp_mag_plot_dB_ax.plot(f, 10*np.log10(abs_h_f_emp))
    emp_mag_plot_dB_ax.set_xlim([f_begin, f_end])
    emp_mag_plot_dB_ax.axvline(f_c - delta_f, color='r', linestyle='--')
    emp_mag_plot_dB_ax.axvline(f_c, color='k')
    emp_mag_plot_dB_ax.axvline(f_c + delta_f, color='r', linestyle='--')
    emp_mag_plot_dB_ax.grid()
    emp_mag_plot_dB_ax.ticklabel_format(style='sci', axis='x', scilimits=(exponent,exponent))
    emp_mag_plot_dB_ax.set_xlabel('Frequency (Hz)')
    emp_mag_plot_dB_ax.set_ylabel(r'$|h_(f)|_{dB}$')
    emp_mag_plot_dB_ax.set_title('Empirical Frequency Magnitude Response (dB)')
    
    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):

    withinUnitCircle = False
    MA_or_AR = 'MA'
    lowpass_or_highpass = 'lowpass'

    exponent = 4
    f_s     = (90.0)*(10**exponent)
    f_c     = (10.0)*(10**exponent)
    delta_f = ( 2.5)*(10**exponent)
    min_dB = -20

    f_bin_width = 10
    AUC = 0.99
    min_approximation_dB = -10

    [f, abs_h_f] = calculateFrequencyMagnitudeResponse(lowpass_or_highpass, f_s, f_c, delta_f, f_bin_width, AUC, min_dB)

    norm_f = f/f_s
    norm_f_bin_width = f_bin_width/f_s

    [abs_h_f_dB, 
     root_repeating_factor, 
     squared_reduced_abs_h_f_cheb_poly_root_dicts_list] = \
        utility.spectralEstimation.estimateUniqueSquaredSpectralChebyshevPolynomialRoots(norm_f_bin_width, 
                                                                                         norm_f, 
                                                                                         abs_h_f, 
                                                                                         min_approximation_dB, 
                                                                                         withinUnitCircle, 
                                                                                         MA_or_AR,
                                                                                         10**-5)

    [MA_z_coefs, 
     AR_z_coefs] =\
        utility.spectralEstimation.estimateSpectralZTransCoefs(root_repeating_factor, 
                                                               squared_reduced_abs_h_f_cheb_poly_root_dicts_list)

    [abs_h_f_theo, 
     angle_deg_h_f_theo] = \
        utility.spectralEstimation.estimateSpectralMagnitudeAndPhase(norm_f, 
                                                                     root_repeating_factor, 
                                                                     squared_reduced_abs_h_f_cheb_poly_root_dicts_list)
    
    norm_value = abs_h_f_theo[np.floor((f_c - delta_f)/f_bin_width).astype(int)]
    abs_h_f_theo = abs_h_f_theo/norm_value
    MA_z_coefs = MA_z_coefs/norm_value

    [_, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs, 2*np.pi*f)
    abs_h_f_emp = np.abs(h_f_emp)

    print(root_repeating_factor)

    showPlots(exponent, f, abs_h_f, abs_h_f_theo, abs_h_f_emp)
