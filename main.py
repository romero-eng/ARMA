import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import utility


def calculate_frequency_magnitude_response(f_c, delta_f, transition_bandwidth, transition_bandwidth_stop_perc, abs_h_f_c_stop_dB):
    
    f_c_2 = f_c + transition_bandwidth_stop_perc*transition_bandwidth
    f_c_1 = f_c_2 - transition_bandwidth

    abs_h_f_c_2 = 10**(abs_h_f_c_stop_dB/10)
    abs_h_f_c_1 = 1 - abs_h_f_c_2

    sigma = (f_c_2 - f_c_1)/np.log((abs_h_f_c_1/abs_h_f_c_2)*((1 - abs_h_f_c_2)/(1 - abs_h_f_c_1)))
    mu = f_c_1  - sigma*np.log((1 - abs_h_f_c_1)/abs_h_f_c_1)
  
    f = np.arange(0, 0.5 + delta_f, delta_f)
    z_score_f = (f - mu)/sigma
    abs_h_f = 1/(1 + np.exp(z_score_f))

    abs_h_f_dB = 10*np.log10(abs_h_f)

    return [f_c_1, f_c_2, f_c, f, abs_h_f, abs_h_f_dB]


def showPlots(f_c, f_c_1, f_c_2, f, abs_h_f, abs_h_f_dB, abs_h_f_emp, abs_h_f_emp_dB):

    abs_h_f_idxs        = np.array([0, 0])
    abs_h_f_dB_idxs     = np.array([0, 1])
    abs_h_f_emp_idxs    = np.array([1, 0])
    abs_h_f_emp_dB_idxs = np.array([1, 1])

    idxs_matrix = \
        np.vstack((abs_h_f_idxs,
                   abs_h_f_dB_idxs,
                   abs_h_f_emp_idxs,
                   abs_h_f_emp_dB_idxs))
    
    num_rows = np.amax(idxs_matrix[:, 0]) + 1
    num_cols = np.amax(idxs_matrix[:, 1]) + 1
    [fig, axs] = plt.subplots(num_rows, num_cols)

    if(len(axs.shape) == 2):
        abs_h_f_axis        = axs[       abs_h_f_idxs[0],        abs_h_f_idxs[1]]
        abs_h_f_dB_axis     = axs[    abs_h_f_dB_idxs[0],     abs_h_f_dB_idxs[1]]
        abs_h_f_emp_axis    = axs[   abs_h_f_emp_idxs[0],    abs_h_f_emp_idxs[1]]
        abs_h_f_emp_dB_axis = axs[abs_h_f_emp_dB_idxs[0], abs_h_f_emp_dB_idxs[1]]
    else:
        abs_h_f_axis        = axs[np.amax(       abs_h_f_idxs)]
        abs_h_f_dB_axis     = axs[np.amax(    abs_h_f_dB_idxs)]
        abs_h_f_emp_axis    = axs[np.amax(   abs_h_f_emp_idxs)]
        abs_h_f_emp_dB_axis = axs[np.amax(abs_h_f_emp_dB_idxs)]
    
    abs_h_f_axis.plot(f, abs_h_f)
    abs_h_f_axis.set_xlim([0, 0.5])
    #abs_h_f_axis.set_ylim([-0.1, 1.1])
    abs_h_f_axis.axvline(f_c_1, color='r')
    abs_h_f_axis.axvline(f_c, color='k')
    abs_h_f_axis.axvline(f_c_2, color='r')
    abs_h_f_axis.grid()
    abs_h_f_axis.set_xlabel('Frequency (Hz)')
    abs_h_f_axis.set_ylabel(r'$|h_(f)|$')
    abs_h_f_axis.set_title('Target Frequency Magnitude Response')

    abs_h_f_dB_axis.plot(f, abs_h_f_dB)
    abs_h_f_dB_axis.set_xlim([0, 0.5])
    abs_h_f_dB_axis.axvline(f_c_1, color='r')
    abs_h_f_dB_axis.axvline(f_c, color='k')
    abs_h_f_dB_axis.axvline(f_c_2, color='r')
    abs_h_f_dB_axis.grid()
    abs_h_f_dB_axis.set_xlabel('Frequency (Hz)')
    abs_h_f_dB_axis.set_ylabel(r'$|h_(f)|_{dB}$')
    abs_h_f_dB_axis.set_title('Target Frequency Magnitude Decibel Response')

    abs_h_f_emp_axis.plot(f, abs_h_f_emp)
    abs_h_f_emp_axis.set_xlim([0, 0.5])
    #abs_h_f_emp_axis.set_ylim([-0.1, 1.1])
    abs_h_f_emp_axis.axvline(f_c_1, color='r')
    abs_h_f_emp_axis.axvline(f_c, color='k')
    abs_h_f_emp_axis.axvline(f_c_2, color='r')
    abs_h_f_emp_axis.grid()
    abs_h_f_emp_axis.set_xlabel('Frequency (Hz)')
    abs_h_f_emp_axis.set_ylabel(r'$|h_(f)|$')
    abs_h_f_emp_axis.set_title('Acquired Frequency Magnitude Response')

    abs_h_f_emp_dB_axis.plot(f, abs_h_f_emp_dB)
    abs_h_f_emp_dB_axis.set_xlim([0, 0.5])
    abs_h_f_emp_dB_axis.axvline(f_c_1, color='r')
    abs_h_f_emp_dB_axis.axvline(f_c, color='k')
    abs_h_f_emp_dB_axis.axvline(f_c_2, color='r')
    abs_h_f_emp_dB_axis.grid()
    abs_h_f_emp_dB_axis.set_xlabel('Frequency (Hz)')
    abs_h_f_emp_dB_axis.set_ylabel(r'$|h_(f)|_{dB}$')
    abs_h_f_emp_dB_axis.set_title('Acquired Frequency Magnitude Decibel Response')

    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):

    f_c = 0.2
    delta_f = 0.00001
    transition_bandwidth = 0.075
    transition_bandwidth_stop_perc = 0.7
    abs_h_f_c_stop_dB = -15
    min_approximation_dB = -10

    [f_c_1, f_c_2, f_c, f, abs_h_f, abs_h_f_dB] = \
        calculate_frequency_magnitude_response(f_c, delta_f, transition_bandwidth, transition_bandwidth_stop_perc, abs_h_f_c_stop_dB)

    reduction_factor = np.ceil(-np.amax(np.abs(abs_h_f_dB))/min_approximation_dB)
    reduced_abs_h_f = np.power(abs_h_f, 1/reduction_factor)
    squared_reduced_abs_h_f = np.square(reduced_abs_h_f)
    squared_reduced_abs_h_f_cheb_poly_roots = utility.chebyshevSpectrumCalculations.calculateChebyshevSpectrumPolynomialRoots(f, delta_f, squared_reduced_abs_h_f, 40)
    squared_reduced_abs_h_f_cheb_poly_root_dicts_list = utility.magnitudeDomainRoots.convertLimitedRootsArrayToRootsDictList(False, 'MA', squared_reduced_abs_h_f_cheb_poly_roots)

    MA_z_coefs = np.array([1])
    AR_z_coefs = np.array([1])
    for root_dict in squared_reduced_abs_h_f_cheb_poly_root_dicts_list:
        [tmp_MA_z_coefs, tmp_AR_z_coefs] = utility.frequencyResponseAndZTransformCalculations.z_trans_coefs(root_dict)
        MA_z_coefs = np.polynomial.polynomial.polymul(MA_z_coefs, tmp_MA_z_coefs)
        AR_z_coefs = np.polynomial.polynomial.polymul(AR_z_coefs, tmp_AR_z_coefs)

    tmp_MA_z_coefs = MA_z_coefs
    tmp_AR_z_coefs = AR_z_coefs
    for tmp_idx in np.arange(0, reduction_factor, 1):
        MA_z_coefs = np.polynomial.polynomial.polymul(MA_z_coefs, tmp_MA_z_coefs)
        AR_z_coefs = np.polynomial.polynomial.polymul(AR_z_coefs, tmp_AR_z_coefs)

    [_, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs, 2*np.pi*f)
    abs_h_f_emp = np.abs(h_f_emp)
    MA_z_coefs = MA_z_coefs/abs_h_f_emp[np.floor(f_c_1/delta_f).astype(int)]

    [_, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs, 2*np.pi*f)
    abs_h_f_emp = np.abs(h_f_emp)
    abs_h_f_emp_dB = 10*np.log10(abs_h_f_emp)

    showPlots(f_c, f_c_1, f_c_2, f, abs_h_f, abs_h_f_dB, abs_h_f_emp, abs_h_f_emp_dB)
