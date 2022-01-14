import numpy as np
import matplotlib.pyplot as plt
import utility


def calculate_frequency_magnitude_response(f, f_c, f_delta, f_delta_stop_perc, abs_h_f_c_stop_dB):
  
    f_c_2 = f_c + f_delta_stop_perc*f_delta
    f_c_1 = f_c_2 - f_delta

    abs_h_f_c_2 = 10**(abs_h_f_c_stop_dB/10)
    abs_h_f_c_1 = 1 - abs_h_f_c_2

    sigma = (f_c_2 - f_c_1)/np.log((abs_h_f_c_1/abs_h_f_c_2)*((1 - abs_h_f_c_2)/(1 - abs_h_f_c_1)))
    mu = f_c_1  - sigma*np.log((1 - abs_h_f_c_1)/abs_h_f_c_1)
  
    z_score_f = (f - mu)/sigma
    abs_h_f = 1/(1 + np.exp(z_score_f))

    return [f_c_1, f_c_2, f_c, abs_h_f]


def calculateChebyshevPolynomialCoefficients(t, t_delta, f_t, N):

    flipped_cheb_series_n = np.zeros(N)
    flipped_cheb_series_n[0] = 2*np.sum(f_t)*t_delta
    for n in np.arange(1, N, 1):
        flipped_cheb_series_n[n] = 4*np.sum(f_t*np.cos(2*np.pi*n*t))*t_delta

    flipped_cheb_poly_coeffs = np.polynomial.chebyshev.cheb2poly(flipped_cheb_series_n)

    #approx_f_t = np.zeros(f_t.shape)
    #for n in np.arange(0, N, 1):
    #    approx_f_t = approx_f_t + flipped_cheb_poly_coeffs[n]*np.power(np.cos(2*np.pi*t), n)
    
    cheb_poly_coeffs = flipped_cheb_poly_coeffs
    scale = cheb_poly_coeffs[0]
    cheb_poly_coeffs = cheb_poly_coeffs/scale

    cheb_poly_roots = np.polynomial.polynomial.polyroots(cheb_poly_coeffs)
  
    return [scale, cheb_poly_roots]


def showPlots(f, f_c, f_c_1, f_c_2, abs_h_f, exponent_normed_squared_abs_h_f, approximated_exponent_normed_squared_abs_h_f, approximated_abs_h_f):

    abs_h_f_dB                                      = 10*np.log10(abs_h_f)
    exponent_normed_squared_abs_h_f_dB              = 10*np.log10(exponent_normed_squared_abs_h_f)
    approximated_exponent_normed_squared_abs_h_f_dB = 10*np.log10(approximated_exponent_normed_squared_abs_h_f)
    approximated_abs_h_f_dB                         = 10*np.log10(approximated_abs_h_f)

    abs_h_f_idxs                                         = np.array([0, 0])
    abs_h_f_dB_idxs                                      = np.array([0, 1])
    exponent_normed_squared_abs_h_f_idxs                 = np.array([1, 0])
    exponent_normed_squared_abs_h_f_dB_idxs              = np.array([1, 1])
    approximated_exponent_normed_squared_abs_h_f_idxs    = np.array([2, 0])
    approximated_exponent_normed_squared_abs_h_f_dB_idxs = np.array([2, 1])
    approximated_abs_h_f_idxs                            = np.array([3, 0])
    approximated_abs_h_f_dB_idxs                         = np.array([3, 1])
    
    idxs_matrix = \
        np.vstack((abs_h_f_idxs, 
                   abs_h_f_dB_idxs, 
                   exponent_normed_squared_abs_h_f_idxs, 
                   exponent_normed_squared_abs_h_f_dB_idxs, 
                   approximated_exponent_normed_squared_abs_h_f_idxs,
                   approximated_exponent_normed_squared_abs_h_f_dB_idxs,
                   approximated_abs_h_f_idxs,
                   approximated_abs_h_f_dB_idxs))
    num_rows = np.amax(idxs_matrix[:, 0]) + 1
    num_cols = np.amax(idxs_matrix[:, 1]) + 1
    [fig, axs] = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    if(len(axs.shape) == 2):
        abs_h_f_axis                                         = axs[                                        abs_h_f_idxs[0],                                         abs_h_f_idxs[1]]
        abs_h_f_dB_axis                                      = axs[                                     abs_h_f_dB_idxs[0],                                      abs_h_f_dB_idxs[1]]
        exponent_normed_squared_abs_h_f_axis                 = axs[                exponent_normed_squared_abs_h_f_idxs[0],                 exponent_normed_squared_abs_h_f_idxs[1]]
        exponent_normed_squared_abs_h_f_dB_axis              = axs[             exponent_normed_squared_abs_h_f_dB_idxs[0],              exponent_normed_squared_abs_h_f_dB_idxs[1]]
        approximated_exponent_normed_squared_abs_h_f_axis    = axs[   approximated_exponent_normed_squared_abs_h_f_idxs[0],    approximated_exponent_normed_squared_abs_h_f_idxs[1]]
        approximated_exponent_normed_squared_abs_h_f_dB_axis = axs[approximated_exponent_normed_squared_abs_h_f_dB_idxs[0], approximated_exponent_normed_squared_abs_h_f_dB_idxs[1]]
        approximated_abs_h_f_axis                            = axs[                           approximated_abs_h_f_idxs[0],                            approximated_abs_h_f_idxs[1]]
        approximated_abs_h_f_dB_axis                         = axs[                        approximated_abs_h_f_dB_idxs[0],                         approximated_abs_h_f_dB_idxs[1]]
    else:
        abs_h_f_axis                                         = axs[np.amax(                                        abs_h_f_idxs)]
        abs_h_f_dB_axis                                      = axs[np.amax(                                     abs_h_f_dB_idxs)]
        exponent_normed_squared_abs_h_f_axis                 = axs[np.amax(                exponent_normed_squared_abs_h_f_idxs)]
        exponent_normed_squared_abs_h_f_dB_axis              = axs[np.amax(             exponent_normed_squared_abs_h_f_dB_idxs)]
        approximated_exponent_normed_squared_abs_h_f_axis    = axs[np.amax(   approximated_exponent_normed_squared_abs_h_f_idxs)]
        approximated_exponent_normed_squared_abs_h_f_dB_axis = axs[np.amax(approximated_exponent_normed_squared_abs_h_f_dB_idxs)]
        approximated_abs_h_f_axis                            = axs[np.amax(                           approximated_abs_h_f_idxs)]
        approximated_abs_h_f_dB_axis                         = axs[np.amax(                        approximated_abs_h_f_dB_idxs)]

    abs_h_f_axis.plot(f, abs_h_f)
    abs_h_f_axis.set_xlim([0, 0.5])
    abs_h_f_axis.set_ylim(bottom=-0.1)
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

    exponent_normed_squared_abs_h_f_axis.plot(f, exponent_normed_squared_abs_h_f)
    exponent_normed_squared_abs_h_f_axis.set_xlim([0, 0.5])
    exponent_normed_squared_abs_h_f_axis.set_ylim(bottom=-0.1)
    exponent_normed_squared_abs_h_f_axis.axvline(f_c_1, color='r')
    exponent_normed_squared_abs_h_f_axis.axvline(f_c, color='k')
    exponent_normed_squared_abs_h_f_axis.axvline(f_c_2, color='r')
    exponent_normed_squared_abs_h_f_axis.grid()
    exponent_normed_squared_abs_h_f_axis.set_xlabel('Frequency (Hz)')
    exponent_normed_squared_abs_h_f_axis.set_ylabel(r'$|h_(f)|$')
    exponent_normed_squared_abs_h_f_axis.set_title('Reduced Target Frequency Magnitude Response')

    exponent_normed_squared_abs_h_f_dB_axis.plot(f, exponent_normed_squared_abs_h_f_dB)
    exponent_normed_squared_abs_h_f_dB_axis.set_xlim([0, 0.5])
    exponent_normed_squared_abs_h_f_dB_axis.axvline(f_c_1, color='r')
    exponent_normed_squared_abs_h_f_dB_axis.axvline(f_c, color='k')
    exponent_normed_squared_abs_h_f_dB_axis.axvline(f_c_2, color='r')
    exponent_normed_squared_abs_h_f_dB_axis.grid()
    exponent_normed_squared_abs_h_f_dB_axis.set_xlabel('Frequency (Hz)')
    exponent_normed_squared_abs_h_f_dB_axis.set_ylabel(r'$|h_(f)|_{dB}$')
    exponent_normed_squared_abs_h_f_dB_axis.set_title('Reduced Target Frequency Magnitude Decibel Response')

    approximated_exponent_normed_squared_abs_h_f_axis.plot(f, approximated_exponent_normed_squared_abs_h_f)
    approximated_exponent_normed_squared_abs_h_f_axis.set_xlim([0, 0.5])
    approximated_exponent_normed_squared_abs_h_f_axis.set_ylim(bottom=-0.1)
    approximated_exponent_normed_squared_abs_h_f_axis.axvline(f_c_1, color='r')
    approximated_exponent_normed_squared_abs_h_f_axis.axvline(f_c, color='k')
    approximated_exponent_normed_squared_abs_h_f_axis.axvline(f_c_2, color='r')
    approximated_exponent_normed_squared_abs_h_f_axis.grid()
    approximated_exponent_normed_squared_abs_h_f_axis.set_xlabel('Frequency (Hz)')
    approximated_exponent_normed_squared_abs_h_f_axis.set_ylabel(r'$|h_(f)|$')
    approximated_exponent_normed_squared_abs_h_f_axis.set_title('Approximated Reduced Target Frequency Magnitude Response')

    approximated_exponent_normed_squared_abs_h_f_dB_axis.plot(f, approximated_exponent_normed_squared_abs_h_f_dB)
    approximated_exponent_normed_squared_abs_h_f_dB_axis.set_xlim([0, 0.5])
    approximated_exponent_normed_squared_abs_h_f_dB_axis.axvline(f_c_1, color='r')
    approximated_exponent_normed_squared_abs_h_f_dB_axis.axvline(f_c, color='k')
    approximated_exponent_normed_squared_abs_h_f_dB_axis.axvline(f_c_2, color='r')
    approximated_exponent_normed_squared_abs_h_f_dB_axis.grid()
    approximated_exponent_normed_squared_abs_h_f_dB_axis.set_xlabel('Frequency (Hz)')
    approximated_exponent_normed_squared_abs_h_f_dB_axis.set_ylabel(r'$|h_(f)|_{dB}$')
    approximated_exponent_normed_squared_abs_h_f_dB_axis.set_title('Approximated Reduced Target Frequency Magnitude Decibel Response')

    approximated_abs_h_f_axis.plot(f, approximated_abs_h_f)
    approximated_abs_h_f_axis.set_xlim([0, 0.5])
    approximated_abs_h_f_axis.set_ylim(bottom=-0.1)
    approximated_abs_h_f_axis.axvline(f_c_1, color='r')
    approximated_abs_h_f_axis.axvline(f_c, color='k')
    approximated_abs_h_f_axis.axvline(f_c_2, color='r')
    approximated_abs_h_f_axis.grid()
    approximated_abs_h_f_axis.set_xlabel('Frequency (Hz)')
    approximated_abs_h_f_axis.set_ylabel(r'$|h_(f)|$')
    approximated_abs_h_f_axis.set_title('Approximated Frequency Magnitude Response')

    approximated_abs_h_f_dB_axis.plot(f, approximated_abs_h_f_dB)
    approximated_abs_h_f_dB_axis.set_xlim([0, 0.5])
    approximated_abs_h_f_dB_axis.axvline(f_c_1, color='r')
    approximated_abs_h_f_dB_axis.axvline(f_c, color='k')
    approximated_abs_h_f_dB_axis.axvline(f_c_2, color='r')
    approximated_abs_h_f_dB_axis.grid()
    approximated_abs_h_f_dB_axis.set_xlabel('Frequency (Hz)')
    approximated_abs_h_f_dB_axis.set_ylabel(r'$|h_(f)|_{dB}$')
    approximated_abs_h_f_dB_axis.set_title('Approximated Frequency Magnitude Decibel Response')

    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):

    f_c = 0.2
    f_delta = 0.075
    f_delta_stop_perc = 0.6
    abs_h_f_c_stop_dB = -25

    reduction_factor = 40
    N = 40

    f_step = 0.0001

    f = np.arange(0, 0.5 + f_step, f_step)

    [f_c_1, f_c_2, f_c, abs_h_f] = calculate_frequency_magnitude_response(f, f_c, f_delta, f_delta_stop_perc, abs_h_f_c_stop_dB)
    exponent_normed_squared_abs_h_f = np.square(np.power(abs_h_f, 1/reduction_factor))
    [_, cheb_poly_roots] = calculateChebyshevPolynomialCoefficients(f, f_delta, exponent_normed_squared_abs_h_f, N)

    approximated_exponent_normed_squared_abs_h_f = \
        utility.chebyshevSpectrumCalculations.calculatePartialChebyshevPowerSpectrum(2*np.pi*f, utility.magnitudeDomainRoots.convertLimitedRootsArrayToRootsDictList(False, 'MA', cheb_poly_roots))
    approximated_exponent_normed_squared_abs_h_f = approximated_exponent_normed_squared_abs_h_f/approximated_exponent_normed_squared_abs_h_f[f == f_c + (f_delta_stop_perc - 1)*f_delta]

    approximated_abs_h_f = np.power(np.sqrt(approximated_exponent_normed_squared_abs_h_f), reduction_factor)

    showPlots(f, f_c, f_c_1, f_c_2, abs_h_f, exponent_normed_squared_abs_h_f, approximated_exponent_normed_squared_abs_h_f, approximated_abs_h_f)

