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


def showPlots(f, f_c, f_c_1, f_c_2, abs_h_f):

    abs_h_f_dB = 10*np.log10(abs_h_f)

    abs_h_f_idxs    = np.array([0, 0])
    abs_h_f_dB_idxs = np.array([0, 1])
    
    idxs_matrix = np.vstack((abs_h_f_idxs, abs_h_f_dB_idxs))
    num_rows = np.amax(idxs_matrix[:, 0]) + 1
    num_cols = np.amax(idxs_matrix[:, 1]) + 1
    [fig, axs] = plt.subplots(num_rows, num_cols, figsize=(10, 4))
    if(len(axs.shape) == 2):
        abs_h_f_axis    = axs[   abs_h_f_idxs[0],    abs_h_f_idxs[1]]
        abs_h_f_dB_axis = axs[abs_h_f_dB_idxs[0], abs_h_f_dB_idxs[1]]
    else:
        abs_h_f_axis    = axs[np.amax(   abs_h_f_idxs)]
        abs_h_f_dB_axis = axs[np.amax(abs_h_f_dB_idxs)]

    abs_h_f_axis.plot(f, abs_h_f)
    abs_h_f_axis.set_xlim([0, 0.5])
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

    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):

    f_c = 0.2
    f_delta = 0.075
    f_delta_stop_perc = 0.6
    abs_h_f_c_stop_dB = -25

    reduction_factor = 20
    N = 40

    f_step = 0.0001

    f = np.arange(0, 0.5 + f_step, f_step)

    [f_c_1, f_c_2, f_c, abs_h_f] = calculate_frequency_magnitude_response(f, f_c, f_delta, f_delta_stop_perc, abs_h_f_c_stop_dB)

    showPlots(f, f_c, f_c_1, f_c_2, abs_h_f)

