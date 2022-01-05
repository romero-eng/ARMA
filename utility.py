import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

if(__name__=='__main__'):

    eta_func                = lambda gamma_real, abs_gamma_imag : 0.5*(np.square(gamma_real) + np.square(abs_gamma_imag) + 1)
    gamma_func              = lambda eta, gamma_real : np.sqrt(eta + np.sqrt(np.square(eta) - np.square(gamma_real)))
    abs_rho_func            = lambda gamma, abs_rho_power : (gamma + np.sqrt(np.square(gamma) - 1))**abs_rho_power
    z_transform_coeffs_func = lambda gamma_real, gamma, abs_rho : np.array([1, 2*abs_rho*(gamma_real/gamma), np.square(abs_rho)])
    magnitude_response_func = lambda omega, gamma_real, abs_gamma_imag, abs_rho : 2*abs_rho*np.sqrt(np.square(np.cos(omega) + gamma_real) + np.square(abs_gamma_imag))

    semi_final_z_transform_coeffs_func = \
        lambda gamma_real, abs_gamma_imag, abs_rho_power : z_transform_coeffs_func(gamma_real,
                                                                                   gamma_func(eta_func(gamma_real,
                                                                                                       abs_gamma_imag), 
                                                                                              gamma_real),
                                                                                   abs_rho_func(gamma_func(eta_func(gamma_real,
                                                                                                                    abs_gamma_imag),
                                                                                                           gamma_real), 
                                                                                                abs_rho_power))
    final_magnitude_response_func = \
        lambda omega, gamma_real, abs_gamma_imag, abs_rho_power : magnitude_response_func(omega,
                                                                                          gamma_real,
                                                                                          abs_gamma_imag, 
                                                                                          abs_rho_func(gamma_func(eta_func(gamma_real,
                                                                                                                           abs_gamma_imag), 
                                                                                                                  gamma_real), 
                                                                                                       abs_rho_power))**abs_rho_power

    ##############################################################################################################################################################################################################################

    z_domain_root_within_unit_circle = True
    gamma_real = 0.5
    abs_gamma_imag = 0.05

    abs_rho_power = 1 - 2*z_domain_root_within_unit_circle
    
    [omega, h_f_emp] = dsp.freqz(semi_final_z_transform_coeffs_func(gamma_real, abs_gamma_imag, abs_rho_power))
    abs_h_f_emp = np.abs(h_f_emp)

    abs_h_f_theo = final_magnitude_response_func(omega, gamma_real, abs_gamma_imag, abs_rho_power)

    f = omega/(2*np.pi)

    plt.figure()
    plt.plot(f, abs_h_f_emp)
    plt.grid()
    plt.title('Empirical')

    plt.figure()
    plt.plot(f, abs_h_f_theo)
    plt.grid()
    plt.title('Theoretical')

    plt.show()

