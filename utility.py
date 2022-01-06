import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt


class zDomainComplexConjugateRootPair:

    # lambda for converting boolean flag (decides whether z-domain roots are inside or outside unit circle) into corresponding exponent
    convert_boolean_to_abs_rho_power_func = lambda z_domain_root_within_unit_circle : 1 - 2*z_domain_root_within_unit_circle

    # Basic math equations for mapping a complex conjugate root pair from the squared frequency magnitude cosine 
    # polynomial to the magnitudes of the corresponding complex conjugate root pair within the z-transform
    eta_func     = lambda gamma_real, abs_gamma_imag : 0.5*(np.square(gamma_real) + np.square(abs_gamma_imag) + 1)
    gamma_func   = lambda eta, gamma_real : np.sqrt(eta + np.sqrt(np.square(eta) - np.square(gamma_real)))
    abs_rho_func = lambda gamma, abs_rho_power : (gamma + np.sqrt(np.square(gamma) - 1))**abs_rho_power

    # Array of coefficients for a z-transform with only one specified pair of complex conjugate roots in the squared frequency
    # magnitude cosine polynomial
    z_transform_coeffs_func = lambda gamma_real, gamma, abs_rho : np.array([1, 2*abs_rho*(gamma_real/gamma), np.square(abs_rho)])

    # Frequency magnitude response for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
    magnitude_response_func = lambda omega, gamma_real, abs_gamma_imag, abs_rho : 2*abs_rho*np.sqrt(np.square(np.cos(omega) + gamma_real) + np.square(abs_gamma_imag))

    # Frequency phase response (in degrees) for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
    arctan_num_func            = lambda omega, gamma, gamma_real, abs_rho : 2*np.sin(omega)*((gamma_real/gamma) + np.cos(omega))
    arctan_den_func            = lambda omega, gamma, gamma_real, abs_rho : 2*np.square(np.cos(omega) + (gamma_real/(2*abs_rho*gamma))) - 2*np.square(gamma_real/(2*abs_rho*gamma)) + np.square(1/abs_rho) - 1
    phase_degree_response_func = lambda arctan_num, arctan_den : -np.rad2deg(np.arctan2(arctan_num, arctan_den))

    @classmethod
    def z_trans_coefs(cls, gamma_real, abs_gamma_imag, z_domain_root_within_unit_circle):

        abs_rho_power = cls.convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)
        gamma = cls.gamma_func(cls.eta_func(gamma_real, abs_gamma_imag), gamma_real)
        z_coefs = cls.z_transform_coeffs_func(gamma_real, gamma, cls.abs_rho_func(gamma, abs_rho_power))

        if z_domain_root_within_unit_circle:
            MA_z_coefs = np.array([1])
            AR_z_coefs = z_coefs
        else:
            MA_z_coefs = z_coefs
            AR_z_coefs = np.array([1])

        return [MA_z_coefs, AR_z_coefs]

    @classmethod
    def freqz(cls, omega, gamma_real, abs_gamma_imag, z_domain_root_within_unit_circle):

        abs_rho_power = cls.convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)
        gamma = cls.gamma_func(cls.eta_func(gamma_real, abs_gamma_imag), gamma_real)
        abs_rho = cls.abs_rho_func(gamma, abs_rho_power)

        arctan_num = cls.arctan_num_func(omega, gamma_real, abs_gamma_imag, abs_rho)
        arctan_den = cls.arctan_den_func(omega, gamma_real, abs_gamma_imag, abs_rho)

        abs_h_f = cls.magnitude_response_func(omega, gamma_real, abs_gamma_imag, abs_rho)**abs_rho_power
        angle_deg_h_f = abs_rho_power*cls.phase_degree_response_func(arctan_num, arctan_den)

        return [abs_h_f, angle_deg_h_f]


class zDomainSingleRealRoot:
    
    # lambda for converting boolean flag (decides whether z-domain roots are inside or outside unit circle) into corresponding exponent
    convert_boolean_to_abs_rho_power_func = lambda z_domain_root_within_unit_circle : 1 - 2*z_domain_root_within_unit_circle

    # Basic math equation for mapping a single root from the squared frequency magnitude cosine 
    # polynomial to the magnitudes of the corresponding single real root within the z-transform
    abs_rho_func = lambda gamma, abs_rho_power : (gamma + np.sqrt(np.square(gamma) - 1))**abs_rho_power

    # Array of coefficients for a z-transform with only one specified single real root 
    # in the squared frequency magnitude cosine polynomial
    z_transform_coeffs_func = lambda abs_rho, rho_sign : np.array([1, rho_sign*abs_rho])

    # Frequency magnitude response for the corresponding discrete fourier transform of a z-transform with only 
    # one specified single real root
    magnitude_response_func = lambda omega, gamma, abs_rho, rho_sign : np.sqrt(2*abs_rho*(gamma + rho_sign*np.cos(omega)))

    # Frequency phase response (in degrees) for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
    phase_degree_response_func = lambda omega, abs_rho, rho_sign : -rho_sign*np.rad2deg(np.arctan2(np.sin(omega), (1/abs_rho) + rho_sign*np.cos(omega)))

    @classmethod
    def z_trans_coefs(cls, gamma, rho_sign, z_domain_root_within_unit_circle):
        
        z_coefs = cls.z_transform_coeffs_func(cls.abs_rho_func(gamma, cls.convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)), rho_sign)

        if z_domain_root_within_unit_circle:
            MA_z_coefs = np.array([1])
            AR_z_coefs = z_coefs
        else:
            MA_z_coefs = z_coefs
            AR_z_coefs = np.array([1])

        return [MA_z_coefs, AR_z_coefs]
    
    @classmethod
    def freqz(cls, omega, gamma, rho_sign, z_domain_root_within_unit_circle):

        abs_rho_power = cls.convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)
        abs_rho = cls.abs_rho_func(gamma, abs_rho_power)

        abs_h_f = cls.magnitude_response_func(omega, gamma, abs_rho, rho_sign)**abs_rho_power
        angle_deg_h_f = abs_rho_power*cls.phase_degree_response_func(omega, abs_rho, rho_sign)

        return [abs_h_f, angle_deg_h_f]


if(__name__=='__main__'):

    ##############################################################################################################################################################################################################################

    z_domain_root_within_unit_circle = False
    #gamma_real = 0.5
    #abs_gamma_imag = 0.05
    gamma = 1.1
    rho_sign = 1
    
    #[MA_z_coefs, AR_z_coefs] = zDomainComplexConjugateRootPair.z_trans_coefs(gamma_real, abs_gamma_imag, z_domain_root_within_unit_circle)
    [MA_z_coefs, AR_z_coefs] = zDomainSingleRealRoot.z_trans_coefs(gamma, rho_sign, z_domain_root_within_unit_circle)

    [omega, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs)
    abs_h_f_emp = np.abs(h_f_emp)
    angle_deg_h_f_emp = np.rad2deg(np.angle(h_f_emp))
    
    #[abs_h_f_theo, angle_deg_h_f_theo] = zDomainComplexConjugateRootPair.freqz(omega, gamma_real, abs_gamma_imag, z_domain_root_within_unit_circle)
    [abs_h_f_theo, angle_deg_h_f_theo] = zDomainSingleRealRoot.freqz(omega, gamma, rho_sign, z_domain_root_within_unit_circle)
    
    f = omega/(2*np.pi)
    
    plt.figure()
    plt.plot(f, abs_h_f_emp)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.title('Empirical Magnitude')
    
    plt.figure()
    plt.plot(f, angle_deg_h_f_emp)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.title('Empirical Phase')
    
    plt.figure()
    plt.plot(f, abs_h_f_theo)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.title('Theoretical Magnitude')
    
    plt.figure()
    plt.plot(f, angle_deg_h_f_theo)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.title('Theoretical Phase')

    plt.show()

