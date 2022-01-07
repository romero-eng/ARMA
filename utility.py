import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt


class zDomainComplexConjugateRootPair:

    @staticmethod
    def get_math_functions():

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
        arctan_num_func            = lambda omega, gamma, gamma_real, abs_rho : 2*np.sin(omega)*((gamma_real/(abs_rho*gamma)) + np.cos(omega))
        arctan_den_func            = lambda omega, gamma, gamma_real, abs_rho : 2*np.square(np.cos(omega) + (gamma_real/(2*abs_rho*gamma))) - 2*np.square(gamma_real/(2*abs_rho*gamma)) + np.square(1/abs_rho) - 1
        phase_degree_response_func = lambda arctan_num, arctan_den : -np.rad2deg(np.arctan2(arctan_num, arctan_den))

        return [convert_boolean_to_abs_rho_power_func,
                eta_func, gamma_func, abs_rho_func, 
                z_transform_coeffs_func, 
                magnitude_response_func, 
                arctan_num_func, arctan_den_func, 
                phase_degree_response_func]

    @staticmethod
    def interpret_squared_frequency_magnitude_cosine_polynomial_root(square_freq_mag_cos_poly_root):

        square_freq_mag_cos_poly_root = -1*square_freq_mag_cos_poly_root

        gamma_real     = np.real(square_freq_mag_cos_poly_root)
        abs_gamma_imag = np.imag(square_freq_mag_cos_poly_root)

        if(abs_gamma_imag <= 0):
            error('The given complex root of a squared frequency magnitude cosine polynomial must have a negative imaginary component.')

        return [gamma_real, abs_gamma_imag]

    @staticmethod
    def z_trans_coefs(square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR):

        [convert_boolean_to_abs_rho_power_func,
         eta_func, gamma_func, abs_rho_func, 
         z_transform_coeffs_func, _, _, _, _] = \
            zDomainComplexConjugateRootPair.get_math_functions()

        [MA_flag, AR_flag] = zDomainRoot.MA_or_AR_interpretation(MA_or_AR, z_domain_root_within_unit_circle)

        [gamma_real, abs_gamma_imag] = zDomainComplexConjugateRootPair.interpret_squared_frequency_magnitude_cosine_polynomial_root(square_freq_mag_cos_poly_root)

        abs_rho_power = convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)
        gamma = gamma_func(eta_func(gamma_real, abs_gamma_imag), gamma_real)
        z_coefs = z_transform_coeffs_func(gamma_real, gamma, abs_rho_func(gamma, abs_rho_power))

        if(AR_flag):
            MA_z_coefs = np.array([1])
            AR_z_coefs = z_coefs
        elif(MA_flag):
            MA_z_coefs = z_coefs
            AR_z_coefs = np.array([1])

        return [MA_z_coefs, AR_z_coefs]

    @staticmethod
    def freqz(omega, square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR):

        [convert_boolean_to_abs_rho_power_func,
         eta_func, gamma_func, abs_rho_func, 
         _, magnitude_response_func, 
         arctan_num_func, arctan_den_func, 
         phase_degree_response_func] = \
            zDomainComplexConjugateRootPair.get_math_functions()

        [_, AR_flag] = zDomainRoot.MA_or_AR_interpretation(MA_or_AR, z_domain_root_within_unit_circle)

        [gamma_real, abs_gamma_imag] = zDomainComplexConjugateRootPair.interpret_squared_frequency_magnitude_cosine_polynomial_root(square_freq_mag_cos_poly_root)

        abs_rho_power = convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)
        gamma = gamma_func(eta_func(gamma_real, abs_gamma_imag), gamma_real)
        abs_rho = abs_rho_func(gamma, abs_rho_power)

        arctan_num = arctan_num_func(omega, gamma, gamma_real, abs_rho)
        arctan_den = arctan_den_func(omega, gamma, gamma_real, abs_rho)

        abs_h_f = magnitude_response_func(omega, gamma_real, abs_gamma_imag, abs_rho)
        angle_deg_h_f = phase_degree_response_func(arctan_num, arctan_den)

        if(AR_flag):
            abs_h_f = 1/abs_h_f
            angle_deg_h_f = -angle_deg_h_f

        return [abs_h_f, angle_deg_h_f]


class zDomainSingleRealRoot:

    @staticmethod
    def get_math_functions():

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

        return [convert_boolean_to_abs_rho_power_func,
                abs_rho_func,
                z_transform_coeffs_func,
                magnitude_response_func,
                phase_degree_response_func]

    @staticmethod
    def interpret_squared_frequency_magnitude_cosine_polynomial_root(square_freq_mag_cos_poly_root):

        rho_sign = -1*np.sign(square_freq_mag_cos_poly_root)
        gamma = np.abs(square_freq_mag_cos_poly_root)

        if(gamma <= 1):
            error('The given single real root must have an absolute value greater than one.')

        return [gamma, rho_sign]

    @staticmethod
    def z_trans_coefs(square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR):

        [convert_boolean_to_abs_rho_power_func,
         abs_rho_func,
         z_transform_coeffs_func,
         _, _] = \
            zDomainSingleRealRoot.get_math_functions()

        [MA_flag, AR_flag] = zDomainRoot.MA_or_AR_interpretation(MA_or_AR, z_domain_root_within_unit_circle)

        [gamma, rho_sign] = zDomainSingleRealRoot.interpret_squared_frequency_magnitude_cosine_polynomial_root(square_freq_mag_cos_poly_root)
        
        z_coefs = z_transform_coeffs_func(abs_rho_func(gamma, convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle)), rho_sign)

        if(AR_flag):
            MA_z_coefs = np.array([1])
            AR_z_coefs = z_coefs
        elif(MA_flag):
            MA_z_coefs = z_coefs
            AR_z_coefs = np.array([1])

        return [MA_z_coefs, AR_z_coefs]
    
    @staticmethod
    def freqz(omega, square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR):

        [convert_boolean_to_abs_rho_power_func,
         abs_rho_func, _,
         magnitude_response_func,
         phase_degree_response_func] = \
            zDomainSingleRealRoot.get_math_functions()

        [_, AR_flag] = zDomainRoot.MA_or_AR_interpretation(MA_or_AR, z_domain_root_within_unit_circle)

        [gamma, rho_sign] = zDomainSingleRealRoot.interpret_squared_frequency_magnitude_cosine_polynomial_root(square_freq_mag_cos_poly_root)

        abs_rho = abs_rho_func(gamma, convert_boolean_to_abs_rho_power_func(z_domain_root_within_unit_circle))

        abs_h_f = magnitude_response_func(omega, gamma, abs_rho, rho_sign)
        angle_deg_h_f = phase_degree_response_func(omega, abs_rho, rho_sign)

        if(AR_flag):
            abs_h_f = 1/abs_h_f
            angle_deg_h_f = -angle_deg_h_f

        return [abs_h_f, angle_deg_h_f]


class zDomainRoot():

    @staticmethod
    def MA_or_AR_interpretation(MA_or_AR, z_domain_root_within_unit_circle):

        MA_flag = MA_or_AR == 'MA'
        AR_flag = MA_or_AR == 'AR'

        if((not MA_flag) and (not AR_flag)):
            raise ValueError("The 'MA_or_AR' variable must either be 'MA' or 'AR'")
        
        if(AR_flag and (not z_domain_root_within_unit_circle)):
            raise ValueError("Cannot have autoregressive (IIR) coefficients while the roots of the corresponding z-transform are outside the unit circle")

        return [MA_flag, AR_flag]
    
    @staticmethod
    def z_trans_coefs(square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR):

        if(isinstance(square_freq_mag_cos_poly_root, complex)):
            [MA_z_coefs, AR_z_coefs] = zDomainComplexConjugateRootPair.z_trans_coefs(square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR)
        else:
            [MA_z_coefs, AR_z_coefs] =           zDomainSingleRealRoot.z_trans_coefs(square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR)
    
        return [MA_z_coefs, AR_z_coefs]

    @staticmethod
    def freqz(omega, square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR):

        if(isinstance(square_freq_mag_cos_poly_root, complex)):
            [abs_h_f_theo, angle_deg_h_f_theo] = zDomainComplexConjugateRootPair.freqz(omega, square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR)
        else:
            [abs_h_f_theo, angle_deg_h_f_theo] =           zDomainSingleRealRoot.freqz(omega, square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR)
    
        return [abs_h_f_theo, angle_deg_h_f_theo]


def testByPlotting(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo):

    f = omega/(2*np.pi)
    
    [fig, axs] = plt.subplots(2, 2)

    axs[0, 0].plot(f, abs_h_f_emp)
    axs[0, 0].set_xlim([0, 0.5])
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].grid()
    axs[0, 0].set_xlabel('Frequency (Hz)')
    axs[0, 0].set_title('Empirical Magnitude')
    
    axs[0, 1].plot(f, angle_deg_h_f_emp)
    axs[0, 1].set_xlim([0, 0.5])
    axs[0, 1].grid()
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].set_title('Empirical Phase')
    
    axs[1, 0].plot(f, abs_h_f_theo)
    axs[1, 0].set_xlim([0, 0.5])
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].grid()
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_title('Theoretical Magnitude')
    
    axs[1, 1].plot(f, angle_deg_h_f_theo)
    axs[1, 1].set_xlim([0, 0.5])
    axs[1, 1].grid()
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_title('Theoretical Phase')

    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):

    z_domain_root_within_unit_circle =  True; MA_or_AR = 'MA'; square_freq_mag_cos_poly_root = -(-1.1)
    #z_domain_root_within_unit_circle = False; MA_or_AR = 'MA'; square_freq_mag_cos_poly_root = -(-1.1)
    #z_domain_root_within_unit_circle =  True; MA_or_AR = 'AR'; square_freq_mag_cos_poly_root = -(-1.1)
    #z_domain_root_within_unit_circle = False; MA_or_AR = 'AR'; square_freq_mag_cos_poly_root = -(-1.1)         # should generate a ValueError
    #z_domain_root_within_unit_circle =  True; MA_or_AR = 'MA'; square_freq_mag_cos_poly_root = -(-0.5 + 0.05j)
    #z_domain_root_within_unit_circle = False; MA_or_AR = 'MA'; square_freq_mag_cos_poly_root = -(-0.5 + 0.05j)
    #z_domain_root_within_unit_circle =  True; MA_or_AR = 'AR'; square_freq_mag_cos_poly_root = -(-0.5 + 0.05j)
    #z_domain_root_within_unit_circle = False; MA_or_AR = 'AR'; square_freq_mag_cos_poly_root = -(-0.5 + 0.05j) # should generate a ValueError

    [MA_z_coefs, AR_z_coefs] = zDomainRoot.z_trans_coefs(square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR)

    [omega, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs)
    abs_h_f_emp = np.abs(h_f_emp)
    angle_deg_h_f_emp = np.rad2deg(np.angle(h_f_emp))
    
    [abs_h_f_theo, angle_deg_h_f_theo] = zDomainRoot.freqz(omega, square_freq_mag_cos_poly_root, z_domain_root_within_unit_circle, MA_or_AR)
    
    testByPlotting(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo)

