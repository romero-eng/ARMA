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
            raise ValueError('The given complex root of a squared frequency magnitude cosine polynomial must have a negative imaginary component.')

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
            raise ValueError('The given single real root must have an absolute value greater than one.')

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


class magnitudeDomainRoots():

    @staticmethod
    def calculatePartialChebyshevPowerSpectrum(omega, MA_or_AR_root_tuples_list):

        real_roots = []
        complex_roots = []
        for root_tuple in MA_or_AR_root_tuples_list:
            if(isinstance(root_tuple[2], complex)):
                complex_roots.append(root_tuple[2])
            else:
                real_roots.append(root_tuple[2])

        real_roots = np.array(real_roots)
        complex_roots = np.array(complex_roots)
        complex_roots = np.hstack((complex_roots, np.conjugate(complex_roots)))
        roots = np.hstack((real_roots, complex_roots))

        cheb_series_coefs = np.polynomial.chebyshev.poly2cheb(np.real(np.polynomial.polynomial.polyfromroots(roots)))
    
        squared_abs_h_f_cheb_theo = np.zeros(omega.shape)
        for n in np.arange(0, len(cheb_series_coefs), 1):
            squared_abs_h_f_cheb_theo = squared_abs_h_f_cheb_theo + cheb_series_coefs[n]*np.cos(n*omega)
    
        if( np.sum(np.sign(real_roots) == 1) % 2  == 1 ):
            squared_abs_h_f_cheb_theo = -1*squared_abs_h_f_cheb_theo

        abs_h_f_cheb_theo = np.sqrt(squared_abs_h_f_cheb_theo)

        return abs_h_f_cheb_theo

    @staticmethod
    def calculateEntireChebyshevPowerSpectrum(omega, root_tuples_list):

        AR_root_tuples_list = []
        MA_root_tuples_list = []
        for root_tuple in root_tuples_list:
            if(root_tuple[1] == 'MA'):
                MA_root_tuples_list.append(root_tuple)
            elif(root_tuple[1] == 'AR'):
                AR_root_tuples_list.append(root_tuple)
            else:
                raise ValueError('Unexpected value for MA/AR description for the following root: ' + str(root_tuple))

        AR_abs_h_f_cheb_theo = magnitudeDomainRoots.calculatePartialChebyshevPowerSpectrum(omega, AR_root_tuples_list)
        MA_abs_h_f_cheb_theo = magnitudeDomainRoots.calculatePartialChebyshevPowerSpectrum(omega, MA_root_tuples_list)
        abs_h_f_cheb_theo = MA_abs_h_f_cheb_theo/AR_abs_h_f_cheb_theo

        return abs_h_f_cheb_theo

    @staticmethod
    def generateEmpiricalAndTheoreticalResponses(root_tuples_list):

        num_roots = len(root_tuples_list)

        MA_z_coefs = np.array([1])
        AR_z_coefs = np.array([1])
        for root_idx in np.arange(0, num_roots):
            [tmp_MA_z_coefs,
             tmp_AR_z_coefs] = \
                 zDomainRoot.z_trans_coefs(root_tuples_list[root_idx][2], 
                                           root_tuples_list[root_idx][0], 
                                           root_tuples_list[root_idx][1])
            MA_z_coefs = np.polynomial.polynomial.polymul(MA_z_coefs, tmp_MA_z_coefs)
            AR_z_coefs = np.polynomial.polynomial.polymul(AR_z_coefs, tmp_AR_z_coefs)

        [omega, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs)
        abs_h_f_emp = np.abs(h_f_emp)
        angle_deg_h_f_emp = np.rad2deg(np.angle(h_f_emp))

        abs_h_f_theo = np.ones(abs_h_f_emp.shape)
        angle_deg_h_f_theo = np.zeros(angle_deg_h_f_emp.shape)
        for root_idx in np.arange(0, num_roots):
            [tmp_abs_h_f_theo,
             tmp_angle_deg_h_f_theo] = \
                 zDomainRoot.freqz(omega,
                                   root_tuples_list[root_idx][2], 
                                   root_tuples_list[root_idx][0], 
                                   root_tuples_list[root_idx][1])
            abs_h_f_theo = tmp_abs_h_f_theo*abs_h_f_theo
            angle_deg_h_f_theo = angle_deg_h_f_theo + tmp_angle_deg_h_f_theo
        angle_deg_h_f_theo = np.rad2deg(np.arctan2(np.sin(np.deg2rad(angle_deg_h_f_theo)), np.cos(np.deg2rad(angle_deg_h_f_theo)))) # this is done to wrap the phase response

        abs_h_f_cheb_theo = magnitudeDomainRoots.calculateEntireChebyshevPowerSpectrum(omega, root_tuples_list)

        return [omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo, abs_h_f_cheb_theo]


def generateSpectralPlots(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo, abs_h_f_cheb_theo):

    f = omega/(2*np.pi)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    [fig1, axs] = plt.subplots(2, 3)

    axs[0, 0].plot(f, abs_h_f_emp)
    axs[0, 0].set_xlim([0, 0.5])
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].grid()
    axs[0, 0].set_xlabel('Frequency (Hz)')
    axs[0, 0].set_ylabel(r'$\big|h(f)\big|$')
    axs[0, 0].set_title('Empirical Magnitude')
    
    axs[1, 0].plot(f, angle_deg_h_f_emp)
    axs[1, 0].set_xlim([0, 0.5])
    axs[1, 0].grid()
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel(r'${\angle}h(f)^{\circ}$')
    axs[1, 0].set_title('Empirical Phase')
    
    axs[0, 1].plot(f, abs_h_f_theo)
    axs[0, 1].set_xlim([0, 0.5])
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].grid()
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].set_ylabel(r'$\big|h(f)\big|$')
    axs[0, 1].set_title('Theoretical Magnitude')
    
    axs[1, 1].plot(f, angle_deg_h_f_theo)
    axs[1, 1].set_xlim([0, 0.5])
    axs[1, 1].grid()
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel(r'${\angle}h(f)^{\circ}$')
    axs[1, 1].set_title('Theoretical Phase')

    axs[0, 2].plot(f, abs_h_f_cheb_theo)
    axs[0, 2].set_xlim([0, 0.5])
    if(not np.any(abs_h_f_cheb_theo < 0)):
        axs[0, 2].set_ylim(bottom=0)
    axs[0, 2].grid()
    axs[0, 2].set_xlabel('Frequency (Hz)')
    axs[0, 2].set_ylabel(r'$\big|h(f)\big||$')
    axs[0, 2].set_title('Chebyshev Magnitude')

    fig1.tight_layout()

    plt.show()


if(__name__=='__main__'):

    root_tuples_list = \
        [(False, 'MA',    1.1), 
         ( True, 'MA',   -1.4), 
         ( True, 'AR', -( 0.45 + 0.86j)), 
         ( True, 'AR', -(-5.45 + 4.76j))]

    [omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo, abs_h_f_cheb_theo] = magnitudeDomainRoots.generateEmpiricalAndTheoreticalResponses(root_tuples_list)

    #################################################################################################################################################################################################
    #################################################################################################################################################################################################
    #################################################################################################################################################################################################

    generateSpectralPlots(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo, abs_h_f_cheb_theo)

    #normed_abs_h_f_emp = abs_h_f_emp/np.amax(abs_h_f_emp)
    #normed_abs_h_f_theo = abs_h_f_theo/np.amax(abs_h_f_theo)
    #normed_abs_h_f_cheb_theo = abs_h_f_cheb_theo/np.amax(abs_h_f_cheb_theo)
    #generateSpectralPlots(omega, normed_abs_h_f_emp, angle_deg_h_f_emp, normed_abs_h_f_theo, angle_deg_h_f_theo, normed_abs_h_f_cheb_theo)
